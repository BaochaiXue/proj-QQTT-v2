"""Pure-numpy disparity/depth/confidence product builders for the FFS runner.

Extracted verbatim from ``fast_foundation_stereo.py`` (behavior-preserving split).
These helpers turn raw TensorRT disparity output into the metric-depth product
dicts consumed downstream. The only heavier dependency is
``disparity_to_metric_depth`` from ``demo_v7.runtime.utils.depth_geometry``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from demo_v7.runtime.utils.depth_geometry import disparity_to_metric_depth


def compute_disparity_audit_stats(disparity_raw: np.ndarray) -> dict[str, float]:
    disparity = np.asarray(disparity_raw, dtype=np.float32)
    finite = np.isfinite(disparity)
    finite_values = disparity[finite]
    total_count = int(disparity.size)
    finite_count = int(np.count_nonzero(finite))
    positive = finite & (disparity > 0)
    nonpositive = finite & (disparity <= 0)
    stats = {
        "pixel_count": float(total_count),
        "finite_ratio": float(finite_count / max(1, total_count)),
        "positive_ratio": float(np.count_nonzero(positive) / max(1, total_count)),
        "nonpositive_ratio": float(np.count_nonzero(nonpositive) / max(1, total_count)),
        "positive_fraction_of_finite": 0.0,
        "nonpositive_fraction_of_finite": 0.0,
        "min_disparity": 0.0,
        "max_disparity": 0.0,
        "mean_disparity": 0.0,
        "mean_abs_disparity": 0.0,
        "p50_abs_disparity": 0.0,
        "p90_abs_disparity": 0.0,
    }
    if finite_count <= 0:
        return stats
    stats["positive_fraction_of_finite"] = float(np.count_nonzero(positive) / finite_count)
    stats["nonpositive_fraction_of_finite"] = float(np.count_nonzero(nonpositive) / finite_count)
    stats["min_disparity"] = float(np.min(finite_values))
    stats["max_disparity"] = float(np.max(finite_values))
    stats["mean_disparity"] = float(np.mean(finite_values))
    abs_values = np.abs(finite_values)
    stats["mean_abs_disparity"] = float(np.mean(abs_values))
    stats["p50_abs_disparity"] = float(np.quantile(abs_values, 0.50))
    stats["p90_abs_disparity"] = float(np.quantile(abs_values, 0.90))
    return stats


def build_disparity_products(
    disparity_raw: np.ndarray,
    *,
    K_ir_left: np.ndarray,
    baseline_m: float,
    scale: float,
    scale_x: float | None = None,
    scale_y: float | None = None,
    valid_iters: int,
    max_disp: int,
    audit_mode: bool,
) -> dict[str, np.ndarray | float | list[list[float]]]:
    disparity_raw = np.asarray(disparity_raw, dtype=np.float32)
    disparity = disparity_raw.clip(0, None).astype(np.float32)
    scale_x = float(scale if scale_x is None else scale_x)
    scale_y = float(scale if scale_y is None else scale_y)
    K_used = np.asarray(K_ir_left, dtype=np.float32).copy()
    K_used[0, :] *= scale_x
    K_used[1, :] *= scale_y
    depth_ir_left_m = disparity_to_metric_depth(
        disparity,
        fx_ir=float(K_used[0, 0]),
        baseline_m=float(baseline_m),
    )
    result = {
        "disparity": disparity,
        "depth_ir_left_m": depth_ir_left_m,
        "K_ir_left_used": K_used,
        "baseline_m": float(baseline_m),
        "scale": float(scale),
        "resize_scale_x": float(scale_x),
        "resize_scale_y": float(scale_y),
        "valid_iters": int(valid_iters),
        "max_disp": int(max_disp),
    }
    if audit_mode:
        result["disparity_raw"] = disparity_raw
        result["audit_stats"] = compute_disparity_audit_stats(disparity_raw)
    return result


def split_disparity_batch_output_maps(
    disparity_raw: np.ndarray,
    *,
    expected_batch_size: int,
) -> list[np.ndarray]:
    disparity_raw = np.asarray(disparity_raw, dtype=np.float32)
    batch_size = int(expected_batch_size)
    if batch_size <= 0:
        raise ValueError(f"expected_batch_size must be positive, got {expected_batch_size}.")

    if disparity_raw.ndim == 4:
        if disparity_raw.shape[0] != batch_size:
            raise ValueError(
                "Expected TensorRT disparity batch dimension to match requested batch size. "
                f"Got shape={disparity_raw.shape} expected_batch_size={batch_size}."
            )
        if disparity_raw.shape[1] != 1:
            raise ValueError(f"Expected single-channel disparity output, got shape={disparity_raw.shape}.")
        return [np.asarray(disparity_raw[idx, 0], dtype=np.float32) for idx in range(batch_size)]

    if disparity_raw.ndim == 3:
        if disparity_raw.shape[0] != batch_size:
            raise ValueError(
                "Expected disparity batch dimension to match requested batch size. "
                f"Got shape={disparity_raw.shape} expected_batch_size={batch_size}."
            )
        return [np.asarray(disparity_raw[idx], dtype=np.float32) for idx in range(batch_size)]

    if disparity_raw.ndim == 2:
        if batch_size != 1:
            raise ValueError(
                "Expected a batched disparity output but received a single map. "
                f"Got shape={disparity_raw.shape} expected_batch_size={batch_size}."
            )
        return [np.asarray(disparity_raw, dtype=np.float32)]

    raise ValueError(f"Expected 2D/3D/4D disparity output, got shape={disparity_raw.shape}.")


def undo_tensorrt_disparity_transform(
    disparity_raw: np.ndarray,
    *,
    transform: dict[str, int | float | str],
) -> np.ndarray:
    disparity_raw = np.asarray(disparity_raw, dtype=np.float32)
    mode = str(transform["mode"])
    if mode in {"match", "resize"}:
        return disparity_raw
    if mode == "pad":
        pad_top = int(transform["pad_top"])
        pad_bottom = int(transform["pad_bottom"])
        pad_left = int(transform["pad_left"])
        pad_right = int(transform["pad_right"])
        height_end = disparity_raw.shape[0] - pad_bottom
        width_end = disparity_raw.shape[1] - pad_right
        return disparity_raw[pad_top:height_end, pad_left:width_end]
    raise ValueError(f"Unsupported TensorRT disparity transform mode: {mode}")


def finalize_tensorrt_disparity_batch_outputs(
    disparity_raw: np.ndarray,
    *,
    transform: dict[str, int | float | str],
    batch_samples: list[dict[str, Any]],
    valid_iters: int,
    max_disp: int,
) -> list[dict[str, np.ndarray | float | list[list[float]]]]:
    if not batch_samples:
        raise ValueError("Expected at least one batch sample.")

    disparity_maps = split_disparity_batch_output_maps(
        disparity_raw,
        expected_batch_size=len(batch_samples),
    )
    scale_x = float(transform["scale_x"])
    scale_y = float(transform["scale_y"])
    uniform_scale = scale_x if abs(scale_x - scale_y) <= 1e-6 else 1.0

    outputs: list[dict[str, np.ndarray | float | list[list[float]]]] = []
    for sample, disparity_map in zip(batch_samples, disparity_maps):
        disparity_map = undo_tensorrt_disparity_transform(disparity_map, transform=transform)
        outputs.append(
            build_disparity_products(
                disparity_map,
                K_ir_left=np.asarray(sample["K_ir_left"], dtype=np.float32),
                baseline_m=float(sample["baseline_m"]),
                scale=uniform_scale,
                scale_x=scale_x,
                scale_y=scale_y,
                valid_iters=int(valid_iters),
                max_disp=int(max_disp),
                audit_mode=bool(sample.get("audit_mode", False)),
            )
        )
    return outputs
