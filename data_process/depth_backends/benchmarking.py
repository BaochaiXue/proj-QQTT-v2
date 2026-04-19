from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class FfsBenchmarkConfig:
    model_path: str
    model_label: str
    scale: float
    valid_iters: int
    max_disp: int

    @property
    def config_id(self) -> str:
        scale_label = f"{self.scale:.2f}".rstrip("0").rstrip(".")
        return (
            f"{self.model_label}"
            f"_scale{scale_label}"
            f"_iters{int(self.valid_iters)}"
            f"_disp{int(self.max_disp)}"
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["config_id"] = self.config_id
        return payload


def infer_model_label(model_path: str | Path) -> str:
    path = Path(model_path)
    if path.parent.name:
        return path.parent.name
    if path.stem:
        return path.stem
    return str(path)


def expand_benchmark_configs(
    *,
    model_paths: Sequence[str | Path],
    scales: Sequence[float],
    valid_iters_values: Sequence[int],
    max_disp_values: Sequence[int],
) -> list[FfsBenchmarkConfig]:
    if not model_paths:
        raise ValueError("At least one model path is required.")
    if not scales:
        raise ValueError("At least one scale is required.")
    if not valid_iters_values:
        raise ValueError("At least one valid_iters value is required.")
    if not max_disp_values:
        raise ValueError("At least one max_disp value is required.")

    configs: list[FfsBenchmarkConfig] = []
    for model_path, scale, valid_iters, max_disp in product(
        model_paths,
        scales,
        valid_iters_values,
        max_disp_values,
    ):
        configs.append(
            FfsBenchmarkConfig(
                model_path=str(Path(model_path).resolve()),
                model_label=infer_model_label(model_path),
                scale=float(scale),
                valid_iters=int(valid_iters),
                max_disp=int(max_disp),
            )
        )
    return configs


def summarize_latency_samples_ms(samples_ms: Sequence[float]) -> dict[str, float | int]:
    samples = np.asarray(list(samples_ms), dtype=np.float64)
    if samples.size <= 0:
        raise ValueError("Latency samples must not be empty.")

    mean_ms = float(np.mean(samples))
    median_ms = float(np.median(samples))
    return {
        "sample_count": int(samples.size),
        "latency_mean_ms": mean_ms,
        "latency_median_ms": median_ms,
        "latency_std_ms": float(np.std(samples)),
        "latency_min_ms": float(np.min(samples)),
        "latency_max_ms": float(np.max(samples)),
        "latency_p90_ms": float(np.quantile(samples, 0.90)),
        "fps_from_mean": float(1000.0 / mean_ms) if mean_ms > 0 else 0.0,
        "fps_from_median": float(1000.0 / median_ms) if median_ms > 0 else 0.0,
    }


def resize_depth_nearest(depth_m: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"Expected 2D depth map, got shape={depth.shape}.")
    target_h = int(target_shape[0])
    target_w = int(target_shape[1])
    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"Invalid target_shape={target_shape}.")
    if depth.shape == (target_h, target_w):
        return depth

    src_h, src_w = depth.shape
    yy = np.clip(
        np.round(((np.arange(target_h, dtype=np.float32) + 0.5) * src_h / target_h) - 0.5).astype(np.int32),
        0,
        src_h - 1,
    )
    xx = np.clip(
        np.round(((np.arange(target_w, dtype=np.float32) + 0.5) * src_w / target_w) - 0.5).astype(np.int32),
        0,
        src_w - 1,
    )
    return depth[yy[:, None], xx[None, :]]


def compute_reference_depth_metrics(
    reference_depth_m: np.ndarray,
    candidate_depth_m: np.ndarray,
) -> dict[str, float]:
    reference = np.asarray(reference_depth_m, dtype=np.float32)
    candidate = np.asarray(candidate_depth_m, dtype=np.float32)
    if reference.shape != candidate.shape:
        candidate = resize_depth_nearest(candidate, reference.shape)

    total = int(reference.size)
    valid_reference = np.isfinite(reference) & (reference > 0)
    valid_candidate = np.isfinite(candidate) & (candidate > 0)
    overlap = valid_reference & valid_candidate
    overlap_count = int(np.count_nonzero(overlap))
    metrics = {
        "pixel_count": float(total),
        "reference_valid_ratio": float(np.count_nonzero(valid_reference) / max(1, total)),
        "candidate_valid_ratio": float(np.count_nonzero(valid_candidate) / max(1, total)),
        "overlap_valid_ratio": float(overlap_count / max(1, total)),
        "mean_abs_depth_diff_m": 0.0,
        "median_abs_depth_diff_m": 0.0,
        "p90_abs_depth_diff_m": 0.0,
        "max_abs_depth_diff_m": 0.0,
    }
    if overlap_count <= 0:
        return metrics

    abs_diff = np.abs(candidate[overlap] - reference[overlap]).astype(np.float32)
    metrics["mean_abs_depth_diff_m"] = float(np.mean(abs_diff))
    metrics["median_abs_depth_diff_m"] = float(np.median(abs_diff))
    metrics["p90_abs_depth_diff_m"] = float(np.quantile(abs_diff, 0.90))
    metrics["max_abs_depth_diff_m"] = float(np.max(abs_diff))
    return metrics


def find_result_by_config_id(
    results: Iterable[dict[str, Any]],
    *,
    config_id: str,
) -> dict[str, Any] | None:
    for result in results:
        if str(result.get("config", {}).get("config_id")) == str(config_id):
            return result
    return None


def select_fastest_result(results: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    if not results:
        return None
    return min(
        results,
        key=lambda item: (
            float(item["latency_summary"]["latency_mean_ms"]),
            float(item["latency_summary"]["latency_p90_ms"]),
        ),
    )


def select_most_reference_like_result(results: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    if not results:
        return None
    return min(
        results,
        key=lambda item: (
            float(item["reference_metrics"]["median_abs_depth_diff_m"]),
            float(item["reference_metrics"]["p90_abs_depth_diff_m"]),
            -float(item["reference_metrics"]["overlap_valid_ratio"]),
            float(item["latency_summary"]["latency_mean_ms"]),
        ),
    )


def select_tradeoff_result(
    results: Sequence[dict[str, Any]],
    *,
    target_fps: float,
) -> dict[str, Any] | None:
    eligible = [
        item
        for item in results
        if float(item["latency_summary"]["fps_from_mean"]) >= float(target_fps)
    ]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda item: (
            float(item["reference_metrics"]["median_abs_depth_diff_m"]),
            float(item["reference_metrics"]["p90_abs_depth_diff_m"]),
            -float(item["reference_metrics"]["overlap_valid_ratio"]),
            float(item["latency_summary"]["latency_mean_ms"]),
        ),
    )


def build_tradeoff_summary(
    results: Sequence[dict[str, Any]],
    *,
    target_fps_values: Sequence[float],
) -> dict[str, Any]:
    fastest = select_fastest_result(results)
    most_reference_like = select_most_reference_like_result(results)
    targets: dict[str, Any] = {}
    for target_fps in target_fps_values:
        selected = select_tradeoff_result(results, target_fps=float(target_fps))
        targets[f"{float(target_fps):.1f}"] = None if selected is None else {
            "config_id": str(selected["config"]["config_id"]),
            "fps_from_mean": float(selected["latency_summary"]["fps_from_mean"]),
            "latency_mean_ms": float(selected["latency_summary"]["latency_mean_ms"]),
            "median_abs_depth_diff_m": float(selected["reference_metrics"]["median_abs_depth_diff_m"]),
            "p90_abs_depth_diff_m": float(selected["reference_metrics"]["p90_abs_depth_diff_m"]),
        }
    return {
        "fastest_overall": None if fastest is None else {
            "config_id": str(fastest["config"]["config_id"]),
            "fps_from_mean": float(fastest["latency_summary"]["fps_from_mean"]),
            "latency_mean_ms": float(fastest["latency_summary"]["latency_mean_ms"]),
        },
        "most_reference_like": None if most_reference_like is None else {
            "config_id": str(most_reference_like["config"]["config_id"]),
            "fps_from_mean": float(most_reference_like["latency_summary"]["fps_from_mean"]),
            "latency_mean_ms": float(most_reference_like["latency_summary"]["latency_mean_ms"]),
            "median_abs_depth_diff_m": float(most_reference_like["reference_metrics"]["median_abs_depth_diff_m"]),
        },
        "targets": targets,
    }
