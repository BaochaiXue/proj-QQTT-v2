from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from .geometry import disparity_to_metric_depth

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _disable_torch_compile(torch_module) -> None:
    def identity_compile(fn=None, *args, **kwargs):
        if fn is None:
            def decorator(inner):
                return inner
            return decorator
        return fn

    torch_module.compile = identity_compile


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


def apply_remove_invisible_mask(disparity_raw: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    disparity = np.asarray(disparity_raw, dtype=np.float32).clip(0, None)
    if disparity.ndim != 2:
        raise ValueError(f"Expected 2D disparity map, got shape={disparity.shape}.")

    _, width = disparity.shape
    xx = np.broadcast_to(np.arange(width, dtype=np.float32)[None, :], disparity.shape)
    remove_mask = np.isfinite(disparity) & (disparity > 0) & ((xx - disparity) < 0)
    masked = disparity.copy()
    masked[remove_mask] = np.inf

    pixel_count = int(disparity.size)
    removed_count = int(np.count_nonzero(remove_mask))
    return masked, {
        "pixel_count": float(pixel_count),
        "remove_invisible_pixel_count": float(removed_count),
        "remove_invisible_ratio": float(removed_count / max(1, pixel_count)),
    }


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


def compute_confidence_proxies_from_logits(logits: np.ndarray) -> dict[str, np.ndarray]:
    logits = np.asarray(logits, dtype=np.float32)
    if logits.ndim == 5 and logits.shape[1] == 1:
        logits = logits[:, 0]
    if logits.ndim != 4:
        raise ValueError(f"Expected logits shaped [B, D, H, W], got {logits.shape}.")
    if logits.shape[1] <= 0:
        raise ValueError(f"Expected positive disparity-bin axis, got {logits.shape}.")

    logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits_shifted)
    prob_denom = np.sum(exp_logits, axis=1, keepdims=True)
    prob = np.divide(
        exp_logits,
        prob_denom,
        out=np.zeros_like(exp_logits, dtype=np.float32),
        where=prob_denom > 0,
    )
    prob = np.nan_to_num(prob, nan=0.0, posinf=0.0, neginf=0.0)
    prob = prob.astype(np.float32, copy=False)
    sorted_prob = np.sort(prob, axis=1)
    top1 = sorted_prob[:, -1]
    top2 = sorted_prob[:, -2] if logits.shape[1] > 1 else np.zeros_like(top1, dtype=np.float32)
    margin = np.nan_to_num(np.clip(top1 - top2, 0.0, 1.0), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    max_softmax = np.nan_to_num(np.clip(top1, 0.0, 1.0), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    disparity_bins = int(logits.shape[1])
    entropy = -(prob * np.log(prob + 1e-8)).sum(axis=1)
    max_entropy = max(float(np.log(disparity_bins)), 1e-8)
    entropy_confidence = 1.0 - (entropy / max_entropy)
    entropy_confidence = np.nan_to_num(entropy_confidence, nan=0.0, posinf=0.0, neginf=0.0)
    entropy_confidence = np.clip(entropy_confidence, 0.0, 1.0).astype(np.float32, copy=False)

    d_values = np.arange(disparity_bins, dtype=np.float32).reshape(1, disparity_bins, 1, 1)
    pred = (prob * d_values).sum(axis=1)
    pred2 = (prob * (d_values ** 2)).sum(axis=1)
    variance = np.maximum(pred2 - (pred ** 2), 0.0)
    inverse_variance = 1.0 / (variance + 1e-4)
    finite = np.isfinite(inverse_variance)
    variance_confidence = np.zeros(inverse_variance.shape, dtype=np.float32)
    if np.any(finite):
        finite_values = inverse_variance[finite]
        p05 = float(np.quantile(finite_values, 0.05))
        p95 = float(np.quantile(finite_values, 0.95))
        denom = max(p95 - p05, 1e-8)
        variance_confidence = ((inverse_variance - p05) / denom).astype(np.float32, copy=False)
    variance_confidence = np.nan_to_num(variance_confidence, nan=0.0, posinf=0.0, neginf=0.0)
    variance_confidence = np.clip(variance_confidence, 0.0, 1.0).astype(np.float32, copy=False)
    return {
        "margin": margin,
        "max_softmax": max_softmax,
        "entropy": entropy_confidence,
        "variance": variance_confidence,
    }


def resize_confidence_maps_to_shape(
    confidence_maps: np.ndarray,
    *,
    output_shape: tuple[int, int],
) -> np.ndarray:
    maps = np.asarray(confidence_maps, dtype=np.float32)
    if maps.ndim != 3:
        raise ValueError(f"Expected confidence maps shaped [B, H, W], got {maps.shape}.")

    target_h, target_w = [int(item) for item in output_shape]
    resized = [
        cv2.resize(np.asarray(confidence_map, dtype=np.float32), (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        for confidence_map in maps
    ]
    if not resized:
        return np.empty((0, target_h, target_w), dtype=np.float32)
    return np.stack(resized, axis=0).astype(np.float32, copy=False)


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


def _ensure_ffs_repo_on_sys_path(ffs_repo: Path) -> None:
    repo_path = str(ffs_repo)
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)


def _configure_tensorrt_runtime_search_paths(trt_root: Path | None) -> list[Any]:
    dll_handles: list[Any] = []
    if trt_root is None:
        return dll_handles
    if not trt_root.exists():
        raise FileNotFoundError(f"TensorRT runtime root not found: {trt_root}")

    search_paths = [trt_root / "lib", trt_root / "bin"]
    existing_search_paths = [path for path in search_paths if path.exists()]
    if existing_search_paths:
        os.environ["PATH"] = os.pathsep.join(
            [*(str(path) for path in existing_search_paths), os.environ.get("PATH", "")]
        )
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        for path in existing_search_paths:
            dll_handles.append(os.add_dll_directory(str(path)))
    return dll_handles


def load_tensorrt_model_config(
    model_dir: str | Path,
    *,
    model_path: str | Path | None = None,
) -> dict[str, Any]:
    import yaml

    model_dir = Path(model_dir).resolve()
    cfg_path = resolve_tensorrt_model_config_path(model_dir, model_path=model_path)
    with open(cfg_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    image_size = cfg.get("image_size")
    if not isinstance(image_size, (list, tuple)) or len(image_size) != 2:
        raise ValueError(f"Expected TensorRT config image_size=[H, W], got {image_size!r} in {cfg_path}")
    cfg["image_size"] = [int(image_size[0]), int(image_size[1])]
    return cfg


def resolve_tensorrt_model_config_path(
    model_dir: str | Path,
    *,
    model_path: str | Path | None = None,
) -> Path:
    model_dir = Path(model_dir).resolve()
    candidates: list[Path] = []
    if model_path is not None:
        model_path = Path(model_path)
        candidates.append(model_dir / f"{model_path.stem}.yaml")
    candidates.extend(
        [
            model_dir / "config.yaml",
            model_dir / "onnx.yaml",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "TensorRT metadata not found. Looked in: "
        + ", ".join(str(path) for path in candidates)
    )


def resolve_single_engine_tensorrt_model_path(model_dir: str | Path) -> Path:
    model_dir = Path(model_dir).resolve()
    engine_paths = sorted(path for path in model_dir.glob("*.engine") if path.is_file())
    if not engine_paths:
        raise FileNotFoundError(f"No TensorRT single-engine model found under {model_dir}.")
    if len(engine_paths) > 1:
        raise ValueError(
            "Expected exactly one TensorRT single-engine model under "
            f"{model_dir}, found {len(engine_paths)}: "
            + ", ".join(path.name for path in engine_paths)
        )
    return engine_paths[0]


def resolve_tensorrt_engine_static_batch_size(
    *,
    trt_mode: str,
    model_dir: str | Path,
    trt_root: str | Path | None = None,
) -> int:
    model_dir = Path(model_dir).resolve()
    trt_root_path = None if trt_root is None else Path(trt_root).resolve()
    dll_handles = _configure_tensorrt_runtime_search_paths(trt_root_path)
    del dll_handles

    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)

    def _load_engine(engine_path: Path):
        with open(engine_path, "rb") as handle:
            engine = trt.Runtime(logger).deserialize_cuda_engine(handle.read())
        if engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine from {engine_path}.")
        return engine

    if trt_mode == "two_stage":
        feature_engine = _load_engine(model_dir / "feature_runner.engine")
        post_engine = _load_engine(model_dir / "post_runner.engine")
        left_batch = int(feature_engine.get_tensor_shape("left")[0])
        right_batch = int(feature_engine.get_tensor_shape("right")[0])
        disp_batch = int(post_engine.get_tensor_shape("disp")[0])
        if left_batch != right_batch or left_batch != disp_batch:
            raise ValueError(
                "Two-stage TensorRT engine batch dimensions are inconsistent. "
                f"feature_left={left_batch} feature_right={right_batch} post_disp={disp_batch}"
            )
        return left_batch
    if trt_mode == "single_engine":
        model_path = resolve_single_engine_tensorrt_model_path(model_dir)
        engine = _load_engine(model_path)
        left_batch = int(engine.get_tensor_shape("left_image")[0])
        right_batch = int(engine.get_tensor_shape("right_image")[0])
        disp_batch = int(engine.get_tensor_shape("disparity")[0])
        if left_batch != right_batch or left_batch != disp_batch:
            raise ValueError(
                "Single-engine TensorRT engine batch dimensions are inconsistent. "
                f"left={left_batch} right={right_batch} disparity={disp_batch}"
            )
        return left_batch
    raise ValueError(f"Unsupported TensorRT mode for batch-size resolution: {trt_mode}")


def normalize_single_engine_tensorrt_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    return (image / 255.0 - IMAGENET_MEAN) / IMAGENET_STD


def select_tensorrt_disparity_output(outputs: dict[str, Any]) -> Any:
    if "disparity" in outputs:
        return outputs["disparity"]
    if "disp" in outputs:
        return outputs["disp"]
    if len(outputs) == 1:
        return next(iter(outputs.values()))
    raise ValueError(
        "Could not resolve TensorRT disparity output from outputs: "
        + ", ".join(sorted(outputs))
    )


def run_forward_on_non_default_cuda_stream(
    *,
    torch_module: Any,
    stream: Any,
    forward_fn: Callable[..., Any],
    **forward_kwargs: Any,
) -> Any:
    current_stream = torch_module.cuda.current_stream()
    stream.wait_stream(current_stream)
    with torch_module.cuda.stream(stream):
        output = forward_fn(**forward_kwargs)
    current_stream.wait_stream(stream)
    return output


def resolve_tensorrt_image_transform(
    *,
    input_height: int,
    input_width: int,
    engine_height: int,
    engine_width: int,
) -> dict[str, int | float | str]:
    input_height = int(input_height)
    input_width = int(input_width)
    engine_height = int(engine_height)
    engine_width = int(engine_width)
    if input_height == engine_height and input_width == engine_width:
        return {
            "mode": "match",
            "engine_height": engine_height,
            "engine_width": engine_width,
            "output_height": input_height,
            "output_width": input_width,
            "scale_x": 1.0,
            "scale_y": 1.0,
            "pad_top": 0,
            "pad_bottom": 0,
            "pad_left": 0,
            "pad_right": 0,
        }
    if input_height == 480 and input_width == 848 and engine_height == 480 and engine_width == 864:
        pad_total = engine_width - input_width
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        return {
            "mode": "pad",
            "engine_height": engine_height,
            "engine_width": engine_width,
            "output_height": input_height,
            "output_width": input_width,
            "scale_x": 1.0,
            "scale_y": 1.0,
            "pad_top": 0,
            "pad_bottom": 0,
            "pad_left": pad_left,
            "pad_right": pad_right,
        }
    return {
        "mode": "resize",
        "engine_height": engine_height,
        "engine_width": engine_width,
        "output_height": engine_height,
        "output_width": engine_width,
        "scale_x": float(engine_width / input_width),
        "scale_y": float(engine_height / input_height),
        "pad_top": 0,
        "pad_bottom": 0,
        "pad_left": 0,
        "pad_right": 0,
    }


def apply_tensorrt_image_transform(
    image: np.ndarray,
    *,
    transform: dict[str, int | float | str],
) -> np.ndarray:
    image = np.asarray(image)
    mode = str(transform["mode"])
    if mode == "match":
        return image
    if mode == "pad":
        pad_top = int(transform["pad_top"])
        pad_bottom = int(transform["pad_bottom"])
        pad_left = int(transform["pad_left"])
        pad_right = int(transform["pad_right"])
        pad_spec: list[tuple[int, int]] = [
            (pad_top, pad_bottom),
            (pad_left, pad_right),
        ]
        if image.ndim == 3:
            pad_spec.append((0, 0))
        return np.pad(image, tuple(pad_spec), mode="edge")
    if mode == "resize":
        return cv2.resize(
            image,
            dsize=(int(transform["engine_width"]), int(transform["engine_height"])),
            interpolation=cv2.INTER_LINEAR,
        )
    raise ValueError(f"Unsupported TensorRT image transform mode: {mode}")


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


def finalize_single_engine_tensorrt_output(
    disparity_raw: np.ndarray,
    *,
    transform: dict[str, int | float | str],
    K_ir_left: np.ndarray,
    baseline_m: float,
    valid_iters: int,
    max_disp: int,
    audit_mode: bool,
) -> dict[str, np.ndarray | float | list[list[float]]]:
    outputs = finalize_tensorrt_disparity_batch_outputs(
        disparity_raw,
        transform=transform,
        batch_samples=[
            {
                "K_ir_left": np.asarray(K_ir_left, dtype=np.float32),
                "baseline_m": float(baseline_m),
                "audit_mode": bool(audit_mode),
            }
        ],
        valid_iters=valid_iters,
        max_disp=max_disp,
    )
    return outputs[0]


def _load_official_tensorrt_foundation_stereo(*, ffs_repo: Path) -> Any:
    _ensure_ffs_repo_on_sys_path(ffs_repo)
    import core.foundation_stereo as foundation_stereo
    import core.submodule as submodule

    if getattr(submodule, "triton", None) is None:
        raise RuntimeError(
            "Official Fast-FoundationStereo two-stage TensorRT requires Triton for the "
            "intermediate GWC volume kernel. Install a compatible official FFS environment "
            "or use --ffs_trt_mode single_engine."
        )
    foundation_stereo.build_gwc_volume_triton = submodule.build_gwc_volume_triton
    return foundation_stereo


class FastFoundationStereoRunner:
    def __init__(
        self,
        *,
        ffs_repo: str | Path,
        model_path: str | Path,
        scale: float = 1.0,
        valid_iters: int = 8,
        max_disp: int = 192,
    ) -> None:
        self.ffs_repo = Path(ffs_repo).resolve()
        self.model_path = Path(model_path).resolve()
        self.scale = float(scale)
        self.valid_iters = int(valid_iters)
        self.max_disp = int(max_disp)

        if not self.ffs_repo.exists():
            raise FileNotFoundError(f"Fast-FoundationStereo repo not found: {self.ffs_repo}")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Fast-FoundationStereo model not found: {self.model_path}")

        import torch
        import yaml
        from omegaconf import OmegaConf

        _disable_torch_compile(torch)
        if not torch.cuda.is_available():
            raise RuntimeError("Fast-FoundationStereoRunner requires CUDA.")

        _ensure_ffs_repo_on_sys_path(self.ffs_repo)
        from core.utils.utils import InputPadder
        from Utils import AMP_DTYPE, set_logging_format, set_seed

        with open(self.model_path.parent / "cfg.yaml", "r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
        cfg.update(
            {
                "model_dir": str(self.model_path),
                "scale": self.scale,
                "valid_iters": self.valid_iters,
                "max_disp": self.max_disp,
            }
        )
        self.cfg = OmegaConf.create(cfg)
        self.torch = torch
        self.InputPadder = InputPadder
        self.AMP_DTYPE = AMP_DTYPE
        set_logging_format()
        set_seed(0)
        torch.autograd.set_grad_enabled(False)

        model = torch.load(str(self.model_path), map_location="cpu", weights_only=False)
        model.args.valid_iters = self.valid_iters
        model.args.max_disp = self.max_disp
        self.model = model.cuda().eval()

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        image = np.asarray(image)
        if image.ndim == 2:
            image = np.tile(image[..., None], (1, 1, 3))
        image = image[..., :3]
        if self.scale != 1.0:
            image = cv2.resize(image, dsize=None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LINEAR)
        return image

    def _forward_with_optional_classifier_logits(
        self,
        *,
        left_tensor: Any,
        right_tensor: Any,
        capture_classifier_logits: bool,
    ) -> tuple[Any, np.ndarray | None]:
        captured_logits: list[Any] = []
        hook_handle = None
        if capture_classifier_logits:
            def _capture_logits(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
                captured_logits.append(output.detach().float())

            hook_handle = self.model.classifier.register_forward_hook(_capture_logits)

        try:
            with self.torch.amp.autocast("cuda", enabled=True, dtype=self.AMP_DTYPE):
                disparity = self.model.forward(
                    left_tensor,
                    right_tensor,
                    iters=self.valid_iters,
                    test_mode=True,
                    optimize_build_volume="pytorch1",
                )
        finally:
            if hook_handle is not None:
                hook_handle.remove()

        if not capture_classifier_logits:
            return disparity, None
        if len(captured_logits) != 1:
            raise RuntimeError(
                "Expected exactly one classifier logits capture during PyTorch FFS forward. "
                f"Got {len(captured_logits)} captures."
            )
        return disparity, np.asarray(captured_logits[0].cpu().numpy(), dtype=np.float32)

    def run_batch(
        self,
        batch_samples: list[dict[str, Any]],
    ) -> list[dict[str, np.ndarray | float | list[list[float]]]]:
        if not batch_samples:
            raise ValueError("Expected at least one batch sample.")

        prepared_left: list[np.ndarray] = []
        prepared_right: list[np.ndarray] = []
        target_shape: tuple[int, int] | None = None
        for sample in batch_samples:
            left = self._prepare_image(sample["left_image"])
            right = self._prepare_image(sample["right_image"])
            if right.shape[:2] != left.shape[:2]:
                right = cv2.resize(right, dsize=(left.shape[1], left.shape[0]), interpolation=cv2.INTER_LINEAR)
            if target_shape is None:
                target_shape = (int(left.shape[0]), int(left.shape[1]))
            elif (int(left.shape[0]), int(left.shape[1])) != target_shape:
                raise ValueError(
                    "All PyTorch FFS batch samples must have the same preprocessed image shape. "
                    f"Got {(int(left.shape[0]), int(left.shape[1]))} vs {target_shape}."
                )
            prepared_left.append(left)
            prepared_right.append(right)

        torch = self.torch
        left_tensor = torch.stack(
            [torch.as_tensor(left).cuda().float().permute(2, 0, 1) for left in prepared_left],
            dim=0,
        )
        right_tensor = torch.stack(
            [torch.as_tensor(right).cuda().float().permute(2, 0, 1) for right in prepared_right],
            dim=0,
        )
        padder = self.InputPadder(left_tensor.shape, divis_by=32, force_square=False)
        left_tensor, right_tensor = padder.pad(left_tensor, right_tensor)

        disparity, _ = self._forward_with_optional_classifier_logits(
            left_tensor=left_tensor,
            right_tensor=right_tensor,
            capture_classifier_logits=False,
        )
        disparity = padder.unpad(disparity.float())
        disparity_maps = split_disparity_batch_output_maps(
            disparity.data.cpu().numpy(),
            expected_batch_size=len(batch_samples),
        )
        return [
            build_disparity_products(
                disparity_map,
                K_ir_left=np.asarray(sample["K_ir_left"], dtype=np.float32),
                baseline_m=float(sample["baseline_m"]),
                scale=self.scale,
                valid_iters=self.valid_iters,
                max_disp=self.max_disp,
                audit_mode=bool(sample.get("audit_mode", False)),
            )
            for sample, disparity_map in zip(batch_samples, disparity_maps)
        ]

    def run_pair(
        self,
        left_image: np.ndarray,
        right_image: np.ndarray,
        *,
        K_ir_left: np.ndarray,
        baseline_m: float,
        audit_mode: bool = False,
    ) -> dict[str, np.ndarray | float | list[list[float]]]:
        return self.run_batch(
            [
                {
                    "left_image": left_image,
                    "right_image": right_image,
                    "K_ir_left": K_ir_left,
                    "baseline_m": baseline_m,
                    "audit_mode": audit_mode,
                }
            ]
        )[0]

    def run_pair_with_confidence(
        self,
        left_image: np.ndarray,
        right_image: np.ndarray,
        *,
        K_ir_left: np.ndarray,
        baseline_m: float,
        audit_mode: bool = False,
    ) -> dict[str, np.ndarray | float | list[list[float]]]:
        left = self._prepare_image(left_image)
        right = self._prepare_image(right_image)
        if right.shape[:2] != left.shape[:2]:
            right = cv2.resize(right, dsize=(left.shape[1], left.shape[0]), interpolation=cv2.INTER_LINEAR)

        torch = self.torch
        left_tensor = torch.as_tensor(left).cuda().float()[None].permute(0, 3, 1, 2)
        right_tensor = torch.as_tensor(right).cuda().float()[None].permute(0, 3, 1, 2)
        padder = self.InputPadder(left_tensor.shape, divis_by=32, force_square=False)
        left_tensor, right_tensor = padder.pad(left_tensor, right_tensor)
        padded_output_shape = (int(left_tensor.shape[-2]), int(left_tensor.shape[-1]))

        disparity, logits = self._forward_with_optional_classifier_logits(
            left_tensor=left_tensor,
            right_tensor=right_tensor,
            capture_classifier_logits=True,
        )
        if logits is None:
            raise RuntimeError("PyTorch FFS confidence path expected classifier logits but none were captured.")
        confidence_coarse = compute_confidence_proxies_from_logits(logits)
        confidence_resized = {
            metric_name: resize_confidence_maps_to_shape(
                confidence_map,
                output_shape=padded_output_shape,
            )
            for metric_name, confidence_map in confidence_coarse.items()
        }

        disparity = padder.unpad(disparity.float())
        confidence_tensors = {
            metric_name: padder.unpad(torch.as_tensor(confidence_map[:, None, :, :]))
            for metric_name, confidence_map in confidence_resized.items()
        }
        disparity_map = split_disparity_batch_output_maps(disparity.data.cpu().numpy(), expected_batch_size=1)[0]

        result = build_disparity_products(
            disparity_map,
            K_ir_left=np.asarray(K_ir_left, dtype=np.float32),
            baseline_m=float(baseline_m),
            scale=self.scale,
            valid_iters=self.valid_iters,
            max_disp=self.max_disp,
            audit_mode=bool(audit_mode),
        )
        for metric_name, confidence_tensor in confidence_tensors.items():
            confidence_map = split_disparity_batch_output_maps(
                confidence_tensor.data.cpu().numpy(),
                expected_batch_size=1,
            )[0]
            result[f"confidence_{metric_name}_ir_left"] = np.clip(confidence_map, 0.0, 1.0).astype(np.float32)
        return result


class FastFoundationStereoTensorRTRunner:
    def __init__(
        self,
        *,
        ffs_repo: str | Path,
        model_dir: str | Path,
        trt_root: str | Path | None = None,
    ) -> None:
        self.ffs_repo = Path(ffs_repo).resolve()
        self.model_dir = Path(model_dir).resolve()
        self.trt_root = None if trt_root is None else Path(trt_root).resolve()
        self.feature_engine_path = self.model_dir / "feature_runner.engine"
        self.post_engine_path = self.model_dir / "post_runner.engine"

        if not self.ffs_repo.exists():
            raise FileNotFoundError(f"Fast-FoundationStereo repo not found: {self.ffs_repo}")
        for path in (self.feature_engine_path, self.post_engine_path):
            if not path.exists():
                raise FileNotFoundError(f"TensorRT engine not found: {path}")

        import torch
        from omegaconf import OmegaConf

        _disable_torch_compile(torch)
        if not torch.cuda.is_available():
            raise RuntimeError("FastFoundationStereoTensorRTRunner requires CUDA.")

        self._dll_handles = _configure_tensorrt_runtime_search_paths(self.trt_root)
        foundation_stereo = _load_official_tensorrt_foundation_stereo(ffs_repo=self.ffs_repo)
        from Utils import set_logging_format, set_seed

        cfg_dict = load_tensorrt_model_config(self.model_dir)
        self.cfg = OmegaConf.create(cfg_dict)
        self.engine_height = int(self.cfg.image_size[0])
        self.engine_width = int(self.cfg.image_size[1])
        self.valid_iters = int(self.cfg.valid_iters)
        self.max_disp = int(self.cfg.max_disp)
        self.torch = torch
        self.inference_stream = torch.cuda.Stream()
        set_logging_format()
        set_seed(0)
        torch.autograd.set_grad_enabled(False)
        self.model = foundation_stereo.TrtRunner(
            self.cfg,
            str(self.feature_engine_path),
            str(self.post_engine_path),
        )

    def _prepare_image(self, image: np.ndarray) -> tuple[np.ndarray, dict[str, int | float | str]]:
        image = np.asarray(image)
        if image.ndim == 2:
            image = np.tile(image[..., None], (1, 1, 3))
        image = image[..., :3]
        transform = resolve_tensorrt_image_transform(
            input_height=int(image.shape[0]),
            input_width=int(image.shape[1]),
            engine_height=self.engine_height,
            engine_width=self.engine_width,
        )
        image = apply_tensorrt_image_transform(image, transform=transform)
        return image, transform

    def run_batch(
        self,
        batch_samples: list[dict[str, Any]],
    ) -> list[dict[str, np.ndarray | float | list[list[float]]]]:
        if not batch_samples:
            raise ValueError("Expected at least one batch sample.")

        prepared_left: list[np.ndarray] = []
        prepared_right: list[np.ndarray] = []
        batch_transform: dict[str, int | float | str] | None = None
        for sample in batch_samples:
            left, left_transform = self._prepare_image(sample["left_image"])
            right, right_transform = self._prepare_image(sample["right_image"])
            if left_transform != right_transform:
                raise ValueError(
                    "Left/right TensorRT preprocessing transforms must match. "
                    f"Got {left_transform!r} vs {right_transform!r}."
                )
            if batch_transform is None:
                batch_transform = left_transform
            elif left_transform != batch_transform:
                raise ValueError(
                    "All two-stage TensorRT batch samples must share the same preprocessing transform. "
                    f"Got {left_transform!r} vs {batch_transform!r}."
                )
            prepared_left.append(left)
            prepared_right.append(right)

        torch = self.torch
        left_tensor = torch.stack(
            [torch.as_tensor(left).cuda().float().permute(2, 0, 1) for left in prepared_left],
            dim=0,
        )
        right_tensor = torch.stack(
            [torch.as_tensor(right).cuda().float().permute(2, 0, 1) for right in prepared_right],
            dim=0,
        )
        disparity = run_forward_on_non_default_cuda_stream(
            torch_module=torch,
            stream=self.inference_stream,
            forward_fn=self.model.forward,
            image1=left_tensor,
            image2=right_tensor,
        )
        return finalize_tensorrt_disparity_batch_outputs(
            disparity.data.cpu().numpy(),
            transform=batch_transform or resolve_tensorrt_image_transform(
                input_height=self.engine_height,
                input_width=self.engine_width,
                engine_height=self.engine_height,
                engine_width=self.engine_width,
            ),
            batch_samples=batch_samples,
            valid_iters=self.valid_iters,
            max_disp=self.max_disp,
        )

    def run_pair(
        self,
        left_image: np.ndarray,
        right_image: np.ndarray,
        *,
        K_ir_left: np.ndarray,
        baseline_m: float,
        audit_mode: bool = False,
    ) -> dict[str, np.ndarray | float | list[list[float]]]:
        return self.run_batch(
            [
                {
                    "left_image": left_image,
                    "right_image": right_image,
                    "K_ir_left": K_ir_left,
                    "baseline_m": baseline_m,
                    "audit_mode": audit_mode,
                }
            ]
        )[0]


class _SingleEngineTensorRTRuntime:
    def __init__(self, *, engine_path: Path) -> None:
        import tensorrt as trt
        import torch

        self.trt = trt
        self.torch = torch
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as handle:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(handle.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine from {engine_path}.")
        self.context = self.engine.create_execution_context()

    def _trt_to_torch_dtype(self, tensor_dtype: Any) -> Any:
        trt = self.trt
        mapping = {
            trt.DataType.FLOAT: self.torch.float32,
            trt.DataType.HALF: self.torch.float16,
            trt.DataType.BF16: self.torch.bfloat16,
            trt.DataType.INT32: self.torch.int32,
            trt.DataType.INT8: self.torch.int8,
            trt.DataType.BOOL: self.torch.bool,
        }
        if tensor_dtype not in mapping:
            raise RuntimeError(f"Unsupported TensorRT dtype: {tensor_dtype}")
        return mapping[tensor_dtype]

    def __call__(self, *, left_image: Any, right_image: Any) -> dict[str, Any]:
        trt = self.trt
        input_names = [
            self.engine.get_tensor_name(idx)
            for idx in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(idx)) == trt.TensorIOMode.INPUT
        ]
        if len(input_names) != 2:
            raise RuntimeError(f"Expected exactly two TensorRT inputs, got {input_names!r}.")
        lower_input_names = {name.lower(): name for name in input_names}
        left_input_name = next(
            (name for key, name in lower_input_names.items() if "left" in key),
            input_names[0],
        )
        right_input_name = next(
            (
                name
                for key, name in lower_input_names.items()
                if "right" in key and name != left_input_name
            ),
            input_names[1] if input_names[1] != left_input_name else input_names[0],
        )
        if left_input_name == right_input_name:
            raise RuntimeError(f"Could not disambiguate TensorRT input names: {input_names!r}.")
        inputs = {
            left_input_name: left_image,
            right_input_name: right_image,
        }
        for name, tensor in list(inputs.items()):
            expected_dtype = self._trt_to_torch_dtype(self.engine.get_tensor_dtype(name))
            if tensor.dtype != expected_dtype:
                tensor = tensor.to(expected_dtype)
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            inputs[name] = tensor
            self.context.set_input_shape(name, tuple(tensor.shape))

        outputs: dict[str, Any] = {}
        for idx in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(idx)
            if self.engine.get_tensor_mode(name) != trt.TensorIOMode.OUTPUT:
                continue
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = self._trt_to_torch_dtype(self.engine.get_tensor_dtype(name))
            outputs[name] = self.torch.empty(shape, device="cuda", dtype=dtype)

        for name, tensor in inputs.items():
            self.context.set_tensor_address(name, int(tensor.data_ptr()))
        for name, tensor in outputs.items():
            self.context.set_tensor_address(name, int(tensor.data_ptr()))

        stream = self.torch.cuda.current_stream().cuda_stream
        ok = self.context.execute_async_v3(stream)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 returned failure.")
        return outputs


class FastFoundationStereoSingleEngineTensorRTRunner:
    def __init__(
        self,
        *,
        ffs_repo: str | Path,
        model_dir: str | Path,
        trt_root: str | Path | None = None,
    ) -> None:
        self.ffs_repo = Path(ffs_repo).resolve()
        self.model_dir = Path(model_dir).resolve()
        self.trt_root = None if trt_root is None else Path(trt_root).resolve()
        self.model_path = resolve_single_engine_tensorrt_model_path(self.model_dir)

        if not self.ffs_repo.exists():
            raise FileNotFoundError(f"Fast-FoundationStereo repo not found: {self.ffs_repo}")

        import torch
        from omegaconf import OmegaConf

        _disable_torch_compile(torch)
        if not torch.cuda.is_available():
            raise RuntimeError("FastFoundationStereoSingleEngineTensorRTRunner requires CUDA.")

        _ensure_ffs_repo_on_sys_path(self.ffs_repo)
        self._dll_handles = _configure_tensorrt_runtime_search_paths(self.trt_root)
        from Utils import set_logging_format, set_seed

        cfg_dict = load_tensorrt_model_config(self.model_dir, model_path=self.model_path)
        self.cfg = OmegaConf.create(cfg_dict)
        self.engine_height = int(self.cfg.image_size[0])
        self.engine_width = int(self.cfg.image_size[1])
        self.valid_iters = int(self.cfg.valid_iters)
        self.max_disp = int(self.cfg.max_disp)
        self.torch = torch
        self.inference_stream = torch.cuda.Stream()
        set_logging_format()
        set_seed(0)
        torch.autograd.set_grad_enabled(False)
        self.model = _SingleEngineTensorRTRuntime(engine_path=self.model_path)

    def _prepare_image(self, image: np.ndarray) -> tuple[np.ndarray, dict[str, int | float | str]]:
        image = np.asarray(image)
        if image.ndim == 2:
            image = np.tile(image[..., None], (1, 1, 3))
        image = image[..., :3]
        transform = resolve_tensorrt_image_transform(
            input_height=int(image.shape[0]),
            input_width=int(image.shape[1]),
            engine_height=self.engine_height,
            engine_width=self.engine_width,
        )
        image = apply_tensorrt_image_transform(image, transform=transform)
        image = normalize_single_engine_tensorrt_image(image)
        return image, transform

    def run_batch(
        self,
        batch_samples: list[dict[str, Any]],
    ) -> list[dict[str, np.ndarray | float | list[list[float]]]]:
        if not batch_samples:
            raise ValueError("Expected at least one batch sample.")

        prepared_left: list[np.ndarray] = []
        prepared_right: list[np.ndarray] = []
        batch_transform: dict[str, int | float | str] | None = None
        for sample in batch_samples:
            left, left_transform = self._prepare_image(sample["left_image"])
            right, right_transform = self._prepare_image(sample["right_image"])
            if left_transform != right_transform:
                raise ValueError(
                    "Left/right TensorRT preprocessing transforms must match. "
                    f"Got {left_transform!r} vs {right_transform!r}."
                )
            if batch_transform is None:
                batch_transform = left_transform
            elif left_transform != batch_transform:
                raise ValueError(
                    "All single-engine TensorRT batch samples must share the same preprocessing transform. "
                    f"Got {left_transform!r} vs {batch_transform!r}."
                )
            prepared_left.append(left)
            prepared_right.append(right)

        torch = self.torch
        left_tensor = torch.stack(
            [torch.as_tensor(left).cuda().float().permute(2, 0, 1) for left in prepared_left],
            dim=0,
        )
        right_tensor = torch.stack(
            [torch.as_tensor(right).cuda().float().permute(2, 0, 1) for right in prepared_right],
            dim=0,
        )
        outputs = run_forward_on_non_default_cuda_stream(
            torch_module=torch,
            stream=self.inference_stream,
            forward_fn=self.model,
            left_image=left_tensor,
            right_image=right_tensor,
        )
        disparity_output = select_tensorrt_disparity_output(outputs)
        return finalize_tensorrt_disparity_batch_outputs(
            disparity_output.data.cpu().numpy(),
            transform=batch_transform or resolve_tensorrt_image_transform(
                input_height=self.engine_height,
                input_width=self.engine_width,
                engine_height=self.engine_height,
                engine_width=self.engine_width,
            ),
            batch_samples=batch_samples,
            valid_iters=self.valid_iters,
            max_disp=self.max_disp,
        )

    def run_pair(
        self,
        left_image: np.ndarray,
        right_image: np.ndarray,
        *,
        K_ir_left: np.ndarray,
        baseline_m: float,
        audit_mode: bool = False,
    ) -> dict[str, np.ndarray | float | list[list[float]]]:
        return self.run_batch(
            [
                {
                    "left_image": left_image,
                    "right_image": right_image,
                    "K_ir_left": K_ir_left,
                    "baseline_m": baseline_m,
                    "audit_mode": audit_mode,
                }
            ]
        )[0]
