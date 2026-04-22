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
    disparity_raw = np.asarray(disparity_raw, dtype=np.float32)
    if disparity_raw.ndim == 4:
        if disparity_raw.shape[0] != 1:
            raise ValueError(f"Expected batch size 1 disparity output, got shape={disparity_raw.shape}.")
        disparity_raw = disparity_raw.reshape(disparity_raw.shape[-2], disparity_raw.shape[-1])
    elif disparity_raw.ndim == 3:
        if disparity_raw.shape[0] != 1:
            raise ValueError(f"Expected single-channel disparity output, got shape={disparity_raw.shape}.")
        disparity_raw = disparity_raw.reshape(disparity_raw.shape[-2], disparity_raw.shape[-1])
    elif disparity_raw.ndim != 2:
        raise ValueError(f"Expected 2D/3D/4D disparity output, got shape={disparity_raw.shape}.")

    disparity_raw = undo_tensorrt_disparity_transform(disparity_raw, transform=transform)
    scale_x = float(transform["scale_x"])
    scale_y = float(transform["scale_y"])
    uniform_scale = scale_x if abs(scale_x - scale_y) <= 1e-6 else 1.0
    return build_disparity_products(
        disparity_raw,
        K_ir_left=K_ir_left,
        baseline_m=baseline_m,
        scale=uniform_scale,
        scale_x=scale_x,
        scale_y=scale_y,
        valid_iters=valid_iters,
        max_disp=max_disp,
        audit_mode=audit_mode,
    )


def _patch_tensorrt_triton_cost_volume(*, ffs_repo: Path) -> Any:
    _ensure_ffs_repo_on_sys_path(ffs_repo)
    import core.foundation_stereo as foundation_stereo
    import core.submodule as submodule

    replacement = submodule.build_gwc_volume_optimized_pytorch1
    submodule.build_gwc_volume_triton = replacement
    foundation_stereo.build_gwc_volume_triton = replacement
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

    def run_pair(
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

        with torch.amp.autocast("cuda", enabled=True, dtype=self.AMP_DTYPE):
            # This is the external Fast-FoundationStereo network forward pass.
            # Input: this camera's left/right stereo images. Output: disparity.
            disparity = self.model.forward(
                left_tensor,
                right_tensor,
                iters=self.valid_iters,
                test_mode=True,
                optimize_build_volume="pytorch1",
            )

        disparity = padder.unpad(disparity.float())
        disparity_raw = disparity.data.cpu().numpy().reshape(left.shape[0], left.shape[1]).astype(np.float32)
        return build_disparity_products(
            disparity_raw,
            K_ir_left=K_ir_left,
            baseline_m=baseline_m,
            scale=self.scale,
            valid_iters=self.valid_iters,
            max_disp=self.max_disp,
            audit_mode=audit_mode,
        )


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
        foundation_stereo = _patch_tensorrt_triton_cost_volume(ffs_repo=self.ffs_repo)
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

    def run_pair(
        self,
        left_image: np.ndarray,
        right_image: np.ndarray,
        *,
        K_ir_left: np.ndarray,
        baseline_m: float,
        audit_mode: bool = False,
    ) -> dict[str, np.ndarray | float | list[list[float]]]:
        left, left_transform = self._prepare_image(left_image)
        right, right_transform = self._prepare_image(right_image)
        if left_transform != right_transform:
            raise ValueError(
                "Left/right TensorRT preprocessing transforms must match. "
                f"Got {left_transform!r} vs {right_transform!r}."
            )

        torch = self.torch
        left_tensor = torch.as_tensor(left).cuda().float()[None].permute(0, 3, 1, 2)
        right_tensor = torch.as_tensor(right).cuda().float()[None].permute(0, 3, 1, 2)
        disparity = run_forward_on_non_default_cuda_stream(
            torch_module=torch,
            stream=self.inference_stream,
            forward_fn=self.model.forward,
            image1=left_tensor,
            image2=right_tensor,
        )
        disparity_raw = disparity.data.cpu().numpy().reshape(self.engine_height, self.engine_width).astype(np.float32)
        disparity_raw = undo_tensorrt_disparity_transform(disparity_raw, transform=left_transform)
        scale_x = float(left_transform["scale_x"])
        scale_y = float(left_transform["scale_y"])
        uniform_scale = scale_x if abs(scale_x - scale_y) <= 1e-6 else 1.0
        return build_disparity_products(
            disparity_raw,
            K_ir_left=K_ir_left,
            baseline_m=baseline_m,
            scale=uniform_scale,
            scale_x=scale_x,
            scale_y=scale_y,
            valid_iters=self.valid_iters,
            max_disp=self.max_disp,
            audit_mode=audit_mode,
        )


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

    def run_pair(
        self,
        left_image: np.ndarray,
        right_image: np.ndarray,
        *,
        K_ir_left: np.ndarray,
        baseline_m: float,
        audit_mode: bool = False,
    ) -> dict[str, np.ndarray | float | list[list[float]]]:
        left, left_transform = self._prepare_image(left_image)
        right, right_transform = self._prepare_image(right_image)
        if left_transform != right_transform:
            raise ValueError(
                "Left/right TensorRT preprocessing transforms must match. "
                f"Got {left_transform!r} vs {right_transform!r}."
            )

        torch = self.torch
        left_tensor = torch.as_tensor(left).cuda().float()[None].permute(0, 3, 1, 2)
        right_tensor = torch.as_tensor(right).cuda().float()[None].permute(0, 3, 1, 2)
        outputs = run_forward_on_non_default_cuda_stream(
            torch_module=torch,
            stream=self.inference_stream,
            forward_fn=self.model,
            left_image=left_tensor,
            right_image=right_tensor,
        )
        disparity_output = select_tensorrt_disparity_output(outputs)
        disparity_raw = disparity_output.data.cpu().numpy()
        return finalize_single_engine_tensorrt_output(
            disparity_raw,
            transform=left_transform,
            K_ir_left=K_ir_left,
            baseline_m=baseline_m,
            valid_iters=self.valid_iters,
            max_disp=self.max_disp,
            audit_mode=audit_mode,
        )
