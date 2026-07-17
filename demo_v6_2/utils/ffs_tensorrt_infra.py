"""TensorRT runtime plumbing, image transforms, and input staging buffers.

Extracted verbatim from ``fast_foundation_stereo.py`` (behavior-preserving split).
The heavy ``tensorrt`` / ``yaml`` dependencies remain lazy in-method imports so
importing this module never pulls in the TensorRT stack; only the two-stage FFS
runner path exercises them at runtime.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


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


FFS_INPUT_STAGING_PINNED = "pinned"
FFS_INPUT_STAGING_PAGEABLE = "pageable"
FFS_INPUT_STAGING_MODES = (FFS_INPUT_STAGING_PINNED, FFS_INPUT_STAGING_PAGEABLE)


class _PinnedBatchPairImageInputBuffers:
    def __init__(self, *, torch_module: Any, batch_size: int, image_shape: tuple[int, int, int]) -> None:
        self.torch = torch_module
        self.batch_size = int(batch_size)
        self.image_shape = tuple(int(item) for item in image_shape)
        if self.batch_size <= 0:
            raise ValueError(f"Expected positive batch size, got {batch_size}.")
        if len(self.image_shape) != 3 or self.image_shape[2] != 3:
            raise ValueError(f"Expected HxWx3 image shape, got {self.image_shape!r}.")
        height, width, channels = self.image_shape
        host_shape = (self.batch_size, height, width, channels)
        device_shape = (self.batch_size, channels, height, width)
        self.left_host = torch_module.empty(host_shape, dtype=torch_module.uint8, pin_memory=True)
        self.right_host = torch_module.empty(host_shape, dtype=torch_module.uint8, pin_memory=True)
        self.left_device = torch_module.empty(device_shape, device="cuda", dtype=torch_module.float32)
        self.right_device = torch_module.empty(device_shape, device="cuda", dtype=torch_module.float32)
        self.last_profile: dict[str, float | str | bool] = {
            "input_staging": FFS_INPUT_STAGING_PINNED,
            "stage_ms": 0.0,
            "h2d_enqueue_ms": 0.0,
            "h2d_wait_ms": 0.0,
            "pin_memory": True,
        }

    def load(self, left_images: list[np.ndarray], right_images: list[np.ndarray]) -> tuple[Any, Any]:
        if len(left_images) != self.batch_size or len(right_images) != self.batch_size:
            raise ValueError(
                "Pinned batch TensorRT input buffer batch mismatch. "
                f"expected={self.batch_size} left={len(left_images)} right={len(right_images)}"
            )
        torch = self.torch
        stage_start_s = time.perf_counter()
        for idx, (left_image, right_image) in enumerate(zip(left_images, right_images)):
            left = np.ascontiguousarray(left_image)
            right = np.ascontiguousarray(right_image)
            if tuple(left.shape) != self.image_shape or tuple(right.shape) != self.image_shape:
                raise ValueError(
                    "Pinned batch TensorRT input buffer shape mismatch. "
                    f"expected={self.image_shape!r} left={left.shape!r} right={right.shape!r}"
                )
            if left.dtype != np.uint8 or right.dtype != np.uint8:
                raise ValueError(f"Expected uint8 TensorRT inputs, got {left.dtype!r} and {right.dtype!r}.")
            self.left_host[idx].copy_(torch.as_tensor(left, dtype=torch.uint8))
            self.right_host[idx].copy_(torch.as_tensor(right, dtype=torch.uint8))
        stage_ms = (time.perf_counter() - stage_start_s) * 1000.0
        h2d_start_s = time.perf_counter()
        self.left_device.copy_(self.left_host.permute(0, 3, 1, 2), non_blocking=True)
        self.right_device.copy_(self.right_host.permute(0, 3, 1, 2), non_blocking=True)
        h2d_enqueue_ms = (time.perf_counter() - h2d_start_s) * 1000.0
        self.last_profile = {
            "input_staging": FFS_INPUT_STAGING_PINNED,
            "stage_ms": float(stage_ms),
            "h2d_enqueue_ms": float(h2d_enqueue_ms),
            "h2d_wait_ms": 0.0,
            "pin_memory": True,
        }
        return self.left_device, self.right_device


class _CachedTensorRTRun:
    def __init__(self, *, torch_module: Any, trt_module: Any, trt_runner: Any) -> None:
        self.torch = torch_module
        self.trt = trt_module
        self.trt_runner = trt_runner
        self._outputs: dict[tuple[int, str, tuple[int, ...], Any], Any] = {}
        self._input_shapes: dict[tuple[int, str], tuple[int, ...]] = {}
        self._tensor_addresses: dict[tuple[int, str], int] = {}

    def _set_input_shape_if_needed(self, context: Any, name: str, shape: tuple[int, ...]) -> bool:
        key = (id(context), name)
        if self._input_shapes.get(key) != shape:
            context.set_input_shape(name, shape)
            self._input_shapes[key] = shape
            return True
        return False

    def _set_tensor_address_if_needed(self, context: Any, name: str, tensor: Any, *, force: bool = False) -> None:
        address = int(tensor.data_ptr())
        key = (id(context), name)
        if force or self._tensor_addresses.get(key) != address:
            context.set_tensor_address(name, address)
            self._tensor_addresses[key] = address

    def _cached_output_tensor(self, engine: Any, context: Any, name: str) -> Any:
        shape = tuple(int(item) for item in context.get_tensor_shape(name))
        dtype = self.trt_runner.trt_dtype_to_torch(engine.get_tensor_dtype(name))
        key = (id(context), name, shape, dtype)
        tensor = self._outputs.get(key)
        if tensor is None:
            tensor = self.torch.empty(shape, device="cuda", dtype=dtype)
            self._outputs[key] = tensor
        return tensor

    def run_trt(self, engine: Any, context: Any, inputs_by_name: dict[str, Any]) -> dict[str, Any]:
        prepared_inputs: dict[str, Any] = {}
        shape_changed = False
        for name, tensor in inputs_by_name.items():
            expected_dtype = self.trt_runner.trt_dtype_to_torch(engine.get_tensor_dtype(name))
            if tensor.dtype != expected_dtype:
                tensor = tensor.to(expected_dtype)
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            prepared_inputs[name] = tensor
            shape_changed = (
                self._set_input_shape_if_needed(context, name, tuple(int(item) for item in tensor.shape))
                or shape_changed
            )

        output_names = self.trt_runner.get_io_tensor_names(engine, self.trt.TensorIOMode.OUTPUT)
        outputs = {
            name: self._cached_output_tensor(engine, context, name)
            for name in output_names
        }

        for name, tensor in prepared_inputs.items():
            self._set_tensor_address_if_needed(context, name, tensor, force=shape_changed)
        for name, tensor in outputs.items():
            self._set_tensor_address_if_needed(context, name, tensor, force=shape_changed)

        stream = self.torch.cuda.current_stream().cuda_stream
        ok = context.execute_async_v3(stream)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 returned failure.")
        return dict(outputs)


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
