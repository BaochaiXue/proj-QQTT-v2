"""Two-stage Fast-FoundationStereo TensorRT runner (the only live FFS runner).

Extracted verbatim from ``fast_foundation_stereo.py`` (behavior-preserving split).
This is the runner constructed by ``main_data_processing.py`` for the ``ir-ffs``
depth backend. ``torch`` / ``tensorrt`` / ``omegaconf`` / FoundationStereo remain
lazy in-method imports resolved from the ``--ffs-repo`` checkout at runtime.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from demo_v6_2.utils.ffs_disparity_products import finalize_tensorrt_disparity_batch_outputs
from demo_v6_2.utils.ffs_foundation_loader import (
    _disable_torch_compile,
    _load_official_tensorrt_foundation_stereo,
)
from demo_v6_2.utils.ffs_tensorrt_infra import (
    FFS_INPUT_STAGING_MODES,
    FFS_INPUT_STAGING_PAGEABLE,
    FFS_INPUT_STAGING_PINNED,
    _CachedTensorRTRun,
    _PinnedBatchPairImageInputBuffers,
    _PinnedSinglePairImageInputBuffers,
    _configure_tensorrt_runtime_search_paths,
    apply_tensorrt_image_transform,
    load_tensorrt_model_config,
    resolve_tensorrt_engine_static_batch_size,
    resolve_tensorrt_image_transform,
)


class FastFoundationStereoTensorRTRunner:
    def __init__(
        self,
        *,
        ffs_repo: str | Path,
        model_dir: str | Path,
        trt_root: str | Path | None = None,
        input_staging: str = FFS_INPUT_STAGING_PINNED,
    ) -> None:
        self.ffs_repo = Path(ffs_repo).resolve()
        self.model_dir = Path(model_dir).resolve()
        self.trt_root = None if trt_root is None else Path(trt_root).resolve()
        self.input_staging = str(input_staging)
        if self.input_staging not in FFS_INPUT_STAGING_MODES:
            raise ValueError(
                f"Unsupported FFS TensorRT input staging mode: {self.input_staging!r}. "
                f"Expected one of {FFS_INPUT_STAGING_MODES!r}."
            )
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
        self.static_batch_size = resolve_tensorrt_engine_static_batch_size(
            trt_mode="two_stage",
            model_dir=self.model_dir,
            trt_root=self.trt_root,
        )
        foundation_stereo = _load_official_tensorrt_foundation_stereo(
            ffs_repo=self.ffs_repo,
            batch_safe_gwc_volume=int(self.static_batch_size) > 1,
        )
        import tensorrt as trt
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
        self._input_buffers: _PinnedSinglePairImageInputBuffers | None = None
        self._batch_input_buffers: _PinnedBatchPairImageInputBuffers | None = None
        self._disparity_host_buffer: Any | None = None
        self._last_h2d_profile: dict[str, float | str | bool] = {
            "input_staging": self.input_staging,
            "stage_ms": 0.0,
            "h2d_enqueue_ms": 0.0,
            "h2d_wait_ms": 0.0,
            "pin_memory": self.input_staging == FFS_INPUT_STAGING_PINNED,
        }
        self._cached_trt_run = _CachedTensorRTRun(
            torch_module=torch,
            trt_module=trt,
            trt_runner=self.model,
        )
        self.model.run_trt = self._cached_trt_run.run_trt

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

    def _build_input_tensors(
        self,
        prepared_left: list[np.ndarray],
        prepared_right: list[np.ndarray],
    ) -> tuple[Any, Any]:
        torch = self.torch
        if (
            self.input_staging == FFS_INPUT_STAGING_PINNED
            and len(prepared_left) == 1
            and len(prepared_right) == 1
            and prepared_left[0].ndim == 3
            and prepared_right[0].ndim == 3
            and prepared_left[0].shape == prepared_right[0].shape
            and prepared_left[0].dtype == np.uint8
            and prepared_right[0].dtype == np.uint8
        ):
            image_shape = tuple(int(item) for item in prepared_left[0].shape)
            if self._input_buffers is None or self._input_buffers.image_shape != image_shape:
                self._input_buffers = _PinnedSinglePairImageInputBuffers(
                    torch_module=torch,
                    image_shape=image_shape,
                )
            left_tensor, right_tensor = self._input_buffers.load(prepared_left[0], prepared_right[0])
            self._last_h2d_profile = dict(self._input_buffers.last_profile)
            return left_tensor, right_tensor
        if (
            self.input_staging == FFS_INPUT_STAGING_PINNED
            and len(prepared_left) == len(prepared_right)
            and len(prepared_left) > 1
            and all(left.ndim == 3 for left in prepared_left)
            and all(right.ndim == 3 for right in prepared_right)
            and all(left.shape == prepared_left[0].shape for left in prepared_left)
            and all(right.shape == prepared_left[0].shape for right in prepared_right)
            and all(left.dtype == np.uint8 for left in prepared_left)
            and all(right.dtype == np.uint8 for right in prepared_right)
        ):
            image_shape = tuple(int(item) for item in prepared_left[0].shape)
            batch_size = int(len(prepared_left))
            if (
                self._batch_input_buffers is None
                or self._batch_input_buffers.batch_size != batch_size
                or self._batch_input_buffers.image_shape != image_shape
            ):
                self._batch_input_buffers = _PinnedBatchPairImageInputBuffers(
                    torch_module=torch,
                    batch_size=batch_size,
                    image_shape=image_shape,
                )
            left_tensor, right_tensor = self._batch_input_buffers.load(prepared_left, prepared_right)
            self._last_h2d_profile = dict(self._batch_input_buffers.last_profile)
            return left_tensor, right_tensor

        h2d_start_s = time.perf_counter()
        left_tensor = torch.stack(
            [torch.as_tensor(left).cuda().float().permute(2, 0, 1) for left in prepared_left],
            dim=0,
        )
        right_tensor = torch.stack(
            [torch.as_tensor(right).cuda().float().permute(2, 0, 1) for right in prepared_right],
            dim=0,
        )
        h2d_enqueue_ms = (time.perf_counter() - h2d_start_s) * 1000.0
        self._last_h2d_profile = {
            "input_staging": FFS_INPUT_STAGING_PAGEABLE,
            "stage_ms": 0.0,
            "h2d_enqueue_ms": float(h2d_enqueue_ms),
            "h2d_wait_ms": 0.0,
            "pin_memory": False,
        }
        return left_tensor, right_tensor

    def _copy_disparity_to_numpy(
        self,
        disparity: Any,
        *,
        stable_copy: bool,
        sync_stream: Any | None = None,
    ) -> np.ndarray:
        tensor = disparity.detach()
        if not getattr(tensor, "is_cuda", False):
            array = tensor.cpu().numpy()
            return array.copy() if stable_copy else array
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        shape = tuple(int(item) for item in tensor.shape)
        if (
            self._disparity_host_buffer is None
            or tuple(int(item) for item in self._disparity_host_buffer.shape) != shape
            or self._disparity_host_buffer.dtype != tensor.dtype
        ):
            self._disparity_host_buffer = self.torch.empty(
                shape,
                dtype=tensor.dtype,
                pin_memory=True,
            )
        self._disparity_host_buffer.copy_(tensor, non_blocking=True)
        (sync_stream or self.torch.cuda.current_stream()).synchronize()
        array = self._disparity_host_buffer.numpy()
        return array.copy() if stable_copy else array

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

        stable_copy = any(bool(sample.get("audit_mode", False)) for sample in batch_samples)
        with self.torch.cuda.stream(self.inference_stream):
            left_tensor, right_tensor = self._build_input_tensors(prepared_left, prepared_right)
            disparity = self.model.forward(image1=left_tensor, image2=right_tensor)
            disparity_raw = self._copy_disparity_to_numpy(
                disparity,
                stable_copy=stable_copy,
                sync_stream=self.inference_stream,
            )
        outputs = finalize_tensorrt_disparity_batch_outputs(
            disparity_raw,
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
        h2d_profile = dict(self._last_h2d_profile)
        for output in outputs:
            output["h2d_profile"] = h2d_profile
        return outputs

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
