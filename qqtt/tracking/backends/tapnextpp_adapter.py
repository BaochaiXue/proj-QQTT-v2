from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
import os
import sys
import time
from typing import Any

import numpy as np

from qqtt.tracking.base import BackendAvailability, BackendUnavailableError, TrackingResult
from qqtt.tracking.backends.point_tracker_adapter import (
    TRACKER_BACKEND_TAPNEXTPP,
    PointTrackerBackendSpec,
)


INSTALL_HINT = "Run scripts/env/install_tapnextpp_demo_3_1_max.sh"


class TAPNextPPAdapter:
    """Stateful online TAPNext++ adapter for Demo 3.1 point tracking."""

    name = TRACKER_BACKEND_TAPNEXTPP
    spec = PointTrackerBackendSpec(
        name=TRACKER_BACKEND_TAPNEXTPP,
        family="tapnext",
        supports_batch_views=True,
        supports_online=True,
        supports_prewarm=True,
        query_format="yx",
        batch_support_status="true_online_batch_views",
    )

    def __init__(
        self,
        *,
        device: str = "cuda",
        camera_idx: int | None = None,
        repo_dir: str | None = None,
        checkpoint: str | None = None,
        image_size: tuple[int, int] | Sequence[int] | str = (256, 256),
        autocast_dtype: str = "fp16",
        use_certainty: bool = False,
        certainty_radius: int = 8,
        certainty_threshold: float = 0.5,
        compile_model: bool = False,
        reset_on_reinitialize: bool = True,
        fast_postprocess: bool = True,
    ) -> None:
        self.device = str(device)
        self.camera_idx = camera_idx
        self.repo_dir = self._expand_path(
            repo_dir
            or os.environ.get("QQTT_TAPNET_REPO_DIR")
            or os.environ.get("TAPNET_REPO_DIR")
        )
        self.checkpoint = self._expand_path(
            checkpoint
            or os.environ.get("QQTT_TAPNEXTPP_CHECKPOINT")
            or os.environ.get("TAPNEXTPP_CHECKPOINT")
        )
        self.image_size = self._normalize_image_size(image_size)
        self.autocast_dtype = self._normalize_autocast_dtype(autocast_dtype)
        self.use_certainty = bool(use_certainty)
        self.certainty_radius = self._normalize_positive_int(certainty_radius, "tapnextpp_certainty_radius")
        self.certainty_threshold = float(certainty_threshold)
        self.compile_model = bool(compile_model)
        self.reset_on_reinitialize = bool(reset_on_reinitialize)
        self.fast_postprocess = bool(fast_postprocess)
        self.frame_value_range = "minus1_1_float"
        self._model: Any | None = None
        self._tracker_certainty: Any | None = None
        self._model_load_ms = 0.0
        self._model_load_missing: list[str] = []
        self._model_load_unexpected: list[str] = []
        self._query_points_yx: np.ndarray | None = None
        self._query_points_tyx: Any | None = None
        self._tracking_state: Any | None = None
        self._original_frame_shape_hw: tuple[int, int] | None = None
        self._frames_seen = 0
        self._batch_camera_ids: tuple[int, ...] = ()
        self._batch_query_points_yx_by_camera: dict[int, np.ndarray] = {}
        self._batch_query_points_tyx: Any | None = None
        self._batch_original_frame_shape_hw_by_camera: dict[int, tuple[int, int]] = {}
        self._batch_tracking_state: Any | None = None
        self._batch_frames_seen = 0
        self._batch_model_call_groups = 0

    @staticmethod
    def _expand_path(value: str | None) -> str | None:
        return str(Path(value).expanduser()) if value else None

    @staticmethod
    def _normalize_positive_int(value: int, name: str) -> int:
        parsed = int(value)
        if parsed <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
        return parsed

    @staticmethod
    def _normalize_image_size(value: tuple[int, int] | Sequence[int] | str) -> tuple[int, int]:
        if isinstance(value, str):
            raw = value.strip().lower().replace("x", ",")
            parts = [part.strip() for part in raw.split(",") if part.strip()]
            if len(parts) == 1:
                height = width = int(parts[0])
            elif len(parts) == 2:
                height, width = int(parts[0]), int(parts[1])
            else:
                raise ValueError("--tapnextpp-image-size must be HxW, H,W, or a single square size.")
        else:
            parts = tuple(int(item) for item in value)
            if len(parts) == 1:
                height = width = parts[0]
            elif len(parts) == 2:
                height, width = parts
            else:
                raise ValueError("--tapnextpp-image-size must contain one or two integers.")
        if height <= 0 or width <= 0:
            raise ValueError("--tapnextpp-image-size dimensions must be positive.")
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError("--tapnextpp-image-size dimensions must be multiples of 8.")
        return (int(height), int(width))

    @staticmethod
    def _normalize_autocast_dtype(value: str) -> str:
        normalized = str(value).strip().lower()
        aliases = {"float16": "fp16", "half": "fp16", "float32": "fp32", "bfloat16": "bf16"}
        normalized = aliases.get(normalized, normalized)
        if normalized not in {"fp16", "bf16", "fp32"}:
            raise ValueError("--tapnextpp-autocast-dtype must be one of {'fp16', 'bf16', 'fp32'}.")
        return normalized

    @staticmethod
    def _resolve_repo_sys_path(repo_dir: str | None) -> Path | None:
        if not repo_dir:
            return None
        path = Path(repo_dir).expanduser()
        if (path / "tapnet" / "tapnext" / "tapnext_torch.py").is_file():
            return path
        if (path / "tapnext" / "tapnext_torch.py").is_file() and path.name == "tapnet":
            return path.parent
        return path

    def _prepend_repo_dir(self) -> Path | None:
        repo = self._resolve_repo_sys_path(self.repo_dir)
        if repo is None:
            return None
        path = str(repo)
        if path not in sys.path:
            sys.path.insert(0, path)
        return repo

    def availability(self) -> BackendAvailability:
        if self._model is not None:
            return BackendAvailability(self.name, True, "Injected TAPNext++ model is available")
        missing: list[str] = []
        repo = self._resolve_repo_sys_path(self.repo_dir)
        if repo is None:
            missing.append("--tapnet-repo-dir")
        elif not (repo / "tapnet" / "tapnext" / "tapnext_torch.py").is_file():
            missing.append(
                f"--tapnet-repo-dir {self.repo_dir!r} must point to external/tapnet "
                "or a path containing the tapnet package"
            )
        if not self.checkpoint:
            missing.append("--tapnextpp-checkpoint")
        elif not Path(self.checkpoint).is_file():
            missing.append(f"--tapnextpp-checkpoint {self.checkpoint!r} does not exist")
        if missing:
            return BackendAvailability(self.name, False, "; ".join(missing) + f". {INSTALL_HINT}.")
        self._prepend_repo_dir()
        try:
            import torch  # noqa: F401
            from tapnet.tapnext.tapnext_torch import TAPNext  # noqa: F401
            from tapnet.tapnext.tapnext_torch_utils import tracker_certainty  # noqa: F401
        except Exception as exc:
            return BackendAvailability(
                self.name,
                False,
                f"TAPNext++ runtime import failed: {type(exc).__name__}: {exc}. {INSTALL_HINT}.",
            )
        return BackendAvailability(self.name, True, "TAPNext++ import and checkpoint path are available")

    def is_available(self) -> bool:
        return self.availability().available

    def availability_reason(self) -> str:
        return self.availability().reason

    def is_initialized(self) -> bool:
        return self._query_points_yx is not None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        availability = self.availability()
        if not availability.available:
            raise BackendUnavailableError(availability.reason)
        import torch
        from tapnet.tapnext.tapnext_torch import TAPNext
        from tapnet.tapnext.tapnext_torch_utils import tracker_certainty

        assert self.checkpoint is not None
        started_s = time.perf_counter()
        model = TAPNext(image_size=tuple(self.image_size))
        try:
            ckpt = torch.load(str(self.checkpoint), map_location="cpu")
        except TypeError:
            ckpt = torch.load(str(self.checkpoint), map_location="cpu", weights_only=False)
        raw_state = ckpt["state_dict"] if isinstance(ckpt, Mapping) and "state_dict" in ckpt else ckpt
        state = {str(k).replace("tapnext.", ""): v for k, v in raw_state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        self._model_load_missing = [str(item) for item in missing]
        self._model_load_unexpected = [str(item) for item in unexpected]
        model.to(self.device).eval()
        if self.compile_model:
            model = torch.compile(model)
        self._sync_cuda_if_needed()
        self._model_load_ms = float((time.perf_counter() - started_s) * 1000.0)
        self._tracker_certainty = tracker_certainty
        self._model = model
        return model

    def _sync_cuda_if_needed(self) -> None:
        if not str(self.device).startswith("cuda"):
            return
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize(torch.device(self.device))
        except Exception:
            return

    def _autocast_context(self):
        if not str(self.device).startswith("cuda") or self.autocast_dtype == "fp32":
            return nullcontext()
        import torch

        dtype = torch.float16 if self.autocast_dtype == "fp16" else torch.bfloat16
        return torch.autocast(device_type="cuda", dtype=dtype, enabled=True)

    def warmup(self) -> dict[str, Any]:
        started_s = time.perf_counter()
        self._load_model()
        return {
            "model_load_ms": float(self._model_load_ms),
            "total_ms": float((time.perf_counter() - started_s) * 1000.0),
            "tracker_backend": self.name,
            "adapter": type(self).__name__,
            "image_size": [int(self.image_size[0]), int(self.image_size[1])],
            "tapnextpp_image_size": [int(self.image_size[0]), int(self.image_size[1])],
            "autocast_dtype": self.autocast_dtype,
            "tapnextpp_autocast_dtype": self.autocast_dtype,
            "tapnextpp_use_certainty": bool(self.use_certainty),
            "tapnextpp_compile": bool(self.compile_model),
            "tapnextpp_fast_postprocess": bool(self.fast_postprocess),
            "tapnextpp_frame_value_range": self.frame_value_range,
            "batch_support_status": self.spec.batch_support_status,
            "state_dict_missing": list(self._model_load_missing),
            "state_dict_unexpected": list(self._model_load_unexpected),
        }

    @staticmethod
    def _validate_query_points(query_points_yx: np.ndarray, *, camera_idx: int | None = None) -> np.ndarray:
        points = np.asarray(query_points_yx, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 2:
            prefix = "query_points_yx" if camera_idx is None else f"query_points_yx for camera {camera_idx}"
            raise ValueError(f"{prefix} must have shape (N,2); got {points.shape}")
        if len(points) == 0:
            raise ValueError("TAPNext++ requires at least one query point.")
        return np.ascontiguousarray(points)

    @staticmethod
    def _validate_frame(frame: np.ndarray, *, camera_idx: int | None = None) -> np.ndarray:
        arr = np.asarray(frame, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            prefix = "frame" if camera_idx is None else f"frame for camera {camera_idx}"
            raise ValueError(f"TAPNext++ {prefix} must be HxWx3 RGB uint8; got {arr.shape}")
        return np.ascontiguousarray(arr)

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        detach = getattr(value, "detach", None)
        if callable(detach):
            value = detach()
        cpu = getattr(value, "cpu", None)
        if callable(cpu):
            value = cpu()
        numpy = getattr(value, "numpy", None)
        if callable(numpy):
            return np.asarray(numpy())
        return np.asarray(value)

    @staticmethod
    def _shape_list(value: Any) -> list[int]:
        shape = getattr(value, "shape", None)
        if shape is None:
            return list(np.asarray(value).shape)
        return [int(item) for item in tuple(shape)]

    @staticmethod
    def _dtype_name(value: Any) -> str:
        dtype = getattr(value, "dtype", None)
        return "" if dtype is None else str(dtype)

    @staticmethod
    def _device_name(value: Any) -> str:
        device = getattr(value, "device", None)
        return "" if device is None else str(device)

    @staticmethod
    def _numel(value: Any) -> int:
        numel = getattr(value, "numel", None)
        if callable(numel):
            return int(numel())
        return int(np.asarray(value).size)

    @classmethod
    def _to_numpy_timed(cls, value: Any) -> tuple[np.ndarray, float]:
        started_s = time.perf_counter()
        array = cls._to_numpy(value)
        return array, float((time.perf_counter() - started_s) * 1000.0)

    @staticmethod
    def _is_torch_tensor(value: Any) -> bool:
        return callable(getattr(value, "detach", None)) and hasattr(value, "device") and hasattr(value, "shape")

    @staticmethod
    def _torch_to_numpy_copy_timed(value: Any) -> tuple[np.ndarray, float, float]:
        tensor = value.detach()
        started_s = time.perf_counter()
        cpu_tensor = tensor.cpu()
        to_cpu_ms = float((time.perf_counter() - started_s) * 1000.0)
        started_s = time.perf_counter()
        array = np.asarray(cpu_tensor.numpy())
        numpy_ms = float((time.perf_counter() - started_s) * 1000.0)
        return array, to_cpu_ms, numpy_ms

    def _frames_to_video_tensor(self, frames: Sequence[np.ndarray], *, camera_ids: Sequence[int] | None = None):
        import torch
        import torch.nn.functional as F

        arrays: list[np.ndarray] = []
        for idx, frame in enumerate(frames):
            camera_idx = None if camera_ids is None else int(camera_ids[idx])
            arrays.append(self._validate_frame(frame, camera_idx=camera_idx))
        if not arrays:
            raise ValueError("TAPNext++ update requires at least one frame.")
        shape = arrays[0].shape
        for idx, arr in enumerate(arrays):
            if arr.shape != shape:
                camera_text = "" if camera_ids is None else f" for camera {int(camera_ids[idx])}"
                raise ValueError(
                    "TAPNext++ batch-views requires equal frame shapes; "
                    f"frame 0 has {shape}, frame {idx}{camera_text} has {arr.shape}"
                )
        stacked = np.stack(arrays, axis=0)
        tensor = torch.from_numpy(stacked).to(device=self.device, dtype=torch.float32)
        tensor = tensor.mul(2.0 / 255.0).sub(1.0)
        source_h, source_w = int(shape[0]), int(shape[1])
        target_h, target_w = self.image_size
        if (source_h, source_w) != (target_h, target_w):
            tensor = F.interpolate(
                tensor.permute(0, 3, 1, 2).contiguous(),
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1).contiguous()
        return tensor[:, None], (source_h, source_w)

    def _queries_yx_to_tyx_tensor(self, query_points_yx: np.ndarray, *, source_shape_hw: tuple[int, int]):
        import torch

        points = np.asarray(query_points_yx, dtype=np.float32).reshape(-1, 2)
        source_h, source_w = int(source_shape_hw[0]), int(source_shape_hw[1])
        target_h, target_w = self.image_size
        y_scale = float(target_h) / float(max(source_h, 1))
        x_scale = float(target_w) / float(max(source_w, 1))
        y_scaled = points[:, 0:1] * np.float32(y_scale)
        x_scaled = points[:, 1:2] * np.float32(x_scale)
        t = np.zeros((len(points), 1), dtype=np.float32)
        tyx = np.ascontiguousarray(np.concatenate([t, y_scaled, x_scaled], axis=1), dtype=np.float32)
        return torch.from_numpy(tyx).to(device=self.device, dtype=torch.float32)

    def _run_model(self, *, video: Any, query_points: Any | None, state: Any | None) -> tuple[Any, float]:
        model = self._load_model()
        self._sync_cuda_if_needed()
        started_s = time.perf_counter()
        import torch

        with torch.no_grad(), self._autocast_context():
            if state is None:
                output = model(video=video, query_points=query_points)
            else:
                output = model(video=video, state=state)
        self._sync_cuda_if_needed()
        return output, float((time.perf_counter() - started_s) * 1000.0)

    @staticmethod
    def _extract_output(output: Any) -> tuple[Any, Any, Any, Any]:
        if isinstance(output, Mapping):
            tracks = output.get("tracks", output.get("pred_tracks"))
            track_logits = output.get("track_logits")
            visible_logits = output.get("visible_logits")
            state = output.get("tracking_state", output.get("state"))
            if tracks is None or visible_logits is None or state is None:
                raise ValueError("TAPNext++ output mapping must contain tracks, visible_logits, and state.")
            return tracks, track_logits, visible_logits, state
        if isinstance(output, tuple) and len(output) >= 4:
            return output[0], output[1], output[2], output[3]
        raise ValueError("TAPNext++ model output must be a 4-tuple or compatible mapping.")

    @staticmethod
    def _normalize_tracks_time_shape(
        tracks_xy: np.ndarray,
        *,
        batch_size: int,
        query_count: int,
        time_steps: int,
    ) -> np.ndarray:
        tracks = np.asarray(tracks_xy, dtype=np.float32)
        if tracks.ndim == 3 and tracks.shape == (batch_size, query_count, 2):
            tracks = tracks[:, None, :, :]
        elif tracks.ndim == 4 and tracks.shape == (batch_size, query_count, time_steps, 2):
            tracks = np.transpose(tracks, (0, 2, 1, 3))
        expected = (int(batch_size), int(time_steps), int(query_count), 2)
        if tracks.shape != expected:
            raise ValueError(f"TAPNext++ tracks shape {tracks.shape} does not match expected {expected}")
        return np.ascontiguousarray(tracks, dtype=np.float32)

    @staticmethod
    def _normalize_visibility_time_shape(
        visible_logits: np.ndarray,
        *,
        batch_size: int,
        query_count: int,
        time_steps: int,
    ) -> np.ndarray:
        visible = np.asarray(visible_logits, dtype=np.float32)
        if visible.ndim == 4 and visible.shape[-1] == 1:
            visible = visible[..., 0]
        if visible.ndim == 2 and visible.shape == (batch_size, query_count):
            visible = visible[:, None, :]
        elif visible.ndim == 3 and visible.shape == (batch_size, query_count, time_steps):
            visible = np.transpose(visible, (0, 2, 1))
        expected = (int(batch_size), int(time_steps), int(query_count))
        if visible.shape != expected:
            raise ValueError(f"TAPNext++ visible logits shape {visible.shape} does not match expected {expected}")
        return (visible > 0.0).astype(np.float32)

    def _scaled_tracks_yx_to_original_yx(
        self,
        tracks_yx_b_t_n: np.ndarray,
        *,
        source_shapes_hw: Sequence[tuple[int, int]],
    ) -> np.ndarray:
        tracks_yx_scaled = np.asarray(tracks_yx_b_t_n, dtype=np.float32)
        batch_size = tracks_yx_scaled.shape[0]
        if len(source_shapes_hw) != batch_size:
            raise ValueError("source_shapes_hw length must match TAPNext++ batch size.")
        target_h, target_w = self.image_size
        tracks_yx = np.empty_like(tracks_yx_scaled, dtype=np.float32)
        for batch_idx, (source_h, source_w) in enumerate(source_shapes_hw):
            y_scale = float(max(int(source_h), 1)) / float(max(target_h, 1))
            x_scale = float(max(int(source_w), 1)) / float(max(target_w, 1))
            tracks_yx[batch_idx, ..., 0] = tracks_yx_scaled[batch_idx, ..., 0] * np.float32(y_scale)
            tracks_yx[batch_idx, ..., 1] = tracks_yx_scaled[batch_idx, ..., 1] * np.float32(x_scale)
        return np.ascontiguousarray(tracks_yx, dtype=np.float32)

    def _scale_latest_yx_to_original_yx(
        self,
        tracks_yx_b_n: np.ndarray,
        *,
        source_shapes_hw: Sequence[tuple[int, int]],
    ) -> np.ndarray:
        tracks_yx_scaled = np.asarray(tracks_yx_b_n, dtype=np.float32)
        if tracks_yx_scaled.ndim != 3 or tracks_yx_scaled.shape[-1] != 2:
            raise ValueError(f"TAPNext++ latest tracks must have shape (B,N,2); got {tracks_yx_scaled.shape}")
        batch_size = int(tracks_yx_scaled.shape[0])
        if len(source_shapes_hw) != batch_size:
            raise ValueError("source_shapes_hw length must match TAPNext++ batch size.")
        target_h, target_w = self.image_size
        scale = np.asarray(
            [
                [
                    float(max(int(source_h), 1)) / float(max(target_h, 1)),
                    float(max(int(source_w), 1)) / float(max(target_w, 1)),
                ]
                for source_h, source_w in source_shapes_hw
            ],
            dtype=np.float32,
        )
        tracks_yx = tracks_yx_scaled * scale[:, None, :]
        return np.ascontiguousarray(tracks_yx[:, None, :, :], dtype=np.float32)

    @staticmethod
    def _latest_tracks_tensor(value: Any, *, batch_size: int, query_count: int) -> Any:
        shape = tuple(int(item) for item in value.shape)
        if len(shape) == 3 and shape == (batch_size, query_count, 2):
            return value
        if len(shape) == 4:
            if shape[0] != batch_size or shape[-1] != 2:
                raise ValueError(f"TAPNext++ tracks shape {shape} does not match batch/query expectations")
            if shape[2] == query_count:
                return value[:, -1, :, :]
            if shape[1] == query_count:
                return value[:, :, -1, :]
        raise ValueError(f"TAPNext++ tracks shape {shape} does not contain a recognizable latest frame")

    @staticmethod
    def _latest_visibility_tensor(value: Any, *, batch_size: int, query_count: int) -> Any:
        visible = value
        shape = tuple(int(item) for item in visible.shape)
        if len(shape) == 4 and shape[-1] == 1:
            visible = visible[..., 0]
            shape = tuple(int(item) for item in visible.shape)
        if len(shape) == 2 and shape == (batch_size, query_count):
            return visible
        if len(shape) == 3:
            if shape[0] != batch_size:
                raise ValueError(f"TAPNext++ visible logits shape {shape} does not match batch size {batch_size}")
            if shape[2] == query_count:
                return visible[:, -1, :]
            if shape[1] == query_count:
                return visible[:, :, -1]
        raise ValueError(f"TAPNext++ visible logits shape {shape} does not contain a recognizable latest frame")

    def _parse_output_fast_profiled(
        self,
        tracks_raw: Any,
        visible_raw: Any,
        state: Any,
        profile: dict[str, Any],
        *,
        batch_size: int,
        query_count: int,
        source_shapes_hw: Sequence[tuple[int, int]],
    ) -> tuple[np.ndarray, np.ndarray, Any, dict[str, Any]]:
        if not self._is_torch_tensor(tracks_raw) or not self._is_torch_tensor(visible_raw):
            profile["fast_postprocess_fallback"] = "non_torch_output"
            return self._parse_output_slow_profiled(
                tracks_raw,
                visible_raw,
                state,
                profile,
                batch_size=batch_size,
                query_count=query_count,
                source_shapes_hw=source_shapes_hw,
            )

        import torch

        started_s = time.perf_counter()
        tracks_latest = self._latest_tracks_tensor(
            tracks_raw,
            batch_size=int(batch_size),
            query_count=int(query_count),
        ).to(dtype=torch.float32).contiguous()
        visible_latest = self._latest_visibility_tensor(
            visible_raw,
            batch_size=int(batch_size),
            query_count=int(query_count),
        )
        visible_latest = (visible_latest > 0.0).contiguous()
        profile["slice_latest_on_gpu_ms"] = float((time.perf_counter() - started_s) * 1000.0)
        profile["tracks_latest_shape"] = self._shape_list(tracks_latest)
        profile["visible_latest_shape"] = self._shape_list(visible_latest)

        started_s = time.perf_counter()
        self._sync_cuda_if_needed()
        profile["gpu_wait_before_cpu_copy_ms"] = float((time.perf_counter() - started_s) * 1000.0)

        tracks_np, tracks_to_cpu_ms, tracks_numpy_ms = self._torch_to_numpy_copy_timed(tracks_latest)
        profile["tracks_to_cpu_ms"] = float(tracks_to_cpu_ms)
        profile["tracks_cpu_bytes"] = int(np.asarray(tracks_np).nbytes)
        visible_np, visibility_to_cpu_ms, visible_numpy_ms = self._torch_to_numpy_copy_timed(visible_latest)
        profile["visibility_to_cpu_ms"] = float(visibility_to_cpu_ms)
        profile["visible_cpu_bytes"] = int(np.asarray(visible_np).nbytes)
        profile["visibility_cpu_bytes"] = int(np.asarray(visible_np).nbytes)
        profile["numpy_conversion_ms"] = float(tracks_numpy_ms + visible_numpy_ms)
        profile["tracks_normalize_shape_ms"] = 0.0
        profile["visibility_normalize_shape_ms"] = 0.0
        profile["normalize_shape_ms"] = 0.0

        started_s = time.perf_counter()
        tracks_yx_scaled = np.asarray(tracks_np, dtype=np.float32)
        profile["xy_to_yx_ms"] = float((time.perf_counter() - started_s) * 1000.0)
        started_s = time.perf_counter()
        tracks_yx = self._scale_latest_yx_to_original_yx(
            tracks_yx_scaled,
            source_shapes_hw=source_shapes_hw,
        )
        profile["scale_xy_to_original_ms"] = float((time.perf_counter() - started_s) * 1000.0)
        profile["scale_to_original_ms"] = float(profile["scale_xy_to_original_ms"])
        visibility = np.ascontiguousarray(np.asarray(visible_np, dtype=np.float32)[:, None, :], dtype=np.float32)
        profile["postprocess_cpu_bytes"] = int(profile["tracks_cpu_bytes"] + profile["visibility_cpu_bytes"])
        return tracks_yx, visibility, state, profile

    def _parse_output_slow_profiled(
        self,
        tracks_raw: Any,
        visible_raw: Any,
        state: Any,
        profile: dict[str, Any],
        *,
        batch_size: int,
        query_count: int,
        source_shapes_hw: Sequence[tuple[int, int]],
    ) -> tuple[np.ndarray, np.ndarray, Any, dict[str, Any]]:
        profile.setdefault("slice_latest_on_gpu_ms", 0.0)
        profile.setdefault("gpu_wait_before_cpu_copy_ms", 0.0)
        profile.setdefault("xy_to_yx_ms", 0.0)
        tracks_np, tracks_to_cpu_ms = self._to_numpy_timed(tracks_raw)
        profile["tracks_to_cpu_ms"] = float(tracks_to_cpu_ms)
        profile["tracks_cpu_bytes"] = int(np.asarray(tracks_np).nbytes)
        profile["tracks_latest_shape"] = []
        started_s = time.perf_counter()
        tracks_yx_scaled = self._normalize_tracks_time_shape(
            tracks_np,
            batch_size=batch_size,
            query_count=query_count,
            time_steps=1,
        )
        profile["tracks_normalize_shape_ms"] = float((time.perf_counter() - started_s) * 1000.0)

        visible_np, visibility_to_cpu_ms = self._to_numpy_timed(visible_raw)
        profile["visibility_to_cpu_ms"] = float(visibility_to_cpu_ms)
        profile["visible_cpu_bytes"] = int(np.asarray(visible_np).nbytes)
        profile["visibility_cpu_bytes"] = int(np.asarray(visible_np).nbytes)
        profile["visible_latest_shape"] = []
        profile["numpy_conversion_ms"] = 0.0
        started_s = time.perf_counter()
        visibility = self._normalize_visibility_time_shape(
            visible_np,
            batch_size=batch_size,
            query_count=query_count,
            time_steps=1,
        )
        profile["visibility_normalize_shape_ms"] = float((time.perf_counter() - started_s) * 1000.0)
        profile["normalize_shape_ms"] = float(
            profile["tracks_normalize_shape_ms"] + profile["visibility_normalize_shape_ms"]
        )
        started_s = time.perf_counter()
        tracks_yx = self._scaled_tracks_yx_to_original_yx(
            tracks_yx_scaled,
            source_shapes_hw=source_shapes_hw,
        )
        profile["scale_to_original_ms"] = float((time.perf_counter() - started_s) * 1000.0)
        profile["scale_xy_to_original_ms"] = float(profile["scale_to_original_ms"])
        profile["postprocess_cpu_bytes"] = int(profile["tracks_cpu_bytes"] + profile["visibility_cpu_bytes"])
        return tracks_yx, visibility, state, profile

    def _parse_output(
        self,
        output: Any,
        *,
        batch_size: int,
        query_count: int,
        source_shapes_hw: Sequence[tuple[int, int]],
    ) -> tuple[np.ndarray, np.ndarray, Any]:
        tracks_yx, visibility, state, _profile = self._parse_output_profiled(
            output,
            batch_size=batch_size,
            query_count=query_count,
            source_shapes_hw=source_shapes_hw,
        )
        return tracks_yx, visibility, state

    def _parse_output_profiled(
        self,
        output: Any,
        *,
        batch_size: int,
        query_count: int,
        source_shapes_hw: Sequence[tuple[int, int]],
    ) -> tuple[np.ndarray, np.ndarray, Any, dict[str, Any]]:
        total_started_s = time.perf_counter()
        profile: dict[str, Any] = {}
        started_s = time.perf_counter()
        tracks_raw, _track_logits, visible_raw, state = self._extract_output(output)
        profile["output_extract_ms"] = float((time.perf_counter() - started_s) * 1000.0)
        started_s = time.perf_counter()
        profile["tracks_raw_shape"] = self._shape_list(tracks_raw)
        profile["visible_raw_shape"] = self._shape_list(visible_raw)
        profile["tracks_raw_dtype"] = self._dtype_name(tracks_raw)
        profile["visible_raw_dtype"] = self._dtype_name(visible_raw)
        profile["tracks_raw_device"] = self._device_name(tracks_raw)
        profile["visible_raw_device"] = self._device_name(visible_raw)
        profile["tracks_raw_numel"] = int(self._numel(tracks_raw))
        profile["visible_raw_numel"] = int(self._numel(visible_raw))
        profile["output_shape_inspect_ms"] = float((time.perf_counter() - started_s) * 1000.0)
        profile["fast_postprocess"] = bool(self.fast_postprocess)
        if self.fast_postprocess:
            tracks_yx, visibility, state, profile = self._parse_output_fast_profiled(
                tracks_raw,
                visible_raw,
                state,
                profile,
                batch_size=batch_size,
                query_count=query_count,
                source_shapes_hw=source_shapes_hw,
            )
        else:
            tracks_yx, visibility, state, profile = self._parse_output_slow_profiled(
                tracks_raw,
                visible_raw,
                state,
                profile,
                batch_size=batch_size,
                query_count=query_count,
                source_shapes_hw=source_shapes_hw,
            )
        profile["total_postprocess_ms"] = float((time.perf_counter() - total_started_s) * 1000.0)
        return tracks_yx, visibility, state, profile

    def initialize(
        self,
        frames: Sequence[np.ndarray],
        query_points_yx: np.ndarray,
        masks: Sequence[np.ndarray] | None = None,
    ) -> None:
        _ = masks
        self._load_model()
        if self.reset_on_reinitialize:
            self._tracking_state = None
            self._query_points_tyx = None
            self._original_frame_shape_hw = None
        self._query_points_yx = self._validate_query_points(query_points_yx, camera_idx=self.camera_idx)
        self._frames_seen = 0
        for frame in frames:
            self.update(frame)

    def initialize_camera(self, camera_idx: int, query_points_yx: np.ndarray) -> None:
        self.camera_idx = int(camera_idx)
        self.initialize([], query_points_yx)

    def update(self, frame: np.ndarray) -> TrackingResult:
        if self._query_points_yx is None:
            raise RuntimeError("Call initialize(..., query_points_yx=...) before update().")
        wall_started_s = time.perf_counter()
        preprocess_started_s = wall_started_s
        video, source_shape = self._frames_to_video_tensor([frame])
        if self._original_frame_shape_hw is None:
            self._original_frame_shape_hw = source_shape
            self._query_points_tyx = self._queries_yx_to_tyx_tensor(
                self._query_points_yx,
                source_shape_hw=source_shape,
            )[None]
        elif self._original_frame_shape_hw != source_shape:
            raise ValueError(
                "TAPNext++ serial stream frame shape changed after initialization; "
                f"expected {self._original_frame_shape_hw}, got {source_shape}"
            )
        preprocess_ms = float((time.perf_counter() - preprocess_started_s) * 1000.0)
        first_update = self._tracking_state is None
        output, run_ms = self._run_model(
            video=video,
            query_points=self._query_points_tyx if first_update else None,
            state=None if first_update else self._tracking_state,
        )
        postprocess_started_s = time.perf_counter()
        tracks_yx_b, visibility_b, self._tracking_state, postprocess_profile = self._parse_output_profiled(
            output,
            batch_size=1,
            query_count=len(self._query_points_yx),
            source_shapes_hw=[self._original_frame_shape_hw],
        )
        postprocess_ms = float((time.perf_counter() - postprocess_started_s) * 1000.0)
        wall_ms = float((time.perf_counter() - wall_started_s) * 1000.0)
        frame_idx = int(self._frames_seen)
        self._frames_seen += 1
        result_pack_started_s = time.perf_counter()
        stats = {
            "backend": self.name,
            "tracker_backend": self.name,
            "adapter": type(self).__name__,
            "mode": "tapnextpp_online_serial",
            "stream_status": "published",
            "update_mode": "serial",
            "chunk_start_idx": frame_idx,
            "chunk_end_idx": frame_idx,
            "frames_seen": int(self._frames_seen),
            "num_query_points": int(len(self._query_points_yx)),
            "model_run_ms": float(run_ms),
            "cuda_event_ms": float(run_ms),
            "preprocess_ms": float(preprocess_ms),
            "postprocess_ms": float(postprocess_ms),
            "wall_ms": float(wall_ms),
            "fps_model_only": float(1000.0 / run_ms) if run_ms > 0.0 else 0.0,
            "image_size": [int(self.image_size[0]), int(self.image_size[1])],
            "tapnextpp_image_size": [int(self.image_size[0]), int(self.image_size[1])],
            "autocast_dtype": self.autocast_dtype,
            "tapnextpp_autocast_dtype": self.autocast_dtype,
            "tapnextpp_frame_value_range": self.frame_value_range,
            "tapnextpp_fast_postprocess": bool(self.fast_postprocess),
            "tapnextpp_state_active": self._tracking_state is not None,
            "tapnextpp_model_calls": 1,
            "device": self.device,
        }
        stats.update(postprocess_profile)
        result = TrackingResult(
            tracks_yx=tracks_yx_b[0],
            visibility=visibility_b[0],
            backend=self.name,
            camera_idx=self.camera_idx,
            query_points_yx=self._query_points_yx,
            stats=stats,
        )
        result_pack_ms = float((time.perf_counter() - result_pack_started_s) * 1000.0)
        result.stats["result_pack_ms"] = float(result_pack_ms)
        result.stats["postprocess_with_pack_ms"] = float(postprocess_ms + result_pack_ms)
        result.stats["wall_with_pack_ms"] = float(wall_ms + result_pack_ms)
        return result

    def initialize_batch(self, query_points_yx_by_camera: Mapping[int, np.ndarray]) -> None:
        self._load_model()
        query_points: dict[int, np.ndarray] = {}
        for camera_idx, points in sorted(query_points_yx_by_camera.items()):
            query_points[int(camera_idx)] = self._validate_query_points(points, camera_idx=int(camera_idx))
        if not query_points:
            raise ValueError("TAPNext++ batch-views requires at least one camera.")
        counts = {int(camera_idx): int(len(points)) for camera_idx, points in query_points.items()}
        if len(set(counts.values())) != 1:
            raise ValueError(
                "TAPNext++ batch-views requires equal query counts per camera; "
                f"got {counts}. Use --tracker-batch-query-count-policy min-common."
            )
        self._batch_camera_ids = tuple(sorted(query_points))
        self._batch_query_points_yx_by_camera = query_points
        self._batch_query_points_tyx = None
        self._batch_original_frame_shape_hw_by_camera = {}
        self._batch_tracking_state = None
        self._batch_frames_seen = 0
        self._batch_model_call_groups = 0

    def update_batch(self, frames_by_camera: Mapping[int, np.ndarray]) -> dict[int, TrackingResult]:
        if not self._batch_camera_ids:
            raise RuntimeError("Call initialize_batch(...) before update_batch().")
        wall_started_s = time.perf_counter()
        preprocess_started_s = wall_started_s
        expected = set(int(item) for item in self._batch_camera_ids)
        received = set(int(item) for item in frames_by_camera)
        missing = sorted(expected - received)
        extra = sorted(received - expected)
        if missing or extra:
            raise ValueError(f"TAPNext++ batch-views camera set mismatch; missing={missing}, extra={extra}")
        camera_ids = tuple(int(item) for item in self._batch_camera_ids)
        frames = [np.asarray(frames_by_camera[idx], dtype=np.uint8) for idx in camera_ids]
        video, source_shape = self._frames_to_video_tensor(frames, camera_ids=camera_ids)
        if not self._batch_original_frame_shape_hw_by_camera:
            self._batch_original_frame_shape_hw_by_camera = {idx: source_shape for idx in camera_ids}
            queries = [
                self._queries_yx_to_tyx_tensor(
                    self._batch_query_points_yx_by_camera[idx],
                    source_shape_hw=source_shape,
                )
                for idx in camera_ids
            ]
            import torch

            self._batch_query_points_tyx = torch.stack(queries, dim=0).contiguous()
        else:
            expected_shapes = {idx: source_shape for idx in camera_ids}
            if self._batch_original_frame_shape_hw_by_camera != expected_shapes:
                raise ValueError(
                    "TAPNext++ batch-views frame shape changed after initialization; "
                    f"expected {self._batch_original_frame_shape_hw_by_camera}, got {expected_shapes}"
                )
        preprocess_ms = float((time.perf_counter() - preprocess_started_s) * 1000.0)
        first_update = self._batch_tracking_state is None
        try:
            output, run_ms = self._run_model(
                video=video,
                query_points=self._batch_query_points_tyx if first_update else None,
                state=None if first_update else self._batch_tracking_state,
            )
        except RuntimeError as exc:
            if not first_update and len(camera_ids) > 1:
                raise BackendUnavailableError(
                    "TAPNext++ PyTorch state does not support B=3 batch-views in the current adapter."
                ) from exc
            raise
        query_count = int(len(next(iter(self._batch_query_points_yx_by_camera.values()))))
        source_shapes = [self._batch_original_frame_shape_hw_by_camera[idx] for idx in camera_ids]
        postprocess_started_s = time.perf_counter()
        tracks_yx_b, visibility_b, self._batch_tracking_state, postprocess_profile = self._parse_output_profiled(
            output,
            batch_size=len(camera_ids),
            query_count=query_count,
            source_shapes_hw=source_shapes,
        )
        postprocess_ms = float((time.perf_counter() - postprocess_started_s) * 1000.0)
        wall_ms = float((time.perf_counter() - wall_started_s) * 1000.0)
        frame_idx = int(self._batch_frames_seen)
        self._batch_frames_seen += 1
        self._batch_model_call_groups += 1
        results: dict[int, TrackingResult] = {}
        result_pack_started_s = time.perf_counter()
        for batch_idx, camera_idx in enumerate(camera_ids):
            query_points_yx = self._batch_query_points_yx_by_camera[int(camera_idx)]
            stats = {
                "backend": self.name,
                "tracker_backend": self.name,
                "adapter": type(self).__name__,
                "mode": "tapnextpp_online_batch_views",
                "stream_status": "published",
                "update_mode": "batch",
                "chunk_start_idx": frame_idx,
                "chunk_end_idx": frame_idx,
                "frames_seen": int(self._batch_frames_seen),
                "num_query_points": int(len(query_points_yx)),
                "model_run_ms": float(run_ms),
                "cuda_event_ms": float(run_ms),
                "preprocess_ms": float(preprocess_ms),
                "postprocess_ms": float(postprocess_ms),
                "wall_ms": float(wall_ms),
                "fps_model_only": float(1000.0 / run_ms) if run_ms > 0.0 else 0.0,
                "image_size": [int(self.image_size[0]), int(self.image_size[1])],
                "tapnextpp_image_size": [int(self.image_size[0]), int(self.image_size[1])],
                "autocast_dtype": self.autocast_dtype,
                "tapnextpp_autocast_dtype": self.autocast_dtype,
                "tapnextpp_frame_value_range": self.frame_value_range,
                "tapnextpp_fast_postprocess": bool(self.fast_postprocess),
                "tapnextpp_state_active": self._batch_tracking_state is not None,
                "batch_size": int(len(camera_ids)),
                "batch_camera_ids": [int(item) for item in camera_ids],
                "batch_index": int(batch_idx),
                "tapnextpp_batch_size": int(len(camera_ids)),
                "tapnextpp_model_calls": 1,
                "tapnextpp_model_calls_per_group": 1,
                "tapnextpp_model_call_groups": int(self._batch_model_call_groups),
                "device": self.device,
            }
            stats.update(postprocess_profile)
            results[int(camera_idx)] = TrackingResult(
                tracks_yx=tracks_yx_b[batch_idx],
                visibility=visibility_b[batch_idx],
                backend=self.name,
                camera_idx=int(camera_idx),
                query_points_yx=query_points_yx,
                stats=stats,
            )
        result_pack_ms = float((time.perf_counter() - result_pack_started_s) * 1000.0)
        for result in results.values():
            result.stats["result_pack_ms"] = float(result_pack_ms)
            result.stats["postprocess_with_pack_ms"] = float(postprocess_ms + result_pack_ms)
            result.stats["wall_with_pack_ms"] = float(wall_ms + result_pack_ms)
        return results


__all__ = ["TAPNextPPAdapter"]
