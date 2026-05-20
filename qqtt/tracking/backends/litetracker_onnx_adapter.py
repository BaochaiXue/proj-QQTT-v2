from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
import importlib
import sys
from pathlib import Path
import time
from typing import Any

import numpy as np

from qqtt.tracking.base import BackendAvailability, BackendUnavailableError, TrackingResult
from qqtt.tracking.backends.point_tracker_adapter import (
    LITETRACKER_RUNTIME_ONNX_CUDA,
    TRACKER_BACKEND_LITETRACKER,
    PointTrackerBackendSpec,
)


REQUIRED_LITETRACKER_ONNX_FILES = ("fnet.onnx", "corr_mlp.onnx", "updateformer.onnx")


class OnnxLiteTrackerAdapter:
    """Serial LiteTracker adapter using the external repo's ONNX Runtime wrapper."""

    name = TRACKER_BACKEND_LITETRACKER
    runtime = LITETRACKER_RUNTIME_ONNX_CUDA
    spec = PointTrackerBackendSpec(
        name=TRACKER_BACKEND_LITETRACKER,
        family="litetracker",
        supports_batch_views=False,
        supports_online=True,
        supports_prewarm=True,
        query_format="yx",
        batch_support_status="onnx_serial_only",
    )

    def __init__(
        self,
        *,
        device: str = "cuda",
        camera_idx: int | None = None,
        weights: str | None = None,
        repo_dir: str | None = None,
        onnx_dir: str | None = None,
        providers: Sequence[str] | None = None,
        export_onnx: bool = False,
        opset: int = 17,
        optimization_level: int = 5,
        tracker_cls: Any | None = None,
    ) -> None:
        self.device = str(device)
        self.camera_idx = camera_idx
        self.weights = str(Path(weights).expanduser()) if weights else None
        self.repo_dir = str(Path(repo_dir).expanduser()) if repo_dir else None
        self.onnx_dir = str(Path(onnx_dir).expanduser()) if onnx_dir else None
        self.providers = tuple(providers or ("CUDAExecutionProvider", "CPUExecutionProvider"))
        self.export_onnx = bool(export_onnx)
        self.opset = int(opset)
        self._actual_opset = max(int(opset), 18)
        self.optimization_level = int(optimization_level)
        self._tracker_cls = tracker_cls
        self._tracker: Any | None = None
        self._query_points_yx: np.ndarray | None = None
        self._queries_xyf: Any | None = None
        self._frame_count = 0
        self._model_load_ms = 0.0
        self._is_first_frame = True
        self._actual_providers: tuple[str, ...] = ()
        self._actual_optimization_level = "uninitialized"

    def availability(self) -> BackendAvailability:
        missing = self._static_missing_reasons()
        if missing:
            return BackendAvailability(self.name, False, "; ".join(missing))
        if self._tracker_cls is not None:
            return BackendAvailability(self.name, True, "Injected OnnxLiteTracker test wrapper is available")
        try:
            import torch  # noqa: F401
            import onnxruntime as ort
        except Exception as exc:
            return BackendAvailability(
                self.name,
                False,
                f"LiteTracker ONNX runtime import failed: {type(exc).__name__}: {exc}",
            )
        available_providers = set(str(item) for item in ort.get_available_providers())
        if "CUDAExecutionProvider" not in available_providers:
            return BackendAvailability(
                self.name,
                False,
                "LiteTracker ONNX-CUDA requires onnxruntime-gpu with CUDAExecutionProvider",
            )
        try:
            self._import_onnx_example_attr("OnnxLiteTracker")
        except Exception as exc:
            return BackendAvailability(
                self.name,
                False,
                (
                    "LiteTracker ONNX runtime requires external lite-tracker repo with "
                    f"onnx_example.py. Pass --litetracker-repo-dir /path/to/lite-tracker. "
                    f"Import failed: {type(exc).__name__}: {exc}"
                ),
            )
        return BackendAvailability(self.name, True, "LiteTracker ONNX-CUDA wrapper and model files are available")

    def is_available(self) -> bool:
        return self.availability().available

    def availability_reason(self) -> str:
        return self.availability().reason

    def is_initialized(self) -> bool:
        return self._query_points_yx is not None

    def warmup(self) -> dict[str, Any]:
        started_s = time.perf_counter()
        tracker = self._load_tracker()
        return {
            "model_load_ms": float(self._model_load_ms),
            "total_ms": float((time.perf_counter() - started_s) * 1000.0),
            "tracker_backend": self.name,
            "adapter": type(self).__name__,
            "device": str(self.device),
            "litetracker_runtime": self.runtime,
            "litetracker_onnx_dir": self.onnx_dir,
            "litetracker_onnx_opset": int(self.opset),
            "litetracker_onnx_opset_actual": int(self._actual_opset),
            "litetracker_onnx_optimization_level": int(self.optimization_level),
            "litetracker_onnx_optimization_level_actual": str(self._actual_optimization_level),
            "onnx_providers": list(self._provider_names(tracker)),
            "onnx_provider": self._primary_provider(tracker),
            "batch_support_status": self.spec.batch_support_status,
        }

    def initialize(
        self,
        frames: Sequence[np.ndarray],
        query_points_yx: np.ndarray,
        masks: Sequence[np.ndarray] | None = None,
    ) -> None:
        _ = masks
        tracker = self._load_tracker()
        reset = getattr(tracker, "reset", None)
        if callable(reset):
            reset()
        self._query_points_yx = self._validate_query_points(query_points_yx, camera_idx=self.camera_idx)
        self._queries_xyf = self._queries_yx_to_xyf(self._query_points_yx)
        self._frame_count = 0
        self._is_first_frame = True
        for frame in frames:
            self.update(frame)

    def initialize_camera(self, camera_idx: int, query_points_yx: np.ndarray) -> None:
        self.camera_idx = int(camera_idx)
        self.initialize([], query_points_yx)

    def initialize_batch(self, query_points_yx_by_camera: Mapping[int, np.ndarray]) -> None:
        _ = query_points_yx_by_camera
        raise NotImplementedError("LiteTracker ONNX serial adapter does not support batch-views yet.")

    def update_batch(self, frames_by_camera: Mapping[int, np.ndarray]) -> dict[int, TrackingResult]:
        _ = frames_by_camera
        raise NotImplementedError("LiteTracker ONNX serial adapter does not support batch-views yet.")

    def update(self, frame: np.ndarray) -> TrackingResult:
        if self._query_points_yx is None or self._queries_xyf is None:
            raise RuntimeError("Call initialize(..., query_points_yx=...) before update().")
        tracker = self._load_tracker()
        e2e_start_s = time.perf_counter()
        frame_start_s = time.perf_counter()
        frame_tensor = self._frame_to_torch_chw(frame)
        frame_to_tensor_ms = float((time.perf_counter() - frame_start_s) * 1000.0)
        query_start_s = time.perf_counter()
        queries_xyf = self._queries_xyf
        query_to_tensor_ms = float((time.perf_counter() - query_start_s) * 1000.0)

        model_start_s = time.perf_counter()
        if self._is_first_frame:
            tracker(frame_tensor, queries_xyf)
            self._is_first_frame = False
        coords, visibility, *rest = tracker(frame_tensor, queries_xyf)
        model_run_ms = float((time.perf_counter() - model_start_s) * 1000.0)

        output_start_s = time.perf_counter()
        tracks_xy = self._to_numpy(coords).astype(np.float32)
        visibility_np = self._to_numpy(visibility).astype(np.float32)
        confidence_np = None if not rest else self._to_numpy(rest[0]).astype(np.float32)
        if tracks_xy.ndim != 4 or tracks_xy.shape[0] != 1 or tracks_xy.shape[-1] != 2:
            raise ValueError(f"LiteTracker ONNX returned invalid coords shape {tracks_xy.shape}")
        if visibility_np.ndim == 4 and visibility_np.shape[-1] == 1:
            visibility_np = visibility_np[..., 0]
        if visibility_np.ndim != 3 or visibility_np.shape[:3] != tracks_xy.shape[:3]:
            raise ValueError(
                f"LiteTracker ONNX returned visibility shape {visibility_np.shape}, expected {tracks_xy.shape[:-1]}"
            )
        if confidence_np is not None and confidence_np.ndim == 4 and confidence_np.shape[-1] == 1:
            confidence_np = confidence_np[..., 0]
        tracks_yx = tracks_xy[0, -1:, :, ::-1].astype(np.float32)
        visibility_t = visibility_np[0, -1:, :].astype(np.float32)
        confidence_t = None if confidence_np is None else confidence_np[0, -1:, :].astype(np.float32)
        output_to_numpy_ms = float((time.perf_counter() - output_start_s) * 1000.0)
        e2e_ms = float((time.perf_counter() - e2e_start_s) * 1000.0)

        frame_idx = int(self._frame_count)
        self._frame_count += 1
        provider_names = self._provider_names(tracker)
        return TrackingResult(
            tracks_yx=tracks_yx,
            visibility=visibility_t,
            confidence=confidence_t,
            backend=self.name,
            camera_idx=self.camera_idx,
            query_points_yx=self._query_points_yx,
            stats={
                "backend": self.name,
                "tracker_backend": self.name,
                "adapter": type(self).__name__,
                "mode": "litetracker_onnx_serial",
                "stream_status": "published",
                "update_mode": "serial",
                "chunk_start_idx": frame_idx,
                "chunk_end_idx": frame_idx,
                "frames_seen": self._frame_count,
                "num_query_points": int(len(self._query_points_yx)),
                "model_run_ms": float(model_run_ms),
                "fps_model_only": float(1000.0 / model_run_ms) if model_run_ms > 0 else 0.0,
                "device": str(self.device),
                "litetracker_runtime": self.runtime,
                "litetracker_onnx_dir": self.onnx_dir,
                "litetracker_provider": self._primary_provider(tracker),
                "onnx_provider": self._primary_provider(tracker),
                "onnx_providers": list(provider_names),
                "litetracker_model_ms": float(model_run_ms),
                "litetracker_e2e_ms": float(e2e_ms),
                "litetracker_frame_to_tensor_ms": float(frame_to_tensor_ms),
                "litetracker_query_to_tensor_ms": float(query_to_tensor_ms),
                "litetracker_output_to_numpy_ms": float(output_to_numpy_ms),
                "litetracker_onnx_opset": int(self.opset),
                "litetracker_onnx_opset_actual": int(self._actual_opset),
                "litetracker_onnx_optimization_level": int(self.optimization_level),
                "litetracker_onnx_optimization_level_actual": str(self._actual_optimization_level),
            },
        )

    def _static_missing_reasons(self) -> list[str]:
        missing: list[str] = []
        if not self.onnx_dir:
            missing.append("--litetracker-onnx-dir")
        elif not self._onnx_dir_path().is_dir() and not self.export_onnx:
            missing.append(f"--litetracker-onnx-dir {self.onnx_dir!r} does not exist")
        if self.repo_dir and not Path(self.repo_dir).is_dir():
            missing.append(f"--litetracker-repo-dir {self.repo_dir!r} does not exist")
        if self.onnx_dir:
            missing_files = self._missing_onnx_files()
            if missing_files and not self.export_onnx:
                missing.append(
                    "LiteTracker ONNX files missing in "
                    f"{self.onnx_dir!r}: {', '.join(missing_files)}. "
                    "Pass --litetracker-export-onnx to export them."
                )
            if missing_files and self.export_onnx and not self.weights:
                missing.append("--litetracker-weights is required when --litetracker-export-onnx is used")
        return missing

    def _load_tracker(self) -> Any:
        if self._tracker is not None:
            return self._tracker
        availability = self.availability()
        if not availability.available:
            raise BackendUnavailableError(availability.reason)
        self._ensure_onnx_files()
        started_s = time.perf_counter()
        tracker_cls = self._tracker_cls or self._import_onnx_example_attr("OnnxLiteTracker")
        with self._session_options_patch():
            tracker = tracker_cls(str(self._onnx_dir_path()), providers=list(self.providers))
        self._model_load_ms = float((time.perf_counter() - started_s) * 1000.0)
        self._actual_providers = self._provider_names(tracker)
        self._tracker = tracker
        return tracker

    def _ensure_onnx_files(self) -> None:
        missing = self._missing_onnx_files()
        if not missing and not self.export_onnx:
            return
        if not self.export_onnx:
            raise BackendUnavailableError(
                f"LiteTracker ONNX files missing in {self.onnx_dir!r}: {', '.join(missing)}"
            )
        self._export_onnx_files()
        missing_after_export = self._missing_onnx_files()
        if missing_after_export:
            raise BackendUnavailableError(
                f"LiteTracker ONNX export did not create: {', '.join(missing_after_export)}"
            )

    def _export_onnx_files(self) -> None:
        if not self.weights:
            raise BackendUnavailableError("--litetracker-weights is required when --litetracker-export-onnx is used")
        self._onnx_dir_path().mkdir(parents=True, exist_ok=True)
        load_model = self._import_onnx_example_attr("load_model")
        export_all = self._import_onnx_example_attr("export_all")
        model = load_model(str(Path(self.weights).expanduser()))
        export_all(model, str(self._onnx_dir_path()), opset_version=int(self._actual_opset))

    def _import_onnx_example_attr(self, attr: str) -> Any:
        if self.repo_dir:
            path = str(Path(self.repo_dir).expanduser())
            if path and path not in sys.path:
                sys.path.insert(0, path)
        module = importlib.import_module("onnx_example")
        return getattr(module, attr)

    @contextmanager
    def _session_options_patch(self):
        if self._tracker_cls is not None:
            self._actual_optimization_level = "injected-test-wrapper"
            yield
            return
        try:
            import torch  # noqa: F401
            import onnxruntime as ort
        except Exception:
            self._actual_optimization_level = "unavailable"
            yield
            return
        preload = getattr(ort, "preload_dlls", None)
        if callable(preload):
            try:
                preload()
            except Exception:
                pass
        session_options, actual_level = self._session_options(ort)
        self._actual_optimization_level = actual_level
        if session_options is None:
            yield
            return
        original = ort.InferenceSession

        def patched_inference_session(path_or_bytes: Any, *args: Any, **kwargs: Any) -> Any:
            if args:
                return original(path_or_bytes, *args, **kwargs)
            return original(path_or_bytes, session_options, **kwargs)

        ort.InferenceSession = patched_inference_session
        try:
            yield
        finally:
            ort.InferenceSession = original

    def _session_options(self, ort: Any) -> tuple[Any | None, str]:
        options = ort.SessionOptions()
        level = int(self.optimization_level)
        graph_levels = ort.GraphOptimizationLevel
        if level <= 0:
            options.graph_optimization_level = graph_levels.ORT_DISABLE_ALL
            return options, "ORT_DISABLE_ALL"
        if level == 1:
            options.graph_optimization_level = graph_levels.ORT_ENABLE_BASIC
            return options, "ORT_ENABLE_BASIC"
        if level < 5:
            options.graph_optimization_level = graph_levels.ORT_ENABLE_EXTENDED
            return options, "ORT_ENABLE_EXTENDED"
        options.graph_optimization_level = graph_levels.ORT_ENABLE_ALL
        return options, "ORT_ENABLE_ALL"

    def _onnx_dir_path(self) -> Path:
        if not self.onnx_dir:
            raise BackendUnavailableError("--litetracker-onnx-dir is required for LiteTracker ONNX runtime")
        return Path(self.onnx_dir).expanduser()

    def _missing_onnx_files(self) -> list[str]:
        if not self.onnx_dir:
            return list(REQUIRED_LITETRACKER_ONNX_FILES)
        root = self._onnx_dir_path()
        return [name for name in REQUIRED_LITETRACKER_ONNX_FILES if not (root / name).is_file()]

    @staticmethod
    def _validate_query_points(query_points_yx: np.ndarray, *, camera_idx: int | None = None) -> np.ndarray:
        points = np.asarray(query_points_yx, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 2:
            prefix = "query_points_yx" if camera_idx is None else f"query_points_yx for camera {camera_idx}"
            raise ValueError(f"{prefix} must have shape (N,2); got {points.shape}")
        if len(points) == 0:
            raise ValueError("LiteTracker ONNX requires at least one query point.")
        return np.ascontiguousarray(points)

    @staticmethod
    def _frame_to_torch_chw(frame: np.ndarray) -> Any:
        import torch

        arr = np.asarray(frame, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(f"LiteTracker ONNX frame must be HxWx3 RGB uint8; got {arr.shape}")
        return torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1)[None].float().contiguous()

    @staticmethod
    def _queries_yx_to_xyf(query_points_yx: np.ndarray) -> Any:
        import torch

        points_yx = np.asarray(query_points_yx, dtype=np.float32).reshape(-1, 2)
        points_xy = np.ascontiguousarray(points_yx[:, ::-1])
        frame_index = np.zeros((len(points_xy), 1), dtype=np.float32)
        queries = np.concatenate([frame_index, points_xy], axis=1)[None]
        return torch.from_numpy(queries).float().contiguous()

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def _provider_names(self, tracker: Any) -> tuple[str, ...]:
        session = getattr(tracker, "fnet_session", None)
        get_providers = getattr(session, "get_providers", None)
        if callable(get_providers):
            return tuple(str(item) for item in get_providers())
        if self._actual_providers:
            return self._actual_providers
        return tuple(str(item) for item in self.providers)

    def _primary_provider(self, tracker: Any) -> str:
        providers = self._provider_names(tracker)
        return str(providers[0]) if providers else ""


__all__ = ["OnnxLiteTrackerAdapter", "REQUIRED_LITETRACKER_ONNX_FILES"]
