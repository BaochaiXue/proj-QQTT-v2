from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import numpy as np

from qqtt.tracking.base import BackendAvailability, BackendUnavailableError, TrackingResult
from qqtt.tracking.backends.point_tracker_adapter import (
    TRACKER_BACKEND_LITETRACKER,
    PointTrackerBackendSpec,
)


class LiteTrackerAdapter:
    """Serial live LiteTracker adapter over the external LiteTracker repo."""

    name = TRACKER_BACKEND_LITETRACKER
    spec = PointTrackerBackendSpec(
        name=TRACKER_BACKEND_LITETRACKER,
        family="litetracker",
        supports_batch_views=False,
        supports_online=True,
        supports_prewarm=True,
        query_format="yx",
        batch_support_status="serial_only",
    )

    def __init__(
        self,
        *,
        device: str = "cuda",
        camera_idx: int | None = None,
        weights: str | None = None,
        repo_dir: str | None = None,
    ) -> None:
        self.device = str(device)
        self.camera_idx = camera_idx
        self.weights = str(Path(weights).expanduser()) if weights else None
        self.repo_dir = str(Path(repo_dir).expanduser()) if repo_dir else None
        self._tracker: Any | None = None
        self._model_load_ms = 0.0
        self._query_points_yx: np.ndarray | None = None
        self._frame_count = 0

    def availability(self) -> BackendAvailability:
        missing: list[str] = []
        if not self.weights:
            missing.append("--litetracker-weights")
        elif not Path(self.weights).is_file():
            missing.append(f"--litetracker-weights {self.weights!r} does not exist")
        if self.repo_dir and not Path(self.repo_dir).is_dir():
            missing.append(f"--litetracker-repo-dir {self.repo_dir!r} does not exist")
        if missing:
            return BackendAvailability(self.name, False, "; ".join(missing))
        try:
            import torch  # noqa: F401
            from src.eval.lite_tracker_wrapper import LiteTrackerWrapper  # noqa: F401
        except Exception as exc:
            return BackendAvailability(
                self.name,
                False,
                f"LiteTracker runtime import failed: {type(exc).__name__}: {exc}",
            )
        return BackendAvailability(self.name, True, "LiteTracker wrapper import and weights path are available")

    def is_available(self) -> bool:
        return self.availability().available

    def availability_reason(self) -> str:
        return self.availability().reason

    def is_initialized(self) -> bool:
        return self._query_points_yx is not None

    def _load_tracker(self) -> Any:
        if self._tracker is not None:
            return self._tracker
        availability = self.availability()
        if not availability.available:
            raise BackendUnavailableError(availability.reason)
        import torch
        from src.eval.lite_tracker_wrapper import LiteTrackerWrapper

        started_s = time.perf_counter()
        tracker = LiteTrackerWrapper(Path(str(self.weights)), return_vis=True)
        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        self._model_load_ms = float((time.perf_counter() - started_s) * 1000.0)
        self._tracker = tracker
        return tracker

    def warmup(self) -> dict[str, Any]:
        started_s = time.perf_counter()
        tracker = self._load_tracker()
        return {
            "model_load_ms": float(self._model_load_ms),
            "total_ms": float((time.perf_counter() - started_s) * 1000.0),
            "tracker_backend": self.name,
            "adapter": type(self).__name__,
            "device": str(getattr(tracker, "device", self.device)),
            "dtype": str(getattr(tracker, "dtype", "")),
            "batch_support_status": self.spec.batch_support_status,
        }

    @staticmethod
    def _validate_query_points(query_points_yx: np.ndarray, *, camera_idx: int | None = None) -> np.ndarray:
        points = np.asarray(query_points_yx, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 2:
            prefix = "query_points_yx" if camera_idx is None else f"query_points_yx for camera {camera_idx}"
            raise ValueError(f"{prefix} must have shape (N,2); got {points.shape}")
        if len(points) == 0:
            raise ValueError("LiteTracker requires at least one query point.")
        return np.ascontiguousarray(points)

    @staticmethod
    def _frame_to_torch_uint8(frame: np.ndarray):
        import torch

        arr = np.asarray(frame, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(f"LiteTracker frame must be HxWx3 RGB uint8; got {arr.shape}")
        return torch.from_numpy(np.ascontiguousarray(arr))[None]

    def initialize(
        self,
        frames: list[np.ndarray] | tuple[np.ndarray, ...],
        query_points_yx: np.ndarray,
        masks: list[np.ndarray] | tuple[np.ndarray, ...] | None = None,
    ) -> None:
        _ = masks
        self._load_tracker()
        self._query_points_yx = self._validate_query_points(query_points_yx, camera_idx=self.camera_idx)
        self._frame_count = 0
        for frame in frames:
            self.update(frame)

    def initialize_camera(self, camera_idx: int, query_points_yx: np.ndarray) -> None:
        self.camera_idx = int(camera_idx)
        self.initialize([], query_points_yx)

    def update(self, frame: np.ndarray) -> TrackingResult:
        if self._query_points_yx is None:
            raise RuntimeError("Call initialize(..., query_points_yx=...) before update().")
        import torch

        tracker = self._load_tracker()
        frame_tensor = self._frame_to_torch_uint8(frame)
        queries_xy = np.ascontiguousarray(self._query_points_yx[:, ::-1])
        started_s = time.perf_counter()
        points_xy, visibility = tracker.trackpoints2D(queries_xy, [frame_tensor, frame_tensor])
        if str(getattr(tracker, "device", self.device)).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        run_ms = float((time.perf_counter() - started_s) * 1000.0)
        tracks_xy = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
        tracks_yx = tracks_xy[:, ::-1][None, :, :].astype(np.float32)
        visibility_np = np.asarray(visibility, dtype=np.float32).reshape(-1)
        visibility_t = visibility_np[None, :].astype(np.float32)
        frame_idx = int(self._frame_count)
        self._frame_count += 1
        return TrackingResult(
            tracks_yx=tracks_yx,
            visibility=visibility_t,
            backend=self.name,
            camera_idx=self.camera_idx,
            query_points_yx=self._query_points_yx,
            stats={
                "backend": self.name,
                "tracker_backend": self.name,
                "adapter": type(self).__name__,
                "mode": "litetracker_trackpoints2D",
                "stream_status": "published",
                "update_mode": "serial",
                "chunk_start_idx": frame_idx,
                "chunk_end_idx": frame_idx,
                "frames_seen": self._frame_count,
                "num_query_points": int(len(self._query_points_yx)),
                "model_run_ms": float(run_ms),
                "fps_model_only": float(1000.0 / run_ms) if run_ms > 0 else 0.0,
                "device": str(getattr(tracker, "device", self.device)),
                "dtype": str(getattr(tracker, "dtype", "")),
            },
        )


__all__ = ["LiteTrackerAdapter"]
