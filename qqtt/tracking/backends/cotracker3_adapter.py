from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from qqtt.tracking.base import BackendAvailability, TrackingResult
from qqtt.tracking.backends.cotracker3_online import CoTracker3OnlineBackend
from qqtt.tracking.backends.point_tracker_adapter import (
    TRACKER_BACKEND_COTRACKER3,
    PointTrackerBackendSpec,
)


class CoTracker3Adapter:
    """Point-tracker adapter over the existing CoTracker3 online backend."""

    name = TRACKER_BACKEND_COTRACKER3
    spec = PointTrackerBackendSpec(
        name=TRACKER_BACKEND_COTRACKER3,
        family="cotracker",
        supports_batch_views=True,
        supports_online=True,
        supports_prewarm=True,
        query_format="yx",
        batch_support_status="true",
    )

    def __init__(
        self,
        *,
        device: str = "cuda",
        camera_idx: int | None = None,
        backend: CoTracker3OnlineBackend | None = None,
    ) -> None:
        self.device = str(device)
        self.camera_idx = camera_idx
        self.backend = backend or CoTracker3OnlineBackend(device=self.device)

    def availability(self) -> BackendAvailability:
        return self.backend.availability()

    def is_available(self) -> bool:
        return self.backend.is_available()

    def availability_reason(self) -> str:
        return self.backend.availability_reason()

    def warmup(self) -> dict[str, Any]:
        result = self.backend.warmup()
        result.update({"tracker_backend": self.name, "adapter": type(self).__name__})
        return result

    def initialize(
        self,
        frames: Sequence[np.ndarray],
        query_points_yx: np.ndarray,
        masks: Sequence[np.ndarray] | None = None,
    ) -> None:
        self.backend.initialize(frames, query_points_yx, masks=masks)

    def update(self, frame: np.ndarray) -> TrackingResult:
        result = self.backend.update(frame)
        result.backend = self.name
        if result.camera_idx is None:
            result.camera_idx = self.camera_idx
        result.stats.setdefault("tracker_backend", self.name)
        result.stats.setdefault("adapter", type(self).__name__)
        return result

    def initialize_camera(self, camera_idx: int, query_points_yx: np.ndarray) -> None:
        self.camera_idx = int(camera_idx)
        self.initialize([], query_points_yx)

    def update_camera(self, camera_idx: int, frame_rgb: np.ndarray) -> TrackingResult:
        self.camera_idx = int(camera_idx)
        return self.update(frame_rgb)

    def initialize_batch(self, query_points_yx_by_camera: Mapping[int, np.ndarray]) -> None:
        self.backend.initialize_batch(query_points_yx_by_camera)

    def update_batch(self, frames_by_camera: Mapping[int, np.ndarray]) -> dict[int, TrackingResult]:
        results = self.backend.update_batch(frames_by_camera)
        for camera_idx, result in results.items():
            result.backend = self.name
            result.camera_idx = int(camera_idx)
            result.stats.setdefault("tracker_backend", self.name)
            result.stats.setdefault("adapter", type(self).__name__)
        return results


__all__ = ["CoTracker3Adapter"]
