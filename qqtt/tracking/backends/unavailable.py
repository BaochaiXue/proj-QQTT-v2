from __future__ import annotations

from typing import Sequence

import numpy as np

from qqtt.tracking.base import BackendAvailability, BackendUnavailableError, TrackingResult


class UnavailableBackend:
    def __init__(self, name: str, reason: str):
        self.name = str(name)
        self._reason = str(reason)

    def availability(self) -> BackendAvailability:
        return BackendAvailability(self.name, False, self._reason)

    def is_available(self) -> bool:
        return False

    def availability_reason(self) -> str:
        return self._reason

    def initialize(self, frames: Sequence[np.ndarray], query_points_yx: np.ndarray, masks: Sequence[np.ndarray] | None = None) -> None:
        _ = frames, query_points_yx, masks
        raise BackendUnavailableError(self._reason)

    def track_sequence(
        self,
        frames: Sequence[np.ndarray] | None = None,
        query_points_yx: np.ndarray | None = None,
        *,
        frames_rgb: Sequence[np.ndarray] | None = None,
        camera_idx: int | None = None,
        output_shape_hw: tuple[int, int] | None = None,
    ) -> TrackingResult:
        _ = frames, query_points_yx, frames_rgb, camera_idx, output_shape_hw
        raise BackendUnavailableError(self._reason)

    def update(self, frame: np.ndarray) -> TrackingResult:
        _ = frame
        raise BackendUnavailableError(self._reason)


PlannedUnavailableBackend = UnavailableBackend
