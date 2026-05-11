from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from qqtt.tracking.base import BackendAvailability, BackendUnavailableError, TrackingResult


class VpiLkBackend:
    name = "vpi_lk"

    def __init__(self, *, device: str = "cuda") -> None:
        self.device = str(device)
        self._vpi_import_error: str | None = None

    def _import_vpi(self):
        try:
            import vpi  # type: ignore
        except Exception as exc:
            self._vpi_import_error = str(exc)
            return None
        return vpi

    def availability(self) -> BackendAvailability:
        vpi = self._import_vpi()
        if vpi is None:
            return BackendAvailability(self.name, False, f"import vpi failed: {self._vpi_import_error}")
        has_backend = hasattr(vpi, "Backend") and hasattr(getattr(vpi, "Backend"), "CUDA")
        if not has_backend:
            return BackendAvailability(self.name, False, "vpi import works but vpi.Backend.CUDA is unavailable")
        return BackendAvailability(self.name, True, "vpi import works and CUDA backend is visible")

    def is_available(self) -> bool:
        return self.availability().available

    def availability_reason(self) -> str:
        return self.availability().reason

    def initialize(self, frames: Sequence[np.ndarray], query_points_yx: np.ndarray, masks: Sequence[np.ndarray] | None = None) -> None:
        _ = frames, query_points_yx, masks
        if not self.is_available():
            raise BackendUnavailableError(self.availability_reason())

    def track_sequence(
        self,
        frames: Sequence[np.ndarray] | None = None,
        query_points_yx: np.ndarray | None = None,
        *,
        frames_rgb: Sequence[np.ndarray] | None = None,
        camera_idx: int | None = None,
        output_shape_hw: tuple[int, int] | None = None,
    ) -> TrackingResult:
        _ = output_shape_hw
        if not self.is_available():
            raise BackendUnavailableError(self.availability_reason())
        video_frames = list(frames_rgb if frames_rgb is not None else frames or [])
        if query_points_yx is None:
            raise ValueError("query_points_yx is required.")
        if len(video_frames) < 1:
            raise ValueError("VPI LK requires at least one frame.")
        queries = np.asarray(query_points_yx, dtype=np.float32)
        tracks = np.zeros((len(video_frames), queries.shape[0], 2), dtype=np.float32)
        tracks[0] = queries
        visibility = np.ones((len(video_frames), queries.shape[0]), dtype=np.float32)
        prev_gray = cv2.cvtColor(video_frames[0], cv2.COLOR_RGB2GRAY)
        prev_xy = queries[:, ::-1].astype(np.float32).reshape(-1, 1, 2)
        for frame_idx in range(1, len(video_frames)):
            next_gray = cv2.cvtColor(video_frames[frame_idx], cv2.COLOR_RGB2GRAY)
            next_xy, status, _err = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, prev_xy, None)
            if next_xy is None or status is None:
                visibility[frame_idx] = 0.0
                tracks[frame_idx] = tracks[frame_idx - 1]
                continue
            status_flat = status.reshape(-1).astype(bool)
            visibility[frame_idx] = visibility[frame_idx - 1] * status_flat.astype(np.float32)
            tracks[frame_idx] = next_xy.reshape(-1, 2)[:, ::-1]
            prev_gray = next_gray
            prev_xy = next_xy.astype(np.float32)
        return TrackingResult(
            tracks_yx=tracks,
            visibility=visibility,
            backend=self.name,
            camera_idx=camera_idx,
            query_points_yx=queries,
            stats={"backend": self.name, "mode": "sparse_lk_point_tracking", "implementation": "opencv_lk_runtime_after_vpi_probe"},
        )

    def update(self, frame: np.ndarray) -> TrackingResult:
        _ = frame
        raise NotImplementedError("VPI LK live update is reserved for a later realtime integration slice.")
