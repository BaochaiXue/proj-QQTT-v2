from __future__ import annotations

from typing import Sequence

import numpy as np

from qqtt.tracking.base import BackendAvailability, TrackingResult


class FakeTrackingBackend:
    name = "fake"

    def availability(self) -> BackendAvailability:
        return BackendAvailability(self.name, True, "deterministic in-repo fake backend for CI")

    def is_available(self) -> bool:
        return True

    def availability_reason(self) -> str:
        return self.availability().reason

    def initialize(self, frames: Sequence[np.ndarray], query_points_yx: np.ndarray, masks: Sequence[np.ndarray] | None = None) -> None:
        _ = frames, query_points_yx, masks

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
        video_frames = list(frames_rgb if frames_rgb is not None else frames or [])
        if query_points_yx is None:
            raise ValueError("query_points_yx is required.")
        if not video_frames:
            raise ValueError("frames_rgb is required.")
        queries = np.asarray(query_points_yx, dtype=np.float32)
        tracks = np.repeat(queries[None, :, :], len(video_frames), axis=0)
        visibility = np.ones((len(video_frames), queries.shape[0]), dtype=np.float32)
        return TrackingResult(
            tracks_yx=tracks,
            visibility=visibility,
            backend=self.name,
            camera_idx=camera_idx,
            query_points_yx=queries,
            stats={
                "backend": self.name,
                "camera_idx": None if camera_idx is None else int(camera_idx),
                "num_frames": int(len(video_frames)),
                "num_query_points": int(queries.shape[0]),
                "model_load_ms": 0.0,
                "model_run_ms": 0.0,
                "fps_model_only": 0.0,
                "device": "cpu",
            },
        )

    def update(self, frame: np.ndarray) -> TrackingResult:
        _ = frame
        raise NotImplementedError("FakeTrackingBackend only supports track_sequence.")
