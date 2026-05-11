from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

import numpy as np


@dataclass(frozen=True)
class BackendAvailability:
    backend: str
    available: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"backend": self.backend, "available": bool(self.available), "reason": str(self.reason)}


@dataclass
class TrackingResult:
    tracks_yx: np.ndarray
    visibility: np.ndarray
    confidence: np.ndarray | None = None
    backend: str = ""
    camera_idx: int | None = None
    coordinate_order: str = "yx"
    stats: dict[str, Any] = field(default_factory=dict)
    track_ids: np.ndarray | None = None
    query_points_yx: np.ndarray | None = None

    def __post_init__(self) -> None:
        if str(self.coordinate_order).lower() != "yx":
            raise ValueError(f"TrackingResult coordinate_order must be 'yx'; got {self.coordinate_order!r}")
        self.coordinate_order = "yx"
        self.tracks_yx = np.asarray(self.tracks_yx, dtype=np.float32)
        self.visibility = np.asarray(self.visibility, dtype=np.float32)
        if self.tracks_yx.ndim not in {3, 4} or self.tracks_yx.shape[-1] != 2:
            raise ValueError(f"tracks_yx must have shape (T,N,2) or (C,T,N,2); got {self.tracks_yx.shape}")
        expected_prefix = self.tracks_yx.shape[:-1]
        if self.visibility.shape != expected_prefix and self.visibility.shape != (*expected_prefix, 1):
            raise ValueError(f"visibility shape {self.visibility.shape} does not match tracks {self.tracks_yx.shape}")
        if self.confidence is not None:
            self.confidence = np.asarray(self.confidence, dtype=np.float32)
            if self.confidence.shape != expected_prefix and self.confidence.shape != (*expected_prefix, 1):
                raise ValueError(f"confidence shape {self.confidence.shape} does not match tracks {self.tracks_yx.shape}")
        num_points = int(self.tracks_yx.shape[-2])
        if self.track_ids is not None:
            self.track_ids = np.asarray(self.track_ids)
            if self.track_ids.shape[0] != num_points:
                raise ValueError("track_ids length must match number of query points.")
        if self.query_points_yx is not None:
            self.query_points_yx = np.asarray(self.query_points_yx, dtype=np.float32)
            if self.query_points_yx.shape != (num_points, 2):
                raise ValueError("query_points_yx must have shape (N,2).")


class TrackingBackend(Protocol):
    name: str

    def availability(self) -> BackendAvailability:
        ...

    def is_available(self) -> bool:
        ...

    def availability_reason(self) -> str:
        ...

    def initialize(
        self,
        frames: Sequence[np.ndarray],
        query_points_yx: np.ndarray,
        masks: Sequence[np.ndarray] | None = None,
    ) -> None:
        ...

    def track_sequence(
        self,
        *,
        frames_rgb: Sequence[np.ndarray],
        query_points_yx: np.ndarray,
        camera_idx: int,
        output_shape_hw: tuple[int, int] | None = None,
    ) -> TrackingResult:
        ...

    def update(self, frame: np.ndarray) -> TrackingResult:
        ...


class BackendUnavailableError(RuntimeError):
    pass
