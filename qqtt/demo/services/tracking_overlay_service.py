from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from qqtt.demo.services.service_types import RenderLayer


@dataclass(frozen=True)
class TrackingOverlay2D:
    group_id: int
    frame_idx: int
    source_timestamp_s: float
    publish_timestamp_s: float
    tracks_yx_by_camera: dict[int, np.ndarray]
    visibility_by_camera: dict[int, np.ndarray]
    query_points_yx_by_camera: dict[int, np.ndarray]
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DepthFrameRef:
    group_id: int
    timestamp_s: float
    depth_by_camera: dict[int, np.ndarray]
    intrinsics_by_camera: dict[int, Any]
    c2w_by_camera: dict[int, np.ndarray]


class DepthRingBuffer:
    def __init__(self, max_items: int = 8) -> None:
        self.max_items = int(max_items)
        self._items: list[DepthFrameRef] = []

    def publish(self, ref: DepthFrameRef) -> None:
        self._items.append(ref)
        if len(self._items) > self.max_items:
            self._items = self._items[-self.max_items :]

    def nearest(self, *, group_id: int, timestamp_s: float, max_age_ms: float) -> DepthFrameRef | None:
        if not self._items:
            return None
        exact = [item for item in self._items if int(item.group_id) == int(group_id)]
        if exact:
            return exact[-1]
        best = min(self._items, key=lambda item: abs(float(item.timestamp_s) - float(timestamp_s)))
        age_ms = abs(float(best.timestamp_s) - float(timestamp_s)) * 1000.0
        return best if age_ms <= float(max_age_ms) else None


class TrackingOverlayService:
    def __init__(self, *, max_points_per_camera: int = 30, max_depth_age_ms: float = 250.0) -> None:
        self.max_points_per_camera = int(max_points_per_camera)
        self.max_depth_age_ms = float(max_depth_age_ms)
        self.depth_ring = DepthRingBuffer()
        self.depth_mismatch_count = 0

    def lift_to_world(
        self,
        overlay: TrackingOverlay2D,
        depth_ref: DepthFrameRef | None = None,
    ) -> tuple[tuple[RenderLayer, ...], dict[str, Any]]:
        ref = depth_ref or self.depth_ring.nearest(
            group_id=overlay.group_id,
            timestamp_s=overlay.source_timestamp_s,
            max_age_ms=self.max_depth_age_ms,
        )
        if ref is None:
            self.depth_mismatch_count += 1
            return (), {"tracking_overlay_depth_mismatch_count": int(self.depth_mismatch_count)}
        layers: list[RenderLayer] = []
        for camera_idx, tracks in overlay.tracks_yx_by_camera.items():
            visible = np.asarray(overlay.visibility_by_camera.get(camera_idx, []), dtype=np.float32)
            count = min(int(tracks.shape[0]), int(visible.shape[0]), self.max_points_per_camera)
            points = np.zeros((count, 3), dtype=np.float32)
            colors = np.full((count, 3), 255, dtype=np.uint8)
            layers.append(
                RenderLayer(
                    name=f"tracking_overlay_cam{int(camera_idx)}",
                    points_xyz=points,
                    colors_rgb=colors,
                    source_group_id=int(ref.group_id),
                    source_timestamp_s=float(ref.timestamp_s),
                )
            )
        return tuple(layers), {
            "tracking_overlay_depth_group_id": int(ref.group_id),
            "tracking_overlay_depth_age_ms": abs(float(ref.timestamp_s) - float(overlay.source_timestamp_s)) * 1000.0,
            "tracking_overlay_depth_mismatch_count": int(self.depth_mismatch_count),
        }


__all__ = [
    "DepthFrameRef",
    "DepthRingBuffer",
    "TrackingOverlay2D",
    "TrackingOverlayService",
]
