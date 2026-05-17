from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import threading
import time
from typing import Any

import numpy as np

from qqtt.demo.tracking_overlay_render import lift_tracks_yx_to_world, select_overlay_point_indices
from qqtt.tracking.backends.cotracker3_online import CoTracker3OnlineBackend
from qqtt.tracking.sampling import sample_query_points_from_mask


BackendFactory = Callable[[int], Any]


@dataclass(frozen=True)
class TrackingOverlayInputPacket:
    group_id: int
    frame_idx: int
    timestamp_s: float
    rgb_by_camera: Mapping[int, np.ndarray]
    mask_by_camera: Mapping[int, np.ndarray]
    depth_by_camera: Mapping[int, np.ndarray] | None = None
    intrinsics_by_camera: Mapping[int, Any] | None = None
    c2w_by_camera: Mapping[int, np.ndarray] | None = None
    depth_scale_m_per_unit: float | Mapping[int, float] = 0.001

    @property
    def seq(self) -> int:
        return int(self.group_id)


@dataclass(frozen=True)
class TrackingOverlayPacket:
    group_id: int
    frame_idx: int
    timestamp_s: float
    camera_tracks_yx: dict[int, np.ndarray]
    camera_visibility: dict[int, np.ndarray]
    camera_tracks_world: dict[int, np.ndarray] = field(default_factory=dict)
    query_points_yx: dict[int, np.ndarray] = field(default_factory=dict)
    publish_range: tuple[int, int] = (0, 0)
    model_ms: float = 0.0
    e2e_ms: float = 0.0
    stale: bool = False

    @property
    def seq(self) -> int:
        return int(self.group_id)


class LatestTrackingOverlaySlot:
    """Thread-safe latest-wins slot for optional tracking overlays."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._packet: TrackingOverlayPacket | None = None
        self.published = 0
        self.taken = 0
        self.dropped = 0

    def publish(self, packet: TrackingOverlayPacket) -> None:
        with self._lock:
            if self._packet is not None:
                self.dropped += 1
            self._packet = packet
            self.published += 1

    def get_optional(self) -> TrackingOverlayPacket | None:
        with self._lock:
            return self._packet

    def take_latest(self) -> TrackingOverlayPacket | None:
        with self._lock:
            packet = self._packet
            self._packet = None
            if packet is not None:
                self.taken += 1
            return packet

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "published": int(self.published),
                "taken": int(self.taken),
                "dropped": int(self.dropped),
                "pending": int(self._packet is not None),
            }


class CoTracker3OverlayWorker:
    """Async-stage contract for Demo 3 CoTracker3 overlay publishing.

    The worker is intentionally usable synchronously in tests. The runtime owns
    the actual thread or executor and calls ``process_group`` with latest camera
    groups; this class never blocks the renderer and only publishes latest
    overlay packets.
    """

    def __init__(
        self,
        *,
        camera_ids: tuple[int, ...] = (0, 1, 2),
        backend_factory: BackendFactory | None = None,
        output_slot: LatestTrackingOverlaySlot | None = None,
        query_count: int = 128,
        overlay_max_points_per_camera: int = 30,
        seed: int = 42,
        device: str = "cuda",
    ) -> None:
        self.camera_ids = tuple(int(camera_id) for camera_id in camera_ids)
        self.backend_factory = backend_factory or (
            lambda _camera_idx: CoTracker3OnlineBackend(device=device)
        )
        self.output_slot = output_slot or LatestTrackingOverlaySlot()
        self.query_count = int(query_count)
        self.overlay_max_points_per_camera = int(overlay_max_points_per_camera)
        self.seed = int(seed)
        self._backends: dict[int, Any] = {}
        self._query_points_yx: dict[int, np.ndarray] = {}
        self._published_packets = 0
        if self.query_count <= 0:
            raise ValueError("query_count must be positive.")
        if self.overlay_max_points_per_camera <= 0:
            raise ValueError("overlay_max_points_per_camera must be positive.")

    @property
    def published_packets(self) -> int:
        return int(self._published_packets)

    def _depth_scale_for_camera(self, packet: TrackingOverlayInputPacket, camera_idx: int) -> float:
        scale = packet.depth_scale_m_per_unit
        if isinstance(scale, Mapping):
            return float(scale.get(camera_idx, 0.001))
        return float(scale)

    def _ensure_camera_stream(self, camera_idx: int, mask: np.ndarray) -> np.ndarray:
        if camera_idx in self._query_points_yx:
            return self._query_points_yx[camera_idx]
        query_points = sample_query_points_from_mask(
            mask,
            num_points=self.query_count,
            strategy="phystwin_random",
            seed=self.seed + int(camera_idx),
            strict=False,
        )
        self._query_points_yx[camera_idx] = query_points.astype(np.float32)
        if len(query_points) > 0:
            backend = self.backend_factory(int(camera_idx))
            backend.initialize([], query_points.astype(np.float32))
            self._backends[camera_idx] = backend
        return self._query_points_yx[camera_idx]

    def process_group(self, packet: TrackingOverlayInputPacket) -> TrackingOverlayPacket | None:
        started_s = time.perf_counter()
        camera_tracks_yx: dict[int, np.ndarray] = {}
        camera_visibility: dict[int, np.ndarray] = {}
        camera_tracks_world: dict[int, np.ndarray] = {}
        query_points_yx: dict[int, np.ndarray] = {}
        publish_starts: list[int] = []
        publish_ends: list[int] = []
        model_ms = 0.0

        for camera_idx in self.camera_ids:
            frame = packet.rgb_by_camera.get(camera_idx)
            mask = packet.mask_by_camera.get(camera_idx)
            if frame is None or mask is None:
                continue
            query_points = self._ensure_camera_stream(camera_idx, mask)
            query_points_yx[camera_idx] = query_points
            if len(query_points) == 0:
                continue
            backend = self._backends[camera_idx]
            result = backend.update(np.asarray(frame, dtype=np.uint8))
            if str(result.stats.get("stream_status", "")) != "published" or result.tracks_yx.shape[0] == 0:
                continue
            tracks_t = np.asarray(result.tracks_yx[-1], dtype=np.float32)
            visibility_t = np.asarray(result.visibility[-1], dtype=np.float32).reshape(-1)
            selected = select_overlay_point_indices(
                visibility_t,
                max_points=self.overlay_max_points_per_camera,
            )
            if len(selected) == 0:
                continue
            camera_tracks_yx[camera_idx] = tracks_t[selected]
            camera_visibility[camera_idx] = visibility_t[selected]
            query_points_yx[camera_idx] = query_points[selected]
            model_ms += float(result.stats.get("model_run_ms", 0.0))
            publish_starts.append(int(result.stats.get("chunk_start_idx", packet.frame_idx)))
            publish_ends.append(int(result.stats.get("chunk_end_idx", packet.frame_idx)))

            if (
                packet.depth_by_camera is not None
                and packet.intrinsics_by_camera is not None
                and packet.c2w_by_camera is not None
                and camera_idx in packet.depth_by_camera
                and camera_idx in packet.intrinsics_by_camera
                and camera_idx in packet.c2w_by_camera
            ):
                lifted = lift_tracks_yx_to_world(
                    tracks_yx=camera_tracks_yx[camera_idx],
                    visibility=camera_visibility[camera_idx],
                    depth=packet.depth_by_camera[camera_idx],
                    intrinsics=packet.intrinsics_by_camera[camera_idx],
                    c2w=packet.c2w_by_camera[camera_idx],
                    depth_scale_m_per_unit=self._depth_scale_for_camera(packet, camera_idx),
                    mask=np.asarray(mask, dtype=bool),
                )
                camera_tracks_world[camera_idx] = lifted.points_world

        if not camera_tracks_yx:
            return None
        e2e_ms = (time.perf_counter() - started_s) * 1000.0
        overlay = TrackingOverlayPacket(
            group_id=int(packet.group_id),
            frame_idx=int(packet.frame_idx),
            timestamp_s=float(packet.timestamp_s),
            camera_tracks_yx=camera_tracks_yx,
            camera_visibility=camera_visibility,
            camera_tracks_world=camera_tracks_world,
            query_points_yx=query_points_yx,
            publish_range=(min(publish_starts), max(publish_ends)),
            model_ms=float(model_ms),
            e2e_ms=float(e2e_ms),
            stale=False,
        )
        self.output_slot.publish(overlay)
        self._published_packets += 1
        return overlay

    def latest_overlay(self) -> TrackingOverlayPacket | None:
        return self.output_slot.get_optional()

    def snapshot(self) -> dict[str, Any]:
        return {
            "camera_ids": list(self.camera_ids),
            "query_count": int(self.query_count),
            "overlay_max_points_per_camera": int(self.overlay_max_points_per_camera),
            "published_packets": int(self._published_packets),
            "slot": self.output_slot.snapshot(),
        }


__all__ = [
    "CoTracker3OverlayWorker",
    "LatestTrackingOverlaySlot",
    "TrackingOverlayInputPacket",
    "TrackingOverlayPacket",
]
