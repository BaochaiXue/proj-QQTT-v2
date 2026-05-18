from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
import threading
import time
from typing import Any

import numpy as np

from qqtt.demo.tracking_overlay_render import lift_tracks_yx_to_world, select_overlay_point_indices
from qqtt.tracking.backends.cotracker3_online import CoTracker3OnlineBackend
from qqtt.tracking.sampling import (
    PHYSTWIN_DENSE_QUERY_POINTS,
    sample_phystwin_dense,
)


BackendFactory = Callable[[int], Any]


@dataclass(frozen=True)
class TrackingOverlayInputPacket:
    group_id: int
    frame_idx: int
    timestamp_s: float
    rgb_by_camera: Mapping[int, np.ndarray]
    mask_by_camera: Mapping[int, np.ndarray]
    object_mask_by_camera: Mapping[int, np.ndarray] | None = None
    controller_mask_by_camera: Mapping[int, np.ndarray] | None = None
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
    source_timestamp_s: float | None = None
    publish_range: tuple[int, int] = (0, 0)
    model_ms: float = 0.0
    e2e_ms: float = 0.0
    stale: bool = False
    tracking_query_count_actual_by_camera: dict[int, int] = field(default_factory=dict)
    tracking_union_pixels_by_camera: dict[int, int] = field(default_factory=dict)
    tracking_object_pixels_by_camera: dict[int, int] = field(default_factory=dict)
    tracking_controller_pixels_by_camera: dict[int, int] = field(default_factory=dict)
    tracking_sample_object_hits_by_camera: dict[int, int] = field(default_factory=dict)
    tracking_sample_controller_hits_by_camera: dict[int, int] = field(default_factory=dict)
    tracking_sample_overlap_hits_by_camera: dict[int, int] = field(default_factory=dict)
    tracking_sample_background_hits_by_camera: dict[int, int] = field(default_factory=dict)
    overlay_display_count_by_camera: dict[int, int] = field(default_factory=dict)

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

    def get_optional(
        self,
        *,
        now_s: float | None = None,
        stale_timeout_s: float | None = None,
    ) -> TrackingOverlayPacket | None:
        with self._lock:
            packet = self._packet
        if (
            packet is not None
            and now_s is not None
            and stale_timeout_s is not None
            and float(stale_timeout_s) >= 0.0
            and float(now_s) - float(packet.timestamp_s) > float(stale_timeout_s)
        ):
            return replace(packet, stale=True)
        return packet

    def get_fresh(self, *, now_s: float, stale_timeout_s: float) -> TrackingOverlayPacket | None:
        packet = self.get_optional(now_s=now_s, stale_timeout_s=stale_timeout_s)
        if packet is None or packet.stale:
            return None
        return packet

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


class LatestTrackingInputSlot:
    """Latest-wins input queue for CoTracker work that must not block render."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._packet: TrackingOverlayInputPacket | None = None
        self.published = 0
        self.taken = 0
        self.dropped = 0

    def publish(self, packet: TrackingOverlayInputPacket) -> None:
        with self._lock:
            if self._packet is not None:
                self.dropped += 1
            self._packet = packet
            self.published += 1

    def take_latest(self) -> TrackingOverlayInputPacket | None:
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


class CoTracker3OverlayThread:
    """Background loop that converts latest tracking inputs into overlay packets."""

    def __init__(
        self,
        *,
        worker: "CoTracker3OverlayWorker",
        input_slot: LatestTrackingInputSlot,
        stop_event: threading.Event | None = None,
        poll_interval_s: float = 0.001,
    ) -> None:
        self.worker = worker
        self.input_slot = input_slot
        self.stop_event = stop_event or threading.Event()
        self.poll_interval_s = float(poll_interval_s)
        self.thread: threading.Thread | None = None
        self.processed_packets = 0
        self.error_count = 0
        self.last_error: str | None = None

    def start(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            return
        self.thread = threading.Thread(target=self.run, name="demo3-cotracker-overlay", daemon=True)
        self.thread.start()

    def stop(self, *, timeout_s: float = 1.0) -> None:
        self.stop_event.set()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=float(timeout_s))

    def run(self) -> None:
        while not self.stop_event.is_set():
            packet = self.input_slot.take_latest()
            if packet is None:
                time.sleep(self.poll_interval_s)
                continue
            try:
                self.worker.process_group(packet)
                self.processed_packets += 1
            except Exception as exc:
                self.error_count += 1
                self.last_error = f"{type(exc).__name__}: {exc}"

    def snapshot(self) -> dict[str, Any]:
        return {
            "processed_packets": int(self.processed_packets),
            "error_count": int(self.error_count),
            "last_error": self.last_error,
            "input_slot": self.input_slot.snapshot(),
            "worker": self.worker.snapshot(),
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
        query_mode: str = "phystwin_dense",
        query_count_request: int | str = "auto",
        query_count: int | None = None,
        overlay_max_points_per_camera: int = 30,
        seed: int = 42,
        device: str = "cuda",
        sampling_device: str = "cpu",
        init_requires_object_and_controller: bool = True,
    ) -> None:
        self.camera_ids = tuple(int(camera_id) for camera_id in camera_ids)
        self.backend_factory = backend_factory or (
            lambda _camera_idx: CoTracker3OnlineBackend(device=device)
        )
        self.output_slot = output_slot or LatestTrackingOverlaySlot()
        self.query_mode = str(query_mode)
        self.query_count_request = str(query_count if query_count is not None else query_count_request)
        self.query_count = self._normalize_query_count_request(self.query_count_request)
        self.overlay_max_points_per_camera = int(overlay_max_points_per_camera)
        self.seed = int(seed)
        self.sampling_device = str(sampling_device)
        self.init_requires_object_and_controller = bool(init_requires_object_and_controller)
        self._backends: dict[int, Any] = {}
        self._query_points_yx: dict[int, np.ndarray] = {}
        self._camera_stats: dict[int, dict[str, int | bool]] = {}
        self._overlay_display_count_by_camera: dict[int, int] = {}
        self._published_packets = 0
        self._model_ms_samples: list[float] = []
        self._e2e_ms_samples: list[float] = []
        self._publish_times_s: list[float] = []
        if self.query_mode != "phystwin_dense":
            raise ValueError("CoTracker3OverlayWorker currently supports only query_mode='phystwin_dense'.")
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

    @staticmethod
    def _normalize_query_count_request(value: int | str) -> int | str:
        raw = str(value).strip().lower()
        if raw == "auto":
            return "auto"
        count = int(raw)
        if count <= 0:
            raise ValueError("query_count_request must be 'auto' or a positive integer.")
        return count

    def _component_mask_or_empty(
        self,
        masks: Mapping[int, np.ndarray] | None,
        camera_idx: int,
        shape: tuple[int, ...],
    ) -> np.ndarray:
        if masks is None or camera_idx not in masks:
            return np.zeros(shape, dtype=bool)
        return np.asarray(masks[camera_idx], dtype=bool)

    def _sample_query_points(self, *, camera_idx: int, union_mask: np.ndarray) -> np.ndarray:
        if self.query_count == "auto":
            return sample_phystwin_dense(
                union_mask,
                seed=self.seed,
                camera_idx=int(camera_idx),
                torch_device=self.sampling_device,
            ).astype(np.float32)
        dense = sample_phystwin_dense(
            union_mask,
            seed=self.seed,
            camera_idx=int(camera_idx),
            torch_device=self.sampling_device,
        ).astype(np.float32)
        return dense[: int(self.query_count)].astype(np.float32)

    def _record_sampling_stats(
        self,
        *,
        camera_idx: int,
        union_mask: np.ndarray,
        object_mask: np.ndarray,
        controller_mask: np.ndarray,
        query_points: np.ndarray,
        waiting_for_object_controller: bool,
    ) -> None:
        points = np.asarray(query_points, dtype=np.float32).reshape(-1, 2)
        yi = np.clip(np.rint(points[:, 0]).astype(np.int64), 0, union_mask.shape[0] - 1) if len(points) else np.empty(0, dtype=np.int64)
        xi = np.clip(np.rint(points[:, 1]).astype(np.int64), 0, union_mask.shape[1] - 1) if len(points) else np.empty(0, dtype=np.int64)
        object_hits = int(np.count_nonzero(object_mask[yi, xi])) if len(points) else 0
        controller_hits = int(np.count_nonzero(controller_mask[yi, xi])) if len(points) else 0
        overlap_hits = int(np.count_nonzero(object_mask[yi, xi] & controller_mask[yi, xi])) if len(points) else 0
        background_hits = int(np.count_nonzero(~union_mask[yi, xi])) if len(points) else 0
        self._camera_stats[int(camera_idx)] = {
            "tracking_query_count_actual": int(len(points)),
            "tracking_union_pixels": int(np.count_nonzero(union_mask)),
            "tracking_object_pixels": int(np.count_nonzero(object_mask)),
            "tracking_controller_pixels": int(np.count_nonzero(controller_mask)),
            "tracking_sample_object_hits": object_hits,
            "tracking_sample_controller_hits": controller_hits,
            "tracking_sample_overlap_hits": overlap_hits,
            "tracking_sample_background_hits": background_hits,
            "cotracker_waiting_for_object_controller": bool(waiting_for_object_controller),
            "object_mask_nonempty": bool(np.count_nonzero(object_mask) > 0),
            "controller_mask_nonempty": bool(np.count_nonzero(controller_mask) > 0),
        }

    def _ensure_camera_stream(
        self,
        camera_idx: int,
        union_mask: np.ndarray,
        object_mask: np.ndarray,
        controller_mask: np.ndarray,
    ) -> np.ndarray:
        existing = self._query_points_yx.get(camera_idx)
        if existing is not None and len(existing) > 0:
            return existing
        waiting = bool(
            self.init_requires_object_and_controller
            and (int(np.count_nonzero(object_mask)) == 0 or int(np.count_nonzero(controller_mask)) == 0)
        )
        if waiting:
            query_points = np.empty((0, 2), dtype=np.float32)
            self._record_sampling_stats(
                camera_idx=camera_idx,
                union_mask=union_mask,
                object_mask=object_mask,
                controller_mask=controller_mask,
                query_points=query_points,
                waiting_for_object_controller=True,
            )
            return query_points
        query_points = self._sample_query_points(camera_idx=camera_idx, union_mask=union_mask)
        self._record_sampling_stats(
            camera_idx=camera_idx,
            union_mask=union_mask,
            object_mask=object_mask,
            controller_mask=controller_mask,
            query_points=query_points,
            waiting_for_object_controller=False,
        )
        if len(query_points) == 0:
            return query_points
        self._query_points_yx[camera_idx] = query_points
        backend = self.backend_factory(int(camera_idx))
        backend.initialize([], query_points)
        self._backends[camera_idx] = backend
        return query_points

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
            union_mask = np.asarray(mask, dtype=bool)
            object_mask = self._component_mask_or_empty(
                packet.object_mask_by_camera,
                camera_idx,
                union_mask.shape,
            )
            controller_mask = self._component_mask_or_empty(
                packet.controller_mask_by_camera,
                camera_idx,
                union_mask.shape,
            )
            query_points = self._ensure_camera_stream(camera_idx, union_mask, object_mask, controller_mask)
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
            self._overlay_display_count_by_camera[int(camera_idx)] = int(len(selected))
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
                    mask=union_mask,
                )
                camera_tracks_world[camera_idx] = lifted.points_world

        if not camera_tracks_yx:
            return None
        e2e_ms = (time.perf_counter() - started_s) * 1000.0
        published_s = time.perf_counter()
        overlay = TrackingOverlayPacket(
            group_id=int(packet.group_id),
            frame_idx=int(packet.frame_idx),
            timestamp_s=float(published_s),
            camera_tracks_yx=camera_tracks_yx,
            camera_visibility=camera_visibility,
            camera_tracks_world=camera_tracks_world,
            query_points_yx=query_points_yx,
            source_timestamp_s=float(packet.timestamp_s),
            publish_range=(min(publish_starts), max(publish_ends)),
            model_ms=float(model_ms),
            e2e_ms=float(e2e_ms),
            stale=False,
            tracking_query_count_actual_by_camera={
                int(camera_idx): int(stats.get("tracking_query_count_actual", 0))
                for camera_idx, stats in self._camera_stats.items()
            },
            tracking_union_pixels_by_camera={
                int(camera_idx): int(stats.get("tracking_union_pixels", 0))
                for camera_idx, stats in self._camera_stats.items()
            },
            tracking_object_pixels_by_camera={
                int(camera_idx): int(stats.get("tracking_object_pixels", 0))
                for camera_idx, stats in self._camera_stats.items()
            },
            tracking_controller_pixels_by_camera={
                int(camera_idx): int(stats.get("tracking_controller_pixels", 0))
                for camera_idx, stats in self._camera_stats.items()
            },
            tracking_sample_object_hits_by_camera={
                int(camera_idx): int(stats.get("tracking_sample_object_hits", 0))
                for camera_idx, stats in self._camera_stats.items()
            },
            tracking_sample_controller_hits_by_camera={
                int(camera_idx): int(stats.get("tracking_sample_controller_hits", 0))
                for camera_idx, stats in self._camera_stats.items()
            },
            tracking_sample_overlap_hits_by_camera={
                int(camera_idx): int(stats.get("tracking_sample_overlap_hits", 0))
                for camera_idx, stats in self._camera_stats.items()
            },
            tracking_sample_background_hits_by_camera={
                int(camera_idx): int(stats.get("tracking_sample_background_hits", 0))
                for camera_idx, stats in self._camera_stats.items()
            },
            overlay_display_count_by_camera=dict(self._overlay_display_count_by_camera),
        )
        self.output_slot.publish(overlay)
        self._published_packets += 1
        self._model_ms_samples.append(float(model_ms))
        self._e2e_ms_samples.append(float(e2e_ms))
        self._publish_times_s.append(float(published_s))
        return overlay

    def latest_overlay(self) -> TrackingOverlayPacket | None:
        return self.output_slot.get_optional()

    def snapshot(self) -> dict[str, Any]:
        actual_by_camera = {
            int(camera_idx): int(stats.get("tracking_query_count_actual", 0))
            for camera_idx, stats in self._camera_stats.items()
        }
        return {
            "camera_ids": list(self.camera_ids),
            "query_mode": str(self.query_mode),
            "query_count_request": str(self.query_count),
            "query_count": int(PHYSTWIN_DENSE_QUERY_POINTS if self.query_count == "auto" else self.query_count),
            "tracking_query_count_actual_by_camera": actual_by_camera,
            "tracking_union_pixels_by_camera": {
                int(camera_idx): int(stats.get("tracking_union_pixels", 0))
                for camera_idx, stats in self._camera_stats.items()
            },
            "tracking_object_pixels_by_camera": {
                int(camera_idx): int(stats.get("tracking_object_pixels", 0))
                for camera_idx, stats in self._camera_stats.items()
            },
            "tracking_controller_pixels_by_camera": {
                int(camera_idx): int(stats.get("tracking_controller_pixels", 0))
                for camera_idx, stats in self._camera_stats.items()
            },
            "tracking_sample_object_hits_by_camera": {
                int(camera_idx): int(stats.get("tracking_sample_object_hits", 0))
                for camera_idx, stats in self._camera_stats.items()
            },
            "tracking_sample_controller_hits_by_camera": {
                int(camera_idx): int(stats.get("tracking_sample_controller_hits", 0))
                for camera_idx, stats in self._camera_stats.items()
            },
            "tracking_sample_overlap_hits_by_camera": {
                int(camera_idx): int(stats.get("tracking_sample_overlap_hits", 0))
                for camera_idx, stats in self._camera_stats.items()
            },
            "tracking_sample_background_hits_by_camera": {
                int(camera_idx): int(stats.get("tracking_sample_background_hits", 0))
                for camera_idx, stats in self._camera_stats.items()
            },
            "cotracker_waiting_for_object_controller_by_camera": {
                int(camera_idx): bool(stats.get("cotracker_waiting_for_object_controller", False))
                for camera_idx, stats in self._camera_stats.items()
            },
            "object_mask_nonempty_by_camera": {
                int(camera_idx): bool(stats.get("object_mask_nonempty", False))
                for camera_idx, stats in self._camera_stats.items()
            },
            "controller_mask_nonempty_by_camera": {
                int(camera_idx): bool(stats.get("controller_mask_nonempty", False))
                for camera_idx, stats in self._camera_stats.items()
            },
            "overlay_display_count_by_camera": {
                int(camera_idx): int(count)
                for camera_idx, count in self._overlay_display_count_by_camera.items()
            },
            "overlay_max_points_per_camera": int(self.overlay_max_points_per_camera),
            "published_packets": int(self._published_packets),
            "model_ms_median": _median(self._model_ms_samples),
            "model_ms_p95": _p95(self._model_ms_samples),
            "e2e_ms_median": _median(self._e2e_ms_samples),
            "e2e_ms_p95": _p95(self._e2e_ms_samples),
            "publish_fps": _event_fps(self._publish_times_s),
            "slot": self.output_slot.snapshot(),
        }


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.median(np.asarray(values, dtype=np.float32)))


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float32), 95))


def _event_fps(times_s: list[float]) -> float:
    if len(times_s) < 2:
        return 0.0
    duration_s = float(max(times_s) - min(times_s))
    return float((len(times_s) - 1) / duration_s) if duration_s > 0 else 0.0


__all__ = [
    "CoTracker3OverlayThread",
    "CoTracker3OverlayWorker",
    "LatestTrackingInputSlot",
    "LatestTrackingOverlaySlot",
    "TrackingOverlayInputPacket",
    "TrackingOverlayPacket",
]
