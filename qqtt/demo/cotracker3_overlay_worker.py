from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
import threading
import time
from typing import Any

import numpy as np

from qqtt.demo.tracking_overlay_render import lift_tracks_yx_to_world, select_overlay_point_indices
from qqtt.tracking.backends.cotracker3_online import CoTracker3OnlineBackend
from qqtt.tracking.backends.point_tracker_adapter import (
    TRACKER_BACKEND_COTRACKER3,
    TRACKER_BATCH_QUERY_COUNT_POLICY_FIXED,
    TRACKER_BATCH_QUERY_COUNT_POLICY_MIN_COMMON,
    TRACKER_EXECUTION_MODE_AUTO,
    normalize_tracker_backend,
    normalize_tracker_batch_query_count_policy,
    normalize_tracker_execution_mode,
)
from qqtt.tracking.sampling import (
    PHYSTWIN_DENSE_QUERY_POINTS,
    sample_phystwin_dense,
)


BackendFactory = Callable[[int], Any]

OVERLAY_DISPLAY_SCOPE_CONTROLLER = "controller"
OVERLAY_DISPLAY_SCOPE_OBJECT = "object"
OVERLAY_DISPLAY_SCOPE_UNION = "union"
OVERLAY_DISPLAY_SCOPES = (
    OVERLAY_DISPLAY_SCOPE_CONTROLLER,
    OVERLAY_DISPLAY_SCOPE_OBJECT,
    OVERLAY_DISPLAY_SCOPE_UNION,
)
COTRACKER_UPDATE_MODE_AUTO = "auto"
COTRACKER_UPDATE_MODE_BATCH = "batch"
COTRACKER_UPDATE_MODE_SERIAL = "serial"
COTRACKER_UPDATE_MODES = (
    COTRACKER_UPDATE_MODE_AUTO,
    COTRACKER_UPDATE_MODE_BATCH,
    COTRACKER_UPDATE_MODE_SERIAL,
)


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
    mask_source_group_id: int | None = None
    mask_age_ms: float = 0.0
    mask_reused: bool = False

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
    query_is_object_by_camera: dict[int, np.ndarray] = field(default_factory=dict)
    query_is_controller_by_camera: dict[int, np.ndarray] = field(default_factory=dict)
    source_timestamp_s: float | None = None
    publish_range: tuple[int, int] = (0, 0)
    model_ms: float = 0.0
    e2e_ms: float = 0.0
    stale: bool = False
    cotracker_update_mode: str = COTRACKER_UPDATE_MODE_SERIAL
    cotracker_batch_size: int = 1
    cotracker_batch_update_count: int = 0
    cotracker_serial_group_update_count: int = 0
    cotracker_serial_camera_update_count: int = 0
    cotracker_serial_fallback_count: int = 0
    cotracker_batch_error_count: int = 0
    cotracker_batch_disabled_reason: str | None = None
    mask_source_group_id: int | None = None
    mask_age_ms: float = 0.0
    mask_reused: bool = False
    overlay_display_scope: str = OVERLAY_DISPLAY_SCOPE_CONTROLLER
    tracking_query_count_actual_by_camera: dict[int, int] = field(default_factory=dict)
    tracking_union_pixels_by_camera: dict[int, int] = field(default_factory=dict)
    tracking_object_pixels_by_camera: dict[int, int] = field(default_factory=dict)
    tracking_controller_pixels_by_camera: dict[int, int] = field(default_factory=dict)
    tracking_sample_object_hits_by_camera: dict[int, int] = field(default_factory=dict)
    tracking_sample_controller_hits_by_camera: dict[int, int] = field(default_factory=dict)
    tracking_sample_overlap_hits_by_camera: dict[int, int] = field(default_factory=dict)
    tracking_sample_background_hits_by_camera: dict[int, int] = field(default_factory=dict)
    overlay_display_count_by_camera: dict[int, int] = field(default_factory=dict)
    overlay_display_object_count_by_camera: dict[int, int] = field(default_factory=dict)
    overlay_display_controller_count_by_camera: dict[int, int] = field(default_factory=dict)
    tracker_backend: str = TRACKER_BACKEND_COTRACKER3
    tracking_backend_execution_mode: str = TRACKER_EXECUTION_MODE_AUTO
    tracker_batch_query_count_policy: str = TRACKER_BATCH_QUERY_COUNT_POLICY_FIXED
    tracking_backend_effective_query_count: int = 0
    tracking_backend_query_count_truncated_by_camera: dict[int, int] = field(default_factory=dict)
    tracking_backend_batch_fallback_reason: str | None = None

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
        overlay_display_scope: str = OVERLAY_DISPLAY_SCOPE_CONTROLLER,
        seed: int = 42,
        device: str = "cuda",
        sampling_device: str = "cpu",
        init_requires_object_and_controller: bool = True,
        update_mode: str = COTRACKER_UPDATE_MODE_AUTO,
        tracker_backend: str = TRACKER_BACKEND_COTRACKER3,
        backend_execution_mode: str = TRACKER_EXECUTION_MODE_AUTO,
        tracker_batch_query_count_policy: str = TRACKER_BATCH_QUERY_COUNT_POLICY_FIXED,
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
        self.overlay_display_scope = self._normalize_overlay_display_scope(overlay_display_scope)
        self.update_mode = self._normalize_update_mode(update_mode)
        self.tracker_backend = normalize_tracker_backend(tracker_backend)
        self.backend_execution_mode = normalize_tracker_execution_mode(backend_execution_mode)
        self.tracker_batch_query_count_policy = normalize_tracker_batch_query_count_policy(
            tracker_batch_query_count_policy
        )
        self.seed = int(seed)
        self.sampling_device = str(sampling_device)
        self.init_requires_object_and_controller = bool(init_requires_object_and_controller)
        self._backends: dict[int, Any] = {}
        self._batch_backend: Any | None = None
        self._batch_backend_signature: tuple[tuple[int, int], ...] | None = None
        self._batch_backend_disabled_reason: str | None = None
        self._query_points_yx: dict[int, np.ndarray] = {}
        self._query_is_object: dict[int, np.ndarray] = {}
        self._query_is_controller: dict[int, np.ndarray] = {}
        self._camera_stats: dict[int, dict[str, int | bool]] = {}
        self._overlay_display_count_by_camera: dict[int, int] = {}
        self._overlay_display_object_count_by_camera: dict[int, int] = {}
        self._overlay_display_controller_count_by_camera: dict[int, int] = {}
        self._published_packets = 0
        self._model_ms_samples: list[float] = []
        self._e2e_ms_samples: list[float] = []
        self._publish_times_s: list[float] = []
        self._backend_warmup_profile: dict[int, dict[str, Any]] = {}
        self._batch_warmup_profile: dict[str, Any] = {}
        self._batch_update_count = 0
        self._serial_camera_update_count = 0
        self._serial_group_update_count = 0
        self._serial_fallback_count = 0
        self._batch_error_count = 0
        self._last_batch_error: str | None = None
        self._last_update_mode = COTRACKER_UPDATE_MODE_SERIAL
        self._last_batch_size = 0
        self._last_batch_effective_query_count = 0
        self._last_batch_truncated_by_camera: dict[int, int] = {}
        if self.query_mode != "phystwin_dense":
            raise ValueError("CoTracker3OverlayWorker currently supports only query_mode='phystwin_dense'.")
        # Non-positive means "render all selected visible tracks". Demo 3.1 uses
        # this to show every controller-labeled CoTracker point by default.

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

    @staticmethod
    def _normalize_overlay_display_scope(value: str) -> str:
        normalized = str(value).strip().lower().replace("-", "_")
        if normalized not in OVERLAY_DISPLAY_SCOPES:
            raise ValueError(f"overlay_display_scope must be one of {OVERLAY_DISPLAY_SCOPES}; got {value!r}")
        return normalized

    @staticmethod
    def _normalize_update_mode(value: str) -> str:
        normalized = str(value).strip().lower().replace("_", "-")
        if normalized not in COTRACKER_UPDATE_MODES:
            raise ValueError(f"update_mode must be one of {COTRACKER_UPDATE_MODES}; got {value!r}")
        return normalized

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

    def _classify_query_points(
        self,
        *,
        points: np.ndarray,
        object_mask: np.ndarray,
        controller_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        query_points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if len(query_points) == 0:
            empty = np.empty((0,), dtype=bool)
            return empty, empty
        height, width = object_mask.shape[:2]
        yi = np.clip(np.rint(query_points[:, 0]).astype(np.int64), 0, height - 1)
        xi = np.clip(np.rint(query_points[:, 1]).astype(np.int64), 0, width - 1)
        query_is_object = np.asarray(object_mask[yi, xi], dtype=bool)
        query_is_controller = np.asarray(controller_mask[yi, xi], dtype=bool)
        return query_is_object, query_is_controller

    def _selection_visibility(self, *, visibility: np.ndarray, camera_idx: int) -> np.ndarray:
        vis = np.asarray(visibility, dtype=np.float32).reshape(-1)
        if self.overlay_display_scope == OVERLAY_DISPLAY_SCOPE_UNION:
            return vis
        if self.overlay_display_scope == OVERLAY_DISPLAY_SCOPE_OBJECT:
            label = self._query_is_object.get(int(camera_idx), np.zeros_like(vis, dtype=bool))
        else:
            label = self._query_is_controller.get(int(camera_idx), np.zeros_like(vis, dtype=bool))
        label_bool = np.asarray(label, dtype=bool).reshape(-1)
        if label_bool.shape[0] > vis.shape[0]:
            label_bool = label_bool[: vis.shape[0]]
        elif label_bool.shape[0] < vis.shape[0]:
            fitted = np.zeros_like(vis, dtype=bool)
            fitted[: label_bool.shape[0]] = label_bool
            label_bool = fitted
        return np.where(label_bool, vis, 0.0).astype(np.float32)

    def _query_labels_for_camera(self, *, camera_idx: int, query_count: int) -> tuple[np.ndarray, np.ndarray]:
        count = int(query_count)
        object_labels = np.asarray(self._query_is_object.get(int(camera_idx), ()), dtype=bool).reshape(-1)
        controller_labels = np.asarray(self._query_is_controller.get(int(camera_idx), ()), dtype=bool).reshape(-1)

        def fit(labels: np.ndarray) -> np.ndarray:
            if labels.shape[0] >= count:
                return labels[:count].astype(bool, copy=False)
            fitted = np.zeros(count, dtype=bool)
            fitted[: labels.shape[0]] = labels
            return fitted

        return fit(object_labels), fit(controller_labels)

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

    def warmup_backends(self) -> dict[str, Any]:
        started_s = time.perf_counter()
        if self.update_mode != COTRACKER_UPDATE_MODE_SERIAL:
            construct_start_s = time.perf_counter()
            backend = self._batch_backend
            construct_ms = 0.0
            if backend is None:
                backend = self.backend_factory(-1)
                construct_ms = float((time.perf_counter() - construct_start_s) * 1000.0)
                self._batch_backend = backend
            warmup_start_s = time.perf_counter()
            warmup_stats: dict[str, Any] = {}
            if hasattr(backend, "warmup"):
                result = backend.warmup()
                if isinstance(result, dict):
                    warmup_stats.update(result)
            self._batch_warmup_profile = {
                "construct_ms": float(construct_ms),
                "warmup_ms": float((time.perf_counter() - warmup_start_s) * 1000.0),
                "supports_batch_update": bool(
                    hasattr(backend, "initialize_batch")
                    and hasattr(backend, "update_batch")
                ),
                **warmup_stats,
            }
            return {
                "camera_ids": [int(item) for item in self.camera_ids],
                "update_mode": str(self.update_mode),
                "batch": dict(self._batch_warmup_profile),
                "per_camera": {},
                "total_ms": float((time.perf_counter() - started_s) * 1000.0),
            }
        per_camera: dict[int, dict[str, Any]] = {}
        for camera_idx in self.camera_ids:
            idx = int(camera_idx)
            backend = self._backends.get(idx)
            construct_ms = 0.0
            if backend is None:
                construct_start_s = time.perf_counter()
                backend = self.backend_factory(idx)
                construct_ms = float((time.perf_counter() - construct_start_s) * 1000.0)
                self._backends[idx] = backend
            warmup_start_s = time.perf_counter()
            warmup_stats: dict[str, Any] = {}
            if hasattr(backend, "warmup"):
                result = backend.warmup()
                if isinstance(result, dict):
                    warmup_stats.update(result)
            per_camera[idx] = {
                "construct_ms": float(construct_ms),
                "warmup_ms": float((time.perf_counter() - warmup_start_s) * 1000.0),
                **warmup_stats,
            }
        self._backend_warmup_profile = per_camera
        return {
            "camera_ids": [int(item) for item in self.camera_ids],
            "update_mode": str(self.update_mode),
            "per_camera": {str(idx): dict(value) for idx, value in per_camera.items()},
            "total_ms": float((time.perf_counter() - started_s) * 1000.0),
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
        query_is_object, query_is_controller = self._classify_query_points(
            points=query_points,
            object_mask=object_mask,
            controller_mask=controller_mask,
        )
        self._query_is_object[camera_idx] = query_is_object
        self._query_is_controller[camera_idx] = query_is_controller
        return query_points

    def _ensure_serial_backend(self, camera_idx: int, query_points: np.ndarray) -> Any:
        backend = self._backends.get(int(camera_idx))
        if backend is None:
            backend = self.backend_factory(int(camera_idx))
            backend.initialize([], query_points)
            self._backends[int(camera_idx)] = backend
        return backend

    def _supports_batch_backend(self, backend: Any) -> bool:
        return bool(hasattr(backend, "initialize_batch") and hasattr(backend, "update_batch"))

    def _batch_signature(self, query_points_by_camera: Mapping[int, np.ndarray]) -> tuple[tuple[int, int], ...]:
        return tuple(
            (int(camera_idx), int(len(np.asarray(query_points_by_camera[int(camera_idx)]).reshape(-1, 2))))
            for camera_idx in sorted(query_points_by_camera)
        )

    def _ensure_batch_backend(self, query_points_by_camera: Mapping[int, np.ndarray]) -> Any | None:
        if self._batch_backend_disabled_reason is not None:
            return None
        backend = self._batch_backend
        if backend is None:
            backend = self.backend_factory(-1)
            self._batch_backend = backend
        if not self._supports_batch_backend(backend):
            self._batch_backend_disabled_reason = "backend does not implement initialize_batch/update_batch"
            return None
        signature = self._batch_signature(query_points_by_camera)
        if signature != self._batch_backend_signature:
            backend.initialize_batch(query_points_by_camera)
            self._batch_backend_signature = signature
        return backend

    def _clear_cuda_cache_after_batch_error(self) -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            return

    def _batch_preferred(self) -> bool:
        return self.update_mode in {COTRACKER_UPDATE_MODE_AUTO, COTRACKER_UPDATE_MODE_BATCH} and len(self.camera_ids) > 1

    def _process_batch_updates(
        self,
        active_inputs: Mapping[int, dict[str, Any]],
    ) -> dict[int, Any] | None:
        query_points_by_camera = {
            int(camera_idx): np.asarray(payload["query_points"], dtype=np.float32)
            for camera_idx, payload in active_inputs.items()
        }
        backend = self._ensure_batch_backend(query_points_by_camera)
        if backend is None:
            return None
        frames_by_camera = {
            int(camera_idx): np.asarray(payload["frame"], dtype=np.uint8)
            for camera_idx, payload in active_inputs.items()
        }
        results = backend.update_batch(frames_by_camera)
        self._batch_update_count += 1
        self._last_update_mode = COTRACKER_UPDATE_MODE_BATCH
        self._last_batch_size = int(len(active_inputs))
        return {int(camera_idx): result for camera_idx, result in results.items()}

    def _batch_policy_inputs(
        self,
        active_inputs: Mapping[int, dict[str, Any]],
    ) -> dict[int, dict[str, Any]]:
        counts = {
            int(camera_idx): int(len(np.asarray(payload["query_points"], dtype=np.float32).reshape(-1, 2)))
            for camera_idx, payload in active_inputs.items()
        }
        if not counts:
            self._last_batch_effective_query_count = 0
            self._last_batch_truncated_by_camera = {}
            return {int(camera_idx): dict(payload) for camera_idx, payload in active_inputs.items()}
        if self.tracker_batch_query_count_policy != TRACKER_BATCH_QUERY_COUNT_POLICY_MIN_COMMON:
            self._last_batch_effective_query_count = max(counts.values())
            self._last_batch_truncated_by_camera = {int(camera_idx): 0 for camera_idx in counts}
            return {int(camera_idx): dict(payload) for camera_idx, payload in active_inputs.items()}

        effective = min(counts.values())
        self._last_batch_effective_query_count = int(effective)
        truncated: dict[int, int] = {}
        policy_inputs: dict[int, dict[str, Any]] = {}
        for camera_idx, payload in active_inputs.items():
            idx = int(camera_idx)
            points = np.asarray(payload["query_points"], dtype=np.float32).reshape(-1, 2)
            truncated[idx] = max(0, int(len(points) - effective))
            copied = dict(payload)
            copied["query_points"] = points[:effective]
            policy_inputs[idx] = copied
        self._last_batch_truncated_by_camera = truncated
        return policy_inputs

    def _process_serial_updates(
        self,
        active_inputs: Mapping[int, dict[str, Any]],
    ) -> dict[int, Any]:
        results: dict[int, Any] = {}
        for camera_idx, payload in active_inputs.items():
            idx = int(camera_idx)
            query_points = np.asarray(payload["query_points"], dtype=np.float32)
            backend = self._ensure_serial_backend(idx, query_points)
            results[idx] = backend.update(np.asarray(payload["frame"], dtype=np.uint8))
            self._serial_camera_update_count += 1
        if active_inputs:
            self._serial_group_update_count += 1
        self._last_update_mode = COTRACKER_UPDATE_MODE_SERIAL
        self._last_batch_size = 1 if active_inputs else 0
        return results

    def process_group(self, packet: TrackingOverlayInputPacket) -> TrackingOverlayPacket | None:
        started_s = time.perf_counter()
        camera_tracks_yx: dict[int, np.ndarray] = {}
        camera_visibility: dict[int, np.ndarray] = {}
        camera_tracks_world: dict[int, np.ndarray] = {}
        query_points_yx: dict[int, np.ndarray] = {}
        query_is_object_by_camera: dict[int, np.ndarray] = {}
        query_is_controller_by_camera: dict[int, np.ndarray] = {}
        publish_starts: list[int] = []
        publish_ends: list[int] = []
        model_ms = 0.0
        active_inputs: dict[int, dict[str, Any]] = {}

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
            active_inputs[int(camera_idx)] = {
                "frame": np.asarray(frame, dtype=np.uint8),
                "union_mask": union_mask,
                "query_points": query_points,
            }

        if not active_inputs:
            return None

        results: dict[int, Any] | None = None
        batch_attempt_failed = False
        if self._batch_preferred() and self._batch_backend_disabled_reason is None:
            if len(active_inputs) == len(self.camera_ids):
                batch_inputs = self._batch_policy_inputs(active_inputs)
                try:
                    results = self._process_batch_updates(batch_inputs)
                    if results is not None:
                        active_inputs = batch_inputs
                except BaseException as exc:
                    batch_attempt_failed = True
                    self._batch_error_count += 1
                    self._last_batch_error = f"{type(exc).__name__}: {exc}"
                    self._batch_backend_disabled_reason = self._last_batch_error
                    self._clear_cuda_cache_after_batch_error()
                    if self.update_mode == COTRACKER_UPDATE_MODE_BATCH:
                        raise
                    self._serial_fallback_count += 1
            elif self.update_mode == COTRACKER_UPDATE_MODE_AUTO:
                return None
            elif self.update_mode == COTRACKER_UPDATE_MODE_BATCH:
                return None
        if results is None:
            if self.update_mode == COTRACKER_UPDATE_MODE_BATCH:
                return None
            if self._batch_preferred() and self._batch_backend_disabled_reason is not None and not batch_attempt_failed:
                self._serial_fallback_count += 1
            results = self._process_serial_updates(active_inputs)

        counted_batch_model_ms = False
        for camera_idx in self.camera_ids:
            if camera_idx not in results or camera_idx not in active_inputs:
                continue
            result = results[int(camera_idx)]
            if str(result.stats.get("stream_status", "")) != "published" or result.tracks_yx.shape[0] == 0:
                continue
            query_points = np.asarray(active_inputs[int(camera_idx)]["query_points"], dtype=np.float32)
            union_mask = np.asarray(active_inputs[int(camera_idx)]["union_mask"], dtype=bool)
            tracks_t = np.asarray(result.tracks_yx[-1], dtype=np.float32)
            visibility_t = np.asarray(result.visibility[-1], dtype=np.float32).reshape(-1)
            selection_visibility = self._selection_visibility(visibility=visibility_t, camera_idx=int(camera_idx))
            overlay_cap = None if self.overlay_max_points_per_camera <= 0 else self.overlay_max_points_per_camera
            selected = select_overlay_point_indices(
                selection_visibility,
                max_points=overlay_cap,
            )
            if len(selected) == 0:
                continue
            camera_tracks_yx[camera_idx] = tracks_t[selected]
            camera_visibility[camera_idx] = visibility_t[selected]
            query_points_yx[camera_idx] = query_points[selected]
            is_object, is_controller = self._query_labels_for_camera(
                camera_idx=int(camera_idx),
                query_count=len(query_points),
            )
            selected_is_object = is_object[selected]
            selected_is_controller = is_controller[selected]
            query_is_object_by_camera[camera_idx] = selected_is_object
            query_is_controller_by_camera[camera_idx] = selected_is_controller
            self._overlay_display_count_by_camera[int(camera_idx)] = int(len(selected))
            self._overlay_display_object_count_by_camera[int(camera_idx)] = int(np.count_nonzero(selected_is_object))
            self._overlay_display_controller_count_by_camera[int(camera_idx)] = int(np.count_nonzero(selected_is_controller))
            if str(result.stats.get("update_mode", "")) == COTRACKER_UPDATE_MODE_BATCH:
                if not counted_batch_model_ms:
                    model_ms += float(result.stats.get("model_run_ms", 0.0))
                    counted_batch_model_ms = True
            else:
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
            query_is_object_by_camera=query_is_object_by_camera,
            query_is_controller_by_camera=query_is_controller_by_camera,
            source_timestamp_s=float(packet.timestamp_s),
            publish_range=(min(publish_starts), max(publish_ends)),
            model_ms=float(model_ms),
            e2e_ms=float(e2e_ms),
            stale=False,
            cotracker_update_mode=str(self._last_update_mode),
            cotracker_batch_size=int(self._last_batch_size),
            cotracker_batch_update_count=int(self._batch_update_count),
            cotracker_serial_group_update_count=int(self._serial_group_update_count),
            cotracker_serial_camera_update_count=int(self._serial_camera_update_count),
            cotracker_serial_fallback_count=int(self._serial_fallback_count),
            cotracker_batch_error_count=int(self._batch_error_count),
            cotracker_batch_disabled_reason=self._batch_backend_disabled_reason,
            mask_source_group_id=(
                None if packet.mask_source_group_id is None else int(packet.mask_source_group_id)
            ),
            mask_age_ms=float(packet.mask_age_ms),
            mask_reused=bool(packet.mask_reused),
            overlay_display_scope=str(self.overlay_display_scope),
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
            overlay_display_object_count_by_camera=dict(self._overlay_display_object_count_by_camera),
            overlay_display_controller_count_by_camera=dict(self._overlay_display_controller_count_by_camera),
            tracker_backend=str(self.tracker_backend),
            tracking_backend_execution_mode=str(self.backend_execution_mode),
            tracker_batch_query_count_policy=str(self.tracker_batch_query_count_policy),
            tracking_backend_effective_query_count=int(self._last_batch_effective_query_count),
            tracking_backend_query_count_truncated_by_camera=dict(self._last_batch_truncated_by_camera),
            tracking_backend_batch_fallback_reason=self._batch_backend_disabled_reason,
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
            "tracker_backend": str(self.tracker_backend),
            "tracking_backend_execution_mode": str(self.backend_execution_mode),
            "tracker_batch_query_count_policy": str(self.tracker_batch_query_count_policy),
            "tracking_backend_effective_query_count": int(self._last_batch_effective_query_count),
            "tracking_backend_query_count_truncated_by_camera": dict(self._last_batch_truncated_by_camera),
            "tracking_backend_batch_fallback_reason": self._batch_backend_disabled_reason,
            "query_mode": str(self.query_mode),
            "query_count_request": str(self.query_count),
            "query_count": int(PHYSTWIN_DENSE_QUERY_POINTS if self.query_count == "auto" else self.query_count),
            "cotracker_update_mode_requested": str(self.update_mode),
            "cotracker_update_mode_effective": str(self._last_update_mode),
            "cotracker_batch_size": int(self._last_batch_size),
            "cotracker_batch_update_count": int(self._batch_update_count),
            "cotracker_serial_group_update_count": int(self._serial_group_update_count),
            "cotracker_serial_camera_update_count": int(self._serial_camera_update_count),
            "cotracker_serial_fallback_count": int(self._serial_fallback_count),
            "cotracker_batch_error_count": int(self._batch_error_count),
            "cotracker_batch_disabled_reason": self._batch_backend_disabled_reason,
            "cotracker_last_batch_error": self._last_batch_error,
            "overlay_display_scope": str(self.overlay_display_scope),
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
            "overlay_display_object_count_by_camera": {
                int(camera_idx): int(count)
                for camera_idx, count in self._overlay_display_object_count_by_camera.items()
            },
            "overlay_display_controller_count_by_camera": {
                int(camera_idx): int(count)
                for camera_idx, count in self._overlay_display_controller_count_by_camera.items()
            },
            "overlay_max_points_per_camera": int(self.overlay_max_points_per_camera),
            "backend_warmup": {
                str(camera_idx): dict(profile)
                for camera_idx, profile in self._backend_warmup_profile.items()
            },
            "batch_backend_warmup": dict(self._batch_warmup_profile),
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
    "COTRACKER_UPDATE_MODE_AUTO",
    "COTRACKER_UPDATE_MODE_BATCH",
    "COTRACKER_UPDATE_MODE_SERIAL",
    "COTRACKER_UPDATE_MODES",
    "CoTracker3OverlayThread",
    "CoTracker3OverlayWorker",
    "LatestTrackingInputSlot",
    "LatestTrackingOverlaySlot",
    "OVERLAY_DISPLAY_SCOPE_CONTROLLER",
    "OVERLAY_DISPLAY_SCOPE_OBJECT",
    "OVERLAY_DISPLAY_SCOPE_UNION",
    "OVERLAY_DISPLAY_SCOPES",
    "TrackingOverlayInputPacket",
    "TrackingOverlayPacket",
]
