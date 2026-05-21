from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import time
from typing import Any

import numpy as np

from qqtt.demo.services.latest_wins import LatestWinsQueue


@dataclass(frozen=True)
class TrackingInputLitePacket:
    group_id: int
    frame_idx: int
    timestamp_s: float
    rgb_by_camera: Mapping[int, np.ndarray]
    mask_by_camera: Mapping[int, np.ndarray]
    object_mask_by_camera: Mapping[int, np.ndarray]
    controller_mask_by_camera: Mapping[int, np.ndarray]
    mask_source_group_id: int | None = None
    mask_age_ms: float = 0.0
    mask_reused: bool = False

    @property
    def seq(self) -> int:
        return int(self.group_id)

    def to_overlay_input_packet(self):
        from qqtt.demo.cotracker3_overlay_worker import TrackingOverlayInputPacket

        return TrackingOverlayInputPacket(
            group_id=int(self.group_id),
            frame_idx=int(self.frame_idx),
            timestamp_s=float(self.timestamp_s),
            rgb_by_camera={
                int(camera_idx): np.ascontiguousarray(np.asarray(frame, dtype=np.uint8))
                for camera_idx, frame in self.rgb_by_camera.items()
            },
            mask_by_camera={
                int(camera_idx): np.ascontiguousarray(np.asarray(mask, dtype=bool))
                for camera_idx, mask in self.mask_by_camera.items()
            },
            object_mask_by_camera={
                int(camera_idx): np.ascontiguousarray(np.asarray(mask, dtype=bool))
                for camera_idx, mask in self.object_mask_by_camera.items()
            },
            controller_mask_by_camera={
                int(camera_idx): np.ascontiguousarray(np.asarray(mask, dtype=bool))
                for camera_idx, mask in self.controller_mask_by_camera.items()
            },
            mask_source_group_id=self.mask_source_group_id,
            mask_age_ms=float(self.mask_age_ms),
            mask_reused=bool(self.mask_reused),
        )


@dataclass(frozen=True)
class TrackingResultLitePacket:
    group_id: int
    frame_idx: int
    source_timestamp_s: float
    publish_timestamp_s: float
    camera_tracks_yx: dict[int, np.ndarray]
    camera_visibility: dict[int, np.ndarray]
    query_points_yx: dict[int, np.ndarray]
    publish_range: tuple[int, int]
    query_is_object_by_camera: dict[int, np.ndarray] = field(default_factory=dict)
    query_is_controller_by_camera: dict[int, np.ndarray] = field(default_factory=dict)
    model_ms: float = 0.0
    e2e_ms: float = 0.0
    stale: bool = False
    cotracker_update_mode: str = "batch"
    cotracker_batch_size: int = 3
    cotracker_batch_update_count: int = 0
    cotracker_serial_group_update_count: int = 0
    cotracker_serial_camera_update_count: int = 0
    cotracker_serial_fallback_count: int = 0
    cotracker_batch_error_count: int = 0
    cotracker_batch_disabled_reason: str | None = None
    mask_source_group_id: int | None = None
    mask_age_ms: float = 0.0
    mask_reused: bool = False
    overlay_display_scope: str = "controller"
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
    tracker_backend: str = "cotracker3_online"
    tracking_backend_execution_mode: str = "batch-views"
    tracker_batch_query_count_policy: str = "fixed"
    tracking_backend_effective_query_count: int = 0
    tracking_backend_query_count_truncated_by_camera: dict[int, int] = field(default_factory=dict)
    tracking_backend_batch_fallback_reason: str | None = None
    tracker_group_wall_ms: float = 0.0
    tracker_model_ms_sum_per_group: float = 0.0
    tracker_model_ms_max_per_group: float = 0.0
    per_camera_model_ms_by_camera: dict[int, float] = field(default_factory=dict)
    model_calls_per_group: int = 0
    model_instances_expected: int = 0
    model_instances_actual: int = 0
    query_count_per_camera: int = 0
    total_query_count_across_views: int = 0

    @property
    def seq(self) -> int:
        return int(self.group_id)

    @classmethod
    def from_overlay_packet(cls, packet: Any) -> "TrackingResultLitePacket":
        return cls(
            group_id=int(packet.group_id),
            frame_idx=int(packet.frame_idx),
            source_timestamp_s=float(packet.source_timestamp_s or packet.timestamp_s),
            publish_timestamp_s=float(packet.timestamp_s),
            camera_tracks_yx={
                int(camera_idx): np.asarray(tracks, dtype=np.float32)
                for camera_idx, tracks in packet.camera_tracks_yx.items()
            },
            camera_visibility={
                int(camera_idx): np.asarray(visibility, dtype=np.float32)
                for camera_idx, visibility in packet.camera_visibility.items()
            },
            query_points_yx={
                int(camera_idx): np.asarray(points, dtype=np.float32)
                for camera_idx, points in packet.query_points_yx.items()
            },
            query_is_object_by_camera={
                int(camera_idx): np.asarray(labels, dtype=bool)
                for camera_idx, labels in getattr(packet, "query_is_object_by_camera", {}).items()
            },
            query_is_controller_by_camera={
                int(camera_idx): np.asarray(labels, dtype=bool)
                for camera_idx, labels in getattr(packet, "query_is_controller_by_camera", {}).items()
            },
            publish_range=tuple(int(item) for item in packet.publish_range),
            model_ms=float(packet.model_ms),
            e2e_ms=float(packet.e2e_ms),
            stale=bool(getattr(packet, "stale", False)),
            cotracker_update_mode=str(getattr(packet, "cotracker_update_mode", "batch")),
            cotracker_batch_size=int(getattr(packet, "cotracker_batch_size", 1) or 1),
            cotracker_batch_update_count=int(getattr(packet, "cotracker_batch_update_count", 0) or 0),
            cotracker_serial_group_update_count=int(
                getattr(packet, "cotracker_serial_group_update_count", 0) or 0
            ),
            cotracker_serial_camera_update_count=int(
                getattr(packet, "cotracker_serial_camera_update_count", 0) or 0
            ),
            cotracker_serial_fallback_count=int(getattr(packet, "cotracker_serial_fallback_count", 0) or 0),
            cotracker_batch_error_count=int(getattr(packet, "cotracker_batch_error_count", 0) or 0),
            cotracker_batch_disabled_reason=getattr(packet, "cotracker_batch_disabled_reason", None),
            mask_source_group_id=(
                None
                if getattr(packet, "mask_source_group_id", None) is None
                else int(getattr(packet, "mask_source_group_id"))
            ),
            mask_age_ms=float(getattr(packet, "mask_age_ms", 0.0) or 0.0),
            mask_reused=bool(getattr(packet, "mask_reused", False)),
            overlay_display_scope=str(getattr(packet, "overlay_display_scope", "controller")),
            tracking_query_count_actual_by_camera={
                int(camera_idx): int(count)
                for camera_idx, count in getattr(packet, "tracking_query_count_actual_by_camera", {}).items()
            },
            tracking_union_pixels_by_camera={
                int(camera_idx): int(count)
                for camera_idx, count in getattr(packet, "tracking_union_pixels_by_camera", {}).items()
            },
            tracking_object_pixels_by_camera={
                int(camera_idx): int(count)
                for camera_idx, count in getattr(packet, "tracking_object_pixels_by_camera", {}).items()
            },
            tracking_controller_pixels_by_camera={
                int(camera_idx): int(count)
                for camera_idx, count in getattr(packet, "tracking_controller_pixels_by_camera", {}).items()
            },
            tracking_sample_object_hits_by_camera={
                int(camera_idx): int(count)
                for camera_idx, count in getattr(packet, "tracking_sample_object_hits_by_camera", {}).items()
            },
            tracking_sample_controller_hits_by_camera={
                int(camera_idx): int(count)
                for camera_idx, count in getattr(packet, "tracking_sample_controller_hits_by_camera", {}).items()
            },
            tracking_sample_overlap_hits_by_camera={
                int(camera_idx): int(count)
                for camera_idx, count in getattr(packet, "tracking_sample_overlap_hits_by_camera", {}).items()
            },
            tracking_sample_background_hits_by_camera={
                int(camera_idx): int(count)
                for camera_idx, count in getattr(packet, "tracking_sample_background_hits_by_camera", {}).items()
            },
            overlay_display_count_by_camera={
                int(camera_idx): int(count)
                for camera_idx, count in getattr(packet, "overlay_display_count_by_camera", {}).items()
            },
            overlay_display_object_count_by_camera={
                int(camera_idx): int(count)
                for camera_idx, count in getattr(packet, "overlay_display_object_count_by_camera", {}).items()
            },
            overlay_display_controller_count_by_camera={
                int(camera_idx): int(count)
                for camera_idx, count in getattr(packet, "overlay_display_controller_count_by_camera", {}).items()
            },
            tracker_backend=str(getattr(packet, "tracker_backend", "cotracker3_online")),
            tracking_backend_execution_mode=str(getattr(packet, "tracking_backend_execution_mode", "batch-views")),
            tracker_batch_query_count_policy=str(getattr(packet, "tracker_batch_query_count_policy", "fixed")),
            tracking_backend_effective_query_count=int(getattr(packet, "tracking_backend_effective_query_count", 0)),
            tracking_backend_query_count_truncated_by_camera={
                int(camera_idx): int(count)
                for camera_idx, count in getattr(
                    packet,
                    "tracking_backend_query_count_truncated_by_camera",
                    {},
                ).items()
            },
            tracking_backend_batch_fallback_reason=getattr(packet, "tracking_backend_batch_fallback_reason", None),
            tracker_group_wall_ms=float(getattr(packet, "tracker_group_wall_ms", 0.0) or 0.0),
            tracker_model_ms_sum_per_group=float(
                getattr(packet, "tracker_model_ms_sum_per_group", getattr(packet, "model_ms", 0.0)) or 0.0
            ),
            tracker_model_ms_max_per_group=float(
                getattr(packet, "tracker_model_ms_max_per_group", getattr(packet, "model_ms", 0.0)) or 0.0
            ),
            per_camera_model_ms_by_camera={
                int(camera_idx): float(value)
                for camera_idx, value in getattr(packet, "per_camera_model_ms_by_camera", {}).items()
            },
            model_calls_per_group=int(getattr(packet, "model_calls_per_group", 0) or 0),
            model_instances_expected=int(getattr(packet, "model_instances_expected", 0) or 0),
            model_instances_actual=int(getattr(packet, "model_instances_actual", 0) or 0),
            query_count_per_camera=int(getattr(packet, "query_count_per_camera", 0) or 0),
            total_query_count_across_views=int(getattr(packet, "total_query_count_across_views", 0) or 0),
        )

    def mark_stale(self) -> "TrackingResultLitePacket":
        return TrackingResultLitePacket(
            group_id=self.group_id,
            frame_idx=self.frame_idx,
            source_timestamp_s=self.source_timestamp_s,
            publish_timestamp_s=self.publish_timestamp_s,
            camera_tracks_yx=self.camera_tracks_yx,
            camera_visibility=self.camera_visibility,
            query_points_yx=self.query_points_yx,
            query_is_object_by_camera=self.query_is_object_by_camera,
            query_is_controller_by_camera=self.query_is_controller_by_camera,
            publish_range=self.publish_range,
            model_ms=self.model_ms,
            e2e_ms=self.e2e_ms,
            stale=True,
            cotracker_update_mode=self.cotracker_update_mode,
            cotracker_batch_size=self.cotracker_batch_size,
            cotracker_batch_update_count=self.cotracker_batch_update_count,
            cotracker_serial_group_update_count=self.cotracker_serial_group_update_count,
            cotracker_serial_camera_update_count=self.cotracker_serial_camera_update_count,
            cotracker_serial_fallback_count=self.cotracker_serial_fallback_count,
            cotracker_batch_error_count=self.cotracker_batch_error_count,
            cotracker_batch_disabled_reason=self.cotracker_batch_disabled_reason,
            mask_source_group_id=self.mask_source_group_id,
            mask_age_ms=self.mask_age_ms,
            mask_reused=self.mask_reused,
            overlay_display_scope=self.overlay_display_scope,
            tracking_query_count_actual_by_camera=self.tracking_query_count_actual_by_camera,
            tracking_union_pixels_by_camera=self.tracking_union_pixels_by_camera,
            tracking_object_pixels_by_camera=self.tracking_object_pixels_by_camera,
            tracking_controller_pixels_by_camera=self.tracking_controller_pixels_by_camera,
            tracking_sample_object_hits_by_camera=self.tracking_sample_object_hits_by_camera,
            tracking_sample_controller_hits_by_camera=self.tracking_sample_controller_hits_by_camera,
            tracking_sample_overlap_hits_by_camera=self.tracking_sample_overlap_hits_by_camera,
            tracking_sample_background_hits_by_camera=self.tracking_sample_background_hits_by_camera,
            overlay_display_count_by_camera=self.overlay_display_count_by_camera,
            overlay_display_object_count_by_camera=self.overlay_display_object_count_by_camera,
            overlay_display_controller_count_by_camera=self.overlay_display_controller_count_by_camera,
            tracker_backend=self.tracker_backend,
            tracking_backend_execution_mode=self.tracking_backend_execution_mode,
            tracker_batch_query_count_policy=self.tracker_batch_query_count_policy,
            tracking_backend_effective_query_count=self.tracking_backend_effective_query_count,
            tracking_backend_query_count_truncated_by_camera=dict(
                self.tracking_backend_query_count_truncated_by_camera
            ),
            tracking_backend_batch_fallback_reason=self.tracking_backend_batch_fallback_reason,
            tracker_group_wall_ms=self.tracker_group_wall_ms,
            tracker_model_ms_sum_per_group=self.tracker_model_ms_sum_per_group,
            tracker_model_ms_max_per_group=self.tracker_model_ms_max_per_group,
            per_camera_model_ms_by_camera=dict(self.per_camera_model_ms_by_camera),
            model_calls_per_group=self.model_calls_per_group,
            model_instances_expected=self.model_instances_expected,
            model_instances_actual=self.model_instances_actual,
            query_count_per_camera=self.query_count_per_camera,
            total_query_count_across_views=self.total_query_count_across_views,
        )


def should_publish_tracking_input(
    *,
    now_s: float,
    last_publish_s: float | None,
    target_fps: float,
) -> bool:
    if float(target_fps) <= 0.0:
        return False
    if last_publish_s is None:
        return True
    return float(now_s) - float(last_publish_s) >= 1.0 / float(target_fps)


@dataclass(frozen=True)
class LatestMaskSelection:
    group_id: int
    source_group_id: int
    age_ms: float
    reused: bool
    mask_by_camera: Mapping[int, Any]


class LatestMaskCache:
    """Small policy helper for strict vs latest-reuse mask fusion."""

    def __init__(self) -> None:
        self._latest_group_id: int | None = None
        self._latest_timestamp_s: float | None = None
        self._latest_masks: Mapping[int, Any] | None = None
        self.selection_count = 0
        self.reuse_count = 0
        self.stale_reject_count = 0
        self.age_ms_samples: list[float] = []

    def publish(self, *, group_id: int, timestamp_s: float | None = None, mask_by_camera: Mapping[int, Any]) -> None:
        self._latest_group_id = int(group_id)
        self._latest_timestamp_s = time.perf_counter() if timestamp_s is None else float(timestamp_s)
        self._latest_masks = mask_by_camera

    def select(
        self,
        *,
        group_id: int,
        now_s: float | None = None,
        policy: str = "latest-reuse",
        stale_timeout_ms: float = 250.0,
    ) -> LatestMaskSelection | None:
        if self._latest_group_id is None or self._latest_timestamp_s is None or self._latest_masks is None:
            return None
        normalized = str(policy).strip().lower().replace("_", "-")
        if normalized not in {"strict", "latest-reuse"}:
            raise ValueError(f"Unsupported fusion mask policy: {policy}")
        if normalized == "strict" and int(self._latest_group_id) != int(group_id):
            return None
        now = time.perf_counter() if now_s is None else float(now_s)
        age_ms = max(0.0, (now - float(self._latest_timestamp_s)) * 1000.0)
        if age_ms > float(stale_timeout_ms):
            self.stale_reject_count += 1
            return None
        reused = int(self._latest_group_id) != int(group_id)
        self.selection_count += 1
        self.reuse_count += int(reused)
        self.age_ms_samples.append(float(age_ms))
        return LatestMaskSelection(
            group_id=int(group_id),
            source_group_id=int(self._latest_group_id),
            age_ms=float(age_ms),
            reused=bool(reused),
            mask_by_camera=self._latest_masks,
        )

    def snapshot(self) -> dict[str, Any]:
        if self.age_ms_samples:
            arr = np.asarray(self.age_ms_samples, dtype=np.float32)
            median = float(np.median(arr))
            p95 = float(np.percentile(arr, 95))
        else:
            median = p95 = 0.0
        return {
            "selection_count": int(self.selection_count),
            "reuse_count": int(self.reuse_count),
            "stale_reject_count": int(self.stale_reject_count),
            "mask_reuse_ratio": float(self.reuse_count / self.selection_count) if self.selection_count else 0.0,
            "mask_age_ms_median": median,
            "mask_age_ms_p95": p95,
        }


__all__ = [
    "LatestMaskCache",
    "LatestMaskSelection",
    "LatestWinsQueue",
    "TrackingInputLitePacket",
    "TrackingResultLitePacket",
    "should_publish_tracking_input",
]
