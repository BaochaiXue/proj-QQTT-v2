from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import queue
import time
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TrackingInputLitePacket:
    group_id: int
    frame_idx: int
    timestamp_s: float
    rgb_by_camera: Mapping[int, np.ndarray]
    mask_by_camera: Mapping[int, np.ndarray]
    object_mask_by_camera: Mapping[int, np.ndarray]
    controller_mask_by_camera: Mapping[int, np.ndarray]

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
            publish_range=tuple(int(item) for item in packet.publish_range),
            model_ms=float(packet.model_ms),
            e2e_ms=float(packet.e2e_ms),
            stale=bool(getattr(packet, "stale", False)),
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
            publish_range=self.publish_range,
            model_ms=self.model_ms,
            e2e_ms=self.e2e_ms,
            stale=True,
            tracking_query_count_actual_by_camera=self.tracking_query_count_actual_by_camera,
            tracking_union_pixels_by_camera=self.tracking_union_pixels_by_camera,
            tracking_object_pixels_by_camera=self.tracking_object_pixels_by_camera,
            tracking_controller_pixels_by_camera=self.tracking_controller_pixels_by_camera,
            tracking_sample_object_hits_by_camera=self.tracking_sample_object_hits_by_camera,
            tracking_sample_controller_hits_by_camera=self.tracking_sample_controller_hits_by_camera,
            tracking_sample_overlap_hits_by_camera=self.tracking_sample_overlap_hits_by_camera,
            tracking_sample_background_hits_by_camera=self.tracking_sample_background_hits_by_camera,
            overlay_display_count_by_camera=self.overlay_display_count_by_camera,
        )


class LatestWinsQueue:
    """CPU latest-wins queue wrapper for non-blocking process IPC."""

    def __init__(self, queue_obj: Any | None = None) -> None:
        self.queue = queue_obj if queue_obj is not None else queue.Queue(maxsize=1)
        self.published = 0
        self.taken = 0
        self.replaced = 0
        self.put_failures = 0

    def publish_latest(self, item: Any) -> int:
        replaced = self._drain()
        try:
            self.queue.put_nowait(item)
        except queue.Full:
            replaced += self._drain()
            try:
                self.queue.put_nowait(item)
            except queue.Full:
                self.put_failures += 1
                return replaced
        self.published += 1
        self.replaced += replaced
        return replaced

    def take_latest(self) -> Any | None:
        latest = None
        drained = 0
        while True:
            try:
                latest = self.queue.get_nowait()
                drained += 1
            except queue.Empty:
                break
        if drained:
            self.taken += 1
            self.replaced += max(0, drained - 1)
        return latest

    def snapshot(self) -> dict[str, int]:
        return {
            "published": int(self.published),
            "taken": int(self.taken),
            "replaced": int(self.replaced),
            "put_failures": int(self.put_failures),
        }

    def _drain(self) -> int:
        count = 0
        while True:
            try:
                self.queue.get_nowait()
                count += 1
            except queue.Empty:
                break
        return count


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
