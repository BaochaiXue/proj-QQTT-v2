from __future__ import annotations

from dataclasses import dataclass, field
import threading
import time
from typing import Any

import numpy as np

from qqtt.demo.phystwin_volume_filter import (
    DEFAULT_PHYSTWIN_OBJECT_VOLUME_EMERGENCY_MAX_POINTS,
    DEFAULT_PHYSTWIN_OBJECT_VOLUME_MAX_VOXEL_M,
    DEFAULT_PHYSTWIN_OBJECT_VOLUME_MIN_VOXEL_M,
    DEFAULT_PHYSTWIN_OBJECT_VOLUME_POINTS_PER_VOXEL,
    DEFAULT_PHYSTWIN_OBJECT_VOLUME_TARGET_MS,
    DEFAULT_PHYSTWIN_OBJECT_VOLUME_VOXEL_M,
    PHYSTWIN_VOLUME_ORIGIN_FIRST_STABLE_FRAME_MIN,
    PHYSTWIN_VOLUME_ORIGIN_FRAME_MIN,
    PHYSTWIN_VOLUME_ORIGIN_WORLD,
    PHYSTWIN_VOLUME_ORIGINS,
    ObjectVoxelBudgetController,
    phystwin_volume_sample_indices,
    phystwin_volume_sample_points,
)


OBJECT_POINT_CONTROL_PHYSTWIN_VOLUME = "phystwin-volume"
OBJECT_POINT_CONTROL_FIXED_CAP = "fixed-cap"
OBJECT_VOLUME_RENDER_POLICY_LATEST_OR_CHEAP = "latest-volume-or-cheap"


@dataclass(frozen=True)
class ObjectVolumeFilterInput:
    seq: int
    timestamp_s: float
    object_xyz_world: np.ndarray
    object_rgb: np.ndarray
    controller_xyz_world: np.ndarray | None = None
    controller_rgb: np.ndarray | None = None
    cheap_object_xyz_world: np.ndarray | None = None
    cheap_object_rgb: np.ndarray | None = None


@dataclass(frozen=True)
class ObjectVolumeFilterOutput:
    seq: int
    timestamp_s: float
    object_xyz_world: np.ndarray
    object_rgb: np.ndarray
    controller_xyz_world: np.ndarray | None
    controller_rgb: np.ndarray | None
    voxel_size_m: float
    occupied_voxels: int
    input_points: int
    output_points: int
    filter_ms: float
    safety_cap_triggered: bool
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass
class ObjectVolumeFilterConfig:
    point_control: str = OBJECT_POINT_CONTROL_PHYSTWIN_VOLUME
    base_voxel_m: float = DEFAULT_PHYSTWIN_OBJECT_VOLUME_VOXEL_M
    origin_policy: str = PHYSTWIN_VOLUME_ORIGIN_WORLD
    adaptive: bool = True
    min_voxel_m: float = DEFAULT_PHYSTWIN_OBJECT_VOLUME_MIN_VOXEL_M
    max_voxel_m: float = DEFAULT_PHYSTWIN_OBJECT_VOLUME_MAX_VOXEL_M
    target_ms: float = DEFAULT_PHYSTWIN_OBJECT_VOLUME_TARGET_MS
    emergency_max_points: int = DEFAULT_PHYSTWIN_OBJECT_VOLUME_EMERGENCY_MAX_POINTS
    points_per_voxel: int = DEFAULT_PHYSTWIN_OBJECT_VOLUME_POINTS_PER_VOXEL
    render_policy: str = OBJECT_VOLUME_RENDER_POLICY_LATEST_OR_CHEAP
    stale_timeout_ms: float = 250.0

    def __post_init__(self) -> None:
        if self.point_control not in {OBJECT_POINT_CONTROL_PHYSTWIN_VOLUME, OBJECT_POINT_CONTROL_FIXED_CAP}:
            raise ValueError(f"unsupported object point control: {self.point_control}")
        if self.origin_policy not in PHYSTWIN_VOLUME_ORIGINS:
            raise ValueError(f"unsupported object volume origin policy: {self.origin_policy}")
        if self.base_voxel_m <= 0.0 or self.min_voxel_m <= 0.0 or self.max_voxel_m <= 0.0:
            raise ValueError("voxel sizes must be positive")
        if self.min_voxel_m > self.max_voxel_m:
            raise ValueError("min_voxel_m must be <= max_voxel_m")
        if self.target_ms <= 0.0:
            raise ValueError("target_ms must be positive")
        if self.emergency_max_points < 0:
            raise ValueError("emergency_max_points must be >= 0")
        if self.points_per_voxel < 1:
            raise ValueError("points_per_voxel must be >= 1")


class ObjectVolumeFilterService:
    def __init__(self, config: ObjectVolumeFilterConfig | None = None) -> None:
        self.config = config or ObjectVolumeFilterConfig()
        self._budget = ObjectVoxelBudgetController(
            target_ms=float(self.config.target_ms),
            base_voxel_m=float(self.config.base_voxel_m),
            min_voxel_m=float(self.config.min_voxel_m),
            max_voxel_m=float(self.config.max_voxel_m),
        )
        self._stable_origin_world: np.ndarray | None = None
        self._latest: ObjectVolumeFilterOutput | None = None
        self._lock = threading.Lock()
        self.processed_count = 0
        self.published_count = 0

    @property
    def current_voxel_m(self) -> float:
        return float(self._budget.current_voxel_m if self.config.adaptive else self.config.base_voxel_m)

    def filter_points(
        self,
        points_xyz_world: np.ndarray,
        colors_rgb: np.ndarray,
        *,
        seq: int = 0,
        timestamp_s: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any], float]:
        packet = ObjectVolumeFilterInput(
            seq=int(seq),
            timestamp_s=time.time() if timestamp_s is None else float(timestamp_s),
            object_xyz_world=points_xyz_world,
            object_rgb=colors_rgb,
        )
        output = self.filter_sync(packet)
        return output.object_xyz_world, output.object_rgb, dict(output.stats), float(output.filter_ms)

    def filter_sync(self, packet: ObjectVolumeFilterInput) -> ObjectVolumeFilterOutput:
        started_s = time.perf_counter()
        points = np.asarray(packet.object_xyz_world, dtype=np.float32)
        colors = np.asarray(packet.object_rgb)
        if colors.shape[0] != points.shape[0]:
            raise ValueError("object_rgb must have the same first dimension as object_xyz_world")
        voxel_m = self.current_voxel_m
        sampled_points, sampled_colors_or_none, stats = phystwin_volume_sample_points(
            points,
            colors,
            voxel_size_m=voxel_m,
            origin_world=self._origin_for_points(points),
            origin_policy=self.config.origin_policy,
            points_per_voxel=int(self.config.points_per_voxel),
            emergency_max_points=int(self.config.emergency_max_points),
        )
        filter_ms = float((time.perf_counter() - started_s) * 1000.0)
        if self.config.adaptive:
            self._budget.update(filter_ms)
        sampled_colors = (
            np.empty((0, 3), dtype=colors.dtype)
            if sampled_colors_or_none is None
            else np.asarray(sampled_colors_or_none)
        )
        service_stats = dict(stats)
        service_stats.update(self._profile_fields(stats, filter_ms=filter_ms))
        service_stats.update(
            {
                "enabled": True,
                "point_control": OBJECT_POINT_CONTROL_PHYSTWIN_VOLUME,
                "mode": OBJECT_POINT_CONTROL_PHYSTWIN_VOLUME,
            }
        )
        output = ObjectVolumeFilterOutput(
            seq=int(packet.seq),
            timestamp_s=float(packet.timestamp_s),
            object_xyz_world=np.asarray(sampled_points, dtype=np.float32),
            object_rgb=sampled_colors,
            controller_xyz_world=packet.controller_xyz_world,
            controller_rgb=packet.controller_rgb,
            voxel_size_m=float(stats.get("voxel_size_m", voxel_m)),
            occupied_voxels=int(stats.get("occupied_voxel_count", 0)),
            input_points=int(stats.get("input_point_count", 0)),
            output_points=int(stats.get("output_point_count", 0)),
            filter_ms=filter_ms,
            safety_cap_triggered=bool(stats.get("safety_cap_triggered", False)),
            stats=service_stats,
        )
        with self._lock:
            self._latest = output
            self.processed_count += 1
        return output

    def submit_latest(self, packet: ObjectVolumeFilterInput) -> ObjectVolumeFilterOutput:
        output = self.filter_sync(packet)
        with self._lock:
            self.published_count += 1
        return output

    def get_latest(self) -> ObjectVolumeFilterOutput | None:
        with self._lock:
            return self._latest

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            latest = self._latest
            return {
                "point_control": self.config.point_control,
                "base_voxel_m": float(self.config.base_voxel_m),
                "current_voxel_m": float(self.current_voxel_m),
                "origin_policy": self.config.origin_policy,
                "adaptive": bool(self.config.adaptive),
                "processed_count": int(self.processed_count),
                "published_count": int(self.published_count),
                "latest_seq": None if latest is None else int(latest.seq),
                "latest_output_points": 0 if latest is None else int(latest.output_points),
                "latest_occupied_voxels": 0 if latest is None else int(latest.occupied_voxels),
            }

    def _origin_for_points(self, points: np.ndarray) -> np.ndarray | None:
        policy = self.config.origin_policy
        if policy == PHYSTWIN_VOLUME_ORIGIN_WORLD:
            return np.zeros((3,), dtype=np.float32)
        if policy == PHYSTWIN_VOLUME_ORIGIN_FRAME_MIN:
            return None
        if policy == PHYSTWIN_VOLUME_ORIGIN_FIRST_STABLE_FRAME_MIN:
            if self._stable_origin_world is None and int(points.shape[0]) > 0:
                self._stable_origin_world = np.asarray(points, dtype=np.float32).min(axis=0)
            if self._stable_origin_world is None:
                return None
            return self._stable_origin_world
        raise ValueError(f"unsupported object volume origin policy: {policy}")

    def _profile_fields(self, stats: dict[str, Any], *, filter_ms: float) -> dict[str, Any]:
        return {
            "object_point_control": OBJECT_POINT_CONTROL_PHYSTWIN_VOLUME,
            "object_volume_voxel_m_base": float(self.config.base_voxel_m),
            "object_volume_voxel_m_current": float(stats.get("voxel_size_m", self.current_voxel_m)),
            "object_volume_origin_policy": str(stats.get("origin_policy", self.config.origin_policy)),
            "object_volume_points_per_voxel": int(stats.get("points_per_voxel", self.config.points_per_voxel)),
            "object_volume_input_points": int(stats.get("input_point_count", 0)),
            "object_volume_occupied_voxels": int(stats.get("occupied_voxel_count", 0)),
            "object_volume_output_points": int(stats.get("output_point_count", 0)),
            "object_volume_ms": float(filter_ms),
            "object_volume_safety_cap_triggered": bool(stats.get("safety_cap_triggered", False)),
            "object_volume_safety_cap_points": int(stats.get("safety_cap_points", self.config.emergency_max_points)),
            "object_volume_target_ms": float(self.config.target_ms),
            "object_volume_adaptive_enabled": bool(self.config.adaptive),
        }


__all__ = [
    "OBJECT_POINT_CONTROL_FIXED_CAP",
    "OBJECT_POINT_CONTROL_PHYSTWIN_VOLUME",
    "OBJECT_VOLUME_RENDER_POLICY_LATEST_OR_CHEAP",
    "ObjectVolumeFilterConfig",
    "ObjectVolumeFilterInput",
    "ObjectVolumeFilterOutput",
    "ObjectVolumeFilterService",
    "phystwin_volume_sample_indices",
]
