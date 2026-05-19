from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


DEFAULT_PHYSTWIN_OBJECT_VOLUME_VOXEL_M = 0.005
DEFAULT_PHYSTWIN_OBJECT_VOLUME_MIN_VOXEL_M = 0.005
DEFAULT_PHYSTWIN_OBJECT_VOLUME_MAX_VOXEL_M = 0.012
DEFAULT_PHYSTWIN_OBJECT_VOLUME_TARGET_MS = 8.0
DEFAULT_PHYSTWIN_OBJECT_VOLUME_EMERGENCY_MAX_POINTS = 30_000
DEFAULT_PHYSTWIN_OBJECT_VOLUME_POINTS_PER_VOXEL = 1

PHYSTWIN_VOLUME_ORIGIN_WORLD = "world"
PHYSTWIN_VOLUME_ORIGIN_FRAME_MIN = "frame-min"
PHYSTWIN_VOLUME_ORIGIN_FIRST_STABLE_FRAME_MIN = "first-stable-frame-min"
PHYSTWIN_VOLUME_ORIGINS = (
    PHYSTWIN_VOLUME_ORIGIN_WORLD,
    PHYSTWIN_VOLUME_ORIGIN_FRAME_MIN,
    PHYSTWIN_VOLUME_ORIGIN_FIRST_STABLE_FRAME_MIN,
)


@dataclass
class ObjectVoxelBudgetController:
    target_ms: float = DEFAULT_PHYSTWIN_OBJECT_VOLUME_TARGET_MS
    base_voxel_m: float = DEFAULT_PHYSTWIN_OBJECT_VOLUME_VOXEL_M
    min_voxel_m: float = DEFAULT_PHYSTWIN_OBJECT_VOLUME_MIN_VOXEL_M
    max_voxel_m: float = DEFAULT_PHYSTWIN_OBJECT_VOLUME_MAX_VOXEL_M

    def __post_init__(self) -> None:
        if self.target_ms <= 0:
            raise ValueError("target_ms must be positive")
        if self.base_voxel_m <= 0 or self.min_voxel_m <= 0 or self.max_voxel_m <= 0:
            raise ValueError("voxel sizes must be positive")
        if self.min_voxel_m > self.max_voxel_m:
            raise ValueError("min_voxel_m must be <= max_voxel_m")
        self.current_voxel_m = min(self.max_voxel_m, max(self.min_voxel_m, self.base_voxel_m))

    def update(self, measured_ms: float) -> float:
        measured = float(measured_ms)
        if measured <= 0:
            return float(self.current_voxel_m)
        if measured > self.target_ms * 1.5:
            self.current_voxel_m = min(self.max_voxel_m, self.current_voxel_m * 1.25)
        elif measured > self.target_ms:
            self.current_voxel_m = min(self.max_voxel_m, self.current_voxel_m * 1.10)
        elif measured < self.target_ms * 0.5:
            self.current_voxel_m = max(self.min_voxel_m, self.current_voxel_m / 1.10)
        return float(self.current_voxel_m)


def _empty_indices() -> np.ndarray:
    return np.empty((0,), dtype=np.int64)


def _validate_points(xyz_world: np.ndarray) -> np.ndarray:
    points = np.asarray(xyz_world, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("xyz_world must be an Nx3 array")
    return points


def _voxel_rows(
    points: np.ndarray,
    *,
    voxel_size_m: float,
    origin_world: np.ndarray,
) -> np.ndarray:
    if voxel_size_m <= 0:
        raise ValueError("voxel_size_m must be positive")
    origin = np.asarray(origin_world, dtype=np.float32).reshape(3)
    return np.floor((points - origin[None, :]) / float(voxel_size_m)).astype(np.int64)


def phystwin_volume_sample_indices(
    xyz_world: np.ndarray,
    *,
    voxel_size_m: float = DEFAULT_PHYSTWIN_OBJECT_VOLUME_VOXEL_M,
    origin_world: np.ndarray | None = None,
    points_per_voxel: int = DEFAULT_PHYSTWIN_OBJECT_VOLUME_POINTS_PER_VOXEL,
) -> np.ndarray:
    """
    FuturePhysTwin-style world-volume object sampling.

    Keeps the first N representatives per occupied world-space voxel. With the
    default N=1, output count is exactly the occupied voxel count.
    """

    points = _validate_points(xyz_world)
    if points.shape[0] == 0:
        return _empty_indices()
    if int(points_per_voxel) < 1:
        raise ValueError("points_per_voxel must be >= 1")

    origin = points.min(axis=0) if origin_world is None else np.asarray(origin_world, dtype=np.float32).reshape(3)
    q = _voxel_rows(points, voxel_size_m=float(voxel_size_m), origin_world=origin)
    q_view = np.ascontiguousarray(q).view(np.dtype((np.void, q.dtype.itemsize * q.shape[1]))).reshape(-1)

    order = np.argsort(q_view, kind="stable")
    sorted_keys = q_view[order]
    keep_chunks: list[np.ndarray] = []
    start = 0
    per_voxel = int(points_per_voxel)
    while start < sorted_keys.shape[0]:
        end = start + 1
        while end < sorted_keys.shape[0] and sorted_keys[end] == sorted_keys[start]:
            end += 1
        keep_chunks.append(order[start : min(end, start + per_voxel)])
        start = end
    if not keep_chunks:
        return _empty_indices()
    return np.sort(np.concatenate(keep_chunks).astype(np.int64, copy=False))


def phystwin_volume_sample_points(
    xyz_world: np.ndarray,
    colors: np.ndarray | None = None,
    *,
    voxel_size_m: float = DEFAULT_PHYSTWIN_OBJECT_VOLUME_VOXEL_M,
    origin_world: np.ndarray | None = None,
    origin_policy: str = PHYSTWIN_VOLUME_ORIGIN_WORLD,
    points_per_voxel: int = DEFAULT_PHYSTWIN_OBJECT_VOLUME_POINTS_PER_VOXEL,
    emergency_max_points: int = DEFAULT_PHYSTWIN_OBJECT_VOLUME_EMERGENCY_MAX_POINTS,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    points = _validate_points(xyz_world)
    color_arr = None if colors is None else np.asarray(colors)
    if color_arr is not None and int(color_arr.shape[0]) != int(points.shape[0]):
        raise ValueError("colors must have the same first dimension as xyz_world")
    if int(emergency_max_points) < 0:
        raise ValueError("emergency_max_points must be >= 0")

    if points.shape[0] == 0:
        empty_colors = None if color_arr is None else np.empty((0, 3), dtype=color_arr.dtype)
        return points.copy(), empty_colors, {
            "mode": "phystwin-volume",
            "input_point_count": 0,
            "occupied_voxel_count": 0,
            "output_point_count": 0,
            "voxel_size_m": float(voxel_size_m),
            "origin_policy": str(origin_policy),
            "origin_world": None if origin_world is None else np.asarray(origin_world, dtype=np.float32).reshape(3).tolist(),
            "points_per_voxel": int(points_per_voxel),
            "safety_cap_triggered": False,
            "safety_cap_points": int(emergency_max_points),
        }

    origin = points.min(axis=0) if origin_world is None else np.asarray(origin_world, dtype=np.float32).reshape(3)
    q = _voxel_rows(points, voxel_size_m=float(voxel_size_m), origin_world=origin)
    q_view = np.ascontiguousarray(q).view(np.dtype((np.void, q.dtype.itemsize * q.shape[1]))).reshape(-1)
    occupied = int(np.unique(q_view).shape[0])
    keep_idx = phystwin_volume_sample_indices(
        points,
        voxel_size_m=float(voxel_size_m),
        origin_world=origin,
        points_per_voxel=int(points_per_voxel),
    )
    safety_cap_triggered = False
    cap = int(emergency_max_points)
    if cap > 0 and keep_idx.shape[0] > cap:
        keep_idx = keep_idx[:cap]
        safety_cap_triggered = True
    sampled_points = points[keep_idx]
    sampled_colors = None if color_arr is None else color_arr[keep_idx]
    stats = {
        "mode": "phystwin-volume",
        "input_point_count": int(points.shape[0]),
        "occupied_voxel_count": int(occupied),
        "output_point_count": int(sampled_points.shape[0]),
        "voxel_size_m": float(voxel_size_m),
        "origin_policy": str(origin_policy),
        "origin_world": origin.astype(np.float32).tolist(),
        "points_per_voxel": int(points_per_voxel),
        "safety_cap_triggered": bool(safety_cap_triggered),
        "safety_cap_points": int(cap),
    }
    return sampled_points, sampled_colors, stats
