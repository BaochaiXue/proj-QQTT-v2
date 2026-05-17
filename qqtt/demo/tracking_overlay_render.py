from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class OverlayLiftResult:
    points_world: np.ndarray
    source_indices: np.ndarray
    tracks_yx: np.ndarray
    valid_mask: np.ndarray


def _intrinsics_to_matrix(intrinsics: Any) -> np.ndarray:
    if isinstance(intrinsics, Mapping):
        fx = float(intrinsics["fx"])
        fy = float(intrinsics["fy"])
        cx = float(intrinsics["cx"])
        cy = float(intrinsics["cy"])
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    if all(hasattr(intrinsics, name) for name in ("fx", "fy", "cx", "cy")):
        return np.array(
            [
                [float(intrinsics.fx), 0.0, float(intrinsics.cx)],
                [0.0, float(intrinsics.fy), float(intrinsics.cy)],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
    return np.asarray(intrinsics, dtype=np.float32).reshape(3, 3)


def _depth_to_meters(depth: np.ndarray, depth_scale_m_per_unit: float) -> np.ndarray:
    arr = np.asarray(depth)
    if np.issubdtype(arr.dtype, np.floating):
        depth_m = arr.astype(np.float32)
    else:
        depth_m = arr.astype(np.float32) * float(depth_scale_m_per_unit)
    depth_m[~np.isfinite(depth_m)] = 0.0
    depth_m[depth_m < 0.0] = 0.0
    return depth_m


def select_overlay_point_indices(visibility: np.ndarray, *, max_points: int | None) -> np.ndarray:
    visible = np.asarray(visibility, dtype=np.float32).reshape(-1) > 0.0
    indices = np.where(visible)[0]
    if max_points is None or int(max_points) < 0 or len(indices) <= int(max_points):
        return indices.astype(np.int64)
    selected = indices[np.linspace(0, len(indices) - 1, int(max_points), dtype=np.int64)]
    return selected.astype(np.int64)


def lift_tracks_yx_to_world(
    *,
    tracks_yx: np.ndarray,
    visibility: np.ndarray,
    depth: np.ndarray,
    intrinsics: Any,
    c2w: np.ndarray,
    depth_scale_m_per_unit: float = 0.001,
    mask: np.ndarray | None = None,
    depth_min_m: float = 0.0,
    depth_max_m: float = float("inf"),
    max_points: int | None = None,
) -> OverlayLiftResult:
    tracks = np.asarray(tracks_yx, dtype=np.float32)
    if tracks.ndim != 2 or tracks.shape[1] != 2:
        raise ValueError(f"tracks_yx must have shape (N,2); got {tracks.shape}")
    vis = np.asarray(visibility, dtype=np.float32).reshape(-1) > 0.0
    if vis.shape[0] != tracks.shape[0]:
        raise ValueError("visibility length must match tracks_yx.")

    depth_m = _depth_to_meters(depth, float(depth_scale_m_per_unit))
    height, width = depth_m.shape[:2]
    mask_bool = np.ones((height, width), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if mask_bool.shape[:2] != (height, width):
        raise ValueError(f"mask shape {mask_bool.shape} does not match depth shape {depth_m.shape}")

    yy = np.rint(tracks[:, 0]).astype(np.int64)
    xx = np.rint(tracks[:, 1]).astype(np.int64)
    in_bounds = (yy >= 0) & (yy < height) & (xx >= 0) & (xx < width)

    sampled_depth = np.zeros((tracks.shape[0],), dtype=np.float32)
    inside_mask = np.zeros((tracks.shape[0],), dtype=bool)
    valid_bounds = np.where(in_bounds)[0]
    if len(valid_bounds) > 0:
        sampled_depth[valid_bounds] = depth_m[yy[valid_bounds], xx[valid_bounds]]
        inside_mask[valid_bounds] = mask_bool[yy[valid_bounds], xx[valid_bounds]]

    depth_valid = (
        np.isfinite(sampled_depth)
        & (sampled_depth > 0.0)
        & (sampled_depth >= float(depth_min_m))
        & (sampled_depth <= float(depth_max_m))
    )
    valid = vis & in_bounds & inside_mask & depth_valid
    if max_points is not None and int(max_points) >= 0 and int(valid.sum()) > int(max_points):
        valid_indices = np.where(valid)[0]
        keep = valid_indices[np.linspace(0, len(valid_indices) - 1, int(max_points), dtype=np.int64)]
        capped = np.zeros_like(valid)
        capped[keep] = True
        valid = capped

    source_indices = np.where(valid)[0].astype(np.int64)
    if len(source_indices) == 0:
        return OverlayLiftResult(
            points_world=np.empty((0, 3), dtype=np.float32),
            source_indices=source_indices,
            tracks_yx=np.empty((0, 2), dtype=np.float32),
            valid_mask=valid,
        )

    K = _intrinsics_to_matrix(intrinsics)
    transform = np.asarray(c2w, dtype=np.float32).reshape(4, 4)
    z = sampled_depth[source_indices]
    x_cam = (tracks[source_indices, 1] - float(K[0, 2])) * z / float(K[0, 0])
    y_cam = (tracks[source_indices, 0] - float(K[1, 2])) * z / float(K[1, 1])
    points_camera = np.stack([x_cam, y_cam, z], axis=1).astype(np.float32)
    hom = np.concatenate([points_camera, np.ones((len(points_camera), 1), dtype=np.float32)], axis=1)
    points_world = (hom @ transform.T)[:, :3].astype(np.float32)
    return OverlayLiftResult(
        points_world=points_world,
        source_indices=source_indices,
        tracks_yx=tracks[source_indices].astype(np.float32),
        valid_mask=valid,
    )


__all__ = [
    "OverlayLiftResult",
    "lift_tracks_yx_to_world",
    "select_overlay_point_indices",
]
