"""Pinhole projection grids and 2D-track lifting to world coordinates."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from demo_v6_2.utils.depth_geometry import transform_points


def build_projection_grid_from_matrix(
    *,
    width: int,
    height: int,
    K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build projection grid from matrix."""
    k = np.asarray(K, dtype=np.float32).reshape(3, 3)
    fx = np.float32(k[0, 0])
    fy = np.float32(k[1, 1])
    if fx <= 0 or fy <= 0:
        raise ValueError("intrinsics fx/fy must be positive")
    xs = (np.arange(width, dtype=np.float32) - np.float32(k[0, 2])) / fx
    ys = (np.arange(height, dtype=np.float32) - np.float32(k[1, 2])) / fy
    ray_x, ray_y = np.meshgrid(xs, ys, indexing="xy")
    return np.ascontiguousarray(ray_x, dtype=np.float32), np.ascontiguousarray(ray_y, dtype=np.float32)


@dataclass(frozen=True)
class OverlayLiftResult:
    points_world: np.ndarray
    source_indices: np.ndarray
    tracks_yx: np.ndarray
    valid_mask: np.ndarray


def intrinsics_to_matrix(intrinsics: Any) -> np.ndarray:
    """Normalize a mapping, fx/fy/cx/cy attribute object, or 3x3 array to a float32 K matrix."""
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


def track_lift_valid_mask(
    *,
    tracks_yx: np.ndarray,
    visibility: np.ndarray,
    depth: np.ndarray,
    depth_scale_m_per_unit: float,
    mask: np.ndarray | None,
    depth_min_m: float,
    depth_max_m: float,
) -> np.ndarray:
    """Return the per-track lift validity mask (visible, in-bounds, in-mask,
    depth within [depth_min_m, depth_max_m]), aligned with the input tracks."""
    tracks = np.asarray(tracks_yx, dtype=np.float32).reshape(-1, 2)
    vis = np.asarray(visibility, dtype=np.float32).reshape(-1) > 0.0
    if vis.shape[0] != tracks.shape[0]:
        raise ValueError("visibility length must match tracks_yx")

    depth_arr = np.asarray(depth)
    if np.issubdtype(depth_arr.dtype, np.floating):
        depth_m = depth_arr.astype(np.float32, copy=False)
    else:
        depth_m = depth_arr.astype(np.float32) * np.float32(depth_scale_m_per_unit)
    height, width = depth_m.shape[:2]
    mask_bool = (
        np.ones((height, width), dtype=bool)
        if mask is None
        else np.asarray(mask, dtype=bool)
    )
    if mask_bool.shape[:2] != (height, width):
        raise ValueError("tracker lift mask shape must match depth shape")

    yy = np.rint(tracks[:, 0]).astype(np.int64)
    xx = np.rint(tracks[:, 1]).astype(np.int64)
    finite_tracks = np.isfinite(tracks).all(axis=1)
    in_bounds = (yy >= 0) & (yy < height) & (xx >= 0) & (xx < width)
    valid = vis & finite_tracks & in_bounds
    if not np.any(valid):
        return np.zeros((tracks.shape[0],), dtype=bool)

    valid_indices = np.flatnonzero(valid)
    sampled_depth = depth_m[yy[valid_indices], xx[valid_indices]]
    depth_valid = (
        np.isfinite(sampled_depth)
        & (sampled_depth > 0.0)
        & (sampled_depth >= np.float32(depth_min_m))
    )
    if np.isfinite(float(depth_max_m)):
        depth_valid &= sampled_depth <= np.float32(depth_max_m)
    inside_mask = mask_bool[yy[valid_indices], xx[valid_indices]]
    valid_out = np.zeros((tracks.shape[0],), dtype=bool)
    valid_out[valid_indices] = depth_valid & inside_mask
    return valid_out


def _depth_to_meters(depth: np.ndarray, depth_scale_m_per_unit: float) -> np.ndarray:
    """Return depth in meters: float inputs are taken as meters already, integer
    inputs (e.g. uint16 sensor units) are scaled by depth_scale_m_per_unit.
    Non-finite and negative depths are zeroed so they fail the >0 validity gate."""
    arr = np.asarray(depth)
    if np.issubdtype(arr.dtype, np.floating):
        depth_m = arr.astype(np.float32)
    else:
        depth_m = arr.astype(np.float32) * float(depth_scale_m_per_unit)
    depth_m[~np.isfinite(depth_m)] = 0.0
    depth_m[depth_m < 0.0] = 0.0
    return depth_m


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
    """Lift 2D tracker points (row, col order) to world-frame 3D via the depth map.

    A track survives only if it is visible, lands in-bounds, inside the optional
    mask, and samples a depth within [depth_min_m, depth_max_m]. valid_mask is
    aligned with the input tracks; points_world/tracks_yx hold survivors only.
    """
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

    valid = track_lift_valid_mask(
        tracks_yx=tracks,
        visibility=visibility,
        depth=depth,
        depth_scale_m_per_unit=float(depth_scale_m_per_unit),
        mask=mask,
        depth_min_m=float(depth_min_m),
        depth_max_m=float(depth_max_m),
    )
    # Deterministic even-stride cap keeps the survivor set stable frame to frame.
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

    K = intrinsics_to_matrix(intrinsics)
    ray_x, ray_y = build_projection_grid_from_matrix(width=width, height=height, K=K)
    transform = np.asarray(c2w, dtype=np.float32).reshape(4, 4)
    rows = yy[source_indices]
    cols = xx[source_indices]
    z = depth_m[rows, cols]
    x_cam = ray_x[rows, cols].astype(np.float32, copy=False) * z
    y_cam = ray_y[rows, cols].astype(np.float32, copy=False) * z
    points_camera = np.stack([x_cam, y_cam, z], axis=1).astype(np.float32)
    points_world = transform_points(points_camera, transform).astype(np.float32)
    return OverlayLiftResult(
        points_world=points_world,
        source_indices=source_indices,
        tracks_yx=tracks[source_indices].astype(np.float32),
        valid_mask=valid,
    )
