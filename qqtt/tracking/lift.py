from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class LiftedTrackFrame:
    points_world: np.ndarray
    track_ids: np.ndarray
    camera_ids: np.ndarray
    colors: np.ndarray | None
    valid_track_mask: np.ndarray
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def valid_mask(self) -> np.ndarray:
        return self.valid_track_mask


def _depth_to_meters(depth: np.ndarray, depth_scale_m_per_unit: float) -> np.ndarray:
    arr = np.asarray(depth)
    if np.issubdtype(arr.dtype, np.floating):
        depth_m = arr.astype(np.float32)
    else:
        depth_m = arr.astype(np.float32) * float(depth_scale_m_per_unit)
    depth_m[~np.isfinite(depth_m)] = 0.0
    depth_m[depth_m < 0] = 0.0
    return depth_m


def lift_tracks_to_world(
    *,
    tracks_yx_t: np.ndarray,
    visibility_t: np.ndarray,
    depth_uint16: np.ndarray,
    depth_scale_m_per_unit: float,
    mask: np.ndarray | None = None,
    object_mask: np.ndarray | None = None,
    K: np.ndarray,
    c2w: np.ndarray,
    camera_idx: int = 0,
    colors_rgb: np.ndarray | None = None,
    track_ids: np.ndarray | None = None,
    depth_min_m: float = 0.2,
    depth_max_m: float = 1.5,
    max_tracks: int | None = None,
) -> LiftedTrackFrame:
    tracks = np.asarray(tracks_yx_t, dtype=np.float32)
    visibility = np.asarray(visibility_t, dtype=bool).reshape(-1)
    if tracks.ndim != 2 or tracks.shape[1] != 2:
        raise ValueError(f"tracks_yx_t must have shape (N,2); got {tracks.shape}")
    if visibility.shape[0] != tracks.shape[0]:
        raise ValueError("visibility_t length must match tracks_yx_t.")
    depth_m = _depth_to_meters(depth_uint16, float(depth_scale_m_per_unit))
    height, width = depth_m.shape[:2]
    selected_mask = object_mask if object_mask is not None else mask
    mask_bool = np.ones((height, width), dtype=bool) if selected_mask is None else np.asarray(selected_mask, dtype=bool)
    if mask_bool.shape[:2] != (height, width):
        raise ValueError(f"mask shape {mask_bool.shape} does not match depth shape {depth_m.shape}")
    yy = np.rint(tracks[:, 0]).astype(np.int64)
    xx = np.rint(tracks[:, 1]).astype(np.int64)
    in_bounds = (yy >= 0) & (yy < height) & (xx >= 0) & (xx < width)
    inside_mask = np.zeros_like(visibility, dtype=bool)
    depth_valid = np.zeros_like(visibility, dtype=bool)
    sampled_depth = np.zeros_like(tracks[:, 0], dtype=np.float32)
    valid_indices = np.where(in_bounds)[0]
    if len(valid_indices) > 0:
        vyy = yy[valid_indices]
        vxx = xx[valid_indices]
        inside_mask[valid_indices] = mask_bool[vyy, vxx]
        sampled_depth[valid_indices] = depth_m[vyy, vxx]
        depth_valid[valid_indices] = (
            np.isfinite(sampled_depth[valid_indices])
            & (sampled_depth[valid_indices] >= float(depth_min_m))
            & (sampled_depth[valid_indices] <= float(depth_max_m))
        )
    valid = visibility & in_bounds & inside_mask & depth_valid
    if max_tracks is not None and int(max_tracks) >= 0 and int(valid.sum()) > int(max_tracks):
        valid_for_cap = np.where(valid)[0]
        keep = valid_for_cap[np.linspace(0, len(valid_for_cap) - 1, int(max_tracks), dtype=np.int64)]
        capped = np.zeros_like(valid)
        capped[keep] = True
        valid = capped
    ids = np.arange(tracks.shape[0], dtype=np.int32) if track_ids is None else np.asarray(track_ids)
    if ids.shape[0] != tracks.shape[0]:
        raise ValueError("track_ids length must match tracks_yx_t.")
    stats = _lift_stats(visibility, in_bounds, inside_mask, depth_valid, valid)
    if not np.any(valid):
        return LiftedTrackFrame(
            points_world=np.empty((0, 3), dtype=np.float32),
            track_ids=np.empty((0,), dtype=ids.dtype),
            camera_ids=np.empty((0,), dtype=np.int16),
            colors=None if colors_rgb is None else np.empty((0, 3), dtype=np.uint8),
            valid_track_mask=valid,
            stats=stats,
        )
    K_mat = np.asarray(K, dtype=np.float32).reshape(3, 3)
    transform = np.asarray(c2w, dtype=np.float32).reshape(4, 4)
    z = sampled_depth[valid]
    x_cam = (tracks[valid, 1] - float(K_mat[0, 2])) * z / float(K_mat[0, 0])
    y_cam = (tracks[valid, 0] - float(K_mat[1, 2])) * z / float(K_mat[1, 1])
    points_camera = np.stack([x_cam, y_cam, z], axis=1).astype(np.float32)
    hom = np.concatenate([points_camera, np.ones((len(points_camera), 1), dtype=np.float32)], axis=1)
    points_world = (hom @ transform.T)[:, :3].astype(np.float32)
    lifted_colors = np.asarray(colors_rgb)[yy[valid], xx[valid]].astype(np.uint8) if colors_rgb is not None else None
    return LiftedTrackFrame(
        points_world=points_world,
        track_ids=ids[valid],
        camera_ids=np.full((len(points_world),), int(camera_idx), dtype=np.int16),
        colors=lifted_colors,
        valid_track_mask=valid,
        stats=stats,
    )


def _lift_stats(visibility: np.ndarray, in_bounds: np.ndarray, inside_mask: np.ndarray, depth_valid: np.ndarray, lifted: np.ndarray) -> dict[str, Any]:
    total = int(visibility.size)
    visible_count = int(np.asarray(visibility, dtype=bool).sum())
    in_bounds_count = int((visibility & in_bounds).sum())
    inside_count = int((visibility & in_bounds & inside_mask).sum())
    depth_count = int((visibility & in_bounds & inside_mask & depth_valid).sum())
    lifted_count = int(lifted.sum())
    return {
        "num_tracks_total": total,
        "num_visible": visible_count,
        "num_in_bounds": in_bounds_count,
        "num_inside_mask": inside_count,
        "num_depth_valid": depth_count,
        "num_lifted": lifted_count,
        "visible_ratio": float(visible_count / total) if total else 0.0,
        "in_bounds_ratio": float(in_bounds_count / visible_count) if visible_count else 0.0,
        "inside_mask_ratio": float(inside_count / visible_count) if visible_count else 0.0,
        "depth_valid_ratio": float(depth_count / visible_count) if visible_count else 0.0,
        "lifted_ratio": float(lifted_count / total) if total else 0.0,
    }
