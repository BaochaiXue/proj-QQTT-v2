from __future__ import annotations

import numpy as np

from demo_v5.contracts import SEMANTIC_CONTROLLER, SEMANTIC_OBJECT
from qqtt.demo import phystwin_strict_product as strict


def round_tracks_to_indices(
    tracks_yx: np.ndarray,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tracks = np.asarray(tracks_yx, dtype=np.float32).reshape(-1, 2)
    finite = np.isfinite(tracks).all(axis=1)
    safe = np.where(np.isfinite(tracks), tracks, 0.0)
    yy = np.rint(safe[:, 0]).astype(np.int64)
    xx = np.rint(safe[:, 1]).astype(np.int64)
    valid = finite & (yy >= 0) & (yy < shape[0]) & (xx >= 0) & (xx < shape[1])
    return yy, xx, valid


def first_frame_semantics(frame: strict.PreparedPhysTwinFrame) -> np.ndarray:
    masks = strict.normalize_processed_mask_frame(frame.processed_mask_frame)
    tracks = np.asarray(frame.tracks_yx, dtype=np.float32).reshape(-1, 2)
    visible = np.asarray(frame.visibility, dtype=bool).reshape(-1)
    yy, xx, in_bounds = round_tracks_to_indices(tracks, masks["object"].shape)
    valid = visible & in_bounds
    semantics = np.zeros((tracks.shape[0],), dtype=np.int8)
    if np.any(valid):
        ids = np.flatnonzero(valid)
        controller = masks["controller"][yy[ids], xx[ids]]
        obj = masks["object"][yy[ids], xx[ids]] & ~controller
        semantics[ids[obj]] = SEMANTIC_OBJECT
        semantics[ids[controller]] = SEMANTIC_CONTROLLER
    return np.ascontiguousarray(semantics)


def frame_direct_samples(
    frame: strict.PreparedPhysTwinFrame,
    semantics: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    masks = strict.normalize_processed_mask_frame(frame.processed_mask_frame)
    points_grid = np.asarray(frame.pcd_points, dtype=np.float32)
    colors_grid = np.asarray(frame.pcd_colors, dtype=np.uint8)
    if points_grid.ndim == 4:
        points_grid = points_grid[0]
    if colors_grid.ndim == 4:
        colors_grid = colors_grid[0]
    if points_grid.ndim != 3 or points_grid.shape[-1] != 3:
        raise ValueError(f"invalid pcd_points shape {points_grid.shape}")
    if colors_grid.shape != points_grid.shape:
        raise ValueError("pcd_colors must match pcd_points")
    tracks = np.asarray(frame.tracks_yx, dtype=np.float32).reshape(-1, 2)
    visible = np.asarray(frame.visibility, dtype=bool).reshape(-1)
    if visible.shape[0] != tracks.shape[0] or semantics.shape != (tracks.shape[0],):
        raise ValueError("query arrays do not share count")
    yy, xx, in_bounds = round_tracks_to_indices(tracks, points_grid.shape[:2])
    semantic_inside = np.zeros((tracks.shape[0],), dtype=bool)
    object_ids = in_bounds & (semantics == SEMANTIC_OBJECT)
    controller_ids = in_bounds & (semantics == SEMANTIC_CONTROLLER)
    if np.any(object_ids):
        semantic_inside[object_ids] = masks["object"][yy[object_ids], xx[object_ids]]
    if np.any(controller_ids):
        semantic_inside[controller_ids] = masks["controller"][
            yy[controller_ids], xx[controller_ids]
        ]
    valid = visible & in_bounds & semantic_inside
    points = np.full((tracks.shape[0], 3), np.nan, dtype=np.float32)
    colors = np.zeros((tracks.shape[0], 3), dtype=np.float32)
    ids = np.flatnonzero(valid)
    if ids.size:
        sampled = points_grid[yy[ids], xx[ids]]
        keep = np.isfinite(sampled).all(axis=1) & (
            np.linalg.norm(sampled, axis=1) > 1e-9
        )
        valid[ids[~keep]] = False
        ids = ids[keep]
        if ids.size:
            points[ids] = points_grid[yy[ids], xx[ids]]
            colors[ids] = colors_grid[yy[ids], xx[ids]].astype(np.float32) / 255.0
    return points, colors, np.ascontiguousarray(valid)
