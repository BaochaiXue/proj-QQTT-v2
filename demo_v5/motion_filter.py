from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def motion_valid_for_points(
    points: np.ndarray,
    visibility: np.ndarray,
    *,
    neighbor_dist_m: float = 0.01,
    min_neighbors: int = 5,
    motion_similarity_m: float = 0.005,
    once_false_mask: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float32)
    vis = np.asarray(visibility, dtype=bool)
    if pts.ndim != 3 or pts.shape[-1] != 3 or vis.shape != pts.shape[:2]:
        raise ValueError("expected T,N,3 points and T,N visibility")
    result = np.zeros_like(vis, dtype=bool)
    if pts.shape[0] > 1:
        result[:-1] = vis[:-1] & vis[1:]
    global_mask = (
        np.prod(vis, axis=0).astype(bool)
        if once_false_mask and vis.size
        else np.ones((pts.shape[1],), dtype=bool)
    )
    motion = np.zeros_like(pts)
    if pts.shape[0] > 1:
        motion[:-1] = pts[1:] - pts[:-1]
    for frame_idx in range(max(0, pts.shape[0] - 1)):
        if once_false_mask:
            result[frame_idx] &= global_mask
        valid_ids = np.flatnonzero(result[frame_idx])
        finite_ids = valid_ids[np.isfinite(pts[frame_idx, valid_ids]).all(axis=1)]
        result[frame_idx, valid_ids] = False
        result[frame_idx, finite_ids] = True
        if finite_ids.size == 0:
            continue
        tree = cKDTree(pts[frame_idx, finite_ids])
        neighborhoods = tree.query_ball_point(
            pts[frame_idx, finite_ids], r=neighbor_dist_m, workers=-1
        )
        for local_idx, query_id_value in enumerate(finite_ids):
            query_id = int(query_id_value)
            neighbors = finite_ids[np.asarray(neighborhoods[local_idx], dtype=np.int64)]
            neighbors = neighbors[result[frame_idx, neighbors]]
            if neighbors.size < min_neighbors:
                result[frame_idx, query_id] = False
                if once_false_mask:
                    global_mask[query_id] = False
                continue
            diff = np.linalg.norm(
                motion[frame_idx, query_id] - motion[frame_idx, neighbors], axis=1
            )
            if np.count_nonzero(diff < motion_similarity_m) < 0.5 * neighbors.size:
                result[frame_idx, query_id] = False
                if once_false_mask:
                    global_mask[query_id] = False
        if once_false_mask:
            result[frame_idx] &= global_mask
    return np.ascontiguousarray(result), np.ascontiguousarray(global_mask)
