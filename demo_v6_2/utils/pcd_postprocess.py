"""PhysTwin-compatible radius-outlier detection."""

from __future__ import annotations

import numpy as np


def detect_radius_outlier_indices(
    points_world: np.ndarray,
    *,
    radius_m: float,
    nb_points: int,
) -> dict[str, np.ndarray]:
    """Split points into inliers/outliers by neighbor count within radius_m.

    A point is an inlier when it has at least nb_points neighbors (self included)
    inside radius_m — the same rule as Open3D remove_radius_outlier. Returned
    index arrays are int32 and sorted ascending.
    """
    cloud = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    point_count = int(len(cloud))
    if point_count == 0:
        empty = np.empty((0,), dtype=np.int32)
        return {"inlier_indices": empty, "outlier_indices": empty}

    # Deferred import: keep scipy off the module import path for non-filtering callers.
    from scipy.spatial import cKDTree

    tree = cKDTree(cloud)
    neighbor_counts = tree.query_ball_point(
        cloud, r=float(radius_m), return_length=True
    )
    inliers = np.flatnonzero(
        np.asarray(neighbor_counts, dtype=np.int64) >= int(nb_points)
    ).astype(np.int32)
    if len(inliers) == 0:
        return {
            "inlier_indices": inliers,
            "outlier_indices": np.arange(point_count, dtype=np.int32),
        }
    keep_mask = np.zeros((point_count,), dtype=bool)
    keep_mask[inliers] = True
    outliers = np.flatnonzero(~keep_mask).astype(np.int32)
    return {"inlier_indices": inliers, "outlier_indices": outliers}
