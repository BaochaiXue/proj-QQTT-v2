from __future__ import annotations

import numpy as np


def farthest_point_indices(points_xyz: np.ndarray, count: int) -> np.ndarray:
    points = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    if points.shape[0] < count:
        raise RuntimeError(f"need {count} candidates; got {points.shape[0]}")
    selected = [0]
    min_dist2 = np.sum((points - points[0]) ** 2, axis=1)
    for _ in range(1, count):
        idx = int(np.argmax(min_dist2))
        selected.append(idx)
        min_dist2 = np.minimum(
            min_dist2, np.sum((points - points[idx]) ** 2, axis=1)
        )
    return np.asarray(selected, dtype=np.int64)
