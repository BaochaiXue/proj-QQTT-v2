from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree


DATA_PROCESS_SAM3D_MAX_DIST_CAP_M = 0.035


def effective_shape_prior_max_dist(max_dist: float) -> float:
    value = float(max_dist)
    if value <= 0.0:
        return value
    return min(value, DATA_PROCESS_SAM3D_MAX_DIST_CAP_M)


def _points(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    return np.ascontiguousarray(arr.reshape(-1, 3), dtype=np.float64)


def _grid_index(
    point: np.ndarray,
    min_bound: np.ndarray,
    grid_size: float,
    *,
    force_float32: bool,
) -> tuple[int, int, int]:
    dtype = np.float32 if force_float32 else np.float64
    point_arr = np.asarray(point, dtype=dtype)
    min_arr = np.asarray(min_bound, dtype=dtype)
    return tuple(np.floor((point_arr - min_arr) / dtype(grid_size)).astype(int))


def _grid_indices(
    points: np.ndarray,
    min_bound: np.ndarray,
    grid_size: float,
    *,
    force_float32: bool,
) -> np.ndarray:
    dtype = np.float32 if force_float32 else np.float64
    point_arr = np.asarray(points, dtype=dtype)
    min_arr = np.asarray(min_bound, dtype=dtype)
    return np.floor((point_arr - min_arr) / dtype(grid_size)).astype(np.int64)


def _query_nearest_distances(tree: cKDTree, candidates: np.ndarray) -> np.ndarray:
    try:
        distances, _ = tree.query(candidates, k=1, workers=-1)
    except TypeError:
        distances, _ = tree.query(candidates, k=1)
    return np.asarray(distances, dtype=np.float64)


@dataclass
class ShapePriorBatchSelector:
    reference_points: np.ndarray
    min_bound: np.ndarray
    grid_size: float
    max_dist: float
    force_float32_voxel_keys: bool = False
    reference_tree: cKDTree | None = None
    accepted_candidate_count: int = 0
    _selected: list[np.ndarray] = field(default_factory=list, init=False)
    _occupied: set[tuple[int, int, int]] = field(default_factory=set, init=False)
    _tree: cKDTree | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.reference_points = _points(self.reference_points)
        self.min_bound = np.asarray(self.min_bound, dtype=np.float64).reshape(3)
        self.grid_size = float(self.grid_size)
        self.max_dist = float(self.max_dist)
        if self.grid_size <= 0.0:
            raise ValueError("grid_size must be positive")
        if self.reference_tree is not None:
            self._tree = self.reference_tree
        elif len(self.reference_points):
            self._tree = cKDTree(self.reference_points)

    def add_batch(self, batch: np.ndarray, *, limit: int) -> np.ndarray:
        remaining = int(limit) - len(self._selected)
        if remaining <= 0:
            return np.empty((0, 3), dtype=np.float64)
        candidates = _points(batch)
        if len(candidates) == 0:
            return candidates

        if self._tree is None:
            distances = np.zeros((len(candidates),), dtype=np.float64)
        else:
            distances = _query_nearest_distances(self._tree, candidates)
        if self.max_dist > 0.0:
            keep = distances <= self.max_dist
            candidates = candidates[keep]
            distances = distances[keep]
        self.accepted_candidate_count += int(len(candidates))
        if len(candidates) == 0:
            return np.empty((0, 3), dtype=np.float64)

        order = np.argsort(distances)
        keys = _grid_indices(
            candidates,
            self.min_bound,
            self.grid_size,
            force_float32=bool(self.force_float32_voxel_keys),
        )
        sorted_keys = keys[order]
        _, first_positions = np.unique(sorted_keys, axis=0, return_index=True)
        candidate_indices = order[np.sort(first_positions)]

        selected: list[np.ndarray] = []
        for candidate_index in candidate_indices:
            point = candidates[candidate_index]
            index = tuple(int(value) for value in keys[candidate_index])
            if index in self._occupied:
                continue
            selected.append(np.ascontiguousarray(point, dtype=np.float64))
            if len(selected) >= remaining:
                break
        for point in selected:
            self._occupied.add(
                _grid_index(
                    point,
                    self.min_bound,
                    self.grid_size,
                    force_float32=bool(self.force_float32_voxel_keys),
                )
            )
            self._selected.append(point)
        if not selected:
            return np.empty((0, 3), dtype=np.float64)
        return np.ascontiguousarray(np.asarray(selected, dtype=np.float64))

    def points(self) -> np.ndarray:
        if not self._selected:
            return np.empty((0, 3), dtype=np.float64)
        return np.ascontiguousarray(np.asarray(self._selected, dtype=np.float64))


__all__ = [
    "DATA_PROCESS_SAM3D_MAX_DIST_CAP_M",
    "ShapePriorBatchSelector",
    "effective_shape_prior_max_dist",
]
