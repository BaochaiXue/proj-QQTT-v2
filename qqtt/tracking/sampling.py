from __future__ import annotations

from typing import Literal
import warnings

import numpy as np

SamplingStrategy = Literal["random", "grid", "uniform_grid", "farthest"]


def _mask_coordinates_yx(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(np.asarray(mask).astype(bool))
    if coords.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    return coords.astype(np.float32)


def sample_query_points_from_mask(
    mask: np.ndarray,
    *,
    num_points: int,
    strategy: SamplingStrategy = "grid",
    seed: int = 0,
    strict: bool = False,
) -> np.ndarray:
    coords = _mask_coordinates_yx(mask)
    count = int(num_points)
    if count <= 0 or len(coords) == 0:
        if strict and count > 0:
            raise ValueError("Cannot sample query points from an empty mask.")
        return np.empty((0, 2), dtype=np.float32)
    if len(coords) <= count:
        if strict and len(coords) < count:
            raise ValueError(f"Mask has only {len(coords)} pixels, fewer than requested {count}.")
        if len(coords) < count:
            warnings.warn(
                f"Mask has only {len(coords)} pixels, fewer than requested {count}; returning all available pixels.",
                RuntimeWarning,
                stacklevel=2,
            )
        return coords.copy()
    normalized_strategy = "uniform_grid" if strategy == "grid" else strategy
    if normalized_strategy == "random":
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(len(coords), size=count, replace=False)
        return coords[np.sort(idx)].astype(np.float32)
    if normalized_strategy in {"uniform_grid", "farthest"}:
        order = np.lexsort((coords[:, 1], coords[:, 0]))
        sorted_coords = coords[order]
        idx = np.linspace(0, len(sorted_coords) - 1, count, dtype=np.int64)
        return sorted_coords[idx].astype(np.float32)
    raise ValueError(f"Unsupported query sampling strategy: {strategy}")


def sample_object_sparse(mask: np.ndarray, num_points: int, *, strategy: SamplingStrategy = "grid", seed: int = 0) -> np.ndarray:
    return sample_query_points_from_mask(mask, num_points=num_points, strategy=strategy, seed=seed)


def sample_object_dense(mask: np.ndarray, num_points: int = 5000, *, strategy: SamplingStrategy = "grid", seed: int = 0, strict: bool = False) -> np.ndarray:
    return sample_query_points_from_mask(mask, num_points=num_points, strategy=strategy, seed=seed, strict=strict)


def sample_controller_sparse(mask: np.ndarray, num_points: int = 30, *, strategy: SamplingStrategy = "grid", seed: int = 0) -> np.ndarray:
    return sample_query_points_from_mask(mask, num_points=num_points, strategy=strategy, seed=seed)


def query_counts_for_mode(mode: str) -> tuple[int, ...]:
    normalized = str(mode).strip().lower()
    if normalized == "object_sparse":
        return (100, 256, 512, 1024)
    if normalized in {"object_dense", "phystwin_dense"}:
        return (5000, 10000)
    if normalized == "controller_sparse":
        return (30, 100)
    raise ValueError(f"Unsupported query point mode: {mode}")
