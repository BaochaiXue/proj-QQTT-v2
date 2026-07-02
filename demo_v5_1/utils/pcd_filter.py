"""Voxel-based point-cloud capping/filtering and the async filter worker."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import threading
import time
from typing import Any, Callable

import numpy as np


def _empty_indices() -> np.ndarray:
    return np.empty((0,), dtype=np.int64)


def _voxel_keys(xyz: np.ndarray, *, voxel_size_m: float) -> np.ndarray:
    """Map each point to an int64 key identifying its voxel of size voxel_size_m."""
    if voxel_size_m <= 0:
        raise ValueError("voxel_size_m must be positive")
    xyz_c = np.ascontiguousarray(xyz)
    if xyz_c.ndim != 2 or xyz_c.shape[1] != 3:
        raise ValueError("xyz must be an Nx3 array")
    if xyz_c.shape[0] == 0:
        return np.empty(0, dtype=np.int64)

    q = np.floor(xyz_c / float(voxel_size_m)).astype(np.int64)
    q -= q.min(axis=0, keepdims=True)
    dims = q.max(axis=0) + 1

    # ravel_multi_index needs dims_x*dims_y*dims_z to fit in int64; for extreme
    # extents fall back to a spatial-hash key (classic large primes). Hash keys
    # can collide across voxels, which is acceptable for capping/density use.
    limit = np.iinfo(np.int64).max // 4
    product = 1
    for dim in dims:
        product *= int(dim)
        if product > limit:
            break
    if product > limit:
        return q[:, 0] * np.int64(73856093) ^ q[:, 1] * np.int64(19349663) ^ q[:, 2] * np.int64(83492791)
    return np.ravel_multi_index(q.T, tuple(int(dim) for dim in dims)).astype(np.int64, copy=False)


def voxel_cap_indices(
    xyz: np.ndarray,
    *,
    max_points: int = 20_000,
    voxel_size_m: float = 0.004,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Return indices for spatially capped point clouds before expensive filtering.

    Keeps at most one representative per voxel first. If that still exceeds
    max_points, randomly subsamples voxel representatives.
    """

    points = np.asarray(xyz)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("xyz must be an Nx3 array")
    if max_points < 0:
        raise ValueError("max_points must be >= 0")
    if points.shape[0] == 0:
        return _empty_indices()
    if max_points == 0 or points.shape[0] <= int(max_points):
        return np.arange(points.shape[0], dtype=np.int64)

    keys = _voxel_keys(points, voxel_size_m=float(voxel_size_m))
    _unused_unique, first_idx = np.unique(keys, return_index=True)
    if first_idx.shape[0] > int(max_points):
        # Fixed seed by default: identical input clouds must cap identically.
        generator = rng if rng is not None else np.random.default_rng(0)
        keep_idx = generator.choice(first_idx, size=int(max_points), replace=False)
    else:
        keep_idx = first_idx
    keep_idx = np.sort(keep_idx)
    return keep_idx.astype(np.int64, copy=False)


def voxel_density_indices(
    xyz: np.ndarray,
    *,
    voxel_size_m: float = 0.004,
    min_points_per_voxel: int = 2,
) -> np.ndarray:
    """Return indices of points whose voxel holds at least min_points_per_voxel points."""
    points = np.asarray(xyz)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("xyz must be an Nx3 array")
    if min_points_per_voxel <= 1:
        return np.arange(points.shape[0], dtype=np.int64)
    if points.shape[0] == 0:
        return _empty_indices()
    keys = _voxel_keys(points, voxel_size_m=float(voxel_size_m))
    _unused_unique, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    keep = counts[inverse] >= int(min_points_per_voxel)
    return np.flatnonzero(keep).astype(np.int64, copy=False)


@dataclass(frozen=True)
class FilterInput:
    seq: int
    object_xyz: np.ndarray
    object_rgb: np.ndarray
    controller_xyz: np.ndarray
    controller_rgb: np.ndarray
    created_perf_s: float = field(default_factory=time.perf_counter)
    object_cap: int = 20_000
    controller_cap: int = 20_000
    object_voxel_size_m: float = 0.004
    controller_voxel_size_m: float = 0.003
    object_yx: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.int64))
    controller_yx: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.int64))


@dataclass(frozen=True)
class FilterOutput:
    seq: int
    object_xyz: np.ndarray
    object_rgb: np.ndarray
    controller_xyz: np.ndarray
    controller_rgb: np.ndarray
    filter_ms: float
    created_perf_s: float
    output_perf_s: float = field(default_factory=time.perf_counter)
    stats: dict[str, Any] = field(default_factory=dict)
    object_yx: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.int64))
    controller_yx: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.int64))


class AsyncPcdFilterWorker:
    """Latest-wins worker for non-blocking point-cloud filtering.

    A single pending slot holds at most one FilterInput: submitting while one is
    queued replaces it (counted in pending_replace_count / drop_count). The
    worker thread always filters the newest frame, so consumers polling
    latest_output() never fall more than one frame behind the filter itself.
    """

    def __init__(self, filter_fn: Callable[[FilterInput], FilterOutput], *, stats_window_s: float = 1.0) -> None:
        self.filter_fn = filter_fn
        self.stats_window_s = float(stats_window_s)
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._pending: FilterInput | None = None
        self._latest: FilterOutput | None = None
        self._busy = False
        self._stop = False
        self._started = False
        self._submit_count = 0
        self._output_count = 0
        self._pending_replace_count = 0
        self._thread = threading.Thread(target=self._run, name="pcd-filter-worker", daemon=True)
        self._submit_times: deque[float] = deque()
        self._output_times: deque[float] = deque()

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
        self._thread.start()

    def stop(self) -> None:
        with self._condition:
            self._stop = True
            self._condition.notify_all()
        if self._started and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def submit_latest(self, item: FilterInput) -> bool:
        """Queue item, replacing any not-yet-started one; True when the worker was idle."""
        now_s = time.perf_counter()
        with self._condition:
            if self._pending is not None:
                self._pending_replace_count += 1
            self._pending = item
            self._submit_count += 1
            self._submit_times.append(now_s)
            self._prune_locked(self._submit_times, now_s)
            accepted_idle = not self._busy
            self._condition.notify()
            return accepted_idle

    def latest_output(self) -> FilterOutput | None:
        with self._lock:
            return self._latest

    def is_busy(self) -> bool:
        with self._lock:
            return self._busy

    @property
    def drop_count(self) -> int:
        with self._lock:
            return int(self._pending_replace_count)

    @property
    def submit_fps(self) -> float:
        with self._lock:
            return self._fps_locked(self._submit_times)

    @property
    def output_fps(self) -> float:
        with self._lock:
            return self._fps_locked(self._output_times)

    @property
    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "busy": bool(self._busy),
                "submit_count": int(self._submit_count),
                "output_count": int(self._output_count),
                "pending_replace_count": int(self._pending_replace_count),
                "submit_fps": self._fps_locked(self._submit_times),
                "output_fps": self._fps_locked(self._output_times),
            }

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._stop and self._pending is None:
                    self._busy = False
                    self._condition.wait(timeout=0.01)
                if self._stop:
                    return
                item = self._pending
                self._pending = None
                self._busy = True

            # Run the (potentially slow) filter outside the lock so submits keep flowing.
            assert item is not None
            output = self.filter_fn(item)

            now_s = time.perf_counter()
            with self._condition:
                self._latest = output
                self._busy = False
                self._output_count += 1
                self._output_times.append(now_s)
                self._prune_locked(self._output_times, now_s)

    def _prune_locked(self, values: deque[float], now_s: float) -> None:
        # Keep at least one sample so _fps_locked can still span the window edge.
        cutoff = now_s - self.stats_window_s
        while len(values) > 1 and values[0] < cutoff:
            values.popleft()

    @staticmethod
    def _fps_locked(values: deque[float]) -> float:
        if len(values) < 2:
            return 0.0
        elapsed = values[-1] - values[0]
        if elapsed <= 0:
            return 0.0
        return float((len(values) - 1) / elapsed)


class FilterBudgetController:
    """Adapt the point cap so filter time converges toward target_ms."""

    def __init__(
        self,
        *,
        target_ms: float = 12.0,
        min_cap: int = 5_000,
        max_cap: int = 20_000,
        init_cap: int = 20_000,
    ) -> None:
        self.target_ms = float(target_ms)
        self.min_cap = int(min_cap)
        self.max_cap = int(max_cap)
        self.cap = int(np.clip(int(init_cap), self.min_cap, self.max_cap))

    def update(self, measured_ms: float) -> int:
        if measured_ms <= 0 or self.target_ms <= 0:
            return self.cap
        # Damped proportional step (30% weight on the measurement) to avoid
        # oscillating the cap on noisy per-frame timings.
        ratio = self.target_ms / float(measured_ms)
        new_cap = int(self.cap * (0.7 + 0.3 * ratio))
        self.cap = int(np.clip(new_cap, self.min_cap, self.max_cap))
        return self.cap
