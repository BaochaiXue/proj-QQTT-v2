from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class SingleViewShapePriorSamples:
    surface_points_m: np.ndarray
    interior_points_m: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SimpleShapeMesh:
    vertices: np.ndarray
    faces: np.ndarray

    @property
    def bounds(self) -> np.ndarray:
        verts = _points(self.vertices)
        if len(verts) == 0:
            return np.zeros((2, 3), dtype=np.float32)
        return np.stack([np.min(verts, axis=0), np.max(verts, axis=0)], axis=0).astype(np.float32)

    def copy(self) -> "SimpleShapeMesh":
        return SimpleShapeMesh(
            vertices=np.asarray(self.vertices, dtype=np.float32).copy(),
            faces=np.asarray(self.faces, dtype=np.int64).copy(),
        )


def _points(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float32)
    if arr.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    return np.ascontiguousarray(arr.reshape(-1, 3), dtype=np.float32)


def filter_points_by_nn_distance(points: np.ndarray, reference_points: np.ndarray, max_dist: float) -> np.ndarray:
    pts = _points(points)
    ref = _points(reference_points)
    if float(max_dist) <= 0.0 or pts.size == 0 or ref.size == 0:
        return pts
    tree = cKDTree(ref)
    distances, _ = tree.query(pts, k=1)
    return np.ascontiguousarray(pts[distances <= float(max_dist)], dtype=np.float32)


def _point_grid_index(point: np.ndarray, min_bound: np.ndarray, grid_size: float) -> tuple[int, int, int]:
    return tuple(np.floor((np.asarray(point, dtype=np.float32) - min_bound) / np.float32(grid_size)).astype(int))


def _sample_surface(mesh: Any, count: int) -> np.ndarray:
    try:
        import trimesh

        sampled, _ = trimesh.sample.sample_surface(mesh, int(count))
        return _points(sampled)
    except Exception:
        pass

    vertices = _points(np.asarray(mesh.vertices))
    faces = np.asarray(mesh.faces, dtype=np.int64).reshape(-1, 3)
    if len(vertices) == 0 or len(faces) == 0:
        return vertices
    triangles = vertices[faces]
    a = triangles[:, 0]
    b = triangles[:, 1]
    c = triangles[:, 2]
    areas = np.linalg.norm(np.cross(b - a, c - a), axis=1)
    if not np.isfinite(areas).all() or float(np.sum(areas)) <= 0.0:
        face_indices = np.arange(int(count), dtype=np.int64) % len(faces)
    else:
        probs = areas / np.sum(areas)
        rng = np.random.default_rng(42)
        face_indices = rng.choice(len(faces), size=int(count), replace=True, p=probs)
    rng = np.random.default_rng(43)
    u = rng.random(int(count), dtype=np.float32)
    v = rng.random(int(count), dtype=np.float32)
    flip = u + v > 1.0
    u[flip] = 1.0 - u[flip]
    v[flip] = 1.0 - v[flip]
    selected = triangles[face_indices]
    sampled = selected[:, 0] + u[:, None] * (selected[:, 1] - selected[:, 0]) + v[:, None] * (selected[:, 2] - selected[:, 0])
    return _points(sampled)


def _interior_candidates(mesh: Any, *, sample_count: int) -> np.ndarray:
    try:
        import trimesh

        sampled = trimesh.sample.volume_mesh(mesh, int(sample_count))
        sampled = _points(sampled)
        if len(sampled):
            return sampled
    except Exception:
        pass

    bounds = np.asarray(mesh.bounds, dtype=np.float32)
    span = np.maximum(bounds[1] - bounds[0], np.float32(1e-6))
    grid_count = min(int(sample_count), 10000)
    side = max(2, int(round(grid_count ** (1.0 / 3.0))))
    axes = [
        np.linspace(bounds[0, axis] + span[axis] * 0.1, bounds[1, axis] - span[axis] * 0.1, side)
        for axis in range(3)
    ]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    try:
        contained = mesh.contains(grid)
        grid = grid[contained]
    except Exception:
        center = np.mean(bounds, axis=0, keepdims=True)
        grid = center if grid.size == 0 else grid
    return _points(grid[:grid_count])


def sample_legacy_single_view_shape_prior_points(
    mesh: Any,
    reference_points_m: np.ndarray,
    *,
    num_surface_points: int = 1024,
    volume_sample_size_m: float = 0.005,
    shape_prior_max_dist_m: float = 0.05,
    interior_sample_count: int = 10000,
) -> SingleViewShapePriorSamples:
    reference = _points(reference_points_m)
    if len(reference) == 0:
        return SingleViewShapePriorSamples(
            surface_points_m=np.empty((0, 3), dtype=np.float32),
            interior_points_m=np.empty((0, 3), dtype=np.float32),
            metadata={
                "single_view_shape_prior_sampling_backend": "legacy",
                "uses_mvsam3d": False,
                "shape_prior_sampling_reason": "empty_reference_points",
            },
        )
    mesh = mesh.copy() if hasattr(mesh, "copy") else mesh
    if getattr(mesh, "faces", None) is None or len(mesh.faces) == 0:
        vertices = filter_points_by_nn_distance(np.asarray(mesh.vertices), reference, float(shape_prior_max_dist_m))
        return SingleViewShapePriorSamples(
            surface_points_m=vertices,
            interior_points_m=np.empty((0, 3), dtype=np.float32),
            metadata={
                "single_view_shape_prior_sampling_backend": "legacy",
                "uses_mvsam3d": False,
                "shape_prior_sampling_reason": "mesh_without_faces",
            },
        )

    surface_points = filter_points_by_nn_distance(_sample_surface(mesh, int(num_surface_points)), reference, float(shape_prior_max_dist_m))
    interior_points = filter_points_by_nn_distance(
        _interior_candidates(mesh, sample_count=int(interior_sample_count)),
        reference,
        float(shape_prior_max_dist_m),
    )

    min_bound = np.min(np.concatenate([surface_points, interior_points, reference], axis=0), axis=0)
    grid_size = float(volume_sample_size_m)
    object_grid = {_point_grid_index(point, min_bound, grid_size) for point in reference}

    prior_grid = set(object_grid)
    final_surface: list[np.ndarray] = []
    for point in surface_points:
        grid_index = _point_grid_index(point, min_bound, grid_size)
        if grid_index in prior_grid:
            continue
        prior_grid.add(grid_index)
        final_surface.append(point)

    interior_grid = set(object_grid)
    final_interior: list[np.ndarray] = []
    for point in interior_points:
        grid_index = _point_grid_index(point, min_bound, grid_size)
        if grid_index in interior_grid:
            continue
        interior_grid.add(grid_index)
        final_interior.append(point)

    surface_arr = _points(np.asarray(final_surface, dtype=np.float32))
    interior_arr = _points(np.asarray(final_interior, dtype=np.float32))
    return SingleViewShapePriorSamples(
        surface_points_m=surface_arr,
        interior_points_m=interior_arr,
        metadata={
            "single_view_shape_prior_sampling_backend": "legacy",
            "uses_mvsam3d": False,
            "shape_prior_num_surface_points": int(num_surface_points),
            "shape_prior_interior_sample_count": int(interior_sample_count),
            "shape_prior_max_dist_m": float(shape_prior_max_dist_m),
            "shape_prior_volume_sample_size_m": float(volume_sample_size_m),
            "shape_prior_surface_candidates": int(len(surface_points)),
            "shape_prior_interior_candidates": int(len(interior_points)),
            "shape_prior_surface_points": int(len(surface_arr)),
            "shape_prior_interior_points": int(len(interior_arr)),
        },
    )


__all__ = [
    "SingleViewShapePriorSamples",
    "filter_points_by_nn_distance",
    "sample_legacy_single_view_shape_prior_points",
]
