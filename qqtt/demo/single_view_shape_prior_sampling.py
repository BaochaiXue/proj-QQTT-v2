from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from data_process_sam3d.shape_prior_sampling import (
    DATA_PROCESS_SAM3D_MAX_DIST_CAP_M,
    ShapePriorBatchSelector,
    effective_shape_prior_max_dist,
)


DEFAULT_NUM_SURFACE_POINTS = 1024
DEFAULT_VOLUME_SAMPLE_SIZE_M = 0.005
DEFAULT_SHAPE_PRIOR_MAX_DIST_M = 0.05
DEFAULT_TARGET_SURFACE_POINTS = 700
DEFAULT_TARGET_INTERIOR_POINTS = 1000


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


def _effective_data_process_sam3d_max_dist(max_dist: float) -> float:
    return effective_shape_prior_max_dist(max_dist)


def _sort_by_reference_distance(points: np.ndarray, reference_points: np.ndarray) -> np.ndarray:
    pts = _points(points)
    ref = _points(reference_points)
    if len(pts) == 0:
        return pts
    tree = cKDTree(ref)
    distances, _ = tree.query(pts, k=1)
    return np.ascontiguousarray(pts[np.argsort(distances)], dtype=np.float32)


def _point_grid_index(point: np.ndarray, min_bound: np.ndarray, grid_size: float) -> tuple[int, int, int]:
    return tuple(np.floor((np.asarray(point, dtype=np.float32) - min_bound) / np.float32(grid_size)).astype(int))


def _dedupe_points(
    points: np.ndarray,
    min_bound: np.ndarray,
    *,
    occupied: set[tuple[int, int, int]] | None = None,
    limit: int | None = None,
    grid_size: float,
) -> np.ndarray:
    pts = _points(points)
    if len(pts) == 0:
        return np.empty((0, 3), dtype=np.float32)
    seen = set() if occupied is None else set(occupied)
    selected: list[np.ndarray] = []
    for point in pts:
        grid_index = _point_grid_index(point, min_bound, float(grid_size))
        if grid_index in seen:
            continue
        seen.add(grid_index)
        selected.append(point)
        if limit is not None and len(selected) >= int(limit):
            break
    if not selected:
        return np.empty((0, 3), dtype=np.float32)
    return _points(np.asarray(selected, dtype=np.float32))


def _as_trimesh_mesh(mesh: Any) -> Any:
    try:
        import trimesh

        if isinstance(mesh, trimesh.Trimesh):
            return mesh
        if hasattr(mesh, "vertices") and hasattr(mesh, "faces"):
            return trimesh.Trimesh(
                vertices=np.asarray(mesh.vertices, dtype=np.float32),
                faces=np.asarray(mesh.faces, dtype=np.int64),
                process=False,
            )
    except Exception:
        return mesh
    return mesh


def _sample_surface(mesh: Any, count: int) -> np.ndarray:
    try:
        import trimesh

        sampled, _ = trimesh.sample.sample_surface(_as_trimesh_mesh(mesh), int(count))
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


def _sample_volume(mesh: Any, *, sample_count: int) -> np.ndarray:
    try:
        import trimesh

        sampled = trimesh.sample.volume_mesh(_as_trimesh_mesh(mesh), int(sample_count))
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


def _voxel_interior_candidates(
    mesh: Any,
    reference_points: np.ndarray,
    *,
    volume_sample_size_m: float,
    max_dist_m: float,
) -> np.ndarray:
    try:
        import open3d as o3d
    except Exception:
        return _points(_sample_volume(mesh, sample_count=200000))

    mesh = _as_trimesh_mesh(mesh)
    bounds = np.asarray(mesh.bounds, dtype=np.float32)
    spacing = max(float(volume_sample_size_m), 1e-4)
    axes = [
        np.arange(bounds[0, axis] + spacing * 0.5, bounds[1, axis], spacing)
        for axis in range(3)
    ]
    if any(len(axis) == 0 for axis in axes):
        return np.empty((0, 3), dtype=np.float32)
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    if grid.shape[0] > 250000:
        step = int(np.ceil(grid.shape[0] / 250000))
        grid = grid[::step]

    try:
        scene = o3d.t.geometry.RaycastingScene()
        vertices = o3d.core.Tensor(np.asarray(mesh.vertices), dtype=o3d.core.Dtype.Float32)
        triangles = o3d.core.Tensor(np.asarray(mesh.faces), dtype=o3d.core.Dtype.UInt32)
        scene.add_triangles(vertices, triangles)
        signed = scene.compute_signed_distance(
            o3d.core.Tensor(grid.astype(np.float32), dtype=o3d.core.Dtype.Float32)
        ).numpy()
        interior = grid[signed < 0]
    except Exception:
        try:
            interior = grid[mesh.contains(grid)]
        except Exception:
            interior = np.empty((0, 3), dtype=np.float32)
    return _points(interior)


def sample_data_process_sam3d_single_view_shape_prior_points(
    mesh: Any,
    reference_points_m: np.ndarray,
    *,
    num_surface_points: int = DEFAULT_NUM_SURFACE_POINTS,
    volume_sample_size_m: float = DEFAULT_VOLUME_SAMPLE_SIZE_M,
    shape_prior_max_dist_m: float = DEFAULT_SHAPE_PRIOR_MAX_DIST_M,
    target_surface_points: int = DEFAULT_TARGET_SURFACE_POINTS,
    target_interior_points: int = DEFAULT_TARGET_INTERIOR_POINTS,
) -> SingleViewShapePriorSamples:
    reference = _points(reference_points_m)
    if len(reference) == 0:
        return SingleViewShapePriorSamples(
            surface_points_m=np.empty((0, 3), dtype=np.float32),
            interior_points_m=np.empty((0, 3), dtype=np.float32),
            metadata={
                "single_view_shape_prior_sampling_backend": "sam3d-single-view",
                "uses_mvsam3d": False,
                "shape_prior_sampling_reason": "empty_reference_points",
                "shape_prior_target_surface_points": int(target_surface_points),
                "shape_prior_target_interior_points": int(target_interior_points),
            },
        )
    mesh = _as_trimesh_mesh(mesh.copy() if hasattr(mesh, "copy") else mesh)
    if getattr(mesh, "faces", None) is None or len(mesh.faces) == 0:
        vertices = filter_points_by_nn_distance(
            np.asarray(mesh.vertices),
            reference,
            _effective_data_process_sam3d_max_dist(shape_prior_max_dist_m),
        )
        return SingleViewShapePriorSamples(
            surface_points_m=vertices,
            interior_points_m=np.empty((0, 3), dtype=np.float32),
            metadata={
                "single_view_shape_prior_sampling_backend": "sam3d-single-view",
                "uses_mvsam3d": False,
                "shape_prior_sampling_reason": "mesh_without_faces",
                "shape_prior_target_surface_points": int(target_surface_points),
                "shape_prior_target_interior_points": int(target_interior_points),
                "shape_prior_surface_points": int(len(vertices)),
                "shape_prior_interior_points": 0,
            },
        )

    np.random.seed(42)
    min_bound = np.min(reference, axis=0)
    prior_grid_size = max(float(volume_sample_size_m) * 0.4, 1e-4)
    max_dist = _effective_data_process_sam3d_max_dist(shape_prior_max_dist_m)
    reference_tree = cKDTree(reference)

    surface_selector = ShapePriorBatchSelector(
        reference_points=reference,
        min_bound=min_bound,
        grid_size=prior_grid_size,
        max_dist=max_dist,
        force_float32_voxel_keys=True,
        reference_tree=reference_tree,
    )
    surface_points = np.empty((0, 3), dtype=np.float32)
    for count in [max(int(num_surface_points), 4096), 10000, 50000, 200000]:
        sampled = _sample_surface(mesh, int(count))
        surface_selector.add_batch(sampled, limit=int(target_surface_points))
        surface_points = surface_selector.points()
        if len(surface_points) >= int(target_surface_points):
            break
    if len(surface_points) < int(target_surface_points):
        for _ in range(2):
            sampled = _sample_surface(mesh, 200000)
            surface_selector.add_batch(sampled, limit=int(target_surface_points))
            surface_points = surface_selector.points()
            if len(surface_points) >= int(target_surface_points):
                break

    interior_selector = ShapePriorBatchSelector(
        reference_points=reference,
        min_bound=min_bound,
        grid_size=prior_grid_size,
        max_dist=max_dist,
        force_float32_voxel_keys=True,
        reference_tree=reference_tree,
    )
    interior_points = np.empty((0, 3), dtype=np.float32)
    for count in [10000, 50000, 200000]:
        sampled = _sample_volume(mesh, sample_count=int(count))
        interior_selector.add_batch(sampled, limit=int(target_interior_points))
        interior_points = interior_selector.points()
        if len(interior_points) >= int(target_interior_points):
            break
    if len(interior_points) < int(target_interior_points):
        fallback = _voxel_interior_candidates(
            mesh,
            reference,
            volume_sample_size_m=float(volume_sample_size_m),
            max_dist_m=max_dist,
        )
        if fallback.size:
            interior_selector.add_batch(fallback, limit=int(target_interior_points))
            interior_points = interior_selector.points()

    return SingleViewShapePriorSamples(
        surface_points_m=_points(surface_points),
        interior_points_m=_points(interior_points),
        metadata={
            "single_view_shape_prior_sampling_backend": "sam3d-single-view",
            "single_view_shape_prior_sampling_source": "data_process_sam3d/data_process_sample.py",
            "uses_mvsam3d": False,
            "shape_prior_num_surface_points": int(num_surface_points),
            "shape_prior_target_surface_points": int(target_surface_points),
            "shape_prior_target_interior_points": int(target_interior_points),
            "shape_prior_max_dist_m": float(shape_prior_max_dist_m),
            "shape_prior_effective_max_dist_m": float(max_dist),
            "shape_prior_volume_sample_size_m": float(volume_sample_size_m),
            "shape_prior_surface_candidates": int(surface_selector.accepted_candidate_count),
            "shape_prior_interior_candidates": int(interior_selector.accepted_candidate_count),
            "shape_prior_surface_points": int(len(surface_points)),
            "shape_prior_interior_points": int(len(interior_points)),
        },
    )


def sample_legacy_single_view_shape_prior_points(
    mesh: Any,
    reference_points_m: np.ndarray,
    *,
    num_surface_points: int = 1024,
    volume_sample_size_m: float = 0.005,
    shape_prior_max_dist_m: float = 0.05,
    interior_sample_count: int = 10000,
) -> SingleViewShapePriorSamples:
    return sample_data_process_sam3d_single_view_shape_prior_points(
        mesh,
        reference_points_m,
        num_surface_points=int(num_surface_points),
        volume_sample_size_m=float(volume_sample_size_m),
        shape_prior_max_dist_m=float(shape_prior_max_dist_m),
    )


__all__ = [
    "SingleViewShapePriorSamples",
    "filter_points_by_nn_distance",
    "sample_data_process_sam3d_single_view_shape_prior_points",
    "sample_legacy_single_view_shape_prior_points",
]
