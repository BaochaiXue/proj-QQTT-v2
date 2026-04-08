from __future__ import annotations

from typing import Any

import numpy as np


def _normalize(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    vec = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-6:
        return np.asarray(fallback, dtype=np.float32)
    return vec / norm


def fit_dominant_table_plane(
    points: np.ndarray,
    *,
    low_quantile: float = 0.35,
    max_fit_points: int = 20000,
) -> dict[str, np.ndarray]:
    cloud = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if len(cloud) < 3:
        return {
            "point": np.zeros((3,), dtype=np.float32),
            "normal": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        }

    z_threshold = float(np.quantile(cloud[:, 2], low_quantile))
    candidates = cloud[cloud[:, 2] <= z_threshold]
    if len(candidates) < 128:
        candidates = cloud
    if len(candidates) > max_fit_points:
        idx = np.linspace(0, len(candidates) - 1, max_fit_points, dtype=np.int32)
        candidates = candidates[idx]

    plane_point = np.median(candidates, axis=0).astype(np.float32)
    centered = candidates - plane_point[None, :]
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = _normalize(vh[-1], np.array([0.0, 0.0, 1.0], dtype=np.float32))
    if float(normal[2]) < 0.0:
        normal = -normal
    return {
        "point": plane_point,
        "normal": normal.astype(np.float32),
    }


def estimate_object_support_mask(
    points: np.ndarray,
    *,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
    object_height_min: float,
    object_height_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    cloud = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    signed_height = (cloud - np.asarray(plane_point, dtype=np.float32)[None, :]) @ np.asarray(plane_normal, dtype=np.float32)
    mask = (signed_height >= float(object_height_min)) & (signed_height <= float(object_height_max))
    return mask, signed_height.astype(np.float32)


def _build_voxel_components(points: np.ndarray, *, voxel_size: float) -> list[np.ndarray]:
    cloud = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if len(cloud) == 0:
        return []
    origin = cloud.min(axis=0)
    keys = np.floor((cloud - origin[None, :]) / max(float(voxel_size), 1e-4)).astype(np.int32)
    voxel_to_point_indices: dict[tuple[int, int, int], list[int]] = {}
    for point_idx, key in enumerate(keys):
        voxel_to_point_indices.setdefault((int(key[0]), int(key[1]), int(key[2])), []).append(point_idx)

    visited: set[tuple[int, int, int]] = set()
    components: list[np.ndarray] = []
    neighbor_offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if not (dx == 0 and dy == 0 and dz == 0)
    ]
    for start_key in voxel_to_point_indices:
        if start_key in visited:
            continue
        queue = [start_key]
        visited.add(start_key)
        component_point_indices: list[int] = []
        while queue:
            current = queue.pop()
            component_point_indices.extend(voxel_to_point_indices[current])
            for dx, dy, dz in neighbor_offsets:
                neighbor = (current[0] + dx, current[1] + dy, current[2] + dz)
                if neighbor in voxel_to_point_indices and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        components.append(np.asarray(component_point_indices, dtype=np.int32))
    components.sort(key=len, reverse=True)
    return components


def _build_planar_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z_axis = _normalize(np.asarray(normal, dtype=np.float32), np.array([0.0, 0.0, 1.0], dtype=np.float32))
    trial = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if abs(float(z_axis @ trial)) > 0.9:
        trial = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    basis_x = trial - z_axis * float(trial @ z_axis)
    basis_x = _normalize(basis_x, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    basis_y = _normalize(np.cross(z_axis, basis_x), np.array([0.0, 1.0, 0.0], dtype=np.float32))
    return basis_x, basis_y


def _select_dense_planar_component(
    points: np.ndarray,
    *,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
    min_points: int = 64,
) -> np.ndarray:
    cloud = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if len(cloud) <= min_points:
        return np.arange(len(cloud), dtype=np.int32)

    basis_x, basis_y = _build_planar_basis(plane_normal)
    centered = cloud - np.asarray(plane_point, dtype=np.float32)[None, :]
    planar = np.stack([centered @ basis_x, centered @ basis_y], axis=1).astype(np.float32)
    planar_min = planar.min(axis=0)
    planar_max = planar.max(axis=0)
    planar_extent = planar_max - planar_min
    cell_size = max(0.02, float(np.max(planar_extent)) * 0.06)
    keys = np.floor((planar - planar_min[None, :]) / cell_size).astype(np.int32)

    cell_to_indices: dict[tuple[int, int], list[int]] = {}
    for point_idx, key in enumerate(keys):
        cell_to_indices.setdefault((int(key[0]), int(key[1])), []).append(point_idx)

    if not cell_to_indices:
        return np.arange(len(cloud), dtype=np.int32)

    counts = {cell: len(indices) for cell, indices in cell_to_indices.items()}
    seed_cell = max(counts, key=counts.get)
    seed_count = counts[seed_cell]
    count_threshold = max(2, int(round(seed_count * 0.22)))
    visited: set[tuple[int, int]] = set()
    queue = [seed_cell]
    visited.add(seed_cell)
    selected_cells: set[tuple[int, int]] = set()
    while queue:
        current = queue.pop()
        if counts.get(current, 0) < count_threshold:
            continue
        selected_cells.add(current)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                neighbor = (current[0] + dx, current[1] + dy)
                if neighbor in counts and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

    if not selected_cells:
        selected_cells = {seed_cell}

    selected_indices = np.concatenate(
        [np.asarray(cell_to_indices[cell], dtype=np.int32) for cell in selected_cells],
        axis=0,
    )
    if len(selected_indices) < min_points:
        centroid = planar[np.asarray(cell_to_indices[seed_cell], dtype=np.int32)].mean(axis=0)
        planar_radius = np.linalg.norm(planar - centroid[None, :], axis=1)
        radius_threshold = max(cell_size * 2.5, float(np.quantile(planar_radius, 0.18)))
        selected_indices = np.where(planar_radius <= radius_threshold)[0].astype(np.int32)
        if len(selected_indices) < min_points:
            nearest = np.argsort(planar_radius)[: min(min_points, len(planar_radius))]
            selected_indices = nearest.astype(np.int32)
    return np.unique(selected_indices)


def filter_points_to_object_region(
    points: np.ndarray,
    colors: np.ndarray,
    *,
    object_roi_min: np.ndarray,
    object_roi_max: np.ndarray,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
    min_height: float = 0.0,
    max_height: float = 0.40,
    xy_margin: float = 0.02,
    z_margin: float = 0.02,
    table_color_bgr: np.ndarray | None = None,
    table_color_threshold: float = 26.0,
    table_plane_band: float = 0.03,
) -> tuple[np.ndarray, np.ndarray]:
    cloud = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    color_values = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    if len(cloud) == 0:
        return cloud, color_values
    roi_min = np.asarray(object_roi_min, dtype=np.float32) - np.array([xy_margin, xy_margin, z_margin], dtype=np.float32)
    roi_max = np.asarray(object_roi_max, dtype=np.float32) + np.array([xy_margin, xy_margin, z_margin], dtype=np.float32)
    in_bounds = np.all(cloud >= roi_min[None, :], axis=1) & np.all(cloud <= roi_max[None, :], axis=1)
    signed_height = (cloud - np.asarray(plane_point, dtype=np.float32)[None, :]) @ np.asarray(plane_normal, dtype=np.float32)
    keep = in_bounds & (signed_height >= float(min_height)) & (signed_height <= float(max_height))
    if table_color_bgr is not None and np.any(keep):
        table_color = np.asarray(table_color_bgr, dtype=np.float32).reshape(1, 3)
        color_distance = np.linalg.norm(color_values.astype(np.float32) - table_color, axis=1)
        keep &= ~((color_distance <= float(table_color_threshold)) & (signed_height <= float(table_plane_band)))
    return cloud[keep], color_values[keep]


def estimate_table_color_bgr(
    points: np.ndarray,
    colors: np.ndarray,
    *,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
    height_band: float = 0.015,
) -> np.ndarray:
    cloud = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    color_values = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    if len(cloud) == 0:
        return np.array([128.0, 128.0, 128.0], dtype=np.float32)
    signed_height = (cloud - np.asarray(plane_point, dtype=np.float32)[None, :]) @ np.asarray(plane_normal, dtype=np.float32)
    mask = np.abs(signed_height) <= float(height_band)
    if np.count_nonzero(mask) < 128:
        mask = signed_height <= float(height_band * 2.0)
    if np.count_nonzero(mask) < 32:
        return np.median(color_values.astype(np.float32), axis=0).astype(np.float32)
    return np.median(color_values[mask].astype(np.float32), axis=0).astype(np.float32)


def estimate_object_roi_bounds(
    points: np.ndarray,
    *,
    fallback_bounds: dict[str, np.ndarray],
    full_bounds: dict[str, np.ndarray],
    object_height_min: float = 0.02,
    object_height_max: float = 0.30,
    object_component_mode: str = "largest",
    object_component_topk: int = 2,
    roi_margin_xy: float = 0.05,
    roi_margin_z: float = 0.03,
) -> dict[str, Any]:
    cloud = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if len(cloud) < 64:
        return {
            "mode": "auto_object_bbox",
            "min": np.asarray(fallback_bounds["min"], dtype=np.float32),
            "max": np.asarray(fallback_bounds["max"], dtype=np.float32),
            "object_roi_min": np.asarray(fallback_bounds["min"], dtype=np.float32),
            "object_roi_max": np.asarray(fallback_bounds["max"], dtype=np.float32),
            "plane_point": np.zeros((3,), dtype=np.float32),
            "plane_normal": np.array([0.0, 0.0, 1.0], dtype=np.float32),
            "fallback_used": True,
        }

    plane = fit_dominant_table_plane(cloud)
    object_mask, signed_height = estimate_object_support_mask(
        cloud,
        plane_point=plane["point"],
        plane_normal=plane["normal"],
        object_height_min=object_height_min,
        object_height_max=object_height_max,
    )
    object_points = cloud[object_mask]
    if len(object_points) < 32:
        return {
            "mode": "auto_object_bbox",
            "min": np.asarray(fallback_bounds["min"], dtype=np.float32),
            "max": np.asarray(fallback_bounds["max"], dtype=np.float32),
            "object_roi_min": np.asarray(fallback_bounds["min"], dtype=np.float32),
            "object_roi_max": np.asarray(fallback_bounds["max"], dtype=np.float32),
            "plane_point": plane["point"],
            "plane_normal": plane["normal"],
            "fallback_used": True,
        }

    scene_extent = float(np.linalg.norm(object_points.max(axis=0) - object_points.min(axis=0)))
    voxel_size = max(0.04, scene_extent * 0.10)
    components = _build_voxel_components(object_points, voxel_size=voxel_size)
    if not components:
        component_points = object_points
    elif object_component_mode == "largest":
        component_points = object_points[components[0]]
    else:
        topk = max(1, int(object_component_topk))
        merged_indices = np.concatenate(components[:topk], axis=0)
        component_points = object_points[merged_indices]

    dense_indices = _select_dense_planar_component(
        component_points,
        plane_point=plane["point"],
        plane_normal=plane["normal"],
    )
    selected_points = component_points[dense_indices]
    if len(selected_points) >= 64:
        basis_x, basis_y = _build_planar_basis(plane["normal"])
        selected_centered = selected_points - plane["point"][None, :]
        selected_planar = np.stack([selected_centered @ basis_x, selected_centered @ basis_y], axis=1).astype(np.float32)
        signed_height_selected = (selected_points - plane["point"][None, :]) @ plane["normal"]
        weights = np.clip(signed_height_selected, 1e-3, None).astype(np.float32)
        planar_centroid = np.average(selected_planar, axis=0, weights=weights)
        planar_distance = np.linalg.norm(selected_planar - planar_centroid[None, :], axis=1)
        radius_threshold = float(np.quantile(planar_distance, 0.58))
        radius_threshold = float(np.clip(radius_threshold, 0.08, 0.24))
        compact_mask = planar_distance <= radius_threshold
        compact_points = selected_points[compact_mask]
        if len(compact_points) >= 48:
            selected_points = compact_points

    roi_min = selected_points.min(axis=0).astype(np.float32)
    roi_max = selected_points.max(axis=0).astype(np.float32)
    crop_min = roi_min - np.array([roi_margin_xy, roi_margin_xy, roi_margin_z], dtype=np.float32)
    crop_max = roi_max + np.array([roi_margin_xy, roi_margin_xy, roi_margin_z], dtype=np.float32)
    crop_min = np.maximum(crop_min, np.asarray(full_bounds["min"], dtype=np.float32))
    crop_max = np.minimum(crop_max, np.asarray(full_bounds["max"], dtype=np.float32))
    return {
        "mode": "auto_object_bbox",
        "min": crop_min.astype(np.float32),
        "max": crop_max.astype(np.float32),
        "object_roi_min": roi_min.astype(np.float32),
        "object_roi_max": roi_max.astype(np.float32),
        "plane_point": plane["point"],
        "plane_normal": plane["normal"],
        "object_point_count": int(len(object_points)),
        "component_point_count": int(len(component_points)),
        "selected_object_point_count": int(len(selected_points)),
        "fallback_used": False,
    }
