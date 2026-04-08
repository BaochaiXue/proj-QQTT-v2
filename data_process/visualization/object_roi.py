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
        selected_points = object_points
    elif object_component_mode == "largest":
        selected_points = object_points[components[0]]
    else:
        topk = max(1, int(object_component_topk))
        merged_indices = np.concatenate(components[:topk], axis=0)
        selected_points = object_points[merged_indices]

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
        "selected_object_point_count": int(len(selected_points)),
        "fallback_used": False,
    }
