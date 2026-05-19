from __future__ import annotations

from typing import Any

import numpy as np


def _detect_radius_outlier_indices(
    points_world: np.ndarray,
    *,
    radius_m: float,
    nb_points: int,
) -> dict[str, np.ndarray]:
    cloud = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    point_count = int(len(cloud))
    if point_count == 0:
        empty = np.empty((0,), dtype=np.int32)
        return {"inlier_indices": empty, "outlier_indices": empty}

    if point_count <= 4096:
        radius_sq = float(radius_m) * float(radius_m)
        deltas = cloud[:, None, :] - cloud[None, :, :]
        neighbor_counts = np.count_nonzero(np.sum(deltas * deltas, axis=2) <= radius_sq + 1e-12, axis=1)
        inliers = np.flatnonzero(neighbor_counts >= int(nb_points)).astype(np.int32)
        keep_mask = np.zeros((point_count,), dtype=bool)
        keep_mask[inliers] = True
        outliers = np.flatnonzero(~keep_mask).astype(np.int32)
        return {"inlier_indices": inliers, "outlier_indices": outliers}

    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud)
    _, inlier_indices = pcd.remove_radius_outlier(
        nb_points=int(nb_points),
        radius=float(radius_m),
    )
    inliers = np.asarray(inlier_indices, dtype=np.int32).reshape(-1)
    if len(inliers) == 0:
        return {
            "inlier_indices": inliers,
            "outlier_indices": np.arange(point_count, dtype=np.int32),
        }
    keep_mask = np.zeros((point_count,), dtype=bool)
    keep_mask[inliers] = True
    outliers = np.flatnonzero(~keep_mask).astype(np.int32)
    return {"inlier_indices": inliers, "outlier_indices": outliers}


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


def apply_phystwin_like_radius_postprocess(
    *,
    points: np.ndarray,
    colors: np.ndarray,
    enabled: bool,
    radius_m: float,
    nb_points: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    point_array = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    color_array = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    point_count = int(len(point_array))
    stats = {
        "enabled": bool(enabled),
        "mode": "phystwin_like_radius_neighbor_filter",
        "radius_m": float(radius_m),
        "nb_points": int(nb_points),
        "input_point_count": point_count,
        "inlier_point_count": point_count,
        "outlier_point_count": 0,
        "outlier_ratio": 0.0,
    }
    if not enabled or point_count == 0:
        return point_array, color_array, stats

    result = _detect_radius_outlier_indices(
        point_array,
        radius_m=float(radius_m),
        nb_points=int(nb_points),
    )
    inlier_indices = np.sort(np.asarray(result["inlier_indices"], dtype=np.int32).reshape(-1))
    outlier_count = int(point_count - len(inlier_indices))
    stats.update(
        {
            "inlier_point_count": int(len(inlier_indices)),
            "outlier_point_count": outlier_count,
            "outlier_ratio": float(outlier_count / max(1, point_count)),
        }
    )
    return point_array[inlier_indices], color_array[inlier_indices], stats


def _bbox_gap_m(left_min: np.ndarray, left_max: np.ndarray, right_min: np.ndarray, right_max: np.ndarray) -> float:
    left_lo = np.asarray(left_min, dtype=np.float32).reshape(3)
    left_hi = np.asarray(left_max, dtype=np.float32).reshape(3)
    right_lo = np.asarray(right_min, dtype=np.float32).reshape(3)
    right_hi = np.asarray(right_max, dtype=np.float32).reshape(3)
    axis_gap = np.maximum(np.maximum(left_lo - right_hi, right_lo - left_hi), 0.0)
    return float(np.linalg.norm(axis_gap))


def _component_summary(
    *,
    component_idx: int,
    component_indices: np.ndarray,
    points: np.ndarray,
    main_bbox_min: np.ndarray,
    main_bbox_max: np.ndarray,
    kept: bool,
) -> dict[str, Any]:
    indices = np.asarray(component_indices, dtype=np.int32).reshape(-1)
    component_points = np.asarray(points, dtype=np.float32)[indices]
    bbox_min = component_points.min(axis=0).astype(np.float32)
    bbox_max = component_points.max(axis=0).astype(np.float32)
    centroid = component_points.mean(axis=0).astype(np.float32)
    extent = (bbox_max - bbox_min).astype(np.float32)
    return {
        "component_idx": int(component_idx),
        "kept": bool(kept),
        "point_count": int(len(indices)),
        "bbox_min": [float(item) for item in bbox_min],
        "bbox_max": [float(item) for item in bbox_max],
        "bbox_extent": [float(item) for item in extent],
        "centroid": [float(item) for item in centroid],
        "bbox_gap_to_main_m": float(_bbox_gap_m(bbox_min, bbox_max, main_bbox_min, main_bbox_max)),
    }


def apply_enhanced_phystwin_like_postprocess_with_trace(
    *,
    points: np.ndarray,
    colors: np.ndarray,
    enabled: bool,
    radius_m: float,
    nb_points: int,
    component_voxel_size_m: float,
    keep_near_main_gap_m: float = 0.0,
    max_component_report_count: int = 32,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    point_array = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    color_array = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    input_point_count = int(len(point_array))
    radius_kept_mask = np.ones((input_point_count,), dtype=bool)
    radius_stats = {
        "enabled": bool(enabled),
        "mode": "phystwin_like_radius_neighbor_filter",
        "radius_m": float(radius_m),
        "nb_points": int(nb_points),
        "input_point_count": input_point_count,
        "inlier_point_count": input_point_count,
        "outlier_point_count": 0,
        "outlier_ratio": 0.0,
    }
    if bool(enabled) and input_point_count > 0:
        result = _detect_radius_outlier_indices(
            point_array,
            radius_m=float(radius_m),
            nb_points=int(nb_points),
        )
        inlier_indices = np.sort(np.asarray(result["inlier_indices"], dtype=np.int32).reshape(-1))
        radius_kept_mask[:] = False
        radius_kept_mask[inlier_indices] = True
        outlier_count = int(input_point_count - len(inlier_indices))
        radius_stats.update(
            {
                "inlier_point_count": int(len(inlier_indices)),
                "outlier_point_count": outlier_count,
                "outlier_ratio": float(outlier_count / max(1, input_point_count)),
            }
        )
    radius_indices = np.where(radius_kept_mask)[0].astype(np.int32)
    radius_points = point_array[radius_indices]
    radius_colors = color_array[radius_indices]
    component_removed_mask = np.zeros((input_point_count,), dtype=bool)
    trace = {
        "kept_mask": radius_kept_mask.copy(),
        "radius_removed_mask": ~radius_kept_mask,
        "component_removed_mask": component_removed_mask.copy(),
        "removed_mask": ~radius_kept_mask,
    }
    stats: dict[str, Any] = {
        "enabled": bool(enabled),
        "mode": "enhanced_phystwin_like_radius_then_component_filter",
        "radius_postprocess": dict(radius_stats),
        "component_filter_enabled": bool(enabled),
        "component_voxel_size_m": float(component_voxel_size_m),
        "keep_near_main_gap_m": float(keep_near_main_gap_m),
        "input_point_count": input_point_count,
        "after_radius_point_count": int(len(radius_points)),
        "output_point_count": int(len(radius_points)),
        "component_count": 0,
        "kept_component_indices": [],
        "removed_component_count": 0,
        "removed_point_count": 0,
        "removed_point_ratio_after_radius": 0.0,
        "components": [],
        "removed_components": [],
    }
    if not enabled or len(radius_points) == 0:
        return radius_points, radius_colors, stats, trace
    if float(component_voxel_size_m) <= 0.0:
        raise ValueError(f"component_voxel_size_m must be positive, got {component_voxel_size_m}.")
    if float(keep_near_main_gap_m) < 0.0:
        raise ValueError(f"keep_near_main_gap_m must be >= 0, got {keep_near_main_gap_m}.")

    components = _build_voxel_components(radius_points, voxel_size=float(component_voxel_size_m))
    stats["component_count"] = int(len(components))
    if len(components) <= 1:
        if components:
            stats["kept_component_indices"] = [0]
            main_points = radius_points[np.asarray(components[0], dtype=np.int32)]
            main_bbox_min = main_points.min(axis=0)
            main_bbox_max = main_points.max(axis=0)
            stats["components"] = [
                _component_summary(
                    component_idx=0,
                    component_indices=components[0],
                    points=radius_points,
                    main_bbox_min=main_bbox_min,
                    main_bbox_max=main_bbox_max,
                    kept=True,
                )
            ]
        return radius_points, radius_colors, stats, trace

    main_points = radius_points[np.asarray(components[0], dtype=np.int32)]
    main_bbox_min = main_points.min(axis=0).astype(np.float32)
    main_bbox_max = main_points.max(axis=0).astype(np.float32)
    component_keep_mask = np.zeros((len(radius_points),), dtype=bool)
    kept_component_indices: list[int] = []
    component_summaries: list[dict[str, Any]] = []
    removed_summaries: list[dict[str, Any]] = []
    for component_idx, component_indices in enumerate(components):
        indices = np.asarray(component_indices, dtype=np.int32).reshape(-1)
        if component_idx == 0:
            keep_component = True
        else:
            component_points = radius_points[indices]
            bbox_min = component_points.min(axis=0).astype(np.float32)
            bbox_max = component_points.max(axis=0).astype(np.float32)
            keep_component = _bbox_gap_m(bbox_min, bbox_max, main_bbox_min, main_bbox_max) <= float(keep_near_main_gap_m)
        if keep_component:
            component_keep_mask[indices] = True
            kept_component_indices.append(int(component_idx))
        summary = _component_summary(
            component_idx=int(component_idx),
            component_indices=indices,
            points=radius_points,
            main_bbox_min=main_bbox_min,
            main_bbox_max=main_bbox_max,
            kept=bool(keep_component),
        )
        component_summaries.append(summary)
        if not keep_component:
            removed_summaries.append(summary)

    kept_count = int(np.count_nonzero(component_keep_mask))
    removed_count = int(len(radius_points) - kept_count)
    component_removed_radius_mask = ~component_keep_mask
    component_removed_mask[radius_indices[component_removed_radius_mask]] = True
    kept_mask = radius_kept_mask & ~component_removed_mask
    trace = {
        "kept_mask": kept_mask,
        "radius_removed_mask": ~radius_kept_mask,
        "component_removed_mask": component_removed_mask,
        "removed_mask": (~radius_kept_mask) | component_removed_mask,
    }
    stats.update(
        {
            "output_point_count": kept_count,
            "kept_component_indices": kept_component_indices,
            "removed_component_count": int(len(removed_summaries)),
            "removed_point_count": removed_count,
            "removed_point_ratio_after_radius": float(removed_count / max(1, len(radius_points))),
            "components": component_summaries[: max(1, int(max_component_report_count))],
            "removed_components": removed_summaries[: max(1, int(max_component_report_count))],
        }
    )
    return radius_points[component_keep_mask], radius_colors[component_keep_mask], stats, trace


def apply_enhanced_phystwin_like_postprocess(
    *,
    points: np.ndarray,
    colors: np.ndarray,
    enabled: bool,
    radius_m: float,
    nb_points: int,
    component_voxel_size_m: float,
    keep_near_main_gap_m: float = 0.0,
    max_component_report_count: int = 32,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    filtered_points, filtered_colors, stats, _trace = apply_enhanced_phystwin_like_postprocess_with_trace(
        points=points,
        colors=colors,
        enabled=enabled,
        radius_m=radius_m,
        nb_points=nb_points,
        component_voxel_size_m=component_voxel_size_m,
        keep_near_main_gap_m=keep_near_main_gap_m,
        max_component_report_count=max_component_report_count,
    )
    return filtered_points, filtered_colors, stats


__all__ = [
    "apply_enhanced_phystwin_like_postprocess",
    "apply_enhanced_phystwin_like_postprocess_with_trace",
    "apply_phystwin_like_radius_postprocess",
]
