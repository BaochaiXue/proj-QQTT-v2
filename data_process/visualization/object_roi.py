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

    def _fit_plane_from_points(sample: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        vec0 = sample[1] - sample[0]
        vec1 = sample[2] - sample[0]
        normal = np.cross(vec0, vec1)
        if float(np.linalg.norm(normal)) <= 1e-6:
            return None
        normal = _normalize(normal, np.array([0.0, 0.0, 1.0], dtype=np.float32))
        point = sample[0].astype(np.float32)
        return point, normal.astype(np.float32)

    rng = np.random.default_rng(0)
    best_inlier_mask = None
    best_inlier_count = -1
    best_residual = np.inf
    candidate_count = len(candidates)
    ransac_trials = min(256, max(32, candidate_count // 12))
    distance_threshold = 0.012
    for _ in range(ransac_trials):
        sample_indices = rng.choice(candidate_count, size=3, replace=False)
        plane = _fit_plane_from_points(candidates[sample_indices])
        if plane is None:
            continue
        plane_point, plane_normal = plane
        signed_distance = np.abs((candidates - plane_point[None, :]) @ plane_normal)
        inlier_mask = signed_distance <= distance_threshold
        inlier_count = int(np.count_nonzero(inlier_mask))
        if inlier_count < 64:
            continue
        residual = float(np.median(signed_distance[inlier_mask]))
        if inlier_count > best_inlier_count or (inlier_count == best_inlier_count and residual < best_residual):
            best_inlier_mask = inlier_mask
            best_inlier_count = inlier_count
            best_residual = residual

    fit_points = candidates if best_inlier_mask is None else candidates[best_inlier_mask]
    plane_point = np.median(fit_points, axis=0).astype(np.float32)
    centered = fit_points - plane_point[None, :]
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


def _compute_object_component_stats(
    object_points: np.ndarray,
    components: list[np.ndarray],
    *,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
) -> list[dict[str, Any]]:
    stats: list[dict[str, Any]] = []
    for component_idx, component_indices in enumerate(components):
        component_points = object_points[component_indices]
        heights = (component_points - np.asarray(plane_point, dtype=np.float32)[None, :]) @ np.asarray(plane_normal, dtype=np.float32)
        bbox_min = component_points.min(axis=0).astype(np.float32)
        bbox_max = component_points.max(axis=0).astype(np.float32)
        centroid = component_points.mean(axis=0).astype(np.float32)
        extent = bbox_max - bbox_min
        planar_extent = float(np.linalg.norm(extent[:2]))
        vertical_extent = float(np.max(heights) - np.min(heights))
        min_height = float(np.min(heights))
        max_height = float(np.max(heights))
        median_height = float(np.median(heights))
        p90_height = float(np.quantile(heights, 0.90))
        stability = float(min(vertical_extent, 0.25) + min(planar_extent, 0.35) * 0.35)
        score = (
            float(len(component_indices))
            + p90_height * 800.0
            + median_height * 300.0
            + stability * 120.0
        )
        stats.append(
            {
                "component_idx": int(component_idx),
                "point_count": int(len(component_indices)),
                "bbox_min": bbox_min,
                "bbox_max": bbox_max,
                "centroid": centroid,
                "extent": extent.astype(np.float32),
                "planar_extent": planar_extent,
                "vertical_extent": vertical_extent,
                "min_height": min_height,
                "max_height": max_height,
                "median_height": median_height,
                "p90_height": p90_height,
                "score": score,
            }
        )
    stats.sort(key=lambda item: item["score"], reverse=True)
    return stats


def _bbox_gap_vector(
    bbox_min_a: np.ndarray,
    bbox_max_a: np.ndarray,
    bbox_min_b: np.ndarray,
    bbox_max_b: np.ndarray,
) -> np.ndarray:
    left_gap = np.maximum(bbox_min_b - bbox_max_a, 0.0)
    right_gap = np.maximum(bbox_min_a - bbox_max_b, 0.0)
    return np.maximum(left_gap, right_gap).astype(np.float32)


def _bbox_overlap_with_margin(
    bbox_min_a: np.ndarray,
    bbox_max_a: np.ndarray,
    bbox_min_b: np.ndarray,
    bbox_max_b: np.ndarray,
    *,
    margin_xy: float,
    margin_z: float,
) -> bool:
    margin = np.array([float(margin_xy), float(margin_xy), float(margin_z)], dtype=np.float32)
    expanded_min_a = np.asarray(bbox_min_a, dtype=np.float32) - margin
    expanded_max_a = np.asarray(bbox_max_a, dtype=np.float32) + margin
    expanded_min_b = np.asarray(bbox_min_b, dtype=np.float32) - margin
    expanded_max_b = np.asarray(bbox_max_b, dtype=np.float32) + margin
    return bool(
        np.all(expanded_max_a >= expanded_min_b) and np.all(expanded_min_a <= expanded_max_b)
    )


def _projected_bbox_iou(
    bbox_a: tuple[int, int, int, int] | None,
    bbox_b: tuple[int, int, int, int] | None,
) -> float:
    if bbox_a is None or bbox_b is None:
        return 0.0
    ax0, ay0, ax1, ay1 = bbox_a
    bx0, by0, bx1, by1 = bbox_b
    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    inter_w = max(0, inter_x1 - inter_x0 + 1)
    inter_h = max(0, inter_y1 - inter_y0 + 1)
    inter_area = float(inter_w * inter_h)
    if inter_area <= 0.0:
        return 0.0
    area_a = float(max(0, ax1 - ax0 + 1) * max(0, ay1 - ay0 + 1))
    area_b = float(max(0, bx1 - bx0 + 1) * max(0, by1 - by0 + 1))
    denom = area_a + area_b - inter_area
    if denom <= 1e-6:
        return 0.0
    return inter_area / denom


def _components_are_graph_connected(
    component_a: dict[str, Any],
    component_b: dict[str, Any],
    *,
    projected_bboxes_by_component: dict[int, dict[int, tuple[int, int, int, int]]] | None = None,
) -> bool:
    bbox_min_a = np.asarray(component_a["bbox_min"], dtype=np.float32)
    bbox_max_a = np.asarray(component_a["bbox_max"], dtype=np.float32)
    bbox_min_b = np.asarray(component_b["bbox_min"], dtype=np.float32)
    bbox_max_b = np.asarray(component_b["bbox_max"], dtype=np.float32)
    extent_a = np.asarray(component_a["extent"], dtype=np.float32)
    extent_b = np.asarray(component_b["extent"], dtype=np.float32)
    planar_extent_a = max(float(component_a["planar_extent"]), 1e-3)
    planar_extent_b = max(float(component_b["planar_extent"]), 1e-3)
    vertical_extent_a = max(float(component_a["vertical_extent"]), 1e-3)
    vertical_extent_b = max(float(component_b["vertical_extent"]), 1e-3)
    centroid_a = np.asarray(component_a["centroid"], dtype=np.float32)
    centroid_b = np.asarray(component_b["centroid"], dtype=np.float32)

    gap_vector = _bbox_gap_vector(bbox_min_a, bbox_max_a, bbox_min_b, bbox_max_b)
    planar_gap = float(np.linalg.norm(gap_vector[:2]))
    vertical_gap = float(gap_vector[2])
    centroid_planar = float(np.linalg.norm((centroid_a - centroid_b)[:2]))
    centroid_vertical = float(abs(float(centroid_a[2] - centroid_b[2])))

    margin_xy = max(0.035, min(planar_extent_a, planar_extent_b) * 0.70 + 0.02)
    margin_z = max(0.025, max(vertical_extent_a, vertical_extent_b) * 1.25 + 0.015)
    overlaps_expanded = _bbox_overlap_with_margin(
        bbox_min_a,
        bbox_max_a,
        bbox_min_b,
        bbox_max_b,
        margin_xy=margin_xy,
        margin_z=margin_z,
    )
    centroid_planar_threshold = max(0.08, (planar_extent_a + planar_extent_b) * 0.95)
    centroid_vertical_threshold = max(0.05, (vertical_extent_a + vertical_extent_b) * 1.50 + 0.02)
    height_continuity = not (
        float(component_a["max_height"]) + margin_z < float(component_b["min_height"])
        or float(component_b["max_height"]) + margin_z < float(component_a["min_height"])
    )

    projected_overlap = 0.0
    if projected_bboxes_by_component is not None:
        bboxes_a = projected_bboxes_by_component.get(int(component_a["component_idx"]), {})
        bboxes_b = projected_bboxes_by_component.get(int(component_b["component_idx"]), {})
        shared_camera_ids = set(bboxes_a.keys()) & set(bboxes_b.keys())
        if shared_camera_ids:
            projected_overlap = max(
                _projected_bbox_iou(bboxes_a.get(camera_idx), bboxes_b.get(camera_idx))
                for camera_idx in shared_camera_ids
            )

    return bool(
        projected_overlap >= 0.03
        or overlaps_expanded
        or (
            planar_gap <= margin_xy * 1.35
            and vertical_gap <= margin_z * 1.15
            and centroid_planar <= centroid_planar_threshold
            and centroid_vertical <= centroid_vertical_threshold
            and height_continuity
        )
    )


def _select_graph_union_component_indices(
    component_stats: list[dict[str, Any]],
    *,
    projected_bboxes_by_component: dict[int, dict[int, tuple[int, int, int, int]]] | None = None,
) -> list[int]:
    if not component_stats:
        return []
    component_by_idx = {int(item["component_idx"]): item for item in component_stats}
    adjacency: dict[int, set[int]] = {int(item["component_idx"]): set() for item in component_stats}
    ordered = [int(item["component_idx"]) for item in component_stats]
    for left_offset, left_idx in enumerate(ordered):
        for right_idx in ordered[left_offset + 1:]:
            if _components_are_graph_connected(
                component_by_idx[left_idx],
                component_by_idx[right_idx],
                projected_bboxes_by_component=projected_bboxes_by_component,
            ):
                adjacency[left_idx].add(right_idx)
                adjacency[right_idx].add(left_idx)

    seed_idx = int(component_stats[0]["component_idx"])
    selected = {seed_idx}
    queue = [seed_idx]
    while queue:
        current = queue.pop()
        for neighbor in adjacency[current]:
            if neighbor not in selected:
                selected.add(neighbor)
                queue.append(neighbor)
    return sorted(selected)


def _select_object_component_indices(
    object_points: np.ndarray,
    components: list[np.ndarray],
    *,
    object_component_mode: str,
    object_component_topk: int,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
    projected_bboxes_by_component: dict[int, dict[int, tuple[int, int, int, int]]] | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]], list[int]]:
    if not components:
        fallback_indices = np.arange(len(object_points), dtype=np.int32)
        return fallback_indices, [], [0] if len(object_points) > 0 else []

    component_stats = _compute_object_component_stats(
        object_points,
        components,
        plane_point=plane_point,
        plane_normal=plane_normal,
    )
    components_by_idx = {int(idx): comp for idx, comp in enumerate(components)}
    top_component = component_stats[0]
    selected_component_indices: list[int]
    if object_component_mode == "largest":
        selected_component_indices = [int(top_component["component_idx"])]
    elif object_component_mode == "topk":
        topk = max(1, int(object_component_topk))
        selected_component_indices = [int(item["component_idx"]) for item in component_stats[:topk]]
    elif object_component_mode == "graph_union":
        selected_component_indices = _select_graph_union_component_indices(
            component_stats,
            projected_bboxes_by_component=projected_bboxes_by_component,
        )
    elif object_component_mode == "union":
        largest_count = max(1, int(top_component["point_count"]))
        top_score = max(1e-6, float(top_component["score"]))
        top_bbox_min = np.asarray(top_component["bbox_min"], dtype=np.float32)
        top_bbox_max = np.asarray(top_component["bbox_max"], dtype=np.float32)
        expanded_min = top_bbox_min - np.array([0.08, 0.08, 0.05], dtype=np.float32)
        expanded_max = top_bbox_max + np.array([0.08, 0.08, 0.05], dtype=np.float32)
        selected_component_indices = []
        for item in component_stats:
            bbox_min = np.asarray(item["bbox_min"], dtype=np.float32)
            bbox_max = np.asarray(item["bbox_max"], dtype=np.float32)
            overlaps_anchor = bool(np.all(bbox_max >= expanded_min) and np.all(bbox_min <= expanded_max))
            if (
                float(item["score"]) >= top_score * 0.22
                or int(item["point_count"]) >= max(48, int(round(largest_count * 0.08)))
                or overlaps_anchor
            ):
                selected_component_indices.append(int(item["component_idx"]))
    else:
        raise ValueError(f"Unsupported object_component_mode: {object_component_mode}")

    merged_indices = np.concatenate(
        [np.asarray(components_by_idx[idx], dtype=np.int32) for idx in selected_component_indices],
        axis=0,
    )
    return np.unique(merged_indices), component_stats, selected_component_indices


def compute_object_region_mask(
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
) -> np.ndarray:
    cloud = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    color_values = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    if len(cloud) == 0:
        return np.zeros((0,), dtype=bool)
    roi_min = np.asarray(object_roi_min, dtype=np.float32) - np.array([xy_margin, xy_margin, z_margin], dtype=np.float32)
    roi_max = np.asarray(object_roi_max, dtype=np.float32) + np.array([xy_margin, xy_margin, z_margin], dtype=np.float32)
    in_bounds = np.all(cloud >= roi_min[None, :], axis=1) & np.all(cloud <= roi_max[None, :], axis=1)
    signed_height = (cloud - np.asarray(plane_point, dtype=np.float32)[None, :]) @ np.asarray(plane_normal, dtype=np.float32)
    keep = in_bounds & (signed_height >= float(min_height)) & (signed_height <= float(max_height))
    if table_color_bgr is not None and np.any(keep):
        table_color = np.asarray(table_color_bgr, dtype=np.float32).reshape(1, 3)
        color_distance = np.linalg.norm(color_values.astype(np.float32) - table_color, axis=1)
        keep &= ~((color_distance <= float(table_color_threshold)) & (signed_height <= float(table_plane_band)))
    return keep


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
    keep = compute_object_region_mask(
        cloud,
        color_values,
        object_roi_min=object_roi_min,
        object_roi_max=object_roi_max,
        plane_point=plane_point,
        plane_normal=plane_normal,
        min_height=min_height,
        max_height=max_height,
        xy_margin=xy_margin,
        z_margin=z_margin,
        table_color_bgr=table_color_bgr,
        table_color_threshold=table_color_threshold,
        table_plane_band=table_plane_band,
    )
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
    plane_reference_points: np.ndarray | None = None,
    object_height_min: float = 0.02,
    object_height_max: float = 0.30,
    object_component_mode: str = "graph_union",
    object_component_topk: int = 2,
    roi_margin_xy: float = 0.05,
    roi_margin_z: float = 0.03,
    projected_bboxes_by_component: dict[int, dict[int, tuple[int, int, int, int]]] | None = None,
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

    plane_source = cloud if plane_reference_points is None else np.asarray(plane_reference_points, dtype=np.float32).reshape(-1, 3)
    if len(plane_source) < 64:
        plane_source = cloud
    plane = fit_dominant_table_plane(plane_source)
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
    voxel_size = max(0.025, scene_extent * 0.08)
    components = _build_voxel_components(object_points, voxel_size=voxel_size)
    selected_indices, component_stats, selected_component_indices = _select_object_component_indices(
        object_points,
        components,
        object_component_mode=object_component_mode,
        object_component_topk=object_component_topk,
        plane_point=plane["point"],
        plane_normal=plane["normal"],
        projected_bboxes_by_component=projected_bboxes_by_component,
    )
    component_points = object_points[selected_indices]
    selected_points = component_points

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
        "component_count": int(len(components)),
        "selected_component_indices": [int(item) for item in selected_component_indices],
        "component_scores": [
            {
                "component_idx": int(item["component_idx"]),
                "point_count": int(item["point_count"]),
                "median_height": float(item["median_height"]),
                "p90_height": float(item["p90_height"]),
                "score": float(item["score"]),
            }
            for item in component_stats[: min(len(component_stats), 8)]
        ],
        "fallback_used": False,
    }
