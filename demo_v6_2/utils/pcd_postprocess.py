"""PhysTwin-parity radius/component point-cloud postprocess filters."""

from __future__ import annotations

import time
from typing import Any

import numpy as np


# Component-selection policies for the enhanced postprocess:
# - main-plus-gap: keep the largest component plus anything whose bbox gap to it is small.
# - largest-n: keep only the N largest components that pass the min-size thresholds.
# - largest-n-plus-gap: largest-n, plus components close (bbox gap) to any kept one.
COMPONENT_SELECTION_MAIN_PLUS_GAP = "main-plus-gap"
COMPONENT_SELECTION_LARGEST_N = "largest-n"
COMPONENT_SELECTION_LARGEST_N_PLUS_GAP = "largest-n-plus-gap"
COMPONENT_SELECTION_POLICIES = (
    COMPONENT_SELECTION_MAIN_PLUS_GAP,
    COMPONENT_SELECTION_LARGEST_N,
    COMPONENT_SELECTION_LARGEST_N_PLUS_GAP,
)


def detect_radius_outlier_indices(
    points_world: np.ndarray,
    *,
    radius_m: float,
    nb_points: int,
) -> dict[str, np.ndarray]:
    """Split points into inliers/outliers by neighbor count within radius_m.

    A point is an inlier when it has at least nb_points neighbors (self included)
    inside radius_m — the same rule as Open3D remove_radius_outlier. Returned
    index arrays are int32 and sorted ascending.
    """
    cloud = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    point_count = int(len(cloud))
    if point_count == 0:
        empty = np.empty((0,), dtype=np.int32)
        return {"inlier_indices": empty, "outlier_indices": empty}

    # Deferred import: keep scipy off the module import path for non-filtering callers.
    from scipy.spatial import cKDTree

    tree = cKDTree(cloud)
    neighbor_counts = tree.query_ball_point(cloud, r=float(radius_m), return_length=True)
    inliers = np.flatnonzero(np.asarray(neighbor_counts, dtype=np.int64) >= int(nb_points)).astype(np.int32)
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
    """Cluster points into 26-connected voxel components, sorted largest-first.

    Points are binned into a voxel grid, then occupied voxels are flood-filled
    over their 26-neighborhood; each returned array holds the original point
    indices of one connected component. Voxel size is floored at 0.1 mm to
    guard against a degenerate zero/negative grid.
    """
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


def _radius_outlier_filter_stage(
    points: np.ndarray,
    colors: np.ndarray,
    *,
    enabled: bool,
    radius_m: float,
    nb_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], float]:
    """Shared radius-outlier stage for both postprocess entry points.

    Returns (point_array, color_array, kept_mask, stats, radius_filter_ms):
    ``stats`` carries the phystwin_like_radius_neighbor_filter fields (without
    the timing key, which callers attach as needed), ``kept_mask`` is the
    per-input-point inlier mask, and ``radius_filter_ms`` times only the
    neighbor query (0.0 when disabled or empty). Callers build their own trace
    dicts and slice the surviving points from ``kept_mask``.
    """
    point_array = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    color_array = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    point_count = int(len(point_array))
    kept_mask = np.ones((point_count,), dtype=bool)
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
    radius_filter_ms = 0.0
    if bool(enabled) and point_count > 0:
        radius_started_s = time.perf_counter()
        result = detect_radius_outlier_indices(
            point_array,
            radius_m=float(radius_m),
            nb_points=int(nb_points),
        )
        radius_filter_ms = float((time.perf_counter() - radius_started_s) * 1000.0)
        inlier_indices = np.sort(np.asarray(result["inlier_indices"], dtype=np.int32).reshape(-1))
        kept_mask[:] = False
        kept_mask[inlier_indices] = True
        outlier_count = int(point_count - len(inlier_indices))
        stats.update(
            {
                "inlier_point_count": int(len(inlier_indices)),
                "outlier_point_count": outlier_count,
                "outlier_ratio": float(outlier_count / max(1, point_count)),
            }
        )
    return point_array, color_array, kept_mask, stats, radius_filter_ms


def apply_phystwin_like_radius_postprocess_with_trace(
    *,
    points: np.ndarray,
    colors: np.ndarray,
    enabled: bool,
    radius_m: float,
    nb_points: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    """Radius-outlier filter returning (points, colors, stats, per-point trace masks).

    Trace masks are indexed against the input point order so callers can map
    removals back to source pixels/queries.
    """
    point_array, color_array, kept_mask, stats, _radius_filter_ms = _radius_outlier_filter_stage(
        points, colors, enabled=enabled, radius_m=radius_m, nb_points=nb_points
    )
    point_count = int(len(point_array))
    if not enabled or point_count == 0:
        trace = {
            "kept_mask": kept_mask.copy(),
            "radius_removed_mask": np.zeros((point_count,), dtype=bool),
            "removed_mask": np.zeros((point_count,), dtype=bool),
        }
        return point_array, color_array, stats, trace

    trace = {
        "kept_mask": kept_mask.copy(),
        "radius_removed_mask": ~kept_mask,
        "removed_mask": ~kept_mask,
    }
    return point_array[kept_mask], color_array[kept_mask], stats, trace


def _bbox_gap_m(left_min: np.ndarray, left_max: np.ndarray, right_min: np.ndarray, right_max: np.ndarray) -> float:
    """Euclidean gap (meters) between two axis-aligned boxes; 0.0 when they overlap."""
    left_lo = np.asarray(left_min, dtype=np.float32).reshape(3)
    left_hi = np.asarray(left_max, dtype=np.float32).reshape(3)
    right_lo = np.asarray(right_min, dtype=np.float32).reshape(3)
    right_hi = np.asarray(right_max, dtype=np.float32).reshape(3)
    axis_gap = np.maximum(np.maximum(left_lo - right_hi, right_lo - left_hi), 0.0)
    return float(np.linalg.norm(axis_gap))


def _component_summary(
    *,
    record: dict[str, Any],
    main_bbox_min: np.ndarray,
    main_bbox_max: np.ndarray,
    gap_to_kept_top_n_m: float,
    kept: bool,
) -> dict[str, Any]:
    """Convert an internal component record into the JSON-friendly stats payload shape."""
    bbox_min = np.asarray(record["bbox_min"], dtype=np.float32).reshape(3)
    bbox_max = np.asarray(record["bbox_max"], dtype=np.float32).reshape(3)
    centroid = np.asarray(record["centroid"], dtype=np.float32).reshape(3)
    extent = np.asarray(record["bbox_extent"], dtype=np.float32).reshape(3)
    return {
        "component_idx": int(record["component_idx"]),
        "kept": bool(kept),
        "point_count": int(record["point_count"]),
        "bbox_min": [float(item) for item in bbox_min],
        "bbox_max": [float(item) for item in bbox_max],
        "bbox_extent": [float(item) for item in extent],
        "centroid": [float(item) for item in centroid],
        "bbox_gap_to_main_m": float(_bbox_gap_m(bbox_min, bbox_max, main_bbox_min, main_bbox_max)),
        "bbox_gap_to_kept_top_n_m": float(gap_to_kept_top_n_m),
    }


def _component_records(components: list[np.ndarray], points: np.ndarray) -> list[dict[str, Any]]:
    """Precompute per-component geometry (indices, bbox, centroid) used by selection.

    component_idx follows the largest-first order from _build_voxel_components,
    so index 0 is always the main (largest) component.
    """
    point_array = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    records: list[dict[str, Any]] = []
    for component_idx, component_indices in enumerate(components):
        indices = np.asarray(component_indices, dtype=np.int32).reshape(-1)
        component_points = point_array[indices]
        bbox_min = component_points.min(axis=0).astype(np.float32)
        bbox_max = component_points.max(axis=0).astype(np.float32)
        centroid = component_points.mean(axis=0).astype(np.float32)
        records.append(
            {
                "component_idx": int(component_idx),
                "indices": indices,
                "point_count": int(len(indices)),
                "bbox_min": bbox_min,
                "bbox_max": bbox_max,
                "bbox_extent": (bbox_max - bbox_min).astype(np.float32),
                "centroid": centroid,
            }
        )
    return records


def _component_passes_min_thresholds(
    record: dict[str, Any],
    *,
    after_radius_point_count: int,
    min_component_points: int,
    min_component_ratio: float,
) -> bool:
    """Size gate for top-N candidacy; the main component (idx 0) always qualifies."""
    if int(record["component_idx"]) == 0:
        return True
    point_count = int(record["point_count"])
    if point_count < int(min_component_points):
        return False
    ratio = float(point_count / max(1, int(after_radius_point_count)))
    return ratio >= float(min_component_ratio)


def _gap_to_any_kept_top_n(record: dict[str, Any], top_records: list[dict[str, Any]]) -> float:
    """Smallest bbox gap (meters) from a component to any kept top-N component; inf when none kept."""
    if not top_records:
        return float("inf")
    bbox_min = np.asarray(record["bbox_min"], dtype=np.float32).reshape(3)
    bbox_max = np.asarray(record["bbox_max"], dtype=np.float32).reshape(3)
    return float(
        min(
            _bbox_gap_m(
                bbox_min,
                bbox_max,
                np.asarray(top_record["bbox_min"], dtype=np.float32).reshape(3),
                np.asarray(top_record["bbox_max"], dtype=np.float32).reshape(3),
            )
            for top_record in top_records
        )
    )


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
    keep_top_n_components: int = 1,
    component_selection_policy: str = COMPONENT_SELECTION_LARGEST_N_PLUS_GAP,
    min_component_points: int = 32,
    min_component_ratio: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    """Two-stage filter: radius-outlier removal, then connected-component selection.

    Stage 1 mirrors apply_phystwin_like_radius_postprocess_with_trace. Stage 2
    clusters the survivors into voxel components and keeps them according to
    component_selection_policy (see the policy constants at module top). Returns
    (points, colors, stats, trace); trace masks are indexed against the input
    point order, with component removals folded into removed_mask.
    """
    # ---- Stage 1: radius-outlier filter --------------------------------
    total_started_s = time.perf_counter()
    point_array, color_array, radius_kept_mask, radius_stats, radius_filter_ms = _radius_outlier_filter_stage(
        points, colors, enabled=enabled, radius_m=radius_m, nb_points=nb_points
    )
    radius_stats["radius_filter_ms"] = radius_filter_ms
    input_point_count = int(len(point_array))
    radius_indices = np.where(radius_kept_mask)[0].astype(np.int32)
    radius_points = point_array[radius_indices]
    radius_colors = color_array[radius_indices]

    # Seed trace/stats with the stage-1 result so every early return below is
    # already consistent (component fields stay at their "nothing removed" values).
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
        "component_selection_policy": str(component_selection_policy),
        "keep_top_n_components": int(keep_top_n_components),
        "min_component_points": int(min_component_points),
        "min_component_ratio": float(min_component_ratio),
        "input_point_count": input_point_count,
        "after_radius_point_count": int(len(radius_points)),
        "output_point_count": int(len(radius_points)),
        "component_count": 0,
        "component_point_counts": [],
        "top_n_component_indices": [],
        "top_n_component_point_counts": [],
        "kept_component_indices": [],
        "removed_component_indices": [],
        "kept_component_count": 0,
        "removed_component_count": 0,
        "removed_point_count": 0,
        "removed_point_ratio_after_radius": 0.0,
        "component_bbox_gap_to_kept_top_n_m": {},
        "radius_filter_ms": float(radius_stats.get("radius_filter_ms", 0.0)),
        "voxel_component_ms": 0.0,
        "component_selection_ms": 0.0,
        "total_ms": 0.0,
        "components": [],
        "removed_components": [],
    }
    if not enabled or len(radius_points) == 0:
        stats["total_ms"] = float((time.perf_counter() - total_started_s) * 1000.0)
        return radius_points, radius_colors, stats, trace
    if float(component_voxel_size_m) <= 0.0:
        raise ValueError(f"component_voxel_size_m must be positive, got {component_voxel_size_m}.")
    if float(keep_near_main_gap_m) < 0.0:
        raise ValueError(f"keep_near_main_gap_m must be >= 0, got {keep_near_main_gap_m}.")
    if int(keep_top_n_components) < 1:
        raise ValueError(f"keep_top_n_components must be >= 1, got {keep_top_n_components}.")
    if int(min_component_points) < 1:
        raise ValueError(f"min_component_points must be >= 1, got {min_component_points}.")
    if float(min_component_ratio) < 0.0:
        raise ValueError(f"min_component_ratio must be >= 0, got {min_component_ratio}.")
    policy = str(component_selection_policy)
    if policy not in COMPONENT_SELECTION_POLICIES:
        raise ValueError(f"component_selection_policy must be one of {COMPONENT_SELECTION_POLICIES}, got {policy}.")

    # ---- Stage 2: voxel connected components ----------------------------
    component_started_s = time.perf_counter()
    components = _build_voxel_components(radius_points, voxel_size=float(component_voxel_size_m))
    records = _component_records(components, radius_points)
    voxel_component_ms = float((time.perf_counter() - component_started_s) * 1000.0)
    stats["component_count"] = int(len(components))
    stats["component_point_counts"] = [int(record["point_count"]) for record in records]
    if not records:
        stats["voxel_component_ms"] = voxel_component_ms
        stats["total_ms"] = float((time.perf_counter() - total_started_s) * 1000.0)
        return radius_points, radius_colors, stats, trace

    main_bbox_min = np.asarray(records[0]["bbox_min"], dtype=np.float32).reshape(3)
    main_bbox_max = np.asarray(records[0]["bbox_max"], dtype=np.float32).reshape(3)

    # ---- Component selection --------------------------------------------
    # Pick the top-N candidates in largest-first order; records failing the size
    # gate are skipped rather than ending the scan. Guarantee at least the main
    # component so the output can never go empty at this stage.
    selection_started_s = time.perf_counter()
    top_n_records: list[dict[str, Any]] = []
    for record in records:
        if len(top_n_records) >= int(keep_top_n_components):
            break
        if policy == COMPONENT_SELECTION_MAIN_PLUS_GAP or _component_passes_min_thresholds(
            record,
            after_radius_point_count=int(len(radius_points)),
            min_component_points=int(min_component_points),
            min_component_ratio=float(min_component_ratio),
        ):
            top_n_records.append(record)
    if not top_n_records:
        top_n_records.append(records[0])
    top_n_indices = [int(record["component_idx"]) for record in top_n_records]
    top_n_set = set(top_n_indices)

    gap_to_kept_top_n: dict[int, float] = {
        int(record["component_idx"]): _gap_to_any_kept_top_n(record, top_n_records)
        for record in records
    }

    # Per-policy keep decision for every component (see policy constants).
    keep_component_by_idx: dict[int, bool] = {}
    for record in records:
        component_idx = int(record["component_idx"])
        if policy == COMPONENT_SELECTION_MAIN_PLUS_GAP:
            if component_idx == 0:
                keep_component = True
            else:
                keep_component = (
                    _bbox_gap_m(
                        np.asarray(record["bbox_min"], dtype=np.float32).reshape(3),
                        np.asarray(record["bbox_max"], dtype=np.float32).reshape(3),
                        main_bbox_min,
                        main_bbox_max,
                    )
                    <= float(keep_near_main_gap_m)
                )
        elif policy == COMPONENT_SELECTION_LARGEST_N:
            keep_component = component_idx in top_n_set
        else:
            keep_component = (
                component_idx in top_n_set
                or float(gap_to_kept_top_n[component_idx]) <= float(keep_near_main_gap_m)
            )
        keep_component_by_idx[component_idx] = bool(keep_component)

    component_keep_mask = np.zeros((len(radius_points),), dtype=bool)
    kept_component_indices: list[int] = []
    component_summaries: list[dict[str, Any]] = []
    removed_summaries: list[dict[str, Any]] = []
    removed_component_indices: list[int] = []
    for record in records:
        component_idx = int(record["component_idx"])
        indices = np.asarray(record["indices"], dtype=np.int32).reshape(-1)
        keep_component = bool(keep_component_by_idx[component_idx])
        if keep_component:
            component_keep_mask[indices] = True
            kept_component_indices.append(int(component_idx))
        summary = _component_summary(
            record=record,
            main_bbox_min=main_bbox_min,
            main_bbox_max=main_bbox_max,
            gap_to_kept_top_n_m=float(gap_to_kept_top_n[component_idx]),
            kept=bool(keep_component),
        )
        component_summaries.append(summary)
        if not keep_component:
            removed_component_indices.append(int(component_idx))
            removed_summaries.append(summary)

    # ---- Fold component removals into the input-indexed trace masks -----
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
            "top_n_component_indices": top_n_indices,
            "top_n_component_point_counts": [int(record["point_count"]) for record in top_n_records],
            "kept_component_indices": kept_component_indices,
            "removed_component_indices": removed_component_indices,
            "kept_component_count": int(len(kept_component_indices)),
            "removed_component_count": int(len(removed_summaries)),
            "removed_point_count": removed_count,
            "removed_point_ratio_after_radius": float(removed_count / max(1, len(radius_points))),
            "voxel_component_ms": voxel_component_ms,
            "component_selection_ms": float((time.perf_counter() - selection_started_s) * 1000.0),
            "total_ms": float((time.perf_counter() - total_started_s) * 1000.0),
            "component_bbox_gap_to_kept_top_n_m": {
                str(component_idx): float(gap)
                for component_idx, gap in sorted(gap_to_kept_top_n.items())
            },
            "components": component_summaries[: max(1, int(max_component_report_count))],
            "removed_components": removed_summaries[: max(1, int(max_component_report_count))],
        }
    )
    return radius_points[component_keep_mask], radius_colors[component_keep_mask], stats, trace
