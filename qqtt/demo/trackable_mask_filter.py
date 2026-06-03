from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Mapping

import numpy as np

from data_process.depth_backends.geometry import transform_points
from qqtt.demo.pcd_postprocess import (
    _detect_radius_outlier_indices,
    COMPONENT_SELECTION_LARGEST_N,
    COMPONENT_SELECTION_LARGEST_N_PLUS_GAP,
)
from qqtt.demo.semantic_surface_filter import filter_semantic_surface_points
from qqtt.demo.phystwin_volume_filter import (
    PHYSTWIN_VOLUME_ORIGIN_FRAME_MIN,
    PHYSTWIN_VOLUME_ORIGIN_WORLD,
    phystwin_volume_sample_indices_fast,
)
from qqtt.demo.three_view_masked_fused_pcd_runtime import (
    POSTPROCESS_ENHANCED_PT,
    POSTPROCESS_NONE,
    POSTPROCESS_PT_FILTER,
)


TRACKABLE_MASK_BUILD_POLICY_DISABLED = "disabled"
TRACKABLE_MASK_BUILD_POLICY_INIT_ONLY = "init-only"
TRACKABLE_MASK_BUILD_POLICIES = (
    TRACKABLE_MASK_BUILD_POLICY_INIT_ONLY,
    TRACKABLE_MASK_BUILD_POLICY_DISABLED,
)

TRACKABLE_QUERY_INIT_STRATEGY_STANDARD_FILTER_INIT = "standard-filter-init"
TRACKABLE_QUERY_INIT_STRATEGIES = (
    TRACKABLE_QUERY_INIT_STRATEGY_STANDARD_FILTER_INIT,
)


@dataclass(frozen=True)
class TrackableMaskFilterConfig:
    depth_min_m: float
    depth_max_m: float
    object_point_control: str = "fixed-cap"
    object_volume_voxel_m: float = 0.005
    object_volume_origin: str = PHYSTWIN_VOLUME_ORIGIN_WORLD
    object_volume_points_per_voxel: int = 1
    object_postprocess: str = POSTPROCESS_ENHANCED_PT
    controller_postprocess: str = POSTPROCESS_ENHANCED_PT
    phystwin_radius_m: float = 0.01
    phystwin_nb_points: int = 12
    enhanced_component_voxel_size_m: float = 0.006
    enhanced_keep_near_main_gap_m: float = 0.035
    object_enhanced_keep_top_n_components: int = 1
    controller_enhanced_keep_top_n_components: int = 2
    enhanced_component_selection_policy: str = COMPONENT_SELECTION_LARGEST_N_PLUS_GAP
    enhanced_min_component_points: int = 32
    enhanced_min_component_ratio: float = 0.0
    controller_trackable_max_points_per_camera: int = 4999
    seed: int = 42


@dataclass(frozen=True)
class TrackableMaskResult:
    object_mask: np.ndarray
    controller_mask: np.ndarray
    union_mask: np.ndarray
    stats: dict[str, Any] = field(default_factory=dict)


def _empty_bool_mask(shape: tuple[int, int]) -> np.ndarray:
    return np.zeros(shape, dtype=bool)


def _depth_valid_mask(
    depth_m: np.ndarray,
    semantic_mask: np.ndarray,
    *,
    depth_min_m: float,
    depth_max_m: float,
) -> np.ndarray:
    depth = np.asarray(depth_m, dtype=np.float32)
    mask = np.asarray(semantic_mask, dtype=bool)
    if depth.shape[:2] != mask.shape[:2]:
        raise ValueError("depth and mask shapes must match")
    valid = np.isfinite(depth) & (depth > np.float32(depth_min_m)) & mask
    if float(depth_max_m) > 0.0:
        valid &= depth < np.float32(depth_max_m)
    return np.ascontiguousarray(valid, dtype=bool)


def _source_points_world_from_mask(
    *,
    depth_m: np.ndarray,
    mask: np.ndarray,
    intrinsics: np.ndarray,
    c2w: np.ndarray,
    depth_min_m: float,
    depth_max_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    valid = _depth_valid_mask(
        depth_m,
        mask,
        depth_min_m=float(depth_min_m),
        depth_max_m=float(depth_max_m),
    )
    rows, cols = np.nonzero(valid)
    if rows.size == 0:
        return np.empty((0, 2), dtype=np.int64), np.empty((0, 3), dtype=np.float32)

    depth = np.asarray(depth_m, dtype=np.float32)
    K = np.asarray(intrinsics, dtype=np.float32).reshape(3, 3)
    fx = max(float(K[0, 0]), 1e-6)
    fy = max(float(K[1, 1]), 1e-6)
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    z = depth[rows, cols].astype(np.float32, copy=False)
    x = ((cols.astype(np.float32) - cx) / fx) * z
    y = ((rows.astype(np.float32) - cy) / fy) * z
    points_camera = np.ascontiguousarray(np.stack([x, y, z], axis=1), dtype=np.float32)
    points_world = transform_points(points_camera, np.asarray(c2w, dtype=np.float32).reshape(4, 4)).astype(np.float32)
    yx = np.ascontiguousarray(np.stack([rows, cols], axis=1), dtype=np.int64)
    return yx, np.ascontiguousarray(points_world, dtype=np.float32)


def _indices_from_postprocess(
    points_world: np.ndarray,
    *,
    postprocess_mode: str,
    radius_m: float,
    nb_points: int,
    component_voxel_size_m: float,
    keep_near_main_gap_m: float,
    keep_top_n_components: int,
    component_selection_policy: str,
    min_component_points: int,
    min_component_ratio: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    points = np.asarray(points_world, dtype=np.float32).reshape(-1, 3)
    count = int(len(points))
    if count == 0:
        return np.empty((0,), dtype=np.int64), {
            "mode": str(postprocess_mode),
            "input_point_count": 0,
            "output_point_count": 0,
        }
    mode = str(postprocess_mode)
    if mode == POSTPROCESS_NONE:
        return np.arange(count, dtype=np.int64), {
            "mode": POSTPROCESS_NONE,
            "input_point_count": count,
            "output_point_count": count,
        }
    if mode == POSTPROCESS_PT_FILTER:
        result = _detect_radius_outlier_indices(
            points,
            radius_m=float(radius_m),
            nb_points=int(nb_points),
        )
        keep_idx = np.sort(np.asarray(result["inlier_indices"], dtype=np.int64).reshape(-1))
        return keep_idx, {
            "mode": POSTPROCESS_PT_FILTER,
            "input_point_count": count,
            "output_point_count": int(len(keep_idx)),
            "radius_m": float(radius_m),
            "nb_points": int(nb_points),
        }
    if mode == POSTPROCESS_ENHANCED_PT:
        result = filter_semantic_surface_points(
            points_world=points,
            colors=None,
            enabled=True,
            radius_m=float(radius_m),
            nb_points=int(nb_points),
            component_voxel_size_m=float(component_voxel_size_m),
            keep_near_main_gap_m=float(keep_near_main_gap_m),
            keep_top_n_components=int(keep_top_n_components),
            component_selection_policy=str(component_selection_policy),
            min_component_points=int(min_component_points),
            min_component_ratio=float(min_component_ratio),
        )
        keep_idx = np.asarray(result.survivor_indices, dtype=np.int64).reshape(-1)
        stats = dict(result.stats)
        stats["mode"] = POSTPROCESS_ENHANCED_PT
        stats["input_point_count"] = count
        stats["output_point_count"] = int(len(keep_idx))
        stats["survivor_index_space"] = "source_points_world"
        return keep_idx, stats
    raise ValueError(f"unsupported postprocess mode: {postprocess_mode}")


def _object_survivor_indices(points_world: np.ndarray, config: TrackableMaskFilterConfig) -> tuple[np.ndarray, dict[str, Any]]:
    points = np.asarray(points_world, dtype=np.float32).reshape(-1, 3)
    count = int(len(points))
    if count == 0:
        return np.empty((0,), dtype=np.int64), {
            "mode": str(config.object_point_control),
            "input_point_count": 0,
            "output_point_count": 0,
        }
    component_keep_idx, component_stats = _indices_from_postprocess(
        points,
        postprocess_mode=str(config.object_postprocess),
        radius_m=float(config.phystwin_radius_m),
        nb_points=int(config.phystwin_nb_points),
        component_voxel_size_m=float(config.enhanced_component_voxel_size_m),
        keep_near_main_gap_m=float(config.enhanced_keep_near_main_gap_m),
        keep_top_n_components=int(config.object_enhanced_keep_top_n_components),
        component_selection_policy=COMPONENT_SELECTION_LARGEST_N,
        min_component_points=int(config.enhanced_min_component_points),
        min_component_ratio=float(config.enhanced_min_component_ratio),
    )
    component_points = points[np.asarray(component_keep_idx, dtype=np.int64)]
    if str(config.object_point_control) == "phystwin-volume":
        origin = np.zeros(3, dtype=np.float32)
        if str(config.object_volume_origin) == PHYSTWIN_VOLUME_ORIGIN_FRAME_MIN:
            origin = component_points.min(axis=0).astype(np.float32) if len(component_points) else np.zeros(3, dtype=np.float32)
        elif str(config.object_volume_origin) != PHYSTWIN_VOLUME_ORIGIN_WORLD:
            origin = np.zeros(3, dtype=np.float32)
        started_s = time.perf_counter()
        volume_keep_idx = phystwin_volume_sample_indices_fast(
            component_points,
            voxel_size_m=float(config.object_volume_voxel_m),
            origin_world=origin,
            points_per_voxel=int(config.object_volume_points_per_voxel),
        )
        elapsed_ms = float((time.perf_counter() - started_s) * 1000.0)
        keep_idx = np.asarray(component_keep_idx, dtype=np.int64)[np.asarray(volume_keep_idx, dtype=np.int64)]
        return keep_idx, {
            "mode": "phystwin-volume",
            "input_point_count": count,
            "output_point_count": int(len(keep_idx)),
            "component_filtered_point_count": int(len(component_keep_idx)),
            "occupied_voxel_count": int(len(volume_keep_idx)),
            "voxel_size_m": float(config.object_volume_voxel_m),
            "origin_policy": str(config.object_volume_origin),
            "filter_ms": elapsed_ms,
            "component_filter": component_stats,
        }
    return component_keep_idx, component_stats


def _controller_survivor_indices(points_world: np.ndarray, config: TrackableMaskFilterConfig) -> tuple[np.ndarray, dict[str, Any]]:
    return _indices_from_postprocess(
        points_world,
        postprocess_mode=str(config.controller_postprocess),
        radius_m=float(config.phystwin_radius_m),
        nb_points=int(config.phystwin_nb_points),
        component_voxel_size_m=float(config.enhanced_component_voxel_size_m),
        keep_near_main_gap_m=float(config.enhanced_keep_near_main_gap_m),
        keep_top_n_components=int(config.controller_enhanced_keep_top_n_components),
        component_selection_policy=str(config.enhanced_component_selection_policy),
        min_component_points=int(config.enhanced_min_component_points),
        min_component_ratio=float(config.enhanced_min_component_ratio),
    )


def _indices_to_mask(shape: tuple[int, int], yx: np.ndarray, keep_idx: np.ndarray) -> np.ndarray:
    out = _empty_bool_mask(shape)
    if len(keep_idx) == 0 or len(yx) == 0:
        return out
    selected_yx = np.asarray(yx, dtype=np.int64).reshape(-1, 2)[np.asarray(keep_idx, dtype=np.int64)]
    out[selected_yx[:, 0], selected_yx[:, 1]] = True
    return np.ascontiguousarray(out, dtype=bool)


def _cap_mask_points(
    mask: np.ndarray,
    *,
    max_points: int,
    seed: int,
    camera_idx: int,
) -> tuple[np.ndarray, bool, int, int]:
    mask_bool = np.asarray(mask, dtype=bool)
    before = int(np.count_nonzero(mask_bool))
    cap = int(max_points)
    if cap <= 0 or before <= cap:
        return np.ascontiguousarray(mask_bool, dtype=bool), False, before, before
    rows, cols = np.nonzero(mask_bool)
    rng = np.random.default_rng(int(seed) + int(camera_idx))
    chosen = rng.choice(rows.shape[0], size=cap, replace=False)
    out = np.zeros(mask_bool.shape, dtype=bool)
    out[rows[chosen], cols[chosen]] = True
    return np.ascontiguousarray(out, dtype=bool), True, before, int(cap)


def build_standard_filter_trackable_masks_for_camera(
    *,
    camera_idx: int,
    depth_m: np.ndarray,
    object_mask: np.ndarray,
    controller_mask: np.ndarray,
    intrinsics: np.ndarray,
    c2w: np.ndarray,
    config: TrackableMaskFilterConfig,
) -> TrackableMaskResult:
    started_s = time.perf_counter()
    shape = tuple(np.asarray(depth_m).shape[:2])
    object_yx, object_points = _source_points_world_from_mask(
        depth_m=depth_m,
        mask=object_mask,
        intrinsics=intrinsics,
        c2w=c2w,
        depth_min_m=float(config.depth_min_m),
        depth_max_m=float(config.depth_max_m),
    )
    controller_yx, controller_points = _source_points_world_from_mask(
        depth_m=depth_m,
        mask=controller_mask,
        intrinsics=intrinsics,
        c2w=c2w,
        depth_min_m=float(config.depth_min_m),
        depth_max_m=float(config.depth_max_m),
    )

    object_keep_idx, object_stats = _object_survivor_indices(object_points, config)
    controller_keep_idx, controller_stats = _controller_survivor_indices(controller_points, config)
    object_trackable = _indices_to_mask(shape, object_yx, object_keep_idx)
    controller_before_cap = _indices_to_mask(shape, controller_yx, controller_keep_idx)
    controller_trackable, cap_applied, before_cap, after_cap = _cap_mask_points(
        controller_before_cap,
        max_points=int(config.controller_trackable_max_points_per_camera),
        seed=int(config.seed),
        camera_idx=int(camera_idx),
    )
    union_trackable = np.ascontiguousarray(object_trackable | controller_trackable, dtype=bool)
    elapsed_ms = float((time.perf_counter() - started_s) * 1000.0)
    return TrackableMaskResult(
        object_mask=object_trackable,
        controller_mask=controller_trackable,
        union_mask=union_trackable,
        stats={
            "camera_idx": int(camera_idx),
            "trackable_mask_source": "standard_filter_survivors",
            "trackable_mask_ms": elapsed_ms,
            "raw_object_pixels": int(np.count_nonzero(np.asarray(object_mask, dtype=bool))),
            "raw_controller_pixels": int(np.count_nonzero(np.asarray(controller_mask, dtype=bool))),
            "depth_valid_object_pixels": int(len(object_yx)),
            "depth_valid_controller_pixels": int(len(controller_yx)),
            "object_trackable_pixels": int(np.count_nonzero(object_trackable)),
            "controller_trackable_before_cap": int(before_cap),
            "controller_trackable_after_cap": int(after_cap),
            "controller_trackable_cap_applied": bool(cap_applied),
            "union_trackable_pixels": int(np.count_nonzero(union_trackable)),
            "object_filter": object_stats,
            "controller_filter": controller_stats,
        },
    )


def summarize_trackable_stats(stats_by_camera: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "trackable_mask_source": "standard_filter_survivors",
        "raw_object_pixels_by_camera": {},
        "raw_controller_pixels_by_camera": {},
        "depth_valid_object_pixels_by_camera": {},
        "depth_valid_controller_pixels_by_camera": {},
        "trackable_object_pixels_by_camera": {},
        "controller_trackable_before_cap_by_camera": {},
        "controller_trackable_after_cap_by_camera": {},
        "controller_trackable_cap_applied_by_camera": {},
        "trackable_union_pixels_by_camera": {},
        "trackable_mask_standard_filter_ms_by_camera": {},
        "object_filter_by_camera": {},
        "controller_filter_by_camera": {},
        "object_filter_mode_by_camera": {},
        "controller_filter_mode_by_camera": {},
        "object_component_count_by_camera": {},
        "controller_component_count_by_camera": {},
        "object_component_point_counts_by_camera": {},
        "controller_component_point_counts_by_camera": {},
        "object_top_n_component_point_counts_by_camera": {},
        "controller_top_n_component_point_counts_by_camera": {},
        "object_kept_component_indices_by_camera": {},
        "controller_kept_component_indices_by_camera": {},
        "object_removed_component_count_by_camera": {},
        "controller_removed_component_count_by_camera": {},
        "object_removed_point_count_by_camera": {},
        "controller_removed_point_count_by_camera": {},
        "object_component_filter_removed_ratio_by_camera": {},
        "controller_component_filter_removed_ratio_by_camera": {},
        "query_pcd_filter_reused_result_by_camera": {},
    }

    def _component_stats(filter_stats: Mapping[str, Any]) -> dict[str, Any]:
        nested = filter_stats.get("component_filter")
        return dict(nested) if isinstance(nested, Mapping) else dict(filter_stats)

    for camera_idx, raw_stats in stats_by_camera.items():
        idx = int(camera_idx)
        stats = dict(raw_stats)
        summary["raw_object_pixels_by_camera"][idx] = int(stats.get("raw_object_pixels", 0))
        summary["raw_controller_pixels_by_camera"][idx] = int(stats.get("raw_controller_pixels", 0))
        summary["depth_valid_object_pixels_by_camera"][idx] = int(stats.get("depth_valid_object_pixels", 0))
        summary["depth_valid_controller_pixels_by_camera"][idx] = int(stats.get("depth_valid_controller_pixels", 0))
        summary["trackable_object_pixels_by_camera"][idx] = int(stats.get("object_trackable_pixels", 0))
        summary["controller_trackable_before_cap_by_camera"][idx] = int(stats.get("controller_trackable_before_cap", 0))
        summary["controller_trackable_after_cap_by_camera"][idx] = int(stats.get("controller_trackable_after_cap", 0))
        summary["controller_trackable_cap_applied_by_camera"][idx] = bool(stats.get("controller_trackable_cap_applied", False))
        summary["trackable_union_pixels_by_camera"][idx] = int(stats.get("union_trackable_pixels", 0))
        summary["trackable_mask_standard_filter_ms_by_camera"][idx] = float(stats.get("trackable_mask_ms", 0.0))
        object_filter = dict(stats.get("object_filter", {}) or {})
        controller_filter = dict(stats.get("controller_filter", {}) or {})
        summary["object_filter_by_camera"][idx] = object_filter
        summary["controller_filter_by_camera"][idx] = controller_filter
        summary["object_filter_mode_by_camera"][idx] = str(object_filter.get("mode", ""))
        summary["controller_filter_mode_by_camera"][idx] = str(controller_filter.get("mode", ""))
        object_components = _component_stats(object_filter)
        controller_components = _component_stats(controller_filter)
        summary["object_component_count_by_camera"][idx] = int(object_components.get("component_count", 0) or 0)
        summary["controller_component_count_by_camera"][idx] = int(controller_components.get("component_count", 0) or 0)
        summary["object_component_point_counts_by_camera"][idx] = list(
            object_components.get("component_point_counts", []) or []
        )
        summary["controller_component_point_counts_by_camera"][idx] = list(
            controller_components.get("component_point_counts", []) or []
        )
        summary["object_top_n_component_point_counts_by_camera"][idx] = list(
            object_components.get("top_n_component_point_counts", []) or []
        )
        summary["controller_top_n_component_point_counts_by_camera"][idx] = list(
            controller_components.get("top_n_component_point_counts", []) or []
        )
        summary["object_kept_component_indices_by_camera"][idx] = list(
            object_components.get("kept_component_indices", []) or []
        )
        summary["controller_kept_component_indices_by_camera"][idx] = list(
            controller_components.get("kept_component_indices", []) or []
        )
        summary["object_removed_component_count_by_camera"][idx] = int(
            object_components.get("removed_component_count", 0) or 0
        )
        summary["controller_removed_component_count_by_camera"][idx] = int(
            controller_components.get("removed_component_count", 0) or 0
        )
        summary["object_removed_point_count_by_camera"][idx] = int(
            object_components.get("removed_point_count", 0) or 0
        )
        summary["controller_removed_point_count_by_camera"][idx] = int(
            controller_components.get("removed_point_count", 0) or 0
        )
        summary["object_component_filter_removed_ratio_by_camera"][idx] = float(
            object_components.get("removed_point_ratio_after_radius", 0.0) or 0.0
        )
        summary["controller_component_filter_removed_ratio_by_camera"][idx] = float(
            controller_components.get("removed_point_ratio_after_radius", 0.0) or 0.0
        )
        summary["query_pcd_filter_reused_result_by_camera"][idx] = bool(
            object_components.get("query_pcd_filter_reused_result", False)
            or controller_components.get("query_pcd_filter_reused_result", False)
        )
    return summary


__all__ = [
    "TRACKABLE_MASK_BUILD_POLICIES",
    "TRACKABLE_MASK_BUILD_POLICY_DISABLED",
    "TRACKABLE_MASK_BUILD_POLICY_INIT_ONLY",
    "TRACKABLE_QUERY_INIT_STRATEGIES",
    "TRACKABLE_QUERY_INIT_STRATEGY_STANDARD_FILTER_INIT",
    "TrackableMaskFilterConfig",
    "TrackableMaskResult",
    "build_standard_filter_trackable_masks_for_camera",
    "summarize_trackable_stats",
]
