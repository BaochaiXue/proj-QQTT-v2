from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from qqtt.demo.pcd_postprocess import (
    COMPONENT_SELECTION_LARGEST_N_PLUS_GAP,
    apply_enhanced_phystwin_like_postprocess_with_trace,
)


@dataclass(frozen=True)
class SemanticSurfaceFilterResult:
    survivor_indices: np.ndarray
    kept_mask_in_input: np.ndarray
    filtered_points: np.ndarray
    filtered_colors: np.ndarray | None
    stats: dict[str, Any]
    trace: dict[str, np.ndarray]


def filter_semantic_surface_points(
    *,
    points_world: np.ndarray,
    colors: np.ndarray | None,
    enabled: bool,
    radius_m: float,
    nb_points: int,
    component_voxel_size_m: float,
    keep_near_main_gap_m: float,
    keep_top_n_components: int,
    component_selection_policy: str = COMPONENT_SELECTION_LARGEST_N_PLUS_GAP,
    min_component_points: int = 32,
    min_component_ratio: float = 0.0,
    max_component_report_count: int = 32,
) -> SemanticSurfaceFilterResult:
    points = np.asarray(points_world, dtype=np.float32).reshape(-1, 3)
    input_count = int(len(points))
    color_array: np.ndarray | None
    if colors is None:
        color_array = None
        colors_for_filter = np.zeros((input_count, 3), dtype=np.uint8)
    else:
        color_array = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
        if len(color_array) != input_count:
            raise ValueError("points_world and colors must have the same number of rows")
        colors_for_filter = color_array

    if not enabled:
        kept = np.ones((input_count,), dtype=bool)
        survivor_indices = np.arange(input_count, dtype=np.int64)
        trace = {
            "kept_mask": kept.copy(),
            "radius_removed_mask": np.zeros((input_count,), dtype=bool),
            "component_removed_mask": np.zeros((input_count,), dtype=bool),
            "removed_mask": np.zeros((input_count,), dtype=bool),
        }
        return SemanticSurfaceFilterResult(
            survivor_indices=survivor_indices,
            kept_mask_in_input=kept,
            filtered_points=points,
            filtered_colors=None if color_array is None else color_array,
            stats={
                "enabled": False,
                "mode": "semantic_surface_filter_disabled",
                "input_point_count": input_count,
                "output_point_count": input_count,
                "query_pcd_filter_reused_result": False,
            },
            trace=trace,
        )

    filtered_points, filtered_colors, stats, trace = apply_enhanced_phystwin_like_postprocess_with_trace(
        points=points,
        colors=colors_for_filter,
        enabled=True,
        radius_m=float(radius_m),
        nb_points=int(nb_points),
        component_voxel_size_m=float(component_voxel_size_m),
        keep_near_main_gap_m=float(keep_near_main_gap_m),
        max_component_report_count=int(max_component_report_count),
        keep_top_n_components=int(keep_top_n_components),
        component_selection_policy=str(component_selection_policy),
        min_component_points=int(min_component_points),
        min_component_ratio=float(min_component_ratio),
    )
    kept_mask = np.asarray(trace.get("kept_mask"), dtype=bool).reshape(-1)
    survivor_indices = np.flatnonzero(kept_mask).astype(np.int64)
    return SemanticSurfaceFilterResult(
        survivor_indices=survivor_indices,
        kept_mask_in_input=np.ascontiguousarray(kept_mask, dtype=bool),
        filtered_points=np.asarray(filtered_points, dtype=np.float32).reshape(-1, 3),
        filtered_colors=None if color_array is None else np.asarray(filtered_colors, dtype=np.uint8).reshape(-1, 3),
        stats=dict(stats),
        trace={key: np.asarray(value).copy() for key, value in trace.items()},
    )


__all__ = [
    "SemanticSurfaceFilterResult",
    "filter_semantic_surface_points",
]
