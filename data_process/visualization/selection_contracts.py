from __future__ import annotations

from typing import Any, Callable, Iterable

from .types import AngleSelectionSummary, TruthPairSelectionSummary


AngleMetricGetter = Callable[[dict[str, Any]], tuple[Any, ...]]


def select_angle_candidate(
    step_metrics: list[dict[str, Any]],
    *,
    angle_mode: str,
    angle_deg: float | None,
    ranking_key: AngleMetricGetter,
) -> dict[str, Any]:
    if not step_metrics:
        raise ValueError("No step metrics available for angle selection.")
    if angle_mode == "explicit":
        if angle_deg is None:
            raise ValueError("angle_mode=explicit requires angle_deg.")
        return min(
            step_metrics,
            key=lambda item: (
                abs(float(item["angle_deg"]) - float(angle_deg)),
                int(item["step_idx"]),
            ),
        )
    supported = [item for item in step_metrics if bool(item["is_supported"])]
    candidates = supported if supported else step_metrics
    ranked = sorted(candidates, key=ranking_key)
    return ranked[0]


def build_angle_selection_summary(
    *,
    mode: str,
    selected_step: dict[str, Any],
    candidate_count: int,
) -> dict[str, Any]:
    return AngleSelectionSummary(
        mode=mode,
        selected_step_idx=int(selected_step["step_idx"]),
        selected_angle_deg=float(selected_step["angle_deg"]),
        selected_is_supported=bool(selected_step["is_supported"]),
        object_projected_area_ratio=float(selected_step["object_projected_area_ratio"]),
        object_bbox_fill_ratio=float(selected_step["object_bbox_fill_ratio"]),
        object_multi_camera_support_ratio=float(selected_step["object_multi_camera_support_ratio"]),
        object_mismatch_residual_m=float(selected_step["object_mismatch_residual_m"]),
        context_dominance_penalty=float(selected_step["context_dominance_penalty"]),
        silhouette_penalty=float(selected_step["silhouette_penalty"]),
        final_score=float(selected_step["final_score"]),
        candidate_count=int(candidate_count),
    ).to_dict()


def select_truth_pair_candidate(
    pair_metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    if not pair_metrics:
        raise ValueError("No reprojection pair metrics available.")
    ranked = sorted(
        pair_metrics,
        key=lambda item: (
            -float(item["pair_object_visibility_score"]),
            -float(item["object_overlap_area"]),
            float((float(item["object_edge_weighted_residual_mean_native"]) + float(item["object_edge_weighted_residual_mean_ffs"])) * 0.5),
            float((float(item["object_residual_mean_native"]) + float(item["object_residual_mean_ffs"])) * 0.5),
            int(item["pair"][0]),
            int(item["pair"][1]),
        ),
    )
    return ranked[0]


def build_truth_pair_selection_summary(
    selected_pair: dict[str, Any],
) -> dict[str, Any]:
    return TruthPairSelectionSummary(
        src_camera_idx=int(selected_pair["pair"][0]),
        dst_camera_idx=int(selected_pair["pair"][1]),
        mean_valid_ratio=float(selected_pair["mean_valid_ratio"]),
        residual_gap=float(selected_pair["residual_gap"]),
        object_warp_valid_ratio_native=float(selected_pair["object_warp_valid_ratio_native"]),
        object_warp_valid_ratio_ffs=float(selected_pair["object_warp_valid_ratio_ffs"]),
        object_residual_mean_native=float(selected_pair["object_residual_mean_native"]),
        object_residual_mean_ffs=float(selected_pair["object_residual_mean_ffs"]),
        object_edge_weighted_residual_mean_native=float(selected_pair["object_edge_weighted_residual_mean_native"]),
        object_edge_weighted_residual_mean_ffs=float(selected_pair["object_edge_weighted_residual_mean_ffs"]),
        object_overlap_area=float(selected_pair["object_overlap_area"]),
        pair_object_visibility_score=float(selected_pair["pair_object_visibility_score"]),
        native=dict(selected_pair["native"]),
        ffs=dict(selected_pair["ffs"]),
    ).to_dict()


def build_ranked_candidate_debug(
    *,
    candidates: Iterable[dict[str, Any]],
    selected: dict[str, Any],
) -> dict[str, Any]:
    return {
        "candidates": list(candidates),
        "selected": dict(selected),
    }
