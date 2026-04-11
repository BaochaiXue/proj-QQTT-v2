from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .calibration_frame import SEMANTIC_OVERVIEW_DISPLAY_FRAME_KIND, SEMANTIC_WORLD_FRAME_KIND, build_visualization_frame_contract
from .io_artifacts import build_artifact_sets, write_image, write_json
from .layouts import compose_turntable_board
from .compare_scene import (
    build_orbit_state,
    build_turntable_scene_state,
    compute_object_view_stats,
    ortho_scale_for_view,
)
from .selection_contracts import build_angle_selection_summary, select_angle_candidate
from .source_compare import render_mismatch_residual, render_source_attribution_overlay
from .support_compare import compute_support_count_map, overlay_support_legend, render_support_count_map, summarize_support_counts


def _score_match_candidate(metrics: dict[str, Any]) -> float:
    return float(
        3.2 * float(metrics["object_multi_camera_support_ratio"])
        - 8.0 * float(metrics["object_mismatch_residual_m"])
        + 1.6 * float(metrics["object_projected_area_ratio"])
        + 0.8 * float(metrics["object_bbox_fill_ratio"])
        - 1.2 * float(metrics["silhouette_penalty"])
        - 1.4 * float(metrics["context_dominance_penalty"])
    )


def select_match_angle(
    step_metrics: list[dict[str, Any]],
    *,
    angle_mode: str,
    angle_deg: float | None,
) -> dict[str, Any]:
    return select_angle_candidate(
        step_metrics,
        angle_mode=angle_mode,
        angle_deg=angle_deg,
        ranking_key=lambda item: (
            -float(item["final_score"]),
            -float(item["object_multi_camera_support_ratio"]),
            float(item["object_mismatch_residual_m"]),
            -float(item["object_projected_area_ratio"]),
            float(item["silhouette_penalty"]),
            float(item["context_dominance_penalty"]),
            abs(float(item["angle_deg"])),
            int(item["step_idx"]),
        ),
    )


def _compute_match_step_metrics(
    *,
    scene: dict[str, Any],
    orbit_steps: list[dict[str, Any]],
    projection_mode: str,
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    for orbit_step in orbit_steps:
        view_config = orbit_step["view_config"]
        ortho_scale = ortho_scale_for_view(scene=scene, view_config=view_config, projection_mode=projection_mode)
        native_metrics = compute_object_view_stats(
            object_camera_clouds=scene["native_object_camera_clouds"],
            combined_camera_clouds=scene["native_render_camera_clouds"],
            view_config=view_config,
            projection_mode=projection_mode,
            ortho_scale=ortho_scale,
            width=width,
            height=height,
        )
        ffs_metrics = compute_object_view_stats(
            object_camera_clouds=scene["ffs_object_camera_clouds"],
            combined_camera_clouds=scene["ffs_render_camera_clouds"],
            view_config=view_config,
            projection_mode=projection_mode,
            ortho_scale=ortho_scale,
            width=width,
            height=height,
        )
        merged = {
            "step_idx": int(orbit_step["step_idx"]),
            "angle_deg": float(orbit_step["angle_deg"]),
            "is_supported": bool(view_config.get("is_supported", True)),
            "object_projected_area_ratio": float(np.mean([native_metrics["object_projected_area_ratio"], ffs_metrics["object_projected_area_ratio"]])),
            "object_bbox_fill_ratio": float(np.mean([native_metrics["object_bbox_fill_ratio"], ffs_metrics["object_bbox_fill_ratio"]])),
            "object_multi_camera_support_ratio": float(np.mean([native_metrics["object_multi_camera_support_ratio"], ffs_metrics["object_multi_camera_support_ratio"]])),
            "object_mismatch_residual_m": float(np.mean([native_metrics["object_mismatch_residual_m"], ffs_metrics["object_mismatch_residual_m"]])),
            "context_dominance_penalty": float(np.mean([native_metrics["context_dominance_penalty"], ffs_metrics["context_dominance_penalty"]])),
            "silhouette_penalty": float(np.mean([native_metrics["silhouette_penalty"], ffs_metrics["silhouette_penalty"]])),
        }
        merged["final_score"] = _score_match_candidate(merged)
        metrics.append(merged)
    return metrics


def run_match_board_workflow(
    *,
    aligned_root: Path,
    output_dir: Path,
    case_name: str | None = None,
    realsense_case: str | None = None,
    ffs_case: str | None = None,
    frame_idx: int = 0,
    renderer: str = "fallback",
    projection_mode: str = "perspective",
    scene_crop_mode: str = "auto_object_bbox",
    focus_mode: str = "table",
    crop_margin_xy: float = 0.12,
    crop_min_z: float = -0.15,
    crop_max_z: float = 0.35,
    object_height_min: float = 0.02,
    object_height_max: float = 0.60,
    object_component_mode: str = "graph_union",
    object_component_topk: int = 2,
    roi_x_min: float | None = None,
    roi_x_max: float | None = None,
    roi_y_min: float | None = None,
    roi_y_max: float | None = None,
    roi_z_min: float | None = None,
    roi_z_max: float | None = None,
    manual_image_roi_json: str | Path | None = None,
    voxel_size: float | None = None,
    max_points_per_camera: int | None = 50000,
    depth_min_m: float = 0.2,
    depth_max_m: float = 1.5,
    use_float_ffs_depth_when_available: bool = True,
    camera_ids: list[int] | None = None,
    orbit_mode: str = "observed_hemisphere",
    num_orbit_steps: int = 24,
    orbit_degrees: float = 360.0,
    orbit_radius_scale: float = 1.9,
    view_height_offset: float = 0.0,
    coverage_margin_deg: float = 18.0,
    point_radius_px: int = 4,
    supersample_scale: int = 3,
    angle_mode: str = "auto",
    angle_deg: float | None = None,
    write_debug: bool = False,
    display_frame: str = "semantic_world",
) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    state = build_turntable_scene_state(
        aligned_root=Path(aligned_root).resolve(),
        case_name=case_name,
        realsense_case=realsense_case,
        ffs_case=ffs_case,
        frame_idx=frame_idx,
        camera_ids=camera_ids,
        voxel_size=voxel_size,
        max_points_per_camera=max_points_per_camera,
        depth_min_m=depth_min_m,
        depth_max_m=depth_max_m,
        use_float_ffs_depth_when_available=use_float_ffs_depth_when_available,
        focus_mode=focus_mode,
        scene_crop_mode=scene_crop_mode,
        crop_margin_xy=crop_margin_xy,
        crop_min_z=crop_min_z,
        crop_max_z=crop_max_z,
        object_height_min=object_height_min,
        object_height_max=object_height_max,
        object_component_mode=object_component_mode,
        object_component_topk=object_component_topk,
        roi_x_min=roi_x_min,
        roi_x_max=roi_x_max,
        roi_y_min=roi_y_min,
        roi_y_max=roi_y_max,
        roi_z_min=roi_z_min,
        roi_z_max=roi_z_max,
        manual_image_roi_json=manual_image_roi_json,
        display_frame=display_frame,
    )
    selection = state["selection"]
    scene = state["scene"]
    orbit_state = build_orbit_state(
        selection=selection,
        scene=scene,
        renderer=renderer,
        point_radius_px=point_radius_px,
        supersample_scale=supersample_scale,
        orbit_mode=orbit_mode,
        num_orbit_steps=num_orbit_steps,
        orbit_degrees=orbit_degrees,
        orbit_radius_scale=orbit_radius_scale,
        view_height_offset=view_height_offset,
        coverage_margin_deg=coverage_margin_deg,
        projection_mode=projection_mode,
    )
    step_metrics = _compute_match_step_metrics(
        scene=scene,
        orbit_steps=orbit_state["orbit_steps"],
        projection_mode=projection_mode,
        width=640,
        height=420,
    )
    selected_step = select_match_angle(step_metrics, angle_mode=angle_mode, angle_deg=angle_deg)
    selected_view = orbit_state["orbit_steps"][int(selected_step["step_idx"])]["view_config"]
    ortho_scale = ortho_scale_for_view(scene=scene, view_config=selected_view, projection_mode=projection_mode)

    native_source, _ = render_source_attribution_overlay(
        scene["native_object_camera_clouds"],
        view_config=selected_view,
        width=560,
        height=400,
        projection_mode=projection_mode,
        ortho_scale=ortho_scale,
    )
    ffs_source, _ = render_source_attribution_overlay(
        scene["ffs_object_camera_clouds"],
        view_config=selected_view,
        width=560,
        height=400,
        projection_mode=projection_mode,
        ortho_scale=ortho_scale,
    )
    native_support_map = compute_support_count_map(
        scene["native_object_camera_clouds"],
        view_config=selected_view,
        width=560,
        height=400,
        projection_mode=projection_mode,
        ortho_scale=ortho_scale,
    )
    ffs_support_map = compute_support_count_map(
        scene["ffs_object_camera_clouds"],
        view_config=selected_view,
        width=560,
        height=400,
        projection_mode=projection_mode,
        ortho_scale=ortho_scale,
    )
    native_support = overlay_support_legend(render_support_count_map(native_support_map["support_count"], native_support_map["valid"]))
    ffs_support = overlay_support_legend(render_support_count_map(ffs_support_map["support_count"], ffs_support_map["valid"]))
    native_mismatch, native_mismatch_metrics = render_mismatch_residual(
        scene["native_object_camera_clouds"],
        view_config=selected_view,
        width=560,
        height=400,
        projection_mode=projection_mode,
        ortho_scale=ortho_scale,
    )
    ffs_mismatch, ffs_mismatch_metrics = render_mismatch_residual(
        scene["ffs_object_camera_clouds"],
        view_config=selected_view,
        width=560,
        height=400,
        projection_mode=projection_mode,
        ortho_scale=ortho_scale,
    )

    case_label = (
        selection["native_case_dir"].name
        if selection["same_case_mode"]
        else f"{selection['native_case_dir'].name} vs {selection['ffs_case_dir'].name}"
    )
    board = compose_turntable_board(
        title_lines=[
            f"{case_label} | frame_idx={selection['native_frame_idx']} | 3-view point-cloud match",
            f"angle={selected_step['angle_deg']:+.1f} deg | proj={projection_mode} | crop={scene_crop_mode}",
        ],
        column_headers=["Source attribution", "Support count", "Mismatch residual"],
        row_headers=["Native", "FFS"],
        native_images=[native_source, native_support, native_mismatch],
        ffs_images=[ffs_source, ffs_support, ffs_mismatch],
        overview_inset=None,
    )
    output_path = output_dir / "01_pointcloud_match_board.png"
    write_image(output_path, board)
    summary_path = output_dir / "match_board_summary.json"
    product_artifacts, debug_artifacts = build_artifact_sets(
        output_dir=output_dir,
        product_paths={"pointcloud_match_board": output_path},
        summary_paths={"match_board_summary": summary_path},
        debug_enabled=bool(write_debug),
        debug_dir=output_dir / "debug" if write_debug else None,
        debug_paths={"match_angle_candidates": output_dir / "debug" / "match_angle_candidates.json"} if write_debug else None,
    )

    summary = {
        "same_case_mode": bool(selection["same_case_mode"]),
        "native_case_dir": str(selection["native_case_dir"]),
        "ffs_case_dir": str(selection["ffs_case_dir"]),
        "frame_idx": int(selection["native_frame_idx"]),
        "display_frame": display_frame,
        "projection_mode": projection_mode,
        "scene_crop_mode": scene_crop_mode,
        "visualization_frame_contract": build_visualization_frame_contract(
            display_frame=display_frame,
            uses_semantic_world=display_frame == "semantic_world",
            semantic_world_frame_kind=SEMANTIC_WORLD_FRAME_KIND if display_frame == "semantic_world" else None,
            overview_display_frame_kind=SEMANTIC_OVERVIEW_DISPLAY_FRAME_KIND if display_frame == "semantic_world" else "calibration_world_topdown_display",
            notes=[
                "This board reuses the shared single-frame compare scene and angle-selection contract.",
            ],
        ),
        "match_angle_selection": build_angle_selection_summary(
            mode=angle_mode,
            selected_step=selected_step,
            candidate_count=len(step_metrics),
        ),
        "native_support_summary": {
            **summarize_support_counts(native_support_map["support_count"], native_support_map["valid"]),
            **native_mismatch_metrics["summary"],
        },
        "ffs_support_summary": {
            **summarize_support_counts(ffs_support_map["support_count"], ffs_support_map["valid"]),
            **ffs_mismatch_metrics["summary"],
        },
        "top_level_output": str(output_path),
        "debug_written": bool(write_debug),
        "product_artifacts": product_artifacts.to_dict(),
        "debug_artifacts": debug_artifacts.to_dict(),
    }
    write_json(summary_path, summary)

    if write_debug:
        debug_dir = output_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        write_json(debug_dir / "match_angle_candidates.json", {"steps": step_metrics, "selected": selected_step})

    return {
        "output_dir": str(output_dir),
        "summary": summary,
    }
