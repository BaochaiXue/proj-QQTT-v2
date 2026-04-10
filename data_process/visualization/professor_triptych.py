from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .camera_frusta import build_camera_frustum_geometry, extract_camera_poses
from .depth_diagnostics import (
    compute_photometric_residual,
    get_case_camera_transform,
    label_tile,
    load_color_frame,
    load_depth_frame,
    warp_rgb_between_cameras,
)
from .hero_compare import compose_hero_compare
from .io_artifacts import write_image, write_json
from .layouts import compose_depth_review_board, compose_turntable_board
from .pointcloud_compare import (
    PROJECTION_MODES,
    RENDER_MODES,
    SCENE_CROP_MODES,
    estimate_ortho_scale,
    get_case_intrinsics,
    render_point_cloud,
)
from .reprojection_compare import build_camera_pairs, run_reprojection_compare_workflow
from .source_compare import render_mismatch_residual, render_source_attribution_overlay
from .support_compare import compute_support_count_map, overlay_support_legend, render_support_count_map, summarize_support_counts
from .turntable_compare import (
    _resolve_single_frame_case_selection_model,
    build_scene_overview_state,
    build_single_frame_scene,
    load_single_frame_compare_clouds,
    prepare_object_roi_refinement,
    render_overview_inset,
    run_turntable_compare_workflow,
)
from .views import build_object_centered_orbit_views, estimate_orbit_axis


def _parse_manual_xyz_roi(
    *,
    scene_crop_mode: str,
    roi_x_min: float | None,
    roi_x_max: float | None,
    roi_y_min: float | None,
    roi_y_max: float | None,
    roi_z_min: float | None,
    roi_z_max: float | None,
) -> dict[str, float] | None:
    if scene_crop_mode != "manual_xyz_roi":
        return None
    roi_values = [roi_x_min, roi_x_max, roi_y_min, roi_y_max, roi_z_min, roi_z_max]
    if any(value is None for value in roi_values):
        raise ValueError("manual_xyz_roi requires all roi_x/y/z min/max arguments.")
    return {
        "x_min": float(roi_x_min),
        "x_max": float(roi_x_max),
        "y_min": float(roi_y_min),
        "y_max": float(roi_y_max),
        "z_min": float(roi_z_min),
        "z_max": float(roi_z_max),
    }


def _parse_manual_image_roi_json(manual_image_roi_json: str | Path | None) -> dict[int, tuple[int, int, int, int]] | None:
    if manual_image_roi_json is None:
        return None
    roi_path = Path(manual_image_roi_json).resolve()
    data = json.loads(roi_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data:
        raise ValueError(f"manual_image_roi_json must contain a non-empty camera->bbox mapping: {roi_path}")
    parsed: dict[int, tuple[int, int, int, int]] = {}
    for camera_key, bbox in data.items():
        camera_idx = int(camera_key)
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError(f"manual_image_roi_json entry for camera {camera_key} must be [x_min, y_min, x_max, y_max].")
        x_min, y_min, x_max, y_max = [int(item) for item in bbox]
        if x_min >= x_max or y_min >= y_max:
            raise ValueError(f"manual_image_roi_json entry for camera {camera_key} is invalid: {bbox}")
        parsed[camera_idx] = (x_min, y_min, x_max, y_max)
    return parsed


def _pair_key(pair: tuple[int, int]) -> str:
    return f"{int(pair[0])}_to_{int(pair[1])}"


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.astype(np.float32).tolist()
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def select_professor_angle(
    step_metrics: list[dict[str, Any]],
    *,
    angle_mode: str,
    angle_deg: float | None,
) -> dict[str, Any]:
    if not step_metrics:
        raise ValueError("No step metrics available for hero-angle selection.")
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

    ranked = sorted(
        step_metrics,
        key=lambda item: (
            0 if bool(item["is_supported"]) else 1,
            -float(item["mean_multi_camera_support"]),
            float(item["mean_mismatch_residual_m"]),
            float(item["mean_mismatch_p90_m"]),
            abs(float(item["angle_deg"])),
            int(item["step_idx"]),
        ),
    )
    return ranked[0]


def select_truth_camera_pair(pair_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    if not pair_metrics:
        raise ValueError("No reprojection pair metrics available.")
    ranked = sorted(
        pair_metrics,
        key=lambda item: (
            -float(item["mean_valid_ratio"]),
            -float(item["residual_gap"]),
            float(item["native"]["residual_mean"]),
            int(item["pair"][0]),
            int(item["pair"][1]),
        ),
    )
    return ranked[0]


def _build_turntable_scene(
    *,
    aligned_root: Path,
    case_name: str | None,
    realsense_case: str | None,
    ffs_case: str | None,
    frame_idx: int,
    camera_ids: list[int] | None,
    voxel_size: float | None,
    max_points_per_camera: int | None,
    depth_min_m: float,
    depth_max_m: float,
    use_float_ffs_depth_when_available: bool,
    focus_mode: str,
    scene_crop_mode: str,
    crop_margin_xy: float,
    crop_min_z: float,
    crop_max_z: float,
    object_height_min: float,
    object_height_max: float,
    object_component_mode: str,
    object_component_topk: int,
    roi_x_min: float | None,
    roi_x_max: float | None,
    roi_y_min: float | None,
    roi_y_max: float | None,
    roi_z_min: float | None,
    roi_z_max: float | None,
    manual_image_roi_json: str | Path | None,
) -> dict[str, Any]:
    selection_model = _resolve_single_frame_case_selection_model(
        aligned_root=Path(aligned_root).resolve(),
        case_name=case_name,
        realsense_case=realsense_case,
        ffs_case=ffs_case,
        frame_idx=frame_idx,
        camera_ids=camera_ids,
    )
    selection = selection_model.to_dict()
    manual_image_roi_by_camera = _parse_manual_image_roi_json(manual_image_roi_json)
    raw_scene = load_single_frame_compare_clouds(
        selection,
        voxel_size=voxel_size,
        max_points_per_camera=None,
        depth_min_m=depth_min_m,
        depth_max_m=depth_max_m,
        use_float_ffs_depth_when_available=use_float_ffs_depth_when_available,
        pixel_roi_by_camera=None,
    )
    manual_xyz_roi = _parse_manual_xyz_roi(
        scene_crop_mode=scene_crop_mode,
        roi_x_min=roi_x_min,
        roi_x_max=roi_x_max,
        roi_y_min=roi_y_min,
        roi_y_max=roi_y_max,
        roi_z_min=roi_z_min,
        roi_z_max=roi_z_max,
    )
    refinement = prepare_object_roi_refinement(
        raw_scene=raw_scene,
        focus_mode=focus_mode,
        scene_crop_mode=scene_crop_mode,
        crop_margin_xy=crop_margin_xy,
        crop_min_z=crop_min_z,
        crop_max_z=crop_max_z,
        manual_xyz_roi=manual_xyz_roi,
        manual_image_roi_by_camera=manual_image_roi_by_camera,
        object_height_min=object_height_min,
        object_height_max=object_height_max,
        object_component_mode=object_component_mode,
        object_component_topk=object_component_topk,
    )
    scene = build_single_frame_scene(
        native_points=raw_scene["native_points"],
        native_colors=raw_scene["native_colors"],
        native_camera_clouds=raw_scene["native_camera_clouds"],
        ffs_points=raw_scene["ffs_points"],
        ffs_colors=raw_scene["ffs_colors"],
        ffs_camera_clouds=raw_scene["ffs_camera_clouds"],
        focus_mode=focus_mode,
        scene_crop_mode=scene_crop_mode,
        crop_margin_xy=crop_margin_xy,
        crop_min_z=crop_min_z,
        crop_max_z=crop_max_z,
        manual_xyz_roi=manual_xyz_roi,
        object_seed_point_sets=refinement["pass2_seed_point_sets"],
        native_pixel_mask_by_camera=refinement["final_native_masks"],
        ffs_pixel_mask_by_camera=refinement["final_ffs_masks"],
        object_height_min=object_height_min,
        object_height_max=object_height_max,
        object_component_mode=object_component_mode,
        object_component_topk=object_component_topk,
        context_max_points_per_camera=max_points_per_camera,
        crop_bounds_override=refinement["final_crop"],
    )
    return {
        "selection": selection,
        "manual_image_roi_by_camera": manual_image_roi_by_camera,
        "scene": scene,
        "refinement": refinement,
    }


def _build_orbit_state(
    *,
    selection: dict[str, Any],
    scene: dict[str, Any],
    renderer: str,
    point_radius_px: int,
    supersample_scale: int,
    orbit_mode: str,
    num_orbit_steps: int,
    orbit_degrees: float,
    orbit_radius_scale: float,
    view_height_offset: float,
    coverage_margin_deg: float,
    projection_mode: str,
) -> dict[str, Any]:
    camera_poses = extract_camera_poses(
        selection["native_c2w"],
        serial_numbers=selection["serial_numbers"],
        camera_ids=selection["camera_ids"],
    )
    orbit_axis = estimate_orbit_axis(camera_poses)
    orbit_plan = build_object_centered_orbit_views(
        camera_poses=camera_poses,
        focus_point=scene["focus_point"],
        bounds_min=scene["object_roi_bounds"]["min"],
        bounds_max=scene["object_roi_bounds"]["max"],
        orbit_axis=orbit_axis,
        num_orbit_steps=num_orbit_steps,
        orbit_degrees=orbit_degrees,
        orbit_radius_scale=orbit_radius_scale,
        view_height_offset=view_height_offset,
        orbit_mode=orbit_mode,
        coverage_margin_deg=coverage_margin_deg,
        show_unsupported_warning=True,
    )
    frustum_scale = max(0.06, float(np.linalg.norm(scene["bounds_max"] - scene["bounds_min"])) * 0.14)
    camera_geometries = [build_camera_frustum_geometry(pose, frustum_scale=frustum_scale) for pose in camera_poses]
    if len(scene["native_render_points"]) > 0 and len(scene["ffs_render_points"]) > 0:
        overview_points = np.concatenate([scene["native_render_points"], scene["ffs_render_points"]], axis=0)
        overview_colors = np.concatenate([scene["native_render_colors"], scene["ffs_render_colors"]], axis=0)
    elif len(scene["native_render_points"]) > 0:
        overview_points = scene["native_render_points"]
        overview_colors = scene["native_render_colors"]
    else:
        overview_points = scene["ffs_render_points"]
        overview_colors = scene["ffs_render_colors"]
    overview_state = build_scene_overview_state(
        scene_points=overview_points,
        scene_colors=overview_colors,
        camera_geometries=camera_geometries,
        focus_point=scene["focus_point"],
        render_mode="color_by_height",
        renderer=renderer,
        scalar_bounds=scene["scalar_bounds"],
        point_radius_px=point_radius_px,
        supersample_scale=supersample_scale,
        orbit_path_points=orbit_plan["orbit_path"],
        orbit_path_supported=orbit_plan["orbit_supported_mask"],
        crop_bounds=scene["object_roi_bounds"],
        supported_arc_label=f"Supported arc: {orbit_plan['coverage_arc']['span_deg']:.1f} deg",
    )
    return {
        "camera_poses": camera_poses,
        "orbit_axis": orbit_axis,
        "orbit_steps": orbit_plan["orbit_steps"],
        "overview_state": overview_state,
        "orbit_plan": orbit_plan,
    }


def _ortho_scale_for_view(
    *,
    scene: dict[str, Any],
    view_config: dict[str, Any],
    projection_mode: str,
) -> float | None:
    if projection_mode != "orthographic":
        return None
    return estimate_ortho_scale([scene["native_render_points"], scene["ffs_render_points"]], view_config=view_config)


def _compute_step_metrics(
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
        ortho_scale = _ortho_scale_for_view(scene=scene, view_config=view_config, projection_mode=projection_mode)
        native_support_map = compute_support_count_map(
            scene["native_render_camera_clouds"],
            view_config=view_config,
            width=width,
            height=height,
            projection_mode=projection_mode,
            ortho_scale=ortho_scale,
        )
        ffs_support_map = compute_support_count_map(
            scene["ffs_render_camera_clouds"],
            view_config=view_config,
            width=width,
            height=height,
            projection_mode=projection_mode,
            ortho_scale=ortho_scale,
        )
        native_support = summarize_support_counts(native_support_map["support_count"], native_support_map["valid"])
        ffs_support = summarize_support_counts(ffs_support_map["support_count"], ffs_support_map["valid"])
        native_mismatch = render_mismatch_residual(
            scene["native_render_camera_clouds"],
            view_config=view_config,
            width=width,
            height=height,
            projection_mode=projection_mode,
            ortho_scale=ortho_scale,
        )[1]["summary"]
        ffs_mismatch = render_mismatch_residual(
            scene["ffs_render_camera_clouds"],
            view_config=view_config,
            width=width,
            height=height,
            projection_mode=projection_mode,
            ortho_scale=ortho_scale,
        )[1]["summary"]
        metrics.append(
            {
                "step_idx": int(orbit_step["step_idx"]),
                "angle_deg": float(orbit_step["angle_deg"]),
                "is_supported": bool(view_config.get("is_supported", True)),
                "native_support": native_support,
                "ffs_support": ffs_support,
                "native_mismatch": native_mismatch,
                "ffs_mismatch": ffs_mismatch,
                "mean_multi_camera_support": float(
                    np.mean(
                        [
                            native_support["support_ratio_2"] + native_support["support_ratio_3"],
                            ffs_support["support_ratio_2"] + ffs_support["support_ratio_3"],
                        ]
                    )
                ),
                "mean_mismatch_residual_m": float(np.mean([native_mismatch["residual_mean_m"], ffs_mismatch["residual_mean_m"]])),
                "mean_mismatch_p90_m": float(np.mean([native_mismatch["residual_p90_m"], ffs_mismatch["residual_p90_m"]])),
            }
        )
    return metrics


def _render_hero_compare(
    *,
    output_dir: Path,
    case_label: str,
    frame_idx: int,
    scene_crop_mode: str,
    scene: dict[str, Any],
    view_config: dict[str, Any],
    overview_state: dict[str, Any],
    renderer: str,
    render_mode: str,
    projection_mode: str,
    point_radius_px: int,
    supersample_scale: int,
) -> dict[str, Any]:
    ortho_scale = _ortho_scale_for_view(scene=scene, view_config=view_config, projection_mode=projection_mode)
    native_image, _ = render_point_cloud(
        scene["native_render_points"],
        scene["native_render_colors"],
        renderer=renderer,
        view_config=view_config,
        render_mode=render_mode,
        scalar_bounds=scene["scalar_bounds"],
        width=1280,
        height=900,
        point_radius_px=point_radius_px,
        supersample_scale=supersample_scale,
        projection_mode=projection_mode,
        ortho_scale=ortho_scale,
    )
    ffs_image, _ = render_point_cloud(
        scene["ffs_render_points"],
        scene["ffs_render_colors"],
        renderer=renderer,
        view_config=view_config,
        render_mode=render_mode,
        scalar_bounds=scene["scalar_bounds"],
        width=1280,
        height=900,
        point_radius_px=point_radius_px,
        supersample_scale=supersample_scale,
        projection_mode=projection_mode,
        ortho_scale=ortho_scale,
    )
    overview_inset = render_overview_inset(
        overview_state,
        current_views=[view_config],
        inset_size=(260, 170),
        angle_label=f"{view_config.get('azimuth_deg', view_config.get('angle_deg', 0.0)):+.1f} deg",
    )
    hero = compose_hero_compare(
        title_lines=[
            f"{case_label} | Native vs FFS",
            f"frame={frame_idx} | orbit={view_config.get('angle_deg', 0.0):+.1f} deg | proj={projection_mode} | crop={scene_crop_mode}",
        ],
        native_image=native_image,
        ffs_image=ffs_image,
        overview_inset=overview_inset,
        warning_text=view_config.get("warning_text"),
    )
    output_path = output_dir / "01_hero_compare.png"
    write_image(output_path, hero)
    return {
        "path": str(output_path),
        "native_image": native_image,
        "ffs_image": ffs_image,
    }


def _render_merge_evidence(
    *,
    output_dir: Path,
    case_label: str,
    frame_idx: int,
    scene_crop_mode: str,
    scene: dict[str, Any],
    view_config: dict[str, Any],
    projection_mode: str,
) -> dict[str, Any]:
    ortho_scale = _ortho_scale_for_view(scene=scene, view_config=view_config, projection_mode=projection_mode)
    native_source, native_source_metrics = render_source_attribution_overlay(
        scene["native_render_camera_clouds"],
        view_config=view_config,
        width=560,
        height=400,
        projection_mode=projection_mode,
        ortho_scale=ortho_scale,
    )
    ffs_source, ffs_source_metrics = render_source_attribution_overlay(
        scene["ffs_render_camera_clouds"],
        view_config=view_config,
        width=560,
        height=400,
        projection_mode=projection_mode,
        ortho_scale=ortho_scale,
    )
    native_support_map = compute_support_count_map(
        scene["native_render_camera_clouds"],
        view_config=view_config,
        width=560,
        height=400,
        projection_mode=projection_mode,
        ortho_scale=ortho_scale,
    )
    ffs_support_map = compute_support_count_map(
        scene["ffs_render_camera_clouds"],
        view_config=view_config,
        width=560,
        height=400,
        projection_mode=projection_mode,
        ortho_scale=ortho_scale,
    )
    native_support = overlay_support_legend(render_support_count_map(native_support_map["support_count"], native_support_map["valid"]))
    ffs_support = overlay_support_legend(render_support_count_map(ffs_support_map["support_count"], ffs_support_map["valid"]))
    native_mismatch, native_mismatch_metrics = render_mismatch_residual(
        scene["native_render_camera_clouds"],
        view_config=view_config,
        width=560,
        height=400,
        projection_mode=projection_mode,
        ortho_scale=ortho_scale,
    )
    ffs_mismatch, ffs_mismatch_metrics = render_mismatch_residual(
        scene["ffs_render_camera_clouds"],
        view_config=view_config,
        width=560,
        height=400,
        projection_mode=projection_mode,
        ortho_scale=ortho_scale,
    )
    board = compose_turntable_board(
        title_lines=[
            f"{case_label} | Merge Evidence",
            f"frame={frame_idx} | orbit={view_config.get('angle_deg', 0.0):+.1f} deg | proj={projection_mode} | crop={scene_crop_mode}",
        ],
        column_headers=["Source attribution", "Support count", "Mismatch residual"],
        row_headers=["Native", "FFS"],
        native_images=[native_source, native_support, native_mismatch],
        ffs_images=[ffs_source, ffs_support, ffs_mismatch],
        overview_inset=None,
    )
    output_path = output_dir / "02_merge_evidence.png"
    write_image(output_path, board)
    return {
        "path": str(output_path),
        "source": {"native": native_source_metrics, "ffs": ffs_source_metrics},
        "support": {
            "native": summarize_support_counts(native_support_map["support_count"], native_support_map["valid"]),
            "ffs": summarize_support_counts(ffs_support_map["support_count"], ffs_support_map["valid"]),
        },
        "mismatch": {
            "native": native_mismatch_metrics["summary"],
            "ffs": ffs_mismatch_metrics["summary"],
        },
    }


def _compute_pair_diagnostics(
    *,
    selection: dict[str, Any],
    camera_pairs: list[tuple[int, int]],
    use_float_ffs_depth_when_available: bool,
) -> list[dict[str, Any]]:
    native_case_dir = Path(selection["native_case_dir"])
    ffs_case_dir = Path(selection["ffs_case_dir"])
    native_metadata = selection["native_metadata"]
    ffs_metadata = selection["ffs_metadata"]
    same_case_mode = bool(selection["same_case_mode"])
    native_frame_idx = int(selection["native_frame_idx"])
    ffs_frame_idx = int(selection["ffs_frame_idx"])
    native_intrinsics = get_case_intrinsics(native_metadata)
    ffs_intrinsics = get_case_intrinsics(ffs_metadata)
    native_c2w = get_case_camera_transform(case_dir=native_case_dir, metadata=native_metadata)
    ffs_c2w = get_case_camera_transform(case_dir=ffs_case_dir, metadata=ffs_metadata)

    pair_metrics: list[dict[str, Any]] = []
    for src_idx, dst_idx in camera_pairs:
        native_src_rgb = load_color_frame(native_case_dir, src_idx, native_frame_idx)
        native_dst_rgb = load_color_frame(native_case_dir, dst_idx, native_frame_idx)
        _, native_src_depth_m, native_depth_info = load_depth_frame(
            case_dir=native_case_dir,
            metadata=native_metadata,
            camera_idx=src_idx,
            frame_idx=native_frame_idx,
            depth_source="realsense",
            use_float_ffs_depth_when_available=use_float_ffs_depth_when_available,
        )
        ffs_src_rgb = native_src_rgb if same_case_mode else load_color_frame(ffs_case_dir, src_idx, ffs_frame_idx)
        ffs_dst_rgb = native_dst_rgb if same_case_mode else load_color_frame(ffs_case_dir, dst_idx, ffs_frame_idx)
        _, ffs_src_depth_m, ffs_depth_info = load_depth_frame(
            case_dir=ffs_case_dir,
            metadata=ffs_metadata,
            camera_idx=src_idx,
            frame_idx=ffs_frame_idx,
            depth_source="ffs",
            use_float_ffs_depth_when_available=use_float_ffs_depth_when_available,
        )
        native_warped_rgb, native_valid, _ = warp_rgb_between_cameras(
            source_rgb=native_src_rgb,
            source_depth_m=native_src_depth_m,
            source_K=native_intrinsics[src_idx],
            source_c2w=native_c2w[src_idx],
            target_K=native_intrinsics[dst_idx],
            target_c2w=native_c2w[dst_idx],
            output_shape=native_dst_rgb.shape[:2],
        )
        ffs_warped_rgb, ffs_valid, _ = warp_rgb_between_cameras(
            source_rgb=ffs_src_rgb,
            source_depth_m=ffs_src_depth_m,
            source_K=ffs_intrinsics[src_idx],
            source_c2w=ffs_c2w[src_idx],
            target_K=ffs_intrinsics[dst_idx],
            target_c2w=ffs_c2w[dst_idx],
            output_shape=ffs_dst_rgb.shape[:2],
        )
        native_heatmap, native_stats = compute_photometric_residual(native_warped_rgb, native_dst_rgb, native_valid)
        ffs_heatmap, ffs_stats = compute_photometric_residual(ffs_warped_rgb, ffs_dst_rgb, ffs_valid)
        pair_metrics.append(
            {
                "pair": (int(src_idx), int(dst_idx)),
                "native": native_stats,
                "ffs": ffs_stats,
                "mean_valid_ratio": float(np.mean([native_stats["valid_warped_pixel_ratio"], ffs_stats["valid_warped_pixel_ratio"]])),
                "residual_gap": float(abs(native_stats["edge_weighted_residual_mean"] - ffs_stats["edge_weighted_residual_mean"])),
                "images": {
                    "native_target": native_dst_rgb,
                    "native_warp": native_warped_rgb,
                    "native_residual": native_heatmap,
                    "ffs_target": ffs_dst_rgb,
                    "ffs_warp": ffs_warped_rgb,
                    "ffs_residual": ffs_heatmap,
                },
                "depth_dirs": {
                    "native": native_depth_info["depth_dir_used"],
                    "ffs": ffs_depth_info["depth_dir_used"],
                },
            }
        )
    return pair_metrics


def _render_truth_board(
    *,
    output_dir: Path,
    case_label: str,
    frame_idx: int,
    pair_info: dict[str, Any],
    same_case_mode: bool,
) -> dict[str, Any]:
    src_idx, dst_idx = pair_info["pair"]
    ffs_target_label = f"FFS Tgt C{dst_idx}"
    tiles = [
        label_tile(pair_info["images"]["native_target"], f"Native Tgt C{dst_idx}", (360, 240)),
        label_tile(pair_info["images"]["native_warp"], f"Native Warp ({pair_info['depth_dirs']['native']})", (360, 240)),
        label_tile(pair_info["images"]["native_residual"], "Native Residual", (360, 240)),
        label_tile(pair_info["images"]["native_target"] if same_case_mode else pair_info["images"]["ffs_target"], ffs_target_label, (360, 240)),
        label_tile(pair_info["images"]["ffs_warp"], f"FFS Warp ({pair_info['depth_dirs']['ffs']})", (360, 240)),
        label_tile(pair_info["images"]["ffs_residual"], "FFS Residual", (360, 240)),
    ]
    board = compose_depth_review_board(
        title_lines=[
            f"{case_label} | Truth Board",
            f"frame={frame_idx} | pair=C{src_idx} -> C{dst_idx}",
        ],
        metric_lines=[
            f"Native valid={pair_info['native']['valid_warped_pixel_ratio']:.3f} | mean={pair_info['native']['residual_mean']:.2f} | edge={pair_info['native']['edge_weighted_residual_mean']:.2f}",
            f"FFS valid={pair_info['ffs']['valid_warped_pixel_ratio']:.3f} | mean={pair_info['ffs']['residual_mean']:.2f} | edge={pair_info['ffs']['edge_weighted_residual_mean']:.2f}",
        ],
        rows=[tiles[:3], tiles[3:]],
    )
    output_path = output_dir / "03_truth_board.png"
    write_image(output_path, board)
    return {"path": str(output_path)}


def run_professor_triptych_workflow(
    *,
    aligned_root: Path,
    output_dir: Path,
    case_name: str | None = None,
    realsense_case: str | None = None,
    ffs_case: str | None = None,
    frame_idx: int = 0,
    renderer: str = "fallback",
    render_mode: str = "neutral_gray_shaded",
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
    write_video: bool = False,
    write_keyframes: bool = False,
) -> dict[str, Any]:
    if render_mode not in RENDER_MODES:
        raise ValueError(f"Unsupported render_mode: {render_mode}")
    if projection_mode not in PROJECTION_MODES:
        raise ValueError(f"Unsupported projection_mode: {projection_mode}")
    if scene_crop_mode not in SCENE_CROP_MODES:
        raise ValueError(f"Unsupported scene_crop_mode: {scene_crop_mode}")
    if angle_mode not in ("auto", "explicit"):
        raise ValueError(f"Unsupported angle_mode: {angle_mode}")

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    turntable_state = _build_turntable_scene(
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
    )
    selection = turntable_state["selection"]
    scene = turntable_state["scene"]
    refinement = turntable_state["refinement"]
    orbit_state = _build_orbit_state(
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
    step_metrics = _compute_step_metrics(
        scene=scene,
        orbit_steps=orbit_state["orbit_steps"],
        projection_mode=projection_mode,
        width=640,
        height=420,
    )
    selected_step = select_professor_angle(step_metrics, angle_mode=angle_mode, angle_deg=angle_deg)
    selected_view = orbit_state["orbit_steps"][int(selected_step["step_idx"])]["view_config"]
    case_label = (
        selection["native_case_dir"].name
        if selection["same_case_mode"]
        else f"{selection['native_case_dir'].name} vs {selection['ffs_case_dir'].name}"
    )

    hero_info = _render_hero_compare(
        output_dir=output_dir,
        case_label=case_label,
        frame_idx=int(selection["native_frame_idx"]),
        scene_crop_mode=scene_crop_mode,
        scene=scene,
        view_config=selected_view,
        overview_state=orbit_state["overview_state"],
        renderer=renderer,
        render_mode=render_mode,
        projection_mode=projection_mode,
        point_radius_px=point_radius_px,
        supersample_scale=supersample_scale,
    )
    evidence_info = _render_merge_evidence(
        output_dir=output_dir,
        case_label=case_label,
        frame_idx=int(selection["native_frame_idx"]),
        scene_crop_mode=scene_crop_mode,
        scene=scene,
        view_config=selected_view,
        projection_mode=projection_mode,
    )

    pair_metrics = _compute_pair_diagnostics(
        selection=selection,
        camera_pairs=build_camera_pairs(selection["camera_ids"], None),
        use_float_ffs_depth_when_available=use_float_ffs_depth_when_available,
    )
    selected_pair = select_truth_camera_pair(pair_metrics)
    truth_info = _render_truth_board(
        output_dir=output_dir,
        case_label=case_label,
        frame_idx=int(selection["native_frame_idx"]),
        pair_info=selected_pair,
        same_case_mode=bool(selection["same_case_mode"]),
    )

    summary = {
        "same_case_mode": bool(selection["same_case_mode"]),
        "native_case_dir": str(selection["native_case_dir"]),
        "ffs_case_dir": str(selection["ffs_case_dir"]),
        "frame_idx": int(selection["native_frame_idx"]),
        "projection_mode": projection_mode,
        "scene_crop_mode": scene_crop_mode,
        "render_mode": render_mode,
        "hero_angle_selection": {
            "mode": angle_mode,
            "selected_step_idx": int(selected_step["step_idx"]),
            "selected_angle_deg": float(selected_step["angle_deg"]),
            "selected_is_supported": bool(selected_step["is_supported"]),
            "mean_multi_camera_support": float(selected_step["mean_multi_camera_support"]),
            "mean_mismatch_residual_m": float(selected_step["mean_mismatch_residual_m"]),
            "mean_mismatch_p90_m": float(selected_step["mean_mismatch_p90_m"]),
            "candidate_count": int(len(step_metrics)),
        },
        "truth_camera_pair": {
            "src_camera_idx": int(selected_pair["pair"][0]),
            "dst_camera_idx": int(selected_pair["pair"][1]),
            "mean_valid_ratio": float(selected_pair["mean_valid_ratio"]),
            "residual_gap": float(selected_pair["residual_gap"]),
            "native": selected_pair["native"],
            "ffs": selected_pair["ffs"],
        },
        "top_level_outputs": {
            "hero_compare": hero_info["path"],
            "merge_evidence": evidence_info["path"],
            "truth_board": truth_info["path"],
        },
        "debug_written": bool(write_debug or write_video or write_keyframes),
        "video_written": bool(write_video),
        "keyframes_written": bool(write_keyframes),
    }
    write_json(output_dir / "summary.json", summary)

    if write_debug or write_video or write_keyframes:
        debug_dir = output_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        write_json(debug_dir / "angle_selection_metrics.json", {"steps": step_metrics, "selected": selected_step})
        serializable_pairs = []
        for item in pair_metrics:
            serializable_pairs.append(
                {
                    "pair": list(item["pair"]),
                    "mean_valid_ratio": item["mean_valid_ratio"],
                    "residual_gap": item["residual_gap"],
                    "native": item["native"],
                    "ffs": item["ffs"],
                    "depth_dirs": item["depth_dirs"],
                }
            )
        write_json(
            debug_dir / "truth_pair_metrics.json",
            {
                "pairs": serializable_pairs,
                "selected_pair": list(selected_pair["pair"]),
            },
        )
        if refinement["pass1_crop"] is not None:
            write_json(debug_dir / "object_roi_pass1_world.json", _json_ready(refinement["pass1_crop"]))
        if refinement["pass2_crop"] is not None:
            write_json(debug_dir / "object_roi_pass2_world.json", _json_ready(refinement["pass2_crop"]))
        write_image(debug_dir / "scene_overview_with_cameras.png", orbit_state["overview_state"]["image"])
        if write_video or write_keyframes:
            run_turntable_compare_workflow(
                aligned_root=Path(aligned_root).resolve(),
                output_dir=debug_dir / "turntable_full",
                case_name=case_name,
                realsense_case=realsense_case,
                ffs_case=ffs_case,
                frame_idx=frame_idx,
                renderer=renderer,
                render_mode=render_mode,
                write_mp4=bool(write_video),
                write_gif=bool(write_video),
                write_keyframe_sheet=bool(write_keyframes),
                num_orbit_steps=num_orbit_steps,
                orbit_degrees=orbit_degrees,
                camera_ids=camera_ids,
                scene_crop_mode=scene_crop_mode,
                focus_mode=focus_mode,
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
                projection_mode=projection_mode,
                point_radius_px=point_radius_px,
                supersample_scale=supersample_scale,
                voxel_size=voxel_size,
                max_points_per_camera=max_points_per_camera,
                depth_min_m=depth_min_m,
                depth_max_m=depth_max_m,
                use_float_ffs_depth_when_available=use_float_ffs_depth_when_available,
                fps=8,
                orbit_mode=orbit_mode,
                layout_mode="side_by_side_large",
                orbit_radius_scale=orbit_radius_scale,
                view_height_offset=view_height_offset,
                render_both_modes=True,
                coverage_margin_deg=coverage_margin_deg,
                show_unsupported_warning=True,
            )
            run_reprojection_compare_workflow(
                aligned_root=Path(aligned_root).resolve(),
                output_dir=debug_dir / "reprojection_full",
                case_name=case_name,
                realsense_case=realsense_case,
                ffs_case=ffs_case,
                frame_start=frame_idx,
                frame_end=frame_idx,
                frame_stride=1,
                camera_ids=camera_ids,
                camera_pairs=[selected_pair["pair"]],
                write_mp4=bool(write_video),
                fps=8,
                use_float_ffs_depth_when_available=use_float_ffs_depth_when_available,
            )

    return {
        "output_dir": str(output_dir),
        "summary": summary,
    }
