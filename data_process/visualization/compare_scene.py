from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .camera_frusta import build_camera_frustum_geometry, extract_camera_poses
from .pointcloud_compare import estimate_ortho_scale
from .semantic_world import infer_display_frame_state
from .source_compare import render_mismatch_residual
from .support_compare import compute_support_count_map, summarize_support_counts
from .turntable_compare import (
    _resolve_single_frame_case_selection_model,
    build_scene_overview_state,
    build_single_frame_scene,
    load_single_frame_compare_clouds,
    prepare_object_roi_refinement,
)
from .views import build_object_centered_orbit_views, estimate_orbit_axis


def parse_manual_xyz_roi(
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


def parse_manual_image_roi_json(manual_image_roi_json: str | Path | None) -> dict[int, tuple[int, int, int, int]] | None:
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


def build_turntable_scene_state(
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
    display_frame: str = "semantic_world",
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
    manual_image_roi_by_camera = parse_manual_image_roi_json(manual_image_roi_json)
    raw_scene = load_single_frame_compare_clouds(
        selection,
        voxel_size=voxel_size,
        max_points_per_camera=None,
        depth_min_m=depth_min_m,
        depth_max_m=depth_max_m,
        use_float_ffs_depth_when_available=use_float_ffs_depth_when_available,
        pixel_roi_by_camera=None,
    )
    manual_xyz_roi = parse_manual_xyz_roi(
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
    display_state = infer_display_frame_state(
        selection=selection,
        scene=scene,
        display_frame=display_frame,
    )
    selection = {
        **selection,
        "display_frame": display_frame,
        "display_camera_c2w": display_state["camera_c2w"],
    }
    return {
        "selection": selection,
        "manual_image_roi_by_camera": manual_image_roi_by_camera,
        "scene": display_state["scene"],
        "refinement": refinement,
        "display_state": display_state,
    }


def build_orbit_state(
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
        selection.get("display_camera_c2w", selection["native_c2w"]),
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


def ortho_scale_for_view(
    *,
    scene: dict[str, Any],
    view_config: dict[str, Any],
    projection_mode: str,
) -> float | None:
    if projection_mode != "orthographic":
        return None
    return estimate_ortho_scale([scene["native_render_points"], scene["ffs_render_points"]], view_config=view_config)


def mask_bbox(valid_mask: np.ndarray) -> tuple[int, int, int, int] | None:
    yy, xx = np.nonzero(np.asarray(valid_mask, dtype=bool))
    if len(xx) == 0:
        return None
    return int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1


def bbox_area(bbox: tuple[int, int, int, int] | None) -> int:
    if bbox is None:
        return 0
    x0, y0, x1, y1 = bbox
    return max(0, int(x1 - x0)) * max(0, int(y1 - y0))


def crop_to_bbox(image: np.ndarray, bbox: tuple[int, int, int, int], *, padding: int = 14) -> np.ndarray:
    h, w = image.shape[:2]
    x0, y0, x1, y1 = bbox
    x0 = max(0, x0 - padding)
    y0 = max(0, y0 - padding)
    x1 = min(w, x1 + padding)
    y1 = min(h, y1 + padding)
    return np.asarray(image, dtype=np.uint8)[y0:y1, x0:x1].copy()


def compute_object_view_stats(
    *,
    object_camera_clouds: list[dict[str, Any]],
    combined_camera_clouds: list[dict[str, Any]],
    view_config: dict[str, Any],
    projection_mode: str,
    ortho_scale: float | None,
    width: int,
    height: int,
) -> dict[str, float]:
    object_support_map = compute_support_count_map(
        object_camera_clouds,
        view_config=view_config,
        width=width,
        height=height,
        projection_mode=projection_mode,
        ortho_scale=ortho_scale,
    )
    combined_support_map = compute_support_count_map(
        combined_camera_clouds,
        view_config=view_config,
        width=width,
        height=height,
        projection_mode=projection_mode,
        ortho_scale=ortho_scale,
    )
    object_support = summarize_support_counts(object_support_map["support_count"], object_support_map["valid"])
    object_mismatch = render_mismatch_residual(
        object_camera_clouds,
        view_config=view_config,
        width=width,
        height=height,
        projection_mode=projection_mode,
        ortho_scale=ortho_scale,
    )[1]["summary"]
    frame_area = float(width * height)
    object_bbox = mask_bbox(object_support_map["valid"])
    object_visible_area = int(np.count_nonzero(object_support_map["valid"]))
    combined_visible_area = int(np.count_nonzero(combined_support_map["valid"]))
    object_bbox_area = bbox_area(object_bbox)
    bbox_fill_ratio = float(object_bbox_area / frame_area) if frame_area > 0 else 0.0
    projected_area_ratio = float(object_visible_area / frame_area) if frame_area > 0 else 0.0
    context_penalty = 0.0
    if combined_visible_area > 0:
        context_penalty = float(max(0, combined_visible_area - object_visible_area) / combined_visible_area)
    silhouette_penalty = 1.0
    if object_bbox is not None:
        x0, y0, x1, y1 = object_bbox
        bbox_w = max(1, x1 - x0)
        bbox_h = max(1, y1 - y0)
        silhouette_penalty = 1.0 - float(min(bbox_w, bbox_h) / max(bbox_w, bbox_h))
    return {
        "object_projected_area_ratio": projected_area_ratio,
        "object_bbox_fill_ratio": bbox_fill_ratio,
        "object_multi_camera_support_ratio": float(object_support["support_ratio_2"] + object_support["support_ratio_3"]),
        "object_mismatch_residual_m": float(object_mismatch["residual_mean_m"]),
        "context_dominance_penalty": context_penalty,
        "silhouette_penalty": silhouette_penalty,
        "object_valid_pixel_count": float(object_visible_area),
        "combined_valid_pixel_count": float(combined_visible_area),
        "object_overlap_pixel_count": float(object_mismatch["overlap_pixel_count"]),
    }
