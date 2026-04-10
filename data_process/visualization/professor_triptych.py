from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
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
from .object_compare import project_world_roi_to_camera_bbox
from .pointcloud_compare import (
    PROJECTION_MODES,
    RENDER_MODES,
    SCENE_CROP_MODES,
    estimate_ortho_scale,
    get_case_intrinsics,
    render_point_cloud,
)
from .renderers import rasterize_point_cloud_view
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

    supported_steps = [item for item in step_metrics if bool(item["is_supported"])]
    ranked_candidates = supported_steps if supported_steps else step_metrics
    ranked = sorted(
        ranked_candidates,
        key=lambda item: (
            -float(item["final_score"]),
            -float(item["object_projected_area_ratio"]),
            -float(item["object_bbox_fill_ratio"]),
            -float(item["object_multi_camera_support_ratio"]),
            float(item["object_mismatch_residual_m"]),
            float(item["context_dominance_penalty"]),
            float(item["silhouette_penalty"]),
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
            -float(item["pair_object_visibility_score"]),
            -float(item["object_overlap_area"]),
            float(np.mean([item["object_edge_weighted_residual_mean_native"], item["object_edge_weighted_residual_mean_ffs"]])),
            float(np.mean([item["object_residual_mean_native"], item["object_residual_mean_ffs"]])),
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


def _mask_bbox(valid_mask: np.ndarray) -> tuple[int, int, int, int] | None:
    yy, xx = np.nonzero(np.asarray(valid_mask, dtype=bool))
    if len(xx) == 0:
        return None
    return int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1


def _bbox_area(bbox: tuple[int, int, int, int] | None) -> int:
    if bbox is None:
        return 0
    x0, y0, x1, y1 = bbox
    return max(0, int(x1 - x0)) * max(0, int(y1 - y0))


def _crop_to_bbox(image: np.ndarray, bbox: tuple[int, int, int, int], *, padding: int = 14) -> np.ndarray:
    h, w = image.shape[:2]
    x0, y0, x1, y1 = bbox
    x0 = max(0, x0 - padding)
    y0 = max(0, y0 - padding)
    x1 = min(w, x1 + padding)
    y1 = min(h, y1 + padding)
    return np.asarray(image, dtype=np.uint8)[y0:y1, x0:x1].copy()


def _compute_object_view_stats(
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
    object_bbox = _mask_bbox(object_support_map["valid"])
    object_visible_area = int(np.count_nonzero(object_support_map["valid"]))
    combined_visible_area = int(np.count_nonzero(combined_support_map["valid"]))
    object_bbox_area = _bbox_area(object_bbox)
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


def _score_hero_candidate(metrics: dict[str, Any]) -> float:
    return float(
        2.8 * float(metrics["object_projected_area_ratio"])
        + 2.2 * float(metrics["object_bbox_fill_ratio"])
        + 1.8 * float(metrics["object_multi_camera_support_ratio"])
        - 6.0 * float(metrics["object_mismatch_residual_m"])
        - 1.4 * float(metrics["context_dominance_penalty"])
        - 1.0 * float(metrics["silhouette_penalty"])
    )


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
        native_object_metrics = _compute_object_view_stats(
            object_camera_clouds=scene["native_object_camera_clouds"],
            combined_camera_clouds=scene["native_render_camera_clouds"],
            view_config=view_config,
            projection_mode=projection_mode,
            ortho_scale=ortho_scale,
            width=width,
            height=height,
        )
        ffs_object_metrics = _compute_object_view_stats(
            object_camera_clouds=scene["ffs_object_camera_clouds"],
            combined_camera_clouds=scene["ffs_render_camera_clouds"],
            view_config=view_config,
            projection_mode=projection_mode,
            ortho_scale=ortho_scale,
            width=width,
            height=height,
        )
        aggregate_metrics = {
            "object_projected_area_ratio": float(np.mean([native_object_metrics["object_projected_area_ratio"], ffs_object_metrics["object_projected_area_ratio"]])),
            "object_bbox_fill_ratio": float(np.mean([native_object_metrics["object_bbox_fill_ratio"], ffs_object_metrics["object_bbox_fill_ratio"]])),
            "object_multi_camera_support_ratio": float(np.mean([native_object_metrics["object_multi_camera_support_ratio"], ffs_object_metrics["object_multi_camera_support_ratio"]])),
            "object_mismatch_residual_m": float(np.mean([native_object_metrics["object_mismatch_residual_m"], ffs_object_metrics["object_mismatch_residual_m"]])),
            "context_dominance_penalty": float(np.mean([native_object_metrics["context_dominance_penalty"], ffs_object_metrics["context_dominance_penalty"]])),
            "silhouette_penalty": float(np.mean([native_object_metrics["silhouette_penalty"], ffs_object_metrics["silhouette_penalty"]])),
        }
        metrics.append(
            {
                "step_idx": int(orbit_step["step_idx"]),
                "angle_deg": float(orbit_step["angle_deg"]),
                "is_supported": bool(view_config.get("is_supported", True)),
                "native_object_metrics": native_object_metrics,
                "ffs_object_metrics": ffs_object_metrics,
                **aggregate_metrics,
                "final_score": _score_hero_candidate(aggregate_metrics),
            }
        )
    return metrics


def _summarize_object_reprojection_region(
    *,
    warped_rgb: np.ndarray,
    target_rgb: np.ndarray,
    valid_mask: np.ndarray,
    roi_bbox: tuple[int, int, int, int] | None,
) -> dict[str, float]:
    if roi_bbox is None:
        return {
            "object_warp_valid_ratio": 0.0,
            "object_residual_mean": 0.0,
            "object_edge_weighted_residual_mean": 0.0,
            "object_overlap_area": 0.0,
        }
    x0, y0, x1, y1 = roi_bbox
    warped = np.asarray(warped_rgb, dtype=np.float32)
    target = np.asarray(target_rgb, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)
    valid_roi = valid[y0:y1, x0:x1]
    roi_area = max(1, int((x1 - x0) * (y1 - y0)))
    stats = {
        "object_warp_valid_ratio": float(np.count_nonzero(valid_roi) / roi_area),
        "object_residual_mean": 0.0,
        "object_edge_weighted_residual_mean": 0.0,
        "object_overlap_area": float(np.count_nonzero(valid_roi)),
    }
    if not np.any(valid_roi):
        return stats
    residual = np.abs(warped - target).mean(axis=2)[y0:y1, x0:x1]
    valid_residual = residual[valid_roi]
    stats["object_residual_mean"] = float(valid_residual.mean())
    target_gray = cv2.cvtColor(np.asarray(target_rgb, dtype=np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    target_gray_roi = target_gray[y0:y1, x0:x1]
    gx = cv2.Sobel(target_gray_roi, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(target_gray_roi, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(gx * gx + gy * gy)
    max_gradient = max(1e-6, float(gradient.max()))
    weights = 1.0 + gradient / max_gradient
    edge_weights = weights[valid_roi]
    stats["object_edge_weighted_residual_mean"] = float(np.sum(valid_residual * edge_weights) / np.sum(edge_weights))
    return stats


def _project_object_bbox_for_pair(
    *,
    case_dir: Path,
    camera_idx: int,
    c2w: np.ndarray,
    K_color: np.ndarray,
    color_path: Path,
    object_roi_min: np.ndarray,
    object_roi_max: np.ndarray,
) -> tuple[int, int, int, int] | None:
    bbox, _ = project_world_roi_to_camera_bbox(
        {
            "camera_idx": int(camera_idx),
            "color_path": str(color_path),
            "c2w": np.asarray(c2w, dtype=np.float32),
            "K_color": np.asarray(K_color, dtype=np.float32),
        },
        object_roi_min=np.asarray(object_roi_min, dtype=np.float32),
        object_roi_max=np.asarray(object_roi_max, dtype=np.float32),
    )
    return bbox


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
        scene["native_object_points"],
        scene["native_object_colors"],
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
        scene["ffs_object_points"],
        scene["ffs_object_colors"],
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
        scene["native_object_camera_clouds"],
        view_config=view_config,
        width=560,
        height=400,
        projection_mode=projection_mode,
        ortho_scale=ortho_scale,
    )
    ffs_source, ffs_source_metrics = render_source_attribution_overlay(
        scene["ffs_object_camera_clouds"],
        view_config=view_config,
        width=560,
        height=400,
        projection_mode=projection_mode,
        ortho_scale=ortho_scale,
    )
    native_support_map = compute_support_count_map(
        scene["native_object_camera_clouds"],
        view_config=view_config,
        width=560,
        height=400,
        projection_mode=projection_mode,
        ortho_scale=ortho_scale,
    )
    ffs_support_map = compute_support_count_map(
        scene["ffs_object_camera_clouds"],
        view_config=view_config,
        width=560,
        height=400,
        projection_mode=projection_mode,
        ortho_scale=ortho_scale,
    )
    native_support = overlay_support_legend(render_support_count_map(native_support_map["support_count"], native_support_map["valid"]))
    ffs_support = overlay_support_legend(render_support_count_map(ffs_support_map["support_count"], ffs_support_map["valid"]))
    native_mismatch, native_mismatch_metrics = render_mismatch_residual(
        scene["native_object_camera_clouds"],
        view_config=view_config,
        width=560,
        height=400,
        projection_mode=projection_mode,
        ortho_scale=ortho_scale,
    )
    ffs_mismatch, ffs_mismatch_metrics = render_mismatch_residual(
        scene["ffs_object_camera_clouds"],
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
    scene: dict[str, Any],
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
        native_target_color_path = native_case_dir / "color" / str(dst_idx) / f"{native_frame_idx}.png"
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
        ffs_target_color_path = (
            native_target_color_path
            if same_case_mode
            else ffs_case_dir / "color" / str(dst_idx) / f"{ffs_frame_idx}.png"
        )
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
        native_object_bbox = _project_object_bbox_for_pair(
            case_dir=native_case_dir,
            camera_idx=dst_idx,
            c2w=native_c2w[dst_idx],
            K_color=native_intrinsics[dst_idx],
            color_path=native_target_color_path,
            object_roi_min=scene["object_roi_bounds"]["min"],
            object_roi_max=scene["object_roi_bounds"]["max"],
        )
        ffs_object_bbox = _project_object_bbox_for_pair(
            case_dir=ffs_case_dir,
            camera_idx=dst_idx,
            c2w=ffs_c2w[dst_idx],
            K_color=ffs_intrinsics[dst_idx],
            color_path=ffs_target_color_path,
            object_roi_min=scene["object_roi_bounds"]["min"],
            object_roi_max=scene["object_roi_bounds"]["max"],
        )
        object_bbox = native_object_bbox if native_object_bbox is not None else ffs_object_bbox
        native_object_stats = _summarize_object_reprojection_region(
            warped_rgb=native_warped_rgb,
            target_rgb=native_dst_rgb,
            valid_mask=native_valid,
            roi_bbox=object_bbox,
        )
        ffs_object_stats = _summarize_object_reprojection_region(
            warped_rgb=ffs_warped_rgb,
            target_rgb=ffs_dst_rgb,
            valid_mask=ffs_valid,
            roi_bbox=object_bbox,
        )
        overlap_ratio = 0.0
        if object_bbox is not None:
            overlap_ratio = float(
                np.mean(
                    [
                        native_object_stats["object_overlap_area"],
                        ffs_object_stats["object_overlap_area"],
                    ]
                )
                / max(1.0, float(_bbox_area(object_bbox)))
            )
        pair_object_visibility_score = float(
            np.mean(
                [
                    native_object_stats["object_warp_valid_ratio"],
                    ffs_object_stats["object_warp_valid_ratio"],
                ]
            )
            * max(0.0, overlap_ratio) ** 0.5
        )
        pair_metrics.append(
            {
                "pair": (int(src_idx), int(dst_idx)),
                "native": native_stats,
                "ffs": ffs_stats,
                "mean_valid_ratio": float(np.mean([native_stats["valid_warped_pixel_ratio"], ffs_stats["valid_warped_pixel_ratio"]])),
                "residual_gap": float(abs(native_stats["edge_weighted_residual_mean"] - ffs_stats["edge_weighted_residual_mean"])),
                "object_bbox": list(object_bbox) if object_bbox is not None else None,
                "object_warp_valid_ratio_native": native_object_stats["object_warp_valid_ratio"],
                "object_warp_valid_ratio_ffs": ffs_object_stats["object_warp_valid_ratio"],
                "object_residual_mean_native": native_object_stats["object_residual_mean"],
                "object_residual_mean_ffs": ffs_object_stats["object_residual_mean"],
                "object_edge_weighted_residual_mean_native": native_object_stats["object_edge_weighted_residual_mean"],
                "object_edge_weighted_residual_mean_ffs": ffs_object_stats["object_edge_weighted_residual_mean"],
                "object_overlap_area": float(
                    np.mean(
                        [
                            native_object_stats["object_overlap_area"],
                            ffs_object_stats["object_overlap_area"],
                        ]
                    )
                ),
                "pair_object_visibility_score": pair_object_visibility_score,
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
    object_bbox = tuple(pair_info["object_bbox"]) if pair_info.get("object_bbox") is not None else None
    native_target = pair_info["images"]["native_target"]
    ffs_target = pair_info["images"]["native_target"] if same_case_mode else pair_info["images"]["ffs_target"]
    native_warp = pair_info["images"]["native_warp"]
    native_residual = pair_info["images"]["native_residual"]
    ffs_warp = pair_info["images"]["ffs_warp"]
    ffs_residual = pair_info["images"]["ffs_residual"]
    if object_bbox is not None:
        native_target = _crop_to_bbox(native_target, object_bbox)
        ffs_target = _crop_to_bbox(ffs_target, object_bbox)
        native_warp = _crop_to_bbox(native_warp, object_bbox)
        native_residual = _crop_to_bbox(native_residual, object_bbox)
        ffs_warp = _crop_to_bbox(ffs_warp, object_bbox)
        ffs_residual = _crop_to_bbox(ffs_residual, object_bbox)
    tiles = [
        label_tile(native_target, f"Native Tgt C{dst_idx}", (360, 240)),
        label_tile(native_warp, f"Native Warp ({pair_info['depth_dirs']['native']})", (360, 240)),
        label_tile(native_residual, "Native Residual", (360, 240)),
        label_tile(ffs_target, ffs_target_label, (360, 240)),
        label_tile(ffs_warp, f"FFS Warp ({pair_info['depth_dirs']['ffs']})", (360, 240)),
        label_tile(ffs_residual, "FFS Residual", (360, 240)),
    ]
    board = compose_depth_review_board(
        title_lines=[
            f"{case_label} | Truth Board",
            f"frame={frame_idx} | pair=C{src_idx} -> C{dst_idx}",
        ],
        metric_lines=[
            f"Native obj-valid={pair_info['object_warp_valid_ratio_native']:.3f} | obj-mean={pair_info['object_residual_mean_native']:.2f} | obj-edge={pair_info['object_edge_weighted_residual_mean_native']:.2f}",
            f"FFS obj-valid={pair_info['object_warp_valid_ratio_ffs']:.3f} | obj-mean={pair_info['object_residual_mean_ffs']:.2f} | obj-edge={pair_info['object_edge_weighted_residual_mean_ffs']:.2f}",
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
        scene=scene,
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
            "object_projected_area_ratio": float(selected_step["object_projected_area_ratio"]),
            "object_bbox_fill_ratio": float(selected_step["object_bbox_fill_ratio"]),
            "object_multi_camera_support_ratio": float(selected_step["object_multi_camera_support_ratio"]),
            "object_mismatch_residual_m": float(selected_step["object_mismatch_residual_m"]),
            "context_dominance_penalty": float(selected_step["context_dominance_penalty"]),
            "silhouette_penalty": float(selected_step["silhouette_penalty"]),
            "final_score": float(selected_step["final_score"]),
            "candidate_count": int(len(step_metrics)),
        },
        "truth_camera_pair": {
            "src_camera_idx": int(selected_pair["pair"][0]),
            "dst_camera_idx": int(selected_pair["pair"][1]),
            "mean_valid_ratio": float(selected_pair["mean_valid_ratio"]),
            "residual_gap": float(selected_pair["residual_gap"]),
            "object_warp_valid_ratio_native": float(selected_pair["object_warp_valid_ratio_native"]),
            "object_warp_valid_ratio_ffs": float(selected_pair["object_warp_valid_ratio_ffs"]),
            "object_residual_mean_native": float(selected_pair["object_residual_mean_native"]),
            "object_residual_mean_ffs": float(selected_pair["object_residual_mean_ffs"]),
            "object_edge_weighted_residual_mean_native": float(selected_pair["object_edge_weighted_residual_mean_native"]),
            "object_edge_weighted_residual_mean_ffs": float(selected_pair["object_edge_weighted_residual_mean_ffs"]),
            "object_overlap_area": float(selected_pair["object_overlap_area"]),
            "pair_object_visibility_score": float(selected_pair["pair_object_visibility_score"]),
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
        write_json(debug_dir / "hero_angle_candidates.json", {"steps": step_metrics, "selected": selected_step})
        serializable_pairs = []
        for item in pair_metrics:
            serializable_pairs.append(
                {
                    "pair": list(item["pair"]),
                    "mean_valid_ratio": item["mean_valid_ratio"],
                    "residual_gap": item["residual_gap"],
                    "object_warp_valid_ratio_native": item["object_warp_valid_ratio_native"],
                    "object_warp_valid_ratio_ffs": item["object_warp_valid_ratio_ffs"],
                    "object_residual_mean_native": item["object_residual_mean_native"],
                    "object_residual_mean_ffs": item["object_residual_mean_ffs"],
                    "object_edge_weighted_residual_mean_native": item["object_edge_weighted_residual_mean_native"],
                    "object_edge_weighted_residual_mean_ffs": item["object_edge_weighted_residual_mean_ffs"],
                    "object_overlap_area": item["object_overlap_area"],
                    "pair_object_visibility_score": item["pair_object_visibility_score"],
                    "native": item["native"],
                    "ffs": item["ffs"],
                    "depth_dirs": item["depth_dirs"],
                }
            )
        write_json(
            debug_dir / "truth_pair_candidates.json",
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
