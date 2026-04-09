from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .camera_frusta import build_camera_frustum_geometry, collect_camera_geometry_points, extract_camera_poses
from .object_compare import (
    build_object_first_layers,
    build_refined_pixel_masks_from_bboxes,
    filter_camera_clouds_by_pixel_masks,
    project_world_roi_to_camera_bbox,
    world_bbox_corners,
    write_compare_debug_metrics,
    write_fused_cloud_debug_artifacts,
    write_object_debug_artifacts,
)
from .object_roi import estimate_table_color_bgr
from .pointcloud_compare import (
    PROJECTION_MODES,
    RENDER_MODES,
    SCENE_CROP_MODES,
    compute_scene_crop_bounds,
    compute_view_config,
    crop_points_to_bounds,
    estimate_focus_point,
    estimate_ortho_scale,
    get_frame_count,
    load_case_frame_cloud_with_sources,
    load_case_metadata,
    project_world_points_to_image,
    render_point_cloud,
    resolve_case_dirs,
    write_gif as write_gif_animation,
    write_video,
)
from .source_compare import (
    SOURCE_CAMERA_COLORS_BGR,
    build_source_legend_image,
    render_mismatch_residual,
    render_source_attribution_overlay,
    render_source_split_images,
    write_source_legend_image,
)
from .support_compare import compute_support_count_map, overlay_support_legend, render_support_count_map, summarize_support_counts


DEFAULT_CAMERA_IDS = [0, 1, 2]
ORBIT_MODES = ("observed_hemisphere", "full_360", "camera_neighborhood")
LAYOUT_MODES = ("side_by_side_large", "camera_neighborhood_grid")


def _normalize_vector(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    vec = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-6:
        return np.asarray(fallback, dtype=np.float32)
    return vec / norm


def _compute_bounds(point_sets: list[np.ndarray], *, fallback_center: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    points = [np.asarray(item, dtype=np.float32) for item in point_sets if len(item) > 0]
    if not points:
        center = np.zeros((3,), dtype=np.float32) if fallback_center is None else np.asarray(fallback_center, dtype=np.float32)
        return center - 1.0, center + 1.0
    stacked = np.concatenate(points, axis=0)
    return stacked.min(axis=0).astype(np.float32), stacked.max(axis=0).astype(np.float32)


def _project_vector_to_plane(vector: np.ndarray, axis: np.ndarray) -> np.ndarray:
    vec = np.asarray(vector, dtype=np.float32)
    normal = _normalize_vector(axis, np.array([0.0, 0.0, 1.0], dtype=np.float32))
    return vec - normal * float(vec @ normal)


def _build_orbit_basis(
    *,
    camera_poses: list[dict[str, Any]],
    focus_point: np.ndarray,
    orbit_axis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    focus = np.asarray(focus_point, dtype=np.float32)
    axis = _normalize_vector(orbit_axis, np.array([0.0, 0.0, 1.0], dtype=np.float32))
    basis_x = None
    for pose in camera_poses:
        projected = _project_vector_to_plane(np.asarray(pose["position"], dtype=np.float32) - focus, axis)
        if float(np.linalg.norm(projected)) > 1e-6:
            basis_x = _normalize_vector(projected, np.array([1.0, 0.0, 0.0], dtype=np.float32))
            break
    if basis_x is None:
        fallback = _project_vector_to_plane(np.array([1.0, 0.0, 0.0], dtype=np.float32), axis)
        if float(np.linalg.norm(fallback)) <= 1e-6:
            fallback = _project_vector_to_plane(np.array([0.0, 1.0, 0.0], dtype=np.float32), axis)
        basis_x = _normalize_vector(fallback, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    basis_y = _normalize_vector(np.cross(axis, basis_x), np.array([0.0, 1.0, 0.0], dtype=np.float32))
    return basis_x, basis_y


def _compute_crop_corners(bounds_min: np.ndarray, bounds_max: np.ndarray) -> np.ndarray:
    min_corner = np.asarray(bounds_min, dtype=np.float32)
    max_corner = np.asarray(bounds_max, dtype=np.float32)
    corners: list[np.ndarray] = []
    for x_value in (min_corner[0], max_corner[0]):
        for y_value in (min_corner[1], max_corner[1]):
            for z_value in (min_corner[2], max_corner[2]):
                corners.append(np.array([x_value, y_value, z_value], dtype=np.float32))
    return np.stack(corners, axis=0)


def _wrap_angle_deg(angle_deg: float) -> float:
    wrapped = (float(angle_deg) + 180.0) % 360.0 - 180.0
    return wrapped


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


def select_single_frame_index(*, native_count: int, ffs_count: int, frame_idx: int) -> tuple[int, int]:
    max_index = min(int(native_count), int(ffs_count)) - 1
    selected = int(frame_idx)
    if selected < 0 or selected > max_index:
        raise ValueError(
            f"frame_idx={selected} is out of range for native_count={native_count}, ffs_count={ffs_count}. "
            f"Expected 0 <= frame_idx <= {max_index}."
        )
    return selected, selected


def resolve_single_frame_case_selection(
    *,
    aligned_root: Path,
    case_name: str | None,
    realsense_case: str | None,
    ffs_case: str | None,
    frame_idx: int,
    camera_ids: list[int] | None = None,
) -> dict[str, Any]:
    native_case_dir, ffs_case_dir, same_case_mode = resolve_case_dirs(
        aligned_root=Path(aligned_root).resolve(),
        case_name=case_name,
        realsense_case=realsense_case,
        ffs_case=ffs_case,
    )
    native_metadata = load_case_metadata(native_case_dir)
    ffs_metadata = load_case_metadata(ffs_case_dir)
    if same_case_mode and not ((native_case_dir / "depth_ffs").is_dir() or (native_case_dir / "depth_ffs_float_m").is_dir()):
        raise ValueError("Same-case comparison requires an aligned case containing depth_ffs/ or depth_ffs_float_m/.")
    if len(native_metadata["serial_numbers"]) != len(ffs_metadata["serial_numbers"]):
        raise ValueError("Native and FFS cases must have the same number of cameras.")

    selected_camera_ids = list(DEFAULT_CAMERA_IDS if camera_ids is None else camera_ids)
    if len(selected_camera_ids) != len(set(selected_camera_ids)):
        raise ValueError(f"camera_ids must be unique. Got: {selected_camera_ids}")
    max_camera_index = len(native_metadata["serial_numbers"]) - 1
    for camera_idx in selected_camera_ids:
        if int(camera_idx) < 0 or int(camera_idx) > max_camera_index:
            raise ValueError(f"camera_idx out of range: {camera_idx}")

    native_frame_idx, ffs_frame_idx = select_single_frame_index(
        native_count=get_frame_count(native_metadata),
        ffs_count=get_frame_count(ffs_metadata),
        frame_idx=frame_idx,
    )

    from .calibration_io import load_calibration_transforms

    calibration_reference_serials = native_metadata.get(
        "calibration_reference_serials",
        native_metadata["serial_numbers"],
    )
    native_c2w = load_calibration_transforms(
        native_case_dir / "calibrate.pkl",
        serial_numbers=native_metadata["serial_numbers"],
        calibration_reference_serials=calibration_reference_serials,
    )

    return {
        "aligned_root": str(Path(aligned_root).resolve()),
        "native_case_dir": native_case_dir,
        "ffs_case_dir": ffs_case_dir,
        "same_case_mode": bool(same_case_mode),
        "native_metadata": native_metadata,
        "ffs_metadata": ffs_metadata,
        "native_frame_idx": int(native_frame_idx),
        "ffs_frame_idx": int(ffs_frame_idx),
        "camera_ids": [int(item) for item in selected_camera_ids],
        "serial_numbers": list(native_metadata["serial_numbers"]),
        "native_c2w": native_c2w,
    }


def load_single_frame_compare_clouds(
    selection: dict[str, Any],
    *,
    voxel_size: float | None,
    max_points_per_camera: int | None,
    depth_min_m: float,
    depth_max_m: float,
    use_float_ffs_depth_when_available: bool,
    pixel_roi_by_camera: dict[int, tuple[int, int, int, int]] | None = None,
) -> dict[str, Any]:
    native_points, native_colors, native_stats, native_camera_clouds = load_case_frame_cloud_with_sources(
        case_dir=selection["native_case_dir"],
        metadata=selection["native_metadata"],
        frame_idx=selection["native_frame_idx"],
        depth_source="realsense",
        use_float_ffs_depth_when_available=False,
        voxel_size=voxel_size,
        pixel_roi_by_camera=pixel_roi_by_camera,
        max_points_per_camera=max_points_per_camera,
        depth_min_m=depth_min_m,
        depth_max_m=depth_max_m,
    )
    ffs_points, ffs_colors, ffs_stats, ffs_camera_clouds = load_case_frame_cloud_with_sources(
        case_dir=selection["ffs_case_dir"],
        metadata=selection["ffs_metadata"],
        frame_idx=selection["ffs_frame_idx"],
        depth_source="ffs",
        use_float_ffs_depth_when_available=use_float_ffs_depth_when_available,
        voxel_size=voxel_size,
        pixel_roi_by_camera=pixel_roi_by_camera,
        max_points_per_camera=max_points_per_camera,
        depth_min_m=depth_min_m,
        depth_max_m=depth_max_m,
    )
    return {
        "native_points": native_points,
        "native_colors": native_colors,
        "native_stats": native_stats,
        "native_camera_clouds": native_camera_clouds,
        "ffs_points": ffs_points,
        "ffs_colors": ffs_colors,
        "ffs_stats": ffs_stats,
        "ffs_camera_clouds": ffs_camera_clouds,
    }


def _json_ready_crop_bounds(crop_bounds: dict[str, Any]) -> dict[str, Any]:
    serialized: dict[str, Any] = {}
    for key, value in crop_bounds.items():
        if isinstance(value, np.ndarray):
            serialized[key] = np.asarray(value).astype(np.float32).tolist()
        elif isinstance(value, (np.floating, np.integer)):
            serialized[key] = float(value)
        elif isinstance(value, list):
            serialized[key] = value
        else:
            serialized[key] = value
    return serialized


def _write_json(output_path: Path, payload: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _source_histogram(source_camera_idx: np.ndarray) -> dict[str, int]:
    values = np.asarray(source_camera_idx, dtype=np.int16).reshape(-1)
    if len(values) == 0:
        return {}
    unique_values, counts = np.unique(values, return_counts=True)
    return {str(int(camera_idx)): int(count) for camera_idx, count in zip(unique_values, counts, strict=False)}


def _aggregate_step_metric_series(step_metrics: list[dict[str, Any]], *, key: str) -> float:
    values = [float(item[key]) for item in step_metrics if key in item]
    return float(np.mean(values)) if values else 0.0


def _derive_bbox_by_camera(
    camera_clouds: list[dict[str, Any]],
    *,
    crop_bounds: dict[str, Any],
    manual_bbox_by_camera: dict[int, tuple[int, int, int, int]] | None = None,
    object_height_max: float | None = None,
) -> tuple[dict[int, tuple[int, int, int, int]], dict[int, dict[str, Any]]]:
    bbox_by_camera: dict[int, tuple[int, int, int, int]] = {}
    debug_by_camera: dict[int, dict[str, Any]] = {}
    object_roi_min = np.asarray(crop_bounds.get("object_roi_min", crop_bounds["min"]), dtype=np.float32)
    object_roi_max = np.asarray(crop_bounds.get("object_roi_max", crop_bounds["max"]), dtype=np.float32)
    extra_world_points = None
    if object_height_max is not None and "plane_point" in crop_bounds and "plane_normal" in crop_bounds:
        plane_point = np.asarray(crop_bounds["plane_point"], dtype=np.float32)
        plane_normal = np.asarray(crop_bounds["plane_normal"], dtype=np.float32)
        corners = world_bbox_corners(object_roi_min, object_roi_max)
        current_top = float(np.max((corners - plane_point[None, :]) @ plane_normal))
        target_top = max(current_top + 0.12, float(object_height_max))
        extra_height = max(0.0, target_top - current_top)
        if extra_height > 1e-6:
            extra_world_points = corners + plane_normal[None, :] * extra_height
    for camera_cloud in camera_clouds:
        camera_idx = int(camera_cloud["camera_idx"])
        if manual_bbox_by_camera is not None and camera_idx in manual_bbox_by_camera:
            bbox = tuple(int(item) for item in manual_bbox_by_camera[camera_idx])
            bbox_by_camera[camera_idx] = bbox
            debug_by_camera[camera_idx] = {
                "camera_idx": camera_idx,
                "source": "manual_image_roi_json",
                "bbox": list(bbox),
            }
            continue
        bbox, debug = project_world_roi_to_camera_bbox(
            camera_cloud,
            object_roi_min=object_roi_min,
            object_roi_max=object_roi_max,
            extra_world_points=extra_world_points,
        )
        if bbox is not None:
            bbox_by_camera[camera_idx] = bbox
        debug_by_camera[camera_idx] = debug
    return bbox_by_camera, debug_by_camera


def _collect_seed_points(camera_clouds: list[dict[str, Any]]) -> list[np.ndarray]:
    return [np.asarray(item["points"], dtype=np.float32) for item in camera_clouds if len(item["points"]) > 0]


def _compute_seed_height_summary(
    point_sets: list[np.ndarray],
    *,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
) -> dict[str, float]:
    valid_sets = [np.asarray(item, dtype=np.float32) for item in point_sets if len(item) > 0]
    if not valid_sets:
        return {"point_count": 0.0, "max_height": 0.0, "p90_height": 0.0}
    stacked = np.concatenate(valid_sets, axis=0)
    heights = (stacked - np.asarray(plane_point, dtype=np.float32)[None, :]) @ np.asarray(plane_normal, dtype=np.float32)
    return {
        "point_count": float(len(stacked)),
        "max_height": float(np.max(heights)) if len(heights) > 0 else 0.0,
        "p90_height": float(np.quantile(heights, 0.90)) if len(heights) > 0 else 0.0,
    }


def _object_roi_volume(crop_bounds: dict[str, Any] | None) -> float:
    if crop_bounds is None:
        return 0.0
    roi_min = np.asarray(crop_bounds.get("object_roi_min", crop_bounds["min"]), dtype=np.float32)
    roi_max = np.asarray(crop_bounds.get("object_roi_max", crop_bounds["max"]), dtype=np.float32)
    extent = np.maximum(roi_max - roi_min, 1e-6)
    return float(np.prod(extent))


def _build_seed_union_crop_bounds(
    seed_points: list[np.ndarray],
    *,
    full_bounds_min: np.ndarray,
    full_bounds_max: np.ndarray,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
    crop_margin_xy: float,
    crop_min_z: float,
    crop_max_z: float,
) -> dict[str, Any] | None:
    valid_sets = [np.asarray(item, dtype=np.float32) for item in seed_points if len(item) > 0]
    if not valid_sets:
        return None
    stacked = np.concatenate(valid_sets, axis=0)
    object_roi_min = stacked.min(axis=0).astype(np.float32)
    object_roi_max = stacked.max(axis=0).astype(np.float32)
    roi_margin_xy = max(0.02, float(crop_margin_xy) * 0.45)
    roi_margin_z = max(0.015, abs(float(crop_max_z) - float(crop_min_z)) * 0.08)
    crop_min = np.maximum(
        np.asarray(full_bounds_min, dtype=np.float32),
        object_roi_min - np.array([roi_margin_xy, roi_margin_xy, roi_margin_z], dtype=np.float32),
    ).astype(np.float32)
    crop_max = np.minimum(
        np.asarray(full_bounds_max, dtype=np.float32),
        object_roi_max + np.array([roi_margin_xy, roi_margin_xy, roi_margin_z], dtype=np.float32),
    ).astype(np.float32)
    return {
        "mode": "auto_object_bbox",
        "min": crop_min,
        "max": crop_max,
        "object_roi_min": object_roi_min,
        "object_roi_max": object_roi_max,
        "plane_point": np.asarray(plane_point, dtype=np.float32),
        "plane_normal": np.asarray(plane_normal, dtype=np.float32),
        "object_point_count": int(len(stacked)),
        "component_point_count": int(len(stacked)),
        "selected_object_point_count": int(len(stacked)),
        "component_count": 1,
        "selected_component_indices": [0],
        "component_scores": [],
        "seed_bbox_used": True,
        "manual_seed_union_used": True,
        "fallback_used": False,
    }


def prepare_object_roi_refinement(
    *,
    raw_scene: dict[str, Any],
    focus_mode: str,
    scene_crop_mode: str,
    crop_margin_xy: float,
    crop_min_z: float,
    crop_max_z: float,
    manual_xyz_roi: dict[str, float] | None,
    manual_image_roi_by_camera: dict[int, tuple[int, int, int, int]] | None,
    object_height_min: float,
    object_height_max: float,
    object_component_mode: str,
    object_component_topk: int,
) -> dict[str, Any]:
    raw_bounds_min, raw_bounds_max = _compute_bounds([raw_scene["native_points"], raw_scene["ffs_points"]])
    focus_point = estimate_focus_point(
        [raw_scene["native_points"], raw_scene["ffs_points"]],
        bounds_min=raw_bounds_min,
        bounds_max=raw_bounds_max,
        focus_mode=focus_mode,
    )
    if scene_crop_mode != "auto_object_bbox":
        return {
            "focus_point": focus_point,
            "pass1_crop": None,
            "pass2_crop": None,
            "native_pass1_masks": None,
            "ffs_pass1_masks": None,
            "native_pass2_masks": None,
            "ffs_pass2_masks": None,
            "native_pass1_seed_metrics": [],
            "ffs_pass1_seed_metrics": [],
            "native_pass2_seed_metrics": [],
            "ffs_pass2_seed_metrics": [],
            "bbox_debug_by_camera": {},
            "pass1_mask_debug": {"native": {}, "ffs": {}},
            "pass2_mask_debug": {"native": {}, "ffs": {}},
            "pass1_seed_height_summary": {"point_count": 0.0, "max_height": 0.0, "p90_height": 0.0},
            "pass2_seed_height_summary": {"point_count": 0.0, "max_height": 0.0, "p90_height": 0.0},
        }

    pass1_crop = compute_scene_crop_bounds(
        [raw_scene["native_points"], raw_scene["ffs_points"]],
        focus_point=focus_point,
        scene_crop_mode=scene_crop_mode,
        crop_margin_xy=float(crop_margin_xy),
        crop_min_z=float(crop_min_z),
        crop_max_z=float(crop_max_z),
        manual_xyz_roi=manual_xyz_roi,
        object_seed_point_sets=None,
        object_height_min=float(object_height_min),
        object_height_max=float(object_height_max),
        object_component_mode=object_component_mode,
        object_component_topk=int(object_component_topk),
    )
    pass1_bbox_by_camera, pass1_bbox_debug = _derive_bbox_by_camera(
        raw_scene["native_camera_clouds"],
        crop_bounds=pass1_crop,
        manual_bbox_by_camera=manual_image_roi_by_camera,
        object_height_max=float(object_height_max),
    )
    native_pass1_masks, native_pass1_mask_debug = build_refined_pixel_masks_from_bboxes(
        raw_scene["native_camera_clouds"],
        roi_by_camera=pass1_bbox_by_camera,
        plane_point=np.asarray(pass1_crop.get("plane_point", focus_point), dtype=np.float32),
        plane_normal=np.asarray(pass1_crop.get("plane_normal", np.array([0.0, 0.0, 1.0], dtype=np.float32)), dtype=np.float32),
        object_height_min=float(object_height_min),
        object_height_max=max(0.60, float(object_height_max)),
    )
    ffs_pass1_masks, ffs_pass1_mask_debug = build_refined_pixel_masks_from_bboxes(
        raw_scene["ffs_camera_clouds"],
        roi_by_camera=pass1_bbox_by_camera,
        plane_point=np.asarray(pass1_crop.get("plane_point", focus_point), dtype=np.float32),
        plane_normal=np.asarray(pass1_crop.get("plane_normal", np.array([0.0, 0.0, 1.0], dtype=np.float32)), dtype=np.float32),
        object_height_min=float(object_height_min),
        object_height_max=max(0.60, float(object_height_max)),
    )
    native_pass1_seed_clouds, native_pass1_seed_metrics = filter_camera_clouds_by_pixel_masks(
        raw_scene["native_camera_clouds"],
        pixel_mask_by_camera=native_pass1_masks,
    )
    ffs_pass1_seed_clouds, ffs_pass1_seed_metrics = filter_camera_clouds_by_pixel_masks(
        raw_scene["ffs_camera_clouds"],
        pixel_mask_by_camera=ffs_pass1_masks,
    )
    pass1_seed_points = _collect_seed_points(native_pass1_seed_clouds) + _collect_seed_points(ffs_pass1_seed_clouds)
    if manual_image_roi_by_camera is not None:
        pass2_crop = _build_seed_union_crop_bounds(
            pass1_seed_points,
            full_bounds_min=raw_bounds_min,
            full_bounds_max=raw_bounds_max,
            plane_point=np.asarray(pass1_crop.get("plane_point", focus_point), dtype=np.float32),
            plane_normal=np.asarray(pass1_crop.get("plane_normal", np.array([0.0, 0.0, 1.0], dtype=np.float32)), dtype=np.float32),
            crop_margin_xy=float(crop_margin_xy),
            crop_min_z=float(crop_min_z),
            crop_max_z=float(crop_max_z),
        )
    else:
        pass2_crop = compute_scene_crop_bounds(
            [raw_scene["native_points"], raw_scene["ffs_points"]],
            focus_point=focus_point,
            scene_crop_mode=scene_crop_mode,
            crop_margin_xy=float(crop_margin_xy),
            crop_min_z=float(crop_min_z),
            crop_max_z=float(crop_max_z),
            manual_xyz_roi=manual_xyz_roi,
            object_seed_point_sets=pass1_seed_points,
            object_height_min=float(object_height_min),
            object_height_max=float(object_height_max),
            object_component_mode=object_component_mode,
            object_component_topk=int(object_component_topk),
        )
    if pass2_crop is None:
        pass2_crop = pass1_crop
    pass2_bbox_by_camera, pass2_bbox_debug = _derive_bbox_by_camera(
        raw_scene["native_camera_clouds"],
        crop_bounds=pass2_crop,
        manual_bbox_by_camera=None,
        object_height_max=float(object_height_max),
    )
    native_pass2_masks, native_pass2_mask_debug = build_refined_pixel_masks_from_bboxes(
        raw_scene["native_camera_clouds"],
        roi_by_camera=pass2_bbox_by_camera,
        plane_point=np.asarray(pass2_crop.get("plane_point", focus_point), dtype=np.float32),
        plane_normal=np.asarray(pass2_crop.get("plane_normal", np.array([0.0, 0.0, 1.0], dtype=np.float32)), dtype=np.float32),
        object_height_min=float(object_height_min),
        object_height_max=max(0.60, float(object_height_max)),
    )
    ffs_pass2_masks, ffs_pass2_mask_debug = build_refined_pixel_masks_from_bboxes(
        raw_scene["ffs_camera_clouds"],
        roi_by_camera=pass2_bbox_by_camera,
        plane_point=np.asarray(pass2_crop.get("plane_point", focus_point), dtype=np.float32),
        plane_normal=np.asarray(pass2_crop.get("plane_normal", np.array([0.0, 0.0, 1.0], dtype=np.float32)), dtype=np.float32),
        object_height_min=float(object_height_min),
        object_height_max=max(0.60, float(object_height_max)),
    )
    native_pass2_seed_clouds, native_pass2_seed_metrics = filter_camera_clouds_by_pixel_masks(
        raw_scene["native_camera_clouds"],
        pixel_mask_by_camera=native_pass2_masks,
    )
    ffs_pass2_seed_clouds, ffs_pass2_seed_metrics = filter_camera_clouds_by_pixel_masks(
        raw_scene["ffs_camera_clouds"],
        pixel_mask_by_camera=ffs_pass2_masks,
    )
    pass2_seed_points = _collect_seed_points(native_pass2_seed_clouds) + _collect_seed_points(ffs_pass2_seed_clouds)
    pass1_selected_count = int(pass1_crop.get("selected_object_point_count", pass1_crop.get("object_point_count", 0))) if pass1_crop is not None else 0
    pass2_selected_count = int(pass2_crop.get("selected_object_point_count", pass2_crop.get("object_point_count", 0))) if pass2_crop is not None else 0
    pass1_volume = _object_roi_volume(pass1_crop)
    pass2_volume = _object_roi_volume(pass2_crop)
    pass2_valid_camera_count = int(
        np.count_nonzero(
            [
                int(native_pass2_mask_debug.get(camera_idx, {}).get("refined_mask_pixels", 0)) > 256
                for camera_idx in pass2_bbox_by_camera
            ]
        )
    )
    pass2_refinement_valid = bool(
        pass2_selected_count >= max(64, int(round(max(pass1_selected_count, 1) * 0.02)))
        and pass2_valid_camera_count >= 2
        and pass2_volume >= max(1e-6, pass1_volume * 0.02)
    )
    final_crop = pass2_crop if pass2_refinement_valid else pass1_crop
    final_native_masks = native_pass2_masks if pass2_refinement_valid else native_pass1_masks
    final_ffs_masks = ffs_pass2_masks if pass2_refinement_valid else ffs_pass1_masks
    final_seed_metrics_native = native_pass2_seed_metrics if pass2_refinement_valid else native_pass1_seed_metrics
    final_seed_metrics_ffs = ffs_pass2_seed_metrics if pass2_refinement_valid else ffs_pass1_seed_metrics
    final_mask_debug = {
        "native": native_pass2_mask_debug if pass2_refinement_valid else native_pass1_mask_debug,
        "ffs": ffs_pass2_mask_debug if pass2_refinement_valid else ffs_pass1_mask_debug,
    }
    bbox_debug_by_camera: dict[int, dict[str, Any]] = {}
    for camera_idx in sorted(set(pass1_bbox_debug.keys()) | set(pass2_bbox_debug.keys())):
        bbox_debug_by_camera[camera_idx] = {
            "camera_idx": int(camera_idx),
            "pass1": pass1_bbox_debug.get(camera_idx),
            "pass2": pass2_bbox_debug.get(camera_idx),
        }
    return {
        "focus_point": focus_point.astype(np.float32),
        "pass1_crop": pass1_crop,
        "pass2_crop": pass2_crop,
        "native_pass1_masks": native_pass1_masks,
        "ffs_pass1_masks": ffs_pass1_masks,
        "native_pass2_masks": native_pass2_masks,
        "ffs_pass2_masks": ffs_pass2_masks,
        "native_pass1_seed_metrics": native_pass1_seed_metrics,
        "ffs_pass1_seed_metrics": ffs_pass1_seed_metrics,
        "native_pass2_seed_metrics": native_pass2_seed_metrics,
        "ffs_pass2_seed_metrics": ffs_pass2_seed_metrics,
        "pass2_seed_point_sets": pass2_seed_points,
        "final_crop": final_crop,
        "final_native_masks": final_native_masks,
        "final_ffs_masks": final_ffs_masks,
        "final_seed_metrics_native": final_seed_metrics_native,
        "final_seed_metrics_ffs": final_seed_metrics_ffs,
        "final_mask_debug": final_mask_debug,
        "pass2_refinement_valid": bool(pass2_refinement_valid),
        "pass2_valid_camera_count": int(pass2_valid_camera_count),
        "pass1_object_point_count": int(pass1_selected_count),
        "pass2_object_point_count": int(pass2_selected_count),
        "pass1_object_volume": float(pass1_volume),
        "pass2_object_volume": float(pass2_volume),
        "bbox_debug_by_camera": bbox_debug_by_camera,
        "pass1_mask_debug": {"native": native_pass1_mask_debug, "ffs": ffs_pass1_mask_debug},
        "pass2_mask_debug": {"native": native_pass2_mask_debug, "ffs": ffs_pass2_mask_debug},
        "pass1_seed_height_summary": _compute_seed_height_summary(
            pass1_seed_points,
            plane_point=np.asarray(pass1_crop.get("plane_point", focus_point), dtype=np.float32),
            plane_normal=np.asarray(pass1_crop.get("plane_normal", np.array([0.0, 0.0, 1.0], dtype=np.float32)), dtype=np.float32),
        ),
        "pass2_seed_height_summary": _compute_seed_height_summary(
            pass2_seed_points,
            plane_point=np.asarray(pass2_crop.get("plane_point", focus_point), dtype=np.float32),
            plane_normal=np.asarray(pass2_crop.get("plane_normal", np.array([0.0, 0.0, 1.0], dtype=np.float32)), dtype=np.float32),
        ),
    }


def build_single_frame_scene(
    *,
    native_points: np.ndarray,
    native_colors: np.ndarray,
    native_camera_clouds: list[dict[str, Any]],
    ffs_points: np.ndarray,
    ffs_colors: np.ndarray,
    ffs_camera_clouds: list[dict[str, Any]],
    focus_mode: str,
    scene_crop_mode: str,
    crop_margin_xy: float,
    crop_min_z: float,
    crop_max_z: float,
    manual_xyz_roi: dict[str, float] | None,
    object_seed_point_sets: list[np.ndarray] | None,
    native_pixel_mask_by_camera: dict[int, np.ndarray] | None,
    ffs_pixel_mask_by_camera: dict[int, np.ndarray] | None,
    object_height_min: float,
    object_height_max: float,
    object_component_mode: str,
    object_component_topk: int,
    context_max_points_per_camera: int | None,
    crop_bounds_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raw_bounds_min, raw_bounds_max = _compute_bounds([native_points, ffs_points])
    focus_point = estimate_focus_point(
        [native_points, ffs_points],
        bounds_min=raw_bounds_min,
        bounds_max=raw_bounds_max,
        focus_mode=focus_mode,
    )
    crop_bounds = crop_bounds_override if crop_bounds_override is not None else compute_scene_crop_bounds(
        [native_points, ffs_points],
        focus_point=focus_point,
        scene_crop_mode=scene_crop_mode,
        crop_margin_xy=float(crop_margin_xy),
        crop_min_z=float(crop_min_z),
        crop_max_z=float(crop_max_z),
        manual_xyz_roi=manual_xyz_roi,
        object_seed_point_sets=object_seed_point_sets,
        object_height_min=float(object_height_min),
        object_height_max=float(object_height_max),
        object_component_mode=object_component_mode,
        object_component_topk=int(object_component_topk),
    )
    cropped_native_points, cropped_native_colors = crop_points_to_bounds(native_points, native_colors, crop_bounds)
    cropped_ffs_points, cropped_ffs_colors = crop_points_to_bounds(ffs_points, ffs_colors, crop_bounds)
    cropped_native_camera_clouds = []
    for camera_cloud in native_camera_clouds:
        points = np.asarray(camera_cloud["points"], dtype=np.float32)
        colors = np.asarray(camera_cloud["colors"], dtype=np.uint8)
        valid = np.all(points >= np.asarray(crop_bounds["min"], dtype=np.float32)[None, :], axis=1) & np.all(
            points <= np.asarray(crop_bounds["max"], dtype=np.float32)[None, :],
            axis=1,
        )
        cropped_native_camera_clouds.append(
            {
                **camera_cloud,
                "points": points[valid],
                "colors": colors[valid],
                "source_camera_idx": np.asarray(camera_cloud["source_camera_idx"], dtype=np.int16).reshape(-1)[valid],
                "source_serial": np.asarray(camera_cloud["source_serial"], dtype=object).reshape(-1)[valid],
            }
        )
    cropped_ffs_camera_clouds = []
    for camera_cloud in ffs_camera_clouds:
        points = np.asarray(camera_cloud["points"], dtype=np.float32)
        colors = np.asarray(camera_cloud["colors"], dtype=np.uint8)
        valid = np.all(points >= np.asarray(crop_bounds["min"], dtype=np.float32)[None, :], axis=1) & np.all(
            points <= np.asarray(crop_bounds["max"], dtype=np.float32)[None, :],
            axis=1,
        )
        cropped_ffs_camera_clouds.append(
            {
                **camera_cloud,
                "points": points[valid],
                "colors": colors[valid],
                "source_camera_idx": np.asarray(camera_cloud["source_camera_idx"], dtype=np.int16).reshape(-1)[valid],
                "source_serial": np.asarray(camera_cloud["source_serial"], dtype=object).reshape(-1)[valid],
            }
        )

    cropped_point_sets = [item for item in (cropped_native_points, cropped_ffs_points) if len(item) > 0]
    bounds_min, bounds_max = _compute_bounds(
        cropped_point_sets if cropped_point_sets else [native_points, ffs_points],
        fallback_center=focus_point,
    )
    object_roi_min = np.asarray(crop_bounds.get("object_roi_min", crop_bounds["min"]), dtype=np.float32)
    object_roi_max = np.asarray(crop_bounds.get("object_roi_max", crop_bounds["max"]), dtype=np.float32)
    plane_point = np.asarray(crop_bounds.get("plane_point", focus_point), dtype=np.float32)
    plane_normal = np.asarray(
        crop_bounds.get("plane_normal", np.array([0.0, 0.0, 1.0], dtype=np.float32)),
        dtype=np.float32,
    )
    refined_focus = ((object_roi_min + object_roi_max) * 0.5).astype(np.float32) if scene_crop_mode == "auto_object_bbox" else estimate_focus_point(
        cropped_point_sets if cropped_point_sets else [native_points, ffs_points],
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        focus_mode=focus_mode,
    )
    if len(cropped_native_points) > 0:
        table_color_bgr = estimate_table_color_bgr(
            cropped_native_points,
            cropped_native_colors,
            plane_point=plane_point,
            plane_normal=plane_normal,
        )
    elif len(cropped_ffs_points) > 0:
        table_color_bgr = estimate_table_color_bgr(
            cropped_ffs_points,
            cropped_ffs_colors,
            plane_point=plane_point,
            plane_normal=plane_normal,
        )
    else:
        table_color_bgr = np.array([128.0, 128.0, 128.0], dtype=np.float32)

    object_extent = object_roi_max - object_roi_min
    render_margin_xy = float(np.clip(max(float(object_extent[0]), float(object_extent[1])) * 0.30, 0.05, 0.14))
    render_margin_z = float(np.clip(max(float(object_extent[2]), max(float(object_extent[0]), float(object_extent[1])) * 0.45), 0.08, 0.22))
    render_bounds_min = np.maximum(
        np.asarray(crop_bounds["min"], dtype=np.float32),
        object_roi_min - np.array([render_margin_xy, render_margin_xy, render_margin_z], dtype=np.float32),
    ).astype(np.float32)
    render_bounds_max = np.minimum(
        np.asarray(crop_bounds["max"], dtype=np.float32),
        object_roi_max + np.array([render_margin_xy, render_margin_xy, render_margin_z], dtype=np.float32),
    ).astype(np.float32)

    native_layers = build_object_first_layers(
        cropped_native_camera_clouds,
        object_roi_min=object_roi_min,
        object_roi_max=object_roi_max,
        plane_point=plane_point,
        plane_normal=plane_normal,
        table_color_bgr=table_color_bgr,
        object_height_min=float(object_height_min),
        object_height_max=float(object_height_max),
        context_max_points_per_camera=context_max_points_per_camera,
        pixel_mask_by_camera=native_pixel_mask_by_camera,
    )
    ffs_layers = build_object_first_layers(
        cropped_ffs_camera_clouds,
        object_roi_min=object_roi_min,
        object_roi_max=object_roi_max,
        plane_point=plane_point,
        plane_normal=plane_normal,
        table_color_bgr=table_color_bgr,
        object_height_min=float(object_height_min),
        object_height_max=float(object_height_max),
        context_max_points_per_camera=context_max_points_per_camera,
        pixel_mask_by_camera=ffs_pixel_mask_by_camera,
    )
    render_native_points = native_layers["combined_points"]
    render_native_colors = native_layers["combined_colors"]
    render_native_source_camera_idx = native_layers["combined_source_camera_idx"]
    render_ffs_points = ffs_layers["combined_points"]
    render_ffs_colors = ffs_layers["combined_colors"]
    render_ffs_source_camera_idx = ffs_layers["combined_source_camera_idx"]
    render_native_camera_clouds = native_layers["combined_camera_clouds"]
    render_ffs_camera_clouds = ffs_layers["combined_camera_clouds"]

    scalar_bounds = {
        "height": (float(render_bounds_min[2]), float(render_bounds_max[2])),
        "depth": (0.0, max(float(np.linalg.norm(render_bounds_max - render_bounds_min)) * 2.0, 0.6)),
    }
    return {
        "native_points": cropped_native_points,
        "native_colors": cropped_native_colors,
        "native_camera_clouds": cropped_native_camera_clouds,
        "native_render_points": render_native_points,
        "native_render_colors": render_native_colors,
        "native_render_source_camera_idx": render_native_source_camera_idx,
        "native_render_camera_clouds": render_native_camera_clouds,
        "native_object_points": native_layers["object_points"],
        "native_object_colors": native_layers["object_colors"],
        "native_object_source_camera_idx": native_layers["object_source_camera_idx"],
        "native_context_points": native_layers["context_points"],
        "native_context_colors": native_layers["context_colors"],
        "native_context_source_camera_idx": native_layers["context_source_camera_idx"],
        "native_object_camera_clouds": native_layers["object_camera_clouds"],
        "native_context_camera_clouds": native_layers["context_camera_clouds"],
        "native_debug_metrics": native_layers["per_camera_metrics"],
        "ffs_points": cropped_ffs_points,
        "ffs_colors": cropped_ffs_colors,
        "ffs_camera_clouds": cropped_ffs_camera_clouds,
        "ffs_render_points": render_ffs_points,
        "ffs_render_colors": render_ffs_colors,
        "ffs_render_source_camera_idx": render_ffs_source_camera_idx,
        "ffs_render_camera_clouds": render_ffs_camera_clouds,
        "ffs_object_points": ffs_layers["object_points"],
        "ffs_object_colors": ffs_layers["object_colors"],
        "ffs_object_source_camera_idx": ffs_layers["object_source_camera_idx"],
        "ffs_context_points": ffs_layers["context_points"],
        "ffs_context_colors": ffs_layers["context_colors"],
        "ffs_context_source_camera_idx": ffs_layers["context_source_camera_idx"],
        "ffs_object_camera_clouds": ffs_layers["object_camera_clouds"],
        "ffs_context_camera_clouds": ffs_layers["context_camera_clouds"],
        "ffs_debug_metrics": ffs_layers["per_camera_metrics"],
        "focus_point": refined_focus.astype(np.float32),
        "crop_bounds": {
            "min": np.asarray(crop_bounds["min"], dtype=np.float32),
            "max": np.asarray(crop_bounds["max"], dtype=np.float32),
        },
        "object_roi_bounds": {"min": object_roi_min, "max": object_roi_max},
        "plane_point": plane_point,
        "plane_normal": plane_normal,
        "bounds_min": bounds_min,
        "bounds_max": bounds_max,
        "render_bounds_min": render_bounds_min,
        "render_bounds_max": render_bounds_max,
        "scalar_bounds": scalar_bounds,
        "crop_metadata": {
            key: value
            for key, value in crop_bounds.items()
            if key not in ("min", "max", "object_roi_min", "object_roi_max", "plane_point", "plane_normal")
        },
    }


def estimate_orbit_axis(camera_poses: list[dict[str, Any]]) -> np.ndarray:
    if not camera_poses:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    up_stack = np.stack([np.asarray(pose["up"], dtype=np.float32) for pose in camera_poses], axis=0)
    axis = up_stack.mean(axis=0)
    return _normalize_vector(axis, np.array([0.0, 0.0, 1.0], dtype=np.float32))


def rotate_vector_around_axis(vector: np.ndarray, axis: np.ndarray, angle_deg: float) -> np.ndarray:
    vec = np.asarray(vector, dtype=np.float32)
    axis = _normalize_vector(axis, np.array([0.0, 0.0, 1.0], dtype=np.float32))
    theta = np.deg2rad(float(angle_deg))
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))
    cross = np.cross(axis, vec)
    dot = float(axis @ vec)
    return (vec * cos_t + cross * sin_t + axis * dot * (1.0 - cos_t)).astype(np.float32)


def generate_orbit_angles(*, num_orbit_steps: int, orbit_degrees: float) -> list[float]:
    step_count = max(1, int(num_orbit_steps))
    total_degrees = abs(float(orbit_degrees))
    if step_count == 1 or total_degrees <= 1e-6:
        return [0.0]
    if total_degrees >= 359.5:
        return [float(angle) for angle in np.linspace(0.0, total_degrees, step_count, endpoint=False)]
    half_sweep = total_degrees * 0.5
    return [float(angle) for angle in np.linspace(-half_sweep, half_sweep, step_count)]


def build_camera_anchored_orbit_views(
    *,
    camera_poses: list[dict[str, Any]],
    focus_point: np.ndarray,
    orbit_axis: np.ndarray,
    num_orbit_steps: int,
    orbit_degrees: float,
) -> list[dict[str, Any]]:
    focus = np.asarray(focus_point, dtype=np.float32)
    orbit_steps: list[dict[str, Any]] = []
    for step_idx, angle_deg in enumerate(generate_orbit_angles(num_orbit_steps=num_orbit_steps, orbit_degrees=orbit_degrees)):
        view_configs: list[dict[str, Any]] = []
        for pose in camera_poses:
            anchor_offset = np.asarray(pose["position"], dtype=np.float32) - focus
            if float(np.linalg.norm(anchor_offset)) <= 1e-6:
                anchor_offset = np.asarray(pose["forward"], dtype=np.float32) * -1.0
            rotated_offset = rotate_vector_around_axis(anchor_offset, orbit_axis, angle_deg)
            rotated_up = rotate_vector_around_axis(np.asarray(pose["up"], dtype=np.float32), orbit_axis, angle_deg)
            view_configs.append(
                {
                    "view_name": f"cam{pose['camera_idx']}_step{step_idx:03d}",
                    "label": f"Near Cam{pose['camera_idx']}",
                    "camera_idx": int(pose["camera_idx"]),
                    "serial": pose["serial"],
                    "angle_deg": float(angle_deg),
                    "center": focus.copy(),
                    "camera_position": (focus + rotated_offset).astype(np.float32),
                    "anchor_camera_position": np.asarray(pose["position"], dtype=np.float32),
                    "up": _normalize_vector(rotated_up, orbit_axis),
                    "radius": float(np.linalg.norm(rotated_offset)),
                    "orbit_axis": np.asarray(orbit_axis, dtype=np.float32),
                    "orbit_angle_deg": float(angle_deg),
                    "color_bgr": tuple(int(channel) for channel in pose["color_bgr"]),
                }
            )
        orbit_steps.append(
            {
                "step_idx": int(step_idx),
                "angle_deg": float(angle_deg),
                "view_configs": view_configs,
            }
        )
    return orbit_steps


def compute_camera_azimuths_deg(
    *,
    camera_poses: list[dict[str, Any]],
    focus_point: np.ndarray,
    orbit_axis: np.ndarray,
    basis_x: np.ndarray | None = None,
    basis_y: np.ndarray | None = None,
) -> dict[int, float]:
    focus = np.asarray(focus_point, dtype=np.float32)
    axis = _normalize_vector(orbit_axis, np.array([0.0, 0.0, 1.0], dtype=np.float32))
    if basis_x is None or basis_y is None:
        basis_x, basis_y = _build_orbit_basis(
            camera_poses=camera_poses,
            focus_point=focus,
            orbit_axis=axis,
        )
    azimuths: dict[int, float] = {}
    for pose in camera_poses:
        offset = np.asarray(pose["position"], dtype=np.float32) - focus
        planar = _project_vector_to_plane(offset, axis)
        if float(np.linalg.norm(planar)) <= 1e-6:
            continue
        azimuths[int(pose["camera_idx"])] = float(
            np.rad2deg(np.arctan2(float(planar @ basis_y), float(planar @ basis_x)))
        )
    return azimuths


def estimate_supported_coverage_arc(
    camera_azimuths_deg: dict[int, float],
    *,
    coverage_margin_deg: float,
) -> dict[str, Any]:
    if not camera_azimuths_deg:
        return {
            "start_deg": 0.0,
            "end_deg": 360.0,
            "span_deg": 360.0,
            "largest_gap_deg": 0.0,
            "largest_gap_center_deg": 180.0,
            "camera_azimuths_deg": {},
        }
    angles = sorted(((float(angle) % 360.0) + 360.0) % 360.0 for angle in camera_azimuths_deg.values())
    if len(angles) == 1:
        center = angles[0]
        span = min(360.0, max(90.0, float(coverage_margin_deg) * 2.0 + 90.0))
        start = center - span * 0.5
        end = center + span * 0.5
        return {
            "start_deg": float(start),
            "end_deg": float(end),
            "span_deg": float(span),
            "largest_gap_deg": float(360.0 - span),
            "largest_gap_center_deg": float(center + 180.0),
            "camera_azimuths_deg": camera_azimuths_deg,
        }

    extended = angles + [angles[0] + 360.0]
    gaps = [extended[idx + 1] - extended[idx] for idx in range(len(angles))]
    largest_gap_idx = int(np.argmax(gaps))
    largest_gap = float(gaps[largest_gap_idx])
    supported_start = float(extended[largest_gap_idx + 1] - float(coverage_margin_deg))
    supported_end = float(extended[largest_gap_idx] + 360.0 + float(coverage_margin_deg))
    supported_span = min(360.0, supported_end - supported_start)
    largest_gap_center = float(extended[largest_gap_idx] + largest_gap * 0.5)
    return {
        "start_deg": supported_start,
        "end_deg": supported_start + supported_span,
        "span_deg": supported_span,
        "largest_gap_deg": largest_gap,
        "largest_gap_center_deg": largest_gap_center,
        "camera_azimuths_deg": camera_azimuths_deg,
    }


def angle_is_supported(angle_deg: float, coverage_arc: dict[str, Any]) -> bool:
    relative = (float(angle_deg) - float(coverage_arc["start_deg"])) % 360.0
    return relative <= float(coverage_arc["span_deg"]) + 1e-6


def build_object_centered_orbit_views(
    *,
    camera_poses: list[dict[str, Any]],
    focus_point: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    orbit_axis: np.ndarray,
    num_orbit_steps: int,
    orbit_degrees: float,
    orbit_radius_scale: float,
    view_height_offset: float,
    orbit_mode: str,
    coverage_margin_deg: float,
    show_unsupported_warning: bool,
) -> dict[str, Any]:
    focus = np.asarray(focus_point, dtype=np.float32)
    axis = _normalize_vector(orbit_axis, np.array([0.0, 0.0, 1.0], dtype=np.float32))
    basis_x, basis_y = _build_orbit_basis(
        camera_poses=camera_poses,
        focus_point=focus,
        orbit_axis=axis,
    )

    crop_corners = _compute_crop_corners(bounds_min, bounds_max)
    crop_offsets = crop_corners - focus[None, :]
    crop_planar_x = crop_offsets @ basis_x
    crop_planar_y = crop_offsets @ basis_y
    crop_axis = crop_offsets @ axis
    crop_planar_radius = max(
        float(np.max(np.abs(crop_planar_x))),
        float(np.max(np.abs(crop_planar_y))),
        1e-3,
    )
    crop_axis_extent = max(float(np.max(np.abs(crop_axis))), 1e-3)

    camera_axis_heights: list[float] = []
    camera_reference_azimuths_deg: dict[int, float] = {}
    for pose in camera_poses:
        offset = np.asarray(pose["position"], dtype=np.float32) - focus
        planar = _project_vector_to_plane(offset, axis)
        camera_axis_heights.append(float(offset @ axis))
    camera_reference_azimuths_deg = compute_camera_azimuths_deg(
        camera_poses=camera_poses,
        focus_point=focus,
        orbit_axis=axis,
        basis_x=basis_x,
        basis_y=basis_y,
    )

    orbit_radius = max(crop_planar_radius * float(orbit_radius_scale), 0.25)
    median_camera_height = float(np.median(camera_axis_heights)) if camera_axis_heights else crop_axis_extent * 1.4
    base_object_height = max(crop_axis_extent * 0.85, 0.12)
    camera_guided_height = min(max(median_camera_height * 0.45, base_object_height), orbit_radius * 0.55)
    orbit_height = max(camera_guided_height + crop_axis_extent * float(view_height_offset), 0.10)

    start_azimuth_deg = camera_reference_azimuths_deg.get(0)
    if start_azimuth_deg is None and camera_reference_azimuths_deg:
        start_azimuth_deg = float(next(iter(camera_reference_azimuths_deg.values())))
    if start_azimuth_deg is None:
        start_azimuth_deg = 0.0

    coverage_arc = estimate_supported_coverage_arc(
        camera_reference_azimuths_deg,
        coverage_margin_deg=coverage_margin_deg,
    )

    if orbit_mode == "observed_hemisphere":
        span_deg = max(5.0, float(coverage_arc["span_deg"]))
        azimuth_sequence = [
            float(coverage_arc["start_deg"] + item)
            for item in np.linspace(0.0, span_deg, max(1, int(num_orbit_steps)), endpoint=True)
        ]
        if len(azimuth_sequence) > 1:
            start_idx = int(
                np.argmin([abs(_wrap_angle_deg(angle_deg - float(start_azimuth_deg))) for angle_deg in azimuth_sequence])
            )
            azimuth_sequence = azimuth_sequence[start_idx:] + azimuth_sequence[:start_idx]
    else:
        azimuth_sequence = [float(start_azimuth_deg + item) for item in generate_orbit_angles(num_orbit_steps=num_orbit_steps, orbit_degrees=orbit_degrees)]

    orbit_steps: list[dict[str, Any]] = []
    orbit_path_points: list[np.ndarray] = []
    camera_azimuths = {
        camera_idx: angle_deg
        for camera_idx, angle_deg in camera_reference_azimuths_deg.items()
    }
    orbit_supported_mask: list[bool] = []
    display_angle_origin_deg = float(coverage_arc["start_deg"] if orbit_mode == "observed_hemisphere" else start_azimuth_deg)
    for step_idx, azimuth_deg in enumerate(azimuth_sequence):
        azimuth_rad = np.deg2rad(azimuth_deg)
        planar_offset = basis_x * (np.cos(azimuth_rad) * orbit_radius) + basis_y * (np.sin(azimuth_rad) * orbit_radius)
        camera_position = (focus + planar_offset + axis * orbit_height).astype(np.float32)
        current_angle_deg = float(np.rad2deg(np.arctan2(float(planar_offset @ basis_y), float(planar_offset @ basis_x))))
        is_supported = angle_is_supported(current_angle_deg, coverage_arc)
        if orbit_mode == "observed_hemisphere":
            is_supported = True
        display_angle_deg = float(azimuth_deg - display_angle_origin_deg)
        nearest_camera_idx = None
        nearest_camera_delta_deg = None
        for camera_idx, camera_angle_deg in camera_azimuths.items():
            delta_deg = abs(_wrap_angle_deg(current_angle_deg - float(camera_angle_deg)))
            if nearest_camera_delta_deg is None or delta_deg < nearest_camera_delta_deg:
                nearest_camera_idx = int(camera_idx)
                nearest_camera_delta_deg = float(delta_deg)
        view_config = {
            "view_name": f"orbit_step{step_idx:03d}",
            "label": "Object-Centered Orbit",
            "camera_idx": nearest_camera_idx,
            "serial": None,
            "angle_deg": display_angle_deg,
            "azimuth_deg": float(current_angle_deg),
            "center": focus.copy(),
            "camera_position": camera_position,
            "anchor_camera_position": focus.copy(),
            "up": axis.copy(),
            "radius": float(np.linalg.norm(camera_position - focus)),
            "orbit_axis": axis.copy(),
            "orbit_angle_deg": display_angle_deg,
            "color_bgr": (255, 255, 255),
            "nearest_camera_idx": nearest_camera_idx,
            "nearest_camera_delta_deg": nearest_camera_delta_deg,
            "is_supported": bool(is_supported),
            "warning_text": (
                "Unsupported backside view"
                if orbit_mode == "full_360" and show_unsupported_warning and not is_supported
                else None
            ),
        }
        orbit_steps.append(
            {
                "step_idx": int(step_idx),
                "angle_deg": display_angle_deg,
                "view_config": view_config,
                "view_configs": [view_config],
            }
        )
        orbit_path_points.append(camera_position)
        orbit_supported_mask.append(bool(is_supported))

    orbit_path = np.stack(orbit_path_points, axis=0).astype(np.float32) if orbit_path_points else np.empty((0, 3), dtype=np.float32)
    return {
        "orbit_steps": orbit_steps,
        "orbit_path": orbit_path,
        "orbit_supported_mask": orbit_supported_mask,
        "orbit_axis": axis,
        "orbit_basis_x": basis_x,
        "orbit_basis_y": basis_y,
        "orbit_radius": float(orbit_radius),
        "orbit_height": float(orbit_height),
        "start_azimuth_deg": float(start_azimuth_deg),
        "camera_reference_azimuths_deg": camera_azimuths,
        "coverage_arc": coverage_arc,
    }


def _draw_text_box(
    image: np.ndarray,
    *,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    font_scale: float = 0.58,
    thickness: int = 2,
) -> None:
    text_size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x0 = int(origin[0])
    y0 = int(origin[1])
    top_left = (max(0, x0 - 4), max(0, y0 - text_size[1] - 6))
    bottom_right = (min(image.shape[1] - 1, x0 + text_size[0] + 4), min(image.shape[0] - 1, y0 + baseline + 4))
    cv2.rectangle(image, top_left, bottom_right, (0, 0, 0), -1)
    cv2.putText(
        image,
        text,
        (x0, y0),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def build_render_output_specs(
    *,
    geom_render_mode: str,
    render_both_modes: bool,
) -> list[dict[str, Any]]:
    outputs = [
        {
            "name": "geom",
            "render_mode": str(geom_render_mode),
            "video_name": "orbit_compare_geom.mp4",
            "gif_name": "orbit_compare_geom.gif",
            "sheet_name": "turntable_keyframes_geom.png",
            "frames_dir_name": "frames_geom",
        }
    ]
    if render_both_modes:
        outputs.append(
            {
                "name": "rgb",
                "render_mode": "color_by_rgb",
                "video_name": "orbit_compare_rgb.mp4",
                "gif_name": "orbit_compare_rgb.gif",
                "sheet_name": "turntable_keyframes_rgb.png",
                "frames_dir_name": "frames_rgb",
            }
        )
    outputs.append(
        {
            "name": "support",
            "render_mode": "support_count",
            "video_name": "orbit_compare_support.mp4",
            "gif_name": "orbit_compare_support.gif",
            "sheet_name": "turntable_keyframes_support.png",
            "frames_dir_name": "frames_support",
            "write_video": True,
            "write_gif": True,
        }
    )
    outputs.append(
        {
            "name": "source",
            "render_mode": "source_attribution_alpha",
            "video_name": "orbit_compare_source.mp4",
            "gif_name": "orbit_compare_source.gif",
            "sheet_name": "turntable_keyframes_source.png",
            "frames_dir_name": "frames_source",
            "write_video": True,
            "write_gif": True,
        }
    )
    outputs.append(
        {
            "name": "mismatch",
            "render_mode": "mismatch_residual",
            "video_name": "orbit_compare_mismatch.mp4",
            "gif_name": "orbit_compare_mismatch.gif",
            "sheet_name": "turntable_keyframes_mismatch.png",
            "frames_dir_name": "frames_mismatch",
            "write_video": True,
            "write_gif": True,
        }
    )
    return outputs


def _build_overview_display_basis(
    camera_geometries: list[dict[str, Any]],
    focus_point: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    focus = np.asarray(focus_point, dtype=np.float32)
    if camera_geometries:
        camera_positions = np.stack([np.asarray(item["position"], dtype=np.float32) for item in camera_geometries], axis=0)
        rig_direction = camera_positions.mean(axis=0) - focus
    else:
        rig_direction = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    basis_z = _normalize_vector(rig_direction, np.array([0.0, 0.0, 1.0], dtype=np.float32))

    candidate = None
    if len(camera_geometries) >= 2:
        camera_positions = np.stack([np.asarray(item["position"], dtype=np.float32) for item in camera_geometries], axis=0)
        for idx in range(len(camera_positions)):
            for jdx in range(idx + 1, len(camera_positions)):
                vector = camera_positions[jdx] - camera_positions[idx]
                projected = _project_vector_to_plane(vector, basis_z)
                if float(np.linalg.norm(projected)) > 1e-4:
                    candidate = projected
                    break
            if candidate is not None:
                break
    if candidate is None:
        candidate = _project_vector_to_plane(np.array([1.0, 0.0, 0.0], dtype=np.float32), basis_z)
        if float(np.linalg.norm(candidate)) <= 1e-6:
            candidate = _project_vector_to_plane(np.array([0.0, 1.0, 0.0], dtype=np.float32), basis_z)
    basis_x = _normalize_vector(candidate, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    basis_y = _normalize_vector(np.cross(basis_z, basis_x), np.array([0.0, 1.0, 0.0], dtype=np.float32))
    return basis_x, basis_y, basis_z


def _transform_points_to_display_frame(
    points: np.ndarray,
    *,
    origin: np.ndarray,
    basis_x: np.ndarray,
    basis_y: np.ndarray,
    basis_z: np.ndarray,
) -> np.ndarray:
    cloud = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if len(cloud) == 0:
        return np.empty((0, 3), dtype=np.float32)
    centered = cloud - np.asarray(origin, dtype=np.float32)[None, :]
    transformed = np.stack(
        [
            centered @ np.asarray(basis_x, dtype=np.float32),
            centered @ np.asarray(basis_y, dtype=np.float32),
            centered @ np.asarray(basis_z, dtype=np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    return transformed


def _transform_camera_geometries_to_display_frame(
    camera_geometries: list[dict[str, Any]],
    *,
    origin: np.ndarray,
    basis_x: np.ndarray,
    basis_y: np.ndarray,
    basis_z: np.ndarray,
) -> list[dict[str, Any]]:
    transformed_geometries: list[dict[str, Any]] = []
    for geometry in camera_geometries:
        transformed_segments = []
        for start, end in geometry["segments"]:
            transformed_pair = _transform_points_to_display_frame(
                np.stack([start, end], axis=0),
                origin=origin,
                basis_x=basis_x,
                basis_y=basis_y,
                basis_z=basis_z,
            )
            transformed_segments.append((transformed_pair[0], transformed_pair[1]))
        transformed_geometry = {
            **geometry,
            "position": _transform_points_to_display_frame(
                np.asarray(geometry["position"], dtype=np.float32).reshape(1, 3),
                origin=origin,
                basis_x=basis_x,
                basis_y=basis_y,
                basis_z=basis_z,
            )[0],
            "forward_tip": _transform_points_to_display_frame(
                np.asarray(geometry["forward_tip"], dtype=np.float32).reshape(1, 3),
                origin=origin,
                basis_x=basis_x,
                basis_y=basis_y,
                basis_z=basis_z,
            )[0],
            "label_anchor": _transform_points_to_display_frame(
                np.asarray(geometry["label_anchor"], dtype=np.float32).reshape(1, 3),
                origin=origin,
                basis_x=basis_x,
                basis_y=basis_y,
                basis_z=basis_z,
            )[0],
            "frustum_corners": _transform_points_to_display_frame(
                np.asarray(geometry["frustum_corners"], dtype=np.float32),
                origin=origin,
                basis_x=basis_x,
                basis_y=basis_y,
                basis_z=basis_z,
            ),
            "segments": transformed_segments,
        }
        transformed_geometries.append(transformed_geometry)
    return transformed_geometries


def _transform_views_to_display_frame(
    current_views: list[dict[str, Any]],
    *,
    origin: np.ndarray,
    basis_x: np.ndarray,
    basis_y: np.ndarray,
    basis_z: np.ndarray,
) -> list[dict[str, Any]]:
    transformed_views: list[dict[str, Any]] = []
    for current_view in current_views:
        transformed = dict(current_view)
        for key in ("camera_position", "anchor_camera_position", "center", "original_camera_position"):
            if key in transformed and transformed[key] is not None:
                transformed[key] = _transform_points_to_display_frame(
                    np.asarray(transformed[key], dtype=np.float32).reshape(1, 3),
                    origin=origin,
                    basis_x=basis_x,
                    basis_y=basis_y,
                    basis_z=basis_z,
                )[0]
        if "orbit_axis" in transformed and transformed["orbit_axis"] is not None:
            axis_point = _transform_points_to_display_frame(
                np.asarray(origin, dtype=np.float32).reshape(1, 3) + np.asarray(transformed["orbit_axis"], dtype=np.float32).reshape(1, 3),
                origin=origin,
                basis_x=basis_x,
                basis_y=basis_y,
                basis_z=basis_z,
            )[0]
            transformed["orbit_axis"] = axis_point
        transformed_views.append(transformed)
    return transformed_views


def _transform_crop_bounds_to_display_frame(
    crop_bounds: dict[str, np.ndarray] | None,
    *,
    origin: np.ndarray,
    basis_x: np.ndarray,
    basis_y: np.ndarray,
    basis_z: np.ndarray,
) -> dict[str, np.ndarray] | None:
    if crop_bounds is None:
        return None
    corners = _compute_crop_corners(crop_bounds["min"], crop_bounds["max"])
    transformed = _transform_points_to_display_frame(
        corners,
        origin=origin,
        basis_x=basis_x,
        basis_y=basis_y,
        basis_z=basis_z,
    )
    return {
        "min": transformed.min(axis=0).astype(np.float32),
        "max": transformed.max(axis=0).astype(np.float32),
    }


def _build_overview_triptych_layout(
    pane_images: list[np.ndarray],
    *,
    pane_labels: list[str],
) -> np.ndarray:
    if not pane_images:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    pane_h, pane_w = pane_images[0].shape[:2]
    gap = 12
    header_h = 34
    canvas = np.zeros((header_h + pane_h, pane_w * len(pane_images) + gap * (len(pane_images) - 1), 3), dtype=np.uint8)
    canvas[:] = (18, 18, 22)
    for pane_idx, (pane_image, pane_label) in enumerate(zip(pane_images, pane_labels, strict=False)):
        x0 = pane_idx * (pane_w + gap)
        canvas[header_h:header_h + pane_h, x0:x0 + pane_w] = pane_image
        cv2.rectangle(canvas, (x0, header_h), (x0 + pane_w - 1, header_h + pane_h - 1), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, pane_label, (x0 + 8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def _fit_image_to_canvas(
    image: np.ndarray,
    *,
    canvas_size: tuple[int, int],
    background_bgr: tuple[int, int, int] = (18, 18, 22),
) -> np.ndarray:
    target_w, target_h = [max(1, int(item)) for item in canvas_size]
    source = np.asarray(image, dtype=np.uint8)
    src_h, src_w = source.shape[:2]
    if src_h <= 0 or src_w <= 0:
        return np.full((target_h, target_w, 3), background_bgr, dtype=np.uint8)
    scale = min(float(target_w) / float(src_w), float(target_h) / float(src_h))
    fit_w = max(1, int(round(src_w * scale)))
    fit_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(source, (fit_w, fit_h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
    canvas = np.full((target_h, target_w, 3), background_bgr, dtype=np.uint8)
    x0 = (target_w - fit_w) // 2
    y0 = (target_h - fit_h) // 2
    canvas[y0:y0 + fit_h, x0:x0 + fit_w] = resized
    return canvas


def _make_overview_pane_configs(bounds_min: np.ndarray, bounds_max: np.ndarray) -> list[tuple[str, dict[str, Any]]]:
    center = ((np.asarray(bounds_min, dtype=np.float32) + np.asarray(bounds_max, dtype=np.float32)) * 0.5).astype(np.float32)
    radius = max(1e-3, float(np.linalg.norm(np.asarray(bounds_max, dtype=np.float32) - np.asarray(bounds_min, dtype=np.float32))))
    distance = radius * 2.0
    return [
        (
            "Top",
            {
                "center": center.copy(),
                "camera_position": (center + np.array([0.0, 0.0, distance], dtype=np.float32)).astype(np.float32),
                "up": np.array([0.0, 1.0, 0.0], dtype=np.float32),
            },
        ),
        (
            "Front",
            {
                "center": center.copy(),
                "camera_position": (center + np.array([0.0, -distance, 0.0], dtype=np.float32)).astype(np.float32),
                "up": np.array([0.0, 0.0, 1.0], dtype=np.float32),
            },
        ),
        (
            "Side",
            {
                "center": center.copy(),
                "camera_position": (center + np.array([distance, 0.0, 0.0], dtype=np.float32)).astype(np.float32),
                "up": np.array([0.0, 0.0, 1.0], dtype=np.float32),
            },
        ),
    ]


def draw_scene_overlays(
    image: np.ndarray,
    *,
    camera_geometries: list[dict[str, Any]],
    view_config: dict[str, Any],
    projection_mode: str,
    ortho_scale: float | None,
    focus_point: np.ndarray | None = None,
    current_views: list[dict[str, Any]] | None = None,
    orbit_path_points: np.ndarray | None = None,
    orbit_path_supported: list[bool] | None = None,
    crop_bounds: dict[str, np.ndarray] | None = None,
    angle_label: str | None = None,
    supported_arc_label: str | None = None,
) -> np.ndarray:
    canvas = np.asarray(image, dtype=np.uint8).copy()
    height, width = canvas.shape[:2]

    if orbit_path_points is not None and len(orbit_path_points) >= 2:
        projected_path = project_world_points_to_image(
            np.asarray(orbit_path_points, dtype=np.float32),
            view_config=view_config,
            width=width,
            height=height,
            projection_mode=projection_mode,
            ortho_scale=ortho_scale,
        )
        path_uv = np.rint(projected_path["uv"]).astype(np.int32)
        if orbit_path_supported is None:
            orbit_path_supported = [True] * len(orbit_path_points)
        for idx in range(len(orbit_path_points)):
            next_idx = (idx + 1) % len(orbit_path_points)
            if not bool(projected_path["valid"][idx] and projected_path["valid"][next_idx]):
                continue
            color = (80, 210, 120) if orbit_path_supported[idx] and orbit_path_supported[next_idx] else (90, 90, 220)
            cv2.line(
                canvas,
                tuple(path_uv[idx]),
                tuple(path_uv[next_idx]),
                color,
                2,
                cv2.LINE_AA,
            )

    if crop_bounds is not None:
        corners = _compute_crop_corners(crop_bounds["min"], crop_bounds["max"])
        projected_box = project_world_points_to_image(
            corners,
            view_config=view_config,
            width=width,
            height=height,
            projection_mode=projection_mode,
            ortho_scale=ortho_scale,
        )
        edges = (
            (0, 1), (0, 2), (0, 4),
            (1, 3), (1, 5),
            (2, 3), (2, 6),
            (3, 7),
            (4, 5), (4, 6),
            (5, 7),
            (6, 7),
        )
        for start_idx, end_idx in edges:
            if not bool(projected_box["valid"][start_idx] and projected_box["valid"][end_idx]):
                continue
            pt0 = tuple(np.rint(projected_box["uv"][start_idx]).astype(np.int32))
            pt1 = tuple(np.rint(projected_box["uv"][end_idx]).astype(np.int32))
            cv2.line(canvas, pt0, pt1, (240, 200, 80), 1, cv2.LINE_AA)

    for geometry in camera_geometries:
        for start, end in geometry["segments"]:
            projected = project_world_points_to_image(
                np.stack([start, end], axis=0),
                view_config=view_config,
                width=width,
                height=height,
                projection_mode=projection_mode,
                ortho_scale=ortho_scale,
            )
            if not bool(projected["valid"][0] and projected["valid"][1]):
                continue
            pt0 = tuple(np.rint(projected["uv"][0]).astype(np.int32))
            pt1 = tuple(np.rint(projected["uv"][1]).astype(np.int32))
            cv2.line(canvas, pt0, pt1, geometry["color_bgr"], 2, cv2.LINE_AA)

        projected_pose = project_world_points_to_image(
            np.stack([geometry["position"], geometry["label_anchor"]], axis=0),
            view_config=view_config,
            width=width,
            height=height,
            projection_mode=projection_mode,
            ortho_scale=ortho_scale,
        )
        if bool(projected_pose["valid"][0]):
            center = tuple(np.rint(projected_pose["uv"][0]).astype(np.int32))
            cv2.circle(canvas, center, 6, geometry["color_bgr"], -1, cv2.LINE_AA)
        if bool(projected_pose["valid"][1]):
            label_anchor = tuple(np.rint(projected_pose["uv"][1]).astype(np.int32))
            _draw_text_box(
                canvas,
                text=geometry["label"],
                origin=(label_anchor[0], label_anchor[1]),
                color=geometry["color_bgr"],
                font_scale=0.56,
            )

    if focus_point is not None:
        projected_focus = project_world_points_to_image(
            np.asarray(focus_point, dtype=np.float32).reshape(1, 3),
            view_config=view_config,
            width=width,
            height=height,
            projection_mode=projection_mode,
            ortho_scale=ortho_scale,
        )
        if bool(projected_focus["valid"][0]):
            focus_uv = tuple(np.rint(projected_focus["uv"][0]).astype(np.int32))
            cv2.drawMarker(canvas, focus_uv, (255, 255, 255), cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
            _draw_text_box(canvas, text="ROI", origin=(focus_uv[0] + 8, focus_uv[1] - 8), color=(255, 255, 255), font_scale=0.52)

    if current_views:
        for current_view in current_views:
            anchor_point = np.asarray(
                current_view.get(
                    "anchor_camera_position",
                    focus_point if focus_point is not None else current_view["center"],
                ),
                dtype=np.float32,
            )
            overlay_points = np.stack([anchor_point, np.asarray(current_view["camera_position"], dtype=np.float32)], axis=0)
            projected = project_world_points_to_image(
                overlay_points,
                view_config=view_config,
                width=width,
                height=height,
                projection_mode=projection_mode,
                ortho_scale=ortho_scale,
            )
            if bool(projected["valid"][0] and projected["valid"][1]):
                anchor_uv = tuple(np.rint(projected["uv"][0]).astype(np.int32))
                eye_uv = tuple(np.rint(projected["uv"][1]).astype(np.int32))
                cv2.line(canvas, anchor_uv, eye_uv, current_view["color_bgr"], 2, cv2.LINE_AA)
                cv2.circle(canvas, eye_uv, 6, current_view["color_bgr"], 2, cv2.LINE_AA)
                _draw_text_box(
                    canvas,
                    text=(
                        f"Orbit | near Cam{current_view['nearest_camera_idx']}"
                        if current_view.get("nearest_camera_idx") is not None
                        else "Orbit"
                    ),
                    origin=(eye_uv[0] + 8, eye_uv[1] - 8),
                    color=current_view["color_bgr"],
                    font_scale=0.50,
                    thickness=1,
                )
                if current_view.get("warning_text"):
                    _draw_text_box(
                        canvas,
                        text=str(current_view["warning_text"]),
                        origin=(max(12, eye_uv[0] - 70), min(height - 12, eye_uv[1] + 26)),
                        color=(120, 120, 255),
                        font_scale=0.48,
                        thickness=1,
                    )

    if supported_arc_label:
        _draw_text_box(canvas, text=supported_arc_label, origin=(12, height - 14), color=(120, 235, 120), font_scale=0.50, thickness=1)
    if angle_label:
        _draw_text_box(canvas, text=angle_label, origin=(12, 26), color=(255, 255, 255), font_scale=0.54, thickness=1)

    return canvas


def build_scene_overview_state(
    *,
    scene_points: np.ndarray,
    scene_colors: np.ndarray,
    camera_geometries: list[dict[str, Any]],
    focus_point: np.ndarray,
    render_mode: str,
    renderer: str,
    scalar_bounds: dict[str, tuple[float, float]],
    point_radius_px: int,
    supersample_scale: int,
    orbit_path_points: np.ndarray | None = None,
    orbit_path_supported: list[bool] | None = None,
    crop_bounds: dict[str, np.ndarray] | None = None,
    supported_arc_label: str | None = None,
) -> dict[str, Any]:
    focus = np.asarray(focus_point, dtype=np.float32)
    basis_x, basis_y, basis_z = _build_overview_display_basis(camera_geometries, focus)
    display_scene_points = _transform_points_to_display_frame(
        scene_points,
        origin=focus,
        basis_x=basis_x,
        basis_y=basis_y,
        basis_z=basis_z,
    )
    display_camera_geometries = _transform_camera_geometries_to_display_frame(
        camera_geometries,
        origin=focus,
        basis_x=basis_x,
        basis_y=basis_y,
        basis_z=basis_z,
    )
    display_orbit_path_points = None
    if orbit_path_points is not None:
        display_orbit_path_points = _transform_points_to_display_frame(
            orbit_path_points,
            origin=focus,
            basis_x=basis_x,
            basis_y=basis_y,
            basis_z=basis_z,
        )
    display_crop_bounds = _transform_crop_bounds_to_display_frame(
        crop_bounds,
        origin=focus,
        basis_x=basis_x,
        basis_y=basis_y,
        basis_z=basis_z,
    )
    geometry_points = collect_camera_geometry_points(display_camera_geometries)
    bounds_min, bounds_max = _compute_bounds(
        [display_scene_points, geometry_points, np.zeros((1, 3), dtype=np.float32)],
        fallback_center=np.zeros((3,), dtype=np.float32),
    )
    overview_label, overview_view = _make_overview_pane_configs(bounds_min, bounds_max)[0]
    overview_ortho_scale = estimate_ortho_scale(
        [display_scene_points, geometry_points],
        view_config=overview_view,
    )
    overview_image_raw, renderer_used = render_point_cloud(
        display_scene_points,
        scene_colors,
        renderer=renderer,
        view_config=overview_view,
        render_mode=render_mode,
        scalar_bounds=scalar_bounds,
        width=520,
        height=420,
        point_radius_px=max(2, int(point_radius_px)),
        supersample_scale=max(1, int(supersample_scale)),
        projection_mode="orthographic",
        ortho_scale=overview_ortho_scale,
    )
    overview_image = draw_scene_overlays(
        overview_image_raw,
        camera_geometries=display_camera_geometries,
        view_config=overview_view,
        projection_mode="orthographic",
        ortho_scale=overview_ortho_scale,
        focus_point=np.zeros((3,), dtype=np.float32),
        orbit_path_points=display_orbit_path_points,
        orbit_path_supported=orbit_path_supported,
        crop_bounds=display_crop_bounds,
        supported_arc_label=supported_arc_label,
    )
    return {
        "image": overview_image,
        "renderer_used": renderer_used,
        "label": overview_label,
        "view_config": overview_view,
        "projection_mode": "orthographic",
        "ortho_scale": float(overview_ortho_scale),
        "display_origin": focus,
        "display_basis_x": basis_x,
        "display_basis_y": basis_y,
        "display_basis_z": basis_z,
        "supported_arc_label": supported_arc_label,
    }


def render_overview_inset(
    overview_state: dict[str, Any],
    *,
    current_views: list[dict[str, Any]],
    inset_size: tuple[int, int] = (560, 320),
    angle_label: str | None = None,
) -> np.ndarray:
    transformed_views = _transform_views_to_display_frame(
        current_views,
        origin=np.asarray(overview_state["display_origin"], dtype=np.float32),
        basis_x=np.asarray(overview_state["display_basis_x"], dtype=np.float32),
        basis_y=np.asarray(overview_state["display_basis_y"], dtype=np.float32),
        basis_z=np.asarray(overview_state["display_basis_z"], dtype=np.float32),
    )
    overlay = draw_scene_overlays(
        overview_state["image"],
        camera_geometries=[],
        view_config=overview_state["view_config"],
        projection_mode=overview_state["projection_mode"],
        ortho_scale=overview_state["ortho_scale"],
        focus_point=None,
        current_views=transformed_views,
        orbit_path_points=None,
        orbit_path_supported=None,
        crop_bounds=None,
        angle_label=angle_label,
        supported_arc_label=overview_state["supported_arc_label"],
    )
    return _fit_image_to_canvas(
        overlay,
        canvas_size=inset_size,
        background_bgr=(18, 18, 22),
    )


def _overlay_large_panel_label(
    image: np.ndarray,
    *,
    label: str,
    accent_bgr: tuple[int, int, int],
) -> np.ndarray:
    canvas = np.asarray(image, dtype=np.uint8).copy()
    strip_h = 48
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1] - 1, strip_h), (18, 18, 20), -1)
    cv2.rectangle(canvas, (0, 0), (18, strip_h), accent_bgr, -1)
    cv2.putText(canvas, label, (30, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.92, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def compose_side_by_side_large(
    *,
    title_lines: list[str],
    native_image: np.ndarray,
    ffs_image: np.ndarray,
    overview_inset: np.ndarray,
    warning_text: str | None = None,
) -> np.ndarray:
    native_labeled = _overlay_large_panel_label(native_image, label="Native", accent_bgr=(80, 180, 255))
    ffs_labeled = _overlay_large_panel_label(ffs_image, label="FFS", accent_bgr=(120, 220, 120))
    main_body = np.hstack([native_labeled, ffs_labeled])

    title_h = 86
    title_bar = np.zeros((title_h, main_body.shape[1], 3), dtype=np.uint8)
    title_bar[:] = (12, 14, 18)
    for line_idx, line in enumerate(title_lines[:2]):
        y = 28 + line_idx * 26
        cv2.putText(
            title_bar,
            line,
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.76 if line_idx == 0 else 0.62,
            (255, 255, 255),
            2 if line_idx == 0 else 1,
            cv2.LINE_AA,
        )
    if warning_text:
        cv2.rectangle(title_bar, (title_bar.shape[1] - 430, 14), (title_bar.shape[1] - 16, 44), (48, 48, 120), -1)
        cv2.putText(title_bar, warning_text, (title_bar.shape[1] - 418, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)

    inset = np.asarray(overview_inset, dtype=np.uint8)
    overview_h, overview_w = inset.shape[:2]
    footer_h = max(overview_h + 32, 520)
    footer = np.zeros((footer_h, main_body.shape[1], 3), dtype=np.uint8)
    footer[:] = (16, 18, 22)
    cv2.putText(footer, "Overview", (22, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.02, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(footer, "Real camera frusta, ROI crop, supported arc, and current orbit camera", (22, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (220, 220, 220), 2, cv2.LINE_AA)
    cv2.putText(footer, "The orbit path is identical for Native, FFS, and support render.", (22, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (220, 220, 220), 2, cv2.LINE_AA)
    cv2.putText(footer, "Camera labels show the original calibrated viewpoints used to define the supported viewing arc.", (22, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (220, 220, 220), 2, cv2.LINE_AA)
    max_overview_w = max(260, footer.shape[1] - 420)
    max_overview_h = footer_h - 36
    scale = min(1.0, float(max_overview_w) / max(1, overview_w), float(max_overview_h) / max(1, overview_h))
    if scale < 1.0:
        overview_w = max(160, int(round(overview_w * scale)))
        overview_h = max(120, int(round(overview_h * scale)))
        inset = cv2.resize(inset, (overview_w, overview_h), interpolation=cv2.INTER_AREA)
    x0 = footer.shape[1] - overview_w - 28
    y0 = max(18, (footer_h - overview_h) // 2)
    footer[y0:y0 + overview_h, x0:x0 + overview_w] = inset
    cv2.rectangle(footer, (x0 - 1, y0 - 1), (x0 + overview_w, y0 + overview_h), (255, 255, 255), 1, cv2.LINE_AA)

    return np.vstack([title_bar, main_body, footer])


def compose_turntable_board(
    *,
    title_lines: list[str],
    column_headers: list[str],
    row_headers: list[str],
    native_images: list[np.ndarray],
    ffs_images: list[np.ndarray],
    overview_inset: np.ndarray | None = None,
) -> np.ndarray:
    if len(native_images) != len(ffs_images):
        raise ValueError("native_images and ffs_images must have the same length.")
    if len(row_headers) != 2:
        raise ValueError("turntable board requires exactly 2 row headers.")
    if len(column_headers) != len(native_images):
        raise ValueError("column_headers must match image column count.")

    panel_h, panel_w = native_images[0].shape[:2]
    num_cols = len(native_images)
    row_label_w = 170
    header_h = 40
    body = np.zeros((panel_h * 2, row_label_w + panel_w * num_cols, 3), dtype=np.uint8)

    for row_idx, row_images in enumerate((native_images, ffs_images)):
        y0 = row_idx * panel_h
        body[y0:y0 + panel_h, :row_label_w] = (20, 20, 20)
        header = row_headers[row_idx]
        text_size = cv2.getTextSize(header, cv2.FONT_HERSHEY_SIMPLEX, 0.92, 2)[0]
        text_x = max(10, (row_label_w - text_size[0]) // 2)
        text_y = y0 + (panel_h + text_size[1]) // 2
        cv2.putText(body, header, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.92, (255, 255, 255), 2, cv2.LINE_AA)
        for col_idx, image in enumerate(row_images):
            x0 = row_label_w + col_idx * panel_w
            body[y0:y0 + panel_h, x0:x0 + panel_w] = image

    header_bar = np.zeros((header_h, body.shape[1], 3), dtype=np.uint8)
    header_bar[:, :row_label_w] = (26, 26, 26)
    for col_idx, header in enumerate(column_headers):
        x0 = row_label_w + col_idx * panel_w
        header_bar[:, x0:x0 + panel_w] = (26, 26, 26)
        text_size = cv2.getTextSize(header, cv2.FONT_HERSHEY_SIMPLEX, 0.80, 2)[0]
        text_x = x0 + max(8, (panel_w - text_size[0]) // 2)
        cv2.putText(header_bar, header, (text_x, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.80, (255, 255, 255), 2, cv2.LINE_AA)

    inset_h = 0 if overview_inset is None else int(overview_inset.shape[0])
    title_h = max(74, inset_h + 16)
    title_bar = np.zeros((title_h, body.shape[1], 3), dtype=np.uint8)
    for line_idx, line in enumerate(title_lines[:2]):
        y = 26 + line_idx * 24
        cv2.putText(title_bar, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68 if line_idx == 0 else 0.58, (255, 255, 255), 2 if line_idx == 0 else 1, cv2.LINE_AA)

    if overview_inset is not None:
        inset = np.asarray(overview_inset, dtype=np.uint8)
        inset_h, inset_w = inset.shape[:2]
        x0 = title_bar.shape[1] - inset_w - 12
        y0 = max(8, (title_bar.shape[0] - inset_h) // 2)
        title_bar[y0:y0 + inset_h, x0:x0 + inset_w] = inset
        cv2.rectangle(title_bar, (x0 - 1, y0 - 1), (x0 + inset_w, y0 + inset_h), (255, 255, 255), 1, cv2.LINE_AA)

    return np.vstack([title_bar, header_bar, body])


def compose_keyframe_sheet(
    boards: list[np.ndarray],
    *,
    max_width: int = 4600,
    max_height: int = 4200,
    padding: int = 18,
) -> np.ndarray:
    if not boards:
        raise ValueError("compose_keyframe_sheet requires at least one board.")
    board_h, board_w = boards[0].shape[:2]
    cols = 1 if len(boards) == 1 else 2
    rows = int(math.ceil(len(boards) / cols))
    scale = min(
        1.0,
        float(max_width - padding * (cols + 1)) / max(1.0, cols * board_w),
        float(max_height - padding * (rows + 1)) / max(1.0, rows * board_h),
    )
    tile_w = max(320, int(round(board_w * scale)))
    tile_h = max(240, int(round(board_h * scale)))
    canvas = np.zeros((padding * (rows + 1) + tile_h * rows, padding * (cols + 1) + tile_w * cols, 3), dtype=np.uint8)
    canvas[:] = (10, 10, 10)
    for idx, board in enumerate(boards):
        row = idx // cols
        col = idx % cols
        x0 = padding + col * (tile_w + padding)
        y0 = padding + row * (tile_h + padding)
        resized = cv2.resize(board, (tile_w, tile_h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
        canvas[y0:y0 + tile_h, x0:x0 + tile_w] = resized
    return canvas


def _format_angle_token(angle_deg: float) -> str:
    sign = "p" if angle_deg >= 0 else "m"
    scaled = int(round(abs(float(angle_deg)) * 10.0))
    return f"{sign}{scaled:04d}"


def run_turntable_compare_workflow(
    *,
    aligned_root: Path,
    output_dir: Path,
    case_name: str | None = None,
    realsense_case: str | None = None,
    ffs_case: str | None = None,
    frame_idx: int = 0,
    renderer: str = "auto",
    render_mode: str = "neutral_gray_shaded",
    write_mp4: bool = True,
    write_gif: bool = True,
    write_keyframe_sheet: bool = True,
    num_orbit_steps: int = 72,
    orbit_degrees: float = 360.0,
    camera_ids: list[int] | None = None,
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
    projection_mode: str = "perspective",
    point_radius_px: int = 4,
    supersample_scale: int = 3,
    voxel_size: float | None = None,
    max_points_per_camera: int | None = 50000,
    depth_min_m: float = 0.2,
    depth_max_m: float = 1.5,
    use_float_ffs_depth_when_available: bool = True,
    fps: int = 8,
    orbit_mode: str = "observed_hemisphere",
    layout_mode: str = "side_by_side_large",
    orbit_radius_scale: float = 1.9,
    view_height_offset: float = 0.0,
    render_both_modes: bool = True,
    coverage_margin_deg: float = 18.0,
    show_unsupported_warning: bool = True,
) -> dict[str, Any]:
    if render_mode not in RENDER_MODES:
        raise ValueError(f"Unsupported render_mode: {render_mode}")
    if scene_crop_mode not in SCENE_CROP_MODES:
        raise ValueError(f"Unsupported scene_crop_mode: {scene_crop_mode}")
    if projection_mode not in PROJECTION_MODES:
        raise ValueError(f"Unsupported projection_mode: {projection_mode}")
    if orbit_mode not in ORBIT_MODES:
        raise ValueError(f"Unsupported orbit_mode: {orbit_mode}")
    if layout_mode not in LAYOUT_MODES:
        raise ValueError(f"Unsupported layout_mode: {layout_mode}")

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selection = resolve_single_frame_case_selection(
        aligned_root=Path(aligned_root).resolve(),
        case_name=case_name,
        realsense_case=realsense_case,
        ffs_case=ffs_case,
        frame_idx=frame_idx,
        camera_ids=camera_ids,
    )
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

    camera_poses = extract_camera_poses(
        selection["native_c2w"],
        serial_numbers=selection["serial_numbers"],
        camera_ids=selection["camera_ids"],
    )
    orbit_axis = estimate_orbit_axis(camera_poses)
    if layout_mode == "side_by_side_large" and orbit_mode not in ("observed_hemisphere", "full_360"):
        raise ValueError("side_by_side_large currently requires orbit_mode=observed_hemisphere or full_360.")
    if layout_mode == "camera_neighborhood_grid" and orbit_mode != "camera_neighborhood":
        raise ValueError("camera_neighborhood_grid currently requires orbit_mode=camera_neighborhood.")

    if orbit_mode in ("observed_hemisphere", "full_360"):
        object_orbit = build_object_centered_orbit_views(
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
            show_unsupported_warning=show_unsupported_warning,
        )
        orbit_steps = object_orbit["orbit_steps"]
        orbit_path_points = object_orbit["orbit_path"]
        orbit_path_supported = object_orbit["orbit_supported_mask"]
        orbit_radius = object_orbit["orbit_radius"]
        orbit_height = object_orbit["orbit_height"]
        start_azimuth_deg = object_orbit["start_azimuth_deg"]
        camera_reference_azimuths_deg = object_orbit["camera_reference_azimuths_deg"]
        coverage_arc = object_orbit["coverage_arc"]
    else:
        orbit_steps = build_camera_anchored_orbit_views(
            camera_poses=camera_poses,
            focus_point=scene["focus_point"],
            orbit_axis=orbit_axis,
            num_orbit_steps=num_orbit_steps,
            orbit_degrees=orbit_degrees,
        )
        orbit_path_points = np.stack(
            [step["view_configs"][0]["camera_position"] for step in orbit_steps],
            axis=0,
        ).astype(np.float32)
        orbit_path_supported = [True] * len(orbit_steps)
        orbit_radius = float(np.median([step["view_configs"][0]["radius"] for step in orbit_steps])) if orbit_steps else 0.0
        orbit_height = 0.0
        start_azimuth_deg = 0.0
        camera_reference_azimuths_deg = {}
        coverage_arc = {"start_deg": 0.0, "end_deg": 360.0, "span_deg": 360.0}

    scene_diagonal = float(np.linalg.norm(scene["bounds_max"] - scene["bounds_min"]))
    frustum_scale = max(0.06, scene_diagonal * 0.14)
    camera_geometries = [
        build_camera_frustum_geometry(pose, frustum_scale=frustum_scale)
        for pose in camera_poses
    ]

    if len(scene["native_render_points"]) > 0 and len(scene["ffs_render_points"]) > 0:
        overview_cloud_points = np.concatenate([scene["native_render_points"], scene["ffs_render_points"]], axis=0)
        overview_cloud_colors = np.concatenate([scene["native_render_colors"], scene["ffs_render_colors"]], axis=0)
    elif len(scene["native_render_points"]) > 0:
        overview_cloud_points = scene["native_render_points"]
        overview_cloud_colors = scene["native_render_colors"]
    else:
        overview_cloud_points = scene["ffs_render_points"]
        overview_cloud_colors = scene["ffs_render_colors"]
    overview_state = build_scene_overview_state(
        scene_points=overview_cloud_points,
        scene_colors=overview_cloud_colors,
        camera_geometries=camera_geometries,
        focus_point=scene["focus_point"],
        render_mode="color_by_height",
        renderer=renderer,
        scalar_bounds=scene["scalar_bounds"],
        point_radius_px=point_radius_px,
        supersample_scale=supersample_scale,
        orbit_path_points=orbit_path_points,
        orbit_path_supported=orbit_path_supported,
        crop_bounds=scene["object_roi_bounds"] if scene_crop_mode == "auto_object_bbox" else scene["crop_bounds"],
        supported_arc_label=f"Supported arc: {coverage_arc['span_deg']:.1f} deg",
    )
    overview_image_path = output_dir / "scene_overview_with_cameras.png"
    cv2.imwrite(str(overview_image_path), overview_state["image"])
    source_legend_path = Path(write_source_legend_image(output_dir / "source_attribution_legend.png"))

    output_specs = build_render_output_specs(
        geom_render_mode=render_mode,
        render_both_modes=render_both_modes,
    )
    frame_paths_by_output: dict[str, list[Path]] = {spec["name"]: [] for spec in output_specs}
    frames_dir_by_output: dict[str, Path] = {}
    board_images_by_output: dict[str, list[np.ndarray]] = {spec["name"]: [] for spec in output_specs}
    renderer_used_by_output: dict[str, dict[str, str]] = {spec["name"]: {} for spec in output_specs}
    support_metrics: list[dict[str, Any]] = []
    source_metrics_steps: list[dict[str, Any]] = []
    mismatch_metrics_steps: list[dict[str, Any]] = []
    source_split_frames_dir = output_dir / "frames_source_split"
    source_split_frames_dir.mkdir(parents=True, exist_ok=True)
    source_split_boards: list[np.ndarray] = []
    source_split_frame_paths: list[Path] = []
    compare_mode_label = "same-case" if selection["same_case_mode"] else "two-case fallback"
    case_label = (
        selection["native_case_dir"].name
        if selection["same_case_mode"]
        else f"{selection['native_case_dir'].name} vs {selection['ffs_case_dir'].name}"
    )
    main_width = 1280 if layout_mode == "side_by_side_large" else 960
    main_height = 900 if layout_mode == "side_by_side_large" else 720

    for output_spec in output_specs:
        frames_dir = output_dir / output_spec["frames_dir_name"]
        frames_dir.mkdir(parents=True, exist_ok=True)
        frames_dir_by_output[output_spec["name"]] = frames_dir

    for orbit_step in orbit_steps:
        current_views = orbit_step["view_configs"]
        current_view = orbit_step["view_config"] if "view_config" in orbit_step else orbit_step["view_configs"][0]
        step_source_metrics_entry: dict[str, Any] | None = None
        overview_inset = render_overview_inset(
            overview_state,
            current_views=current_views,
            inset_size=(920, 560) if layout_mode == "side_by_side_large" else (420, 260),
            angle_label=f"Angle: {current_view.get('azimuth_deg', orbit_step['angle_deg']):+.1f} deg",
        )

        per_step_renders: dict[tuple[str, str], np.ndarray] = {}
        for output_spec in output_specs:
            mode_name = output_spec["name"]
            mode_render = output_spec["render_mode"]
            if layout_mode == "side_by_side_large":
                view_configs = [orbit_step["view_config"]]
            else:
                view_configs = orbit_step["view_configs"]

            native_images: list[np.ndarray] = []
            ffs_images: list[np.ndarray] = []
            for view_config in view_configs:
                ortho_scale = None
                if projection_mode == "orthographic":
                    ortho_scale = estimate_ortho_scale(
                        [scene["native_render_points"], scene["ffs_render_points"]],
                        view_config=view_config,
                    )
                if mode_render == "support_count":
                    native_support = compute_support_count_map(
                        scene["native_render_camera_clouds"],
                        view_config=view_config,
                        width=main_width,
                        height=main_height,
                        projection_mode=projection_mode,
                        ortho_scale=ortho_scale,
                    )
                    ffs_support = compute_support_count_map(
                        scene["ffs_render_camera_clouds"],
                        view_config=view_config,
                        width=main_width,
                        height=main_height,
                        projection_mode=projection_mode,
                        ortho_scale=ortho_scale,
                    )
                    native_render = overlay_support_legend(
                        render_support_count_map(native_support["support_count"], native_support["valid"])
                    )
                    ffs_render = overlay_support_legend(
                        render_support_count_map(ffs_support["support_count"], ffs_support["valid"])
                    )
                    native_renderer_used = "support_count"
                    ffs_renderer_used = "support_count"
                    support_metrics.append(
                        {
                            "step_idx": int(orbit_step["step_idx"]),
                            "angle_deg": float(orbit_step["angle_deg"]),
                            "azimuth_deg": float(view_config.get("azimuth_deg", orbit_step["angle_deg"])),
                            "is_supported": bool(view_config.get("is_supported", True)),
                            "native": summarize_support_counts(native_support["support_count"], native_support["valid"]),
                            "ffs": summarize_support_counts(ffs_support["support_count"], ffs_support["valid"]),
                        }
                    )
                elif mode_render == "source_attribution_alpha":
                    native_render, native_source = render_source_attribution_overlay(
                        scene["native_render_camera_clouds"],
                        view_config=view_config,
                        width=main_width,
                        height=main_height,
                        projection_mode=projection_mode,
                        ortho_scale=ortho_scale,
                    )
                    ffs_render, ffs_source = render_source_attribution_overlay(
                        scene["ffs_render_camera_clouds"],
                        view_config=view_config,
                        width=main_width,
                        height=main_height,
                        projection_mode=projection_mode,
                        ortho_scale=ortho_scale,
                    )
                    native_renderer_used = "source_attribution"
                    ffs_renderer_used = "source_attribution"
                    step_source_metrics_entry = {
                        "step_idx": int(orbit_step["step_idx"]),
                        "angle_deg": float(orbit_step["angle_deg"]),
                        "azimuth_deg": float(view_config.get("azimuth_deg", orbit_step["angle_deg"])),
                        "native": native_source,
                        "ffs": ffs_source,
                    }
                    source_metrics_steps.append(step_source_metrics_entry)
                elif mode_render == "mismatch_residual":
                    native_render, native_mismatch = render_mismatch_residual(
                        scene["native_render_camera_clouds"],
                        view_config=view_config,
                        width=main_width,
                        height=main_height,
                        projection_mode=projection_mode,
                        ortho_scale=ortho_scale,
                    )
                    ffs_render, ffs_mismatch = render_mismatch_residual(
                        scene["ffs_render_camera_clouds"],
                        view_config=view_config,
                        width=main_width,
                        height=main_height,
                        projection_mode=projection_mode,
                        ortho_scale=ortho_scale,
                    )
                    native_renderer_used = "mismatch_residual"
                    ffs_renderer_used = "mismatch_residual"
                    mismatch_metrics_steps.append(
                        {
                            "step_idx": int(orbit_step["step_idx"]),
                            "angle_deg": float(orbit_step["angle_deg"]),
                            "azimuth_deg": float(view_config.get("azimuth_deg", orbit_step["angle_deg"])),
                            "native": {
                                "summary": native_mismatch["summary"],
                                "per_camera": native_mismatch["per_camera"],
                            },
                            "ffs": {
                                "summary": ffs_mismatch["summary"],
                                "per_camera": ffs_mismatch["per_camera"],
                            },
                        }
                    )
                else:
                    native_render, native_renderer_used = render_point_cloud(
                        scene["native_render_points"],
                        scene["native_render_colors"],
                        renderer=renderer,
                        view_config=view_config,
                        render_mode=mode_render,
                        scalar_bounds=scene["scalar_bounds"],
                        width=main_width,
                        height=main_height,
                        point_radius_px=point_radius_px,
                        supersample_scale=supersample_scale,
                        projection_mode=projection_mode,
                        ortho_scale=ortho_scale,
                    )
                    ffs_render, ffs_renderer_used = render_point_cloud(
                        scene["ffs_render_points"],
                        scene["ffs_render_colors"],
                        renderer=renderer if renderer != "auto" else native_renderer_used,
                        view_config=view_config,
                        render_mode=mode_render,
                        scalar_bounds=scene["scalar_bounds"],
                        width=main_width,
                        height=main_height,
                        point_radius_px=point_radius_px,
                        supersample_scale=supersample_scale,
                        projection_mode=projection_mode,
                        ortho_scale=ortho_scale,
                    )
                native_images.append(native_render)
                ffs_images.append(ffs_render)
                renderer_used_by_output[mode_name][f"native_{view_config['view_name']}"] = native_renderer_used
                renderer_used_by_output[mode_name][f"ffs_{view_config['view_name']}"] = ffs_renderer_used

            if layout_mode == "side_by_side_large":
                board = compose_side_by_side_large(
                    title_lines=[
                        f"{case_label} | frame_idx={selection['native_frame_idx']} | {compare_mode_label}",
                        f"{mode_name} | orbit={orbit_step['angle_deg']:+.1f} deg | proj={projection_mode} | crop={scene_crop_mode} | coverage={orbit_mode}",
                    ],
                    native_image=native_images[0],
                    ffs_image=ffs_images[0],
                    overview_inset=overview_inset,
                    warning_text=current_view.get("warning_text"),
                )
            else:
                board = compose_turntable_board(
                    title_lines=[
                        f"{case_label} | frame_idx={selection['native_frame_idx']} | {compare_mode_label}",
                        f"{mode_name} | render={mode_render} | orbit={orbit_step['angle_deg']:+.1f} deg | proj={projection_mode}",
                    ],
                    column_headers=[f"Near Cam{view_config['camera_idx']}" for view_config in view_configs],
                    row_headers=["Native", "FFS"],
                    native_images=native_images,
                    ffs_images=ffs_images,
                    overview_inset=overview_inset,
                )
            per_step_renders[(mode_name, "board")] = board

        split_view_config = orbit_step["view_config"] if "view_config" in orbit_step else orbit_step["view_configs"][0]
        split_ortho_scale = None
        if projection_mode == "orthographic":
            split_ortho_scale = estimate_ortho_scale(
                [scene["native_render_points"], scene["ffs_render_points"]],
                view_config=split_view_config,
            )
        native_split_images, native_split_metrics = render_source_split_images(
            scene["native_render_camera_clouds"],
            view_config=split_view_config,
            scalar_bounds=scene["scalar_bounds"],
            renderer=renderer,
            width=640,
            height=420,
            point_radius_px=point_radius_px,
            supersample_scale=supersample_scale,
            projection_mode=projection_mode,
            ortho_scale=split_ortho_scale,
        )
        ffs_split_images, ffs_split_metrics = render_source_split_images(
            scene["ffs_render_camera_clouds"],
            view_config=split_view_config,
            scalar_bounds=scene["scalar_bounds"],
            renderer=renderer,
            width=640,
            height=420,
            point_radius_px=point_radius_px,
            supersample_scale=supersample_scale,
            projection_mode=projection_mode,
            ortho_scale=split_ortho_scale,
        )
        source_split_board = compose_turntable_board(
            title_lines=[
                f"{case_label} | frame_idx={selection['native_frame_idx']} | {compare_mode_label}",
                f"source_split | orbit={orbit_step['angle_deg']:+.1f} deg | proj={projection_mode}",
            ],
            column_headers=["Cam0", "Cam1", "Cam2"],
            row_headers=["Native", "FFS"],
            native_images=native_split_images,
            ffs_images=ffs_split_images,
            overview_inset=overview_inset,
        )
        split_board_path = source_split_frames_dir / f"{orbit_step['step_idx']:03d}_angle_{_format_angle_token(orbit_step['angle_deg'])}.png"
        cv2.imwrite(str(split_board_path), source_split_board)
        source_split_frame_paths.append(split_board_path)
        source_split_boards.append(source_split_board)
        if step_source_metrics_entry is not None:
            step_source_metrics_entry["native_split"] = native_split_metrics
            step_source_metrics_entry["ffs_split"] = ffs_split_metrics

        for output_spec in output_specs:
            mode_name = output_spec["name"]
            board = per_step_renders[(mode_name, "board")]
            board_path = frames_dir_by_output[mode_name] / f"{orbit_step['step_idx']:03d}_angle_{_format_angle_token(orbit_step['angle_deg'])}.png"
            cv2.imwrite(str(board_path), board)
            frame_paths_by_output[mode_name].append(board_path)
            board_images_by_output[mode_name].append(board)

    output_files: dict[str, dict[str, str | None]] = {}
    for output_spec in output_specs:
        mode_name = output_spec["name"]
        video_path = output_dir / output_spec["video_name"]
        gif_path = output_dir / output_spec["gif_name"]
        sheet_path = output_dir / output_spec["sheet_name"]
        if write_mp4 and bool(output_spec.get("write_video", True)):
            write_video(video_path, frame_paths_by_output[mode_name], fps)
        if write_gif and bool(output_spec.get("write_gif", True)):
            write_gif_animation(gif_path, frame_paths_by_output[mode_name], fps)
        if write_keyframe_sheet and board_images_by_output[mode_name]:
            sheet = compose_keyframe_sheet(board_images_by_output[mode_name])
            cv2.imwrite(str(sheet_path), sheet)
        output_files[mode_name] = {
            "frames_dir": str(frames_dir_by_output[mode_name]),
            "video_path": str(video_path) if write_mp4 and bool(output_spec.get("write_video", True)) else None,
            "gif_path": str(gif_path) if write_gif and bool(output_spec.get("write_gif", True)) else None,
            "sheet_path": str(sheet_path) if write_keyframe_sheet else None,
        }
    source_split_sheet_path = output_dir / "turntable_keyframes_source_split.png"
    if write_keyframe_sheet and source_split_boards:
        source_split_sheet = compose_keyframe_sheet(source_split_boards)
        cv2.imwrite(str(source_split_sheet_path), source_split_sheet)
    output_files["source_split"] = {
        "frames_dir": str(source_split_frames_dir),
        "video_path": None,
        "gif_path": None,
        "sheet_path": str(source_split_sheet_path) if write_keyframe_sheet else None,
    }
    support_metrics_path = output_dir / "support_metrics.json"
    support_metrics_path.write_text(json.dumps(support_metrics, indent=2), encoding="utf-8")
    source_metrics_payload = {
        "source_color_map_bgr": {str(key): list(value) for key, value in SOURCE_CAMERA_COLORS_BGR.items()},
        "native": {
            "object_source_histogram": _source_histogram(scene["native_object_source_camera_idx"]),
            "context_source_histogram": _source_histogram(scene["native_context_source_camera_idx"]),
            "combined_source_histogram": _source_histogram(scene["native_render_source_camera_idx"]),
        },
        "ffs": {
            "object_source_histogram": _source_histogram(scene["ffs_object_source_camera_idx"]),
            "context_source_histogram": _source_histogram(scene["ffs_context_source_camera_idx"]),
            "combined_source_histogram": _source_histogram(scene["ffs_render_source_camera_idx"]),
        },
        "steps": source_metrics_steps,
    }
    source_metrics_path = output_dir / "source_metrics.json"
    source_metrics_path.write_text(json.dumps(source_metrics_payload, indent=2), encoding="utf-8")
    mismatch_metrics_payload = {
        "steps": mismatch_metrics_steps,
        "native_overall": {
            "residual_mean_m": _aggregate_step_metric_series([item["native"]["summary"] for item in mismatch_metrics_steps], key="residual_mean_m"),
            "residual_p90_m": _aggregate_step_metric_series([item["native"]["summary"] for item in mismatch_metrics_steps], key="residual_p90_m"),
            "residual_max_m": _aggregate_step_metric_series([item["native"]["summary"] for item in mismatch_metrics_steps], key="residual_max_m"),
        },
        "ffs_overall": {
            "residual_mean_m": _aggregate_step_metric_series([item["ffs"]["summary"] for item in mismatch_metrics_steps], key="residual_mean_m"),
            "residual_p90_m": _aggregate_step_metric_series([item["ffs"]["summary"] for item in mismatch_metrics_steps], key="residual_p90_m"),
            "residual_max_m": _aggregate_step_metric_series([item["ffs"]["summary"] for item in mismatch_metrics_steps], key="residual_max_m"),
        },
    }
    mismatch_metrics_path = output_dir / "mismatch_metrics.json"
    mismatch_metrics_path.write_text(json.dumps(mismatch_metrics_payload, indent=2), encoding="utf-8")
    if refinement["pass1_crop"] is not None:
        _write_json(output_dir / "object_roi_pass1_world.json", _json_ready_crop_bounds(refinement["pass1_crop"]))
    if refinement["pass2_crop"] is not None:
        _write_json(output_dir / "object_roi_pass2_world.json", _json_ready_crop_bounds(refinement["pass2_crop"]))
    per_camera_auto_bbox_dir = output_dir / "per_camera_auto_bbox"
    for camera_idx, bbox_payload in refinement["bbox_debug_by_camera"].items():
        _write_json(per_camera_auto_bbox_dir / f"cam{camera_idx}.json", bbox_payload)

    debug_dir = output_dir / "debug"
    native_debug = write_object_debug_artifacts(
        output_dir=debug_dir,
        source_name="native",
        object_layers={
            "object_camera_clouds": scene["native_object_camera_clouds"],
            "combined_camera_clouds": scene["native_render_camera_clouds"],
            "per_camera_metrics": scene["native_debug_metrics"],
        },
        focus_point=scene["focus_point"],
        renderer=renderer,
        render_mode="color_by_rgb",
    )
    ffs_debug = write_object_debug_artifacts(
        output_dir=debug_dir,
        source_name="ffs",
        object_layers={
            "object_camera_clouds": scene["ffs_object_camera_clouds"],
            "combined_camera_clouds": scene["ffs_render_camera_clouds"],
            "per_camera_metrics": scene["ffs_debug_metrics"],
        },
        focus_point=scene["focus_point"],
        renderer=renderer,
        render_mode="color_by_rgb",
    )
    native_fused_debug = write_fused_cloud_debug_artifacts(
        output_dir=debug_dir,
        source_name="native",
        object_points=scene["native_object_points"],
        object_colors=scene["native_object_colors"],
        combined_points=scene["native_render_points"],
        combined_colors=scene["native_render_colors"],
        focus_point=scene["focus_point"],
        renderer=renderer,
    )
    ffs_fused_debug = write_fused_cloud_debug_artifacts(
        output_dir=debug_dir,
        source_name="ffs",
        object_points=scene["ffs_object_points"],
        object_colors=scene["ffs_object_colors"],
        combined_points=scene["ffs_render_points"],
        combined_colors=scene["ffs_render_colors"],
        focus_point=scene["focus_point"],
        renderer=renderer,
    )
    root_native_object_png = output_dir / "fused_object_only_native.png"
    root_ffs_object_png = output_dir / "fused_object_only_ffs.png"
    root_native_object_png.write_bytes(Path(native_fused_debug["object_png"]).read_bytes())
    root_ffs_object_png.write_bytes(Path(ffs_fused_debug["object_png"]).read_bytes())
    for camera_idx in (0, 1, 2):
        preview_idx = next(
            (idx for idx, metrics in enumerate(native_debug["per_camera_metrics"]) if int(metrics["camera_idx"]) == camera_idx),
            None,
        )
        if preview_idx is not None:
            preview_path = Path(native_debug["preview_paths"][preview_idx])
            (output_dir / f"per_camera_object_cloud_cam{camera_idx}.png").write_bytes(preview_path.read_bytes())

    def _aggregate_support(source_name: str) -> dict[str, float]:
        relevant = [item[source_name] for item in support_metrics if source_name in item]
        if not relevant:
            return {"valid_pixels_mean": 0.0, "single_camera_ratio_mean": 0.0, "two_camera_ratio_mean": 0.0, "three_camera_ratio_mean": 0.0}
        return {
            "valid_pixels_mean": float(np.mean([entry["valid_pixel_count"] for entry in relevant])),
            "single_camera_ratio_mean": float(np.mean([entry["support_ratio_1"] for entry in relevant])),
            "two_camera_ratio_mean": float(np.mean([entry["support_ratio_2"] for entry in relevant])),
            "three_camera_ratio_mean": float(np.mean([entry["support_ratio_3"] for entry in relevant])),
        }

    head_like_recovered = bool(
        refinement["pass2_seed_height_summary"]["max_height"] > refinement["pass1_seed_height_summary"]["max_height"] + 0.01
        or refinement["pass2_seed_height_summary"]["point_count"] > refinement["pass1_seed_height_summary"]["point_count"] * 1.08
    )

    compare_debug_metrics_path = debug_dir / "compare_debug_metrics.json"
    compare_debug_metrics_root_path = output_dir / "compare_debug_metrics.json"
    compare_debug_payload = {
        "scene_crop_mode": scene_crop_mode,
        "object_component_mode": object_component_mode,
        "object_component_topk": int(object_component_topk),
        "object_height_min": float(object_height_min),
        "object_height_max": float(object_height_max),
        "manual_image_roi_by_camera": None
        if manual_image_roi_by_camera is None
        else {str(key): list(value) for key, value in manual_image_roi_by_camera.items()},
        "context_max_points_per_camera": None if max_points_per_camera is None else int(max_points_per_camera),
        "world_roi_pass1": None if refinement["pass1_crop"] is None else _json_ready_crop_bounds(refinement["pass1_crop"]),
        "world_roi_pass2": None if refinement["pass2_crop"] is None else _json_ready_crop_bounds(refinement["pass2_crop"]),
        "per_camera_auto_bbox_dir": str(per_camera_auto_bbox_dir),
        "crop_metadata": scene["crop_metadata"],
        "head_like_protrusion_recovered_heuristic": bool(head_like_recovered),
        "pass2_refinement_valid": bool(refinement["pass2_refinement_valid"]),
        "final_compare_source": "pass2" if refinement["pass2_refinement_valid"] else "pass1",
        "pass2_valid_camera_count": int(refinement["pass2_valid_camera_count"]),
        "pass1_object_point_count": int(refinement["pass1_object_point_count"]),
        "pass2_object_point_count": int(refinement["pass2_object_point_count"]),
        "pass1_object_volume": float(refinement["pass1_object_volume"]),
        "pass2_object_volume": float(refinement["pass2_object_volume"]),
        "pass1_seed_height_summary": refinement["pass1_seed_height_summary"],
        "pass2_seed_height_summary": refinement["pass2_seed_height_summary"],
        "final_support_summary": {
            "native": _aggregate_support("native"),
            "ffs": _aggregate_support("ffs"),
        },
        "source_metrics_path": str(source_metrics_path),
        "mismatch_metrics_path": str(mismatch_metrics_path),
        "source_legend_path": str(source_legend_path),
        "native": {
            "pass1_seed_mask_metrics": refinement["native_pass1_seed_metrics"],
            "pass2_seed_mask_metrics": refinement["native_pass2_seed_metrics"],
            "final_seed_mask_metrics": refinement["final_seed_metrics_native"],
            "pass1_mask_debug": {str(key): value for key, value in refinement["pass1_mask_debug"]["native"].items()},
            "pass2_mask_debug": {str(key): value for key, value in refinement["pass2_mask_debug"]["native"].items()},
            "final_mask_debug": {str(key): value for key, value in refinement["final_mask_debug"]["native"].items()},
            "per_camera": native_debug["per_camera_metrics"],
            "fused_object_point_count": int(len(scene["native_object_points"])),
            "fused_context_point_count": int(len(scene["native_context_points"])),
            "overlay_paths": native_debug["overlay_paths"],
            "preview_paths": native_debug["preview_paths"],
            **native_fused_debug,
        },
        "ffs": {
            "pass1_seed_mask_metrics": refinement["ffs_pass1_seed_metrics"],
            "pass2_seed_mask_metrics": refinement["ffs_pass2_seed_metrics"],
            "final_seed_mask_metrics": refinement["final_seed_metrics_ffs"],
            "pass1_mask_debug": {str(key): value for key, value in refinement["pass1_mask_debug"]["ffs"].items()},
            "pass2_mask_debug": {str(key): value for key, value in refinement["pass2_mask_debug"]["ffs"].items()},
            "final_mask_debug": {str(key): value for key, value in refinement["final_mask_debug"]["ffs"].items()},
            "per_camera": ffs_debug["per_camera_metrics"],
            "fused_object_point_count": int(len(scene["ffs_object_points"])),
            "fused_context_point_count": int(len(scene["ffs_context_points"])),
            "overlay_paths": ffs_debug["overlay_paths"],
            "preview_paths": ffs_debug["preview_paths"],
            **ffs_fused_debug,
        },
    }
    write_compare_debug_metrics(
        compare_debug_metrics_path,
        debug_payload=compare_debug_payload,
    )
    write_compare_debug_metrics(compare_debug_metrics_root_path, debug_payload=compare_debug_payload)

    metadata = {
        "same_case_mode": selection["same_case_mode"],
        "native_case_dir": str(selection["native_case_dir"]),
        "ffs_case_dir": str(selection["ffs_case_dir"]),
        "frame_idx": int(selection["native_frame_idx"]),
        "native_frame_idx": int(selection["native_frame_idx"]),
        "ffs_frame_idx": int(selection["ffs_frame_idx"]),
        "camera_ids": selection["camera_ids"],
        "camera_labels": [geometry["label"] for geometry in camera_geometries],
        "compare_mode_label": compare_mode_label,
        "geom_render_mode": render_mode,
        "render_both_modes": bool(render_both_modes),
        "projection_mode": projection_mode,
        "scene_crop_mode": scene_crop_mode,
        "focus_mode": focus_mode,
        "manual_image_roi_json": None if manual_image_roi_json is None else str(Path(manual_image_roi_json).resolve()),
        "manual_image_roi_by_camera": None
        if manual_image_roi_by_camera is None
        else {str(key): list(value) for key, value in manual_image_roi_by_camera.items()},
        "orbit_mode": orbit_mode,
        "layout_mode": layout_mode,
        "crop_bounds": {
            "min": scene["crop_bounds"]["min"].tolist(),
            "max": scene["crop_bounds"]["max"].tolist(),
        },
        "focus_point": scene["focus_point"].tolist(),
        "orbit_axis": orbit_axis.tolist(),
        "orbit_angles_deg": [step["angle_deg"] for step in orbit_steps],
        "num_orbit_steps": int(num_orbit_steps),
        "orbit_degrees": float(orbit_degrees),
        "coverage_margin_deg": float(coverage_margin_deg),
        "show_unsupported_warning": bool(show_unsupported_warning),
        "orbit_radius": float(orbit_radius),
        "orbit_radius_scale": float(orbit_radius_scale),
        "orbit_height": float(orbit_height),
        "view_height_offset": float(view_height_offset),
        "start_azimuth_deg": float(start_azimuth_deg),
        "camera_reference_azimuths_deg": camera_reference_azimuths_deg,
        "coverage_arc": coverage_arc,
        "unsupported_step_count": int(sum(1 for step in orbit_steps if not bool(step["view_config"].get("is_supported", True)))),
        "point_radius_px": int(point_radius_px),
        "supersample_scale": int(supersample_scale),
        "depth_min_m": float(depth_min_m),
        "depth_max_m": float(depth_max_m),
        "voxel_size": voxel_size,
        "max_points_per_camera": max_points_per_camera,
        "use_float_ffs_depth_when_available": bool(use_float_ffs_depth_when_available),
        "scene_overview_with_cameras": str(overview_image_path),
        "object_roi_pass1_world": None if refinement["pass1_crop"] is None else str((output_dir / "object_roi_pass1_world.json").resolve()),
        "object_roi_pass2_world": None if refinement["pass2_crop"] is None else str((output_dir / "object_roi_pass2_world.json").resolve()),
        "per_camera_auto_bbox_dir": str((output_dir / "per_camera_auto_bbox").resolve()),
        "final_compare_source": "pass2" if refinement["pass2_refinement_valid"] else "pass1",
        "orbit_path_point_count": int(len(orbit_path_points)),
        "object_roi_bounds": {
            "min": scene["object_roi_bounds"]["min"].tolist(),
            "max": scene["object_roi_bounds"]["max"].tolist(),
        },
        "render_bounds": {
            "min": scene["render_bounds_min"].tolist(),
            "max": scene["render_bounds_max"].tolist(),
        },
        "render_point_count": {
            "native": int(len(scene["native_render_points"])),
            "ffs": int(len(scene["ffs_render_points"])),
        },
        "object_point_count": {
            "native": int(len(scene["native_object_points"])),
            "ffs": int(len(scene["ffs_object_points"])),
        },
        "context_point_count": {
            "native": int(len(scene["native_context_points"])),
            "ffs": int(len(scene["ffs_context_points"])),
        },
        "crop_metadata": scene["crop_metadata"],
        "support_metrics_path": str(support_metrics_path),
        "source_metrics_path": str(source_metrics_path),
        "mismatch_metrics_path": str(mismatch_metrics_path),
        "source_legend_path": str(source_legend_path),
        "compare_debug_metrics_path": str(compare_debug_metrics_path),
        "compare_debug_metrics_root_path": str(compare_debug_metrics_root_path),
        "outputs": output_files,
        "renderer_requested": renderer,
        "renderer_used": {
            "overview": overview_state["renderer_used"],
            **renderer_used_by_output,
        },
        "native_stats": raw_scene["native_stats"],
        "ffs_stats": raw_scene["ffs_stats"],
    }
    (output_dir / "turntable_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "output_dir": str(output_dir),
        "metadata": metadata,
    }
