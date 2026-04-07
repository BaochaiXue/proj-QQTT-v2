from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .camera_frusta import build_camera_frustum_geometry, collect_camera_geometry_points, extract_camera_poses
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
    load_case_frame_cloud,
    load_case_metadata,
    project_world_points_to_image,
    render_point_cloud,
    resolve_case_dirs,
    write_video,
)


DEFAULT_CAMERA_IDS = [0, 1, 2]
ORBIT_MODES = ("object_centered_360", "camera_neighborhood")
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
) -> dict[str, Any]:
    native_points, native_colors, native_stats = load_case_frame_cloud(
        case_dir=selection["native_case_dir"],
        metadata=selection["native_metadata"],
        frame_idx=selection["native_frame_idx"],
        depth_source="realsense",
        use_float_ffs_depth_when_available=False,
        voxel_size=voxel_size,
        max_points_per_camera=max_points_per_camera,
        depth_min_m=depth_min_m,
        depth_max_m=depth_max_m,
    )
    ffs_points, ffs_colors, ffs_stats = load_case_frame_cloud(
        case_dir=selection["ffs_case_dir"],
        metadata=selection["ffs_metadata"],
        frame_idx=selection["ffs_frame_idx"],
        depth_source="ffs",
        use_float_ffs_depth_when_available=use_float_ffs_depth_when_available,
        voxel_size=voxel_size,
        max_points_per_camera=max_points_per_camera,
        depth_min_m=depth_min_m,
        depth_max_m=depth_max_m,
    )
    return {
        "native_points": native_points,
        "native_colors": native_colors,
        "native_stats": native_stats,
        "ffs_points": ffs_points,
        "ffs_colors": ffs_colors,
        "ffs_stats": ffs_stats,
    }


def build_single_frame_scene(
    *,
    native_points: np.ndarray,
    native_colors: np.ndarray,
    ffs_points: np.ndarray,
    ffs_colors: np.ndarray,
    focus_mode: str,
    scene_crop_mode: str,
    crop_margin_xy: float,
    crop_min_z: float,
    crop_max_z: float,
    manual_xyz_roi: dict[str, float] | None,
) -> dict[str, Any]:
    raw_bounds_min, raw_bounds_max = _compute_bounds([native_points, ffs_points])
    focus_point = estimate_focus_point(
        [native_points, ffs_points],
        bounds_min=raw_bounds_min,
        bounds_max=raw_bounds_max,
        focus_mode=focus_mode,
    )
    crop_bounds = compute_scene_crop_bounds(
        [native_points, ffs_points],
        focus_point=focus_point,
        scene_crop_mode=scene_crop_mode,
        crop_margin_xy=float(crop_margin_xy),
        crop_min_z=float(crop_min_z),
        crop_max_z=float(crop_max_z),
        manual_xyz_roi=manual_xyz_roi,
    )
    cropped_native_points, cropped_native_colors = crop_points_to_bounds(native_points, native_colors, crop_bounds)
    cropped_ffs_points, cropped_ffs_colors = crop_points_to_bounds(ffs_points, ffs_colors, crop_bounds)

    cropped_point_sets = [item for item in (cropped_native_points, cropped_ffs_points) if len(item) > 0]
    bounds_min, bounds_max = _compute_bounds(
        cropped_point_sets if cropped_point_sets else [native_points, ffs_points],
        fallback_center=focus_point,
    )
    refined_focus = estimate_focus_point(
        cropped_point_sets if cropped_point_sets else [native_points, ffs_points],
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        focus_mode=focus_mode,
    )
    scalar_bounds = {
        "height": (float(bounds_min[2]), float(bounds_max[2])),
        "depth": (0.0, max(float(np.linalg.norm(bounds_max - bounds_min)) * 2.0, 1.0)),
    }
    return {
        "native_points": cropped_native_points,
        "native_colors": cropped_native_colors,
        "ffs_points": cropped_ffs_points,
        "ffs_colors": cropped_ffs_colors,
        "focus_point": refined_focus.astype(np.float32),
        "crop_bounds": {
            "min": np.asarray(crop_bounds["min"], dtype=np.float32),
            "max": np.asarray(crop_bounds["max"], dtype=np.float32),
        },
        "bounds_min": bounds_min,
        "bounds_max": bounds_max,
        "scalar_bounds": scalar_bounds,
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

    camera_planar_radii: list[float] = []
    camera_axis_heights: list[float] = []
    camera_reference_azimuths_deg: dict[int, float] = {}
    for pose in camera_poses:
        offset = np.asarray(pose["position"], dtype=np.float32) - focus
        planar = _project_vector_to_plane(offset, axis)
        camera_planar_radii.append(float(np.linalg.norm(planar)))
        camera_axis_heights.append(float(offset @ axis))
        if float(np.linalg.norm(planar)) > 1e-6:
            camera_reference_azimuths_deg[int(pose["camera_idx"])] = float(
                np.rad2deg(np.arctan2(float(planar @ basis_y), float(planar @ basis_x)))
            )

    orbit_radius = max(crop_planar_radius * float(orbit_radius_scale), 0.25)
    median_camera_height = float(np.median(camera_axis_heights)) if camera_axis_heights else crop_axis_extent * 1.4
    orbit_height = max(median_camera_height + crop_axis_extent * float(view_height_offset), crop_axis_extent * 0.9, 0.12)

    start_azimuth_deg = camera_reference_azimuths_deg.get(0)
    if start_azimuth_deg is None and camera_reference_azimuths_deg:
        start_azimuth_deg = float(next(iter(camera_reference_azimuths_deg.values())))
    if start_azimuth_deg is None:
        start_azimuth_deg = 0.0

    orbit_steps: list[dict[str, Any]] = []
    orbit_path_points: list[np.ndarray] = []
    camera_azimuths = {
        camera_idx: angle_deg
        for camera_idx, angle_deg in camera_reference_azimuths_deg.items()
    }
    for step_idx, angle_deg in enumerate(generate_orbit_angles(num_orbit_steps=num_orbit_steps, orbit_degrees=orbit_degrees)):
        azimuth_deg = float(start_azimuth_deg + angle_deg)
        azimuth_rad = np.deg2rad(azimuth_deg)
        planar_offset = basis_x * (np.cos(azimuth_rad) * orbit_radius) + basis_y * (np.sin(azimuth_rad) * orbit_radius)
        camera_position = (focus + planar_offset + axis * orbit_height).astype(np.float32)
        current_angle_deg = float(np.rad2deg(np.arctan2(float(planar_offset @ basis_y), float(planar_offset @ basis_x))))
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
            "angle_deg": float(angle_deg),
            "azimuth_deg": float(current_angle_deg),
            "center": focus.copy(),
            "camera_position": camera_position,
            "anchor_camera_position": focus.copy(),
            "up": axis.copy(),
            "radius": float(np.linalg.norm(camera_position - focus)),
            "orbit_axis": axis.copy(),
            "orbit_angle_deg": float(angle_deg),
            "color_bgr": (255, 255, 255),
            "nearest_camera_idx": nearest_camera_idx,
            "nearest_camera_delta_deg": nearest_camera_delta_deg,
        }
        orbit_steps.append(
            {
                "step_idx": int(step_idx),
                "angle_deg": float(angle_deg),
                "view_config": view_config,
                "view_configs": [view_config],
            }
        )
        orbit_path_points.append(camera_position)

    orbit_path = np.stack(orbit_path_points, axis=0).astype(np.float32) if orbit_path_points else np.empty((0, 3), dtype=np.float32)
    return {
        "orbit_steps": orbit_steps,
        "orbit_path": orbit_path,
        "orbit_axis": axis,
        "orbit_basis_x": basis_x,
        "orbit_basis_y": basis_y,
        "orbit_radius": float(orbit_radius),
        "orbit_height": float(orbit_height),
        "start_azimuth_deg": float(start_azimuth_deg),
        "camera_reference_azimuths_deg": camera_azimuths,
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
) -> list[dict[str, str]]:
    outputs = [
        {
            "name": "geom",
            "render_mode": str(geom_render_mode),
            "video_name": "orbit_compare_geom.mp4",
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
                "sheet_name": "turntable_keyframes_rgb.png",
                "frames_dir_name": "frames_rgb",
            }
        )
    return outputs


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
    crop_bounds: dict[str, np.ndarray] | None = None,
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
        path_uv = np.rint(projected_path["uv"][projected_path["valid"]]).astype(np.int32)
        if len(path_uv) >= 2:
            cv2.polylines(canvas, [path_uv.reshape(-1, 1, 2)], True, (170, 170, 170), 2, cv2.LINE_AA)

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
    crop_bounds: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    geometry_points = collect_camera_geometry_points(camera_geometries)
    bounds_min, bounds_max = _compute_bounds([scene_points, geometry_points, np.asarray(focus_point, dtype=np.float32).reshape(1, 3)])
    overview_view = compute_view_config(bounds_min, bounds_max, view_name="oblique")
    focus = np.asarray(focus_point, dtype=np.float32)
    direction = _normalize_vector(
        np.asarray(overview_view["camera_position"], dtype=np.float32) - np.asarray(overview_view["center"], dtype=np.float32),
        np.array([1.0, 1.0, 1.0], dtype=np.float32),
    )
    overview_radius = max(float(np.linalg.norm(bounds_max - bounds_min)) * 1.1, float(np.linalg.norm(overview_view["camera_position"] - overview_view["center"])))
    overview_view["center"] = focus
    overview_view["camera_position"] = (focus + direction * overview_radius).astype(np.float32)
    overview_projection_mode = "orthographic"
    overview_ortho_scale = estimate_ortho_scale(
        [scene_points, geometry_points],
        view_config=overview_view,
    )
    base_image, renderer_used = render_point_cloud(
        scene_points,
        scene_colors,
        renderer=renderer,
        view_config=overview_view,
        render_mode=render_mode,
        scalar_bounds=scalar_bounds,
        width=960,
        height=720,
        point_radius_px=max(2, int(point_radius_px)),
        supersample_scale=max(1, int(supersample_scale)),
        projection_mode=overview_projection_mode,
        ortho_scale=overview_ortho_scale,
    )
    base_overlay = draw_scene_overlays(
        base_image,
        camera_geometries=camera_geometries,
        view_config=overview_view,
        projection_mode=overview_projection_mode,
        ortho_scale=overview_ortho_scale,
        focus_point=focus,
        orbit_path_points=orbit_path_points,
        crop_bounds=crop_bounds,
    )
    return {
        "image": base_overlay,
        "renderer_used": renderer_used,
        "view_config": overview_view,
        "projection_mode": overview_projection_mode,
        "ortho_scale": float(overview_ortho_scale),
    }


def render_overview_inset(
    overview_state: dict[str, Any],
    *,
    current_views: list[dict[str, Any]],
    inset_size: tuple[int, int] = (560, 320),
) -> np.ndarray:
    overlay = draw_scene_overlays(
        overview_state["image"],
        camera_geometries=[],
        view_config=overview_state["view_config"],
        projection_mode=overview_state["projection_mode"],
        ortho_scale=overview_state["ortho_scale"],
        focus_point=None,
        current_views=current_views,
        orbit_path_points=None,
        crop_bounds=None,
    )
    return cv2.resize(overlay, inset_size, interpolation=cv2.INTER_AREA)


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

    inset = np.asarray(overview_inset, dtype=np.uint8)
    overview_h, overview_w = inset.shape[:2]
    footer_h = max(overview_h + 24, 340)
    footer = np.zeros((footer_h, main_body.shape[1], 3), dtype=np.uint8)
    footer[:] = (16, 18, 22)
    cv2.putText(footer, "Overview", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.86, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(footer, "Real camera frusta, ROI crop, orbit ring, and current orbit camera", (18, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(footer, "The orbit path is identical for Native and FFS.", (18, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (220, 220, 220), 1, cv2.LINE_AA)
    max_overview_w = max(180, footer.shape[1] - 220)
    max_overview_h = footer_h - 28
    scale = min(1.0, float(max_overview_w) / max(1, overview_w), float(max_overview_h) / max(1, overview_h))
    if scale < 1.0:
        overview_w = max(160, int(round(overview_w * scale)))
        overview_h = max(120, int(round(overview_h * scale)))
        inset = cv2.resize(inset, (overview_w, overview_h), interpolation=cv2.INTER_AREA)
    x0 = footer.shape[1] - overview_w - 20
    y0 = max(14, (footer_h - overview_h) // 2)
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
    write_keyframe_sheet: bool = True,
    num_orbit_steps: int = 72,
    orbit_degrees: float = 360.0,
    camera_ids: list[int] | None = None,
    scene_crop_mode: str = "auto_table_bbox",
    focus_mode: str = "table",
    crop_margin_xy: float = 0.12,
    crop_min_z: float = -0.15,
    crop_max_z: float = 0.35,
    roi_x_min: float | None = None,
    roi_x_max: float | None = None,
    roi_y_min: float | None = None,
    roi_y_max: float | None = None,
    roi_z_min: float | None = None,
    roi_z_max: float | None = None,
    projection_mode: str = "perspective",
    point_radius_px: int = 4,
    supersample_scale: int = 3,
    voxel_size: float | None = None,
    max_points_per_camera: int | None = 50000,
    depth_min_m: float = 0.2,
    depth_max_m: float = 1.5,
    use_float_ffs_depth_when_available: bool = True,
    fps: int = 8,
    orbit_mode: str = "object_centered_360",
    layout_mode: str = "side_by_side_large",
    orbit_radius_scale: float = 1.9,
    view_height_offset: float = 0.0,
    render_both_modes: bool = True,
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
    raw_scene = load_single_frame_compare_clouds(
        selection,
        voxel_size=voxel_size,
        max_points_per_camera=max_points_per_camera,
        depth_min_m=depth_min_m,
        depth_max_m=depth_max_m,
        use_float_ffs_depth_when_available=use_float_ffs_depth_when_available,
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
    scene = build_single_frame_scene(
        native_points=raw_scene["native_points"],
        native_colors=raw_scene["native_colors"],
        ffs_points=raw_scene["ffs_points"],
        ffs_colors=raw_scene["ffs_colors"],
        focus_mode=focus_mode,
        scene_crop_mode=scene_crop_mode,
        crop_margin_xy=crop_margin_xy,
        crop_min_z=crop_min_z,
        crop_max_z=crop_max_z,
        manual_xyz_roi=manual_xyz_roi,
    )

    camera_poses = extract_camera_poses(
        selection["native_c2w"],
        serial_numbers=selection["serial_numbers"],
        camera_ids=selection["camera_ids"],
    )
    orbit_axis = estimate_orbit_axis(camera_poses)
    if layout_mode == "side_by_side_large" and orbit_mode != "object_centered_360":
        raise ValueError("side_by_side_large currently requires orbit_mode=object_centered_360.")
    if layout_mode == "camera_neighborhood_grid" and orbit_mode != "camera_neighborhood":
        raise ValueError("camera_neighborhood_grid currently requires orbit_mode=camera_neighborhood.")

    if orbit_mode == "object_centered_360":
        object_orbit = build_object_centered_orbit_views(
            camera_poses=camera_poses,
            focus_point=scene["focus_point"],
            bounds_min=scene["bounds_min"],
            bounds_max=scene["bounds_max"],
            orbit_axis=orbit_axis,
            num_orbit_steps=num_orbit_steps,
            orbit_degrees=orbit_degrees,
            orbit_radius_scale=orbit_radius_scale,
            view_height_offset=view_height_offset,
        )
        orbit_steps = object_orbit["orbit_steps"]
        orbit_path_points = object_orbit["orbit_path"]
        orbit_radius = object_orbit["orbit_radius"]
        orbit_height = object_orbit["orbit_height"]
        start_azimuth_deg = object_orbit["start_azimuth_deg"]
        camera_reference_azimuths_deg = object_orbit["camera_reference_azimuths_deg"]
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
        orbit_radius = float(np.median([step["view_configs"][0]["radius"] for step in orbit_steps])) if orbit_steps else 0.0
        orbit_height = 0.0
        start_azimuth_deg = 0.0
        camera_reference_azimuths_deg = {}

    scene_diagonal = float(np.linalg.norm(scene["bounds_max"] - scene["bounds_min"]))
    frustum_scale = max(0.06, scene_diagonal * 0.14)
    camera_geometries = [
        build_camera_frustum_geometry(pose, frustum_scale=frustum_scale)
        for pose in camera_poses
    ]

    if len(scene["native_points"]) > 0 and len(scene["ffs_points"]) > 0:
        overview_cloud_points = np.concatenate([scene["native_points"], scene["ffs_points"]], axis=0)
        overview_cloud_colors = np.concatenate([scene["native_colors"], scene["ffs_colors"]], axis=0)
    elif len(scene["native_points"]) > 0:
        overview_cloud_points = scene["native_points"]
        overview_cloud_colors = scene["native_colors"]
    else:
        overview_cloud_points = scene["ffs_points"]
        overview_cloud_colors = scene["ffs_colors"]
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
        crop_bounds=scene["crop_bounds"],
    )
    overview_image_path = output_dir / "scene_overview_with_cameras.png"
    cv2.imwrite(str(overview_image_path), overview_state["image"])

    output_specs = build_render_output_specs(
        geom_render_mode=render_mode,
        render_both_modes=render_both_modes,
    )
    frame_paths_by_output: dict[str, list[Path]] = {spec["name"]: [] for spec in output_specs}
    frames_dir_by_output: dict[str, Path] = {}
    board_images_by_output: dict[str, list[np.ndarray]] = {spec["name"]: [] for spec in output_specs}
    renderer_used_by_output: dict[str, dict[str, str]] = {spec["name"]: {} for spec in output_specs}
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
        overview_inset = render_overview_inset(
            overview_state,
            current_views=current_views,
            inset_size=(560, 320) if layout_mode == "side_by_side_large" else (420, 260),
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
                        [scene["native_points"], scene["ffs_points"]],
                        view_config=view_config,
                    )
                native_render, native_renderer_used = render_point_cloud(
                    scene["native_points"],
                    scene["native_colors"],
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
                    scene["ffs_points"],
                    scene["ffs_colors"],
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
                        f"{mode_name} | render={mode_render} | orbit={orbit_step['angle_deg']:+.1f} deg | proj={projection_mode} | crop={scene_crop_mode}",
                    ],
                    native_image=native_images[0],
                    ffs_image=ffs_images[0],
                    overview_inset=overview_inset,
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
        sheet_path = output_dir / output_spec["sheet_name"]
        if write_mp4:
            write_video(video_path, frame_paths_by_output[mode_name], fps)
        if write_keyframe_sheet and board_images_by_output[mode_name]:
            sheet = compose_keyframe_sheet(board_images_by_output[mode_name])
            cv2.imwrite(str(sheet_path), sheet)
        output_files[mode_name] = {
            "frames_dir": str(frames_dir_by_output[mode_name]),
            "video_path": str(video_path) if write_mp4 else None,
            "sheet_path": str(sheet_path) if write_keyframe_sheet else None,
        }

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
        "orbit_radius": float(orbit_radius),
        "orbit_radius_scale": float(orbit_radius_scale),
        "orbit_height": float(orbit_height),
        "view_height_offset": float(view_height_offset),
        "start_azimuth_deg": float(start_azimuth_deg),
        "camera_reference_azimuths_deg": camera_reference_azimuths_deg,
        "point_radius_px": int(point_radius_px),
        "supersample_scale": int(supersample_scale),
        "depth_min_m": float(depth_min_m),
        "depth_max_m": float(depth_max_m),
        "voxel_size": voxel_size,
        "max_points_per_camera": max_points_per_camera,
        "use_float_ffs_depth_when_available": bool(use_float_ffs_depth_when_available),
        "scene_overview_with_cameras": str(overview_image_path),
        "orbit_path_point_count": int(len(orbit_path_points)),
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
