from __future__ import annotations

from typing import Any

import numpy as np


def normalize_vector(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    vec = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-6:
        return np.asarray(fallback, dtype=np.float32)
    return vec / norm


def compute_bounds(point_sets: list[np.ndarray], *, fallback_center: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    points = [np.asarray(item, dtype=np.float32) for item in point_sets if len(item) > 0]
    if not points:
        center = np.zeros((3,), dtype=np.float32) if fallback_center is None else np.asarray(fallback_center, dtype=np.float32)
        return center - 1.0, center + 1.0
    stacked = np.concatenate(points, axis=0)
    return stacked.min(axis=0).astype(np.float32), stacked.max(axis=0).astype(np.float32)


def project_vector_to_plane(vector: np.ndarray, axis: np.ndarray) -> np.ndarray:
    vec = np.asarray(vector, dtype=np.float32)
    normal = normalize_vector(axis, np.array([0.0, 0.0, 1.0], dtype=np.float32))
    return vec - normal * float(vec @ normal)


def compute_view_config(bounds_min: np.ndarray, bounds_max: np.ndarray, view_name: str = "oblique") -> dict[str, Any]:
    center = (bounds_min + bounds_max) * 0.5
    extents = np.maximum(bounds_max - bounds_min, 1e-6)
    radius = float(np.linalg.norm(extents))
    if view_name == "top":
        camera_position = center + radius * np.array([0.0, 0.0, 1.8], dtype=np.float32)
        up = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    elif view_name == "side":
        camera_position = center + radius * np.array([0.0, -1.8, 0.35], dtype=np.float32)
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        azimuth = np.deg2rad(35.0)
        elevation = np.deg2rad(25.0)
        camera_position = center + radius * np.array(
            [
                np.cos(elevation) * np.cos(azimuth),
                np.cos(elevation) * np.sin(azimuth),
                np.sin(elevation),
            ],
            dtype=np.float32,
        )
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return {
        "view_name": view_name,
        "label": view_name.title(),
        "center": center.astype(np.float32),
        "camera_position": camera_position.astype(np.float32),
        "up": up,
        "radius": radius,
    }


def build_camera_pose_view_configs(
    *,
    c2w_list: list[np.ndarray],
    serial_numbers: list[str],
    focus_point: np.ndarray,
    view_distance_scale: float,
    target_distance: float | None = None,
) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for camera_idx, (serial, c2w) in enumerate(zip(serial_numbers, c2w_list, strict=False)):
        transform = np.asarray(c2w, dtype=np.float32).reshape(4, 4)
        original_camera_position = transform[:3, 3]
        direction = normalize_vector(original_camera_position - focus_point, np.array([0.0, 0.0, 1.0], dtype=np.float32))
        if target_distance is None:
            distance = max(1e-3, float(np.linalg.norm(original_camera_position - focus_point)) * float(view_distance_scale))
        else:
            distance = max(1e-3, float(target_distance) * float(view_distance_scale))
        camera_position = np.asarray(focus_point, dtype=np.float32) + direction * distance
        up_hint = -transform[:3, 1]
        up = normalize_vector(up_hint, np.array([0.0, 0.0, 1.0], dtype=np.float32))
        configs.append(
            {
                "view_name": f"cam{camera_idx}",
                "label": f"Cam{camera_idx} | {serial}",
                "camera_idx": camera_idx,
                "serial": serial,
                "center": np.asarray(focus_point, dtype=np.float32),
                "camera_position": np.asarray(camera_position, dtype=np.float32),
                "original_camera_position": np.asarray(original_camera_position, dtype=np.float32),
                "up": up,
                "radius": float(np.linalg.norm(camera_position - focus_point)),
            }
        )
    return configs


def build_original_camera_view_configs(
    *,
    c2w_list: list[np.ndarray],
    serial_numbers: list[str],
    look_distance: float = 1.0,
    camera_ids: list[int] | None = None,
) -> list[dict[str, Any]]:
    if len(serial_numbers) != len(c2w_list):
        raise ValueError("serial_numbers must match c2w_list length.")
    selected_ids = list(range(len(c2w_list))) if camera_ids is None else [int(idx) for idx in camera_ids]
    if float(look_distance) <= 1e-6:
        raise ValueError("look_distance must be positive.")

    configs: list[dict[str, Any]] = []
    for camera_idx in selected_ids:
        if camera_idx < 0 or camera_idx >= len(c2w_list):
            raise ValueError(f"camera_idx out of range: {camera_idx}")
        transform = np.asarray(c2w_list[camera_idx], dtype=np.float32).reshape(4, 4)
        camera_position = transform[:3, 3].astype(np.float32)
        right = normalize_vector(transform[:3, 0], np.array([1.0, 0.0, 0.0], dtype=np.float32))
        up = normalize_vector(-transform[:3, 1], np.array([0.0, 0.0, 1.0], dtype=np.float32))
        forward = normalize_vector(transform[:3, 2], np.array([0.0, 0.0, 1.0], dtype=np.float32))
        center = camera_position + forward * float(look_distance)
        configs.append(
            {
                "view_name": f"cam{camera_idx}",
                "label": f"Cam{camera_idx} | {serial_numbers[camera_idx]}",
                "camera_idx": int(camera_idx),
                "serial": str(serial_numbers[camera_idx]),
                "center": center.astype(np.float32),
                "camera_position": camera_position.astype(np.float32),
                "right": right.astype(np.float32),
                "up": up.astype(np.float32),
                "forward": forward.astype(np.float32),
                "look_distance": float(look_distance),
                "radius": float(look_distance),
            }
        )
    return configs


def build_orbit_basis(
    *,
    camera_poses: list[dict[str, Any]],
    focus_point: np.ndarray,
    orbit_axis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    focus = np.asarray(focus_point, dtype=np.float32)
    axis = normalize_vector(orbit_axis, np.array([0.0, 0.0, 1.0], dtype=np.float32))
    basis_x = None
    for pose in camera_poses:
        projected = project_vector_to_plane(np.asarray(pose["position"], dtype=np.float32) - focus, axis)
        if float(np.linalg.norm(projected)) > 1e-6:
            basis_x = normalize_vector(projected, np.array([1.0, 0.0, 0.0], dtype=np.float32))
            break
    if basis_x is None:
        fallback = project_vector_to_plane(np.array([1.0, 0.0, 0.0], dtype=np.float32), axis)
        if float(np.linalg.norm(fallback)) <= 1e-6:
            fallback = project_vector_to_plane(np.array([0.0, 1.0, 0.0], dtype=np.float32), axis)
        basis_x = normalize_vector(fallback, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    basis_y = normalize_vector(np.cross(axis, basis_x), np.array([0.0, 1.0, 0.0], dtype=np.float32))
    return basis_x, basis_y


def compute_crop_corners(bounds_min: np.ndarray, bounds_max: np.ndarray) -> np.ndarray:
    min_corner = np.asarray(bounds_min, dtype=np.float32)
    max_corner = np.asarray(bounds_max, dtype=np.float32)
    corners: list[np.ndarray] = []
    for x_value in (min_corner[0], max_corner[0]):
        for y_value in (min_corner[1], max_corner[1]):
            for z_value in (min_corner[2], max_corner[2]):
                corners.append(np.array([x_value, y_value, z_value], dtype=np.float32))
    return np.stack(corners, axis=0)


def wrap_angle_deg(angle_deg: float) -> float:
    return (float(angle_deg) + 180.0) % 360.0 - 180.0


def estimate_orbit_axis(camera_poses: list[dict[str, Any]]) -> np.ndarray:
    if not camera_poses:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    up_stack = np.stack([np.asarray(pose["up"], dtype=np.float32) for pose in camera_poses], axis=0)
    axis = up_stack.mean(axis=0)
    return normalize_vector(axis, np.array([0.0, 0.0, 1.0], dtype=np.float32))


def rotate_vector_around_axis(vector: np.ndarray, axis: np.ndarray, angle_deg: float) -> np.ndarray:
    vec = np.asarray(vector, dtype=np.float32)
    axis = normalize_vector(axis, np.array([0.0, 0.0, 1.0], dtype=np.float32))
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
                    "up": normalize_vector(rotated_up, orbit_axis),
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
    axis = normalize_vector(orbit_axis, np.array([0.0, 0.0, 1.0], dtype=np.float32))
    if basis_x is None or basis_y is None:
        basis_x, basis_y = build_orbit_basis(
            camera_poses=camera_poses,
            focus_point=focus,
            orbit_axis=axis,
        )
    azimuths: dict[int, float] = {}
    for pose in camera_poses:
        offset = np.asarray(pose["position"], dtype=np.float32) - focus
        planar = project_vector_to_plane(offset, axis)
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
        return {
            "start_deg": float(start),
            "end_deg": float(start + span),
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
    axis = normalize_vector(orbit_axis, np.array([0.0, 0.0, 1.0], dtype=np.float32))
    basis_x, basis_y = build_orbit_basis(
        camera_poses=camera_poses,
        focus_point=focus,
        orbit_axis=axis,
    )

    crop_corners = compute_crop_corners(bounds_min, bounds_max)
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
    for pose in camera_poses:
        offset = np.asarray(pose["position"], dtype=np.float32) - focus
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
                np.argmin([abs(wrap_angle_deg(angle_deg - float(start_azimuth_deg))) for angle_deg in azimuth_sequence])
            )
            azimuth_sequence = azimuth_sequence[start_idx:] + azimuth_sequence[:start_idx]
    else:
        azimuth_sequence = [float(start_azimuth_deg + item) for item in generate_orbit_angles(num_orbit_steps=num_orbit_steps, orbit_degrees=orbit_degrees)]

    orbit_steps: list[dict[str, Any]] = []
    orbit_path_points: list[np.ndarray] = []
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
        for camera_idx, camera_angle_deg in camera_reference_azimuths_deg.items():
            delta_deg = abs(wrap_angle_deg(current_angle_deg - float(camera_angle_deg)))
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
        "camera_reference_azimuths_deg": camera_reference_azimuths_deg,
        "coverage_arc": coverage_arc,
    }
