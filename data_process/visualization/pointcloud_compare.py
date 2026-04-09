from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .io_artifacts import write_gif, write_ply_ascii, write_video
from .io_case import (
    choose_depth_stream,
    decode_depth_to_meters,
    depth_to_camera_points,
    get_case_intrinsics,
    get_depth_scale_list,
    get_frame_count,
    load_case_frame_camera_clouds,
    load_case_frame_cloud,
    load_case_frame_cloud_with_sources,
    load_case_metadata,
    resolve_case_dirs,
    select_frame_indices,
    transform_points,
    voxel_downsample,
)
from .layouts import compose_grid_2x3, overlay_panel_label
from .renderers.fallback import (
    apply_image_flip,
    estimate_ortho_scale,
    look_at_view_matrix,
    project_world_points_to_image,
    rasterize_point_cloud_view,
    render_point_cloud,
    render_point_cloud_fallback,
)
from .roi import compute_scene_crop_bounds, crop_points_to_bounds, estimate_focus_point
from .views import build_camera_pose_view_configs, compute_view_config


RENDER_MODES = (
    "color_by_rgb",
    "color_by_depth",
    "color_by_height",
    "color_by_normals",
    "neutral_gray_shaded",
)
VIEW_NAMES = ("oblique", "top", "side")
VIEW_MODES = ("fixed", "camera_poses_table_focus")
FOCUS_MODES = ("none", "table")
LAYOUT_MODES = ("pair", "grid_2x3")
SCENE_CROP_MODES = ("none", "auto_table_bbox", "auto_object_bbox", "manual_xyz_roi")
PROJECTION_MODES = ("perspective", "orthographic")
IMAGE_FLIP_MODES = ("none", "vertical", "horizontal", "both")



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


def estimate_focus_point(
    point_sets: list[np.ndarray],
    *,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    focus_mode: str,
) -> np.ndarray:
    default_center = ((bounds_min + bounds_max) * 0.5).astype(np.float32)
    if focus_mode == "none":
        return default_center

    points = [np.asarray(item, dtype=np.float32) for item in point_sets if len(item) > 0]
    if not points:
        return default_center

    stacked = np.concatenate(points, axis=0)
    if len(stacked) == 0:
        return default_center

    if focus_mode == "table":
        z_low = float(np.percentile(stacked[:, 2], 20))
        z_high = float(np.percentile(stacked[:, 2], 65))
        band = stacked[(stacked[:, 2] >= z_low) & (stacked[:, 2] <= z_high)]
        if len(band) < 128:
            band = stacked
        return np.asarray(
            [
                float(np.median(band[:, 0])),
                float(np.median(band[:, 1])),
                float(np.median(band[:, 2])),
            ],
            dtype=np.float32,
        )

    raise ValueError(f"Unsupported focus_mode: {focus_mode}")


def compute_scene_crop_bounds(
    point_sets: list[np.ndarray],
    *,
    focus_point: np.ndarray,
    scene_crop_mode: str,
    crop_margin_xy: float,
    crop_min_z: float,
    crop_max_z: float,
    manual_xyz_roi: dict[str, float] | None = None,
    object_seed_point_sets: list[np.ndarray] | None = None,
    object_height_min: float = 0.02,
    object_height_max: float = 0.30,
    object_component_mode: str = "graph_union",
    object_component_topk: int = 2,
) -> dict[str, np.ndarray]:
    points = [np.asarray(item, dtype=np.float32) for item in point_sets if len(item) > 0]
    if not points:
        focus = np.asarray(focus_point, dtype=np.float32)
        return {
            "mode": scene_crop_mode,
            "min": focus - np.array([1.0, 1.0, 1.0], dtype=np.float32),
            "max": focus + np.array([1.0, 1.0, 1.0], dtype=np.float32),
        }

    stacked = np.concatenate(points, axis=0)
    full_min = stacked.min(axis=0)
    full_max = stacked.max(axis=0)

    if scene_crop_mode == "none":
        return {"mode": scene_crop_mode, "min": full_min.astype(np.float32), "max": full_max.astype(np.float32)}

    if scene_crop_mode == "manual_xyz_roi":
        if manual_xyz_roi is None:
            raise ValueError("manual_xyz_roi crop mode requires explicit roi bounds.")
        crop_min = np.array(
            [manual_xyz_roi["x_min"], manual_xyz_roi["y_min"], manual_xyz_roi["z_min"]],
            dtype=np.float32,
        )
        crop_max = np.array(
            [manual_xyz_roi["x_max"], manual_xyz_roi["y_max"], manual_xyz_roi["z_max"]],
            dtype=np.float32,
        )
        if np.any(crop_min >= crop_max):
            raise ValueError(f"Invalid manual_xyz_roi bounds: {manual_xyz_roi}")
        return {"mode": scene_crop_mode, "min": crop_min, "max": crop_max}

    if scene_crop_mode == "auto_object_bbox":
        from .object_roi import estimate_object_roi_bounds, fit_dominant_table_plane

        table_bounds = compute_scene_crop_bounds(
            point_sets,
            focus_point=focus_point,
            scene_crop_mode="auto_table_bbox",
            crop_margin_xy=crop_margin_xy,
            crop_min_z=crop_min_z,
            crop_max_z=crop_max_z,
            manual_xyz_roi=None,
        )
        table_valid = (
            np.all(stacked >= table_bounds["min"][None, :], axis=1)
            & np.all(stacked <= table_bounds["max"][None, :], axis=1)
        )
        table_points = stacked[table_valid]
        seed_points = None
        if object_seed_point_sets:
            seed_sets = [np.asarray(item, dtype=np.float32) for item in object_seed_point_sets if len(item) > 0]
            if seed_sets:
                seed_stacked = np.concatenate(seed_sets, axis=0)
                seed_valid = (
                    np.all(seed_stacked >= table_bounds["min"][None, :], axis=1)
                    & np.all(seed_stacked <= table_bounds["max"][None, :], axis=1)
                )
                seed_points = seed_stacked[seed_valid]
        if seed_points is not None and len(seed_points) >= 32:
            seed_object_roi = estimate_object_roi_bounds(
                seed_points,
                fallback_bounds=table_bounds,
                full_bounds={"min": full_min.astype(np.float32), "max": full_max.astype(np.float32)},
                plane_reference_points=table_points if len(table_points) > 0 else stacked,
                object_height_min=float(object_height_min),
                object_height_max=max(0.40, float(object_height_max)),
                object_component_mode=object_component_mode,
                object_component_topk=int(object_component_topk),
                roi_margin_xy=max(0.02, float(crop_margin_xy) * 0.45),
                roi_margin_z=max(0.015, abs(float(crop_max_z) - float(crop_min_z)) * 0.08),
            )
            seed_object_roi["seed_bbox_used"] = True
            seed_object_roi["seed_source_point_count"] = int(len(seed_points))
            return seed_object_roi
        object_roi = estimate_object_roi_bounds(
            seed_points if seed_points is not None and len(seed_points) > 0 else (table_points if len(table_points) > 0 else stacked),
            fallback_bounds=table_bounds,
            full_bounds={"min": full_min.astype(np.float32), "max": full_max.astype(np.float32)},
            plane_reference_points=table_points if len(table_points) > 0 else stacked,
            object_height_min=float(object_height_min),
            object_height_max=float(object_height_max),
            object_component_mode=object_component_mode,
            object_component_topk=int(object_component_topk),
            roi_margin_xy=max(0.02, float(crop_margin_xy) * 0.45),
            roi_margin_z=max(0.015, abs(float(crop_max_z) - float(crop_min_z)) * 0.08),
        )
        return object_roi

    if scene_crop_mode != "auto_table_bbox":
        raise ValueError(f"Unsupported scene_crop_mode: {scene_crop_mode}")

    focus = np.asarray(focus_point, dtype=np.float32)
    z_min = float(focus[2] + crop_min_z)
    z_max = float(focus[2] + crop_max_z)
    band = stacked[(stacked[:, 2] >= z_min) & (stacked[:, 2] <= z_max)]
    if len(band) < 256:
        band = stacked[np.abs(stacked[:, 2] - focus[2]) <= max(abs(crop_min_z), abs(crop_max_z), 0.35)]
    if len(band) < 64:
        band = stacked

    x_min, x_max = np.quantile(band[:, 0], [0.05, 0.95])
    y_min, y_max = np.quantile(band[:, 1], [0.05, 0.95])
    crop_min = np.array(
        [
            float(x_min - crop_margin_xy),
            float(y_min - crop_margin_xy),
            z_min,
        ],
        dtype=np.float32,
    )
    crop_max = np.array(
        [
            float(x_max + crop_margin_xy),
            float(y_max + crop_margin_xy),
            z_max,
        ],
        dtype=np.float32,
    )
    crop_min = np.maximum(crop_min, full_min)
    crop_max = np.minimum(crop_max, full_max)
    return {"mode": scene_crop_mode, "min": crop_min, "max": crop_max}


def crop_points_to_bounds(
    points: np.ndarray,
    colors: np.ndarray,
    crop_bounds: dict[str, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray]:
    if crop_bounds is None or len(points) == 0:
        return points, colors
    crop_min = np.asarray(crop_bounds["min"], dtype=np.float32)
    crop_max = np.asarray(crop_bounds["max"], dtype=np.float32)
    valid = np.all(points >= crop_min[None, :], axis=1) & np.all(points <= crop_max[None, :], axis=1)
    return points[valid], colors[valid]


def _normalize_vector(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    vec = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-6:
        return np.asarray(fallback, dtype=np.float32)
    return vec / norm


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
        direction = _normalize_vector(original_camera_position - focus_point, np.array([0.0, 0.0, 1.0], dtype=np.float32))
        if target_distance is None:
            distance = max(1e-3, float(np.linalg.norm(original_camera_position - focus_point)) * float(view_distance_scale))
        else:
            distance = max(1e-3, float(target_distance) * float(view_distance_scale))
        camera_position = np.asarray(focus_point, dtype=np.float32) + direction * distance
        up_hint = -transform[:3, 1]
        up = _normalize_vector(up_hint, np.array([0.0, 0.0, 1.0], dtype=np.float32))
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


def overlay_panel_label(
    image: np.ndarray,
    *,
    label: str,
    text_color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    canvas = np.asarray(image, dtype=np.uint8).copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1] - 1, 32), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        label,
        (12, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        text_color,
        2,
        cv2.LINE_AA,
    )
    return canvas


def compose_grid_2x3(
    *,
    title: str,
    column_headers: list[str],
    row_headers: list[str],
    native_images: list[np.ndarray],
    ffs_images: list[np.ndarray],
) -> np.ndarray:
    if len(native_images) != 3 or len(ffs_images) != 3:
        raise ValueError("grid_2x3 layout requires exactly 3 native images and 3 ffs images.")

    if len(column_headers) != 3 or len(row_headers) != 2:
        raise ValueError("grid_2x3 layout requires 3 column headers and 2 row headers.")

    panel_h, panel_w = native_images[0].shape[:2]
    row_label_w = 170
    title_h = 42
    header_h = 38
    body = np.zeros((panel_h * 2, row_label_w + panel_w * 3, 3), dtype=np.uint8)

    for row_idx, row_images in enumerate((native_images, ffs_images)):
        y0 = row_idx * panel_h
        body[y0:y0 + panel_h, :row_label_w] = (16, 16, 16)
        header = row_headers[row_idx]
        text_size = cv2.getTextSize(header, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)[0]
        text_x = max(10, (row_label_w - text_size[0]) // 2)
        text_y = y0 + (panel_h + text_size[1]) // 2
        cv2.putText(body, header, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
        for col_idx, image in enumerate(row_images):
            x0 = row_label_w + col_idx * panel_w
            body[y0:y0 + panel_h, x0:x0 + panel_w] = image

    header_bar = np.zeros((header_h, body.shape[1], 3), dtype=np.uint8)
    header_bar[:, :row_label_w] = (24, 24, 24)
    for col_idx, header in enumerate(column_headers):
        x0 = row_label_w + col_idx * panel_w
        header_bar[:, x0:x0 + panel_w] = (24, 24, 24)
        text_size = cv2.getTextSize(header, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)[0]
        text_x = x0 + max(8, (panel_w - text_size[0]) // 2)
        cv2.putText(header_bar, header, (text_x, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    title_bar = np.zeros((title_h, body.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        title_bar,
        title,
        (14, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return np.vstack([title_bar, header_bar, body])


def apply_image_flip(image: np.ndarray, image_flip: str) -> np.ndarray:
    if image_flip == "none":
        return image
    if image_flip == "vertical":
        return cv2.flip(image, 0)
    if image_flip == "horizontal":
        return cv2.flip(image, 1)
    if image_flip == "both":
        return cv2.flip(image, -1)
    raise ValueError(f"Unsupported image_flip: {image_flip}")


def look_at_view_matrix(camera_position: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = center - camera_position
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, forward)
    rotation = np.stack([right, true_up, -forward], axis=0)
    translation = -rotation @ camera_position
    view = np.eye(4, dtype=np.float32)
    view[:3, :3] = rotation
    view[:3, 3] = translation
    return view


def _look_at(camera_position: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
    return look_at_view_matrix(camera_position, center, up)


def _project_view_coordinates(
    xyz: np.ndarray,
    *,
    width: int,
    height: int,
    projection_mode: str,
    ortho_scale: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    if len(xyz) == 0:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32)
    z = -xyz[:, 2]
    if projection_mode == "perspective":
        focal = width / (2.0 * np.tan(np.deg2rad(35.0) / 2.0))
        u = (xyz[:, 0] * focal / z) + width * 0.5
        v = height * 0.5 - (xyz[:, 1] * focal / z)
        return u, v
    if projection_mode == "orthographic":
        scale = float(ortho_scale) if ortho_scale is not None else max(1e-6, float(np.max(np.abs(xyz[:, :2]))) * 1.2)
        u = (xyz[:, 0] / scale) * (width * 0.5) + width * 0.5
        v = height * 0.5 - (xyz[:, 1] / scale) * (height * 0.5)
        return u, v
    raise ValueError(f"Unsupported projection_mode: {projection_mode}")


def project_world_points_to_image(
    points: np.ndarray,
    *,
    view_config: dict[str, Any],
    width: int,
    height: int,
    projection_mode: str,
    ortho_scale: float | None,
) -> dict[str, np.ndarray]:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    uv = np.full((len(points), 2), np.nan, dtype=np.float32)
    xyz_view = np.zeros((len(points), 3), dtype=np.float32)
    valid = np.zeros((len(points),), dtype=bool)
    if len(points) == 0:
        return {
            "uv": uv,
            "xyz_view": xyz_view,
            "valid": valid,
        }

    homogeneous = np.concatenate([points, np.ones((len(points), 1), dtype=np.float32)], axis=1)
    view = look_at_view_matrix(view_config["camera_position"], view_config["center"], view_config["up"])
    camera_points = homogeneous @ view.T
    xyz_view = camera_points[:, :3]
    valid = xyz_view[:, 2] < -1e-6
    if np.any(valid):
        u, v = _project_view_coordinates(
            xyz_view[valid],
            width=width,
            height=height,
            projection_mode=projection_mode,
            ortho_scale=ortho_scale,
        )
        uv[valid, 0] = u
        uv[valid, 1] = v

    return {
        "uv": uv,
        "xyz_view": xyz_view,
        "valid": valid,
    }


def estimate_ortho_scale(
    point_sets: list[np.ndarray],
    *,
    view_config: dict[str, Any],
    margin: float = 1.15,
) -> float:
    points = [np.asarray(item, dtype=np.float32) for item in point_sets if len(item) > 0]
    if not points:
        return 1.0
    stacked = np.concatenate(points, axis=0)
    homogeneous = np.concatenate(
        [stacked.astype(np.float32), np.ones((len(stacked), 1), dtype=np.float32)],
        axis=1,
    )
    view = _look_at(view_config["camera_position"], view_config["center"], view_config["up"])
    camera_points = homogeneous @ view.T
    xyz = camera_points[:, :3]
    valid = xyz[:, 2] < -1e-6
    xyz = xyz[valid]
    if len(xyz) == 0:
        return 1.0
    scale = max(
        float(np.max(np.abs(xyz[:, 0]))),
        float(np.max(np.abs(xyz[:, 1]))),
        1e-3,
    )
    return float(scale * margin)


def _rasterize_view(
    points: np.ndarray,
    colors: np.ndarray,
    *,
    view_config: dict[str, Any],
    width: int,
    height: int,
    projection_mode: str,
    ortho_scale: float | None,
) -> dict[str, np.ndarray]:
    canvas = {
        "rgb": np.zeros((height, width, 3), dtype=np.uint8),
        "depth": np.zeros((height, width), dtype=np.float32),
        "xyz_view": np.zeros((height, width, 3), dtype=np.float32),
        "world_z": np.zeros((height, width), dtype=np.float32),
        "valid": np.zeros((height, width), dtype=bool),
    }
    if len(points) == 0:
        return canvas

    view = _look_at(view_config["camera_position"], view_config["center"], view_config["up"])
    homogeneous = np.concatenate([points.astype(np.float32), np.ones((len(points), 1), dtype=np.float32)], axis=1)
    camera_points = homogeneous @ view.T
    xyz = camera_points[:, :3]
    valid = xyz[:, 2] < -1e-6
    xyz = xyz[valid]
    world_points = points[valid]
    color_values = colors[valid]
    if len(xyz) == 0:
        return canvas

    z = -xyz[:, 2]
    u, v = _project_view_coordinates(
        xyz,
        width=width,
        height=height,
        projection_mode=projection_mode,
        ortho_scale=ortho_scale,
    )
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    if not np.any(inside):
        return canvas
    u = np.rint(u[inside]).astype(np.int32)
    v = np.rint(v[inside]).astype(np.int32)
    u = np.clip(u, 0, width - 1)
    v = np.clip(v, 0, height - 1)
    z = z[inside]
    xyz = xyz[inside]
    world_points = world_points[inside]
    color_values = color_values[inside]

    order = np.argsort(z)[::-1]
    u = u[order]
    v = v[order]
    z = z[order]
    xyz = xyz[order]
    world_points = world_points[order]
    color_values = color_values[order]

    canvas["rgb"][v, u] = color_values
    canvas["depth"][v, u] = z
    canvas["xyz_view"][v, u] = xyz
    canvas["world_z"][v, u] = world_points[:, 2]
    canvas["valid"][v, u] = True
    return canvas


def rasterize_point_cloud_view(
    points: np.ndarray,
    colors: np.ndarray,
    *,
    view_config: dict[str, Any],
    width: int,
    height: int,
    projection_mode: str,
    ortho_scale: float | None,
) -> dict[str, np.ndarray]:
    return _rasterize_view(
        points,
        colors,
        view_config=view_config,
        width=width,
        height=height,
        projection_mode=projection_mode,
        ortho_scale=ortho_scale,
    )


def _colorize_scalar_map(
    scalar_map: np.ndarray,
    valid_mask: np.ndarray,
    *,
    min_value: float,
    max_value: float,
) -> np.ndarray:
    canvas = np.zeros(scalar_map.shape + (3,), dtype=np.uint8)
    if np.any(valid_mask):
        normalized = np.clip((scalar_map - float(min_value)) / max(1e-6, float(max_value) - float(min_value)), 0.0, 1.0)
        colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        canvas[valid_mask] = colored[valid_mask]
    return canvas


def _compute_normals_from_xyz_map(xyz_map: np.ndarray, valid_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normals = np.zeros_like(xyz_map, dtype=np.float32)
    if xyz_map.shape[0] < 3 or xyz_map.shape[1] < 3:
        return normals, np.zeros(valid_mask.shape, dtype=bool)
    left = xyz_map[1:-1, :-2]
    right = xyz_map[1:-1, 2:]
    up = xyz_map[:-2, 1:-1]
    down = xyz_map[2:, 1:-1]
    valid_inner = (
        valid_mask[1:-1, 1:-1]
        & valid_mask[1:-1, :-2]
        & valid_mask[1:-1, 2:]
        & valid_mask[:-2, 1:-1]
        & valid_mask[2:, 1:-1]
    )
    dx = right - left
    dy = down - up
    inner_normals = np.cross(dx, dy)
    norm = np.linalg.norm(inner_normals, axis=2, keepdims=True)
    good = valid_inner & (norm[..., 0] > 1e-6)
    inner_normals[good] = inner_normals[good] / norm[good]
    flip = inner_normals[..., 2] > 0
    inner_normals[flip] *= -1.0
    normals[1:-1, 1:-1] = inner_normals
    valid_normals = np.zeros(valid_mask.shape, dtype=bool)
    valid_normals[1:-1, 1:-1] = good
    return normals, valid_normals


def _densify_xyz_map(
    xyz_map: np.ndarray,
    valid_mask: np.ndarray,
    *,
    blur_radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    kernel = max(3, blur_radius * 2 + 1)
    mask = valid_mask.astype(np.float32)
    blurred_mask = cv2.GaussianBlur(mask, (kernel, kernel), sigmaX=max(1.0, blur_radius * 0.75))
    densified = np.zeros_like(xyz_map, dtype=np.float32)
    for channel_idx in range(3):
        channel = np.asarray(xyz_map[..., channel_idx], dtype=np.float32) * mask
        blurred_channel = cv2.GaussianBlur(channel, (kernel, kernel), sigmaX=max(1.0, blur_radius * 0.75))
        valid = blurred_mask > 1e-4
        densified[..., channel_idx][valid] = blurred_channel[valid] / np.maximum(blurred_mask[valid], 1e-4)
    valid_dense = blurred_mask > 0.05
    return densified, valid_dense


def _render_normals(normals: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    canvas = np.zeros(normals.shape, dtype=np.uint8)
    if np.any(valid_mask):
        rgb = ((normals + 1.0) * 0.5 * 255.0).astype(np.uint8)
        canvas[valid_mask] = rgb[valid_mask][:, ::-1]
    return canvas


def _render_gray_shaded(normals: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    canvas = np.zeros(normals.shape, dtype=np.uint8)
    if np.any(valid_mask):
        light_dir = np.asarray([0.30, -0.18, -1.0], dtype=np.float32)
        light_dir /= np.linalg.norm(light_dir)
        intensity = np.clip((normals @ light_dir) * 0.5 + 0.5, 0.0, 1.0)
        shaded = np.clip(58.0 + intensity * 178.0, 0.0, 255.0).astype(np.uint8)
        gray = np.stack([shaded, shaded, shaded], axis=2)
        canvas[valid_mask] = gray[valid_mask]
    return canvas


def _background_bgr_for_render_mode(render_mode: str) -> tuple[int, int, int]:
    if render_mode == "color_by_rgb":
        return (28, 28, 30)
    if render_mode == "neutral_gray_shaded":
        return (30, 34, 40)
    return (24, 26, 30)


def _apply_zoom(image: np.ndarray, zoom_scale: float) -> np.ndarray:
    if zoom_scale <= 1.0:
        return image
    height, width = image.shape[:2]
    crop_w = max(8, int(round(width / float(zoom_scale))))
    crop_h = max(8, int(round(height / float(zoom_scale))))
    x0 = max(0, (width - crop_w) // 2)
    y0 = max(0, (height - crop_h) // 2)
    cropped = image[y0:y0 + crop_h, x0:x0 + crop_w]
    return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)


def render_point_cloud_fallback(
    points: np.ndarray,
    colors: np.ndarray,
    *,
    view_config: dict[str, Any],
    render_mode: str,
    scalar_bounds: dict[str, tuple[float, float]],
    width: int = 960,
    height: int = 720,
    zoom_scale: float = 1.0,
    point_radius_px: int = 2,
    supersample_scale: int = 2,
    projection_mode: str = "perspective",
    ortho_scale: float | None = None,
) -> np.ndarray:
    ss = max(1, int(supersample_scale))
    ss_width = width * ss
    ss_height = height * ss
    raster = _rasterize_view(
        points,
        colors,
        view_config=view_config,
        width=ss_width,
        height=ss_height,
        projection_mode=projection_mode,
        ortho_scale=ortho_scale,
    )
    valid = raster["valid"]
    valid_render = valid.astype(np.float32)
    radius_ss = max(1, int(point_radius_px) * ss)
    if render_mode == "color_by_rgb":
        canvas = raster["rgb"]
    elif render_mode == "color_by_height":
        min_height, max_height = scalar_bounds["height"]
        canvas = _colorize_scalar_map(raster["world_z"], valid, min_value=min_height, max_value=max_height)
    elif render_mode == "color_by_depth":
        min_depth, max_depth = scalar_bounds["depth"]
        canvas = _colorize_scalar_map(raster["depth"], valid, min_value=min_depth, max_value=max_depth)
    else:
        densified_xyz, densified_valid = _densify_xyz_map(
            raster["xyz_view"],
            valid,
            blur_radius=radius_ss,
        )
        normals, valid_normals = _compute_normals_from_xyz_map(densified_xyz, densified_valid)
        if render_mode == "color_by_normals":
            if np.count_nonzero(valid_normals) < 512:
                min_height, max_height = scalar_bounds["height"]
                canvas = _colorize_scalar_map(raster["world_z"], valid, min_value=min_height, max_value=max_height)
            else:
                canvas = _render_normals(normals, valid_normals)
        else:
            if np.count_nonzero(valid_normals) < 512:
                min_height, max_height = scalar_bounds["height"]
                canvas = _colorize_scalar_map(raster["world_z"], valid, min_value=min_height, max_value=max_height)
            else:
                canvas = _render_gray_shaded(normals, valid_normals)

    if radius_ss > 1:
        kernel = radius_ss * 2 + 1
        mask = valid.astype(np.float32)
        blurred_mask = cv2.GaussianBlur(mask, (kernel, kernel), sigmaX=max(1.0, radius_ss * 0.65))
        valid_render = blurred_mask
        if canvas.ndim == 3:
            blurred_canvas = cv2.GaussianBlur(canvas.astype(np.float32), (kernel, kernel), sigmaX=max(1.0, radius_ss * 0.65))
            canvas = np.zeros_like(canvas)
            valid_blur = blurred_mask > 1e-4
            if np.any(valid_blur):
                normalized = blurred_canvas / np.maximum(blurred_mask[..., None], 1e-4)
                canvas[valid_blur] = np.clip(normalized[valid_blur], 0.0, 255.0).astype(np.uint8)

    if ss > 1:
        canvas = cv2.resize(canvas, (width, height), interpolation=cv2.INTER_AREA)
        valid_render = cv2.resize(valid_render.astype(np.float32), (width, height), interpolation=cv2.INTER_AREA)

    background = np.full(canvas.shape, _background_bgr_for_render_mode(render_mode), dtype=np.uint8)
    final_valid = valid_render > (1e-4 if radius_ss > 1 else 0.0)
    if np.any(final_valid):
        background[final_valid] = canvas[final_valid]
    return _apply_zoom(background, zoom_scale)


def render_point_cloud(
    points: np.ndarray,
    colors: np.ndarray,
    *,
    renderer: str,
    view_config: dict[str, Any],
    render_mode: str,
    scalar_bounds: dict[str, tuple[float, float]],
    width: int = 960,
    height: int = 720,
    zoom_scale: float = 1.0,
    point_radius_px: int = 2,
    supersample_scale: int = 2,
    projection_mode: str = "perspective",
    ortho_scale: float | None = None,
) -> tuple[np.ndarray, str]:
    if (
        renderer == "fallback"
        or render_mode != "color_by_rgb"
        or projection_mode != "perspective"
        or point_radius_px != 1
        or supersample_scale != 1
    ):
        return render_point_cloud_fallback(
            points,
            colors,
            view_config=view_config,
            render_mode=render_mode,
            scalar_bounds=scalar_bounds,
            width=width,
            height=height,
            zoom_scale=zoom_scale,
            point_radius_px=point_radius_px,
            supersample_scale=supersample_scale,
            projection_mode=projection_mode,
            ortho_scale=ortho_scale,
        ), "fallback"

    if renderer == "open3d" or renderer == "auto":
        try:
            import open3d as o3d
            from open3d.visualization import rendering

            if not hasattr(rendering, "OffscreenRenderer"):
                raise RuntimeError("Open3D offscreen renderer unavailable.")
            renderer_o3d = rendering.OffscreenRenderer(width, height)
            material = rendering.MaterialRecord()
            material.shader = "defaultUnlit"
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
            pcd.colors = o3d.utility.Vector3dVector((colors.astype(np.float32) / 255.0)[:, ::-1])
            renderer_o3d.scene.add_geometry("pcd", pcd, material)
            bbox_min = points.min(axis=0)
            bbox_max = points.max(axis=0)
            center = view_config["center"]
            eye = view_config["camera_position"]
            up = view_config["up"]
            renderer_o3d.scene.camera.look_at(center, eye, up)
            renderer_o3d.scene.set_background([0.0, 0.0, 0.0, 1.0])
            image = np.asarray(renderer_o3d.render_to_image())
            renderer_o3d.release()
            return _apply_zoom(cv2.cvtColor(image, cv2.COLOR_RGBA2BGR), zoom_scale), "open3d"
        except Exception:
            if renderer == "open3d":
                raise
    return render_point_cloud_fallback(
        points,
        colors,
        view_config=view_config,
        render_mode=render_mode,
        scalar_bounds=scalar_bounds,
        width=width,
        height=height,
        zoom_scale=zoom_scale,
        point_radius_px=point_radius_px,
        supersample_scale=supersample_scale,
        projection_mode=projection_mode,
        ortho_scale=ortho_scale,
    ), "fallback"


def compose_panel(native_image: np.ndarray, ffs_image: np.ndarray, *, layout: str) -> np.ndarray:
    if layout == "stacked":
        return np.vstack([native_image, ffs_image])
    return np.hstack([native_image, ffs_image])


def write_video(video_path: Path, frame_paths: list[Path], fps: int) -> None:
    if not frame_paths:
        return
    first = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        return
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (first.shape[1], first.shape[0]),
    )
    for path in frame_paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        writer.write(image)
    writer.release()


def write_gif(gif_path: Path, frame_paths: list[Path], fps: int) -> None:
    if not frame_paths:
        return
    frames: list[Image.Image] = []
    for path in frame_paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        frames.append(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
    if not frames:
        return
    duration_ms = max(20, int(round(1000.0 / max(1, int(fps)))))
    frames[0].save(
        str(gif_path),
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def run_depth_comparison_workflow(
    *,
    aligned_root: Path,
    case_name: str | None = None,
    realsense_case: str | None = None,
    ffs_case: str | None = None,
    output_dir: Path,
    frame_start: int | None = None,
    frame_end: int | None = None,
    frame_stride: int = 1,
    voxel_size: float | None = None,
    max_points_per_camera: int | None = None,
    depth_min_m: float = 0.1,
    depth_max_m: float = 3.0,
    renderer: str = "auto",
    write_ply: bool = False,
    write_mp4: bool = False,
    fps: int = 30,
    panel_layout: str = "side_by_side",
    use_float_ffs_depth_when_available: bool = False,
    render_mode: str = "neutral_gray_shaded",
    views: list[str] | None = None,
    zoom_scale: float = 1.0,
    view_mode: str = "fixed",
    focus_mode: str = "none",
    layout_mode: str = "pair",
    scene_crop_mode: str = "none",
    crop_margin_xy: float = 0.12,
    crop_min_z: float = -0.15,
    crop_max_z: float = 0.35,
    roi_x_min: float | None = None,
    roi_x_max: float | None = None,
    roi_y_min: float | None = None,
    roi_y_max: float | None = None,
    roi_z_min: float | None = None,
    roi_z_max: float | None = None,
    view_distance_scale: float = 1.0,
    projection_mode: str = "perspective",
    ortho_scale: float | None = None,
    point_radius_px: int = 2,
    supersample_scale: int = 2,
    image_flip: str = "none",
) -> dict[str, Any]:
    if render_mode not in RENDER_MODES:
        raise ValueError(f"Unsupported render_mode: {render_mode}")
    if view_mode not in VIEW_MODES:
        raise ValueError(f"Unsupported view_mode: {view_mode}")
    if focus_mode not in FOCUS_MODES:
        raise ValueError(f"Unsupported focus_mode: {focus_mode}")
    if layout_mode not in LAYOUT_MODES:
        raise ValueError(f"Unsupported layout_mode: {layout_mode}")
    if scene_crop_mode not in SCENE_CROP_MODES:
        raise ValueError(f"Unsupported scene_crop_mode: {scene_crop_mode}")
    if projection_mode not in PROJECTION_MODES:
        raise ValueError(f"Unsupported projection_mode: {projection_mode}")
    if image_flip not in IMAGE_FLIP_MODES:
        raise ValueError(f"Unsupported image_flip: {image_flip}")
    aligned_root = Path(aligned_root).resolve()
    output_dir = Path(output_dir).resolve()
    native_case_dir, ffs_case_dir, same_case_mode = resolve_case_dirs(
        aligned_root=aligned_root,
        case_name=case_name,
        realsense_case=realsense_case,
        ffs_case=ffs_case,
    )
    native_metadata = load_case_metadata(native_case_dir)
    ffs_metadata = load_case_metadata(ffs_case_dir)

    if same_case_mode and not (native_case_dir / "depth_ffs").exists():
        raise ValueError("Same-case comparison requires an aligned case containing depth_ffs/.")

    if len(native_metadata["serial_numbers"]) != len(ffs_metadata["serial_numbers"]):
        raise ValueError("Native and FFS cases must have the same number of cameras for comparison.")

    frame_pairs = select_frame_indices(
        native_count=get_frame_count(native_metadata),
        ffs_count=get_frame_count(ffs_metadata),
        frame_start=frame_start,
        frame_end=frame_end,
        frame_stride=frame_stride,
    )
    if not frame_pairs:
        raise ValueError("No frame pairs selected for comparison.")

    manual_xyz_roi = None
    if scene_crop_mode == "manual_xyz_roi":
        roi_values = [roi_x_min, roi_x_max, roi_y_min, roi_y_max, roi_z_min, roi_z_max]
        if any(value is None for value in roi_values):
            raise ValueError("manual_xyz_roi requires all roi_x/y/z min/max arguments.")
        manual_xyz_roi = {
            "x_min": float(roi_x_min),
            "x_max": float(roi_x_max),
            "y_min": float(roi_y_min),
            "y_max": float(roi_y_max),
            "z_min": float(roi_z_min),
            "z_max": float(roi_z_max),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    cache: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {}
    raw_bounds_min = np.array([np.inf, np.inf, np.inf], dtype=np.float32)
    raw_bounds_max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float32)
    all_point_sets: list[np.ndarray] = []
    for native_frame_idx, ffs_frame_idx in frame_pairs:
        for source, case_dir, metadata, frame_idx in (
            ("native", native_case_dir, native_metadata, native_frame_idx),
            ("ffs", ffs_case_dir, ffs_metadata, ffs_frame_idx),
        ):
            points, colors, stats = load_case_frame_cloud(
                case_dir=case_dir,
                metadata=metadata,
                frame_idx=frame_idx,
                depth_source="realsense" if source == "native" else "ffs",
                use_float_ffs_depth_when_available=use_float_ffs_depth_when_available,
                voxel_size=voxel_size,
                max_points_per_camera=max_points_per_camera,
                depth_min_m=depth_min_m,
                depth_max_m=depth_max_m,
            )
            cache[(source, frame_idx)] = (points, colors, stats)
            if len(points) > 0:
                all_point_sets.append(points)
                raw_bounds_min = np.minimum(raw_bounds_min, points.min(axis=0))
                raw_bounds_max = np.maximum(raw_bounds_max, points.max(axis=0))

    if not np.isfinite(raw_bounds_min).all():
        raw_bounds_min = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
        raw_bounds_max = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    metrics = []
    renderer_used_by_view: dict[str, str] = {}
    focus_point = estimate_focus_point(
        all_point_sets,
        bounds_min=raw_bounds_min,
        bounds_max=raw_bounds_max,
        focus_mode=focus_mode,
    )
    crop_bounds = compute_scene_crop_bounds(
        all_point_sets,
        focus_point=focus_point,
        scene_crop_mode=scene_crop_mode,
        crop_margin_xy=float(crop_margin_xy),
        crop_min_z=float(crop_min_z),
        crop_max_z=float(crop_max_z),
        manual_xyz_roi=manual_xyz_roi,
    )

    cropped_cache: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {}
    cropped_point_sets: list[np.ndarray] = []
    bounds_min = np.array([np.inf, np.inf, np.inf], dtype=np.float32)
    bounds_max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float32)
    for key, (points, colors, stats) in cache.items():
        cropped_points, cropped_colors = crop_points_to_bounds(points, colors, crop_bounds)
        cropped_stats = dict(stats)
        cropped_stats["fused_point_count_before_crop"] = int(len(points))
        cropped_stats["fused_point_count_after_crop"] = int(len(cropped_points))
        cropped_cache[key] = (cropped_points, cropped_colors, cropped_stats)
        if len(cropped_points) > 0:
            cropped_point_sets.append(cropped_points)
            bounds_min = np.minimum(bounds_min, cropped_points.min(axis=0))
            bounds_max = np.maximum(bounds_max, cropped_points.max(axis=0))

    if not np.isfinite(bounds_min).all():
        bounds_min = raw_bounds_min.copy()
        bounds_max = raw_bounds_max.copy()

    scalar_bounds = {
        "height": (float(bounds_min[2]), float(bounds_max[2])),
        "depth": (0.0, max(float(np.linalg.norm(bounds_max - bounds_min)) * 2.0, 1.0)),
    }
    focus_point = estimate_focus_point(
        cropped_point_sets if cropped_point_sets else all_point_sets,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        focus_mode=focus_mode,
    )

    if view_mode == "fixed":
        selected_view_names = views or ["oblique"]
        for view_name in selected_view_names:
            if view_name not in VIEW_NAMES:
                raise ValueError(f"Unsupported view: {view_name}")
        view_configs = []
        for view_name in selected_view_names:
            config = compute_view_config(bounds_min, bounds_max, view_name=view_name)
            direction = _normalize_vector(config["camera_position"] - focus_point, np.array([0.0, 0.0, 1.0], dtype=np.float32))
            distance = max(1e-3, float(np.linalg.norm(config["camera_position"] - focus_point)) * float(view_distance_scale))
            config["camera_position"] = np.asarray(focus_point, dtype=np.float32) + direction * distance
            config["center"] = np.asarray(focus_point, dtype=np.float32)
            config["radius"] = distance
            view_configs.append(config)
    else:
        if len(native_metadata["serial_numbers"]) != 3:
            raise ValueError("camera_poses_table_focus currently requires exactly 3 cameras.")
        calibration_reference_serials = native_metadata.get("calibration_reference_serials", native_metadata["serial_numbers"])
        native_c2w = load_calibration_transforms(
            native_case_dir / "calibrate.pkl",
            serial_numbers=native_metadata["serial_numbers"],
            calibration_reference_serials=calibration_reference_serials,
        )
        target_distance = max(
            0.6,
            float(np.linalg.norm(bounds_max - bounds_min)),
        )
        view_configs = build_camera_pose_view_configs(
            c2w_list=native_c2w,
            serial_numbers=native_metadata["serial_numbers"],
            focus_point=focus_point,
            view_distance_scale=float(view_distance_scale),
            target_distance=target_distance,
        )

    if layout_mode == "grid_2x3" and len(view_configs) != 3:
        raise ValueError("grid_2x3 layout requires exactly 3 view configs.")

    per_view_outputs: dict[str, dict[str, Any]] = {}
    grid_frame_paths: list[Path] = []
    if layout_mode == "grid_2x3":
        grid_frames_dir = output_dir / "grid_2x3_frames"
        grid_frames_dir.mkdir(parents=True, exist_ok=True)

    for view_config in view_configs:
        view_name = str(view_config["view_name"])
        view_output_dir = output_dir if len(view_configs) == 1 and layout_mode == "pair" else output_dir / f"view_{view_name}"
        view_output_dir.mkdir(parents=True, exist_ok=True)
        native_frames_dir = view_output_dir / "native_frames"
        ffs_frames_dir = view_output_dir / "ffs_frames"
        side_frames_dir = view_output_dir / "side_by_side_frames"
        for directory in (native_frames_dir, ffs_frames_dir, side_frames_dir):
            directory.mkdir(parents=True, exist_ok=True)
        native_clouds_dir = view_output_dir / "native_clouds"
        ffs_clouds_dir = view_output_dir / "ffs_clouds"
        if write_ply:
            native_clouds_dir.mkdir(parents=True, exist_ok=True)
            ffs_clouds_dir.mkdir(parents=True, exist_ok=True)
        per_view_outputs[view_name] = {
            "view_output_dir": view_output_dir,
            "native_frames_dir": native_frames_dir,
            "ffs_frames_dir": ffs_frames_dir,
            "side_frames_dir": side_frames_dir,
            "native_clouds_dir": native_clouds_dir,
            "ffs_clouds_dir": ffs_clouds_dir,
            "native_frame_paths": [],
            "ffs_frame_paths": [],
            "side_frame_paths": [],
            "renderer_used": None,
            "view_config": view_config,
            "ortho_scale": None,
        }

        if projection_mode == "orthographic":
            per_view_outputs[view_name]["ortho_scale"] = float(
                ortho_scale
                if ortho_scale is not None
                else estimate_ortho_scale(
                    cropped_point_sets if cropped_point_sets else all_point_sets,
                    view_config=view_config,
                )
            )

    for panel_idx, (native_frame_idx, ffs_frame_idx) in enumerate(frame_pairs):
        native_points, native_colors, native_stats = cropped_cache[("native", native_frame_idx)]
        ffs_points, ffs_colors, ffs_stats = cropped_cache[("ffs", ffs_frame_idx)]
        grid_native_images: list[np.ndarray] = []
        grid_ffs_images: list[np.ndarray] = []

        for view_config in view_configs:
            view_name = str(view_config["view_name"])
            view_state = per_view_outputs[view_name]
            renderer_used = view_state["renderer_used"]
            native_render, renderer_used = render_point_cloud(
                native_points,
                native_colors,
                renderer=renderer,
                view_config=view_config,
                render_mode=render_mode,
                scalar_bounds=scalar_bounds,
                zoom_scale=zoom_scale,
                point_radius_px=point_radius_px,
                supersample_scale=supersample_scale,
                projection_mode=projection_mode,
                ortho_scale=view_state["ortho_scale"],
            )
            ffs_render, renderer_used = render_point_cloud(
                ffs_points,
                ffs_colors,
                renderer=renderer if renderer != "auto" else renderer_used or "auto",
                view_config=view_config,
                render_mode=render_mode,
                scalar_bounds=scalar_bounds,
                zoom_scale=zoom_scale,
                point_radius_px=point_radius_px,
                supersample_scale=supersample_scale,
                projection_mode=projection_mode,
                ortho_scale=view_state["ortho_scale"],
            )
            view_state["renderer_used"] = renderer_used

            native_render = apply_image_flip(native_render, image_flip)
            ffs_render = apply_image_flip(ffs_render, image_flip)
            native_labeled = overlay_panel_label(native_render, label=f"Native | {view_config['label']}")
            ffs_labeled = overlay_panel_label(ffs_render, label=f"FFS | {view_config['label']}")
            side_render = compose_panel(native_labeled, ffs_labeled, layout=panel_layout)

            native_frame_path = view_state["native_frames_dir"] / f"{panel_idx:06d}.png"
            ffs_frame_path = view_state["ffs_frames_dir"] / f"{panel_idx:06d}.png"
            side_frame_path = view_state["side_frames_dir"] / f"{panel_idx:06d}.png"
            cv2.imwrite(str(native_frame_path), native_labeled)
            cv2.imwrite(str(ffs_frame_path), ffs_labeled)
            cv2.imwrite(str(side_frame_path), side_render)
            view_state["native_frame_paths"].append(native_frame_path)
            view_state["ffs_frame_paths"].append(ffs_frame_path)
            view_state["side_frame_paths"].append(side_frame_path)

            if write_ply:
                write_ply_ascii(view_state["native_clouds_dir"] / f"{panel_idx:06d}.ply", native_points, native_colors)
                write_ply_ascii(view_state["ffs_clouds_dir"] / f"{panel_idx:06d}.ply", ffs_points, ffs_colors)

            if layout_mode == "grid_2x3":
                grid_native_images.append(native_labeled)
                grid_ffs_images.append(ffs_labeled)

            metrics.append(
                {
                    "view_name": view_name,
                    "panel_frame_idx": panel_idx,
                    "native_frame_idx": native_frame_idx,
                    "ffs_frame_idx": ffs_frame_idx,
                    "native": native_stats,
                    "ffs": ffs_stats,
                }
            )

        if layout_mode == "grid_2x3":
            title = (
                f"{native_case_dir.name} vs {ffs_case_dir.name} | frame {panel_idx:06d} | "
                f"{render_mode} | crop={scene_crop_mode} | proj={projection_mode} | "
                f"dist={view_distance_scale:.2f} | depth=[{depth_min_m:.2f}, {depth_max_m:.2f}] m"
            )
            grid_image = compose_grid_2x3(
                title=title,
                column_headers=[f"View from Cam{i}" for i in range(3)],
                row_headers=["Native depth", "FFS depth"],
                native_images=grid_native_images,
                ffs_images=grid_ffs_images,
            )
            grid_frame_path = grid_frames_dir / f"{panel_idx:06d}.png"
            cv2.imwrite(str(grid_frame_path), grid_image)
            grid_frame_paths.append(grid_frame_path)

    for view_name, view_state in per_view_outputs.items():
        renderer_used_by_view[view_name] = view_state["renderer_used"] or "fallback"
        videos_dir = Path(view_state["view_output_dir"]) / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)
        if write_mp4:
            write_video(videos_dir / "native.mp4", view_state["native_frame_paths"], fps)
            write_video(videos_dir / "ffs.mp4", view_state["ffs_frame_paths"], fps)
            write_video(videos_dir / "side_by_side.mp4", view_state["side_frame_paths"], fps)

    if layout_mode == "grid_2x3":
        videos_dir = output_dir / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)
        if write_mp4:
            write_video(videos_dir / "grid_2x3.mp4", grid_frame_paths, fps)

    comparison_metadata = {
        "same_case_mode": same_case_mode,
        "native_case_dir": str(native_case_dir),
        "ffs_case_dir": str(ffs_case_dir),
        "frame_pairs": frame_pairs,
        "renderer_requested": renderer,
        "renderer_used": renderer_used_by_view,
        "views": [str(view_config["view_name"]) for view_config in view_configs],
        "view_labels": [str(view_config["label"]) for view_config in view_configs],
        "view_mode": view_mode,
        "focus_mode": focus_mode,
        "layout_mode": layout_mode,
        "render_mode": render_mode,
        "panel_layout": panel_layout,
        "scene_crop_mode": scene_crop_mode,
        "crop_bounds": {"min": crop_bounds["min"].tolist(), "max": crop_bounds["max"].tolist()},
        "crop_margin_xy": float(crop_margin_xy),
        "crop_min_z": float(crop_min_z),
        "crop_max_z": float(crop_max_z),
        "depth_min_m": float(depth_min_m),
        "depth_max_m": float(depth_max_m),
        "voxel_size": voxel_size,
        "max_points_per_camera": max_points_per_camera,
        "use_float_ffs_depth_when_available": use_float_ffs_depth_when_available,
        "zoom_scale": float(zoom_scale),
        "view_distance_scale": float(view_distance_scale),
        "projection_mode": projection_mode,
        "ortho_scale": ortho_scale,
        "ortho_scale_by_view": {
            view_name: per_view_outputs[view_name]["ortho_scale"]
            for view_name in per_view_outputs
        },
        "point_radius_px": int(point_radius_px),
        "supersample_scale": int(supersample_scale),
        "image_flip": image_flip,
        "scalar_bounds": scalar_bounds,
        "focus_point": focus_point.tolist(),
    }
    (output_dir / "comparison_metadata.json").write_text(json.dumps(comparison_metadata, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return {
        "output_dir": str(output_dir),
        "comparison_metadata": comparison_metadata,
        "metrics": metrics,
    }
