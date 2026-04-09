from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .object_roi import compute_object_region_mask
from .pointcloud_compare import compute_view_config, render_point_cloud, write_ply_ascii


def deterministic_subsample(
    points: np.ndarray,
    colors: np.ndarray,
    *,
    max_points: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    cloud = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    color_values = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    if max_points is None or int(max_points) <= 0 or len(cloud) <= int(max_points):
        return cloud, color_values
    idx = np.linspace(0, len(cloud) - 1, int(max_points), dtype=np.int32)
    return cloud[idx], color_values[idx]


def deterministic_subsample_indices(length: int, *, max_points: int | None) -> np.ndarray:
    if max_points is None or int(max_points) <= 0 or int(length) <= int(max_points):
        return np.arange(int(length), dtype=np.int32)
    return np.linspace(0, int(length) - 1, int(max_points), dtype=np.int32)


def _concat_clouds(camera_clouds: list[dict[str, Any]], *, key_points: str, key_colors: str) -> tuple[np.ndarray, np.ndarray]:
    point_sets = [np.asarray(item[key_points], dtype=np.float32) for item in camera_clouds if len(item[key_points]) > 0]
    color_sets = [np.asarray(item[key_colors], dtype=np.uint8) for item in camera_clouds if len(item[key_points]) > 0]
    if not point_sets:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
    if len(point_sets) == 1:
        return point_sets[0], color_sets[0]
    return np.concatenate(point_sets, axis=0), np.concatenate(color_sets, axis=0)


def _concat_aligned_arrays(camera_clouds: list[dict[str, Any]], *, key: str, dtype: Any) -> np.ndarray:
    arrays = [np.asarray(item[key], dtype=dtype).reshape(-1) for item in camera_clouds if len(item["points"]) > 0]
    if not arrays:
        return np.empty((0,), dtype=dtype)
    if len(arrays) == 1:
        return arrays[0]
    return np.concatenate(arrays, axis=0)


def build_object_first_layers(
    camera_clouds: list[dict[str, Any]],
    *,
    object_roi_min: np.ndarray,
    object_roi_max: np.ndarray,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
    table_color_bgr: np.ndarray | None,
    object_height_min: float,
    object_height_max: float,
    context_max_points_per_camera: int | None,
    pixel_mask_by_camera: dict[int, np.ndarray] | None = None,
    object_max_points_per_camera: int | None = None,
) -> dict[str, Any]:
    object_camera_clouds: list[dict[str, Any]] = []
    context_camera_clouds: list[dict[str, Any]] = []
    combined_camera_clouds: list[dict[str, Any]] = []
    per_camera_metrics: list[dict[str, Any]] = []

    for camera_cloud in camera_clouds:
        points = np.asarray(camera_cloud["points"], dtype=np.float32)
        colors = np.asarray(camera_cloud["colors"], dtype=np.uint8)
        source_camera_idx_all = np.asarray(
            camera_cloud.get("source_camera_idx", np.full((len(points),), int(camera_cloud["camera_idx"]), dtype=np.int16)),
            dtype=np.int16,
        ).reshape(-1)
        source_serial_all = np.asarray(
            camera_cloud.get("source_serial", np.full((len(points),), camera_cloud["serial"], dtype=object)),
            dtype=object,
        ).reshape(-1)
        geometric_object_mask = compute_object_region_mask(
            points,
            colors,
            object_roi_min=object_roi_min,
            object_roi_max=object_roi_max,
            plane_point=plane_point,
            plane_normal=plane_normal,
            min_height=max(0.0, float(object_height_min)),
            max_height=max(0.40, float(object_height_max)),
            table_color_bgr=table_color_bgr,
        )
        object_mask = geometric_object_mask.copy()
        pixel_points_before = 0
        geometric_points_before = int(np.count_nonzero(geometric_object_mask))
        combined_points_before = geometric_points_before
        if pixel_mask_by_camera is not None:
            pixel_mask = pixel_mask_by_camera.get(int(camera_cloud["camera_idx"]))
            if pixel_mask is not None:
                pixel_object_mask = point_mask_from_pixel_mask(camera_cloud, pixel_mask=pixel_mask)
                pixel_points_before = int(np.count_nonzero(pixel_object_mask))
                intersected_mask = geometric_object_mask & pixel_object_mask
                combined_points_before = int(np.count_nonzero(intersected_mask))
                if combined_points_before >= max(256, int(round(pixel_points_before * 0.35))):
                    object_mask = intersected_mask
                else:
                    object_mask = pixel_object_mask
                    combined_points_before = pixel_points_before
        object_points_before = points[object_mask]
        object_colors_before = colors[object_mask]
        object_source_camera_idx_before = source_camera_idx_all[object_mask]
        object_source_serial_before = source_serial_all[object_mask]
        context_points_before = points[~object_mask]
        context_colors_before = colors[~object_mask]
        context_source_camera_idx_before = source_camera_idx_all[~object_mask]
        context_source_serial_before = source_serial_all[~object_mask]

        object_idx = deterministic_subsample_indices(len(object_points_before), max_points=object_max_points_per_camera)
        context_idx = deterministic_subsample_indices(len(context_points_before), max_points=context_max_points_per_camera)
        object_points_after = object_points_before[object_idx]
        object_colors_after = object_colors_before[object_idx]
        object_source_camera_idx_after = object_source_camera_idx_before[object_idx]
        object_source_serial_after = object_source_serial_before[object_idx]
        context_points_after = context_points_before[context_idx]
        context_colors_after = context_colors_before[context_idx]
        context_source_camera_idx_after = context_source_camera_idx_before[context_idx]
        context_source_serial_after = context_source_serial_before[context_idx]

        object_cloud = {
            **camera_cloud,
            "points": object_points_after,
            "colors": object_colors_after,
            "source_camera_idx": object_source_camera_idx_after,
            "source_serial": object_source_serial_after,
        }
        context_cloud = {
            **camera_cloud,
            "points": context_points_after,
            "colors": context_colors_after,
            "source_camera_idx": context_source_camera_idx_after,
            "source_serial": context_source_serial_after,
        }
        combined_cloud = {
            **camera_cloud,
            "points": np.concatenate([object_points_after, context_points_after], axis=0) if len(context_points_after) > 0 else object_points_after,
            "colors": np.concatenate([object_colors_after, context_colors_after], axis=0) if len(context_colors_after) > 0 else object_colors_after,
            "source_camera_idx": (
                np.concatenate([object_source_camera_idx_after, context_source_camera_idx_after], axis=0)
                if len(context_points_after) > 0 else object_source_camera_idx_after
            ),
            "source_serial": (
                np.concatenate([object_source_serial_after, context_source_serial_after], axis=0)
                if len(context_points_after) > 0 else object_source_serial_after
            ),
        }
        object_camera_clouds.append(object_cloud)
        context_camera_clouds.append(context_cloud)
        combined_camera_clouds.append(combined_cloud)
        per_camera_metrics.append(
            {
                "camera_idx": int(camera_cloud["camera_idx"]),
                "serial": camera_cloud["serial"],
                "geometric_object_points_before_sampling": int(geometric_points_before),
                "pixel_object_points_before_sampling": int(pixel_points_before),
                "combined_object_points_before_sampling": int(combined_points_before),
                "object_points_before_sampling": int(len(object_points_before)),
                "object_points_after_sampling": int(len(object_points_after)),
                "context_points_before_sampling": int(len(context_points_before)),
                "context_points_after_sampling": int(len(context_points_after)),
            }
        )

    fused_object_points, fused_object_colors = _concat_clouds(object_camera_clouds, key_points="points", key_colors="colors")
    fused_context_points, fused_context_colors = _concat_clouds(context_camera_clouds, key_points="points", key_colors="colors")
    fused_combined_points, fused_combined_colors = _concat_clouds(combined_camera_clouds, key_points="points", key_colors="colors")
    fused_object_source_camera_idx = _concat_aligned_arrays(object_camera_clouds, key="source_camera_idx", dtype=np.int16)
    fused_context_source_camera_idx = _concat_aligned_arrays(context_camera_clouds, key="source_camera_idx", dtype=np.int16)
    fused_combined_source_camera_idx = _concat_aligned_arrays(combined_camera_clouds, key="source_camera_idx", dtype=np.int16)
    fused_object_source_serial = _concat_aligned_arrays(object_camera_clouds, key="source_serial", dtype=object)
    fused_context_source_serial = _concat_aligned_arrays(context_camera_clouds, key="source_serial", dtype=object)
    fused_combined_source_serial = _concat_aligned_arrays(combined_camera_clouds, key="source_serial", dtype=object)
    return {
        "object_camera_clouds": object_camera_clouds,
        "context_camera_clouds": context_camera_clouds,
        "combined_camera_clouds": combined_camera_clouds,
        "object_points": fused_object_points,
        "object_colors": fused_object_colors,
        "object_source_camera_idx": fused_object_source_camera_idx,
        "object_source_serial": fused_object_source_serial,
        "context_points": fused_context_points,
        "context_colors": fused_context_colors,
        "context_source_camera_idx": fused_context_source_camera_idx,
        "context_source_serial": fused_context_source_serial,
        "combined_points": fused_combined_points,
        "combined_colors": fused_combined_colors,
        "combined_source_camera_idx": fused_combined_source_camera_idx,
        "combined_source_serial": fused_combined_source_serial,
        "per_camera_metrics": per_camera_metrics,
    }


def _project_world_points_to_pixels(
    points_world: np.ndarray,
    *,
    c2w: np.ndarray,
    K_color: np.ndarray,
    image_shape: tuple[int, int],
) -> np.ndarray:
    cloud = np.asarray(points_world, dtype=np.float32).reshape(-1, 3)
    if len(cloud) == 0:
        return np.empty((0, 2), dtype=np.int32)
    w2c = np.linalg.inv(np.asarray(c2w, dtype=np.float32).reshape(4, 4))
    homogeneous = np.concatenate([cloud, np.ones((len(cloud), 1), dtype=np.float32)], axis=1)
    camera_points = homogeneous @ w2c.T
    xyz = camera_points[:, :3]
    valid = xyz[:, 2] > 1e-6
    xyz = xyz[valid]
    if len(xyz) == 0:
        return np.empty((0, 2), dtype=np.int32)
    K = np.asarray(K_color, dtype=np.float32).reshape(3, 3)
    uvw = xyz @ K.T
    u = np.rint(uvw[:, 0] / np.maximum(uvw[:, 2], 1e-6)).astype(np.int32)
    v = np.rint(uvw[:, 1] / np.maximum(uvw[:, 2], 1e-6)).astype(np.int32)
    width = int(image_shape[1])
    height = int(image_shape[0])
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    if not np.any(inside):
        return np.empty((0, 2), dtype=np.int32)
    pixels = np.stack([u[inside], v[inside]], axis=1).astype(np.int32)
    return np.unique(pixels, axis=0)


def build_foreground_mask_from_roi(
    color_path: str | Path,
    roi: tuple[int, int, int, int],
) -> np.ndarray:
    image = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Missing RGB image for foreground mask: {color_path}")
    x_min, y_min, x_max, y_max = [int(item) for item in roi]
    mask = np.full(image.shape[:2], cv2.GC_BGD, dtype=np.uint8)
    mask[y_min:y_max + 1, x_min:x_max + 1] = cv2.GC_PR_BGD
    inner_margin_x = max(4, int(round((x_max - x_min + 1) * 0.22)))
    inner_margin_y = max(4, int(round((y_max - y_min + 1) * 0.18)))
    inner_x0 = min(x_max, x_min + inner_margin_x)
    inner_x1 = max(x_min, x_max - inner_margin_x)
    inner_y0 = min(y_max, y_min + inner_margin_y)
    inner_y1 = max(y_min, y_max - inner_margin_y)
    if inner_x1 > inner_x0 and inner_y1 > inner_y0:
        mask[inner_y0:inner_y1 + 1, inner_x0:inner_x1 + 1] = cv2.GC_PR_FGD
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)
    try:
        cv2.grabCut(image, mask, None, bgd_model, fgd_model, 4, cv2.GC_INIT_WITH_MASK)
        foreground = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
    except cv2.error:
        foreground = np.zeros(image.shape[:2], dtype=bool)
        fallback_margin_x = max(1, int(round((x_max - x_min + 1) * 0.18)))
        fallback_margin_y = max(1, int(round((y_max - y_min + 1) * 0.18)))
        fg_x0 = min(x_max, x_min + fallback_margin_x)
        fg_x1 = max(x_min, x_max - fallback_margin_x)
        fg_y0 = min(y_max, y_min + fallback_margin_y)
        fg_y1 = max(y_min, y_max - fallback_margin_y)
        foreground[fg_y0:fg_y1 + 1, fg_x0:fg_x1 + 1] = True
    roi_binary = np.zeros_like(foreground, dtype=np.uint8)
    roi_binary[y_min:y_max + 1, x_min:x_max + 1] = 255
    foreground &= roi_binary.astype(bool)
    foreground_u8 = foreground.astype(np.uint8) * 255
    foreground_u8 = cv2.morphologyEx(foreground_u8, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    foreground_u8 = cv2.morphologyEx(foreground_u8, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))
    foreground_u8 = cv2.dilate(foreground_u8, np.ones((3, 3), dtype=np.uint8), iterations=1)
    return foreground_u8 > 0


def _odd_kernel_size(value: float, *, min_size: int, max_size: int) -> int:
    size = int(round(float(value)))
    size = max(int(min_size), min(int(max_size), size))
    if size % 2 == 0:
        size += 1
    return size


def _keep_salient_mask_components(
    mask: np.ndarray,
    *,
    max_components: int = 4,
    min_area: int = 96,
) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    if not np.any(binary):
        return binary
    labels, stats = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)[1:3]
    component_areas: list[tuple[int, int]] = []
    for label_idx in range(1, stats.shape[0]):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area > 0:
            component_areas.append((label_idx, area))
    if not component_areas:
        return binary
    component_areas.sort(key=lambda item: item[1], reverse=True)
    largest_area = component_areas[0][1]
    keep_threshold = max(int(min_area), int(round(largest_area * 0.02)))
    keep = np.zeros_like(binary, dtype=bool)
    kept = 0
    for label_idx, area in component_areas:
        if kept >= int(max_components):
            break
        if area < keep_threshold and kept > 0:
            continue
        keep |= labels == label_idx
        kept += 1
    return keep


def build_geometry_constrained_foreground_mask(
    camera_cloud: dict[str, Any],
    *,
    roi: tuple[int, int, int, int],
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
    object_height_min: float,
    object_height_max: float,
) -> tuple[np.ndarray, dict[str, int]]:
    image = cv2.imread(str(camera_cloud["color_path"]), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Missing RGB image for geometry-constrained mask: {camera_cloud['color_path']}")

    grabcut_mask = build_foreground_mask_from_roi(camera_cloud["color_path"], roi)
    x_min, y_min, x_max, y_max = [int(item) for item in roi]
    roi_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    roi_mask[y_min:y_max + 1, x_min:x_max + 1] = 255

    points = np.asarray(camera_cloud["points"], dtype=np.float32).reshape(-1, 3)
    c2w = np.asarray(camera_cloud["c2w"], dtype=np.float32).reshape(4, 4)
    K_color = np.asarray(camera_cloud["K_color"], dtype=np.float32).reshape(3, 3)
    w2c = np.linalg.inv(c2w)
    homogeneous = np.concatenate([points, np.ones((len(points), 1), dtype=np.float32)], axis=1)
    camera_points = homogeneous @ w2c.T
    xyz = camera_points[:, :3]
    valid = xyz[:, 2] > 1e-6
    geometry_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    if np.any(valid):
        xyz_valid = xyz[valid]
        uvw = xyz_valid @ K_color.T
        u = np.rint(uvw[:, 0] / np.maximum(uvw[:, 2], 1e-6)).astype(np.int32)
        v = np.rint(uvw[:, 1] / np.maximum(uvw[:, 2], 1e-6)).astype(np.int32)
        z = xyz_valid[:, 2].astype(np.float32)
        inside = (
            (u >= x_min) & (u <= x_max) &
            (v >= y_min) & (v <= y_max) &
            (u >= 0) & (u < image.shape[1]) &
            (v >= 0) & (v < image.shape[0])
        )
        if np.any(inside):
            u = u[inside]
            v = v[inside]
            z = z[inside]
            sort_idx = np.argsort(z, kind="mergesort")
            pixel_keys = (v[sort_idx].astype(np.int64) * int(image.shape[1]) + u[sort_idx].astype(np.int64))
            _, unique_idx = np.unique(pixel_keys, return_index=True)
            nearest_idx = sort_idx[unique_idx]
            nearest_u = u[nearest_idx]
            nearest_v = v[nearest_idx]
            nearest_z = z[nearest_idx]

            z_near = float(np.quantile(nearest_z, 0.20))
            z_mid = float(np.quantile(nearest_z, 0.55))
            depth_band = float(np.clip((z_mid - z_near) * 0.95 + 0.025, 0.025, 0.11))
            depth_threshold = z_near + depth_band
            keep_depth = nearest_z <= depth_threshold
            if np.any(keep_depth):
                geometry_mask[nearest_v[keep_depth], nearest_u[keep_depth]] = 255

    if np.count_nonzero(geometry_mask) > 0:
        close_kernel = _odd_kernel_size(max(x_max - x_min, y_max - y_min) * 0.05, min_size=5, max_size=19)
        dilate_kernel = _odd_kernel_size(max(x_max - x_min, y_max - y_min) * 0.08, min_size=7, max_size=27)
        geometry_mask = cv2.morphologyEx(
            geometry_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel)),
        )
        geometry_mask = cv2.dilate(
            geometry_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel, dilate_kernel)),
            iterations=1,
        )

    if np.count_nonzero(geometry_mask) > 0:
        constrained_mask = grabcut_mask & geometry_mask.astype(bool)
        min_pixels = max(128, int(round(np.count_nonzero(geometry_mask) * 0.18)))
        if int(np.count_nonzero(constrained_mask)) < min_pixels:
            constrained_mask = geometry_mask.astype(bool)
        else:
            constrained_mask |= geometry_mask.astype(bool)
    else:
        constrained_mask = grabcut_mask.copy()

    constrained_u8 = (constrained_mask.astype(np.uint8) * 255) & roi_mask
    constrained_u8 = cv2.morphologyEx(constrained_u8, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))
    constrained_u8 = cv2.dilate(constrained_u8, np.ones((3, 3), dtype=np.uint8), iterations=1)
    refined_mask = _keep_salient_mask_components(constrained_u8 > 0)
    return refined_mask, {
        "grabcut_mask_pixels": int(np.count_nonzero(grabcut_mask)),
        "geometry_mask_pixels": int(np.count_nonzero(geometry_mask)),
        "refined_mask_pixels": int(np.count_nonzero(refined_mask)),
    }


def filter_camera_clouds_by_pixel_masks(
    camera_clouds: list[dict[str, Any]],
    *,
    pixel_mask_by_camera: dict[int, np.ndarray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    filtered_clouds: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    for camera_cloud in camera_clouds:
        camera_idx = int(camera_cloud["camera_idx"])
        pixel_mask = pixel_mask_by_camera.get(camera_idx)
        points = np.asarray(camera_cloud["points"], dtype=np.float32).reshape(-1, 3)
        colors = np.asarray(camera_cloud["colors"], dtype=np.uint8).reshape(-1, 3)
        if pixel_mask is None or len(points) == 0:
            filtered_clouds.append({**camera_cloud, "points": points, "colors": colors})
            metrics.append({"camera_idx": camera_idx, "seed_mask_pixels": 0, "seed_object_points": int(len(points))})
            continue

        c2w = np.asarray(camera_cloud["c2w"], dtype=np.float32).reshape(4, 4)
        w2c = np.linalg.inv(c2w)
        homogeneous = np.concatenate([points, np.ones((len(points), 1), dtype=np.float32)], axis=1)
        camera_points = homogeneous @ w2c.T
        xyz = camera_points[:, :3]
        valid = xyz[:, 2] > 1e-6
        K = np.asarray(camera_cloud["K_color"], dtype=np.float32).reshape(3, 3)
        uvw = xyz[valid] @ K.T
        u = np.rint(uvw[:, 0] / np.maximum(uvw[:, 2], 1e-6)).astype(np.int32)
        v = np.rint(uvw[:, 1] / np.maximum(uvw[:, 2], 1e-6)).astype(np.int32)
        inside = (u >= 0) & (u < pixel_mask.shape[1]) & (v >= 0) & (v < pixel_mask.shape[0])
        keep = np.zeros((len(points),), dtype=bool)
        valid_indices = np.where(valid)[0]
        selected = np.zeros_like(inside, dtype=bool)
        if np.any(inside):
            selected[inside] = pixel_mask[v[inside], u[inside]]
        keep_indices = valid_indices[selected]
        keep[keep_indices] = True
        filtered_clouds.append({**camera_cloud, "points": points[keep], "colors": colors[keep]})
        metrics.append(
            {
                "camera_idx": camera_idx,
                "seed_mask_pixels": int(np.count_nonzero(pixel_mask)),
                "seed_object_points": int(np.count_nonzero(keep)),
            }
        )
    return filtered_clouds, metrics


def point_mask_from_pixel_mask(
    camera_cloud: dict[str, Any],
    *,
    pixel_mask: np.ndarray | None,
) -> np.ndarray:
    points = np.asarray(camera_cloud["points"], dtype=np.float32).reshape(-1, 3)
    if pixel_mask is None or len(points) == 0:
        return np.zeros((len(points),), dtype=bool)
    c2w = np.asarray(camera_cloud["c2w"], dtype=np.float32).reshape(4, 4)
    w2c = np.linalg.inv(c2w)
    homogeneous = np.concatenate([points, np.ones((len(points), 1), dtype=np.float32)], axis=1)
    camera_points = homogeneous @ w2c.T
    xyz = camera_points[:, :3]
    valid = xyz[:, 2] > 1e-6
    keep = np.zeros((len(points),), dtype=bool)
    if not np.any(valid):
        return keep
    K = np.asarray(camera_cloud["K_color"], dtype=np.float32).reshape(3, 3)
    uvw = xyz[valid] @ K.T
    u = np.rint(uvw[:, 0] / np.maximum(uvw[:, 2], 1e-6)).astype(np.int32)
    v = np.rint(uvw[:, 1] / np.maximum(uvw[:, 2], 1e-6)).astype(np.int32)
    inside = (u >= 0) & (u < pixel_mask.shape[1]) & (v >= 0) & (v < pixel_mask.shape[0])
    selected = np.zeros_like(inside, dtype=bool)
    if np.any(inside):
        selected[inside] = pixel_mask[v[inside], u[inside]]
    keep[np.where(valid)[0][selected]] = True
    return keep


def world_bbox_corners(
    object_roi_min: np.ndarray,
    object_roi_max: np.ndarray,
) -> np.ndarray:
    bbox_min = np.asarray(object_roi_min, dtype=np.float32)
    bbox_max = np.asarray(object_roi_max, dtype=np.float32)
    corners: list[np.ndarray] = []
    for x_value in (bbox_min[0], bbox_max[0]):
        for y_value in (bbox_min[1], bbox_max[1]):
            for z_value in (bbox_min[2], bbox_max[2]):
                corners.append(np.array([x_value, y_value, z_value], dtype=np.float32))
    return np.stack(corners, axis=0)


def project_world_roi_to_camera_bbox(
    camera_cloud: dict[str, Any],
    *,
    object_roi_min: np.ndarray,
    object_roi_max: np.ndarray,
    extra_world_points: np.ndarray | None = None,
    padding_ratio: float = 0.14,
    min_pad_px: int = 12,
    max_pad_px: int = 56,
) -> tuple[tuple[int, int, int, int] | None, dict[str, Any]]:
    color_path = camera_cloud["color_path"]
    image = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Missing RGB image for projected bbox: {color_path}")
    roi_points = world_bbox_corners(object_roi_min, object_roi_max)
    if extra_world_points is not None and len(extra_world_points) > 0:
        roi_points = np.concatenate([roi_points, np.asarray(extra_world_points, dtype=np.float32).reshape(-1, 3)], axis=0)
    projected = _project_world_points_to_pixels(
        roi_points,
        c2w=np.asarray(camera_cloud["c2w"], dtype=np.float32),
        K_color=np.asarray(camera_cloud["K_color"], dtype=np.float32),
        image_shape=image.shape[:2],
    )
    if len(projected) == 0:
        return None, {
            "camera_idx": int(camera_cloud["camera_idx"]),
            "source": "projected_world_roi",
            "visible_corner_count": 0,
            "bbox": None,
        }
    x_min = int(np.min(projected[:, 0]))
    y_min = int(np.min(projected[:, 1]))
    x_max = int(np.max(projected[:, 0]))
    y_max = int(np.max(projected[:, 1]))
    width = max(1, x_max - x_min + 1)
    height = max(1, y_max - y_min + 1)
    pad = int(np.clip(round(max(width, height) * float(padding_ratio)), int(min_pad_px), int(max_pad_px)))
    x_min = max(0, x_min - pad)
    y_min = max(0, y_min - pad)
    x_max = min(int(image.shape[1]) - 1, x_max + pad)
    y_max = min(int(image.shape[0]) - 1, y_max + pad)
    if x_min >= x_max or y_min >= y_max:
        return None, {
            "camera_idx": int(camera_cloud["camera_idx"]),
            "source": "projected_world_roi",
            "visible_corner_count": int(len(projected)),
            "bbox": None,
        }
    bbox = (int(x_min), int(y_min), int(x_max), int(y_max))
    return bbox, {
        "camera_idx": int(camera_cloud["camera_idx"]),
        "source": "projected_world_roi",
        "visible_corner_count": int(len(projected)),
        "bbox": [int(x_min), int(y_min), int(x_max), int(y_max)],
    }


def build_refined_pixel_masks_from_bboxes(
    camera_clouds: list[dict[str, Any]],
    *,
    roi_by_camera: dict[int, tuple[int, int, int, int]],
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
    object_height_min: float,
    object_height_max: float,
) -> tuple[dict[int, np.ndarray], dict[int, dict[str, Any]]]:
    pixel_masks: dict[int, np.ndarray] = {}
    debug_by_camera: dict[int, dict[str, Any]] = {}
    for camera_cloud in camera_clouds:
        camera_idx = int(camera_cloud["camera_idx"])
        roi = roi_by_camera.get(camera_idx)
        if roi is None:
            continue
        mask, mask_metrics = build_geometry_constrained_foreground_mask(
            camera_cloud,
            roi=roi,
            plane_point=plane_point,
            plane_normal=plane_normal,
            object_height_min=float(object_height_min),
            object_height_max=float(object_height_max),
        )
        pixel_masks[camera_idx] = mask
        debug_by_camera[camera_idx] = {
            "camera_idx": camera_idx,
            "bbox": [int(item) for item in roi],
            **mask_metrics,
        }
    return pixel_masks, debug_by_camera


def _overlay_object_pixels_on_rgb(
    color_path: str | Path,
    *,
    object_points: np.ndarray,
    c2w: np.ndarray,
    K_color: np.ndarray,
    title: str,
) -> tuple[np.ndarray, int]:
    image = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Missing debug color frame: {color_path}")
    pixels = _project_world_points_to_pixels(
        object_points,
        c2w=c2w,
        K_color=K_color,
        image_shape=image.shape[:2],
    )
    overlay = image.copy()
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    if len(pixels) > 0:
        mask[pixels[:, 1], pixels[:, 0]] = 255
        mask = cv2.dilate(mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
        green = np.zeros_like(image)
        green[..., 1] = 220
        alpha = (mask.astype(np.float32) / 255.0)[:, :, None] * 0.65
        overlay = np.clip(image.astype(np.float32) * (1.0 - alpha) + green.astype(np.float32) * alpha, 0.0, 255.0).astype(np.uint8)
    cv2.rectangle(overlay, (0, 0), (overlay.shape[1] - 1, 34), (0, 0, 0), -1)
    cv2.putText(overlay, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(overlay, f"object pixels={int(np.count_nonzero(mask))}", (10, overlay.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (80, 220, 120), 2, cv2.LINE_AA)
    return overlay, int(np.count_nonzero(mask))


def _scalar_bounds(points: np.ndarray) -> dict[str, tuple[float, float]]:
    cloud = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if len(cloud) == 0:
        return {"height": (-0.1, 0.1), "depth": (0.0, 1.0)}
    bounds_min = cloud.min(axis=0)
    bounds_max = cloud.max(axis=0)
    return {
        "height": (float(bounds_min[2]), float(bounds_max[2])),
        "depth": (0.0, max(float(np.linalg.norm(bounds_max - bounds_min)) * 2.0, 0.6)),
    }


def _render_cloud_preview(
    points: np.ndarray,
    colors: np.ndarray,
    *,
    focus_point: np.ndarray,
    camera_position: np.ndarray | None,
    up: np.ndarray | None,
    renderer: str,
    render_mode: str,
) -> np.ndarray:
    cloud = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    color_values = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    if len(cloud) == 0:
        return np.zeros((480, 640, 3), dtype=np.uint8)
    if camera_position is None or up is None:
        bounds_min = cloud.min(axis=0)
        bounds_max = cloud.max(axis=0)
        view_config = compute_view_config(bounds_min, bounds_max, view_name="oblique")
        view_config["center"] = np.asarray(focus_point, dtype=np.float32)
    else:
        view_config = {
            "view_name": "debug",
            "label": "Debug",
            "center": np.asarray(focus_point, dtype=np.float32),
            "camera_position": np.asarray(camera_position, dtype=np.float32),
            "up": np.asarray(up, dtype=np.float32),
        }
    image, _ = render_point_cloud(
        cloud,
        color_values,
        renderer=renderer,
        view_config=view_config,
        render_mode=render_mode,
        scalar_bounds=_scalar_bounds(cloud),
        width=640,
        height=480,
        point_radius_px=3,
        supersample_scale=2,
        projection_mode="perspective",
        ortho_scale=None,
    )
    return image


def write_object_debug_artifacts(
    *,
    output_dir: Path,
    source_name: str,
    object_layers: dict[str, Any],
    focus_point: np.ndarray,
    renderer: str,
    render_mode: str,
) -> dict[str, Any]:
    debug_root = Path(output_dir).resolve()
    mask_dir = debug_root / "per_camera_object_mask_overlay"
    camera_cloud_dir = debug_root / "per_camera_object_cloud"
    mask_dir.mkdir(parents=True, exist_ok=True)
    camera_cloud_dir.mkdir(parents=True, exist_ok=True)

    overlay_paths: list[str] = []
    preview_paths: list[str] = []
    metrics_by_camera: list[dict[str, Any]] = []
    for camera_cloud, camera_metrics in zip(
        object_layers["object_camera_clouds"],
        object_layers["per_camera_metrics"],
        strict=False,
    ):
        overlay_image, roi_valid_pixels = _overlay_object_pixels_on_rgb(
            camera_cloud["color_path"],
            object_points=camera_cloud["points"],
            c2w=np.asarray(camera_cloud["c2w"], dtype=np.float32),
            K_color=np.asarray(camera_cloud["K_color"], dtype=np.float32),
            title=f"{source_name} Cam{camera_cloud['camera_idx']} object mask",
        )
        overlay_path = mask_dir / f"{source_name}_cam{camera_cloud['camera_idx']}.png"
        cv2.imwrite(str(overlay_path), overlay_image)
        overlay_paths.append(str(overlay_path))

        preview_image = _render_cloud_preview(
            camera_cloud["points"],
            camera_cloud["colors"],
            focus_point=focus_point,
            camera_position=np.asarray(camera_cloud["c2w"], dtype=np.float32)[:3, 3],
            up=-np.asarray(camera_cloud["c2w"], dtype=np.float32)[:3, 1],
            renderer=renderer,
            render_mode="color_by_rgb",
        )
        preview_path = camera_cloud_dir / f"{source_name}_cam{camera_cloud['camera_idx']}.png"
        cv2.imwrite(str(preview_path), preview_image)
        preview_paths.append(str(preview_path))
        ply_path = camera_cloud_dir / f"{source_name}_cam{camera_cloud['camera_idx']}.ply"
        write_ply_ascii(ply_path, camera_cloud["points"], camera_cloud["colors"])

        metrics_with_pixels = dict(camera_metrics)
        metrics_with_pixels["roi_valid_pixels"] = int(roi_valid_pixels)
        metrics_by_camera.append(metrics_with_pixels)

    return {
        "overlay_paths": overlay_paths,
        "preview_paths": preview_paths,
        "per_camera_metrics": metrics_by_camera,
    }


def write_fused_cloud_debug_artifacts(
    *,
    output_dir: Path,
    source_name: str,
    object_points: np.ndarray,
    object_colors: np.ndarray,
    combined_points: np.ndarray,
    combined_colors: np.ndarray,
    focus_point: np.ndarray,
    renderer: str,
) -> dict[str, str]:
    object_dir = Path(output_dir).resolve() / "fused_object_only"
    context_dir = Path(output_dir).resolve() / "fused_object_context"
    object_dir.mkdir(parents=True, exist_ok=True)
    context_dir.mkdir(parents=True, exist_ok=True)

    object_ply = object_dir / f"{source_name}_object_only.ply"
    combined_ply = context_dir / f"{source_name}_object_context.ply"
    write_ply_ascii(object_ply, object_points, object_colors)
    write_ply_ascii(combined_ply, combined_points, combined_colors)

    object_preview = _render_cloud_preview(
        object_points,
        object_colors,
        focus_point=focus_point,
        camera_position=None,
        up=None,
        renderer=renderer,
        render_mode="color_by_rgb",
    )
    combined_preview = _render_cloud_preview(
        combined_points,
        combined_colors,
        focus_point=focus_point,
        camera_position=None,
        up=None,
        renderer=renderer,
        render_mode="color_by_rgb",
    )
    object_png = object_dir / f"{source_name}_object_only.png"
    combined_png = context_dir / f"{source_name}_object_context.png"
    cv2.imwrite(str(object_png), object_preview)
    cv2.imwrite(str(combined_png), combined_preview)
    return {
        "object_ply": str(object_ply),
        "object_png": str(object_png),
        "combined_ply": str(combined_ply),
        "combined_png": str(combined_png),
    }


def write_compare_debug_metrics(
    output_path: Path,
    *,
    debug_payload: dict[str, Any],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(debug_payload, indent=2), encoding="utf-8")
