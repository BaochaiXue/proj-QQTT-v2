from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .calibration_io import load_calibration_transforms
from .pointcloud_compare import (
    choose_depth_stream,
    decode_depth_to_meters,
    get_case_intrinsics,
    get_depth_scale_list,
    get_frame_count,
    load_case_metadata,
    resolve_case_dirs,
    select_frame_indices,
)

INVALID_COLOR_BGR = (96, 0, 96)
DIFF_INVALID_COLOR_BGR = (32, 32, 32)
VALID_MASK_COLORS = {
    "both_valid": np.array([0, 200, 0], dtype=np.uint8),
    "native_only": np.array([0, 0, 255], dtype=np.uint8),
    "ffs_only": np.array([255, 128, 0], dtype=np.uint8),
    "both_invalid": np.array([0, 0, 0], dtype=np.uint8),
}


def parse_roi_spec(spec: str) -> tuple[int, int, int, int]:
    parts = [part.strip() for part in spec.split(",")]
    if len(parts) != 4:
        raise ValueError(f"ROI must have format x0,y0,x1,y1: {spec}")
    x0, y0, x1, y1 = [int(part) for part in parts]
    if x0 >= x1 or y0 >= y1:
        raise ValueError(f"ROI must satisfy x0<x1 and y0<y1: {spec}")
    return x0, y0, x1, y1


def clamp_roi(roi: tuple[int, int, int, int], image_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    height, width = image_shape[:2]
    x0, y0, x1, y1 = roi
    x0 = max(0, min(width - 1, x0))
    y0 = max(0, min(height - 1, y0))
    x1 = max(x0 + 1, min(width, x1))
    y1 = max(y0 + 1, min(height, y1))
    return x0, y0, x1, y1


def default_rois(image_shape: tuple[int, int]) -> list[tuple[int, int, int, int]]:
    height, width = image_shape[:2]
    rois = [
        (
            int(width * 0.35),
            int(height * 0.35),
            int(width * 0.65),
            int(height * 0.65),
        ),
        (
            int(width * 0.55),
            int(height * 0.15),
            int(width * 0.90),
            int(height * 0.50),
        ),
    ]
    return [clamp_roi(roi, image_shape) for roi in rois]


def resolve_camera_ids(metadata: dict[str, Any], camera_ids: list[int] | None) -> list[int]:
    num_cameras = len(metadata["serial_numbers"])
    if camera_ids is None or len(camera_ids) == 0:
        return list(range(num_cameras))
    resolved = sorted({int(camera_id) for camera_id in camera_ids})
    invalid = [camera_id for camera_id in resolved if camera_id < 0 or camera_id >= num_cameras]
    if invalid:
        raise ValueError(f"Invalid camera ids {invalid}; case has cameras 0..{num_cameras - 1}")
    return resolved


def _resize_tile(image: np.ndarray, tile_size: tuple[int, int]) -> np.ndarray:
    tile_w, tile_h = tile_size
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return cv2.resize(image, (tile_w, tile_h), interpolation=cv2.INTER_AREA)


def label_tile(image: np.ndarray, label: str, tile_size: tuple[int, int]) -> np.ndarray:
    tile = _resize_tile(image, tile_size)
    cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, 22), (0, 0, 0), -1)
    cv2.putText(
        tile,
        label,
        (8, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return tile


def compose_grid(tiles: list[np.ndarray], *, columns: int) -> np.ndarray:
    if not tiles:
        raise ValueError("No tiles provided.")
    columns = max(1, columns)
    rows = int(np.ceil(len(tiles) / columns))
    tile_h, tile_w = tiles[0].shape[:2]
    canvas = np.zeros((rows * tile_h, columns * tile_w, 3), dtype=np.uint8)
    for idx, tile in enumerate(tiles):
        row = idx // columns
        col = idx % columns
        y0 = row * tile_h
        x0 = col * tile_w
        canvas[y0:y0 + tile_h, x0:x0 + tile_w] = tile
    return canvas


def colorize_depth_map(
    depth_m: np.ndarray,
    *,
    depth_min_m: float,
    depth_max_m: float,
    invalid_color: tuple[int, int, int] = INVALID_COLOR_BGR,
) -> np.ndarray:
    depth = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    canvas = np.zeros(depth.shape + (3,), dtype=np.uint8)
    if np.any(valid):
        normalized = np.clip((depth - float(depth_min_m)) / max(1e-6, float(depth_max_m) - float(depth_min_m)), 0.0, 1.0)
        colorized = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        canvas[valid] = colorized[valid]
    canvas[~valid] = np.asarray(invalid_color, dtype=np.uint8)
    return canvas


def absolute_depth_difference_heatmap(
    native_depth_m: np.ndarray,
    ffs_depth_m: np.ndarray,
    *,
    max_diff_m: float | None = None,
) -> np.ndarray:
    native = np.asarray(native_depth_m, dtype=np.float32)
    ffs = np.asarray(ffs_depth_m, dtype=np.float32)
    valid = np.isfinite(native) & np.isfinite(ffs) & (native > 0) & (ffs > 0)
    heatmap = np.zeros(native.shape + (3,), dtype=np.uint8)
    if not np.any(valid):
        heatmap[:] = np.asarray(DIFF_INVALID_COLOR_BGR, dtype=np.uint8)
        return heatmap
    diff = np.abs(native - ffs)
    max_value = max_diff_m
    if max_value is None:
        max_value = max(0.02, float(np.percentile(diff[valid], 95)))
    normalized = np.clip(diff / max(1e-6, float(max_value)), 0.0, 1.0)
    colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    heatmap[valid] = colored[valid]
    heatmap[~valid] = np.asarray(DIFF_INVALID_COLOR_BGR, dtype=np.uint8)
    return heatmap


def valid_mask_comparison(native_depth_m: np.ndarray, ffs_depth_m: np.ndarray) -> np.ndarray:
    native_valid = np.isfinite(native_depth_m) & (np.asarray(native_depth_m) > 0)
    ffs_valid = np.isfinite(ffs_depth_m) & (np.asarray(ffs_depth_m) > 0)
    canvas = np.zeros(native_valid.shape + (3,), dtype=np.uint8)
    canvas[~native_valid & ~ffs_valid] = VALID_MASK_COLORS["both_invalid"]
    canvas[native_valid & ~ffs_valid] = VALID_MASK_COLORS["native_only"]
    canvas[~native_valid & ffs_valid] = VALID_MASK_COLORS["ffs_only"]
    canvas[native_valid & ffs_valid] = VALID_MASK_COLORS["both_valid"]
    return canvas


def _depth_to_xyz_grid(depth_m: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    depth = np.asarray(depth_m, dtype=np.float32)
    K = np.asarray(K, dtype=np.float32).reshape(3, 3)
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    yy, xx = np.indices(depth.shape, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    xyz = np.zeros(depth.shape + (3,), dtype=np.float32)
    xyz[..., 2] = depth
    xyz[..., 0] = (xx - cx) * depth / fx
    xyz[..., 1] = (yy - cy) * depth / fy
    return xyz, valid


def compute_depth_normals(depth_m: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xyz, valid = _depth_to_xyz_grid(depth_m, K)
    normals = np.zeros_like(xyz, dtype=np.float32)
    if xyz.shape[0] < 3 or xyz.shape[1] < 3:
        return normals, np.zeros(valid.shape, dtype=bool)

    left = xyz[1:-1, :-2]
    right = xyz[1:-1, 2:]
    up = xyz[:-2, 1:-1]
    down = xyz[2:, 1:-1]
    valid_inner = (
        valid[1:-1, 1:-1]
        & valid[1:-1, :-2]
        & valid[1:-1, 2:]
        & valid[:-2, 1:-1]
        & valid[2:, 1:-1]
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
    valid_normals = np.zeros(valid.shape, dtype=bool)
    valid_normals[1:-1, 1:-1] = good
    return normals, valid_normals


def normal_rgb_map(depth_m: np.ndarray, K: np.ndarray) -> np.ndarray:
    normals, valid = compute_depth_normals(depth_m, K)
    canvas = np.zeros(normals.shape, dtype=np.uint8)
    if np.any(valid):
        rgb = ((normals + 1.0) * 0.5 * 255.0).astype(np.uint8)
        canvas[valid] = rgb[valid][:, ::-1]
    canvas[~valid] = np.asarray(DIFF_INVALID_COLOR_BGR, dtype=np.uint8)
    return canvas


def shaded_depth_map(depth_m: np.ndarray, K: np.ndarray) -> np.ndarray:
    normals, valid = compute_depth_normals(depth_m, K)
    canvas = np.zeros(normals.shape, dtype=np.uint8)
    if np.any(valid):
        light_dir = np.asarray([0.35, -0.25, -1.0], dtype=np.float32)
        light_dir /= np.linalg.norm(light_dir)
        intensity = np.clip((normals @ light_dir) * 0.5 + 0.5, 0.0, 1.0)
        shaded = np.clip(40.0 + 215.0 * intensity, 0.0, 255.0).astype(np.uint8)
        canvas[valid] = np.stack([shaded, shaded, shaded], axis=2)[valid]
    canvas[~valid] = np.asarray(DIFF_INVALID_COLOR_BGR, dtype=np.uint8)
    return canvas


def annotate_rois(image: np.ndarray, rois: list[tuple[int, int, int, int]]) -> np.ndarray:
    canvas = np.asarray(image).copy()
    colors = [(0, 255, 255), (255, 255, 0), (0, 200, 255), (255, 128, 0)]
    for idx, roi in enumerate(rois):
        x0, y0, x1, y1 = roi
        color = colors[idx % len(colors)]
        cv2.rectangle(canvas, (x0, y0), (x1 - 1, y1 - 1), color, 2)
        cv2.putText(canvas, f"ROI {idx + 1}", (x0 + 4, max(16, y0 + 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return canvas


def make_roi_tile(
    rgb_image: np.ndarray,
    native_depth_vis: np.ndarray,
    ffs_depth_vis: np.ndarray,
    diff_heatmap: np.ndarray,
    roi: tuple[int, int, int, int],
    *,
    tile_size: tuple[int, int],
) -> np.ndarray:
    x0, y0, x1, y1 = roi
    rgb_crop = rgb_image[y0:y1, x0:x1]
    native_crop = native_depth_vis[y0:y1, x0:x1]
    ffs_crop = ffs_depth_vis[y0:y1, x0:x1]
    diff_crop = diff_heatmap[y0:y1, x0:x1]
    crop_w = max(64, tile_size[0] // 2)
    crop_h = max(64, tile_size[1] // 2)
    tiles = [
        label_tile(rgb_crop, "RGB Crop", (crop_w, crop_h)),
        label_tile(native_crop, "Native Crop", (crop_w, crop_h)),
        label_tile(ffs_crop, "FFS Crop", (crop_w, crop_h)),
        label_tile(diff_crop, "|Δ| Crop", (crop_w, crop_h)),
    ]
    return compose_grid(tiles, columns=2)


def load_depth_frame(
    *,
    case_dir: Path,
    metadata: dict[str, Any],
    camera_idx: int,
    frame_idx: int,
    depth_source: str,
    use_float_ffs_depth_when_available: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    depth_dir_name, use_float = choose_depth_stream(case_dir, metadata, depth_source, use_float_ffs_depth_when_available)
    depth_path = case_dir / depth_dir_name / str(camera_idx) / f"{frame_idx}.npy"
    if not depth_path.exists():
        raise FileNotFoundError(f"Missing depth frame: {depth_path}")
    depth_raw = np.load(depth_path)
    depth_scales = get_depth_scale_list(metadata, len(metadata["serial_numbers"]))
    depth_m = decode_depth_to_meters(depth_raw, None if use_float else depth_scales[camera_idx])
    return depth_raw, depth_m, {
        "depth_dir_used": depth_dir_name,
        "used_float_depth": bool(use_float),
        "depth_path": str(depth_path),
    }


def load_color_frame(case_dir: Path, camera_idx: int, frame_idx: int) -> np.ndarray:
    color_path = case_dir / "color" / str(camera_idx) / f"{frame_idx}.png"
    image = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Missing color frame: {color_path}")
    return image


def get_case_camera_transform(
    *,
    case_dir: Path,
    metadata: dict[str, Any],
) -> list[np.ndarray]:
    calibration_reference_serials = metadata.get("calibration_reference_serials", metadata["serial_numbers"])
    return load_calibration_transforms(
        case_dir / "calibrate.pkl",
        serial_numbers=metadata["serial_numbers"],
        calibration_reference_serials=calibration_reference_serials,
    )


def _backproject_depth(depth_m: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    depth = np.asarray(depth_m, dtype=np.float32)
    K = np.asarray(K, dtype=np.float32).reshape(3, 3)
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    yy, xx = np.indices(depth.shape, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    z = depth[valid]
    x = (xx[valid] - cx) * z / fx
    y = (yy[valid] - cy) * z / fy
    points = np.stack([x, y, z], axis=1)
    pixels = np.stack([xx[valid], yy[valid]], axis=1)
    return points, pixels


def _apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    T = np.asarray(transform, dtype=np.float32).reshape(4, 4)
    homogeneous = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float32)], axis=1)
    transformed = homogeneous @ T.T
    return transformed[:, :3]


def _project_points(points: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float32)
    if pts.size == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)
    K = np.asarray(K, dtype=np.float32).reshape(3, 3)
    z = pts[:, 2]
    valid = np.isfinite(z) & (z > 0)
    uv = np.full((pts.shape[0], 2), np.nan, dtype=np.float32)
    if np.any(valid):
        normalized = pts[valid] / z[valid, None]
        projected = normalized @ K.T
        uv[valid] = projected[:, :2]
    return uv, z


def warp_rgb_between_cameras(
    *,
    source_rgb: np.ndarray,
    source_depth_m: np.ndarray,
    source_K: np.ndarray,
    source_c2w: np.ndarray,
    target_K: np.ndarray,
    target_c2w: np.ndarray,
    output_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_rgb = np.asarray(source_rgb, dtype=np.uint8)
    points_src, pixels_src = _backproject_depth(source_depth_m, source_K)
    if len(points_src) == 0:
        height, width = output_shape
        return (
            np.zeros((height, width, 3), dtype=np.uint8),
            np.zeros((height, width), dtype=bool),
            np.zeros((height, width), dtype=np.float32),
        )

    source_colors = source_rgb[pixels_src[:, 1].astype(np.int32), pixels_src[:, 0].astype(np.int32)]
    points_world = _apply_transform(points_src, source_c2w)
    target_w2c = np.linalg.inv(np.asarray(target_c2w, dtype=np.float32).reshape(4, 4))
    points_target = _apply_transform(points_world, target_w2c)
    uv, z_values = _project_points(points_target, target_K)

    height, width = output_shape
    warped = np.zeros((height, width, 3), dtype=np.uint8)
    valid_mask = np.zeros((height, width), dtype=bool)
    depth_map = np.zeros((height, width), dtype=np.float32)
    valid = np.isfinite(uv[:, 0]) & np.isfinite(uv[:, 1]) & np.isfinite(z_values) & (z_values > 0)
    if not np.any(valid):
        return warped, valid_mask, depth_map

    uv_valid = uv[valid]
    z = z_values[valid]
    colors = source_colors[valid]
    finite_uv = np.isfinite(uv_valid[:, 0]) & np.isfinite(uv_valid[:, 1])
    uv_valid = uv_valid[finite_uv]
    z = z[finite_uv]
    colors = colors[finite_uv]
    if len(uv_valid) == 0:
        return warped, valid_mask, depth_map
    coords = np.rint(np.nan_to_num(uv_valid, nan=-1.0)).astype(np.int32)
    inside = (coords[:, 0] >= 0) & (coords[:, 0] < width) & (coords[:, 1] >= 0) & (coords[:, 1] < height)
    coords = coords[inside]
    z = z[inside]
    colors = colors[inside]
    if len(coords) == 0:
        return warped, valid_mask, depth_map

    order = np.argsort(z)[::-1]
    coords = coords[order]
    z = z[order]
    colors = colors[order]
    warped[coords[:, 1], coords[:, 0]] = colors
    depth_map[coords[:, 1], coords[:, 0]] = z
    valid_mask[coords[:, 1], coords[:, 0]] = True
    return warped, valid_mask, depth_map


def compute_photometric_residual(
    warped_rgb: np.ndarray,
    target_rgb: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    warped = np.asarray(warped_rgb, dtype=np.float32)
    target = np.asarray(target_rgb, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)
    residual = np.abs(warped - target).mean(axis=2)
    heatmap = np.zeros(warped.shape, dtype=np.uint8)
    stats = {
        "valid_warped_pixel_ratio": float(valid.mean()),
        "residual_mean": 0.0,
        "residual_median": 0.0,
        "edge_weighted_residual_mean": 0.0,
    }
    if np.any(valid):
        clipped = np.clip(residual / 64.0, 0.0, 1.0)
        colored = cv2.applyColorMap((clipped * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        heatmap[valid] = colored[valid]

        valid_residual = residual[valid]
        stats["residual_mean"] = float(valid_residual.mean())
        stats["residual_median"] = float(np.median(valid_residual))

        target_gray = cv2.cvtColor(np.asarray(target_rgb, dtype=np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx = cv2.Sobel(target_gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(target_gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient = np.sqrt(gx * gx + gy * gy)
        weights = 1.0 + (gradient / max(1e-6, float(gradient.max())))
        edge_weights = weights[valid]
        stats["edge_weighted_residual_mean"] = float(np.sum(valid_residual * edge_weights) / np.sum(edge_weights))
    else:
        heatmap[:] = np.asarray(DIFF_INVALID_COLOR_BGR, dtype=np.uint8)
    heatmap[~valid] = np.asarray(DIFF_INVALID_COLOR_BGR, dtype=np.uint8)
    return heatmap, stats
