from __future__ import annotations

from pathlib import Path

import numpy as np


def format_ffs_intrinsic_text(K: np.ndarray, baseline_m: float) -> str:
    matrix = np.asarray(K, dtype=np.float32).reshape(3, 3)
    flattened = " ".join(f"{value:.8f}" for value in matrix.reshape(-1))
    return f"{flattened}\n{baseline_m:.8f}\n"


def write_ffs_intrinsic_file(path: Path, K: np.ndarray, baseline_m: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_ffs_intrinsic_text(K, baseline_m), encoding="utf-8")
    return path


def disparity_to_metric_depth(
    disparity: np.ndarray,
    fx_ir: float,
    baseline_m: float,
    invalid_value: float = 0.0,
) -> np.ndarray:
    disparity = np.asarray(disparity, dtype=np.float32)
    depth = np.full(disparity.shape, invalid_value, dtype=np.float32)
    valid = np.isfinite(disparity) & (disparity > 0)
    if np.any(valid):
        depth[valid] = (float(fx_ir) * float(baseline_m)) / disparity[valid]
    return depth


def unproject_ir_depth(depth_m: np.ndarray, K_ir_left: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    depth = np.asarray(depth_m, dtype=np.float32)
    K = np.asarray(K_ir_left, dtype=np.float32).reshape(3, 3)
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    height, width = depth.shape
    yy, xx = np.indices((height, width), dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    z = depth[valid]
    x = (xx[valid] - cx) * z / fx
    y = (yy[valid] - cy) * z / fy
    points = np.stack([x, y, z], axis=1)
    return points, valid


def transform_points(points: np.ndarray, T_src_to_dst: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    transform = np.asarray(T_src_to_dst, dtype=np.float32).reshape(4, 4)
    if pts.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    homogeneous = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float32)], axis=1)
    transformed = homogeneous @ transform.T
    return transformed[:, :3]


def project_to_color(points_color: np.ndarray, K_color: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_color, dtype=np.float32)
    K = np.asarray(K_color, dtype=np.float32).reshape(3, 3)
    if points.size == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)

    z = points[:, 2]
    valid = np.isfinite(z) & (z > 0)
    uv = np.full((points.shape[0], 2), np.nan, dtype=np.float32)
    if np.any(valid):
        normalized = points[valid] / z[valid, None]
        pixels = normalized @ K.T
        uv[valid] = pixels[:, :2]
    return uv, z


def rasterize_nearest_depth(
    uv: np.ndarray,
    z_values: np.ndarray,
    output_shape: tuple[int, int],
    invalid_value: float = 0.0,
) -> np.ndarray:
    height, width = output_shape
    depth = np.full((height, width), invalid_value, dtype=np.float32)
    uv = np.asarray(uv, dtype=np.float32)
    z_values = np.asarray(z_values, dtype=np.float32)
    valid = np.isfinite(uv[:, 0]) & np.isfinite(uv[:, 1]) & np.isfinite(z_values) & (z_values > 0)
    if not np.any(valid):
        return depth

    coords = np.rint(uv[valid]).astype(np.int32)
    z = z_values[valid]
    inside = (coords[:, 0] >= 0) & (coords[:, 0] < width) & (coords[:, 1] >= 0) & (coords[:, 1] < height)
    coords = coords[inside]
    z = z[inside]
    for (x_coord, y_coord), depth_value in zip(coords, z, strict=False):
        current = depth[y_coord, x_coord]
        if current == invalid_value or depth_value < current:
            depth[y_coord, x_coord] = depth_value
    return depth


def align_depth_to_color(
    depth_ir_m: np.ndarray,
    K_ir_left: np.ndarray,
    T_ir_left_to_color: np.ndarray,
    K_color: np.ndarray,
    output_shape: tuple[int, int],
    invalid_value: float = 0.0,
) -> np.ndarray:
    points_ir, _ = unproject_ir_depth(depth_ir_m, K_ir_left)
    points_color = transform_points(points_ir, T_ir_left_to_color)
    uv_color, z_color = project_to_color(points_color, K_color)
    return rasterize_nearest_depth(uv_color, z_color, output_shape=output_shape, invalid_value=invalid_value)


def quantize_depth_with_invalid_zero(depth_m: np.ndarray, depth_scale_m_per_unit: float) -> np.ndarray:
    depth = np.asarray(depth_m, dtype=np.float32)
    scale = float(depth_scale_m_per_unit)
    encoded = np.zeros(depth.shape, dtype=np.uint16)
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        return encoded
    scaled = np.rint(depth[valid] / scale)
    scaled = np.clip(scaled, 1, np.iinfo(np.uint16).max)
    encoded[valid] = scaled.astype(np.uint16)
    return encoded
