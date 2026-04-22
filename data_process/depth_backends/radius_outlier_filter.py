from __future__ import annotations

from typing import Any

import numpy as np

from .geometry import quantize_depth_with_invalid_zero


FFS_RADIUS_OUTLIER_FILTER_MODE = "per_camera_color_frame_radius_outlier"
FFS_RADIUS_OUTLIER_FILTER_ARCHIVE_POLICY = "replace_main_and_archive_raw"
FFS_DEPTH_ARCHIVE_DIR_FFS_BACKEND = "depth_original"
FFS_DEPTH_ARCHIVE_DIR_BOTH_BACKEND = "depth_ffs_original"
FFS_FLOAT_ARCHIVE_DIR = "depth_ffs_float_m_original"


def build_ffs_radius_outlier_filter_contract(*, radius_m: float, nb_points: int) -> dict[str, Any]:
    return {
        "mode": FFS_RADIUS_OUTLIER_FILTER_MODE,
        "radius_m": float(radius_m),
        "nb_points": int(nb_points),
        "archive_policy": FFS_RADIUS_OUTLIER_FILTER_ARCHIVE_POLICY,
    }


def _load_open3d():
    try:
        import open3d as o3d
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Open3D is required for --ffs_radius_outlier_filter but is not installed in this environment."
        ) from exc
    return o3d


def _backproject_valid_depth_pixels(depth_m: np.ndarray, K_color: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"Expected 2D depth image, got shape={depth.shape}.")
    K = np.asarray(K_color, dtype=np.float32).reshape(3, 3)
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    yy, xx = np.indices(depth.shape, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 2), dtype=np.int32)

    z = depth[valid]
    x = (xx[valid] - cx) * z / max(fx, 1e-6)
    y = (yy[valid] - cy) * z / max(fy, 1e-6)
    points = np.stack([x, y, z], axis=1).astype(np.float32)
    pixel_uv = np.stack([xx[valid], yy[valid]], axis=1).astype(np.int32)
    return points, pixel_uv


def apply_ffs_radius_outlier_filter_float_m(
    depth_color_m: np.ndarray,
    *,
    K_color: np.ndarray,
    radius_m: float,
    nb_points: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    depth = np.asarray(depth_color_m, dtype=np.float32)
    points, pixel_uv = _backproject_valid_depth_pixels(depth, K_color)
    filtered_depth = depth.copy()
    valid_point_count = int(len(points))
    if valid_point_count == 0:
        return filtered_depth, {
            "filter_enabled": True,
            **build_ffs_radius_outlier_filter_contract(radius_m=radius_m, nb_points=nb_points),
            "pixel_count": int(depth.size),
            "valid_pixel_count": 0,
            "inlier_pixel_count": 0,
            "outlier_pixel_count": 0,
            "outlier_ratio": 0.0,
        }

    o3d = _load_open3d()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    _, inlier_indices = pcd.remove_radius_outlier(
        nb_points=int(nb_points),
        radius=float(radius_m),
    )
    inlier_indices = np.asarray(inlier_indices, dtype=np.int32).reshape(-1)
    keep_mask = np.zeros((valid_point_count,), dtype=bool)
    if len(inlier_indices) > 0:
        keep_mask[inlier_indices] = True
    outlier_uv = pixel_uv[~keep_mask]
    if len(outlier_uv) > 0:
        filtered_depth[outlier_uv[:, 1], outlier_uv[:, 0]] = 0.0

    outlier_count = int(len(outlier_uv))
    return filtered_depth, {
        "filter_enabled": True,
        **build_ffs_radius_outlier_filter_contract(radius_m=radius_m, nb_points=nb_points),
        "pixel_count": int(depth.size),
        "valid_pixel_count": valid_point_count,
        "inlier_pixel_count": int(valid_point_count - outlier_count),
        "outlier_pixel_count": outlier_count,
        "outlier_ratio": float(outlier_count / max(1, valid_point_count)),
    }


def apply_ffs_radius_outlier_filter_u16(
    depth_color_m: np.ndarray,
    *,
    K_color: np.ndarray,
    depth_scale_m_per_unit: float,
    radius_m: float,
    nb_points: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    filtered_depth_m, stats = apply_ffs_radius_outlier_filter_float_m(
        depth_color_m,
        K_color=K_color,
        radius_m=radius_m,
        nb_points=nb_points,
    )
    filtered_depth_u16 = quantize_depth_with_invalid_zero(filtered_depth_m, depth_scale_m_per_unit)
    return filtered_depth_u16, filtered_depth_m, stats
