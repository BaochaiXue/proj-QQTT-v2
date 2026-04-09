from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .renderers import rasterize_point_cloud_view


SUPPORT_COLORS_BGR = {
    0: (38, 42, 48),
    1: (70, 140, 255),
    2: (110, 220, 180),
    3: (120, 235, 120),
}


def compute_support_count_map(
    camera_clouds: list[dict[str, Any]],
    *,
    view_config: dict[str, Any],
    width: int,
    height: int,
    projection_mode: str,
    ortho_scale: float | None,
    depth_tolerance_m: float = 0.02,
    depth_tolerance_ratio: float = 0.03,
) -> dict[str, Any]:
    if not camera_clouds:
        return {
            "support_count": np.zeros((height, width), dtype=np.uint8),
            "valid": np.zeros((height, width), dtype=bool),
            "fused_depth": np.zeros((height, width), dtype=np.float32),
        }

    depth_stack = []
    valid_stack = []
    for cloud in camera_clouds:
        raster = rasterize_point_cloud_view(
            np.asarray(cloud["points"], dtype=np.float32),
            np.asarray(cloud["colors"], dtype=np.uint8),
            view_config=view_config,
            width=width,
            height=height,
            projection_mode=projection_mode,
            ortho_scale=ortho_scale,
        )
        depth_map = np.asarray(raster["depth"], dtype=np.float32)
        valid_map = np.asarray(raster["valid"], dtype=bool)
        depth_map = np.where(valid_map, depth_map, np.inf).astype(np.float32)
        depth_stack.append(depth_map)
        valid_stack.append(valid_map)

    depth_values = np.stack(depth_stack, axis=0)
    valid_values = np.stack(valid_stack, axis=0)
    fused_depth = np.min(depth_values, axis=0)
    fused_valid = np.isfinite(fused_depth)
    tolerance = np.maximum(float(depth_tolerance_m), fused_depth * float(depth_tolerance_ratio)).astype(np.float32)
    agreeing = valid_values & fused_valid[None, :, :]
    delta = np.full(depth_values.shape, np.inf, dtype=np.float32)
    delta[agreeing] = np.abs(depth_values[agreeing] - np.broadcast_to(fused_depth[None, :, :], depth_values.shape)[agreeing])
    agreeing &= delta <= tolerance[None, :, :]
    support_count = np.sum(agreeing, axis=0).astype(np.uint8)
    support_count[~fused_valid] = 0
    return {
        "support_count": support_count,
        "valid": fused_valid,
        "fused_depth": np.where(fused_valid, fused_depth, 0.0).astype(np.float32),
    }


def render_support_count_map(support_count: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    canvas = np.zeros(support_count.shape + (3,), dtype=np.uint8)
    for value, color in SUPPORT_COLORS_BGR.items():
        mask = (support_count == value)
        canvas[mask] = np.asarray(color, dtype=np.uint8)
    canvas[~np.asarray(valid_mask, dtype=bool)] = np.asarray(SUPPORT_COLORS_BGR[0], dtype=np.uint8)
    return canvas


def summarize_support_counts(support_count: np.ndarray, valid_mask: np.ndarray) -> dict[str, Any]:
    valid = np.asarray(valid_mask, dtype=bool)
    total_valid = int(np.count_nonzero(valid))
    summary = {
        "valid_pixel_count": total_valid,
        "support_count_histogram": {},
    }
    for value in (0, 1, 2, 3):
        count = int(np.count_nonzero((support_count == value) & valid))
        summary["support_count_histogram"][str(value)] = count
        summary[f"support_ratio_{value}"] = float(count / total_valid) if total_valid > 0 else 0.0
    return summary


def overlay_support_legend(image: np.ndarray) -> np.ndarray:
    canvas = np.asarray(image, dtype=np.uint8).copy()
    x0 = 18
    y0 = canvas.shape[0] - 96
    cv2.rectangle(canvas, (x0 - 8, y0 - 26), (x0 + 210, y0 + 64), (18, 18, 20), -1)
    cv2.putText(canvas, "Support Count", (x0, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    for idx, value in enumerate((1, 2, 3)):
        y = y0 + idx * 20 + 16
        cv2.rectangle(canvas, (x0, y - 10), (x0 + 18, y + 8), SUPPORT_COLORS_BGR[value], -1)
        cv2.putText(canvas, f"{value} camera", (x0 + 28, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 235, 235), 1, cv2.LINE_AA)
    return canvas
