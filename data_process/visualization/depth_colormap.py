from __future__ import annotations

import cv2
import numpy as np


DEFAULT_DEPTH_VIS_MIN_M = 0.1
DEFAULT_DEPTH_VIS_MAX_M = 3.0
INVALID_DEPTH_COLOR_BGR = (96, 0, 96)


def colorize_depth_meters(
    depth_m: np.ndarray,
    *,
    depth_min_m: float = DEFAULT_DEPTH_VIS_MIN_M,
    depth_max_m: float = DEFAULT_DEPTH_VIS_MAX_M,
    invalid_color: tuple[int, int, int] = INVALID_DEPTH_COLOR_BGR,
) -> np.ndarray:
    depth = np.asarray(depth_m, dtype=np.float32)
    if float(depth_max_m) <= float(depth_min_m):
        raise ValueError(
            f"depth_max_m must be greater than depth_min_m. Got {depth_min_m=} {depth_max_m=}"
        )

    valid = np.isfinite(depth) & (depth > 0.0)
    canvas = np.zeros(depth.shape + (3,), dtype=np.uint8)
    if np.any(valid):
        normalized = np.zeros_like(depth, dtype=np.float32)
        normalized[valid] = np.clip(
            (depth[valid] - float(depth_min_m)) / (float(depth_max_m) - float(depth_min_m)),
            0.0,
            1.0,
        )
        colorized = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        canvas[valid] = colorized[valid]
    canvas[~valid] = np.asarray(invalid_color, dtype=np.uint8)
    return canvas


def colorize_depth_units(
    depth_raw: np.ndarray,
    *,
    depth_scale_m_per_unit: float,
    depth_min_m: float = DEFAULT_DEPTH_VIS_MIN_M,
    depth_max_m: float = DEFAULT_DEPTH_VIS_MAX_M,
    invalid_color: tuple[int, int, int] = INVALID_DEPTH_COLOR_BGR,
) -> np.ndarray:
    raw = np.asarray(depth_raw)
    depth_m = raw.astype(np.float32) * float(depth_scale_m_per_unit)
    if np.issubdtype(raw.dtype, np.integer):
        depth_m[raw == 0] = 0.0
    return colorize_depth_meters(
        depth_m,
        depth_min_m=depth_min_m,
        depth_max_m=depth_max_m,
        invalid_color=invalid_color,
    )
