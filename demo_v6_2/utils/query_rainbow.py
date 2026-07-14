"""Rainbow palettes for tracking query-point visualization."""

from __future__ import annotations

import matplotlib
import numpy as np


def query_rainbow_colors_from_points_yx_rgb_u8(query_points_yx: np.ndarray) -> np.ndarray:
    """Generate query colors sorted by first-frame YX point order."""
    points = np.asarray(query_points_yx, dtype=np.float32).reshape(-1, 2)
    if len(points) == 0:
        return np.empty((0, 3), dtype=np.uint8)
    y = points[:, 0]
    y_min = np.nanmin(y)
    y_max = np.nanmax(y)
    span = y_max - y_min
    if not np.isfinite(span) or span <= np.float32(1e-6):
        normalized = np.zeros((len(points),), dtype=np.float32)
    else:
        normalized = np.clip((y - y_min) / span, 0.0, 1.0).astype(np.float32)
    rgba = matplotlib.colormaps.get_cmap("gist_rainbow")(normalized)
    rgb = np.asarray(rgba[:, :3], dtype=np.float32)
    return np.ascontiguousarray(np.clip(rgb * np.float32(255.0), 0, 255).astype(np.uint8))
