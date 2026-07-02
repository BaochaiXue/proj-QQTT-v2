"""Rainbow palettes for tracking query-point visualization."""

from __future__ import annotations

import matplotlib
import numpy as np


def _hsv_to_rgb_u8(hue: np.ndarray) -> np.ndarray:
    """Vectorized HSV->RGB for hue in [0, 1) at fixed S=0.88, V=1.0.

    Implements the standard six-sector HSV conversion with boolean masks so the
    whole palette is built in one pass without a Python per-color loop.
    """
    h = np.asarray(hue, dtype=np.float32).reshape(-1)
    h6 = (h % np.float32(1.0)) * np.float32(6.0)
    sector = np.floor(h6).astype(np.int32)
    frac = h6 - sector.astype(np.float32)
    sat = np.float32(0.88)
    val = np.float32(1.0)
    p = val * (np.float32(1.0) - sat)
    q = val * (np.float32(1.0) - sat * frac)
    t = val * (np.float32(1.0) - sat * (np.float32(1.0) - frac))
    rgb = np.empty((h.shape[0], 3), dtype=np.float32)
    branch = sector % 6
    rgb[branch == 0] = np.stack([np.full_like(frac[branch == 0], val), t[branch == 0], np.full_like(frac[branch == 0], p)], axis=1)
    rgb[branch == 1] = np.stack([q[branch == 1], np.full_like(frac[branch == 1], val), np.full_like(frac[branch == 1], p)], axis=1)
    rgb[branch == 2] = np.stack([np.full_like(frac[branch == 2], p), np.full_like(frac[branch == 2], val), t[branch == 2]], axis=1)
    rgb[branch == 3] = np.stack([np.full_like(frac[branch == 3], p), q[branch == 3], np.full_like(frac[branch == 3], val)], axis=1)
    rgb[branch == 4] = np.stack([t[branch == 4], np.full_like(frac[branch == 4], p), np.full_like(frac[branch == 4], val)], axis=1)
    rgb[branch == 5] = np.stack([np.full_like(frac[branch == 5], val), np.full_like(frac[branch == 5], p), q[branch == 5]], axis=1)
    return np.clip(np.rint(rgb * np.float32(255.0)), 0, 255).astype(np.uint8)


def query_rainbow_colors_rgb_u8(query_count: int) -> np.ndarray:
    """Generate stable uint8 RGB rainbow colors for query ids."""
    count = int(query_count)
    if count <= 0:
        return np.empty((0, 3), dtype=np.uint8)
    indices = np.arange(count, dtype=np.float32)
    hue = indices / np.float32(max(1, count))
    return np.ascontiguousarray(_hsv_to_rgb_u8(hue), dtype=np.uint8)


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


def query_rainbow_colors_for_indices(query_indices: np.ndarray, *, query_count: int | None = None) -> np.ndarray:
    """Return the query rainbow colors for indices."""
    indices = np.asarray(query_indices, dtype=np.int64).reshape(-1)
    if len(indices) == 0:
        return np.empty((0, 3), dtype=np.uint8)
    resolved_count = int(query_count) if query_count is not None else int(indices.max(initial=-1) + 1)
    # Grow the palette past query_count if an index exceeds it so lookups never go out of range.
    palette = query_rainbow_colors_rgb_u8(max(resolved_count, int(indices.max(initial=-1) + 1)))
    # Out-of-range (negative) indices render black instead of raising.
    valid = (indices >= 0) & (indices < len(palette))
    colors = np.zeros((len(indices), 3), dtype=np.uint8)
    colors[valid] = palette[indices[valid]]
    return np.ascontiguousarray(colors, dtype=np.uint8)
