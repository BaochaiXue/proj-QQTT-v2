"""FFS depth / PCD gates for Demo v5.1 tracking (see demo_v5_1/design_spec.md).

This module owns the "PCD/depth 是否有效" half of the per-frame observation
gate: sampling the dense world PCD grid at track pixels and deciding whether
the lifted 3D point is a usable measurement. Semantic-mask gating lives in
``demo_v5_1/segment.py``; the state machine lives in ``demo_v5_1/tracking.py``.
"""

from __future__ import annotations

import numpy as np

DEPTH_NONZERO_NORM_M = 1e-9


def round_tracks_to_pixels(
    tracks_yx: np.ndarray,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Round float (y, x) tracks to integer pixels, matching origin indexing.

    Offline parity: data_process_origin/data_process_track.py:L53 rounds
    tracks and indexes masks/PCD with them; out-of-image coordinates are
    invalid observations rather than errors.
    """
    tracks = np.asarray(tracks_yx, dtype=np.float32).reshape(-1, 2)
    finite = np.isfinite(tracks).all(axis=1)
    # Substitute -1 for non-finite coordinates before rounding: NaN/inf do not
    # cast to int, and -1 is guaranteed to fail the bounds check below.
    safe = np.where(finite[:, None], tracks, np.float32(-1.0))
    # np.rint rounds half-to-even, the same rule as origin's np.round.
    yy = np.rint(safe[:, 0]).astype(np.int64)
    xx = np.rint(safe[:, 1]).astype(np.int64)
    in_bounds = finite & (yy >= 0) & (yy < int(shape[0])) & (xx >= 0) & (xx < int(shape[1]))
    return yy, xx, in_bounds


def sample_world_pcd_at_pixels(
    points_grid: np.ndarray,
    colors_grid: np.ndarray,
    *,
    yy: np.ndarray,
    xx: np.ndarray,
    sample: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample one frame's world PCD grid at track pixels.

    ``points_grid``/``colors_grid`` are one frame's ``(H, W, 3)`` world PCD
    and RGB grids. ``sample`` marks queries whose pixel may be sampled
    (tracker-visible and in-bounds). Returns ``(points, colors, depth_valid)``
    where invalid entries stay zero and ``depth_valid`` requires a finite,
    nonzero-norm world point — the origin PCD-validity convention
    (data_process_origin/data_process_track.py:L96-L111 samples only pixels
    whose pcd masks marked usable depth).
    """
    points = np.asarray(points_grid, dtype=np.float32)
    colors = np.asarray(colors_grid)
    sample_mask = np.asarray(sample, dtype=bool).reshape(-1)
    query_count = sample_mask.shape[0]
    out_points = np.zeros((query_count, 3), dtype=np.float32)
    out_colors = np.zeros((query_count, 3), dtype=np.float32)
    depth_valid = np.zeros((query_count,), dtype=bool)
    if not np.any(sample_mask):
        return out_points, out_colors, depth_valid
    idx = np.flatnonzero(sample_mask)
    sampled = points[yy[idx], xx[idx]]
    finite = np.isfinite(sampled).all(axis=1)
    nonzero = np.linalg.norm(sampled, axis=1) > DEPTH_NONZERO_NORM_M
    usable = finite & nonzero
    keep = idx[usable]
    depth_valid[keep] = True
    if keep.size:
        out_points[keep] = sampled[usable]
        # Colors arrive as uint8 RGB; published colors are float in [0, 1].
        out_colors[keep] = colors[yy[keep], xx[keep]].astype(np.float32) / 255.0
    return out_points, out_colors, depth_valid
