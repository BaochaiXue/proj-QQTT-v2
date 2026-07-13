"""Depth/stereo geometry helpers (vendored for demo_v6_1).

Vendored verbatim from ``data_process/depth_backends/geometry.py`` so demo_v6_1
does not import the repo-level ``data_process`` package. Pure numpy; demo_v6_1
uses ``transform_points`` directly and the local FFS runner uses
``disparity_to_metric_depth``.
"""

from __future__ import annotations

import numpy as np


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


def transform_points(points: np.ndarray, T_src_to_dst: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    transform = np.asarray(T_src_to_dst, dtype=np.float32).reshape(4, 4)
    if pts.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    homogeneous = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float32)], axis=1)
    transformed = homogeneous @ transform.T
    return transformed[:, :3]
