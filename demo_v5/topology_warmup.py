from __future__ import annotations

import numpy as np
from demo_v5.contracts import SEMANTIC_CONTROLLER, SEMANTIC_OBJECT, as_points
from demo_v5.tracking_samples import first_frame_semantics, frame_direct_samples


def prepare_warmup(frames, surface_points, interior_points, minimum_surface, minimum_interior):
    if len(frames) < 2:
        raise ValueError("topology warmup requires at least two frames")
    surface = as_points(surface_points)
    interior = as_points(interior_points)
    if len(surface) < minimum_surface or len(interior) < minimum_interior:
        raise ValueError(f"shape prior is incomplete: {len(surface)}/{len(interior)}")
    queries0 = np.asarray(frames[0].query_points_yx, dtype=np.float32).reshape(-1, 2)
    for frame in frames[1:]:
        queries = np.asarray(frame.query_points_yx, dtype=np.float32).reshape(-1, 2)
        if queries.shape != queries0.shape or not np.allclose(queries, queries0, atol=1e-4, rtol=0.0):
            raise ValueError("prepared frames changed query identity")
    semantics = first_frame_semantics(frames[0])
    samples = [frame_direct_samples(frame, semantics) for frame in frames]
    points_t = np.stack([item[0] for item in samples])
    valid_t = np.stack([item[2] for item in samples])
    object_ids = np.flatnonzero((semantics == SEMANTIC_OBJECT) & valid_t[0]).astype(np.int64)
    controller_ids = np.flatnonzero((semantics == SEMANTIC_CONTROLLER) & valid_t[0]).astype(np.int64)
    if not len(object_ids):
        raise RuntimeError("no valid first-frame object queries")
    return surface, interior, queries0, semantics, points_t, valid_t, object_ids, controller_ids
