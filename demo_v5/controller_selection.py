from __future__ import annotations

import numpy as np
from demo_v5.fps_sampling import farthest_point_indices
from demo_v5.motion_filter import motion_valid_for_points


def select_controller_query_ids(points_t, valid_t, candidate_ids, count=30):
    if candidate_ids.size < count:
        raise RuntimeError(f"need {count} controller queries; got {candidate_ids.size}")
    points = points_t[:, candidate_ids]
    visibility = valid_t[:, candidate_ids]
    motion, strict_mask = motion_valid_for_points(points, visibility, once_false_mask=True)
    visibility_ratio = visibility.mean(axis=0)
    motion_ratio = motion[:-1].mean(axis=0)
    strict_local = np.flatnonzero(strict_mask)
    if strict_local.size >= count:
        pool = strict_local
    else:
        ranking = np.lexsort((np.arange(candidate_ids.size), -motion_ratio, -visibility_ratio, -strict_mask.astype(np.int8)))
        pool = ranking[: max(count, strict_local.size)]
    selected = farthest_point_indices(points_t[0, candidate_ids[pool]], count)
    return np.ascontiguousarray(candidate_ids[pool[selected]], dtype=np.int64)
