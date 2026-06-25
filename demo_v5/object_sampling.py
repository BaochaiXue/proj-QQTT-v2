from __future__ import annotations

import numpy as np


def select_object_query_ids(query_ids, first_points, surface_points, interior_points, cell_size_m):
    query_ids = np.asarray(query_ids, dtype=np.int64).reshape(-1)
    points = np.asarray(first_points, dtype=np.float32)
    parts = [points[query_ids], np.asarray(surface_points).reshape(-1, 3), np.asarray(interior_points).reshape(-1, 3)]
    minimum = np.min(np.concatenate(parts, axis=0), axis=0)
    seen = set()
    selected = []
    for query_id_value in query_ids:
        query_id = int(query_id_value)
        key = tuple(np.floor((points[query_id] - minimum) / cell_size_m).astype(np.int64).tolist())
        if key not in seen:
            seen.add(key)
            selected.append(query_id)
    return np.asarray(selected, dtype=np.int64)
