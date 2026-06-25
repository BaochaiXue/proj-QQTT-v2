import numpy as np


def recover_selected(rest_points, direct_points, direct_colors, direct_valid, selected_ids, support_ids, last_points=None, last_colors=None, recovery_k=8, recovery_radius_m=0.08):
    output = np.empty((selected_ids.size, 3), dtype=np.float32)
    colors = np.zeros((selected_ids.size, 3), dtype=np.float32)
    recovered = np.zeros((selected_ids.size,), dtype=bool)
    confidence = np.ones((selected_ids.size,), dtype=np.float32)
    support = support_ids[direct_valid[support_ids] & np.isfinite(rest_points[support_ids]).all(axis=1) & np.isfinite(direct_points[support_ids]).all(axis=1)]
    for local_idx, value in enumerate(selected_ids):
        query_id = int(value)
        if direct_valid[query_id]:
            output[local_idx] = direct_points[query_id]
            colors[local_idx] = direct_colors[query_id]
            continue
        recovered[local_idx] = True
        anchor = rest_points[query_id]
        candidates = support[support != query_id]
        if candidates.size:
            distances = np.linalg.norm(rest_points[candidates] - anchor[None], axis=1)
            within = np.flatnonzero(distances <= recovery_radius_m)
            if within.size == 0:
                within = np.arange(candidates.size)
            order = within[np.argsort(distances[within], kind="stable")][:recovery_k]
            neighbors = candidates[order]
            distances = distances[order]
            weights = 1.0 / np.maximum(distances, 1e-4) ** 2
            weights /= weights.sum()
            output[local_idx] = anchor + np.sum((direct_points[neighbors] - rest_points[neighbors]) * weights[:, None], axis=0)
            colors[local_idx] = np.sum(direct_colors[neighbors] * weights[:, None], axis=0)
            confidence[local_idx] = min(1.0, neighbors.size / recovery_k) * np.exp(-distances.mean() / recovery_radius_m)
        elif last_points is not None:
            output[local_idx] = last_points[local_idx]
            if last_colors is not None:
                colors[local_idx] = last_colors[local_idx]
            confidence[local_idx] = 0.1
        else:
            output[local_idx] = anchor
            confidence[local_idx] = 0.0
    if not np.isfinite(output).all():
        raise RuntimeError("anchor recovery produced nonfinite points")
    return output, colors, recovered, confidence
