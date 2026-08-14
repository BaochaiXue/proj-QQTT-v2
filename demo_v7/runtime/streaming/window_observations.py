"""Window-level raw observation lifting for the tracking state machine.

``build_window_observations`` turns one chunk window's tracker, mask, and
PCD products into the per-frame observation arrays consumed by
``demo_v7.runtime.tracking.TrackingRuntime.process_window``.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from demo_v7.runtime.perception import ffs, segment


def build_window_observations(
    *,
    tracks_yx: np.ndarray,
    visibility: np.ndarray,
    mask_frames: Sequence[Mapping[str, Any]],
    pcd_points: np.ndarray,
    pcd_colors: np.ndarray,
    query_ids: np.ndarray | None = None,
    query_semantic_labels: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Build one window's raw per-frame observations for the state machine.

    Shapes and units: ``tracks_yx`` is ``(T, N, 2)`` float pixel coordinates
    in (y, x) = (row, col) order over T window frames and N queries;
    ``visibility`` is ``(T, N)`` bool; ``pcd_points``/``pcd_colors`` are
    ``(T, C, H, W, 3)`` world-frame grids (points in meters, colors uint8
    RGB). Only camera 0 of the C axis is sampled. Every returned point array
    is world-frame meters; returned colors are float RGB in [0, 1].

    Offline parity: data_process_origin/data_process_track.py:L58-L135 —
    frame-0 labeling, per-frame class-mask gating (no controller-mask
    subtraction from the object mask), and PCD lifting at track pixels.
    """
    tracks = np.asarray(tracks_yx, dtype=np.float32)
    vis = np.asarray(visibility, dtype=bool)
    points_grid = np.asarray(pcd_points, dtype=np.float32)
    colors_grid = np.asarray(pcd_colors)
    if tracks.ndim != 3 or tracks.shape[-1] != 2:
        raise ValueError("tracks_yx must have shape T,N,2")
    if vis.shape != tracks.shape[:2]:
        raise ValueError("visibility must have shape T,N")
    if points_grid.ndim != 5 or points_grid.shape[1] < 1 or points_grid.shape[-1] != 3:
        raise ValueError("pcd_points must have shape T,C,H,W,3")
    if colors_grid.shape != points_grid.shape:
        raise ValueError("pcd_colors must match pcd_points shape")
    if len(mask_frames) != tracks.shape[0] or points_grid.shape[0] != tracks.shape[0]:
        raise ValueError("mask_frames, pcd_points, and tracks must share T")

    frame_count = int(tracks.shape[0])
    query_count = int(tracks.shape[1])
    if query_ids is None:
        query_id_arr = np.arange(query_count, dtype=np.int64)
    else:
        query_id_arr = np.asarray(query_ids, dtype=np.int64).reshape(-1)
        if query_id_arr.shape[0] != query_count:
            raise ValueError("query_ids must match track query count")

    first_object_mask = segment.object_mask_from_frame(mask_frames[0])
    first_controller_mask = segment.controller_mask_from_frame(mask_frames[0])
    if query_semantic_labels is None:
        semantic_labels = segment.frame0_semantic_labels(
            tracks[0],
            vis[0],
            object_mask=first_object_mask,
            controller_mask=first_controller_mask,
        )
    else:
        semantic_labels = np.asarray(query_semantic_labels, dtype=np.int8).reshape(-1)
        if semantic_labels.shape[0] != query_count:
            raise ValueError("query_semantic_labels must match track query count")
        valid_labels = np.isin(
            semantic_labels,
            np.array(
                [
                    segment.QUERY_SEMANTIC_NONE,
                    segment.QUERY_SEMANTIC_OBJECT,
                    segment.QUERY_SEMANTIC_CONTROLLER,
                ],
                dtype=np.int8,
            ),
        )
        if not bool(np.all(valid_labels)):
            raise ValueError("query_semantic_labels must contain only 0, 1, or 2")
    object_label = semantic_labels == segment.QUERY_SEMANTIC_OBJECT
    controller_label = semantic_labels == segment.QUERY_SEMANTIC_CONTROLLER

    raw_visible = np.array(vis, dtype=bool, copy=True)
    processed_mask_valid = np.zeros_like(raw_visible, dtype=bool)
    depth_valid = np.zeros_like(raw_visible, dtype=bool)
    raw_points = np.zeros((frame_count, query_count, 3), dtype=np.float32)
    track_points = np.zeros((frame_count, query_count, 3), dtype=np.float32)
    track_colors = np.zeros((frame_count, query_count, 3), dtype=np.float32)

    for frame_idx in range(frame_count):
        object_mask = segment.object_mask_from_frame(mask_frames[frame_idx])
        controller_mask = segment.controller_mask_from_frame(mask_frames[frame_idx])
        processed_mask_valid[frame_idx] = segment.class_mask_valid(
            tracks[frame_idx],
            semantic_labels,
            object_mask=object_mask,
            controller_mask=controller_mask,
            tracker_visible=raw_visible[frame_idx],
        )
        yy, xx, in_bounds = ffs.round_tracks_to_pixels(tracks[frame_idx], object_mask.shape)
        sampled_points, sampled_colors, frame_depth_valid = ffs.sample_world_pcd_at_pixels(
            points_grid[frame_idx, 0],
            colors_grid[frame_idx, 0],
            yy=yy,
            xx=xx,
            sample=raw_visible[frame_idx] & in_bounds,
        )
        depth_valid[frame_idx] = frame_depth_valid
        raw_points[frame_idx] = sampled_points
        measurement = processed_mask_valid[frame_idx] & frame_depth_valid
        track_points[frame_idx, measurement] = sampled_points[measurement]
        track_colors[frame_idx, measurement] = sampled_colors[measurement]

    measurement_valid = raw_visible & processed_mask_valid & depth_valid
    object_indices = np.flatnonzero(object_label)
    controller_indices = np.flatnonzero(controller_label)
    hand_labels = segment.frame0_hand_labels(
        tracks[0], vis[0], mask_frame=mask_frames[0]
    )
    return {
        "query_ids": np.ascontiguousarray(query_id_arr, dtype=np.int64),
        "query_semantic_labels": np.ascontiguousarray(semantic_labels, dtype=np.int8),
        "query_is_object": object_label,
        "query_is_controller": controller_label,
        "object_query_indices": object_indices.astype(np.int64),
        "controller_query_indices": controller_indices.astype(np.int64),
        "object_points": track_points[:, object_indices, :],
        "object_colors": track_colors[:, object_indices, :],
        "object_visibilities": measurement_valid[:, object_indices],
        "controller_points": track_points[:, controller_indices, :],
        "controller_colors": track_colors[:, controller_indices, :],
        "controller_visibilities": measurement_valid[:, controller_indices],
        "controller_raw_points": raw_points[:, controller_indices, :],
        "controller_raw_visible": raw_visible[:, controller_indices],
        "controller_processed_mask_valid": processed_mask_valid[:, controller_indices],
        "controller_depth_valid": depth_valid[:, controller_indices],
        "controller_measurement_valid": measurement_valid[:, controller_indices],
        "controller_hand_labels": np.ascontiguousarray(
            hand_labels[controller_indices], dtype=np.int8
        ),
    }
