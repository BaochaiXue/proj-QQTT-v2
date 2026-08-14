"""Tracker mask classification, visibility, and 3D-lift helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from demo_v7.runtime.mdp.constants import (
    HAND_A_ID,
    HAND_B_ID,
    OBJECT_ID,
    QUERY_CONTROLLER_INSTANCE_HAND_A,
    QUERY_CONTROLLER_INSTANCE_HAND_B,
)
from demo_v7.runtime.mdp.packets import MaskPacket
from demo_v7.runtime.utils.projection import track_lift_valid_mask


def _tracker_union_mask(mask_packet: MaskPacket) -> np.ndarray:
    """Return the tracker union mask."""
    controller = np.asarray(mask_packet.controller_mask, dtype=bool)
    obj = np.asarray(mask_packet.object_mask, dtype=bool)
    if controller.shape != obj.shape:
        raise ValueError("controller/object masks must share a shape")
    return np.logical_or(controller, obj)


def _mask_packet_hand_a_mask(mask_packet: MaskPacket) -> np.ndarray:
    """Return the mask packet hand a mask."""
    if mask_packet.hand_a_mask is None:
        return np.asarray(mask_packet.controller_mask, dtype=bool)
    return np.asarray(mask_packet.hand_a_mask, dtype=bool)


def _mask_packet_hand_b_mask(mask_packet: MaskPacket) -> np.ndarray:
    """Return the mask packet hand b mask."""
    if mask_packet.hand_b_mask is None:
        return np.zeros_like(
            np.asarray(mask_packet.controller_mask, dtype=bool), dtype=bool
        )
    return np.asarray(mask_packet.hand_b_mask, dtype=bool)


def _classify_query_targets_yx(
    query_points_yx: np.ndarray,
    *,
    object_mask: np.ndarray,
    hand_a_mask: np.ndarray,
    hand_b_mask: np.ndarray,
    controller_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Classify query targets YX."""
    points = np.asarray(query_points_yx, dtype=np.float32).reshape(-1, 2)
    if len(points) == 0:
        empty_bool = np.empty((0,), dtype=bool)
        empty_int = np.empty((0,), dtype=np.int64)
        return empty_bool, empty_bool, empty_int, empty_int
    object_bool = np.asarray(object_mask, dtype=bool)
    hand_a_bool = np.asarray(hand_a_mask, dtype=bool)
    hand_b_bool = np.asarray(hand_b_mask, dtype=bool)
    controller_bool = np.asarray(controller_mask, dtype=bool)
    height, width = object_bool.shape[:2]
    yy = np.clip(np.rint(points[:, 0]).astype(np.int64), 0, height - 1)
    xx = np.clip(np.rint(points[:, 1]).astype(np.int64), 0, width - 1)
    in_hand_a = hand_a_bool[yy, xx]
    in_hand_b = hand_b_bool[yy, xx] & ~in_hand_a
    # Origin preserves object/controller overlap. Tracker target IDs are
    # single-valued, so the existing design gives a hand identity priority only
    # in this query-label table; it does not alter either processed mask.
    in_object = object_bool[yy, xx] & ~(in_hand_a | in_hand_b)
    in_controller = controller_bool[yy, xx] | in_hand_a | in_hand_b
    target_id = np.zeros((len(points),), dtype=np.int64)
    target_id[in_object] = OBJECT_ID
    target_id[in_hand_a] = HAND_A_ID
    target_id[in_hand_b] = HAND_B_ID
    controller_instance_id = np.zeros((len(points),), dtype=np.int64)
    controller_instance_id[in_hand_a] = QUERY_CONTROLLER_INSTANCE_HAND_A
    controller_instance_id[in_hand_b] = QUERY_CONTROLLER_INSTANCE_HAND_B
    return (
        in_object.astype(bool),
        in_controller.astype(bool),
        target_id,
        controller_instance_id,
    )


def _tracker_per_target_visibility(
    tracks_yx: np.ndarray,
    visibility: np.ndarray,
    *,
    mask_packet: MaskPacket,
    query_target_id: np.ndarray,
) -> np.ndarray:
    """Return the tracker per target visibility."""
    tracks = np.asarray(tracks_yx, dtype=np.float32).reshape(-1, 2)
    vis = np.asarray(visibility, dtype=np.float32).reshape(-1)
    target_id = np.asarray(query_target_id, dtype=np.int64).reshape(-1)
    count = min(len(tracks), len(vis), len(target_id))
    output = np.zeros((len(vis),), dtype=np.float32)
    if count == 0:
        return output
    object_mask = np.asarray(mask_packet.object_mask, dtype=bool)
    hand_a_mask = _mask_packet_hand_a_mask(mask_packet)
    hand_b_mask = _mask_packet_hand_b_mask(mask_packet)
    height, width = object_mask.shape[:2]
    yy = np.rint(tracks[:count, 0]).astype(np.int64)
    xx = np.rint(tracks[:count, 1]).astype(np.int64)
    finite_tracks = np.isfinite(tracks[:count]).all(axis=1)
    in_bounds = (yy >= 0) & (yy < height) & (xx >= 0) & (xx < width)
    valid = (vis[:count] > 0.0) & finite_tracks & in_bounds
    if not np.any(valid):
        return output
    valid_indices = np.flatnonzero(valid)
    inside_target = np.zeros((count,), dtype=bool)
    valid_targets = target_id[valid_indices]
    hand_a_indices = valid_indices[valid_targets == HAND_A_ID]
    if len(hand_a_indices):
        inside_target[hand_a_indices] = hand_a_mask[
            yy[hand_a_indices], xx[hand_a_indices]
        ]
    hand_b_indices = valid_indices[valid_targets == HAND_B_ID]
    if len(hand_b_indices):
        inside_target[hand_b_indices] = hand_b_mask[
            yy[hand_b_indices], xx[hand_b_indices]
        ]
    object_indices = valid_indices[valid_targets == OBJECT_ID]
    if len(object_indices):
        inside_target[object_indices] = object_mask[
            yy[object_indices], xx[object_indices]
        ]
    output[:count] = np.where(inside_target, vis[:count], 0.0).astype(np.float32)
    return output


def _tracker_lift_valid_mask(
    *,
    tracks_yx: np.ndarray,
    visibility: np.ndarray,
    depth: np.ndarray,
    depth_scale_m_per_unit: float,
    mask: np.ndarray | None,
    depth_min_m: float,
    depth_max_m: float,
) -> np.ndarray:
    """Return the tracker lift valid mask."""
    return track_lift_valid_mask(
        tracks_yx=tracks_yx,
        visibility=visibility,
        depth=depth,
        depth_scale_m_per_unit=depth_scale_m_per_unit,
        mask=mask,
        depth_min_m=depth_min_m,
        depth_max_m=depth_max_m,
    )


def _select_visible_spread_indices(
    tracks_yx: np.ndarray, visibility: np.ndarray, *, max_points: int
) -> np.ndarray:
    """Select visible spread indices."""
    tracks = np.asarray(tracks_yx, dtype=np.float32).reshape(-1, 2)
    visible = np.flatnonzero(np.asarray(visibility, dtype=np.float32).reshape(-1) > 0.0)
    if len(visible) > 0:
        visible = visible[np.isfinite(tracks[visible]).all(axis=1)]
    # overlay cap is fixed to 0 (draw all visible markers); the former
    # farthest-point subsampling for cap > 0 was unreachable and removed.
    return visible.astype(np.int64)


def _latest_tracker_arrays(result: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return the latest tracker arrays."""
    tracks = np.asarray(result.tracks_yx, dtype=np.float32)
    visibility = np.asarray(result.visibility, dtype=np.float32)
    if tracks.ndim == 4:
        tracks_latest = tracks[0, -1]
        visibility_latest = visibility[0, -1]
    elif tracks.ndim == 3:
        tracks_latest = tracks[-1]
        visibility_latest = visibility[-1]
    elif tracks.ndim == 2:
        tracks_latest = tracks
        visibility_latest = visibility
    else:
        raise ValueError(f"tracker tracks_yx must be 2D, 3D, or 4D; got {tracks.shape}")
    return (
        np.ascontiguousarray(
            np.asarray(tracks_latest, dtype=np.float32).reshape(-1, 2)
        ),
        np.ascontiguousarray(
            np.asarray(visibility_latest, dtype=np.float32).reshape(-1)
        ),
    )


__all__ = [
    "_tracker_union_mask",
    "_mask_packet_hand_a_mask",
    "_mask_packet_hand_b_mask",
    "_classify_query_targets_yx",
    "_tracker_per_target_visibility",
    "_tracker_lift_valid_mask",
    "_select_visible_spread_indices",
    "_latest_tracker_arrays",
]
