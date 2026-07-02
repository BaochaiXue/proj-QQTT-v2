"""Semantic-mask gates for Demo v5.1 tracking (see demo_v5_1/design_spec.md).

This module owns the mask half of the per-frame observation gate:

- frame-0 query labeling — a query is an object query iff it is visible at
  frame 0 and its pixel lies inside the frame-0 object processed mask, and a
  controller query iff visible and inside the controller processed mask. The
  labels are frozen for the whole session.
- per-frame class-mask membership for later frames.

Offline parity: data_process_origin/data_process_track.py:L58-L94. Origin
keeps the object/controller mask overlap in both classes, so this module
never subtracts the controller mask from the object mask.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from demo_v5_1 import ffs

QUERY_SEMANTIC_NONE = np.int8(0)
QUERY_SEMANTIC_OBJECT = np.int8(1)
QUERY_SEMANTIC_CONTROLLER = np.int8(2)


def object_mask_from_frame(frame: Mapping[str, Any]) -> np.ndarray:
    """Return the per-class object processed mask without controller subtraction."""
    if "object" not in frame or frame["object"] is None:
        raise ValueError("mask frame is missing the 'object' mask")
    return np.asarray(frame["object"], dtype=bool)


def controller_mask_from_frame(frame: Mapping[str, Any]) -> np.ndarray:
    """Return the controller processed mask, unioning hand masks when needed."""
    if "controller" in frame and frame["controller"] is not None:
        return np.asarray(frame["controller"], dtype=bool)
    obj = object_mask_from_frame(frame)
    hand_a = np.asarray(frame.get("hand_a", np.zeros_like(obj, dtype=bool)), dtype=bool)
    hand_b = np.asarray(frame.get("hand_b", np.zeros_like(obj, dtype=bool)), dtype=bool)
    if hand_a.shape != obj.shape or hand_b.shape != obj.shape:
        raise ValueError("hand masks must match the object mask shape")
    return np.logical_or(hand_a, hand_b)


def frame0_semantic_labels(
    tracks0_yx: np.ndarray,
    visibility0: np.ndarray,
    *,
    object_mask: np.ndarray,
    controller_mask: np.ndarray,
) -> np.ndarray:
    """Label queries once from the warmup frame (design_spec.md line 1).

    A query visible at frame 0 whose pixel is inside both masks keeps the
    existing session convention: the controller label wins.
    """
    obj_mask = np.asarray(object_mask, dtype=bool)
    ctrl_mask = np.asarray(controller_mask, dtype=bool)
    if obj_mask.shape != ctrl_mask.shape:
        raise ValueError("object/controller masks must have the same shape")
    vis0 = np.asarray(visibility0, dtype=bool).reshape(-1)
    yy, xx, in_bounds = ffs.round_tracks_to_pixels(tracks0_yx, obj_mask.shape)
    if vis0.shape[0] != yy.shape[0]:
        raise ValueError("visibility0 must match tracks0_yx query count")
    labels = np.zeros((vis0.shape[0],), dtype=np.int8)
    visible = vis0 & in_bounds
    if np.any(visible):
        # Gather only the in-bounds subset: out-of-image or non-finite frame-0
        # tracks are unlabeled observations, not indexing errors.
        vis_idx = np.flatnonzero(visible)
        obj_hit = obj_mask[yy[vis_idx], xx[vis_idx]].astype(bool)
        ctrl_hit = ctrl_mask[yy[vis_idx], xx[vis_idx]].astype(bool)
        labels[vis_idx[obj_hit]] = QUERY_SEMANTIC_OBJECT
        labels[vis_idx[ctrl_hit]] = QUERY_SEMANTIC_CONTROLLER
    return labels


def class_mask_valid(
    tracks_yx: np.ndarray,
    semantic_labels: np.ndarray,
    *,
    object_mask: np.ndarray,
    controller_mask: np.ndarray,
    tracker_visible: np.ndarray,
) -> np.ndarray:
    """Per-frame class-mask membership gate (design_spec.md 大类一).

    Object queries must land inside the frame's object processed mask,
    controller queries inside the controller processed mask. Queries labeled
    neither are never mask-valid. Out-of-image pixels are invalid.
    """
    obj_mask = np.asarray(object_mask, dtype=bool)
    ctrl_mask = np.asarray(controller_mask, dtype=bool)
    labels = np.asarray(semantic_labels, dtype=np.int8).reshape(-1)
    visible = np.asarray(tracker_visible, dtype=bool).reshape(-1)
    yy, xx, in_bounds = ffs.round_tracks_to_pixels(tracks_yx, obj_mask.shape)
    valid = np.zeros((labels.shape[0],), dtype=bool)
    candidates = visible & in_bounds
    object_queries = candidates & (labels == QUERY_SEMANTIC_OBJECT)
    controller_queries = candidates & (labels == QUERY_SEMANTIC_CONTROLLER)
    if np.any(object_queries):
        valid[object_queries] = obj_mask[yy[object_queries], xx[object_queries]]
    if np.any(controller_queries):
        valid[controller_queries] = ctrl_mask[yy[controller_queries], xx[controller_queries]]
    return valid
