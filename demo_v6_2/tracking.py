"""Demo v6.1 realtime tracking state machine (see demo_v6_1/design_spec.md).

Semantics implemented here, in spec order:

- Frame-0 (warmup frame) query labels are frozen for the whole session.
- Every per-frame failure is one state, ``temporary_invalid``: tracker
  visible but mask / PCD-depth / motion-consistency gate failed, or tracker
  invisible / lost. It never deletes a query and never changes anchor
  identity.
- Chunk 0 selects controller anchors with origin strictness (valid at every
  window frame, origin motion consistency with once-fail removal), then
  farthest-point-samples the final handles. The anchor set, ``query_ids``,
  ``query_semantic_labels``, and ``controller_sample_query_ids`` never
  change afterwards.
- A one-time table stores each controller point's nearest
  ``NEIGHBOR_TABLE_SIZE`` controller points by first-frame 3D positions.
- In later chunks a temporarily-invalid anchor frame is filled by local
  rigid registration from currently-valid same-hand neighbors (first frame
  -> current frame), applied to the anchor's first-frame position. Donor
  selection is a ladder (design_spec.md 特殊情况): nearest 15 valid table
  neighbors, else 10, else 5; with fewer than 5 the donors become the
  nearest 5-15 currently-valid same-hand controller anchors; only when that
  also fails does ``TrackingRecoveryError`` raise.
- No confidence values anywhere.

Motion consistency is a verbatim port of
data_process_origin/data_process_track.py::filter_motion (0.01 m radius,
5 neighbors including self, 0.005 m similarity, 50% agreement).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from demo_v6_2 import ffs, segment

MOTION_NEIGHBOR_DIST_M = 0.01
MOTION_MIN_NEIGHBORS = 5
MOTION_SIMILARITY_M = 0.005
CONTROLLER_FINAL_COUNT = 30
NEIGHBOR_TABLE_SIZE = 100
RECOVERY_NEIGHBOR_COUNT = 15
DEFAULT_VOLUME_SAMPLE_SIZE_M = 0.005

TRACK_STATUS_NORMAL = "normal"
TRACK_STATUS_DEGRADED = "degraded"


class ControllerSelectionError(RuntimeError):
    """Chunk-0 controller selection found fewer survivors than required."""


class TrackingRecoveryError(RuntimeError):
    """A temporarily-invalid anchor had too few valid recovery neighbors."""


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


def motion_consistency(
    points: np.ndarray,
    visibilities: np.ndarray,
    *,
    neighbor_dist: float = MOTION_NEIGHBOR_DIST_M,
    min_neighbors: int = MOTION_MIN_NEIGHBORS,
    motion_similarity_m: float = MOTION_SIMILARITY_M,
    once_false_mask: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Origin neighbor-motion consistency filter.

    Verbatim port of data_process_origin/data_process_track.py::filter_motion:
    forward motion per frame, radius search (self included in the count),
    50% agreement within ``motion_similarity_m``. With ``once_false_mask``
    a single failure anywhere permanently removes the candidate (origin's
    controller semantics, used for chunk-0 selection only).

    ``points`` is ``(T, N, 3)`` world-frame meters, ``visibilities`` is
    ``(T, N)`` bool. Returns ``(motions_valid, global_mask)`` where row t of
    ``motions_valid`` judges the forward motion t -> t+1 (the last row is
    always False) and ``global_mask`` is the ``(N,)`` once-fail survivor set.
    """
    pts = np.asarray(points, dtype=np.float32)
    vis = np.asarray(visibilities, dtype=bool)
    if pts.ndim != 3 or pts.shape[-1] != 3:
        raise ValueError("points must have shape T,N,3")
    if vis.shape != pts.shape[:2]:
        raise ValueError("visibilities must have shape T,N")
    motions_valid = np.zeros_like(vis, dtype=bool)
    if pts.shape[0] > 1:
        motions_valid[:-1] = vis[:-1] & vis[1:]
    if once_false_mask and vis.size:
        global_mask = np.prod(vis, axis=0).astype(bool)
    else:
        global_mask = np.ones((pts.shape[1],), dtype=bool)
    if pts.shape[1] == 0:
        return motions_valid, global_mask
    motions = np.zeros_like(pts, dtype=np.float32)
    motions[:-1] = pts[1:] - pts[:-1]
    from scipy.spatial import cKDTree  # noqa: PLC0415

    for frame_idx in range(max(0, pts.shape[0] - 1)):
        if once_false_mask:
            motions_valid[frame_idx] &= global_mask
        if not np.any(motions_valid[frame_idx]):
            continue
        tree = cKDTree(pts[frame_idx])
        all_neighbors = tree.query_ball_point(
            pts[frame_idx],
            r=float(neighbor_dist),
            workers=-1,
            return_sorted=False,
        )
        for query_idx in range(pts.shape[1]):
            if once_false_mask and not global_mask[query_idx]:
                motions_valid[frame_idx, query_idx] = False
                continue
            if not motions_valid[frame_idx, query_idx]:
                continue
            neighbors = np.asarray(all_neighbors[query_idx], dtype=np.int64)
            neighbors = neighbors[motions_valid[frame_idx, neighbors]]
            if len(neighbors) < int(min_neighbors):
                motions_valid[frame_idx, query_idx] = False
                if once_false_mask:
                    global_mask[query_idx] = False
                continue
            motion_diff = np.linalg.norm(
                motions[frame_idx, query_idx] - motions[frame_idx, neighbors], axis=1
            )
            agreeing = int(np.count_nonzero(motion_diff < float(motion_similarity_m)))
            if agreeing < 0.5 * float(len(neighbors)):
                motions_valid[frame_idx, query_idx] = False
                if once_false_mask:
                    global_mask[query_idx] = False
        if once_false_mask:
            motions_valid[frame_idx] &= global_mask
    return motions_valid, global_mask.astype(bool, copy=False)


def _motion_failed_mask(visibilities: np.ndarray, motions_valid: np.ndarray) -> np.ndarray:
    """Frames where motion was testable and the consistency check failed.

    A frame whose forward motion is untestable (last window frame, or the
    query invisible at t+1) carries no motion evidence and must not become
    ``temporary_invalid`` for it.
    """
    vis = np.asarray(visibilities, dtype=bool)
    tested = np.zeros_like(vis, dtype=bool)
    if vis.shape[0] > 1:
        tested[:-1] = vis[:-1] & vis[1:]
    return tested & ~np.asarray(motions_valid, dtype=bool)


def _farthest_point_sample_indices(points_xyz: np.ndarray, count: int) -> np.ndarray:
    """Deterministic farthest point sampling (origin controller FPS parity)."""
    pts = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    target = int(count)
    if target < 0:
        raise ValueError("count must be >= 0")
    if target == 0:
        return np.empty((0,), dtype=np.int64)
    if len(pts) < target:
        raise ControllerSelectionError(
            f"controller FPS requires at least {target} candidates; got {len(pts)}"
        )
    selected = [0]
    min_dist2 = np.sum((pts - pts[0]) ** 2, axis=1)
    for _ in range(1, target):
        idx = int(np.argmax(min_dist2))
        selected.append(idx)
        dist2 = np.sum((pts - pts[idx]) ** 2, axis=1)
        min_dist2 = np.minimum(min_dist2, dist2)
    return np.asarray(selected, dtype=np.int64)


def _volume_sample_indices(
    first_frame_points: np.ndarray,
    *,
    surface_points: np.ndarray | None,
    interior_points: np.ndarray | None,
    volume_sample_size: float,
) -> np.ndarray:
    """Origin first-frame volume sampling (one point per occupied voxel)."""
    pts = np.asarray(first_frame_points, dtype=np.float32).reshape(-1, 3)
    if pts.shape[0] == 0:
        return np.empty((0,), dtype=np.int64)
    voxel = float(volume_sample_size)
    if voxel <= 0.0:
        raise ValueError("volume_sample_size must be positive")
    bound_inputs = [pts]
    for prior in (surface_points, interior_points):
        if prior is None:
            continue
        prior_arr = np.asarray(prior, dtype=np.float32).reshape(-1, 3)
        if prior_arr.size:
            bound_inputs.append(prior_arr)
    min_bound = np.min(np.concatenate(bound_inputs, axis=0), axis=0)
    seen: set[tuple[int, int, int]] = set()
    keep: list[int] = []
    for idx, point in enumerate(pts):
        grid_index = tuple(
            np.floor((point - min_bound) / np.float32(voxel)).astype(np.int64).tolist()
        )
        if grid_index in seen:
            continue
        seen.add(grid_index)
        keep.append(int(idx))
    return np.asarray(keep, dtype=np.int64)


def _rigid_transform(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares rigid transform (Kabsch) mapping ``src`` onto ``dst``."""
    src_pts = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    dst_pts = np.asarray(dst, dtype=np.float64).reshape(-1, 3)
    if src_pts.shape != dst_pts.shape or src_pts.shape[0] < 3:
        raise ValueError("rigid transform needs matching point sets of size >= 3")
    src_center = src_pts.mean(axis=0)
    dst_center = dst_pts.mean(axis=0)
    covariance = (src_pts - src_center).T @ (dst_pts - dst_center)
    u, _s, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt = vt.copy()
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    translation = dst_center - rotation @ src_center
    return rotation, translation


class TrackingRuntime:
    """Session-lived tracking state machine (design_spec.md).

    Chunk 0 freezes identity (object columns, controller anchors, neighbor
    table); later windows only decide per-frame value sources. There is no
    confidence, no dead-reckoning, and no anchor replacement.
    """

    def __init__(
        self,
        *,
        controller_count: int = CONTROLLER_FINAL_COUNT,
        neighbor_table_size: int = NEIGHBOR_TABLE_SIZE,
        recovery_neighbor_count: int = RECOVERY_NEIGHBOR_COUNT,
        volume_sample_size: float = DEFAULT_VOLUME_SAMPLE_SIZE_M,
    ) -> None:
        """Initialize TrackingRuntime."""
        if int(controller_count) <= 0:
            raise ValueError("controller_count must be positive")
        if int(neighbor_table_size) <= 0:
            raise ValueError("neighbor_table_size must be positive")
        if int(recovery_neighbor_count) < 3:
            raise ValueError(
                "recovery_neighbor_count must be >= 3 (rigid-fit minimum)"
            )
        if int(recovery_neighbor_count) > int(neighbor_table_size):
            raise ValueError("recovery_neighbor_count cannot exceed neighbor_table_size")
        if float(volume_sample_size) <= 0.0:
            raise ValueError("volume_sample_size must be positive")
        self.controller_count = int(controller_count)
        self.neighbor_table_size = int(neighbor_table_size)
        self.recovery_neighbor_count = int(recovery_neighbor_count)
        self.volume_sample_size = float(volume_sample_size)
        self._anchor_indices: np.ndarray | None = None
        self._anchor_first_points: np.ndarray | None = None
        self._controller_first_points: np.ndarray | None = None
        self._controller_hand_labels: np.ndarray | None = None
        self._neighbor_table: dict[int, np.ndarray] = {}
        self._chunk0_controller_mask: np.ndarray | None = None
        self._object_column_indices: np.ndarray | None = None
        self._query_ids: np.ndarray | None = None
        self._query_semantic_labels: np.ndarray | None = None

    @property
    def initialized(self) -> bool:
        """Return the initialized."""
        return self._anchor_indices is not None

    def _freeze_identity(
        self,
        window: Mapping[str, np.ndarray],
        controller_global_mask: np.ndarray,
        *,
        surface_points: np.ndarray | None,
        interior_points: np.ndarray | None,
    ) -> None:
        """Return the freeze identity."""
        ctrl_points = np.asarray(window["controller_points"], dtype=np.float32)
        ctrl_vis = np.asarray(window["controller_visibilities"], dtype=bool)
        candidates = np.flatnonzero(np.asarray(controller_global_mask, dtype=bool))
        if candidates.shape[0] < self.controller_count:
            raise ControllerSelectionError(
                "chunk-0 controller selection requires "
                f"{self.controller_count} whole-window-valid, motion-consistent "
                f"candidates; got {candidates.shape[0]} of {ctrl_points.shape[1]}"
            )
        fps_local = _farthest_point_sample_indices(
            ctrl_points[0, candidates], self.controller_count
        )
        self._anchor_indices = np.ascontiguousarray(candidates[fps_local], dtype=np.int64)
        self._anchor_first_points = np.ascontiguousarray(
            ctrl_points[0, self._anchor_indices], dtype=np.float64
        )
        self._chunk0_controller_mask = np.ascontiguousarray(
            controller_global_mask, dtype=bool
        )

        # design_spec.md: the one-time table stores, for every controller
        # point, its nearest NEIGHBOR_TABLE_SIZE controller points by
        # first-frame 3D positions — never crossing hands. A hand with fewer
        # points than the table size fills with that whole hand. Built once,
        # never updated.
        first_valid = np.flatnonzero(ctrl_vis[0])
        self._controller_first_points = np.ascontiguousarray(
            ctrl_points[0], dtype=np.float64
        )
        hand_labels = np.asarray(window["controller_hand_labels"], dtype=np.int8).reshape(-1)
        self._controller_hand_labels = np.ascontiguousarray(hand_labels, dtype=np.int8)
        self._neighbor_table = {}
        from scipy.spatial import cKDTree  # noqa: PLC0415

        for hand in np.unique(hand_labels[first_valid]):
            pool = first_valid[hand_labels[first_valid] == hand]
            if pool.shape[0] < 2:
                continue
            pool_points = ctrl_points[0, pool]
            tree = cKDTree(pool_points)
            k = min(self.neighbor_table_size + 1, pool.shape[0])
            _dists, local_neighbors = tree.query(pool_points, k=k, workers=-1)
            local_neighbors = np.atleast_2d(local_neighbors)
            for row, candidate_idx in enumerate(pool):
                neighbor_local = [
                    int(n) for n in np.asarray(local_neighbors[row]).reshape(-1) if int(n) != row
                ]
                self._neighbor_table[int(candidate_idx)] = np.asarray(
                    [int(pool[n]) for n in neighbor_local[: self.neighbor_table_size]],
                    dtype=np.int64,
                )

        obj_points = np.asarray(window["object_points"], dtype=np.float32)
        obj_vis = np.asarray(window["object_visibilities"], dtype=bool)
        if obj_points.shape[1]:
            # Zero-filled rows are the "no measurement" placeholder, so a
            # frame-0 point must be finite and nonzero to seed an object column.
            first_finite = np.isfinite(obj_points[0]).all(axis=1)
            first_nonzero = np.linalg.norm(obj_points[0], axis=1) > 1e-9
            valid_first = np.flatnonzero(obj_vis[0] & first_finite & first_nonzero)
        else:
            valid_first = np.empty((0,), dtype=np.int64)
        sample_local = _volume_sample_indices(
            obj_points[0, valid_first],
            surface_points=surface_points,
            interior_points=interior_points,
            volume_sample_size=self.volume_sample_size,
        )
        self._object_column_indices = np.ascontiguousarray(
            valid_first[sample_local], dtype=np.int64
        )
        self._query_ids = np.ascontiguousarray(window["query_ids"], dtype=np.int64)
        self._query_semantic_labels = np.ascontiguousarray(
            window["query_semantic_labels"], dtype=np.int8
        )

    def _check_frozen_identity(self, window: Mapping[str, np.ndarray]) -> None:
        """Check frozen identity."""
        if self._query_ids is None or self._query_semantic_labels is None:
            raise RuntimeError("tracking runtime identity is not frozen yet")
        if not np.array_equal(self._query_ids, np.asarray(window["query_ids"], dtype=np.int64)):
            raise ValueError("Demo v6.1 session query_ids changed across chunks")
        if not np.array_equal(
            self._query_semantic_labels,
            np.asarray(window["query_semantic_labels"], dtype=np.int8),
        ):
            raise ValueError("Demo v6.1 session query_semantic_labels changed across chunks")

    def _recovery_tiers(self) -> list[int]:
        """design_spec.md 特殊情况 donor-count ladder: 15 -> 10 -> 5 by default.

        Tiers scale with ``recovery_neighbor_count`` (n, 2n/3, n/3) and are
        clamped to the rigid-fit minimum of 3 points.
        """
        n = int(self.recovery_neighbor_count)
        tiers: list[int] = []
        for count in (n, round(2 * n / 3), round(n / 3)):
            tier = max(3, int(count))
            if tier not in tiers:
                tiers.append(tier)
        return tiers

    def _recovery_tier(self, valid_count: int) -> int | None:
        """Return the recovery tier."""
        for tier in self._recovery_tiers():
            if int(valid_count) >= tier:
                return tier
        return None

    def _fallback_anchor_donors(
        self, anchor_column: int, usable_frame: np.ndarray
    ) -> np.ndarray | None:
        """design_spec.md 特殊情况 last resort: nearest currently-valid
        same-hand controller anchors (the more the better, never crossing
        hands), or None when even those are fewer than the lowest tier."""
        assert self._anchor_indices is not None
        assert self._anchor_first_points is not None
        assert self._controller_first_points is not None
        assert self._controller_hand_labels is not None
        anchors = self._anchor_indices
        hand = self._controller_hand_labels[int(anchors[anchor_column])]
        donor_indices = np.asarray(
            [
                int(candidate_idx)
                for column, candidate_idx in enumerate(anchors.tolist())
                if column != int(anchor_column)
                and usable_frame[int(candidate_idx)]
                and self._controller_hand_labels[int(candidate_idx)] == hand
            ],
            dtype=np.int64,
        )
        if donor_indices.shape[0] < self._recovery_tiers()[-1]:
            return None
        distances = np.linalg.norm(
            self._controller_first_points[donor_indices]
            - self._anchor_first_points[int(anchor_column)],
            axis=1,
        )
        order = np.argsort(distances, kind="stable")
        take = min(int(self.recovery_neighbor_count), donor_indices.shape[0])
        return donor_indices[order[:take]]

    def _recover_anchor(
        self,
        anchor_column: int,
        frame_idx: int,
        usable_frame: np.ndarray,
        ctrl_points_frame: np.ndarray,
    ) -> np.ndarray:
        """Return the recover anchor."""
        assert self._anchor_indices is not None
        assert self._anchor_first_points is not None
        assert self._controller_first_points is not None
        anchor_idx = int(self._anchor_indices[anchor_column])
        neighbors = self._neighbor_table.get(anchor_idx)
        valid_neighbors: list[int] = []
        if neighbors is not None:
            for neighbor_idx in neighbors:
                if usable_frame[int(neighbor_idx)]:
                    valid_neighbors.append(int(neighbor_idx))
                    if len(valid_neighbors) >= self.recovery_neighbor_count:
                        break
        tier = self._recovery_tier(len(valid_neighbors))
        if tier is not None:
            donors = np.asarray(valid_neighbors[:tier], dtype=np.int64)
        else:
            donors = self._fallback_anchor_donors(anchor_column, usable_frame)
        if donors is None:
            min_count = self._recovery_tiers()[-1]
            raise TrackingRecoveryError(
                f"controller anchor recovery found neither {min_count} "
                f"valid neighbors among the nearest {self.neighbor_table_size} "
                f"nor {min_count} valid same-hand fallback anchors; anchor "
                f"column {anchor_column} (query index {anchor_idx}) had "
                f"{len(valid_neighbors)} valid neighbors at window frame {frame_idx}"
            )
        rotation, translation = _rigid_transform(
            self._controller_first_points[donors],
            ctrl_points_frame[donors],
        )
        recovered = rotation @ self._anchor_first_points[anchor_column] + translation
        return recovered.astype(np.float32)

    def process_window(
        self,
        window: Mapping[str, np.ndarray],
        *,
        surface_points: np.ndarray | None = None,
        interior_points: np.ndarray | None = None,
        lookahead_frames: int = 0,
    ) -> dict[str, np.ndarray]:
        """Run the per-window state machine and return track_process arrays.

        ``window`` may carry ``lookahead_frames`` extra rows at the tail (the
        next window's first row(s), the "borrow" frames). Motion consistency
        — including the chunk-0 once-fail selection filter — is computed over
        the extended window so every published row, the tail included, gets a
        real forward-motion verdict. Published outputs are sliced back to the
        window rows; borrow data never reaches published arrays. With
        ``lookahead_frames=0`` (capture end / offline tail) the last row
        publishes origin's end-of-sequence semantics
        (``motions_valid = False``).
        """
        result = {key: np.asarray(value).copy() for key, value in window.items()}
        # Point arrays are (T_ext, N, 3) world-frame meters; visibility masks
        # are (T_ext, N) bool on the same extended-window-frame x query axes.
        obj_points = np.asarray(result["object_points"], dtype=np.float32)
        obj_vis = np.asarray(result["object_visibilities"], dtype=bool)
        ctrl_points = np.asarray(result["controller_points"], dtype=np.float32)
        ctrl_vis = np.asarray(result["controller_visibilities"], dtype=bool)
        extended_count = int(ctrl_points.shape[0])
        lookahead = int(lookahead_frames)
        if lookahead < 0 or lookahead >= max(1, extended_count):
            raise ValueError(
                "lookahead_frames must be >= 0 and smaller than the extended "
                f"window; got {lookahead} of {extended_count} frames"
            )
        frame_count = extended_count - lookahead

        # Phase 1 — motion gates over the extended window. Chunk 0 runs the
        # origin whole-window gates and freezes session identity; later
        # windows verify the frozen query schema. The borrow row makes the
        # tail-row motion (window last -> next window first) a real test in
        # the publishing chunk, which is origin's own indexing for boundary
        # jumps.
        if not self.initialized:
            object_motions_valid, _ = motion_consistency(
                obj_points, obj_vis, once_false_mask=False
            )
            ctrl_motions_valid, ctrl_global = motion_consistency(
                ctrl_points, ctrl_vis, once_false_mask=True
            )
            self._freeze_identity(
                result,
                ctrl_global,
                surface_points=surface_points,
                interior_points=interior_points,
            )
        else:
            self._check_frozen_identity(result)
            if int(ctrl_points.shape[1]) != int(self._controller_first_points.shape[0]):
                raise ValueError(
                    "controller candidate count changed across chunks; session "
                    "query schema is frozen"
                )
            object_motions_valid, _ = motion_consistency(
                obj_points, obj_vis, once_false_mask=False
            )
            ctrl_motions_valid, _ = motion_consistency(
                ctrl_points, ctrl_vis, once_false_mask=False
            )

        assert self._anchor_indices is not None
        assert self._object_column_indices is not None
        assert self._chunk0_controller_mask is not None

        # Phase 2 — mark temporary_invalid frames.
        # design_spec.md temporary_invalid: no direct observation this frame.
        # The motion term only applies where forward motion was testable; the
        # published tail row is testable exactly when a borrow row is present.
        ctrl_motion_failed = _motion_failed_mask(ctrl_vis, ctrl_motions_valid)
        ctrl_usable = ctrl_vis & ~ctrl_motion_failed

        anchors = self._anchor_indices
        anchor_count = int(anchors.shape[0])
        out_points = np.ascontiguousarray(
            ctrl_points[:frame_count, anchors, :], dtype=np.float32
        )
        # Published visibility means "this value is a direct measurement":
        # motion-gate failures are temporary_invalid, so their frames get a
        # rigid proxy value and must not read as visible.
        out_vis = np.ascontiguousarray(ctrl_usable[:frame_count, anchors], dtype=bool)
        out_colors = np.ascontiguousarray(
            np.asarray(result["controller_colors"], dtype=np.float32)[
                :frame_count, anchors, :
            ]
        )
        # Phase 3 — recovery loop over the published rows only (borrow rows
        # are re-processed as the next window's first frames). Each
        # temporarily-invalid anchor frame is filled with a rigid proxy from
        # currently-valid donors (first-frame -> current-frame registration
        # applied to the anchor's first-frame position); when even the
        # fallback donor ladder is too thin, _recover_anchor raises
        # TrackingRecoveryError and the window aborts.
        proxied = np.zeros((frame_count, anchor_count), dtype=bool)
        for frame_idx in range(frame_count):
            invalid_columns = np.flatnonzero(~ctrl_usable[frame_idx, anchors])
            for column in invalid_columns:
                out_points[frame_idx, column] = self._recover_anchor(
                    int(column),
                    frame_idx,
                    ctrl_usable[frame_idx],
                    ctrl_points[frame_idx],
                )
                proxied[frame_idx, column] = True

        # With a borrow row the published tail carries a real forward-motion
        # verdict; without one (capture end) it stays False — origin's
        # end-of-sequence semantics.
        published_object_motions_valid = np.ascontiguousarray(
            object_motions_valid[:frame_count], dtype=bool
        )
        published_controller_candidate_motions_valid = np.ascontiguousarray(
            ctrl_motions_valid[:frame_count], dtype=bool
        )
        published_controller_motions_valid = np.ascontiguousarray(
            published_controller_candidate_motions_valid[:, anchors], dtype=bool
        )

        # Phase 4 — publish: frozen-identity metadata plus this window's
        # values, re-indexed onto the anchor / object-column axes.
        controller_query_indices = np.asarray(
            result["controller_query_indices"], dtype=np.int64
        ).reshape(-1)
        anchor_query_indices = controller_query_indices[anchors]
        # Candidate-axis diagnostics keep the pre-selection arrays inspectable.
        result["controller_candidate_motions_valid"] = np.ascontiguousarray(
            published_controller_candidate_motions_valid, dtype=bool
        )
        result["controller_candidate_mask"] = np.ascontiguousarray(
            self._chunk0_controller_mask, dtype=bool
        )
        result["controller_candidate_query_ids"] = np.ascontiguousarray(
            controller_query_indices, dtype=np.int64
        )
        anchor_status = np.asarray(
            [
                "proxied" if bool(np.any(proxied[:, column])) else "direct"
                for column in range(anchor_count)
            ],
            dtype="<U8",
        )
        neighbor_table_ids = np.full(
            (anchor_count, self.neighbor_table_size), -1, dtype=np.int64
        )
        for column in range(anchor_count):
            neighbors = self._neighbor_table.get(int(anchors[column]))
            if neighbors is None or neighbors.size == 0:
                continue
            ids = controller_query_indices[neighbors]
            neighbor_table_ids[column, : ids.shape[0]] = ids

        cols = self._object_column_indices
        object_query_indices = np.asarray(
            result["object_query_indices"], dtype=np.int64
        ).reshape(-1)
        object_sample_query_ids = object_query_indices[cols]
        object_status = np.asarray(
            [
                "direct" if bool(np.any(obj_vis[:frame_count, int(col)])) else "missing"
                for col in cols
            ],
            dtype="<U8",
        )

        result["object_points"] = np.ascontiguousarray(obj_points[:frame_count, cols, :])
        result["object_colors"] = np.ascontiguousarray(
            np.asarray(result["object_colors"], dtype=np.float32)[:frame_count, cols, :]
        )
        result["object_visibilities"] = np.ascontiguousarray(obj_vis[:frame_count, cols])
        result["object_motions_valid"] = np.ascontiguousarray(
            published_object_motions_valid[:, cols]
        )
        result["object_volume_sample_indices"] = np.ascontiguousarray(cols, dtype=np.int64)
        result["object_sample_indices"] = np.ascontiguousarray(cols, dtype=np.int64)
        result["object_sample_query_ids"] = np.ascontiguousarray(
            object_sample_query_ids, dtype=np.int64
        )
        result["object_selected_query_ids"] = np.ascontiguousarray(
            object_sample_query_ids, dtype=np.int64
        )
        result["object_track_query_indices"] = np.ascontiguousarray(
            object_sample_query_ids, dtype=np.int64
        )
        result["object_track_active_query_indices"] = np.ascontiguousarray(
            object_sample_query_ids, dtype=np.int64
        )
        result["object_track_status"] = object_status

        result["controller_points"] = out_points
        result["controller_colors"] = out_colors
        result["controller_visibilities"] = out_vis
        result["controller_motions_valid"] = published_controller_motions_valid
        result["controller_proxied"] = proxied
        result["controller_mask"] = np.ascontiguousarray(
            self._chunk0_controller_mask, dtype=bool
        )
        result["controller_final_indices"] = np.ascontiguousarray(anchors, dtype=np.int64)
        result["controller_sample_query_ids"] = np.ascontiguousarray(
            anchor_query_indices, dtype=np.int64
        )
        result["controller_track_query_indices"] = np.ascontiguousarray(
            anchor_query_indices, dtype=np.int64
        )
        result["controller_track_active_query_indices"] = np.ascontiguousarray(
            anchor_query_indices, dtype=np.int64
        )
        result["controller_track_status"] = anchor_status
        result["controller_neighbor_query_ids"] = neighbor_table_ids
        result["track_process_status"] = np.asarray(
            TRACK_STATUS_DEGRADED if bool(np.any(proxied)) else TRACK_STATUS_NORMAL
        )
        # Candidate-axis raw diagnostics pass through from the window copy;
        # slice off the borrow rows so no published array carries them.
        for raw_key in (
            "controller_raw_points",
            "controller_raw_visible",
            "controller_processed_mask_valid",
            "controller_depth_valid",
            "controller_measurement_valid",
        ):
            if raw_key in result:
                result[raw_key] = np.ascontiguousarray(
                    np.asarray(result[raw_key])[:frame_count]
                )
        return result
