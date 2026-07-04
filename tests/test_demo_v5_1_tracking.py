"""Unit tests for the design_spec.md tracking state machine."""

from __future__ import annotations

import unittest

import numpy as np

from demo_v5_1 import ffs, segment, tracking


IMAGE_SIZE = 64
# 1 px = 4 mm keeps the 0.01 m motion radius safely between 2 px (inside) and
# 3 px (outside) so cKDTree boundary inclusion never depends on fp rounding.
PIXEL_TO_M = 0.004


def _world_grid() -> tuple[np.ndarray, np.ndarray]:
    rows, cols = np.meshgrid(
        np.arange(IMAGE_SIZE, dtype=np.float32),
        np.arange(IMAGE_SIZE, dtype=np.float32),
        indexing="ij",
    )
    points = np.stack(
        [cols * PIXEL_TO_M, rows * PIXEL_TO_M, np.full_like(rows, -0.1)], axis=-1
    )
    colors = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 128, dtype=np.uint8)
    return points.astype(np.float32), colors


OBJECT_PATCH_YX = [(y, x) for y in range(8, 11) for x in range(8, 11)]
CONTROLLER_PATCH_YX = [(y, x) for y in range(40, 44) for x in range(20, 30)]


def _query_pixels() -> np.ndarray:
    return np.asarray(OBJECT_PATCH_YX + CONTROLLER_PATCH_YX, dtype=np.float32)


def _masks(*, controller_hole_yx: tuple[int, int] | None = None) -> dict[str, np.ndarray]:
    obj = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
    obj[:32, :] = True
    ctrl = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
    ctrl[32:, :] = True
    if controller_hole_yx is not None:
        ctrl[controller_hole_yx] = False
    return {"object": obj, "controller": ctrl}


HAND_SPLIT_COL = 25


def _hand_masks() -> dict[str, np.ndarray]:
    """Controller region split into hand_a (x < 25) and hand_b (x >= 25)."""
    frame = _masks()
    hand_a = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
    hand_b = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
    hand_a[32:, :HAND_SPLIT_COL] = True
    hand_b[32:, HAND_SPLIT_COL:] = True
    frame["hand_a"] = hand_a
    frame["hand_b"] = hand_b
    return frame


def _window(
    *,
    frame_count: int = 4,
    shift_px_per_window: int = 0,
    visibility_override: np.ndarray | None = None,
    mask_frames: list[dict[str, np.ndarray]] | None = None,
    query_ids: np.ndarray | None = None,
    query_semantic_labels: np.ndarray | None = None,
    pixel_offsets: dict[int, dict[int, tuple[float, float]]] | None = None,
) -> dict[str, np.ndarray]:
    """Build a synthetic static (or uniformly shifted) observation window."""
    base = _query_pixels()
    query_count = base.shape[0]
    tracks = np.repeat(base[None], frame_count, axis=0).astype(np.float32)
    tracks[:, :, 1] += float(shift_px_per_window)
    if pixel_offsets:
        for frame_idx, per_query in pixel_offsets.items():
            for query_idx, (dy, dx) in per_query.items():
                tracks[frame_idx, query_idx, 0] += float(dy)
                tracks[frame_idx, query_idx, 1] += float(dx)
    visibility = (
        np.ones((frame_count, query_count), dtype=bool)
        if visibility_override is None
        else visibility_override
    )
    points, colors = _world_grid()
    pcd_points = np.repeat(points[None][None], frame_count, axis=0)
    pcd_colors = np.repeat(colors[None][None], frame_count, axis=0)
    frames = mask_frames if mask_frames is not None else [_masks()] * frame_count
    return tracking.build_window_observations(
        tracks_yx=tracks,
        visibility=visibility,
        mask_frames=frames,
        pcd_points=pcd_points,
        pcd_colors=pcd_colors,
        query_ids=query_ids,
        query_semantic_labels=query_semantic_labels,
    )


def _runtime() -> tracking.TrackingRuntime:
    return tracking.TrackingRuntime(
        controller_count=5,
        neighbor_table_size=10,
        recovery_neighbor_count=3,
    )


class Frame0LabelingTests(unittest.TestCase):
    def test_labels_require_visible_and_class_mask(self) -> None:
        masks = _masks()
        tracks0 = np.asarray(
            [[9.0, 9.0], [41.0, 21.0], [42.0, 22.0]], dtype=np.float32
        )
        visibility0 = np.asarray([True, False, True])
        labels = segment.frame0_semantic_labels(
            tracks0,
            visibility0,
            object_mask=masks["object"],
            controller_mask=masks["controller"],
        )
        self.assertEqual(int(labels[0]), int(segment.QUERY_SEMANTIC_OBJECT))
        # Visible-at-frame-0 is required: invisible query stays unlabeled.
        self.assertEqual(int(labels[1]), int(segment.QUERY_SEMANTIC_NONE))
        self.assertEqual(int(labels[2]), int(segment.QUERY_SEMANTIC_CONTROLLER))

    def test_out_of_bounds_frame0_track_is_unlabeled_not_an_error(self) -> None:
        masks = _masks()
        tracks0 = np.asarray(
            [[63.6, 10.0], [np.nan, np.nan], [9.0, 9.0]], dtype=np.float32
        )
        visibility0 = np.asarray([True, True, True])
        labels = segment.frame0_semantic_labels(
            tracks0,
            visibility0,
            object_mask=masks["object"],
            controller_mask=masks["controller"],
        )
        self.assertEqual(int(labels[0]), int(segment.QUERY_SEMANTIC_NONE))
        self.assertEqual(int(labels[1]), int(segment.QUERY_SEMANTIC_NONE))
        self.assertEqual(int(labels[2]), int(segment.QUERY_SEMANTIC_OBJECT))

    def test_mask_overlap_keeps_object_valid(self) -> None:
        # Origin parity: no controller-mask subtraction from the object mask.
        obj = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
        ctrl = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
        obj[10, 10] = True
        ctrl[10, 10] = True
        valid = segment.class_mask_valid(
            np.asarray([[10.0, 10.0]], dtype=np.float32),
            np.asarray([segment.QUERY_SEMANTIC_OBJECT], dtype=np.int8),
            object_mask=obj,
            controller_mask=ctrl,
            tracker_visible=np.asarray([True]),
        )
        self.assertTrue(bool(valid[0]))


class Frame0HandLabelTests(unittest.TestCase):
    def test_hand_labels_follow_masks_with_hand_a_winning_overlap(self) -> None:
        frame = _hand_masks()
        frame["hand_a"][40, 26] = True  # overlap with hand_b at (40, 26)
        tracks0 = np.asarray(
            [[40.0, 21.0], [40.0, 27.0], [40.0, 26.0], [9.0, 9.0], [40.0, 22.0]],
            dtype=np.float32,
        )
        visibility0 = np.asarray([True, True, True, True, False])
        labels = segment.frame0_hand_labels(tracks0, visibility0, mask_frame=frame)
        self.assertEqual(int(labels[0]), int(segment.QUERY_HAND_A))
        self.assertEqual(int(labels[1]), int(segment.QUERY_HAND_B))
        self.assertEqual(int(labels[2]), int(segment.QUERY_HAND_A))  # overlap
        self.assertEqual(int(labels[3]), int(segment.QUERY_HAND_NONE))  # object area
        self.assertEqual(int(labels[4]), int(segment.QUERY_HAND_NONE))  # invisible

    def test_frames_without_hand_masks_label_none(self) -> None:
        labels = segment.frame0_hand_labels(
            np.asarray([[40.0, 21.0]], dtype=np.float32),
            np.asarray([True]),
            mask_frame=_masks(),
        )
        self.assertEqual(int(labels[0]), int(segment.QUERY_HAND_NONE))


class SameHandNeighborTableTests(unittest.TestCase):
    def _frozen_runtime(self) -> tuple[tracking.TrackingRuntime, dict[str, np.ndarray]]:
        runtime = _runtime()
        window0 = _window(frame_count=4, mask_frames=[_hand_masks()] * 4)
        runtime.process_window(window0)
        return runtime, window0

    def test_neighbor_table_never_crosses_hands(self) -> None:
        runtime, window0 = self._frozen_runtime()
        hand_labels = np.asarray(window0["controller_hand_labels"], dtype=np.int8)
        self.assertTrue(bool(np.any(hand_labels == segment.QUERY_HAND_A)))
        self.assertTrue(bool(np.any(hand_labels == segment.QUERY_HAND_B)))
        self.assertTrue(runtime._neighbor_table)
        for candidate_idx, neighbors in runtime._neighbor_table.items():
            self.assertGreater(neighbors.size, 0)
            self.assertTrue(
                bool(np.all(hand_labels[neighbors] == hand_labels[candidate_idx])),
                f"candidate {candidate_idx} has cross-hand neighbors",
            )

    def test_small_hand_fills_with_whole_hand(self) -> None:
        # Patch columns 20-29 split at 25: each hand holds 4x5 = 20 candidates,
        # fewer than a table_size of 25, so entries hold the hand minus self.
        runtime = tracking.TrackingRuntime(
            controller_count=5, neighbor_table_size=25, recovery_neighbor_count=3
        )
        window0 = _window(frame_count=4, mask_frames=[_hand_masks()] * 4)
        runtime.process_window(window0)
        for neighbors in runtime._neighbor_table.values():
            self.assertEqual(int(neighbors.size), 19)

    def test_recovery_cannot_borrow_the_other_hand(self) -> None:
        runtime, window0 = self._frozen_runtime()
        hand_labels = np.asarray(window0["controller_hand_labels"], dtype=np.int8)
        anchors = np.asarray(runtime._anchor_indices)
        hand_a_columns = np.flatnonzero(
            hand_labels[anchors] == segment.QUERY_HAND_A
        )
        self.assertGreater(hand_a_columns.size, 0)

        # Frame 1: every hand_a controller query loses tracking while hand_b
        # stays fully valid. Same-hand tables must refuse hand_b donors.
        frame_count = 3
        query_count = len(OBJECT_PATCH_YX) + len(CONTROLLER_PATCH_YX)
        visibility = np.ones((frame_count, query_count), dtype=bool)
        controller_query_indices = np.asarray(window0["controller_query_indices"])
        hand_a_queries = controller_query_indices[
            np.flatnonzero(hand_labels == segment.QUERY_HAND_A)
        ]
        visibility[1, hand_a_queries] = False
        window1 = _window(
            frame_count=frame_count,
            visibility_override=visibility,
            mask_frames=[_hand_masks()] * frame_count,
        )
        with self.assertRaisesRegex(
            tracking.TrackingRecoveryError, "valid neighbors among the nearest"
        ):
            runtime.process_window(window1)


class TemporaryInvalidGateTests(unittest.TestCase):
    def test_mask_depth_and_visibility_gates(self) -> None:
        frame_count = 4
        query_count = len(OBJECT_PATCH_YX) + len(CONTROLLER_PATCH_YX)
        ctrl0_column = 0  # first controller candidate
        ctrl0_query = len(OBJECT_PATCH_YX)

        visibility = np.ones((frame_count, query_count), dtype=bool)
        visibility[2, ctrl0_query] = False  # tracker invisible at frame 2
        mask_frames = [_masks() for _ in range(frame_count)]
        mask_frames[1] = _masks(controller_hole_yx=CONTROLLER_PATCH_YX[0])

        window = _window(
            frame_count=frame_count,
            visibility_override=visibility,
            mask_frames=mask_frames,
        )
        measurement = window["controller_measurement_valid"]
        self.assertTrue(bool(measurement[0, ctrl0_column]))
        self.assertFalse(bool(measurement[1, ctrl0_column]))  # mask reject
        self.assertFalse(bool(measurement[2, ctrl0_column]))  # tracker invisible
        self.assertTrue(bool(measurement[3, ctrl0_column]))

    def test_zero_depth_invalidates(self) -> None:
        points, colors = _world_grid()
        y, x = CONTROLLER_PATCH_YX[0]
        points = points.copy()
        points[y, x] = 0.0
        yy, xx, in_bounds = ffs.round_tracks_to_pixels(
            np.asarray([[float(y), float(x)]], dtype=np.float32), points.shape[:2]
        )
        _pts, _cols, depth_valid = ffs.sample_world_pcd_at_pixels(
            points, colors, yy=yy, xx=xx, sample=in_bounds
        )
        self.assertFalse(bool(depth_valid[0]))


class MotionConsistencyTests(unittest.TestCase):
    def test_origin_motion_consistency_keeps_tail_row_false(self) -> None:
        window = _window(frame_count=4)
        pts = np.asarray(window["object_points"], dtype=np.float32)
        vis = np.asarray(window["object_visibilities"], dtype=bool)
        motions_valid, _ = tracking.motion_consistency(
            pts, vis, once_false_mask=False
        )
        self.assertFalse(bool(np.any(motions_valid[-1])))

    def test_divergent_motion_is_rejected_per_frame(self) -> None:
        frame_count = 3
        window = _window(frame_count=frame_count)
        pts = np.asarray(window["object_points"], dtype=np.float32).copy()
        vis = np.asarray(window["object_visibilities"], dtype=bool)
        # Column 4 (patch center) moves 25 mm between frames 0 and 1 while
        # every neighbor stays still: origin similarity gate (5 mm) fails.
        pts[1, 4, 0] += 0.025
        motions_valid, _ = tracking.motion_consistency(pts, vis, once_false_mask=False)
        self.assertFalse(bool(motions_valid[0, 4]))
        self.assertTrue(bool(motions_valid[0, 0]))

    def test_once_false_mask_removes_candidate_globally(self) -> None:
        frame_count = 3
        window = _window(frame_count=frame_count)
        pts = np.asarray(window["controller_points"], dtype=np.float32).copy()
        vis = np.asarray(window["controller_visibilities"], dtype=bool)
        pts[1, 5, 0] += 0.025
        _valid, global_mask = tracking.motion_consistency(
            pts, vis, once_false_mask=True
        )
        self.assertFalse(bool(global_mask[5]))
        self.assertTrue(bool(global_mask[0]))


class Chunk0SelectionTests(unittest.TestCase):
    def test_whole_window_validity_excludes_candidate(self) -> None:
        frame_count = 4
        query_count = len(OBJECT_PATCH_YX) + len(CONTROLLER_PATCH_YX)
        flaky_query = len(OBJECT_PATCH_YX) + 7
        visibility = np.ones((frame_count, query_count), dtype=bool)
        visibility[2, flaky_query] = False
        window = _window(frame_count=frame_count, visibility_override=visibility)
        runtime = _runtime()
        result = runtime.process_window(window)
        controller_query_indices = np.asarray(window["controller_query_indices"])
        flaky_candidate = int(np.flatnonzero(controller_query_indices == flaky_query)[0])
        self.assertNotIn(
            flaky_candidate, result["controller_final_indices"].tolist()
        )
        self.assertEqual(result["controller_points"].shape[1], 5)
        self.assertEqual(str(result["track_process_status"]), "normal")
        self.assertFalse(bool(result["controller_proxied"].any()))

    def test_too_few_survivors_raises(self) -> None:
        frame_count = 3
        query_count = len(OBJECT_PATCH_YX) + len(CONTROLLER_PATCH_YX)
        visibility = np.ones((frame_count, query_count), dtype=bool)
        # Leave only three controller candidates whole-window visible.
        visibility[1, len(OBJECT_PATCH_YX) + 3 :] = False
        window = _window(frame_count=frame_count, visibility_override=visibility)
        with self.assertRaises(tracking.ControllerSelectionError):
            _runtime().process_window(window)


class CrossWindowTests(unittest.TestCase):
    def _init_runtime(self) -> tuple[tracking.TrackingRuntime, dict[str, np.ndarray]]:
        runtime = _runtime()
        window0 = _window(frame_count=4)
        result0 = runtime.process_window(window0)
        return runtime, result0

    def test_identity_frozen_across_windows(self) -> None:
        runtime, result0 = self._init_runtime()
        window1 = _window(frame_count=4, shift_px_per_window=4)
        result1 = runtime.process_window(window1)
        np.testing.assert_array_equal(
            result0["controller_sample_query_ids"],
            result1["controller_sample_query_ids"],
        )
        np.testing.assert_array_equal(
            result0["object_sample_query_ids"], result1["object_sample_query_ids"]
        )

    def test_published_motion_valid_marks_chunk_tail_valid(self) -> None:
        runtime, result0 = self._init_runtime()
        result1 = runtime.process_window(
            _window(frame_count=4, shift_px_per_window=4)
        )

        for result in (result0, result1):
            object_valid = np.asarray(result["object_motions_valid"], dtype=bool)
            controller_valid = np.asarray(
                result["controller_motions_valid"], dtype=bool
            )
            candidate_valid = np.asarray(
                result["controller_candidate_motions_valid"], dtype=bool
            )
            self.assertTrue(bool(np.all(object_valid[-1])))
            self.assertTrue(bool(np.all(controller_valid[-1])))
            self.assertTrue(bool(np.all(candidate_valid[-1])))

        self.assertTrue(
            bool(np.all(np.asarray(result1["object_motions_valid"], dtype=bool)[0]))
        )
        self.assertTrue(
            bool(
                np.all(
                    np.asarray(result1["controller_motions_valid"], dtype=bool)[0]
                )
            )
        )

    def test_rigid_recovery_fills_temporary_invalid_anchor(self) -> None:
        runtime, result0 = self._init_runtime()
        anchor_candidates = np.asarray(result0["controller_final_indices"])
        anchor_column = 0
        anchor_candidate = int(anchor_candidates[anchor_column])
        anchor_pixel = CONTROLLER_PATCH_YX[anchor_candidate]

        frame_count = 4
        hole_frame = 2
        shift_px = 4
        mask_frames = [_masks() for _ in range(frame_count)]
        mask_frames[hole_frame] = _masks(
            controller_hole_yx=(anchor_pixel[0], anchor_pixel[1] + shift_px)
        )
        window1 = _window(
            frame_count=frame_count,
            shift_px_per_window=shift_px,
            mask_frames=mask_frames,
        )
        result1 = runtime.process_window(window1)

        proxied = np.asarray(result1["controller_proxied"], dtype=bool)
        self.assertTrue(bool(proxied[hole_frame, anchor_column]))
        self.assertEqual(int(np.count_nonzero(proxied)), 1)
        self.assertFalse(
            bool(result1["controller_visibilities"][hole_frame, anchor_column])
        )
        self.assertEqual(str(result1["track_process_status"]), "degraded")

        # Neighbors moved by a pure +x translation of shift_px pixels, so the
        # rigid fit must place the anchor at first-frame + shift.
        first_point = np.asarray(result0["controller_points"])[0, anchor_column]
        expected = first_point + np.asarray(
            [shift_px * PIXEL_TO_M, 0.0, 0.0], dtype=np.float32
        )
        recovered = np.asarray(result1["controller_points"])[hole_frame, anchor_column]
        np.testing.assert_allclose(recovered, expected, atol=1e-5)

    def test_recovery_with_too_few_neighbors_raises(self) -> None:
        runtime, _result0 = self._init_runtime()
        frame_count = 3
        query_count = len(OBJECT_PATCH_YX) + len(CONTROLLER_PATCH_YX)
        visibility = np.ones((frame_count, query_count), dtype=bool)
        # Frame 1: every controller candidate loses tracking.
        visibility[1, len(OBJECT_PATCH_YX) :] = False
        window1 = _window(frame_count=frame_count, visibility_override=visibility)
        with self.assertRaisesRegex(
            tracking.TrackingRecoveryError, "valid neighbors among the nearest"
        ):
            runtime.process_window(window1)

    def test_neighbor_table_and_first_points_never_update(self) -> None:
        runtime, _result0 = self._init_runtime()
        first_points_snapshot = np.array(runtime._controller_first_points, copy=True)
        table_snapshot = {
            key: np.array(value, copy=True)
            for key, value in runtime._neighbor_table.items()
        }
        runtime.process_window(_window(frame_count=4, shift_px_per_window=4))
        np.testing.assert_array_equal(
            first_points_snapshot, runtime._controller_first_points
        )
        self.assertEqual(set(table_snapshot), set(runtime._neighbor_table))
        for key, value in table_snapshot.items():
            np.testing.assert_array_equal(value, runtime._neighbor_table[key])

    def test_motion_failure_alone_triggers_proxying(self) -> None:
        runtime, result0 = self._init_runtime()
        anchors = np.asarray(result0["controller_final_indices"])
        anchor_column = 0
        jump_query = len(OBJECT_PATCH_YX) + int(anchors[anchor_column])

        frame_count = 3
        # Frame 1 only: the anchor's track jumps 6 px (24 mm) while every
        # neighbor stays still — mask and depth stay valid, so the failure is
        # purely the origin motion-consistency gate.
        window1 = _window(
            frame_count=frame_count,
            pixel_offsets={1: {jump_query: (0.0, 6.0)}},
        )
        result1 = runtime.process_window(window1)
        proxied = np.asarray(result1["controller_proxied"], dtype=bool)
        self.assertTrue(bool(proxied[0, anchor_column]))
        self.assertFalse(
            bool(result1["controller_visibilities"][0, anchor_column])
        )
        # Static neighbors give an identity rigid fit: the proxied value is
        # the anchor's first-frame position.
        first_point = np.asarray(result0["controller_points"])[0, anchor_column]
        np.testing.assert_allclose(
            np.asarray(result1["controller_points"])[0, anchor_column],
            first_point,
            atol=1e-5,
        )

    def test_chunk_boundary_jump_is_caught_by_seam_carry(self) -> None:
        runtime, result0 = self._init_runtime()
        anchors = np.asarray(result0["controller_final_indices"])
        anchor_column = None
        for column, candidate in enumerate(anchors.tolist()):
            if CONTROLLER_PATCH_YX[candidate][1] <= 23:
                anchor_column = column
                break
        self.assertIsNotNone(anchor_column)
        jump_candidate = int(anchors[anchor_column])
        jump_query = len(OBJECT_PATCH_YX) + jump_candidate

        # The anchor jumps 6 px between window 0 and window 1 and then stays
        # static inside the cluster. Within window 1 its motion is zero and
        # consistent, so only the carried seam test can catch the jump.
        frame_count = 3
        window1 = _window(
            frame_count=frame_count,
            pixel_offsets={
                frame: {jump_query: (0.0, 6.0)} for frame in range(frame_count)
            },
        )
        result1 = runtime.process_window(window1)
        proxied = np.asarray(result1["controller_proxied"], dtype=bool)
        self.assertTrue(bool(proxied[0, anchor_column]))
        self.assertFalse(bool(proxied[1, anchor_column]))
        self.assertEqual(str(result1["track_process_status"]), "degraded")

    def test_object_frames_without_observation_stay_unsynthesized(self) -> None:
        runtime, result0 = self._init_runtime()
        object_columns = np.asarray(result0["object_volume_sample_indices"])
        target_column = 0
        target_query = int(np.asarray(result0["object_sample_query_ids"])[target_column])

        frame_count = 3
        query_count = len(OBJECT_PATCH_YX) + len(CONTROLLER_PATCH_YX)
        visibility = np.ones((frame_count, query_count), dtype=bool)
        visibility[1, target_query] = False
        window1 = _window(frame_count=frame_count, visibility_override=visibility)
        result1 = runtime.process_window(window1)

        self.assertFalse(bool(result1["object_visibilities"][1, target_column]))
        np.testing.assert_array_equal(
            result1["object_points"][1, target_column], np.zeros((3,), dtype=np.float32)
        )
        self.assertTrue(bool(result1["object_visibilities"][0, target_column]))
        # 3x3 patch at 4 mm spacing collapses deterministically onto a 2x2 set
        # of occupied 5 mm voxels (offsets 0/0.004/0.008 -> voxels 0/0/1).
        self.assertEqual(len(object_columns), 4)


class RecoveryLadderTests(unittest.TestCase):
    def test_tier_selection_matches_spec_ladder(self) -> None:
        runtime = tracking.TrackingRuntime(
            controller_count=5, neighbor_table_size=100, recovery_neighbor_count=15
        )
        self.assertEqual(runtime._recovery_tiers(), [15, 10, 5])
        self.assertEqual(runtime._recovery_tier(20), 15)
        self.assertEqual(runtime._recovery_tier(15), 15)
        self.assertEqual(runtime._recovery_tier(12), 10)
        self.assertEqual(runtime._recovery_tier(10), 10)
        self.assertEqual(runtime._recovery_tier(7), 5)
        self.assertEqual(runtime._recovery_tier(5), 5)
        self.assertIsNone(runtime._recovery_tier(4))
        self.assertIsNone(runtime._recovery_tier(0))

    def test_degraded_tier_still_recovers(self) -> None:
        # 40 candidates, 20-entry tables, ladder [15, 10, 5]: killing the
        # anchor plus its 13 nearest neighbors leaves 7 valid -> tier 5.
        runtime = tracking.TrackingRuntime(
            controller_count=5, neighbor_table_size=20, recovery_neighbor_count=15
        )
        window0 = _window(frame_count=4)
        result0 = runtime.process_window(window0)
        anchor_column = 0
        anchor_candidate = int(result0["controller_final_indices"][anchor_column])
        table = np.asarray(runtime._neighbor_table[anchor_candidate])
        self.assertEqual(int(table.size), 20)

        frame_count = 3
        query_count = len(OBJECT_PATCH_YX) + len(CONTROLLER_PATCH_YX)
        visibility = np.ones((frame_count, query_count), dtype=bool)
        for candidate in [anchor_candidate] + [int(i) for i in table[:13]]:
            visibility[1, len(OBJECT_PATCH_YX) + candidate] = False
        window1 = _window(frame_count=frame_count, visibility_override=visibility)
        result1 = runtime.process_window(window1)

        proxied = np.asarray(result1["controller_proxied"], dtype=bool)
        self.assertTrue(bool(proxied[1, anchor_column]))
        self.assertEqual(str(result1["track_process_status"]), "degraded")
        # Static neighbors give an identity rigid fit.
        first_point = np.asarray(result0["controller_points"])[0, anchor_column]
        np.testing.assert_allclose(
            np.asarray(result1["controller_points"])[1, anchor_column],
            first_point,
            atol=1e-5,
        )

    def test_anchor_fallback_recovers_when_table_is_exhausted(self) -> None:
        # 6-entry tables with ladder [6, 4, 3]: killing the anchor plus four
        # table neighbors leaves at most 2 valid -> ladder exhausted -> the
        # nearest currently-valid anchors take over as donors.
        runtime = tracking.TrackingRuntime(
            controller_count=8, neighbor_table_size=6, recovery_neighbor_count=6
        )
        self.assertEqual(runtime._recovery_tiers(), [6, 4, 3])
        window0 = _window(frame_count=4)
        result0 = runtime.process_window(window0)
        anchors = np.asarray(result0["controller_final_indices"])
        anchor_column = 0
        anchor_candidate = int(anchors[anchor_column])
        table = np.asarray(runtime._neighbor_table[anchor_candidate])

        frame_count = 3
        query_count = len(OBJECT_PATCH_YX) + len(CONTROLLER_PATCH_YX)
        visibility = np.ones((frame_count, query_count), dtype=bool)
        dead = {anchor_candidate} | {int(i) for i in table[:4]}
        for candidate in dead:
            visibility[1, len(OBJECT_PATCH_YX) + candidate] = False
        # Precondition: at least 3 anchors stay usable as fallback donors.
        self.assertGreaterEqual(len([a for a in anchors.tolist() if a not in dead]), 3)

        window1 = _window(frame_count=frame_count, visibility_override=visibility)
        result1 = runtime.process_window(window1)
        proxied = np.asarray(result1["controller_proxied"], dtype=bool)
        self.assertTrue(bool(proxied[1, anchor_column]))
        first_point = np.asarray(result0["controller_points"])[0, anchor_column]
        np.testing.assert_allclose(
            np.asarray(result1["controller_points"])[1, anchor_column],
            first_point,
            atol=1e-5,
        )

    def test_anchor_fallback_never_crosses_hands(self) -> None:
        runtime = _runtime()
        window0 = _window(frame_count=4, mask_frames=[_hand_masks()] * 4)
        result0 = runtime.process_window(window0)
        hand_labels = np.asarray(window0["controller_hand_labels"], dtype=np.int8)
        anchors = np.asarray(result0["controller_final_indices"])
        hand_a_anchor_columns = np.flatnonzero(
            hand_labels[anchors] == segment.QUERY_HAND_A
        )
        self.assertGreater(hand_a_anchor_columns.size, 0)
        anchor_column = int(hand_a_anchor_columns[0])

        usable = np.zeros((len(hand_labels),), dtype=bool)
        usable[hand_labels == segment.QUERY_HAND_B] = True  # only hand_b valid
        self.assertIsNone(runtime._fallback_anchor_donors(anchor_column, usable))

        usable[anchors[hand_a_anchor_columns]] = True  # same-hand anchors valid
        donors = runtime._fallback_anchor_donors(anchor_column, usable)
        if hand_a_anchor_columns.size - 1 >= runtime._recovery_tiers()[-1]:
            self.assertIsNotNone(donors)
            self.assertTrue(
                bool(np.all(hand_labels[donors] == segment.QUERY_HAND_A))
            )


class NoConfidenceContractTests(unittest.TestCase):
    def test_output_has_proxied_mask_and_no_confidence(self) -> None:
        runtime = _runtime()
        result = runtime.process_window(_window(frame_count=3))
        self.assertIn("controller_proxied", result)
        for key in result:
            self.assertNotIn("confidence", key)
            self.assertNotIn("track_mode", key)
            self.assertNotIn("filter_reason", key)


if __name__ == "__main__":
    unittest.main()
