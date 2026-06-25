from __future__ import annotations

import pickle
from pathlib import Path
import tempfile
import unittest

import numpy as np

from qqtt.demo import phystwin_strict_product as strict


def _reference_motion_valid_for_class(
    points: np.ndarray,
    visibilities: np.ndarray,
    *,
    neighbor_dist: float,
    min_neighbors: int,
    motion_similarity_m: float,
    once_false_mask: bool,
) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float32)
    vis = np.asarray(visibilities, dtype=bool)
    motions_valid = np.zeros_like(vis, dtype=bool)
    if pts.shape[0] > 1:
        motions_valid[:-1] = vis[:-1] & vis[1:]
    global_mask = np.prod(vis, axis=0).astype(bool) if once_false_mask and vis.size else np.ones((pts.shape[1],), dtype=bool)
    if pts.shape[1] == 0:
        return motions_valid, global_mask
    motions = np.zeros_like(pts, dtype=np.float32)
    motions[:-1] = pts[1:] - pts[:-1]
    for frame_idx in range(max(0, pts.shape[0] - 1)):
        if once_false_mask:
            motions_valid[frame_idx] &= global_mask
        for query_idx in range(pts.shape[1]):
            if once_false_mask and not global_mask[query_idx]:
                motions_valid[frame_idx, query_idx] = False
                continue
            if not motions_valid[frame_idx, query_idx]:
                continue
            distances = np.linalg.norm(pts[frame_idx] - pts[frame_idx, query_idx], axis=1)
            neighbors = np.flatnonzero((distances <= float(neighbor_dist)) & motions_valid[frame_idx])
            if len(neighbors) < int(min_neighbors):
                motions_valid[frame_idx, query_idx] = False
                if once_false_mask:
                    global_mask[query_idx] = False
                continue
            motion_diff = np.linalg.norm(motions[frame_idx, query_idx] - motions[frame_idx, neighbors], axis=1)
            if int(np.count_nonzero(motion_diff < float(motion_similarity_m))) < 0.5 * float(len(neighbors)):
                motions_valid[frame_idx, query_idx] = False
                if once_false_mask:
                    global_mask[query_idx] = False
        if once_false_mask:
            motions_valid[frame_idx] &= global_mask
    return motions_valid, global_mask.astype(bool, copy=False)


def _filtered_controller_track(
    points: np.ndarray,
    *,
    query_ids: np.ndarray | None = None,
    controller_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    pts = np.ascontiguousarray(np.asarray(points, dtype=np.float32))
    if pts.ndim != 3 or pts.shape[-1] != 3:
        raise ValueError("points must have shape T,N,3")
    frame_count, point_count, _ = pts.shape
    if query_ids is None:
        query_ids = np.arange(point_count, dtype=np.int64)
    if controller_mask is None:
        controller_mask = np.ones((point_count,), dtype=bool)
    return {
        "object_points": np.zeros((frame_count, 0, 3), dtype=np.float32),
        "object_colors": np.zeros((frame_count, 0, 3), dtype=np.float32),
        "object_visibilities": np.zeros((frame_count, 0), dtype=bool),
        "object_motions_valid": np.zeros((frame_count, 0), dtype=bool),
        "controller_points": pts,
        "controller_colors": np.ones_like(pts, dtype=np.float32),
        "controller_visibilities": np.ones((frame_count, point_count), dtype=bool),
        "controller_motions_valid": np.ones((frame_count, point_count), dtype=bool),
        "controller_mask": np.ascontiguousarray(np.asarray(controller_mask, dtype=bool)),
        "controller_query_indices": np.ascontiguousarray(np.asarray(query_ids, dtype=np.int64)),
    }


def _filtered_object_track(
    points: np.ndarray,
    *,
    query_ids: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    pts = np.ascontiguousarray(np.asarray(points, dtype=np.float32))
    if pts.ndim != 3 or pts.shape[-1] != 3:
        raise ValueError("points must have shape T,N,3")
    frame_count, point_count, _ = pts.shape
    if query_ids is None:
        query_ids = np.arange(point_count, dtype=np.int64)
    return {
        "object_points": pts,
        "object_colors": np.ones_like(pts, dtype=np.float32),
        "object_visibilities": np.ones((frame_count, point_count), dtype=bool),
        "object_motions_valid": np.ones((frame_count, point_count), dtype=bool),
        "object_query_indices": np.ascontiguousarray(np.asarray(query_ids, dtype=np.int64)),
        "controller_points": np.zeros((frame_count, 0, 3), dtype=np.float32),
        "controller_colors": np.zeros((frame_count, 0, 3), dtype=np.float32),
        "controller_visibilities": np.zeros((frame_count, 0), dtype=bool),
        "controller_motions_valid": np.zeros((frame_count, 0), dtype=bool),
        "controller_mask": np.zeros((0,), dtype=bool),
    }


class PhysTwinStrictProductTest(unittest.TestCase):
    def test_first_frame_union_sampler_exports_txy_and_internal_yx(self) -> None:
        object_mask = np.zeros((3, 4), dtype=bool)
        controller_mask = np.zeros((3, 4), dtype=bool)
        object_mask[0, 1] = True
        object_mask[1, 2] = True
        controller_mask[2, 0] = True
        controller_mask[1, 2] = True

        sample = strict.sample_first_frame_union_queries(
            object_mask,
            controller_mask,
            max_queries=10,
            seed=7,
            camera_idx=0,
        )

        self.assertEqual(sample.query_txy.shape, (3, 3))
        self.assertEqual(sample.query_points_yx.shape, (3, 2))
        self.assertTrue(np.all(sample.query_txy[:, 0] == 0.0))
        np.testing.assert_array_equal(sample.query_txy[:, 1:], sample.query_points_yx[:, ::-1])
        self.assertEqual({tuple(row) for row in sample.query_points_yx.astype(int)}, {(0, 1), (1, 2), (2, 0)})

    def test_dense_world_pcd_grid_zeroes_invalid_depth(self) -> None:
        depth_m = np.array([[1.0, 0.0], [2.0, np.nan]], dtype=np.float32)
        color_rgb = np.array(
            [
                [[10, 20, 30], [40, 50, 60]],
                [[70, 80, 90], [100, 110, 120]],
            ],
            dtype=np.uint8,
        )
        intrinsics = {"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0}
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, 3] = np.array([10.0, 0.0, 0.5], dtype=np.float32)

        points, colors = strict.dense_world_pcd_grid(
            depth_m=depth_m,
            color_rgb_u8=color_rgb,
            intrinsics=intrinsics,
            c2w=c2w,
        )

        self.assertEqual(points.shape, (1, 2, 2, 3))
        self.assertEqual(colors.shape, (1, 2, 2, 3))
        np.testing.assert_allclose(points[0, 0, 0], np.array([10.0, 0.0, 1.5], dtype=np.float32))
        np.testing.assert_allclose(points[0, 1, 0], np.array([10.0, 2.0, 2.5], dtype=np.float32))
        np.testing.assert_allclose(points[0, 0, 1], np.zeros((3,), dtype=np.float32))
        np.testing.assert_allclose(points[0, 1, 1], np.zeros((3,), dtype=np.float32))
        np.testing.assert_array_equal(colors[0], color_rgb)

    def test_write_processed_masks_pickle_uses_controller_union_and_keeps_hands(self) -> None:
        hand_a = np.array([[True, False], [False, False]])
        hand_b = np.array([[False, False], [True, False]])
        obj = np.array([[False, True], [False, False]])

        with tempfile.TemporaryDirectory() as tmp:
            path = strict.write_processed_masks(
                Path(tmp),
                [
                    {
                        "object": obj,
                        "hand_a": hand_a,
                        "hand_b": hand_b,
                    }
                ],
            )
            with path.open("rb") as handle:
                payload = pickle.load(handle)

        self.assertEqual(len(payload), 1)
        frame = payload[0][0]
        np.testing.assert_array_equal(frame["object"], obj)
        np.testing.assert_array_equal(frame["controller"], hand_a | hand_b)
        np.testing.assert_array_equal(frame["hand_a"], hand_a)
        np.testing.assert_array_equal(frame["hand_b"], hand_b)

    def test_object_motion_valid_is_per_transition_not_once_false(self) -> None:
        object_points = np.zeros((3, 11, 3), dtype=np.float32)
        object_points[0, :6, 0] = np.arange(6, dtype=np.float32) * 0.001
        object_points[0, 6:, 0] = np.arange(5, dtype=np.float32) * 0.001
        object_points[1, :6, 0] = object_points[0, :6, 0] + 0.001
        object_points[1, 0, 0] = 0.050
        object_points[1, 6:, 0] = 0.050 + np.arange(5, dtype=np.float32) * 0.001
        object_points[2] = object_points[1] + np.array([0.001, 0.0, 0.0], dtype=np.float32)
        object_vis = np.ones((3, 11), dtype=bool)
        object_vis[0, 6:] = False
        track_data = {
            "object_points": object_points,
            "object_colors": np.ones((3, 11, 3), dtype=np.float32),
            "object_visibilities": object_vis,
            "controller_points": np.zeros((3, 0, 3), dtype=np.float32),
            "controller_colors": np.zeros((3, 0, 3), dtype=np.float32),
            "controller_visibilities": np.zeros((3, 0), dtype=bool),
        }

        filtered = strict.apply_phystwin_motion_filters(track_data)

        self.assertFalse(bool(filtered["object_motions_valid"][0, 0]))
        self.assertTrue(bool(filtered["object_motions_valid"][1, 0]))

    def test_controller_requires_whole_window_visibility_and_motion_then_fps30(self) -> None:
        points = np.zeros((2, 32, 3), dtype=np.float32)
        points[0, :, 0] = np.arange(32, dtype=np.float32) * 0.001
        points[1] = points[0] + np.array([0.001, 0.0, 0.0], dtype=np.float32)
        points[1, 31] = points[0, 31] + np.array([0.2, 0.0, 0.0], dtype=np.float32)
        vis = np.ones((2, 32), dtype=bool)
        vis[1, 30] = False
        track_data = {
            "object_points": np.zeros((2, 0, 3), dtype=np.float32),
            "object_colors": np.zeros((2, 0, 3), dtype=np.float32),
            "object_visibilities": np.zeros((2, 0), dtype=bool),
            "controller_points": points,
            "controller_colors": np.ones((2, 32, 3), dtype=np.float32),
            "controller_visibilities": vis,
        }

        filtered = strict.apply_phystwin_motion_filters(track_data)
        final_data = strict.select_final_controller_points(filtered, count=30)

        self.assertEqual(int(np.count_nonzero(filtered["controller_mask"])), 30)
        self.assertEqual(final_data["controller_points"].shape, (2, 30, 3))

    def test_streaming_controller_anchor_selector_reuses_initial_query_ids(self) -> None:
        query_ids = np.arange(100, 108, dtype=np.int64)
        first_points = np.zeros((2, len(query_ids), 3), dtype=np.float32)
        first_points[0, :, 0] = np.linspace(0.00, 0.07, len(query_ids), dtype=np.float32)
        first_points[0, :, 2] = -0.10
        first_points[1] = first_points[0] + np.array([0.01, 0.0, 0.0], dtype=np.float32)
        second_points = np.zeros_like(first_points)
        second_points[0, :, 0] = np.linspace(0.20, 0.27, len(query_ids), dtype=np.float32)[::-1]
        second_points[0, :, 1] = np.linspace(0.00, 0.04, len(query_ids), dtype=np.float32)
        second_points[0, :, 2] = -0.10
        second_points[1] = second_points[0] + np.array([0.01, 0.0, 0.0], dtype=np.float32)

        selector = strict.StreamingControllerAnchorSelector(count=3)
        first = selector.select(_filtered_controller_track(first_points, query_ids=query_ids))
        selected_ids = np.asarray(first["controller_anchor_query_indices"], dtype=np.int64)
        second = selector.select(_filtered_controller_track(second_points, query_ids=query_ids))

        self.assertEqual(second["controller_points"].shape, (2, 3, 3))
        np.testing.assert_array_equal(second["controller_anchor_query_indices"], selected_ids)
        np.testing.assert_array_equal(second["controller_anchor_status"], np.asarray(["direct", "direct", "direct"]))
        for anchor_idx, query_id in enumerate(selected_ids):
            source_idx = int(np.flatnonzero(query_ids == query_id)[0])
            np.testing.assert_allclose(second["controller_points"][:, anchor_idx, :], second_points[:, source_idx, :])

    def test_streaming_controller_anchor_selector_marks_lost_anchor_missing_without_replacement(self) -> None:
        query_ids = np.arange(200, 208, dtype=np.int64)
        first_points = np.zeros((2, len(query_ids), 3), dtype=np.float32)
        first_points[0, :, 0] = np.linspace(0.00, 0.07, len(query_ids), dtype=np.float32)
        first_points[0, :, 2] = -0.10
        first_points[1] = first_points[0] + np.array([0.01, 0.0, 0.0], dtype=np.float32)
        selector = strict.StreamingControllerAnchorSelector(count=3)
        first = selector.select(_filtered_controller_track(first_points, query_ids=query_ids))
        selected_ids = np.asarray(first["controller_anchor_query_indices"], dtype=np.int64)

        second_points = first_points + np.array([0.03, 0.0, 0.0], dtype=np.float32)
        lost_anchor_idx = 1
        lost_query_id = int(selected_ids[lost_anchor_idx])
        lost_source_idx = int(np.flatnonzero(query_ids == lost_query_id)[0])
        controller_mask = np.ones((len(query_ids),), dtype=bool)
        controller_mask[lost_source_idx] = False
        second_points[:, lost_source_idx, :] = 0.0

        second = selector.select(
            _filtered_controller_track(
                second_points,
                query_ids=query_ids,
                controller_mask=controller_mask,
            )
        )

        self.assertEqual(second["controller_points"].shape, (2, 3, 3))
        np.testing.assert_array_equal(second["controller_anchor_query_indices"], selected_ids)
        self.assertEqual(str(second["controller_anchor_status"][lost_anchor_idx]), "missing")
        self.assertEqual(int(second["controller_anchor_active_query_indices"][lost_anchor_idx]), -1)
        self.assertEqual(int(second["controller_fps_indices"][lost_anchor_idx]), -1)
        self.assertEqual(second["controller_mask"].shape, (len(query_ids),))
        self.assertFalse(bool(second["controller_mask"][lost_source_idx]))
        self.assertEqual(second["controller_visibilities"].shape, (2, 3))
        self.assertEqual(second["controller_motions_valid"].shape, (2, 3))
        self.assertFalse(np.any(second["controller_visibilities"][:, lost_anchor_idx]))
        self.assertFalse(np.any(second["controller_motions_valid"][:, lost_anchor_idx]))
        self.assertTrue(np.isfinite(second["controller_points"]).all())
        self.assertTrue(np.all(np.linalg.norm(second["controller_points"][:, lost_anchor_idx, :], axis=1) > 1e-9))
        np.testing.assert_allclose(
            second["controller_points"][:, lost_anchor_idx, :],
            np.repeat(first["controller_points"][-1:, lost_anchor_idx, :], 2, axis=0),
        )

    def test_streaming_object_anchor_selector_reuses_first_chunk_volume_sample_query_ids(self) -> None:
        query_ids = np.arange(300, 306, dtype=np.int64)
        first_points = np.zeros((2, len(query_ids), 3), dtype=np.float32)
        first_points[0, :, 0] = np.array([0.00, 0.003, 0.012, 0.024, 0.036, 0.048], dtype=np.float32)
        first_points[0, :, 2] = -0.10
        first_points[1] = first_points[0] + np.array([0.001, 0.0, 0.0], dtype=np.float32)
        second_points = first_points + np.array([0.05, 0.0, 0.0], dtype=np.float32)
        second_points[:, :, 1] = np.linspace(0.0, 0.03, len(query_ids), dtype=np.float32)

        selector = strict.StreamingObjectAnchorSelector(volume_sample_size=0.01)
        first = selector.select(_filtered_object_track(first_points, query_ids=query_ids))
        selected_ids = np.asarray(first["object_anchor_query_indices"], dtype=np.int64)
        second = selector.select(_filtered_object_track(second_points, query_ids=query_ids))

        self.assertGreaterEqual(len(selected_ids), 4)
        np.testing.assert_array_equal(second["object_anchor_query_indices"], selected_ids)
        self.assertEqual(second["object_points"].shape[1], len(selected_ids))
        for anchor_idx, query_id in enumerate(selected_ids):
            source_idx = int(np.flatnonzero(query_ids == query_id)[0])
            np.testing.assert_allclose(second["object_points"][:, anchor_idx, :], second_points[:, source_idx, :])

    def test_streaming_object_anchor_selector_marks_lost_anchor_missing_without_replacement(self) -> None:
        query_ids = np.arange(400, 408, dtype=np.int64)
        first_points = np.zeros((2, len(query_ids), 3), dtype=np.float32)
        first_points[0, :, 0] = np.array([0.00, 0.003, 0.012, 0.024, 0.036, 0.048, 0.060, 0.072], dtype=np.float32)
        first_points[0, :, 2] = -0.10
        first_points[1] = first_points[0] + np.array([0.001, 0.0, 0.0], dtype=np.float32)

        selector = strict.StreamingObjectAnchorSelector(volume_sample_size=0.01)
        first = selector.select(_filtered_object_track(first_points, query_ids=query_ids))
        selected_ids = np.asarray(first["object_anchor_query_indices"], dtype=np.int64)
        lost_anchor_idx = 1
        lost_query_id = int(selected_ids[lost_anchor_idx])

        keep_mask = query_ids != lost_query_id
        second_query_ids = query_ids[keep_mask]
        second_points = first_points[:, keep_mask, :] + np.array([0.05, 0.02, 0.0], dtype=np.float32)
        second = selector.select(_filtered_object_track(second_points, query_ids=second_query_ids))

        self.assertEqual(second["object_points"].shape[1], len(selected_ids))
        np.testing.assert_array_equal(second["object_anchor_query_indices"], selected_ids)
        self.assertEqual(str(second["object_anchor_status"][lost_anchor_idx]), "missing")
        self.assertEqual(int(second["object_anchor_active_query_indices"][lost_anchor_idx]), -1)
        self.assertEqual(int(second["object_volume_sample_indices"][lost_anchor_idx]), -1)
        self.assertTrue(np.isfinite(second["object_points"]).all())
        self.assertTrue(np.all(np.linalg.norm(second["object_points"][:, lost_anchor_idx, :], axis=1) > 1e-9))
        np.testing.assert_allclose(
            second["object_points"][:, lost_anchor_idx, :],
            np.repeat(first["object_points"][-1:, lost_anchor_idx, :], 2, axis=0),
        )
        self.assertFalse(np.any(second["object_visibilities"][:, lost_anchor_idx]))
        self.assertFalse(np.any(second["object_motions_valid"][:, lost_anchor_idx]))

    def test_motion_filter_matches_reference_neighbor_semantics(self) -> None:
        rng = np.random.default_rng(7)
        points = (rng.normal(size=(5, 37, 3)) * 0.003).astype(np.float32)
        points[1:] += np.array([0.001, 0.0, 0.0], dtype=np.float32)
        points[2, 5] += np.array([0.03, 0.0, 0.0], dtype=np.float32)
        points[3, 13] += np.array([0.0, 0.03, 0.0], dtype=np.float32)
        vis = rng.random((5, 37)) > 0.08
        vis[:, :10] = True

        object_expected, _ = _reference_motion_valid_for_class(
            points,
            vis,
            neighbor_dist=0.01,
            min_neighbors=5,
            motion_similarity_m=0.005,
            once_false_mask=False,
        )
        controller_expected, controller_mask_expected = _reference_motion_valid_for_class(
            points,
            vis,
            neighbor_dist=0.01,
            min_neighbors=5,
            motion_similarity_m=0.005,
            once_false_mask=True,
        )

        track_data = {
            "object_points": points,
            "object_colors": np.ones_like(points),
            "object_visibilities": vis,
            "controller_points": points,
            "controller_colors": np.ones_like(points),
            "controller_visibilities": vis,
        }
        filtered = strict.apply_phystwin_motion_filters(track_data)

        np.testing.assert_array_equal(filtered["object_motions_valid"], object_expected)
        np.testing.assert_array_equal(filtered["controller_motions_valid"], controller_expected)
        np.testing.assert_array_equal(filtered["controller_mask"], controller_mask_expected)

    def test_motion_filter_removes_same_frame_failed_queries_from_later_neighbors(self) -> None:
        frame0 = np.array(
            [
                [0.000, 0.0, 0.0],
                [0.001, 0.0, 0.0],
                [0.002, 0.0, 0.0],
                [0.003, 0.0, 0.0],
                [0.004, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        frame1 = frame0.copy()
        frame1[0] += np.array([0.02, 0.0, 0.0], dtype=np.float32)
        points = np.stack([frame0, frame1], axis=0)
        vis = np.ones((2, 5), dtype=bool)
        track_data = {
            "object_points": points,
            "object_colors": np.ones_like(points),
            "object_visibilities": vis,
            "controller_points": points,
            "controller_colors": np.ones_like(points),
            "controller_visibilities": vis,
        }

        filtered = strict.apply_phystwin_motion_filters(track_data)

        self.assertFalse(np.any(filtered["object_motions_valid"][0]))
        self.assertFalse(np.any(filtered["controller_motions_valid"][0]))
        self.assertFalse(np.any(filtered["controller_mask"]))

    def test_object_volume_sampling_slices_all_object_arrays(self) -> None:
        object_points = np.array(
            [
                [
                    [0.000, 0.0, 0.0],
                    [0.001, 0.0, 0.0],
                    [0.006, 0.0, 0.0],
                ],
                [
                    [0.010, 0.0, 0.0],
                    [0.011, 0.0, 0.0],
                    [0.016, 0.0, 0.0],
                ],
            ],
            dtype=np.float32,
        )
        track_data = {
            "object_points": object_points,
            "object_colors": np.arange(18, dtype=np.float32).reshape(2, 3, 3),
            "object_visibilities": np.ones((2, 3), dtype=bool),
            "object_motions_valid": np.ones((2, 3), dtype=bool),
            "controller_points": np.zeros((2, 30, 3), dtype=np.float32),
        }

        sampled = strict.sample_object_first_frame_volume(track_data, volume_sample_size=0.005)

        self.assertEqual(sampled["object_points"].shape, (2, 2, 3))
        np.testing.assert_allclose(sampled["object_points"][:, 0], object_points[:, 0])
        np.testing.assert_allclose(sampled["object_points"][:, 1], object_points[:, 2])
        self.assertEqual(sampled["object_visibilities"].shape, (2, 2))
        self.assertEqual(sampled["object_motions_valid"].shape, (2, 2))

    def test_finalize_headless_capture_writes_phystwin_like_artifacts(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            capture = Path(tmp) / "capture"
            for name in ("masks", "ffs_depth", "rgb", "query_trajectory"):
                (capture / name).mkdir(parents=True, exist_ok=True)
            metadata = {
                "depth_source": "ffs",
                "intrinsics": {"fx": 1000.0, "fy": 1000.0, "cx": 0.0, "cy": 0.0},
                "camera_to_world_c2w": np.eye(4, dtype=np.float32).tolist(),
            }
            (capture / "metadata.json").write_text(__import__("json").dumps(metadata), encoding="utf-8")
            height, width = 8, 40
            object_mask = np.zeros((height, width), dtype=bool)
            controller_mask = np.zeros((height, width), dtype=bool)
            object_mask[1, :6] = True
            controller_mask[3, :32] = True
            query_points = np.array(
                [[1.0, float(x)] for x in range(6)] + [[3.0, float(x)] for x in range(32)],
                dtype=np.float32,
            )
            frames_rows = []
            for seq in range(2):
                np.save(capture / "ffs_depth" / f"{seq:06d}.npy", np.ones((height, width), dtype=np.float32))
                Image.fromarray(np.full((height, width, 3), 120 + seq, dtype=np.uint8), mode="RGB").save(
                    capture / "rgb" / f"{seq:06d}.png"
                )
                np.savez(
                    capture / "masks" / f"{seq:06d}.npz",
                    object_mask=object_mask,
                    controller_mask=controller_mask,
                    hand_a_mask=controller_mask,
                    hand_b_mask=np.zeros_like(controller_mask),
                )
                np.savez(
                    capture / "query_trajectory" / f"{seq:06d}.npz",
                    seq=np.asarray([seq], dtype=np.int64),
                    query_points_yx=query_points,
                    all_tracks_yx=query_points,
                    all_tracker_visibility=np.ones((len(query_points),), dtype=np.float32),
                )
                frames_rows.append(
                    {
                        "seq": seq,
                        "ffs_depth_path": f"ffs_depth/{seq:06d}.npy",
                        "rgb_path": f"rgb/{seq:06d}.png",
                        "mask_path": f"masks/{seq:06d}.npz",
                        "query_trajectory_path": f"query_trajectory/{seq:06d}.npz",
                    }
                )
            with (capture / "frames.jsonl").open("w", encoding="utf-8") as handle:
                for row in frames_rows:
                    handle.write(__import__("json").dumps(row) + "\n")

            manifest = strict.finalize_headless_capture(capture)
            out = capture / "phystwin_like"

            self.assertEqual(manifest["compatibility_target"], "PhysTwin")
            self.assertEqual(manifest["tracker_backend"], "tapnextpp")
            self.assertEqual(manifest["mask_backend"], "edgetam")
            self.assertEqual(manifest["depth_backend"], "ffs")
            self.assertEqual(manifest["execution_mode"], "workstation_strict")
            self.assertTrue((out / "manifest.json").is_file())
            self.assertTrue((out / "mask" / "processed_masks.pkl").is_file())
            self.assertTrue((out / "tracking" / "0.npz").is_file())
            self.assertTrue((out / "cotracker" / "0.npz").is_file())
            self.assertTrue((out / "track_process_data.pkl").is_file())
            self.assertTrue((out / "final_data.pkl").is_file())
            for name in ("tracking_2d", "track_process_data", "final_data", "final_pcd"):
                mp4 = out / f"{name}.mp4"
                self.assertTrue(mp4.is_file(), mp4)
                self.assertGreater(mp4.stat().st_size, 0, mp4)
            with (out / "track_process_data.pkl").open("rb") as handle:
                track_process = pickle.load(handle)
            with (out / "final_data.pkl").open("rb") as handle:
                final_data = pickle.load(handle)
            self.assertEqual(track_process["controller_points"].shape, (2, 30, 3))
            self.assertEqual(final_data["controller_points"].shape, (2, 30, 3))
            self.assertEqual(final_data["object_points"].shape[0], 2)

    def test_finalize_headless_capture_accepts_depth_color_m_path_and_native_manifest(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            capture = Path(tmp) / "capture"
            for name in ("masks", "depth_color_m", "rgb", "query_trajectory"):
                (capture / name).mkdir(parents=True, exist_ok=True)
            metadata = {
                "depth_backend": "native-realsense",
                "depth_source_internal": "realsense",
                "intrinsics": {"fx": 1000.0, "fy": 1000.0, "cx": 0.0, "cy": 0.0},
                "camera_to_world_c2w": np.eye(4, dtype=np.float32).tolist(),
            }
            (capture / "metadata.json").write_text(__import__("json").dumps(metadata), encoding="utf-8")
            height, width = 8, 40
            object_mask = np.zeros((height, width), dtype=bool)
            controller_mask = np.zeros((height, width), dtype=bool)
            object_mask[1, :6] = True
            controller_mask[3, :32] = True
            query_points = np.array(
                [[1.0, float(x)] for x in range(6)] + [[3.0, float(x)] for x in range(32)],
                dtype=np.float32,
            )
            with (capture / "frames.jsonl").open("w", encoding="utf-8") as handle:
                for seq in range(2):
                    np.save(capture / "depth_color_m" / f"{seq:06d}.npy", np.ones((height, width), dtype=np.float32))
                    Image.fromarray(np.full((height, width, 3), 120 + seq, dtype=np.uint8), mode="RGB").save(
                        capture / "rgb" / f"{seq:06d}.png"
                    )
                    np.savez(
                        capture / "masks" / f"{seq:06d}.npz",
                        object_mask=object_mask,
                        controller_mask=controller_mask,
                        hand_a_mask=controller_mask,
                        hand_b_mask=np.zeros_like(controller_mask),
                    )
                    np.savez(
                        capture / "query_trajectory" / f"{seq:06d}.npz",
                        seq=np.asarray([seq], dtype=np.int64),
                        query_points_yx=query_points,
                        all_tracks_yx=query_points,
                        all_tracker_visibility=np.ones((len(query_points),), dtype=np.float32),
                    )
                    row = {
                        "seq": seq,
                        "depth_color_m_path": f"depth_color_m/{seq:06d}.npy",
                        "rgb_path": f"rgb/{seq:06d}.png",
                        "mask_path": f"masks/{seq:06d}.npz",
                        "query_trajectory_path": f"query_trajectory/{seq:06d}.npz",
                    }
                    handle.write(__import__("json").dumps(row) + "\n")

            manifest = strict.finalize_headless_capture(capture)

            self.assertEqual(manifest["depth_backend"], "native-realsense")
            self.assertEqual(manifest["depth_source_internal"], "realsense")
            self.assertEqual(manifest["frame_count"], 2)
            self.assertTrue((capture / "phystwin_like" / "final_data.pkl").is_file())

    def test_finalize_headless_capture_accepts_prepared_only_rows(self) -> None:
        import json

        with tempfile.TemporaryDirectory() as tmp:
            capture = Path(tmp) / "capture"
            (capture / "prepared_phystwin").mkdir(parents=True, exist_ok=True)
            metadata = {
                "headless_prepared_only": True,
                "depth_backend": "native-realsense",
                "depth_source_internal": "realsense",
            }
            (capture / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            query_points = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            rows = []
            for seq in range(2):
                frame = strict.PreparedPhysTwinFrame(
                    seq=seq,
                    rgb_frame=np.full((3, 4, 3), 120 + seq, dtype=np.uint8),
                    processed_mask_frame={
                        "object": np.ones((3, 4), dtype=bool),
                        "controller": np.zeros((3, 4), dtype=bool),
                    },
                    pcd_points=np.zeros((3, 4, 3), dtype=np.float32),
                    pcd_colors=np.zeros((3, 4, 3), dtype=np.uint8),
                    tracks_yx=query_points,
                    visibility=np.ones((2,), dtype=bool),
                    query_points_yx=query_points,
                )
                path = capture / "prepared_phystwin" / f"{seq:06d}.npz"
                strict.write_prepared_phystwin_frame(path, frame)
                rows.append({"seq": seq, "prepared_phystwin_frame_path": f"prepared_phystwin/{seq:06d}.npz"})
            with (capture / "frames.jsonl").open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")

            manifest = strict.finalize_headless_capture(capture)
            out = capture / "phystwin_like"

            self.assertTrue((out / "manifest.json").is_file())
            self.assertEqual(manifest["headless_prepared_only"], True)
            self.assertEqual(manifest["chunk_materialization_source"], "prepared_phystwin_frame")
            self.assertEqual(manifest["depth_backend"], "native-realsense")
            self.assertEqual(manifest["depth_source_internal"], "realsense")
            self.assertEqual(manifest["frame_count"], 2)
            self.assertEqual(manifest["query_count"], 2)
            self.assertIsNone(manifest["final_data_path"])
            self.assertFalse((out / "final_data.pkl").exists())


if __name__ == "__main__":
    unittest.main()
