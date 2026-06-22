from __future__ import annotations

import pickle
from pathlib import Path
import tempfile
import unittest

import numpy as np

from qqtt.demo import phystwin_strict_product as strict


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


if __name__ == "__main__":
    unittest.main()
