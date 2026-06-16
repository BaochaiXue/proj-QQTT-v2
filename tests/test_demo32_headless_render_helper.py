from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from scripts.harness.diagnostics.demo.render_demo32_headless_capture import (
    TRACKING_BACKGROUND_MASK_RGB,
    TRACKING_BACKGROUND_MASK_TARGET_UNION,
    _apply_tracking_background_mask,
    _read_target_union_mask,
    render_capture_to_video,
)


class Demo32HeadlessRenderHelperTest(unittest.TestCase):
    def test_apply_tracking_background_mask_blacks_pixels_outside_union(self) -> None:
        image = np.full((4, 5, 3), 80, dtype=np.uint8)
        image[1, 2] = np.array([10, 20, 30], dtype=np.uint8)
        mask = np.zeros((4, 5), dtype=bool)
        mask[1, 2] = True
        mask[3, 4] = True

        kept = _apply_tracking_background_mask(image, mask)

        self.assertEqual(kept, 2)
        np.testing.assert_array_equal(image[1, 2], np.array([10, 20, 30], dtype=np.uint8))
        np.testing.assert_array_equal(image[3, 4], np.array([80, 80, 80], dtype=np.uint8))
        np.testing.assert_array_equal(image[0, 0], np.array([0, 0, 0], dtype=np.uint8))

    def test_read_target_union_mask_uses_object_or_controller_mask(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            capture_dir = Path(tmp) / "capture"
            (capture_dir / "masks").mkdir(parents=True)
            controller_mask = np.zeros((4, 5), dtype=bool)
            object_mask = np.zeros((4, 5), dtype=bool)
            controller_mask[1, 2] = True
            object_mask[3, 4] = True
            np.savez(
                capture_dir / "masks" / "000000.npz",
                controller_mask=controller_mask,
                object_mask=object_mask,
            )
            frame = {"seq": 0, "mask_path": "masks/000000.npz"}

            union = _read_target_union_mask(capture_dir=capture_dir, frame=frame, width=5, height=4)

            expected = np.logical_or(controller_mask, object_mask)
            np.testing.assert_array_equal(union, expected)

    def test_render_synthetic_capture_to_video_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            capture_dir = Path(tmp) / "capture"
            (capture_dir / "pcd").mkdir(parents=True)
            (capture_dir / "ffs_depth").mkdir()
            (capture_dir / "rgb").mkdir()
            (capture_dir / "query_trajectory").mkdir()
            metadata = {
                "width": 32,
                "height": 24,
                "saved_pcd_source": "enhanced_pt_filtered",
                "intrinsics": {"fx": 20.0, "fy": 20.0, "cx": 16.0, "cy": 12.0},
            }
            (capture_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            np.savez(
                capture_dir / "pcd" / "000000.npz",
                controller_xyz_m=np.array([[0.0, 0.0, 0.5]], dtype=np.float32),
                controller_rgb_u8=np.array([[255, 0, 0]], dtype=np.uint8),
                object_xyz_m=np.array([[0.05, 0.0, 0.6]], dtype=np.float32),
                object_rgb_u8=np.array([[0, 255, 0]], dtype=np.uint8),
            )
            np.save(capture_dir / "ffs_depth" / "000000.npy", np.ones((24, 32), dtype=np.float32))
            Image.fromarray(np.full((24, 32, 3), 64, dtype=np.uint8)).save(capture_dir / "rgb" / "000000.png")
            np.savez(
                capture_dir / "query_trajectory" / "000000.npz",
                marker_xyz_m=np.array([[0.0, 0.0, 0.5], [0.05, 0.0, 0.6]], dtype=np.float32),
                marker_rgb_u8=np.array([[255, 32, 32], [32, 255, 255]], dtype=np.uint8),
                query_rgb_u8=np.array([[255, 32, 32], [32, 255, 255]], dtype=np.uint8),
                tracks_yx=np.array([[12.0, 16.0], [12.0, 18.0]], dtype=np.float32),
                visibility=np.ones((2,), dtype=np.float32),
                query_indices=np.array([0, 1], dtype=np.int64),
                query_is_object=np.array([False, True], dtype=bool),
                query_is_controller=np.array([True, False], dtype=bool),
                query_controller_instance_id=np.array([1, 0], dtype=np.int64),
                query_count=np.array([2], dtype=np.int64),
            )
            (capture_dir / "masks").mkdir()
            controller_mask = np.zeros((24, 32), dtype=bool)
            object_mask = np.zeros((24, 32), dtype=bool)
            controller_mask[11:14, 15:18] = True
            object_mask[11:14, 17:20] = True
            np.savez(
                capture_dir / "masks" / "000000.npz",
                controller_mask=controller_mask,
                object_mask=object_mask,
            )
            row = {
                "seq": 0,
                "pcd_path": "pcd/000000.npz",
                "ffs_depth_path": "ffs_depth/000000.npy",
                "rgb_path": "rgb/000000.png",
                "query_trajectory_path": "query_trajectory/000000.npz",
                "mask_path": "masks/000000.npz",
            }
            (capture_dir / "frames.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

            output = capture_dir / "video.mp4"
            summary = render_capture_to_video(capture_dir=capture_dir, output=output, fps=30.0)

            self.assertTrue(output.is_file())
            self.assertEqual(summary["frame_count"], 1)
            self.assertEqual(summary["saved_pcd_source"], "enhanced_pt_filtered")
            self.assertEqual(summary["query_overlay"], "phystwin_rgb_current_points_only")
            self.assertEqual(summary["query_color_mode"], "phystwin_rainbow_identity")
            self.assertEqual(summary["query_match_policy"], "exact_same_seq_only")
            self.assertEqual(summary["missing_query_frames"], 0)
            self.assertEqual(summary["rendered_counts"][0]["controller_points"], 0)
            self.assertEqual(summary["rendered_counts"][0]["object_points"], 0)
            self.assertEqual(summary["rendered_counts"][0]["query_controller_points"], 1)
            self.assertEqual(summary["rendered_counts"][0]["query_object_points"], 1)
            self.assertEqual(summary["rendered_counts"][0]["query_hand_a_points"], 1)
            self.assertEqual(summary["rendered_counts"][0]["query_hand_b_points"], 0)
            self.assertEqual(summary["query_count_totals"]["hand_a"], 1)
            self.assertEqual(summary["query_count_totals"]["hand_b"], 0)
            self.assertEqual(summary["query_count_totals"]["object"], 1)
            self.assertTrue((capture_dir / "video.render_summary.json").is_file())
            self.assertEqual(summary["tracking_background_mask"], TRACKING_BACKGROUND_MASK_TARGET_UNION)
            self.assertEqual(summary["tracking_background_mask_source"], "object_mask|controller_mask")
            self.assertEqual(
                summary["rendered_counts"][0]["tracking_background_mask_pixels"],
                int(np.count_nonzero(np.logical_or(controller_mask, object_mask))),
            )
            self.assertEqual(
                summary["tracking_background_mask_pixel_total"],
                int(np.count_nonzero(np.logical_or(controller_mask, object_mask))),
            )

    def test_render_does_not_fallback_to_previous_query_trajectory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            capture_dir = Path(tmp) / "capture"
            (capture_dir / "pcd").mkdir(parents=True)
            (capture_dir / "ffs_depth").mkdir()
            (capture_dir / "rgb").mkdir()
            (capture_dir / "query_trajectory").mkdir()
            (capture_dir / "masks").mkdir()
            metadata = {
                "width": 32,
                "height": 24,
                "saved_pcd_source": "enhanced_pt_filtered",
                "intrinsics": {"fx": 20.0, "fy": 20.0, "cx": 16.0, "cy": 12.0},
            }
            (capture_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            for seq in (0, 1):
                np.savez(
                    capture_dir / "pcd" / f"{seq:06d}.npz",
                    controller_xyz_m=np.array([[0.0, 0.0, 0.5]], dtype=np.float32),
                    controller_rgb_u8=np.array([[255, 0, 0]], dtype=np.uint8),
                    object_xyz_m=np.array([[0.05, 0.0, 0.6]], dtype=np.float32),
                    object_rgb_u8=np.array([[0, 255, 0]], dtype=np.uint8),
                )
                np.save(capture_dir / "ffs_depth" / f"{seq:06d}.npy", np.ones((24, 32), dtype=np.float32))
                Image.fromarray(np.full((24, 32, 3), 32 + seq, dtype=np.uint8)).save(
                    capture_dir / "rgb" / f"{seq:06d}.png"
                )
                controller_mask = np.zeros((24, 32), dtype=bool)
                object_mask = np.zeros((24, 32), dtype=bool)
                controller_mask[10:13, 14:18] = True
                object_mask[10:13, 18:21] = True
                np.savez(
                    capture_dir / "masks" / f"{seq:06d}.npz",
                    controller_mask=controller_mask,
                    object_mask=object_mask,
                )
            np.savez(
                capture_dir / "query_trajectory" / "000000.npz",
                marker_xyz_m=np.array([[0.0, 0.0, 0.5]], dtype=np.float32),
                marker_rgb_u8=np.array([[255, 32, 32]], dtype=np.uint8),
                tracks_yx=np.array([[12.0, 16.0]], dtype=np.float32),
                visibility=np.ones((1,), dtype=np.float32),
                query_indices=np.array([0], dtype=np.int64),
                query_is_object=np.array([False], dtype=bool),
                query_is_controller=np.array([True], dtype=bool),
                query_count=np.array([1], dtype=np.int64),
            )
            rows = [
                {
                    "seq": 0,
                    "pcd_path": "pcd/000000.npz",
                    "ffs_depth_path": "ffs_depth/000000.npy",
                    "rgb_path": "rgb/000000.png",
                    "query_trajectory_path": "query_trajectory/000000.npz",
                    "mask_path": "masks/000000.npz",
                },
                {
                    "seq": 1,
                    "pcd_path": "pcd/000001.npz",
                    "ffs_depth_path": "ffs_depth/000001.npy",
                    "rgb_path": "rgb/000001.png",
                    "query_trajectory_path": "query_trajectory/000001.npz",
                    "mask_path": "masks/000001.npz",
                },
            ]
            (capture_dir / "frames.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )

            summary = render_capture_to_video(capture_dir=capture_dir, output=capture_dir / "video.mp4", fps=30.0)

            self.assertEqual(summary["missing_query_frames"], 1)
            self.assertEqual(summary["rendered_counts"][0]["query_trajectory_exact"], 1)
            self.assertEqual(summary["rendered_counts"][1]["query_trajectory_exact"], 0)
            self.assertEqual(summary["rendered_counts"][1]["query_points"], 0)

    def test_tracking_rgb_background_mask_does_not_require_mask_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            capture_dir = Path(tmp) / "capture"
            (capture_dir / "pcd").mkdir(parents=True)
            (capture_dir / "ffs_depth").mkdir()
            (capture_dir / "rgb").mkdir()
            (capture_dir / "query_trajectory").mkdir()
            metadata = {
                "width": 16,
                "height": 12,
                "saved_pcd_source": "enhanced_pt_filtered",
                "intrinsics": {"fx": 10.0, "fy": 10.0, "cx": 8.0, "cy": 6.0},
            }
            (capture_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            np.savez(
                capture_dir / "pcd" / "000000.npz",
                controller_xyz_m=np.empty((0, 3), dtype=np.float32),
                controller_rgb_u8=np.empty((0, 3), dtype=np.uint8),
                object_xyz_m=np.empty((0, 3), dtype=np.float32),
                object_rgb_u8=np.empty((0, 3), dtype=np.uint8),
            )
            np.save(capture_dir / "ffs_depth" / "000000.npy", np.ones((12, 16), dtype=np.float32))
            Image.fromarray(np.full((12, 16, 3), 90, dtype=np.uint8)).save(capture_dir / "rgb" / "000000.png")
            np.savez(
                capture_dir / "query_trajectory" / "000000.npz",
                tracks_yx=np.array([[6.0, 8.0]], dtype=np.float32),
                visibility=np.ones((1,), dtype=np.float32),
                query_indices=np.array([0], dtype=np.int64),
                query_is_object=np.array([True], dtype=bool),
                query_is_controller=np.array([False], dtype=bool),
                query_count=np.array([1], dtype=np.int64),
            )
            row = {
                "seq": 0,
                "pcd_path": "pcd/000000.npz",
                "ffs_depth_path": "ffs_depth/000000.npy",
                "rgb_path": "rgb/000000.png",
                "query_trajectory_path": "query_trajectory/000000.npz",
            }
            (capture_dir / "frames.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

            summary = render_capture_to_video(
                capture_dir=capture_dir,
                output=capture_dir / "video.mp4",
                fps=30.0,
                tracking_background_mask=TRACKING_BACKGROUND_MASK_RGB,
            )

            self.assertEqual(summary["tracking_background_mask"], TRACKING_BACKGROUND_MASK_RGB)
            self.assertEqual(summary["tracking_background_mask_source"], "full_rgb")
            self.assertEqual(summary["tracking_background_mask_pixel_total"], 0)
            self.assertEqual(summary["rendered_counts"][0]["tracking_background_mask_pixels"], 0)
            self.assertEqual(summary["rendered_counts"][0]["query_points"], 1)

    def test_pcd_visual_mode_suppresses_query_overlay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            capture_dir = Path(tmp) / "capture"
            (capture_dir / "pcd").mkdir(parents=True)
            (capture_dir / "ffs_depth").mkdir()
            (capture_dir / "query_trajectory").mkdir()
            metadata = {
                "width": 32,
                "height": 24,
                "saved_pcd_source": "enhanced_pt_filtered",
                "intrinsics": {"fx": 20.0, "fy": 20.0, "cx": 16.0, "cy": 12.0},
            }
            (capture_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            np.savez(
                capture_dir / "pcd" / "000000.npz",
                controller_xyz_m=np.array([[0.0, 0.0, 0.5]], dtype=np.float32),
                controller_rgb_u8=np.array([[255, 0, 0]], dtype=np.uint8),
                object_xyz_m=np.array([[0.05, 0.0, 0.6]], dtype=np.float32),
                object_rgb_u8=np.array([[0, 255, 0]], dtype=np.uint8),
            )
            np.save(capture_dir / "ffs_depth" / "000000.npy", np.ones((24, 32), dtype=np.float32))
            np.savez(
                capture_dir / "query_trajectory" / "000000.npz",
                marker_xyz_m=np.array([[0.0, 0.0, 0.5]], dtype=np.float32),
                marker_rgb_u8=np.array([[255, 32, 32]], dtype=np.uint8),
                query_indices=np.array([0], dtype=np.int64),
                query_is_object=np.array([False], dtype=bool),
                query_is_controller=np.array([True], dtype=bool),
            )
            row = {
                "seq": 0,
                "pcd_path": "pcd/000000.npz",
                "ffs_depth_path": "ffs_depth/000000.npy",
                "query_trajectory_path": "query_trajectory/000000.npz",
            }
            (capture_dir / "frames.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

            summary = render_capture_to_video(
                capture_dir=capture_dir,
                output=capture_dir / "video.mp4",
                fps=30.0,
                demo_visual_mode="pcd",
            )

            self.assertEqual(summary["demo_visual_mode"], "pcd")
            self.assertEqual(summary["query_overlay"], "none")
            self.assertEqual(summary["rendered_counts"][0]["query_points"], 0)


if __name__ == "__main__":
    unittest.main()
