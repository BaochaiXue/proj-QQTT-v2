from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from scripts.harness.render_demo32_headless_capture import render_capture_to_video


class Demo32HeadlessRenderHelperTest(unittest.TestCase):
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
                query_count=np.array([2], dtype=np.int64),
            )
            row = {
                "seq": 0,
                "pcd_path": "pcd/000000.npz",
                "ffs_depth_path": "ffs_depth/000000.npy",
                "rgb_path": "rgb/000000.png",
                "query_trajectory_path": "query_trajectory/000000.npz",
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
            self.assertTrue((capture_dir / "video.render_summary.json").is_file())

    def test_render_does_not_fallback_to_previous_query_trajectory(self) -> None:
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
                },
                {
                    "seq": 1,
                    "pcd_path": "pcd/000001.npz",
                    "ffs_depth_path": "ffs_depth/000001.npy",
                    "rgb_path": "rgb/000001.png",
                    "query_trajectory_path": "query_trajectory/000001.npz",
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
