from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.harness.render_demo32_headless_capture import render_capture_to_video


class Demo32HeadlessRenderHelperTest(unittest.TestCase):
    def test_render_synthetic_capture_to_video_summary(self) -> None:
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
                query_indices=np.array([0], dtype=np.int64),
            )
            row = {
                "seq": 0,
                "pcd_path": "pcd/000000.npz",
                "ffs_depth_path": "ffs_depth/000000.npy",
                "query_trajectory_path": "query_trajectory/000000.npz",
            }
            (capture_dir / "frames.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

            output = capture_dir / "video.mp4"
            summary = render_capture_to_video(capture_dir=capture_dir, output=output, fps=30.0)

            self.assertTrue(output.is_file())
            self.assertEqual(summary["frame_count"], 1)
            self.assertEqual(summary["saved_pcd_source"], "enhanced_pt_filtered")
            self.assertTrue((capture_dir / "render_summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
