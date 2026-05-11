from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from qqtt.tracking.base import TrackingResult
from qqtt.tracking.io import save_cotracker_like_npz
from tests.visualization_test_utils import make_visualization_case


class Demo3TrackingHarnessSmokeTest(unittest.TestCase):
    def test_fake_backend_benchmark_writes_three_camera_outputs(self) -> None:
        from scripts.harness.experiments.run_demo3_tracking_backend_benchmark import RESULT_COLUMNS, parse_args, run_benchmark

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            case_root = root / "case"
            make_visualization_case(case_root, frame_num=2)
            args = parse_args(
                [
                    "--case-root",
                    str(case_root),
                    "--output-root",
                    str(root / "out"),
                    "--backends",
                    "fake",
                    "--cameras",
                    "0,1,2",
                    "--num-query-points",
                    "3",
                    "--frames",
                    "2",
                    "--write-phystwin-cotracker-dir",
                ]
            )

            summary = run_benchmark(args)
            output_dir = Path(summary["output_dir"])

            self.assertTrue((output_dir / "results.csv").is_file())
            self.assertTrue((output_dir / "summary.json").is_file())
            self.assertTrue((output_dir / "fake" / "points_3" / "cam0.npz").is_file())
            self.assertTrue((output_dir / "fake" / "points_3" / "benchmark_cam0.json").is_file())
            self.assertTrue((output_dir / "cotracker" / "0.npz").is_file())
            header = (output_dir / "results.csv").read_text(encoding="utf-8").splitlines()[0].split(",")
            self.assertEqual(header, RESULT_COLUMNS)

    def test_overlay_export_writes_ply_frame_video_and_stats(self) -> None:
        from scripts.harness.visualize_demo3_tracking_pcd_overlay import parse_args, run_overlay_export

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            case_root = root / "case"
            make_visualization_case(case_root, frame_num=2)
            tracking_root = root / "tracking"
            result = TrackingResult(
                tracks_yx=np.array([[[2.0, 2.0], [20.0, 20.0]]], dtype=np.float32),
                visibility=np.array([[1.0, 1.0]], dtype=np.float32),
                backend="unit_backend",
                camera_idx=0,
            )
            save_cotracker_like_npz(result, tracking_root / "cotracker_like" / "0.npz", camera_idx=0)
            args = parse_args(
                [
                    "--case-root",
                    str(case_root),
                    "--tracking-root",
                    str(tracking_root),
                    "--output",
                    str(root / "overlay"),
                    "--cameras",
                    "0",
                    "--frame-idx",
                    "0",
                    "--viewpoints",
                    "cam0,cam1,cam2",
                ]
            )

            summary = run_overlay_export(args)

            self.assertTrue((root / "overlay" / "lifted_anchors.ply").is_file())
            self.assertTrue((root / "overlay" / "lifted_trails.ply").is_file())
            self.assertTrue((root / "overlay" / "overlay_stats.json").is_file())
            self.assertTrue((root / "overlay" / "frames" / "frame_000000.png").is_file())
            self.assertTrue((root / "overlay" / "overlay_3view.mp4").is_file())
            self.assertGreaterEqual(summary["anchor_point_count"], 1)


if __name__ == "__main__":
    unittest.main()
