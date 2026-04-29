from __future__ import annotations

from dataclasses import dataclass
import contextlib
import io
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

from scripts.harness import realtime_single_camera_pointcloud as demo


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DummyPacket:
    seq: int


class RealtimeSingleCameraPointCloudSmokeTest(unittest.TestCase):
    def test_help_exposes_supported_capture_rates_and_profiles(self) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/harness/realtime_single_camera_pointcloud.py", "--help"],
            cwd=ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertIn("--fps {5,15,30}", result.stdout)
        self.assertIn("--profile {848x480,640x480}", result.stdout)
        self.assertIn("--view-mode {camera,orbit}", result.stdout)
        self.assertIn(demo.COORDINATE_FRAME, result.stdout)
        self.assertIn("Use <=0 to disable", result.stdout)
        self.assertIn("far clipping", result.stdout)

    def test_profile_parsing_and_argparse_rejection(self) -> None:
        self.assertEqual(demo.parse_profile("848x480"), (848, 480))
        self.assertEqual(demo.parse_profile("640x480"), (640, 480))
        with self.assertRaises(ValueError):
            demo.parse_profile("320x240")
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                demo.build_parser().parse_args(["--profile", "320x240"])

    def test_default_depth_max_disables_far_clipping(self) -> None:
        args = demo.build_parser().parse_args([])
        self.assertEqual(args.depth_max_m, 0.0)
        self.assertEqual(args.view_mode, "camera")
        demo.validate_args(args)
        color_bgr = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
        depth_u16 = np.array([[1000, 6000]], dtype=np.uint16)
        points, _ = demo.backproject_aligned_rgbd(
            color_bgr=color_bgr,
            depth_u16=depth_u16,
            intrinsics=demo.CameraIntrinsics(fx=1.0, fy=1.0, cx=0.0, cy=0.0),
            depth_scale_m_per_unit=0.001,
            depth_min_m=0.1,
            depth_max_m=args.depth_max_m,
            stride=1,
            max_points=0,
        )
        self.assertEqual(points.shape[0], 2)
        np.testing.assert_allclose(points[:, 2], np.array([1.0, 6.0], dtype=np.float32))

    def test_synthetic_backprojection_returns_expected_xyz_and_rgb(self) -> None:
        color_bgr = np.array(
            [
                [[1, 2, 3], [4, 5, 6]],
                [[7, 8, 9], [10, 11, 12]],
            ],
            dtype=np.uint8,
        )
        depth_u16 = np.array([[1000, 0], [2000, 3000]], dtype=np.uint16)
        points, colors = demo.backproject_aligned_rgbd(
            color_bgr=color_bgr,
            depth_u16=depth_u16,
            intrinsics=demo.CameraIntrinsics(fx=1.0, fy=1.0, cx=0.0, cy=0.0),
            depth_scale_m_per_unit=0.001,
            depth_min_m=0.1,
            depth_max_m=5.0,
            stride=1,
            max_points=0,
            pixel_grid=demo.build_pixel_grid(width=2, height=2, stride=1),
        )
        np.testing.assert_allclose(
            points,
            np.array([[0.0, 0.0, 1.0], [0.0, 2.0, 2.0], [3.0, 3.0, 3.0]], dtype=np.float32),
        )
        np.testing.assert_array_equal(colors, np.array([[3, 2, 1], [9, 8, 7], [12, 11, 10]], dtype=np.uint8))

    def test_latest_slot_drops_stale_packets(self) -> None:
        slot: demo.LatestSlot[DummyPacket] = demo.LatestSlot()
        self.assertEqual(slot.dropped_count, 0)
        slot.put(DummyPacket(seq=1))
        slot.put(DummyPacket(seq=2))
        self.assertEqual(slot.dropped_count, 1)
        packet = slot.get_latest_after(-1)
        self.assertIsNotNone(packet)
        self.assertEqual(packet.seq, 2)
        self.assertIsNone(slot.get_latest_after(2))
        slot.put(DummyPacket(seq=3))
        self.assertEqual(slot.dropped_count, 1)
        self.assertEqual(slot.get_latest_after(2).seq, 3)  # type: ignore[union-attr]

    def test_render_stats_are_deterministic(self) -> None:
        stats = demo.RenderStats(window_s=1.0)
        stats.record_render(render_time_s=0.0, latency_ms=10.0)
        stats.record_render(render_time_s=0.5, latency_ms=20.0)
        stats.record_render(render_time_s=1.0, latency_ms=30.0)
        self.assertAlmostEqual(stats.render_fps, 2.0)
        self.assertAlmostEqual(stats.latest_latency_ms, 30.0)
        self.assertAlmostEqual(stats.mean_latency_ms, 20.0)


if __name__ == "__main__":
    unittest.main()
