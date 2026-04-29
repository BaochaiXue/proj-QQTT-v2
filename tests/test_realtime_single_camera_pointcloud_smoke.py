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
        self.assertIn("--fps {5,15,30,60}", result.stdout)
        self.assertIn("--profile {848x480,640x480}", result.stdout)
        self.assertIn("--view-mode {camera,orbit}", result.stdout)
        self.assertIn("--render-backend {auto,image,pointcloud}", result.stdout)
        self.assertIn("--image-splat-px IMAGE_SPLAT_PX", result.stdout)
        self.assertIn("--debug", result.stdout)
        self.assertIn("orbit=200000", result.stdout)
        self.assertIn(demo.COORDINATE_FRAME, result.stdout)
        self.assertIn("Use <=0 to disable", result.stdout)
        self.assertIn("far clipping", result.stdout)

    def test_profile_parsing_and_argparse_rejection(self) -> None:
        self.assertEqual(demo.parse_profile("848x480"), (848, 480))
        self.assertEqual(demo.parse_profile("640x480"), (640, 480))
        args = demo.build_parser().parse_args(["--fps", "60"])
        self.assertEqual(args.fps, 60)
        with self.assertRaises(ValueError):
            demo.parse_profile("320x240")
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                demo.build_parser().parse_args(["--profile", "320x240"])

    def test_default_depth_max_disables_far_clipping(self) -> None:
        args = demo.build_parser().parse_args([])
        self.assertEqual(args.fps, 60)
        self.assertEqual(args.depth_max_m, 0.0)
        self.assertEqual(args.view_mode, "camera")
        self.assertEqual(demo.resolve_render_backend(args), "image")
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

    def test_auto_backend_uses_pointcloud_outside_camera_view(self) -> None:
        args = demo.build_parser().parse_args(["--view-mode", "orbit"])
        demo.validate_args(args)
        self.assertEqual(demo.resolve_render_backend(args), "pointcloud")

    def test_orbit_view_defaults_to_200k_points_unless_explicit(self) -> None:
        camera_args = demo.build_parser().parse_args([])
        orbit_args = demo.build_parser().parse_args(["--view-mode", "orbit"])
        uncapped_orbit_args = demo.build_parser().parse_args(["--view-mode", "orbit", "--max-points", "0"])

        demo.validate_args(camera_args)
        demo.validate_args(orbit_args)
        demo.validate_args(uncapped_orbit_args)

        self.assertEqual(demo.resolve_max_points(camera_args), 0)
        self.assertEqual(demo.resolve_max_points(orbit_args), 200000)
        self.assertEqual(demo.resolve_max_points(uncapped_orbit_args), 0)

        demo.apply_view_defaults(orbit_args)
        self.assertEqual(orbit_args.max_points, 200000)

    def test_pointcloud_update_readds_only_when_capacity_is_too_small(self) -> None:
        self.assertTrue(
            demo.pointcloud_update_requires_readd(geometry_added=False, current_capacity=0, point_count=10)
        )
        self.assertFalse(
            demo.pointcloud_update_requires_readd(
                geometry_added=True,
                current_capacity=200000,
                point_count=187949,
            )
        )
        self.assertFalse(
            demo.pointcloud_update_requires_readd(
                geometry_added=True,
                current_capacity=200000,
                point_count=200000,
            )
        )
        self.assertTrue(
            demo.pointcloud_update_requires_readd(
                geometry_added=True,
                current_capacity=187949,
                point_count=200000,
            )
        )

    def test_image_backend_is_rejected_outside_camera_view(self) -> None:
        args = demo.build_parser().parse_args(["--view-mode", "orbit", "--render-backend", "image"])
        with self.assertRaises(ValueError):
            demo.validate_args(args)

    def test_image_backend_preserves_valid_depth_pixels_in_camera_view(self) -> None:
        color_bgr = np.array(
            [
                [[1, 2, 3], [4, 5, 6]],
                [[7, 8, 9], [10, 11, 12]],
            ],
            dtype=np.uint8,
        )
        depth_u16 = np.array([[1000, 0], [2000, 3000]], dtype=np.uint16)
        image_rgb, valid_count = demo.build_camera_view_image(
            color_bgr=color_bgr,
            depth_u16=depth_u16,
            depth_scale_m_per_unit=0.001,
            depth_min_m=0.1,
            depth_max_m=2.5,
            splat_px=0,
        )
        self.assertEqual(valid_count, 2)
        np.testing.assert_array_equal(
            image_rgb,
            np.array(
                [
                    [[3, 2, 1], [0, 0, 0]],
                    [[9, 8, 7], [0, 0, 0]],
                ],
                dtype=np.uint8,
            ),
        )

    def test_image_backend_depth_bounds_preserve_raw_threshold_edges(self) -> None:
        self.assertEqual(
            demo._depth_bounds_to_u16(
                depth_scale_m_per_unit=0.001,
                depth_min_m=0.1,
                depth_max_m=0.102,
            ),
            (100, 101),
        )
        color_bgr = np.array(
            [[[10, 20, 30], [11, 21, 31], [12, 22, 32], [13, 23, 33], [14, 24, 34]]],
            dtype=np.uint8,
        )
        depth_u16 = np.array([[99, 100, 101, 102, 103]], dtype=np.uint16)
        image_rgb, valid_count = demo.build_camera_view_image(
            color_bgr=color_bgr,
            depth_u16=depth_u16,
            depth_scale_m_per_unit=0.001,
            depth_min_m=0.1,
            depth_max_m=0.102,
            splat_px=0,
        )
        self.assertEqual(valid_count, 2)
        np.testing.assert_array_equal(
            image_rgb,
            np.array([[[0, 0, 0], [31, 21, 11], [32, 22, 12], [0, 0, 0], [0, 0, 0]]], dtype=np.uint8),
        )

    def test_image_backend_numpy_fallback_matches_default_path(self) -> None:
        color_bgr = np.array(
            [
                [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
            ],
            dtype=np.uint8,
        )
        depth_u16 = np.array([[0, 100, 600], [999, 1000, 2000]], dtype=np.uint16)
        default_image, default_count = demo.build_camera_view_image(
            color_bgr=color_bgr,
            depth_u16=depth_u16,
            depth_scale_m_per_unit=0.001,
            depth_min_m=0.1,
            depth_max_m=1.0,
            splat_px=0,
        )
        original_cv2 = demo.cv2
        try:
            demo.cv2 = None
            fallback_image, fallback_count = demo.build_camera_view_image(
                color_bgr=color_bgr,
                depth_u16=depth_u16,
                depth_scale_m_per_unit=0.001,
                depth_min_m=0.1,
                depth_max_m=1.0,
                splat_px=0,
            )
        finally:
            demo.cv2 = original_cv2
        self.assertEqual(default_count, fallback_count)
        np.testing.assert_array_equal(default_image, fallback_image)

    def test_image_backend_splat_keeps_original_valid_count(self) -> None:
        color_bgr = np.zeros((3, 3, 3), dtype=np.uint8)
        color_bgr[1, 1] = np.array([10, 20, 30], dtype=np.uint8)
        depth_u16 = np.zeros((3, 3), dtype=np.uint16)
        depth_u16[1, 1] = 1000
        image_rgb, valid_count = demo.build_camera_view_image(
            color_bgr=color_bgr,
            depth_u16=depth_u16,
            depth_scale_m_per_unit=0.001,
            depth_min_m=0.1,
            depth_max_m=0.0,
            splat_px=1,
        )
        self.assertEqual(valid_count, 1)
        self.assertEqual(int(np.count_nonzero(np.any(image_rgb != 0, axis=2))), 9)
        np.testing.assert_array_equal(image_rgb[0, 0], np.array([30, 20, 10], dtype=np.uint8))

    def test_pointcloud_upload_helpers_keep_float32_and_reuse_color_buffer(self) -> None:
        points = np.arange(12, dtype=np.float32).reshape(4, 3)
        same_points = demo.ensure_float32_c_contiguous(points)
        self.assertIs(same_points, points)

        sliced = np.arange(24, dtype=np.float64).reshape(8, 3)[::2]
        converted = demo.ensure_float32_c_contiguous(sliced)
        self.assertEqual(converted.dtype, np.float32)
        self.assertTrue(converted.flags["C_CONTIGUOUS"])

        color_buffer = demo.ColorFloat32Buffer()
        colors = np.array([[0, 127, 255], [255, 0, 64]], dtype=np.uint8)
        colors_float = color_buffer.convert(colors)
        self.assertEqual(colors_float.dtype, np.float32)
        np.testing.assert_allclose(
            colors_float,
            np.array([[0.0, 127.0 / 255.0, 1.0], [1.0, 0.0, 64.0 / 255.0]], dtype=np.float32),
        )
        self.assertIs(color_buffer.convert(np.zeros_like(colors)), colors_float)

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

    def test_backprojection_max_points_preserves_linspace_valid_sampling(self) -> None:
        color_bgr = np.arange(3 * 4 * 3, dtype=np.uint8).reshape(3, 4, 3)
        depth_u16 = np.arange(1, 13, dtype=np.uint16).reshape(3, 4)
        intrinsics = demo.CameraIntrinsics(fx=1.0, fy=1.0, cx=0.0, cy=0.0)
        full_points, full_colors = demo.backproject_aligned_rgbd(
            color_bgr=color_bgr,
            depth_u16=depth_u16,
            intrinsics=intrinsics,
            depth_scale_m_per_unit=0.001,
            depth_min_m=0.001,
            depth_max_m=0.0,
            stride=1,
            max_points=0,
            projection_grid=demo.build_projection_grid(width=4, height=3, stride=1, intrinsics=intrinsics),
        )
        capped_points, capped_colors = demo.backproject_aligned_rgbd(
            color_bgr=color_bgr,
            depth_u16=depth_u16,
            intrinsics=intrinsics,
            depth_scale_m_per_unit=0.001,
            depth_min_m=0.001,
            depth_max_m=0.0,
            stride=1,
            max_points=5,
            projection_grid=demo.build_projection_grid(width=4, height=3, stride=1, intrinsics=intrinsics),
        )
        expected_indices = np.linspace(0, full_points.shape[0] - 1, 5, dtype=np.int64)
        np.testing.assert_allclose(capped_points, full_points[expected_indices])
        np.testing.assert_array_equal(capped_colors, full_colors[expected_indices])

    def test_projection_grid_matches_pixel_grid_backprojection(self) -> None:
        color_bgr = np.array(
            [
                [[1, 2, 3], [4, 5, 6]],
                [[7, 8, 9], [10, 11, 12]],
            ],
            dtype=np.uint8,
        )
        depth_u16 = np.array([[1000, 0], [2000, 3000]], dtype=np.uint16)
        intrinsics = demo.CameraIntrinsics(fx=2.0, fy=4.0, cx=0.5, cy=0.25)
        pixel_points, pixel_colors = demo.backproject_aligned_rgbd(
            color_bgr=color_bgr,
            depth_u16=depth_u16,
            intrinsics=intrinsics,
            depth_scale_m_per_unit=0.001,
            depth_min_m=0.1,
            depth_max_m=5.0,
            stride=1,
            max_points=0,
            pixel_grid=demo.build_pixel_grid(width=2, height=2, stride=1),
        )
        projection_points, projection_colors = demo.backproject_aligned_rgbd(
            color_bgr=color_bgr,
            depth_u16=depth_u16,
            intrinsics=intrinsics,
            depth_scale_m_per_unit=0.001,
            depth_min_m=0.1,
            depth_max_m=5.0,
            stride=1,
            max_points=0,
            projection_grid=demo.build_projection_grid(width=2, height=2, stride=1, intrinsics=intrinsics),
        )
        np.testing.assert_allclose(projection_points, pixel_points)
        np.testing.assert_array_equal(projection_colors, pixel_colors)

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

    def test_depth_to_render_profiler_sum_excludes_camera_wait(self) -> None:
        timing = demo.PipelineTiming(
            wait_ms=33.0,
            align_ms=1.0,
            frame_copy_ms=2.0,
            image_mask_ms=0.5,
            backproject_ms=3.0,
            open3d_convert_ms=4.0,
            open3d_update_ms=5.0,
            receive_to_render_ms=20.0,
        )
        self.assertAlmostEqual(demo.depth_to_render_ms(timing), 15.5)


if __name__ == "__main__":
    unittest.main()
