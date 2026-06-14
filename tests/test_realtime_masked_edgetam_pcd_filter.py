from __future__ import annotations

import contextlib
import io
import unittest

import numpy as np

from qqtt.demo import realtime_masked_edgetam_pcd as masked_demo


class RealtimeMaskedEdgeTamPcdFilterTest(unittest.TestCase):
    def test_filter_keep_component_cli_defaults(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            with self.assertRaises(SystemExit):
                masked_demo.build_parser().parse_args(["--help"])
        help_text = stdout.getvalue()
        self.assertIn("--object-filter-keep-components OBJECT_FILTER_KEEP_COMPONENTS", help_text)
        self.assertIn("--controller-filter-keep-components CONTROLLER_FILTER_KEEP_COMPONENTS", help_text)
        self.assertIn("--filter-max-age-frames FILTER_MAX_AGE_FRAMES", help_text)

        args = masked_demo.build_parser().parse_args([])
        self.assertEqual(args.object_filter, "enhanced-pt")
        self.assertEqual(args.controller_filter, "enhanced-pt")
        self.assertEqual(args.object_filter_keep_components, 1)
        self.assertEqual(args.controller_filter_keep_components, 2)
        self.assertEqual(args.filter_max_age_frames, 3)
        self.assertEqual(args.view_mode, "orbit")

        bad_object = masked_demo.build_parser().parse_args(["--object-filter-keep-components", "0"])
        with self.assertRaisesRegex(ValueError, "object-filter-keep-components"):
            masked_demo.validate_args(bad_object)

        bad_controller = masked_demo.build_parser().parse_args(["--controller-filter-keep-components", "0"])
        with self.assertRaisesRegex(ValueError, "controller-filter-keep-components"):
            masked_demo.validate_args(bad_controller)

        bad_age = masked_demo.build_parser().parse_args(["--filter-max-age-frames", "-1"])
        with self.assertRaisesRegex(ValueError, "filter-max-age-frames"):
            masked_demo.validate_args(bad_age)

    def test_enhanced_controller_keeps_two_components_by_default(self) -> None:
        args = masked_demo.build_parser().parse_args(
            [
                "--enable-pcd-filter",
                "--pcd-filter-mode",
                "sync",
                "--object-filter",
                "enhanced-pt",
                "--controller-filter",
                "enhanced-pt",
                "--object-filter-cap",
                "0",
                "--controller-filter-cap",
                "0",
                "--filter-nb-points",
                "1",
                "--filter-radius-m",
                "0.05",
                "--enhanced-component-voxel-size-m",
                "0.05",
                "--enhanced-keep-near-main-gap-m",
                "0",
            ]
        )
        demo_instance = masked_demo.RealtimeMaskedEdgeTamPcdDemo(args)
        left_hand = np.array(
            [[float(col) * 0.005, float(row) * 0.005, 0.50] for row in range(5) for col in range(8)],
            dtype=np.float32,
        )
        right_hand = left_hand + np.array([0.50, 0.0, 0.0], dtype=np.float32)
        two_components = np.vstack([left_hand, right_hand])
        colors = (np.arange(two_components.shape[0] * 3, dtype=np.uint16).reshape(-1, 3) % 255).astype(np.uint8)

        output = demo_instance._filter_pcd_input(
            demo_instance._make_filter_input(
                seq=5,
                object_xyz=two_components,
                object_colors=colors,
                controller_xyz=two_components,
                controller_colors=colors,
            )
        )

        self.assertEqual(output.object_xyz.shape[0], 40)
        self.assertEqual(output.controller_xyz.shape[0], 80)
        self.assertEqual(output.stats["object"]["keep_components"], 1)
        self.assertEqual(output.stats["controller"]["keep_components"], 2)

    def test_filter_falls_back_to_capped_points_when_radius_filter_outputs_empty(self) -> None:
        args = masked_demo.build_parser().parse_args(
            [
                "--enable-pcd-filter",
                "--pcd-filter-mode",
                "sync",
                "--object-filter",
                "pt-filter",
                "--controller-filter",
                "pt-filter",
                "--object-filter-cap",
                "0",
                "--controller-filter-cap",
                "0",
                "--filter-nb-points",
                "2",
                "--filter-radius-m",
                "0.001",
            ]
        )
        demo_instance = masked_demo.RealtimeMaskedEdgeTamPcdDemo(args)
        points = np.array(
            [
                [0.00, 0.00, 0.50],
                [0.10, 0.00, 0.50],
                [0.20, 0.00, 0.50],
                [0.30, 0.00, 0.50],
            ],
            dtype=np.float32,
        )
        colors = np.full((points.shape[0], 3), 127, dtype=np.uint8)

        output = demo_instance._filter_pcd_input(
            demo_instance._make_filter_input(
                seq=9,
                object_xyz=points,
                object_colors=colors,
                controller_xyz=points,
                controller_colors=colors,
            )
        )

        self.assertEqual(output.object_xyz.shape[0], points.shape[0])
        self.assertEqual(output.controller_xyz.shape[0], points.shape[0])
        self.assertEqual(output.stats["object"]["filter_output_points"], 0)
        self.assertEqual(output.stats["controller"]["filter_output_points"], 0)
        self.assertTrue(output.stats["object"]["fallback_to_capped"])
        self.assertTrue(output.stats["controller"]["fallback_to_capped"])
        self.assertEqual(output.stats["object"]["fallback_reason"], "empty_filter_output")
        self.assertEqual(output.stats["controller"]["fallback_reason"], "empty_filter_output")

    def test_controller_filter_falls_back_when_retain_ratio_is_too_low(self) -> None:
        args = masked_demo.build_parser().parse_args(
            [
                "--enable-pcd-filter",
                "--pcd-filter-mode",
                "sync",
                "--object-filter",
                "pt-filter",
                "--controller-filter",
                "pt-filter",
                "--object-filter-cap",
                "0",
                "--controller-filter-cap",
                "0",
                "--filter-nb-points",
                "2",
                "--filter-radius-m",
                "0.01",
            ]
        )
        demo_instance = masked_demo.RealtimeMaskedEdgeTamPcdDemo(args)
        cluster = np.array(
            [
                [0.000, 0.000, 0.50],
                [0.004, 0.000, 0.50],
                [0.008, 0.000, 0.50],
            ],
            dtype=np.float32,
        )
        sparse = np.array(
            [[0.20 + float(idx) * 0.04, 0.0, 0.50] for idx in range(7)],
            dtype=np.float32,
        )
        points = np.vstack([cluster, sparse])
        colors = np.full((points.shape[0], 3), 191, dtype=np.uint8)

        output = demo_instance._filter_pcd_input(
            demo_instance._make_filter_input(
                seq=11,
                object_xyz=points,
                object_colors=colors,
                controller_xyz=points,
                controller_colors=colors,
            )
        )

        self.assertEqual(output.object_xyz.shape[0], 3)
        self.assertFalse(output.stats["object"]["fallback_to_capped"])
        self.assertEqual(output.controller_xyz.shape[0], points.shape[0])
        self.assertEqual(output.stats["controller"]["filter_output_points"], 3)
        self.assertAlmostEqual(output.stats["controller"]["filter_retain_ratio"], 0.3)
        self.assertTrue(output.stats["controller"]["fallback_to_capped"])
        self.assertEqual(output.stats["controller"]["fallback_reason"], "low_filter_retain_ratio")

    def test_async_filter_output_must_be_recent_enough_to_render(self) -> None:
        args = masked_demo.build_parser().parse_args(["--filter-max-age-frames", "3"])
        demo_instance = masked_demo.RealtimeMaskedEdgeTamPcdDemo(args)
        empty_points = np.empty((0, 3), dtype=np.float32)
        empty_colors = np.empty((0, 3), dtype=np.uint8)
        output = masked_demo.FilterOutput(
            seq=7,
            object_xyz=empty_points,
            object_rgb=empty_colors,
            controller_xyz=empty_points,
            controller_rgb=empty_colors,
            filter_ms=0.0,
            created_perf_s=0.0,
            output_perf_s=0.0,
        )

        self.assertTrue(demo_instance._filter_output_is_fresh(packet_seq=10, output=output))
        self.assertFalse(demo_instance._filter_output_is_fresh(packet_seq=11, output=output))


if __name__ == "__main__":
    unittest.main()
