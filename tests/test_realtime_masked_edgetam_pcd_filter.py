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

        args = masked_demo.build_parser().parse_args([])
        self.assertEqual(args.object_filter_keep_components, 1)
        self.assertEqual(args.controller_filter_keep_components, 2)

        bad_object = masked_demo.build_parser().parse_args(["--object-filter-keep-components", "0"])
        with self.assertRaisesRegex(ValueError, "object-filter-keep-components"):
            masked_demo.validate_args(bad_object)

        bad_controller = masked_demo.build_parser().parse_args(["--controller-filter-keep-components", "0"])
        with self.assertRaisesRegex(ValueError, "controller-filter-keep-components"):
            masked_demo.validate_args(bad_controller)

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


if __name__ == "__main__":
    unittest.main()
