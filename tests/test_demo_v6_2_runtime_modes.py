from __future__ import annotations

from contextlib import redirect_stderr
import io
import inspect
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from demo_v6_2 import main_cli
from demo_v6_2 import mdp_cli
from demo_v6_2 import mdp_demo_pcd
from demo_v6_2 import mdp_packets
from demo_v6_2.main_subprocess import _contract, build_main_data_processing_command
from demo_v6_2.mdp_packets import PairedBuildResult
from demo_v6_2.mdp_pcd_depth import backproject_masked_rgbd_profiled
from demo_v6_2.phystwin_strict_product import (
    apply_radius_outlier_to_mask_frame,
    prepare_phystwin_frame,
)


class RuntimeInputModeTests(unittest.TestCase):
    def test_orchestrator_accepts_only_fake_live_and_live(self) -> None:
        parser = main_cli.build_parser()
        supported_modes = {"fake-live", "live"}
        input_source_action = next(
            action for action in parser._actions if action.dest == "input_source"
        )
        self.assertEqual(set(input_source_action.choices or ()), supported_modes)
        self.assertIn(parser.parse_args([]).input_source, supported_modes)

        for input_source in supported_modes:
            args = parser.parse_args(["--input-source", input_source])
            self.assertEqual(args.input_source, input_source)
            self.assertFalse(hasattr(args, "source_headless_capture"))
        self.assertNotIn("source_headless_capture", _contract(parser.parse_args([])))

        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                parser.parse_args(["--source-headless-capture", "/tmp/capture"])
        self.assertEqual(error.exception.code, 2)

    def test_camera_runtime_accepts_only_fake_live_and_live(self) -> None:
        parser = mdp_cli.build_parser()
        supported_modes = {"fake-live", "live"}
        input_source_action = next(
            action for action in parser._actions if action.dest == "input_source"
        )
        self.assertEqual(set(input_source_action.choices or ()), supported_modes)
        self.assertIn(parser.parse_args([]).input_source, supported_modes)

        for input_source in supported_modes:
            args = parser.parse_args(["--input-source", input_source])
            self.assertEqual(args.input_source, input_source)

        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                parser.parse_args(["--input-source", "recording"])
        self.assertEqual(error.exception.code, 2)

        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                parser.parse_args(["--recording-case", "/tmp/case"])
        self.assertEqual(error.exception.code, 2)

        self.assertTrue(mdp_cli._is_replay_input_source("fake-live"))
        self.assertFalse(mdp_cli._is_replay_input_source("live"))
        self.assertFalse(mdp_cli._is_replay_input_source("recording"))

    def test_masked_pcd_requires_tracker(self) -> None:
        parser = mdp_cli.build_parser()
        defaults = parser.parse_args([])
        self.assertEqual(defaults.pcd_mode, "masked")
        self.assertEqual(defaults.tracker_backend, "tapnextpp")

        no_tracker = parser.parse_args(["--tracker-backend", "none"])
        with self.assertRaisesRegex(
            ValueError,
            "--pcd-mode masked requires --tracker-backend tapnextpp",
        ):
            mdp_cli.validate_args(no_tracker)

        no_track_mode = parser.parse_args(["--track-mode", "none"])
        with self.assertRaisesRegex(
            ValueError,
            "--pcd-mode masked requires an enabled --track-mode",
        ):
            mdp_cli.validate_args(no_track_mode)

    def test_legacy_latest_frame_pcd_worker_is_absent(self) -> None:
        self.assertFalse(hasattr(mdp_demo_pcd._PcdMixin, "_pcd_worker"))

    def test_runtime_pcd_filter_surface_is_absent(self) -> None:
        parser = mdp_cli.build_parser()
        options = set(parser._option_string_actions)
        removed_options = {
            "--enable-pcd-filter",
            "--pcd-filter-mode",
            "--pcd-filter-preset",
            "--object-filter",
            "--controller-filter",
            "--object-filter-cap",
            "--controller-filter-cap",
            "--object-filter-keep-components",
            "--controller-filter-keep-components",
            "--object-filter-voxel-m",
            "--controller-filter-voxel-m",
            "--filter-every-n",
            "--filter-max-age-frames",
            "--voxel-density-min-points",
            "--filter-radius-m",
            "--filter-nb-points",
            "--enhanced-component-voxel-size-m",
            "--enhanced-keep-near-main-gap-m",
            "--tracker-retire-filtered-markers",
        }
        self.assertFalse(removed_options & options)
        self.assertFalse(hasattr(mdp_demo_pcd._PcdMixin, "_make_filter_input"))
        self.assertFalse(hasattr(mdp_demo_pcd._PcdMixin, "_filter_pcd_input"))
        self.assertFalse(hasattr(mdp_demo_pcd._PcdMixin, "_apply_single_pcd_filter"))
        self.assertFalse(hasattr(mdp_packets, "PcdFilterTelemetry"))

        orchestrator_args = main_cli.build_parser().parse_args([])
        command = build_main_data_processing_command(
            orchestrator_args,
            capture_dir=Path("/tmp/demo-v6-2-capture"),
            profile_json=Path("/tmp/demo-v6-2-profile.json"),
            chunk_frame_count=5,
        )
        self.assertFalse(removed_options & set(command))

    def test_runtime_backprojection_keeps_every_valid_masked_point(self) -> None:
        height, width = 250, 241
        color_bgr = np.zeros((height, width, 3), dtype=np.uint8)
        depth_m = np.ones((height, width), dtype=np.float32)
        mask = np.ones((height, width), dtype=bool)
        ray_x = np.zeros_like(depth_m)
        ray_y = np.zeros_like(depth_m)

        points, colors, pixels_yx, _timing = backproject_masked_rgbd_profiled(
            color_bgr=color_bgr,
            depth_m=depth_m,
            mask=mask,
            ray_x=ray_x,
            ray_y=ray_y,
            depth_min_m=0.2,
            depth_max_m=1.5,
            color_mode="rgb",
            class_rgb=(0, 0, 0),
            return_yx=True,
        )

        expected = height * width
        self.assertGreater(expected, 60_000)
        self.assertEqual(len(points), expected)
        self.assertEqual(len(colors), expected)
        self.assertEqual(len(pixels_yx), expected)
        self.assertNotIn(
            "max_points",
            inspect.signature(backproject_masked_rgbd_profiled).parameters,
        )

    def test_strict_product_keeps_the_only_pt_mask_filter(self) -> None:
        prepare_parameters = inspect.signature(prepare_phystwin_frame).parameters
        self.assertNotIn("mask_radius_outlier_filter", prepare_parameters)
        self.assertNotIn("mask_radius_outlier_radius_m", prepare_parameters)
        self.assertNotIn("mask_radius_outlier_nb_points", prepare_parameters)

        points_grid = np.zeros((1, 41, 3), dtype=np.float32)
        points_grid[0, :40, 2] = 1.0
        points_grid[0, 40] = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        object_mask = np.ones((1, 41), dtype=bool)
        controller_mask = np.zeros((1, 41), dtype=bool)

        cleaned = apply_radius_outlier_to_mask_frame(
            {"object": object_mask, "controller": controller_mask},
            points_grid,
        )

        self.assertTrue(np.all(cleaned["object"][0, :40]))
        self.assertFalse(cleaned["object"][0, 40])


class PairedBuildResultTests(unittest.TestCase):
    def test_rejects_mixed_sequence_results(self) -> None:
        for component in ("pcd", "mask", "tracker"):
            sequences = {"pcd": 7, "mask": 7, "tracker": 7}
            sequences[component] = 8
            with self.subTest(component=component):
                with self.assertRaisesRegex(
                    ValueError,
                    "strict same-seq build result mismatch",
                ):
                    PairedBuildResult(
                        seq=7,
                        pcd_result=SimpleNamespace(
                            packet=SimpleNamespace(seq=sequences["pcd"]),
                            mask_packet=SimpleNamespace(seq=sequences["mask"]),
                        ),
                        tracker_packet=SimpleNamespace(seq=sequences["tracker"]),
                    )

    def test_legacy_render_packet_is_absent(self) -> None:
        self.assertFalse(hasattr(mdp_packets, "PairedRenderPacket"))


if __name__ == "__main__":
    unittest.main()
