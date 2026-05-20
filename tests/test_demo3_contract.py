from __future__ import annotations

import argparse
import contextlib
import io
from pathlib import Path
import pickle
import tempfile
import unittest

from qqtt.demo import demo3_runtime
from qqtt.env.camera.calibration_metadata import build_calibration_metadata, write_calibration_metadata


class Demo3RuntimeContractTest(unittest.TestCase):
    def _parse(self, argv: list[str]):
        parser = demo3_runtime.build_arg_parser()
        args = parser.parse_args(argv)
        return demo3_runtime.apply_preset_defaults(args, explicit_options=set(argv))

    def test_dry_run_exactly_three_cameras_passes(self) -> None:
        args = self._parse(["--dry-run", "--camera-ids", "0,1,2"])
        demo3_runtime.validate_args(args)
        contract = demo3_runtime.build_contract(args)

        self.assertEqual(contract["demo"], "demo3")
        self.assertTrue(contract["requires_three_realsense"])
        self.assertEqual(contract["num_cameras"], 3)
        self.assertEqual(contract["depth_source"], "realsense")
        self.assertFalse(contract["uses_ffs"])
        self.assertEqual(contract["mask_source"], "hf_edgetam")
        self.assertTrue(contract["edgetam_batch_vision_encoder"])
        self.assertEqual(contract["edgetam_live_session_keep_frames"], 64)
        self.assertTrue(contract["edgetam_live_session_pruning"])
        self.assertEqual(contract["input_source"], "live_realsense")
        self.assertFalse(contract["offline_mode_available"])
        self.assertFalse(contract["offline_tracking_available"])
        self.assertEqual(contract["init_mode"], "sam31_first_frame")
        self.assertEqual(contract["mask_propagation"], "hf_edgetam_online")
        self.assertEqual(contract["semantic_mode"], "exp")
        self.assertEqual(contract["controller_prompt"], "towel")
        self.assertEqual(contract["tracking_mask_scope"], "object_controller_union")
        self.assertEqual(contract["tracking_query_mode"], "phystwin_dense")
        self.assertEqual(contract["tracking_query_count_requested"], "auto")
        self.assertEqual(contract["tracking_query_count_rule"], "min(union_mask_pixels, 5000)")
        self.assertEqual(contract["tracking_sampling"], "torch_randperm_seed_plus_camera_idx")
        self.assertEqual(contract["cotracker_seed"], 42)
        self.assertEqual(contract["overlay_max_points_per_camera"], 30)
        self.assertEqual(contract["overlay_display_scope"], "controller")
        self.assertEqual(contract["overlay_display_classification"], "first_frame_mask_membership")
        self.assertTrue(contract["phystwin_dense_compatible"])
        self.assertEqual(contract["cotracker_backend"], "cotracker3_online")
        self.assertTrue(contract["cotracker_async"])
        self.assertEqual(contract["cotracker_update_mode"], "batch")
        self.assertEqual(contract["cotracker_batch_size_target"], 3)
        self.assertFalse(contract["cotracker_batch_fallback_enabled"])
        self.assertTrue(contract["render_latest_wins"])
        self.assertFalse(contract["debug_fusion"]["color_by_camera"])
        self.assertFalse(contract["gpu_sampling"]["enabled"])

    def test_mode_demo_uses_hand_controller(self) -> None:
        args = self._parse(["--dry-run", "--camera-ids", "0,1,2", "--mode", "demo"])
        contract = demo3_runtime.build_contract(args)

        self.assertEqual(contract["semantic_mode"], "demo")
        self.assertEqual(contract["shared_experiment_mode"], "demo-mode")
        self.assertEqual(contract["controller_prompt"], "hand")
        self.assertEqual(contract["tracking_controller_label"], "hand")

    def test_track_mode_is_not_public_cli(self) -> None:
        parser = demo3_runtime.build_arg_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["--dry-run", "--camera-ids", "0,1,2", "--track-mode", "object-only"])
        with self.assertRaises(SystemExit):
            parser.parse_args(["--dry-run", "--camera-ids", "0,1,2", "--track-mode", "controller-only"])

    def test_offline_inputs_are_not_public_cli(self) -> None:
        parser = demo3_runtime.build_arg_parser()
        for flag, value in (("--input-video", "foo.mp4"), ("--case-root", "case"), ("--saved-masks", "masks")):
            with self.subTest(flag=flag):
                with self.assertRaises(SystemExit):
                    parser.parse_args(["--dry-run", "--camera-ids", "0,1,2", flag, value])

    def test_one_or_two_cameras_fail_fast(self) -> None:
        for camera_ids in ("0", "0,1"):
            with self.subTest(camera_ids=camera_ids):
                args = self._parse(["--dry-run", "--camera-ids", camera_ids])
                with self.assertRaisesRegex(ValueError, "Demo 3 requires exactly three RealSense cameras"):
                    demo3_runtime.validate_args(args)

    def test_depth_source_ffs_fails_fast(self) -> None:
        args = self._parse(["--dry-run", "--camera-ids", "0,1,2", "--depth-source", "ffs"])
        with self.assertRaisesRegex(ValueError, "Demo 3 does not support FFS"):
            demo3_runtime.validate_args(args)

    def test_main_dry_run_prints_acceptance_contract(self) -> None:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = demo3_runtime.main(["--dry-run", "--camera-ids", "0,1,2"])

        self.assertEqual(exit_code, 0)
        output = stdout.getvalue()
        self.assertIn("demo = demo3", output)
        self.assertIn("requires_three_realsense = true", output)
        self.assertIn("num_cameras = 3", output)
        self.assertIn("depth_source = realsense", output)
        self.assertIn("uses_ffs = false", output)
        self.assertIn("mask_source = hf_edgetam", output)
        self.assertIn("edgetam_batch_vision_encoder = true", output)
        self.assertIn("edgetam_live_session_keep_frames = 64", output)
        self.assertIn("edgetam_live_session_pruning = true", output)
        self.assertIn("input_source = live_realsense", output)
        self.assertIn("offline_mode_available = false", output)
        self.assertIn("init_mode = sam31_first_frame", output)
        self.assertIn("mask_propagation = hf_edgetam_online", output)
        self.assertIn("semantic_mode = exp", output)
        self.assertIn("tracking_mask_scope = object_controller_union", output)
        self.assertIn("tracking_query_mode = phystwin_dense", output)
        self.assertIn("tracking_query_count_requested = auto", output)
        self.assertIn("tracking_sampling = torch_randperm_seed_plus_camera_idx", output)
        self.assertIn("overlay_display_scope = controller", output)
        self.assertIn("phystwin_dense_compatible = true", output)
        self.assertIn("cotracker_backend = cotracker3_online", output)
        self.assertIn("cotracker_async = true", output)
        self.assertIn("cotracker_update_mode = batch", output)
        self.assertIn("render_latest_wins = true", output)

    def test_mask_only_preset_disables_cotracker(self) -> None:
        args = self._parse(["--preset", "demo3-realsense-mask-only", "--dry-run", "--camera-ids", "0,1,2"])
        contract = demo3_runtime.build_contract(args)
        self.assertFalse(contract["cotracker_enabled"])
        self.assertFalse(contract["uses_ffs"])

    def test_live_realsense_validation_checks_connected_count_and_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            calibrate_path = Path(tmp_dir) / "calibrate.pkl"
            with calibrate_path.open("wb") as handle:
                pickle.dump([[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]] * 3, handle)
            write_calibration_metadata(
                calibrate_path,
                build_calibration_metadata(
                    serial_numbers=["s0", "s1", "s2"],
                    WH=(848, 480),
                    fps=30,
                    transform_count=3,
                ),
            )
            args = self._parse(["--camera-ids", "0,1,2", "--calibrate-path", str(calibrate_path)])

            validation = demo3_runtime.validate_live_realsense_contract(
                args,
                connected_serials_provider=lambda: ["s0", "s1", "s2"],
            )
            self.assertEqual(validation["active_serials"], ["s0", "s1", "s2"])

            with self.assertRaisesRegex(RuntimeError, "exactly three connected RealSense"):
                demo3_runtime.validate_live_realsense_contract(
                    args,
                    connected_serials_provider=lambda: ["s0", "s1"],
                )

    def test_non_dry_run_invokes_shared_runtime_adapter_with_realsense_depth(self) -> None:
        class _Stats:
            def __init__(self, fps: float) -> None:
                self.fps = fps

        class _FakeSharedRuntime:
            last_args = None

            def __init__(self, args) -> None:
                type(self).last_args = args
                self._summary = {
                    "final": {
                        "render_fps": 12.0,
                        "capture_group_fps": 30.0,
                        "fusion_fps": 28.0,
                    }
                }
                self.edge_stats = {0: _Stats(24.0), 1: _Stats(25.0), 2: _Stats(26.0)}
                self.render_stats = _Stats(12.0)
                self.capture_group_stats = _Stats(30.0)
                self.fusion_stats = _Stats(28.0)
                self.demo3_overlay_ms_samples = [0.10, 0.20, 0.30]

            def run(self) -> int:
                return 0

        class _FakeSharedModule:
            @staticmethod
            def build_arg_parser():
                parser = argparse.ArgumentParser()
                parser.add_argument("--preset")
                parser.add_argument("--demo-version-override")
                parser.add_argument("--demo-display-name-override")
                parser.add_argument("--profile")
                parser.add_argument("--fps", type=int)
                parser.add_argument("--fusion-target-fps", type=float)
                parser.add_argument("--capture-group-target-fps", type=float)
                parser.add_argument("--camera-ids", type=demo3_runtime.parse_camera_ids)
                parser.add_argument("--calibrate-path")
                parser.add_argument("--serials", nargs="*")
                parser.add_argument("--calibration-reference-serials", nargs="*")
                parser.add_argument("--depth-source")
                parser.add_argument("--edgetam-batch-vision-encoder", action="store_true")
                parser.add_argument("--edgetam-live-session-keep-frames", type=int)
                parser.add_argument("--render-mode")
                parser.add_argument("--point-size", type=float)
                parser.add_argument("--render-every-n", type=int)
                parser.add_argument("--render-backend")
                parser.add_argument("--render-layer-mode")
                parser.add_argument("--render-copy-mode")
                parser.add_argument("--pcd-color-mode")
                parser.add_argument("--no-render-async-latest-only", action="store_true")
                parser.add_argument("--track-mode")
                parser.add_argument("--object-prompt")
                parser.add_argument("--controller-prompt")
                parser.add_argument("--experiment-mode")
                parser.add_argument("--duration-s", type=float)
                parser.add_argument("--output-root")
                parser.add_argument("--profile-pipeline", action="store_true")
                parser.add_argument("--profile-visualization", action="store_true")
                parser.add_argument("--render-micro-profile", action="store_true")
                parser.add_argument("--object-point-control")
                parser.add_argument("--object-volume-voxel-m", type=float)
                parser.add_argument("--object-volume-origin")
                parser.add_argument("--object-volume-adaptive", action=argparse.BooleanOptionalAction)
                parser.add_argument("--object-volume-min-voxel-m", type=float)
                parser.add_argument("--object-volume-max-voxel-m", type=float)
                parser.add_argument("--object-volume-target-ms", type=float)
                parser.add_argument("--object-volume-emergency-max-points", type=int)
                parser.add_argument("--object-volume-points-per-voxel", type=int)
                parser.add_argument("--debug-color-by-camera", action="store_true")
                parser.add_argument("--debug-save-per-camera-pcd", action="store_true")
                parser.add_argument("--debug-save-mask-overlays", action="store_true")
                parser.add_argument("--debug-identity-c2w", action="store_true")
                parser.add_argument("--debug-invert-c2w", action="store_true")
                parser.add_argument("--debug-only-camera-idx", type=int)
                parser.add_argument("--debug-fusion-max-saved-groups", type=int)
                parser.add_argument("--gpu-sampling", action="store_true")
                parser.add_argument("--gpu-sampling-interval-s", type=float)
                parser.add_argument("--gpu-sampling-backend")
                parser.add_argument("--gpu-sampling-device-index", type=int)
                parser.add_argument("--gpu-sampling-device-indexes", type=demo3_runtime.parse_gpu_sampling_device_indexes)
                parser.add_argument("--tracking-backend")
                parser.add_argument("--tracking-source")
                parser.add_argument("--tracking-num-points", type=int)
                parser.add_argument("--tracking-overlay-max-points", type=int)
                parser.add_argument("--tracking-trail-len", type=int)
                parser.add_argument("--tracking-depth-source")
                parser.add_argument("--profile-json-output")
                parser.add_argument("--debug", action="store_true")
                parser.add_argument("--show-tracking-overlay", action="store_true")
                return parser

            @staticmethod
            def apply_preset_defaults(args, *, explicit_options=None):
                _ = explicit_options
                return args

            @staticmethod
            def _explicit_cli_options(argv):
                return {item for item in argv if str(item).startswith("--")}

        with tempfile.TemporaryDirectory() as tmp_dir:
            calibrate_path = Path(tmp_dir) / "calibrate.pkl"
            with calibrate_path.open("wb") as handle:
                pickle.dump([[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]] * 3, handle)
            write_calibration_metadata(
                calibrate_path,
                build_calibration_metadata(
                    serial_numbers=["s0", "s1", "s2"],
                    WH=(848, 480),
                    fps=30,
                    transform_count=3,
                ),
            )
            args = self._parse(
                [
                    "--camera-ids",
                    "0,1,2",
                    "--calibrate-path",
                    str(calibrate_path),
                    "--duration-s",
                    "0.01",
                    "--debug-color-by-camera",
                    "--debug-only-camera-idx",
                    "1",
                    "--gpu-sampling",
                    "--gpu-sampling-device-indexes",
                    "0,1",
                    "--point-size",
                    "1.5",
                    "--object-volume-points-per-voxel",
                    "3",
                ]
            )
            runtime = demo3_runtime.Demo3Runtime(
                args,
                shared_runtime_module=_FakeSharedModule,
                shared_runtime_cls=_FakeSharedRuntime,
                connected_serials_provider=lambda: ["s0", "s1", "s2"],
            )

            profile = runtime.run()

            self.assertEqual(profile["summary"]["exit_code"], 0)
            self.assertEqual(profile["summary"]["depth_source"], "realsense")
            self.assertFalse(profile["summary"]["uses_ffs"])
            self.assertEqual(profile["summary"]["rendered_fps"], 12.0)
            self.assertEqual(profile["summary"]["capture_group_fps"], 30.0)
            self.assertEqual(profile["summary"]["fusion_fps"], 28.0)
            self.assertGreater(profile["summary"]["edgetam_mask_fps"], 0.0)
            self.assertTrue(profile["summary"]["edgetam_batch_vision_encoder"])
            self.assertEqual(_FakeSharedRuntime.last_args.depth_source, "realsense")
            self.assertTrue(_FakeSharedRuntime.last_args.edgetam_batch_vision_encoder)
            self.assertEqual(_FakeSharedRuntime.last_args.edgetam_live_session_keep_frames, 64)
            self.assertEqual(_FakeSharedRuntime.last_args.demo_version_override, "demo3")
            self.assertEqual(_FakeSharedRuntime.last_args.demo_display_name_override, "Demo 3")
            self.assertTrue(_FakeSharedRuntime.last_args.debug_color_by_camera)
            self.assertEqual(_FakeSharedRuntime.last_args.debug_only_camera_idx, 1)
            self.assertTrue(_FakeSharedRuntime.last_args.gpu_sampling)
            self.assertEqual(_FakeSharedRuntime.last_args.gpu_sampling_device_indexes, (0, 1))
            self.assertEqual(_FakeSharedRuntime.last_args.point_size, 1.5)
            self.assertEqual(_FakeSharedRuntime.last_args.object_point_control, "phystwin-volume")
            self.assertEqual(_FakeSharedRuntime.last_args.object_volume_voxel_m, 0.005)
            self.assertEqual(_FakeSharedRuntime.last_args.object_volume_points_per_voxel, 3)
            self.assertEqual(_FakeSharedRuntime.last_args.track_mode, "controller-object")
            self.assertEqual(_FakeSharedRuntime.last_args.experiment_mode, "controller-object-exp")
            self.assertEqual(_FakeSharedRuntime.last_args.controller_prompt, "towel")
            self.assertEqual(_FakeSharedRuntime.last_args.tracking_backend, "none")
            self.assertEqual(_FakeSharedRuntime.last_args.tracking_source, "cached")
            self.assertFalse(_FakeSharedRuntime.last_args.show_tracking_overlay)
            self.assertTrue(profile["contract"]["cotracker_enabled"])


if __name__ == "__main__":
    unittest.main()
