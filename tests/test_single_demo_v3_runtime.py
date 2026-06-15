from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from qqtt.demo import single_demo_v3_runtime as runtime


def _explicit(argv: list[str]) -> set[str]:
    return {item.split("=", 1)[0] for item in argv if item.startswith("--")}


def _option_value(argv: list[str], option: str) -> str:
    return argv[argv.index(option) + 1]


LEGACY_MULTI_CAMERA_FIELDS = {
    "requires_three_realsense",
    "num_realsense_cameras",
    "camera_ids",
    "camera_sync_required",
    "multi_camera_world_fusion",
    "multi_camera_calibration_required",
    "calibrate_path",
    "calibrate_pkl_required",
    "calibrate_pkl_loaded",
    "calibration_transform_count",
    "point_tracker_enabled",
    "point_tracker_live_stage",
    "tracking_backend_execution_mode",
    "tracking_backend_batch_size",
    "tracking_backend_model_instances_expected",
    "required_cuda_devices",
    "require_two_cuda",
    "cross_gpu_cuda_tensor_transfer",
    "strict_source_three_camera_bundle",
    "removed_three_camera_work",
}


class SingleDemoV3RuntimeTest(unittest.TestCase):
    def _parse(self, version: str, argv: list[str]):
        parser = runtime.build_arg_parser(demo_version=version)
        args = parser.parse_args(argv)
        return runtime.apply_preset_defaults(args, explicit_options=_explicit(argv))

    def assert_legacy_fields_removed(self, contract: dict[str, object]) -> None:
        for field in LEGACY_MULTI_CAMERA_FIELDS:
            self.assertNotIn(field, contract)

    def test_demo3_contract_is_single_camera_only(self) -> None:
        args = self._parse(runtime.DEMO_VERSION_3, ["--dry-run"])
        runtime.validate_args(args)
        contract = runtime.build_contract(args)

        self.assertEqual(contract["demo"], "single-demo3")
        self.assertEqual(contract["camera_count"], 1)
        self.assertEqual(contract["input_source"], "live_realsense_single_camera")
        self.assertEqual(contract["input_source_mode"], "live")
        self.assertIsNone(contract["recording_case"])
        self.assertIsNone(contract["replay_fps"])
        self.assertEqual(contract["live_delegate_module"], "qqtt.demo.realtime_masked_edgetam_pcd")
        self.assertEqual(contract["depth_source"], "realsense")
        self.assertEqual(contract["depth_pipeline"], "realsense_native")
        self.assertFalse(contract["uses_ffs"])
        self.assertIsNone(contract["ffs_trt_batch_size"])
        self.assertEqual(contract["tracker_backend"], "tapnextpp")
        self.assertEqual(contract["tracker_backend_family"], "tapnext")
        self.assertEqual(contract["tracker_query_count"], 4096)
        self.assertEqual(contract["tracker_display_scope"], "union")
        self.assertEqual(contract["tracker_visualization_mode"], "all_tracks_3d_lift")
        self.assertEqual(contract["pcd_max_points"], 60000)
        self.assertEqual(contract["pcd_stride"], 1)
        self.assertEqual(contract["render_max_points_per_layer"], 5000)
        self.assertFalse(contract["pcd_filter_enabled"])
        self.assertEqual(contract["point_size"], 2.0)
        self.assertEqual(contract["object_prompt"], "stuffed animal")
        self.assertEqual(contract["controller_prompt"], "towel")
        self.assert_legacy_fields_removed(contract)

    def test_ffs_contract_is_single_camera_ffs_batch_one(self) -> None:
        args = self._parse(runtime.DEMO_VERSION_3_2, ["--dry-run"])
        contract = runtime.build_contract(args)

        self.assertEqual(contract["demo"], "single-demo3.2")
        self.assertEqual(contract["camera_count"], 1)
        self.assertEqual(contract["depth_source"], "ffs")
        self.assertEqual(contract["depth_pipeline"], "ffs_tensorrt_batch1")
        self.assertTrue(contract["uses_ffs"])
        self.assertEqual(contract["ffs_trt_batch_size"], 1)
        self.assertEqual(contract["tracker_backend"], "tapnextpp")
        self.assert_legacy_fields_removed(contract)

        delegate = runtime.build_live_delegate_argv(args, active_serial="s0")
        self.assertIn("--serial", delegate)
        self.assertIn("s0", delegate)
        self.assertIn("--depth-source", delegate)
        self.assertIn("ffs", delegate)
        self.assertIn("--controller-prompt", delegate)
        self.assertIn("towel", delegate)
        self.assertIn("--tracker-backend", delegate)
        self.assertIn("tapnextpp", delegate)

    def test_old_multi_camera_options_are_not_public_cli(self) -> None:
        parser = runtime.build_arg_parser(demo_version=runtime.DEMO_VERSION_3)
        removed_args = (
            "--camera-ids",
            "--serials",
            "--calibrate-path",
            "--depth-source",
            "--live-delegate",
        )
        for option in removed_args:
            with self.subTest(option=option):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        parser.parse_args(["--dry-run", option, "0"])

    def test_ffs_options_are_exposed_only_for_ffs_versions(self) -> None:
        realsense_parser = runtime.build_arg_parser(demo_version=runtime.DEMO_VERSION_3_1)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                realsense_parser.parse_args(["--ffs-repo", "/tmp/ffs"])

        ffs_parser = runtime.build_arg_parser(demo_version=runtime.DEMO_VERSION_3_2)
        args = runtime.apply_preset_defaults(
            ffs_parser.parse_args(["--ffs-repo", "/tmp/ffs"]),
            explicit_options={"--ffs-repo"},
        )
        self.assertEqual(str(args.ffs_repo), "/tmp/ffs")
        self.assertEqual(args.depth_source, "ffs")

    def test_dry_run_main_prints_reduced_single_camera_contract_and_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            profile = Path(tmp_dir) / "profile.json"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = runtime.main(
                    [
                        "--dry-run",
                        "--profile-json-output",
                        str(profile),
                    ],
                    demo_version=runtime.DEMO_VERSION_3_2,
                )

            self.assertEqual(code, 0)
            output = stdout.getvalue()
            self.assertIn("camera_count = 1", output)
            self.assertIn("depth_source = ffs", output)
            self.assertIn("ffs_trt_batch_size = 1", output)
            self.assertNotIn("requires_three_realsense", output)
            self.assertNotIn("multi_camera_world_fusion", output)

            payload = json.loads(profile.read_text(encoding="utf-8"))
            self.assertEqual(payload["contract"]["camera_count"], 1)
            self.assert_legacy_fields_removed(payload["contract"])

    def test_live_validation_uses_one_connected_serial(self) -> None:
        args = self._parse(runtime.DEMO_VERSION_3, [])
        validation = runtime.validate_live_contract(
            args,
            connected_serials_provider=lambda: ["s0", "s1"],
        )

        self.assertEqual(validation["active_serial"], "s0")
        self.assertNotIn("active_serials", validation)

    def test_recording_contract_uses_rgbd_input_source_and_metadata_fps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            case_dir = Path(tmp_dir) / "case"
            case_dir.mkdir()
            (case_dir / "metadata.json").write_text(json.dumps({"fps": 30}), encoding="utf-8")
            args = self._parse(
                runtime.DEMO_VERSION_3_1,
                [
                    "--input-source",
                    "recording",
                    "--recording-case",
                    str(case_dir),
                    "--serial",
                    "s0",
                    "--mode",
                    "demo",
                ],
            )

            runtime.validate_args(args)
            contract = runtime.build_contract(args)

            self.assertEqual(contract["input_source"], "recording_single_camera")
            self.assertEqual(contract["input_source_mode"], "recording")
            self.assertEqual(contract["recording_case"], str(case_dir))
            self.assertEqual(contract["replay_fps"], 30.0)
            self.assertEqual(contract["replay_fps_source"], "metadata")
            self.assertIsNone(contract["serial"])
            self.assertEqual(contract["controller_prompt"], "human hand")
            self.assertEqual(contract["track_mode"], "controller-object")
            self.assertEqual(contract["render_mode"], "pointcloud")
            self.assertEqual(contract["view_mode"], "orbit")
            self.assertEqual(contract["edgetam_live_session_keep_frames"], 64)
            self.assertEqual(contract["tracker_backend"], "tapnextpp")
            self.assertEqual(contract["tracker_device"], "cuda:1")
            self.assertEqual(contract["tracker_query_source"], "object_controller_union_mask")
            self.assertEqual(contract["tracker_visualization_mode"], "all_tracks_3d_lift")

            delegate = runtime.build_live_delegate_argv(args)
            self.assertIn("--input-source", delegate)
            self.assertIn("recording", delegate)
            self.assertIn("--recording-case", delegate)
            self.assertIn(str(case_dir), delegate)
            self.assertIn("--render-mode", delegate)
            self.assertIn("pointcloud", delegate)
            self.assertIn("--track-mode", delegate)
            self.assertIn("controller-object", delegate)
            self.assertIn("--tracker-backend", delegate)
            self.assertIn("tapnextpp", delegate)
            self.assertIn("--tracker-display-scope", delegate)
            self.assertIn("union", delegate)
            self.assertEqual(_option_value(delegate, "--pcd-max-points"), "60000")
            self.assertEqual(_option_value(delegate, "--pcd-stride"), "1")
            self.assertEqual(_option_value(delegate, "--render-max-points-per-layer"), "5000")
            self.assertNotIn("--serial", delegate)

    def test_fake_live_uses_default_case_and_allows_ffs_versions(self) -> None:
        for version, expected_depth in (
            (runtime.DEMO_VERSION_3, "realsense"),
            (runtime.DEMO_VERSION_3_1, "realsense"),
            (runtime.DEMO_VERSION_3_2, "ffs"),
            (runtime.DEMO_VERSION_3_3, "ffs"),
        ):
            with self.subTest(version=version):
                args = self._parse(version, ["--input-source", "fake-live"])
                runtime.validate_args(args)
                contract = runtime.build_contract(args)
                delegate = runtime.build_live_delegate_argv(args)

                self.assertEqual(contract["input_source"], "fake_live_recorded_single_camera")
                self.assertEqual(contract["input_source_mode"], "fake-live")
                self.assertEqual(contract["recording_case"], str(runtime.DEFAULT_FAKE_LIVE_CASE))
                self.assertIsNone(contract["serial"])
                self.assertEqual(contract["semantic_mode"], "demo")
                self.assertEqual(contract["controller_prompt"], "human hand")
                self.assertEqual(contract["depth_source"], expected_depth)
                self.assertIn("--input-source", delegate)
                self.assertIn("fake-live", delegate)
                self.assertIn("--recording-case", delegate)
                self.assertIn(str(runtime.DEFAULT_FAKE_LIVE_CASE), delegate)
                self.assertEqual(_option_value(delegate, "--controller-prompt"), "human hand")
                self.assertNotIn("--serial", delegate)

    def test_fake_live_case_alias_is_forwarded_to_delegate(self) -> None:
        args = self._parse(
            runtime.DEMO_VERSION_3_2,
            [
                "--input-source",
                "fake-live",
                "--fake-live-case",
                "data_collect/custom_case",
                "--replay-fps",
                "30",
            ],
        )
        runtime.validate_args(args)
        contract = runtime.build_contract(args)
        delegate = runtime.build_live_delegate_argv(args)

        self.assertEqual(contract["recording_case"], "data_collect/custom_case")
        self.assertEqual(contract["replay_fps"], 30.0)
        self.assertEqual(contract["semantic_mode"], "demo")
        self.assertEqual(contract["controller_prompt"], "human hand")
        self.assertEqual(_option_value(delegate, "--recording-case"), "data_collect/custom_case")
        self.assertEqual(_option_value(delegate, "--replay-fps"), "30.0")
        self.assertEqual(_option_value(delegate, "--controller-prompt"), "human hand")

    def test_fake_live_forces_demo_mode_even_with_stale_exp_mode(self) -> None:
        args = self._parse(
            runtime.DEMO_VERSION_3_1,
            [
                "--input-source",
                "fake-live",
                "--mode",
                "exp",
            ],
        )
        runtime.validate_args(args)
        contract = runtime.build_contract(args)
        delegate = runtime.build_live_delegate_argv(args)

        self.assertEqual(args.mode, "demo")
        self.assertEqual(contract["semantic_mode"], "demo")
        self.assertEqual(contract["controller_prompt"], "human hand")
        self.assertEqual(_option_value(delegate, "--controller-prompt"), "human hand")

    def test_pointcloud_load_controls_are_forwarded_to_delegate(self) -> None:
        args = self._parse(
            runtime.DEMO_VERSION_3_1,
            [
                "--pcd-max-points",
                "20000",
                "--pcd-stride",
                "2",
                "--pcd-mask-erode-pixels",
                "4",
                "--depth-max-m",
                "1.2",
                "--pcd-color-mode",
                "class",
                "--render-max-points-per-layer",
                "4096",
                "--view-mode",
                "camera",
                "--enable-pcd-filter",
                "--pcd-filter-mode",
                "sync",
                "--object-filter",
                "pt-filter",
                "--controller-filter",
                "enhanced-pt",
                "--object-filter-cap",
                "12000",
                "--controller-filter-cap",
                "14000",
                "--object-filter-keep-components",
                "1",
                "--controller-filter-keep-components",
                "2",
                "--object-filter-voxel-m",
                "0.006",
                "--controller-filter-voxel-m",
                "0.007",
                "--filter-every-n",
                "2",
                "--filter-max-age-frames",
                "2",
                "--filter-budget-ms",
                "9",
                "--filter-min-cap",
                "4000",
                "--voxel-density-min-points",
                "3",
                "--filter-radius-m",
                "0.012",
                "--filter-nb-points",
                "12",
                "--enhanced-component-voxel-size-m",
                "0.014",
                "--enhanced-keep-near-main-gap-m",
                "0.02",
                "--edgetam-live-session-keep-frames",
                "32",
                "--point-size",
                "1.5",
            ],
        )
        runtime.validate_args(args)
        contract = runtime.build_contract(args)
        delegate = runtime.build_live_delegate_argv(args, active_serial="s0")

        self.assertEqual(contract["pcd_max_points"], 20000)
        self.assertEqual(contract["pcd_stride"], 2)
        self.assertEqual(contract["pcd_mask_erode_pixels"], 4)
        self.assertEqual(contract["depth_max_m"], 1.2)
        self.assertEqual(contract["pcd_color_mode"], "class")
        self.assertEqual(contract["render_max_points_per_layer"], 4096)
        self.assertEqual(contract["view_mode"], "camera")
        self.assertTrue(contract["pcd_filter_enabled"])
        self.assertEqual(contract["pcd_filter_mode"], "sync")
        self.assertEqual(contract["object_filter"], "pt-filter")
        self.assertEqual(contract["controller_filter"], "enhanced-pt")
        self.assertEqual(contract["object_filter_cap"], 12000)
        self.assertEqual(contract["controller_filter_cap"], 14000)
        self.assertEqual(contract["object_filter_keep_components"], 1)
        self.assertEqual(contract["controller_filter_keep_components"], 2)
        self.assertEqual(contract["object_filter_voxel_m"], 0.006)
        self.assertEqual(contract["controller_filter_voxel_m"], 0.007)
        self.assertEqual(contract["filter_every_n"], 2)
        self.assertEqual(contract["filter_max_age_frames"], 2)
        self.assertEqual(contract["filter_budget_ms"], 9.0)
        self.assertEqual(contract["filter_min_cap"], 4000)
        self.assertEqual(contract["voxel_density_min_points"], 3)
        self.assertEqual(contract["filter_radius_m"], 0.012)
        self.assertEqual(contract["filter_nb_points"], 12)
        self.assertEqual(contract["enhanced_component_voxel_size_m"], 0.014)
        self.assertEqual(contract["enhanced_keep_near_main_gap_m"], 0.02)
        self.assertEqual(contract["edgetam_live_session_keep_frames"], 32)
        self.assertEqual(contract["point_size"], 1.5)
        self.assertEqual(_option_value(delegate, "--pcd-max-points"), "20000")
        self.assertEqual(_option_value(delegate, "--pcd-stride"), "2")
        self.assertEqual(_option_value(delegate, "--pcd-mask-erode-pixels"), "4")
        self.assertEqual(_option_value(delegate, "--depth-max-m"), "1.2")
        self.assertEqual(_option_value(delegate, "--pcd-color-mode"), "class")
        self.assertEqual(_option_value(delegate, "--render-max-points-per-layer"), "4096")
        self.assertEqual(_option_value(delegate, "--view-mode"), "camera")
        self.assertEqual(_option_value(delegate, "--pcd-filter-mode"), "sync")
        self.assertEqual(_option_value(delegate, "--object-filter"), "pt-filter")
        self.assertEqual(_option_value(delegate, "--controller-filter"), "enhanced-pt")
        self.assertEqual(_option_value(delegate, "--object-filter-cap"), "12000")
        self.assertEqual(_option_value(delegate, "--controller-filter-cap"), "14000")
        self.assertEqual(_option_value(delegate, "--object-filter-keep-components"), "1")
        self.assertEqual(_option_value(delegate, "--controller-filter-keep-components"), "2")
        self.assertEqual(_option_value(delegate, "--object-filter-voxel-m"), "0.006")
        self.assertEqual(_option_value(delegate, "--controller-filter-voxel-m"), "0.007")
        self.assertEqual(_option_value(delegate, "--filter-every-n"), "2")
        self.assertEqual(_option_value(delegate, "--filter-max-age-frames"), "2")
        self.assertEqual(_option_value(delegate, "--filter-budget-ms"), "9.0")
        self.assertEqual(_option_value(delegate, "--filter-min-cap"), "4000")
        self.assertEqual(_option_value(delegate, "--voxel-density-min-points"), "3")
        self.assertEqual(_option_value(delegate, "--filter-radius-m"), "0.012")
        self.assertEqual(_option_value(delegate, "--filter-nb-points"), "12")
        self.assertEqual(_option_value(delegate, "--enhanced-component-voxel-size-m"), "0.014")
        self.assertEqual(_option_value(delegate, "--enhanced-keep-near-main-gap-m"), "0.02")
        self.assertEqual(_option_value(delegate, "--edgetam-live-session-keep-frames"), "32")
        self.assertEqual(_option_value(delegate, "--point-size"), "1.5")
        self.assertIn("--enable-pcd-filter", delegate)

    def test_invalid_pointcloud_load_controls_are_rejected(self) -> None:
        bad_points = self._parse(runtime.DEMO_VERSION_3_1, ["--pcd-max-points", "-1"])
        with self.assertRaisesRegex(ValueError, "pcd-max-points"):
            runtime.validate_args(bad_points)

        bad_stride = self._parse(runtime.DEMO_VERSION_3_1, ["--pcd-stride", "0"])
        with self.assertRaisesRegex(ValueError, "pcd-stride"):
            runtime.validate_args(bad_stride)

        bad_render_cap = self._parse(runtime.DEMO_VERSION_3_1, ["--render-max-points-per-layer", "-1"])
        with self.assertRaisesRegex(ValueError, "render-max-points-per-layer"):
            runtime.validate_args(bad_render_cap)

        bad_object_filter_cap = self._parse(runtime.DEMO_VERSION_3_1, ["--object-filter-cap", "-1"])
        with self.assertRaisesRegex(ValueError, "object-filter-cap"):
            runtime.validate_args(bad_object_filter_cap)

        bad_controller_filter_cap = self._parse(runtime.DEMO_VERSION_3_1, ["--controller-filter-cap", "-1"])
        with self.assertRaisesRegex(ValueError, "controller-filter-cap"):
            runtime.validate_args(bad_controller_filter_cap)

        bad_object_components = self._parse(runtime.DEMO_VERSION_3_1, ["--object-filter-keep-components", "0"])
        with self.assertRaisesRegex(ValueError, "object-filter-keep-components"):
            runtime.validate_args(bad_object_components)

        bad_controller_components = self._parse(runtime.DEMO_VERSION_3_1, ["--controller-filter-keep-components", "0"])
        with self.assertRaisesRegex(ValueError, "controller-filter-keep-components"):
            runtime.validate_args(bad_controller_components)

        bad_filter_age = self._parse(runtime.DEMO_VERSION_3_1, ["--filter-max-age-frames", "-1"])
        with self.assertRaisesRegex(ValueError, "filter-max-age-frames"):
            runtime.validate_args(bad_filter_age)

        bad_mask_erode = self._parse(runtime.DEMO_VERSION_3_1, ["--pcd-mask-erode-pixels", "-1"])
        with self.assertRaisesRegex(ValueError, "pcd-mask-erode-pixels"):
            runtime.validate_args(bad_mask_erode)

        bad_filter_every = self._parse(runtime.DEMO_VERSION_3_1, ["--filter-every-n", "0"])
        with self.assertRaisesRegex(ValueError, "filter-every-n"):
            runtime.validate_args(bad_filter_every)

        bad_filter_budget = self._parse(runtime.DEMO_VERSION_3_1, ["--filter-budget-ms", "-1"])
        with self.assertRaisesRegex(ValueError, "filter-budget-ms"):
            runtime.validate_args(bad_filter_budget)

        bad_filter_min_cap = self._parse(runtime.DEMO_VERSION_3_1, ["--filter-min-cap", "-1"])
        with self.assertRaisesRegex(ValueError, "filter-min-cap"):
            runtime.validate_args(bad_filter_min_cap)

        bad_voxel_density = self._parse(runtime.DEMO_VERSION_3_1, ["--voxel-density-min-points", "0"])
        with self.assertRaisesRegex(ValueError, "voxel-density-min-points"):
            runtime.validate_args(bad_voxel_density)

        bad_radius = self._parse(runtime.DEMO_VERSION_3_1, ["--filter-radius-m", "0"])
        with self.assertRaisesRegex(ValueError, "filter-radius-m"):
            runtime.validate_args(bad_radius)

        bad_nb_points = self._parse(runtime.DEMO_VERSION_3_1, ["--filter-nb-points", "0"])
        with self.assertRaisesRegex(ValueError, "filter-nb-points"):
            runtime.validate_args(bad_nb_points)

        bad_component_voxel = self._parse(runtime.DEMO_VERSION_3_1, ["--enhanced-component-voxel-size-m", "0"])
        with self.assertRaisesRegex(ValueError, "enhanced-component-voxel-size-m"):
            runtime.validate_args(bad_component_voxel)

        bad_gap = self._parse(runtime.DEMO_VERSION_3_1, ["--enhanced-keep-near-main-gap-m", "-0.001"])
        with self.assertRaisesRegex(ValueError, "enhanced-keep-near-main-gap-m"):
            runtime.validate_args(bad_gap)

        bad_keep_frames = self._parse(runtime.DEMO_VERSION_3_1, ["--edgetam-live-session-keep-frames", "-1"])
        with self.assertRaisesRegex(ValueError, "edgetam-live-session-keep-frames"):
            runtime.validate_args(bad_keep_frames)

        headless_filter = self._parse(runtime.DEMO_VERSION_3_1, ["--render-mode", "none", "--enable-pcd-filter"])
        with self.assertRaisesRegex(ValueError, "enable-pcd-filter"):
            runtime.validate_args(headless_filter)

    def test_demo31_demo32_demo33_default_to_5000_render_points_per_layer(self) -> None:
        for version in (runtime.DEMO_VERSION_3_1, runtime.DEMO_VERSION_3_2, runtime.DEMO_VERSION_3_3):
            with self.subTest(version=version):
                args = self._parse(version, [])
                runtime.validate_args(args)
                contract = runtime.build_contract(args)
                delegate = runtime.build_live_delegate_argv(args, active_serial="s0")

                self.assertEqual(contract["render_max_points_per_layer"], 5000)
                self.assertEqual(contract["view_mode"], "orbit")
                self.assertEqual(contract["pcd_mask_erode_pixels"], 0)
                self.assertEqual(contract["object_filter_keep_components"], 1)
                self.assertEqual(contract["controller_filter_keep_components"], 2)
                self.assertEqual(contract["filter_max_age_frames"], 3)
                self.assertEqual(contract["edgetam_live_session_keep_frames"], 64)
                self.assertEqual(_option_value(delegate, "--render-max-points-per-layer"), "5000")
                self.assertEqual(_option_value(delegate, "--view-mode"), "orbit")
                self.assertEqual(_option_value(delegate, "--pcd-mask-erode-pixels"), "0")
                self.assertEqual(_option_value(delegate, "--object-filter-keep-components"), "1")
                self.assertEqual(_option_value(delegate, "--controller-filter-keep-components"), "2")
                self.assertEqual(_option_value(delegate, "--filter-max-age-frames"), "3")
                self.assertEqual(_option_value(delegate, "--edgetam-live-session-keep-frames"), "64")

    def test_ffs_filter_surface_defaults_apply_only_when_filter_enabled(self) -> None:
        disabled = self._parse(runtime.DEMO_VERSION_3_2, [])
        self.assertEqual(disabled.filter_radius_m, runtime.masked_pcd.DEFAULT_FILTER_RADIUS_M)
        self.assertEqual(disabled.filter_nb_points, runtime.masked_pcd.DEFAULT_FILTER_NB_POINTS)
        self.assertEqual(disabled.filter_every_n, 3)
        self.assertEqual(disabled.filter_max_age_frames, 3)
        self.assertEqual(disabled.pcd_mask_erode_pixels, 0)

        enabled = self._parse(runtime.DEMO_VERSION_3_2, ["--enable-pcd-filter"])
        runtime.validate_args(enabled)
        contract = runtime.build_contract(enabled)
        delegate = runtime.build_live_delegate_argv(enabled, active_serial="s0")

        self.assertTrue(contract["pcd_filter_enabled"])
        self.assertEqual(contract["pcd_filter_mode"], "async")
        self.assertEqual(contract["object_filter"], "enhanced-pt")
        self.assertEqual(contract["controller_filter"], "enhanced-pt")
        self.assertEqual(contract["filter_radius_m"], runtime.FFS_SURFACE_FILTER_RADIUS_M)
        self.assertEqual(contract["filter_nb_points"], runtime.FFS_SURFACE_FILTER_NB_POINTS)
        self.assertEqual(contract["enhanced_component_voxel_size_m"], runtime.FFS_SURFACE_COMPONENT_VOXEL_SIZE_M)
        self.assertEqual(contract["filter_every_n"], runtime.FFS_SURFACE_FILTER_EVERY_N)
        self.assertEqual(contract["filter_max_age_frames"], runtime.FFS_SURFACE_FILTER_MAX_AGE_FRAMES)
        self.assertEqual(contract["pcd_mask_erode_pixels"], runtime.FFS_SURFACE_MASK_ERODE_PIXELS)
        self.assertEqual(_option_value(delegate, "--filter-radius-m"), str(runtime.FFS_SURFACE_FILTER_RADIUS_M))
        self.assertEqual(_option_value(delegate, "--filter-nb-points"), str(runtime.FFS_SURFACE_FILTER_NB_POINTS))
        self.assertEqual(
            _option_value(delegate, "--enhanced-component-voxel-size-m"),
            str(runtime.FFS_SURFACE_COMPONENT_VOXEL_SIZE_M),
        )
        self.assertEqual(_option_value(delegate, "--filter-every-n"), str(runtime.FFS_SURFACE_FILTER_EVERY_N))
        self.assertEqual(
            _option_value(delegate, "--filter-max-age-frames"),
            str(runtime.FFS_SURFACE_FILTER_MAX_AGE_FRAMES),
        )
        self.assertEqual(
            _option_value(delegate, "--pcd-mask-erode-pixels"),
            str(runtime.FFS_SURFACE_MASK_ERODE_PIXELS),
        )

        overridden = self._parse(
            runtime.DEMO_VERSION_3_2,
            [
                "--enable-pcd-filter",
                "--filter-radius-m",
                "0.02",
                "--filter-nb-points",
                "6",
                "--enhanced-component-voxel-size-m",
                "0.025",
                "--filter-every-n",
                "4",
                "--filter-max-age-frames",
                "5",
                "--pcd-mask-erode-pixels",
                "1",
            ],
        )
        self.assertEqual(overridden.filter_radius_m, 0.02)
        self.assertEqual(overridden.filter_nb_points, 6)
        self.assertEqual(overridden.enhanced_component_voxel_size_m, 0.025)
        self.assertEqual(overridden.filter_every_n, 4)
        self.assertEqual(overridden.filter_max_age_frames, 5)
        self.assertEqual(overridden.pcd_mask_erode_pixels, 1)

    def test_recording_mode_skips_live_serial_validation(self) -> None:
        with mock.patch.object(runtime.masked_pcd, "main", return_value=0) as masked_main:
            code = runtime.main(
                [
                    "--input-source",
                    "recording",
                    "--recording-case",
                    "data_collect/example_rgbd",
                ],
                demo_version=runtime.DEMO_VERSION_3_1,
                connected_serials_provider=lambda: (_ for _ in ()).throw(AssertionError("serial check should not run")),
            )

        self.assertEqual(code, 0)
        self.assertIn("--input-source", masked_main.call_args.args[0])
        self.assertIn("recording", masked_main.call_args.args[0])
        self.assertIn("--render-mode", masked_main.call_args.args[0])
        self.assertIn("pointcloud", masked_main.call_args.args[0])
        self.assertIn("--tracker-backend", masked_main.call_args.args[0])
        self.assertIn("tapnextpp", masked_main.call_args.args[0])

    def test_recording_mode_requires_pointcloud_render_path(self) -> None:
        args = self._parse(
            runtime.DEMO_VERSION_3_1,
            [
                "--input-source",
                "recording",
                "--recording-case",
                "data_collect/example_rgbd",
                "--render-mode",
                "none",
            ],
        )

        with self.assertRaisesRegex(ValueError, "requires --render-mode pointcloud"):
            runtime.validate_args(args)

    def test_recording_mode_requires_controller_object_tracking(self) -> None:
        args = self._parse(
            runtime.DEMO_VERSION_3_1,
            [
                "--input-source",
                "recording",
                "--recording-case",
                "data_collect/example_rgbd",
                "--track-mode",
                "none",
            ],
        )

        with self.assertRaisesRegex(ValueError, "requires --track-mode controller-object"):
            runtime.validate_args(args)

    def test_recording_mode_requires_tapnextpp_tracker(self) -> None:
        args = self._parse(
            runtime.DEMO_VERSION_3_1,
            [
                "--input-source",
                "recording",
                "--recording-case",
                "data_collect/example_rgbd",
                "--tracker-backend",
                "none",
            ],
        )

        with self.assertRaisesRegex(ValueError, "requires --tracker-backend tapnextpp"):
            runtime.validate_args(args)

    def test_headless_render_auto_disables_tracker_when_not_explicit(self) -> None:
        args = self._parse(runtime.DEMO_VERSION_3_1, ["--render-mode", "none"])
        runtime.validate_args(args)
        contract = runtime.build_contract(args)

        self.assertEqual(contract["track_mode"], "none")
        self.assertEqual(contract["tracker_backend"], "none")
        self.assertEqual(contract["tracker_visualization_mode"], "none")
        delegate = runtime.build_live_delegate_argv(args, active_serial="s0")
        self.assertIn("--tracker-backend", delegate)
        self.assertIn("none", delegate)
        self.assertIn("--pcd-mode", delegate)
        self.assertIn("none", delegate)

    def test_recording_alias_accepts_ffs_versions_when_case_is_explicit(self) -> None:
        args = self._parse(
            runtime.DEMO_VERSION_3_2,
            [
                "--input-source",
                "recording",
                "--recording-case",
                "data_collect/example_rgbd",
            ],
        )

        runtime.validate_args(args)
        contract = runtime.build_contract(args)
        delegate = runtime.build_live_delegate_argv(args)

        self.assertEqual(contract["input_source_mode"], "recording")
        self.assertEqual(contract["depth_source"], "ffs")
        self.assertIn("recording", delegate)
        self.assertIn("--recording-case", delegate)


if __name__ == "__main__":
    unittest.main()
