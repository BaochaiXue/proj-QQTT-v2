from __future__ import annotations

from pathlib import Path
import unittest

from scripts.harness.validation import run as validation_run


class ValidationRunnerSmokeTest(unittest.TestCase):
    def test_parse_args_defaults_to_smoke_profile(self) -> None:
        args = validation_run.parse_args([])
        self.assertEqual(args.profile, "smoke")
        self.assertFalse(args.run_hardware)

    def test_parse_args_accepts_new_profiles(self) -> None:
        for profile in ("smoke", "deterministic", "hardware", "exhaustive"):
            with self.subTest(profile=profile):
                args = validation_run.parse_args(["--profile", profile])
                self.assertEqual(args.profile, profile)

    def test_parse_args_rejects_old_profile_names(self) -> None:
        with self.assertRaises(SystemExit):
            validation_run.parse_args(["--full"])
        with self.assertRaises(SystemExit):
            validation_run.parse_args(["--profile", "quick"])

    def test_smoke_profile_uses_curated_batched_commands(self) -> None:
        commands = validation_run.build_commands(python="python", profile="smoke")
        self.assertEqual(len(commands), 14)
        self.assertIn(["python", "cameras_viewer.py", "--help"], commands)
        self.assertIn(["python", "record_data_realtime_align.py", "--help"], commands)
        self.assertIn(["python", "data_process/record_data_align.py", "--help"], commands)
        self.assertIn(["python", "scripts/harness/render_demo32_headless_capture.py", "--help"], commands)
        self.assertIn(["python", "scripts/harness/visual_compare_depth_panels.py", "--help"], commands)
        self.assertIn(["python", "scripts/harness/visual_compare_reprojection.py", "--help"], commands)
        self.assertIn(["python", "scripts/harness/visual_compare_turntable.py", "--help"], commands)
        self.assertIn(["python", "scripts/harness/check_harness_catalog.py"], commands)
        self.assertIn(["python", "scripts/harness/check_experiment_boundaries.py"], commands)
        self.assertIn(["python", "scripts/harness/check_visual_architecture.py"], commands)
        unittest_commands = [cmd for cmd in commands if cmd[1:4] == ["-m", "unittest", "-v"]]
        self.assertEqual(
            unittest_commands,
            [["python", "-m", "unittest", "-v", *validation_run.SMOKE_UNITTEST_MODULES]],
        )
        flat_items = [item for command in commands for item in command]
        self.assertFalse(any(cmd[:3] == ["python", "-m", "pytest"] for cmd in commands))
        self.assertNotIn("tests.test_visual_compare_depth_panels_smoke", flat_items)
        self.assertNotIn("tests.test_visual_compare_reprojection_smoke", flat_items)
        self.assertNotIn("tests.test_visual_compare_turntable_smoke", flat_items)
        self.assertIn("tests.test_single_camera_defaults_smoke", flat_items)
        self.assertIn("tests.test_recorded_rgbd_replay_source", flat_items)
        self.assertIn("tests.test_single_demo_v3_runtime", flat_items)
        self.assertIn("tests.test_realtime_masked_edgetam_pcd_filter", flat_items)
        self.assertIn("tests.test_single_demo_tapnextpp_overlay", flat_items)
        self.assertIn("tests.test_demo32_headless_render_helper", flat_items)

    def test_deterministic_profile_broadens_command_surface(self) -> None:
        commands = validation_run.build_commands(python="python", profile="deterministic")
        smoke_commands = validation_run.build_commands(python="python", profile="smoke")
        deterministic_modules = validation_run._unique(
            (*validation_run.SMOKE_UNITTEST_MODULES, *validation_run.DETERMINISTIC_ONLY_UNITTEST_MODULES)
        )
        self.assertGreater(len(commands), len(smoke_commands))
        self.assertEqual(len(deterministic_modules), len(set(deterministic_modules)))
        self.assertTrue(set(validation_run.SMOKE_UNITTEST_MODULES).issubset(deterministic_modules))
        self.assertIn(["python", "cameras_viewer_FFS.py", "--help"], commands)
        self.assertIn(["python", "demo_v3/realtime_single_camera_realsense_masked_pcd.py", "--help"], commands)
        self.assertIn(["python", "demo_v3_1/realtime_single_camera_realsense_masked_pcd.py", "--help"], commands)
        self.assertIn(["python", "demo_v3_2/realtime_single_camera_ffs_masked_pcd.py", "--help"], commands)
        self.assertIn(["python", "demo_v3_3/realtime_single_camera_ffs_masked_pcd.py", "--help"], commands)
        flat_items = [item for command in commands for item in command]
        self.assertIn("tests.test_visual_compare_depth_panels_smoke", flat_items)
        self.assertIn("tests.test_visual_compare_reprojection_smoke", flat_items)
        self.assertIn("tests.test_visual_compare_turntable_smoke", flat_items)
        self.assertFalse(any(cmd[:3] == ["python", "-m", "pytest"] for cmd in commands))

    def test_exhaustive_profile_keeps_pytest_and_exhaustive_help_surface(self) -> None:
        commands = validation_run.build_commands(python="python", profile="exhaustive")
        deterministic_commands = validation_run.build_commands(python="python", profile="deterministic")
        self.assertGreater(len(commands), len(deterministic_commands))
        self.assertIn(["python", "scripts/harness/experiments/run_ffs_confidence_filter_sweep.py", "--help"], commands)
        self.assertIn(
            ["python", "-m", "pytest", "tests/test_d455_probe_matrix_builder.py", "tests/test_d455_probe_result_schema.py"],
            commands,
        )

    def test_hardware_profile_lists_manual_commands_only_when_requested(self) -> None:
        self.assertEqual(validation_run.build_commands(python="python", profile="hardware"), [])
        commands = validation_run.build_commands(python="python", profile="hardware", run_hardware=True)
        self.assertIn(["python", "scripts/harness/realtime_single_camera_pointcloud.py", "--help"], commands)
        self.assertIn(["python", "scripts/harness/verify_ffs_demo.py", "--help"], commands)
        self.assertFalse(any(cmd[1:4] == ["-m", "unittest", "-v"] for cmd in commands))

    def test_generated_script_paths_exist(self) -> None:
        for profile in ("smoke", "deterministic", "exhaustive"):
            with self.subTest(profile=profile):
                commands = validation_run.build_commands(python="python", profile=profile)
                for cmd in commands:
                    for item in cmd[1:]:
                        if item.endswith(".py") and not item.startswith("-"):
                            script_path = validation_run.ROOT / item
                            self.assertTrue(script_path.is_file(), f"missing script path in {profile}: {item}")

    def test_generated_unittest_modules_exist(self) -> None:
        for profile in ("smoke", "deterministic", "exhaustive"):
            with self.subTest(profile=profile):
                commands = validation_run.build_commands(python="python", profile=profile)
                for cmd in commands:
                    if cmd[1:4] != ["-m", "unittest", "-v"]:
                        continue
                    for module_name in cmd[4:]:
                        module_path = validation_run.ROOT / Path(*module_name.split(".")).with_suffix(".py")
                        self.assertTrue(module_path.is_file(), f"missing unittest module in {profile}: {module_name}")


if __name__ == "__main__":
    unittest.main()
