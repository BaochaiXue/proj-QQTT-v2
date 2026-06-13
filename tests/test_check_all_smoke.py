from __future__ import annotations

from pathlib import Path
import unittest

from scripts.harness import check_all


class CheckAllSmokeTest(unittest.TestCase):
    def test_parse_args_defaults_to_quick_profile(self) -> None:
        args = check_all.parse_args([])
        self.assertEqual(args.profile, "quick")

    def test_full_flag_selects_full_profile(self) -> None:
        args = check_all.parse_args(["--full"])
        self.assertEqual(args.profile, "full")

    def test_quick_profile_uses_curated_batched_commands(self) -> None:
        commands = check_all.build_commands(python="python", profile="quick")
        self.assertEqual(len(commands), 13)
        self.assertIn(["python", "cameras_viewer.py", "--help"], commands)
        self.assertIn(["python", "record_data_realtime_align.py", "--help"], commands)
        self.assertIn(["python", "data_process/record_data_align.py", "--help"], commands)
        self.assertIn(["python", "scripts/harness/visual_compare_depth_panels.py", "--help"], commands)
        self.assertIn(["python", "scripts/harness/visual_compare_reprojection.py", "--help"], commands)
        self.assertIn(["python", "scripts/harness/visual_compare_turntable.py", "--help"], commands)
        self.assertIn(["python", "scripts/harness/check_harness_catalog.py"], commands)
        self.assertIn(["python", "scripts/harness/check_experiment_boundaries.py"], commands)
        self.assertIn(["python", "scripts/harness/check_visual_architecture.py"], commands)
        unittest_commands = [cmd for cmd in commands if cmd[1:4] == ["-m", "unittest", "-v"]]
        self.assertEqual(unittest_commands, [["python", "-m", "unittest", "-v", *check_all.QUICK_UNITTEST_MODULES]])
        flat_items = [item for command in commands for item in command]
        self.assertFalse(any(cmd[:3] == ["python", "-m", "pytest"] for cmd in commands))
        self.assertNotIn("tests.test_visual_compare_depth_panels_smoke", flat_items)
        self.assertNotIn("tests.test_visual_compare_reprojection_smoke", flat_items)
        self.assertNotIn("tests.test_visual_compare_turntable_smoke", flat_items)
        self.assertIn("tests.test_single_camera_defaults_smoke", flat_items)
        self.assertIn("tests.test_recorded_rgbd_replay_source", flat_items)
        self.assertIn("tests.test_single_demo_v3_runtime", flat_items)

    def test_full_profile_keeps_pytest_and_broader_command_surface(self) -> None:
        commands = check_all.build_commands(python="python", profile="full")
        self.assertGreater(len(commands), len(check_all.build_commands(python="python", profile="quick")))
        self.assertEqual(len(check_all.FULL_UNITTEST_MODULES), len(set(check_all.FULL_UNITTEST_MODULES)))
        self.assertTrue(set(check_all.QUICK_UNITTEST_MODULES).issubset(check_all.FULL_UNITTEST_MODULES))
        self.assertIn(["python", "cameras_viewer_FFS.py", "--help"], commands)
        self.assertIn(["python", "single_demo_v3/realtime_single_camera_realsense_masked_pcd.py", "--help"], commands)
        self.assertIn(["python", "single_demo_v3_1/realtime_single_camera_realsense_masked_pcd.py", "--help"], commands)
        self.assertIn(["python", "single_demo_v3_2/realtime_single_camera_ffs_masked_pcd.py", "--help"], commands)
        self.assertIn(["python", "single_demo_v3_3/realtime_single_camera_ffs_masked_pcd.py", "--help"], commands)
        self.assertIn(["python", "scripts/harness/realtime_single_camera_pointcloud.py", "--help"], commands)
        self.assertIn(["python", "scripts/harness/experiments/run_ffs_confidence_filter_sweep.py", "--help"], commands)
        self.assertIn(["python", "scripts/harness/verify_ffs_demo.py", "--help"], commands)
        self.assertIn(["python", "scripts/harness/verify_ffs_single_engine_tensorrt_wsl.py", "--help"], commands)
        flat_items = [item for command in commands for item in command]
        self.assertIn("tests.test_visual_compare_depth_panels_smoke", flat_items)
        self.assertIn("tests.test_visual_compare_reprojection_smoke", flat_items)
        self.assertIn("tests.test_visual_compare_turntable_smoke", flat_items)
        self.assertIn(
            ["python", "-m", "pytest", "tests/test_d455_probe_matrix_builder.py", "tests/test_d455_probe_result_schema.py"],
            commands,
        )

    def test_generated_script_paths_exist(self) -> None:
        for profile in ("quick", "full"):
            with self.subTest(profile=profile):
                commands = check_all.build_commands(python="python", profile=profile)
                for cmd in commands:
                    for item in cmd[1:]:
                        if item.endswith(".py") and not item.startswith("-"):
                            script_path = check_all.ROOT / item
                            self.assertTrue(script_path.is_file(), f"missing script path in {profile}: {item}")

    def test_generated_unittest_modules_exist(self) -> None:
        for profile in ("quick", "full"):
            with self.subTest(profile=profile):
                commands = check_all.build_commands(python="python", profile=profile)
                for cmd in commands:
                    if cmd[1:4] != ["-m", "unittest", "-v"]:
                        continue
                    for module_name in cmd[4:]:
                        module_path = check_all.ROOT / Path(*module_name.split(".")).with_suffix(".py")
                        self.assertTrue(module_path.is_file(), f"missing unittest module in {profile}: {module_name}")


if __name__ == "__main__":
    unittest.main()
