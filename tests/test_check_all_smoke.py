from __future__ import annotations

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
        self.assertEqual(len(commands), 25)
        self.assertIn(["python", "cameras_viewer.py", "--help"], commands)
        self.assertIn(["python", "record_data_realtime_align.py", "--help"], commands)
        self.assertIn(["python", "scripts/harness/experiments/visualize_ffs_static_confidence_panels.py", "--help"], commands)
        self.assertIn(["python", "scripts/harness/experiments/visualize_ffs_static_confidence_pcd_panels.py", "--help"], commands)
        self.assertIn(["python", "scripts/harness/experiments/run_ffs_confidence_filter_sweep.py", "--help"], commands)
        self.assertIn(["python", "scripts/harness/experiments/visual_compare_ffs_confidence_filter_pcd.py", "--help"], commands)
        self.assertIn(["python", "scripts/harness/experiments/visual_compare_ffs_confidence_threshold_sweep_pcd.py", "--help"], commands)
        self.assertIn(["python", "scripts/harness/experiments/visual_compare_ffs_mask_erode_sweep_pcd.py", "--help"], commands)
        self.assertIn(["python", "scripts/harness/experiments/visual_compare_native_ffs_fused_pcd.py", "--help"], commands)
        self.assertIn(["python", "scripts/harness/check_experiment_boundaries.py"], commands)
        self.assertIn(["python", "scripts/harness/check_visual_architecture.py"], commands)
        self.assertIn(
            [
                "python",
                "-m",
                "unittest",
                "-v",
                "tests.test_record_data_align_smoke",
                "tests.test_cameras_viewer_ffs_smoke",
                "tests.test_depth_backend_contract_smoke",
                "tests.test_ffs_confidence_filtering_smoke",
                "tests.test_ffs_confidence_filter_pcd_compare_smoke",
                "tests.test_ffs_confidence_threshold_sweep_pcd_compare_smoke",
                "tests.test_ffs_mask_erode_sweep_pcd_compare_smoke",
                "tests.test_native_ffs_fused_pcd_compare_smoke",
                "tests.test_ffs_intrinsic_file_format",
                "tests.test_ffs_reprojection_smoke",
                "tests.test_ffs_remove_invisible_mask_smoke",
                "tests.test_ffs_tensorrt_single_engine_smoke",
                "tests.test_ffs_confidence_panels_smoke",
                "tests.test_ffs_confidence_pcd_panels_smoke",
                "tests.test_ffs_static_replay_matrix_smoke",
            ],
            commands,
        )
        self.assertFalse(any(cmd[:3] == ["python", "-m", "pytest"] for cmd in commands))

    def test_full_profile_keeps_pytest_and_broader_command_surface(self) -> None:
        commands = check_all.build_commands(python="python", profile="full")
        self.assertGreater(len(commands), len(check_all.build_commands(python="python", profile="quick")))
        self.assertIn(["python", "scripts/harness/verify_ffs_demo.py", "--help"], commands)
        self.assertIn(["python", "scripts/harness/verify_ffs_single_engine_tensorrt_wsl.py", "--help"], commands)
        self.assertIn(
            ["python", "-m", "pytest", "tests/test_d455_probe_matrix_builder.py", "tests/test_d455_probe_result_schema.py"],
            commands,
        )


if __name__ == "__main__":
    unittest.main()
