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
        self.assertEqual(len(commands), 15)
        self.assertIn(["python", "cameras_viewer.py", "--help"], commands)
        self.assertIn(["python", "scripts/harness/check_visual_architecture.py"], commands)
        self.assertIn(
            [
                "python",
                "-m",
                "unittest",
                "-v",
                "tests.test_visual_compare_depth_panels_smoke",
                "tests.test_visual_compare_reprojection_smoke",
                "tests.test_visual_compare_turntable_smoke",
            ],
            commands,
        )
        self.assertFalse(any(cmd[:3] == ["python", "-m", "pytest"] for cmd in commands))

    def test_full_profile_keeps_pytest_and_broader_command_surface(self) -> None:
        commands = check_all.build_commands(python="python", profile="full")
        self.assertGreater(len(commands), len(check_all.build_commands(python="python", profile="quick")))
        self.assertIn(["python", "scripts/harness/verify_ffs_demo.py", "--help"], commands)
        self.assertIn(
            ["python", "-m", "pytest", "tests/test_d455_probe_matrix_builder.py", "tests/test_d455_probe_result_schema.py"],
            commands,
        )


if __name__ == "__main__":
    unittest.main()
