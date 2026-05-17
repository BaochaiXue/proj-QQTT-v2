from __future__ import annotations

import contextlib
import io
import unittest

from qqtt.demo import demo3_runtime


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
        self.assertEqual(contract["cotracker_backend"], "cotracker3_online")
        self.assertTrue(contract["cotracker_async"])
        self.assertTrue(contract["render_latest_wins"])

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
        self.assertIn("cotracker_backend = cotracker3_online", output)
        self.assertIn("cotracker_async = true", output)
        self.assertIn("render_latest_wins = true", output)

    def test_mask_only_preset_disables_cotracker(self) -> None:
        args = self._parse(["--preset", "demo3-realsense-mask-only", "--dry-run", "--camera-ids", "0,1,2"])
        contract = demo3_runtime.build_contract(args)
        self.assertFalse(contract["cotracker_enabled"])
        self.assertFalse(contract["uses_ffs"])


if __name__ == "__main__":
    unittest.main()
