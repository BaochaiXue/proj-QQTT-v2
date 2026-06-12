from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from qqtt.demo import single_demo_v3_runtime as runtime


def _explicit(argv: list[str]) -> set[str]:
    return {item.split("=", 1)[0] for item in argv if item.startswith("--")}


class SingleDemoV3RuntimeTest(unittest.TestCase):
    def _parse(self, version: str, argv: list[str]):
        parser = runtime.build_arg_parser(demo_version=version)
        args = parser.parse_args(argv)
        return runtime.apply_preset_defaults(args, explicit_options=_explicit(argv))

    def test_demo3_defaults_to_one_realsense_camera(self) -> None:
        args = self._parse(
            runtime.DEMO_VERSION_3,
            ["--dry-run", "--calibrate-path", "/tmp/missing-single-demo-calibrate.pkl"],
        )
        runtime.validate_args(args)
        contract = runtime.build_contract(args)

        self.assertEqual(contract["demo"], "single-demo3")
        self.assertTrue(contract["requires_single_realsense"])
        self.assertFalse(contract["requires_three_realsense"])
        self.assertEqual(contract["num_cameras"], 1)
        self.assertEqual(contract["camera_ids"], [0])
        self.assertFalse(contract["camera_sync_required"])
        self.assertFalse(contract["multi_camera_world_fusion"])
        self.assertFalse(contract["calibrate_pkl_required"])
        self.assertEqual(contract["depth_source"], "realsense")
        self.assertFalse(contract["uses_ffs"])
        self.assertEqual(contract["ffs_trt_batch_size"], 0)
        self.assertFalse(contract["dual_gpu_enabled"])
        self.assertEqual(contract["required_cuda_devices"], 1)
        self.assertEqual(contract["object_prompt"], "stuffed animal")
        self.assertEqual(contract["controller_prompt"], "towel")
        self.assertFalse(contract["point_tracker_enabled"])
        self.assertIn("batch3_tensor_rt_depth", contract["removed_three_camera_work"])

    def test_demo32_defaults_to_single_camera_ffs_batch_one(self) -> None:
        args = self._parse(
            runtime.DEMO_VERSION_3_2,
            ["--dry-run", "--calibrate-path", "/tmp/missing-single-demo-calibrate.pkl"],
        )
        contract = runtime.build_contract(args)

        self.assertEqual(contract["demo"], "single-demo3.2")
        self.assertEqual(contract["depth_source"], "ffs")
        self.assertTrue(contract["uses_ffs"])
        self.assertEqual(contract["ffs_trt_batch_size"], 1)
        self.assertEqual(contract["ffs_schedule"], "single-camera-latest")
        self.assertFalse(contract["ffs_batch3_required"])
        self.assertFalse(contract["strict_source_three_camera_bundle"])

        delegate = runtime.build_live_delegate_argv(args, active_serial="s0")
        self.assertIn("--serial", delegate)
        self.assertIn("s0", delegate)
        self.assertIn("--depth-source", delegate)
        self.assertIn("ffs", delegate)
        self.assertIn("--controller-prompt", delegate)
        self.assertIn("towel", delegate)

    def test_parser_rejects_multi_camera_ids(self) -> None:
        parser = runtime.build_arg_parser(demo_version=runtime.DEMO_VERSION_3)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(["--dry-run", "--camera-ids", "0,1"])

    def test_realsense_versions_reject_ffs(self) -> None:
        args = self._parse(
            runtime.DEMO_VERSION_3_1,
            [
                "--dry-run",
                "--depth-source",
                "ffs",
                "--calibrate-path",
                "/tmp/missing-single-demo-calibrate.pkl",
            ],
        )
        with self.assertRaisesRegex(ValueError, "RealSense-depth only"):
            runtime.validate_args(args)

    def test_dry_run_main_prints_single_camera_contract_and_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            profile = Path(tmp_dir) / "profile.json"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = runtime.main(
                    [
                        "--dry-run",
                        "--profile-json-output",
                        str(profile),
                        "--calibrate-path",
                        str(Path(tmp_dir) / "missing.pkl"),
                    ],
                    demo_version=runtime.DEMO_VERSION_3_2,
                )

            self.assertEqual(code, 0)
            output = stdout.getvalue()
            self.assertIn("requires_three_realsense = false", output)
            self.assertIn("num_cameras = 1", output)
            self.assertIn("ffs_trt_batch_size = 1", output)

            payload = json.loads(profile.read_text(encoding="utf-8"))
            self.assertEqual(payload["contract"]["num_cameras"], 1)
            self.assertFalse(payload["contract"]["requires_three_realsense"])

    def test_live_validation_uses_one_connected_serial(self) -> None:
        args = self._parse(
            runtime.DEMO_VERSION_3,
            ["--calibrate-path", "/tmp/missing-single-demo-calibrate.pkl"],
        )
        validation = runtime.validate_live_contract(
            args,
            connected_serials_provider=lambda: ["s0", "s1"],
        )

        self.assertEqual(validation["active_serials"], ["s0"])
        self.assertEqual(validation["active_serial"], "s0")


if __name__ == "__main__":
    unittest.main()
