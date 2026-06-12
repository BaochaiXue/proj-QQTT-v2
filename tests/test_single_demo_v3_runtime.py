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
    "ffs_batch3_required",
    "point_tracker_enabled",
    "point_tracker_live_stage",
    "tracking_backend_execution_mode",
    "tracking_backend_batch_size",
    "tracking_backend_model_instances_expected",
    "dual_gpu_enabled",
    "required_cuda_devices",
    "require_two_cuda",
    "cross_gpu_cuda_tensor_transfer",
    "strict_source_three_camera_bundle",
    "shape_prior_warmup_enabled",
    "shape_prior_live_stage",
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
        self.assertEqual(contract["live_delegate_module"], "qqtt.demo.realtime_masked_edgetam_pcd")
        self.assertEqual(contract["depth_source"], "realsense")
        self.assertEqual(contract["depth_pipeline"], "realsense_native")
        self.assertFalse(contract["uses_ffs"])
        self.assertIsNone(contract["ffs_trt_batch_size"])
        self.assertEqual(contract["object_prompt"], "stuffed animal")
        self.assertEqual(contract["controller_prompt"], "towel")
        self.assert_legacy_fields_removed(contract)

    def test_demo32_contract_is_single_camera_ffs_batch_one(self) -> None:
        args = self._parse(runtime.DEMO_VERSION_3_2, ["--dry-run"])
        contract = runtime.build_contract(args)

        self.assertEqual(contract["demo"], "single-demo3.2")
        self.assertEqual(contract["camera_count"], 1)
        self.assertEqual(contract["depth_source"], "ffs")
        self.assertEqual(contract["depth_pipeline"], "ffs_tensorrt_batch1")
        self.assertTrue(contract["uses_ffs"])
        self.assertEqual(contract["ffs_trt_batch_size"], 1)
        self.assert_legacy_fields_removed(contract)

        delegate = runtime.build_live_delegate_argv(args, active_serial="s0")
        self.assertIn("--serial", delegate)
        self.assertIn("s0", delegate)
        self.assertIn("--depth-source", delegate)
        self.assertIn("ffs", delegate)
        self.assertIn("--controller-prompt", delegate)
        self.assertIn("towel", delegate)

    def test_old_multi_camera_options_are_not_public_cli(self) -> None:
        parser = runtime.build_arg_parser(demo_version=runtime.DEMO_VERSION_3)
        removed_args = (
            "--camera-ids",
            "--serials",
            "--calibrate-path",
            "--depth-source",
            "--live-delegate",
            "--shape-prior-warmup",
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
            self.assertNotIn("shape_prior", output)

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


if __name__ == "__main__":
    unittest.main()
