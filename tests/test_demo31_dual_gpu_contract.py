from __future__ import annotations

import contextlib
import io
import unittest

from qqtt.demo import demo31_runtime
from qqtt.demo.demo31_dual_gpu_ipc import TrackingResultLitePacket


class Demo31DualGpuContractTest(unittest.TestCase):
    def _parse(self, argv: list[str]):
        parser = demo31_runtime.build_arg_parser()
        args = parser.parse_args(argv)
        return demo31_runtime.apply_preset_defaults(args, explicit_options=set(argv))

    def test_dry_run_contract_maps_gpu0_and_gpu1(self) -> None:
        args = self._parse(["--dry-run", "--camera-ids", "0,1,2", "--mask-gpu", "0", "--cotracker-gpu", "1"])
        demo31_runtime.validate_args(args, cuda_device_count_provider=lambda: 2)
        contract = demo31_runtime.build_contract(args, cuda_device_count_provider=lambda: 2)

        self.assertEqual(contract["demo"], "demo3.1")
        self.assertTrue(contract["dual_gpu_enabled"])
        self.assertEqual(contract["required_cuda_devices"], 2)
        self.assertEqual(contract["mask_gpu_physical"], 0)
        self.assertEqual(contract["cotracker_gpu_physical"], 1)
        self.assertEqual(contract["main_cuda_visible_devices"], "0")
        self.assertEqual(contract["cotracker_cuda_visible_devices"], "1")
        self.assertEqual(contract["depth_source"], "realsense")
        self.assertFalse(contract["uses_ffs"])
        self.assertEqual(contract["mask_source"], "hf_edgetam")
        self.assertTrue(contract["edgetam_batch_vision_encoder"])
        self.assertEqual(contract["cotracker_backend"], "cotracker3_online")
        self.assertEqual(contract["cotracker_owner"], "process")
        self.assertEqual(contract["cotracker_process_mode"], "subprocess")
        self.assertFalse(contract["cross_gpu_cuda_tensor_transfer"])
        self.assertEqual(contract["ipc_payload"], "cpu_numpy_latest_wins")
        self.assertFalse(contract["tracking_input_contains_depth"])
        self.assertEqual(contract["shared_runtime_tracking_backend"], "none")
        self.assertFalse(contract["render_waited_for_cotracker"])
        self.assertFalse(contract["render_waited_for_mask"])

    def test_main_dry_run_prints_dual_gpu_contract(self) -> None:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = demo31_runtime.main(
                ["--dry-run", "--camera-ids", "0,1,2", "--mask-gpu", "0", "--cotracker-gpu", "1"],
                cuda_device_count_provider=lambda: 2,
            )

        self.assertEqual(exit_code, 0)
        output = stdout.getvalue()
        self.assertIn("demo = demo3.1", output)
        self.assertIn("dual_gpu_enabled = true", output)
        self.assertIn("main_cuda_visible_devices = 0", output)
        self.assertIn("cotracker_cuda_visible_devices = 1", output)
        self.assertIn("uses_ffs = false", output)
        self.assertIn("cotracker_owner = process", output)
        self.assertIn("cross_gpu_cuda_tensor_transfer = false", output)
        self.assertIn("render_waited_for_cotracker = false", output)

    def test_requires_two_cuda_unless_debug_override(self) -> None:
        args = self._parse(["--dry-run", "--camera-ids", "0,1,2", "--mask-gpu", "0", "--cotracker-gpu", "1"])
        with self.assertRaisesRegex(RuntimeError, "requires at least two CUDA devices"):
            demo31_runtime.validate_args(args, cuda_device_count_provider=lambda: 1)

        debug_args = self._parse(
            [
                "--dry-run",
                "--camera-ids",
                "0,1,2",
                "--mask-gpu",
                "0",
                "--cotracker-gpu",
                "0",
                "--allow-single-gpu-debug",
            ]
        )
        demo31_runtime.validate_args(debug_args, cuda_device_count_provider=lambda: 1)

    def test_same_gpu_fails_without_debug_override(self) -> None:
        args = self._parse(["--dry-run", "--camera-ids", "0,1,2", "--mask-gpu", "0", "--cotracker-gpu", "0"])
        with self.assertRaisesRegex(ValueError, "requires distinct --mask-gpu and --cotracker-gpu"):
            demo31_runtime.validate_args(args, cuda_device_count_provider=lambda: 2)

    def test_no_ffs_depth_source_is_accepted(self) -> None:
        args = self._parse(["--dry-run", "--camera-ids", "0,1,2", "--depth-source", "ffs"])
        with self.assertRaisesRegex(ValueError, "does not support FFS"):
            demo31_runtime.validate_args(args, cuda_device_count_provider=lambda: 2)

    def test_strict_mask_policy_reports_renderer_waits_for_mask(self) -> None:
        args = self._parse(
            [
                "--dry-run",
                "--camera-ids",
                "0,1,2",
                "--mask-gpu",
                "0",
                "--cotracker-gpu",
                "1",
                "--fusion-mask-policy",
                "strict",
            ]
        )
        contract = demo31_runtime.build_contract(args, cuda_device_count_provider=lambda: 2)

        self.assertEqual(contract["fusion_mask_policy"], "strict")
        self.assertTrue(contract["render_waited_for_mask"])

    def test_cotracker_process_config_uses_gpu1(self) -> None:
        args = self._parse(["--dry-run", "--camera-ids", "0,1,2", "--mask-gpu", "0", "--cotracker-gpu", "1"])
        config = demo31_runtime.build_cotracker_process_config(args)

        self.assertEqual(config.cotracker_gpu, "1")
        self.assertEqual(config.camera_ids, (0, 1, 2))
        self.assertEqual(config.cotracker_backend, "cotracker3_online")

    def test_renderer_can_proceed_with_no_cotracker_result(self) -> None:
        self.assertIsNone(
            demo31_runtime.fresh_tracking_result_or_none(
                None,
                now_s=10.0,
                stale_timeout_ms=1500.0,
            )
        )

    def test_renderer_skips_stale_cotracker_result(self) -> None:
        stale = TrackingResultLitePacket(
            group_id=1,
            frame_idx=1,
            source_timestamp_s=8.0,
            publish_timestamp_s=8.0,
            camera_tracks_yx={},
            camera_visibility={},
            query_points_yx={},
            publish_range=(0, 1),
        )
        fresh = TrackingResultLitePacket(
            group_id=2,
            frame_idx=2,
            source_timestamp_s=9.9,
            publish_timestamp_s=9.9,
            camera_tracks_yx={},
            camera_visibility={},
            query_points_yx={},
            publish_range=(1, 2),
        )

        self.assertIsNone(
            demo31_runtime.fresh_tracking_result_or_none(
                stale,
                now_s=10.0,
                stale_timeout_ms=1500.0,
            )
        )
        self.assertIs(
            fresh,
            demo31_runtime.fresh_tracking_result_or_none(
                fresh,
                now_s=10.0,
                stale_timeout_ms=1500.0,
            ),
        )


if __name__ == "__main__":
    unittest.main()
