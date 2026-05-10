from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from demo_v2_2 import runtime as demo
from demo_v2_1_5 import realtime_three_view_async_filtered_fused_pcd as demo215


def _frame(camera_idx: int, *, group_id: int = 7) -> demo.CameraFramePacket:
    depth_u16 = np.array([[0, 100], [250, 500]], dtype=np.uint16) + np.uint16(camera_idx)
    return demo.CameraFramePacket(
        group_id=group_id,
        camera_idx=int(camera_idx),
        frame_seq=int(camera_idx),
        timestamp_ns=int(1_000_000 + camera_idx),
        realsense_timestamp_ms=float(1.0 + camera_idx),
        realsense_frame_number=int(10 + camera_idx),
        timestamp_domain="hardware_clock",
        capture_arrival_perf_ns=int(2_000_000 + camera_idx),
        color_bgr=np.zeros((2, 2, 3), dtype=np.uint8),
        ir_left_u8=None,
        ir_right_u8=None,
        k_color=np.eye(3, dtype=np.float32),
        k_ir_left=None,
        t_ir_left_to_color=None,
        baseline_m=0.0,
        intrinsics=demo.CameraIntrinsics(fx=1.0, fy=1.0, cx=0.0, cy=0.0),
        c2w=np.eye(4, dtype=np.float32),
        depth_u16=depth_u16,
        depth_scale_m_per_unit=0.001,
    )


class DemoV215RealSenseDepthSmoke(unittest.TestCase):
    def test_demo215_wrapper_defaults_to_realsense_async_filter_preset(self) -> None:
        argv = demo215._with_default_preset(["--dry-run"])

        self.assertEqual(argv[:3], ["--preset", demo.PRESET_DEMO215_ASYNC_FILTER_5FPS, "--dry-run"])

    def test_demo215_public_help_is_native_depth_and_hides_ffs_batch_override(self) -> None:
        help_text = demo215.build_arg_parser().format_help()

        self.assertIn("native aligned RealSense", help_text)
        self.assertIn("--warmup-s", help_text)
        self.assertIn("--gpu-sampling", help_text)
        self.assertIn("--gpu-sampling-interval-s", help_text)
        self.assertIn("--min-depth-m", help_text)
        self.assertIn("--experimental-edgetam-batch-vision", help_text)
        self.assertIn("--advanced-help", help_text)
        self.assertIn("--warm-cache-only", help_text)
        self.assertIn("--warm-cache-repeat", help_text)
        self.assertNotIn("--ffs-batch-size", help_text)
        self.assertNotIn("--fusion-target-fps", help_text)
        self.assertNotIn("--gpu-pipeline-mode", help_text)

    def test_demo215_wrapper_imports_dedicated_runtime_boundary(self) -> None:
        source = Path(demo215.__file__).read_text(encoding="utf-8")

        self.assertIn("from demo_v2_2 import runtime", source)
        self.assertNotIn("from demo_v2_1 import realtime_three_view_masked_fused_pcd", source)

    def test_demo215_warm_cache_flags_do_not_pass_to_runtime_parser(self) -> None:
        argv = demo215._to_demo215_argv(
            [
                "--warm-cache-only",
                "--warm-cache-repeat",
                "2",
                "--warm-cache-json-output",
                "docs/generated/demo2_1_5_init_cache_warmup_probe.json",
            ]
        )

        self.assertEqual(argv[:2], ["--preset", demo.PRESET_DEMO215_ASYNC_FILTER_5FPS])
        self.assertNotIn("--warm-cache-only", argv)
        self.assertNotIn("--warm-cache-repeat", argv)
        self.assertNotIn("--warm-cache-json-output", argv)

    def test_demo215_public_cli_aliases_translate_to_runtime_flags(self) -> None:
        argv = demo215._to_demo215_argv(
            [
                "--dry-run",
                "--duration-s",
                "10",
                "--warmup-s",
                "4",
                "--fps",
                "15",
                "--camera-ids",
                "0,1,2",
                "--object-only",
                "--min-depth-m",
                "0.12",
                "--max-depth-m",
                "1.25",
                "--no-parallel-init",
                "--no-compile-prewarm",
                "--experimental-edgetam-batch-vision",
                "--gpu-sampling",
                "--gpu-sampling-device-index",
                "0",
            ]
        )

        self.assertEqual(argv[:2], ["--preset", demo.PRESET_DEMO215_ASYNC_FILTER_5FPS])
        self.assertIn("--profile-warmup-exclude-s", argv)
        self.assertIn("--depth-min-m", argv)
        self.assertIn("--depth-max-m", argv)
        self.assertIn("--track-mode", argv)
        self.assertIn(demo.TRACK_MODE_OBJECT_ONLY, argv)
        self.assertIn("--no-parallel-init", argv)
        self.assertIn("--no-edgetam-prewarm-compile", argv)
        self.assertIn("--edgetam-batch-vision-encoder", argv)
        self.assertIn("--gpu-sampling", argv)
        self.assertIn("--gpu-sampling-device-index", argv)

    def test_demo215_public_experimental_staged_parallel_selects_probe_preset(self) -> None:
        argv = demo215._to_demo215_argv(["--dry-run", "--experimental-staged-parallel"])

        self.assertEqual(argv[:2], ["--preset", demo.PRESET_DEMO215_STAGED_PARALLEL_5FPS])

    def test_demo215_preset_contract_is_realsense_depth_single_owner(self) -> None:
        parser = demo.build_arg_parser()
        args = parser.parse_args(["--dry-run", "--preset", demo.PRESET_DEMO215_ASYNC_FILTER_5FPS])
        args = demo.apply_preset_defaults(args, explicit_options={"--dry-run", "--preset"})
        contract = demo.build_contract(args)

        self.assertEqual(contract["demo"], "demo_2_1_5_realsense_async_filtered_fused_pcd")
        self.assertEqual(contract["preset_canonical"], demo.PRESET_DEMO215_ASYNC_FILTER_5FPS)
        self.assertEqual(contract["fps"], 15)
        self.assertEqual(contract["fusion_target_fps"], 15.0)
        self.assertEqual(contract["capture_group_target_fps"], 15.0)
        self.assertEqual(contract["depth_source"], demo.DEPTH_SOURCE_REALSENSE)
        self.assertFalse(contract["official_quality_depth"])
        self.assertEqual(contract["native_realsense_depth_role"], "primary")
        self.assertEqual(contract["compile_mode"], demo.DEFAULT_COMPILE_MODE)
        self.assertEqual(contract["track_mode"], demo.TRACK_MODE_CONTROLLER_OBJECT)
        self.assertEqual(args.depth_min_m, demo.DEFAULT_DEMO22_DEPTH_MIN_M)
        self.assertEqual(contract["gpu_pipeline"]["mode"], demo.GPU_PIPELINE_MODE_SINGLE_OWNER)
        self.assertEqual(contract["edgetam"]["model_topology"], demo.EDGETAM_MODEL_TOPOLOGY_SHARED)
        self.assertTrue(contract["filter_scheduler"]["render_filtered_only"])
        self.assertEqual(contract["ffs_contract"]["trt_batch_size"], 1)

    def test_demo215_staged_parallel_preset_contract_is_realsense_depth(self) -> None:
        parser = demo.build_arg_parser()
        args = parser.parse_args(["--dry-run", "--preset", demo.PRESET_DEMO215_STAGED_PARALLEL_5FPS])
        args = demo.apply_preset_defaults(args, explicit_options={"--dry-run", "--preset"})
        contract = demo.build_contract(args)

        self.assertEqual(contract["demo"], "demo_2_1_5_realsense_async_filtered_fused_pcd")
        self.assertEqual(contract["preset_canonical"], demo.PRESET_DEMO215_STAGED_PARALLEL_5FPS)
        self.assertEqual(contract["depth_source"], demo.DEPTH_SOURCE_REALSENSE)
        self.assertEqual(contract["gpu_pipeline"]["mode"], demo.GPU_PIPELINE_MODE_STAGED)
        self.assertEqual(contract["edgetam"]["stream_mode"], demo.EDGETAM_STREAM_MODE_PER_CAMERA)
        self.assertTrue(contract["filter_scheduler"]["render_filtered_only"])

    def test_demo215_thread_specs_include_gpu_owner_and_filter(self) -> None:
        parser = demo.build_arg_parser()
        args = parser.parse_args(["--dry-run", "--preset", demo.PRESET_DEMO215_ASYNC_FILTER_5FPS])
        args = demo.apply_preset_defaults(args, explicit_options={"--dry-run", "--preset"})
        runtime = demo.Demo22Runtime(args)
        names = [name for name, _target in runtime._thread_specs()]

        self.assertEqual(names, ["capture-group", "gpu-owner", "fusion", "filter"])

    def test_parallel_init_task_records_start_finish_and_duration(self) -> None:
        parser = demo.build_arg_parser()
        args = parser.parse_args(["--dry-run", "--preset", demo.PRESET_DEMO215_ASYNC_FILTER_5FPS])
        args = demo.apply_preset_defaults(args, explicit_options={"--dry-run", "--preset"})
        runtime = demo.Demo22Runtime(args)

        result = runtime._run_parallel_init_task("unit", lambda: "ok")
        task_profile = runtime._init_profile_snapshot()["parallel_init"]["unit"]

        self.assertEqual(result, "ok")
        self.assertIn("started_s", task_profile)
        self.assertIn("finished_s", task_profile)
        self.assertIn("duration_ms", task_profile)
        self.assertGreaterEqual(task_profile["finished_s"], task_profile["started_s"])
        self.assertGreaterEqual(task_profile["duration_ms"], 0.0)

    def test_demo215_staged_parallel_thread_specs_include_staged_gpu_and_filter(self) -> None:
        parser = demo.build_arg_parser()
        args = parser.parse_args(["--dry-run", "--preset", demo.PRESET_DEMO215_STAGED_PARALLEL_5FPS])
        args = demo.apply_preset_defaults(args, explicit_options={"--dry-run", "--preset"})
        runtime = demo.Demo22Runtime(args)
        names = [name for name, _target in runtime._thread_specs()]

        self.assertEqual(names, ["capture-group", "staged-gpu", "fusion", "filter"])

    def test_realsense_depth_cycle_converts_uint16_to_meter_depth_group(self) -> None:
        parser = demo.build_arg_parser()
        args = parser.parse_args(["--dry-run", "--preset", demo.PRESET_DEMO215_ASYNC_FILTER_5FPS])
        args = demo.apply_preset_defaults(args, explicit_options={"--dry-run", "--preset"})
        runtime = demo.Demo22Runtime(args)
        group = demo.CaptureGroup(
            group_id=7,
            created_perf_s=1.0,
            frames={idx: _frame(idx) for idx in args.camera_ids},
            group_timestamp_ns=1_000_000,
            max_temporal_skew_ms=0.0,
            per_camera_time_offset_ms={idx: 0.0 for idx in args.camera_ids},
            per_camera_frame_seq={idx: idx for idx in args.camera_ids},
            timestamp_source="test",
        )

        depth_group, h2d = runtime._run_depth_cycle_for_group(group=group, runner=None, aligners={})

        self.assertEqual(h2d, {})
        self.assertEqual(depth_group.group_id, 7)
        self.assertEqual(set(depth_group.depths), {0, 1, 2})
        np.testing.assert_allclose(
            depth_group.depths[0].depth_m,
            np.array([[0.0, 0.1], [0.25, 0.5]], dtype=np.float32),
        )
        self.assertEqual(depth_group.per_camera_ms[0]["ffs_ms"], 0.0)
        self.assertGreaterEqual(depth_group.per_camera_ms[0]["realsense_depth_ms"], 0.0)


if __name__ == "__main__":
    unittest.main()
