from __future__ import annotations

import contextlib
import io
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
        self.assertIn("--parallel-edgetam", help_text)
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

        preset_idx = argv.index("--preset")
        self.assertEqual(argv[preset_idx + 1], demo.PRESET_DEMO215_ASYNC_FILTER_5FPS)
        self.assertNotIn(demo.PRESET_DEMO215_STAGED_PARALLEL_5FPS, argv)
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

        preset_idx = argv.index("--preset")
        self.assertEqual(argv[preset_idx + 1], demo.PRESET_DEMO215_ASYNC_FILTER_5FPS)
        self.assertNotIn(demo.PRESET_DEMO215_STAGED_PARALLEL_5FPS, argv)
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

    def test_demo215_gpu_sampling_rejects_nvidia_smi_backend(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            demo215._to_demo215_argv(["--gpu-sampling-backend", "nvidia-smi"])

    def test_demo215_runtime_gpu_sampling_backend_is_nvml_only(self) -> None:
        self.assertEqual(demo.GPU_SAMPLING_BACKENDS, ("nvml",))

    def test_demo215_public_experimental_staged_parallel_selects_probe_preset(self) -> None:
        argv = demo215._to_demo215_argv(["--dry-run", "--experimental-staged-parallel"])

        self.assertEqual(argv[:2], ["--preset", demo.PRESET_DEMO215_STAGED_PARALLEL_5FPS])

    def test_demo215_public_parallel_edgetam_selects_compiled_worker_preset(self) -> None:
        argv = demo215._to_demo215_argv(["--dry-run", "--parallel-edgetam"])

        self.assertEqual(argv[:2], ["--preset", demo.PRESET_DEMO215_COMPILED_PARALLEL_EDGETAM_5FPS])
        self.assertIn("--dry-run", argv)

    def test_demo215_public_parallel_edgetam_can_disable_compile(self) -> None:
        argv = demo215._to_demo215_argv(["--dry-run", "--parallel-edgetam", "--no-compile-edgetam"])

        self.assertEqual(argv[:2], ["--preset", demo.PRESET_DEMO215_COMPILED_PARALLEL_EDGETAM_5FPS])
        compile_idx = argv.index("--compile-mode")
        self.assertEqual(argv[compile_idx + 1], demo.COMPILE_MODE_NONE)

    def test_demo215_public_profile_and_postprocess_flags_translate(self) -> None:
        argv = demo215._to_demo215_argv(
            [
                "--dry-run",
                "--mask-only-debug",
                "--compile-mode",
                demo.COMPILE_MODE_COMPONENTS_REDUCE_OVERHEAD,
                "--dtype",
                "float16",
                "--mask-postprocess",
                demo.MASK_POSTPROCESS_CUDA_INLINE,
                "--profile-cuda-events",
                "--profile-sync",
                "--profile-edgetam-stages",
                "--profile-nsys-markers",
            ]
        )

        self.assertEqual(argv[:2], ["--preset", demo.PRESET_DEMO215_MASK_ONLY_DEBUG])
        self.assertIn("--profile-cuda-events", argv)
        self.assertIn("--profile-sync", argv)
        self.assertIn("--profile-edgetam-stages", argv)
        self.assertIn("--profile-nsys-markers", argv)
        self.assertEqual(argv[argv.index("--compile-mode") + 1], demo.COMPILE_MODE_COMPONENTS_REDUCE_OVERHEAD)
        self.assertEqual(argv[argv.index("--dtype") + 1], "float16")
        self.assertEqual(argv[argv.index("--mask-postprocess") + 1], demo.MASK_POSTPROCESS_CUDA_INLINE)

    def test_demo215_public_clear_presets_translate(self) -> None:
        self.assertEqual(
            demo215._to_demo215_argv(["--dry-run", "--live-fast-native"])[:2],
            ["--preset", demo.PRESET_DEMO215_LIVE_FAST_NATIVE],
        )
        self.assertEqual(
            demo215._to_demo215_argv(["--dry-run", "--live-quality-ffs"])[:2],
            ["--preset", demo.PRESET_DEMO215_LIVE_QUALITY_FFS],
        )
        self.assertEqual(
            demo215._to_demo215_argv(["--dry-run", "--mask-only-debug"])[:2],
            ["--preset", demo.PRESET_DEMO215_MASK_ONLY_DEBUG],
        )

    def test_demo215_public_parallel_edgetam_preserves_explicit_runtime_preset(self) -> None:
        argv = demo215._to_demo215_argv(
            [
                "--dry-run",
                "--parallel-edgetam",
                "--preset",
                demo.PRESET_DEMO215_ASYNC_FILTER_5FPS,
            ]
        )

        preset_idx = argv.index("--preset")
        self.assertEqual(argv[preset_idx + 1], demo.PRESET_DEMO215_ASYNC_FILTER_5FPS)
        self.assertNotIn(demo.PRESET_DEMO215_COMPILED_PARALLEL_EDGETAM_5FPS, argv)
        self.assertNotIn(demo.PRESET_DEMO215_STAGED_PARALLEL_5FPS, argv)

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
        self.assertEqual(contract["compile_mode"], demo.COMPILE_MODE_VISION_DEFAULT)
        self.assertEqual(contract["gpu_pipeline"]["mode"], demo.GPU_PIPELINE_MODE_STAGED)
        self.assertEqual(contract["edgetam"]["stream_mode"], demo.EDGETAM_STREAM_MODE_PER_CAMERA)
        self.assertTrue(contract["filter_scheduler"]["render_filtered_only"])

    def test_demo215_compiled_parallel_edgetam_contract_is_worker_path(self) -> None:
        parser = demo.build_arg_parser()
        args = parser.parse_args(
            ["--dry-run", "--preset", demo.PRESET_DEMO215_COMPILED_PARALLEL_EDGETAM_5FPS]
        )
        args = demo.apply_preset_defaults(args, explicit_options={"--dry-run", "--preset"})
        contract = demo.build_contract(args)

        self.assertEqual(contract["demo"], "demo_2_1_5_realsense_async_filtered_fused_pcd")
        self.assertEqual(contract["preset_canonical"], demo.PRESET_DEMO215_COMPILED_PARALLEL_EDGETAM_5FPS)
        self.assertEqual(contract["depth_source"], demo.DEPTH_SOURCE_REALSENSE)
        self.assertEqual(contract["compile_mode"], demo.DEFAULT_COMPILE_MODE)
        self.assertEqual(contract["gpu_pipeline"]["mode"], demo.GPU_PIPELINE_MODE_SEPARATE_WORKERS)
        self.assertTrue(contract["gpu_pipeline"]["separate_ffs_and_edgetam_workers"])
        self.assertEqual(contract["gpu_gate"]["mode"], "off")
        self.assertEqual(contract["edgetam"]["model_topology"], demo.EDGETAM_MODEL_TOPOLOGY_REPLICATED)
        self.assertEqual(contract["edgetam"]["stream_mode"], demo.EDGETAM_STREAM_MODE_PER_CAMERA)
        self.assertFalse(contract["edgetam"]["prewarm_compile"])
        self.assertTrue(contract["edgetam"]["serialize_first_compiled_forward"])
        self.assertTrue(contract["memory_for_speed"]["models_loaded_once_per_worker"])
        self.assertTrue(contract["h2d_transfer"]["edge_pin_enabled"])
        self.assertEqual(contract["h2d_transfer"]["pin_memory_mode"], "edge")
        self.assertEqual(contract["h2d_transfer"]["pinned_ring_size"], 1)
        self.assertTrue(contract["filter_scheduler"]["render_filtered_only"])

    def test_demo215_parallel_edgetam_contract_accepts_eager_mode(self) -> None:
        parser = demo.build_arg_parser()
        args = parser.parse_args(
            [
                "--dry-run",
                "--preset",
                demo.PRESET_DEMO215_COMPILED_PARALLEL_EDGETAM_5FPS,
                "--compile-mode",
                demo.COMPILE_MODE_NONE,
            ]
        )
        args = demo.apply_preset_defaults(
            args,
            explicit_options={"--dry-run", "--preset", "--compile-mode"},
        )
        contract = demo.build_contract(args)

        self.assertEqual(contract["compile_mode"], demo.COMPILE_MODE_NONE)
        self.assertEqual(contract["gpu_pipeline"]["mode"], demo.GPU_PIPELINE_MODE_SEPARATE_WORKERS)
        self.assertFalse(contract["edgetam"]["prewarm_compile"])
        self.assertFalse(contract["edgetam"]["serialize_first_compiled_forward"])
        self.assertTrue(contract["filter_scheduler"]["render_filtered_only"])

    def test_demo215_mask_only_debug_contract_is_mask_only(self) -> None:
        parser = demo.build_arg_parser()
        args = parser.parse_args(
            [
                "--dry-run",
                "--preset",
                demo.PRESET_DEMO215_MASK_ONLY_DEBUG,
                "--mask-postprocess",
                demo.MASK_POSTPROCESS_CUDA_INLINE,
                "--profile-edgetam-stages",
            ]
        )
        args = demo.apply_preset_defaults(
            args,
            explicit_options={"--dry-run", "--preset", "--mask-postprocess", "--profile-edgetam-stages"},
        )
        contract = demo.build_contract(args)

        self.assertEqual(contract["depth_source"], "none")
        self.assertEqual(contract["render_mode"], "none")
        self.assertEqual(contract["mask_postprocess"], demo.MASK_POSTPROCESS_CUDA_INLINE)
        self.assertTrue(contract["profiling"]["profile_edgetam_stages"])
        self.assertEqual(contract["edgetam"]["active_object_ids"], [1, 2])

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

    def test_demo215_compiled_parallel_edgetam_thread_specs_are_parallel_workers(self) -> None:
        parser = demo.build_arg_parser()
        args = parser.parse_args(
            ["--dry-run", "--preset", demo.PRESET_DEMO215_COMPILED_PARALLEL_EDGETAM_5FPS]
        )
        args = demo.apply_preset_defaults(args, explicit_options={"--dry-run", "--preset"})
        runtime = demo.Demo22Runtime(args)
        names = [name for name, _target in runtime._thread_specs()]

        self.assertEqual(
            names,
            [
                "capture-group",
                "realsense-depth",
                "edgetam-cam0",
                "edgetam-cam1",
                "edgetam-cam2",
                "fusion",
                "filter",
            ],
        )

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
