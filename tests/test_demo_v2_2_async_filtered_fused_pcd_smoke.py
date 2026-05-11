from __future__ import annotations

import contextlib
import io
import threading
import time
import unittest
from pathlib import Path

import numpy as np

from demo_v2_2 import runtime as demo
from demo_v2_2 import realtime_three_view_async_filtered_fused_pcd as demo22


def _raw_packet(seq: int) -> demo.RawFusedPcdPacket:
    raw_object = demo.FusedLayerCloud(
        label="stuffed animal",
        postprocess_mode=demo.POSTPROCESS_NONE,
        points_m=np.array([[0.0, 0.0, 0.4], [0.01, 0.0, 0.4], [0.20, 0.0, 0.4]], dtype=np.float32),
        colors_rgb=np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8),
        per_camera=({"camera_idx": 0, "point_count": 3},),
    )
    raw_controller = demo.FusedLayerCloud(
        label="towel",
        postprocess_mode=demo.POSTPROCESS_NONE,
        points_m=np.array([[0.0, 0.1, 0.4]], dtype=np.float32),
        colors_rgb=np.array([[255, 255, 0]], dtype=np.uint8),
        per_camera=({"camera_idx": 0, "point_count": 1},),
    )
    return demo.RawFusedPcdPacket(
        group_id=int(seq),
        created_perf_s=float(seq),
        raw_object=raw_object,
        raw_controller=raw_controller,
        raw_fusion_ms=2.0,
        build_object_raw_ms=1.0,
        build_controller_raw_ms=1.0,
        object_raw_points=raw_object.point_count,
        controller_raw_points=raw_controller.point_count,
        ffs_cycle_ms=10.0,
        edgetam_ms_by_camera={0: 1.0, 1: 1.0, 2: 1.0},
        ffs_gpu_gate_wait_ms=0.0,
        edgetam_gpu_gate_wait_ms_by_camera={0: 0.0, 1: 0.0, 2: 0.0},
        capture_temporal_skew_ms=0.0,
        capture_time_offsets_ms_by_camera={0: 0.0, 1: 0.0, 2: 0.0},
        timestamp_source="test",
    )


class DemoV22AsyncFilteredFusedPcdSmoke(unittest.TestCase):
    def test_demo22_wrapper_defaults_to_async_filter_preset(self) -> None:
        argv = demo22._with_default_preset(["--dry-run"])

        self.assertEqual(argv[:3], ["--preset", demo.PRESET_DEMO22_ASYNC_FILTER_5FPS, "--dry-run"])

    def test_demo22_public_help_hides_legacy_runtime_knobs(self) -> None:
        help_text = demo22.build_arg_parser().format_help()

        self.assertIn("--warmup-s", help_text)
        self.assertIn("--gpu-sampling", help_text)
        self.assertIn("--gpu-sampling-interval-s", help_text)
        self.assertIn("--min-depth-m", help_text)
        self.assertIn("--experimental-edgetam-batch-vision", help_text)
        self.assertIn("--single-object-batchvision-edgetam", help_text)
        self.assertIn("--edgetam-backend", help_text)
        self.assertIn("--advanced-help", help_text)
        self.assertNotIn("--fusion-target-fps", help_text)
        self.assertNotIn("--gpu-pipeline-mode", help_text)
        self.assertNotIn("--single-owner-order", help_text)

    def test_demo22_wrapper_imports_dedicated_runtime_boundary(self) -> None:
        source = Path(demo22.__file__).read_text(encoding="utf-8")

        self.assertIn("from demo_v2_2 import runtime", source)
        self.assertNotIn("from demo_v2_1 import realtime_three_view_masked_fused_pcd", source)

    def test_demo22_public_cli_aliases_translate_to_runtime_flags(self) -> None:
        argv = demo22._to_demo22_argv(
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
                "--edgetam-backend",
                demo.EDGETAM_BACKEND_HF_BATCH_VISION_SEQ_SESSION,
                "--edgetam-external-path",
                "/home/zhangxinjie/EdgeTAM-HF-batched",
                "--mask-postprocess",
                demo.MASK_POSTPROCESS_CUDA_INLINE,
                "--compile-mode",
                demo.COMPILE_MODE_VISION_REDUCE_OVERHEAD,
                "--filter-mode",
                "async",
                "--gpu-sampling",
                "--gpu-sampling-interval-s",
                "0.25",
                "--gpu-sampling-backend",
                "nvml",
                "--ffs-batch-size",
                "1",
            ]
        )

        self.assertEqual(argv[:2], ["--preset", demo.PRESET_DEMO22_ASYNC_FILTER_5FPS])
        self.assertIn("--profile-warmup-exclude-s", argv)
        self.assertIn("--depth-min-m", argv)
        self.assertIn("--depth-max-m", argv)
        self.assertIn("--track-mode", argv)
        self.assertIn(demo.TRACK_MODE_OBJECT_ONLY, argv)
        self.assertIn("--no-parallel-init", argv)
        self.assertIn("--no-edgetam-prewarm-compile", argv)
        self.assertIn("--edgetam-batch-vision-encoder", argv)
        self.assertIn("--edgetam-backend", argv)
        self.assertIn(demo.EDGETAM_BACKEND_HF_BATCH_VISION_SEQ_SESSION, argv)
        self.assertIn("--edgetam-external-path", argv)
        self.assertIn("/home/zhangxinjie/EdgeTAM-HF-batched", argv)
        self.assertIn("--mask-postprocess", argv)
        self.assertIn(demo.MASK_POSTPROCESS_CUDA_INLINE, argv)
        self.assertIn("--compile-mode", argv)
        self.assertIn(demo.COMPILE_MODE_VISION_REDUCE_OVERHEAD, argv)
        self.assertIn("--pcd-filter-mode", argv)
        self.assertIn("async", argv)
        self.assertIn("--gpu-sampling", argv)
        self.assertIn("--gpu-sampling-interval-s", argv)
        self.assertIn("0.25", argv)
        self.assertIn("--gpu-sampling-backend", argv)
        self.assertIn("nvml", argv)
        self.assertIn("--ffs-trt-batch-size", argv)
        self.assertIn("1", argv)

    def test_demo22_gpu_sampling_rejects_nvidia_smi_backend(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            demo22._to_demo22_argv(["--gpu-sampling-backend", "nvidia-smi"])

    def test_demo22_runtime_gpu_sampling_backend_is_nvml_only(self) -> None:
        self.assertEqual(demo.GPU_SAMPLING_BACKENDS, ("nvml",))

    def test_demo22_public_cli_still_passes_legacy_flags_through(self) -> None:
        argv = demo22._to_demo22_argv(["--dry-run", "--gpu-pipeline-mode", demo.GPU_PIPELINE_MODE_SINGLE_OWNER])

        self.assertEqual(argv[:2], ["--preset", demo.PRESET_DEMO22_ASYNC_FILTER_5FPS])
        self.assertIn("--gpu-pipeline-mode", argv)
        self.assertIn(demo.GPU_PIPELINE_MODE_SINGLE_OWNER, argv)

    def test_demo22_public_experimental_staged_parallel_selects_probe_preset(self) -> None:
        argv = demo22._to_demo22_argv(["--dry-run", "--experimental-staged-parallel"])

        self.assertEqual(argv[:2], ["--preset", demo.PRESET_DEMO22_STAGED_PARALLEL_5FPS])

    def test_demo22_preset_contract_is_filtered_only_single_owner(self) -> None:
        parser = demo.build_arg_parser()
        args = parser.parse_args(["--dry-run", "--preset", demo.PRESET_DEMO22_ASYNC_FILTER_5FPS])
        args = demo.apply_preset_defaults(args, explicit_options={"--dry-run", "--preset"})
        contract = demo.build_contract(args)

        self.assertEqual(contract["demo"], "demo_2_2_async_filtered_fused_pcd")
        self.assertEqual(contract["preset_canonical"], demo.PRESET_DEMO22_ASYNC_FILTER_5FPS)
        self.assertEqual(contract["fps"], 15)
        self.assertEqual(contract["fusion_target_fps"], 15.0)
        self.assertEqual(contract["capture_group_target_fps"], 15.0)
        self.assertFalse(contract["gpu_sampling"]["enabled"])
        self.assertEqual(contract["depth_source"], demo.DEPTH_SOURCE_FFS)
        self.assertEqual(contract["compile_mode"], demo.DEFAULT_COMPILE_MODE)
        self.assertEqual(contract["track_mode"], demo.TRACK_MODE_CONTROLLER_OBJECT)
        self.assertEqual(args.depth_min_m, demo.DEFAULT_DEMO22_DEPTH_MIN_M)
        self.assertEqual(contract["init"]["mode"], "sam31-first-frame")
        self.assertTrue(contract["init"]["parallel_init"])
        self.assertTrue(contract["init"]["sam31_cache_init_model"])
        self.assertTrue(contract["init"]["sam31_keep_runtime_until_all_cameras_init"])
        self.assertEqual(contract["edgetam"]["model_topology"], demo.EDGETAM_MODEL_TOPOLOGY_SHARED)
        self.assertTrue(contract["edgetam"]["prewarm_compile"])
        self.assertEqual(contract["edgetam"]["prewarm_runs"], 1)
        self.assertFalse(contract["edgetam"]["batch_vision_encoder"])
        self.assertEqual(contract["gpu_pipeline"]["mode"], demo.GPU_PIPELINE_MODE_SINGLE_OWNER)
        self.assertEqual(contract["gpu_pipeline"]["internal_order"], demo.SINGLE_OWNER_ORDER_FFS_THEN_EDGETAM)
        self.assertEqual(contract["ffs_contract"]["trt_batch_size"], 3)
        self.assertTrue(contract["ffs_contract"]["batch3_isolated_artifact"])
        self.assertEqual(
            Path(contract["ffs_contract"]["trt_model_dir"]),
            demo.DEFAULT_FFS_TRT_BATCH3_TWO_STAGE_MODEL_DIR,
        )
        self.assertEqual(contract["filter_scheduler"]["mode"], "async")
        self.assertTrue(contract["filter_scheduler"]["render_filtered_only"])
        self.assertFalse(contract["filter_scheduler"]["render_accepts_raw_fused_pcd"])
        self.assertEqual([layer["label"] for layer in contract["semantic_layers"]], ["towel", "stuffed animal"])

    def test_demo22_preset_allows_explicit_batch1_rollback(self) -> None:
        parser = demo.build_arg_parser()
        args = parser.parse_args(
            [
                "--dry-run",
                "--preset",
                demo.PRESET_DEMO22_ASYNC_FILTER_5FPS,
                "--ffs-trt-batch-size",
                "1",
            ]
        )
        args = demo.apply_preset_defaults(
            args,
            explicit_options={"--dry-run", "--preset", "--ffs-trt-batch-size"},
        )
        contract = demo.build_contract(args)

        self.assertEqual(contract["ffs_contract"]["trt_batch_size"], 1)
        self.assertFalse(contract["ffs_contract"]["batch3_isolated_artifact"])
        self.assertEqual(
            Path(contract["ffs_contract"]["trt_model_dir"]),
            demo.DEFAULT_FFS_TRT_TWO_STAGE_MODEL_DIR,
        )

    def test_demo22_batch_vision_option_is_explicit(self) -> None:
        parser = demo.build_arg_parser()
        args = parser.parse_args(
            [
                "--dry-run",
                "--preset",
                demo.PRESET_DEMO22_ASYNC_FILTER_5FPS,
                "--edgetam-batch-vision-encoder",
            ]
        )
        args = demo.apply_preset_defaults(
            args,
            explicit_options={"--dry-run", "--preset", "--edgetam-batch-vision-encoder"},
        )
        contract = demo.build_contract(args)

        self.assertTrue(contract["edgetam"]["batch_vision_encoder"])
        self.assertEqual(contract["edgetam"]["batch_vision_batch_size"], 3)
        self.assertEqual(contract["edgetam"]["model_topology"], demo.EDGETAM_MODEL_TOPOLOGY_SHARED)
        self.assertEqual(contract["gpu_pipeline"]["mode"], demo.GPU_PIPELINE_MODE_SINGLE_OWNER)

    def test_demo22_single_object_batchvision_preset_contract(self) -> None:
        parser = demo.build_arg_parser()
        args = parser.parse_args(["--dry-run", "--preset", demo.PRESET_DEMO22_SINGLE_OBJECT_BATCHVISION_EDGETAM])
        args = demo.apply_preset_defaults(args, explicit_options={"--dry-run", "--preset"})
        contract = demo.build_contract(args)

        self.assertEqual(contract["demo"], "demo_2_2_async_filtered_fused_pcd")
        self.assertEqual(contract["preset_canonical"], demo.PRESET_DEMO22_SINGLE_OBJECT_BATCHVISION_EDGETAM)
        self.assertEqual(contract["track_mode"], demo.TRACK_MODE_OBJECT_ONLY)
        self.assertEqual(contract["compile_mode"], demo.COMPILE_MODE_VISION_REDUCE_OVERHEAD)
        self.assertEqual(contract["mask_postprocess"], demo.MASK_POSTPROCESS_CUDA_INLINE)
        self.assertEqual(contract["edgetam"]["backend"], demo.EDGETAM_BACKEND_HF_BATCH_VISION_SEQ_SESSION)
        self.assertTrue(contract["edgetam"]["hf_batch_vision_seq_session"])
        self.assertFalse(contract["edgetam"]["true_batched_multisession_runtime"])
        self.assertTrue(contract["edgetam"]["batch_vision_encoder"])
        self.assertEqual(contract["edgetam"]["batch_vision_batch_size"], 3)
        self.assertEqual(contract["edgetam"]["model_topology"], demo.EDGETAM_MODEL_TOPOLOGY_SHARED)
        self.assertEqual(contract["gpu_pipeline"]["mode"], demo.GPU_PIPELINE_MODE_SINGLE_OWNER)
        self.assertEqual(contract["depth_source"], demo.DEPTH_SOURCE_FFS)
        self.assertEqual([layer["label"] for layer in contract["semantic_layers"]], ["stuffed animal"])

    def test_demo22_single_object_batchvision_public_alias(self) -> None:
        argv = demo22._to_demo22_argv(
            [
                "--dry-run",
                "--single-object-batchvision-edgetam",
                "--edgetam-external-path",
                "/home/zhangxinjie/EdgeTAM-HF-batched",
            ]
        )

        self.assertEqual(argv[:2], ["--preset", demo.PRESET_DEMO22_SINGLE_OBJECT_BATCHVISION_EDGETAM])
        self.assertIn("--edgetam-external-path", argv)
        self.assertIn("/home/zhangxinjie/EdgeTAM-HF-batched", argv)

    def test_demo22_parallel_edgetam_alias_selects_batchvision_backend(self) -> None:
        argv = demo22._to_demo22_argv(["--dry-run", "--parallel-edgetam"])

        self.assertEqual(argv[:2], ["--preset", demo.PRESET_DEMO22_ASYNC_FILTER_5FPS])
        self.assertIn("--edgetam-backend", argv)
        self.assertIn(demo.EDGETAM_BACKEND_HF_BATCH_VISION_SEQ_SESSION, argv)
        self.assertIn("--edgetam-batch-vision-encoder", argv)

    def test_explicit_batchvision_backend_resolves_contract_batch_encoder(self) -> None:
        parser = demo.build_arg_parser()
        args = parser.parse_args(
            [
                "--dry-run",
                "--edgetam-backend",
                demo.EDGETAM_BACKEND_HF_BATCH_VISION_SEQ_SESSION,
            ]
        )
        args = demo.apply_preset_defaults(
            args,
            explicit_options={"--dry-run", "--edgetam-backend"},
        )
        contract = demo.build_contract(args)

        self.assertEqual(contract["edgetam"]["backend"], demo.EDGETAM_BACKEND_HF_BATCH_VISION_SEQ_SESSION)
        self.assertTrue(contract["edgetam"]["batch_vision_encoder"])

    def test_batch_vision_rejects_staged_parallel_contract(self) -> None:
        parser = demo.build_arg_parser()
        args = parser.parse_args(
            [
                "--dry-run",
                "--preset",
                demo.PRESET_DEMO22_STAGED_PARALLEL_5FPS,
                "--edgetam-batch-vision-encoder",
            ]
        )
        args = demo.apply_preset_defaults(
            args,
            explicit_options={"--dry-run", "--preset", "--edgetam-batch-vision-encoder"},
        )
        runtime = demo.Demo22Runtime(args)

        with self.assertRaisesRegex(RuntimeError, "batch-vision-encoder requires single-owner"):
            runtime._validate_live_contract()

    def test_demo22_staged_parallel_preset_contract(self) -> None:
        parser = demo.build_arg_parser()
        args = parser.parse_args(["--dry-run", "--preset", demo.PRESET_DEMO22_STAGED_PARALLEL_5FPS])
        args = demo.apply_preset_defaults(args, explicit_options={"--dry-run", "--preset"})
        contract = demo.build_contract(args)

        self.assertEqual(contract["demo"], "demo_2_2_async_filtered_fused_pcd")
        self.assertEqual(contract["preset_canonical"], demo.PRESET_DEMO22_STAGED_PARALLEL_5FPS)
        self.assertEqual(contract["fps"], 15)
        self.assertEqual(contract["fusion_target_fps"], 15.0)
        self.assertEqual(contract["capture_group_target_fps"], 15.0)
        self.assertEqual(contract["depth_source"], demo.DEPTH_SOURCE_FFS)
        self.assertEqual(contract["compile_mode"], demo.COMPILE_MODE_VISION_DEFAULT)
        self.assertEqual(contract["track_mode"], demo.TRACK_MODE_CONTROLLER_OBJECT)
        self.assertEqual(args.depth_min_m, demo.DEFAULT_DEMO22_DEPTH_MIN_M)
        self.assertTrue(contract["init"]["parallel_init"])
        self.assertEqual(contract["gpu_pipeline"]["mode"], demo.GPU_PIPELINE_MODE_STAGED)
        self.assertEqual(contract["gpu_pipeline"]["internal_order"], demo.STAGED_ORDER_FFS_THEN_PARALLEL_EDGETAM)
        self.assertEqual(contract["gpu_pipeline"]["ffs_stage"], "sequential_cam0_cam1_cam2")
        self.assertEqual(contract["gpu_pipeline"]["edgetam_stage"], "parallel_cam0_cam1_cam2")
        self.assertEqual(contract["edgetam"]["model_topology"], demo.EDGETAM_MODEL_TOPOLOGY_REPLICATED)
        self.assertEqual(contract["edgetam"]["stream_mode"], demo.EDGETAM_STREAM_MODE_PER_CAMERA)
        self.assertTrue(contract["edgetam"]["prewarm_compile"])
        self.assertEqual(contract["edgetam"]["prewarm_runs"], 1)
        self.assertEqual(contract["h2d_transfer"]["pin_memory_mode"], demo.PIN_MEMORY_MODE_ALL)
        self.assertTrue(contract["h2d_transfer"]["edge_pin_enabled"])
        self.assertTrue(contract["h2d_transfer"]["ffs_pin_requested"])
        self.assertEqual(contract["h2d_transfer"]["h2d_stream_mode"], demo.H2D_STREAM_MODE_DEDICATED)
        self.assertTrue(contract["memory_for_speed"]["static_device_buffers"])
        self.assertTrue(contract["memory_for_speed"]["ffs_reusable_cuda_input_buffers"])
        self.assertTrue(contract["memory_for_speed"]["edgetam_reusable_cuda_pixel_slots"])
        self.assertTrue(contract["filter_scheduler"]["render_filtered_only"])

    def test_demo22_depth_min_m_preserves_explicit_override(self) -> None:
        parser = demo.build_arg_parser()
        args = parser.parse_args(
            [
                "--dry-run",
                "--preset",
                demo.PRESET_DEMO22_ASYNC_FILTER_5FPS,
                "--depth-min-m",
                "0.25",
            ]
        )
        args = demo.apply_preset_defaults(
            args,
            explicit_options={"--dry-run", "--preset", "--depth-min-m"},
        )

        self.assertEqual(args.depth_min_m, 0.25)

    def test_demo22_thread_specs_include_filter_worker(self) -> None:
        parser = demo.build_arg_parser()
        args = parser.parse_args(["--dry-run", "--preset", demo.PRESET_DEMO22_ASYNC_FILTER_5FPS])
        args = demo.apply_preset_defaults(args, explicit_options={"--dry-run", "--preset"})
        runtime = demo.Demo22Runtime(args)
        names = [name for name, _target in runtime._thread_specs()]

        self.assertEqual(names, ["capture-group", "gpu-owner", "fusion", "filter"])

    def test_demo22_staged_parallel_thread_specs_include_staged_gpu_and_filter(self) -> None:
        parser = demo.build_arg_parser()
        args = parser.parse_args(["--dry-run", "--preset", demo.PRESET_DEMO22_STAGED_PARALLEL_5FPS])
        args = demo.apply_preset_defaults(args, explicit_options={"--dry-run", "--preset"})
        runtime = demo.Demo22Runtime(args)
        names = [name for name, _target in runtime._thread_specs()]

        self.assertEqual(names, ["capture-group", "staged-gpu", "fusion", "filter"])

    def test_raw_fused_packet_is_not_published_to_render_slot(self) -> None:
        parser = demo.build_arg_parser()
        args = parser.parse_args(["--dry-run", "--preset", demo.PRESET_DEMO22_ASYNC_FILTER_5FPS])
        args = demo.apply_preset_defaults(args, explicit_options={"--dry-run", "--preset"})
        runtime = demo.Demo22Runtime(args)

        runtime._publish_raw_fused_for_async_filter(_raw_packet(1))

        self.assertEqual(runtime.raw_fused_slot.latest_seq(), 1)
        self.assertEqual(runtime.render_slot.latest_seq(), -1)

    def test_filter_output_is_the_only_render_packet(self) -> None:
        parser = demo.build_arg_parser()
        args = parser.parse_args(["--dry-run", "--preset", demo.PRESET_DEMO22_ASYNC_FILTER_5FPS])
        args = demo.apply_preset_defaults(args, explicit_options={"--dry-run", "--preset"})
        runtime = demo.Demo22Runtime(args)
        raw = _raw_packet(2)

        packet = runtime._filter_raw_fused_packet(raw)
        runtime.render_slot.put(packet)

        self.assertEqual(runtime.render_slot.get_latest_after(-1).seq, 2)  # type: ignore[union-attr]
        self.assertEqual(raw.object_raw_points, 3)
        self.assertEqual(packet.object_point_count, 3)
        self.assertEqual(packet.controller_point_count, 1)

    def test_raw_slot_latest_wins_replaces_pending_work(self) -> None:
        parser = demo.build_arg_parser()
        args = parser.parse_args(["--dry-run", "--preset", demo.PRESET_DEMO22_ASYNC_FILTER_5FPS])
        args = demo.apply_preset_defaults(args, explicit_options={"--dry-run", "--preset"})
        runtime = demo.Demo22Runtime(args)

        runtime.raw_fused_slot.put(_raw_packet(1))
        runtime.raw_fused_slot.put(_raw_packet(2))

        self.assertEqual(runtime.raw_fused_slot.dropped_count, 1)
        self.assertEqual(runtime.raw_fused_slot.get_latest_after(-1).seq, 2)  # type: ignore[union-attr]

    def test_async_filter_worker_publishes_filtered_latest_packet(self) -> None:
        parser = demo.build_arg_parser()
        args = parser.parse_args(["--dry-run", "--preset", demo.PRESET_DEMO22_ASYNC_FILTER_5FPS])
        args = demo.apply_preset_defaults(args, explicit_options={"--dry-run", "--preset"})
        runtime = demo.Demo22Runtime(args)
        worker = threading.Thread(target=runtime._async_filter_worker, daemon=True)
        worker.start()
        try:
            runtime._publish_raw_fused_for_async_filter(_raw_packet(3))
            deadline_s = time.perf_counter() + 1.0
            while runtime.render_slot.latest_seq() < 3 and time.perf_counter() < deadline_s:
                time.sleep(0.001)
        finally:
            runtime.stop_event.set()
            worker.join(timeout=1.0)

        self.assertEqual(runtime.render_slot.latest_seq(), 3)
        self.assertEqual(runtime.filter_output_stats.fps, 0.0)

    def test_profile_summary_separates_raw_filter_and_render_fps(self) -> None:
        parser = demo.build_arg_parser()
        args = parser.parse_args(["--dry-run", "--preset", demo.PRESET_DEMO22_ASYNC_FILTER_5FPS, "--profile-pipeline"])
        args = demo.apply_preset_defaults(args, explicit_options={"--dry-run", "--preset", "--profile-pipeline"})
        runtime = demo.Demo22Runtime(args)
        records = [
            {
                "group_id": 0,
                "t_group_created": 0.0,
                "complete": True,
                "raw_fusion": {"publish_s": 0.0, "total_ms": 2.0},
                "filter": {"publish_s": 0.1, "total_ms": 5.0},
                "fusion": {"publish_s": 0.1, "total_ms": 7.0},
                "render": {"render_s": 0.2},
            },
            {
                "group_id": 1,
                "t_group_created": 0.2,
                "complete": True,
                "raw_fusion": {"publish_s": 0.2, "total_ms": 2.0},
                "filter": {"publish_s": 0.3, "total_ms": 5.0},
                "fusion": {"publish_s": 0.3, "total_ms": 7.0},
                "render": {"render_s": 0.4},
            },
        ]

        summary = runtime._profile_summary_for_records(records)

        self.assertAlmostEqual(summary["raw_fusion_fps"], 5.0)
        self.assertAlmostEqual(summary["filter_output_fps"], 5.0)
        self.assertAlmostEqual(summary["render_fps"], 5.0)

    def test_profile_payload_includes_init_profile_breakdown(self) -> None:
        parser = demo.build_arg_parser()
        args = parser.parse_args(["--dry-run", "--preset", demo.PRESET_DEMO22_ASYNC_FILTER_5FPS, "--profile-pipeline"])
        args = demo.apply_preset_defaults(args, explicit_options={"--dry-run", "--preset", "--profile-pipeline"})
        runtime = demo.Demo22Runtime(args)

        runtime._init_profile_set(("camera_startup_ms",), 12.0)
        runtime._init_profile_update(("sam31", "cam0"), {"total_ms": 3.0})
        runtime._init_profile_add(("edgetam", "model_load_ms_total"), 4.0)
        payload = runtime._build_profile_payload()

        self.assertEqual(payload["init_profile"]["camera_startup_ms"], 12.0)
        self.assertEqual(payload["init_profile"]["sam31"]["cam0"]["total_ms"], 3.0)
        self.assertEqual(payload["init_profile"]["edgetam"]["model_load_ms_total"], 4.0)

    def test_gpu_sample_summary_filters_by_warmup(self) -> None:
        samples = [
            {"sample_s": 1.0, "gpu_util_pct": 10.0, "memory_used_mb": 100.0},
            {"sample_s": 20.0, "gpu_util_pct": 40.0, "memory_used_mb": 200.0},
            {"sample_s": 21.0, "gpu_util_pct": 80.0, "memory_used_mb": 300.0},
        ]

        summary = demo.summarize_gpu_samples(samples, start_s=20.0)

        self.assertEqual(summary["sample_count"], 2)
        self.assertEqual(summary["metrics"]["gpu_util_pct"]["median"], 60.0)
        self.assertEqual(summary["metrics"]["gpu_util_pct"]["max"], 80.0)
        self.assertEqual(summary["metrics"]["memory_used_mb"]["max"], 300.0)

    def test_profile_payload_includes_gpu_sampling_summary(self) -> None:
        class FakeSampler:
            def samples_snapshot(self) -> list[dict[str, float]]:
                return [
                    {"sample_s": 1.0, "gpu_util_pct": 5.0, "power_w": 20.0},
                    {"sample_s": 25.0, "gpu_util_pct": 75.0, "power_w": 180.0},
                ]

            def diagnostics(self) -> dict[str, object]:
                return {
                    "enabled": True,
                    "requested_backend": "nvml",
                    "backend_used": "nvml",
                    "device_index": 0,
                    "interval_s": 0.5,
                    "sample_count": 2,
                    "errors": [],
                }

        parser = demo.build_arg_parser()
        args = parser.parse_args(
            [
                "--dry-run",
                "--preset",
                demo.PRESET_DEMO22_ASYNC_FILTER_5FPS,
                "--gpu-sampling",
                "--profile-warmup-exclude-s",
                "20",
            ]
        )
        args = demo.apply_preset_defaults(
            args,
            explicit_options={"--dry-run", "--preset", "--gpu-sampling", "--profile-warmup-exclude-s"},
        )
        runtime = demo.Demo22Runtime(args)
        runtime._gpu_sampler = FakeSampler()  # type: ignore[assignment]

        payload = runtime._build_profile_payload()

        self.assertTrue(payload["gpu_sampling"]["enabled"])
        self.assertEqual(payload["gpu_sampling"]["summary_after_warmup"]["sample_count"], 1)
        self.assertEqual(
            payload["gpu_sampling"]["summary_after_warmup"]["metrics"]["gpu_util_pct"]["median"],
            75.0,
        )


if __name__ == "__main__":
    unittest.main()
