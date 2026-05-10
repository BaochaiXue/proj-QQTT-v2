from __future__ import annotations

import threading
import time
import unittest

import numpy as np

from demo_v2_1 import realtime_three_view_masked_fused_pcd as demo
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

        self.assertEqual(argv[:3], ["--preset", demo.PRESET_DEMO22_STAGED_PARALLEL_5FPS, "--dry-run"])

    def test_demo22_preset_contract_is_filtered_only_single_owner(self) -> None:
        parser = demo.build_arg_parser()
        args = parser.parse_args(["--dry-run", "--preset", demo.PRESET_DEMO22_ASYNC_FILTER_5FPS])
        args = demo.apply_preset_defaults(args, explicit_options={"--dry-run", "--preset"})
        contract = demo.build_contract(args)

        self.assertEqual(contract["demo"], "demo_2_2_async_filtered_fused_pcd")
        self.assertEqual(contract["preset_canonical"], demo.PRESET_DEMO22_ASYNC_FILTER_5FPS)
        self.assertEqual(contract["fps"], 5)
        self.assertEqual(contract["fusion_target_fps"], 5.0)
        self.assertEqual(contract["depth_source"], demo.DEPTH_SOURCE_FFS)
        self.assertEqual(contract["compile_mode"], demo.DEFAULT_COMPILE_MODE)
        self.assertEqual(contract["track_mode"], demo.TRACK_MODE_CONTROLLER_OBJECT)
        self.assertEqual(contract["init"]["mode"], "sam31-first-frame")
        self.assertEqual(contract["gpu_pipeline"]["mode"], demo.GPU_PIPELINE_MODE_SINGLE_OWNER)
        self.assertEqual(contract["gpu_pipeline"]["internal_order"], demo.SINGLE_OWNER_ORDER_FFS_THEN_EDGETAM)
        self.assertEqual(contract["filter_scheduler"]["mode"], "async")
        self.assertTrue(contract["filter_scheduler"]["render_filtered_only"])
        self.assertFalse(contract["filter_scheduler"]["render_accepts_raw_fused_pcd"])
        self.assertEqual([layer["label"] for layer in contract["semantic_layers"]], ["towel", "stuffed animal"])

    def test_demo22_staged_parallel_preset_contract(self) -> None:
        parser = demo.build_arg_parser()
        args = parser.parse_args(["--dry-run", "--preset", demo.PRESET_DEMO22_STAGED_PARALLEL_5FPS])
        args = demo.apply_preset_defaults(args, explicit_options={"--dry-run", "--preset"})
        contract = demo.build_contract(args)

        self.assertEqual(contract["demo"], "demo_2_2_async_filtered_fused_pcd")
        self.assertEqual(contract["preset_canonical"], demo.PRESET_DEMO22_STAGED_PARALLEL_5FPS)
        self.assertEqual(contract["fps"], 5)
        self.assertEqual(contract["fusion_target_fps"], 5.0)
        self.assertEqual(contract["depth_source"], demo.DEPTH_SOURCE_FFS)
        self.assertEqual(contract["compile_mode"], demo.DEFAULT_COMPILE_MODE)
        self.assertEqual(contract["track_mode"], demo.TRACK_MODE_CONTROLLER_OBJECT)
        self.assertEqual(contract["gpu_pipeline"]["mode"], demo.GPU_PIPELINE_MODE_STAGED)
        self.assertEqual(contract["gpu_pipeline"]["internal_order"], demo.STAGED_ORDER_FFS_THEN_PARALLEL_EDGETAM)
        self.assertEqual(contract["gpu_pipeline"]["ffs_stage"], "sequential_cam0_cam1_cam2")
        self.assertEqual(contract["gpu_pipeline"]["edgetam_stage"], "parallel_cam0_cam1_cam2")
        self.assertEqual(contract["edgetam"]["model_topology"], demo.EDGETAM_MODEL_TOPOLOGY_REPLICATED)
        self.assertEqual(contract["edgetam"]["stream_mode"], demo.EDGETAM_STREAM_MODE_PER_CAMERA)
        self.assertEqual(contract["h2d_transfer"]["pin_memory_mode"], demo.PIN_MEMORY_MODE_ALL)
        self.assertTrue(contract["h2d_transfer"]["edge_pin_enabled"])
        self.assertTrue(contract["h2d_transfer"]["ffs_pin_requested"])
        self.assertEqual(contract["h2d_transfer"]["h2d_stream_mode"], demo.H2D_STREAM_MODE_DEDICATED)
        self.assertTrue(contract["memory_for_speed"]["static_device_buffers"])
        self.assertTrue(contract["memory_for_speed"]["ffs_reusable_cuda_input_buffers"])
        self.assertTrue(contract["memory_for_speed"]["edgetam_reusable_cuda_pixel_slots"])
        self.assertTrue(contract["filter_scheduler"]["render_filtered_only"])

    def test_demo22_thread_specs_include_filter_worker(self) -> None:
        parser = demo.build_arg_parser()
        args = parser.parse_args(["--dry-run", "--preset", demo.PRESET_DEMO22_ASYNC_FILTER_5FPS])
        args = demo.apply_preset_defaults(args, explicit_options={"--dry-run", "--preset"})
        runtime = demo.Demo21Runtime(args)
        names = [name for name, _target in runtime._thread_specs()]

        self.assertEqual(names, ["capture-group", "gpu-owner", "fusion", "filter"])

    def test_demo22_staged_parallel_thread_specs_include_staged_gpu_and_filter(self) -> None:
        parser = demo.build_arg_parser()
        args = parser.parse_args(["--dry-run", "--preset", demo.PRESET_DEMO22_STAGED_PARALLEL_5FPS])
        args = demo.apply_preset_defaults(args, explicit_options={"--dry-run", "--preset"})
        runtime = demo.Demo21Runtime(args)
        names = [name for name, _target in runtime._thread_specs()]

        self.assertEqual(names, ["capture-group", "staged-gpu", "fusion", "filter"])

    def test_raw_fused_packet_is_not_published_to_render_slot(self) -> None:
        parser = demo.build_arg_parser()
        args = parser.parse_args(["--dry-run", "--preset", demo.PRESET_DEMO22_ASYNC_FILTER_5FPS])
        args = demo.apply_preset_defaults(args, explicit_options={"--dry-run", "--preset"})
        runtime = demo.Demo21Runtime(args)

        runtime._publish_raw_fused_for_async_filter(_raw_packet(1))

        self.assertEqual(runtime.raw_fused_slot.latest_seq(), 1)
        self.assertEqual(runtime.render_slot.latest_seq(), -1)

    def test_filter_output_is_the_only_render_packet(self) -> None:
        parser = demo.build_arg_parser()
        args = parser.parse_args(["--dry-run", "--preset", demo.PRESET_DEMO22_ASYNC_FILTER_5FPS])
        args = demo.apply_preset_defaults(args, explicit_options={"--dry-run", "--preset"})
        runtime = demo.Demo21Runtime(args)
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
        runtime = demo.Demo21Runtime(args)

        runtime.raw_fused_slot.put(_raw_packet(1))
        runtime.raw_fused_slot.put(_raw_packet(2))

        self.assertEqual(runtime.raw_fused_slot.dropped_count, 1)
        self.assertEqual(runtime.raw_fused_slot.get_latest_after(-1).seq, 2)  # type: ignore[union-attr]

    def test_async_filter_worker_publishes_filtered_latest_packet(self) -> None:
        parser = demo.build_arg_parser()
        args = parser.parse_args(["--dry-run", "--preset", demo.PRESET_DEMO22_ASYNC_FILTER_5FPS])
        args = demo.apply_preset_defaults(args, explicit_options={"--dry-run", "--preset"})
        runtime = demo.Demo21Runtime(args)
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
        runtime = demo.Demo21Runtime(args)
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


if __name__ == "__main__":
    unittest.main()
