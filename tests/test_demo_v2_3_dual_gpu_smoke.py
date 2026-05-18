from __future__ import annotations

import os
import pickle
import queue
import types
import unittest
from pathlib import Path

import numpy as np

from demo_v2_2 import runtime as demo22
from demo_v2_3 import realtime_three_view_dual_gpu_async_filtered_fused_pcd as demo23_entry
from qqtt.demo import demo23_dual_gpu_workers as workers
from qqtt.demo import demo23_runtime as demo23


def _capture_group(seq: int) -> demo23.CaptureGroup:
    return demo23.CaptureGroup(
        group_id=int(seq),
        created_perf_s=float(seq),
        frames={},
        group_timestamp_ns=int(seq),
        max_temporal_skew_ms=0.0,
        per_camera_time_offset_ms={},
        per_camera_frame_seq={},
        timestamp_source="test",
    )


def _depth_group(seq: int) -> demo23.DepthGroup:
    return demo23.DepthGroup(
        group_id=int(seq),
        depths={},
        total_ms=10.0,
        per_camera_ms={},
        gpu_gate_wait_ms=0.0,
        max_temporal_skew_ms=0.0,
        per_camera_time_offset_ms={},
        per_camera_frame_seq={},
        timestamp_source="test",
    )


def _mask_group(seq: int) -> demo23.MaskGroup:
    return demo23.MaskGroup(
        group_id=int(seq),
        mask_packets={},
        edgetam_stage_wall_ms=20.0,
        edgetam_stage_sum_model_ms=18.0,
        edgetam_stage_mode="dual-gpu-batch-vision",
    )


class DemoV23DualGpuSmoke(unittest.TestCase):
    def test_demo23_wrapper_defaults_to_dual4090_preset(self) -> None:
        argv = demo23_entry._with_default_preset(["--dry-run"])

        self.assertEqual(argv[:3], ["--preset", demo23.PRESET_DEMO23_DUAL4090_MAXFPS, "--dry-run"])

    def test_demo23_public_help_exposes_dual_gpu_controls(self) -> None:
        help_text = demo23_entry.build_arg_parser().format_help()

        self.assertIn("--ffs-device", help_text)
        self.assertIn("--edgetam-device", help_text)
        self.assertIn("--sam31-device", help_text)
        self.assertIn("--dual-gpu-queue-size", help_text)
        self.assertIn("--dual-gpu-start-method", help_text)
        self.assertIn("--dual-gpu-profile-workers", help_text)
        self.assertIn("--no-dual-gpu-processes", help_text)
        self.assertIn("--gpu-sampling-device-indexes", help_text)
        self.assertIn("--edgetam-batch-vision", help_text)
        self.assertIn("--render-micro-profile", help_text)
        self.assertNotIn("--experimental-overlapped-stages", help_text)

    def test_demo23_entrypoint_uses_dedicated_runtime_boundary(self) -> None:
        source = Path(demo23_entry.__file__).read_text(encoding="utf-8")

        self.assertIn("from qqtt.demo import demo23_runtime as runtime", source)
        self.assertNotIn("from demo_v2_2", source)
        self.assertNotIn("from demo_v2_1", source)

    def test_demo23_public_cli_translates_dual_gpu_flags(self) -> None:
        argv = demo23_entry._to_demo23_argv(
            [
                "--dry-run",
                "--ffs-device",
                "cuda:0",
                "--edgetam-device",
                "cuda:1",
                "--sam31-device",
                "cuda:1",
                "--dual-gpu-queue-size",
                "3",
                "--dual-gpu-start-method",
                "forkserver",
                "--dual-gpu-profile-workers",
                "--gpu-sampling",
                "--gpu-sampling-device-indexes",
                "0,1",
                "--render-mode",
                "none",
            ]
        )

        self.assertEqual(argv[:2], ["--preset", demo23.PRESET_DEMO23_DUAL4090_MAXFPS])
        self.assertIn("--ffs-device", argv)
        self.assertIn("cuda:0", argv)
        self.assertIn("--edgetam-device", argv)
        self.assertIn("cuda:1", argv)
        self.assertIn("--sam31-device", argv)
        self.assertIn("--dual-gpu-queue-size", argv)
        self.assertIn("3", argv)
        self.assertIn("--dual-gpu-start-method", argv)
        self.assertIn("forkserver", argv)
        self.assertIn("--dual-gpu-profile-workers", argv)
        self.assertIn("--gpu-sampling-device-indexes", argv)
        self.assertIn("0,1", argv)
        self.assertIn("--render-mode", argv)
        self.assertIn("none", argv)

    def test_demo23_dry_run_contract_is_dual_gpu_split_batch3(self) -> None:
        parser = demo23.build_arg_parser()
        args = parser.parse_args(["--dry-run", "--preset", demo23.PRESET_DEMO23_DUAL4090_MAXFPS])
        args = demo23.apply_preset_defaults(args, explicit_options={"--dry-run", "--preset"})
        contract = demo23.build_contract(args)

        self.assertEqual(contract["demo_version"], "demo2.3")
        self.assertEqual(contract["demo"], "demo_2_3_dual_gpu_async_filtered_fused_pcd")
        self.assertEqual(contract["fps"], 30)
        self.assertEqual(contract["fusion_target_fps"], 15.0)
        self.assertEqual(contract["gpu_pipeline"]["mode"], demo23.GPU_PIPELINE_MODE_DUAL_GPU_SPLIT)
        self.assertTrue(contract["gpu_pipeline"]["same_group_join_required"])
        self.assertEqual(contract["dual_gpu"]["ffs_device"], "cuda:0")
        self.assertEqual(contract["dual_gpu"]["edgetam_device"], "cuda:1")
        self.assertEqual(contract["dual_gpu"]["sam31_device"], "cuda:1")
        self.assertEqual(contract["ffs_contract"]["trt_batch_size"], 3)
        self.assertTrue(contract["ffs_contract"]["batch3_isolated_artifact"])
        self.assertIn("batch3", contract["ffs_contract"]["trt_model_dir"])
        self.assertEqual(contract["controller_prompt"], "towel")
        self.assertEqual(contract["object_prompt"], "stuffed animal")

    def test_demo23_thread_specs_do_not_start_same_gpu_overlap_workers(self) -> None:
        parser = demo23.build_arg_parser()
        args = parser.parse_args(["--dry-run", "--preset", demo23.PRESET_DEMO23_DUAL4090_MAXFPS])
        args = demo23.apply_preset_defaults(args, explicit_options={"--dry-run", "--preset"})
        runtime = demo23.Demo23Runtime(args)
        names = [name for name, _target in runtime._thread_specs()]

        self.assertEqual(names, ["capture-group", "dual-gpu-dispatch", "dual-gpu-result-collector", "filter"])
        self.assertNotIn("gpu-owner", names)
        self.assertNotIn("stage-dispatch", names)
        self.assertNotIn("ffs-stage", names)
        self.assertNotIn("edgetam-stage", names)

    def test_worker_result_dataclasses_are_pickleable(self) -> None:
        depth_result = workers.WorkerDepthResult(
            group_id=7,
            depth_group=_depth_group(7),
            worker_profile={"ffs": {"cycle_ms": 10.0}},
            worker_timing={"worker_period_ms": 11.0},
        )
        mask_result = workers.WorkerMaskResult(
            group_id=7,
            mask_group=_mask_group(7),
            worker_profile={"edgetam": {"batch_vision": {"total_ms": 20.0}}},
            worker_timing={"worker_period_ms": 21.0},
        )

        self.assertEqual(pickle.loads(pickle.dumps(depth_result)).group_id, 7)
        self.assertEqual(pickle.loads(pickle.dumps(mask_result)).mask_group.group_id, 7)

    def test_worker_child_args_use_cuda_alias_after_visible_device_isolation(self) -> None:
        args = types.SimpleNamespace(
            device="cuda:1",
            gpu_sampling=True,
            parallel_init=True,
            sam31_device="cuda:1",
        )

        prior_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        try:
            child_args = workers._prepare_child_args(args, physical_device="cuda:1", stage="edgetam")
        finally:
            if prior_visible is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = prior_visible

        self.assertEqual(child_args.device, "cuda")
        self.assertEqual(child_args.sam31_device, "cuda")
        self.assertFalse(child_args.gpu_sampling)
        self.assertFalse(child_args.parallel_init)

    def test_bounded_latest_task_queue_drops_oldest(self) -> None:
        task_queue: queue.Queue[workers.WorkerCaptureTask] = queue.Queue(maxsize=2)
        latest_queue = workers.BoundedLatestTaskQueue(task_queue, maxsize=2)

        self.assertEqual(latest_queue.put_latest(workers.WorkerCaptureTask(1, _capture_group(1))), 0)
        self.assertEqual(latest_queue.put_latest(workers.WorkerCaptureTask(2, _capture_group(2))), 0)
        self.assertEqual(latest_queue.put_latest(workers.WorkerCaptureTask(3, _capture_group(3))), 1)

        self.assertEqual(task_queue.get_nowait().group_id, 2)
        self.assertEqual(task_queue.get_nowait().group_id, 3)
        self.assertEqual(latest_queue.drop_count, 1)

    def test_same_group_join_buffer_never_joins_mismatched_group_ids(self) -> None:
        buffer = demo23.SameGroupJoinBuffer(max_groups=8)

        buffer.put_capture(_capture_group(1))
        buffer.put_depth(_depth_group(1))
        buffer.put_mask(_mask_group(2))

        self.assertIsNone(buffer.pop_latest_ready())
        self.assertEqual(buffer.snapshot()["ready_join_count"], 0)

    def test_demo22_default_contract_is_unchanged(self) -> None:
        parser = demo22.build_arg_parser()
        args = parser.parse_args(["--dry-run", "--preset", demo22.PRESET_DEMO22_ASYNC_FILTER_5FPS])
        args = demo22.apply_preset_defaults(args, explicit_options={"--dry-run", "--preset"})
        contract = demo22.build_contract(args)

        self.assertEqual(contract["demo_version"], "demo2.2")
        self.assertEqual(contract["gpu_pipeline"]["mode"], demo22.GPU_PIPELINE_MODE_SINGLE_OWNER)
        self.assertEqual(contract["ffs_contract"]["trt_batch_size"], 3)
        self.assertFalse(contract["dual_gpu"]["enabled"])


if __name__ == "__main__":
    unittest.main()
