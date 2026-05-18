from __future__ import annotations

import os
import pickle
import queue
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from demo_v2_2 import runtime as demo22
from demo_v2_3 import realtime_three_view_dual_gpu_async_filtered_fused_pcd as demo23_entry
from qqtt.demo import demo23_dual_gpu_workers as workers
from qqtt.demo import demo23_runtime as demo23
from qqtt.demo import realtime_masked_edgetam_pcd as masked_runtime
from qqtt.demo import three_view_masked_fused_pcd_runtime as shared_runtime


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
        self.assertIn("--depth-source", help_text)
        self.assertIn("--debug-color-by-camera", help_text)
        self.assertIn("--debug-save-per-camera-pcd", help_text)
        self.assertIn("--debug-save-mask-overlays", help_text)
        self.assertIn("--debug-identity-c2w", help_text)
        self.assertIn("--debug-invert-c2w", help_text)
        self.assertIn("--debug-only-camera-idx", help_text)
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
                "--depth-source",
                "realsense",
                "--debug-color-by-camera",
                "--debug-save-per-camera-pcd",
                "--debug-save-mask-overlays",
                "--debug-identity-c2w",
                "--debug-only-camera-idx",
                "1",
                "--debug-fusion-max-saved-groups",
                "2",
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
        self.assertIn("--depth-source", argv)
        self.assertIn("realsense", argv)
        self.assertIn("--debug-color-by-camera", argv)
        self.assertIn("--debug-save-per-camera-pcd", argv)
        self.assertIn("--debug-save-mask-overlays", argv)
        self.assertIn("--debug-identity-c2w", argv)
        self.assertIn("--debug-only-camera-idx", argv)
        self.assertIn("1", argv)
        self.assertIn("--debug-fusion-max-saved-groups", argv)
        self.assertIn("2", argv)

    def test_demo23_dry_run_contract_is_dual_gpu_split_batch3(self) -> None:
        parser = demo23.build_arg_parser()
        args = parser.parse_args(["--dry-run", "--preset", demo23.PRESET_DEMO23_DUAL4090_MAXFPS])
        args = demo23.apply_preset_defaults(args, explicit_options={"--dry-run", "--preset"})
        contract = demo23.build_contract(args)

        self.assertEqual(contract["demo_version"], "demo2.3")
        self.assertEqual(contract["demo"], "demo_2_3_dual_gpu_async_filtered_fused_pcd")
        self.assertEqual(contract["fps"], 30)
        self.assertEqual(contract["fusion_target_fps"], 30.0)
        self.assertEqual(contract["capture_group_target_fps"], 30.0)
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
        self.assertFalse(contract["fusion_debug"]["color_by_camera"])

    def test_demo23_explicit_fps_drives_capture_group_default(self) -> None:
        parser = demo23.build_arg_parser()
        args = parser.parse_args(["--dry-run", "--preset", demo23.PRESET_DEMO23_DUAL4090_MAXFPS, "--fps", "15"])
        args = demo23.apply_preset_defaults(args, explicit_options={"--dry-run", "--preset", "--fps"})
        contract = demo23.build_contract(args)

        self.assertEqual(contract["fps"], 15)
        self.assertEqual(contract["capture_group_target_fps"], 15.0)
        self.assertEqual(contract["fusion_target_fps"], 30.0)

    def test_demo23_native_depth_debug_contract_skips_ffs_batch_requirement(self) -> None:
        with tempfile.NamedTemporaryFile() as calibration_file:
            parser = demo23.build_arg_parser()
            args = parser.parse_args(
                [
                    "--dry-run",
                    "--preset",
                    demo23.PRESET_DEMO23_DUAL4090_MAXFPS,
                    "--calibrate-path",
                    calibration_file.name,
                    "--depth-source",
                    demo23.DEPTH_SOURCE_REALSENSE,
                    "--debug-color-by-camera",
                ]
            )
            args = demo23.apply_preset_defaults(
                args,
                explicit_options={
                    "--dry-run",
                    "--preset",
                    "--calibrate-path",
                    "--depth-source",
                    "--debug-color-by-camera",
                },
            )
            runtime = demo23.Demo23Runtime(args)

            runtime._validate_live_contract()
            self.assertEqual(demo23.build_contract(args)["depth_source"], demo23.DEPTH_SOURCE_REALSENSE)

    def test_fusion_debug_writes_per_camera_artifacts_and_profile_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            parser = demo23.build_arg_parser()
            args = parser.parse_args(
                [
                    "--dry-run",
                    "--preset",
                    demo23.PRESET_DEMO23_DUAL4090_MAXFPS,
                    "--render-mode",
                    "none",
                    "--profile-json-output",
                    str(tmp / "profile.json"),
                    "--object-postprocess",
                    demo23.POSTPROCESS_NONE,
                    "--controller-postprocess",
                    demo23.POSTPROCESS_NONE,
                    "--debug-color-by-camera",
                    "--debug-save-per-camera-pcd",
                    "--debug-save-mask-overlays",
                    "--debug-identity-c2w",
                    "--debug-only-camera-idx",
                    "1",
                ]
            )
            args = demo23.apply_preset_defaults(
                args,
                explicit_options={
                    "--dry-run",
                    "--preset",
                    "--render-mode",
                    "--profile-json-output",
                    "--object-postprocess",
                    "--controller-postprocess",
                    "--debug-color-by-camera",
                    "--debug-save-per-camera-pcd",
                    "--debug-save-mask-overlays",
                    "--debug-identity-c2w",
                    "--debug-only-camera-idx",
                },
            )
            runtime = demo23.Demo23Runtime(args)
            runtime._debug_fusion_dir = tmp / "debug_fusion"
            runtime._debug_effective_c2w_mapping_mode = "debug-identity-c2w"
            identity = np.eye(4, dtype=np.float32)
            runtime._c2w_by_camera = {0: identity, 1: identity, 2: identity}
            runtime._debug_original_c2w_by_camera = {0: identity, 1: identity, 2: identity}
            runtime.camera_system = types.SimpleNamespace(
                serial_numbers=["serial0", "serial1", "serial2"],
                calibration_reference_serials=["serial0", "serial1", "serial2"],
            )
            runtime._write_calibration_debug_report()

            object_mask = np.asarray([[True, False], [False, True]], dtype=bool)
            controller_mask = np.asarray([[False, True], [False, False]], dtype=bool)
            mask_packet = demo23.CameraMaskPacket(
                group_id=7,
                camera_idx=1,
                color_bgr=np.zeros((2, 2, 3), dtype=np.uint8),
                controller_mask=controller_mask,
                object_mask=object_mask,
                model_ms=1.0,
                cuda_event_model_ms=1.0,
                mask_ms=1.0,
                gpu_gate_wait_ms=0.0,
            )
            depth_group = demo23.DepthGroup(
                group_id=7,
                depths={
                    1: demo23.DepthPacket(
                        group_id=7,
                        camera_idx=1,
                        depth_m=np.full((2, 2), 0.5, dtype=np.float32),
                        ffs_ms=1.0,
                        align_ms=0.0,
                    )
                },
                total_ms=1.0,
                per_camera_ms={1: {"ffs_ms": 1.0, "align_ms": 0.0}},
                gpu_gate_wait_ms=0.0,
                max_temporal_skew_ms=0.0,
                per_camera_time_offset_ms={},
                per_camera_frame_seq={},
                timestamp_source="test",
            )
            ray_cache = {
                1: (
                    np.asarray([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32),
                    np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
                )
            }

            packet = runtime._build_fused_packet(
                depth_group=depth_group,
                masks={1: mask_packet},
                ray_cache=ray_cache,
                rng=np.random.default_rng(0),
            )
            record = runtime.pop_profile_record(7)

            np.testing.assert_array_equal(
                packet.object_colors_rgb,
                np.full_like(packet.object_colors_rgb, [0, 255, 0]),
            )
            fusion = record["fusion"]
            self.assertEqual(fusion["active_camera_ids"], [1])
            self.assertEqual(fusion["per_camera_point_counts"]["cam0"]["total"], 0)
            self.assertFalse(fusion["per_camera_point_counts"]["cam0"]["active_for_fusion"])
            self.assertEqual(fusion["per_camera_point_counts"]["cam1"]["object"], 2)
            self.assertEqual(fusion["per_camera_point_counts"]["cam1"]["controller"], 1)
            self.assertEqual(fusion["per_camera_mask_pixel_counts"]["cam1"]["union"], 3)
            self.assertIsNotNone(fusion["per_camera_cloud_centroids"]["cam1"]["object"])
            self.assertTrue((tmp / "debug_fusion" / "calibration_report.json").is_file())
            self.assertTrue((tmp / "debug_fusion" / "group_000007_cam1_object.ply").is_file())
            self.assertTrue((tmp / "debug_fusion" / "group_000007_cam1_controller.ply").is_file())
            self.assertTrue((tmp / "debug_fusion" / "group_000007_cam1_mask_overlay.png").is_file())

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

    def test_open3d_duration_shutdown_closes_window_and_quits_app(self) -> None:
        parser = demo23.build_arg_parser()
        args = parser.parse_args(
            ["--dry-run", "--preset", demo23.PRESET_DEMO23_DUAL4090_MAXFPS, "--duration-s", "0.01"]
        )
        args = demo23.apply_preset_defaults(
            args,
            explicit_options={"--dry-run", "--preset", "--duration-s"},
        )
        runtime = demo23.Demo23Runtime(args)
        runtime._start_threads = lambda: None  # type: ignore[method-assign]

        class FakeWindow:
            def __init__(self) -> None:
                self.renderer = object()
                self.content_rect = types.SimpleNamespace(x=0, y=0, width=1280, height=800)
                self.theme = types.SimpleNamespace(font_size=12)
                self.close_calls = 0
                self.on_close = None

            def add_child(self, _child):
                return None

            def set_on_layout(self, _callback):
                return None

            def set_on_close(self, callback):
                self.on_close = callback

            def close(self):
                self.close_calls += 1
                if self.on_close is not None:
                    self.on_close()

        class FakeApp:
            def __init__(self) -> None:
                self.window = FakeWindow()
                self.quit_calls = 0
                self.run_calls = 0
                self.post_calls = 0

            def initialize(self):
                return None

            def create_window(self, *_args):
                return self.window

            def post_to_main_thread(self, _window, callback):
                self.post_calls += 1
                callback()

            def run(self):
                self.run_calls += 1

            def quit(self):
                self.quit_calls += 1

        class FakeSceneWidget:
            def __init__(self) -> None:
                self.scene = None
                self.frame = None

            def setup_camera(self, *_args):
                return None

        class FakePanel:
            def __init__(self, *_args) -> None:
                return None

            def add_child(self, _child):
                return None

            def calc_preferred_size(self, *_args):
                return types.SimpleNamespace(width=760, height=120)

        class FakeOpen3DScene:
            def __init__(self, _renderer) -> None:
                self.scene = types.SimpleNamespace()

            def set_background(self, _color):
                return None

        class FakeTimer:
            def __init__(self, _interval, callback) -> None:
                self.callback = callback
                self.daemon = False
                self.cancelled = False

            def start(self):
                self.callback()

            def cancel(self):
                self.cancelled = True

        fake_app = FakeApp()
        fake_o3d = types.SimpleNamespace(core=types.SimpleNamespace(Device=lambda _name: object()))
        fake_gui = types.SimpleNamespace(
            Application=types.SimpleNamespace(instance=fake_app),
            SceneWidget=FakeSceneWidget,
            Label=lambda text: types.SimpleNamespace(text=text, text_color=None),
            Color=lambda *_args: object(),
            Vert=FakePanel,
            Margins=lambda *_args: object(),
            Widget=types.SimpleNamespace(Constraints=lambda: object()),
            Rect=lambda *args: args,
        )
        fake_rendering = types.SimpleNamespace(
            Open3DScene=FakeOpen3DScene,
            MaterialRecord=lambda: types.SimpleNamespace(shader="", point_size=0.0),
        )

        with mock.patch.dict(os.environ, {"QQTT_WSLG_OPEN3D_FAST_EXIT": "0"}), mock.patch(
            "qqtt.demo.three_view_masked_fused_pcd_runtime._load_open3d_modules",
            return_value=(fake_o3d, fake_gui, fake_rendering),
        ), mock.patch(
            "qqtt.demo.three_view_masked_fused_pcd_runtime.Open3DSceneTensorLayer",
            return_value=types.SimpleNamespace(update=lambda *_args: None),
        ), mock.patch.object(
            shared_runtime.threading,
            "Timer",
            FakeTimer,
        ):
            runtime._run_open3d()

        self.assertTrue(runtime.stop_event.is_set())
        self.assertGreaterEqual(fake_app.window.close_calls, 1)
        self.assertGreaterEqual(fake_app.quit_calls, 1)
        self.assertEqual(fake_app.post_calls, 1)
        self.assertTrue(runtime._summary["open3d_shutdown_requested"])
        self.assertTrue(runtime._summary["open3d_window_close_requested"])
        self.assertTrue(runtime._summary["open3d_app_quit_requested"])

    def test_sam31_runtime_releases_before_dual_gpu_steady_state(self) -> None:
        parser = demo23.build_arg_parser()
        args = parser.parse_args(["--dry-run", "--preset", demo23.PRESET_DEMO23_DUAL4090_MAXFPS])
        args = demo23.apply_preset_defaults(args, explicit_options={"--dry-run", "--preset"})
        runtime = demo23.Demo23Runtime(args)
        states = {
            0: {"initialized": True},
            1: {"initialized": True},
            2: {"initialized": True},
        }

        with mock.patch(
            "qqtt.demo.three_view_masked_fused_pcd_runtime.release_sam31_runtime_resources",
            return_value=12.5,
        ) as release:
            self.assertTrue(runtime._release_sam31_runtime_after_all_cameras_init_if_needed(states))
            self.assertFalse(runtime._release_sam31_runtime_after_all_cameras_init_if_needed(states))

        release.assert_called_once_with("cuda")
        self.assertTrue(runtime._summary["sam31_runtime_released_after_all_cameras_init"])
        self.assertEqual(runtime._init_profile_snapshot()["sam31"]["release_cleanup_ms"], 12.5)

    def test_cached_sam31_init_trims_cuda_cache_after_each_frame(self) -> None:
        args = types.SimpleNamespace(
            track_mode=masked_runtime.TRACK_MODE_OBJECT_ONLY,
            object_prompt="stuffed animal",
            controller_prompt="towel",
            device="cuda",
            sam31_cache_init_model=True,
            sam31_keep_runtime_until_all_cameras_init=True,
        )

        def fake_run_image_segmentation(**_kwargs):
            return {
                "masks_by_label": {
                    "stuffed animal": [
                        np.asarray([[False, True, False], [True, True, False]], dtype=bool)
                    ]
                },
                "timing_ms": {"total_ms": 1.0},
            }

        with mock.patch(
            "scripts.harness.sam31_mask_helper.run_image_segmentation",
            side_effect=fake_run_image_segmentation,
        ), mock.patch.object(
            masked_runtime,
            "trim_sam31_cuda_allocator",
            return_value=7.5,
        ) as trim, mock.patch.object(
            masked_runtime,
            "release_sam31_runtime_resources",
        ) as release:
            controller_mask, object_mask = masked_runtime.run_sam31_first_frame_masks(
                np.zeros((2, 3, 3), dtype=np.uint8),
                args,
            )

        trim.assert_called_once_with("cuda")
        release.assert_not_called()
        self.assertEqual(args._sam31_last_trim_cleanup_ms, 7.5)
        self.assertFalse(np.any(controller_mask))
        self.assertEqual(int(np.count_nonzero(object_mask)), 3)

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
