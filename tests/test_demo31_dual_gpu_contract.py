from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
import io
from types import SimpleNamespace
import unittest

import numpy as np

from qqtt.demo import demo31_runtime
from qqtt.demo.demo31_dual_gpu_ipc import TrackingResultLitePacket


@dataclass(frozen=True)
class _FakePacket:
    group_id: int
    camera_idx: int
    object_mask: object | None = None
    controller_mask: object | None = None


@dataclass(frozen=True)
class _FakeMaskGroup:
    group_id: int
    mask_packets: dict[int, _FakePacket]
    edgetam_stage_wall_ms: float = 1.0
    edgetam_stage_sum_model_ms: float = 1.0
    edgetam_stage_mode: str = "fake"


@dataclass(frozen=True)
class _FakeGroup:
    group_id: int


@dataclass(frozen=True)
class _FakeRenderPacket:
    group_id: int
    controller_points_m: np.ndarray
    controller_colors_rgb: np.ndarray


class _FakeProcessClient:
    def __init__(self, result: TrackingResultLitePacket | None) -> None:
        self.result = result

    def get_result(self) -> TrackingResultLitePacket | None:
        return self.result

    def snapshot(self) -> dict[str, int]:
        return {}

    def stop(self, *, timeout_s: float) -> None:
        return None


class _FakeSharedRuntimeModule:
    @staticmethod
    def build_arg_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument("--preset")
        parser.add_argument("--demo-version-override")
        parser.add_argument("--demo-display-name-override")
        parser.add_argument("--profile")
        parser.add_argument("--fps", type=int)
        parser.add_argument("--fusion-target-fps", type=float)
        parser.add_argument("--capture-group-target-fps", type=float)
        parser.add_argument("--camera-ids", type=demo31_runtime.demo3_runtime.parse_camera_ids)
        parser.add_argument("--calibrate-path")
        parser.add_argument("--serials", nargs="*")
        parser.add_argument("--calibration-reference-serials", nargs="*")
        parser.add_argument("--depth-source")
        parser.add_argument("--edgetam-batch-vision-encoder", action="store_true")
        parser.add_argument("--edgetam-live-session-keep-frames", type=int)
        parser.add_argument("--render-mode")
        parser.add_argument("--point-size", type=float)
        parser.add_argument("--render-every-n", type=int)
        parser.add_argument("--render-backend")
        parser.add_argument("--render-layer-mode")
        parser.add_argument("--render-copy-mode")
        parser.add_argument("--pcd-color-mode")
        parser.add_argument("--no-render-async-latest-only", action="store_true")
        parser.add_argument("--track-mode")
        parser.add_argument("--object-prompt")
        parser.add_argument("--controller-prompt")
        parser.add_argument("--experiment-mode")
        parser.add_argument("--duration-s", type=float)
        parser.add_argument("--output-root")
        parser.add_argument("--profile-pipeline", action="store_true")
        parser.add_argument("--profile-visualization", action="store_true")
        parser.add_argument("--render-micro-profile", action="store_true")
        parser.add_argument("--object-point-control")
        parser.add_argument("--object-volume-voxel-m", type=float)
        parser.add_argument("--object-volume-origin")
        parser.add_argument("--object-volume-adaptive", action=argparse.BooleanOptionalAction)
        parser.add_argument("--object-volume-min-voxel-m", type=float)
        parser.add_argument("--object-volume-max-voxel-m", type=float)
        parser.add_argument("--object-volume-target-ms", type=float)
        parser.add_argument("--object-volume-emergency-max-points", type=int)
        parser.add_argument("--object-volume-points-per-voxel", type=int)
        parser.add_argument("--debug-color-by-camera", action="store_true")
        parser.add_argument("--debug-save-per-camera-pcd", action="store_true")
        parser.add_argument("--debug-save-mask-overlays", action="store_true")
        parser.add_argument("--debug-identity-c2w", action="store_true")
        parser.add_argument("--debug-invert-c2w", action="store_true")
        parser.add_argument("--debug-only-camera-idx", type=int)
        parser.add_argument("--debug-fusion-max-saved-groups", type=int)
        parser.add_argument("--gpu-sampling", action="store_true")
        parser.add_argument("--gpu-sampling-interval-s", type=float)
        parser.add_argument("--gpu-sampling-backend")
        parser.add_argument("--gpu-sampling-device-index", type=int)
        parser.add_argument("--gpu-sampling-device-indexes", type=demo31_runtime.demo3_runtime.parse_gpu_sampling_device_indexes)
        parser.add_argument("--tracking-backend")
        parser.add_argument("--tracking-source")
        parser.add_argument("--tracking-num-points", type=int)
        parser.add_argument("--tracking-overlay-max-points", type=int)
        parser.add_argument("--tracking-trail-len", type=int)
        parser.add_argument("--tracking-depth-source")
        parser.add_argument("--profile-json-output")
        parser.add_argument("--debug", action="store_true")
        parser.add_argument("--show-tracking-overlay", action="store_true")
        return parser

    @staticmethod
    def apply_preset_defaults(args, *, explicit_options=None):
        _ = explicit_options
        return args

    @staticmethod
    def _explicit_cli_options(argv):
        return {item for item in argv if str(item).startswith("--")}

    class Demo21Runtime:
        def __init__(self, args: object) -> None:
            self.args = args
            self.profile_updates: list[tuple[int, dict[str, object]]] = []
            self.published_packet: object | None = None

        def _profile_update(self, group_id: int, **kwargs: object) -> None:
            self.profile_updates.append((int(group_id), kwargs))

        def _publish_render_packet(self, packet: object) -> None:
            self.published_packet = packet

        def stop(self) -> None:
            return None


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
        self.assertEqual(contract["edgetam_live_session_keep_frames"], 64)
        self.assertTrue(contract["edgetam_live_session_pruning"])
        self.assertFalse(contract["debug_fusion"]["color_by_camera"])
        self.assertFalse(contract["gpu_sampling"]["enabled"])
        self.assertEqual(contract["gpu_sampling"]["device_indexes"], [0, 1])
        self.assertEqual(contract["input_source"], "live_realsense")
        self.assertFalse(contract["offline_mode_available"])
        self.assertFalse(contract["offline_tracking_available"])
        self.assertEqual(contract["init_mode"], "sam31_first_frame")
        self.assertEqual(contract["mask_propagation"], "hf_edgetam_online")
        self.assertEqual(contract["semantic_mode"], "exp")
        self.assertEqual(contract["controller_prompt"], "towel")
        self.assertEqual(contract["tracking_mask_scope"], "object_controller_union")
        self.assertEqual(contract["tracking_query_mode"], "phystwin_dense")
        self.assertEqual(contract["tracking_query_count_requested"], "auto")
        self.assertEqual(contract["tracking_sampling"], "torch_randperm_seed_plus_camera_idx")
        self.assertEqual(contract["cotracker_seed"], 42)
        self.assertEqual(contract["overlay_max_points_per_camera"], 30)
        self.assertEqual(contract["overlay_display_scope"], "controller")
        self.assertEqual(contract["overlay_display_classification"], "first_frame_mask_membership")
        self.assertTrue(contract["phystwin_dense_compatible"])
        self.assertEqual(contract["cotracker_backend"], "cotracker3_online")
        self.assertEqual(contract["cotracker_owner"], "process")
        self.assertEqual(contract["cotracker_process_mode"], "subprocess")
        self.assertEqual(contract["cotracker_update_mode"], "auto")
        self.assertEqual(contract["cotracker_batch_size_target"], 3)
        self.assertTrue(contract["cotracker_batch_fallback_enabled"])
        self.assertFalse(contract["cross_gpu_cuda_tensor_transfer"])
        self.assertEqual(contract["ipc_payload"], "cpu_numpy_latest_wins")
        self.assertFalse(contract["tracking_input_contains_depth"])
        self.assertEqual(contract["shared_runtime_tracking_backend"], "none")
        self.assertFalse(contract["render_waited_for_cotracker"])
        self.assertFalse(contract["render_waited_for_mask"])
        self.assertEqual(contract["pcd_color_mode"], "rgb")

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
        self.assertIn("edgetam_live_session_keep_frames = 64", output)
        self.assertIn("edgetam_live_session_pruning = true", output)
        self.assertIn("input_source = live_realsense", output)
        self.assertIn("offline_mode_available = false", output)
        self.assertIn("semantic_mode = exp", output)
        self.assertIn("tracking_mask_scope = object_controller_union", output)
        self.assertIn("tracking_query_mode = phystwin_dense", output)
        self.assertIn("tracking_query_count_requested = auto", output)
        self.assertIn("overlay_display_scope = controller", output)
        self.assertIn("phystwin_dense_compatible = true", output)
        self.assertIn("cotracker_owner = process", output)
        self.assertIn("cross_gpu_cuda_tensor_transfer = false", output)
        self.assertIn("pcd_color_mode = rgb", output)
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
        self.assertEqual(config.query_mode, "phystwin_dense")
        self.assertEqual(config.query_count_request, "auto")
        self.assertEqual(config.seed, 42)
        self.assertEqual(config.update_mode, "auto")
        self.assertTrue(config.init_requires_object_and_controller)
        self.assertEqual(config.overlay_display_scope, "controller")

    def test_mode_demo_uses_hand_controller_without_changing_gpu_split(self) -> None:
        args = self._parse(
            [
                "--dry-run",
                "--camera-ids",
                "0,1,2",
                "--mask-gpu",
                "0",
                "--cotracker-gpu",
                "1",
                "--mode",
                "demo",
            ]
        )
        contract = demo31_runtime.build_contract(args, cuda_device_count_provider=lambda: 2)

        self.assertEqual(contract["semantic_mode"], "demo")
        self.assertEqual(contract["shared_experiment_mode"], "demo-mode")
        self.assertEqual(contract["controller_prompt"], "hand")
        self.assertEqual(contract["main_cuda_visible_devices"], "0")
        self.assertEqual(contract["cotracker_cuda_visible_devices"], "1")

    def test_shared_runtime_args_forward_edgetam_session_limit(self) -> None:
        args = self._parse(
            [
                "--camera-ids",
                "0,1,2",
                "--mask-gpu",
                "0",
                "--cotracker-gpu",
                "1",
                "--edgetam-live-session-keep-frames",
                "32",
                "--debug-color-by-camera",
                "--debug-only-camera-idx",
                "1",
                "--gpu-sampling",
                "--point-size",
                "1.5",
                "--object-volume-points-per-voxel",
                "3",
            ]
        )

        shared_args = demo31_runtime.build_shared_runtime_args(
            args,
            shared_runtime_module=_FakeSharedRuntimeModule,
            live_validation={
                "active_serials": ["s0", "s1", "s2"],
                "calibration_reference_serials": ["s0", "s1", "s2"],
            },
            shared_profile_path=None,
        )

        self.assertEqual(shared_args.edgetam_live_session_keep_frames, 32)
        self.assertEqual(shared_args.demo_version_override, "demo3.1")
        self.assertEqual(shared_args.demo_display_name_override, "Demo 3.1")
        self.assertTrue(shared_args.debug_color_by_camera)
        self.assertEqual(shared_args.debug_only_camera_idx, 1)
        self.assertTrue(shared_args.gpu_sampling)
        self.assertEqual(shared_args.gpu_sampling_device_indexes, (0, 1))
        self.assertEqual(shared_args.point_size, 1.5)
        self.assertEqual(shared_args.pcd_color_mode, "rgb")
        self.assertEqual(shared_args.object_point_control, "phystwin-volume")
        self.assertEqual(shared_args.object_volume_voxel_m, 0.005)
        self.assertEqual(shared_args.object_volume_points_per_voxel, 3)
        self.assertEqual(shared_args.depth_source, "realsense")
        self.assertTrue(shared_args.edgetam_batch_vision_encoder)

    def test_explicit_class_pcd_color_mode_is_forwarded(self) -> None:
        args = self._parse(["--camera-ids", "0,1,2", "--pcd-color-mode", "class"])

        shared_args = demo31_runtime.build_shared_runtime_args(
            args,
            shared_runtime_module=_FakeSharedRuntimeModule,
            live_validation={
                "active_serials": ["s0", "s1", "s2"],
                "calibration_reference_serials": ["s0", "s1", "s2"],
            },
            shared_profile_path=None,
        )

        self.assertEqual(shared_args.pcd_color_mode, "class")

    def test_summary_extracts_shared_gpu_sampling_by_physical_device(self) -> None:
        args = self._parse(["--camera-ids", "0,1,2", "--mask-gpu", "0", "--cotracker-gpu", "1"])
        runtime = demo31_runtime.Demo31Runtime(
            args,
            cuda_device_count_provider=lambda: 2,
            connected_serials_provider=lambda: ["s0", "s1", "s2"],
        )

        summary = runtime._build_summary(
            runtime=SimpleNamespace(_summary={"final": {}}),
            exit_code=0,
            snapshot=None,
            shared_payload={
                "summary_after_warmup": {"render_fps": 12.0, "fusion_fps": 11.0, "capture_group_fps": 30.0},
                "gpu_sampling": {
                    "summary_by_device_after_warmup": {
                        "0": {
                            "metrics": {
                                "gpu_util_pct": {"median": 55.0, "p95": 75.0},
                                "memory_used_mb": {"median": 6144.0},
                            }
                        },
                        "1": {
                            "metrics": {
                                "gpu_util_pct": {"median": 66.0, "p95": 88.0},
                                "memory_used_mb": {"median": 8192.0},
                            }
                        },
                    }
                },
            },
        )

        self.assertEqual(summary["gpu0_util_median"], 55.0)
        self.assertEqual(summary["gpu0_util_p95"], 75.0)
        self.assertEqual(summary["gpu0_mem_used_gb"], 6.0)
        self.assertEqual(summary["gpu1_util_median"], 66.0)
        self.assertEqual(summary["gpu1_util_p95"], 88.0)
        self.assertEqual(summary["gpu1_mem_used_gb"], 8.0)

    def test_track_mode_is_not_public_cli(self) -> None:
        parser = demo31_runtime.build_arg_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["--dry-run", "--camera-ids", "0,1,2", "--track-mode", "object-only"])

    def test_latest_reuse_join_buffer_retargets_latest_mask_to_fresh_depth(self) -> None:
        buffer = demo31_runtime.Demo31MaskPolicyJoinBuffer(
            policy="latest-reuse",
            stale_timeout_ms=1000.0,
        )
        buffer.put_mask(_FakeMaskGroup(group_id=4, mask_packets={0: _FakePacket(group_id=4, camera_idx=0)}))
        buffer.put_capture(_FakeGroup(group_id=5))
        buffer.put_depth(_FakeGroup(group_id=5))

        ready = buffer.pop_latest_ready()

        self.assertIsNotNone(ready)
        capture, depth, mask = ready  # type: ignore[misc]
        self.assertEqual(capture.group_id, 5)
        self.assertEqual(depth.group_id, 5)
        self.assertEqual(mask.group_id, 5)
        self.assertEqual(mask.mask_packets[0].group_id, 5)
        self.assertEqual(mask.source_group_id, 4)
        self.assertTrue(mask.mask_reused)
        self.assertEqual(buffer.selection_for_group(5)["source_group_id"], 4)  # type: ignore[index]
        self.assertEqual(buffer.snapshot()["mask_reuse_count"], 1)
        self.assertEqual(buffer.snapshot()["mask_group_delta_median"], 1.0)

    def test_strict_join_buffer_requires_matching_mask_group(self) -> None:
        buffer = demo31_runtime.Demo31MaskPolicyJoinBuffer(policy="strict")
        buffer.put_mask(_FakeMaskGroup(group_id=4, mask_packets={0: _FakePacket(group_id=4, camera_idx=0)}))
        buffer.put_capture(_FakeGroup(group_id=5))
        buffer.put_depth(_FakeGroup(group_id=5))

        self.assertIsNone(buffer.pop_latest_ready())

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

    def test_lift_input_cache_returns_group_aligned_copies(self) -> None:
        cache = demo31_runtime.Demo31LiftInputCache(max_groups=2)
        depth1 = np.full((1, 1), 1.0, dtype=np.float32)
        depth2 = np.full((1, 1), 2.0, dtype=np.float32)
        cache.publish(
            group_id=1,
            timestamp_s=1.0,
            depth_by_camera={0: depth1},
            intrinsics_by_camera={0: np.eye(3, dtype=np.float32)},
            c2w_by_camera={0: np.eye(4, dtype=np.float32)},
            mask_by_camera={0: np.ones((1, 1), dtype=bool)},
        )
        cache.publish(
            group_id=2,
            timestamp_s=2.0,
            depth_by_camera={0: depth2},
            intrinsics_by_camera={0: np.eye(3, dtype=np.float32)},
            c2w_by_camera={0: np.eye(4, dtype=np.float32)},
            mask_by_camera={0: np.ones((1, 1), dtype=bool)},
        )
        depth1[0, 0] = 9.0

        snapshot1 = cache.get(1)
        self.assertIsNotNone(snapshot1)
        self.assertEqual(float(snapshot1.depth_by_camera[0][0, 0]), 1.0)  # type: ignore[union-attr]

        cache.publish(
            group_id=3,
            timestamp_s=3.0,
            depth_by_camera={0: np.full((1, 1), 3.0, dtype=np.float32)},
            intrinsics_by_camera={0: np.eye(3, dtype=np.float32)},
            c2w_by_camera={0: np.eye(4, dtype=np.float32)},
            mask_by_camera={0: np.ones((1, 1), dtype=bool)},
        )
        self.assertIsNone(cache.get(1))
        self.assertIsNotNone(cache.get(2))
        self.assertEqual(cache.snapshot()["evicted"], 1)

    def test_renderer_lifts_overlay_with_matching_group_depth_not_latest_depth(self) -> None:
        now_s = demo31_runtime.time.perf_counter()
        result = TrackingResultLitePacket(
            group_id=1,
            frame_idx=1,
            source_timestamp_s=now_s,
            publish_timestamp_s=now_s,
            camera_tracks_yx={0: np.array([[0.0, 0.0]], dtype=np.float32)},
            camera_visibility={0: np.array([1.0], dtype=np.float32)},
            query_points_yx={0: np.array([[0.0, 0.0]], dtype=np.float32)},
            publish_range=(1, 1),
        )
        runtime_cls = demo31_runtime.make_demo31_live_runtime_class(
            _FakeSharedRuntimeModule,
            process_client_factory=lambda _config: _FakeProcessClient(result),
        )
        runtime = runtime_cls(
            SimpleNamespace(camera_ids=(0,)),
            demo31_contract={
                "fusion_mask_policy": "latest-reuse",
                "mask_stale_timeout_ms": 250.0,
                "cotracker_result_stale_timeout_ms": 1500.0,
            },
            cotracker_process_config=SimpleNamespace(),
        )
        runtime.demo31_lift_input_cache.publish(
            group_id=1,
            timestamp_s=now_s,
            depth_by_camera={0: np.full((1, 1), 1.0, dtype=np.float32)},
            intrinsics_by_camera={0: np.eye(3, dtype=np.float32)},
            c2w_by_camera={0: np.eye(4, dtype=np.float32)},
            mask_by_camera={0: np.ones((1, 1), dtype=bool)},
        )
        runtime.demo31_lift_input_cache.publish(
            group_id=2,
            timestamp_s=now_s,
            depth_by_camera={0: np.full((1, 1), 2.0, dtype=np.float32)},
            intrinsics_by_camera={0: np.eye(3, dtype=np.float32)},
            c2w_by_camera={0: np.eye(4, dtype=np.float32)},
            mask_by_camera={0: np.ones((1, 1), dtype=bool)},
        )
        packet = _FakeRenderPacket(
            group_id=2,
            controller_points_m=np.empty((0, 3), dtype=np.float32),
            controller_colors_rgb=np.empty((0, 3), dtype=np.uint8),
        )

        runtime._publish_render_packet(packet)

        published = runtime.published_packet
        self.assertIsNotNone(published)
        np.testing.assert_allclose(published.controller_points_m[-1], np.array([0.0, 0.0, 1.0], dtype=np.float32))  # type: ignore[union-attr]
        overlay_profile = runtime.profile_updates[-1][1]["demo31_tracking_overlay"]
        self.assertEqual(overlay_profile["overlay_group_id"], 1)
        self.assertEqual(overlay_profile["render_group_id"], 2)
        self.assertTrue(overlay_profile["overlay_lift_cache_hit"])


if __name__ == "__main__":
    unittest.main()
