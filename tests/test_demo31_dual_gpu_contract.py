from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
import io
import pickle
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from qqtt.demo import demo31_runtime, demo32_runtime
from qqtt.demo.demo31_dual_gpu_ipc import TrackingInputLitePacket, TrackingResultLitePacket


@dataclass(frozen=True)
class _FakePacket:
    group_id: int
    camera_idx: int
    object_mask: object | None = None
    controller_mask: object | None = None
    color_bgr: object | None = None


@dataclass(frozen=True)
class _FakeDepthFrame:
    group_id: int
    depth_m: np.ndarray


@dataclass(frozen=True)
class _FakeDepthGroup:
    group_id: int
    depths: dict[int, _FakeDepthFrame]
    per_camera_frame_seq: dict[int, int]


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
    tracker_backend: str | None = None
    tracker_update_mode: str | None = None
    tracker_batch_size: int | None = None
    tracker_model_ms: float | None = None
    tracker_e2e_ms: float | None = None
    tracker_publish_to_render_ms: float | None = None
    tracker_source_to_render_ms: float | None = None
    tracker_overlay_group_id: int | None = None


class _FakeProcessClient:
    def __init__(self, result: TrackingResultLitePacket | None) -> None:
        self.result = result
        self.inputs: list[TrackingInputLitePacket] = []

    def get_result(self) -> TrackingResultLitePacket | None:
        return self.result

    def publish_input(self, packet: TrackingInputLitePacket) -> int:
        self.inputs.append(packet)
        return 0

    def snapshot(self) -> dict[str, int]:
        return {}

    def stop(self, *, timeout_s: float) -> None:
        return None


class _FakeStatusProcessClient(_FakeProcessClient):
    def __init__(self, events: list[dict[str, object]], *, started_s: float) -> None:
        super().__init__(None)
        self.events = list(events)
        self.started_s = float(started_s)
        self.pid = 12345

    def drain_status_events(self) -> list[dict[str, object]]:
        events = list(self.events)
        self.events.clear()
        return events


class _FakeSharedRuntimeModule:
    PRESET_DEMO23_DUAL4090_MAXFPS = "demo2.3-dual4090-maxfps"
    DEPTH_SOURCE_FFS = "ffs"
    GPU_PIPELINE_MODE_DUAL_GPU_SPLIT = "dual-gpu-split"
    DEFAULT_DEMO22_DEPTH_MIN_M = 0.15

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
        parser.add_argument("--render-backend")
        parser.add_argument("--render-layer-mode")
        parser.add_argument("--render-copy-mode")
        parser.add_argument("--pcd-color-mode")
        parser.add_argument("--no-render-async-latest-only", action="store_true")
        parser.add_argument("--track-mode")
        parser.add_argument("--object-postprocess")
        parser.add_argument("--controller-postprocess")
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
        parser.add_argument("--phystwin-radius-m", type=float)
        parser.add_argument("--phystwin-nb-points", type=int)
        parser.add_argument("--enhanced-component-voxel-size-m", type=float)
        parser.add_argument("--enhanced-keep-near-main-gap-m", type=float)
        parser.add_argument("--object-enhanced-keep-top-n-components", type=int)
        parser.add_argument("--controller-enhanced-keep-top-n-components", type=int)
        parser.add_argument("--enhanced-component-selection-policy")
        parser.add_argument("--enhanced-min-component-points", type=int)
        parser.add_argument("--enhanced-min-component-ratio", type=float)
        parser.add_argument("--apply-enhanced-component-filter-to-pcd", action=argparse.BooleanOptionalAction)
        parser.add_argument("--controller-render-voxel-m", type=float)
        parser.add_argument("--controller-render-max-points", type=int)
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

        def _build_raw_fused_packet(self, *, depth_group: object, masks: dict[int, object], ray_cache: dict[int, object], rng: object) -> object:
            self.raw_fused_call = {
                "depth_group": depth_group,
                "masks": masks,
                "ray_cache": ray_cache,
                "rng": rng,
            }
            return SimpleNamespace(
                group_id=getattr(depth_group, "group_id", 0),
                created_perf_s=demo31_runtime.time.perf_counter(),
                masks=masks,
            )

        def _filter_raw_fused_packet(self, raw: object) -> object:
            self.filter_raw_call = raw
            return _FakeRenderPacket(
                group_id=getattr(raw, "group_id", 0),
                controller_points_m=np.empty((0, 3), dtype=np.float32),
                controller_colors_rgb=np.empty((0, 3), dtype=np.uint8),
            )

        def _build_fused_packet(self, *, depth_group: object, masks: dict[int, object], ray_cache: dict[int, object], rng: object) -> object:
            self.fused_call = {
                "depth_group": depth_group,
                "masks": masks,
                "ray_cache": ray_cache,
                "rng": rng,
            }
            return _FakeRenderPacket(
                group_id=getattr(depth_group, "group_id", 0),
                controller_points_m=np.empty((0, 3), dtype=np.float32),
                controller_colors_rgb=np.empty((0, 3), dtype=np.uint8),
            )

        def stop(self) -> None:
            return None


class Demo31DualGpuContractTest(unittest.TestCase):
    def _parse(self, argv: list[str], *, default_preset: str = demo31_runtime.PRESET_DEMO31_DUAL4090_HIGHFPS):
        parser = demo31_runtime.build_arg_parser(default_preset=default_preset)
        args = parser.parse_args(argv)
        return demo31_runtime.apply_preset_defaults(args, explicit_options=set(argv))

    def test_tracker_ready_status_records_event_time_not_teardown_time(self) -> None:
        process_started_s = demo31_runtime.time.perf_counter() - 1.0
        ready_perf_s = process_started_s + 0.25
        ready_event = {
            "type": "ready",
            "ready_to_receive_inputs": True,
            "ready_state": "ready_to_receive_inputs",
            "ready_perf_s": ready_perf_s,
            "total_init_ms": 250.0,
            "prewarm_backends": True,
            "tracker_prewarm_mode": "backend_model_prewarm",
            "warmup_profile": {"total_ms": 200.0},
        }
        runtime_cls = demo31_runtime.make_demo31_live_runtime_class(
            _FakeSharedRuntimeModule,
            process_client_factory=lambda _config: _FakeStatusProcessClient(
                [ready_event],
                started_s=process_started_s,
            ),
        )
        runtime = runtime_cls(
            SimpleNamespace(camera_ids=(0,)),
            demo31_contract={
                "fusion_mask_policy": "latest-reuse",
                "mask_stale_timeout_ms": 250.0,
                "cotracker_result_stale_timeout_ms": 1500.0,
                "demo31_process_status_background_drain": False,
            },
            cotracker_process_config=SimpleNamespace(prewarm_backends=True),
        )

        events = runtime._drain_demo31_process_status()

        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertAlmostEqual(float(event["ready_event_after_process_start_s"]), 0.25, places=3)
        self.assertIn("ready_receive_s", event)
        self.assertGreaterEqual(float(event["ready_receive_after_process_start_s"]), 0.25)
        self.assertGreaterEqual(float(event["ready_queue_lag_ms"]), 0.0)
        self.assertAlmostEqual(runtime._summary["demo31_tracker_process_init_ms"], 250.0)
        self.assertAlmostEqual(runtime._summary["demo31_tracker_backend_warmup_ms"], 200.0)
        self.assertAlmostEqual(
            runtime._summary["demo31_tracker_ready_event_after_process_start_s"],
            0.25,
            places=3,
        )

    def test_dry_run_contract_maps_gpu0_and_gpu1(self) -> None:
        args = self._parse(["--dry-run", "--camera-ids", "0,1,2", "--mask-gpu", "0", "--cotracker-gpu", "1"])
        demo31_runtime.validate_args(args, cuda_device_count_provider=lambda: 2)
        contract = demo31_runtime.build_contract(args, cuda_device_count_provider=lambda: 2)

        self.assertEqual(contract["demo"], "demo3.1")
        self.assertEqual(contract["output_root"], "result/demo31_dual4090_realsense_tapnextpp")
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
        self.assertEqual(contract["tracking_query_count_requested"], "4096")
        self.assertEqual(contract["trackable_mask_source"], "standard_filter_survivors")
        self.assertEqual(contract["tracking_input_mask_semantics"], "standard_filter_trackable_masks")
        self.assertEqual(contract["tracker_query_source"], "union_trackable_mask")
        self.assertEqual(contract["object_mask_semantics"], "object_trackable_mask")
        self.assertEqual(contract["controller_mask_semantics"], "controller_trackable_mask")
        self.assertEqual(contract["object_point_control"], "fixed-cap")
        self.assertEqual(contract["object_postprocess"], "enhanced-pt")
        self.assertEqual(contract["controller_postprocess"], "enhanced-pt")
        self.assertEqual(contract["trackable_object_filter"]["mode"], "enhanced-pt")
        self.assertEqual(contract["trackable_object_filter"]["point_control"], "fixed-cap")
        self.assertEqual(contract["trackable_controller_filter"]["mode"], "enhanced-pt")
        self.assertEqual(contract["object_filter"]["mode"], "enhanced-pt")
        self.assertEqual(contract["object_filter"]["point_control"], "fixed-cap")
        self.assertEqual(contract["controller_filter"]["mode"], "enhanced-pt")
        self.assertEqual(contract["object_enhanced_keep_top_n_components"], 1)
        self.assertEqual(contract["controller_enhanced_keep_top_n_components"], 2)
        self.assertEqual(contract["enhanced_component_selection_policy"], "largest-n-plus-gap")
        self.assertEqual(contract["enhanced_min_component_points"], 32)
        self.assertEqual(contract["enhanced_min_component_ratio"], 0.0)
        self.assertTrue(contract["apply_enhanced_component_filter_to_pcd"])
        self.assertEqual(
            contract["query_and_pcd_surface_filter_shared"],
            "same_config_reuse_when_source_points_identical",
        )
        self.assertEqual(contract["trackable_controller_filter"]["keep_top_n_components"], 2)
        self.assertEqual(contract["render_controller_filter"]["keep_top_n_components"], 2)
        self.assertEqual(contract["render_object_filter"]["point_control"], "fixed-cap")
        self.assertEqual(contract["render_object_filter"]["postprocess"], "enhanced-pt")
        self.assertEqual(contract["render_controller_filter"]["postprocess"], "enhanced-pt")
        self.assertEqual(contract["tracking_sampling"], "controller_pcd_cap_then_torch_randperm_seed_plus_camera_idx")
        self.assertEqual(contract["controller_pcd_max_points_per_camera"], 4999)
        self.assertEqual(contract["controller_pcd_cap_stage"], "before_tracking_query_and_fusion")
        self.assertEqual(contract["cotracker_seed"], 42)
        self.assertTrue(contract["wait_for_tracking_overlay"])
        self.assertTrue(contract["tracking_overlay_required_before_first_render"])
        self.assertTrue(contract["tracking_overlay_required_for_render"])
        self.assertTrue(contract["render_requires_new_cotracker_result"])
        self.assertFalse(contract["render_reuses_cached_cotracker_result"])
        self.assertEqual(contract["tracking_overlay_color_rgb"], [255, 0, 0])
        self.assertEqual(contract["tracking_overlay_color_mode"], "solid")
        self.assertFalse(contract["tracking_overlay_debug_color_by_camera"])
        self.assertEqual(contract["tracker_visualization_mode"], "3d-surface-markers")
        self.assertEqual(contract["tracker_3d_marker_mode"], "surface_snap")
        self.assertEqual(contract["tracker_3d_marker_shape"], "sphere")
        self.assertFalse(contract["tracker_legacy_lift_used"])
        self.assertEqual(contract["tracker_3d_snap_radius_px"], 4.0)
        self.assertEqual(contract["tracker_3d_marker_radius_m"], 0.006)
        self.assertEqual(contract["tracker_control_points_per_camera"], 16)
        self.assertEqual(contract["tracker_control_point_selection"], "visible-spread")
        self.assertEqual(contract["tracking_overlay_lift_method"], "surface_snap")
        self.assertEqual(contract["overlay_max_points_per_camera"], 0)
        self.assertEqual(contract["overlay_display_scope"], "controller")
        self.assertEqual(contract["overlay_display_classification"], "first_frame_mask_membership")
        self.assertTrue(contract["overlay_bbox_filter_enabled"])
        self.assertEqual(contract["overlay_bbox_filter_scope"], "controller")
        self.assertEqual(contract["overlay_bbox_filter_margin_m"], 0.15)
        self.assertTrue(contract["tracking_control_point_markers"])
        self.assertEqual(contract["tracking_control_point_count_requested"], 48)
        self.assertEqual(contract["tracking_control_points_per_camera"], 16)
        self.assertEqual(contract["tracking_control_point_radius_m"], 0.006)
        self.assertFalse(contract["overlay_render_raw_track_points"])
        self.assertEqual(contract["tracking_pending_render_packet_max_groups"], 128)
        self.assertEqual(contract["tracking_pending_fusion_bundle_max_groups"], 128)
        self.assertEqual(
            contract["tracking_render_packet_match_policy"],
            "exact-target-bundle",
        )
        self.assertEqual(contract["frame_bundle_policy"], "exact-target")
        self.assertTrue(contract["render_bundle_exact_target_default"])
        self.assertFalse(contract["tracker_child_receives_full_frame_bundle"])
        self.assertFalse(contract["tracker_child_receives_depth"])
        self.assertFalse(contract["tracker_child_receives_intrinsics"])
        self.assertFalse(contract["tracker_child_receives_c2w"])
        self.assertEqual(contract["pcd_fusion_trigger"], "tracker-result")
        self.assertTrue(contract["tracker_result_gated_fusion"])
        self.assertTrue(contract["render_triggers_pcd_fusion"])
        self.assertFalse(contract["phystwin_dense_compatible"])
        self.assertEqual(contract["cotracker_backend"], "tapnextpp")
        self.assertEqual(contract["tracker_backend"], "tapnextpp")
        self.assertEqual(contract["tracker_backend_family"], "tapnext")
        self.assertEqual(contract["tracking_backend_online_semantics"], "stateful_frame_by_frame")
        self.assertEqual(contract["tapnextpp_checkpoint"], "checkpoints/tapnextpp/tapnextpp_ckpt.pt")
        self.assertEqual(contract["tapnextpp_image_size"], [256, 256])
        self.assertEqual(contract["tapnextpp_autocast_dtype"], "fp16")
        self.assertTrue(contract["tapnextpp_fast_postprocess"])
        self.assertEqual(contract["tracking_backend_execution_mode"], "batch-views")
        self.assertEqual(contract["tracking_backend_batch_dimension"], "camera")
        self.assertEqual(contract["tracking_backend_batch_size"], 3)
        self.assertTrue(contract["tracking_backend_batch_supported"])
        self.assertFalse(contract["tracking_backend_batch_auto_selected"])
        self.assertEqual(contract["tracker_batch_query_count_policy"], "fixed")
        self.assertEqual(contract["cotracker_owner"], "process")
        self.assertEqual(contract["cotracker_process_mode"], "subprocess")
        self.assertEqual(contract["cotracker_update_mode"], "batch")
        self.assertEqual(contract["cotracker_batch_size_target"], 3)
        self.assertFalse(contract["cotracker_batch_fallback_enabled"])
        self.assertFalse(contract["cross_gpu_cuda_tensor_transfer"])
        self.assertEqual(contract["ipc_payload"], "cpu_numpy_latest_wins")
        self.assertFalse(contract["tracking_input_contains_depth"])
        self.assertEqual(contract["shared_runtime_tracking_backend"], "none")
        self.assertTrue(contract["render_waited_for_cotracker"])
        self.assertTrue(contract["render_waited_for_fresh_cotracker_result"])
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
        self.assertIn("tracking_query_count_requested = 4096", output)
        self.assertIn("trackable_mask_source = standard_filter_survivors", output)
        self.assertIn("tracking_input_mask_semantics = standard_filter_trackable_masks", output)
        self.assertIn("tracker_query_source = union_trackable_mask", output)
        self.assertIn("object_point_control = fixed-cap", output)
        self.assertIn("object_postprocess = enhanced-pt", output)
        self.assertIn("controller_postprocess = enhanced-pt", output)
        self.assertIn("object_enhanced_keep_top_n_components = 1", output)
        self.assertIn("controller_enhanced_keep_top_n_components = 2", output)
        self.assertIn("enhanced_component_selection_policy = largest-n-plus-gap", output)
        self.assertIn("apply_enhanced_component_filter_to_pcd = true", output)
        self.assertIn("controller_pcd_max_points_per_camera = 4999", output)
        self.assertIn("controller_pcd_cap_stage = before_tracking_query_and_fusion", output)
        self.assertIn("wait_for_tracking_overlay = true", output)
        self.assertIn("render_requires_new_cotracker_result = true", output)
        self.assertIn("render_reuses_cached_cotracker_result = false", output)
        self.assertIn("tracker_visualization_mode = 3d-surface-markers", output)
        self.assertIn("tracker_3d_marker_mode = surface_snap", output)
        self.assertIn("tracker_3d_marker_shape = sphere", output)
        self.assertIn("tracker_legacy_lift_used = false", output)
        self.assertIn("tracker_3d_snap_radius_px = 4.0", output)
        self.assertIn("tracker_3d_marker_radius_m = 0.006", output)
        self.assertIn("tracker_control_points_per_camera = 16", output)
        self.assertIn("overlay_display_scope = controller", output)
        self.assertIn("overlay_bbox_filter_enabled = true", output)
        self.assertIn("overlay_bbox_filter_margin_m = 0.15", output)
        self.assertIn("tracking_control_point_markers = true", output)
        self.assertIn("tracking_control_point_count_requested = 48", output)
        self.assertIn("tracking_control_point_radius_m = 0.006", output)
        self.assertIn("overlay_render_raw_track_points = false", output)
        self.assertIn("tracking_pending_render_packet_max_groups = 128", output)
        self.assertIn("tracking_pending_fusion_bundle_max_groups = 128", output)
        self.assertIn("frame_bundle_policy = exact-target", output)
        self.assertIn("tracking_render_packet_match_policy = exact-target-bundle", output)
        self.assertIn("pcd_fusion_trigger = tracker-result", output)
        self.assertIn("tracker_result_gated_fusion = true", output)
        self.assertIn("render_triggers_pcd_fusion = true", output)
        self.assertIn("phystwin_dense_compatible = false", output)
        self.assertIn("cotracker_owner = process", output)
        self.assertIn("tracker_backend = tapnextpp", output)
        self.assertIn("tracker_backend_family = tapnext", output)
        self.assertIn("tracking_backend_online_semantics = stateful_frame_by_frame", output)
        self.assertIn("tapnextpp_checkpoint = checkpoints/tapnextpp/tapnextpp_ckpt.pt", output)
        self.assertIn("output_root = result/demo31_dual4090_realsense_tapnextpp", output)
        self.assertIn("tracking_backend_execution_mode = batch-views", output)
        self.assertIn("tracking_backend_batch_dimension = camera", output)
        self.assertIn("cotracker_update_mode = batch", output)
        self.assertIn("cross_gpu_cuda_tensor_transfer = false", output)
        self.assertIn("pcd_color_mode = rgb", output)
        self.assertIn("render_waited_for_cotracker = true", output)
        self.assertIn("render_waited_for_fresh_cotracker_result = true", output)

    def test_topn_enhanced_pt_cli_fields_are_in_contract(self) -> None:
        args = self._parse(
            [
                "--camera-ids",
                "0,1,2",
                "--controller-enhanced-keep-top-n-components",
                "3",
                "--object-enhanced-keep-top-n-components",
                "2",
                "--enhanced-component-selection-policy",
                "largest-n",
                "--enhanced-min-component-points",
                "64",
                "--enhanced-min-component-ratio",
                "0.01",
                "--no-apply-enhanced-component-filter-to-pcd",
            ]
        )

        contract = demo31_runtime.build_contract(args, cuda_device_count_provider=lambda: 2)

        self.assertEqual(contract["object_enhanced_keep_top_n_components"], 2)
        self.assertEqual(contract["controller_enhanced_keep_top_n_components"], 3)
        self.assertEqual(contract["enhanced_component_selection_policy"], "largest-n")
        self.assertEqual(contract["enhanced_min_component_points"], 64)
        self.assertEqual(contract["enhanced_min_component_ratio"], 0.01)
        self.assertFalse(contract["apply_enhanced_component_filter_to_pcd"])
        self.assertEqual(contract["trackable_object_filter"]["keep_top_n_components"], 2)
        self.assertEqual(contract["trackable_controller_filter"]["keep_top_n_components"], 3)
        self.assertEqual(contract["render_object_filter"]["keep_top_n_components"], 2)
        self.assertEqual(contract["render_controller_filter"]["keep_top_n_components"], 3)

    def test_trackable_mask_policy_disabled_reports_raw_semantic_query_masks(self) -> None:
        args = self._parse(
            [
                "--dry-run",
                "--camera-ids",
                "0,1,2",
                "--mask-gpu",
                "0",
                "--cotracker-gpu",
                "1",
                "--trackable-mask-build-policy",
                "disabled",
            ]
        )
        contract = demo31_runtime.build_contract(args, cuda_device_count_provider=lambda: 2)

        self.assertEqual(contract["trackable_mask_source"], "raw_semantic_union")
        self.assertEqual(contract["tracking_input_mask_semantics"], "raw_semantic_masks")
        self.assertEqual(contract["tracker_query_source"], "object_controller_union_mask")
        self.assertEqual(contract["object_mask_semantics"], "raw_object_mask")
        self.assertEqual(contract["controller_mask_semantics"], "raw_controller_mask")

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
        self.assertEqual(config.cotracker_backend, "tapnextpp")
        self.assertEqual(config.tapnet_repo_dir, "external/tapnet")
        self.assertEqual(config.tapnextpp_checkpoint, "checkpoints/tapnextpp/tapnextpp_ckpt.pt")
        self.assertEqual(config.tapnextpp_image_size, (256, 256))
        self.assertEqual(config.backend_execution_mode, "batch-views")
        self.assertEqual(config.tracker_batch_query_count_policy, "fixed")
        self.assertEqual(config.query_mode, "phystwin_dense")
        self.assertEqual(config.query_count_request, "4096")
        self.assertEqual(config.seed, 42)
        self.assertEqual(config.update_mode, "batch")
        self.assertTrue(config.init_requires_object_and_controller)
        self.assertEqual(config.overlay_display_scope, "controller")

    def test_trackon2_backend_contract_is_accepted(self) -> None:
        args = self._parse(
            [
                "--dry-run",
                "--camera-ids",
                "0,1,2",
                "--mask-gpu",
                "0",
                "--cotracker-gpu",
                "1",
                "--cotracker-backend",
                "trackon2",
                "--tracking-backend-execution-mode",
                "batch-views",
                "--tracker-batch-query-count-policy",
                "min-common",
                "--trackon2-checkpoint",
                "/tmp/trackon2.pth",
                "--trackon2-repo-dir",
                "/tmp/track_on",
            ]
        )
        demo31_runtime.validate_args(args, cuda_device_count_provider=lambda: 2)
        contract = demo31_runtime.build_contract(args, cuda_device_count_provider=lambda: 2)
        config = demo31_runtime.build_cotracker_process_config(args)

        self.assertEqual(contract["cotracker_backend"], "trackon2")
        self.assertEqual(contract["tracker_backend"], "trackon2")
        self.assertEqual(contract["tracker_backend_family"], "trackon")
        self.assertEqual(contract["tracking_backend_execution_mode"], "batch-views")
        self.assertEqual(contract["tracker_batch_query_count_policy"], "min-common")
        self.assertEqual(config.cotracker_backend, "trackon2")
        self.assertEqual(config.backend_execution_mode, "batch-views")
        self.assertEqual(config.tracker_batch_query_count_policy, "min-common")

    def test_trackon2_live_realsense_validation_uses_demo31_backend_rules(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            calibrate_path = f"{tmp_dir}/calibrate.pkl"
            with open(calibrate_path, "wb") as handle:
                pickle.dump(np.repeat(np.eye(4, dtype=np.float32)[None, :, :], 3, axis=0), handle)
            args = self._parse(
                [
                    "--camera-ids",
                    "0,1,2",
                    "--mask-gpu",
                    "0",
                    "--cotracker-gpu",
                    "1",
                    "--cotracker-backend",
                    "trackon2",
                    "--trackon2-checkpoint",
                    "/tmp/trackon2.pth",
                    "--trackon2-repo-dir",
                    "/tmp/track_on",
                    "--calibrate-path",
                    calibrate_path,
                ]
            )

            validation = demo31_runtime.validate_live_realsense_contract(
                args,
                connected_serials_provider=lambda: ["s0", "s1", "s2"],
                cuda_device_count_provider=lambda: 2,
            )

        self.assertEqual(validation["active_serials"], ["s0", "s1", "s2"])
        self.assertEqual(validation["calibration_transform_count"], 3)

    def test_litetracker_auto_contract_marks_experimental_batch_support(self) -> None:
        args = self._parse(
            [
                "--dry-run",
                "--camera-ids",
                "0,1,2",
                "--mask-gpu",
                "0",
                "--cotracker-gpu",
                "1",
                "--cotracker-backend",
                "litetracker",
                "--litetracker-weights",
                "/tmp/litetracker.pth",
            ]
        )
        contract = demo31_runtime.build_contract(args, cuda_device_count_provider=lambda: 2)

        self.assertEqual(contract["tracker_backend"], "litetracker")
        self.assertEqual(contract["tracker_backend_family"], "litetracker")
        self.assertTrue(contract["tracking_backend_batch_supported"])
        self.assertEqual(contract["tracking_backend_batch_support_status"], "experimental_batch_views")
        self.assertEqual(contract["tracking_backend_batch_dimension"], "camera")

    def test_locotrack_backend_contract_accepts_serial_and_batch_views(self) -> None:
        common = [
            "--dry-run",
            "--camera-ids",
            "0,1,2",
            "--mask-gpu",
            "0",
            "--cotracker-gpu",
            "1",
            "--cotracker-backend",
            "locotrack",
            "--locotrack-repo-dir",
            "external/locotrack/locotrack_pytorch",
            "--locotrack-checkpoint",
            "checkpoints/locotrack/locotrack_small.ckpt",
            "--locotrack-model-size",
            "small",
            "--locotrack-window-frames",
            "8",
            "--locotrack-resolution",
            "256x256",
            "--locotrack-query-chunk-size",
            "256",
            "--locotrack-autocast-dtype",
            "bf16",
        ]
        for mode, batch_dimension, batch_size, update_mode in (
            ("serial", "none", 1, "serial"),
            ("batch-views", "camera", 3, "batch"),
        ):
            with self.subTest(mode=mode):
                args = self._parse([*common, "--tracking-backend-execution-mode", mode])
                demo31_runtime.validate_args(args, cuda_device_count_provider=lambda: 2)
                contract = demo31_runtime.build_contract(args, cuda_device_count_provider=lambda: 2)
                config = demo31_runtime.build_cotracker_process_config(args)

                self.assertEqual(contract["cotracker_backend"], "locotrack")
                self.assertEqual(contract["tracker_backend"], "locotrack")
                self.assertEqual(contract["tracker_backend_family"], "locotrack")
                self.assertEqual(contract["tracking_backend_execution_mode"], mode)
                self.assertEqual(contract["tracking_backend_batch_dimension"], batch_dimension)
                self.assertEqual(contract["tracking_backend_batch_size"], batch_size)
                self.assertTrue(contract["tracking_backend_supports_batch_views"])
                self.assertEqual(contract["tracking_backend_online_semantics"], "windowed")
                self.assertEqual(contract["tracking_backend_batch_support_status"], "windowed_batch_views")
                self.assertEqual(contract["locotrack_model_size"], "small")
                self.assertEqual(contract["locotrack_window_frames"], 8)
                self.assertEqual(contract["locotrack_resolution"], [256, 256])
                self.assertEqual(contract["locotrack_query_chunk_size"], 256)
                self.assertEqual(contract["locotrack_autocast_dtype"], "bf16")
                self.assertEqual(contract["locotrack_checkpoint"], "checkpoints/locotrack/locotrack_small.ckpt")
                self.assertEqual(contract["locotrack_repo_dir"], "external/locotrack/locotrack_pytorch")
                self.assertEqual(config.cotracker_backend, "locotrack")
                self.assertEqual(config.backend_execution_mode, mode)
                self.assertEqual(config.update_mode, update_mode)
                self.assertEqual(config.locotrack_resolution, (256, 256))

    def test_tracking_backend_alias_selects_locotrack(self) -> None:
        args = self._parse(
            [
                "--dry-run",
                "--camera-ids",
                "0,1,2",
                "--mask-gpu",
                "0",
                "--cotracker-gpu",
                "1",
                "--tracking-backend",
                "locotrack",
            ]
        )
        contract = demo31_runtime.build_contract(args, cuda_device_count_provider=lambda: 2)

        self.assertEqual(contract["tracker_backend"], "locotrack")
        self.assertEqual(contract["tracker_backend_family"], "locotrack")

    def test_tapnextpp_backend_contract_accepts_serial_and_batch_views(self) -> None:
        common = [
            "--dry-run",
            "--camera-ids",
            "0,1,2",
            "--mask-gpu",
            "0",
            "--cotracker-gpu",
            "1",
            "--cotracker-backend",
            "tapnextpp",
            "--tapnet-repo-dir",
            "external/tapnet",
            "--tapnextpp-checkpoint",
            "checkpoints/tapnextpp/tapnextpp_ckpt.pt",
            "--tapnextpp-image-size",
            "256,256",
            "--tapnextpp-autocast-dtype",
            "fp16",
        ]
        for mode, batch_dimension, batch_size, update_mode, expected_instances in (
            ("serial", "none", 1, "serial", 3),
            ("batch-views", "camera", 3, "batch", 1),
        ):
            with self.subTest(mode=mode):
                args = self._parse([*common, "--tracking-backend-execution-mode", mode])
                demo31_runtime.validate_args(args, cuda_device_count_provider=lambda: 2)
                contract = demo31_runtime.build_contract(args, cuda_device_count_provider=lambda: 2)
                config = demo31_runtime.build_cotracker_process_config(args)

                self.assertEqual(contract["cotracker_backend"], "tapnextpp")
                self.assertEqual(contract["tracker_backend"], "tapnextpp")
                self.assertEqual(contract["tracker_backend_family"], "tapnext")
                self.assertEqual(contract["tracking_backend_execution_mode"], mode)
                self.assertEqual(contract["tracking_backend_batch_dimension"], batch_dimension)
                self.assertEqual(contract["tracking_backend_batch_size"], batch_size)
                self.assertEqual(contract["tracking_backend_model_instances_expected"], expected_instances)
                self.assertTrue(contract["tracking_backend_supports_batch_views"])
                self.assertTrue(contract["tracking_backend_supports_online"])
                self.assertEqual(contract["tracking_backend_online_semantics"], "stateful_frame_by_frame")
                self.assertEqual(contract["tracking_backend_batch_support_status"], "true_online_batch_views")
                self.assertEqual(contract["tapnet_repo_dir"], "external/tapnet")
                self.assertEqual(contract["tapnextpp_checkpoint"], "checkpoints/tapnextpp/tapnextpp_ckpt.pt")
                self.assertEqual(contract["tapnextpp_image_size"], [256, 256])
                self.assertEqual(contract["tapnextpp_autocast_dtype"], "fp16")
                self.assertTrue(contract["tapnextpp_fast_postprocess"])
                self.assertEqual(contract["tapnextpp_frame_value_range"], "minus1_1_float")
                self.assertEqual(config.cotracker_backend, "tapnextpp")
                self.assertEqual(config.backend_execution_mode, mode)
                self.assertEqual(config.update_mode, update_mode)
                self.assertEqual(config.tapnextpp_image_size, (256, 256))
                self.assertTrue(config.tapnextpp_fast_postprocess)

        args = self._parse(
            [
                *common,
                "--tracking-backend-execution-mode",
                "batch-views",
                "--no-tapnextpp-fast-postprocess",
            ]
        )
        contract = demo31_runtime.build_contract(args, cuda_device_count_provider=lambda: 2)
        config = demo31_runtime.build_cotracker_process_config(args)
        self.assertFalse(contract["tapnextpp_fast_postprocess"])
        self.assertFalse(config.tapnextpp_fast_postprocess)

    def test_demo32_defaults_to_ffs_batch3_litetracker_batch3(self) -> None:
        args = self._parse(
            ["--dry-run", "--camera-ids", "0,1,2", "--mask-gpu", "0", "--cotracker-gpu", "1"],
            default_preset=demo31_runtime.PRESET_DEMO32_FFS_LITETRACKER,
        )
        demo31_runtime.validate_args(args, cuda_device_count_provider=lambda: 2)
        contract = demo31_runtime.build_contract(args, cuda_device_count_provider=lambda: 2)
        config = demo31_runtime.build_cotracker_process_config(args)

        self.assertEqual(contract["demo"], "demo3.2")
        self.assertEqual(contract["preset"], "demo3.2-ffs-litetracker")
        self.assertEqual(contract["depth_source"], "ffs")
        self.assertTrue(contract["uses_ffs"])
        self.assertTrue(contract["async_depth_pipeline"])
        self.assertEqual(
            contract["pipeline_order"],
            [
                "capture",
                "ffs_batch3_opt5_depth",
                "edgetam",
                "litetracker_batch3",
                "render_and_diagnostics",
            ],
        )
        self.assertEqual(contract["shared_runtime_preset"], "demo2.3-dual4090-maxfps")
        self.assertEqual(contract["shared_runtime_gpu_pipeline_mode"], "dual-gpu-split")
        self.assertEqual(contract["shared_runtime_gpu_placement"], "ffs_edgetam_gpu0_litetracker_gpu1")
        self.assertEqual(contract["ffs_gpu_physical"], 0)
        self.assertEqual(contract["edgetam_gpu_physical"], 0)
        self.assertEqual(contract["sam31_gpu_physical"], 0)
        self.assertEqual(contract["litetracker_gpu_physical"], 1)
        self.assertTrue(contract["ffs_edgetam_same_gpu"])
        self.assertEqual(contract["cotracker_backend"], "litetracker")
        self.assertEqual(contract["tracker_backend"], "litetracker")
        self.assertEqual(contract["litetracker_runtime"], "pytorch")
        self.assertIsNone(contract["litetracker_onnx_dir"])
        self.assertFalse(contract["litetracker_export_onnx"])
        self.assertEqual(contract["litetracker_onnx_opset"], 17)
        self.assertEqual(contract["litetracker_onnx_opset_actual"], 18)
        self.assertEqual(contract["litetracker_onnx_optimization_level"], 5)
        self.assertEqual(contract["tracking_backend_execution_mode"], "batch-views")
        self.assertEqual(contract["cotracker_update_mode"], "batch")
        self.assertFalse(contract["cotracker_prewarm_backends"])
        self.assertFalse(contract["tracker_prewarm_backends"])
        self.assertEqual(contract["tracker_prewarm_mode"], "lazy_query_init")
        self.assertEqual(contract["tracker_ready_state"], "ready_to_receive_inputs")
        self.assertTrue(contract["tracker_query_dependent_init"])
        self.assertTrue(contract["tracker_query_dependent_init_pending_until_first_input"])
        self.assertTrue(contract["sam31_init_quick_fail_empty_masks"])
        self.assertEqual(contract["sam31_init_min_mask_pixels"], 1)
        self.assertEqual(contract["sam31_init_required_masks"], ["stuffed animal", "towel"])
        self.assertEqual(contract["trackable_mask_build_policy"], "init-only")
        self.assertEqual(contract["trackable_query_init_strategy"], "standard-filter-init")
        self.assertEqual(contract["trackable_mask_source"], "standard_filter_survivors")
        self.assertEqual(contract["tracking_input_mask_semantics"], "standard_filter_trackable_masks")
        self.assertEqual(contract["tracker_query_source"], "union_trackable_mask")
        self.assertEqual(contract["object_point_control"], "fixed-cap")
        self.assertEqual(contract["object_postprocess"], "enhanced-pt")
        self.assertEqual(contract["controller_postprocess"], "enhanced-pt")
        self.assertEqual(contract["trackable_object_filter"]["mode"], "enhanced-pt")
        self.assertEqual(contract["trackable_object_filter"]["point_control"], "fixed-cap")
        self.assertEqual(contract["trackable_controller_filter"]["mode"], "enhanced-pt")
        self.assertEqual(contract["object_filter"]["mode"], "enhanced-pt")
        self.assertEqual(contract["object_filter"]["point_control"], "fixed-cap")
        self.assertEqual(contract["controller_filter"]["mode"], "enhanced-pt")
        self.assertEqual(contract["object_enhanced_keep_top_n_components"], 1)
        self.assertEqual(contract["controller_enhanced_keep_top_n_components"], 2)
        self.assertEqual(contract["enhanced_component_selection_policy"], "largest-n-plus-gap")
        self.assertTrue(contract["apply_enhanced_component_filter_to_pcd"])
        self.assertEqual(contract["render_object_filter"]["point_control"], "fixed-cap")
        self.assertEqual(contract["render_object_filter"]["postprocess"], "enhanced-pt")
        self.assertEqual(contract["controller_trackable_max_points_per_camera"], 4999)
        self.assertEqual(contract["controller_trackable_cap_stage"], "after_enhanced_pt_top_n_component_filter")
        self.assertEqual(contract["controller_mask_erode_px"], 0)
        self.assertEqual(contract["controller_mask_erode_stage"], "before_tracking_union_and_trackable_filter")
        self.assertEqual(contract["controller_mask_erode_applies_to"], "tracking_input_and_anchor_masks")
        self.assertEqual(contract["render_controller_filter"]["render_voxel_m"], 0.003)
        self.assertTrue(contract["render_controller_filter"]["render_voxel_downsample"])
        self.assertEqual(contract["render_controller_filter"]["render_max_points"], 10000)
        self.assertTrue(contract["render_controller_filter"]["render_cap_enabled"])
        self.assertTrue(contract["render_controller_filter"]["render_only"])
        self.assertFalse(contract["render_controller_filter"]["affects_tracking_markers"])
        self.assertEqual(contract["tracker_visualization_mode"], "all-tracks-3d-lift")
        self.assertEqual(contract["tracker_3d_marker_mode"], "all-tracks-3d-lift")
        self.assertTrue(contract["tracker_direct_depth_lift_used"])
        self.assertTrue(contract["tracker_all_tracks_anchor_mode"])
        self.assertFalse(contract["tracker_surface_gate_enabled"])
        self.assertFalse(contract["overlay_bbox_filter_enabled"])
        self.assertEqual(contract["tracking_overlay_lift_method"], "all_tracks_depth_lift")
        self.assertEqual(contract["tracking_control_point_count_requested"], 0)
        self.assertEqual(
            contract["tracking_control_point_sampling"],
            "all_visible_depth_valid_tracks_no_surface_or_bbox_gate",
        )
        self.assertEqual(contract["tracker_env_name"], "demo_3_1_max")
        self.assertEqual(contract["tracking_backend_batch_dimension"], "camera")
        self.assertTrue(contract["tracking_backend_batch_supported"])
        self.assertEqual(contract["tracking_backend_batch_size"], 3)
        self.assertEqual(contract["tracking_backend_batch_support_status"], "experimental_batch_views")
        self.assertFalse(contract["tracking_backend_batch_auto_selected"])
        self.assertEqual(contract["tracker_batch_query_count_policy"], "min-common")
        self.assertNotIn("ffs", contract["hot_path_forbids"])
        self.assertNotIn("ffs_tensorrt", contract["hot_path_forbids"])
        self.assertEqual(contract["ffs_contract"]["builderOptimizationLevel"], 5)
        self.assertEqual(contract["ffs_contract"]["trt_batch_size"], 3)
        self.assertTrue(contract["ffs_contract"]["batch3_isolated_artifact"])
        self.assertIn("batch3", contract["ffs_contract"]["trt_model_dir"])
        self.assertTrue(contract["profile_summary_fields"]["uses_ffs"])
        self.assertEqual(contract["profile_summary_fields"]["depth_source"], "ffs")
        self.assertEqual(config.cotracker_backend, "litetracker")
        self.assertEqual(config.backend_execution_mode, "batch-views")
        self.assertEqual(config.update_mode, "batch")
        self.assertEqual(config.tracker_batch_query_count_policy, "min-common")
        self.assertFalse(config.prewarm_backends)
        self.assertEqual(config.tracker_prewarm_mode, "lazy_query_init")
        self.assertTrue(config.tracker_query_dependent_init)
        self.assertEqual(config.litetracker_repo_dir, "/home/xinjie/external/lite-tracker")
        self.assertEqual(config.litetracker_weights, "/home/xinjie/external/weights/cotracker3/scaled_online.pth")
        self.assertEqual(config.litetracker_runtime, "pytorch")
        self.assertIsNone(config.litetracker_onnx_dir)
        self.assertFalse(config.litetracker_export_onnx)
        self.assertEqual(config.litetracker_onnx_opset, 17)
        self.assertEqual(config.litetracker_onnx_optimization_level, 5)

    def test_demo32_accepts_litetracker_serial_onnx_runtime(self) -> None:
        args = self._parse(
            [
                "--dry-run",
                "--camera-ids",
                "0,1,2",
                "--mask-gpu",
                "0",
                "--cotracker-gpu",
                "1",
                "--tracking-backend-execution-mode",
                "serial",
                "--litetracker-runtime",
                "onnx-cuda",
                "--litetracker-onnx-dir",
                "result/litetracker_onnx",
                "--litetracker-export-onnx",
            ],
            default_preset=demo31_runtime.PRESET_DEMO32_FFS_LITETRACKER,
        )
        demo31_runtime.validate_args(args, cuda_device_count_provider=lambda: 2)
        contract = demo31_runtime.build_contract(args, cuda_device_count_provider=lambda: 2)
        config = demo31_runtime.build_cotracker_process_config(args)

        self.assertEqual(contract["pipeline_order"][3], "litetracker_serial")
        self.assertEqual(contract["tracking_backend_execution_mode"], "serial")
        self.assertEqual(contract["cotracker_update_mode"], "serial")
        self.assertEqual(contract["tracking_backend_batch_dimension"], "none")
        self.assertEqual(contract["tracking_backend_batch_size"], 1)
        self.assertEqual(contract["litetracker_runtime"], "onnx-cuda")
        self.assertEqual(contract["litetracker_onnx_dir"], "result/litetracker_onnx")
        self.assertTrue(contract["litetracker_export_onnx"])
        self.assertEqual(contract["litetracker_onnx_opset"], 17)
        self.assertEqual(contract["litetracker_onnx_opset_actual"], 18)
        self.assertEqual(contract["litetracker_onnx_optimization_level"], 5)
        self.assertEqual(contract["profile_summary_fields"]["litetracker_runtime"], "onnx-cuda")
        self.assertEqual(contract["profile_summary_fields"]["litetracker_onnx_dir"], "result/litetracker_onnx")
        self.assertEqual(contract["profile_summary_fields"]["litetracker_onnx_opset_actual"], 18)
        self.assertEqual(config.backend_execution_mode, "serial")
        self.assertEqual(config.update_mode, "serial")
        self.assertEqual(config.litetracker_runtime, "onnx-cuda")
        self.assertEqual(config.litetracker_onnx_dir, "result/litetracker_onnx")
        self.assertTrue(config.litetracker_export_onnx)
        self.assertEqual(config.litetracker_onnx_opset, 17)
        self.assertEqual(config.litetracker_onnx_optimization_level, 5)

    def test_demo32_rejects_litetracker_onnx_batch_runtime(self) -> None:
        args = self._parse(
            [
                "--dry-run",
                "--camera-ids",
                "0,1,2",
                "--mask-gpu",
                "0",
                "--cotracker-gpu",
                "1",
                "--litetracker-runtime",
                "onnx-cuda",
                "--litetracker-onnx-dir",
                "result/litetracker_onnx",
            ],
            default_preset=demo31_runtime.PRESET_DEMO32_FFS_LITETRACKER,
        )

        with self.assertRaisesRegex(ValueError, "serial-only"):
            demo31_runtime.validate_args(args, cuda_device_count_provider=lambda: 2)

    def test_controller_mask_erode_defaults_to_zero_in_demo_mode(self) -> None:
        for preset in (
            demo31_runtime.PRESET_DEMO31_DUAL4090_HIGHFPS,
            demo31_runtime.PRESET_DEMO32_FFS_LITETRACKER,
        ):
            with self.subTest(preset=preset):
                args = self._parse(
                    ["--dry-run", "--mode", "demo", "--camera-ids", "0,1,2", "--mask-gpu", "0", "--cotracker-gpu", "1"],
                    default_preset=preset,
                )
                contract = demo31_runtime.build_contract(args, cuda_device_count_provider=lambda: 2)

                self.assertEqual(args.controller_mask_erode_px, 0)
                self.assertEqual(contract["semantic_mode"], "demo")
                self.assertEqual(contract["controller_prompt"], "human hand")
                self.assertEqual(contract["tracking_controller_label"], "hand")
                self.assertEqual(contract["controller_mask_erode_px"], 0)

    def test_controller_mask_erode_explicit_override_wins_in_demo_mode(self) -> None:
        args = self._parse(
            [
                "--dry-run",
                "--mode",
                "demo",
                "--controller-mask-erode-px",
                "2",
                "--camera-ids",
                "0,1,2",
                "--mask-gpu",
                "0",
                "--cotracker-gpu",
                "1",
            ],
            default_preset=demo31_runtime.PRESET_DEMO32_FFS_LITETRACKER,
        )
        contract = demo31_runtime.build_contract(args, cuda_device_count_provider=lambda: 2)

        self.assertEqual(args.controller_mask_erode_px, 2)
        self.assertEqual(contract["semantic_mode"], "demo")
        self.assertEqual(contract["controller_mask_erode_px"], 2)

    def test_controller_mask_erode_happens_before_tracking_union(self) -> None:
        object_mask = np.zeros((5, 5), dtype=bool)
        object_mask[0, 0] = True
        controller_mask = np.zeros((5, 5), dtype=bool)
        controller_mask[1:4, 1:4] = True
        packet = _FakePacket(
            group_id=1,
            camera_idx=0,
            object_mask=object_mask,
            controller_mask=controller_mask,
            color_bgr=np.zeros((5, 5, 3), dtype=np.uint8),
        )

        union_mask, returned_object, eroded_controller = demo31_runtime._phystwin_union_tracking_masks(
            packet,
            controller_mask_erode_px=1,
        )

        expected_controller = np.zeros((5, 5), dtype=bool)
        expected_controller[2, 2] = True
        np.testing.assert_array_equal(returned_object, object_mask)
        np.testing.assert_array_equal(eroded_controller, expected_controller)
        np.testing.assert_array_equal(union_mask, object_mask | expected_controller)

    def test_demo32_has_independent_runtime_contract(self) -> None:
        parser = demo32_runtime.build_arg_parser()
        args = parser.parse_args(["--dry-run", "--camera-ids", "0,1,2", "--mask-gpu", "0", "--cotracker-gpu", "1"])
        args = demo32_runtime.apply_preset_defaults(args, explicit_options={"--dry-run", "--camera-ids", "--mask-gpu", "--cotracker-gpu"})
        contract = demo32_runtime.build_contract(args, cuda_device_count_provider=lambda: 2)

        self.assertEqual(contract["demo"], "demo3.2")
        self.assertEqual(contract["runtime_module"], "qqtt.demo.demo32_runtime")
        self.assertEqual(contract["runtime_owner"], "demo32_litetracker_ffs")
        self.assertTrue(contract["independent_demo_runtime"])
        self.assertFalse(contract["derived_from_demo31_preset"])
        self.assertFalse(contract["delegates_to_demo23_entrypoint"])
        self.assertTrue(contract["tracker_result_required_for_render"])
        self.assertTrue(contract["tracker_marker_required_for_render"])
        self.assertIn("raw_fused_async", contract["tracker_input_publish_hooks"])
        self.assertEqual(contract["tracker_prewarm_mode"], "lazy_query_init")
        self.assertEqual(contract["tracker_ready_state"], "ready_to_receive_inputs")

    def test_demo32_raw_async_path_publishes_litetracker_input_and_surface_anchors(self) -> None:
        client = _FakeProcessClient(None)
        runtime_cls = demo32_runtime.make_demo32_live_runtime_class(
            _FakeSharedRuntimeModule,
            process_client_factory=lambda _config: client,
        )
        runtime = runtime_cls(
            SimpleNamespace(
                camera_ids=(0, 1),
                depth_min_m=0.0,
                depth_max_m=3.0,
                object_point_control="none",
                object_postprocess="none",
                controller_postprocess="none",
                object_volume_voxel_m=0.005,
                object_volume_origin="world",
                object_volume_points_per_voxel=1,
                phystwin_radius_m=0.01,
                phystwin_nb_points=1,
                enhanced_component_voxel_size_m=0.006,
                enhanced_keep_near_main_gap_m=0.035,
            ),
            demo31_contract={
                "fusion_mask_policy": "latest-reuse",
                "mask_stale_timeout_ms": 250.0,
                "cotracker_result_stale_timeout_ms": 1500.0,
                "cotracker_input_fps": 10.0,
                "controller_pcd_max_points_per_camera": 100,
                "controller_trackable_max_points_per_camera": 100,
                "trackable_mask_build_policy": "init-only",
                "trackable_query_init_strategy": "standard-filter-init",
                "demo": "demo3.2",
                "cotracker_seed": 42,
                "wait_for_tracking_overlay": True,
            },
            cotracker_process_config=SimpleNamespace(),
        )
        runtime._stream_metadata = [
            {"K_color": np.eye(3, dtype=np.float32)},
            {"K_color": np.eye(3, dtype=np.float32)},
        ]
        runtime._c2w_by_camera = {
            0: np.eye(4, dtype=np.float32),
            1: np.eye(4, dtype=np.float32),
        }
        depth_group = _FakeDepthGroup(
            group_id=7,
            depths={
                0: _FakeDepthFrame(7, np.ones((2, 2), dtype=np.float32)),
                1: _FakeDepthFrame(7, np.ones((2, 2), dtype=np.float32) * 2.0),
            },
            per_camera_frame_seq={0: 70, 1: 71},
        )
        object0 = np.array([[True, False], [False, False]])
        controller0 = np.array([[False, True], [False, False]])
        object1 = np.array([[False, False], [True, False]])
        controller1 = np.array([[False, False], [False, True]])
        color = np.zeros((2, 2, 3), dtype=np.uint8)
        masks = {
            0: _FakePacket(7, 0, object0, controller0, color),
            1: _FakePacket(7, 1, object1, controller1, color),
        }

        raw_packet = runtime._build_raw_fused_packet(
            depth_group=depth_group,
            masks=masks,
            ray_cache={},
            rng=np.random.default_rng(0),
        )

        self.assertEqual(raw_packet.group_id, 7)
        self.assertEqual(len(client.inputs), 1)
        tracking_input = client.inputs[0]
        self.assertEqual(tracking_input.group_id, 7)
        self.assertEqual(tracking_input.frame_idx, 71)
        np.testing.assert_array_equal(tracking_input.object_mask_by_camera[0], object0)
        np.testing.assert_array_equal(tracking_input.controller_mask_by_camera[1], controller1)
        np.testing.assert_array_equal(tracking_input.mask_by_camera[0], object0 | controller0)
        self.assertEqual(runtime.demo31_lift_input_cache.snapshot()["published"], 1)
        self.assertEqual(runtime.demo31_surface_anchor_cache.snapshot()["published"], 1)
        self.assertIn(7, runtime.demo31_surface_anchor_cache.cached_group_ids())
        hook_profiles = [
            update["demo31_tracking_input"]
            for _group_id, update in runtime.profile_updates
            if "demo31_tracking_input" in update
        ]
        self.assertTrue(hook_profiles)
        self.assertEqual(hook_profiles[-1]["publish_hook"], "raw_fused_async")
        self.assertTrue(hook_profiles[-1]["published"])
        self.assertTrue(hook_profiles[-1]["surface_anchor_cache_published"])

    def test_demo32_explicit_preset_uses_demo32_default_output_root(self) -> None:
        args = self._parse(
            ["--dry-run", "--preset", demo31_runtime.PRESET_DEMO32_FFS_LITETRACKER, "--camera-ids", "0,1,2"],
        )
        self.assertEqual(str(args.output_root), "result/demo32_ffs_litetracker")

    def test_demo32_rejects_non_litetracker_or_non_ffs(self) -> None:
        realsense_args = self._parse(
            ["--dry-run", "--camera-ids", "0,1,2", "--depth-source", "realsense"],
            default_preset=demo31_runtime.PRESET_DEMO32_FFS_LITETRACKER,
        )
        with self.assertRaisesRegex(ValueError, "requires FFS depth"):
            demo31_runtime.validate_args(realsense_args, cuda_device_count_provider=lambda: 2)

        cotracker_args = self._parse(
            ["--dry-run", "--camera-ids", "0,1,2", "--cotracker-backend", "cotracker3_online"],
            default_preset=demo31_runtime.PRESET_DEMO32_FFS_LITETRACKER,
        )
        with self.assertRaisesRegex(ValueError, "requires --cotracker-backend litetracker"):
            demo31_runtime.validate_args(cotracker_args, cuda_device_count_provider=lambda: 2)

    def test_demo32_rejects_removed_query_init_choices(self) -> None:
        parser = demo31_runtime.build_arg_parser(default_preset=demo31_runtime.PRESET_DEMO32_FFS_LITETRACKER)
        with self.assertRaises(SystemExit):
            parser.parse_args(["--dry-run", "--trackable-mask-build-policy", "every-n"])
        with self.assertRaises(SystemExit):
            parser.parse_args(["--dry-run", "--trackable-query-init-strategy", "minimal-latency"])

    def test_demo32_shared_runtime_args_use_demo23_ffs_batch3(self) -> None:
        args = self._parse(
            ["--camera-ids", "0,1,2", "--mask-gpu", "0", "--cotracker-gpu", "1"],
            default_preset=demo31_runtime.PRESET_DEMO32_FFS_LITETRACKER,
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

        self.assertEqual(shared_args.preset, "demo2.3-dual4090-maxfps")
        self.assertEqual(shared_args.preset_canonical, "demo2.3-dual4090-maxfps")
        self.assertEqual(shared_args.demo_version_override, "demo3.2")
        self.assertEqual(shared_args.demo_display_name_override, "Demo 3.2")
        self.assertEqual(shared_args.depth_source, "ffs")
        self.assertEqual(shared_args.ffs_trt_batch_size, 3)
        self.assertIn("batch3", str(shared_args.ffs_trt_model_dir))
        self.assertEqual(shared_args.gpu_pipeline_mode, "dual-gpu-split")
        self.assertEqual(shared_args.ffs_schedule, "strict3-latest")
        self.assertEqual(shared_args.ffs_device, "cuda:0")
        self.assertEqual(shared_args.edgetam_device, "cuda:0")
        self.assertEqual(shared_args.sam31_device, "cuda:0")
        self.assertTrue(shared_args.demo32_ffs_edgetam_same_gpu)
        self.assertEqual(shared_args.demo32_gpu_placement, "ffs_edgetam_gpu0_litetracker_gpu1")
        self.assertTrue(shared_args.dual_gpu_processes)
        self.assertTrue(shared_args.enable_pcd_filter)
        self.assertEqual(shared_args.pcd_filter_mode, "async")
        self.assertEqual(shared_args.controller_render_voxel_m, 0.003)
        self.assertEqual(shared_args.controller_render_max_points, 10000)

    def test_demo32_main_dry_run_prints_ffs_litetracker_contract(self) -> None:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = demo31_runtime.main(
                ["--dry-run", "--camera-ids", "0,1,2", "--mask-gpu", "0", "--cotracker-gpu", "1"],
                cuda_device_count_provider=lambda: 2,
                default_preset=demo31_runtime.PRESET_DEMO32_FFS_LITETRACKER,
            )

        self.assertEqual(exit_code, 0)
        output = stdout.getvalue()
        self.assertIn("demo = demo3.2", output)
        self.assertIn("depth_source = ffs", output)
        self.assertIn("uses_ffs = true", output)
        self.assertIn("async_depth_pipeline = true", output)
        self.assertIn("shared_runtime_preset = demo2.3-dual4090-maxfps", output)
        self.assertIn("shared_runtime_gpu_pipeline_mode = dual-gpu-split", output)
        self.assertIn("shared_runtime_gpu_placement = ffs_edgetam_gpu0_litetracker_gpu1", output)
        self.assertIn("ffs_gpu_physical = 0", output)
        self.assertIn("edgetam_gpu_physical = 0", output)
        self.assertIn("sam31_gpu_physical = 0", output)
        self.assertIn("sam31_init_quick_fail_empty_masks = true", output)
        self.assertIn("sam31_init_min_mask_pixels = 1", output)
        self.assertIn("litetracker_gpu_physical = 1", output)
        self.assertIn("ffs_edgetam_same_gpu = true", output)
        self.assertIn("tracker_backend = litetracker", output)
        self.assertIn("tracker_env_name = demo_3_1_max", output)
        self.assertIn("trackable_mask_build_policy = init-only", output)
        self.assertIn("trackable_query_init_strategy = standard-filter-init", output)
        self.assertIn("trackable_mask_source = standard_filter_survivors", output)
        self.assertIn("tracker_query_source = union_trackable_mask", output)
        self.assertIn("controller_mask_erode_px = 0", output)
        self.assertIn("object_point_control = fixed-cap", output)
        self.assertIn("object_postprocess = enhanced-pt", output)
        self.assertIn("controller_postprocess = enhanced-pt", output)
        self.assertIn("render_object_filter = {'point_control': 'fixed-cap', 'postprocess': 'enhanced-pt'", output)
        self.assertIn("render_controller_filter = {'postprocess': 'enhanced-pt'", output)
        self.assertIn("'render_voxel_m': 0.003", output)
        self.assertIn("'render_max_points': 10000", output)
        self.assertIn("tracker_visualization_mode = all-tracks-3d-lift", output)
        self.assertIn("tracker_3d_marker_mode = all-tracks-3d-lift", output)
        self.assertIn("tracker_all_tracks_anchor_mode = true", output)
        self.assertIn("tracker_surface_gate_enabled = false", output)
        self.assertIn("overlay_bbox_filter_enabled = false", output)
        self.assertIn("tracking_overlay_lift_method = all_tracks_depth_lift", output)
        self.assertIn("tracking_control_point_count_requested = 0", output)
        self.assertIn("tracking_backend_execution_mode = batch-views", output)
        self.assertIn("tracking_backend_batch_size = 3", output)
        self.assertIn("tracker_batch_query_count_policy = min-common", output)
        self.assertIn("cotracker_update_mode = batch", output)
        self.assertIn("output_root = result/demo32_ffs_litetracker", output)
        self.assertIn("'trt_batch_size': 3", output)

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
        self.assertEqual(contract["controller_prompt"], "human hand")
        self.assertEqual(contract["tracking_controller_label"], "hand")
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
                "--overlay-debug-color-by-camera",
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
        self.assertEqual(shared_args.object_point_control, "fixed-cap")
        self.assertEqual(shared_args.object_postprocess, "enhanced-pt")
        self.assertEqual(shared_args.controller_postprocess, "enhanced-pt")
        self.assertEqual(shared_args.phystwin_radius_m, 0.01)
        self.assertEqual(shared_args.phystwin_nb_points, 12)
        self.assertEqual(shared_args.enhanced_component_voxel_size_m, 0.006)
        self.assertEqual(shared_args.enhanced_keep_near_main_gap_m, 0.035)
        self.assertEqual(shared_args.object_enhanced_keep_top_n_components, 1)
        self.assertEqual(shared_args.controller_enhanced_keep_top_n_components, 2)
        self.assertEqual(shared_args.enhanced_component_selection_policy, "largest-n-plus-gap")
        self.assertEqual(shared_args.enhanced_min_component_points, 32)
        self.assertEqual(shared_args.enhanced_min_component_ratio, 0.0)
        self.assertTrue(shared_args.apply_enhanced_component_filter_to_pcd)
        self.assertEqual(shared_args.object_volume_voxel_m, 0.005)
        self.assertEqual(shared_args.object_volume_points_per_voxel, 3)
        self.assertEqual(shared_args.depth_source, "realsense")
        self.assertTrue(shared_args.edgetam_batch_vision_encoder)
        self.assertTrue(shared_args.overlay_debug_color_by_camera)

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

    def test_summary_uses_real_cotracker_snapshot_metrics(self) -> None:
        args = self._parse(["--camera-ids", "0,1,2", "--cotracker-input-fps", "17"])
        runtime = demo31_runtime.Demo31Runtime(
            args,
            cuda_device_count_provider=lambda: 2,
            connected_serials_provider=lambda: ["s0", "s1", "s2"],
        )

        summary = runtime._build_summary(
            runtime=SimpleNamespace(_summary={"final": {}}),
            exit_code=0,
            snapshot={
                "cotracker_input_fps": 8.5,
                "cotracker_publish_fps": 2.25,
                "cotracker_input_count": 9,
                "cotracker_result_count": 3,
                "cotracker_model_ms_median": 123.0,
                "cotracker_model_ms_p95": 150.0,
                "cotracker_e2e_ms_median": 180.0,
                "cotracker_e2e_ms_p95": 210.0,
                "tracker_group_wall_ms_p50": 140.0,
                "tracker_group_wall_ms_p95": 190.0,
                "tracker_model_ms_sum_per_group_p50": 136.0,
                "tracker_model_ms_sum_per_group_p95": 180.0,
                "tracker_model_ms_max_per_group_p50": 48.0,
                "tracker_model_ms_max_per_group_p95": 60.0,
                "per_camera_model_ms_p50_by_camera": {0: 45.0, 1: 46.0, 2: 48.0},
                "per_camera_model_ms_p95_by_camera": {0: 58.0, 1: 59.0, 2: 60.0},
                "model_calls_per_group": 3,
                "model_instances_expected": 3,
                "model_instances_actual": 3,
                "query_count_per_camera": 1365,
                "total_query_count_across_views": 4095,
                "process": {"pid": 42, "ready": {"total_init_ms": 1.0}},
            },
            shared_payload={"tracking_update_hz": 99.0},
        )

        self.assertEqual(summary["cotracker_input_fps"], 8.5)
        self.assertEqual(summary["cotracker_publish_fps"], 2.25)
        self.assertEqual(summary["cotracker_input_count"], 9)
        self.assertEqual(summary["cotracker_result_count"], 3)
        self.assertEqual(summary["cotracker_model_ms_median"], 123.0)
        self.assertEqual(summary["cotracker_e2e_ms_p95"], 210.0)
        self.assertEqual(summary["tracker_group_wall_ms_p50"], 140.0)
        self.assertEqual(summary["tracker_model_ms_sum_per_group_p50"], 136.0)
        self.assertEqual(summary["tracker_model_ms_max_per_group_p50"], 48.0)
        self.assertEqual(summary["per_camera_model_ms_p50_by_camera"], {0: 45.0, 1: 46.0, 2: 48.0})
        self.assertEqual(summary["model_calls_per_group"], 3)
        self.assertEqual(summary["model_instances_expected"], 3)
        self.assertEqual(summary["query_count_per_camera"], 1365)
        self.assertEqual(summary["total_query_count_across_views"], 4095)
        self.assertNotEqual(summary["cotracker_publish_fps"], 99.0)

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

    def test_surface_snap_accepts_existing_anchor_and_rejects_outside_radius(self) -> None:
        layer = demo31_runtime.SurfaceAnchorLayer(
            camera_idx=0,
            label="controller",
            yx=np.array([[0.0, 0.0], [0.0, 10.0]], dtype=np.float32),
            points_world=np.array([[1.0, 2.0, 3.0], [9.0, 9.0, 9.0]], dtype=np.float32),
        )

        points, stats = demo31_runtime.snap_tracks_to_surface(
            tracks_yx=np.array([[0.5, 0.5], [0.0, 7.0]], dtype=np.float32),
            visibility=np.array([1.0, 1.0], dtype=np.float32),
            surface_layer=layer,
            radius_px=1.0,
            max_points=2,
            selection="top-visible",
        )

        np.testing.assert_allclose(points, np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
        self.assertEqual(stats["accepted"], 1)
        self.assertEqual(stats["rejected"], 1)

    def test_surface_marker_mode_snaps_to_surface_without_legacy_lift(self) -> None:
        now_s = demo31_runtime.time.perf_counter()
        result = TrackingResultLitePacket(
            group_id=1,
            frame_idx=1,
            source_timestamp_s=now_s,
            publish_timestamp_s=now_s,
            camera_tracks_yx={0: np.array([[0.25, 0.25], [0.0, 8.0]], dtype=np.float32)},
            camera_visibility={0: np.array([1.0, 1.0], dtype=np.float32)},
            query_points_yx={0: np.array([[0.25, 0.25], [0.0, 8.0]], dtype=np.float32)},
            publish_range=(1, 1),
        )
        runtime_cls = demo31_runtime.make_demo31_live_runtime_class(
            _FakeSharedRuntimeModule,
            process_client_factory=lambda _config: _FakeProcessClient(result),
        )
        runtime = runtime_cls(
            SimpleNamespace(
                camera_ids=(0,),
                tracker_visualization_mode="3d-surface-markers",
                tracker_3d_snap_radius_px=1.0,
                tracker_3d_marker_radius_m=0.006,
                tracker_control_points_per_camera=2,
                tracker_control_point_selection="top-visible",
                overlay_display_scope="controller",
                overlay_debug_color_by_camera=False,
                overlay_reject_outside_semantic_bbox=False,
            ),
            demo31_contract={
                "fusion_mask_policy": "latest-reuse",
                "mask_stale_timeout_ms": 250.0,
                "cotracker_result_stale_timeout_ms": 1500.0,
                "tracker_visualization_mode": "3d-surface-markers",
                "tracking_control_point_markers": True,
                "tracking_control_points_per_camera": 2,
                "tracking_control_point_count_requested": 2,
            },
            cotracker_process_config=SimpleNamespace(),
        )
        runtime.demo31_surface_anchor_cache.publish(
            demo31_runtime.SurfaceAnchorIndexSnapshot(
                group_id=1,
                timestamp_s=now_s,
                layers={
                    (0, "controller"): demo31_runtime.SurfaceAnchorLayer(
                        camera_idx=0,
                        label="controller",
                        yx=np.array([[0.0, 0.0], [0.0, 10.0]], dtype=np.float32),
                        points_world=np.array([[1.0, 2.0, 3.0], [9.0, 9.0, 9.0]], dtype=np.float32),
                    )
                },
            )
        )
        packet = _FakeRenderPacket(
            group_id=1,
            controller_points_m=np.empty((0, 3), dtype=np.float32),
            controller_colors_rgb=np.empty((0, 3), dtype=np.uint8),
        )
        original_lift = demo31_runtime.lift_tracks_yx_to_world

        def _fail_lift(**_kwargs):
            raise AssertionError("legacy lift must not be called in surface marker mode")

        try:
            demo31_runtime.lift_tracks_yx_to_world = _fail_lift
            runtime._publish_render_packet(packet)
            runtime._publish_next_tracker_driven_render_once(now_s=now_s)
        finally:
            demo31_runtime.lift_tracks_yx_to_world = original_lift

        published = runtime.published_packet
        self.assertIsNotNone(published)
        marker_vertices = len(demo31_runtime._SPHERE_MARKER_OFFSETS)
        self.assertEqual(len(published.controller_points_m), marker_vertices)  # type: ignore[arg-type]
        np.testing.assert_allclose(published.controller_points_m[0], np.array([1.0, 2.0, 3.0], dtype=np.float32))  # type: ignore[index]
        np.testing.assert_array_equal(  # type: ignore[union-attr]
            np.unique(published.controller_colors_rgb, axis=0),
            np.array([[255, 0, 0]], dtype=np.uint8),
        )
        overlay_profile = runtime.profile_updates[-1][1]["demo31_tracking_overlay"]
        self.assertEqual(overlay_profile["tracker_visualization_mode"], "3d-surface-markers")
        self.assertEqual(overlay_profile["tracker_3d_marker_mode"], "surface_snap")
        self.assertEqual(overlay_profile["tracker_3d_marker_shape"], "sphere")
        self.assertFalse(overlay_profile["tracker_legacy_lift_used"])
        self.assertTrue(overlay_profile["tracker_surface_anchor_cache_hit"])
        self.assertEqual(overlay_profile["tracker_surface_anchor_group_id"], 1)
        self.assertEqual(overlay_profile["tracker_marker_accepted_by_camera"], {0: 1})
        self.assertEqual(overlay_profile["tracker_marker_rejected_by_camera"], {0: 1})
        self.assertEqual(overlay_profile["tracker_marker_layer_by_camera"], {0: "controller"})
        self.assertEqual(overlay_profile["tracking_control_point_count"], 1)
        self.assertEqual(overlay_profile["tracking_control_marker_points"], marker_vertices)
        self.assertEqual(overlay_profile["tracker_marker_points_rendered"], marker_vertices)

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
            group_id=1,
            controller_points_m=np.empty((0, 3), dtype=np.float32),
            controller_colors_rgb=np.empty((0, 3), dtype=np.uint8),
        )

        runtime._publish_render_packet(packet)
        runtime._publish_next_tracker_driven_render_once(now_s=now_s)

        published = runtime.published_packet
        self.assertIsNotNone(published)
        np.testing.assert_allclose(published.controller_points_m[-1], np.array([0.0, 0.0, 1.0], dtype=np.float32))  # type: ignore[union-attr]
        self.assertEqual(published.tracker_update_mode, "batch")  # type: ignore[union-attr]
        self.assertEqual(published.tracker_batch_size, 3)  # type: ignore[union-attr]
        self.assertEqual(published.tracker_model_ms, 0.0)  # type: ignore[union-attr]
        overlay_profile = runtime.profile_updates[-1][1]["demo31_tracking_overlay"]
        self.assertEqual(overlay_profile["overlay_group_id"], 1)
        self.assertEqual(overlay_profile["incoming_render_group_id"], 1)
        self.assertEqual(overlay_profile["render_group_id"], 1)
        self.assertTrue(overlay_profile["overlay_lift_cache_hit"])
        self.assertTrue(overlay_profile["tracking_result_has_matching_render_packet"])
        self.assertTrue(overlay_profile["tracking_result_used_render_packet"])
        self.assertFalse(overlay_profile["tracking_result_used_nearest_render_packet"])
        self.assertEqual(overlay_profile["tracking_render_packet_match_mode"], "exact")
        self.assertEqual(overlay_profile["tracking_render_packet_group_id"], 1)
        self.assertEqual(overlay_profile["tracking_render_packet_group_delta"], 0)
        self.assertEqual(overlay_profile["overlay_lift_method"], "semantic_projection_grid")
        self.assertEqual(overlay_profile["overlay_points_by_camera"], {0: 1})
        self.assertEqual(overlay_profile["overlay_color_mode"], "solid")
        self.assertTrue(overlay_profile["render_requires_new_cotracker_result"])
        self.assertFalse(overlay_profile["render_reuses_cached_cotracker_result"])

    def test_renderer_lifts_overlay_with_semantic_projection_grid_and_camera_debug_color(self) -> None:
        now_s = demo31_runtime.time.perf_counter()
        result = TrackingResultLitePacket(
            group_id=1,
            frame_idx=1,
            source_timestamp_s=now_s,
            publish_timestamp_s=now_s,
            camera_tracks_yx={1: np.array([[0.49, 1.49]], dtype=np.float32)},
            camera_visibility={1: np.array([1.0], dtype=np.float32)},
            query_points_yx={1: np.array([[0.0, 1.0]], dtype=np.float32)},
            publish_range=(1, 1),
        )
        runtime_cls = demo31_runtime.make_demo31_live_runtime_class(
            _FakeSharedRuntimeModule,
            process_client_factory=lambda _config: _FakeProcessClient(result),
        )
        runtime = runtime_cls(
            SimpleNamespace(camera_ids=(1,), overlay_debug_color_by_camera=True),
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
            depth_by_camera={1: np.full((2, 2), 2.0, dtype=np.float32)},
            intrinsics_by_camera={1: np.eye(3, dtype=np.float32)},
            c2w_by_camera={1: np.eye(4, dtype=np.float32)},
            mask_by_camera={1: np.ones((2, 2), dtype=bool)},
        )
        packet = _FakeRenderPacket(
            group_id=1,
            controller_points_m=np.empty((0, 3), dtype=np.float32),
            controller_colors_rgb=np.empty((0, 3), dtype=np.uint8),
        )

        runtime._publish_render_packet(packet)
        runtime._publish_next_tracker_driven_render_once(now_s=now_s)

        published = runtime.published_packet
        self.assertIsNotNone(published)
        np.testing.assert_allclose(  # type: ignore[union-attr]
            published.controller_points_m[-1],
            np.array([2.0, 0.0, 2.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(  # type: ignore[union-attr]
            published.controller_colors_rgb[-1],
            np.array([0, 255, 0], dtype=np.uint8),
        )
        overlay_profile = runtime.profile_updates[-1][1]["demo31_tracking_overlay"]
        self.assertEqual(overlay_profile["overlay_color_mode"], "by_camera")
        self.assertEqual(overlay_profile["overlay_points_by_camera"], {1: 1})
        self.assertEqual(overlay_profile["overlay_world_centroid_by_camera"], {1: [2.0, 0.0, 2.0]})

    def test_controller_overlay_lift_uses_current_controller_mask_not_union_mask(self) -> None:
        now_s = demo31_runtime.time.perf_counter()
        result = TrackingResultLitePacket(
            group_id=1,
            frame_idx=1,
            source_timestamp_s=now_s,
            publish_timestamp_s=now_s,
            camera_tracks_yx={0: np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.float32)},
            camera_visibility={0: np.array([1.0, 1.0], dtype=np.float32)},
            query_points_yx={0: np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.float32)},
            publish_range=(1, 1),
        )
        runtime_cls = demo31_runtime.make_demo31_live_runtime_class(
            _FakeSharedRuntimeModule,
            process_client_factory=lambda _config: _FakeProcessClient(result),
        )
        runtime = runtime_cls(
            SimpleNamespace(camera_ids=(0,), overlay_display_scope="controller"),
            demo31_contract={
                "fusion_mask_policy": "latest-reuse",
                "mask_stale_timeout_ms": 250.0,
                "cotracker_result_stale_timeout_ms": 1500.0,
            },
            cotracker_process_config=SimpleNamespace(),
        )
        controller_mask = np.array([[True, False]], dtype=bool)
        runtime.demo31_lift_input_cache.publish(
            group_id=1,
            timestamp_s=now_s,
            depth_by_camera={0: np.full((1, 2), 1.0, dtype=np.float32)},
            intrinsics_by_camera={0: np.eye(3, dtype=np.float32)},
            c2w_by_camera={0: np.eye(4, dtype=np.float32)},
            mask_by_camera={0: np.ones((1, 2), dtype=bool)},
            controller_mask_by_camera={0: controller_mask},
        )
        packet = _FakeRenderPacket(
            group_id=1,
            controller_points_m=np.empty((0, 3), dtype=np.float32),
            controller_colors_rgb=np.empty((0, 3), dtype=np.uint8),
        )

        runtime._publish_render_packet(packet)
        runtime._publish_next_tracker_driven_render_once(now_s=now_s)

        published = runtime.published_packet
        self.assertIsNotNone(published)
        self.assertEqual(len(published.controller_points_m), 1)  # type: ignore[arg-type]
        np.testing.assert_allclose(  # type: ignore[union-attr]
            published.controller_points_m[-1],
            np.array([0.0, 0.0, 1.0], dtype=np.float32),
        )
        overlay_profile = runtime.profile_updates[-1][1]["demo31_tracking_overlay"]
        self.assertEqual(overlay_profile["overlay_lift_mask_scope"], "controller")
        self.assertEqual(overlay_profile["overlay_input_points_by_camera"], {0: 2})
        self.assertEqual(overlay_profile["overlay_points_by_camera"], {0: 1})
        self.assertEqual(overlay_profile["overlay_rejected_by_scope_mask_by_camera"], {0: 1})

    def test_controller_overlay_rejects_lifted_outliers_outside_semantic_bbox(self) -> None:
        now_s = demo31_runtime.time.perf_counter()
        result = TrackingResultLitePacket(
            group_id=1,
            frame_idx=1,
            source_timestamp_s=now_s,
            publish_timestamp_s=now_s,
            camera_tracks_yx={0: np.array([[0.0, 0.0], [0.0, 9.0]], dtype=np.float32)},
            camera_visibility={0: np.array([1.0, 1.0], dtype=np.float32)},
            query_points_yx={0: np.array([[0.0, 0.0], [0.0, 9.0]], dtype=np.float32)},
            publish_range=(1, 1),
        )
        runtime_cls = demo31_runtime.make_demo31_live_runtime_class(
            _FakeSharedRuntimeModule,
            process_client_factory=lambda _config: _FakeProcessClient(result),
        )
        runtime = runtime_cls(
            SimpleNamespace(
                camera_ids=(0,),
                overlay_display_scope="controller",
                overlay_reject_outside_semantic_bbox=True,
                overlay_max_distance_from_controller_m=0.05,
            ),
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
            depth_by_camera={0: np.full((1, 10), 1.0, dtype=np.float32)},
            intrinsics_by_camera={0: np.eye(3, dtype=np.float32)},
            c2w_by_camera={0: np.eye(4, dtype=np.float32)},
            mask_by_camera={0: np.ones((1, 10), dtype=bool)},
            controller_mask_by_camera={0: np.ones((1, 10), dtype=bool)},
        )
        packet = _FakeRenderPacket(
            group_id=1,
            controller_points_m=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
            controller_colors_rgb=np.array([[200, 200, 200]], dtype=np.uint8),
        )

        runtime._publish_render_packet(packet)
        runtime._publish_next_tracker_driven_render_once(now_s=now_s)

        published = runtime.published_packet
        self.assertIsNotNone(published)
        self.assertEqual(len(published.controller_points_m), 2)  # type: ignore[arg-type]
        np.testing.assert_allclose(  # type: ignore[union-attr]
            published.controller_points_m[-1],
            np.array([0.0, 0.0, 1.0], dtype=np.float32),
        )
        overlay_profile = runtime.profile_updates[-1][1]["demo31_tracking_overlay"]
        self.assertEqual(overlay_profile["overlay_rejected_by_scope_mask_by_camera"], {0: 0})
        self.assertEqual(overlay_profile["overlay_bbox_rejected_by_camera"], {0: 1})
        self.assertEqual(overlay_profile["overlay_bbox_kept_points_by_camera"], {0: 1})
        self.assertEqual(overlay_profile["overlay_points_by_camera"], {0: 1})
        self.assertEqual(overlay_profile["overlay_world_centroid_by_camera"], {0: [0.0, 0.0, 1.0]})

    def test_all_tracks_anchor_mode_disables_surface_mask_and_bbox_gates(self) -> None:
        now_s = demo31_runtime.time.perf_counter()
        result = TrackingResultLitePacket(
            group_id=1,
            frame_idx=1,
            source_timestamp_s=now_s,
            publish_timestamp_s=now_s,
            camera_tracks_yx={0: np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]], dtype=np.float32)},
            camera_visibility={0: np.array([1.0, 1.0, 1.0], dtype=np.float32)},
            query_points_yx={0: np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]], dtype=np.float32)},
            publish_range=(1, 1),
        )
        runtime_cls = demo31_runtime.make_demo31_live_runtime_class(
            _FakeSharedRuntimeModule,
            process_client_factory=lambda _config: _FakeProcessClient(result),
        )
        runtime = runtime_cls(
            SimpleNamespace(
                camera_ids=(0,),
                tracker_visualization_mode="all-tracks-3d-lift",
                tracker_3d_marker_radius_m=0.006,
                overlay_display_scope="controller",
                overlay_reject_outside_semantic_bbox=True,
                overlay_max_distance_from_controller_m=0.0,
            ),
            demo31_contract={
                "fusion_mask_policy": "latest-reuse",
                "mask_stale_timeout_ms": 250.0,
                "cotracker_result_stale_timeout_ms": 1500.0,
            },
            cotracker_process_config=SimpleNamespace(),
        )
        controller_mask = np.array([[True, False, False]], dtype=bool)
        runtime.demo31_lift_input_cache.publish(
            group_id=1,
            timestamp_s=now_s,
            depth_by_camera={0: np.full((1, 3), 1.0, dtype=np.float32)},
            intrinsics_by_camera={0: np.eye(3, dtype=np.float32)},
            c2w_by_camera={0: np.eye(4, dtype=np.float32)},
            mask_by_camera={0: controller_mask},
            controller_mask_by_camera={0: controller_mask},
        )
        packet = _FakeRenderPacket(
            group_id=1,
            controller_points_m=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
            controller_colors_rgb=np.array([[200, 200, 200]], dtype=np.uint8),
        )

        runtime._publish_render_packet(packet)
        runtime._publish_next_tracker_driven_render_once(now_s=now_s)

        published = runtime.published_packet
        self.assertIsNotNone(published)
        marker_vertices = len(demo31_runtime._SPHERE_MARKER_OFFSETS)
        self.assertEqual(len(published.controller_points_m), 1 + 3 * marker_vertices)  # type: ignore[arg-type]
        overlay_profile = runtime.profile_updates[-1][1]["demo31_tracking_overlay"]
        self.assertEqual(overlay_profile["tracker_visualization_mode"], "all-tracks-3d-lift")
        self.assertEqual(overlay_profile["tracker_3d_marker_mode"], "all-tracks-3d-lift")
        self.assertTrue(overlay_profile["tracker_direct_depth_lift_used"])
        self.assertTrue(overlay_profile["tracker_all_tracks_anchor_mode"])
        self.assertFalse(overlay_profile["tracker_surface_gate_enabled"])
        self.assertEqual(overlay_profile["overlay_lift_method"], "all_tracks_depth_lift")
        self.assertEqual(overlay_profile["overlay_lift_mask_scope"], "none")
        self.assertFalse(overlay_profile["overlay_bbox_filter_enabled"])
        self.assertEqual(overlay_profile["overlay_input_points_by_camera"], {0: 3})
        self.assertEqual(overlay_profile["overlay_points_by_camera"], {0: 3})
        self.assertEqual(overlay_profile["overlay_rejected_by_scope_mask_by_camera"], {0: 0})
        self.assertEqual(overlay_profile["overlay_bbox_rejected_by_camera"], {0: 0})
        self.assertEqual(overlay_profile["tracker_marker_accepted_by_camera"], {0: 3})
        self.assertEqual(overlay_profile["tracker_marker_rejected_by_camera"], {0: 0})
        self.assertEqual(overlay_profile["tracker_marker_layer_by_camera"], {0: "all-tracks"})
        self.assertEqual(overlay_profile["tracking_control_point_count"], 3)
        self.assertEqual(overlay_profile["tracking_control_marker_points"], 3 * marker_vertices)
        self.assertEqual(
            overlay_profile["tracking_control_point_sampling"],
            "all_visible_depth_valid_tracks_no_surface_or_bbox_gate",
        )

    def test_renderer_marks_sampled_tracking_control_points_as_3d_marker_cloud(self) -> None:
        now_s = demo31_runtime.time.perf_counter()
        result = TrackingResultLitePacket(
            group_id=1,
            frame_idx=1,
            source_timestamp_s=now_s,
            publish_timestamp_s=now_s,
            camera_tracks_yx={0: np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]], dtype=np.float32)},
            camera_visibility={0: np.array([1.0, 1.0, 1.0], dtype=np.float32)},
            query_points_yx={0: np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]], dtype=np.float32)},
            publish_range=(1, 1),
        )
        runtime_cls = demo31_runtime.make_demo31_live_runtime_class(
            _FakeSharedRuntimeModule,
            process_client_factory=lambda _config: _FakeProcessClient(result),
        )
        runtime = runtime_cls(
            SimpleNamespace(
                camera_ids=(0,),
                overlay_control_point_markers=True,
                overlay_control_point_count=2,
                overlay_control_point_radius_m=0.01,
                overlay_render_raw_track_points=False,
            ),
            demo31_contract={
                "fusion_mask_policy": "latest-reuse",
                "mask_stale_timeout_ms": 250.0,
                "cotracker_result_stale_timeout_ms": 1500.0,
                "tracking_control_point_markers": True,
                "tracking_control_point_count_requested": 2,
                "tracking_control_point_radius_m": 0.01,
                "overlay_render_raw_track_points": False,
            },
            cotracker_process_config=SimpleNamespace(),
        )
        runtime.demo31_lift_input_cache.publish(
            group_id=1,
            timestamp_s=now_s,
            depth_by_camera={0: np.full((1, 3), 1.0, dtype=np.float32)},
            intrinsics_by_camera={0: np.eye(3, dtype=np.float32)},
            c2w_by_camera={0: np.eye(4, dtype=np.float32)},
            mask_by_camera={0: np.ones((1, 3), dtype=bool)},
            controller_mask_by_camera={0: np.ones((1, 3), dtype=bool)},
        )
        packet = _FakeRenderPacket(
            group_id=1,
            controller_points_m=np.empty((0, 3), dtype=np.float32),
            controller_colors_rgb=np.empty((0, 3), dtype=np.uint8),
        )

        runtime._publish_render_packet(packet)
        runtime._publish_next_tracker_driven_render_once(now_s=now_s)

        published = runtime.published_packet
        self.assertIsNotNone(published)
        marker_vertices = len(demo31_runtime._SPHERE_MARKER_OFFSETS)
        self.assertEqual(len(published.controller_points_m), 2 * marker_vertices)  # type: ignore[arg-type]
        np.testing.assert_array_equal(  # type: ignore[union-attr]
            np.unique(published.controller_colors_rgb, axis=0),
            np.array([[255, 0, 0]], dtype=np.uint8),
        )
        overlay_profile = runtime.profile_updates[-1][1]["demo31_tracking_overlay"]
        self.assertEqual(overlay_profile["overlay_track_points"], 3)
        self.assertEqual(overlay_profile["overlay_points"], 2 * marker_vertices)
        self.assertEqual(overlay_profile["tracking_control_point_count"], 2)
        self.assertEqual(overlay_profile["tracking_control_points_by_camera"], {0: 2})
        self.assertEqual(overlay_profile["tracking_control_marker_points"], 2 * marker_vertices)
        self.assertFalse(overlay_profile["overlay_render_raw_track_points"])

    def test_renderer_does_not_reuse_old_tracking_result_as_new_render_result(self) -> None:
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

        class _OneShotProcessClient(_FakeProcessClient):
            def get_result(self) -> TrackingResultLitePacket | None:
                current = self.result
                self.result = None
                return current

        runtime_cls = demo31_runtime.make_demo31_live_runtime_class(
            _FakeSharedRuntimeModule,
            process_client_factory=lambda _config: _OneShotProcessClient(result),
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

        runtime._publish_render_packet(
            _FakeRenderPacket(
                group_id=1,
                controller_points_m=np.empty((0, 3), dtype=np.float32),
                controller_colors_rgb=np.empty((0, 3), dtype=np.uint8),
            )
        )
        runtime._publish_next_tracker_driven_render_once(now_s=now_s)
        first = runtime.published_packet
        self.assertIsNotNone(first)
        np.testing.assert_allclose(  # type: ignore[union-attr]
            first.controller_points_m[-1],
            np.array([0.0, 0.0, 1.0], dtype=np.float32),
        )
        first_profile = runtime.profile_updates[-1][1]["demo31_tracking_overlay"]
        self.assertTrue(first_profile["overlay_available"])
        self.assertFalse(first_profile["tracking_overlay_render_blocked"])
        self.assertEqual(first_profile["render_group_id"], 1)

        runtime.published_packet = None
        runtime._publish_render_packet(
            _FakeRenderPacket(
                group_id=2,
                controller_points_m=np.empty((0, 3), dtype=np.float32),
                controller_colors_rgb=np.empty((0, 3), dtype=np.uint8),
            )
        )
        runtime._publish_next_tracker_driven_render_once(now_s=now_s)
        self.assertIsNone(runtime.published_packet)
        second_profile = runtime.profile_updates[-1][1]["demo31_tracking_overlay"]
        self.assertFalse(second_profile["overlay_available"])
        self.assertTrue(second_profile["tracking_overlay_render_blocked"])
        self.assertEqual(runtime.demo31_tracking_overlay_render_blocked_count, 0)

    def test_renderer_blocks_nearest_pending_pcd_by_default(self) -> None:
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

        runtime._publish_render_packet(
            _FakeRenderPacket(
                group_id=2,
                controller_points_m=np.empty((0, 3), dtype=np.float32),
                controller_colors_rgb=np.empty((0, 3), dtype=np.uint8),
            )
        )
        runtime._publish_next_tracker_driven_render_once(now_s=now_s)

        self.assertIsNone(runtime.published_packet)
        self.assertEqual(runtime.demo31_tracking_result_without_render_packet_count, 1)
        self.assertEqual(runtime.demo31_tracking_result_nearest_render_packet_count, 0)
        self.assertEqual(runtime.demo31_tracking_result_without_lift_input_count, 0)
        self.assertEqual(runtime.demo31_frame_bundle_missing_exact_count, 1)
        overlay_profile = runtime.profile_updates[-1][1]["demo31_tracking_overlay"]
        self.assertTrue(overlay_profile["overlay_available"])
        self.assertFalse(overlay_profile["tracking_result_has_matching_render_packet"])
        self.assertFalse(overlay_profile["tracking_result_used_render_packet"])
        self.assertFalse(overlay_profile["tracking_result_used_nearest_render_packet"])
        self.assertEqual(overlay_profile["tracking_render_packet_match_mode"], "missing-exact")
        self.assertIsNone(overlay_profile["tracking_render_packet_group_id"])
        self.assertIsNone(overlay_profile["tracking_render_packet_group_delta"])
        self.assertIsNone(overlay_profile["tracking_nearest_render_packet_abs_delta"])
        self.assertEqual(overlay_profile["tracking_render_packet_match_policy"], "exact-target-bundle")
        self.assertEqual(overlay_profile["frame_bundle_policy"], "exact-target")
        self.assertFalse(overlay_profile["same_target_group"])
        self.assertTrue(overlay_profile["tracking_overlay_render_blocked"])

    def test_renderer_blocks_when_exact_group_was_evicted_by_default(self) -> None:
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
            group_id=2,
            timestamp_s=now_s,
            depth_by_camera={0: np.full((1, 1), 2.0, dtype=np.float32)},
            intrinsics_by_camera={0: np.eye(3, dtype=np.float32)},
            c2w_by_camera={0: np.eye(4, dtype=np.float32)},
            mask_by_camera={0: np.ones((1, 1), dtype=bool)},
        )

        runtime._publish_render_packet(
            _FakeRenderPacket(
                group_id=2,
                controller_points_m=np.empty((0, 3), dtype=np.float32),
                controller_colors_rgb=np.empty((0, 3), dtype=np.uint8),
            )
        )
        runtime._publish_next_tracker_driven_render_once(now_s=now_s)

        self.assertIsNone(runtime.published_packet)
        self.assertEqual(runtime.demo31_tracking_result_exact_render_packet_count, 0)
        self.assertEqual(runtime.demo31_tracking_result_nearest_render_packet_count, 0)
        self.assertEqual(runtime.demo31_frame_bundle_missing_exact_count, 1)
        overlay_profile = runtime.profile_updates[-1][1]["demo31_tracking_overlay"]
        self.assertFalse(overlay_profile["tracking_result_has_matching_render_packet"])
        self.assertFalse(overlay_profile["tracking_result_used_render_packet"])
        self.assertFalse(overlay_profile["tracking_result_used_nearest_render_packet"])
        self.assertEqual(overlay_profile["tracking_render_packet_match_mode"], "missing-exact")
        self.assertIsNone(overlay_profile["tracking_render_packet_group_id"])
        self.assertIsNone(overlay_profile["tracking_render_packet_group_delta"])
        self.assertTrue(overlay_profile["tracking_overlay_render_blocked"])

    def test_renderer_allows_nearest_pending_pcd_only_under_debug_policy(self) -> None:
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
                "frame_bundle_policy": "latest-reuse-debug",
                "tracking_render_packet_match_policy": "exact-then-nearest-debug",
            },
            cotracker_process_config=SimpleNamespace(),
        )
        runtime.demo31_lift_input_cache.publish(
            group_id=2,
            timestamp_s=now_s,
            depth_by_camera={0: np.full((1, 1), 2.0, dtype=np.float32)},
            intrinsics_by_camera={0: np.eye(3, dtype=np.float32)},
            c2w_by_camera={0: np.eye(4, dtype=np.float32)},
            mask_by_camera={0: np.ones((1, 1), dtype=bool)},
        )

        runtime._publish_render_packet(
            _FakeRenderPacket(
                group_id=2,
                controller_points_m=np.empty((0, 3), dtype=np.float32),
                controller_colors_rgb=np.empty((0, 3), dtype=np.uint8),
            )
        )
        runtime._publish_next_tracker_driven_render_once(now_s=now_s)

        published = runtime.published_packet
        self.assertIsNotNone(published)
        np.testing.assert_allclose(  # type: ignore[union-attr]
            published.controller_points_m[-1],
            np.array([0.0, 0.0, 2.0], dtype=np.float32),
        )
        self.assertEqual(runtime.demo31_tracking_result_exact_render_packet_count, 0)
        self.assertEqual(runtime.demo31_tracking_result_nearest_render_packet_count, 1)
        self.assertEqual(runtime.demo31_frame_bundle_nearest_fallback_debug_count, 1)
        overlay_profile = runtime.profile_updates[-1][1]["demo31_tracking_overlay"]
        self.assertFalse(overlay_profile["tracking_result_has_matching_render_packet"])
        self.assertTrue(overlay_profile["tracking_result_used_render_packet"])
        self.assertTrue(overlay_profile["tracking_result_used_nearest_render_packet"])
        self.assertEqual(overlay_profile["tracking_render_packet_match_mode"], "nearest")
        self.assertEqual(overlay_profile["tracking_render_packet_match_policy"], "exact-then-nearest-debug")
        self.assertEqual(overlay_profile["frame_bundle_policy"], "latest-reuse-debug")
        self.assertEqual(overlay_profile["tracking_render_packet_group_id"], 2)
        self.assertEqual(overlay_profile["tracking_render_packet_group_delta"], 1)
        self.assertEqual(overlay_profile["render_group_id"], 2)
        self.assertFalse(overlay_profile["same_target_group"])
        self.assertFalse(overlay_profile["tracking_overlay_render_blocked"])

    def test_strict_source_policy_rejects_reused_mask_tracking_result(self) -> None:
        now_s = demo31_runtime.time.perf_counter()
        result = TrackingResultLitePacket(
            group_id=5,
            frame_idx=5,
            source_timestamp_s=now_s,
            publish_timestamp_s=now_s,
            camera_tracks_yx={0: np.array([[0.0, 0.0]], dtype=np.float32)},
            camera_visibility={0: np.array([1.0], dtype=np.float32)},
            query_points_yx={0: np.array([[0.0, 0.0]], dtype=np.float32)},
            publish_range=(5, 5),
            mask_source_group_id=4,
            mask_age_ms=33.0,
            mask_reused=True,
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
                "frame_bundle_policy": "strict-source",
            },
            cotracker_process_config=SimpleNamespace(),
        )
        runtime._publish_render_packet(
            _FakeRenderPacket(
                group_id=5,
                controller_points_m=np.empty((0, 3), dtype=np.float32),
                controller_colors_rgb=np.empty((0, 3), dtype=np.uint8),
            )
        )

        runtime._publish_next_tracker_driven_render_once(now_s=now_s)

        self.assertIsNone(runtime.published_packet)
        self.assertEqual(runtime.demo31_frame_bundle_strict_source_reject_count, 1)
        overlay_profile = runtime.profile_updates[-1][1]["demo31_tracking_overlay"]
        self.assertEqual(overlay_profile["frame_bundle_policy"], "strict-source")
        self.assertTrue(overlay_profile["frame_bundle_strict_source_rejected"])
        self.assertEqual(overlay_profile["tracking_mask_source_group_id"], 4)
        self.assertTrue(overlay_profile["tracking_mask_reused"])
        self.assertFalse(overlay_profile["same_target_group"])
        self.assertFalse(overlay_profile["strict_same_source_frame"])
        self.assertTrue(overlay_profile["tracking_overlay_render_blocked"])

    def test_pending_render_packet_cache_is_bounded(self) -> None:
        runtime_cls = demo31_runtime.make_demo31_live_runtime_class(
            _FakeSharedRuntimeModule,
            process_client_factory=lambda _config: _FakeProcessClient(None),
        )
        runtime = runtime_cls(
            SimpleNamespace(camera_ids=(0,)),
            demo31_contract={
                "fusion_mask_policy": "latest-reuse",
                "mask_stale_timeout_ms": 250.0,
                "cotracker_result_stale_timeout_ms": 1500.0,
                "tracking_pending_render_packet_max_groups": 2,
            },
            cotracker_process_config=SimpleNamespace(),
        )

        for group_id in (1, 2, 3):
            runtime._publish_render_packet(
                _FakeRenderPacket(
                    group_id=group_id,
                    controller_points_m=np.empty((0, 3), dtype=np.float32),
                    controller_colors_rgb=np.empty((0, 3), dtype=np.uint8),
                )
            )

        with runtime.demo31_pending_render_lock:
            pending_ids = sorted(runtime.demo31_pending_render_packets)
        self.assertEqual(pending_ids, [2, 3])
        self.assertEqual(runtime.demo31_pending_render_packet_drop_count, 1)
        self.assertEqual(runtime.demo31_snapshot()["tracking_pending_render_packet_max_groups"], 2)

    def test_protected_pending_render_packet_survives_cache_pruning(self) -> None:
        runtime_cls = demo31_runtime.make_demo31_live_runtime_class(
            _FakeSharedRuntimeModule,
            process_client_factory=lambda _config: _FakeProcessClient(None),
        )
        runtime = runtime_cls(
            SimpleNamespace(camera_ids=(0,)),
            demo31_contract={
                "fusion_mask_policy": "latest-reuse",
                "mask_stale_timeout_ms": 250.0,
                "cotracker_result_stale_timeout_ms": 1500.0,
                "tracking_pending_render_packet_max_groups": 2,
            },
            cotracker_process_config=SimpleNamespace(),
        )

        runtime._protect_frame_bundle(1)
        for group_id in (1, 2, 3):
            runtime._publish_render_packet(
                _FakeRenderPacket(
                    group_id=group_id,
                    controller_points_m=np.empty((0, 3), dtype=np.float32),
                    controller_colors_rgb=np.empty((0, 3), dtype=np.uint8),
                )
            )

        with runtime.demo31_pending_render_lock:
            pending_ids = sorted(runtime.demo31_pending_render_packets)
        self.assertEqual(pending_ids, [1, 3])
        self.assertEqual(runtime.demo31_pending_render_packet_drop_count, 1)
        snapshot = runtime.demo31_snapshot()
        self.assertEqual(snapshot["protected_bundle_count"], 1)
        self.assertIn(1, runtime.demo31_protected_frame_bundle_group_ids)

    def test_pending_fusion_bundle_cache_is_bounded(self) -> None:
        runtime_cls = demo31_runtime.make_demo31_live_runtime_class(
            _FakeSharedRuntimeModule,
            process_client_factory=lambda _config: _FakeProcessClient(None),
        )
        runtime = runtime_cls(
            SimpleNamespace(camera_ids=(0,)),
            demo31_contract={
                "fusion_mask_policy": "latest-reuse",
                "mask_stale_timeout_ms": 250.0,
                "cotracker_result_stale_timeout_ms": 1500.0,
                "tracking_pending_render_packet_max_groups": 2,
                "pcd_fusion_trigger": "tracker-result",
            },
            cotracker_process_config=SimpleNamespace(),
        )

        for group_id in (1, 2, 3):
            runtime._remember_pending_fusion_bundle(
                demo31_runtime.Demo31PendingFusionBundle(
                    group_id=group_id,
                    created_perf_s=demo31_runtime.time.perf_counter(),
                    depth_group=SimpleNamespace(group_id=group_id),
                    masks={},
                    publish_hook="test",
                )
            )

        with runtime.demo31_pending_fusion_lock:
            pending_ids = sorted(runtime.demo31_pending_fusion_bundles)
        self.assertEqual(pending_ids, [2, 3])
        self.assertEqual(runtime.demo31_pending_fusion_bundle_drop_count, 1)
        snapshot = runtime.demo31_snapshot()
        self.assertEqual(snapshot["tracking_pending_fusion_bundle_max_groups"], 2)
        self.assertEqual(snapshot["tracking_pending_fusion_bundles"], 2)

    def test_tracker_result_gated_fusion_defers_work_until_tracker_result(self) -> None:
        now_s = demo31_runtime.time.perf_counter()
        result = TrackingResultLitePacket(
            group_id=7,
            frame_idx=7,
            source_timestamp_s=now_s,
            publish_timestamp_s=now_s,
            camera_tracks_yx={0: np.array([[0.0, 0.0]], dtype=np.float32)},
            camera_visibility={0: np.array([1.0], dtype=np.float32)},
            query_points_yx={0: np.array([[0.0, 0.0]], dtype=np.float32)},
            publish_range=(7, 7),
        )
        client = _FakeProcessClient(result)
        runtime_cls = demo31_runtime.make_demo31_live_runtime_class(
            _FakeSharedRuntimeModule,
            process_client_factory=lambda _config: client,
        )
        runtime = runtime_cls(
            SimpleNamespace(
                camera_ids=(0,),
                tracker_visualization_mode="none",
                enable_pcd_filter=True,
                pcd_filter_mode="async",
            ),
            demo31_contract={
                "fusion_mask_policy": "latest-reuse",
                "mask_stale_timeout_ms": 250.0,
                "cotracker_result_stale_timeout_ms": 1500.0,
                "cotracker_input_fps": 30.0,
                "pcd_fusion_trigger": "tracker-result",
                "tracker_visualization_mode": "none",
            },
            cotracker_process_config=SimpleNamespace(),
        )
        runtime._stream_metadata = [{"K_color": np.eye(3, dtype=np.float32)}]
        runtime._c2w_by_camera = {0: np.eye(4, dtype=np.float32)}
        depth_group = _FakeDepthGroup(
            group_id=7,
            depths={0: _FakeDepthFrame(7, np.ones((1, 1), dtype=np.float32))},
            per_camera_frame_seq={0: 7},
        )
        masks = {
            0: _FakePacket(
                7,
                0,
                np.array([[False]], dtype=bool),
                np.array([[True]], dtype=bool),
                np.zeros((1, 1, 3), dtype=np.uint8),
            )
        }

        raw_packet = runtime._build_raw_fused_packet(
            depth_group=depth_group,
            masks=masks,
            ray_cache={},
            rng=np.random.default_rng(0),
        )

        self.assertIsInstance(raw_packet, demo31_runtime.Demo31PendingFusionBundle)
        self.assertFalse(hasattr(runtime, "raw_fused_call"))
        self.assertEqual(len(client.inputs), 1)
        with runtime.demo31_pending_fusion_lock:
            self.assertEqual(sorted(runtime.demo31_pending_fusion_bundles), [7])

        handled = runtime._publish_next_tracker_driven_render_once(now_s=now_s)

        self.assertTrue(handled)
        self.assertTrue(hasattr(runtime, "raw_fused_call"))
        self.assertTrue(hasattr(runtime, "filter_raw_call"))
        self.assertIsNotNone(runtime.published_packet)
        self.assertEqual(runtime.published_packet.group_id, 7)  # type: ignore[union-attr]
        self.assertEqual(runtime.demo31_tracker_result_triggered_fusion_count, 1)
        with runtime.demo31_pending_fusion_lock:
            self.assertEqual(runtime.demo31_pending_fusion_bundles, {})
        overlay_profile = runtime.profile_updates[-1][1]["demo31_tracking_overlay"]
        self.assertTrue(overlay_profile["tracker_result_gated_fusion"])
        self.assertTrue(overlay_profile["tracker_result_triggered_fusion"])
        self.assertEqual(overlay_profile["tracking_render_packet_match_mode"], "exact")

    def test_tracker_result_gated_fused_packet_path_caches_exact_batch_bundle(self) -> None:
        now_s = demo31_runtime.time.perf_counter()
        result = TrackingResultLitePacket(
            group_id=11,
            frame_idx=11,
            source_timestamp_s=now_s,
            publish_timestamp_s=now_s,
            camera_tracks_yx={0: np.array([[0.0, 0.0]], dtype=np.float32)},
            camera_visibility={0: np.array([1.0], dtype=np.float32)},
            query_points_yx={0: np.array([[0.0, 0.0]], dtype=np.float32)},
            publish_range=(11, 11),
        )
        client = _FakeProcessClient(result)
        runtime_cls = demo31_runtime.make_demo31_live_runtime_class(
            _FakeSharedRuntimeModule,
            process_client_factory=lambda _config: client,
        )
        runtime = runtime_cls(
            SimpleNamespace(
                camera_ids=(0,),
                tracker_visualization_mode="none",
                enable_pcd_filter=True,
                pcd_filter_mode="async",
            ),
            demo31_contract={
                "fusion_mask_policy": "latest-reuse",
                "mask_stale_timeout_ms": 250.0,
                "cotracker_result_stale_timeout_ms": 1500.0,
                "cotracker_input_fps": 30.0,
                "pcd_fusion_trigger": "tracker-result",
                "tracker_visualization_mode": "none",
            },
            cotracker_process_config=SimpleNamespace(),
        )
        runtime._stream_metadata = [{"K_color": np.eye(3, dtype=np.float32)}]
        runtime._c2w_by_camera = {0: np.eye(4, dtype=np.float32)}
        depth_group = _FakeDepthGroup(
            group_id=11,
            depths={0: _FakeDepthFrame(11, np.ones((1, 1), dtype=np.float32))},
            per_camera_frame_seq={0: 11},
        )
        masks = {
            0: _FakePacket(
                11,
                0,
                np.array([[False]], dtype=bool),
                np.array([[True]], dtype=bool),
                np.zeros((1, 1, 3), dtype=np.uint8),
            )
        }

        fused_packet = runtime._build_fused_packet(
            depth_group=depth_group,
            masks=masks,
            ray_cache={},
            rng=np.random.default_rng(0),
        )

        self.assertIsInstance(fused_packet, demo31_runtime.Demo31PendingFusionBundle)
        self.assertFalse(hasattr(runtime, "fused_call"))
        self.assertEqual(len(client.inputs), 1)
        self.assertEqual(client.inputs[0].group_id, 11)
        with runtime.demo31_pending_fusion_lock:
            self.assertEqual(sorted(runtime.demo31_pending_fusion_bundles), [11])

        runtime._publish_render_packet(fused_packet)
        self.assertIsNone(runtime.published_packet)

        handled = runtime._publish_next_tracker_driven_render_once(now_s=now_s)

        self.assertTrue(handled)
        self.assertTrue(hasattr(runtime, "raw_fused_call"))
        self.assertTrue(hasattr(runtime, "filter_raw_call"))
        self.assertIsNotNone(runtime.published_packet)
        self.assertEqual(runtime.published_packet.group_id, 11)  # type: ignore[union-attr]
        overlay_profile = runtime.profile_updates[-1][1]["demo31_tracking_overlay"]
        self.assertTrue(overlay_profile["tracker_result_gated_fusion"])
        self.assertTrue(overlay_profile["tracker_result_triggered_fusion"])
        self.assertEqual(overlay_profile["tracking_render_packet_match_mode"], "exact")
        self.assertEqual(overlay_profile["tracking_render_packet_group_id"], 11)
        self.assertEqual(overlay_profile["tracking_render_packet_group_delta"], 0)

    def test_renderer_warmup_blocks_first_frame_until_tracking_overlay_points_exist(self) -> None:
        runtime_cls = demo31_runtime.make_demo31_live_runtime_class(
            _FakeSharedRuntimeModule,
            process_client_factory=lambda _config: _FakeProcessClient(None),
        )
        runtime = runtime_cls(
            SimpleNamespace(camera_ids=(0,)),
            demo31_contract={
                "fusion_mask_policy": "latest-reuse",
                "mask_stale_timeout_ms": 250.0,
                "cotracker_result_stale_timeout_ms": 1500.0,
                "wait_for_tracking_overlay": True,
            },
            cotracker_process_config=SimpleNamespace(),
        )

        runtime._publish_render_packet(
            _FakeRenderPacket(
                group_id=2,
                controller_points_m=np.empty((0, 3), dtype=np.float32),
                controller_colors_rgb=np.empty((0, 3), dtype=np.uint8),
            )
        )

        self.assertIsNone(runtime.published_packet)
        self.assertEqual(runtime.demo31_tracking_overlay_warmup_skipped_render_count, 0)
        self.assertEqual(runtime.demo31_tracking_overlay_render_blocked_count, 0)
        overlay_profile = runtime.profile_updates[-1][1]["demo31_tracking_overlay"]
        self.assertTrue(overlay_profile["tracking_overlay_warmup_blocked"])
        self.assertTrue(overlay_profile["tracking_overlay_render_blocked"])
        self.assertFalse(overlay_profile["overlay_available"])

    def test_renderer_does_not_publish_pcd_only_when_legacy_wait_flag_is_false(self) -> None:
        runtime_cls = demo31_runtime.make_demo31_live_runtime_class(
            _FakeSharedRuntimeModule,
            process_client_factory=lambda _config: _FakeProcessClient(None),
        )
        runtime = runtime_cls(
            SimpleNamespace(camera_ids=(0,)),
            demo31_contract={
                "fusion_mask_policy": "latest-reuse",
                "mask_stale_timeout_ms": 250.0,
                "cotracker_result_stale_timeout_ms": 1500.0,
                "wait_for_tracking_overlay": False,
            },
            cotracker_process_config=SimpleNamespace(),
        )

        packet = _FakeRenderPacket(
            group_id=2,
            controller_points_m=np.empty((0, 3), dtype=np.float32),
            controller_colors_rgb=np.empty((0, 3), dtype=np.uint8),
        )
        runtime._publish_render_packet(packet)

        self.assertIsNone(runtime.published_packet)
        overlay_profile = runtime.profile_updates[-1][1]["demo31_tracking_overlay"]
        self.assertTrue(overlay_profile["tracking_overlay_warmup_blocked"])
        self.assertTrue(overlay_profile["tracking_overlay_render_blocked"])
        self.assertTrue(overlay_profile["render_requires_new_cotracker_result"])


if __name__ == "__main__":
    unittest.main()
