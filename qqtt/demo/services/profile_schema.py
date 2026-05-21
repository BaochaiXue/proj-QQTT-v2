from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json

import numpy as np


class ProfileKeys:
    RENDERED_FPS = "rendered_fps"
    RENDER_LOOP_FPS = "render_loop_fps"
    NEW_FUSED_PCD_FPS = "new_fused_pcd_fps"
    CAPTURE_GROUP_FPS = "capture_group_fps"
    FRESH_MASK_FPS = "fresh_mask_fps"
    MASK_REUSE_RATIO = "mask_reuse_ratio"
    OBJECT_POINT_CONTROL = "object_point_control"
    OBJECT_VOLUME_MS = "object_volume_ms"
    OBJECT_VOLUME_MS_P50 = "object_volume_ms_p50"
    OBJECT_VOLUME_TOTAL_MS = "object_volume_total_ms"
    OBJECT_VOLUME_KEY_MS = "object_volume_key_ms"
    OBJECT_VOLUME_UNIQUE_MS = "object_volume_unique_ms"
    OBJECT_VOLUME_GATHER_MS = "object_volume_gather_ms"
    OBJECT_VOLUME_OCCUPIED_VOXELS = "object_volume_occupied_voxels"
    OBJECT_VOLUME_OUTPUT_POINTS = "object_volume_output_points"
    OBJECT_VOLUME_INPUT_POINTS = "object_volume_input_points"
    OBJECT_VOLUME_TARGET_MS = "object_volume_target_ms"
    OBJECT_VOLUME_EXACT = "object_volume_exact"
    OBJECT_VOLUME_SAFETY_CAP_TRIGGERED = "object_volume_safety_cap_triggered"
    COTRACKER_PUBLISH_FPS = "cotracker_publish_fps"
    COTRACKER_INPUT_FPS = "cotracker_input_fps"
    GPU0_UTIL_MEDIAN = "gpu0_util_median"
    GPU1_UTIL_MEDIAN = "gpu1_util_median"
    BOTTLENECK_CLASS = "bottleneck_class"


DEMO31_REQUIRED_PROFILE_KEYS = (
    "dual_gpu_enabled",
    "mask_gpu_physical",
    "cotracker_gpu_physical",
    "ffs_gpu_physical",
    "edgetam_gpu_physical",
    "sam31_gpu_physical",
    "litetracker_gpu_physical",
    "ffs_edgetam_same_gpu",
    "shared_runtime_gpu_placement",
    "main_cuda_visible_devices",
    "cotracker_cuda_visible_devices",
    "cross_gpu_cuda_tensor_transfer",
    "ipc_payload",
    "render_waited_for_cotracker",
    "render_waited_for_fresh_cotracker_result",
    "render_requires_new_cotracker_result",
    "render_reuses_cached_cotracker_result",
    "render_driver",
    "render_trigger",
    "rendered_on_new_cotracker_result",
    "render_waited_for_mask",
    ProfileKeys.RENDER_LOOP_FPS,
    ProfileKeys.RENDERED_FPS,
    ProfileKeys.NEW_FUSED_PCD_FPS,
    ProfileKeys.CAPTURE_GROUP_FPS,
    ProfileKeys.FRESH_MASK_FPS,
    ProfileKeys.MASK_REUSE_RATIO,
    ProfileKeys.OBJECT_POINT_CONTROL,
    ProfileKeys.OBJECT_VOLUME_MS,
    ProfileKeys.OBJECT_VOLUME_TOTAL_MS,
    ProfileKeys.OBJECT_VOLUME_KEY_MS,
    ProfileKeys.OBJECT_VOLUME_UNIQUE_MS,
    ProfileKeys.OBJECT_VOLUME_GATHER_MS,
    ProfileKeys.OBJECT_VOLUME_EXACT,
    ProfileKeys.OBJECT_VOLUME_OCCUPIED_VOXELS,
    ProfileKeys.OBJECT_VOLUME_OUTPUT_POINTS,
    ProfileKeys.COTRACKER_INPUT_FPS,
    ProfileKeys.COTRACKER_PUBLISH_FPS,
    ProfileKeys.GPU0_UTIL_MEDIAN,
    ProfileKeys.GPU1_UTIL_MEDIAN,
    "input_source",
    "offline_mode_available",
    "tracking_mask_scope",
    "tracking_query_mode",
    "tracking_query_count_requested",
    "controller_pcd_max_points_per_camera",
    "controller_pcd_cap_stage",
    "overlay_max_points_per_camera",
    "wait_for_tracking_overlay",
    "tracking_overlay_required_before_first_render",
    "tracking_overlay_required_for_render",
    "tracking_overlay_color_rgb",
    "tracking_overlay_warmup_skipped_render_count",
    "tracking_overlay_render_blocked_count",
    "tracking_overlay_first_render_group_id",
    "tracking_pending_render_packets",
    "tracking_pending_render_packet_max_groups",
    "tracking_pending_render_packet_drop_count",
    "tracking_render_packet_match_policy",
    "tracking_result_exact_render_packet_count",
    "tracking_result_nearest_render_packet_count",
    "tracking_result_without_render_packet_count",
    "tracking_result_without_lift_input_count",
)


DEMO23_REQUIRED_PROFILE_KEYS = (
    ProfileKeys.CAPTURE_GROUP_FPS,
    "raw_fusion_fps",
    "filter_output_fps",
    "fusion_fps",
    "render_fps",
    ProfileKeys.OBJECT_POINT_CONTROL,
    ProfileKeys.OBJECT_VOLUME_MS,
    ProfileKeys.OBJECT_VOLUME_TOTAL_MS,
    ProfileKeys.OBJECT_VOLUME_KEY_MS,
    ProfileKeys.OBJECT_VOLUME_UNIQUE_MS,
    ProfileKeys.OBJECT_VOLUME_GATHER_MS,
    ProfileKeys.OBJECT_VOLUME_EXACT,
    ProfileKeys.OBJECT_VOLUME_OCCUPIED_VOXELS,
    ProfileKeys.OBJECT_VOLUME_OUTPUT_POINTS,
)


@dataclass
class RuntimeProfile:
    demo: str
    preset: str
    contract: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    service_profiles: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "demo": self.demo,
            "preset": self.preset,
            "contract": dict(self.contract),
            "summary": dict(self.summary),
            "service_profiles": {
                str(name): dict(payload) for name, payload in self.service_profiles.items()
            },
        }


def percentile_summary(values: Sequence[float]) -> dict[str, float]:
    arr = np.asarray([float(value) for value in values], dtype=np.float32)
    if arr.size == 0:
        return {"median": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def event_fps(times_s: Sequence[float]) -> float:
    if len(times_s) < 2:
        return 0.0
    duration_s = float(max(times_s) - min(times_s))
    return float((len(times_s) - 1) / duration_s) if duration_s > 0.0 else 0.0


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def build_empty_demo31_profile_summary(contract: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "dual_gpu_enabled": True,
        "mask_gpu_physical": int(contract.get("mask_gpu_physical", 0)),
        "cotracker_gpu_physical": int(contract.get("cotracker_gpu_physical", 1)),
        "ffs_gpu_physical": _optional_int(contract.get("ffs_gpu_physical")),
        "edgetam_gpu_physical": int(contract.get("edgetam_gpu_physical", contract.get("mask_gpu_physical", 0))),
        "sam31_gpu_physical": int(contract.get("sam31_gpu_physical", contract.get("mask_gpu_physical", 0))),
        "litetracker_gpu_physical": _optional_int(contract.get("litetracker_gpu_physical")),
        "ffs_edgetam_same_gpu": bool(contract.get("ffs_edgetam_same_gpu", False)),
        "shared_runtime_gpu_placement": str(contract.get("shared_runtime_gpu_placement", "mask_gpu0_track_gpu1")),
        "main_cuda_visible_devices": str(contract.get("main_cuda_visible_devices", "0")),
        "cotracker_cuda_visible_devices": str(contract.get("cotracker_cuda_visible_devices", "1")),
        "cross_gpu_cuda_tensor_transfer": False,
        "ipc_payload": "cpu_numpy_latest_wins",
        "cotracker_owner": "process",
        "cotracker_process_mode": str(contract.get("cotracker_process_mode", "subprocess")),
        "cotracker_prewarm_backends": bool(contract.get("cotracker_prewarm_backends", True)),
        "cotracker_update_mode": str(contract.get("cotracker_update_mode", "batch")),
        "cotracker_update_mode_effective": str(contract.get("cotracker_update_mode", "batch")),
        "tracker_backend": str(contract.get("tracker_backend", contract.get("cotracker_backend", "cotracker3_online"))),
        "tracker_backend_family": str(contract.get("tracker_backend_family", "cotracker")),
        "tracking_backend_execution_mode": str(contract.get("tracking_backend_execution_mode", "batch-views")),
        "tracking_backend_batch_dimension": str(contract.get("tracking_backend_batch_dimension", "camera")),
        "tracking_backend_batch_size": int(contract.get("tracking_backend_batch_size", 3)),
        "tracking_backend_batch_enabled": str(contract.get("cotracker_update_mode", "batch")) == "batch",
        "tracking_backend_batch_supported": bool(contract.get("tracking_backend_batch_supported", True)),
        "tracking_backend_supports_batch_views": bool(
            contract.get("tracking_backend_supports_batch_views", contract.get("tracking_backend_batch_supported", True))
        ),
        "tracking_backend_supports_online": bool(contract.get("tracking_backend_supports_online", True)),
        "tracking_backend_online_semantics": str(contract.get("tracking_backend_online_semantics", "online")),
        "tracking_backend_batch_support_status": str(contract.get("tracking_backend_batch_support_status", "true")),
        "tracking_backend_batch_auto_selected": bool(contract.get("tracking_backend_batch_auto_selected", False)),
        "tracker_batch_query_count_policy": str(contract.get("tracker_batch_query_count_policy", "fixed")),
        "tracking_backend_effective_query_count": 0,
        "tracking_backend_query_count_truncated_by_camera": {},
        "tracking_backend_batch_fallback_reason": None,
        "cotracker_batch_size_target": int(contract.get("cotracker_batch_size_target", 3)),
        "cotracker_batch_size": 0,
        "cotracker_batch_update_count": 0,
        "cotracker_serial_group_update_count": 0,
        "cotracker_serial_camera_update_count": 0,
        "cotracker_serial_fallback_count": 0,
        "cotracker_batch_error_count": 0,
        "cotracker_batch_disabled_reason": None,
        "cotracker_process_ready": False,
        "tracker_process_ready": False,
        "tracker_ready_to_receive_inputs": False,
        "tracker_process_ready_s": 0.0,
        "tracker_ready_to_receive_inputs_s": 0.0,
        "tracker_ready_state": str(contract.get("tracker_ready_state", "ready_to_receive_inputs")),
        "cotracker_process_total_init_ms": 0.0,
        "tracker_process_total_init_ms": 0.0,
        "cotracker_backend_warmup_ms": 0.0,
        "tracker_backend_warmup_ms": 0.0,
        "cotracker_backend_warmup_by_camera": {},
        "tracker_backend_warmup_by_camera": {},
        "tracker_prewarm_backends": bool(contract.get("tracker_prewarm_backends", contract.get("cotracker_prewarm_backends", True))),
        "tracker_prewarm_mode": str(contract.get("tracker_prewarm_mode", "unknown")),
        "tracker_query_dependent_init": bool(contract.get("tracker_query_dependent_init", False)),
        "tracker_query_dependent_init_pending": bool(
            contract.get("tracker_query_dependent_init_pending_until_first_input", False)
        ),
        "render_waited_for_cotracker": bool(contract.get("render_waited_for_cotracker", False)),
        "render_waited_for_fresh_cotracker_result": bool(
            contract.get("render_waited_for_fresh_cotracker_result", False)
        ),
        "render_waited_for_mask": bool(contract.get("render_waited_for_mask", False)),
        "fusion_mask_policy": str(contract.get("fusion_mask_policy", "latest-reuse")),
        ProfileKeys.RENDER_LOOP_FPS: 0.0,
        ProfileKeys.RENDERED_FPS: 0.0,
        ProfileKeys.NEW_FUSED_PCD_FPS: 0.0,
        "render_waited_for_object_volume_filter": False,
        "object_volume_filter_source_counts": {},
        "object_volume_worker_fps": 0.0,
        "object_volume_age_ms_p50": 0.0,
        "object_volume_age_ms_p95": 0.0,
        ProfileKeys.OBJECT_POINT_CONTROL: "phystwin-volume",
        ProfileKeys.OBJECT_VOLUME_MS: 0.0,
        ProfileKeys.OBJECT_VOLUME_TOTAL_MS: 0.0,
        ProfileKeys.OBJECT_VOLUME_KEY_MS: 0.0,
        ProfileKeys.OBJECT_VOLUME_UNIQUE_MS: 0.0,
        ProfileKeys.OBJECT_VOLUME_GATHER_MS: 0.0,
        ProfileKeys.OBJECT_VOLUME_INPUT_POINTS: 0,
        ProfileKeys.OBJECT_VOLUME_OCCUPIED_VOXELS: 0,
        ProfileKeys.OBJECT_VOLUME_OUTPUT_POINTS: 0,
        ProfileKeys.OBJECT_VOLUME_TARGET_MS: 8.0,
        ProfileKeys.OBJECT_VOLUME_EXACT: False,
        "object_volume_adaptive_active": False,
        "object_volume_sampler_impl": "",
        ProfileKeys.OBJECT_VOLUME_SAFETY_CAP_TRIGGERED: False,
        "controller_render_voxel_enabled": bool(
            contract.get("render_controller_filter", {}).get("render_voxel_downsample", False)
            if isinstance(contract.get("render_controller_filter"), dict)
            else False
        ),
        "controller_render_voxel_m": float(
            contract.get("render_controller_filter", {}).get("render_voxel_m", 0.0)
            if isinstance(contract.get("render_controller_filter"), dict)
            else 0.0
        ),
        "controller_render_voxel_input_points": 0,
        "controller_render_voxel_output_points": 0,
        "controller_render_voxel_removed_points": 0,
        "controller_render_voxel_ms": 0.0,
        "controller_render_voxel_stage": "render_pcd_only_after_controller_postprocess",
        "controller_render_voxel_affects_tracking_markers": False,
        "controller_render_cap_enabled": bool(
            contract.get("render_controller_filter", {}).get("render_cap_enabled", False)
            if isinstance(contract.get("render_controller_filter"), dict)
            else False
        ),
        "controller_render_max_points": int(
            contract.get("render_controller_filter", {}).get("render_max_points", 0)
            if isinstance(contract.get("render_controller_filter"), dict)
            else 0
        ),
        "controller_render_cap_input_points": 0,
        "controller_render_cap_output_points": 0,
        "controller_render_cap_removed_points": 0,
        "controller_render_cap_ms": 0.0,
        "controller_render_cap_stage": "render_pcd_only_after_controller_render_voxel",
        "controller_render_cap_affects_tracking_markers": False,
        ProfileKeys.CAPTURE_GROUP_FPS: 0.0,
        ProfileKeys.FRESH_MASK_FPS: 0.0,
        ProfileKeys.MASK_REUSE_RATIO: 0.0,
        "mask_age_ms_median": 0.0,
        "mask_age_ms_p95": 0.0,
        "mask_group_delta_median": 0.0,
        "mask_group_delta_p95": 0.0,
        ProfileKeys.COTRACKER_INPUT_FPS: 0.0,
        "tracker_input_fps": 0.0,
        "cotracker_input_drop_count": 0,
        "tracker_input_drop_count": 0,
        "cotracker_input_queue_replace_count": 0,
        "tracker_input_queue_replace_count": 0,
        ProfileKeys.COTRACKER_PUBLISH_FPS: 0.0,
        "tracker_publish_fps": 0.0,
        "cotracker_model_ms_median": 0.0,
        "tracker_model_ms_median": 0.0,
        "cotracker_model_ms_p95": 0.0,
        "tracker_model_ms_p95": 0.0,
        "cotracker_e2e_ms_median": 0.0,
        "tracker_e2e_ms_median": 0.0,
        "cotracker_e2e_ms_p95": 0.0,
        "tracker_e2e_ms_p95": 0.0,
        "overlay_age_ms_median": 0.0,
        "overlay_age_ms_p95": 0.0,
        "overlay_render_group_delta_median": 0.0,
        "overlay_render_group_delta_p95": 0.0,
        "overlay_render_group_mismatch_count": 0,
        "tracking_input_mask_reuse_ratio": 0.0,
        "tracking_input_mask_age_ms_median": 0.0,
        "tracking_input_mask_age_ms_p95": 0.0,
        ProfileKeys.GPU0_UTIL_MEDIAN: 0.0,
        "gpu0_util_p95": 0.0,
        "gpu0_mem_used_gb": 0.0,
        ProfileKeys.GPU1_UTIL_MEDIAN: 0.0,
        "gpu1_util_p95": 0.0,
        "gpu1_mem_used_gb": 0.0,
        "main_process_pid": 0,
        "cotracker_process_pid": 0,
        "uses_ffs": bool(contract.get("uses_ffs", False)),
        "depth_source": str(contract.get("depth_source", "realsense")),
        "mask_source": "hf_edgetam",
        "edgetam_live_session_keep_frames": int(contract.get("edgetam_live_session_keep_frames", 64)),
        "edgetam_live_session_pruning": bool(contract.get("edgetam_live_session_pruning", True)),
        "cotracker_backend": str(contract.get("cotracker_backend", "cotracker3_online")),
        "litetracker_runtime": str(contract.get("litetracker_runtime", "pytorch")),
        "litetracker_onnx_dir": contract.get("litetracker_onnx_dir"),
        "litetracker_export_onnx": bool(contract.get("litetracker_export_onnx", False)),
        "litetracker_onnx_opset": int(contract.get("litetracker_onnx_opset", 17)),
        "litetracker_onnx_opset_actual": int(contract.get("litetracker_onnx_opset_actual", 18)),
        "litetracker_onnx_optimization_level": int(contract.get("litetracker_onnx_optimization_level", 5)),
        "locotrack_model_size": str(contract.get("locotrack_model_size", "small")),
        "locotrack_window_frames": int(contract.get("locotrack_window_frames", 8)),
        "locotrack_resolution": list(contract.get("locotrack_resolution", [256, 256])),
        "locotrack_query_chunk_size": int(contract.get("locotrack_query_chunk_size", 256)),
        "locotrack_autocast_dtype": str(contract.get("locotrack_autocast_dtype", "bf16")),
        "locotrack_checkpoint": contract.get("locotrack_checkpoint"),
        "locotrack_repo_dir": contract.get("locotrack_repo_dir"),
        "input_source": "live_realsense",
        "offline_mode_available": False,
        "offline_tracking_available": False,
        "init_mode": "sam31_first_frame",
        "mask_propagation": "hf_edgetam_online",
        "semantic_mode": str(contract.get("semantic_mode", "exp")),
        "tracking_mask_scope": str(contract.get("tracking_mask_scope", "object_controller_union")),
        "tracking_controller_label": str(contract.get("tracking_controller_label", "towel")),
        "tracking_query_mode": str(contract.get("tracking_query_mode", "phystwin_dense")),
        "tracking_query_count_requested": str(contract.get("tracking_query_count_requested", "auto")),
        "tracking_query_count_rule": str(
            contract.get("tracking_query_count_rule", "min(capped_object_controller_union_pixels, 5000)")
        ),
        "tracking_sampling": str(
            contract.get("tracking_sampling", "controller_pcd_cap_then_torch_randperm_seed_plus_camera_idx")
        ),
        "controller_pcd_max_points_per_camera": int(contract.get("controller_pcd_max_points_per_camera", 4999)),
        "controller_pcd_cap_stage": str(contract.get("controller_pcd_cap_stage", "before_tracking_query_and_fusion")),
        "controller_pcd_cap_sampling": str(
            contract.get("controller_pcd_cap_sampling", "stable_coordinate_hash_seed_plus_camera_idx")
        ),
        "cotracker_seed": int(contract.get("cotracker_seed", 42)),
        "phystwin_dense_compatible": bool(contract.get("phystwin_dense_compatible", False)),
        "tracking_query_count_actual_by_camera": {},
        "tracking_union_pixels_by_camera": {},
        "tracking_object_pixels_by_camera": {},
        "tracking_controller_pixels_by_camera": {},
        "trackable_mask_build_policy": str(contract.get("trackable_mask_build_policy", "init-only")),
        "trackable_mask_build_stage": str(contract.get("trackable_mask_build_stage", "first_valid_tracking_input")),
        "trackable_query_init_strategy": str(contract.get("trackable_query_init_strategy", "standard-filter-init")),
        "trackable_mask_source": str(contract.get("trackable_mask_source", "standard_filter_survivors")),
        "tracking_input_mask_semantics": str(contract.get("tracking_input_mask_semantics", "standard_filter_trackable_masks")),
        "tracker_query_source": str(contract.get("tracker_query_source", "union_trackable_mask")),
        "object_mask_semantics": str(contract.get("object_mask_semantics", "object_trackable_mask")),
        "controller_mask_semantics": str(contract.get("controller_mask_semantics", "controller_trackable_mask")),
        "controller_trackable_max_points_per_camera": int(contract.get("controller_trackable_max_points_per_camera", 4999)),
        "controller_trackable_cap_stage": str(contract.get("controller_trackable_cap_stage", "after_standard_filter")),
        "controller_mask_erode_px": int(contract.get("controller_mask_erode_px", 0)),
        "controller_mask_erode_unit": str(contract.get("controller_mask_erode_unit", "px")),
        "controller_mask_erode_stage": str(
            contract.get("controller_mask_erode_stage", "before_tracking_union_and_trackable_filter")
        ),
        "controller_mask_erode_applies_to": str(
            contract.get("controller_mask_erode_applies_to", "tracking_input_and_anchor_masks")
        ),
        "controller_mask_pixels_before_erode_by_camera": {},
        "controller_mask_pixels_after_erode_by_camera": {},
        "first_trackable_mask_group_id": None,
        "first_trackable_mask_s": None,
        "first_tracking_input_publish_s": None,
        "trackable_mask_initialized_cameras": [],
        "raw_object_pixels_by_camera": {},
        "raw_controller_pixels_by_camera": {},
        "depth_valid_object_pixels_by_camera": {},
        "depth_valid_controller_pixels_by_camera": {},
        "trackable_object_pixels_by_camera": {},
        "controller_trackable_before_cap_by_camera": {},
        "controller_trackable_after_cap_by_camera": {},
        "controller_trackable_cap_applied_by_camera": {},
        "trackable_union_pixels_by_camera": {},
        "trackable_mask_standard_filter_ms_by_camera": {},
        "tracking_sample_object_hits_by_camera": {},
        "tracking_sample_controller_hits_by_camera": {},
        "tracking_sample_overlap_hits_by_camera": {},
        "tracking_sample_background_hits_by_camera": {},
        "overlay_max_points_per_camera": int(contract.get("overlay_max_points_per_camera", 0)),
        "wait_for_tracking_overlay": bool(contract.get("wait_for_tracking_overlay", True)),
        "tracking_overlay_required_before_first_render": bool(
            contract.get("tracking_overlay_required_before_first_render", True)
        ),
        "tracking_overlay_required_for_render": bool(contract.get("tracking_overlay_required_for_render", True)),
        "render_requires_new_cotracker_result": bool(contract.get("render_requires_new_cotracker_result", True)),
        "render_reuses_cached_cotracker_result": bool(contract.get("render_reuses_cached_cotracker_result", False)),
        "render_driver": str(contract.get("render_driver", "cotracker_child_output")),
        "render_trigger": str(contract.get("render_trigger", "new_cotracker_result")),
        "rendered_on_new_cotracker_result": False,
        "tracker_visualization_mode": str(contract.get("tracker_visualization_mode", "3d-surface-markers")),
        "tracker_3d_marker_mode": str(contract.get("tracker_3d_marker_mode", "surface_snap")),
        "tracker_3d_marker_shape": str(contract.get("tracker_3d_marker_shape", "sphere")),
        "tracker_legacy_lift_used": bool(contract.get("tracker_legacy_lift_used", False)),
        "tracker_direct_depth_lift_used": bool(contract.get("tracker_direct_depth_lift_used", False)),
        "tracker_all_tracks_anchor_mode": bool(contract.get("tracker_all_tracks_anchor_mode", False)),
        "tracker_surface_gate_enabled": bool(contract.get("tracker_surface_gate_enabled", False)),
        "tracker_3d_snap_radius_px": float(contract.get("tracker_3d_snap_radius_px", 4.0)),
        "tracker_3d_marker_radius_m": float(contract.get("tracker_3d_marker_radius_m", 0.006)),
        "tracker_control_points_per_camera": int(contract.get("tracker_control_points_per_camera", 16)),
        "tracker_control_point_selection": str(contract.get("tracker_control_point_selection", "visible-spread")),
        "tracker_surface_anchor_cache_hit": False,
        "tracker_surface_anchor_group_id": None,
        "tracker_marker_accepted_by_camera": {},
        "tracker_marker_rejected_by_camera": {},
        "tracker_marker_pixel_error_median_by_camera": {},
        "tracker_marker_pixel_error_p95_by_camera": {},
        "tracker_marker_layer_by_camera": {},
        "tracker_marker_points_rendered": 0,
        "tracker_marker_points_appended": False,
        "tracking_overlay_color_rgb": list(contract.get("tracking_overlay_color_rgb", [255, 0, 0])),
        "tracking_overlay_warmup_skipped_render_count": 0,
        "tracking_overlay_render_blocked_count": 0,
        "tracking_overlay_first_render_group_id": None,
        "tracking_pending_render_packets": 0,
        "tracking_pending_render_packet_max_groups": int(contract.get("tracking_pending_render_packet_max_groups", 128)),
        "tracking_pending_render_packet_drop_count": 0,
        "tracking_render_packet_match_policy": str(
            contract.get("tracking_render_packet_match_policy", "exact-then-nearest-pending-pcd-by-group-id")
        ),
        "tracking_result_exact_render_packet_count": 0,
        "tracking_result_nearest_render_packet_count": 0,
        "tracking_result_without_render_packet_count": 0,
        "tracking_result_without_lift_input_count": 0,
        "overlay_display_scope": str(contract.get("overlay_display_scope", "controller")),
        "overlay_display_classification": str(
            contract.get("overlay_display_classification", "first_frame_mask_membership")
        ),
        "overlay_bbox_filter_enabled": bool(contract.get("overlay_bbox_filter_enabled", True)),
        "overlay_bbox_filter_scope": str(contract.get("overlay_bbox_filter_scope", "controller")),
        "overlay_bbox_filter_margin_m": float(contract.get("overlay_bbox_filter_margin_m", 0.15)),
        "overlay_bbox_input_points_by_camera": {},
        "overlay_bbox_kept_points_by_camera": {},
        "overlay_bbox_rejected_by_camera": {},
        "overlay_rejected_by_depth_or_bounds_by_camera": {},
        "overlay_world_centroid_by_camera_before_bbox": {},
        "tracking_control_point_markers": bool(contract.get("tracking_control_point_markers", True)),
        "tracking_control_point_count_requested": int(contract.get("tracking_control_point_count_requested", 30)),
        "tracking_control_point_count": 0,
        "tracking_control_points_per_camera": int(contract.get("tracking_control_points_per_camera", 16)),
        "tracking_control_point_radius_m": float(contract.get("tracking_control_point_radius_m", 0.006)),
        "tracking_control_point_sampling": str(
            contract.get("tracking_control_point_sampling", "farthest_point_sample_after_lift_scope_and_bbox")
        ),
        "tracking_control_points_by_camera": {},
        "tracking_control_marker_points": 0,
        "overlay_render_raw_track_points": bool(contract.get("overlay_render_raw_track_points", False)),
        "overlay_display_count_by_camera": {},
        "overlay_display_object_count_by_camera": {},
        "overlay_display_controller_count_by_camera": {},
    }


def build_empty_dual_gpu_profile_summary(contract: Mapping[str, Any]) -> dict[str, Any]:
    return build_empty_demo31_profile_summary(contract)


def merge_service_profiles(*profiles: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for profile in profiles:
        merged.update(dict(profile))
    return merged


def classify_bottleneck(summary: Mapping[str, Any]) -> str:
    if bool(summary.get("render_waited_for_object_volume_filter", False)):
        return "object_volume_filter_blocking_render"
    if float(summary.get("render_total_ms_p50", 0.0) or 0.0) > 20.0:
        return "renderer"
    if float(summary.get(ProfileKeys.OBJECT_VOLUME_MS_P50, summary.get(ProfileKeys.OBJECT_VOLUME_MS, 0.0)) or 0.0) > 15.0:
        return "object_volume_filter"
    render_loop_fps = float(summary.get(ProfileKeys.RENDER_LOOP_FPS, summary.get("render_fps", 0.0)) or 0.0)
    fresh_mask_fps = float(summary.get(ProfileKeys.FRESH_MASK_FPS, 0.0) or 0.0)
    if render_loop_fps > 0.0 and fresh_mask_fps > 0.0 and fresh_mask_fps < render_loop_fps * 0.5:
        return "mask_supply"
    if bool(summary.get("cotracker_enabled", False)) and float(summary.get(ProfileKeys.COTRACKER_PUBLISH_FPS, 0.0) or 0.0) == 0.0:
        return "tracking_supply"
    return "upstream_supply"


def write_profile_json(path: Path, profile: RuntimeProfile) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def write_profile_markdown(path: Path, profile: RuntimeProfile) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = profile.summary
    lines = [
        f"# {profile.demo} Profile",
        "",
        f"- preset: `{profile.preset}`",
        f"- rendered FPS: `{float(summary.get(ProfileKeys.RENDERED_FPS, summary.get('render_fps', 0.0)) or 0.0):.2f}`",
        f"- object volume ms: `{float(summary.get(ProfileKeys.OBJECT_VOLUME_MS, 0.0) or 0.0):.2f}`",
        f"- bottleneck: `{classify_bottleneck(summary)}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


__all__ = [
    "DEMO23_REQUIRED_PROFILE_KEYS",
    "DEMO31_REQUIRED_PROFILE_KEYS",
    "ProfileKeys",
    "RuntimeProfile",
    "build_empty_demo31_profile_summary",
    "build_empty_dual_gpu_profile_summary",
    "classify_bottleneck",
    "event_fps",
    "merge_service_profiles",
    "percentile_summary",
    "write_profile_json",
    "write_profile_markdown",
]
