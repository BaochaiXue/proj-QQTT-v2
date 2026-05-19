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
    "main_cuda_visible_devices",
    "cotracker_cuda_visible_devices",
    "cross_gpu_cuda_tensor_transfer",
    "ipc_payload",
    "render_waited_for_cotracker",
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
    "overlay_max_points_per_camera",
    "wait_for_tracking_overlay",
    "tracking_overlay_required_before_first_render",
    "tracking_overlay_required_for_render",
    "tracking_overlay_color_rgb",
    "tracking_overlay_warmup_skipped_render_count",
    "tracking_overlay_render_blocked_count",
    "tracking_overlay_first_render_group_id",
    "tracking_pending_render_packets",
    "tracking_pending_render_packet_drop_count",
    "tracking_result_without_render_packet_count",
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


def build_empty_demo31_profile_summary(contract: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "dual_gpu_enabled": True,
        "mask_gpu_physical": int(contract.get("mask_gpu_physical", 0)),
        "cotracker_gpu_physical": int(contract.get("cotracker_gpu_physical", 1)),
        "main_cuda_visible_devices": str(contract.get("main_cuda_visible_devices", "0")),
        "cotracker_cuda_visible_devices": str(contract.get("cotracker_cuda_visible_devices", "1")),
        "cross_gpu_cuda_tensor_transfer": False,
        "ipc_payload": "cpu_numpy_latest_wins",
        "cotracker_owner": "process",
        "cotracker_process_mode": str(contract.get("cotracker_process_mode", "subprocess")),
        "cotracker_prewarm_backends": bool(contract.get("cotracker_prewarm_backends", True)),
        "cotracker_update_mode": str(contract.get("cotracker_update_mode", "auto")),
        "cotracker_update_mode_effective": str(contract.get("cotracker_update_mode", "auto")),
        "tracker_backend": str(contract.get("tracker_backend", contract.get("cotracker_backend", "cotracker3_online"))),
        "tracker_backend_family": str(contract.get("tracker_backend_family", "cotracker")),
        "tracking_backend_execution_mode": str(contract.get("tracking_backend_execution_mode", "auto")),
        "tracking_backend_batch_dimension": str(contract.get("tracking_backend_batch_dimension", "camera")),
        "tracking_backend_batch_size": int(contract.get("tracking_backend_batch_size", 3)),
        "tracking_backend_batch_enabled": False,
        "tracking_backend_batch_supported": bool(contract.get("tracking_backend_batch_supported", True)),
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
        "cotracker_process_total_init_ms": 0.0,
        "cotracker_backend_warmup_ms": 0.0,
        "cotracker_backend_warmup_by_camera": {},
        "render_waited_for_cotracker": False,
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
        ProfileKeys.CAPTURE_GROUP_FPS: 0.0,
        ProfileKeys.FRESH_MASK_FPS: 0.0,
        ProfileKeys.MASK_REUSE_RATIO: 0.0,
        "mask_age_ms_median": 0.0,
        "mask_age_ms_p95": 0.0,
        "mask_group_delta_median": 0.0,
        "mask_group_delta_p95": 0.0,
        ProfileKeys.COTRACKER_INPUT_FPS: 0.0,
        "cotracker_input_drop_count": 0,
        "cotracker_input_queue_replace_count": 0,
        ProfileKeys.COTRACKER_PUBLISH_FPS: 0.0,
        "cotracker_model_ms_median": 0.0,
        "cotracker_model_ms_p95": 0.0,
        "cotracker_e2e_ms_median": 0.0,
        "cotracker_e2e_ms_p95": 0.0,
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
        "uses_ffs": False,
        "depth_source": "realsense",
        "mask_source": "hf_edgetam",
        "edgetam_live_session_keep_frames": int(contract.get("edgetam_live_session_keep_frames", 64)),
        "edgetam_live_session_pruning": bool(contract.get("edgetam_live_session_pruning", True)),
        "cotracker_backend": str(contract.get("cotracker_backend", "cotracker3_online")),
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
        "tracking_query_count_rule": str(contract.get("tracking_query_count_rule", "min(union_mask_pixels, 5000)")),
        "tracking_sampling": str(contract.get("tracking_sampling", "torch_randperm_seed_plus_camera_idx")),
        "cotracker_seed": int(contract.get("cotracker_seed", 42)),
        "phystwin_dense_compatible": bool(contract.get("phystwin_dense_compatible", False)),
        "tracking_query_count_actual_by_camera": {},
        "tracking_union_pixels_by_camera": {},
        "tracking_object_pixels_by_camera": {},
        "tracking_controller_pixels_by_camera": {},
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
        "tracking_overlay_color_rgb": list(contract.get("tracking_overlay_color_rgb", [255, 0, 0])),
        "tracking_overlay_warmup_skipped_render_count": 0,
        "tracking_overlay_render_blocked_count": 0,
        "tracking_overlay_first_render_group_id": None,
        "tracking_pending_render_packets": 0,
        "tracking_pending_render_packet_drop_count": 0,
        "tracking_result_without_render_packet_count": 0,
        "overlay_display_scope": str(contract.get("overlay_display_scope", "controller")),
        "overlay_display_classification": str(
            contract.get("overlay_display_classification", "first_frame_mask_membership")
        ),
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
