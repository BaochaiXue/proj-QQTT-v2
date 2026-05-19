from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


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


def build_empty_dual_gpu_profile_summary(contract: dict[str, Any]) -> dict[str, Any]:
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
        "render_waited_for_cotracker": False,
        "render_waited_for_mask": bool(contract.get("render_waited_for_mask", False)),
        "fusion_mask_policy": str(contract.get("fusion_mask_policy", "latest-reuse")),
        "render_loop_fps": 0.0,
        "rendered_fps": 0.0,
        "new_fused_pcd_fps": 0.0,
        "capture_group_fps": 0.0,
        "fresh_mask_fps": 0.0,
        "mask_reuse_ratio": 0.0,
        "mask_age_ms_median": 0.0,
        "mask_age_ms_p95": 0.0,
        "cotracker_input_fps": 0.0,
        "cotracker_input_drop_count": 0,
        "cotracker_input_queue_replace_count": 0,
        "cotracker_publish_fps": 0.0,
        "cotracker_model_ms_median": 0.0,
        "cotracker_model_ms_p95": 0.0,
        "cotracker_e2e_ms_median": 0.0,
        "cotracker_e2e_ms_p95": 0.0,
        "overlay_age_ms_median": 0.0,
        "overlay_age_ms_p95": 0.0,
        "gpu0_util_median": 0.0,
        "gpu0_util_p95": 0.0,
        "gpu0_mem_used_gb": 0.0,
        "gpu1_util_median": 0.0,
        "gpu1_util_p95": 0.0,
        "gpu1_mem_used_gb": 0.0,
        "main_process_pid": 0,
        "cotracker_process_pid": 0,
        "uses_ffs": False,
        "depth_source": "realsense",
        "mask_source": "hf_edgetam",
        "edgetam_live_session_keep_frames": int(contract.get("edgetam_live_session_keep_frames", 64)),
        "edgetam_live_session_pruning": bool(contract.get("edgetam_live_session_pruning", True)),
        "cotracker_backend": "cotracker3_online",
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
        "overlay_max_points_per_camera": int(contract.get("overlay_max_points_per_camera", 30)),
        "overlay_display_count_by_camera": {},
    }


__all__ = [
    "build_empty_dual_gpu_profile_summary",
    "event_fps",
    "percentile_summary",
]
