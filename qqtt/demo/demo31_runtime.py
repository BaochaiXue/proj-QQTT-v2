from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable, Sequence

import numpy as np

from qqtt.demo import demo3_runtime
from qqtt.demo.demo31_cotracker_process import (
    CoTrackerProcessConfig,
    PROCESS_MODE_SUBPROCESS,
    PROCESS_MODES,
    start_cotracker_process,
)
from qqtt.demo.demo31_dual_gpu_ipc import (
    LatestMaskCache,
    TrackingInputLitePacket,
    TrackingResultLitePacket,
    should_publish_tracking_input,
)
from qqtt.demo.demo31_profile import build_empty_dual_gpu_profile_summary, percentile_summary
from qqtt.demo.tracking_overlay_render import lift_tracks_yx_to_world


PRESET_DEMO31_DUAL4090_HIGHFPS = "demo3.1-dual4090-highfps"
PRESETS = (PRESET_DEMO31_DUAL4090_HIGHFPS,)

FUSION_MASK_POLICY_STRICT = "strict"
FUSION_MASK_POLICY_LATEST_REUSE = "latest-reuse"
FUSION_MASK_POLICIES = (FUSION_MASK_POLICY_STRICT, FUSION_MASK_POLICY_LATEST_REUSE)

GPU_PLAN_SPLIT_MASK0_TRACK1 = "split-mask0-track1"
GPU_PLANS = (GPU_PLAN_SPLIT_MASK0_TRACK1,)

DEFAULT_OUTPUT_ROOT = Path("result/demo31_dual4090_realsense_cotracker")
DEFAULT_RENDER_TARGET_FPS = 60.0
DEFAULT_COTRACKER_INPUT_FPS = 10.0
DEFAULT_COTRACKER_INPUT_MAX_AGE_MS = 250.0
DEFAULT_COTRACKER_RESULT_STALE_TIMEOUT_MS = 1500.0
DEFAULT_MASK_STALE_TIMEOUT_MS = 250.0
DEFAULT_MASK_GPU = "0"
DEFAULT_COTRACKER_GPU = "1"
DEFAULT_LIFT_INPUT_CACHE_GROUPS = 32

ConnectedSerialsProvider = Callable[[], Sequence[str]]
CudaDeviceCountProvider = Callable[[], int]
ProcessClientFactory = Callable[[CoTrackerProcessConfig], Any]


@dataclass(frozen=True)
class Demo31LiftInputSnapshot:
    group_id: int
    timestamp_s: float
    depth_by_camera: dict[int, np.ndarray]
    intrinsics_by_camera: dict[int, np.ndarray]
    c2w_by_camera: dict[int, np.ndarray]
    mask_by_camera: dict[int, np.ndarray]


class Demo31LiftInputCache:
    """Bounded main-process cache for group-aligned 2D-to-world lift inputs."""

    def __init__(self, *, max_groups: int = DEFAULT_LIFT_INPUT_CACHE_GROUPS) -> None:
        self.max_groups = int(max_groups)
        self._snapshots: dict[int, Demo31LiftInputSnapshot] = {}
        self.published = 0
        self.evicted = 0
        self.hit_count = 0
        self.miss_count = 0

    def publish(
        self,
        *,
        group_id: int,
        timestamp_s: float,
        depth_by_camera: dict[int, np.ndarray],
        intrinsics_by_camera: dict[int, np.ndarray],
        c2w_by_camera: dict[int, np.ndarray],
        mask_by_camera: dict[int, np.ndarray],
    ) -> None:
        self._snapshots[int(group_id)] = Demo31LiftInputSnapshot(
            group_id=int(group_id),
            timestamp_s=float(timestamp_s),
            depth_by_camera={
                int(camera_idx): np.ascontiguousarray(np.asarray(depth, dtype=np.float32)).copy()
                for camera_idx, depth in depth_by_camera.items()
            },
            intrinsics_by_camera={
                int(camera_idx): np.ascontiguousarray(np.asarray(intrinsics, dtype=np.float32).reshape(3, 3)).copy()
                for camera_idx, intrinsics in intrinsics_by_camera.items()
            },
            c2w_by_camera={
                int(camera_idx): np.ascontiguousarray(np.asarray(c2w, dtype=np.float32).reshape(4, 4)).copy()
                for camera_idx, c2w in c2w_by_camera.items()
            },
            mask_by_camera={
                int(camera_idx): np.ascontiguousarray(np.asarray(mask, dtype=bool)).copy()
                for camera_idx, mask in mask_by_camera.items()
            },
        )
        self.published += 1
        self._prune()

    def get(self, group_id: int) -> Demo31LiftInputSnapshot | None:
        snapshot = self._snapshots.get(int(group_id))
        if snapshot is None:
            self.miss_count += 1
            return None
        self.hit_count += 1
        return snapshot

    def snapshot(self) -> dict[str, Any]:
        return {
            "max_groups": int(self.max_groups),
            "cached_groups": int(len(self._snapshots)),
            "oldest_group_id": int(min(self._snapshots)) if self._snapshots else None,
            "newest_group_id": int(max(self._snapshots)) if self._snapshots else None,
            "published": int(self.published),
            "evicted": int(self.evicted),
            "hit_count": int(self.hit_count),
            "miss_count": int(self.miss_count),
        }

    def _prune(self) -> None:
        while len(self._snapshots) > max(1, int(self.max_groups)):
            oldest = min(self._snapshots)
            self._snapshots.pop(oldest, None)
            self.evicted += 1


def _normalize_mask_source(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized != demo3_runtime.MASK_SOURCE_HF_EDGETAM:
        raise ValueError("Demo 3.1 mask source must be hf-edgetam.")
    return demo3_runtime.MASK_SOURCE_HF_EDGETAM


def _physical_cuda_device_count_from_nvidia_smi() -> int:
    override = os.environ.get("QQTT_DEMO31_TEST_CUDA_COUNT")
    if override:
        return int(override)
    try:
        completed = subprocess.run(
            ["nvidia-smi", "-L"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except Exception:
        return 0
    if completed.returncode != 0:
        return 0
    return sum(1 for line in completed.stdout.splitlines() if line.strip().startswith("GPU "))


def _cuda_count(provider: CudaDeviceCountProvider | None = None) -> int:
    if provider is not None:
        return int(provider())
    return _physical_cuda_device_count_from_nvidia_smi()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Demo 3.1 dual-4090 realtime visualization: GPU0 owns RealSense, "
            "HF EdgeTAM masks, RealSense-depth fusion, and render; GPU1 owns "
            "CoTracker3 online in an isolated latest-wins process."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--preset", choices=PRESETS, default=PRESET_DEMO31_DUAL4090_HIGHFPS)
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved Demo 3.1 runtime contract and exit.")
    parser.add_argument("--duration-s", type=float, default=120.0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--profile-json-output", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--camera-ids", type=demo3_runtime.parse_camera_ids, default=demo3_runtime.DEFAULT_CAMERA_IDS)
    parser.add_argument("--serials", nargs="*", default=None)
    parser.add_argument("--calibrate-path", type=Path, default=Path("calibrate.pkl"))
    parser.add_argument("--width", type=int, default=demo3_runtime.DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=demo3_runtime.DEFAULT_HEIGHT)
    parser.add_argument("--fps", type=int, default=demo3_runtime.DEFAULT_FPS)
    parser.add_argument("--depth-source", default=demo3_runtime.DEPTH_SOURCE_REALSENSE)
    parser.add_argument("--mask-source", default=demo3_runtime.MASK_SOURCE_HF_EDGETAM_CLI)
    parser.add_argument(
        "--edgetam-live-session-keep-frames",
        type=int,
        default=demo3_runtime.DEFAULT_EDGETAM_LIVE_SESSION_KEEP_FRAMES,
        help=(
            "Maximum recent HF EdgeTAM live-session frames kept per camera in "
            "the shared live runtime. This bounds long-run GPU memory growth."
        ),
    )
    parser.add_argument("--mode", choices=demo3_runtime.MODES, default=demo3_runtime.DEFAULT_MODE)
    parser.add_argument("--object-prompt", default=demo3_runtime.DEFAULT_OBJECT_PROMPT)
    parser.add_argument("--cotracker-backend", default=demo3_runtime.COTRACKER3_ONLINE)
    parser.add_argument("--cotracker-query-mode", choices=(demo3_runtime.TRACKING_QUERY_MODE_PHYSTWIN_DENSE,), default=demo3_runtime.TRACKING_QUERY_MODE_PHYSTWIN_DENSE)
    parser.add_argument("--cotracker-query-count", default=demo3_runtime.DEFAULT_COTRACKER_QUERY_COUNT_REQUEST)
    parser.add_argument("--cotracker-seed", type=int, default=demo3_runtime.DEFAULT_COTRACKER_SEED)
    parser.add_argument("--disable-cotracker", action="store_true")
    parser.add_argument("--render-mode", choices=demo3_runtime.RENDER_MODES, default=demo3_runtime.RENDER_MODE_POINTCLOUD)
    parser.add_argument("--point-size", type=float, default=None)
    parser.add_argument("--render-every-n", type=int, default=None)
    parser.add_argument("--render-backend", default=None)
    parser.add_argument("--render-layer-mode", default=None)
    parser.add_argument("--render-copy-mode", default=None)
    parser.add_argument("--no-render-async-latest-only", action="store_true")
    parser.add_argument("--render-micro-profile", action="store_true")
    parser.add_argument(
        "--object-point-control",
        choices=demo3_runtime.OBJECT_POINT_CONTROLS,
        default=demo3_runtime.OBJECT_POINT_CONTROL_PHYSTWIN_VOLUME,
    )
    parser.add_argument(
        "--object-volume-voxel-m",
        type=float,
        default=demo3_runtime.DEFAULT_PHYSTWIN_OBJECT_VOLUME_VOXEL_M,
    )
    parser.add_argument(
        "--object-volume-origin",
        choices=demo3_runtime.PHYSTWIN_VOLUME_ORIGINS,
        default=demo3_runtime.PHYSTWIN_VOLUME_ORIGIN_WORLD,
    )
    parser.add_argument("--object-volume-adaptive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--object-volume-min-voxel-m",
        type=float,
        default=demo3_runtime.DEFAULT_PHYSTWIN_OBJECT_VOLUME_MIN_VOXEL_M,
    )
    parser.add_argument(
        "--object-volume-max-voxel-m",
        type=float,
        default=demo3_runtime.DEFAULT_PHYSTWIN_OBJECT_VOLUME_MAX_VOXEL_M,
    )
    parser.add_argument(
        "--object-volume-target-ms",
        type=float,
        default=demo3_runtime.DEFAULT_PHYSTWIN_OBJECT_VOLUME_TARGET_MS,
    )
    parser.add_argument(
        "--object-volume-emergency-max-points",
        type=int,
        default=demo3_runtime.DEFAULT_PHYSTWIN_OBJECT_VOLUME_EMERGENCY_MAX_POINTS,
    )
    parser.add_argument(
        "--object-volume-points-per-voxel",
        type=int,
        default=demo3_runtime.DEFAULT_PHYSTWIN_OBJECT_VOLUME_POINTS_PER_VOXEL,
    )
    parser.add_argument("--debug-color-by-camera", action="store_true")
    parser.add_argument("--debug-save-per-camera-pcd", action="store_true")
    parser.add_argument("--debug-save-mask-overlays", action="store_true")
    parser.add_argument("--debug-identity-c2w", action="store_true")
    parser.add_argument("--debug-invert-c2w", action="store_true")
    parser.add_argument("--debug-only-camera-idx", type=int, choices=demo3_runtime.DEFAULT_CAMERA_IDS, default=None)
    parser.add_argument("--debug-fusion-max-saved-groups", type=int, default=None)
    parser.add_argument("--gpu-sampling", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gpu-sampling-interval-s", type=float, default=0.5)
    parser.add_argument("--gpu-sampling-backend", choices=demo3_runtime.GPU_SAMPLING_BACKENDS, default="nvml")
    parser.add_argument("--gpu-sampling-device-index", type=int, default=0)
    parser.add_argument("--gpu-sampling-device-indexes", type=demo3_runtime.parse_gpu_sampling_device_indexes, default=None)
    parser.add_argument("--overlay-max-points-per-camera", type=int, default=demo3_runtime.DEFAULT_OVERLAY_MAX_POINTS_PER_CAMERA)
    parser.add_argument("--overlay-trail-len", type=int, default=demo3_runtime.DEFAULT_OVERLAY_TRAIL_LEN)
    parser.add_argument("--overlay-stale-timeout-ms", type=float, default=demo3_runtime.DEFAULT_OVERLAY_STALE_TIMEOUT_MS)
    parser.add_argument("--mask-gpu", default=DEFAULT_MASK_GPU)
    parser.add_argument("--cotracker-gpu", default=DEFAULT_COTRACKER_GPU)
    parser.add_argument("--require-two-cuda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-single-gpu-debug", action="store_true")
    parser.add_argument("--gpu-plan", choices=GPU_PLANS, default=GPU_PLAN_SPLIT_MASK0_TRACK1)
    parser.add_argument("--cotracker-process-mode", choices=PROCESS_MODES, default=PROCESS_MODE_SUBPROCESS)
    parser.add_argument("--cotracker-input-fps", type=float, default=DEFAULT_COTRACKER_INPUT_FPS)
    parser.add_argument("--cotracker-input-max-age-ms", type=float, default=DEFAULT_COTRACKER_INPUT_MAX_AGE_MS)
    parser.add_argument("--cotracker-result-stale-timeout-ms", type=float, default=DEFAULT_COTRACKER_RESULT_STALE_TIMEOUT_MS)
    parser.add_argument("--fusion-mask-policy", choices=FUSION_MASK_POLICIES, default=FUSION_MASK_POLICY_LATEST_REUSE)
    parser.add_argument("--mask-stale-timeout-ms", type=float, default=DEFAULT_MASK_STALE_TIMEOUT_MS)
    parser.add_argument("--render-target-fps", type=float, default=DEFAULT_RENDER_TARGET_FPS)
    parser.add_argument("--render-resample-latest", action=argparse.BooleanOptionalAction, default=True)
    return parser


def apply_preset_defaults(args: argparse.Namespace, *, explicit_options: set[str] | None = None) -> argparse.Namespace:
    explicit = explicit_options or set()
    if args.preset == PRESET_DEMO31_DUAL4090_HIGHFPS:
        if "--fusion-mask-policy" not in explicit:
            args.fusion_mask_policy = FUSION_MASK_POLICY_LATEST_REUSE
        if "--cotracker-input-fps" not in explicit:
            args.cotracker_input_fps = DEFAULT_COTRACKER_INPUT_FPS
        if "--render-target-fps" not in explicit:
            args.render_target_fps = DEFAULT_RENDER_TARGET_FPS
    return args


def validate_args(
    args: argparse.Namespace,
    *,
    require_calibration: bool = False,
    cuda_device_count_provider: CudaDeviceCountProvider | None = None,
) -> None:
    camera_ids = demo3_runtime.parse_camera_ids(args.camera_ids)
    if len(camera_ids) != 3:
        raise ValueError("Demo 3.1 requires exactly three RealSense cameras.")
    if len(set(camera_ids)) != 3:
        raise ValueError("Demo 3.1 requires exactly three distinct RealSense cameras.")
    depth_source = str(args.depth_source).strip().lower()
    if depth_source == demo3_runtime.DEPTH_SOURCE_FFS or depth_source.startswith("ffs"):
        raise ValueError("Demo 3.1 does not support FFS. Use --depth-source realsense.")
    if depth_source != demo3_runtime.DEPTH_SOURCE_REALSENSE:
        raise ValueError("Demo 3.1 depth source must be realsense.")
    _normalize_mask_source(str(args.mask_source))
    if str(args.cotracker_backend) != demo3_runtime.COTRACKER3_ONLINE:
        raise ValueError("Demo 3.1 currently supports only --cotracker-backend cotracker3_online.")
    if str(args.cotracker_query_mode) != demo3_runtime.TRACKING_QUERY_MODE_PHYSTWIN_DENSE:
        raise ValueError("Demo 3.1 currently supports only --cotracker-query-mode phystwin_dense.")
    demo3_runtime.normalize_cotracker_query_count_request(args.cotracker_query_count)
    if str(args.object_point_control) not in demo3_runtime.OBJECT_POINT_CONTROLS:
        raise ValueError(f"Demo 3.1 unsupported --object-point-control {args.object_point_control}")
    if str(args.object_volume_origin) not in demo3_runtime.PHYSTWIN_VOLUME_ORIGINS:
        raise ValueError(f"Demo 3.1 unsupported --object-volume-origin {args.object_volume_origin}")
    if float(args.object_volume_voxel_m) <= 0.0:
        raise ValueError("--object-volume-voxel-m must be positive.")
    if float(args.object_volume_min_voxel_m) <= 0.0 or float(args.object_volume_max_voxel_m) <= 0.0:
        raise ValueError("--object-volume-min-voxel-m and --object-volume-max-voxel-m must be positive.")
    if float(args.object_volume_min_voxel_m) > float(args.object_volume_max_voxel_m):
        raise ValueError("--object-volume-min-voxel-m must be <= --object-volume-max-voxel-m.")
    if float(args.object_volume_target_ms) <= 0.0:
        raise ValueError("--object-volume-target-ms must be > 0.")
    if int(args.object_volume_emergency_max_points) < 0:
        raise ValueError("--object-volume-emergency-max-points must be >= 0.")
    if int(args.object_volume_points_per_voxel) < 1:
        raise ValueError("--object-volume-points-per-voxel must be >= 1.")
    if int(args.edgetam_live_session_keep_frames) < 1:
        raise ValueError("--edgetam-live-session-keep-frames must be >= 1.")
    if bool(args.debug_identity_c2w) and bool(args.debug_invert_c2w):
        raise ValueError("Demo 3.1 accepts only one of --debug-identity-c2w or --debug-invert-c2w.")
    if args.debug_only_camera_idx is not None and int(args.debug_only_camera_idx) not in set(camera_ids):
        raise ValueError(f"--debug-only-camera-idx {args.debug_only_camera_idx} is not in --camera-ids {camera_ids}.")
    if int(args.gpu_sampling_device_index) < 0:
        raise ValueError("--gpu-sampling-device-index must be >= 0.")
    if args.gpu_sampling_device_indexes is not None and any(int(index) < 0 for index in args.gpu_sampling_device_indexes):
        raise ValueError("--gpu-sampling-device-indexes must be >= 0.")
    if float(args.gpu_sampling_interval_s) <= 0.0:
        raise ValueError("--gpu-sampling-interval-s must be > 0.")
    if int(args.overlay_max_points_per_camera) <= 0:
        raise ValueError("--overlay-max-points-per-camera must be positive.")
    if float(args.cotracker_input_fps) < 0.0:
        raise ValueError("--cotracker-input-fps must be non-negative.")
    if str(args.mask_gpu) == str(args.cotracker_gpu) and not bool(args.allow_single_gpu_debug):
        raise ValueError("Demo 3.1 requires distinct --mask-gpu and --cotracker-gpu unless --allow-single-gpu-debug is passed.")
    if bool(args.require_two_cuda) and not bool(args.allow_single_gpu_debug):
        count = _cuda_count(cuda_device_count_provider)
        if count < 2:
            raise RuntimeError(f"Demo 3.1 requires at least two CUDA devices before process isolation; found {count}.")
    if require_calibration and not Path(args.calibrate_path).is_file():
        raise FileNotFoundError(f"Demo 3.1 requires calibrate.pkl for three-camera world fusion: {args.calibrate_path}")


def build_cotracker_process_config(args: argparse.Namespace) -> CoTrackerProcessConfig:
    return CoTrackerProcessConfig(
        camera_ids=demo3_runtime.parse_camera_ids(args.camera_ids),
        cotracker_gpu=str(args.cotracker_gpu),
        cotracker_backend=str(args.cotracker_backend),
        query_mode=str(args.cotracker_query_mode),
        query_count_request=demo3_runtime.normalize_cotracker_query_count_request(args.cotracker_query_count),
        seed=int(args.cotracker_seed),
        sampling_device="cuda",
        init_requires_object_and_controller=True,
        overlay_max_points_per_camera=int(args.overlay_max_points_per_camera),
        input_max_age_ms=float(args.cotracker_input_max_age_ms),
        process_mode=str(args.cotracker_process_mode),
        device="cuda",
    )


def build_contract(
    args: argparse.Namespace,
    *,
    cuda_device_count_provider: CudaDeviceCountProvider | None = None,
) -> dict[str, Any]:
    camera_ids = demo3_runtime.parse_camera_ids(args.camera_ids)
    render_waited_for_mask = str(args.fusion_mask_policy) == FUSION_MASK_POLICY_STRICT
    mode = demo3_runtime.resolve_demo3_mode(str(args.mode))
    query_count_request = demo3_runtime.normalize_cotracker_query_count_request(args.cotracker_query_count)
    contract: dict[str, Any] = {
        "demo": "demo3.1",
        "preset": str(args.preset),
        "input_source": "live_realsense",
        "offline_mode_available": False,
        "offline_tracking_available": False,
        "init_mode": "sam31_first_frame",
        "mask_propagation": "hf_edgetam_online",
        "dual_gpu_enabled": True,
        "required_cuda_devices": 2,
        "physical_cuda_device_count": int(_cuda_count(cuda_device_count_provider)),
        "requires_three_realsense": True,
        "num_cameras": int(len(camera_ids)),
        "num_realsense_cameras": int(len(camera_ids)),
        "camera_ids": list(camera_ids),
        "serials": list(args.serials or []),
        "calibrate_path": str(args.calibrate_path),
        "calibrate_pkl_loaded": bool(Path(args.calibrate_path).is_file()),
        "mask_gpu_physical": int(args.mask_gpu),
        "cotracker_gpu_physical": int(args.cotracker_gpu),
        "main_cuda_visible_devices": str(args.mask_gpu),
        "cotracker_cuda_visible_devices": str(args.cotracker_gpu),
        "gpu_plan": str(args.gpu_plan),
        "depth_source": demo3_runtime.DEPTH_SOURCE_REALSENSE,
        "uses_ffs": False,
        "mask_source": demo3_runtime.MASK_SOURCE_HF_EDGETAM,
        "edgetam_batch_vision_encoder": True,
        "edgetam_live_session_keep_frames": int(args.edgetam_live_session_keep_frames),
        "edgetam_live_session_pruning": True,
        "semantic_mode": str(mode["semantic_mode"]),
        "shared_experiment_mode": str(mode["experiment_mode"]),
        "shared_runtime_track_mode": demo3_runtime.SHARED_TRACK_MODE_CONTROLLER_OBJECT,
        "tracking_mask_scope": demo3_runtime.TRACK_SCOPE_OBJECT_CONTROLLER_UNION,
        "object_prompt": str(args.object_prompt),
        "controller_prompt": str(mode["controller_prompt"]),
        "tracking_controller_label": str(mode["controller_label"]),
        "cotracker_enabled": not bool(args.disable_cotracker),
        "cotracker_backend": demo3_runtime.COTRACKER3_ONLINE,
        "cotracker_owner": "process",
        "cotracker_process_mode": str(args.cotracker_process_mode),
        "cotracker_input_fps": float(args.cotracker_input_fps),
        "cotracker_input_max_age_ms": float(args.cotracker_input_max_age_ms),
        "cotracker_result_stale_timeout_ms": float(args.cotracker_result_stale_timeout_ms),
        "tracking_query_mode": demo3_runtime.TRACKING_QUERY_MODE_PHYSTWIN_DENSE,
        "tracking_query_count_requested": str(query_count_request),
        "tracking_query_count_rule": demo3_runtime.TRACKING_QUERY_COUNT_RULE_PHYSTWIN_DENSE,
        "tracking_sampling": demo3_runtime.TRACKING_SAMPLING_TORCH_RANDPERM,
        "tracking_max_query_points_per_camera": demo3_runtime.PHYSTWIN_DENSE_MAX_POINTS,
        "cotracker_seed": int(args.cotracker_seed),
        "phystwin_dense_compatible": bool(demo3_runtime.phystwin_dense_compatible_for_args(args)),
        "cotracker_window_len": demo3_runtime.DEFAULT_COTRACKER_WINDOW_LEN,
        "cotracker_publish_step": demo3_runtime.DEFAULT_COTRACKER_PUBLISH_STEP,
        "ipc_payload": "cpu_numpy_latest_wins",
        "tracking_input_contains_depth": False,
        "tracking_input_contains_intrinsics": False,
        "tracking_input_contains_c2w": False,
        "world_lift_owner": "main_process",
        "cross_gpu_cuda_tensor_transfer": False,
        "shared_runtime_tracking_backend": "none",
        "overlay_max_points_per_camera": int(args.overlay_max_points_per_camera),
        "overlay_trail_len": int(args.overlay_trail_len),
        "overlay_stale_timeout_ms": float(args.overlay_stale_timeout_ms),
        "fusion_mask_policy": str(args.fusion_mask_policy),
        "mask_stale_timeout_ms": float(args.mask_stale_timeout_ms),
        "render_mode": str(args.render_mode),
        "render_target_fps": float(args.render_target_fps),
        "render_resample_latest": bool(args.render_resample_latest),
        "render_backend": None if args.render_backend is None else str(args.render_backend),
        "render_layer_mode": None if args.render_layer_mode is None else str(args.render_layer_mode),
        "render_copy_mode": None if args.render_copy_mode is None else str(args.render_copy_mode),
        "render_micro_profile": True,
        "render_latest_wins": True,
        "render_waited_for_cotracker": False,
        "render_waited_for_mask": bool(render_waited_for_mask),
        "render_object_filter": {
            "point_control": str(args.object_point_control),
            "voxel_m": float(args.object_volume_voxel_m),
            "origin_policy": str(args.object_volume_origin),
            "adaptive": bool(args.object_volume_adaptive),
            "min_voxel_m": float(args.object_volume_min_voxel_m),
            "max_voxel_m": float(args.object_volume_max_voxel_m),
            "target_ms": float(args.object_volume_target_ms),
            "emergency_max_points": int(args.object_volume_emergency_max_points),
            "points_per_voxel": int(args.object_volume_points_per_voxel),
        },
        "debug_fusion": {
            "color_by_camera": bool(args.debug_color_by_camera),
            "save_per_camera_pcd": bool(args.debug_save_per_camera_pcd),
            "save_mask_overlays": bool(args.debug_save_mask_overlays),
            "identity_c2w": bool(args.debug_identity_c2w),
            "invert_c2w": bool(args.debug_invert_c2w),
            "only_camera_idx": None if args.debug_only_camera_idx is None else int(args.debug_only_camera_idx),
            "max_saved_groups": (
                None if args.debug_fusion_max_saved_groups is None else int(args.debug_fusion_max_saved_groups)
            ),
        },
        "gpu_sampling": {
            "enabled": bool(args.gpu_sampling),
            "interval_s": float(args.gpu_sampling_interval_s),
            "backend": str(args.gpu_sampling_backend),
            "device_index": int(args.gpu_sampling_device_index),
            "device_indexes": (
                list(demo3_runtime._gpu_sampling_device_indexes_for_args(args))
                if demo3_runtime._gpu_sampling_device_indexes_for_args(args) is not None
                else None
            ),
        },
        "width": int(args.width),
        "height": int(args.height),
        "fps": int(args.fps),
        "output_root": str(args.output_root),
        "hot_path_forbids": [
            "ffs",
            "ffs_tensorrt",
            "ffs_remote",
            "ffs_ir_alignment",
            "track_process_data.pkl",
            "inverse_physics",
            "cross_gpu_cuda_tensor_transfer",
        ],
    }
    contract["profile_summary_fields"] = build_empty_dual_gpu_profile_summary(contract)
    return contract


def format_contract(contract: dict[str, Any]) -> str:
    keys = (
        "demo",
        "input_source",
        "offline_mode_available",
        "dual_gpu_enabled",
        "required_cuda_devices",
        "mask_gpu_physical",
        "cotracker_gpu_physical",
        "main_cuda_visible_devices",
        "cotracker_cuda_visible_devices",
        "depth_source",
        "uses_ffs",
        "mask_source",
        "edgetam_batch_vision_encoder",
        "edgetam_live_session_keep_frames",
        "edgetam_live_session_pruning",
        "init_mode",
        "mask_propagation",
        "semantic_mode",
        "tracking_mask_scope",
        "tracking_query_mode",
        "tracking_query_count_requested",
        "tracking_query_count_rule",
        "tracking_sampling",
        "cotracker_seed",
        "overlay_max_points_per_camera",
        "phystwin_dense_compatible",
        "cotracker_backend",
        "cotracker_owner",
        "cotracker_process_mode",
        "cross_gpu_cuda_tensor_transfer",
        "ipc_payload",
        "fusion_mask_policy",
        "render_waited_for_cotracker",
        "render_waited_for_mask",
    )
    lines = []
    for key in keys:
        value = contract[key]
        rendered = str(value).lower() if isinstance(value, bool) else str(value)
        lines.append(f"{key} = {rendered}")
    lines.append(json.dumps(contract, indent=2, sort_keys=True))
    return "\n".join(lines)


def _write_profile(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fresh_tracking_result_or_none(
    result: TrackingResultLitePacket | None,
    *,
    now_s: float,
    stale_timeout_ms: float,
) -> TrackingResultLitePacket | None:
    if result is None:
        return None
    age_ms = max(0.0, (float(now_s) - float(result.publish_timestamp_s)) * 1000.0)
    if age_ms > float(stale_timeout_ms):
        return None
    return result


def build_shared_runtime_args(
    args: argparse.Namespace,
    *,
    shared_runtime_module: Any | None,
    live_validation: dict[str, Any],
    shared_profile_path: Path | None,
) -> argparse.Namespace:
    shared = shared_runtime_module or demo3_runtime._load_shared_runtime_module()
    shared_args = demo3_runtime.build_shared_runtime_args(
        args,
        shared_runtime_module=shared,
        live_validation=live_validation,
        shared_profile_path=shared_profile_path,
    )
    shared_args.tracking_backend = "none"
    shared_args.tracking_source = "cached"
    shared_args.show_tracking_overlay = False
    shared_args.depth_source = demo3_runtime.DEPTH_SOURCE_REALSENSE
    shared_args.edgetam_batch_vision_encoder = True
    if hasattr(shared_args, "render_target_fps"):
        shared_args.render_target_fps = float(args.render_target_fps)
    return shared_args


def _phystwin_union_tracking_masks(mask_packet: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    object_mask = np.asarray(mask_packet.object_mask, dtype=bool)
    controller_mask = np.asarray(mask_packet.controller_mask, dtype=bool)
    union_mask = np.asarray(object_mask | controller_mask, dtype=bool)
    return union_mask, object_mask, controller_mask


class Demo31MaskPolicyJoinBuffer:
    """Join capture/depth with strict or latest-reuse mask semantics."""

    def __init__(
        self,
        *,
        max_groups: int = 8,
        policy: str = FUSION_MASK_POLICY_LATEST_REUSE,
        stale_timeout_ms: float = DEFAULT_MASK_STALE_TIMEOUT_MS,
    ) -> None:
        self.max_groups = int(max_groups)
        self.policy = str(policy)
        self.stale_timeout_ms = float(stale_timeout_ms)
        self._captures: dict[int, Any] = {}
        self._depths: dict[int, Any] = {}
        self._masks: dict[int, tuple[Any, float]] = {}
        self.capture_stale_drops = 0
        self.depth_stale_drops = 0
        self.mask_stale_drops = 0
        self.ready_join_count = 0
        self.mask_selection_count = 0
        self.mask_reuse_count = 0
        self.mask_age_ms_samples: list[float] = []

    def put_capture(self, group: Any) -> None:
        self._captures[int(group.group_id)] = group
        self._prune()

    def put_depth(self, depth: Any) -> None:
        self._depths[int(depth.group_id)] = depth
        self._prune()

    def put_mask(self, mask: Any) -> None:
        self._masks[int(mask.group_id)] = (mask, time.perf_counter())
        self._prune()

    def pop_latest_ready(self) -> tuple[Any, Any, Any] | None:
        ready_depth = sorted(set(self._captures) & set(self._depths))
        if not ready_depth:
            return None
        now_s = time.perf_counter()
        for group_id in reversed(ready_depth):
            mask = self._select_mask_for_group(group_id=group_id, now_s=now_s)
            if mask is None:
                continue
            capture = self._captures.pop(group_id)
            depth = self._depths.pop(group_id)
            self.ready_join_count += 1
            self._drop_older_capture_depth(group_id)
            return capture, depth, mask
        return None

    def snapshot(self) -> dict[str, Any]:
        age = percentile_summary(self.mask_age_ms_samples)
        return {
            "max_groups": int(self.max_groups),
            "policy": str(self.policy),
            "capture_pending": int(len(self._captures)),
            "depth_pending": int(len(self._depths)),
            "mask_pending": int(len(self._masks)),
            "capture_stale_drops": int(self.capture_stale_drops),
            "depth_stale_drops": int(self.depth_stale_drops),
            "mask_stale_drops": int(self.mask_stale_drops),
            "ready_join_count": int(self.ready_join_count),
            "mask_selection_count": int(self.mask_selection_count),
            "mask_reuse_count": int(self.mask_reuse_count),
            "mask_reuse_ratio": float(self.mask_reuse_count / self.mask_selection_count)
            if self.mask_selection_count
            else 0.0,
            "mask_age_ms_median": float(age["median"]),
            "mask_age_ms_p95": float(age["p95"]),
        }

    def _select_mask_for_group(self, *, group_id: int, now_s: float) -> Any | None:
        if not self._masks:
            return None
        if self.policy == FUSION_MASK_POLICY_STRICT:
            entry = self._masks.get(int(group_id))
            if entry is None:
                return None
            mask_group, arrival_s = entry
        else:
            source_group_id = max(self._masks)
            mask_group, arrival_s = self._masks[source_group_id]
        age_ms = max(0.0, (float(now_s) - float(arrival_s)) * 1000.0)
        if age_ms > self.stale_timeout_ms:
            self.mask_stale_drops += 1
            return None
        self.mask_selection_count += 1
        self.mask_age_ms_samples.append(float(age_ms))
        if int(mask_group.group_id) != int(group_id):
            self.mask_reuse_count += 1
        return self._retarget_mask_group(mask_group, target_group_id=int(group_id))

    def _retarget_mask_group(self, mask_group: Any, *, target_group_id: int) -> Any:
        if int(mask_group.group_id) == int(target_group_id):
            return mask_group
        packets = {
            int(camera_idx): replace(packet, group_id=int(target_group_id))
            for camera_idx, packet in mask_group.mask_packets.items()
        }
        return mask_group.__class__(
            group_id=int(target_group_id),
            mask_packets=packets,
            edgetam_stage_wall_ms=float(mask_group.edgetam_stage_wall_ms),
            edgetam_stage_sum_model_ms=float(mask_group.edgetam_stage_sum_model_ms),
            edgetam_stage_mode=str(mask_group.edgetam_stage_mode),
        )

    def _drop_older_capture_depth(self, group_id: int) -> None:
        for table, counter_name in (
            (self._captures, "capture_stale_drops"),
            (self._depths, "depth_stale_drops"),
        ):
            stale = [old_group_id for old_group_id in table if old_group_id < group_id]
            for old_group_id in stale:
                table.pop(old_group_id, None)
            setattr(self, counter_name, getattr(self, counter_name) + len(stale))

    def _prune(self) -> None:
        for table, counter_name in (
            (self._captures, "capture_stale_drops"),
            (self._depths, "depth_stale_drops"),
            (self._masks, "mask_stale_drops"),
        ):
            while len(table) > self.max_groups:
                oldest = min(table)
                table.pop(oldest, None)
                setattr(self, counter_name, getattr(self, counter_name) + 1)


def make_demo31_live_runtime_class(shared_runtime_module: Any, *, process_client_factory: ProcessClientFactory | None = None):
    base_cls = shared_runtime_module.Demo21Runtime

    class Demo31LiveRuntime(base_cls):
        def __init__(
            self,
            args: argparse.Namespace,
            *,
            demo31_contract: dict[str, Any],
            cotracker_process_config: CoTrackerProcessConfig,
            cotracker_enabled: bool = True,
        ) -> None:
            super().__init__(args)
            self.demo31_contract = dict(demo31_contract)
            self.demo31_cotracker_enabled = bool(cotracker_enabled)
            self.demo31_cotracker_config = cotracker_process_config
            self.demo31_process_client = (
                (process_client_factory or start_cotracker_process)(cotracker_process_config)
            if self.demo31_cotracker_enabled
            else None
            )
            self.stage_join_buffer = Demo31MaskPolicyJoinBuffer(
                max_groups=8,
                policy=str(self.demo31_contract["fusion_mask_policy"]),
                stale_timeout_ms=float(self.demo31_contract["mask_stale_timeout_ms"]),
            )
            self.demo31_lift_input_cache = Demo31LiftInputCache()
            self.demo31_mask_cache = LatestMaskCache()
            self.demo31_last_tracking_input_s: float | None = None
            self.demo31_tracking_input_skip_count = 0
            self.demo31_tracking_input_queue_replace_count = 0
            self.demo31_tracking_input_drop_count = 0
            self.demo31_overlay_age_ms_samples: list[float] = []
            self.demo31_overlay_model_ms_samples: list[float] = []
            self.demo31_overlay_e2e_ms_samples: list[float] = []
            self.demo31_tracking_stats: dict[str, dict[int, int]] = {}

        def stop(self) -> None:
            if self.demo31_process_client is not None:
                self.demo31_process_client.stop(timeout_s=2.0)
            super().stop()

        def _build_fused_packet(self, *, depth_group: Any, masks: dict[int, Any], ray_cache: dict[int, Any], rng: np.random.Generator):
            now_s = time.perf_counter()
            rgb_by_camera: dict[int, np.ndarray] = {}
            mask_by_camera: dict[int, np.ndarray] = {}
            object_mask_by_camera: dict[int, np.ndarray] = {}
            controller_mask_by_camera: dict[int, np.ndarray] = {}
            depth_by_camera: dict[int, np.ndarray] = {}
            intrinsics_by_camera: dict[int, np.ndarray] = {}
            c2w_by_camera: dict[int, np.ndarray] = {}
            for camera_idx in self.args.camera_ids:
                idx = int(camera_idx)
                if idx not in masks or idx not in depth_group.depths:
                    continue
                mask_packet = masks[idx]
                rgb_by_camera[idx] = np.ascontiguousarray(np.asarray(mask_packet.color_bgr)[..., ::-1])
                union_mask, object_mask, controller_mask = _phystwin_union_tracking_masks(mask_packet)
                mask_by_camera[idx] = union_mask
                object_mask_by_camera[idx] = object_mask
                controller_mask_by_camera[idx] = controller_mask
                depth_by_camera[idx] = np.asarray(depth_group.depths[idx].depth_m, dtype=np.float32)
                if getattr(self, "_stream_metadata", None) and idx < len(self._stream_metadata):
                    intrinsics_by_camera[idx] = np.asarray(
                        self._stream_metadata[idx]["K_color"],
                        dtype=np.float32,
                    ).reshape(3, 3)
                if idx in getattr(self, "_c2w_by_camera", {}):
                    c2w_by_camera[idx] = np.asarray(self._c2w_by_camera[idx], dtype=np.float32).reshape(4, 4)
            if mask_by_camera:
                self.demo31_mask_cache.publish(
                    group_id=int(depth_group.group_id),
                    timestamp_s=now_s,
                    mask_by_camera=mask_by_camera,
                )
            if self.demo31_process_client is not None and rgb_by_camera and mask_by_camera:
                if should_publish_tracking_input(
                    now_s=now_s,
                    last_publish_s=self.demo31_last_tracking_input_s,
                    target_fps=float(self.demo31_contract["cotracker_input_fps"]),
                ):
                    frame_idx = int(max(depth_group.per_camera_frame_seq.values()) if depth_group.per_camera_frame_seq else depth_group.group_id)
                    self.demo31_lift_input_cache.publish(
                        group_id=int(depth_group.group_id),
                        timestamp_s=now_s,
                        depth_by_camera=depth_by_camera,
                        intrinsics_by_camera=intrinsics_by_camera,
                        c2w_by_camera=c2w_by_camera,
                        mask_by_camera=mask_by_camera,
                    )
                    replaced_count = self.demo31_process_client.publish_input(
                        TrackingInputLitePacket(
                            group_id=int(depth_group.group_id),
                            frame_idx=frame_idx,
                            timestamp_s=now_s,
                            rgb_by_camera=rgb_by_camera,
                            mask_by_camera=mask_by_camera,
                            object_mask_by_camera=object_mask_by_camera,
                            controller_mask_by_camera=controller_mask_by_camera,
                        )
                    )
                    self.demo31_tracking_input_queue_replace_count += int(replaced_count)
                    self.demo31_last_tracking_input_s = now_s
                else:
                    self.demo31_tracking_input_skip_count += 1
            return super()._build_fused_packet(depth_group=depth_group, masks=masks, ray_cache=ray_cache, rng=rng)

        def _publish_render_packet(self, packet: Any) -> None:
            overlay_start_s = time.perf_counter()
            overlay = self._take_fresh_tracking_result(now_s=overlay_start_s)
            overlay_points = np.empty((0, 3), dtype=np.float32)
            overlay_lift_cache_hit = False
            overlay_group_id: int | None = None
            if overlay is not None:
                overlay_group_id = int(overlay.group_id)
                lift_inputs = self.demo31_lift_input_cache.get(overlay_group_id)
                if lift_inputs is not None:
                    overlay_lift_cache_hit = True
                    lifted_points = []
                    for camera_idx, tracks_yx in overlay.camera_tracks_yx.items():
                        idx = int(camera_idx)
                        if (
                            idx not in lift_inputs.depth_by_camera
                            or idx not in lift_inputs.intrinsics_by_camera
                            or idx not in lift_inputs.c2w_by_camera
                        ):
                            continue
                        lifted = lift_tracks_yx_to_world(
                            tracks_yx=tracks_yx,
                            visibility=overlay.camera_visibility[idx],
                            depth=lift_inputs.depth_by_camera[idx],
                            intrinsics=lift_inputs.intrinsics_by_camera[idx],
                            c2w=lift_inputs.c2w_by_camera[idx],
                            depth_scale_m_per_unit=1.0,
                            mask=lift_inputs.mask_by_camera.get(idx),
                        )
                        if lifted.points_world.size:
                            lifted_points.append(lifted.points_world)
                    if lifted_points:
                        overlay_points = np.concatenate(lifted_points, axis=0).astype(np.float32)
                        overlay_colors = np.repeat(demo3_runtime.OVERLAY_COLOR_RGB[None, :], len(overlay_points), axis=0)
                        packet = replace(
                            packet,
                            controller_points_m=np.concatenate([packet.controller_points_m, overlay_points], axis=0),
                            controller_colors_rgb=np.concatenate([packet.controller_colors_rgb, overlay_colors], axis=0),
                        )
            overlay_ms = float((time.perf_counter() - overlay_start_s) * 1000.0)
            self._profile_update(
                packet.group_id,
                demo31_tracking_overlay={
                    "overlay_available": bool(overlay is not None),
                    "overlay_points": int(len(overlay_points)),
                    "overlay_ms": overlay_ms,
                    "overlay_group_id": overlay_group_id,
                    "render_group_id": int(packet.group_id),
                    "overlay_lift_cache_hit": bool(overlay_lift_cache_hit),
                    "render_waited_for_cotracker": False,
                    "cross_gpu_cuda_tensor_transfer": False,
                },
            )
            super()._publish_render_packet(packet)

        def _take_fresh_tracking_result(self, *, now_s: float) -> TrackingResultLitePacket | None:
            if self.demo31_process_client is None:
                return None
            result = self.demo31_process_client.get_result()
            fresh = fresh_tracking_result_or_none(
                result,
                now_s=now_s,
                stale_timeout_ms=float(self.demo31_contract["cotracker_result_stale_timeout_ms"]),
            )
            if fresh is None:
                if result is not None:
                    self.demo31_tracking_input_drop_count += 1
                return None
            result = fresh
            age_ms = max(0.0, (now_s - float(result.publish_timestamp_s)) * 1000.0)
            self.demo31_overlay_age_ms_samples.append(float(age_ms))
            self.demo31_overlay_model_ms_samples.append(float(result.model_ms))
            self.demo31_overlay_e2e_ms_samples.append(float(result.e2e_ms))
            self.demo31_tracking_stats = {
                "tracking_query_count_actual_by_camera": dict(result.tracking_query_count_actual_by_camera),
                "tracking_union_pixels_by_camera": dict(result.tracking_union_pixels_by_camera),
                "tracking_object_pixels_by_camera": dict(result.tracking_object_pixels_by_camera),
                "tracking_controller_pixels_by_camera": dict(result.tracking_controller_pixels_by_camera),
                "tracking_sample_object_hits_by_camera": dict(result.tracking_sample_object_hits_by_camera),
                "tracking_sample_controller_hits_by_camera": dict(result.tracking_sample_controller_hits_by_camera),
                "tracking_sample_overlap_hits_by_camera": dict(result.tracking_sample_overlap_hits_by_camera),
                "tracking_sample_background_hits_by_camera": dict(result.tracking_sample_background_hits_by_camera),
                "overlay_display_count_by_camera": dict(result.overlay_display_count_by_camera),
            }
            return result

        def demo31_snapshot(self) -> dict[str, Any]:
            process_snapshot = (
                self.demo31_process_client.snapshot()
                if self.demo31_process_client is not None and hasattr(self.demo31_process_client, "snapshot")
                else None
            )
            age = percentile_summary(self.demo31_overlay_age_ms_samples)
            model = percentile_summary(self.demo31_overlay_model_ms_samples)
            e2e = percentile_summary(self.demo31_overlay_e2e_ms_samples)
            return {
                "process": process_snapshot,
                "stage_join_buffer": self.stage_join_buffer.snapshot()
                if hasattr(self.stage_join_buffer, "snapshot")
                else {},
                "tracking_input_skip_count": int(self.demo31_tracking_input_skip_count),
                "tracking_input_queue_replace_count": int(self.demo31_tracking_input_queue_replace_count),
                "tracking_input_drop_count": int(self.demo31_tracking_input_drop_count),
                "overlay_age_ms_median": float(age["median"]),
                "overlay_age_ms_p95": float(age["p95"]),
                "cotracker_model_ms_median": float(model["median"]),
                "cotracker_model_ms_p95": float(model["p95"]),
                "cotracker_e2e_ms_median": float(e2e["median"]),
                "cotracker_e2e_ms_p95": float(e2e["p95"]),
                "mask_cache": self.demo31_mask_cache.snapshot(),
                "lift_input_cache": self.demo31_lift_input_cache.snapshot(),
                "tracking_stats": dict(self.demo31_tracking_stats),
            }

    return Demo31LiveRuntime


class Demo31Runtime:
    def __init__(
        self,
        args: argparse.Namespace,
        *,
        shared_runtime_module: Any | None = None,
        shared_runtime_cls: type | None = None,
        connected_serials_provider: ConnectedSerialsProvider | None = None,
        cuda_device_count_provider: CudaDeviceCountProvider | None = None,
        process_client_factory: ProcessClientFactory | None = None,
    ) -> None:
        self.args = args
        self.cuda_device_count_provider = cuda_device_count_provider
        self.contract = build_contract(args, cuda_device_count_provider=cuda_device_count_provider)
        self.shared_runtime_module = shared_runtime_module
        self.shared_runtime_cls = shared_runtime_cls
        self.connected_serials_provider = connected_serials_provider
        self.process_client_factory = process_client_factory

    def run(self) -> dict[str, Any]:
        live_validation = demo3_runtime.validate_live_realsense_contract(
            self.args,
            connected_serials_provider=self.connected_serials_provider,
        )
        shared = self.shared_runtime_module or demo3_runtime._load_shared_runtime_module()
        shared_profile = demo3_runtime._shared_profile_path(self.args)
        shared_args = build_shared_runtime_args(
            self.args,
            shared_runtime_module=shared,
            live_validation=live_validation,
            shared_profile_path=shared_profile,
        )
        runtime_cls = self.shared_runtime_cls or make_demo31_live_runtime_class(
            shared,
            process_client_factory=self.process_client_factory,
        )
        if self.shared_runtime_cls is None:
            runtime = runtime_cls(
                shared_args,
                demo31_contract=self.contract,
                cotracker_process_config=build_cotracker_process_config(self.args),
                cotracker_enabled=not bool(self.args.disable_cotracker),
            )
        else:
            runtime = runtime_cls(shared_args)
        exit_code = int(runtime.run())
        shared_payload = demo3_runtime._load_json_if_exists(shared_profile)
        snapshot = runtime.demo31_snapshot() if hasattr(runtime, "demo31_snapshot") else None
        summary = self._build_summary(runtime=runtime, exit_code=exit_code, snapshot=snapshot, shared_payload=shared_payload)
        profile = {
            "contract": self.contract,
            "summary": summary,
            "live_validation": live_validation,
            "shared_runtime_profile": None if shared_profile is None else str(shared_profile),
            "shared_runtime_profile_payload": shared_payload,
            "cotracker_process_snapshot": snapshot,
            "runtime_note": "Demo 3.1 delegates capture/mask/fusion/render to the shared runtime and runs CoTracker3 in an isolated latest-wins process.",
            "exit_code": exit_code,
        }
        _write_profile(self.args.profile_json_output, profile)
        return profile

    def _build_summary(
        self,
        *,
        runtime: Any,
        exit_code: int,
        snapshot: dict[str, Any] | None,
        shared_payload: dict[str, Any] | None,
    ) -> dict[str, Any]:
        summary = build_empty_dual_gpu_profile_summary(self.contract)
        final = getattr(runtime, "_summary", {}).get("final", {}) if hasattr(runtime, "_summary") else {}
        warm = (shared_payload or {}).get("summary_after_warmup", {})
        gpu_by_device = (shared_payload or {}).get("gpu_sampling", {}).get("summary_by_device_after_warmup", {})

        def _gpu_metric(device_index: int, metric: str, stat: str) -> float:
            if not isinstance(gpu_by_device, dict):
                return 0.0
            device_summary = gpu_by_device.get(str(int(device_index)), {})
            if not isinstance(device_summary, dict):
                return 0.0
            value = demo3_runtime._nested_get(device_summary, ("metrics", metric, stat), 0.0)
            return float(value or 0.0)

        summary.update(
            {
                "exit_code": int(exit_code),
                "rendered_fps": float(final.get("render_fps", warm.get("render_fps", 0.0)) or 0.0),
                "render_loop_fps": float(final.get("render_fps", warm.get("render_fps", 0.0)) or 0.0),
                "new_fused_pcd_fps": float(final.get("fusion_fps", warm.get("fusion_fps", 0.0)) or 0.0),
                "capture_group_fps": float(final.get("capture_group_fps", warm.get("capture_group_fps", 0.0)) or 0.0),
                "gpu0_util_median": _gpu_metric(0, "gpu_util_pct", "median"),
                "gpu0_util_p95": _gpu_metric(0, "gpu_util_pct", "p95"),
                "gpu0_mem_used_gb": _gpu_metric(0, "memory_used_mb", "median") / 1024.0,
                "gpu1_util_median": _gpu_metric(1, "gpu_util_pct", "median"),
                "gpu1_util_p95": _gpu_metric(1, "gpu_util_pct", "p95"),
                "gpu1_mem_used_gb": _gpu_metric(1, "memory_used_mb", "median") / 1024.0,
                "main_process_pid": int(os.getpid()),
            }
        )
        if snapshot:
            process = snapshot.get("process") or {}
            mask_cache = snapshot.get("stage_join_buffer") or snapshot.get("mask_cache") or {}
            input_endpoint = process.get("input_endpoint") or {}
            tracking_stats = snapshot.get("tracking_stats") or {}
            summary.update(
                {
                    "cotracker_process_pid": int(process.get("pid", 0) or 0),
                    "cotracker_input_drop_count": int(snapshot.get("tracking_input_drop_count", 0) or 0),
                    "cotracker_input_queue_replace_count": int(
                        snapshot.get("tracking_input_queue_replace_count", 0)
                        or input_endpoint.get("replaced", 0)
                        or 0
                    ),
                    "cotracker_model_ms_median": float(snapshot.get("cotracker_model_ms_median", 0.0) or 0.0),
                    "cotracker_model_ms_p95": float(snapshot.get("cotracker_model_ms_p95", 0.0) or 0.0),
                    "cotracker_e2e_ms_median": float(snapshot.get("cotracker_e2e_ms_median", 0.0) or 0.0),
                    "cotracker_e2e_ms_p95": float(snapshot.get("cotracker_e2e_ms_p95", 0.0) or 0.0),
                    "overlay_age_ms_median": float(snapshot.get("overlay_age_ms_median", 0.0) or 0.0),
                    "overlay_age_ms_p95": float(snapshot.get("overlay_age_ms_p95", 0.0) or 0.0),
                    "mask_reuse_ratio": float(mask_cache.get("mask_reuse_ratio", 0.0) or 0.0),
                    "mask_age_ms_median": float(mask_cache.get("mask_age_ms_median", 0.0) or 0.0),
                    "mask_age_ms_p95": float(mask_cache.get("mask_age_ms_p95", 0.0) or 0.0),
                    "tracking_query_count_actual_by_camera": tracking_stats.get("tracking_query_count_actual_by_camera", {}),
                    "tracking_union_pixels_by_camera": tracking_stats.get("tracking_union_pixels_by_camera", {}),
                    "tracking_object_pixels_by_camera": tracking_stats.get("tracking_object_pixels_by_camera", {}),
                    "tracking_controller_pixels_by_camera": tracking_stats.get("tracking_controller_pixels_by_camera", {}),
                    "tracking_sample_object_hits_by_camera": tracking_stats.get("tracking_sample_object_hits_by_camera", {}),
                    "tracking_sample_controller_hits_by_camera": tracking_stats.get("tracking_sample_controller_hits_by_camera", {}),
                    "tracking_sample_overlap_hits_by_camera": tracking_stats.get("tracking_sample_overlap_hits_by_camera", {}),
                    "tracking_sample_background_hits_by_camera": tracking_stats.get("tracking_sample_background_hits_by_camera", {}),
                    "overlay_display_count_by_camera": tracking_stats.get("overlay_display_count_by_camera", {}),
                }
            )
        return summary


def main(
    argv: Sequence[str] | None = None,
    *,
    cuda_device_count_provider: CudaDeviceCountProvider | None = None,
) -> int:
    parser = build_arg_parser()
    try:
        args = parser.parse_args(argv)
        args = apply_preset_defaults(args, explicit_options=demo3_runtime._explicit_cli_options(argv))
        validate_args(args, require_calibration=False, cuda_device_count_provider=cuda_device_count_provider)
        contract = build_contract(args, cuda_device_count_provider=cuda_device_count_provider)
        if args.dry_run:
            print(format_contract(contract))
            _write_profile(args.profile_json_output, {"contract": contract, "summary": contract["profile_summary_fields"]})
            return 0
        profile = Demo31Runtime(args, cuda_device_count_provider=cuda_device_count_provider).run()
        print(json.dumps(profile["summary"], indent=2, sort_keys=True))
        return int(profile.get("exit_code", 0))
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


__all__ = [
    "Demo31Runtime",
    "FUSION_MASK_POLICY_LATEST_REUSE",
    "FUSION_MASK_POLICY_STRICT",
    "PRESET_DEMO31_DUAL4090_HIGHFPS",
    "apply_preset_defaults",
    "build_arg_parser",
    "build_contract",
    "build_cotracker_process_config",
    "format_contract",
    "fresh_tracking_result_or_none",
    "main",
    "make_demo31_live_runtime_class",
    "Demo31MaskPolicyJoinBuffer",
    "validate_args",
]
