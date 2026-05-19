from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import pickle
import sys
import time
from typing import Any, Callable, Sequence

import numpy as np

from qqtt.demo.cotracker3_overlay_worker import (
    COTRACKER_UPDATE_MODE_AUTO,
    COTRACKER_UPDATE_MODES,
    CoTracker3OverlayThread,
    CoTracker3OverlayWorker,
    LatestTrackingInputSlot,
    LatestTrackingOverlaySlot,
    OVERLAY_DISPLAY_SCOPE_CONTROLLER,
    OVERLAY_DISPLAY_SCOPES,
    TrackingOverlayInputPacket,
)
from qqtt.demo.phystwin_volume_filter import (
    DEFAULT_PHYSTWIN_OBJECT_VOLUME_EMERGENCY_MAX_POINTS,
    DEFAULT_PHYSTWIN_OBJECT_VOLUME_MAX_VOXEL_M,
    DEFAULT_PHYSTWIN_OBJECT_VOLUME_MIN_VOXEL_M,
    DEFAULT_PHYSTWIN_OBJECT_VOLUME_POINTS_PER_VOXEL,
    DEFAULT_PHYSTWIN_OBJECT_VOLUME_TARGET_MS,
    DEFAULT_PHYSTWIN_OBJECT_VOLUME_VOXEL_M,
    PHYSTWIN_VOLUME_ORIGIN_WORLD,
    PHYSTWIN_VOLUME_ORIGINS,
)


PRESET_DEMO3_REALSENSE_COTRACKER_HIGHFPS = "demo3-realsense-cotracker-highfps"
PRESET_DEMO3_REALSENSE_MASK_ONLY = "demo3-realsense-mask-only"
PRESET_DEMO3_REALSENSE_COTRACKER_PROFILE = "demo3-realsense-cotracker-profile"
PRESETS = (
    PRESET_DEMO3_REALSENSE_COTRACKER_HIGHFPS,
    PRESET_DEMO3_REALSENSE_MASK_ONLY,
    PRESET_DEMO3_REALSENSE_COTRACKER_PROFILE,
)

DEPTH_SOURCE_REALSENSE = "realsense"
DEPTH_SOURCE_FFS = "ffs"
MASK_SOURCE_HF_EDGETAM = "hf_edgetam"
MASK_SOURCE_HF_EDGETAM_CLI = "hf-edgetam"
COTRACKER3_ONLINE = "cotracker3_online"

MODE_EXP = "exp"
MODE_DEMO = "demo"
MODES = (MODE_EXP, MODE_DEMO)

SHARED_EXPERIMENT_MODE_CONTROLLER_OBJECT = "controller-object-exp"
SHARED_EXPERIMENT_MODE_DEMO = "demo-mode"
SHARED_TRACK_MODE_CONTROLLER_OBJECT = "controller-object"
TRACK_SCOPE_OBJECT_CONTROLLER_UNION = "object_controller_union"
TRACKING_QUERY_MODE_PHYSTWIN_DENSE = "phystwin_dense"
TRACKING_QUERY_COUNT_AUTO = "auto"
TRACKING_QUERY_COUNT_RULE_PHYSTWIN_DENSE = "min(union_mask_pixels, 5000)"
TRACKING_SAMPLING_TORCH_RANDPERM = "torch_randperm_seed_plus_camera_idx"
DEFAULT_COTRACKER_SEED = 42
PHYSTWIN_DENSE_MAX_POINTS = 5000

RENDER_MODE_POINTCLOUD = "pointcloud"
RENDER_MODE_NONE = "none"
RENDER_MODES = (RENDER_MODE_POINTCLOUD, RENDER_MODE_NONE)
GPU_SAMPLING_BACKENDS = ("nvml",)
OBJECT_POINT_CONTROL_FIXED_CAP = "fixed-cap"
OBJECT_POINT_CONTROL_PHYSTWIN_VOLUME = "phystwin-volume"
OBJECT_POINT_CONTROLS = (OBJECT_POINT_CONTROL_FIXED_CAP, OBJECT_POINT_CONTROL_PHYSTWIN_VOLUME)

DEFAULT_WIDTH = 848
DEFAULT_HEIGHT = 480
DEFAULT_FPS = 30
DEFAULT_CAMERA_IDS = (0, 1, 2)
DEFAULT_OBJECT_PROMPT = "stuffed animal"
EXP_CONTROLLER_PROMPT = "towel"
DEMO_CONTROLLER_PROMPT = "hand"
DEFAULT_MODE = MODE_EXP
DEFAULT_COTRACKER_QUERY_COUNT_REQUEST = TRACKING_QUERY_COUNT_AUTO
DEFAULT_COTRACKER_UPDATE_MODE = COTRACKER_UPDATE_MODE_AUTO
DEFAULT_OVERLAY_MAX_POINTS_PER_CAMERA = 30
DEFAULT_OVERLAY_DISPLAY_SCOPE = OVERLAY_DISPLAY_SCOPE_CONTROLLER
DEFAULT_OVERLAY_TRAIL_LEN = 16
DEFAULT_OVERLAY_STALE_TIMEOUT_MS = 500.0
DEFAULT_COTRACKER_WINDOW_LEN = 16
DEFAULT_COTRACKER_PUBLISH_STEP = 8
DEFAULT_OUTPUT_ROOT = Path("result/demo3_realsense_cotracker")
DEFAULT_EDGETAM_LIVE_SESSION_KEEP_FRAMES = 64
EDGETAM_BATCH_VISION_ENCODER_REQUIRED = True
OVERLAY_COLOR_RGB = np.array([255, 230, 32], dtype=np.uint8)


ConnectedSerialsProvider = Callable[[], Sequence[str]]
BackendFactory = Callable[[int], Any]


def parse_camera_ids(value: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
        return tuple(int(item) for item in items)
    return tuple(int(item) for item in value)


def parse_gpu_sampling_device_indexes(value: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        indexes = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    else:
        indexes = tuple(int(item) for item in value)
    if not indexes:
        raise argparse.ArgumentTypeError("GPU sampling indexes must look like 0 or 0,1")
    if any(index < 0 for index in indexes):
        raise argparse.ArgumentTypeError(f"GPU sampling indexes must be non-negative: {indexes}")
    if len(set(indexes)) != len(indexes):
        raise argparse.ArgumentTypeError(f"GPU sampling indexes must be unique: {indexes}")
    return indexes


def _explicit_cli_options(argv: Sequence[str] | None) -> set[str]:
    explicit: set[str] = set()
    if argv is None:
        argv = sys.argv[1:]
    for item in argv:
        if item.startswith("--"):
            explicit.add(item.split("=", 1)[0])
    return explicit


def _normalize_mask_source(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized != MASK_SOURCE_HF_EDGETAM:
        raise ValueError("Demo 3 mask source must be hf-edgetam.")
    return MASK_SOURCE_HF_EDGETAM


def resolve_demo3_mode(mode: str) -> dict[str, str]:
    normalized = str(mode).strip().lower()
    if normalized == MODE_EXP:
        return {
            "semantic_mode": MODE_EXP,
            "experiment_mode": SHARED_EXPERIMENT_MODE_CONTROLLER_OBJECT,
            "controller_prompt": EXP_CONTROLLER_PROMPT,
            "controller_label": EXP_CONTROLLER_PROMPT,
        }
    if normalized == MODE_DEMO:
        return {
            "semantic_mode": MODE_DEMO,
            "experiment_mode": SHARED_EXPERIMENT_MODE_DEMO,
            "controller_prompt": DEMO_CONTROLLER_PROMPT,
            "controller_label": DEMO_CONTROLLER_PROMPT,
        }
    raise ValueError(f"Unsupported Demo 3 mode: {mode}")


def normalize_cotracker_query_count_request(value: Any) -> str:
    raw = str(value).strip().lower()
    if raw == TRACKING_QUERY_COUNT_AUTO:
        return TRACKING_QUERY_COUNT_AUTO
    try:
        count = int(raw)
    except ValueError as exc:
        raise ValueError("--cotracker-query-count must be 'auto' or a positive integer.") from exc
    if count <= 0:
        raise ValueError("--cotracker-query-count must be 'auto' or a positive integer.")
    return str(count)


def phystwin_dense_compatible_for_args(args: argparse.Namespace) -> bool:
    return (
        str(args.cotracker_query_mode) == TRACKING_QUERY_MODE_PHYSTWIN_DENSE
        and normalize_cotracker_query_count_request(args.cotracker_query_count) == TRACKING_QUERY_COUNT_AUTO
        and int(args.cotracker_seed) == DEFAULT_COTRACKER_SEED
        and str(args.cotracker_backend) == COTRACKER3_ONLINE
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Demo 3 realtime visualization: three RealSense RGB-D cameras, HF "
            "EdgeTAM masks, RealSense fused PCD, and async CoTracker3 overlay."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--preset", choices=PRESETS, default=PRESET_DEMO3_REALSENSE_COTRACKER_HIGHFPS)
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved Demo 3 runtime contract and exit.")
    parser.add_argument("--duration-s", type=float, default=120.0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--profile-json-output", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--camera-ids", type=parse_camera_ids, default=DEFAULT_CAMERA_IDS)
    parser.add_argument("--serials", nargs="*", default=None)
    parser.add_argument("--calibrate-path", type=Path, default=Path("calibrate.pkl"))
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--depth-source", default=DEPTH_SOURCE_REALSENSE)
    parser.add_argument("--mask-source", default=MASK_SOURCE_HF_EDGETAM_CLI)
    parser.add_argument(
        "--edgetam-live-session-keep-frames",
        type=int,
        default=DEFAULT_EDGETAM_LIVE_SESSION_KEEP_FRAMES,
        help=(
            "Maximum recent HF EdgeTAM live-session frames kept per camera in "
            "the shared live runtime. This bounds long-run GPU memory growth."
        ),
    )
    parser.add_argument("--mode", choices=MODES, default=DEFAULT_MODE)
    parser.add_argument("--object-prompt", default=DEFAULT_OBJECT_PROMPT)
    parser.add_argument("--cotracker-backend", default=COTRACKER3_ONLINE)
    parser.add_argument("--cotracker-query-mode", choices=(TRACKING_QUERY_MODE_PHYSTWIN_DENSE,), default=TRACKING_QUERY_MODE_PHYSTWIN_DENSE)
    parser.add_argument("--cotracker-query-count", default=DEFAULT_COTRACKER_QUERY_COUNT_REQUEST)
    parser.add_argument("--cotracker-seed", type=int, default=DEFAULT_COTRACKER_SEED)
    parser.add_argument("--cotracker-update-mode", choices=COTRACKER_UPDATE_MODES, default=DEFAULT_COTRACKER_UPDATE_MODE)
    parser.add_argument("--disable-cotracker", action="store_true")
    parser.add_argument("--render-mode", choices=RENDER_MODES, default=RENDER_MODE_POINTCLOUD)
    parser.add_argument("--point-size", type=float, default=None)
    parser.add_argument("--render-every-n", type=int, default=None)
    parser.add_argument("--render-backend", default=None)
    parser.add_argument("--render-layer-mode", default=None)
    parser.add_argument("--render-copy-mode", default=None)
    parser.add_argument("--no-render-async-latest-only", action="store_true")
    parser.add_argument("--render-micro-profile", action="store_true")
    parser.add_argument("--object-point-control", choices=OBJECT_POINT_CONTROLS, default=OBJECT_POINT_CONTROL_PHYSTWIN_VOLUME)
    parser.add_argument("--object-volume-voxel-m", type=float, default=DEFAULT_PHYSTWIN_OBJECT_VOLUME_VOXEL_M)
    parser.add_argument("--object-volume-origin", choices=PHYSTWIN_VOLUME_ORIGINS, default=PHYSTWIN_VOLUME_ORIGIN_WORLD)
    parser.add_argument("--object-volume-adaptive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--object-volume-min-voxel-m", type=float, default=DEFAULT_PHYSTWIN_OBJECT_VOLUME_MIN_VOXEL_M)
    parser.add_argument("--object-volume-max-voxel-m", type=float, default=DEFAULT_PHYSTWIN_OBJECT_VOLUME_MAX_VOXEL_M)
    parser.add_argument("--object-volume-target-ms", type=float, default=DEFAULT_PHYSTWIN_OBJECT_VOLUME_TARGET_MS)
    parser.add_argument(
        "--object-volume-emergency-max-points",
        type=int,
        default=DEFAULT_PHYSTWIN_OBJECT_VOLUME_EMERGENCY_MAX_POINTS,
    )
    parser.add_argument(
        "--object-volume-points-per-voxel",
        type=int,
        default=DEFAULT_PHYSTWIN_OBJECT_VOLUME_POINTS_PER_VOXEL,
    )
    parser.add_argument("--debug-color-by-camera", action="store_true")
    parser.add_argument("--debug-save-per-camera-pcd", action="store_true")
    parser.add_argument("--debug-save-mask-overlays", action="store_true")
    parser.add_argument("--debug-identity-c2w", action="store_true")
    parser.add_argument("--debug-invert-c2w", action="store_true")
    parser.add_argument("--debug-only-camera-idx", type=int, choices=DEFAULT_CAMERA_IDS, default=None)
    parser.add_argument("--debug-fusion-max-saved-groups", type=int, default=None)
    parser.add_argument("--gpu-sampling", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gpu-sampling-interval-s", type=float, default=0.5)
    parser.add_argument("--gpu-sampling-backend", choices=GPU_SAMPLING_BACKENDS, default="nvml")
    parser.add_argument("--gpu-sampling-device-index", type=int, default=0)
    parser.add_argument("--gpu-sampling-device-indexes", type=parse_gpu_sampling_device_indexes, default=None)
    parser.add_argument("--overlay-max-points-per-camera", type=int, default=DEFAULT_OVERLAY_MAX_POINTS_PER_CAMERA)
    parser.add_argument("--overlay-display-scope", choices=OVERLAY_DISPLAY_SCOPES, default=DEFAULT_OVERLAY_DISPLAY_SCOPE)
    parser.add_argument("--overlay-trail-len", type=int, default=DEFAULT_OVERLAY_TRAIL_LEN)
    parser.add_argument("--overlay-stale-timeout-ms", type=float, default=DEFAULT_OVERLAY_STALE_TIMEOUT_MS)
    return parser


def apply_preset_defaults(args: argparse.Namespace, *, explicit_options: set[str] | None = None) -> argparse.Namespace:
    explicit = set(explicit_options or set())
    if args.preset == PRESET_DEMO3_REALSENSE_MASK_ONLY:
        if "--disable-cotracker" not in explicit:
            args.disable_cotracker = True
    elif args.preset == PRESET_DEMO3_REALSENSE_COTRACKER_PROFILE:
        if "--render-mode" not in explicit:
            args.render_mode = RENDER_MODE_NONE
        if "--duration-s" not in explicit:
            args.duration_s = 60.0
    elif args.preset == PRESET_DEMO3_REALSENSE_COTRACKER_HIGHFPS:
        if "--cotracker-query-count" not in explicit:
            args.cotracker_query_count = DEFAULT_COTRACKER_QUERY_COUNT_REQUEST
        if "--overlay-max-points-per-camera" not in explicit:
            args.overlay_max_points_per_camera = DEFAULT_OVERLAY_MAX_POINTS_PER_CAMERA
    return args


def validate_args(args: argparse.Namespace, *, require_calibration: bool = False) -> None:
    camera_ids = parse_camera_ids(args.camera_ids)
    if len(camera_ids) != 3:
        raise ValueError("Demo 3 requires exactly three RealSense cameras.")
    if len(set(camera_ids)) != 3:
        raise ValueError("Demo 3 requires exactly three distinct RealSense cameras.")
    depth_source = str(args.depth_source).strip().lower()
    if depth_source == DEPTH_SOURCE_FFS or depth_source.startswith("ffs"):
        raise ValueError("Demo 3 does not support FFS. Use --depth-source realsense.")
    if depth_source != DEPTH_SOURCE_REALSENSE:
        raise ValueError("Demo 3 depth source must be realsense.")
    _normalize_mask_source(str(args.mask_source))
    if str(args.cotracker_backend) != COTRACKER3_ONLINE:
        raise ValueError("Demo 3 currently supports only --cotracker-backend cotracker3_online.")
    if str(args.cotracker_query_mode) != TRACKING_QUERY_MODE_PHYSTWIN_DENSE:
        raise ValueError("Demo 3 currently supports only --cotracker-query-mode phystwin_dense.")
    normalize_cotracker_query_count_request(args.cotracker_query_count)
    if str(args.cotracker_update_mode) not in COTRACKER_UPDATE_MODES:
        raise ValueError(f"--cotracker-update-mode must be one of {COTRACKER_UPDATE_MODES}.")
    if str(args.object_point_control) not in OBJECT_POINT_CONTROLS:
        raise ValueError(f"Demo 3 unsupported --object-point-control {args.object_point_control}")
    if str(args.object_volume_origin) not in PHYSTWIN_VOLUME_ORIGINS:
        raise ValueError(f"Demo 3 unsupported --object-volume-origin {args.object_volume_origin}")
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
        raise ValueError("Demo 3 accepts only one of --debug-identity-c2w or --debug-invert-c2w.")
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
    if str(args.overlay_display_scope) not in OVERLAY_DISPLAY_SCOPES:
        raise ValueError(f"--overlay-display-scope must be one of {OVERLAY_DISPLAY_SCOPES}.")
    if require_calibration and not Path(args.calibrate_path).is_file():
        raise FileNotFoundError(f"Demo 3 requires calibrate.pkl for three-camera world fusion: {args.calibrate_path}")


def build_contract(args: argparse.Namespace) -> dict[str, Any]:
    camera_ids = parse_camera_ids(args.camera_ids)
    cotracker_enabled = not bool(args.disable_cotracker)
    mode = resolve_demo3_mode(str(args.mode))
    query_count_request = normalize_cotracker_query_count_request(args.cotracker_query_count)
    contract: dict[str, Any] = {
        "demo": "demo3",
        "preset": str(args.preset),
        "input_source": "live_realsense",
        "offline_mode_available": False,
        "offline_tracking_available": False,
        "init_mode": "sam31_first_frame",
        "mask_propagation": "hf_edgetam_online",
        "requires_three_realsense": True,
        "num_cameras": int(len(camera_ids)),
        "num_realsense_cameras": int(len(camera_ids)),
        "camera_ids": list(camera_ids),
        "serials": list(args.serials or []),
        "calibrate_path": str(args.calibrate_path),
        "calibrate_pkl_loaded": bool(Path(args.calibrate_path).is_file()),
        "depth_source": DEPTH_SOURCE_REALSENSE,
        "uses_ffs": False,
        "mask_source": MASK_SOURCE_HF_EDGETAM,
        "edgetam_batch_vision_encoder": EDGETAM_BATCH_VISION_ENCODER_REQUIRED,
        "edgetam_live_session_keep_frames": int(args.edgetam_live_session_keep_frames),
        "edgetam_live_session_pruning": True,
        "edgetam_sessions_per_camera": 1,
        "semantic_mode": str(mode["semantic_mode"]),
        "shared_experiment_mode": str(mode["experiment_mode"]),
        "shared_runtime_track_mode": SHARED_TRACK_MODE_CONTROLLER_OBJECT,
        "tracking_mask_scope": TRACK_SCOPE_OBJECT_CONTROLLER_UNION,
        "object_prompt": str(args.object_prompt),
        "controller_prompt": str(mode["controller_prompt"]),
        "tracking_controller_label": str(mode["controller_label"]),
        "cotracker_enabled": bool(cotracker_enabled),
        "cotracker_backend": COTRACKER3_ONLINE,
        "cotracker_async": True,
        "cotracker_latest_wins": True,
        "cotracker_update_mode": str(args.cotracker_update_mode),
        "cotracker_batch_size_target": int(len(camera_ids)),
        "cotracker_batch_fallback_enabled": str(args.cotracker_update_mode) == COTRACKER_UPDATE_MODE_AUTO,
        "tracking_query_mode": TRACKING_QUERY_MODE_PHYSTWIN_DENSE,
        "tracking_query_count_requested": str(query_count_request),
        "tracking_query_count_rule": TRACKING_QUERY_COUNT_RULE_PHYSTWIN_DENSE,
        "tracking_sampling": TRACKING_SAMPLING_TORCH_RANDPERM,
        "tracking_max_query_points_per_camera": PHYSTWIN_DENSE_MAX_POINTS,
        "cotracker_seed": int(args.cotracker_seed),
        "phystwin_dense_compatible": bool(phystwin_dense_compatible_for_args(args)),
        "cotracker_window_len": DEFAULT_COTRACKER_WINDOW_LEN,
        "cotracker_publish_step": DEFAULT_COTRACKER_PUBLISH_STEP,
        "overlay_max_points_per_camera": int(args.overlay_max_points_per_camera),
        "overlay_display_scope": str(args.overlay_display_scope),
        "overlay_display_classification": "first_frame_mask_membership",
        "overlay_trail_len": int(args.overlay_trail_len),
        "overlay_stale_timeout_ms": float(args.overlay_stale_timeout_ms),
        "render_mode": str(args.render_mode),
        "render_latest_wins": True,
        "render_waited_for_cotracker": False,
        "render_backend": None if args.render_backend is None else str(args.render_backend),
        "render_layer_mode": None if args.render_layer_mode is None else str(args.render_layer_mode),
        "render_copy_mode": None if args.render_copy_mode is None else str(args.render_copy_mode),
        "render_micro_profile": True,
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
            "device_indexes": None if args.gpu_sampling_device_indexes is None else list(args.gpu_sampling_device_indexes),
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
        ],
    }
    contract["profile_summary_fields"] = build_empty_profile_summary(contract)
    return contract


def build_empty_profile_summary(contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "rendered_fps": 0.0,
        "render_loop_fps": 0.0,
        "capture_group_fps": 0.0,
        "edgetam_mask_fps": 0.0,
        "fusion_fps": 0.0,
        "cotracker_publish_fps": 0.0,
        "cotracker_model_ms_median": 0.0,
        "cotracker_model_ms_p95": 0.0,
        "cotracker_e2e_ms_median": 0.0,
        "cotracker_e2e_ms_p95": 0.0,
        "overlay_ms_median": 0.0,
        "overlay_ms_p95": 0.0,
        "pcd_fusion_ms_median": 0.0,
        "pcd_render_ms_median": 0.0,
        "render_waited_for_cotracker": False,
        "uses_ffs": False,
        "depth_source": DEPTH_SOURCE_REALSENSE,
        "mask_source": MASK_SOURCE_HF_EDGETAM,
        "edgetam_batch_vision_encoder": EDGETAM_BATCH_VISION_ENCODER_REQUIRED,
        "edgetam_live_session_keep_frames": int(
            contract.get("edgetam_live_session_keep_frames", DEFAULT_EDGETAM_LIVE_SESSION_KEEP_FRAMES)
        ),
        "edgetam_live_session_pruning": bool(contract.get("edgetam_live_session_pruning", True)),
        "num_realsense_cameras": int(contract.get("num_realsense_cameras", 3)),
        "calibrate_pkl_loaded": bool(contract.get("calibrate_pkl_loaded", False)),
        "cotracker_backend": COTRACKER3_ONLINE,
        "cotracker_window_len": DEFAULT_COTRACKER_WINDOW_LEN,
        "cotracker_publish_step": DEFAULT_COTRACKER_PUBLISH_STEP,
        "input_source": "live_realsense",
        "offline_mode_available": False,
        "offline_tracking_available": False,
        "init_mode": "sam31_first_frame",
        "mask_propagation": "hf_edgetam_online",
        "semantic_mode": str(contract.get("semantic_mode", DEFAULT_MODE)),
        "tracking_mask_scope": TRACK_SCOPE_OBJECT_CONTROLLER_UNION,
        "tracking_query_mode": TRACKING_QUERY_MODE_PHYSTWIN_DENSE,
        "tracking_query_count_requested": str(contract.get("tracking_query_count_requested", TRACKING_QUERY_COUNT_AUTO)),
        "tracking_query_count_rule": TRACKING_QUERY_COUNT_RULE_PHYSTWIN_DENSE,
        "tracking_sampling": TRACKING_SAMPLING_TORCH_RANDPERM,
        "cotracker_seed": int(contract.get("cotracker_seed", DEFAULT_COTRACKER_SEED)),
        "cotracker_update_mode": str(contract.get("cotracker_update_mode", DEFAULT_COTRACKER_UPDATE_MODE)),
        "cotracker_update_mode_effective": str(contract.get("cotracker_update_mode", DEFAULT_COTRACKER_UPDATE_MODE)),
        "cotracker_batch_size_target": int(contract.get("cotracker_batch_size_target", 3)),
        "cotracker_batch_size": 0,
        "cotracker_batch_update_count": 0,
        "cotracker_serial_group_update_count": 0,
        "cotracker_serial_camera_update_count": 0,
        "cotracker_serial_fallback_count": 0,
        "cotracker_batch_error_count": 0,
        "cotracker_batch_disabled_reason": None,
        "phystwin_dense_compatible": bool(contract.get("phystwin_dense_compatible", False)),
        "tracking_query_count_actual_by_camera": {},
        "tracking_union_pixels_by_camera": {},
        "tracking_object_pixels_by_camera": {},
        "tracking_controller_pixels_by_camera": {},
        "overlay_max_points_per_camera": int(contract.get("overlay_max_points_per_camera", DEFAULT_OVERLAY_MAX_POINTS_PER_CAMERA)),
        "overlay_display_scope": str(contract.get("overlay_display_scope", DEFAULT_OVERLAY_DISPLAY_SCOPE)),
        "overlay_display_classification": str(
            contract.get("overlay_display_classification", "first_frame_mask_membership")
        ),
        "overlay_display_count_by_camera": {},
        "overlay_display_object_count_by_camera": {},
        "overlay_display_controller_count_by_camera": {},
    }


def format_contract(contract: dict[str, Any]) -> str:
    keys = (
        "demo",
        "input_source",
        "offline_mode_available",
        "requires_three_realsense",
        "num_cameras",
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
        "cotracker_update_mode",
        "overlay_max_points_per_camera",
        "overlay_display_scope",
        "phystwin_dense_compatible",
        "cotracker_backend",
        "cotracker_async",
        "render_latest_wins",
        "render_waited_for_cotracker",
    )
    lines = []
    for key in keys:
        value = contract[key]
        if isinstance(value, bool):
            rendered = str(value).lower()
        else:
            rendered = str(value)
        lines.append(f"{key} = {rendered}")
    lines.append(json.dumps(contract, indent=2, sort_keys=True))
    return "\n".join(lines)


def _write_profile(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_shared_runtime_module():
    from qqtt.demo import three_view_masked_fused_pcd_runtime as shared_runtime

    return shared_runtime


def _get_connected_realsense_serials() -> list[str]:
    from qqtt.env.camera.realsense.single_realsense import SingleRealsense

    return list(SingleRealsense.get_connected_devices_serial())


def validate_live_realsense_contract(
    args: argparse.Namespace,
    *,
    connected_serials_provider: ConnectedSerialsProvider | None = None,
) -> dict[str, Any]:
    validate_args(args, require_calibration=True)
    provider = connected_serials_provider or _get_connected_realsense_serials
    connected_serials = list(provider())
    requested_serials = list(args.serials or [])
    if requested_serials:
        if len(requested_serials) != 3:
            raise RuntimeError("Demo 3 requires exactly three requested RealSense serials when --serials is used.")
        missing = [serial for serial in requested_serials if serial not in connected_serials]
        if missing:
            raise RuntimeError(f"Demo 3 requested RealSense serials are not connected: {missing}")
        active_serials = requested_serials
    else:
        if len(connected_serials) != 3:
            raise RuntimeError(
                "Demo 3 requires exactly three connected RealSense cameras when --serials is not provided. "
                f"connected={len(connected_serials)}"
            )
        active_serials = connected_serials

    from qqtt.env.camera.calibration_metadata import load_calibration_reference_serials

    calibration_reference_serials = load_calibration_reference_serials(args.calibrate_path)
    if calibration_reference_serials is not None:
        if len(calibration_reference_serials) != 3:
            raise RuntimeError(
                "Demo 3 requires calibrate.pkl metadata for exactly three cameras. "
                f"calibration_reference_serials={len(calibration_reference_serials)}"
            )
        missing_from_calibration = [serial for serial in active_serials if serial not in calibration_reference_serials]
        if missing_from_calibration:
            raise RuntimeError(
                "Demo 3 active RealSense serials are not covered by calibrate.pkl metadata. "
                f"missing={missing_from_calibration}"
            )
    try:
        calibration_transform_count = _calibration_transform_count(args.calibrate_path)
    except Exception as exc:
        raise RuntimeError(f"Demo 3 calibration validation failed: {exc}") from exc
    if calibration_transform_count != 3:
        raise RuntimeError(
            "Demo 3 requires calibrate.pkl to contain exactly three camera-to-world transforms. "
            f"transform_count={calibration_transform_count}"
        )
    return {
        "connected_serials": connected_serials,
        "active_serials": active_serials,
        "calibration_reference_serials": calibration_reference_serials,
        "calibration_transform_count": int(calibration_transform_count),
    }


def _calibration_transform_count(calibrate_path: str | Path) -> int:
    with Path(calibrate_path).open("rb") as handle:
        raw = pickle.load(handle)
    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[1:] != (4, 4):
        raise ValueError(f"Unsupported calibrate.pkl transform shape: {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("calibrate.pkl contains non-finite transform values.")
    expected_bottom = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    if not np.allclose(arr[:, 3, :], expected_bottom[None, :], atol=1e-4):
        raise ValueError("calibrate.pkl contains an invalid homogeneous bottom row.")
    return int(arr.shape[0])


def _shared_profile_path(args: argparse.Namespace) -> Path | None:
    if args.profile_json_output is None:
        return Path(args.output_root) / "demo3_shared_runtime_profile.json"
    path = Path(args.profile_json_output)
    return path.with_name(f"{path.stem}_shared_runtime{path.suffix or '.json'}")


def _shared_runtime_label_for_args(args: argparse.Namespace) -> tuple[str, str]:
    if str(getattr(args, "preset", "")).startswith("demo3.1"):
        return "demo3.1", "Demo 3.1"
    return "demo3", "Demo 3"


def _gpu_sampling_device_indexes_for_args(args: argparse.Namespace) -> tuple[int, ...] | None:
    indexes = getattr(args, "gpu_sampling_device_indexes", None)
    if indexes is not None:
        return tuple(int(index) for index in indexes)
    if str(getattr(args, "preset", "")).startswith("demo3.1") and hasattr(args, "mask_gpu") and hasattr(args, "cotracker_gpu"):
        return tuple(dict.fromkeys((int(args.mask_gpu), int(args.cotracker_gpu))))
    return None


def build_shared_runtime_argv(
    args: argparse.Namespace,
    *,
    active_serials: Sequence[str],
    calibration_reference_serials: Sequence[str] | None,
    shared_profile_path: Path | None,
) -> list[str]:
    render_mode = RENDER_MODE_NONE if str(args.render_mode) == RENDER_MODE_NONE else RENDER_MODE_POINTCLOUD
    mode = resolve_demo3_mode(str(args.mode))
    shared_demo_version, shared_display_name = _shared_runtime_label_for_args(args)
    gpu_sampling_device_indexes = _gpu_sampling_device_indexes_for_args(args)
    argv = [
        "--preset",
        "demo2.1.5-live-fast-native",
        "--demo-version-override",
        shared_demo_version,
        "--demo-display-name-override",
        shared_display_name,
        "--profile",
        f"{int(args.width)}x{int(args.height)}",
        "--fps",
        str(int(args.fps)),
        "--fusion-target-fps",
        str(float(args.fps)),
        "--capture-group-target-fps",
        str(float(args.fps)),
        "--camera-ids",
        ",".join(str(item) for item in parse_camera_ids(args.camera_ids)),
        "--calibrate-path",
        str(args.calibrate_path),
        "--serials",
        *[str(serial) for serial in active_serials],
        "--depth-source",
        DEPTH_SOURCE_REALSENSE,
        "--edgetam-batch-vision-encoder",
        "--edgetam-live-session-keep-frames",
        str(int(args.edgetam_live_session_keep_frames)),
        "--render-mode",
        render_mode,
        "--pcd-color-mode",
        str(getattr(args, "pcd_color_mode", "rgb")),
        "--track-mode",
        SHARED_TRACK_MODE_CONTROLLER_OBJECT,
        "--object-prompt",
        str(args.object_prompt),
        "--controller-prompt",
        str(mode["controller_prompt"]),
        "--experiment-mode",
        str(mode["experiment_mode"]),
        "--duration-s",
        str(float(args.duration_s)),
        "--output-root",
        str(args.output_root),
        "--profile-pipeline",
        "--profile-visualization",
        "--render-micro-profile",
        "--tracking-backend",
        "none",
        "--tracking-source",
        "cached",
        "--tracking-num-points",
        str(PHYSTWIN_DENSE_MAX_POINTS),
        "--tracking-overlay-max-points",
        str(int(args.overlay_max_points_per_camera)),
        "--tracking-trail-len",
        str(int(args.overlay_trail_len)),
        "--tracking-depth-source",
        "native",
        "--gpu-sampling-interval-s",
        str(float(args.gpu_sampling_interval_s)),
        "--gpu-sampling-backend",
        str(args.gpu_sampling_backend),
        "--gpu-sampling-device-index",
        str(int(args.gpu_sampling_device_index)),
    ]
    for flag, value in (
        ("--point-size", getattr(args, "point_size", None)),
        ("--render-every-n", getattr(args, "render_every_n", None)),
        ("--render-backend", getattr(args, "render_backend", None)),
        ("--render-layer-mode", getattr(args, "render_layer_mode", None)),
        ("--render-copy-mode", getattr(args, "render_copy_mode", None)),
        ("--object-point-control", getattr(args, "object_point_control", OBJECT_POINT_CONTROL_PHYSTWIN_VOLUME)),
        ("--object-volume-voxel-m", getattr(args, "object_volume_voxel_m", DEFAULT_PHYSTWIN_OBJECT_VOLUME_VOXEL_M)),
        ("--object-volume-origin", getattr(args, "object_volume_origin", PHYSTWIN_VOLUME_ORIGIN_WORLD)),
        (
            "--object-volume-min-voxel-m",
            getattr(args, "object_volume_min_voxel_m", DEFAULT_PHYSTWIN_OBJECT_VOLUME_MIN_VOXEL_M),
        ),
        (
            "--object-volume-max-voxel-m",
            getattr(args, "object_volume_max_voxel_m", DEFAULT_PHYSTWIN_OBJECT_VOLUME_MAX_VOXEL_M),
        ),
        ("--object-volume-target-ms", getattr(args, "object_volume_target_ms", DEFAULT_PHYSTWIN_OBJECT_VOLUME_TARGET_MS)),
        (
            "--object-volume-emergency-max-points",
            getattr(
                args,
                "object_volume_emergency_max_points",
                DEFAULT_PHYSTWIN_OBJECT_VOLUME_EMERGENCY_MAX_POINTS,
            ),
        ),
        (
            "--object-volume-points-per-voxel",
            getattr(args, "object_volume_points_per_voxel", DEFAULT_PHYSTWIN_OBJECT_VOLUME_POINTS_PER_VOXEL),
        ),
        ("--debug-only-camera-idx", getattr(args, "debug_only_camera_idx", None)),
        ("--debug-fusion-max-saved-groups", getattr(args, "debug_fusion_max_saved_groups", None)),
    ):
        if value is not None:
            argv.extend([flag, str(value)])
    if bool(getattr(args, "object_volume_adaptive", True)):
        argv.append("--object-volume-adaptive")
    else:
        argv.append("--no-object-volume-adaptive")
    if gpu_sampling_device_indexes is not None:
        argv.extend(["--gpu-sampling-device-indexes", ",".join(str(index) for index in gpu_sampling_device_indexes)])
    if calibration_reference_serials:
        argv.extend(["--calibration-reference-serials", *[str(serial) for serial in calibration_reference_serials]])
    if shared_profile_path is not None:
        argv.extend(["--profile-json-output", str(shared_profile_path)])
    if bool(args.debug):
        argv.append("--debug")
    if bool(args.gpu_sampling):
        argv.append("--gpu-sampling")
    if bool(args.render_micro_profile):
        argv.append("--render-micro-profile")
    if bool(args.no_render_async_latest_only):
        argv.append("--no-render-async-latest-only")
    if bool(args.debug_color_by_camera):
        argv.append("--debug-color-by-camera")
    if bool(args.debug_save_per_camera_pcd):
        argv.append("--debug-save-per-camera-pcd")
    if bool(args.debug_save_mask_overlays):
        argv.append("--debug-save-mask-overlays")
    if bool(args.debug_identity_c2w):
        argv.append("--debug-identity-c2w")
    if bool(args.debug_invert_c2w):
        argv.append("--debug-invert-c2w")
    return argv


def build_shared_runtime_args(
    args: argparse.Namespace,
    *,
    shared_runtime_module: Any | None = None,
    live_validation: dict[str, Any],
    shared_profile_path: Path | None,
) -> argparse.Namespace:
    shared = shared_runtime_module or _load_shared_runtime_module()
    parser = shared.build_arg_parser()
    shared_argv = build_shared_runtime_argv(
        args,
        active_serials=live_validation["active_serials"],
        calibration_reference_serials=live_validation.get("calibration_reference_serials"),
        shared_profile_path=shared_profile_path,
    )
    shared_args = parser.parse_args(shared_argv)
    return shared.apply_preset_defaults(shared_args, explicit_options=shared._explicit_cli_options(shared_argv))


def _phystwin_union_tracking_masks(mask_packet: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    object_mask = np.asarray(mask_packet.object_mask, dtype=bool)
    controller_mask = np.asarray(mask_packet.controller_mask, dtype=bool)
    union_mask = np.asarray(object_mask | controller_mask, dtype=bool)
    return union_mask, object_mask, controller_mask


def _runtime_stat_fps(runtime: Any, attr: str) -> float:
    return float(getattr(getattr(runtime, attr, None), "fps", 0.0) or 0.0)


def _stats(values: Sequence[float]) -> tuple[float, float]:
    arr = np.asarray([float(value) for value in values], dtype=np.float32)
    if arr.size == 0:
        return 0.0, 0.0
    return float(np.median(arr)), float(np.percentile(arr, 95))


def _nested_get(payload: dict[str, Any], path: Sequence[str], default: Any = None) -> Any:
    cursor: Any = payload
    for key in path:
        if not isinstance(cursor, dict) or key not in cursor:
            return default
        cursor = cursor[key]
    return cursor


def _load_json_if_exists(path: Path | None) -> dict[str, Any] | None:
    if path is None or not Path(path).is_file():
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _build_demo3_live_summary(
    *,
    contract: dict[str, Any],
    runtime: Any,
    exit_code: int,
    tracking_snapshot: dict[str, Any] | None,
    overlay_ms_samples: Sequence[float],
    shared_profile: dict[str, Any] | None,
) -> dict[str, Any]:
    summary = build_empty_profile_summary(contract)
    final = getattr(runtime, "_summary", {}).get("final", {}) if hasattr(runtime, "_summary") else {}
    warm = (shared_profile or {}).get("summary_after_warmup", {})
    metrics = warm.get("metrics", {}) if isinstance(warm, dict) else {}
    edge_fps_values = [
        float(getattr(stat, "fps", 0.0) or 0.0)
        for stat in getattr(runtime, "edge_stats", {}).values()
    ] if isinstance(getattr(runtime, "edge_stats", None), dict) else []
    overlay_median, overlay_p95 = _stats(overlay_ms_samples)
    tracking_worker = (tracking_snapshot or {}).get("worker", {}) if isinstance(tracking_snapshot, dict) else {}
    summary.update(
        {
            "exit_code": int(exit_code),
            "rendered_fps": float(final.get("render_fps", warm.get("render_fps", _runtime_stat_fps(runtime, "render_stats"))) or 0.0),
            "render_loop_fps": float(final.get("render_fps", warm.get("render_fps", _runtime_stat_fps(runtime, "render_stats"))) or 0.0),
            "capture_group_fps": float(final.get("capture_group_fps", warm.get("capture_group_fps", _runtime_stat_fps(runtime, "capture_group_stats"))) or 0.0),
            "edgetam_mask_fps": float(np.mean(edge_fps_values)) if edge_fps_values else 0.0,
            "fusion_fps": float(final.get("fusion_fps", warm.get("fusion_fps", _runtime_stat_fps(runtime, "fusion_stats"))) or 0.0),
            "cotracker_publish_fps": float(tracking_worker.get("publish_fps", 0.0) or 0.0),
            "cotracker_update_mode": str(
                tracking_worker.get(
                    "cotracker_update_mode_requested",
                    contract.get("cotracker_update_mode", DEFAULT_COTRACKER_UPDATE_MODE),
                )
            ),
            "cotracker_update_mode_effective": str(
                tracking_worker.get(
                    "cotracker_update_mode_effective",
                    contract.get("cotracker_update_mode", DEFAULT_COTRACKER_UPDATE_MODE),
                )
            ),
            "cotracker_batch_size": int(tracking_worker.get("cotracker_batch_size", 0) or 0),
            "cotracker_batch_update_count": int(tracking_worker.get("cotracker_batch_update_count", 0) or 0),
            "cotracker_serial_group_update_count": int(
                tracking_worker.get("cotracker_serial_group_update_count", 0) or 0
            ),
            "cotracker_serial_camera_update_count": int(
                tracking_worker.get("cotracker_serial_camera_update_count", 0) or 0
            ),
            "cotracker_serial_fallback_count": int(tracking_worker.get("cotracker_serial_fallback_count", 0) or 0),
            "cotracker_batch_error_count": int(tracking_worker.get("cotracker_batch_error_count", 0) or 0),
            "cotracker_batch_disabled_reason": tracking_worker.get("cotracker_batch_disabled_reason"),
            "cotracker_model_ms_median": float(tracking_worker.get("model_ms_median", 0.0) or 0.0),
            "cotracker_model_ms_p95": float(tracking_worker.get("model_ms_p95", 0.0) or 0.0),
            "cotracker_e2e_ms_median": float(tracking_worker.get("e2e_ms_median", 0.0) or 0.0),
            "cotracker_e2e_ms_p95": float(tracking_worker.get("e2e_ms_p95", 0.0) or 0.0),
            "overlay_ms_median": float(overlay_median),
            "overlay_ms_p95": float(overlay_p95),
            "pcd_fusion_ms_median": float(_nested_get(metrics, ("fusion_total_ms", "median"), 0.0) or 0.0),
            "pcd_render_ms_median": float(_nested_get(metrics, ("render_total_ms", "median"), 0.0) or 0.0),
            "render_waited_for_cotracker": False,
            "uses_ffs": False,
            "depth_source": DEPTH_SOURCE_REALSENSE,
            "mask_source": MASK_SOURCE_HF_EDGETAM,
            "num_realsense_cameras": 3,
            "calibrate_pkl_loaded": True,
            "cotracker_backend": COTRACKER3_ONLINE,
            "cotracker_window_len": DEFAULT_COTRACKER_WINDOW_LEN,
            "cotracker_publish_step": DEFAULT_COTRACKER_PUBLISH_STEP,
            "tracking_query_count_actual_by_camera": tracking_worker.get("tracking_query_count_actual_by_camera", {}),
            "tracking_union_pixels_by_camera": tracking_worker.get("tracking_union_pixels_by_camera", {}),
            "tracking_object_pixels_by_camera": tracking_worker.get("tracking_object_pixels_by_camera", {}),
            "tracking_controller_pixels_by_camera": tracking_worker.get("tracking_controller_pixels_by_camera", {}),
            "tracking_sample_object_hits_by_camera": tracking_worker.get("tracking_sample_object_hits_by_camera", {}),
            "tracking_sample_controller_hits_by_camera": tracking_worker.get("tracking_sample_controller_hits_by_camera", {}),
            "tracking_sample_overlap_hits_by_camera": tracking_worker.get("tracking_sample_overlap_hits_by_camera", {}),
            "tracking_sample_background_hits_by_camera": tracking_worker.get("tracking_sample_background_hits_by_camera", {}),
            "cotracker_waiting_for_object_controller_by_camera": tracking_worker.get("cotracker_waiting_for_object_controller_by_camera", {}),
            "object_mask_nonempty_by_camera": tracking_worker.get("object_mask_nonempty_by_camera", {}),
            "controller_mask_nonempty_by_camera": tracking_worker.get("controller_mask_nonempty_by_camera", {}),
            "overlay_display_scope": tracking_worker.get(
                "overlay_display_scope",
                contract.get("overlay_display_scope", DEFAULT_OVERLAY_DISPLAY_SCOPE),
            ),
            "overlay_display_count_by_camera": tracking_worker.get("overlay_display_count_by_camera", {}),
            "overlay_display_object_count_by_camera": tracking_worker.get(
                "overlay_display_object_count_by_camera",
                {},
            ),
            "overlay_display_controller_count_by_camera": tracking_worker.get(
                "overlay_display_controller_count_by_camera",
                {},
            ),
        }
    )
    return summary


def make_demo3_live_runtime_class(shared_runtime_module: Any):
    base_cls = shared_runtime_module.Demo21Runtime

    class Demo3LiveRuntime(base_cls):
        def __init__(
            self,
            args: argparse.Namespace,
            *,
            demo3_contract: dict[str, Any],
            backend_factory: BackendFactory | None = None,
            cotracker_enabled: bool = True,
            stale_timeout_s: float = 0.5,
        ) -> None:
            super().__init__(args)
            self.demo3_contract = dict(demo3_contract)
            self.demo3_cotracker_enabled = bool(cotracker_enabled)
            self.demo3_stale_timeout_s = float(stale_timeout_s)
            self.demo3_overlay_ms_samples: list[float] = []
            self.demo3_tracking_input_slot: LatestTrackingInputSlot | None = None
            self.demo3_tracking_output_slot: LatestTrackingOverlaySlot | None = None
            self.demo3_cotracker_worker: CoTracker3OverlayWorker | None = None
            self.demo3_cotracker_thread: CoTracker3OverlayThread | None = None
            if self.demo3_cotracker_enabled:
                self.demo3_tracking_input_slot = LatestTrackingInputSlot()
                self.demo3_tracking_output_slot = LatestTrackingOverlaySlot()
                self.demo3_cotracker_worker = CoTracker3OverlayWorker(
                    camera_ids=tuple(int(item) for item in args.camera_ids),
                    backend_factory=backend_factory,
                    output_slot=self.demo3_tracking_output_slot,
                    query_mode=str(self.demo3_contract.get("tracking_query_mode", TRACKING_QUERY_MODE_PHYSTWIN_DENSE)),
                    query_count_request=str(
                        self.demo3_contract.get("tracking_query_count_requested", TRACKING_QUERY_COUNT_AUTO)
                    ),
                    seed=int(self.demo3_contract.get("cotracker_seed", DEFAULT_COTRACKER_SEED)),
                    sampling_device=str(getattr(args, "device", "cuda")),
                    init_requires_object_and_controller=True,
                    overlay_max_points_per_camera=int(
                        self.demo3_contract.get("overlay_max_points_per_camera", DEFAULT_OVERLAY_MAX_POINTS_PER_CAMERA)
                    ),
                    overlay_display_scope=str(
                        self.demo3_contract.get("overlay_display_scope", DEFAULT_OVERLAY_DISPLAY_SCOPE)
                    ),
                    device=str(getattr(args, "device", "cuda")),
                    update_mode=str(self.demo3_contract.get("cotracker_update_mode", DEFAULT_COTRACKER_UPDATE_MODE)),
                )
                self.demo3_cotracker_thread = CoTracker3OverlayThread(
                    worker=self.demo3_cotracker_worker,
                    input_slot=self.demo3_tracking_input_slot,
                    stop_event=self.stop_event,
                )

        def _start_threads(self) -> None:
            if self.demo3_cotracker_thread is not None:
                self.demo3_cotracker_thread.start()
            super()._start_threads()

        def stop(self) -> None:
            if self.demo3_cotracker_thread is not None:
                self.demo3_cotracker_thread.stop(timeout_s=1.0)
            super().stop()

        def _build_fused_packet(self, *, depth_group: Any, masks: dict[int, Any], ray_cache: dict[int, Any], rng: np.random.Generator):
            if self.demo3_tracking_input_slot is not None:
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
                        intrinsics_by_camera[idx] = np.asarray(self._stream_metadata[idx]["K_color"], dtype=np.float32).reshape(3, 3)
                    if idx in getattr(self, "_c2w_by_camera", {}):
                        c2w_by_camera[idx] = np.asarray(self._c2w_by_camera[idx], dtype=np.float32).reshape(4, 4)
                if rgb_by_camera and mask_by_camera:
                    self.demo3_tracking_input_slot.publish(
                        TrackingOverlayInputPacket(
                            group_id=int(depth_group.group_id),
                            frame_idx=int(max(depth_group.per_camera_frame_seq.values()) if depth_group.per_camera_frame_seq else depth_group.group_id),
                            timestamp_s=float(time.perf_counter()),
                            rgb_by_camera=rgb_by_camera,
                            mask_by_camera=mask_by_camera,
                            object_mask_by_camera=object_mask_by_camera,
                            controller_mask_by_camera=controller_mask_by_camera,
                            depth_by_camera=depth_by_camera,
                            intrinsics_by_camera=intrinsics_by_camera,
                            c2w_by_camera=c2w_by_camera,
                            depth_scale_m_per_unit=1.0,
                        )
                    )
            return super()._build_fused_packet(depth_group=depth_group, masks=masks, ray_cache=ray_cache, rng=rng)

        def _publish_render_packet(self, packet: Any) -> None:
            overlay_start_s = time.perf_counter()
            overlay = None
            overlay_points = np.empty((0, 3), dtype=np.float32)
            if self.demo3_tracking_output_slot is not None:
                overlay = self.demo3_tracking_output_slot.get_fresh(
                    now_s=float(time.perf_counter()),
                    stale_timeout_s=self.demo3_stale_timeout_s,
                )
                if overlay is not None and overlay.camera_tracks_world:
                    nonempty = [
                        np.asarray(points, dtype=np.float32).reshape(-1, 3)
                        for points in overlay.camera_tracks_world.values()
                        if np.asarray(points).size > 0
                    ]
                    if nonempty:
                        overlay_points = np.concatenate(nonempty, axis=0).astype(np.float32)
                        overlay_colors = np.repeat(OVERLAY_COLOR_RGB[None, :], len(overlay_points), axis=0)
                        packet = replace(
                            packet,
                            controller_points_m=np.concatenate([packet.controller_points_m, overlay_points], axis=0),
                            controller_colors_rgb=np.concatenate([packet.controller_colors_rgb, overlay_colors], axis=0),
                        )
            overlay_ms = float((time.perf_counter() - overlay_start_s) * 1000.0)
            self.demo3_overlay_ms_samples.append(overlay_ms)
            self._profile_update(
                packet.group_id,
                demo3_tracking_overlay={
                    "overlay_available": bool(overlay is not None),
                    "overlay_points": int(len(overlay_points)),
                    "overlay_ms": overlay_ms,
                    "render_waited_for_cotracker": False,
                },
            )
            super()._publish_render_packet(packet)

        def demo3_tracking_snapshot(self) -> dict[str, Any] | None:
            if self.demo3_cotracker_thread is None:
                return None
            return self.demo3_cotracker_thread.snapshot()

    return Demo3LiveRuntime


class Demo3Runtime:
    """Runtime facade for the Demo 3 realtime visualization contract.

    The hardware loop is intentionally composed from smaller shared helpers and
    workers. Dry-run and profile-only paths are deterministic so CI can validate
    the contract without RealSense hardware or CoTracker weights.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        *,
        shared_runtime_module: Any | None = None,
        shared_runtime_cls: type | None = None,
        connected_serials_provider: ConnectedSerialsProvider | None = None,
        backend_factory: BackendFactory | None = None,
    ) -> None:
        self.args = args
        self.contract = build_contract(args)
        self.shared_runtime_module = shared_runtime_module
        self.shared_runtime_cls = shared_runtime_cls
        self.connected_serials_provider = connected_serials_provider
        self.backend_factory = backend_factory

    def run(self) -> dict[str, Any]:
        live_validation = validate_live_realsense_contract(
            self.args,
            connected_serials_provider=self.connected_serials_provider,
        )
        shared = self.shared_runtime_module or _load_shared_runtime_module()
        shared_profile = _shared_profile_path(self.args)
        shared_args = build_shared_runtime_args(
            self.args,
            shared_runtime_module=shared,
            live_validation=live_validation,
            shared_profile_path=shared_profile,
        )
        runtime_cls = self.shared_runtime_cls or make_demo3_live_runtime_class(shared)
        if self.shared_runtime_cls is None:
            runtime = runtime_cls(
                shared_args,
                demo3_contract=self.contract,
                backend_factory=self.backend_factory,
                cotracker_enabled=not bool(self.args.disable_cotracker),
                stale_timeout_s=float(self.args.overlay_stale_timeout_ms) / 1000.0,
            )
        else:
            runtime = runtime_cls(shared_args)
        exit_code = int(runtime.run())
        shared_payload = _load_json_if_exists(shared_profile)
        tracking_snapshot = runtime.demo3_tracking_snapshot() if hasattr(runtime, "demo3_tracking_snapshot") else None
        overlay_samples = getattr(runtime, "demo3_overlay_ms_samples", [])
        summary = _build_demo3_live_summary(
            contract=self.contract,
            runtime=runtime,
            exit_code=exit_code,
            tracking_snapshot=tracking_snapshot,
            overlay_ms_samples=overlay_samples,
            shared_profile=shared_payload,
        )
        profile = {
            "contract": self.contract,
            "summary": summary,
            "live_validation": live_validation,
            "shared_runtime_profile": None if shared_profile is None else str(shared_profile),
            "shared_runtime_profile_payload": shared_payload,
            "tracking_snapshot": tracking_snapshot,
            "runtime_note": "Demo 3 non-dry-run delegates capture/mask/fusion/render to the shared three-view runtime and runs CoTracker as an async latest-wins overlay stage.",
            "exit_code": exit_code,
        }
        _write_profile(self.args.profile_json_output, profile)
        return profile


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    try:
        args = parser.parse_args(argv)
        args = apply_preset_defaults(args, explicit_options=_explicit_cli_options(argv))
        validate_args(args, require_calibration=False)
        contract = build_contract(args)
        if args.dry_run:
            print(format_contract(contract))
            _write_profile(args.profile_json_output, {"contract": contract, "summary": contract["profile_summary_fields"]})
            return 0
        profile = Demo3Runtime(args).run()
        print(json.dumps(profile["summary"], indent=2, sort_keys=True))
        return int(profile.get("exit_code", 0))
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


__all__ = [
    "COTRACKER3_ONLINE",
    "DEFAULT_COTRACKER_QUERY_COUNT_REQUEST",
    "DEFAULT_COTRACKER_PUBLISH_STEP",
    "DEFAULT_COTRACKER_SEED",
    "DEFAULT_COTRACKER_WINDOW_LEN",
    "DEFAULT_MODE",
    "DEPTH_SOURCE_REALSENSE",
    "Demo3Runtime",
    "MASK_SOURCE_HF_EDGETAM",
    "MODE_DEMO",
    "MODE_EXP",
    "MODES",
    "PHYSTWIN_DENSE_MAX_POINTS",
    "PRESET_DEMO3_REALSENSE_COTRACKER_HIGHFPS",
    "PRESET_DEMO3_REALSENSE_COTRACKER_PROFILE",
    "PRESET_DEMO3_REALSENSE_MASK_ONLY",
    "SHARED_TRACK_MODE_CONTROLLER_OBJECT",
    "TRACKING_QUERY_COUNT_AUTO",
    "TRACKING_QUERY_MODE_PHYSTWIN_DENSE",
    "TRACK_SCOPE_OBJECT_CONTROLLER_UNION",
    "apply_preset_defaults",
    "build_arg_parser",
    "build_contract",
    "build_shared_runtime_args",
    "build_shared_runtime_argv",
    "format_contract",
    "main",
    "make_demo3_live_runtime_class",
    "normalize_cotracker_query_count_request",
    "parse_camera_ids",
    "resolve_demo3_mode",
    "validate_live_realsense_contract",
    "validate_args",
]
