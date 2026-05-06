#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
import gc
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from typing import Any, Callable

import numpy as np


def _resolve_repo_root() -> Path:
    candidates: list[Path] = []
    env_root = os.environ.get("QQTT_REPO_ROOT")
    if env_root:
        candidates.append(Path(env_root))
    candidates.extend([Path(__file__).resolve().parents[1], Path.cwd()])
    for candidate in candidates:
        root = candidate.expanduser().resolve()
        if (root / "data_process").is_dir() and (root / "demo_v2").is_dir():
            return root
    return Path(__file__).resolve().parents[1]


REPO_ROOT = _resolve_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo_v2.realtime_single_camera_pointcloud import (  # noqa: E402
    CameraIntrinsics,
    CoalescedPostGate,
    ColorFloat32Buffer,
    DEFAULT_FFS_REPO,
    DEFAULT_FFS_TRT_TWO_STAGE_MODEL_DIR,
    FfsIrToColorAligner,
    LatestSlot,
    RenderStats,
    SUPPORTED_CAPTURE_FPS,
    SUPPORTED_PROFILES,
    _apply_emitter,
    _elapsed_ms,
    _load_open3d_modules,
    _load_realsense_module,
    apply_wslg_open3d_env_defaults,
    build_projection_grid,
    camera_intrinsics_from_rs,
    ensure_float32_c_contiguous,
    parse_profile,
    pointcloud_update_requires_readd,
    resolve_serial,
    rs_extrinsics_to_matrix,
    rs_intrinsics_to_matrix,
    rs_translation_norm,
    validate_ffs_paths,
    warm_up_numba_ffs_align,
)
from demo_v2.pcd_filter_fast import (  # noqa: E402
    AsyncPcdFilterWorker,
    FilterBudgetController,
    FilterInput,
    FilterOutput,
    voxel_cap_points,
    voxel_density_filter,
)
from services.ffs_remote import FfsRemoteDepthClient  # noqa: E402
from services.ffs_remote.protocol import (  # noqa: E402
    COMPRESSION_MODES,
    RETURN_TYPES,
    SPARSE_RETURN_TYPES,
)
from data_process.depth_backends.ffs_defaults import (  # noqa: E402
    DEFAULT_FFS_MAX_DISP,
    DEFAULT_FFS_MODEL_NAME,
    DEFAULT_FFS_TRT_BUILDER_OPTIMIZATION_LEVEL,
    DEFAULT_FFS_TRT_ENGINE_SIZE,
    DEFAULT_FFS_VALID_ITERS,
)


DEFAULT_MODEL_ID = "yonigozlan/EdgeTAM-hf"
DEFAULT_PROFILE = "848x480"
DEFAULT_FPS = 60
DEFAULT_DEVICE = "cuda"
DEFAULT_DTYPE = "bfloat16"
DEFAULT_COMPILE_MODE = "vision-reduce-overhead"
COMPILE_MODES = ("vision-reduce-overhead",)
INIT_MODES = ("sam31-first-frame", "saved-masks")
DEFAULT_INIT_MODE = "sam31-first-frame"
TRACK_MODES = ("controller-object", "object-only", "none")
DEFAULT_TRACK_MODE = "controller-object"
DEPTH_SOURCES = ("ffs", "ffs_remote", "realsense", "none")
DEFAULT_DEPTH_SOURCE = "ffs"
PCD_MODES = ("masked", "none")
DEFAULT_PCD_MODE = "masked"
RENDER_MODES = ("pointcloud", "none")
DEFAULT_RENDER_MODE = "pointcloud"
PCD_FILTER_MODES = ("async", "sync", "none")
PCD_FILTER_NONE = "none"
PCD_FILTER_PT_FILTER = "pt-filter"
PCD_FILTER_ENHANCED_PT = "enhanced-pt"
PCD_FILTER_VOXEL_DENSITY = "voxel-density"
PCD_FILTERS = (PCD_FILTER_NONE, PCD_FILTER_PT_FILTER, PCD_FILTER_ENHANCED_PT, PCD_FILTER_VOXEL_DENSITY)
DEMO_PRESETS = ("none", "local-ffs-professor")
DEFAULT_DEMO_PRESET = "none"
LOCAL_FFS_PROFESSOR_MAX_POINTS = 20000
LOCAL_FFS_PROFESSOR_POINT_SIZE = 2.5
LOCAL_FFS_PROFESSOR_LATENCY_TARGET_MS = 120.0
LOCAL_FFS_PROFESSOR_FILTER_CAP = 20000
DEFAULT_FILTER_RADIUS_M = 0.01
DEFAULT_FILTER_NB_POINTS = 40
DEFAULT_ENHANCED_COMPONENT_VOXEL_SIZE_M = 0.01
DEFAULT_ENHANCED_KEEP_NEAR_MAIN_GAP_M = 0.0
CONTROLLER_ID = 1
OBJECT_ID = 2
OBJECT_LABELS = {CONTROLLER_ID: "controller", OBJECT_ID: "object"}
CONTROLLER_COLOR_RGB = (255, 96, 32)
OBJECT_COLOR_RGB = (64, 180, 255)
GEOMETRY_CONTROLLER = "masked_edgetam_controller"
GEOMETRY_OBJECT = "masked_edgetam_object"
COORDINATE_FRAME = "camera_color_frame"
DEBUG_LOG_INTERVAL_S = 1.0
WARMUP_HUD_TEXT = (
    "System warming up. Keep one steady pose.\n"
    "SAM3.1 first-frame initialization and compiled EdgeTAM startup are running..."
)


@dataclass(frozen=True)
class PipelineTiming:
    wait_ms: float = 0.0
    align_ms: float = 0.0
    frame_copy_ms: float = 0.0
    ffs_ms: float = 0.0
    ffs_align_ms: float = 0.0
    remote_rtt_ms: float = 0.0
    remote_server_total_ms: float = 0.0
    remote_request_kb: float = 0.0
    remote_response_kb: float = 0.0
    depth_convert_ms: float = 0.0
    preprocess_ms: float = 0.0
    prompt_ms: float = 0.0
    model_ms: float = 0.0
    wall_model_ms: float = 0.0
    cuda_event_model_ms: float = 0.0
    pre_sync_wait_ms: float = 0.0
    post_sync_wait_ms: float = 0.0
    postprocess_ms: float = 0.0
    mask_ms: float = 0.0
    pcd_mask_intersection_ms: float = 0.0
    pcd_select_ms: float = 0.0
    pcd_point_cap_ms: float = 0.0
    pcd_backproject_ms: float = 0.0
    pcd_color_gather_ms: float = 0.0
    pcd_ms: float = 0.0
    pcd_filter_ms: float = 0.0
    object_filter_ms: float = 0.0
    controller_filter_ms: float = 0.0
    open3d_convert_ms: float = 0.0
    open3d_update_ms: float = 0.0
    receive_to_render_ms: float = 0.0


@dataclass(frozen=True)
class RealtimeCameraRuntime:
    pipeline: object
    align: object | None
    serial: str
    intrinsics: CameraIntrinsics
    depth_scale_m_per_unit: float
    k_color: np.ndarray
    k_ir_left: np.ndarray | None = None
    t_ir_left_to_color: np.ndarray | None = None
    ir_baseline_m: float = 0.0


@dataclass(frozen=True)
class FramePacket:
    seq: int
    color_bgr: np.ndarray
    depth_source: str
    intrinsics: CameraIntrinsics
    depth_scale_m_per_unit: float
    receive_perf_s: float
    timing: PipelineTiming
    depth_u16: np.ndarray | None = None
    ir_left_u8: np.ndarray | None = None
    ir_right_u8: np.ndarray | None = None
    k_ir_left: np.ndarray | None = None
    t_ir_left_to_color: np.ndarray | None = None
    k_color: np.ndarray | None = None
    ir_baseline_m: float = 0.0


@dataclass(frozen=True)
class MaskPacket:
    seq: int
    color_bgr: np.ndarray
    depth_source: str
    intrinsics: CameraIntrinsics
    depth_scale_m_per_unit: float
    receive_perf_s: float
    process_done_perf_s: float
    dropped_capture_frames: int
    timing: PipelineTiming
    controller_mask: np.ndarray
    object_mask: np.ndarray
    depth_u16: np.ndarray | None = None
    ir_left_u8: np.ndarray | None = None
    ir_right_u8: np.ndarray | None = None
    k_ir_left: np.ndarray | None = None
    t_ir_left_to_color: np.ndarray | None = None
    k_color: np.ndarray | None = None
    ir_baseline_m: float = 0.0


@dataclass(frozen=True)
class MaskedPcdPacket:
    seq: int
    controller_xyz_m: np.ndarray
    controller_colors_rgb_u8: np.ndarray
    object_xyz_m: np.ndarray
    object_colors_rgb_u8: np.ndarray
    intrinsics: CameraIntrinsics
    receive_perf_s: float
    process_done_perf_s: float
    dropped_capture_frames: int
    dropped_seg_frames: int
    timing: PipelineTiming
    filter_telemetry: PcdFilterTelemetry = field(default_factory=lambda: PcdFilterTelemetry())

    @property
    def controller_point_count(self) -> int:
        return int(self.controller_xyz_m.shape[0])

    @property
    def object_point_count(self) -> int:
        return int(self.object_xyz_m.shape[0])

    @property
    def point_count(self) -> int:
        return self.controller_point_count + self.object_point_count


@dataclass(frozen=True)
class PcdFilterTelemetry:
    enabled: bool = False
    mode: str = PCD_FILTER_NONE
    render_using_filtered: bool = False
    filter_seq: int = -1
    filter_age_frames: int = 0
    filter_age_ms: float = 0.0
    filter_ms: float = 0.0
    object_filter_ms: float = 0.0
    controller_filter_ms: float = 0.0
    object_raw_points: int = 0
    object_cap_points: int = 0
    object_output_points: int = 0
    controller_raw_points: int = 0
    controller_cap_points: int = 0
    controller_output_points: int = 0
    object_filter_cap: int = 0
    controller_filter_cap: int = 0
    filter_submit_fps: float = 0.0
    filter_output_fps: float = 0.0
    filter_queue_drop: int = 0
    filter_busy: bool = False


@dataclass(frozen=True)
class DepthProfilePacket:
    seq: int
    receive_perf_s: float
    process_done_perf_s: float
    dropped_capture_frames: int
    timing: PipelineTiming


@dataclass(frozen=True)
class RemoteFfsQualityPacket:
    seq: int
    receive_perf_s: float
    process_done_perf_s: float
    timing: PipelineTiming
    return_type: str
    sparse_points: int = 0


class StageStats:
    def __init__(self, window_s: float = 1.0) -> None:
        self.window_s = float(window_s)
        self._lock = threading.Lock()
        self._times: deque[float] = deque()

    def record(self, now_s: float | None = None) -> None:
        now = time.perf_counter() if now_s is None else float(now_s)
        with self._lock:
            self._times.append(now)
            cutoff = now - self.window_s
            while len(self._times) > 1 and self._times[0] < cutoff:
                self._times.popleft()

    @property
    def fps(self) -> float:
        with self._lock:
            if len(self._times) < 2:
                return 0.0
            elapsed = self._times[-1] - self._times[0]
            if elapsed <= 0:
                return 0.0
            return float((len(self._times) - 1) / elapsed)


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _parse_rgb_triplet(value: str) -> tuple[int, int, int]:
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    if len(items) != 3:
        raise argparse.ArgumentTypeError("expected R,G,B")
    try:
        rgb = tuple(int(item) for item in items)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected integer R,G,B") from exc
    if any(item < 0 or item > 255 for item in rgb):
        raise argparse.ArgumentTypeError("R,G,B values must be in [0, 255]")
    return rgb  # type: ignore[return-value]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Single-D455 realtime HF EdgeTAM masked point-cloud demo. Captures live "
            "RealSense color plus FFS stereo depth by default, tracks controller/object "
            "or object-only with one HF EdgeTAM streaming session, and renders only the masked PCD."
        )
    )
    parser.add_argument("--serial", default=None, help="Optional RealSense D400 serial. Defaults to first detected D400.")
    parser.add_argument("--profile", choices=SUPPORTED_PROFILES, default=DEFAULT_PROFILE, help="Capture profile.")
    parser.add_argument("--fps", choices=SUPPORTED_CAPTURE_FPS, type=int, default=DEFAULT_FPS, help="Capture FPS.")
    parser.add_argument(
        "--depth-source",
        choices=DEPTH_SOURCES,
        default=DEFAULT_DEPTH_SOURCE,
        help=(
            "Depth source. ffs streams color+IR stereo and runs local TensorRT FFS; "
            "ffs_remote streams color+IR stereo and requests color-aligned FFS depth from a remote service."
        ),
    )
    parser.add_argument(
        "--ffs-repo",
        type=Path,
        default=DEFAULT_FFS_REPO,
        help="Fast-FoundationStereo repo path. Used when --depth-source ffs.",
    )
    parser.add_argument(
        "--ffs-trt-model-dir",
        type=Path,
        default=DEFAULT_FFS_TRT_TWO_STAGE_MODEL_DIR,
        help=(
            "Two-stage TensorRT FFS engine directory. Default is the 20-30-48 / "
            "valid_iters=4 / 848x480->864x480 / builderOptimizationLevel=5 artifact."
        ),
    )
    parser.add_argument(
        "--ffs-trt-root",
        type=Path,
        default=None,
        help="Optional TensorRT Python package/root override forwarded to the FFS runner.",
    )
    parser.add_argument(
        "--ffs-remote-endpoint",
        default=None,
        help="ZeroMQ endpoint for --depth-source ffs_remote, for example tcp://100.x.y.z:7001.",
    )
    parser.add_argument(
        "--ffs-remote-max-inflight",
        type=int,
        default=1,
        help="Remote FFS request inflight cap. The first implementation supports only 1.",
    )
    parser.add_argument(
        "--ffs-remote-timeout-ms",
        type=int,
        default=80,
        help="Remote FFS send/receive timeout in milliseconds.",
    )
    parser.add_argument(
        "--ffs-remote-return",
        choices=RETURN_TYPES,
        default="depth_u16",
        help=(
            "Remote FFS response payload type. Full-frame types work as --depth-source ffs_remote; "
            "sparse types are for quality/protocol experiments."
        ),
    )
    parser.add_argument(
        "--ffs-remote-compress",
        choices=COMPRESSION_MODES,
        default="none",
        help="Compress remote FFS IR request payloads for --depth-source ffs_remote.",
    )
    parser.add_argument(
        "--enable-remote-ffs-quality",
        action="store_true",
        help="Run remote FFS as an asynchronous low-FPS quality side channel while main depth remains RealSense.",
    )
    parser.add_argument(
        "--remote-ffs-quality-endpoint",
        default=None,
        help="Remote FFS quality endpoint. Defaults to --ffs-remote-endpoint when omitted.",
    )
    parser.add_argument(
        "--remote-ffs-quality-return",
        choices=RETURN_TYPES,
        default="depth_u16",
        help="Remote quality side-channel return type.",
    )
    parser.add_argument(
        "--remote-ffs-quality-compress",
        choices=COMPRESSION_MODES,
        default="none",
        help="Compress remote quality side-channel IR/mask request payloads.",
    )
    parser.add_argument(
        "--remote-ffs-quality-timeout-ms",
        type=int,
        default=5000,
        help="Remote quality side-channel timeout in milliseconds.",
    )
    parser.add_argument(
        "--remote-ffs-quality-interval-ms",
        type=float,
        default=200.0,
        help="Minimum interval between remote FFS quality requests.",
    )
    parser.add_argument(
        "--emitter",
        choices=("auto", "on", "off"),
        default="auto",
        help="RealSense emitter policy. Defaults to leaving the current device setting unchanged.",
    )
    parser.add_argument(
        "--init-mode",
        choices=INIT_MODES,
        default=DEFAULT_INIT_MODE,
        help="Frame-0 initialization mode. Default runs SAM3.1 once on the live first frame.",
    )
    parser.add_argument(
        "--track-mode",
        choices=TRACK_MODES,
        default=DEFAULT_TRACK_MODE,
        help="Objects tracked by EdgeTAM. Use none for capture/depth isolation profiling.",
    )
    parser.add_argument(
        "--pcd-mode",
        choices=PCD_MODES,
        default=DEFAULT_PCD_MODE,
        help="Point-cloud stage mode. Use none for EdgeTAM/depth isolation profiling.",
    )
    parser.add_argument(
        "--render-mode",
        choices=RENDER_MODES,
        default=DEFAULT_RENDER_MODE,
        help="Render stage mode. Use none for headless profiling.",
    )
    parser.add_argument(
        "--demo-preset",
        choices=DEMO_PRESETS,
        default=DEFAULT_DEMO_PRESET,
        help=(
            "Optional display preset. local-ffs-professor keeps FFS-derived depth "
            "and compiled EdgeTAM, while capping rendered points for a steadier local demo."
        ),
    )
    parser.add_argument(
        "--controller-init-mask",
        default=None,
        help="Binary frame-0 controller mask PNG for explicit saved-masks debugging mode.",
    )
    parser.add_argument(
        "--object-init-mask",
        default=None,
        help="Binary frame-0 object mask PNG for explicit saved-masks debugging mode.",
    )
    parser.add_argument(
        "--controller-prompt",
        default="hand",
        help="SAM3.1 prompt label to union as controller in sam31-first-frame mode.",
    )
    parser.add_argument(
        "--object-prompt",
        default="stuffed animal",
        help="SAM3.1 prompt label to use as object in sam31-first-frame mode.",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="HF EdgeTAM model id.")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="Inference device, usually cuda.")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default=DEFAULT_DTYPE, help="Inference dtype.")
    parser.add_argument(
        "--compile-mode",
        choices=COMPILE_MODES,
        default=DEFAULT_COMPILE_MODE,
        help="Required EdgeTAM compile mode. Compiles only vision_encoder with reduce-overhead.",
    )
    parser.add_argument("--depth-min-m", type=float, default=0.2, help="Minimum valid depth in meters.")
    parser.add_argument("--depth-max-m", type=float, default=1.5, help="Maximum valid depth in meters. Use <=0 to disable.")
    parser.add_argument(
        "--pcd-max-points",
        type=int,
        default=60000,
        help="Max masked points per object. Use 0 to keep every masked valid depth pixel.",
    )
    parser.add_argument(
        "--pcd-stride",
        type=int,
        default=1,
        help="Optional masked PCD pixel stride. Use 2 for a faster lower-density profiling path.",
    )
    parser.add_argument(
        "--pcd-color-mode",
        choices=("rgb", "class"),
        default="rgb",
        help="Point-cloud colors. rgb uses the live color frame; class uses fixed controller/object colors.",
    )
    parser.add_argument(
        "--enable-pcd-filter",
        action="store_true",
        help="Enable capped point-cloud filtering. Async mode never blocks capture, EdgeTAM, FFS, or render.",
    )
    parser.add_argument(
        "--pcd-filter-mode",
        choices=PCD_FILTER_MODES,
        default="async",
        help="Point-cloud filter scheduling mode. Requires --enable-pcd-filter unless set to none.",
    )
    parser.add_argument("--object-filter", choices=PCD_FILTERS, default=PCD_FILTER_ENHANCED_PT)
    parser.add_argument("--controller-filter", choices=PCD_FILTERS, default=PCD_FILTER_PT_FILTER)
    parser.add_argument("--object-filter-cap", type=int, default=20_000)
    parser.add_argument("--controller-filter-cap", type=int, default=20_000)
    parser.add_argument("--object-filter-voxel-m", type=float, default=0.004)
    parser.add_argument("--controller-filter-voxel-m", type=float, default=0.003)
    parser.add_argument(
        "--filter-every-n",
        type=int,
        default=3,
        help="Submit capped PCD filtering every N PCD packets. Async mode renders the latest available filtered output.",
    )
    parser.add_argument(
        "--filter-budget-ms",
        type=float,
        default=12.0,
        help="Soft async filter budget. Caps are reduced conservatively when the worker exceeds this budget.",
    )
    parser.add_argument("--filter-min-cap", type=int, default=5_000)
    parser.add_argument(
        "--voxel-density-min-points",
        type=int,
        default=2,
        help="Minimum points per voxel for the realtime voxel-density approximate filter.",
    )
    parser.add_argument("--filter-radius-m", type=float, default=DEFAULT_FILTER_RADIUS_M)
    parser.add_argument("--filter-nb-points", type=int, default=DEFAULT_FILTER_NB_POINTS)
    parser.add_argument("--enhanced-component-voxel-size-m", type=float, default=DEFAULT_ENHANCED_COMPONENT_VOXEL_SIZE_M)
    parser.add_argument("--enhanced-keep-near-main-gap-m", type=float, default=DEFAULT_ENHANCED_KEEP_NEAR_MAIN_GAP_M)
    parser.add_argument("--point-size", type=float, default=2.0, help="Open3D point size.")
    parser.add_argument("--render-every-n", type=int, default=1, help="Render every Nth PCD packet.")
    parser.add_argument("--latency-target-ms", type=float, default=80.0, help="HUD latency target.")
    parser.add_argument("--duration-s", type=float, default=0.0, help="Optional auto-stop duration. Use 0 to run until closed.")
    parser.add_argument("--controller-color", type=_parse_rgb_triplet, default=CONTROLLER_COLOR_RGB, help="Controller RGB color.")
    parser.add_argument("--object-color", type=_parse_rgb_triplet, default=OBJECT_COLOR_RGB, help="Object RGB color.")
    parser.add_argument(
        "--profile-sync",
        action="store_true",
        help="Enable device-wide CUDA synchronizes around timed stages. Off by default for live hot path.",
    )
    parser.add_argument(
        "--profile-cuda-events",
        action="store_true",
        help="Record CUDA-event EdgeTAM model timing. Profiling-only; synchronizes the model end event.",
    )
    parser.add_argument("--debug", action="store_true", help="Print once-per-second timing/debug stats.")
    return parser


def apply_demo_preset(args: argparse.Namespace) -> argparse.Namespace:
    if args.demo_preset == "local-ffs-professor":
        if int(args.pcd_max_points) == 60000:
            args.pcd_max_points = LOCAL_FFS_PROFESSOR_MAX_POINTS
        if float(args.point_size) == 2.0:
            args.point_size = LOCAL_FFS_PROFESSOR_POINT_SIZE
        if float(args.latency_target_ms) == 80.0:
            args.latency_target_ms = LOCAL_FFS_PROFESSOR_LATENCY_TARGET_MS
        if int(args.object_filter_cap) == 20_000:
            args.object_filter_cap = LOCAL_FFS_PROFESSOR_FILTER_CAP
        if int(args.controller_filter_cap) == 20_000:
            args.controller_filter_cap = LOCAL_FFS_PROFESSOR_FILTER_CAP
    return args


def pcd_filter_enabled(args: argparse.Namespace) -> bool:
    return bool(args.enable_pcd_filter) and str(args.pcd_filter_mode) != "none"


def validate_args(args: argparse.Namespace) -> None:
    parse_profile(args.profile)
    if args.depth_source not in DEPTH_SOURCES:
        raise ValueError(f"--depth-source must be one of {', '.join(DEPTH_SOURCES)}")
    if args.demo_preset == "local-ffs-professor" and args.depth_source != "ffs":
        raise ValueError("--demo-preset local-ffs-professor requires --depth-source ffs")
    if args.depth_min_m < 0:
        raise ValueError("--depth-min-m must be >= 0")
    if args.depth_max_m > 0 and args.depth_max_m <= args.depth_min_m:
        raise ValueError("--depth-max-m must be <=0 or greater than --depth-min-m")
    if args.pcd_max_points < 0:
        raise ValueError("--pcd-max-points must be >= 0")
    if args.pcd_stride < 1:
        raise ValueError("--pcd-stride must be >= 1")
    if args.render_every_n < 1:
        raise ValueError("--render-every-n must be >= 1")
    if args.point_size <= 0:
        raise ValueError("--point-size must be positive")
    if args.pcd_filter_mode not in PCD_FILTER_MODES:
        raise ValueError(f"--pcd-filter-mode must be one of {', '.join(PCD_FILTER_MODES)}")
    for flag in ("object_filter_cap", "controller_filter_cap", "filter_min_cap"):
        if int(getattr(args, flag)) < 0:
            raise ValueError(f"--{flag.replace('_', '-')} must be >= 0")
    if int(args.object_filter_cap) > 0 and int(args.filter_min_cap) > int(args.object_filter_cap):
        raise ValueError("--filter-min-cap must be <= --object-filter-cap when object cap is enabled")
    if int(args.controller_filter_cap) > 0 and int(args.filter_min_cap) > int(args.controller_filter_cap):
        raise ValueError("--filter-min-cap must be <= --controller-filter-cap when controller cap is enabled")
    if float(args.object_filter_voxel_m) <= 0:
        raise ValueError("--object-filter-voxel-m must be positive")
    if float(args.controller_filter_voxel_m) <= 0:
        raise ValueError("--controller-filter-voxel-m must be positive")
    if int(args.filter_every_n) < 1:
        raise ValueError("--filter-every-n must be >= 1")
    if float(args.filter_budget_ms) < 0:
        raise ValueError("--filter-budget-ms must be >= 0")
    if int(args.voxel_density_min_points) < 1:
        raise ValueError("--voxel-density-min-points must be >= 1")
    if float(args.filter_radius_m) <= 0:
        raise ValueError("--filter-radius-m must be positive")
    if int(args.filter_nb_points) < 1:
        raise ValueError("--filter-nb-points must be >= 1")
    if float(args.enhanced_component_voxel_size_m) <= 0:
        raise ValueError("--enhanced-component-voxel-size-m must be positive")
    if float(args.enhanced_keep_near_main_gap_m) < 0:
        raise ValueError("--enhanced-keep-near-main-gap-m must be >= 0")
    if pcd_filter_enabled(args) and args.pcd_mode != "masked":
        raise ValueError("--enable-pcd-filter requires --pcd-mode masked")
    if args.compile_mode != DEFAULT_COMPILE_MODE:
        raise ValueError("Demo 2.0 requires compiled EdgeTAM: --compile-mode vision-reduce-overhead")
    if args.track_mode == "none" and args.pcd_mode == "masked":
        raise ValueError("--track-mode none requires --pcd-mode none")
    if args.depth_source == "none" and args.pcd_mode == "masked":
        raise ValueError("--depth-source none requires --pcd-mode none")
    if args.render_mode == "pointcloud" and args.pcd_mode == "none":
        raise ValueError("--render-mode pointcloud requires --pcd-mode masked")
    if args.depth_source == "ffs":
        validate_ffs_paths(ffs_repo=Path(args.ffs_repo), model_dir=Path(args.ffs_trt_model_dir))
    if args.depth_source == "ffs_remote":
        if not args.ffs_remote_endpoint:
            raise ValueError("--depth-source ffs_remote requires --ffs-remote-endpoint")
        if int(args.ffs_remote_max_inflight) != 1:
            raise ValueError("first ffs_remote implementation requires --ffs-remote-max-inflight 1")
        if int(args.ffs_remote_timeout_ms) <= 0:
            raise ValueError("--ffs-remote-timeout-ms must be positive")
        if args.ffs_remote_return in SPARSE_RETURN_TYPES:
            if args.track_mode == "none":
                raise ValueError("sparse --depth-source ffs_remote requires EdgeTAM masks; use --track-mode object-only or controller-object")
            if args.pcd_mode != "masked":
                raise ValueError("sparse --depth-source ffs_remote requires --pcd-mode masked")
    if args.enable_remote_ffs_quality:
        if args.depth_source != "realsense":
            raise ValueError("--enable-remote-ffs-quality requires --depth-source realsense")
        if not (args.remote_ffs_quality_endpoint or args.ffs_remote_endpoint):
            raise ValueError("--enable-remote-ffs-quality requires --remote-ffs-quality-endpoint or --ffs-remote-endpoint")
        if int(args.remote_ffs_quality_timeout_ms) <= 0:
            raise ValueError("--remote-ffs-quality-timeout-ms must be positive")
        if float(args.remote_ffs_quality_interval_ms) <= 0:
            raise ValueError("--remote-ffs-quality-interval-ms must be positive")
        if args.remote_ffs_quality_return in SPARSE_RETURN_TYPES and args.track_mode == "none":
            raise ValueError("sparse remote quality returns require EdgeTAM masks; use --track-mode object-only or controller-object")
    if args.track_mode not in TRACK_MODES:
        raise ValueError(f"--track-mode must be one of {', '.join(TRACK_MODES)}")
    if args.init_mode == "saved-masks":
        if not args.object_init_mask:
            raise ValueError("saved-masks mode requires --object-init-mask")
        if controller_tracking_enabled(args) and not args.controller_init_mask:
            raise ValueError("saved-masks controller-object mode requires --controller-init-mask")
        required_masks = [("--object-init-mask", args.object_init_mask)]
        if controller_tracking_enabled(args):
            required_masks.append(("--controller-init-mask", args.controller_init_mask))
        for flag, value in required_masks:
            path = _resolve_path(value)
            if not path.is_file():
                raise ValueError(f"{flag} does not exist: {path}")


def _start_realsense_pipeline(args: argparse.Namespace) -> RealtimeCameraRuntime:
    rs = _load_realsense_module()
    width, height = parse_profile(args.profile)
    serial = resolve_serial(rs, args.serial)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, int(args.fps))
    if args.depth_source in {"ffs", "ffs_remote"} or bool(getattr(args, "enable_remote_ffs_quality", False)):
        config.enable_stream(rs.stream.infrared, 1, width, height, rs.format.y8, int(args.fps))
        config.enable_stream(rs.stream.infrared, 2, width, height, rs.format.y8, int(args.fps))
    if args.depth_source == "realsense":
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, int(args.fps))
    profile = pipeline.start(config)
    try:
        _apply_emitter(profile, args.emitter, rs)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = float(depth_sensor.get_depth_scale())
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intrinsics = camera_intrinsics_from_rs(color_stream.get_intrinsics())
        k_color = rs_intrinsics_to_matrix(color_stream.get_intrinsics())
        if args.depth_source in {"ffs", "ffs_remote"} or bool(getattr(args, "enable_remote_ffs_quality", False)):
            ir_left_profile = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
            ir_right_profile = profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()
            ir_left_to_right = ir_left_profile.get_extrinsics_to(ir_right_profile)
            ir_left_to_color = ir_left_profile.get_extrinsics_to(color_stream)
            if args.depth_source == "realsense":
                align = rs.align(rs.stream.color)
                return RealtimeCameraRuntime(
                    pipeline=pipeline,
                    align=align,
                    serial=serial,
                    intrinsics=intrinsics,
                    depth_scale_m_per_unit=depth_scale,
                    k_color=k_color,
                    k_ir_left=rs_intrinsics_to_matrix(ir_left_profile.get_intrinsics()),
                    t_ir_left_to_color=rs_extrinsics_to_matrix(ir_left_to_color),
                    ir_baseline_m=rs_translation_norm(ir_left_to_right),
                )
            return RealtimeCameraRuntime(
                pipeline=pipeline,
                align=None,
                serial=serial,
                intrinsics=intrinsics,
                depth_scale_m_per_unit=depth_scale,
                k_color=k_color,
                k_ir_left=rs_intrinsics_to_matrix(ir_left_profile.get_intrinsics()),
                t_ir_left_to_color=rs_extrinsics_to_matrix(ir_left_to_color),
                ir_baseline_m=rs_translation_norm(ir_left_to_right),
            )
        if args.depth_source == "none":
            return RealtimeCameraRuntime(
                pipeline=pipeline,
                align=None,
                serial=serial,
                intrinsics=intrinsics,
                depth_scale_m_per_unit=depth_scale,
                k_color=k_color,
            )
        align = rs.align(rs.stream.color)
    except Exception:
        pipeline.stop()
        raise
    return RealtimeCameraRuntime(
        pipeline=pipeline,
        align=align,
        serial=serial,
        intrinsics=intrinsics,
        depth_scale_m_per_unit=depth_scale,
        k_color=k_color,
    )


def _load_gray_image(path: Path) -> np.ndarray:
    try:
        from PIL import Image

        return np.asarray(Image.open(path).convert("L"))
    except Exception as exc:
        raise ValueError(f"failed to load mask image {path}: {exc}") from exc


def load_binary_mask(path: str | Path, *, expected_shape: tuple[int, int]) -> np.ndarray:
    mask_path = _resolve_path(path)
    image = _load_gray_image(mask_path)
    if image.ndim != 2:
        raise ValueError(f"mask must be a 2D image: {mask_path}")
    if tuple(image.shape) != tuple(expected_shape):
        raise ValueError(f"mask shape {tuple(image.shape)} does not match frame shape {tuple(expected_shape)}: {mask_path}")
    return np.ascontiguousarray(image > 0)


def _masked_sample_indices(
    *,
    depth_m: np.ndarray,
    mask: np.ndarray,
    depth_min_m: float,
    depth_max_m: float,
    max_points: int,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if depth_m.ndim != 2 or mask.ndim != 2:
        raise ValueError("depth_m and mask must be 2D arrays")
    if depth_m.shape != mask.shape:
        raise ValueError("depth and mask shapes must match")
    if max_points < 0:
        raise ValueError("max_points must be >= 0")
    valid = np.isfinite(depth_m) & (depth_m > np.float32(depth_min_m))
    if depth_max_m > 0:
        valid &= depth_m < np.float32(depth_max_m)
    selected = valid & np.asarray(mask, dtype=bool)
    if not np.any(selected):
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    rows, cols = np.nonzero(selected)
    if max_points > 0 and rows.shape[0] > max_points:
        generator = rng if rng is not None else np.random.default_rng()
        indices = generator.choice(rows.shape[0], int(max_points), replace=False)
        rows = rows[indices]
        cols = cols[indices]
    return rows.astype(np.int64, copy=False), cols.astype(np.int64, copy=False)


def backproject_masked_rgbd(
    *,
    color_bgr: np.ndarray,
    depth_m: np.ndarray,
    mask: np.ndarray,
    ray_x: np.ndarray,
    ray_y: np.ndarray,
    depth_min_m: float,
    depth_max_m: float,
    max_points: int,
    color_mode: str,
    class_rgb: tuple[int, int, int],
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if color_bgr.ndim != 3 or color_bgr.shape[2] != 3:
        raise ValueError("color_bgr must be an HxWx3 array")
    if depth_m.shape != color_bgr.shape[:2]:
        raise ValueError("color and depth shapes must match")
    if depth_m.shape != ray_x.shape or depth_m.shape != ray_y.shape:
        raise ValueError("depth and projection grids must have matching shapes")
    if color_mode not in {"rgb", "class"}:
        raise ValueError("color_mode must be 'rgb' or 'class'")

    rows, cols = _masked_sample_indices(
        depth_m=depth_m,
        mask=mask,
        depth_min_m=depth_min_m,
        depth_max_m=depth_max_m,
        max_points=max_points,
        rng=rng,
    )
    if rows.size == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    z = depth_m[rows, cols].astype(np.float32, copy=False)
    x = ray_x[rows, cols].astype(np.float32, copy=False) * z
    y = ray_y[rows, cols].astype(np.float32, copy=False) * z
    points = np.ascontiguousarray(np.stack([x, y, z], axis=1), dtype=np.float32)
    if color_mode == "rgb":
        colors = np.ascontiguousarray(color_bgr[rows, cols, ::-1], dtype=np.uint8)
    else:
        colors = make_solid_colors(points.shape[0], class_rgb)
    return points, colors


def backproject_masked_rgbd_profiled(
    *,
    color_bgr: np.ndarray,
    depth_m: np.ndarray,
    mask: np.ndarray,
    ray_x: np.ndarray,
    ray_y: np.ndarray,
    depth_min_m: float,
    depth_max_m: float,
    max_points: int,
    color_mode: str,
    class_rgb: tuple[int, int, int],
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    if color_bgr.ndim != 3 or color_bgr.shape[2] != 3:
        raise ValueError("color_bgr must be an HxWx3 array")
    if depth_m.shape != color_bgr.shape[:2] or depth_m.shape != mask.shape:
        raise ValueError("color, depth, and mask shapes must match")
    if depth_m.shape != ray_x.shape or depth_m.shape != ray_y.shape:
        raise ValueError("depth and projection grids must have matching shapes")
    if max_points < 0:
        raise ValueError("max_points must be >= 0")
    if color_mode not in {"rgb", "class"}:
        raise ValueError("color_mode must be 'rgb' or 'class'")

    timing: dict[str, float] = {}
    started_s = time.perf_counter()
    valid = np.isfinite(depth_m) & (depth_m > np.float32(depth_min_m))
    if depth_max_m > 0:
        valid &= depth_m < np.float32(depth_max_m)
    selected = valid & np.asarray(mask, dtype=bool)
    timing["pcd_mask_intersection_ms"] = _elapsed_ms(started_s, time.perf_counter())

    started_s = time.perf_counter()
    if not np.any(selected):
        timing["pcd_select_ms"] = _elapsed_ms(started_s, time.perf_counter())
        timing["pcd_point_cap_ms"] = 0.0
        timing["pcd_backproject_ms"] = 0.0
        timing["pcd_color_gather_ms"] = 0.0
        timing["pcd_raw_points"] = 0.0
        timing["pcd_cap_points"] = 0.0
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8), timing
    rows, cols = np.nonzero(selected)
    timing["pcd_raw_points"] = float(rows.shape[0])
    timing["pcd_select_ms"] = _elapsed_ms(started_s, time.perf_counter())

    started_s = time.perf_counter()
    if max_points > 0 and rows.shape[0] > max_points:
        generator = rng if rng is not None else np.random.default_rng()
        indices = generator.choice(rows.shape[0], int(max_points), replace=False)
        rows = rows[indices]
        cols = cols[indices]
    rows = rows.astype(np.int64, copy=False)
    cols = cols.astype(np.int64, copy=False)
    timing["pcd_cap_points"] = float(rows.shape[0])
    timing["pcd_point_cap_ms"] = _elapsed_ms(started_s, time.perf_counter())

    started_s = time.perf_counter()
    z = depth_m[rows, cols].astype(np.float32, copy=False)
    x = ray_x[rows, cols].astype(np.float32, copy=False) * z
    y = ray_y[rows, cols].astype(np.float32, copy=False) * z
    points = np.ascontiguousarray(np.stack([x, y, z], axis=1), dtype=np.float32)
    timing["pcd_backproject_ms"] = _elapsed_ms(started_s, time.perf_counter())

    started_s = time.perf_counter()
    if color_mode == "rgb":
        colors = np.ascontiguousarray(color_bgr[rows, cols, ::-1], dtype=np.uint8)
    else:
        colors = make_solid_colors(points.shape[0], class_rgb)
    timing["pcd_color_gather_ms"] = _elapsed_ms(started_s, time.perf_counter())
    return points, colors, timing


def backproject_masked(
    *,
    depth_m: np.ndarray,
    mask: np.ndarray,
    ray_x: np.ndarray,
    ray_y: np.ndarray,
    depth_min_m: float,
    depth_max_m: float,
    max_points: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if depth_m.shape != ray_x.shape or depth_m.shape != ray_y.shape:
        raise ValueError("depth and projection grids must have matching shapes")
    rows, cols = _masked_sample_indices(
        depth_m=depth_m,
        mask=mask,
        depth_min_m=depth_min_m,
        depth_max_m=depth_max_m,
        max_points=max_points,
        rng=rng,
    )
    if rows.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    z = depth_m[rows, cols].astype(np.float32, copy=False)
    x = ray_x[rows, cols].astype(np.float32, copy=False) * z
    y = ray_y[rows, cols].astype(np.float32, copy=False) * z
    return np.ascontiguousarray(np.stack([x, y, z], axis=1), dtype=np.float32)


def make_solid_colors(point_count: int, rgb: tuple[int, int, int]) -> np.ndarray:
    if point_count <= 0:
        return np.empty((0, 3), dtype=np.uint8)
    color = np.asarray(rgb, dtype=np.uint8).reshape(1, 3)
    return np.repeat(color, int(point_count), axis=0)


def controller_tracking_enabled(args_or_track_mode: argparse.Namespace | str) -> bool:
    track_mode = args_or_track_mode if isinstance(args_or_track_mode, str) else args_or_track_mode.track_mode
    return str(track_mode) == "controller-object"


def object_id_labels(track_mode: str = DEFAULT_TRACK_MODE) -> dict[int, str]:
    if track_mode == "none":
        return {}
    if track_mode == "object-only":
        return {OBJECT_ID: OBJECT_LABELS[OBJECT_ID]}
    if track_mode == "controller-object":
        return dict(OBJECT_LABELS)
    raise ValueError(f"unsupported track mode: {track_mode}")


def active_object_ids(args: argparse.Namespace) -> list[int]:
    return list(object_id_labels(str(args.track_mode)).keys())


def _coerce_object_ids(value: Any) -> list[int]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (int, np.integer)):
        return [int(value)]
    return [int(item) for item in list(value)]


def _extract_binary_mask(mask_tensor: Any) -> np.ndarray:
    value = mask_tensor
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    array = np.asarray(value)
    array = np.squeeze(array)
    if array.ndim != 2:
        raise RuntimeError(f"expected 2D mask after squeeze, got {array.shape}")
    return np.ascontiguousarray(array > 0)


def extract_object_masks_from_hf_output(output: Any, post_masks: Any) -> dict[int, np.ndarray]:
    object_ids = _coerce_object_ids(getattr(output, "object_ids"))
    if len(object_ids) != len(post_masks):
        raise RuntimeError(f"HF output object_ids length {len(object_ids)} != mask length {len(post_masks)}")
    return {int(obj_id): _extract_binary_mask(post_masks[idx]) for idx, obj_id in enumerate(object_ids)}


def _load_hf_streaming_runtime() -> Any:
    from scripts.harness.experiments import run_hf_edgetam_streaming_realcase as hf_stream

    hf_stream._load_runtime_dependencies()
    return hf_stream


def _sync_if_needed(torch_module: Any, device: str) -> None:
    if str(device).startswith("cuda") and torch_module.cuda.is_available():
        torch_module.cuda.synchronize()


def _time_runtime_ms(
    torch_module: Any,
    device: str,
    fn: Callable[[], Any],
    *,
    sync_enabled: bool = False,
) -> tuple[Any, float, float, float]:
    pre_sync_ms = 0.0
    post_sync_ms = 0.0
    if sync_enabled:
        sync_start_s = time.perf_counter()
        _sync_if_needed(torch_module, device)
        pre_sync_ms = _elapsed_ms(sync_start_s, time.perf_counter())
    started = time.perf_counter()
    value = fn()
    elapsed_ms = _elapsed_ms(started, time.perf_counter())
    if sync_enabled:
        sync_start_s = time.perf_counter()
        _sync_if_needed(torch_module, device)
        post_sync_ms = _elapsed_ms(sync_start_s, time.perf_counter())
    return value, elapsed_ms, pre_sync_ms, post_sync_ms


def _time_model_forward(
    *,
    torch_module: Any,
    device: str,
    profile_sync: bool,
    profile_cuda_events: bool,
    fn: Callable[[], Any],
) -> tuple[Any, float, float, float, float]:
    pre_sync_ms = 0.0
    post_sync_ms = 0.0
    if profile_sync:
        sync_start_s = time.perf_counter()
        _sync_if_needed(torch_module, device)
        pre_sync_ms = _elapsed_ms(sync_start_s, time.perf_counter())

    start_event = None
    end_event = None
    if profile_cuda_events and str(device).startswith("cuda") and torch_module.cuda.is_available():
        start_event = torch_module.cuda.Event(enable_timing=True)
        end_event = torch_module.cuda.Event(enable_timing=True)
        start_event.record()

    started_s = time.perf_counter()
    value = fn()
    wall_ms = _elapsed_ms(started_s, time.perf_counter())

    cuda_event_ms = 0.0
    if end_event is not None and start_event is not None:
        end_event.record()
        end_event.synchronize()
        cuda_event_ms = float(start_event.elapsed_time(end_event))

    if profile_sync:
        sync_start_s = time.perf_counter()
        _sync_if_needed(torch_module, device)
        post_sync_ms = _elapsed_ms(sync_start_s, time.perf_counter())
    return value, wall_ms, cuda_event_ms, pre_sync_ms, post_sync_ms


def _bgr_to_pil_rgb(color_bgr: np.ndarray) -> Any:
    from PIL import Image

    return Image.fromarray(np.ascontiguousarray(color_bgr[:, :, ::-1]))


def _write_first_frame_case(color_bgr: np.ndarray, root: Path) -> Path:
    case_dir = root / "sam31_frame0_case"
    color_dir = case_dir / "color" / "0"
    color_dir.mkdir(parents=True, exist_ok=True)
    image = _bgr_to_pil_rgb(color_bgr)
    image.save(color_dir / "0.png")
    return case_dir


def _load_label_masks_from_sam31_root(
    *,
    mask_root: Path,
    label: str,
    frame_token: str = "0",
    camera_idx: int = 0,
) -> list[np.ndarray]:
    info_path = mask_root / "mask" / f"mask_info_{int(camera_idx)}.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"SAM3.1 mask info not found: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    label_norm = str(label).strip().lower()
    masks: list[np.ndarray] = []
    for obj_id, obj_label in info.items():
        if str(obj_label).strip().lower() != label_norm:
            continue
        mask_path = mask_root / "mask" / str(int(camera_idx)) / str(obj_id) / f"{frame_token}.png"
        if not mask_path.is_file():
            raise FileNotFoundError(f"SAM3.1 mask image not found: {mask_path}")
        masks.append(np.ascontiguousarray(_load_gray_image(mask_path) > 0))
    return masks


def _union_masks(masks: list[np.ndarray], *, label: str) -> np.ndarray:
    if not masks:
        raise RuntimeError(f"SAM3.1 did not produce a mask for label {label!r}")
    output = np.zeros_like(masks[0], dtype=bool)
    for mask in masks:
        if mask.shape != output.shape:
            raise RuntimeError("SAM3.1 masks for one label have inconsistent shapes")
        output |= mask.astype(bool)
    return np.ascontiguousarray(output)


def release_sam31_runtime_resources(device: str = DEFAULT_DEVICE) -> None:
    helper = sys.modules.get("scripts.harness.sam31_mask_helper")
    autocast_context = getattr(helper, "_CUDA_AUTOCAST_CONTEXT", None) if helper is not None else None
    if autocast_context is not None:
        try:
            autocast_context.__exit__(None, None, None)
        except Exception as exc:
            print(f"[WARN] SAM3.1 autocast cleanup failed: {type(exc).__name__}: {exc}", flush=True)
        if helper is not None:
            setattr(helper, "_CUDA_AUTOCAST_CONTEXT", None)

    gc.collect()
    try:
        import torch  # noqa: PLC0415

        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception as exc:
        print(f"[WARN] SAM3.1 CUDA cleanup failed: {type(exc).__name__}: {exc}", flush=True)


def run_sam31_first_frame_masks(color_bgr: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    with tempfile.TemporaryDirectory(prefix="qqtt_sam31_first_frame_") as tmp:
        root = Path(tmp)
        case_dir = _write_first_frame_case(color_bgr, root)
        output_dir = root / "sam31_masks"
        from scripts.harness.sam31_mask_helper import run_case_segmentation

        prompt_labels = [str(args.object_prompt)]
        if controller_tracking_enabled(args):
            prompt_labels.append(str(args.controller_prompt))
        try:
            run_case_segmentation(
                case_root=case_dir,
                text_prompt=",".join(prompt_labels),
                camera_ids=(0,),
                output_dir=output_dir,
                source_mode="frames",
                checkpoint_path=None,
                ann_frame_index=0,
                keep_session_frames=False,
                session_root=None,
                overwrite=True,
                async_loading_frames=False,
                compile_model=False,
                max_num_objects=16,
            )
        finally:
            release_sam31_runtime_resources(str(args.device))
        object_masks = _load_label_masks_from_sam31_root(mask_root=output_dir, label=args.object_prompt)
        object_mask = _union_masks(
            object_masks,
            label=args.object_prompt,
        )
        if not controller_tracking_enabled(args):
            return np.zeros_like(object_mask, dtype=bool), object_mask
        controller_masks = _load_label_masks_from_sam31_root(mask_root=output_dir, label=args.controller_prompt)
        return _union_masks(controller_masks, label=args.controller_prompt), object_mask


def resolve_initial_masks(frame: FramePacket, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    expected_shape = tuple(frame.color_bgr.shape[:2])
    if args.init_mode == "saved-masks":
        object_mask = load_binary_mask(args.object_init_mask, expected_shape=expected_shape)
        if not controller_tracking_enabled(args):
            return np.zeros_like(object_mask, dtype=bool), object_mask
        controller_mask = load_binary_mask(args.controller_init_mask, expected_shape=expected_shape)
        return controller_mask, object_mask
    if args.init_mode == "sam31-first-frame":
        controller_mask, object_mask = run_sam31_first_frame_masks(frame.color_bgr, args)
        if controller_mask.shape != expected_shape or object_mask.shape != expected_shape:
            raise RuntimeError("SAM3.1 frame-0 masks do not match captured frame shape")
        return controller_mask, object_mask
    raise ValueError(f"unsupported init mode: {args.init_mode}")


class RealtimeMaskedEdgeTamPcdDemo:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.width, self.height = parse_profile(args.profile)
        self.runtime: RealtimeCameraRuntime | None = None
        self.ray_x: np.ndarray | None = None
        self.ray_y: np.ndarray | None = None
        self.capture_slot: LatestSlot[FramePacket] = LatestSlot()
        self.mask_slot: LatestSlot[MaskPacket] = LatestSlot()
        self.depth_profile_slot: LatestSlot[DepthProfilePacket] = LatestSlot()
        self.remote_quality_slot: LatestSlot[RemoteFfsQualityPacket] = LatestSlot()
        self.render_slot: LatestSlot[MaskedPcdPacket] = LatestSlot()
        self.stop_event = threading.Event()
        self._threads: list[threading.Thread] = []
        self._request_render_update: Callable[[], None] = lambda: None
        self.capture_stats = StageStats()
        self.seg_stats = StageStats()
        self.depth_stats = StageStats()
        self.remote_quality_stats = StageStats()
        self.pcd_stats = StageStats()
        self.filter_submit_stats = StageStats()
        self.filter_output_stats = StageStats()
        self.render_stats = RenderStats()
        self.filter_worker: AsyncPcdFilterWorker | None = None
        self._filter_submit_skip_count = 0
        self._last_filter_output_seq_recorded = -1
        self.object_filter_budget = FilterBudgetController(
            target_ms=max(0.0, float(args.filter_budget_ms)) * 0.5,
            min_cap=int(args.filter_min_cap),
            max_cap=max(int(args.filter_min_cap), int(args.object_filter_cap) if int(args.object_filter_cap) > 0 else 200_000),
            init_cap=int(args.object_filter_cap) if int(args.object_filter_cap) > 0 else 200_000,
        )
        self.controller_filter_budget = FilterBudgetController(
            target_ms=max(0.0, float(args.filter_budget_ms)) * 0.5,
            min_cap=int(args.filter_min_cap),
            max_cap=max(int(args.filter_min_cap), int(args.controller_filter_cap) if int(args.controller_filter_cap) > 0 else 200_000),
            init_cap=int(args.controller_filter_cap) if int(args.controller_filter_cap) > 0 else 200_000,
        )
        self._last_debug_log_s = 0.0
        self.ffs_runner: object | None = None
        self.ir_to_color_aligner: FfsIrToColorAligner | None = None
        self._ir_to_color_aligner_key: tuple[
            tuple[int, int],
            tuple[int, int],
            tuple[float, ...],
            tuple[float, ...],
            tuple[float, ...],
        ] | None = None
        self.ffs_remote_client: FfsRemoteDepthClient | None = None
        self.remote_quality_client: FfsRemoteDepthClient | None = None
        self._warned_remote_engine_contract = False

    @property
    def intrinsics(self) -> CameraIntrinsics:
        if self.runtime is None:
            raise RuntimeError("camera runtime is not initialized")
        return self.runtime.intrinsics

    @property
    def serial(self) -> str:
        if self.runtime is None:
            return "<not-started>"
        return self.runtime.serial

    def run(self) -> int:
        apply_wslg_open3d_env_defaults()
        if self.args.depth_source == "ffs":
            self.ffs_runner = self._create_ffs_runner()
            warm_up_numba_ffs_align()
        elif self.args.depth_source == "ffs_remote":
            self.ffs_remote_client = FfsRemoteDepthClient(
                endpoint=str(self.args.ffs_remote_endpoint),
                timeout_ms=int(self.args.ffs_remote_timeout_ms),
                return_type=str(self.args.ffs_remote_return),
                compression=str(self.args.ffs_remote_compress),
                max_inflight=int(self.args.ffs_remote_max_inflight),
            )
        if self.args.enable_remote_ffs_quality:
            endpoint = str(self.args.remote_ffs_quality_endpoint or self.args.ffs_remote_endpoint)
            self.remote_quality_client = FfsRemoteDepthClient(
                endpoint=endpoint,
                timeout_ms=int(self.args.remote_ffs_quality_timeout_ms),
                return_type=str(self.args.remote_ffs_quality_return),
                compression=str(self.args.remote_ffs_quality_compress),
                max_inflight=1,
            )
        if pcd_filter_enabled(self.args) and str(self.args.pcd_filter_mode) == "async":
            self.filter_worker = AsyncPcdFilterWorker(self._filter_pcd_input)
            self.filter_worker.start()
        self.runtime = _start_realsense_pipeline(self.args)
        try:
            self.ray_x, self.ray_y = build_projection_grid(
                width=self.width,
                height=self.height,
                stride=1,
                intrinsics=self.runtime.intrinsics,
            )
            if self.args.render_mode == "none":
                self._run_headless()
            else:
                self._run_open3d_viewer()
        finally:
            self.stop()
        return 0

    def stop(self) -> None:
        self.stop_event.set()
        for thread in list(self._threads):
            if thread.is_alive():
                thread.join(timeout=1.0)
        self._threads.clear()
        if self.runtime is not None:
            try:
                self.runtime.pipeline.stop()
            except Exception:
                pass
            self.runtime = None
        if self.ffs_remote_client is not None:
            self.ffs_remote_client.close()
            self.ffs_remote_client = None
        if self.remote_quality_client is not None:
            self.remote_quality_client.close()
            self.remote_quality_client = None
        if self.filter_worker is not None:
            self.filter_worker.stop()
            self.filter_worker = None

    def _create_ffs_runner(self) -> object:
        try:
            from data_process.depth_backends import FastFoundationStereoTensorRTRunner

            return FastFoundationStereoTensorRTRunner(
                ffs_repo=Path(self.args.ffs_repo),
                model_dir=Path(self.args.ffs_trt_model_dir),
                trt_root=None if self.args.ffs_trt_root is None else Path(self.args.ffs_trt_root),
            )
        except Exception as exc:
            raise RuntimeError(f"failed to start FFS TensorRT runner: {type(exc).__name__}: {exc}") from exc

    def _get_ir_to_color_aligner(
        self,
        *,
        depth_shape: tuple[int, int],
        color_shape: tuple[int, int],
        k_ir_left: np.ndarray,
        t_ir_left_to_color: np.ndarray,
        k_color: np.ndarray,
    ) -> FfsIrToColorAligner:
        k_ir = np.asarray(k_ir_left, dtype=np.float32).reshape(3, 3)
        transform = np.asarray(t_ir_left_to_color, dtype=np.float32).reshape(4, 4)
        k_col = np.asarray(k_color, dtype=np.float32).reshape(3, 3)
        key = (
            (int(depth_shape[0]), int(depth_shape[1])),
            (int(color_shape[0]), int(color_shape[1])),
            tuple(float(v) for v in k_ir.ravel()),
            tuple(float(v) for v in transform.ravel()),
            tuple(float(v) for v in k_col.ravel()),
        )
        if self._ir_to_color_aligner_key != key or self.ir_to_color_aligner is None:
            self.ir_to_color_aligner = FfsIrToColorAligner(
                k_ir_left=k_ir,
                t_ir_left_to_color=transform,
                k_color=k_col,
                ir_shape=depth_shape,
                color_shape=color_shape,
            )
            self._ir_to_color_aligner_key = key
        return self.ir_to_color_aligner

    def _start_threads(self) -> None:
        workers: list[tuple[str, Callable[[], None]]] = [("capture", self._capture_worker)]
        if self.args.track_mode != "none":
            workers.append(("seg", self._seg_worker))
        if self.args.pcd_mode == "masked":
            workers.append(("pcd", self._pcd_worker))
        elif self.args.depth_source in {"ffs", "ffs_remote"}:
            workers.append(("depth", self._depth_profile_worker))
        if self.args.enable_remote_ffs_quality:
            workers.append(("remote-quality", self._remote_ffs_quality_worker))
        if self.args.debug and self.args.render_mode == "none":
            workers.append(("debug", self._headless_debug_worker))
        for name, target in workers:
            thread = threading.Thread(target=target, name=f"masked-edgetam-{name}", daemon=True)
            thread.start()
            self._threads.append(thread)

    def _run_headless(self) -> None:
        self._start_threads()
        started_s = time.perf_counter()
        try:
            while not self.stop_event.is_set():
                if self.args.duration_s > 0 and time.perf_counter() - started_s >= float(self.args.duration_s):
                    self.stop_event.set()
                    break
                time.sleep(0.05)
        except KeyboardInterrupt:
            self.stop_event.set()

    def _capture_worker(self) -> None:
        assert self.runtime is not None
        seq = 0
        pipeline = self.runtime.pipeline
        align = self.runtime.align
        while not self.stop_event.is_set():
            wait_start_s = time.perf_counter()
            try:
                frames = pipeline.wait_for_frames()
            except Exception as exc:
                if not self.stop_event.is_set():
                    print(f"[ERROR] RealSense capture failed: {type(exc).__name__}: {exc}", flush=True)
                self.stop_event.set()
                break
            receive_perf_s = time.perf_counter()
            align_start_s = receive_perf_s
            if self.args.depth_source in {"ffs", "ffs_remote"}:
                align_done_s = receive_perf_s
                color_frame = frames.get_color_frame()
                ir_left_frame = frames.get_infrared_frame(1)
                ir_right_frame = frames.get_infrared_frame(2)
                if not color_frame or not ir_left_frame or not ir_right_frame:
                    continue
                depth_frame = None
            elif self.args.depth_source == "none":
                align_done_s = receive_perf_s
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                depth_frame = None
                ir_left_frame = None
                ir_right_frame = None
            else:
                assert align is not None
                aligned = align.process(frames)
                align_done_s = time.perf_counter()
                color_frame = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue
                if self.args.enable_remote_ffs_quality:
                    ir_left_frame = frames.get_infrared_frame(1)
                    ir_right_frame = frames.get_infrared_frame(2)
                    if not ir_left_frame or not ir_right_frame:
                        continue
                else:
                    ir_left_frame = None
                    ir_right_frame = None
            copy_start_s = time.perf_counter()
            color_bgr = np.ascontiguousarray(np.asanyarray(color_frame.get_data()).copy())
            if self.args.depth_source in {"ffs", "ffs_remote"}:
                assert ir_left_frame is not None and ir_right_frame is not None
                depth_u16 = None
                ir_left_u8 = np.ascontiguousarray(np.asanyarray(ir_left_frame.get_data()).copy())
                ir_right_u8 = np.ascontiguousarray(np.asanyarray(ir_right_frame.get_data()).copy())
            elif self.args.depth_source == "none":
                depth_u16 = None
                ir_left_u8 = None
                ir_right_u8 = None
            else:
                assert depth_frame is not None
                depth_u16 = np.ascontiguousarray(np.asanyarray(depth_frame.get_data()).copy())
                if self.args.enable_remote_ffs_quality:
                    assert ir_left_frame is not None and ir_right_frame is not None
                    ir_left_u8 = np.ascontiguousarray(np.asanyarray(ir_left_frame.get_data()).copy())
                    ir_right_u8 = np.ascontiguousarray(np.asanyarray(ir_right_frame.get_data()).copy())
                else:
                    ir_left_u8 = None
                    ir_right_u8 = None
            copy_done_s = time.perf_counter()
            packet = FramePacket(
                seq=seq,
                color_bgr=color_bgr,
                depth_source=str(self.args.depth_source),
                intrinsics=self.runtime.intrinsics,
                depth_scale_m_per_unit=self.runtime.depth_scale_m_per_unit,
                receive_perf_s=receive_perf_s,
                timing=PipelineTiming(
                    wait_ms=_elapsed_ms(wait_start_s, receive_perf_s),
                    align_ms=_elapsed_ms(align_start_s, align_done_s),
                    frame_copy_ms=_elapsed_ms(copy_start_s, copy_done_s),
                ),
                depth_u16=depth_u16,
                ir_left_u8=ir_left_u8,
                ir_right_u8=ir_right_u8,
                k_ir_left=self.runtime.k_ir_left,
                t_ir_left_to_color=self.runtime.t_ir_left_to_color,
                k_color=self.runtime.k_color,
                ir_baseline_m=self.runtime.ir_baseline_m,
            )
            self.capture_slot.put(packet)
            self.capture_stats.record(copy_done_s)
            seq += 1

    def _init_hf_model(self) -> tuple[Any, Any, Any, Any, Any]:
        hf_stream = _load_hf_streaming_runtime()
        torch_module = hf_stream.torch
        if str(self.args.device).startswith("cuda") and not torch_module.cuda.is_available():
            raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
        dtype = hf_stream._dtype_from_name(self.args.dtype)
        model = hf_stream.EdgeTamVideoModel.from_pretrained(self.args.model_id).to(
            self.args.device,
            dtype=dtype,
        )
        model.eval()
        model, compile_metadata = hf_stream._apply_compile_mode(model, self.args.compile_mode)
        processor = hf_stream.Sam2VideoProcessor.from_pretrained(self.args.model_id)
        metadata = {
            "edge_model": self.args.model_id,
            "demo_preset": self.args.demo_preset,
            "compile_mode": self.args.compile_mode,
            "applied_targets": compile_metadata.get("applied_targets", []),
            "dtype": self.args.dtype,
            "inference_device": self.args.device,
            "inference_state_device": self.args.device,
            "video_storage_device": self.args.device,
            "frame_by_frame_streaming": True,
            "offline_video_input_used": False,
            "track_mode": self.args.track_mode,
            "depth_source": self.args.depth_source,
            "ffs_remote_endpoint": self.args.ffs_remote_endpoint if self.args.depth_source == "ffs_remote" else None,
            "ffs_remote_return": self.args.ffs_remote_return if self.args.depth_source == "ffs_remote" else None,
            "ffs_remote_compress": self.args.ffs_remote_compress if self.args.depth_source == "ffs_remote" else None,
            "remote_ffs_quality_enabled": bool(self.args.enable_remote_ffs_quality),
            "remote_ffs_quality_endpoint": (
                self.args.remote_ffs_quality_endpoint or self.args.ffs_remote_endpoint
                if self.args.enable_remote_ffs_quality
                else None
            ),
            "remote_ffs_quality_return": (
                self.args.remote_ffs_quality_return if self.args.enable_remote_ffs_quality else None
            ),
            "remote_ffs_quality_compress": (
                self.args.remote_ffs_quality_compress if self.args.enable_remote_ffs_quality else None
            ),
            "pcd_mode": self.args.pcd_mode,
            "pcd_max_points": int(self.args.pcd_max_points),
            "pcd_stride": int(self.args.pcd_stride),
            "pcd_filter_enabled": pcd_filter_enabled(self.args),
            "pcd_filter_mode": self.args.pcd_filter_mode if pcd_filter_enabled(self.args) else PCD_FILTER_NONE,
            "object_filter": self.args.object_filter,
            "controller_filter": self.args.controller_filter,
            "object_filter_cap": int(self.args.object_filter_cap),
            "controller_filter_cap": int(self.args.controller_filter_cap),
            "filter_every_n": int(self.args.filter_every_n),
            "filter_budget_ms": float(self.args.filter_budget_ms),
            "render_mode": self.args.render_mode,
            "render_every_n": int(self.args.render_every_n),
        }
        print(
            "[edgetam] "
            f"model={self.args.model_id} device={self.args.device} dtype={self.args.dtype} "
            f"track_mode={self.args.track_mode} compile_mode={self.args.compile_mode} "
            f"applied={compile_metadata.get('applied_targets', [])}",
            flush=True,
        )
        print(f"[edgetam-metadata] {json.dumps(metadata, sort_keys=True)}", flush=True)
        return hf_stream, torch_module, dtype, model, processor

    def _seg_worker(self) -> None:
        try:
            hf_stream, torch_module, dtype, model, processor = self._init_hf_model()
            first_frame = self._wait_for_first_frame()
            if first_frame is None:
                return
            controller_mask, object_mask = resolve_initial_masks(first_frame, self.args)
            session = hf_stream.EdgeTamVideoInferenceSession(
                video=None,
                video_height=int(first_frame.color_bgr.shape[0]),
                video_width=int(first_frame.color_bgr.shape[1]),
                inference_device=self.args.device,
                inference_state_device=self.args.device,
                video_storage_device=self.args.device,
                dtype=dtype,
            )
            with torch_module.inference_mode():
                first_packet = self._run_segmentation_frame(
                    hf_stream=hf_stream,
                    torch_module=torch_module,
                    dtype=dtype,
                    model=model,
                    processor=processor,
                    session=session,
                    frame=first_frame,
                    initial_controller_mask=controller_mask,
                    initial_object_mask=object_mask,
                    add_prompt=True,
                )
                self.mask_slot.put(first_packet)
                self.seg_stats.record(first_packet.process_done_perf_s)
                last_seq = first_frame.seq
                while not self.stop_event.is_set():
                    frame = self.capture_slot.get_latest_after(last_seq)
                    if frame is None:
                        time.sleep(0.001)
                        continue
                    last_seq = frame.seq
                    try:
                        packet = self._run_segmentation_frame(
                            hf_stream=hf_stream,
                            torch_module=torch_module,
                            dtype=dtype,
                            model=model,
                            processor=processor,
                            session=session,
                            frame=frame,
                            initial_controller_mask=controller_mask,
                            initial_object_mask=object_mask,
                            add_prompt=False,
                        )
                    except Exception as exc:
                        print(f"[ERROR] EdgeTAM segmentation failed: {type(exc).__name__}: {exc}", flush=True)
                        self.stop_event.set()
                        break
                    self.mask_slot.put(packet)
                    self.seg_stats.record(packet.process_done_perf_s)
        except Exception as exc:
            if not self.stop_event.is_set():
                print(f"[ERROR] segmentation worker failed: {type(exc).__name__}: {exc}", flush=True)
            self.stop_event.set()

    def _wait_for_first_frame(self) -> FramePacket | None:
        while not self.stop_event.is_set():
            frame = self.capture_slot.get_latest_after(-1)
            if frame is not None:
                return frame
            time.sleep(0.005)
        return None

    def _autocast_context(self, torch_module: Any) -> Any:
        if not str(self.args.device).startswith("cuda") or self.args.dtype == "float32":
            return nullcontext()
        dtype = torch_module.bfloat16 if self.args.dtype == "bfloat16" else torch_module.float16
        return torch_module.autocast("cuda", dtype=dtype)

    def _run_segmentation_frame(
        self,
        *,
        hf_stream: Any,
        torch_module: Any,
        dtype: Any,
        model: Any,
        processor: Any,
        session: Any,
        frame: FramePacket,
        initial_controller_mask: np.ndarray,
        initial_object_mask: np.ndarray,
        add_prompt: bool,
    ) -> MaskPacket:
        image = _bgr_to_pil_rgb(frame.color_bgr)
        inputs, preprocess_ms, preprocess_pre_sync_ms, preprocess_post_sync_ms = _time_runtime_ms(
            torch_module,
            self.args.device,
            lambda: processor(images=image, device=self.args.device, return_tensors="pt"),
            sync_enabled=bool(self.args.profile_sync),
        )
        pixel_values = inputs.pixel_values[0].to(device=self.args.device, dtype=dtype)
        prompt_ms = 0.0
        with self._autocast_context(torch_module):
            if add_prompt:
                prompt_obj_ids: list[int] = []
                prompt_masks: list[np.ndarray] = []
                if controller_tracking_enabled(self.args):
                    prompt_obj_ids.append(CONTROLLER_ID)
                    prompt_masks.append(np.asarray(initial_controller_mask, dtype=bool))
                prompt_obj_ids.append(OBJECT_ID)
                prompt_masks.append(np.asarray(initial_object_mask, dtype=bool))
                _unused, prompt_ms, prompt_pre_sync_ms, prompt_post_sync_ms = _time_runtime_ms(
                    torch_module,
                    self.args.device,
                    lambda: processor.add_inputs_to_inference_session(
                        inference_session=session,
                        frame_idx=0,
                        obj_ids=prompt_obj_ids,
                        input_masks=prompt_masks,
                    ),
                    sync_enabled=bool(self.args.profile_sync),
                )
            else:
                prompt_pre_sync_ms = 0.0
                prompt_post_sync_ms = 0.0
            output, wall_model_ms, cuda_event_model_ms, model_pre_sync_ms, model_post_sync_ms = _time_model_forward(
                torch_module=torch_module,
                device=self.args.device,
                profile_sync=bool(self.args.profile_sync),
                profile_cuda_events=bool(self.args.profile_cuda_events),
                fn=lambda: model(inference_session=session, frame=pixel_values),
            )
            post_masks, postprocess_ms, postprocess_pre_sync_ms, postprocess_post_sync_ms = _time_runtime_ms(
                torch_module,
                self.args.device,
                lambda: processor.post_process_masks(
                    [output.pred_masks],
                    original_sizes=inputs.original_sizes,
                    binarize=False,
                )[0],
                sync_enabled=bool(self.args.profile_sync),
            )
        masks_by_id = extract_object_masks_from_hf_output(output, post_masks)
        missing = [obj_id for obj_id in active_object_ids(self.args) if obj_id not in masks_by_id]
        if missing:
            raise RuntimeError(f"HF output missing tracked object ids: {missing}")
        object_mask = masks_by_id[OBJECT_ID]
        controller_mask = masks_by_id.get(CONTROLLER_ID)
        if controller_mask is None:
            controller_mask = np.zeros_like(object_mask, dtype=bool)
        process_done_s = time.perf_counter()
        timing = replace(
            frame.timing,
            preprocess_ms=preprocess_ms,
            prompt_ms=prompt_ms,
            model_ms=wall_model_ms,
            wall_model_ms=wall_model_ms,
            cuda_event_model_ms=cuda_event_model_ms,
            pre_sync_wait_ms=float(preprocess_pre_sync_ms + prompt_pre_sync_ms + model_pre_sync_ms + postprocess_pre_sync_ms),
            post_sync_wait_ms=float(preprocess_post_sync_ms + prompt_post_sync_ms + model_post_sync_ms + postprocess_post_sync_ms),
            postprocess_ms=postprocess_ms,
            mask_ms=float(preprocess_ms + prompt_ms + wall_model_ms + postprocess_ms),
        )
        return MaskPacket(
            seq=frame.seq,
            color_bgr=frame.color_bgr,
            depth_source=frame.depth_source,
            intrinsics=frame.intrinsics,
            depth_scale_m_per_unit=frame.depth_scale_m_per_unit,
            receive_perf_s=frame.receive_perf_s,
            process_done_perf_s=process_done_s,
            dropped_capture_frames=self.capture_slot.dropped_count,
            timing=timing,
            controller_mask=controller_mask,
            object_mask=object_mask,
            depth_u16=frame.depth_u16,
            ir_left_u8=frame.ir_left_u8,
            ir_right_u8=frame.ir_right_u8,
            k_ir_left=frame.k_ir_left,
            t_ir_left_to_color=frame.t_ir_left_to_color,
            k_color=frame.k_color,
            ir_baseline_m=frame.ir_baseline_m,
        )

    def _make_filter_input(
        self,
        *,
        seq: int,
        object_xyz: np.ndarray,
        object_colors: np.ndarray,
        controller_xyz: np.ndarray,
        controller_colors: np.ndarray,
    ) -> FilterInput:
        object_cap = 0 if int(self.args.object_filter_cap) == 0 else int(self.object_filter_budget.cap)
        controller_cap = 0 if int(self.args.controller_filter_cap) == 0 else int(self.controller_filter_budget.cap)
        return FilterInput(
            seq=int(seq),
            object_xyz=np.asarray(object_xyz, dtype=np.float32),
            object_rgb=np.asarray(object_colors, dtype=np.uint8),
            controller_xyz=np.asarray(controller_xyz, dtype=np.float32),
            controller_rgb=np.asarray(controller_colors, dtype=np.uint8),
            object_cap=object_cap,
            controller_cap=controller_cap,
            object_voxel_size_m=float(self.args.object_filter_voxel_m),
            controller_voxel_size_m=float(self.args.controller_filter_voxel_m),
        )

    def _apply_single_pcd_filter(
        self,
        *,
        points: np.ndarray,
        colors: np.ndarray,
        mode: str,
        cap: int,
        voxel_size_m: float,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        raw_points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        raw_colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
        cap_start_s = time.perf_counter()
        capped_points, capped_colors_or_none = voxel_cap_points(
            raw_points,
            raw_colors,
            max_points=int(cap),
            voxel_size_m=float(voxel_size_m),
            rng=rng,
        )
        capped_colors = np.asarray(capped_colors_or_none, dtype=np.uint8).reshape(-1, 3)
        cap_ms = _elapsed_ms(cap_start_s, time.perf_counter())

        filter_start_s = time.perf_counter()
        if mode == PCD_FILTER_NONE:
            filtered_points = np.asarray(capped_points, dtype=np.float32).reshape(-1, 3)
            filtered_colors = capped_colors
        elif mode == PCD_FILTER_VOXEL_DENSITY:
            density_points, density_colors_or_none = voxel_density_filter(
                capped_points,
                capped_colors,
                voxel_size_m=float(voxel_size_m),
                min_points_per_voxel=int(self.args.voxel_density_min_points),
            )
            filtered_points = np.asarray(density_points, dtype=np.float32).reshape(-1, 3)
            filtered_colors = np.asarray(density_colors_or_none, dtype=np.uint8).reshape(-1, 3)
        elif mode == PCD_FILTER_PT_FILTER:
            from data_process.visualization.experiments.ffs_confidence_filter_pcd_compare import (
                _apply_phystwin_like_radius_postprocess,
            )

            filtered_points, filtered_colors, _unused_stats = _apply_phystwin_like_radius_postprocess(
                points=capped_points,
                colors=capped_colors,
                enabled=True,
                radius_m=float(self.args.filter_radius_m),
                nb_points=int(self.args.filter_nb_points),
            )
        elif mode == PCD_FILTER_ENHANCED_PT:
            from data_process.visualization.experiments.ffs_confidence_filter_pcd_compare import (
                _apply_enhanced_phystwin_like_postprocess,
            )

            filtered_points, filtered_colors, _unused_stats = _apply_enhanced_phystwin_like_postprocess(
                points=capped_points,
                colors=capped_colors,
                enabled=True,
                radius_m=float(self.args.filter_radius_m),
                nb_points=int(self.args.filter_nb_points),
                component_voxel_size_m=float(self.args.enhanced_component_voxel_size_m),
                keep_near_main_gap_m=float(self.args.enhanced_keep_near_main_gap_m),
            )
        else:
            raise ValueError(f"unsupported PCD filter mode: {mode}")

        filter_ms = _elapsed_ms(filter_start_s, time.perf_counter())
        filtered_points = np.ascontiguousarray(filtered_points, dtype=np.float32).reshape(-1, 3)
        filtered_colors = np.ascontiguousarray(filtered_colors, dtype=np.uint8).reshape(-1, 3)
        return filtered_points, filtered_colors, {
            "mode": str(mode),
            "raw_points": int(len(raw_points)),
            "cap_points": int(len(capped_points)),
            "output_points": int(len(filtered_points)),
            "cap": int(cap),
            "voxel_size_m": float(voxel_size_m),
            "cap_ms": float(cap_ms),
            "filter_ms": float(filter_ms),
        }

    def _filter_pcd_input(self, item: FilterInput) -> FilterOutput:
        started_s = time.perf_counter()
        object_points, object_colors, object_stats = self._apply_single_pcd_filter(
            points=item.object_xyz,
            colors=item.object_rgb,
            mode=str(self.args.object_filter),
            cap=int(item.object_cap),
            voxel_size_m=float(item.object_voxel_size_m),
            rng=np.random.default_rng(int(item.seq) * 2 + 17),
        )
        controller_points, controller_colors, controller_stats = self._apply_single_pcd_filter(
            points=item.controller_xyz,
            colors=item.controller_rgb,
            mode=str(self.args.controller_filter),
            cap=int(item.controller_cap),
            voxel_size_m=float(item.controller_voxel_size_m),
            rng=np.random.default_rng(int(item.seq) * 2 + 19),
        )
        done_s = time.perf_counter()
        filter_ms = _elapsed_ms(started_s, done_s)
        if float(self.args.filter_budget_ms) > 0:
            self.object_filter_budget.update(float(object_stats["filter_ms"] + object_stats["cap_ms"]))
            self.controller_filter_budget.update(float(controller_stats["filter_ms"] + controller_stats["cap_ms"]))
        return FilterOutput(
            seq=int(item.seq),
            object_xyz=object_points,
            object_rgb=object_colors,
            controller_xyz=controller_points,
            controller_rgb=controller_colors,
            filter_ms=float(filter_ms),
            created_perf_s=float(item.created_perf_s),
            output_perf_s=done_s,
            stats={
                "object": object_stats,
                "controller": controller_stats,
                "object_filter": str(self.args.object_filter),
                "controller_filter": str(self.args.controller_filter),
            },
        )

    def _filter_worker_stats(self) -> dict[str, Any]:
        worker = self.filter_worker
        if worker is None:
            return {
                "busy": False,
                "submit_fps": self.filter_submit_stats.fps,
                "output_fps": self.filter_output_stats.fps,
                "pending_replace_count": 0,
            }
        stats = worker.stats
        return {
            "busy": bool(stats.get("busy", False)),
            "submit_fps": float(stats.get("submit_fps", self.filter_submit_stats.fps)),
            "output_fps": float(stats.get("output_fps", self.filter_output_stats.fps)),
            "pending_replace_count": int(stats.get("pending_replace_count", 0)) + int(self._filter_submit_skip_count),
        }

    def _filter_telemetry_from_output(
        self,
        *,
        packet_seq: int,
        output: FilterOutput | None,
        using_filtered: bool,
        object_raw_points: int,
        object_cap_points: int,
        controller_raw_points: int,
        controller_cap_points: int,
    ) -> PcdFilterTelemetry:
        worker_stats = self._filter_worker_stats()
        if output is None:
            return PcdFilterTelemetry(
                enabled=pcd_filter_enabled(self.args),
                mode=str(self.args.pcd_filter_mode if pcd_filter_enabled(self.args) else PCD_FILTER_NONE),
                object_raw_points=int(object_raw_points),
                object_cap_points=int(object_cap_points),
                object_output_points=int(object_cap_points),
                controller_raw_points=int(controller_raw_points),
                controller_cap_points=int(controller_cap_points),
                controller_output_points=int(controller_cap_points),
                object_filter_cap=int(self.object_filter_budget.cap),
                controller_filter_cap=int(self.controller_filter_budget.cap),
                filter_submit_fps=float(worker_stats["submit_fps"]),
                filter_output_fps=float(worker_stats["output_fps"]),
                filter_queue_drop=int(worker_stats["pending_replace_count"]),
                filter_busy=bool(worker_stats["busy"]),
            )

        object_stats = dict(output.stats.get("object", {}))
        controller_stats = dict(output.stats.get("controller", {}))
        age_ms = max(0.0, _elapsed_ms(output.output_perf_s, time.perf_counter()))
        return PcdFilterTelemetry(
            enabled=pcd_filter_enabled(self.args),
            mode=str(self.args.pcd_filter_mode),
            render_using_filtered=bool(using_filtered),
            filter_seq=int(output.seq),
            filter_age_frames=max(0, int(packet_seq) - int(output.seq)),
            filter_age_ms=float(age_ms),
            filter_ms=float(output.filter_ms),
            object_filter_ms=float(object_stats.get("filter_ms", 0.0)),
            controller_filter_ms=float(controller_stats.get("filter_ms", 0.0)),
            object_raw_points=int(object_stats.get("raw_points", object_raw_points)),
            object_cap_points=int(object_stats.get("cap_points", object_cap_points)),
            object_output_points=int(object_stats.get("output_points", object_cap_points)),
            controller_raw_points=int(controller_stats.get("raw_points", controller_raw_points)),
            controller_cap_points=int(controller_stats.get("cap_points", controller_cap_points)),
            controller_output_points=int(controller_stats.get("output_points", controller_cap_points)),
            object_filter_cap=int(object_stats.get("cap", self.object_filter_budget.cap)),
            controller_filter_cap=int(controller_stats.get("cap", self.controller_filter_budget.cap)),
            filter_submit_fps=float(worker_stats["submit_fps"]),
            filter_output_fps=float(worker_stats["output_fps"]),
            filter_queue_drop=int(worker_stats["pending_replace_count"]),
            filter_busy=bool(worker_stats["busy"]),
        )

    def _pcd_worker(self) -> None:
        last_seq = -1
        rng = np.random.default_rng()
        assert self.ray_x is not None and self.ray_y is not None
        ray_x = self.ray_x
        ray_y = self.ray_y
        while not self.stop_event.is_set():
            mask_packet = self.mask_slot.get_latest_after(last_seq)
            if mask_packet is None:
                time.sleep(0.001)
                continue
            last_seq = mask_packet.seq
            start_s = time.perf_counter()
            ffs_ms = 0.0
            ffs_align_ms = 0.0
            remote_rtt_ms = 0.0
            remote_server_total_ms = 0.0
            remote_request_kb = 0.0
            remote_response_kb = 0.0
            depth_convert_ms = 0.0
            if mask_packet.depth_source in {"ffs", "ffs_remote"}:
                if mask_packet.depth_source == "ffs_remote" and self.args.ffs_remote_return in SPARSE_RETURN_TYPES:
                    try:
                        packet = self._compute_remote_sparse_pcd_packet(
                            mask_packet=mask_packet,
                            start_s=start_s,
                            rng=rng,
                        )
                    except Exception as exc:
                        if not self.stop_event.is_set():
                            print(f"[WARN] sparse remote FFS frame {mask_packet.seq} failed: {type(exc).__name__}: {exc}", flush=True)
                        continue
                    self.render_slot.put(packet)
                    self.pcd_stats.record(packet.process_done_perf_s)
                    if packet.seq % int(self.args.render_every_n) == 0:
                        self._request_render_update()
                    continue
                try:
                    (
                        depth_m,
                        ffs_ms,
                        ffs_align_ms,
                        remote_rtt_ms,
                        remote_server_total_ms,
                        remote_request_kb,
                        remote_response_kb,
                    ) = self._compute_external_ffs_depth_color_m(mask_packet)
                except Exception as exc:
                    if not self.stop_event.is_set():
                        print(f"[WARN] FFS depth frame {mask_packet.seq} failed: {type(exc).__name__}: {exc}", flush=True)
                    continue
            else:
                if mask_packet.depth_u16 is None:
                    continue
                depth_convert_start_s = time.perf_counter()
                depth_m = np.ascontiguousarray(
                    mask_packet.depth_u16.astype(np.float32) * np.float32(mask_packet.depth_scale_m_per_unit)
                )
                depth_convert_ms = _elapsed_ms(depth_convert_start_s, time.perf_counter())

            stride = int(self.args.pcd_stride)
            if stride > 1:
                color_bgr = mask_packet.color_bgr[::stride, ::stride]
                depth_for_pcd = depth_m[::stride, ::stride]
                controller_mask = mask_packet.controller_mask[::stride, ::stride]
                object_mask = mask_packet.object_mask[::stride, ::stride]
                ray_x_for_pcd = ray_x[::stride, ::stride]
                ray_y_for_pcd = ray_y[::stride, ::stride]
            else:
                color_bgr = mask_packet.color_bgr
                depth_for_pcd = depth_m
                controller_mask = mask_packet.controller_mask
                object_mask = mask_packet.object_mask
                ray_x_for_pcd = ray_x
                ray_y_for_pcd = ray_y
            if controller_tracking_enabled(self.args):
                controller_xyz, controller_colors, controller_pcd_timing = backproject_masked_rgbd_profiled(
                    color_bgr=color_bgr,
                    depth_m=depth_for_pcd,
                    mask=controller_mask,
                    ray_x=ray_x_for_pcd,
                    ray_y=ray_y_for_pcd,
                    depth_min_m=float(self.args.depth_min_m),
                    depth_max_m=float(self.args.depth_max_m),
                    max_points=int(self.args.pcd_max_points),
                    color_mode=str(self.args.pcd_color_mode),
                    class_rgb=tuple(self.args.controller_color),
                    rng=rng,
                )
            else:
                controller_xyz = np.empty((0, 3), dtype=np.float32)
                controller_colors = np.empty((0, 3), dtype=np.uint8)
                controller_pcd_timing = {
                    "pcd_mask_intersection_ms": 0.0,
                    "pcd_select_ms": 0.0,
                    "pcd_point_cap_ms": 0.0,
                    "pcd_backproject_ms": 0.0,
                    "pcd_color_gather_ms": 0.0,
                    "pcd_raw_points": 0.0,
                    "pcd_cap_points": 0.0,
                }
            object_xyz, object_colors, object_pcd_timing = backproject_masked_rgbd_profiled(
                color_bgr=color_bgr,
                depth_m=depth_for_pcd,
                mask=object_mask,
                ray_x=ray_x_for_pcd,
                ray_y=ray_y_for_pcd,
                depth_min_m=float(self.args.depth_min_m),
                depth_max_m=float(self.args.depth_max_m),
                max_points=int(self.args.pcd_max_points),
                color_mode=str(self.args.pcd_color_mode),
                class_rgb=tuple(self.args.object_color),
                rng=rng,
            )
            controller_raw_points = int(controller_pcd_timing.get("pcd_raw_points", len(controller_xyz)))
            controller_cap_points = int(controller_pcd_timing.get("pcd_cap_points", len(controller_xyz)))
            object_raw_points = int(object_pcd_timing.get("pcd_raw_points", len(object_xyz)))
            object_cap_points = int(object_pcd_timing.get("pcd_cap_points", len(object_xyz)))
            render_controller_xyz = controller_xyz
            render_controller_colors = controller_colors
            render_object_xyz = object_xyz
            render_object_colors = object_colors
            filter_output: FilterOutput | None = None
            using_filtered = False

            if pcd_filter_enabled(self.args):
                if str(self.args.pcd_filter_mode) == "sync":
                    filter_input = self._make_filter_input(
                        seq=mask_packet.seq,
                        object_xyz=object_xyz,
                        object_colors=object_colors,
                        controller_xyz=controller_xyz,
                        controller_colors=controller_colors,
                    )
                    self.filter_submit_stats.record()
                    filter_output = self._filter_pcd_input(filter_input)
                    self.filter_output_stats.record(filter_output.output_perf_s)
                    render_controller_xyz = filter_output.controller_xyz
                    render_controller_colors = filter_output.controller_rgb
                    render_object_xyz = filter_output.object_xyz
                    render_object_colors = filter_output.object_rgb
                    using_filtered = True
                elif str(self.args.pcd_filter_mode) == "async":
                    worker = self.filter_worker
                    if worker is not None:
                        latest = worker.latest_output()
                        if latest is not None:
                            filter_output = latest
                            if int(latest.seq) != self._last_filter_output_seq_recorded:
                                self.filter_output_stats.record(latest.output_perf_s)
                                self._last_filter_output_seq_recorded = int(latest.seq)
                            render_controller_xyz = latest.controller_xyz
                            render_controller_colors = latest.controller_rgb
                            render_object_xyz = latest.object_xyz
                            render_object_colors = latest.object_rgb
                            using_filtered = True
                        if mask_packet.seq % int(self.args.filter_every_n) == 0:
                            if not worker.is_busy():
                                worker.submit_latest(
                                    self._make_filter_input(
                                        seq=mask_packet.seq,
                                        object_xyz=object_xyz,
                                        object_colors=object_colors,
                                        controller_xyz=controller_xyz,
                                        controller_colors=controller_colors,
                                    )
                                )
                                self.filter_submit_stats.record()
                            else:
                                self._filter_submit_skip_count += 1
                elif str(self.args.pcd_filter_mode) != "none":
                    raise ValueError(f"unsupported --pcd-filter-mode {self.args.pcd_filter_mode!r}")

            filter_telemetry = self._filter_telemetry_from_output(
                packet_seq=mask_packet.seq,
                output=filter_output,
                using_filtered=using_filtered,
                object_raw_points=object_raw_points,
                object_cap_points=object_cap_points,
                controller_raw_points=controller_raw_points,
                controller_cap_points=controller_cap_points,
            )
            done_s = time.perf_counter()
            timing = replace(
                mask_packet.timing,
                ffs_ms=ffs_ms,
                ffs_align_ms=ffs_align_ms,
                remote_rtt_ms=remote_rtt_ms,
                remote_server_total_ms=remote_server_total_ms,
                remote_request_kb=remote_request_kb,
                remote_response_kb=remote_response_kb,
                depth_convert_ms=depth_convert_ms,
                pcd_mask_intersection_ms=float(
                    controller_pcd_timing["pcd_mask_intersection_ms"]
                    + object_pcd_timing["pcd_mask_intersection_ms"]
                ),
                pcd_select_ms=float(controller_pcd_timing["pcd_select_ms"] + object_pcd_timing["pcd_select_ms"]),
                pcd_point_cap_ms=float(
                    controller_pcd_timing["pcd_point_cap_ms"] + object_pcd_timing["pcd_point_cap_ms"]
                ),
                pcd_backproject_ms=float(
                    controller_pcd_timing["pcd_backproject_ms"] + object_pcd_timing["pcd_backproject_ms"]
                ),
                pcd_color_gather_ms=float(
                    controller_pcd_timing["pcd_color_gather_ms"] + object_pcd_timing["pcd_color_gather_ms"]
                ),
                pcd_filter_ms=float(filter_telemetry.filter_ms),
                object_filter_ms=float(filter_telemetry.object_filter_ms),
                controller_filter_ms=float(filter_telemetry.controller_filter_ms),
                pcd_ms=_elapsed_ms(start_s, done_s),
            )
            packet = MaskedPcdPacket(
                seq=mask_packet.seq,
                controller_xyz_m=render_controller_xyz,
                controller_colors_rgb_u8=render_controller_colors,
                object_xyz_m=render_object_xyz,
                object_colors_rgb_u8=render_object_colors,
                intrinsics=mask_packet.intrinsics,
                receive_perf_s=mask_packet.receive_perf_s,
                process_done_perf_s=done_s,
                dropped_capture_frames=mask_packet.dropped_capture_frames,
                dropped_seg_frames=self.mask_slot.dropped_count,
                timing=timing,
                filter_telemetry=filter_telemetry,
            )
            self.render_slot.put(packet)
            self.pcd_stats.record(done_s)
            if packet.seq % int(self.args.render_every_n) == 0:
                self._request_render_update()

    def _depth_profile_worker(self) -> None:
        last_seq = -1
        while not self.stop_event.is_set():
            frame = self.capture_slot.get_latest_after(last_seq)
            if frame is None:
                time.sleep(0.001)
                continue
            last_seq = frame.seq
            if frame.depth_source not in {"ffs", "ffs_remote"}:
                continue
            try:
                (
                    _depth_m,
                    ffs_ms,
                    ffs_align_ms,
                    remote_rtt_ms,
                    remote_server_total_ms,
                    remote_request_kb,
                    remote_response_kb,
                ) = self._compute_external_ffs_depth_color_m(frame)
            except Exception as exc:
                if not self.stop_event.is_set():
                    print(f"[WARN] FFS depth profile frame {frame.seq} failed: {type(exc).__name__}: {exc}", flush=True)
                continue
            done_s = time.perf_counter()
            packet = DepthProfilePacket(
                seq=frame.seq,
                receive_perf_s=frame.receive_perf_s,
                process_done_perf_s=done_s,
                dropped_capture_frames=self.capture_slot.dropped_count,
                timing=replace(
                    frame.timing,
                    ffs_ms=ffs_ms,
                    ffs_align_ms=ffs_align_ms,
                    remote_rtt_ms=remote_rtt_ms,
                    remote_server_total_ms=remote_server_total_ms,
                    remote_request_kb=remote_request_kb,
                    remote_response_kb=remote_response_kb,
                ),
            )
            self.depth_profile_slot.put(packet)
            self.depth_stats.record(done_s)

    def _compute_external_ffs_depth_color_m(
        self,
        packet: MaskPacket | FramePacket,
    ) -> tuple[np.ndarray, float, float, float, float, float, float]:
        if packet.depth_source == "ffs_remote":
            return self._compute_remote_ffs_depth_color_m(packet)
        depth_color_m, ffs_ms, ffs_align_ms = self._compute_ffs_depth_color_m(packet)
        return depth_color_m, ffs_ms, ffs_align_ms, 0.0, 0.0, 0.0, 0.0

    def _compute_ffs_depth_color_m(self, packet: MaskPacket | FramePacket) -> tuple[np.ndarray, float, float]:
        runner = self.ffs_runner
        if runner is None:
            raise RuntimeError("FFS runner is not initialized")
        if (
            packet.ir_left_u8 is None
            or packet.ir_right_u8 is None
            or packet.k_ir_left is None
            or packet.t_ir_left_to_color is None
            or packet.k_color is None
            or packet.ir_baseline_m <= 0
        ):
            raise RuntimeError("FFS packet is missing IR stereo calibration/data")

        ffs_start_s = time.perf_counter()
        output = runner.run_pair(
            packet.ir_left_u8,
            packet.ir_right_u8,
            K_ir_left=packet.k_ir_left,
            baseline_m=float(packet.ir_baseline_m),
        )
        ffs_done_s = time.perf_counter()
        depth_ir_left_m = np.asarray(output["depth_ir_left_m"], dtype=np.float32)
        k_ir_left_used = np.asarray(output.get("K_ir_left_used", packet.k_ir_left), dtype=np.float32)
        align_start_s = time.perf_counter()
        aligner = self._get_ir_to_color_aligner(
            depth_shape=depth_ir_left_m.shape,
            color_shape=packet.color_bgr.shape[:2],
            k_ir_left=k_ir_left_used,
            t_ir_left_to_color=packet.t_ir_left_to_color,
            k_color=packet.k_color,
        )
        depth_color_m = np.ascontiguousarray(aligner.align(depth_ir_left_m), dtype=np.float32)
        align_done_s = time.perf_counter()
        return (
            depth_color_m,
            _elapsed_ms(ffs_start_s, ffs_done_s),
            _elapsed_ms(align_start_s, align_done_s),
        )

    def _compute_remote_ffs_depth_color_m(
        self,
        packet: MaskPacket | FramePacket,
    ) -> tuple[np.ndarray, float, float, float, float, float, float]:
        if self.args.ffs_remote_return in SPARSE_RETURN_TYPES:
            raise RuntimeError("sparse ffs_remote returns must be consumed by the sparse PCD path")
        result = self._request_remote_ffs_result(packet, mask_u8=None)
        self._warn_if_remote_engine_contract_missing(result)
        return (
            result.depth_color_m,
            result.server_ffs_ms,
            result.server_align_ms,
            result.rtt_ms,
            result.server_total_ms,
            float(result.request_bytes) / 1024.0,
            float(result.response_bytes) / 1024.0,
        )

    def _request_remote_ffs_result(
        self,
        packet: MaskPacket | FramePacket,
        *,
        mask_u8: np.ndarray | None,
    ) -> Any:
        client = self.ffs_remote_client
        if client is None:
            raise RuntimeError("remote FFS client is not initialized")
        if (
            packet.ir_left_u8 is None
            or packet.ir_right_u8 is None
            or packet.k_ir_left is None
            or packet.t_ir_left_to_color is None
            or packet.k_color is None
            or packet.ir_baseline_m <= 0
        ):
            raise RuntimeError("remote FFS packet is missing IR stereo calibration/data")
        result = client.request_depth_color_m(
            frame_id=int(packet.seq),
            ir_left_u8=packet.ir_left_u8,
            ir_right_u8=packet.ir_right_u8,
            color_shape=tuple(packet.color_bgr.shape[:2]),
            k_ir_left=packet.k_ir_left,
            k_color=packet.k_color,
            t_ir_left_to_color=packet.t_ir_left_to_color,
            baseline_m=float(packet.ir_baseline_m),
            depth_scale_m_per_unit=float(packet.depth_scale_m_per_unit),
            mask_u8=mask_u8,
        )
        return result

    def _warn_if_remote_engine_contract_missing(self, result: Any) -> None:
        if self._warned_remote_engine_contract:
            return
        metadata = getattr(result, "metadata", None) or {}
        expected = {
            "ffs_contract_model": DEFAULT_FFS_MODEL_NAME,
            "ffs_contract_valid_iters": DEFAULT_FFS_VALID_ITERS,
            "ffs_contract_engine_height": DEFAULT_FFS_TRT_ENGINE_SIZE[0],
            "ffs_contract_engine_width": DEFAULT_FFS_TRT_ENGINE_SIZE[1],
            "ffs_contract_builder_optimization_level": DEFAULT_FFS_TRT_BUILDER_OPTIMIZATION_LEVEL,
            "ffs_contract_max_disp": DEFAULT_FFS_MAX_DISP,
        }
        missing = [key for key in expected if key not in metadata]
        mismatched = [
            key
            for key, value in expected.items()
            if key in metadata and str(metadata[key]) != str(value)
        ]
        if missing or mismatched:
            print(
                "[WARN] remote FFS server did not prove required Demo 2 engine identity: "
                f"missing={missing} mismatched={mismatched}",
                flush=True,
            )
        self._warned_remote_engine_contract = True

    def _split_sparse_remote_pcd(
        self,
        *,
        payload: np.ndarray,
        return_type: str,
        color_bgr: np.ndarray,
        ray_x: np.ndarray,
        ray_y: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
        timing: dict[str, float] = {
            "pcd_mask_intersection_ms": 0.0,
            "pcd_select_ms": 0.0,
            "pcd_point_cap_ms": 0.0,
            "pcd_backproject_ms": 0.0,
            "pcd_color_gather_ms": 0.0,
        }
        sparse = np.asarray(payload, dtype=np.float32)
        if sparse.ndim != 2 or sparse.shape[1] < 4:
            raise RuntimeError(f"expected sparse payload Nx4+, got {sparse.shape}")
        if sparse.shape[0] == 0:
            return (
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.uint8),
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.uint8),
                timing,
            )

        select_start_s = time.perf_counter()
        if return_type == "masked_uv_depth":
            cols = np.rint(sparse[:, 0]).astype(np.int64)
            rows = np.rint(sparse[:, 1]).astype(np.int64)
            z = sparse[:, 2].astype(np.float32, copy=False)
            labels = np.rint(sparse[:, 3]).astype(np.int64)
            in_bounds = (rows >= 0) & (rows < color_bgr.shape[0]) & (cols >= 0) & (cols < color_bgr.shape[1])
            valid = in_bounds & np.isfinite(z) & (z > np.float32(self.args.depth_min_m))
            if float(self.args.depth_max_m) > 0:
                valid &= z < np.float32(self.args.depth_max_m)
            rows = rows[valid]
            cols = cols[valid]
            z = z[valid]
            labels = labels[valid]
            timing["pcd_select_ms"] = _elapsed_ms(select_start_s, time.perf_counter())

            backproject_start_s = time.perf_counter()
            points = np.ascontiguousarray(
                np.stack([ray_x[rows, cols] * z, ray_y[rows, cols] * z, z], axis=1),
                dtype=np.float32,
            )
            timing["pcd_backproject_ms"] = _elapsed_ms(backproject_start_s, time.perf_counter())
        elif return_type == "masked_xyz":
            if sparse.shape[1] < 6:
                raise RuntimeError("masked_xyz payload must be Nx6 [x,y,z,label,u,v]")
            points = np.ascontiguousarray(sparse[:, :3], dtype=np.float32)
            labels = np.rint(sparse[:, 3]).astype(np.int64)
            cols = np.rint(sparse[:, 4]).astype(np.int64)
            rows = np.rint(sparse[:, 5]).astype(np.int64)
            z = points[:, 2]
            in_bounds = (rows >= 0) & (rows < color_bgr.shape[0]) & (cols >= 0) & (cols < color_bgr.shape[1])
            valid = in_bounds & np.isfinite(z) & (z > np.float32(self.args.depth_min_m))
            if float(self.args.depth_max_m) > 0:
                valid &= z < np.float32(self.args.depth_max_m)
            points = points[valid]
            labels = labels[valid]
            rows = rows[valid]
            cols = cols[valid]
            timing["pcd_select_ms"] = _elapsed_ms(select_start_s, time.perf_counter())
        else:
            raise RuntimeError(f"unsupported sparse return type: {return_type}")

        def build_for_label(label: int, class_rgb: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
            label_indices = np.nonzero(labels == int(label))[0]
            cap_start_s = time.perf_counter()
            max_points = int(self.args.pcd_max_points)
            if max_points > 0 and label_indices.shape[0] > max_points:
                label_indices = rng.choice(label_indices, max_points, replace=False)
            timing["pcd_point_cap_ms"] += _elapsed_ms(cap_start_s, time.perf_counter())
            xyz = np.ascontiguousarray(points[label_indices], dtype=np.float32)
            color_start_s = time.perf_counter()
            if str(self.args.pcd_color_mode) == "rgb":
                colors = np.ascontiguousarray(color_bgr[rows[label_indices], cols[label_indices], ::-1], dtype=np.uint8)
            else:
                colors = make_solid_colors(xyz.shape[0], class_rgb)
            timing["pcd_color_gather_ms"] += _elapsed_ms(color_start_s, time.perf_counter())
            return xyz, colors

        controller_xyz, controller_colors = build_for_label(CONTROLLER_ID, tuple(self.args.controller_color))
        object_xyz, object_colors = build_for_label(OBJECT_ID, tuple(self.args.object_color))
        return controller_xyz, controller_colors, object_xyz, object_colors, timing

    def _compute_remote_sparse_pcd_packet(
        self,
        *,
        mask_packet: MaskPacket,
        start_s: float,
        rng: np.random.Generator,
    ) -> MaskedPcdPacket:
        assert self.ray_x is not None and self.ray_y is not None
        mask_u8 = self._remote_quality_mask_u8(mask_packet)
        result = self._request_remote_ffs_result(mask_packet, mask_u8=mask_u8)
        self._warn_if_remote_engine_contract_missing(result)
        if result.sparse_payload is None:
            raise RuntimeError("remote sparse FFS response did not include sparse payload")
        controller_xyz, controller_colors, object_xyz, object_colors, sparse_timing = self._split_sparse_remote_pcd(
            payload=result.sparse_payload,
            return_type=str(result.return_type),
            color_bgr=mask_packet.color_bgr,
            ray_x=self.ray_x,
            ray_y=self.ray_y,
            rng=rng,
        )
        render_controller_xyz = controller_xyz
        render_controller_colors = controller_colors
        render_object_xyz = object_xyz
        render_object_colors = object_colors
        filter_output: FilterOutput | None = None
        using_filtered = False
        if pcd_filter_enabled(self.args):
            if str(self.args.pcd_filter_mode) == "sync":
                filter_input = self._make_filter_input(
                    seq=mask_packet.seq,
                    object_xyz=object_xyz,
                    object_colors=object_colors,
                    controller_xyz=controller_xyz,
                    controller_colors=controller_colors,
                )
                self.filter_submit_stats.record()
                filter_output = self._filter_pcd_input(filter_input)
                self.filter_output_stats.record(filter_output.output_perf_s)
                render_controller_xyz = filter_output.controller_xyz
                render_controller_colors = filter_output.controller_rgb
                render_object_xyz = filter_output.object_xyz
                render_object_colors = filter_output.object_rgb
                using_filtered = True
            elif str(self.args.pcd_filter_mode) == "async" and self.filter_worker is not None:
                latest = self.filter_worker.latest_output()
                if latest is not None:
                    filter_output = latest
                    render_controller_xyz = latest.controller_xyz
                    render_controller_colors = latest.controller_rgb
                    render_object_xyz = latest.object_xyz
                    render_object_colors = latest.object_rgb
                    using_filtered = True
                if mask_packet.seq % int(self.args.filter_every_n) == 0:
                    if not self.filter_worker.is_busy():
                        self.filter_worker.submit_latest(
                            self._make_filter_input(
                                seq=mask_packet.seq,
                                object_xyz=object_xyz,
                                object_colors=object_colors,
                                controller_xyz=controller_xyz,
                                controller_colors=controller_colors,
                            )
                        )
                        self.filter_submit_stats.record()
                    else:
                        self._filter_submit_skip_count += 1
        filter_telemetry = self._filter_telemetry_from_output(
            packet_seq=mask_packet.seq,
            output=filter_output,
            using_filtered=using_filtered,
            object_raw_points=int(len(object_xyz)),
            object_cap_points=int(len(object_xyz)),
            controller_raw_points=int(len(controller_xyz)),
            controller_cap_points=int(len(controller_xyz)),
        )
        done_s = time.perf_counter()
        timing = replace(
            mask_packet.timing,
            ffs_ms=float(result.server_ffs_ms),
            ffs_align_ms=float(result.server_align_ms),
            remote_rtt_ms=float(result.rtt_ms),
            remote_server_total_ms=float(result.server_total_ms),
            remote_request_kb=float(result.request_bytes) / 1024.0,
            remote_response_kb=float(result.response_bytes) / 1024.0,
            pcd_mask_intersection_ms=float(sparse_timing["pcd_mask_intersection_ms"]),
            pcd_select_ms=float(sparse_timing["pcd_select_ms"]),
            pcd_point_cap_ms=float(sparse_timing["pcd_point_cap_ms"]),
            pcd_backproject_ms=float(sparse_timing["pcd_backproject_ms"]),
            pcd_color_gather_ms=float(sparse_timing["pcd_color_gather_ms"]),
            pcd_filter_ms=float(filter_telemetry.filter_ms),
            object_filter_ms=float(filter_telemetry.object_filter_ms),
            controller_filter_ms=float(filter_telemetry.controller_filter_ms),
            pcd_ms=_elapsed_ms(start_s, done_s),
        )
        return MaskedPcdPacket(
            seq=mask_packet.seq,
            controller_xyz_m=render_controller_xyz,
            controller_colors_rgb_u8=render_controller_colors,
            object_xyz_m=render_object_xyz,
            object_colors_rgb_u8=render_object_colors,
            intrinsics=mask_packet.intrinsics,
            receive_perf_s=mask_packet.receive_perf_s,
            process_done_perf_s=done_s,
            dropped_capture_frames=mask_packet.dropped_capture_frames,
            dropped_seg_frames=self.mask_slot.dropped_count,
            timing=timing,
            filter_telemetry=filter_telemetry,
        )

    def _remote_quality_mask_u8(self, packet: MaskPacket) -> np.ndarray:
        mask = np.zeros(tuple(packet.object_mask.shape), dtype=np.uint8)
        if controller_tracking_enabled(self.args):
            mask[np.asarray(packet.controller_mask, dtype=bool)] = CONTROLLER_ID
        mask[np.asarray(packet.object_mask, dtype=bool)] = OBJECT_ID
        return np.ascontiguousarray(mask)

    def _request_remote_quality(self, packet: MaskPacket | FramePacket, *, mask_u8: np.ndarray | None) -> RemoteFfsQualityPacket:
        client = self.remote_quality_client
        if client is None:
            raise RuntimeError("remote FFS quality client is not initialized")
        if (
            packet.ir_left_u8 is None
            or packet.ir_right_u8 is None
            or packet.k_ir_left is None
            or packet.t_ir_left_to_color is None
            or packet.k_color is None
            or packet.ir_baseline_m <= 0
        ):
            raise RuntimeError("remote FFS quality packet is missing IR stereo calibration/data")
        result = client.request_depth_color_m(
            frame_id=int(packet.seq),
            ir_left_u8=packet.ir_left_u8,
            ir_right_u8=packet.ir_right_u8,
            color_shape=tuple(packet.color_bgr.shape[:2]),
            k_ir_left=packet.k_ir_left,
            k_color=packet.k_color,
            t_ir_left_to_color=packet.t_ir_left_to_color,
            baseline_m=float(packet.ir_baseline_m),
            depth_scale_m_per_unit=float(packet.depth_scale_m_per_unit),
            mask_u8=mask_u8,
        )
        self._warn_if_remote_engine_contract_missing(result)
        done_s = time.perf_counter()
        timing = replace(
            packet.timing,
            ffs_ms=float(result.server_ffs_ms),
            ffs_align_ms=float(result.server_align_ms),
            remote_rtt_ms=float(result.rtt_ms),
            remote_server_total_ms=float(result.server_total_ms),
            remote_request_kb=float(result.request_bytes) / 1024.0,
            remote_response_kb=float(result.response_bytes) / 1024.0,
        )
        sparse_points = 0
        if result.sparse_payload is not None:
            sparse_points = int(result.sparse_payload.shape[0])
        return RemoteFfsQualityPacket(
            seq=int(packet.seq),
            receive_perf_s=float(packet.receive_perf_s),
            process_done_perf_s=done_s,
            timing=timing,
            return_type=str(result.return_type),
            sparse_points=sparse_points,
        )

    def _remote_ffs_quality_worker(self) -> None:
        last_seq = -1
        next_request_s = 0.0
        sparse = str(self.args.remote_ffs_quality_return) in SPARSE_RETURN_TYPES
        interval_s = float(self.args.remote_ffs_quality_interval_ms) / 1000.0
        while not self.stop_event.is_set():
            now_s = time.perf_counter()
            if now_s < next_request_s:
                time.sleep(min(0.01, next_request_s - now_s))
                continue
            if sparse:
                source = self.mask_slot.get_latest_after(last_seq)
                if source is None:
                    time.sleep(0.002)
                    continue
                mask_u8 = self._remote_quality_mask_u8(source)
            else:
                source = self.capture_slot.get_latest_after(last_seq)
                if source is None:
                    time.sleep(0.002)
                    continue
                mask_u8 = None
            last_seq = source.seq
            next_request_s = time.perf_counter() + interval_s
            try:
                packet = self._request_remote_quality(source, mask_u8=mask_u8)
            except Exception as exc:
                if not self.stop_event.is_set() and self.args.debug:
                    print(f"[remote-ffs-quality] seq={source.seq} status=error error={type(exc).__name__}: {exc}", flush=True)
                continue
            self.remote_quality_slot.put(packet)
            self.remote_quality_stats.record(packet.process_done_perf_s)
            if self.args.debug:
                age_ms = _elapsed_ms(packet.receive_perf_s, packet.process_done_perf_s)
                print(
                    "[remote-ffs-quality] "
                    f"seq={packet.seq} fps={self.remote_quality_stats.fps:.2f} "
                    f"return={packet.return_type} sparse_points={packet.sparse_points} "
                    f"age_ms={age_ms:.1f} rtt_ms={packet.timing.remote_rtt_ms:.1f} "
                    f"server_total_ms={packet.timing.remote_server_total_ms:.1f} "
                    f"request_kb={packet.timing.remote_request_kb:.1f} "
                    f"response_kb={packet.timing.remote_response_kb:.1f}",
                    flush=True,
                )

    def _run_open3d_viewer(self) -> None:
        o3d, gui, rendering = _load_open3d_modules()
        o3c = o3d.core
        device = o3c.Device("CPU:0")
        app = gui.Application.instance
        app.initialize()
        window = app.create_window("Demo 2.0 Realtime EdgeTAM Masked PCD", 1280, 800)
        scene_widget = gui.SceneWidget()
        scene_widget.scene = rendering.Open3DScene(window.renderer)
        scene_widget.scene.set_background([0.02, 0.02, 0.02, 1.0])
        hud_label = gui.Label(WARMUP_HUD_TEXT)
        hud_label.text_color = gui.Color(1.0, 1.0, 1.0)
        hud_panel = gui.Vert(0, gui.Margins(8, 8, 8, 8))
        hud_panel.add_child(hud_label)
        window.add_child(scene_widget)
        window.add_child(hud_panel)

        def on_layout(layout_context: object) -> None:
            rect = window.content_rect
            scene_widget.frame = rect
            em = window.theme.font_size
            preferred = hud_panel.calc_preferred_size(layout_context, gui.Widget.Constraints())
            hud_panel.frame = gui.Rect(
                rect.x + 0.5 * em,
                rect.y + 0.5 * em,
                max(preferred.width, 660),
                max(preferred.height, (15.0 if self.args.debug else 11.0) * em),
            )

        window.set_on_layout(on_layout)
        material = rendering.MaterialRecord()
        material.shader = "defaultUnlit"
        material.point_size = float(self.args.point_size)

        class GeometryState:
            def __init__(self, name: str) -> None:
                self.name = name
                self.pcd = o3d.t.geometry.PointCloud(device)
                self.color_buffer = ColorFloat32Buffer()
                self.refs: dict[str, np.ndarray | None] = {"points": None, "colors": None}
                self.added = False
                self.capacity = 0
                self.warned = False

            def update(self, points_xyz_m: np.ndarray, colors_rgb_u8: np.ndarray) -> tuple[float, float]:
                convert_start_s = time.perf_counter()
                points = ensure_float32_c_contiguous(points_xyz_m)
                colors = self.color_buffer.convert(colors_rgb_u8)
                self.refs["points"] = points
                self.refs["colors"] = colors
                self.pcd.point.positions = o3c.Tensor.from_numpy(points)
                self.pcd.point.colors = o3c.Tensor.from_numpy(colors)
                convert_ms = _elapsed_ms(convert_start_s, time.perf_counter())

                update_start_s = time.perf_counter()
                if points.shape[0] == 0:
                    if self.added:
                        try:
                            scene_widget.scene.remove_geometry(self.name)
                        except Exception:
                            pass
                    self.added = False
                    self.capacity = 0
                    return convert_ms, _elapsed_ms(update_start_s, time.perf_counter())

                if pointcloud_update_requires_readd(
                    geometry_added=self.added,
                    current_capacity=self.capacity,
                    point_count=int(points.shape[0]),
                ):
                    if self.added:
                        try:
                            scene_widget.scene.remove_geometry(self.name)
                        except Exception:
                            pass
                    scene_widget.scene.add_geometry(self.name, self.pcd, material)
                    self.added = True
                    self.capacity = int(points.shape[0])
                else:
                    try:
                        flags = rendering.Scene.UPDATE_POINTS_FLAG | rendering.Scene.UPDATE_COLORS_FLAG
                        scene_widget.scene.scene.update_geometry(self.name, self.pcd, flags)
                        self.capacity = max(self.capacity, int(points.shape[0]))
                    except Exception as exc:
                        if not self.warned:
                            print(f"[WARN] update_geometry fallback for {self.name}: {type(exc).__name__}: {exc}", flush=True)
                            self.warned = True
                        try:
                            scene_widget.scene.remove_geometry(self.name)
                        except Exception:
                            pass
                        scene_widget.scene.add_geometry(self.name, self.pcd, material)
                        self.added = True
                        self.capacity = int(points.shape[0])
                return convert_ms, _elapsed_ms(update_start_s, time.perf_counter())

        controller_state = GeometryState(GEOMETRY_CONTROLLER)
        object_state = GeometryState(GEOMETRY_OBJECT)
        camera_initialized = {"value": False}
        render_post_gate = CoalescedPostGate()
        last_render_seq = {"value": -1}

        def reset_camera() -> None:
            intrinsic_matrix = np.array(
                [
                    [self.intrinsics.fx, 0.0, self.intrinsics.cx],
                    [0.0, self.intrinsics.fy, self.intrinsics.cy],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            extrinsic = np.eye(4, dtype=np.float64)
            bounds = o3d.geometry.AxisAlignedBoundingBox([-10.0, -10.0, 0.01], [10.0, 10.0, 20.0])
            scene_widget.setup_camera(intrinsic_matrix, extrinsic, self.width, self.height, bounds)

        def render_latest() -> bool:
            packet = self.render_slot.get_latest_after(last_render_seq["value"])
            if packet is None:
                return False
            last_render_seq["value"] = packet.seq
            controller_convert_ms, controller_update_ms = controller_state.update(
                packet.controller_xyz_m,
                packet.controller_colors_rgb_u8,
            )
            object_convert_ms, object_update_ms = object_state.update(
                packet.object_xyz_m,
                packet.object_colors_rgb_u8,
            )
            if not camera_initialized["value"] and packet.point_count > 0:
                reset_camera()
                camera_initialized["value"] = True
            render_time_s = time.perf_counter()
            latency_ms = _elapsed_ms(packet.receive_perf_s, render_time_s)
            timing = replace(
                packet.timing,
                open3d_convert_ms=float(controller_convert_ms + object_convert_ms),
                open3d_update_ms=float(controller_update_ms + object_update_ms),
                receive_to_render_ms=latency_ms,
            )
            self.render_stats.record_render(render_time_s=render_time_s, latency_ms=latency_ms)
            hud_label.text = self._format_hud(packet=packet, timing=timing)
            self._maybe_log_debug(packet=packet, timing=timing, now_s=render_time_s)
            return True

        def render_latest_on_main_thread() -> None:
            try:
                if self.stop_event.is_set():
                    return
                rendered = render_latest()
                if rendered and hasattr(window, "post_redraw"):
                    try:
                        window.post_redraw()
                    except Exception:
                        pass
            finally:
                render_post_gate.mark_done()
                if not self.stop_event.is_set() and self.render_slot.latest_seq() > last_render_seq["value"]:
                    request_render_update()

        def request_render_update() -> None:
            if self.stop_event.is_set():
                return
            if not render_post_gate.try_mark_pending():
                return
            try:
                app.post_to_main_thread(window, render_latest_on_main_thread)
            except Exception:
                render_post_gate.mark_done()

        fast_exit_after_open3d = os.environ.get("QQTT_WSLG_OPEN3D_FAST_EXIT") == "1"

        def stop_and_quit_open3d() -> None:
            self.stop_event.set()
            self._request_render_update = lambda: None
            if fast_exit_after_open3d:
                self.stop()
                os._exit(0)
            try:
                app.quit()
            except Exception:
                pass

        def on_close() -> bool:
            stop_and_quit_open3d()
            return True

        window.set_on_close(on_close)
        self._request_render_update = request_render_update
        self._start_threads()

        timer: threading.Timer | None = None
        if self.args.duration_s > 0:
            timer = threading.Timer(
                float(self.args.duration_s),
                lambda: app.post_to_main_thread(window, stop_and_quit_open3d),
            )
            timer.daemon = True
            timer.start()
        try:
            app.run()
        finally:
            self._request_render_update = lambda: None
            if timer is not None:
                timer.cancel()

    def _format_hud(self, *, packet: MaskedPcdPacket, timing: PipelineTiming) -> str:
        status = "late" if timing.receive_to_render_ms > self.args.latency_target_ms else "ok"
        max_points = "uncapped" if self.args.pcd_max_points == 0 else str(self.args.pcd_max_points)
        depth_line = f"depth: {self.args.depth_source}  color={self.args.pcd_color_mode}"
        preset_text = "" if self.args.demo_preset == "none" else f"  preset={self.args.demo_preset}"
        filter_info = packet.filter_telemetry
        if filter_info.enabled:
            filter_line = (
                f"filter: {filter_info.mode}  render={'filtered' if filter_info.render_using_filtered else 'raw'}  "
                f"fps submit/out={filter_info.filter_submit_fps:.1f}/{filter_info.filter_output_fps:.1f}  "
                f"age={filter_info.filter_age_frames}f/{filter_info.filter_age_ms:.0f}ms  "
                f"busy={int(filter_info.filter_busy)} drop={filter_info.filter_queue_drop}"
            )
            filter_points_line = (
                "filter pts controller raw/cap/out="
                f"{filter_info.controller_raw_points}/{filter_info.controller_cap_points}/{filter_info.controller_output_points}  "
                "object raw/cap/out="
                f"{filter_info.object_raw_points}/{filter_info.object_cap_points}/{filter_info.object_output_points}"
            )
        else:
            filter_line = "filter: off"
            filter_points_line = ""
        if self.args.depth_source == "ffs_remote":
            depth_line += f"  remote={self.args.ffs_remote_endpoint}"
        quality_line = ""
        if self.args.enable_remote_ffs_quality:
            quality_packet = self.remote_quality_slot.get_latest_after(-1)
            if quality_packet is None:
                quality_line = (
                    "\nremote FFS quality: waiting  "
                    f"return={self.args.remote_ffs_quality_return}  endpoint="
                    f"{self.args.remote_ffs_quality_endpoint or self.args.ffs_remote_endpoint}"
                )
            else:
                age_ms = _elapsed_ms(quality_packet.process_done_perf_s, time.perf_counter())
                quality_line = (
                    "\nremote FFS quality: "
                    f"{self.remote_quality_stats.fps:.1f} FPS  age={age_ms:.0f} ms  "
                    f"rtt={quality_packet.timing.remote_rtt_ms:.0f} ms  "
                    f"resp={quality_packet.timing.remote_response_kb:.0f} KB  "
                    f"return={quality_packet.return_type}"
                )
        return (
            f"capture/seg/pcd/render FPS: {self.capture_stats.fps:.1f} / {self.seg_stats.fps:.1f} / "
            f"{self.pcd_stats.fps:.1f} / {self.render_stats.render_fps:.1f}\n"
            f"latency: {timing.receive_to_render_ms:.1f} ms ({status}, target {self.args.latency_target_ms:.1f} ms)\n"
            f"points controller/object: {packet.controller_point_count} / {packet.object_point_count}  max/object: {max_points}\n"
            f"{filter_line}\n"
            f"{filter_points_line}\n"
            f"dropped capture/seg/pcd: {packet.dropped_capture_frames} / {packet.dropped_seg_frames} / "
            f"{self.render_slot.dropped_count}\n"
            f"EdgeTAM: {self.args.model_id}  mode={self.args.track_mode}  compile={self.args.compile_mode}  "
            f"dtype={self.args.dtype}{preset_text}\n"
            f"{depth_line}{quality_line}\n"
            f"serial/profile/fps: {self.serial}  {self.args.profile}@{self.args.fps}\n"
            f"frame: {COORDINATE_FRAME}  meters  x right / y down / z forward"
        )

    def _emit_debug_line(
        self,
        *,
        seq: int,
        timing: PipelineTiming,
        controller_points: int = 0,
        object_points: int = 0,
        dropped_capture_frames: int = 0,
        dropped_seg_frames: int = 0,
        filter_telemetry: PcdFilterTelemetry | None = None,
    ) -> None:
        filter_info = filter_telemetry or PcdFilterTelemetry()
        print(
            "[masked-edgetam-debug] "
            f"seq={int(seq)} "
            f"capture_fps={self.capture_stats.fps:.1f} "
            f"seg_fps={self.seg_stats.fps:.1f} "
            f"depth_fps={self.depth_stats.fps:.1f} "
            f"remote_quality_fps={self.remote_quality_stats.fps:.1f} "
            f"pcd_fps={self.pcd_stats.fps:.1f} "
            f"render_fps={self.render_stats.render_fps:.1f} "
            f"profile_sync_enabled={int(bool(self.args.profile_sync))} "
            f"profile_cuda_events={int(bool(self.args.profile_cuda_events))} "
            f"mask_ms={timing.mask_ms:.2f} "
            f"preprocess_ms={timing.preprocess_ms:.2f} "
            f"prompt_ms={timing.prompt_ms:.2f} "
            f"model_ms={timing.model_ms:.2f} "
            f"wall_model_ms={timing.wall_model_ms:.2f} "
            f"cuda_event_model_ms={timing.cuda_event_model_ms:.2f} "
            f"pre_sync_wait_ms={timing.pre_sync_wait_ms:.2f} "
            f"post_sync_wait_ms={timing.post_sync_wait_ms:.2f} "
            f"postprocess_ms={timing.postprocess_ms:.2f} "
            f"ffs_ms={timing.ffs_ms:.2f} "
            f"ffs_align_ms={timing.ffs_align_ms:.2f} "
            f"remote_rtt_ms={timing.remote_rtt_ms:.2f} "
            f"remote_server_total_ms={timing.remote_server_total_ms:.2f} "
            f"remote_request_kb={timing.remote_request_kb:.1f} "
            f"remote_response_kb={timing.remote_response_kb:.1f} "
            f"depth_convert_ms={timing.depth_convert_ms:.2f} "
            f"pcd_mask_intersection_ms={timing.pcd_mask_intersection_ms:.2f} "
            f"pcd_select_ms={timing.pcd_select_ms:.2f} "
            f"pcd_point_cap_ms={timing.pcd_point_cap_ms:.2f} "
            f"pcd_backproject_ms={timing.pcd_backproject_ms:.2f} "
            f"pcd_color_gather_ms={timing.pcd_color_gather_ms:.2f} "
            f"pcd_filter_ms={timing.pcd_filter_ms:.2f} "
            f"object_filter_ms={timing.object_filter_ms:.2f} "
            f"controller_filter_ms={timing.controller_filter_ms:.2f} "
            f"pcd_ms={timing.pcd_ms:.2f} "
            f"render_ms={timing.open3d_update_ms:.2f} "
            f"e2e_latency_ms={timing.receive_to_render_ms:.2f} "
            f"filter_enabled={int(filter_info.enabled)} "
            f"filter_mode={filter_info.mode} "
            f"render_using_filtered={int(filter_info.render_using_filtered)} "
            f"filter_submit_fps={filter_info.filter_submit_fps:.1f} "
            f"filter_output_fps={filter_info.filter_output_fps:.1f} "
            f"filter_queue_drop={int(filter_info.filter_queue_drop)} "
            f"filter_busy={int(filter_info.filter_busy)} "
            f"filter_age_frames={int(filter_info.filter_age_frames)} "
            f"filter_age_ms={filter_info.filter_age_ms:.1f} "
            f"object_filter_input_points={int(filter_info.object_raw_points)} "
            f"object_filter_cap_points={int(filter_info.object_cap_points)} "
            f"object_filter_output_points={int(filter_info.object_output_points)} "
            f"controller_filter_input_points={int(filter_info.controller_raw_points)} "
            f"controller_filter_cap_points={int(filter_info.controller_cap_points)} "
            f"controller_filter_output_points={int(filter_info.controller_output_points)} "
            f"controller_points={int(controller_points)} "
            f"object_points={int(object_points)} "
            f"dropped_capture={int(dropped_capture_frames)} "
            f"dropped_seg={int(dropped_seg_frames)} "
            f"dropped_pcd={self.render_slot.dropped_count}",
            flush=True,
        )

    def _headless_debug_worker(self) -> None:
        last_logged_seq = -1
        while not self.stop_event.is_set():
            now_s = time.perf_counter()
            if now_s - self._last_debug_log_s < DEBUG_LOG_INTERVAL_S:
                time.sleep(0.05)
                continue
            self._last_debug_log_s = now_s
            pcd_packet = self.render_slot.get_latest_after(last_logged_seq)
            if pcd_packet is not None:
                last_logged_seq = pcd_packet.seq
                timing = replace(
                    pcd_packet.timing,
                    receive_to_render_ms=_elapsed_ms(pcd_packet.receive_perf_s, pcd_packet.process_done_perf_s),
                )
                self._emit_debug_line(
                    seq=pcd_packet.seq,
                    timing=timing,
                    controller_points=pcd_packet.controller_point_count,
                    object_points=pcd_packet.object_point_count,
                    dropped_capture_frames=pcd_packet.dropped_capture_frames,
                    dropped_seg_frames=pcd_packet.dropped_seg_frames,
                    filter_telemetry=pcd_packet.filter_telemetry,
                )
                continue

            mask_packet = self.mask_slot.get_latest_after(last_logged_seq)
            if mask_packet is not None:
                last_logged_seq = mask_packet.seq
                timing = replace(
                    mask_packet.timing,
                    receive_to_render_ms=_elapsed_ms(mask_packet.receive_perf_s, mask_packet.process_done_perf_s),
                )
                self._emit_debug_line(
                    seq=mask_packet.seq,
                    timing=timing,
                    controller_points=int(np.count_nonzero(mask_packet.controller_mask)),
                    object_points=int(np.count_nonzero(mask_packet.object_mask)),
                    dropped_capture_frames=mask_packet.dropped_capture_frames,
                    dropped_seg_frames=self.mask_slot.dropped_count,
                )
                continue

            depth_packet = self.depth_profile_slot.get_latest_after(last_logged_seq)
            if depth_packet is not None:
                last_logged_seq = depth_packet.seq
                timing = replace(
                    depth_packet.timing,
                    receive_to_render_ms=_elapsed_ms(depth_packet.receive_perf_s, depth_packet.process_done_perf_s),
                )
                self._emit_debug_line(
                    seq=depth_packet.seq,
                    timing=timing,
                    dropped_capture_frames=depth_packet.dropped_capture_frames,
                )
                continue

            frame = self.capture_slot.get_latest_after(last_logged_seq)
            if frame is not None:
                last_logged_seq = frame.seq
                timing = replace(frame.timing, receive_to_render_ms=_elapsed_ms(frame.receive_perf_s, now_s))
                self._emit_debug_line(
                    seq=frame.seq,
                    timing=timing,
                    dropped_capture_frames=self.capture_slot.dropped_count,
                )

    def _maybe_log_debug(self, *, packet: MaskedPcdPacket, timing: PipelineTiming, now_s: float) -> None:
        if not self.args.debug or now_s - self._last_debug_log_s < DEBUG_LOG_INTERVAL_S:
            return
        self._last_debug_log_s = now_s
        self._emit_debug_line(
            seq=packet.seq,
            timing=timing,
            controller_points=packet.controller_point_count,
            object_points=packet.object_point_count,
            dropped_capture_frames=packet.dropped_capture_frames,
            dropped_seg_frames=packet.dropped_seg_frames,
            filter_telemetry=packet.filter_telemetry,
        )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        apply_demo_preset(args)
        validate_args(args)
        return RealtimeMaskedEdgeTamPcdDemo(args).run()
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"{parser.prog}: error: {exc}\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
