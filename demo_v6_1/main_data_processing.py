#!/usr/bin/env python3
"""Demo v6.1 main data processing runtime."""
# ---------------------------------------------------------------------------
# Table of contents (top-level regions, in file order)
# ---------------------------------------------------------------------------
# 1.  Path bootstrap, imports, module constants
# 2.  Shared dataclasses & packet types (FramePacket, MaskPacket, TrackerMarkerPacket, ...)
#     including RecordedRgbdFrameSource for recording replay input
# 3.  HeadlessCaptureWriter: on-disk artifacts for headless capture runs
# 4.  Lossless pipeline plumbing: StageStats, OrderedPacketQueue, SameSeqPairer
# 5.  CLI: build_parser, apply_demo_preset, derived-mode accessors, validate_args
# 6.  RealSense capture startup
# 7.  Depth backprojection & mask erosion (masked RGB-D -> point clouds)
# 8.  Segmentation (EdgeTAM) helpers & model timing
# 9.  World-Z diagnostics & table-Z filtering
# 10. Tracker query classification, visibility & marker gating
# 11. MainDataProcessingDemo: workers for capture -> segmentation -> tracker/pcd ->
#     filter -> pairing -> headless capture
# 12. main() entry point
from __future__ import annotations

import argparse
from collections import OrderedDict, deque
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, replace
import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Callable, Generic, TypeVar

import numpy as np


def _resolve_repo_root() -> Path:
    """Resolve repo root."""
    candidates: list[Path] = []
    candidates.extend([Path(__file__).resolve().parents[1], Path.cwd()])
    env_root = os.environ.get("QQTT_REPO_ROOT")
    if env_root:
        candidates.append(Path(env_root))
    for candidate in candidates:
        root = candidate.expanduser().resolve()
        if (
            (root / "data_process").is_dir()
            and (root / "demo_v6_1").is_dir()
            and (root / "qqtt").is_dir()
        ):
            return root
    return Path(__file__).resolve().parents[1]


REPO_ROOT = _resolve_repo_root()
REPO_ROOT_STR = str(REPO_ROOT)
if REPO_ROOT_STR in sys.path:
    sys.path.remove(REPO_ROOT_STR)
sys.path.insert(0, REPO_ROOT_STR)


def _repo_relative_path_text(path: str | Path | None) -> str | None:
    """Return the repo relative path text."""
    if path is None:
        return None
    original = Path(path)
    try:
        resolved = original.expanduser().resolve()
    except OSError:
        return str(path)
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


from demo_v6_1.utils.camera import (  # noqa: E402
    CameraIntrinsics,
    SUPPORTED_CAPTURE_FPS,
    SUPPORTED_PROFILES,
    apply_emitter,
    camera_intrinsics_from_rs,
    load_realsense_module,
    parse_profile,
    resolve_serial,
    rs_extrinsics_to_matrix,
    rs_intrinsics_to_matrix,
    rs_translation_norm,
)
from demo_v6_1.utils.concurrency import (  # noqa: E402
    LatestSlot,
    elapsed_ms as _elapsed_ms,
    packet_seq as _packet_seq,
)
from demo_v6_1.utils.ffs_align import FfsIrToColorAligner, validate_ffs_paths  # noqa: E402
from demo_v6_1.utils.pcd_filter import (  # noqa: E402
    FilterBudgetController,
    FilterInput,
    FilterOutput,
    voxel_cap_indices,
    voxel_density_indices,
)
from services.ffs_remote.protocol import (  # noqa: E402
    COMPRESSION_MODES,
    RETURN_TYPES,
    SPARSE_RETURN_TYPES,
)
from demo_v6_1 import main_warmup  # noqa: E402
from demo_v6_1 import shape_prior_warmup  # noqa: E402
from demo_v6_1.tracking import CONTROLLER_FINAL_COUNT  # noqa: E402
from demo_v6_1.main_warmup import InitialMaskBundle  # noqa: E402
from data_process.depth_backends.ffs_defaults import (  # noqa: E402
    DEFAULT_FFS_MAX_DISP,
    DEFAULT_FFS_MODEL_NAME,
    DEFAULT_FFS_REPO,
    DEFAULT_FFS_TRT_BUILDER_OPTIMIZATION_LEVEL,
    DEFAULT_FFS_TRT_ENGINE_SIZE,
    DEFAULT_FFS_TRT_TWO_STAGE_MODEL_DIR,
    DEFAULT_FFS_VALID_ITERS,
)
from qqtt.env.camera.table_calibration import (  # noqa: E402
    TABLE_WORLD_FRAME_KIND,
    TableCalibrationLoadError,
    load_table_calibration_transforms,
)
from demo_v6_1.utils.projection import lift_tracks_yx_to_world  # noqa: E402
from demo_v6_1.utils.query_rainbow import query_rainbow_colors_from_points_yx_rgb_u8  # noqa: E402
from demo_v6_1.phystwin_strict_product import (  # noqa: E402
    COMPATIBILITY_TARGET_PHYSTWIN,
    DEFAULT_TRACKING_PRODUCT_BACKEND,
    PHYSTWIN_STRICT_EXECUTION_MODE,
    TRACKING_PRODUCT_BACKEND_REALTIME_OVERLAY,
    TRACKING_PRODUCT_BACKENDS,
    finalize_headless_capture,
    normalize_tracking_product_backend,
    prepare_phystwin_frame,
    tracking_product_backend_is_strict,
    write_prepared_phystwin_frame,
)
from qqtt.tracking.backends.point_tracker_adapter import (  # noqa: E402
    TRACKER_BACKEND_NONE,
    TRACKER_BACKEND_TAPNEXTPP,
    TRACKER_BACKENDS,
    PointTrackerAdapterConfig,
    build_point_tracker_adapter_factory,
    normalize_tracker_backend,
)
from qqtt.tracking.sampling import PHYSTWIN_DENSE_QUERY_POINTS, sample_phystwin_dense  # noqa: E402


# ---------------------------------------------------------------------------
# Module constants: modes, defaults, geometry layer names, object/track ids
# ---------------------------------------------------------------------------
DEFAULT_MODEL_ID = str(Path("vendor") / "demo_runtime" / "EdgeTAM-hf")
DEFAULT_PROFILE = "848x480"
DEFAULT_FPS = 60
DEFAULT_DEVICE = "cuda"
DEFAULT_DTYPE = "bfloat16"
DEFAULT_COMPILE_MODE = "vision-reduce-overhead"
COMPILE_MODES = ("vision-reduce-overhead",)
INIT_MODES = ("sam31-first-frame", "saved-masks")
DEFAULT_INIT_MODE = "sam31-first-frame"
TRACK_MODE_CONTROLLER_OBJECT = "controller-object"
TRACK_MODE_OBJECT_ONLY = "object-only"
TRACK_MODE_CONTROLLER_ONLY = "controller-only"
TRACK_MODE_NONE = "none"
TRACK_MODES = (TRACK_MODE_CONTROLLER_OBJECT, TRACK_MODE_OBJECT_ONLY, TRACK_MODE_CONTROLLER_ONLY, TRACK_MODE_NONE)
DEFAULT_TRACK_MODE = "controller-object"
DEPTH_SOURCES = ("ffs", "ffs_remote", "realsense", "none")
DEFAULT_DEPTH_SOURCE = "ffs"
INPUT_SOURCE_LIVE = "live"
INPUT_SOURCE_FAKE_LIVE = "fake-live"
INPUT_SOURCE_RECORDING = "recording"
INPUT_SOURCES = (INPUT_SOURCE_LIVE, INPUT_SOURCE_FAKE_LIVE, INPUT_SOURCE_RECORDING)
DEFAULT_FAKE_LIVE_CASE = Path("data_collect/stuffed_animal_hand_both_eval_5fps_normal")
PCD_MODES = ("masked", "none")
DEFAULT_PCD_MODE = "masked"
DEMO_VISUAL_MODE_PCD = "pcd"
DEMO_VISUAL_MODE_TRACKING = "tracking"
DEMO_VISUAL_MODES = (DEMO_VISUAL_MODE_PCD, DEMO_VISUAL_MODE_TRACKING)
DEFAULT_DEMO_VISUAL_MODE = DEMO_VISUAL_MODE_TRACKING
PCD_FILTER_MODES = ("async", "sync", "none")
PCD_FILTER_NONE = "none"
PCD_FILTER_PT_FILTER = "pt-filter"
PCD_FILTER_ENHANCED_PT = "enhanced-pt"
PCD_FILTER_VOXEL_DENSITY = "voxel-density"
PCD_FILTERS = (PCD_FILTER_NONE, PCD_FILTER_PT_FILTER, PCD_FILTER_ENHANCED_PT, PCD_FILTER_VOXEL_DENSITY)
PCD_FILTER_PRESET_ORIGINAL = "original"
PCD_FILTER_PRESET_PT = "pt"
PCD_FILTER_PRESET_ENHANCED_PT = PCD_FILTER_ENHANCED_PT
PCD_FILTER_PRESETS = (PCD_FILTER_PRESET_ORIGINAL, PCD_FILTER_PRESET_PT, PCD_FILTER_PRESET_ENHANCED_PT)
TRACKER_QUERY_SOURCE_UNION_MASK = "object_controller_union_mask"
TRACKER_QUERY_SOURCE_PCD_FILTER_RESIDUAL = "pcd_filter_residual"
TRACKER_MARKER_GATE_TARGET_MASK_DEPTH = "target_mask_depth"
TRACKER_MARKER_GATE_PCD_FILTER_RESIDUAL_TABLE_Z = "pcd_filter_residual_table_z"
TRACKER_MARKER_RETIREMENT_POLICY_DISABLED = "disabled"
TRACKER_MARKER_RETIREMENT_POLICY_PCD_FILTER_RESIDUAL_TABLE_Z_ONCE_FALSE = (
    "pcd_filter_residual_table_z_once_false"
)
FAKE_LIVE_FRAME_SELECTION_POLICY = "drop_source_frames_preserve_recording_time"
DEMO_PRESETS = ("none", "local-ffs-professor")
DEFAULT_DEMO_PRESET = "none"
LOCAL_FFS_PROFESSOR_MAX_POINTS = 20000
LOCAL_FFS_PROFESSOR_FILTER_CAP = 20000
DEFAULT_FILTER_RADIUS_M = 0.01
DEFAULT_FILTER_NB_POINTS = 40
DEFAULT_PCD_MASK_ERODE_PIXELS = 0
DEFAULT_OBJECT_PCD_MASK_ERODE_PIXELS: int | None = None
DEFAULT_CONTROLLER_PCD_MASK_ERODE_PIXELS: int | None = None
DEFAULT_ENHANCED_COMPONENT_VOXEL_SIZE_M = 0.01
DEFAULT_ENHANCED_KEEP_NEAR_MAIN_GAP_M = 0.0
DEFAULT_OBJECT_FILTER = PCD_FILTER_NONE
DEFAULT_CONTROLLER_FILTER = PCD_FILTER_NONE
DEFAULT_OBJECT_FILTER_CAP = 0
DEFAULT_CONTROLLER_FILTER_CAP = 0
DEFAULT_OBJECT_FILTER_KEEP_COMPONENTS = 1
DEFAULT_CONTROLLER_FILTER_KEEP_COMPONENTS = 2
DEFAULT_OBJECT_FILTER_MIN_RETAIN_RATIO = 0.0
DEFAULT_CONTROLLER_FILTER_MIN_RETAIN_RATIO = 0.5
DEFAULT_OBJECT_FILTER_MIN_RAW_RETAIN_RATIO = 0.0
DEFAULT_CONTROLLER_FILTER_MIN_RAW_RETAIN_RATIO = 0.5
DEFAULT_FILTER_MAX_AGE_FRAMES = 3
DEFAULT_LOSSLESS_CONTROLLER_FILTER_MIN_CAP = 2500
DEFAULT_EDGETAM_LIVE_SESSION_KEEP_FRAMES = 64
DEFAULT_EDGETAM_MASK_LOGIT_THRESHOLD = 0.0
DEFAULT_LOCAL_FFS_DEPTH_CACHE_FRAMES = 8
HAND_A_ID = 1
OBJECT_ID = 2
HAND_B_ID = 3
CONTROLLER_ID = HAND_A_ID
EDGE_TAM_OBJECT_LABELS = {
    HAND_A_ID: "hand_a",
    OBJECT_ID: "object",
    HAND_B_ID: "hand_b",
}
CONTROLLER_COLOR_RGB = (255, 96, 32)
OBJECT_COLOR_RGB = (64, 180, 255)
GEOMETRY_CONTROLLER = "masked_edgetam_controller"
GEOMETRY_OBJECT = "masked_edgetam_object"
GEOMETRY_TRACKER_OBJECT = "tapnextpp_tracker_markers_object"
GEOMETRY_TRACKER_CONTROLLER = "tapnextpp_tracker_markers_controller"
COORDINATE_FRAME = "camera_color_frame"
TABLE_Z_M = 0.0
DEFAULT_TABLE_Z_DIAGNOSTIC_THRESHOLDS_M = (0.005, 0.010, 0.020, 0.030)
DEFAULT_TABLE_Z_FILTER_THRESHOLD_M = 0.0
# Origin/data_process table frame: z < 0 is above the table, z > 0 is invalid.
TABLE_Z_ABOVE_DIRECTION = "negative"
TABLE_Z_FILTER_CLASS_OBJECT = "object"
TABLE_Z_FILTER_CLASS_CONTROLLER = "controller"
TABLE_Z_FILTER_CLASS_BOTH = "both"
TABLE_Z_FILTER_CLASSES = (
    TABLE_Z_FILTER_CLASS_OBJECT,
    TABLE_Z_FILTER_CLASS_CONTROLLER,
    TABLE_Z_FILTER_CLASS_BOTH,
)
TRACKER_DISPLAY_SCOPE_CONTROLLER = "controller"
TRACKER_DISPLAY_SCOPE_OBJECT = "object"
TRACKER_DISPLAY_SCOPE_UNION = "union"
TRACKER_DISPLAY_SCOPES = (
    TRACKER_DISPLAY_SCOPE_CONTROLLER,
    TRACKER_DISPLAY_SCOPE_OBJECT,
    TRACKER_DISPLAY_SCOPE_UNION,
)
DEFAULT_TRACKER_DISPLAY_SCOPE = TRACKER_DISPLAY_SCOPE_UNION
DEFAULT_TRACKER_BACKEND = TRACKER_BACKEND_NONE
DEFAULT_TRACKER_QUERY_COUNT = PHYSTWIN_DENSE_QUERY_POINTS
DEFAULT_TRACKER_SEED = 42
DEFAULT_TRACKER_MARKER_POINT_SIZE = 8.0
QUERY_CONTROLLER_INSTANCE_NONE = 0
QUERY_CONTROLLER_INSTANCE_HAND_A = 1
QUERY_CONTROLLER_INSTANCE_HAND_B = 2
HEADLESS_CAPTURE_SAVED_PCD_SOURCE = "none_filtered"
HEADLESS_CAPTURE_ALLOWED_PCD_FILTERS = (PCD_FILTER_ENHANCED_PT, PCD_FILTER_PT_FILTER, PCD_FILTER_NONE)
DEBUG_LOG_INTERVAL_S = 1.0
DEFAULT_LOSSLESS_INPUT_FPS = 5.0


# ---------------------------------------------------------------------------
# Shared dataclasses & packet types flowing between pipeline stages
# ---------------------------------------------------------------------------
DEFAULT_LOSSLESS_MAX_BACKLOG_SECONDS = 3.0


DEFAULT_RUNTIME_ASSET_ROOT = Path("vendor") / "demo_runtime"
DEFAULT_TAPNET_REPO_DIR = DEFAULT_RUNTIME_ASSET_ROOT / "tapnet"
DEFAULT_TAPNEXTPP_CHECKPOINT = (
    DEFAULT_RUNTIME_ASSET_ROOT / "checkpoints" / "tapnextpp" / "tapnextpp_ckpt.pt"
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
    source_timestamp_s: float | None = None
    source_frame_index: int | None = None
    source_step: int | None = None


class LiveLatestFrameSampler:
    """Sample the latest live camera frame on a fixed output cadence."""

    def __init__(self, sample_fps: float) -> None:
        """Initialize LiveLatestFrameSampler."""
        fps = float(sample_fps)
        if fps <= 0.0:
            raise ValueError("live latest sampler FPS must be positive")
        self.period_s = 1.0 / fps
        self._next_sample_s: float | None = None
        self._pending_packet: FramePacket | None = None

    def start(self, *, first_publish_s: float) -> None:
        """Start fixed-cadence sampling after the first published frame."""
        self._next_sample_s = float(first_publish_s) + self.period_s
        self._pending_packet = None

    def put_latest(self, packet: FramePacket) -> None:
        """Store the newest live input frame."""
        if self._next_sample_s is None:
            raise RuntimeError("live latest sampler must be started before use")
        self._pending_packet = packet

    def pop_due(self, *, now_s: float) -> tuple[FramePacket, float] | None:
        """Return the pending packet if its fixed output tick is due."""
        if self._next_sample_s is None:
            return None
        if self._pending_packet is None or float(now_s) < self._next_sample_s:
            return None
        packet = self._pending_packet
        sample_s = self._next_sample_s
        self._pending_packet = None
        while self._next_sample_s <= float(now_s):
            self._next_sample_s += self.period_s
        return packet, sample_s


@dataclass(frozen=True)
class FatalWorkerError:
    stage: str
    exc_type: str
    message: str

    def log_message(self) -> str:
        """Format the worker failure for logs and HUD output."""
        return f"{self.stage} failed: {self.exc_type}: {self.message}"


@dataclass(frozen=True)
class RecordedRgbdFrameRef:
    step: int
    timestamp_s: float
    color_path: Path
    depth_path: Path | None = None
    ir_left_path: Path | None = None
    ir_right_path: Path | None = None


class _NoopPipeline:
    def stop(self) -> None:
        """Stop _NoopPipeline."""
        return


class RecordedRgbdFrameSource:
    def __init__(
        self,
        case_path: str | Path,
        *,
        replay_fps: float = 0.0,
        camera_index: int = 0,
        depth_source: str = "realsense",
    ) -> None:
        """Initialize RecordedRgbdFrameSource."""
        self.case_path = _resolve_path(case_path)
        self.camera_index = int(camera_index)
        self.depth_source = str(depth_source)
        if self.depth_source not in DEPTH_SOURCES:
            raise ValueError(f"recording replay depth_source must be one of {DEPTH_SOURCES}")
        self.requires_depth = self.depth_source == "realsense"
        self.requires_ir = self.depth_source in {"ffs", "ffs_remote"}
        self.metadata_path = self.case_path / "metadata.json"
        if not self.metadata_path.is_file():
            raise FileNotFoundError(f"recording metadata not found: {self.metadata_path}")
        try:
            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"recording metadata is not valid JSON: {self.metadata_path}") from exc
        self.metadata: dict[str, Any] = metadata
        streams_present = {str(item) for item in metadata.get("streams_present", [])}
        if "color" not in streams_present:
            raise ValueError("recording replay requires streams_present to include color")
        if self.requires_depth and "depth" not in streams_present:
            raise ValueError("RealSense recording replay requires streams_present to include depth")
        if self.requires_ir and not {"ir_left", "ir_right"}.issubset(streams_present):
            raise ValueError("FFS fake-live replay requires streams_present to include ir_left and ir_right")
        recording_by_camera = metadata.get("recording")
        if not isinstance(recording_by_camera, dict):
            raise ValueError("recording metadata must contain a recording object")
        camera_key = str(self.camera_index)
        camera_recording = recording_by_camera.get(camera_key)
        if not isinstance(camera_recording, dict) or not camera_recording:
            raise ValueError(f"recording metadata has no frames for camera {self.camera_index}")
        self.k_color = self._camera_matrix(metadata, "K_color", fallback_key="intrinsics")
        self.intrinsics = CameraIntrinsics(
            fx=float(self.k_color[0, 0]),
            fy=float(self.k_color[1, 1]),
            cx=float(self.k_color[0, 2]),
            cy=float(self.k_color[1, 2]),
        )
        self.depth_scale_m_per_unit = self._camera_float(metadata, "depth_scale_m_per_unit")
        self.serial = self._camera_string(metadata, "serial_numbers", default=f"recording-cam{self.camera_index}")
        self.width, self.height = self._resolve_dimensions(metadata)
        self.recording_fps = self._resolve_recording_fps(metadata)
        self.effective_fps = self._resolve_replay_fps(float(replay_fps))
        self.k_ir_left: np.ndarray | None = None
        self.t_ir_left_to_color: np.ndarray | None = None
        self.ir_baseline_m = 0.0
        self.has_ir_stereo = False
        if {"ir_left", "ir_right"}.issubset(streams_present):
            try:
                self.k_ir_left = self._camera_matrix(metadata, "K_ir_left")
                self.t_ir_left_to_color = self._camera_transform(metadata, "T_ir_left_to_color")
                self.ir_baseline_m = self._camera_baseline(metadata)
                self.has_ir_stereo = True
            except ValueError:
                if self.requires_ir:
                    raise
        if self.requires_ir and not self.has_ir_stereo:
            raise ValueError("FFS fake-live replay requires IR stereo calibration in metadata")
        self.frames = self._build_frame_refs(camera_recording)
        self._recording_elapsed_s = self._build_recording_elapsed_s(self.frames)

    @property
    def frame_count(self) -> int:
        """Return the frame count."""
        return len(self.frames)

    @property
    def steps(self) -> list[int]:
        """Return the steps."""
        return [frame.step for frame in self.frames]

    def make_runtime(self) -> RealtimeCameraRuntime:
        """Create a replay runtime wrapper around recorded RGB-D frames."""
        return RealtimeCameraRuntime(
            pipeline=_NoopPipeline(),
            align=None,
            serial=self.serial,
            intrinsics=self.intrinsics,
            depth_scale_m_per_unit=self.depth_scale_m_per_unit,
            k_color=self.k_color,
            k_ir_left=self.k_ir_left,
            t_ir_left_to_color=self.t_ir_left_to_color,
            ir_baseline_m=float(self.ir_baseline_m),
        )

    def read_packet(
        self,
        *,
        seq: int,
        frame_index: int | None = None,
        wait_ms: float = 0.0,
        receive_perf_s: float | None = None,
        frame_copy_ms: float | None = None,
    ) -> FramePacket:
        """Read packet."""
        packet_seq = int(seq)
        source_index = packet_seq if frame_index is None else int(frame_index)
        if source_index < 0 or source_index >= len(self.frames):
            raise IndexError(
                f"recording replay frame_index {source_index} out of range for {len(self.frames)} frames"
            )
        ref = self.frames[source_index]
        copy_start_s = time.perf_counter()
        color_bgr = self._load_color_bgr(ref.color_path)
        depth_u16 = self._load_depth_u16(ref.depth_path) if ref.depth_path is not None else None
        ir_left_u8 = self._load_gray_u8(ref.ir_left_path) if ref.ir_left_path is not None else None
        ir_right_u8 = self._load_gray_u8(ref.ir_right_path) if ref.ir_right_path is not None else None
        copy_done_s = time.perf_counter()
        if depth_u16 is not None and color_bgr.shape[:2] != depth_u16.shape:
            raise ValueError(
                f"recording color/depth shape mismatch for step {ref.step}: "
                f"{tuple(color_bgr.shape[:2])} vs {tuple(depth_u16.shape)}"
            )
        if (ir_left_u8 is None) != (ir_right_u8 is None):
            raise ValueError(f"recording IR pair is incomplete for step {ref.step}")
        if ir_left_u8 is not None and ir_left_u8.shape != ir_right_u8.shape:
            raise ValueError(
                f"recording IR left/right shape mismatch for step {ref.step}: "
                f"{tuple(ir_left_u8.shape)} vs {tuple(ir_right_u8.shape)}"
            )
        if tuple(color_bgr.shape[:2]) != (self.height, self.width):
            raise ValueError(
                f"recording frame shape {tuple(color_bgr.shape[:2])} does not match metadata "
                f"height/width {(self.height, self.width)} for step {ref.step}"
            )
        receive_s = copy_done_s if receive_perf_s is None else float(receive_perf_s)
        copy_ms = _elapsed_ms(copy_start_s, copy_done_s) if frame_copy_ms is None else float(frame_copy_ms)
        return FramePacket(
            seq=packet_seq,
            color_bgr=color_bgr,
            depth_source=self.depth_source,
            intrinsics=self.intrinsics,
            depth_scale_m_per_unit=self.depth_scale_m_per_unit,
            receive_perf_s=receive_s,
            timing=PipelineTiming(wait_ms=float(wait_ms), align_ms=0.0, frame_copy_ms=copy_ms),
            depth_u16=depth_u16,
            ir_left_u8=ir_left_u8,
            ir_right_u8=ir_right_u8,
            k_ir_left=self.k_ir_left if ir_left_u8 is not None else None,
            t_ir_left_to_color=self.t_ir_left_to_color if ir_left_u8 is not None else None,
            k_color=self.k_color,
            ir_baseline_m=float(self.ir_baseline_m) if ir_left_u8 is not None else 0.0,
            source_timestamp_s=float(ref.timestamp_s),
            source_frame_index=int(source_index),
            source_step=int(ref.step),
        )

    def read_preview_packet(
        self,
        *,
        seq: int,
        frame_index: int | None = None,
        wait_ms: float = 0.0,
        receive_perf_s: float | None = None,
    ) -> FramePacket:
        """Read preview packet."""
        packet_seq = int(seq)
        source_index = packet_seq if frame_index is None else int(frame_index)
        if source_index < 0 or source_index >= len(self.frames):
            raise IndexError(
                f"recording preview frame_index {source_index} out of range for {len(self.frames)} frames"
            )
        ref = self.frames[source_index]
        copy_start_s = time.perf_counter()
        color_bgr = self._load_color_bgr(ref.color_path)
        copy_done_s = time.perf_counter()
        if tuple(color_bgr.shape[:2]) != (self.height, self.width):
            raise ValueError(
                f"recording preview frame shape {tuple(color_bgr.shape[:2])} does not match metadata "
                f"height/width {(self.height, self.width)} for step {ref.step}"
            )
        receive_s = copy_done_s if receive_perf_s is None else float(receive_perf_s)
        return FramePacket(
            seq=packet_seq,
            color_bgr=color_bgr,
            depth_source=self.depth_source,
            intrinsics=self.intrinsics,
            depth_scale_m_per_unit=self.depth_scale_m_per_unit,
            receive_perf_s=receive_s,
            timing=PipelineTiming(
                wait_ms=float(wait_ms),
                align_ms=0.0,
                frame_copy_ms=_elapsed_ms(copy_start_s, copy_done_s),
            ),
            k_color=self.k_color,
            source_timestamp_s=float(ref.timestamp_s),
            source_frame_index=int(source_index),
            source_step=int(ref.step),
        )

    def _camera_matrix(self, metadata: dict[str, Any], key: str, *, fallback_key: str | None = None) -> np.ndarray:
        """Return the camera matrix."""
        values = metadata.get(key)
        if values is None and fallback_key is not None:
            values = metadata.get(fallback_key)
        if not isinstance(values, list) or self.camera_index >= len(values) or values[self.camera_index] is None:
            raise ValueError(f"recording metadata missing {key} for camera {self.camera_index}")
        matrix = np.asarray(values[self.camera_index], dtype=np.float32)
        if matrix.shape != (3, 3):
            raise ValueError(f"recording metadata {key}[{self.camera_index}] must be 3x3")
        if float(matrix[0, 0]) <= 0.0 or float(matrix[1, 1]) <= 0.0:
            raise ValueError(f"recording metadata {key}[{self.camera_index}] has non-positive focal length")
        return np.ascontiguousarray(matrix, dtype=np.float32)

    def _camera_transform(self, metadata: dict[str, Any], key: str) -> np.ndarray:
        """Return the camera transform."""
        values = metadata.get(key)
        if not isinstance(values, list) or self.camera_index >= len(values) or values[self.camera_index] is None:
            raise ValueError(f"recording metadata missing {key} for camera {self.camera_index}")
        matrix = np.asarray(values[self.camera_index], dtype=np.float32)
        if matrix.shape != (4, 4):
            raise ValueError(f"recording metadata {key}[{self.camera_index}] must be 4x4")
        return np.ascontiguousarray(matrix, dtype=np.float32)

    def _camera_baseline(self, metadata: dict[str, Any]) -> float:
        """Return the camera baseline."""
        values = metadata.get("ir_baseline_m")
        if isinstance(values, list) and self.camera_index < len(values) and values[self.camera_index] is not None:
            value = float(values[self.camera_index])
            if value <= 0.0:
                raise ValueError(f"recording metadata ir_baseline_m[{self.camera_index}] must be positive")
            return value
        transform = self._camera_transform(metadata, "T_ir_left_to_right")
        baseline = float(np.linalg.norm(transform[:3, 3]))
        if baseline <= 0.0:
            raise ValueError(f"recording metadata T_ir_left_to_right[{self.camera_index}] has non-positive baseline")
        return baseline

    def _camera_float(self, metadata: dict[str, Any], key: str) -> float:
        """Return the camera float."""
        values = metadata.get(key)
        if not isinstance(values, list) or self.camera_index >= len(values) or values[self.camera_index] is None:
            raise ValueError(f"recording metadata missing {key} for camera {self.camera_index}")
        value = float(values[self.camera_index])
        if value <= 0.0:
            raise ValueError(f"recording metadata {key}[{self.camera_index}] must be positive")
        return value

    def _camera_string(self, metadata: dict[str, Any], key: str, *, default: str) -> str:
        """Return the camera string."""
        values = metadata.get(key)
        if isinstance(values, list) and self.camera_index < len(values) and values[self.camera_index] is not None:
            return str(values[self.camera_index])
        return default

    def _resolve_dimensions(self, metadata: dict[str, Any]) -> tuple[int, int]:
        """Resolve dimensions."""
        wh = metadata.get("WH")
        if not isinstance(wh, list) or len(wh) != 2:
            raise ValueError("recording metadata missing WH")
        width = int(wh[0])
        height = int(wh[1])
        if width <= 0 or height <= 0:
            raise ValueError("recording metadata WH must be positive")
        return width, height

    def _resolve_recording_fps(self, metadata: dict[str, Any]) -> float:
        """Resolve the recording FPS from case metadata."""
        try:
            fps = float(metadata.get("fps", 0.0))
        except (TypeError, ValueError):
            fps = 0.0
        return fps if fps > 0.0 else 30.0

    def _resolve_replay_fps(self, replay_fps: float) -> float:
        """Resolve the effective replay FPS for fake-live playback."""
        return float(replay_fps) if float(replay_fps) > 0.0 else float(self.recording_fps)

    def _build_recording_elapsed_s(self, frames: list[RecordedRgbdFrameRef]) -> np.ndarray:
        """Build recording elapsed s."""
        timestamps = np.asarray([float(frame.timestamp_s) for frame in frames], dtype=np.float64)
        if len(timestamps) and np.isfinite(timestamps).all() and np.all(np.diff(timestamps) >= 0.0):
            return np.ascontiguousarray(timestamps - timestamps[0], dtype=np.float64)
        frame_indices = np.arange(len(frames), dtype=np.float64)
        return np.ascontiguousarray(frame_indices / float(self.recording_fps), dtype=np.float64)

    def source_index_for_recording_elapsed_s(self, elapsed_s: float) -> int:
        """Return the source frame index nearest a recording elapsed time."""
        if len(self.frames) <= 1:
            return 0
        elapsed = max(0.0, float(elapsed_s))
        index = int(np.searchsorted(self._recording_elapsed_s, elapsed + 1e-9, side="right") - 1)
        return max(0, min(index, len(self.frames) - 1))

    def _build_frame_refs(self, camera_recording: dict[str, Any]) -> list[RecordedRgbdFrameRef]:
        """Build frame refs."""
        refs: list[RecordedRgbdFrameRef] = []
        color_dir = self.case_path / "color" / str(self.camera_index)
        depth_dir = self.case_path / "depth" / str(self.camera_index)
        ir_left_dir = self.case_path / "ir_left" / str(self.camera_index)
        ir_right_dir = self.case_path / "ir_right" / str(self.camera_index)
        for step_text, timestamp in sorted(camera_recording.items(), key=lambda item: int(item[0])):
            step = int(step_text)
            color_path = color_dir / f"{step}.png"
            depth_path = depth_dir / f"{step}.npy"
            if not color_path.is_file():
                raise FileNotFoundError(f"recording color frame missing: {color_path}")
            if self.requires_depth and not depth_path.is_file():
                raise FileNotFoundError(f"recording depth frame missing: {depth_path}")
            ir_left_path = ir_left_dir / f"{step}.png"
            ir_right_path = ir_right_dir / f"{step}.png"
            if self.requires_ir:
                if not ir_left_path.is_file():
                    raise FileNotFoundError(f"recording IR left frame missing: {ir_left_path}")
                if not ir_right_path.is_file():
                    raise FileNotFoundError(f"recording IR right frame missing: {ir_right_path}")
            optional_ir_pair = self.has_ir_stereo and ir_left_path.is_file() and ir_right_path.is_file()
            refs.append(
                RecordedRgbdFrameRef(
                    step=step,
                    timestamp_s=float(timestamp),
                    color_path=color_path,
                    depth_path=depth_path if self.requires_depth else None,
                    ir_left_path=ir_left_path if self.requires_ir or optional_ir_pair else None,
                    ir_right_path=ir_right_path if self.requires_ir or optional_ir_pair else None,
                )
            )
        if not refs:
            raise ValueError(f"recording has no complete fake-live frames for camera {self.camera_index}")
        return refs

    def _load_color_bgr(self, path: Path) -> np.ndarray:
        """Load color BGR."""
        try:
            from PIL import Image

            with Image.open(path) as image:
                rgb = np.asarray(image.convert("RGB"))
        except Exception as exc:
            raise ValueError(f"failed to load recording color frame {path}: {exc}") from exc
        return np.ascontiguousarray(rgb[:, :, ::-1], dtype=np.uint8)

    def _load_depth_u16(self, path: Path) -> np.ndarray:
        """Load depth u16."""
        try:
            depth = np.load(path)
        except Exception as exc:
            raise ValueError(f"failed to load recording depth frame {path}: {exc}") from exc
        depth_u16 = np.asarray(depth)
        if depth_u16.ndim != 2:
            raise ValueError(f"recording depth frame must be 2D: {path}")
        if depth_u16.dtype != np.uint16:
            depth_u16 = depth_u16.astype(np.uint16, copy=False)
        return np.ascontiguousarray(depth_u16)

    def _load_gray_u8(self, path: Path) -> np.ndarray:
        """Load gray u8."""
        try:
            from PIL import Image

            with Image.open(path) as image:
                gray = np.asarray(image.convert("L"))
        except Exception as exc:
            raise ValueError(f"failed to load recording IR frame {path}: {exc}") from exc
        return np.ascontiguousarray(gray, dtype=np.uint8)


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
    hand_a_mask: np.ndarray | None = None
    hand_b_mask: np.ndarray | None = None
    depth_u16: np.ndarray | None = None
    ir_left_u8: np.ndarray | None = None
    ir_right_u8: np.ndarray | None = None
    k_ir_left: np.ndarray | None = None
    t_ir_left_to_color: np.ndarray | None = None
    k_color: np.ndarray | None = None
    ir_baseline_m: float = 0.0
    source_timestamp_s: float | None = None
    source_frame_index: int | None = None
    source_step: int | None = None


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
    coordinate_frame: str = COORDINATE_FRAME
    source_timestamp_s: float | None = None
    source_frame_index: int | None = None
    source_step: int | None = None
    shape_prior_points_m: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float32))
    shape_prior_colors_rgb_u8: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.uint8))
    shape_prior_status: str = shape_prior_warmup.STATUS_DISABLED
    shape_prior_profile: dict[str, Any] = field(default_factory=dict)

    @property
    def controller_point_count(self) -> int:
        """Return the controller point count."""
        return int(self.controller_xyz_m.shape[0])

    @property
    def object_point_count(self) -> int:
        """Return the object point count."""
        return int(self.object_xyz_m.shape[0])

    @property
    def point_count(self) -> int:
        """Return the point count."""
        return self.controller_point_count + self.object_point_count

    @property
    def shape_prior_point_count(self) -> int:
        """Return the shape prior point count."""
        return int(np.asarray(self.shape_prior_points_m, dtype=np.float32).reshape(-1, 3).shape[0])


@dataclass(frozen=True)
class MarkerResidualAudit:
    pixels_yx: np.ndarray
    valid: np.ndarray
    violation: np.ndarray
    checked_count: int
    violation_count: int
    gate: str = TRACKER_MARKER_GATE_PCD_FILTER_RESIDUAL_TABLE_Z


def _fit_bool_array(values: np.ndarray, length: int, *, fill: bool = False) -> np.ndarray:
    """Fit a boolean vector to the requested length."""
    output = np.full((max(0, int(length)),), bool(fill), dtype=bool)
    arr = np.asarray(values, dtype=bool).reshape(-1)
    count = min(len(arr), len(output))
    if count:
        output[:count] = arr[:count]
    return output


def _remaining_query_class_counts(
    alive_mask: np.ndarray,
    *,
    query_is_object: np.ndarray,
    query_is_controller: np.ndarray,
    query_controller_instance_id: np.ndarray,
) -> tuple[int, int, int, int]:
    """Return the remaining query class counts."""
    alive = np.asarray(alive_mask, dtype=bool).reshape(-1)
    count = int(alive.shape[0])
    is_object = _fit_bool_array(query_is_object, count)
    is_controller = _fit_bool_array(query_is_controller, count)
    # Instance ids fitted to the alive-mask length (truncate or zero-pad), int analog of _fit_bool_array.
    instance_id = np.zeros((count,), dtype=np.int64)
    ids = np.asarray(query_controller_instance_id, dtype=np.int64).reshape(-1)
    fit_count = min(len(ids), count)
    if fit_count:
        instance_id[:fit_count] = ids[:fit_count]
    hand_a = alive & (instance_id == QUERY_CONTROLLER_INSTANCE_HAND_A)
    hand_b = alive & (instance_id == QUERY_CONTROLLER_INSTANCE_HAND_B)
    controller = alive & (is_controller | hand_a | hand_b)
    obj = alive & is_object & ~controller
    return (
        int(np.count_nonzero(obj)),
        int(np.count_nonzero(controller)),
        int(np.count_nonzero(hand_a)),
        int(np.count_nonzero(hand_b)),
    )


@dataclass(frozen=True)
class TrackerMarkerPacket:
    seq: int
    marker_xyz_m: np.ndarray
    marker_colors_rgb_u8: np.ndarray
    query_rgb_u8: np.ndarray
    query_points_yx: np.ndarray
    tracks_yx: np.ndarray
    visibility: np.ndarray
    query_is_object: np.ndarray
    query_is_controller: np.ndarray
    receive_perf_s: float
    process_done_perf_s: float
    query_count: int
    consistent_visible_count: int = 0
    model_ms: float = 0.0
    lift_ms: float = 0.0
    e2e_ms: float = 0.0
    backend: str = TRACKER_BACKEND_TAPNEXTPP
    display_scope: str = DEFAULT_TRACKER_DISPLAY_SCOPE
    query_indices: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.int64))
    query_target_id: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.int64))
    query_controller_instance_id: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.int64))
    query_all_target_id: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.int64))
    query_all_controller_instance_id: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.int64))
    hand_a_query_count: int = 0
    hand_b_query_count: int = 0
    object_query_count: int = 0
    marker_pixels_yx: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.int64))
    marker_residual_valid: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=bool))
    marker_residual_violation: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=bool))
    marker_residual_checked_count: int = 0
    marker_residual_violation_count: int = 0
    marker_residual_gate: str = TRACKER_MARKER_GATE_PCD_FILTER_RESIDUAL_TABLE_Z
    query_alive_mask: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=bool))
    remaining_query_count: int = -1
    remaining_object_query_count: int = -1
    remaining_controller_query_count: int = -1
    remaining_hand_a_query_count: int = -1
    remaining_hand_b_query_count: int = -1
    retired_query_count: int = -1
    all_tracks_yx: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.float32))
    all_tracker_visibility: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.float32))
    coordinate_frame: str = COORDINATE_FRAME

    def __post_init__(self) -> None:
        """Validate and normalize the dataclass state after initialization."""
        alive = np.asarray(self.query_alive_mask, dtype=bool).reshape(-1)
        query_count = max(0, int(self.query_count))
        if alive.size == 0 and query_count > 0:
            alive = np.ones((query_count,), dtype=bool)
        elif alive.size != query_count and query_count > 0:
            fitted = np.zeros((query_count,), dtype=bool)
            count = min(int(alive.size), query_count)
            if count:
                fitted[:count] = alive[:count]
            alive = fitted
        alive = np.ascontiguousarray(alive, dtype=bool)
        object.__setattr__(self, "query_alive_mask", alive)
        if int(self.remaining_query_count) < 0:
            object.__setattr__(self, "remaining_query_count", int(np.count_nonzero(alive)))
        if int(self.retired_query_count) < 0:
            object.__setattr__(
                self,
                "retired_query_count",
                max(0, query_count - int(np.count_nonzero(alive))),
            )
        if (
            int(self.remaining_object_query_count) < 0
            or int(self.remaining_controller_query_count) < 0
            or int(self.remaining_hand_a_query_count) < 0
            or int(self.remaining_hand_b_query_count) < 0
        ):
            object_count, controller_count, hand_a_count, hand_b_count = _remaining_query_class_counts(
                alive,
                query_is_object=np.empty((0,), dtype=bool),
                query_is_controller=np.empty((0,), dtype=bool),
                query_controller_instance_id=self.query_all_controller_instance_id,
            )
            if int(self.remaining_object_query_count) < 0:
                object.__setattr__(self, "remaining_object_query_count", object_count)
            if int(self.remaining_controller_query_count) < 0:
                object.__setattr__(self, "remaining_controller_query_count", controller_count)
            if int(self.remaining_hand_a_query_count) < 0:
                object.__setattr__(self, "remaining_hand_a_query_count", hand_a_count)
            if int(self.remaining_hand_b_query_count) < 0:
                object.__setattr__(self, "remaining_hand_b_query_count", hand_b_count)

    @property
    def marker_count(self) -> int:
        """Return the marker count."""
        return int(self.marker_xyz_m.shape[0])


def _formal_chunk_rows_gated(*, warmup_anchor_written: bool, shape_prior_status: str) -> bool:
    """design_spec.md warmup/formal timeline split.

    Rows always write until a chunk-ready warmup anchor row has landed (live
    RealSense can emit an invalid strict pair before color-aligned PCD is
    ready; the bridge trims those, so they must not consume the frame-0
    slot). After the anchor, frames processed while the shape prior is still
    computing stay OUT of the formal final_data chunk timeline (they keep
    feeding the trackers and the left preview, which pace by
    input_frames.jsonl). The first frame after the prior is ready becomes
    output frame 1, stitched directly after warmup frame 0 under the
    operator hold-still convention. Terminal states (ready/disabled/failed)
    lift the gate — a failed prior must surface through the chunk bridge's
    shape-prior error path instead of silently stalling the row stream.
    """
    if not warmup_anchor_written:
        return False
    return str(shape_prior_status) in (
        shape_prior_warmup.STATUS_PENDING,
        shape_prior_warmup.STATUS_RUNNING,
    )


def _full_tracker_arrays_for_prepared_frame(packet: TrackerMarkerPacket) -> tuple[np.ndarray, np.ndarray]:
    """Return the full tracker arrays for prepared frame."""
    query_count = int(np.asarray(packet.query_points_yx, dtype=np.float32).reshape(-1, 2).shape[0])
    all_tracks = np.asarray(packet.all_tracks_yx, dtype=np.float32).reshape(-1, 2)
    all_visibility = np.asarray(packet.all_tracker_visibility, dtype=bool).reshape(-1)
    if all_tracks.shape[0] == query_count and all_visibility.shape[0] == query_count:
        return (
            np.ascontiguousarray(all_tracks, dtype=np.float32),
            np.ascontiguousarray(all_visibility, dtype=bool),
        )

    active_tracks = np.asarray(packet.tracks_yx, dtype=np.float32).reshape(-1, 2)
    active_visibility = np.asarray(packet.visibility, dtype=bool).reshape(-1)
    if active_tracks.shape[0] == query_count and active_visibility.shape[0] == query_count:
        return (
            np.ascontiguousarray(active_tracks, dtype=np.float32),
            np.ascontiguousarray(active_visibility, dtype=bool),
        )

    indices = np.asarray(packet.query_indices, dtype=np.int64).reshape(-1)
    if indices.shape[0] != active_tracks.shape[0] or active_tracks.shape[0] != active_visibility.shape[0]:
        raise ValueError("sparse tracker packet must have query_indices, tracks_yx, and visibility with matching lengths")
    if np.any(indices < 0) or np.any(indices >= query_count):
        raise ValueError("tracker packet query_indices contains out-of-range values")
    tracks = np.zeros((query_count, 2), dtype=np.float32)
    visibility = np.zeros((query_count,), dtype=bool)
    tracks[indices] = active_tracks
    visibility[indices] = active_visibility
    return np.ascontiguousarray(tracks, dtype=np.float32), np.ascontiguousarray(visibility, dtype=bool)


@dataclass(frozen=True)
class PairedRenderPacket:
    seq: int
    pcd_packet: MaskedPcdPacket
    tracker_packet: TrackerMarkerPacket
    mask_packet: MaskPacket | None = None

    def __post_init__(self) -> None:
        """Validate and normalize the dataclass state after initialization."""
        pcd_seq = int(self.pcd_packet.seq)
        tracker_seq = int(self.tracker_packet.seq)
        mask_seq = None if self.mask_packet is None else int(self.mask_packet.seq)
        seq = int(self.seq)
        if pcd_seq != tracker_seq or seq != pcd_seq or (mask_seq is not None and mask_seq != seq):
            raise ValueError(
                "strict same-seq render packet mismatch: "
                f"pair={seq} pcd={pcd_seq} tracker={tracker_seq} mask={mask_seq}"
            )


@dataclass(frozen=True)
class PcdBuildResult:
    packet: MaskedPcdPacket
    depth_m: np.ndarray | None
    mask_packet: MaskPacket
    controller_pcd_mask: np.ndarray | None = None
    object_pcd_mask: np.ndarray | None = None
    object_observation_mask: np.ndarray | None = None
    pcd_stride: int = 1
    pcd_mask_erode_pixels: int = 0
    object_pcd_mask_erode_pixels: int = 0
    controller_pcd_mask_erode_pixels: int = 0
    world_z_diagnostics: dict[str, Any] = field(default_factory=dict)


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
    object_prefallback_points: int = 0
    object_raw_retain_ratio: float = 0.0
    object_fallback_reason: str = ""
    controller_raw_points: int = 0
    controller_cap_points: int = 0
    controller_output_points: int = 0
    controller_prefallback_points: int = 0
    controller_raw_retain_ratio: float = 0.0
    controller_fallback_reason: str = ""
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


# ---------------------------------------------------------------------------
# Headless capture: on-disk artifact writer (frames, pcds, tracker, metadata)
# ---------------------------------------------------------------------------
class HeadlessCaptureWriter:
    def __init__(self, output_dir: str | Path, *, metadata: dict[str, Any]) -> None:
        """Initialize HeadlessCaptureWriter."""
        self.output_dir = _resolve_path(output_dir)
        self.prepared_only = bool(metadata.get("headless_prepared_only", False))
        self.write_input_rgb_timeline = bool(metadata.get("write_input_rgb_timeline", False))
        self.saved_pcd_source = str(metadata.get("saved_pcd_source") or HEADLESS_CAPTURE_SAVED_PCD_SOURCE)
        self.pcd_coordinate_frame = str(
            metadata.get("pcd_coordinate_frame")
            or metadata.get("coordinate_frame")
            or COORDINATE_FRAME
        )
        self.pcd_dir = self.output_dir / "pcd"
        self.depth_dir = self.output_dir / "depth_color_m"
        self.rgb_dir = self.output_dir / "rgb"
        self.trajectory_dir = self.output_dir / "query_trajectory"
        self.mask_dir = self.output_dir / "masks"
        self.shape_prior_dir = self.output_dir / "shape_prior"
        self.prepared_phystwin_dir = self.output_dir / "prepared_phystwin"
        self.input_rgb_dir = self.output_dir / "input_rgb"
        self.frames_path = self.output_dir / "frames.jsonl"
        self.input_frames_path = self.output_dir / "input_frames.jsonl"
        self.world_z_stats_path = self.output_dir / "world_z_stats.jsonl"
        self.metadata_path = self.output_dir / "metadata.json"
        self._lock = threading.Lock()
        self._saved_pcd_count = 0
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pcd_dir.mkdir(parents=True, exist_ok=True)
        self.depth_dir.mkdir(parents=True, exist_ok=True)
        self.rgb_dir.mkdir(parents=True, exist_ok=True)
        self.trajectory_dir.mkdir(parents=True, exist_ok=True)
        self.mask_dir.mkdir(parents=True, exist_ok=True)
        self.prepared_phystwin_dir.mkdir(parents=True, exist_ok=True)
        self.input_rgb_dir.mkdir(parents=True, exist_ok=True)
        self.frames_path.write_text("", encoding="utf-8")
        self.input_frames_path.write_text("", encoding="utf-8")
        self.world_z_stats_path.write_text("", encoding="utf-8")
        payload = dict(metadata)
        payload["headless_capture_enabled"] = True
        payload["headless_prepared_only"] = bool(self.prepared_only)
        payload["write_input_rgb_timeline"] = bool(self.write_input_rgb_timeline)
        payload["saved_pcd_source"] = self.saved_pcd_source
        payload["saved_mask_source"] = "edgetam_binary_masks"
        payload["saved_rgb_source"] = "segmentation_color_bgr"
        payload["input_rgb_timeline"] = "input_frames.jsonl"
        payload["startup_hold_s"] = float(payload.get("startup_hold_s") or 0.0)
        payload["output_dir"] = _repo_relative_path_text(self.output_dir)
        self._metadata_payload = payload
        self._write_metadata_payload(payload)

    def _relative(self, path: Path) -> str:
        """Return the relative."""
        try:
            return str(path.relative_to(self.output_dir))
        except ValueError:
            return str(path)

    def _write_metadata_payload(self, payload: dict[str, Any]) -> None:
        """Write metadata payload."""
        tmp_path = self.metadata_path.with_name(f"{self.metadata_path.name}.tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp_path.replace(self.metadata_path)

    def update_metadata(self, values: dict[str, Any]) -> None:
        """Update metadata."""
        with self._lock:
            payload = dict(self._metadata_payload)
            payload.update(values)
            self._metadata_payload = payload
            self._write_metadata_payload(payload)

    def write_shape_prior_result(self, result: shape_prior_warmup.ShapePriorResult) -> None:
        """Write shape prior result."""
        self.shape_prior_dir.mkdir(parents=True, exist_ok=True)
        path = self.shape_prior_dir / "points.npz"
        np.savez_compressed(
            path,
            seq=np.asarray([int(result.seq)], dtype=np.int64),
            source_seq=np.asarray([-1 if result.source_seq is None else int(result.source_seq)], dtype=np.int64),
            source_timestamp_s=np.asarray(
                [np.nan if result.source_timestamp_s is None else float(result.source_timestamp_s)],
                dtype=np.float64,
            ),
            points_m=np.ascontiguousarray(result.points_m, dtype=np.float32).reshape(-1, 3),
            colors_rgb_u8=np.ascontiguousarray(result.colors_rgb_u8, dtype=np.uint8).reshape(-1, 3),
            surface_points_m=np.ascontiguousarray(result.surface_points_m, dtype=np.float32).reshape(-1, 3),
            interior_points_m=np.ascontiguousarray(result.interior_points_m, dtype=np.float32).reshape(-1, 3),
            metadata_json=np.asarray([json.dumps(dict(result.metadata), sort_keys=True)]),
        )
        values = dict(result.metadata)
        values.update(
            {
                "shape_prior_status": str(result.status),
                "shape_prior_source_seq": result.source_seq,
                "shape_prior_source_time_s": result.source_timestamp_s,
                "shape_prior_ready_seq": int(result.seq),
                "shape_prior_path": self._relative(path),
                "shape_prior_point_count": int(np.asarray(result.points_m).reshape(-1, 3).shape[0]),
                "shape_prior_surface_point_count": int(np.asarray(result.surface_points_m).reshape(-1, 3).shape[0]),
                "shape_prior_interior_point_count": int(np.asarray(result.interior_points_m).reshape(-1, 3).shape[0]),
            }
        )
        self.update_metadata(values)

    def write_input_frame(self, packet: FramePacket) -> None:
        """Write input frame."""
        seq_name = f"{int(packet.seq):06d}"
        rgb_path = self.input_rgb_dir / f"{seq_name}.png"
        row = {
            "seq": int(packet.seq),
            "source_timestamp_s": (
                None if packet.source_timestamp_s is None else float(packet.source_timestamp_s)
            ),
            "source_frame_index": (
                None if packet.source_frame_index is None else int(packet.source_frame_index)
            ),
            "source_step": None if packet.source_step is None else int(packet.source_step),
            "receive_perf_s": float(packet.receive_perf_s),
        }
        if self.write_input_rgb_timeline or not self.prepared_only:
            main_warmup.bgr_to_pil_rgb(packet.color_bgr).save(rgb_path)
            row["input_rgb_path"] = self._relative(rgb_path)
        with self._lock:
            with self.input_frames_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    def write_pcd(
        self,
        packet: MaskedPcdPacket,
        *,
        depth_m: np.ndarray,
        mask_packet: MaskPacket,
        controller_pcd_mask: np.ndarray,
        object_pcd_mask: np.ndarray,
        pcd_stride: int,
        pcd_mask_erode_pixels: int,
        object_pcd_mask_erode_pixels: int,
        controller_pcd_mask_erode_pixels: int,
        tracker_packet: TrackerMarkerPacket | None = None,
        stage_fps: dict[str, float] | None = None,
        world_z_diagnostics: dict[str, Any] | None = None,
        startup_hold_s: float = 0.0,
    ) -> None:
        """Write RGB-D, PCD, masks, tracking, and prepared PhysTwin artifacts."""
        filter_info = packet.filter_telemetry
        if not (filter_info.enabled and filter_info.mode == "sync" and filter_info.render_using_filtered):
            raise RuntimeError("headless capture refuses to save non-filtered PCD output")
        if self.prepared_only and tracker_packet is None:
            raise RuntimeError("prepared-only headless capture requires a tracker packet")
        fps_info = stage_fps or {}
        seq_name = f"{int(packet.seq):06d}"
        pcd_path = self.pcd_dir / f"{seq_name}.npz"
        depth_path = self.depth_dir / f"{seq_name}.npy"
        rgb_path = self.rgb_dir / f"{seq_name}.png"
        query_path = self.trajectory_dir / f"{seq_name}.npz"
        mask_path = self.mask_dir / f"{seq_name}.npz"
        prepared_phystwin_path = self.prepared_phystwin_dir / f"{seq_name}.npz"
        if not self.prepared_only:
            main_warmup.bgr_to_pil_rgb(mask_packet.color_bgr).save(rgb_path)
            np.save(
                depth_path,
                np.ascontiguousarray(depth_m, dtype=np.float32),
            )
            np.savez_compressed(
                mask_path,
                seq=np.asarray([int(packet.seq)], dtype=np.int64),
                controller_mask=np.ascontiguousarray(mask_packet.controller_mask, dtype=bool),
                object_mask=np.ascontiguousarray(mask_packet.object_mask, dtype=bool),
                hand_a_mask=np.ascontiguousarray(_mask_packet_hand_a_mask(mask_packet), dtype=bool),
                hand_b_mask=np.ascontiguousarray(_mask_packet_hand_b_mask(mask_packet), dtype=bool),
                controller_pcd_mask=np.ascontiguousarray(controller_pcd_mask, dtype=bool),
                object_pcd_mask=np.ascontiguousarray(object_pcd_mask, dtype=bool),
                pcd_stride=np.asarray([int(pcd_stride)], dtype=np.int64),
                pcd_mask_erode_pixels=np.asarray([int(pcd_mask_erode_pixels)], dtype=np.int64),
                object_pcd_mask_erode_pixels=np.asarray([int(object_pcd_mask_erode_pixels)], dtype=np.int64),
                controller_pcd_mask_erode_pixels=np.asarray([int(controller_pcd_mask_erode_pixels)], dtype=np.int64),
                mask_source=np.asarray(["edgetam_binary_masks"]),
            )
            np.savez(
                pcd_path,
                seq=np.asarray([int(packet.seq)], dtype=np.int64),
                controller_xyz_m=np.ascontiguousarray(packet.controller_xyz_m, dtype=np.float32),
                controller_rgb_u8=np.ascontiguousarray(packet.controller_colors_rgb_u8, dtype=np.uint8),
                object_xyz_m=np.ascontiguousarray(packet.object_xyz_m, dtype=np.float32),
                object_rgb_u8=np.ascontiguousarray(packet.object_colors_rgb_u8, dtype=np.uint8),
                intrinsics=np.asarray(
                    [
                        float(packet.intrinsics.fx),
                        float(packet.intrinsics.fy),
                        float(packet.intrinsics.cx),
                        float(packet.intrinsics.cy),
                    ],
                    dtype=np.float32,
                ),
                saved_pcd_source=np.asarray([self.saved_pcd_source]),
                coordinate_frame=np.asarray([str(packet.coordinate_frame or self.pcd_coordinate_frame)]),
            )
        prepared_phystwin_frame_path: str | None = None
        if tracker_packet is not None:
            c2w = np.asarray(self._metadata_payload.get("camera_to_world_c2w", np.eye(4)), dtype=np.float32).reshape(4, 4)
            full_tracks_yx, full_visibility = _full_tracker_arrays_for_prepared_frame(tracker_packet)
            mask_frame = {
                "object": np.asarray(mask_packet.object_mask, dtype=bool),
                "controller": np.asarray(mask_packet.controller_mask, dtype=bool),
                "hand_a": np.asarray(_mask_packet_hand_a_mask(mask_packet), dtype=bool),
                "hand_b": np.asarray(_mask_packet_hand_b_mask(mask_packet), dtype=bool),
            }
            prepared = prepare_phystwin_frame(
                seq=int(packet.seq),
                rgb_frame=np.ascontiguousarray(mask_packet.color_bgr[:, :, ::-1], dtype=np.uint8),
                depth_m=np.asarray(depth_m, dtype=np.float32),
                mask_frame=mask_frame,
                tracks_yx=full_tracks_yx,
                visibility=full_visibility,
                query_points_yx=np.asarray(tracker_packet.query_points_yx, dtype=np.float32),
                intrinsics=packet.intrinsics,
                c2w=c2w,
                mask_radius_outlier_filter=bool(self._metadata_payload.get("mask_radius_outlier_filter", True)),
                mask_radius_outlier_radius_m=float(self._metadata_payload.get("mask_radius_outlier_radius_m", 0.01)),
                mask_radius_outlier_nb_points=int(self._metadata_payload.get("mask_radius_outlier_nb_points", 40)),
                source_timestamp_s=packet.source_timestamp_s,
                source_frame_index=packet.source_frame_index,
                source_step=packet.source_step,
            )
            write_prepared_phystwin_frame(prepared_phystwin_path, prepared)
            prepared_phystwin_frame_path = self._relative(prepared_phystwin_path)
        pair_process_done_s = (
            max(float(packet.process_done_perf_s), float(tracker_packet.process_done_perf_s))
            if tracker_packet is not None
            else float(packet.process_done_perf_s)
        )
        row = {
            "seq": int(packet.seq),
            "source_timestamp_s": (
                None if packet.source_timestamp_s is None else float(packet.source_timestamp_s)
            ),
            "source_frame_index": (
                None if packet.source_frame_index is None else int(packet.source_frame_index)
            ),
            "source_step": None if packet.source_step is None else int(packet.source_step),
            "startup_hold_s": float(startup_hold_s),
            "pipeline_latency_ms": float(pair_process_done_s - float(packet.receive_perf_s)) * 1000.0,
            "capture_fps": float(fps_info.get("capture_fps", 0.0)),
            "seg_fps": float(fps_info.get("seg_fps", 0.0)),
            "depth_fps": float(fps_info.get("depth_fps", 0.0)),
            "pcd_fps": float(fps_info.get("pcd_fps", 0.0)),
            "tracker_fps": float(fps_info.get("tracker_fps", 0.0)),
            "filter_preset": self.saved_pcd_source,
            "marker_count": int(tracker_packet.marker_count) if tracker_packet is not None else 0,
            "marker_residual_checked_count": (
                int(tracker_packet.marker_residual_checked_count) if tracker_packet is not None else 0
            ),
            "marker_residual_violation_count": (
                int(tracker_packet.marker_residual_violation_count) if tracker_packet is not None else 0
            ),
            "marker_residual_gate": (
                str(tracker_packet.marker_residual_gate) if tracker_packet is not None else "none"
            ),
            "remaining_query_count": int(tracker_packet.remaining_query_count) if tracker_packet is not None else 0,
            "remaining_object_query_count": (
                int(tracker_packet.remaining_object_query_count) if tracker_packet is not None else 0
            ),
            "remaining_controller_query_count": (
                int(tracker_packet.remaining_controller_query_count) if tracker_packet is not None else 0
            ),
            "remaining_hand_a_query_count": (
                int(tracker_packet.remaining_hand_a_query_count) if tracker_packet is not None else 0
            ),
            "remaining_hand_b_query_count": (
                int(tracker_packet.remaining_hand_b_query_count) if tracker_packet is not None else 0
            ),
            "retired_query_count": int(tracker_packet.retired_query_count) if tracker_packet is not None else 0,
            "controller_point_count": int(packet.controller_point_count),
            "object_point_count": int(packet.object_point_count),
            "controller_mask_pixels": int(np.count_nonzero(mask_packet.controller_mask)),
            "object_mask_pixels": int(np.count_nonzero(mask_packet.object_mask)),
            "hand_a_mask_pixels": int(np.count_nonzero(_mask_packet_hand_a_mask(mask_packet))),
            "hand_b_mask_pixels": int(np.count_nonzero(_mask_packet_hand_b_mask(mask_packet))),
            "controller_pcd_mask_pixels": int(np.count_nonzero(controller_pcd_mask)),
            "object_pcd_mask_pixels": int(np.count_nonzero(object_pcd_mask)),
            "pcd_mask_erode_pixels": int(pcd_mask_erode_pixels),
            "controller_pcd_mask_erode_pixels": int(controller_pcd_mask_erode_pixels),
            "object_pcd_mask_erode_pixels": int(object_pcd_mask_erode_pixels),
            "hand_a_query_count": int(tracker_packet.hand_a_query_count) if tracker_packet is not None else 0,
            "hand_b_query_count": int(tracker_packet.hand_b_query_count) if tracker_packet is not None else 0,
            "object_query_count": int(tracker_packet.object_query_count) if tracker_packet is not None else 0,
            "query_count": int(tracker_packet.query_count) if tracker_packet is not None else 0,
            "receive_perf_s": float(packet.receive_perf_s),
            "process_done_perf_s": float(packet.process_done_perf_s),
            "pair_process_done_perf_s": float(pair_process_done_s),
            "timing": asdict(packet.timing),
            "filter_telemetry": asdict(packet.filter_telemetry),
        }
        if not self.prepared_only:
            row.update(
                {
                    "pcd_path": self._relative(pcd_path),
                    "depth_color_m_path": self._relative(depth_path),
                    "rgb_path": self._relative(rgb_path),
                    "query_trajectory_path": self._relative(query_path),
                    "mask_path": self._relative(mask_path),
                    "world_z_stats_path": self._relative(self.world_z_stats_path),
                }
            )
        if prepared_phystwin_frame_path is not None:
            row["prepared_phystwin_frame_path"] = prepared_phystwin_frame_path
        line = json.dumps(row, sort_keys=True)
        with self._lock:
            with self.frames_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
            if world_z_diagnostics is not None and not self.prepared_only:
                z_payload = dict(world_z_diagnostics)
                z_payload.setdefault("seq", int(packet.seq))
                with self.world_z_stats_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(z_payload, sort_keys=True) + "\n")
            self._saved_pcd_count += 1

    def write_tracker(self, packet: TrackerMarkerPacket) -> None:
        """Write tracker."""
        if self.prepared_only:
            return
        seq_name = f"{int(packet.seq):06d}"
        path = self.trajectory_dir / f"{seq_name}.npz"
        np.savez(
            path,
            seq=np.asarray([int(packet.seq)], dtype=np.int64),
            query_points_yx=np.ascontiguousarray(packet.query_points_yx, dtype=np.float32),
            query_indices=np.ascontiguousarray(packet.query_indices, dtype=np.int64),
            query_rgb_u8=np.ascontiguousarray(packet.query_rgb_u8, dtype=np.uint8),
            marker_xyz_m=np.ascontiguousarray(packet.marker_xyz_m, dtype=np.float32),
            marker_rgb_u8=np.ascontiguousarray(packet.marker_colors_rgb_u8, dtype=np.uint8),
            tracks_yx=np.ascontiguousarray(packet.tracks_yx, dtype=np.float32),
            visibility=np.ascontiguousarray(packet.visibility, dtype=np.float32),
            query_is_object=np.ascontiguousarray(packet.query_is_object, dtype=bool),
            query_is_controller=np.ascontiguousarray(packet.query_is_controller, dtype=bool),
            query_target_id=np.ascontiguousarray(packet.query_target_id, dtype=np.int64),
            query_controller_instance_id=np.ascontiguousarray(packet.query_controller_instance_id, dtype=np.int64),
            query_all_target_id=np.ascontiguousarray(packet.query_all_target_id, dtype=np.int64),
            query_all_controller_instance_id=np.ascontiguousarray(
                packet.query_all_controller_instance_id,
                dtype=np.int64,
            ),
            marker_pixels_yx=np.ascontiguousarray(packet.marker_pixels_yx, dtype=np.int64).reshape(-1, 2),
            marker_residual_valid=np.ascontiguousarray(packet.marker_residual_valid, dtype=bool),
            marker_residual_violation=np.ascontiguousarray(packet.marker_residual_violation, dtype=bool),
            marker_residual_checked_count=np.asarray([int(packet.marker_residual_checked_count)], dtype=np.int64),
            marker_residual_violation_count=np.asarray([int(packet.marker_residual_violation_count)], dtype=np.int64),
            marker_residual_gate=np.asarray([str(packet.marker_residual_gate)]),
            query_alive_mask=np.ascontiguousarray(packet.query_alive_mask, dtype=bool),
            all_tracks_yx=np.ascontiguousarray(packet.all_tracks_yx, dtype=np.float32).reshape(-1, 2),
            all_tracker_visibility=np.ascontiguousarray(packet.all_tracker_visibility, dtype=np.float32).reshape(-1),
            remaining_query_count=np.asarray([int(packet.remaining_query_count)], dtype=np.int64),
            remaining_object_query_count=np.asarray([int(packet.remaining_object_query_count)], dtype=np.int64),
            remaining_controller_query_count=np.asarray([int(packet.remaining_controller_query_count)], dtype=np.int64),
            remaining_hand_a_query_count=np.asarray([int(packet.remaining_hand_a_query_count)], dtype=np.int64),
            remaining_hand_b_query_count=np.asarray([int(packet.remaining_hand_b_query_count)], dtype=np.int64),
            retired_query_count=np.asarray([int(packet.retired_query_count)], dtype=np.int64),
            query_count=np.asarray([int(packet.query_count)], dtype=np.int64),
            consistent_visible_count=np.asarray([int(packet.consistent_visible_count)], dtype=np.int64),
            hand_a_query_count=np.asarray([int(packet.hand_a_query_count)], dtype=np.int64),
            hand_b_query_count=np.asarray([int(packet.hand_b_query_count)], dtype=np.int64),
            object_query_count=np.asarray([int(packet.object_query_count)], dtype=np.int64),
            model_ms=np.asarray([float(packet.model_ms)], dtype=np.float32),
            lift_ms=np.asarray([float(packet.lift_ms)], dtype=np.float32),
            e2e_ms=np.asarray([float(packet.e2e_ms)], dtype=np.float32),
            coordinate_frame=np.asarray([str(packet.coordinate_frame or self.pcd_coordinate_frame)]),
        )

    @property
    def saved_pcd_count(self) -> int:
        """Return the saved PCD count."""
        with self._lock:
            return int(self._saved_pcd_count)


# ---------------------------------------------------------------------------
# Lossless pipeline plumbing: stage FPS, ordered queues, same-seq pairing
# ---------------------------------------------------------------------------
class StageStats:
    def __init__(self, window_s: float = 1.0) -> None:
        """Initialize StageStats."""
        self.window_s = float(window_s)
        self._lock = threading.Lock()
        self._times: deque[float] = deque()

    def record(self, now_s: float | None = None) -> None:
        """Record StageStats."""
        now = time.perf_counter() if now_s is None else float(now_s)
        with self._lock:
            self._times.append(now)
            cutoff = now - self.window_s
            while len(self._times) > 1 and self._times[0] < cutoff:
                self._times.popleft()

    @property
    def fps(self) -> float:
        """Return the FPS."""
        with self._lock:
            if len(self._times) < 2:
                return 0.0
            elapsed = self._times[-1] - self._times[0]
            if elapsed <= 0:
                return 0.0
            return float((len(self._times) - 1) / elapsed)


PacketT = TypeVar("PacketT")


class LosslessPipelineError(RuntimeError):
    """Fatal contract violation in the lossless Demo 3.x pipeline."""


@dataclass(frozen=True)
class OrderedQueueStats:
    name: str
    size: int
    max_size: int
    last_put_seq: int
    last_get_seq: int
    closed: bool


class OrderedPacketQueue(Generic[PacketT]):
    """Bounded FIFO packet queue that rejects gaps and silent overwrites."""

    def __init__(self, *, name: str, max_backlog_frames: int) -> None:
        """Initialize OrderedPacketQueue."""
        self.name = str(name)
        self.max_backlog_frames = max(1, int(max_backlog_frames))
        self._condition = threading.Condition()
        self._items: deque[PacketT] = deque()
        self._last_put_seq = -1
        self._last_get_seq = -1
        self._closed = False
        self._max_size_seen = 0

    def put(self, packet: PacketT) -> int:
        """Return the put."""
        seq = int(_packet_seq(packet))
        with self._condition:
            if self._closed:
                raise LosslessPipelineError(f"{self.name} queue is closed")
            expected = self._last_put_seq + 1
            if seq != expected:
                raise LosslessPipelineError(
                    f"{self.name} queue expected seq {expected}, got {seq}"
                )
            if len(self._items) >= self.max_backlog_frames:
                raise LosslessPipelineError(
                    "lossless input FPS backlog exceeded "
                    f"stage={self.name} queue_len={len(self._items) + 1} "
                    f"max={self.max_backlog_frames} expected_seq={self._last_get_seq + 1} "
                    f"latest_seq={seq}"
                )
            self._items.append(packet)
            self._last_put_seq = seq
            self._max_size_seen = max(self._max_size_seen, len(self._items))
            self._condition.notify_all()
            return len(self._items)

    def wait_for_capacity(self, *, stop_event: threading.Event, timeout_s: float = 0.05) -> bool:
        """Wait for for capacity."""
        with self._condition:
            while not stop_event.is_set():
                if self._closed:
                    raise LosslessPipelineError(f"{self.name} queue is closed")
                if len(self._items) < self.max_backlog_frames:
                    return True
                self._condition.wait(timeout=float(timeout_s))
            return False

    def put_wait(self, packet: PacketT, *, stop_event: threading.Event, timeout_s: float = 0.05) -> int:
        """Return the put wait."""
        seq = int(_packet_seq(packet))
        with self._condition:
            if self._closed:
                raise LosslessPipelineError(f"{self.name} queue is closed")
            expected = self._last_put_seq + 1
            if seq != expected:
                raise LosslessPipelineError(
                    f"{self.name} queue expected seq {expected}, got {seq}"
                )
            while len(self._items) >= self.max_backlog_frames:
                if stop_event.is_set():
                    return 0
                if self._closed:
                    raise LosslessPipelineError(f"{self.name} queue is closed")
                self._condition.wait(timeout=float(timeout_s))
            self._items.append(packet)
            self._last_put_seq = seq
            self._max_size_seen = max(self._max_size_seen, len(self._items))
            self._condition.notify_all()
            return len(self._items)

    def get(self, *, stop_event: threading.Event, timeout_s: float = 0.05) -> PacketT | None:
        """Return the get."""
        with self._condition:
            while not self._items:
                if self._closed or stop_event.is_set():
                    return None
                self._condition.wait(timeout=float(timeout_s))
            packet = self._items.popleft()
            seq = int(_packet_seq(packet))
            expected = self._last_get_seq + 1
            if seq != expected:
                raise LosslessPipelineError(
                    f"{self.name} queue consumer expected seq {expected}, got {seq}"
                )
            self._last_get_seq = seq
            self._condition.notify_all()
            return packet

    def get_nowait(self) -> PacketT | None:
        """Return the get nowait."""
        with self._condition:
            if not self._items:
                return None
            packet = self._items.popleft()
            seq = int(_packet_seq(packet))
            expected = self._last_get_seq + 1
            if seq != expected:
                raise LosslessPipelineError(
                    f"{self.name} queue consumer expected seq {expected}, got {seq}"
                )
            self._last_get_seq = seq
            self._condition.notify_all()
            return packet

    def close(self) -> None:
        """Close OrderedPacketQueue."""
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def reset(self) -> None:
        """Reset OrderedPacketQueue."""
        with self._condition:
            self._items.clear()
            self._last_put_seq = -1
            self._last_get_seq = -1
            self._closed = False
            self._max_size_seen = 0
            self._condition.notify_all()

    @property
    def stats(self) -> OrderedQueueStats:
        """Return the stats."""
        with self._condition:
            return OrderedQueueStats(
                name=self.name,
                size=len(self._items),
                max_size=int(self._max_size_seen),
                last_put_seq=int(self._last_put_seq),
                last_get_seq=int(self._last_get_seq),
                closed=bool(self._closed),
            )

    def latest_seq(self) -> int:
        """Return the latest seq."""
        with self._condition:
            return int(self._last_put_seq)

    def pending_count(self) -> int:
        """Return the pending count."""
        with self._condition:
            return len(self._items)

    def is_closed_and_empty(self) -> bool:
        """Return whether closed and empty."""
        with self._condition:
            return bool(self._closed and not self._items)


@dataclass(frozen=True)
class PairedBuildResult:
    seq: int
    pcd_result: PcdBuildResult
    tracker_packet: TrackerMarkerPacket

    @property
    def render_packet(self) -> PairedRenderPacket:
        """Return the paired packet used by the renderer."""
        return PairedRenderPacket(
            seq=int(self.seq),
            pcd_packet=self.pcd_result.packet,
            tracker_packet=self.tracker_packet,
            mask_packet=self.pcd_result.mask_packet,
        )


@dataclass(frozen=True)
class PairerStats:
    expected_seq: int
    pending_pcd: int
    pending_tracker: int
    emitted_seq: int
    pcd_closed: bool
    tracker_closed: bool


class SameSeqPairer:
    def __init__(self, *, max_backlog_frames: int) -> None:
        """Initialize SameSeqPairer."""
        self.max_backlog_frames = max(1, int(max_backlog_frames))
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._pending_pcd: dict[int, PcdBuildResult] = {}
        self._pending_tracker: dict[int, TrackerMarkerPacket] = {}
        self._expected_seq = 0
        self._emitted_seq = -1
        self._pcd_closed = False
        self._tracker_closed = False

    def reset(self) -> None:
        """Reset SameSeqPairer."""
        with self._condition:
            self._pending_pcd.clear()
            self._pending_tracker.clear()
            self._expected_seq = 0
            self._emitted_seq = -1
            self._pcd_closed = False
            self._tracker_closed = False
            self._condition.notify_all()

    def wait_for_side_capacity(
        self,
        side: str,
        *,
        stop_event: threading.Event,
        timeout_s: float = 0.05,
    ) -> bool:
        """Wait for for side capacity."""
        side_name = str(side)
        if side_name not in {"pcd", "tracker"}:
            raise ValueError("side must be 'pcd' or 'tracker'")
        with self._condition:
            while not stop_event.is_set():
                if side_name == "pcd":
                    if self._pcd_closed:
                        raise LosslessPipelineError("same-seq pairer PCD side is closed")
                    pending = len(self._pending_pcd)
                else:
                    if self._tracker_closed:
                        raise LosslessPipelineError("same-seq pairer tracker side is closed")
                    pending = len(self._pending_tracker)
                if pending < self.max_backlog_frames:
                    return True
                self._condition.wait(timeout=float(timeout_s))
            return False

    def add_pcd_result(self, result: PcdBuildResult) -> list[PairedBuildResult]:
        """Add PCD result."""
        seq = int(result.packet.seq)
        with self._condition:
            if self._pcd_closed:
                raise LosslessPipelineError("same-seq pairer PCD side is closed")
            if seq < self._expected_seq:
                raise LosslessPipelineError(
                    f"same-seq pairer received stale PCD seq {seq}, expected {self._expected_seq}"
                )
            if seq in self._pending_pcd:
                raise LosslessPipelineError(f"same-seq pairer duplicate PCD seq {seq}")
            self._pending_pcd[seq] = result
            self._check_backlog_locked()
            pairs = self._flush_ready_locked()
            self._condition.notify_all()
            return pairs

    def add_tracker_packet(self, packet: TrackerMarkerPacket) -> list[PairedBuildResult]:
        """Add tracker packet."""
        seq = int(packet.seq)
        with self._condition:
            if self._tracker_closed:
                raise LosslessPipelineError("same-seq pairer tracker side is closed")
            if seq < self._expected_seq:
                raise LosslessPipelineError(
                    f"same-seq pairer received stale tracker seq {seq}, expected {self._expected_seq}"
                )
            if seq in self._pending_tracker:
                raise LosslessPipelineError(f"same-seq pairer duplicate tracker seq {seq}")
            self._pending_tracker[seq] = packet
            self._check_backlog_locked()
            pairs = self._flush_ready_locked()
            self._condition.notify_all()
            return pairs

    def close_pcd(self) -> list[PairedBuildResult]:
        """Close PCD."""
        with self._condition:
            self._pcd_closed = True
            pairs = self._flush_ready_locked()
            self._check_closed_locked()
            self._condition.notify_all()
            return pairs

    def close_tracker(self) -> list[PairedBuildResult]:
        """Close tracker."""
        with self._condition:
            self._tracker_closed = True
            pairs = self._flush_ready_locked()
            self._check_closed_locked()
            self._condition.notify_all()
            return pairs

    @property
    def done(self) -> bool:
        """Return the done."""
        with self._condition:
            return (
                self._pcd_closed
                and self._tracker_closed
                and not self._pending_pcd
                and not self._pending_tracker
            )

    @property
    def stats(self) -> PairerStats:
        """Return the stats."""
        with self._condition:
            return PairerStats(
                expected_seq=int(self._expected_seq),
                pending_pcd=len(self._pending_pcd),
                pending_tracker=len(self._pending_tracker),
                emitted_seq=int(self._emitted_seq),
                pcd_closed=bool(self._pcd_closed),
                tracker_closed=bool(self._tracker_closed),
            )

    def _flush_ready_locked(self) -> list[PairedBuildResult]:
        """Return the flush ready locked."""
        pairs: list[PairedBuildResult] = []
        while self._expected_seq in self._pending_pcd and self._expected_seq in self._pending_tracker:
            seq = int(self._expected_seq)
            pcd_result = self._pending_pcd.pop(seq)
            tracker_packet = self._pending_tracker.pop(seq)
            pairs.append(PairedBuildResult(seq=seq, pcd_result=pcd_result, tracker_packet=tracker_packet))
            self._emitted_seq = seq
            self._expected_seq += 1
        return pairs

    def _check_backlog_locked(self) -> None:
        """Check backlog locked."""
        if len(self._pending_pcd) > self.max_backlog_frames or len(self._pending_tracker) > self.max_backlog_frames:
            raise LosslessPipelineError(
                "lossless input FPS backlog exceeded "
                f"stage=pairer expected_seq={self._expected_seq} "
                f"pending_pcd={len(self._pending_pcd)} pending_tracker={len(self._pending_tracker)} "
                f"max={self.max_backlog_frames}"
            )

    def _check_closed_locked(self) -> None:
        """Check closed locked."""
        if not (self._pcd_closed and self._tracker_closed):
            return
        if self._pending_pcd or self._pending_tracker:
            raise LosslessPipelineError(
                "same-seq pairer closed with unmatched packets "
                f"expected_seq={self._expected_seq} "
                f"pending_pcd={sorted(self._pending_pcd)} "
                f"pending_tracker={sorted(self._pending_tracker)}"
            )


# ---------------------------------------------------------------------------
# CLI: argument parsing, demo presets, derived-mode accessors, validation
# ---------------------------------------------------------------------------
def _resolve_path(value: str | Path) -> Path:
    """Resolve a filesystem path to an absolute expanded path."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _parse_rgb_triplet(value: str) -> tuple[int, int, int]:
    """Parse RGB triplet."""
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


def _is_replay_input_source(input_source: str) -> bool:
    """Return whether replay input source."""
    return str(input_source) in {INPUT_SOURCE_FAKE_LIVE, INPUT_SOURCE_RECORDING}


def depth_backend_label(args: argparse.Namespace) -> str:
    """Return the depth backend label."""
    label = getattr(args, "depth_backend_label", None)
    if label is not None and str(label):
        return str(label)
    return str(args.depth_source)


def runtime_metadata_identity(args: argparse.Namespace) -> dict[str, str]:
    """Return the runtime metadata identity."""
    payload: dict[str, str] = {}
    product_name = getattr(args, "runtime_product_name", None)
    if product_name is not None and str(product_name).strip():
        payload["runtime_product_name"] = str(product_name).strip()
    demo_version = getattr(args, "metadata_demo_version", None)
    if demo_version is not None and str(demo_version).strip():
        payload["demo_version"] = str(demo_version).strip()
    reference_pipeline = getattr(args, "metadata_reference_pipeline", None)
    if reference_pipeline is not None and str(reference_pipeline).strip():
        payload["reference_pipeline"] = str(reference_pipeline).strip()
    return payload


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Single-D455 realtime HF EdgeTAM masked point-cloud demo. Captures live "
            "RealSense color plus FFS stereo depth by default, tracks controller/object "
            "or object-only with one HF EdgeTAM streaming session, and writes headless "
            "masked-PCD/tracking capture products."
        )
    )
    parser.add_argument("--serial", default=None, help="Optional RealSense D400 serial. Defaults to first detected D400.")
    parser.add_argument("--profile", choices=SUPPORTED_PROFILES, default=DEFAULT_PROFILE, help="Capture profile.")
    parser.add_argument("--fps", choices=SUPPORTED_CAPTURE_FPS, type=int, default=DEFAULT_FPS, help="Capture FPS.")
    parser.add_argument(
        "--input-source",
        choices=INPUT_SOURCES,
        default=INPUT_SOURCE_LIVE,
        help=(
            "Frame source. fake-live replays a raw single-camera data_collect case at camera cadence, "
            "dropping source frames to preserve recording time when replay FPS is lower; recording is kept "
            "as a compatibility alias."
        ),
    )
    parser.add_argument(
        "--recording-case",
        type=Path,
        default=None,
        help="Raw data_collect case folder for --input-source recording or fake-live.",
    )
    parser.add_argument(
        "--fake-live-case",
        dest="recording_case",
        type=Path,
        default=None,
        help=f"Alias for --recording-case. fake-live defaults to {DEFAULT_FAKE_LIVE_CASE}.",
    )
    parser.add_argument(
        "--replay-fps",
        type=float,
        default=0.0,
        help=(
            "Replay FPS for --input-source recording or fake-live. For fake-live this is the emitted "
            "sample cadence; lower values drop source frames rather than slow motion. Use 0 to read metadata fps."
        ),
    )
    parser.add_argument(
        "--lossless-max-backlog-seconds",
        type=float,
        default=DEFAULT_LOSSLESS_MAX_BACKLOG_SECONDS,
        help=(
            "Maximum strict lossless input-FPS backlog window before treating "
            "the run as stalled."
        ),
    )
    parser.add_argument(
        "--lossless-input-fps",
        type=float,
        default=DEFAULT_LOSSLESS_INPUT_FPS,
        help="Strict lossless camera/fake-live cadence used by tracker-synchronized masked PCD replay.",
    )
    parser.add_argument(
        "--table-calibrate",
        type=Path,
        default=None,
        help=(
            "Optional single-camera table Z=0 calibration pickle. When provided, Demo 3.x PCD "
            "and 3D tracker markers are transformed from camera_color_frame into table_world_z0."
        ),
    )
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
        "--depth-backend-label",
        default=None,
        help=argparse.SUPPRESS,
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
        "--color-exposure",
        type=float,
        default=None,
        help="Optional manual RealSense RGB exposure. When set, RGB auto exposure is disabled.",
    )
    parser.add_argument(
        "--color-gain",
        type=float,
        default=None,
        help="Optional manual RealSense RGB gain. When set, RGB auto exposure is disabled.",
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
        "--demo-visual-mode",
        choices=DEMO_VISUAL_MODES,
        default=DEFAULT_DEMO_VISUAL_MODE,
        help="Visual presentation hint forwarded from single-camera wrappers.",
    )
    parser.add_argument("--runtime-product-name", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--metadata-demo-version", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--metadata-reference-pipeline", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--tracker-backend",
        choices=TRACKER_BACKENDS,
        default=DEFAULT_TRACKER_BACKEND,
        help="Optional point-tracker overlay backend. tapnextpp adds 3D query/track markers.",
    )
    parser.add_argument(
        "--tracker-device",
        default="cuda:1",
        help="Device for the point-tracker backend. Use cuda:1 on the dual-4090 demo machine.",
    )
    parser.add_argument(
        "--tracker-query-count",
        type=int,
        default=DEFAULT_TRACKER_QUERY_COUNT,
        help="TAPNext++ query points sampled from object/controller union mask. Use 0 for PhysTwin dense auto.",
    )
    parser.add_argument("--tracker-seed", type=int, default=DEFAULT_TRACKER_SEED)
    parser.add_argument(
        "--tracker-display-scope",
        choices=TRACKER_DISPLAY_SCOPES,
        default=DEFAULT_TRACKER_DISPLAY_SCOPE,
        help="Which query labels are rendered as 3D markers.",
    )
    parser.add_argument(
        "--tracker-overlay-max-points",
        type=int,
        default=512,
        help="Maximum visible tracker markers rendered per frame. 0 renders all visible selected points.",
    )
    parser.add_argument(
        "--tracker-marker-point-size",
        type=float,
        default=DEFAULT_TRACKER_MARKER_POINT_SIZE,
        help="TAPNext++ marker point size recorded in capture metadata.",
    )
    parser.add_argument(
        "--tracking-product-backend",
        choices=TRACKING_PRODUCT_BACKENDS,
        default=DEFAULT_TRACKING_PRODUCT_BACKEND,
        help=(
            "Final tracking product backend. realtime-overlay keeps the live marker product; "
            "phystwin-strict-tracking writes PhysTwin-compatible headless artifacts using TAPNext++ tracks."
        ),
    )
    parser.add_argument(
        "--phystwin-strict-output-dir",
        type=Path,
        default=None,
        help="Output directory for --tracking-product-backend phystwin-strict-tracking. Defaults to <headless-capture-dir>/phystwin_like.",
    )
    parser.add_argument(
        "--shape-prior-warmup",
        dest="shape_prior_warmup",
        action="store_true",
        help="Enable the optional SAM3D shape-prior warmup request path.",
    )
    parser.add_argument(
        "--no-shape-prior-warmup",
        dest="shape_prior_warmup",
        action="store_false",
        help="Disable the optional SAM3D shape-prior warmup request path.",
    )
    parser.set_defaults(shape_prior_warmup=False)
    parser.add_argument(
        "--shape-prior-prewarm-stage-workers",
        dest="shape_prior_prewarm_stage_workers",
        action="store_true",
        help=(
            "Spawn pre-warmed one-shot upscale/generate/align workers at boot "
            "so shape-prior model loading happens before frame 0."
        ),
    )
    parser.add_argument(
        "--no-shape-prior-prewarm-stage-workers",
        dest="shape_prior_prewarm_stage_workers",
        action="store_false",
        help="Load shape-prior stage models only when the frame-0 request runs.",
    )
    parser.set_defaults(shape_prior_prewarm_stage_workers=True)
    parser.add_argument(
        "--shape-prior-timeout-ms",
        type=int,
        default=shape_prior_warmup.DEFAULT_SHAPE_PRIOR_TIMEOUT_MS,
    )
    parser.add_argument("--shape-prior-profile-json", type=Path, default=None)
    parser.add_argument(
        "--shape-prior-case-root",
        type=Path,
        default=Path("outputs_v6_1") / "shape_prior_case",
    )
    parser.add_argument(
        "--shape-prior-points-npz",
        type=Path,
        default=shape_prior_warmup.POINTS_NPZ,
    )
    parser.add_argument(
        "--shape-prior-warmup-cuda-visible-devices",
        default=shape_prior_warmup.DEFAULT_SHAPE_PRIOR_WARMUP_CUDA_VISIBLE_DEVICES,
    )
    parser.add_argument(
        "--shape-prior-controller-name",
        default=None,
    )
    parser.add_argument("--shape-prior-sam3d-root", type=Path, default=None)
    parser.add_argument("--shape-prior-config", type=Path, default=None)
    parser.add_argument(
        "--shape-prior-skip-route-visualizations",
        dest="shape_prior_skip_route_visualizations",
        action="store_true",
    )
    parser.add_argument(
        "--shape-prior-render-route-visualizations",
        dest="shape_prior_skip_route_visualizations",
        action="store_false",
    )
    parser.set_defaults(shape_prior_skip_route_visualizations=True)
    parser.add_argument(
        "--tracker-retire-filtered-markers",
        dest="tracker_retire_filtered_markers",
        action="store_true",
        help="Opt in to permanently hiding any query marker after it fails the active PCD residual/table-Z gate.",
    )
    parser.add_argument(
        "--no-tracker-retire-filtered-markers",
        dest="tracker_retire_filtered_markers",
        action="store_false",
        help="Use the default per-frame marker gate; filtered markers may reappear later.",
    )
    parser.set_defaults(tracker_retire_filtered_markers=False)
    parser.add_argument(
        "--tapnet-repo-dir",
        type=Path,
        default=DEFAULT_TAPNET_REPO_DIR,
        help="External tapnet repo containing tapnet/tapnext/tapnext_torch.py.",
    )
    parser.add_argument(
        "--tapnextpp-checkpoint",
        type=Path,
        default=DEFAULT_TAPNEXTPP_CHECKPOINT,
        help="External TAPNext++ checkpoint.",
    )
    parser.add_argument("--tapnextpp-image-size", default="256,256")
    parser.add_argument("--tapnextpp-autocast-dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--tapnextpp-compile", action="store_true")
    parser.add_argument("--no-tapnextpp-fast-postprocess", dest="tapnextpp_fast_postprocess", action="store_false")
    parser.set_defaults(tapnextpp_fast_postprocess=True)
    parser.add_argument(
        "--demo-preset",
        choices=DEMO_PRESETS,
        default=DEFAULT_DEMO_PRESET,
        help=(
            "Optional demo preset. local-ffs-professor keeps FFS-derived depth "
            "and compiled EdgeTAM, while capping PCD/filter points for a steadier local demo."
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
        "--edgetam-live-session-keep-frames",
        type=int,
        default=DEFAULT_EDGETAM_LIVE_SESSION_KEEP_FRAMES,
        help="Keep this many recent streamed EdgeTAM frames/outputs in live session state; 0 disables pruning.",
    )
    parser.add_argument(
        "--edgetam-mask-logit-threshold",
        type=float,
        default=DEFAULT_EDGETAM_MASK_LOGIT_THRESHOLD,
        help=(
            "Logit threshold used to binarize EdgeTAM masks. "
            "Lower values make masks more permissive."
        ),
    )
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
        "--pcd-mask-erode-pixels",
        type=int,
        default=DEFAULT_PCD_MASK_ERODE_PIXELS,
        help=(
            "Legacy common erosion for controller/object masks before RGB-D point-cloud backprojection. "
            "Per-class erosion options override this value."
        ),
    )
    parser.add_argument(
        "--object-pcd-mask-erode-pixels",
        type=int,
        default=DEFAULT_OBJECT_PCD_MASK_ERODE_PIXELS,
        help="Object-only mask erosion before RGB-D point-cloud backprojection. Defaults to --pcd-mask-erode-pixels.",
    )
    parser.add_argument(
        "--controller-pcd-mask-erode-pixels",
        type=int,
        default=DEFAULT_CONTROLLER_PCD_MASK_ERODE_PIXELS,
        help="Controller-only mask erosion before RGB-D point-cloud backprojection. Defaults to --pcd-mask-erode-pixels.",
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
    parser.add_argument(
        "--pcd-filter-preset",
        choices=PCD_FILTER_PRESETS,
        default=None,
        help=(
            "High-level PCD surface preset. When set, the same preset controls object/controller PCD "
            "and TAPNext++ initial query sampling from filtered residual pixels."
        ),
    )
    parser.add_argument("--object-filter", choices=PCD_FILTERS, default=DEFAULT_OBJECT_FILTER)
    parser.add_argument("--controller-filter", choices=PCD_FILTERS, default=DEFAULT_CONTROLLER_FILTER)
    parser.add_argument("--object-filter-cap", type=int, default=DEFAULT_OBJECT_FILTER_CAP)
    parser.add_argument("--controller-filter-cap", type=int, default=DEFAULT_CONTROLLER_FILTER_CAP)
    parser.add_argument(
        "--object-filter-keep-components",
        type=int,
        default=DEFAULT_OBJECT_FILTER_KEEP_COMPONENTS,
        help="Connected components to keep when --object-filter enhanced-pt is used.",
    )
    parser.add_argument(
        "--controller-filter-keep-components",
        type=int,
        default=DEFAULT_CONTROLLER_FILTER_KEEP_COMPONENTS,
        help="Connected components to keep when --controller-filter enhanced-pt is used.",
    )
    parser.add_argument("--object-filter-voxel-m", type=float, default=0.004)
    parser.add_argument("--controller-filter-voxel-m", type=float, default=0.003)
    parser.add_argument(
        "--filter-every-n",
        type=int,
        default=3,
        help="Submit capped PCD filtering every N PCD packets. Async mode renders the latest available filtered output.",
    )
    parser.add_argument(
        "--filter-max-age-frames",
        type=int,
        default=DEFAULT_FILTER_MAX_AGE_FRAMES,
        help="Maximum async filtered-output age in frames before rendering raw current PCD instead.",
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
    parser.add_argument(
        "--enable-table-z-filter",
        action="store_true",
        help=(
            "Enable table-world Z filter. Removes target PCD points whose "
            "signed table clearance is <= threshold after PT filtering."
        ),
    )
    parser.add_argument(
        "--disable-table-z-filter",
        action="store_true",
        help="Disable the table-world Z filter when a demo visual preset would enable it by default.",
    )
    parser.add_argument(
        "--table-z-filter-threshold-m",
        type=float,
        default=DEFAULT_TABLE_Z_FILTER_THRESHOLD_M,
        help="World-Z clearance threshold above table_z for --enable-table-z-filter.",
    )
    parser.add_argument(
        "--table-z-filter-classes",
        choices=TABLE_Z_FILTER_CLASSES,
        default=TABLE_Z_FILTER_CLASS_BOTH,
        help="Semantic classes affected by --enable-table-z-filter.",
    )
    parser.add_argument("--duration-s", type=float, default=0.0, help="Optional auto-stop duration. Use 0 to run until closed.")
    parser.add_argument(
        "--headless-capture-dir",
        type=Path,
        default=None,
        help=(
            "Save the selected sync PCD preset, color-aligned depth, and TAPNext++ query "
            "trajectory artifacts here. "
            "With --table-calibrate, the default demo preset uses filter none plus the 0 mm table-Z filter."
        ),
    )
    parser.add_argument(
        "--headless-prepared-only",
        action="store_true",
        help="For strict PhysTwin chunk preprocessing, save prepared_phystwin frames and frames.jsonl without legacy per-frame artifacts.",
    )
    parser.add_argument(
        "--write-input-rgb-timeline",
        action="store_true",
        help=(
            "Write input_rgb/*.png and input_frames.jsonl for Demo v6.1 "
            "realtime side-by-side viewing."
        ),
    )
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
    """Apply demo preset."""
    if args.demo_preset == "local-ffs-professor":
        if int(args.pcd_max_points) == 60000:
            args.pcd_max_points = LOCAL_FFS_PROFESSOR_MAX_POINTS
        if int(args.object_filter_cap) == 20_000:
            args.object_filter_cap = LOCAL_FFS_PROFESSOR_FILTER_CAP
        if int(args.controller_filter_cap) == 20_000:
            args.controller_filter_cap = LOCAL_FFS_PROFESSOR_FILTER_CAP
    if (
        not bool(getattr(args, "disable_table_z_filter", False))
        and not bool(getattr(args, "enable_table_z_filter", False))
        and getattr(args, "table_calibrate", None) is not None
        and str(getattr(args, "demo_visual_mode", DEFAULT_DEMO_VISUAL_MODE)) in DEMO_VISUAL_MODES
        and headless_capture_enabled(args)
    ):
        args.enable_table_z_filter = True
    return args


def pcd_filter_enabled(args: argparse.Namespace) -> bool:
    """Return whether PCD filter is enabled."""
    return bool(args.enable_pcd_filter) and str(args.pcd_filter_mode) != "none"


def pcd_filter_preset_to_filter(preset: str | None) -> str | None:
    """Return the PCD filter preset to filter."""
    if preset is None:
        return None
    normalized = str(preset).strip().lower()
    if not normalized:
        return None
    if normalized == PCD_FILTER_PRESET_ORIGINAL:
        return PCD_FILTER_NONE
    if normalized == PCD_FILTER_PRESET_PT:
        return PCD_FILTER_PT_FILTER
    if normalized == PCD_FILTER_PRESET_ENHANCED_PT:
        return PCD_FILTER_ENHANCED_PT
    raise ValueError(f"--pcd-filter-preset must be one of {', '.join(PCD_FILTER_PRESETS)}")


def tracker_query_source(args: argparse.Namespace) -> str:
    """Return the tracker query source."""
    if tracking_product_backend_is_strict(getattr(args, "tracking_product_backend", DEFAULT_TRACKING_PRODUCT_BACKEND)):
        return TRACKER_QUERY_SOURCE_UNION_MASK
    return (
        TRACKER_QUERY_SOURCE_PCD_FILTER_RESIDUAL
        if pcd_filter_preset_to_filter(getattr(args, "pcd_filter_preset", None)) is not None
        else TRACKER_QUERY_SOURCE_UNION_MASK
    )


def tracker_marker_gate(args: argparse.Namespace) -> str:
    """Return the tracker marker gate."""
    return (
        TRACKER_MARKER_GATE_PCD_FILTER_RESIDUAL_TABLE_Z
        if tracker_query_source(args) == TRACKER_QUERY_SOURCE_PCD_FILTER_RESIDUAL
        else TRACKER_MARKER_GATE_TARGET_MASK_DEPTH
    )


def tracker_retire_filtered_markers(args: argparse.Namespace) -> bool:
    """Return the tracker retire filtered markers."""
    return bool(getattr(args, "tracker_retire_filtered_markers", False))


def tracker_marker_retirement_policy(args: argparse.Namespace) -> str:
    """Return the tracker marker retirement policy."""
    if (
        tracker_retire_filtered_markers(args)
        and tracker_marker_gate(args) == TRACKER_MARKER_GATE_PCD_FILTER_RESIDUAL_TABLE_Z
    ):
        return TRACKER_MARKER_RETIREMENT_POLICY_PCD_FILTER_RESIDUAL_TABLE_Z_ONCE_FALSE
    return TRACKER_MARKER_RETIREMENT_POLICY_DISABLED


def headless_capture_enabled(args: argparse.Namespace) -> bool:
    """Return whether headless capture is enabled."""
    return args.headless_capture_dir is not None


def headless_capture_saved_pcd_source(args: argparse.Namespace) -> str:
    """Return the headless capture saved PCD source."""
    object_filter = str(getattr(args, "object_filter", DEFAULT_OBJECT_FILTER)).replace("-", "_")
    controller_filter = str(getattr(args, "controller_filter", DEFAULT_CONTROLLER_FILTER)).replace("-", "_")
    if object_filter == controller_filter:
        return f"{object_filter}_filtered"
    return f"object_{object_filter}_controller_{controller_filter}_filtered"


def validate_args(args: argparse.Namespace) -> None:
    """Validate args."""
    parse_profile(args.profile)
    if args.input_source not in INPUT_SOURCES:
        raise ValueError(f"--input-source must be one of {', '.join(INPUT_SOURCES)}")
    if args.depth_source not in DEPTH_SOURCES:
        raise ValueError(f"--depth-source must be one of {', '.join(DEPTH_SOURCES)}")
    if float(args.replay_fps) < 0.0:
        raise ValueError("--replay-fps must be >= 0")
    if float(args.lossless_max_backlog_seconds) <= 0.0:
        raise ValueError("--lossless-max-backlog-seconds must be positive")
    if float(args.lossless_input_fps) <= 0.0:
        raise ValueError("--lossless-input-fps must be positive")
    if args.table_calibrate is not None:
        table_path = Path(args.table_calibrate).expanduser()
        if not table_path.is_absolute():
            table_path = REPO_ROOT / table_path
        table_path = table_path.resolve(strict=False)
        try:
            load_table_calibration_transforms(table_path)
        except TableCalibrationLoadError as exc:
            message = str(exc)
            if "Missing table calibration file" in message:
                raise ValueError(message) from exc
            raise ValueError(f"Invalid table calibration file: {message}") from exc
        args.table_calibrate = table_path
    if args.input_source == INPUT_SOURCE_FAKE_LIVE and args.recording_case is None:
        args.recording_case = DEFAULT_FAKE_LIVE_CASE
    if _is_replay_input_source(str(args.input_source)):
        if args.recording_case is None:
            raise ValueError(f"--input-source {args.input_source} requires --recording-case or --fake-live-case")
    elif args.recording_case is not None:
        raise ValueError("--recording-case/--fake-live-case requires --input-source recording or fake-live")
    if args.demo_preset == "local-ffs-professor" and args.depth_source != "ffs":
        raise ValueError("--demo-preset local-ffs-professor requires --depth-source ffs")
    if bool(args.shape_prior_warmup) and not str(args.shape_prior_controller_name or "").strip():
        raise ValueError(
            "--shape-prior-controller-name is required when --shape-prior-warmup "
            "is enabled"
        )
    if args.depth_min_m < 0:
        raise ValueError("--depth-min-m must be >= 0")
    if args.depth_max_m > 0 and args.depth_max_m <= args.depth_min_m:
        raise ValueError("--depth-max-m must be <=0 or greater than --depth-min-m")
    if args.pcd_max_points < 0:
        raise ValueError("--pcd-max-points must be >= 0")
    if args.pcd_stride < 1:
        raise ValueError("--pcd-stride must be >= 1")
    if int(args.pcd_mask_erode_pixels) < 0:
        raise ValueError("--pcd-mask-erode-pixels must be >= 0")
    if args.object_pcd_mask_erode_pixels is not None and int(args.object_pcd_mask_erode_pixels) < 0:
        raise ValueError("--object-pcd-mask-erode-pixels must be >= 0")
    if args.controller_pcd_mask_erode_pixels is not None and int(args.controller_pcd_mask_erode_pixels) < 0:
        raise ValueError("--controller-pcd-mask-erode-pixels must be >= 0")
    if int(args.edgetam_live_session_keep_frames) < 0:
        raise ValueError("--edgetam-live-session-keep-frames must be >= 0")
    if not np.isfinite(float(args.edgetam_mask_logit_threshold)):
        raise ValueError("--edgetam-mask-logit-threshold must be finite")
    if float(args.table_z_filter_threshold_m) < 0:
        raise ValueError("--table-z-filter-threshold-m must be >= 0")
    if (
        int(
            getattr(
                args,
                "shape_prior_timeout_ms",
                shape_prior_warmup.DEFAULT_SHAPE_PRIOR_TIMEOUT_MS,
            )
        )
        <= 0
    ):
        raise ValueError("--shape-prior-timeout-ms must be positive")
    if bool(getattr(args, "shape_prior_warmup", False)) and not getattr(
        args, "table_calibrate", None
    ):
        # Without the table world frame the frame-0 shape-prior request can
        # never be built, so the prior would sit in 'pending' forever and the
        # formal chunk timeline would never start.
        raise ValueError("--shape-prior-warmup requires --table-calibrate")
    if bool(getattr(args, "enable_table_z_filter", False)) and bool(
        getattr(args, "disable_table_z_filter", False)
    ):
        raise ValueError("--enable-table-z-filter conflicts with --disable-table-z-filter")
    if str(args.table_z_filter_classes) not in TABLE_Z_FILTER_CLASSES:
        raise ValueError(
            f"--table-z-filter-classes must be one of {', '.join(TABLE_Z_FILTER_CLASSES)}"
        )
    if args.pcd_filter_mode not in PCD_FILTER_MODES:
        raise ValueError(f"--pcd-filter-mode must be one of {', '.join(PCD_FILTER_MODES)}")
    preset_filter = pcd_filter_preset_to_filter(getattr(args, "pcd_filter_preset", None))
    if preset_filter is not None:
        args.enable_pcd_filter = True
        args.pcd_filter_mode = "sync"
        args.object_filter = preset_filter
        args.controller_filter = preset_filter
        if str(getattr(args, "pcd_filter_preset", "")) == PCD_FILTER_PRESET_ORIGINAL:
            args.object_filter_cap = 0
            args.controller_filter_cap = 0
    for flag in (
        "object_filter_cap",
        "controller_filter_cap",
        "filter_min_cap",
        "object_filter_keep_components",
        "controller_filter_keep_components",
        "filter_max_age_frames",
    ):
        if int(getattr(args, flag)) < 0:
            raise ValueError(f"--{flag.replace('_', '-')} must be >= 0")
    if int(args.object_filter_keep_components) < 1:
        raise ValueError("--object-filter-keep-components must be >= 1")
    if int(args.controller_filter_keep_components) < 1:
        raise ValueError("--controller-filter-keep-components must be >= 1")
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
    if str(args.demo_visual_mode) not in DEMO_VISUAL_MODES:
        raise ValueError(f"--demo-visual-mode must be one of {', '.join(DEMO_VISUAL_MODES)}")
    if headless_capture_enabled(args):
        if args.input_source not in {INPUT_SOURCE_FAKE_LIVE, INPUT_SOURCE_LIVE}:
            raise ValueError("--headless-capture-dir requires --input-source live or fake-live")
        if args.depth_source not in {"ffs", "realsense"}:
            raise ValueError("--headless-capture-dir requires --depth-source ffs or realsense")
        if args.pcd_mode != "masked":
            raise ValueError("--headless-capture-dir requires --pcd-mode masked")
        if not pcd_filter_enabled(args):
            raise ValueError("--headless-capture-dir requires --enable-pcd-filter")
        if args.pcd_filter_mode != "sync":
            raise ValueError("--headless-capture-dir requires --pcd-filter-mode sync")
        if args.object_filter not in HEADLESS_CAPTURE_ALLOWED_PCD_FILTERS:
            allowed = ", ".join(HEADLESS_CAPTURE_ALLOWED_PCD_FILTERS)
            raise ValueError(f"--headless-capture-dir requires --object-filter one of {allowed}")
        if args.controller_filter not in HEADLESS_CAPTURE_ALLOWED_PCD_FILTERS:
            allowed = ", ".join(HEADLESS_CAPTURE_ALLOWED_PCD_FILTERS)
            raise ValueError(f"--headless-capture-dir requires --controller-filter one of {allowed}")
    args.tracker_backend = normalize_tracker_backend(str(args.tracker_backend))
    args.tracking_product_backend = normalize_tracking_product_backend(
        getattr(args, "tracking_product_backend", DEFAULT_TRACKING_PRODUCT_BACKEND)
    )
    if tracking_product_backend_is_strict(args.tracking_product_backend):
        if str(args.input_source) not in {INPUT_SOURCE_FAKE_LIVE, INPUT_SOURCE_LIVE}:
            raise ValueError("phystwin-strict-tracking requires --input-source live or fake-live")
        if args.headless_capture_dir is None:
            raise ValueError("phystwin-strict-tracking requires --headless-capture-dir")
        if str(args.track_mode) != TRACK_MODE_CONTROLLER_OBJECT:
            raise ValueError("phystwin-strict-tracking requires --track-mode controller-object")
        if str(args.tracker_backend) != TRACKER_BACKEND_TAPNEXTPP:
            raise ValueError("phystwin-strict-tracking requires --tracker-backend tapnextpp")
        if args.phystwin_strict_output_dir is None:
            args.phystwin_strict_output_dir = Path(args.headless_capture_dir) / "phystwin_like"
    elif bool(getattr(args, "headless_prepared_only", False)):
        raise ValueError("--headless-prepared-only requires --tracking-product-backend phystwin-strict-tracking")
    if int(args.tracker_query_count) < 0:
        raise ValueError("--tracker-query-count must be >= 0")
    if int(args.tracker_overlay_max_points) < 0:
        raise ValueError("--tracker-overlay-max-points must be >= 0")
    if float(args.tracker_marker_point_size) <= 0:
        raise ValueError("--tracker-marker-point-size must be positive")
    if getattr(args, "color_exposure", None) is not None and float(args.color_exposure) <= 0.0:
        raise ValueError("--color-exposure must be positive")
    if getattr(args, "color_gain", None) is not None and float(args.color_gain) < 0.0:
        raise ValueError("--color-gain must be >= 0")
    if tracker_enabled(args):
        if args.tracker_backend != TRACKER_BACKEND_TAPNEXTPP:
            raise ValueError("single-camera tracker overlay currently supports only tapnextpp")
        if args.track_mode != TRACK_MODE_CONTROLLER_OBJECT:
            raise ValueError("--tracker-backend tapnextpp requires --track-mode controller-object")
        if args.depth_source == "none":
            raise ValueError("--tracker-backend tapnextpp requires RGB-D depth for 3D marker lift")
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
                raise ValueError(
                    "sparse --depth-source ffs_remote requires EdgeTAM masks; "
                    "use --track-mode object-only, controller-only, or controller-object"
                )
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
            raise ValueError(
                "sparse remote quality returns require EdgeTAM masks; "
                "use --track-mode object-only, controller-only, or controller-object"
            )
    if args.track_mode not in TRACK_MODES:
        raise ValueError(f"--track-mode must be one of {', '.join(TRACK_MODES)}")
    if args.init_mode == "saved-masks":
        if object_tracking_enabled(args) and not args.object_init_mask:
            raise ValueError("saved-masks object tracking requires --object-init-mask")
        if controller_tracking_enabled(args) and not args.controller_init_mask:
            raise ValueError("saved-masks controller tracking requires --controller-init-mask")
        required_masks = []
        if object_tracking_enabled(args):
            required_masks.append(("--object-init-mask", args.object_init_mask))
        if controller_tracking_enabled(args):
            required_masks.append(("--controller-init-mask", args.controller_init_mask))
        for flag, value in required_masks:
            path = _resolve_path(value)
            if not path.is_file():
                raise ValueError(f"{flag} does not exist: {path}")


# ---------------------------------------------------------------------------
# RealSense capture startup
# ---------------------------------------------------------------------------
def _start_realsense_pipeline(args: argparse.Namespace) -> RealtimeCameraRuntime:
    """Start realsense pipeline."""
    rs = load_realsense_module()
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
        apply_emitter(profile, args.emitter, rs)
        # Fixed RGB exposure/gain only stick after auto-exposure is disabled on the sensor.
        exposure = getattr(args, "color_exposure", None)
        gain = getattr(args, "color_gain", None)
        if exposure is not None or gain is not None:
            color_sensor = profile.get_device().first_color_sensor()
            if color_sensor.supports(rs.option.enable_auto_exposure):
                color_sensor.set_option(rs.option.enable_auto_exposure, 0.0)
            if exposure is not None:
                if not color_sensor.supports(rs.option.exposure):
                    raise RuntimeError("RealSense RGB sensor does not support exposure control")
                color_sensor.set_option(rs.option.exposure, float(exposure))
            if gain is not None:
                if not color_sensor.supports(rs.option.gain):
                    raise RuntimeError("RealSense RGB sensor does not support gain control")
                color_sensor.set_option(rs.option.gain, float(gain))
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


# ---------------------------------------------------------------------------
# Depth backprojection & mask erosion (masked RGB-D -> per-class point clouds)
# ---------------------------------------------------------------------------
def _masked_sample_indices(
    *,
    depth_m: np.ndarray,
    mask: np.ndarray,
    depth_min_m: float,
    depth_max_m: float,
    max_points: int,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the masked sample indices."""
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


def erode_binary_mask(mask: np.ndarray, *, erode_pixels: int) -> np.ndarray:
    """Erode a binary mask by the requested pixel radius."""
    pixels = int(erode_pixels)
    if pixels < 0:
        raise ValueError("erode_pixels must be >= 0")
    mask_bool = np.asarray(mask, dtype=bool)
    if pixels == 0 or mask_bool.size == 0 or not np.any(mask_bool):
        return np.ascontiguousarray(mask_bool)

    eroded = mask_bool
    for _ in range(pixels):
        padded = np.pad(eroded, 1, mode="constant", constant_values=False)
        eroded = (
            padded[:-2, :-2]
            & padded[:-2, 1:-1]
            & padded[:-2, 2:]
            & padded[1:-1, :-2]
            & padded[1:-1, 1:-1]
            & padded[1:-1, 2:]
            & padded[2:, :-2]
            & padded[2:, 1:-1]
            & padded[2:, 2:]
        )
        if not np.any(eroded):
            break
    return np.ascontiguousarray(eroded)


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
    """Back-project masked rgbd."""
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
    return_yx: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]] | tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Back-project masked rgbd profiled."""
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
        empty_points = np.empty((0, 3), dtype=np.float32)
        empty_colors = np.empty((0, 3), dtype=np.uint8)
        empty_yx = np.empty((0, 2), dtype=np.int64)
        if return_yx:
            return empty_points, empty_colors, empty_yx, timing
        return empty_points, empty_colors, timing
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
    if return_yx:
        yx = np.ascontiguousarray(np.stack([rows, cols], axis=1), dtype=np.int64)
        return points, colors, yx, timing
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
    """Back-project masked."""
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
    """Create solid colors."""
    if point_count <= 0:
        return np.empty((0, 3), dtype=np.uint8)
    color = np.asarray(rgb, dtype=np.uint8).reshape(1, 3)
    return np.repeat(color, int(point_count), axis=0)


# ---------------------------------------------------------------------------
# Segmentation (EdgeTAM) helpers & model timing
# ---------------------------------------------------------------------------
def controller_tracking_enabled(args_or_track_mode: argparse.Namespace | str) -> bool:
    """Return whether controller tracking is enabled."""
    track_mode = args_or_track_mode if isinstance(args_or_track_mode, str) else args_or_track_mode.track_mode
    return str(track_mode) in {TRACK_MODE_CONTROLLER_OBJECT, TRACK_MODE_CONTROLLER_ONLY}


def object_tracking_enabled(args_or_track_mode: argparse.Namespace | str) -> bool:
    """Return whether object tracking is enabled."""
    track_mode = args_or_track_mode if isinstance(args_or_track_mode, str) else args_or_track_mode.track_mode
    return str(track_mode) in {TRACK_MODE_CONTROLLER_OBJECT, TRACK_MODE_OBJECT_ONLY}


def object_id_labels(track_mode: str = DEFAULT_TRACK_MODE) -> dict[int, str]:
    """Return the object id labels."""
    if track_mode == TRACK_MODE_NONE:
        return {}
    if track_mode == TRACK_MODE_OBJECT_ONLY:
        return {OBJECT_ID: EDGE_TAM_OBJECT_LABELS[OBJECT_ID]}
    if track_mode == TRACK_MODE_CONTROLLER_ONLY:
        return {
            HAND_A_ID: EDGE_TAM_OBJECT_LABELS[HAND_A_ID],
            HAND_B_ID: EDGE_TAM_OBJECT_LABELS[HAND_B_ID],
        }
    if track_mode == TRACK_MODE_CONTROLLER_OBJECT:
        return dict(EDGE_TAM_OBJECT_LABELS)
    raise ValueError(f"unsupported track mode: {track_mode}")


def active_object_id_labels(args: argparse.Namespace) -> dict[int, str]:
    """Return the active object id labels."""
    return object_id_labels(str(args.track_mode))


def active_object_ids(args: argparse.Namespace) -> list[int]:
    """Return the active object ids."""
    return list(active_object_id_labels(args).keys())


def extract_object_masks_from_hf_output(
    output: Any,
    post_masks: Any,
    *,
    mask_logit_threshold: float = DEFAULT_EDGETAM_MASK_LOGIT_THRESHOLD,
) -> dict[int, np.ndarray]:
    # HF EdgeTAM may hand back object ids as a torch tensor, ndarray, scalar, or list.
    """Extract object masks from HF output."""
    ids_value = getattr(output, "object_ids")
    if hasattr(ids_value, "detach"):
        ids_value = ids_value.detach().cpu().tolist()
    if isinstance(ids_value, np.ndarray):
        ids_value = ids_value.tolist()
    if isinstance(ids_value, (int, np.integer)):
        object_ids = [int(ids_value)]
    else:
        object_ids = [int(item) for item in list(ids_value)]
    if len(object_ids) != len(post_masks):
        raise RuntimeError(f"HF output object_ids length {len(object_ids)} != mask length {len(post_masks)}")
    masks: dict[int, np.ndarray] = {}
    for idx, obj_id in enumerate(object_ids):
        # Masks may be GPU tensors with singleton dims; normalize each to a contiguous HxW bool array.
        value = post_masks[idx]
        if hasattr(value, "detach"):
            value = value.detach().float().cpu().numpy()
        array = np.squeeze(np.asarray(value))
        if array.ndim != 2:
            raise RuntimeError(f"expected 2D mask after squeeze, got {array.shape}")
        masks[int(obj_id)] = np.ascontiguousarray(
            array > float(mask_logit_threshold)
        )
    return masks


def _load_hf_streaming_runtime() -> Any:
    """Load HF streaming runtime."""
    from scripts.harness.experiments.edgetam import run_hf_edgetam_streaming_realcase as hf_stream

    hf_stream._load_runtime_dependencies()
    return hf_stream


def _sync_if_needed(torch_module: Any, device: str) -> None:
    """Synchronize if needed."""
    if str(device).startswith("cuda") and torch_module.cuda.is_available():
        torch_module.cuda.synchronize()


def _time_runtime_ms(
    torch_module: Any,
    device: str,
    fn: Callable[[], Any],
    *,
    sync_enabled: bool = False,
) -> tuple[Any, float, float, float]:
    """Measure runtime ms."""
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


# Like _time_runtime_ms, but additionally brackets fn() with CUDA events so GPU time can be
# separated from wall time on async launches. Returns (value, wall_ms, cuda_event_ms,
# pre_sync_ms, post_sync_ms); cuda_event_ms is 0.0 when events are disabled or CUDA is absent.
def _time_model_forward(
    *,
    torch_module: Any,
    device: str,
    profile_sync: bool,
    profile_cuda_events: bool,
    fn: Callable[[], Any],
) -> tuple[Any, float, float, float, float]:
    """Measure model forward."""
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


def tracker_enabled(args: argparse.Namespace) -> bool:
    """Return whether tracker is enabled."""
    return normalize_tracker_backend(str(getattr(args, "tracker_backend", TRACKER_BACKEND_NONE))) != TRACKER_BACKEND_NONE


def object_pcd_mask_erode_pixels(args: argparse.Namespace) -> int:
    """Return the object PCD mask erode pixels."""
    value = getattr(args, "object_pcd_mask_erode_pixels", None)
    if value is None:
        value = getattr(args, "pcd_mask_erode_pixels", DEFAULT_PCD_MASK_ERODE_PIXELS)
    return int(value)


def controller_pcd_mask_erode_pixels(args: argparse.Namespace) -> int:
    """Return the controller PCD mask erode pixels."""
    value = getattr(args, "controller_pcd_mask_erode_pixels", None)
    if value is None:
        value = getattr(args, "pcd_mask_erode_pixels", DEFAULT_PCD_MASK_ERODE_PIXELS)
    return int(value)


# ---------------------------------------------------------------------------
# World-Z diagnostics & table-Z filtering
# ---------------------------------------------------------------------------
def _camera_intrinsics_matrix(intrinsics: CameraIntrinsics) -> np.ndarray:
    """Return the camera intrinsics matrix."""
    return np.array(
        [
            [float(intrinsics.fx), 0.0, float(intrinsics.cx)],
            [0.0, float(intrinsics.fy), float(intrinsics.cy)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _transform_points_c2w(points_xyz_m: np.ndarray, c2w: np.ndarray | None) -> np.ndarray:
    """Transform points C2W."""
    points = np.asarray(points_xyz_m, dtype=np.float32).reshape(-1, 3)
    if c2w is None or points.size == 0:
        return np.ascontiguousarray(points, dtype=np.float32)
    matrix = np.asarray(c2w, dtype=np.float32)
    if matrix.shape != (4, 4):
        raise ValueError(f"camera-to-world transform must be 4x4, got {matrix.shape}")
    homogeneous = np.concatenate(
        [points, np.ones((points.shape[0], 1), dtype=np.float32)],
        axis=1,
    )
    world = (matrix @ homogeneous.T).T[:, :3]
    return np.ascontiguousarray(world, dtype=np.float32)


def _z_quantiles(points_xyz_m: np.ndarray) -> dict[str, float | None]:
    """Return the z quantiles."""
    keys = ("min", "p01", "p05", "p10", "p50", "p90", "p95", "p99", "max")
    points = np.asarray(points_xyz_m, dtype=np.float32).reshape(-1, 3)
    z = points[:, 2]
    finite = z[np.isfinite(z)]
    if finite.size == 0:
        # Covers both empty input and all-NaN/inf depth: every quantile is None.
        return {key: None for key in keys}
    quantiles = np.quantile(
        finite.astype(np.float64),
        [0.0, 0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99, 1.0],
    )
    return {key: float(value) for key, value in zip(keys, quantiles)}


def table_z_clearance_m(
    points_xyz_m: np.ndarray,
    *,
    table_z_m: float = TABLE_Z_M,
) -> np.ndarray:
    """Return the table z clearance m."""
    points = np.asarray(points_xyz_m, dtype=np.float32).reshape(-1, 3)
    return np.ascontiguousarray(
        np.float32(table_z_m) - points[:, 2],
        dtype=np.float32,
    )


def _world_z_class_stats(
    points_xyz_m: np.ndarray,
    *,
    table_z_m: float,
    thresholds_m: tuple[float, ...],
) -> dict[str, Any]:
    """Return the world z class stats."""
    points = np.asarray(points_xyz_m, dtype=np.float32).reshape(-1, 3)
    finite = np.isfinite(points).all(axis=1) if len(points) else np.zeros((0,), dtype=bool)
    clearance = table_z_clearance_m(points, table_z_m=table_z_m)
    threshold_rows: list[dict[str, float | int]] = []
    for threshold_m in thresholds_m:
        candidate = finite & (clearance <= np.float32(float(threshold_m)))
        count = int(np.count_nonzero(candidate))
        threshold_rows.append(
            {
                "threshold_m": float(threshold_m),
                "candidate_count": count,
                "candidate_ratio": float(count / max(1, len(points))),
            }
        )
    return {
        "count": int(len(points)),
        "finite_count": int(np.count_nonzero(finite)),
        "z_m": _z_quantiles(points),
        "table_thresholds": threshold_rows,
    }


def build_world_z_diagnostics(
    *,
    object_xyz_m: np.ndarray,
    controller_xyz_m: np.ndarray,
    hand_a_xyz_m: np.ndarray | None = None,
    hand_b_xyz_m: np.ndarray | None = None,
    table_z_m: float = TABLE_Z_M,
    thresholds_m: tuple[float, ...] = DEFAULT_TABLE_Z_DIAGNOSTIC_THRESHOLDS_M,
) -> dict[str, Any]:
    """Build world z diagnostics."""
    thresholds = tuple(float(value) for value in thresholds_m)
    classes: dict[str, Any] = {
        "object": _world_z_class_stats(
            object_xyz_m,
            table_z_m=float(table_z_m),
            thresholds_m=thresholds,
        ),
        "controller": _world_z_class_stats(
            controller_xyz_m,
            table_z_m=float(table_z_m),
            thresholds_m=thresholds,
        ),
    }
    if hand_a_xyz_m is not None:
        classes["hand_a"] = _world_z_class_stats(
            hand_a_xyz_m,
            table_z_m=float(table_z_m),
            thresholds_m=thresholds,
        )
    if hand_b_xyz_m is not None:
        classes["hand_b"] = _world_z_class_stats(
            hand_b_xyz_m,
            table_z_m=float(table_z_m),
            thresholds_m=thresholds,
        )
    return {
        "table_z_m": float(table_z_m),
        "table_z_above_direction": TABLE_Z_ABOVE_DIRECTION,
        "thresholds_m": [float(value) for value in thresholds],
        "classes": classes,
    }


def apply_table_z_filter(
    points_xyz_m: np.ndarray,
    colors_rgb_u8: np.ndarray,
    *,
    enabled: bool,
    threshold_m: float,
    table_z_m: float = TABLE_Z_M,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Apply table z filter."""
    points = np.asarray(points_xyz_m, dtype=np.float32).reshape(-1, 3)
    colors = np.asarray(colors_rgb_u8, dtype=np.uint8).reshape(-1, 3)
    if len(points) != len(colors):
        raise ValueError("points and colors must have the same first dimension")
    if not bool(enabled) or len(points) == 0:
        return np.ascontiguousarray(points, dtype=np.float32), np.ascontiguousarray(colors, dtype=np.uint8), {
            "enabled": bool(enabled),
            "threshold_m": float(threshold_m),
            "table_z_m": float(table_z_m),
            "table_z_above_direction": TABLE_Z_ABOVE_DIRECTION,
            "input_points": int(len(points)),
            "removed_points": 0,
            "output_points": int(len(points)),
            "removed_ratio": 0.0,
        }
    finite = np.isfinite(points).all(axis=1)
    clearance = table_z_clearance_m(points, table_z_m=table_z_m)
    remove = finite & (clearance <= np.float32(float(threshold_m)))
    keep = ~remove
    removed = int(np.count_nonzero(remove))
    return (
        np.ascontiguousarray(points[keep], dtype=np.float32),
        np.ascontiguousarray(colors[keep], dtype=np.uint8),
        {
            "enabled": True,
            "threshold_m": float(threshold_m),
            "table_z_m": float(table_z_m),
            "table_z_above_direction": TABLE_Z_ABOVE_DIRECTION,
            "input_points": int(len(points)),
            "removed_points": removed,
            "output_points": int(np.count_nonzero(keep)),
            "removed_ratio": float(removed / max(1, len(points))),
        },
    )


def apply_table_z_filter_with_yx(
    points_xyz_m: np.ndarray,
    colors_rgb_u8: np.ndarray,
    yx: np.ndarray,
    *,
    enabled: bool,
    threshold_m: float,
    table_z_m: float = TABLE_Z_M,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Apply table z filter with YX."""
    points = np.asarray(points_xyz_m, dtype=np.float32).reshape(-1, 3)
    colors = np.asarray(colors_rgb_u8, dtype=np.uint8).reshape(-1, 3)
    yx_arr = np.asarray(yx, dtype=np.int64).reshape(-1, 2)
    if len(points) != len(colors) or len(points) != len(yx_arr):
        raise ValueError("points, colors, and yx must have the same first dimension")
    if not bool(enabled) or len(points) == 0:
        return (
            np.ascontiguousarray(points, dtype=np.float32),
            np.ascontiguousarray(colors, dtype=np.uint8),
            np.ascontiguousarray(yx_arr, dtype=np.int64),
            {
                "enabled": bool(enabled),
                "threshold_m": float(threshold_m),
                "table_z_m": float(table_z_m),
                "table_z_above_direction": TABLE_Z_ABOVE_DIRECTION,
                "input_points": int(len(points)),
                "removed_points": 0,
                "output_points": int(len(points)),
                "removed_ratio": 0.0,
            },
        )
    finite = np.isfinite(points).all(axis=1)
    clearance = table_z_clearance_m(points, table_z_m=table_z_m)
    remove = finite & (clearance <= np.float32(float(threshold_m)))
    keep = ~remove
    removed = int(np.count_nonzero(remove))
    return (
        np.ascontiguousarray(points[keep], dtype=np.float32),
        np.ascontiguousarray(colors[keep], dtype=np.uint8),
        np.ascontiguousarray(yx_arr[keep], dtype=np.int64),
        {
            "enabled": True,
            "threshold_m": float(threshold_m),
            "table_z_m": float(table_z_m),
            "table_z_above_direction": TABLE_Z_ABOVE_DIRECTION,
            "input_points": int(len(points)),
            "removed_points": removed,
            "output_points": int(np.count_nonzero(keep)),
            "removed_ratio": float(removed / max(1, len(points))),
        },
    )


# ---------------------------------------------------------------------------
# Tracker query classification, visibility & marker gating
# ---------------------------------------------------------------------------
def _tracker_union_mask(mask_packet: MaskPacket) -> np.ndarray:
    """Return the tracker union mask."""
    controller = np.asarray(mask_packet.controller_mask, dtype=bool)
    obj = np.asarray(mask_packet.object_mask, dtype=bool)
    if controller.shape != obj.shape:
        raise ValueError("controller/object masks must share a shape")
    return np.logical_or(controller, obj)


def _mask_packet_hand_a_mask(mask_packet: MaskPacket) -> np.ndarray:
    """Return the mask packet hand a mask."""
    if mask_packet.hand_a_mask is None:
        return np.asarray(mask_packet.controller_mask, dtype=bool)
    return np.asarray(mask_packet.hand_a_mask, dtype=bool)


def _mask_packet_hand_b_mask(mask_packet: MaskPacket) -> np.ndarray:
    """Return the mask packet hand b mask."""
    if mask_packet.hand_b_mask is None:
        return np.zeros_like(np.asarray(mask_packet.controller_mask, dtype=bool), dtype=bool)
    return np.asarray(mask_packet.hand_b_mask, dtype=bool)


def _classify_query_points_yx(
    query_points_yx: np.ndarray,
    *,
    object_mask: np.ndarray,
    controller_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Classify query points YX."""
    points = np.asarray(query_points_yx, dtype=np.float32).reshape(-1, 2)
    if len(points) == 0:
        empty = np.empty((0,), dtype=bool)
        return empty, empty
    object_bool = np.asarray(object_mask, dtype=bool)
    controller_bool = np.asarray(controller_mask, dtype=bool)
    height, width = object_bool.shape[:2]
    yy = np.clip(np.rint(points[:, 0]).astype(np.int64), 0, height - 1)
    xx = np.clip(np.rint(points[:, 1]).astype(np.int64), 0, width - 1)
    return object_bool[yy, xx].astype(bool), controller_bool[yy, xx].astype(bool)


def _classify_query_targets_yx(
    query_points_yx: np.ndarray,
    *,
    object_mask: np.ndarray,
    hand_a_mask: np.ndarray,
    hand_b_mask: np.ndarray,
    controller_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Classify query targets YX."""
    points = np.asarray(query_points_yx, dtype=np.float32).reshape(-1, 2)
    if len(points) == 0:
        empty_bool = np.empty((0,), dtype=bool)
        empty_int = np.empty((0,), dtype=np.int64)
        return empty_bool, empty_bool, empty_int, empty_int
    object_bool = np.asarray(object_mask, dtype=bool)
    hand_a_bool = np.asarray(hand_a_mask, dtype=bool)
    hand_b_bool = np.asarray(hand_b_mask, dtype=bool)
    controller_bool = np.asarray(controller_mask, dtype=bool)
    height, width = object_bool.shape[:2]
    yy = np.clip(np.rint(points[:, 0]).astype(np.int64), 0, height - 1)
    xx = np.clip(np.rint(points[:, 1]).astype(np.int64), 0, width - 1)
    in_hand_a = hand_a_bool[yy, xx]
    in_hand_b = hand_b_bool[yy, xx] & ~in_hand_a
    in_object = object_bool[yy, xx] & ~(in_hand_a | in_hand_b)
    in_controller = controller_bool[yy, xx] | in_hand_a | in_hand_b
    target_id = np.zeros((len(points),), dtype=np.int64)
    target_id[in_object] = OBJECT_ID
    target_id[in_hand_a] = HAND_A_ID
    target_id[in_hand_b] = HAND_B_ID
    controller_instance_id = np.zeros((len(points),), dtype=np.int64)
    controller_instance_id[in_hand_a] = QUERY_CONTROLLER_INSTANCE_HAND_A
    controller_instance_id[in_hand_b] = QUERY_CONTROLLER_INSTANCE_HAND_B
    return in_object.astype(bool), in_controller.astype(bool), target_id, controller_instance_id


def _mask_from_yx(shape: tuple[int, int], yx: np.ndarray) -> np.ndarray:
    """Return the mask from YX."""
    mask = np.zeros(tuple(shape), dtype=bool)
    coords = np.asarray(yx, dtype=np.int64).reshape(-1, 2)
    if len(coords) == 0:
        return mask
    rows = coords[:, 0]
    cols = coords[:, 1]
    valid = (rows >= 0) & (rows < shape[0]) & (cols >= 0) & (cols < shape[1])
    if np.any(valid):
        mask[rows[valid], cols[valid]] = True
    return np.ascontiguousarray(mask)


def _select_points_by_yx_mask(points_xyz_m: np.ndarray, yx: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Select 3D points whose YX pixel locations fall inside a mask."""
    points = np.asarray(points_xyz_m, dtype=np.float32).reshape(-1, 3)
    coords = np.asarray(yx, dtype=np.int64).reshape(-1, 2)
    if len(points) == 0 or len(coords) == 0:
        return np.empty((0, 3), dtype=np.float32)
    count = min(len(points), len(coords))
    target = np.asarray(mask, dtype=bool)
    rows = coords[:count, 0]
    cols = coords[:count, 1]
    valid = (rows >= 0) & (rows < target.shape[0]) & (cols >= 0) & (cols < target.shape[1])
    keep = np.zeros((count,), dtype=bool)
    if np.any(valid):
        keep[valid] = target[rows[valid], cols[valid]]
    return np.ascontiguousarray(points[:count][keep], dtype=np.float32)


# Zeroes visibility for markers outside --tracker-display-scope (display-only; the tracker
# itself keeps tracking every query point). Label arrays shorter/longer than the visibility
# vector are fitted with False padding so scope filtering never raises on length drift.
def _tracker_display_visibility(
    visibility: np.ndarray,
    *,
    query_is_object: np.ndarray,
    query_is_controller: np.ndarray,
    display_scope: str,
) -> np.ndarray:
    """Return the tracker display visibility."""
    vis = np.asarray(visibility, dtype=np.float32).reshape(-1)
    scope = str(display_scope)
    if scope == TRACKER_DISPLAY_SCOPE_UNION:
        return vis
    if scope == TRACKER_DISPLAY_SCOPE_OBJECT:
        labels = np.asarray(query_is_object, dtype=bool).reshape(-1)
    else:
        labels = np.asarray(query_is_controller, dtype=bool).reshape(-1)
    if labels.shape[0] != vis.shape[0]:
        fitted = np.zeros_like(vis, dtype=bool)
        fitted[: min(len(labels), len(fitted))] = labels[: min(len(labels), len(fitted))]
        labels = fitted
    return np.where(labels, vis, 0.0).astype(np.float32)


def _tracker_per_target_visibility(
    tracks_yx: np.ndarray,
    visibility: np.ndarray,
    *,
    mask_packet: MaskPacket,
    query_target_id: np.ndarray,
) -> np.ndarray:
    """Return the tracker per target visibility."""
    tracks = np.asarray(tracks_yx, dtype=np.float32).reshape(-1, 2)
    vis = np.asarray(visibility, dtype=np.float32).reshape(-1)
    target_id = np.asarray(query_target_id, dtype=np.int64).reshape(-1)
    count = min(len(tracks), len(vis), len(target_id))
    output = np.zeros((len(vis),), dtype=np.float32)
    if count == 0:
        return output
    object_mask = np.asarray(mask_packet.object_mask, dtype=bool)
    hand_a_mask = _mask_packet_hand_a_mask(mask_packet)
    hand_b_mask = _mask_packet_hand_b_mask(mask_packet)
    height, width = object_mask.shape[:2]
    yy = np.rint(tracks[:count, 0]).astype(np.int64)
    xx = np.rint(tracks[:count, 1]).astype(np.int64)
    finite_tracks = np.isfinite(tracks[:count]).all(axis=1)
    in_bounds = (yy >= 0) & (yy < height) & (xx >= 0) & (xx < width)
    valid = (vis[:count] > 0.0) & finite_tracks & in_bounds
    if not np.any(valid):
        return output
    valid_indices = np.flatnonzero(valid)
    inside_target = np.zeros((count,), dtype=bool)
    valid_targets = target_id[valid_indices]
    hand_a_indices = valid_indices[valid_targets == HAND_A_ID]
    if len(hand_a_indices):
        inside_target[hand_a_indices] = hand_a_mask[yy[hand_a_indices], xx[hand_a_indices]]
    hand_b_indices = valid_indices[valid_targets == HAND_B_ID]
    if len(hand_b_indices):
        inside_target[hand_b_indices] = hand_b_mask[yy[hand_b_indices], xx[hand_b_indices]]
    object_indices = valid_indices[valid_targets == OBJECT_ID]
    if len(object_indices):
        inside_target[object_indices] = object_mask[yy[object_indices], xx[object_indices]]
    output[:count] = np.where(inside_target, vis[:count], 0.0).astype(np.float32)
    return output


def _tracker_lift_valid_mask(
    *,
    tracks_yx: np.ndarray,
    visibility: np.ndarray,
    depth: np.ndarray,
    depth_scale_m_per_unit: float,
    mask: np.ndarray | None,
    depth_min_m: float,
    depth_max_m: float,
) -> np.ndarray:
    """Return the tracker lift valid mask."""
    tracks = np.asarray(tracks_yx, dtype=np.float32).reshape(-1, 2)
    vis = np.asarray(visibility, dtype=np.float32).reshape(-1) > 0.0
    if vis.shape[0] != tracks.shape[0]:
        raise ValueError("visibility length must match tracks_yx")

    depth_arr = np.asarray(depth)
    if np.issubdtype(depth_arr.dtype, np.floating):
        depth_m = depth_arr.astype(np.float32, copy=False)
    else:
        depth_m = depth_arr.astype(np.float32) * np.float32(depth_scale_m_per_unit)
    height, width = depth_m.shape[:2]
    mask_bool = np.ones((height, width), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if mask_bool.shape[:2] != (height, width):
        raise ValueError("tracker lift mask shape must match depth shape")

    yy = np.rint(tracks[:, 0]).astype(np.int64)
    xx = np.rint(tracks[:, 1]).astype(np.int64)
    finite_tracks = np.isfinite(tracks).all(axis=1)
    in_bounds = (yy >= 0) & (yy < height) & (xx >= 0) & (xx < width)
    valid = vis & finite_tracks & in_bounds
    if not np.any(valid):
        return np.zeros((tracks.shape[0],), dtype=bool)

    valid_indices = np.flatnonzero(valid)
    sampled_depth = depth_m[yy[valid_indices], xx[valid_indices]]
    depth_valid = np.isfinite(sampled_depth) & (sampled_depth > 0.0) & (sampled_depth >= np.float32(depth_min_m))
    if np.isfinite(float(depth_max_m)):
        depth_valid &= sampled_depth <= np.float32(depth_max_m)
    inside_mask = mask_bool[yy[valid_indices], xx[valid_indices]]
    valid_out = np.zeros((tracks.shape[0],), dtype=bool)
    valid_out[valid_indices] = depth_valid & inside_mask
    return valid_out


def _query_current_residual_visibility(
    tracks_yx: np.ndarray,
    *,
    query_is_object: np.ndarray,
    query_is_controller: np.ndarray,
    object_residual_mask: np.ndarray,
    controller_residual_mask: np.ndarray,
) -> np.ndarray:
    """Return the query current residual visibility."""
    tracks = np.asarray(tracks_yx, dtype=np.float32).reshape(-1, 2)
    is_object = np.asarray(query_is_object, dtype=bool).reshape(-1)
    is_controller = np.asarray(query_is_controller, dtype=bool).reshape(-1)
    count = min(len(tracks), len(is_object), len(is_controller))
    visible = np.zeros((len(tracks),), dtype=bool)
    if count <= 0:
        return visible

    object_mask = np.asarray(object_residual_mask, dtype=bool)
    controller_mask = np.asarray(controller_residual_mask, dtype=bool)
    if object_mask.shape != controller_mask.shape:
        raise ValueError("object/controller residual masks must share a shape")
    height, width = object_mask.shape[:2]
    yy = np.rint(tracks[:count, 0]).astype(np.int64)
    xx = np.rint(tracks[:count, 1]).astype(np.int64)
    finite = np.isfinite(tracks[:count]).all(axis=1)
    in_bounds = finite & (yy >= 0) & (yy < height) & (xx >= 0) & (xx < width)
    if not np.any(in_bounds):
        return visible

    valid_indices = np.flatnonzero(in_bounds)
    object_indices = valid_indices[is_object[:count][valid_indices]]
    if len(object_indices):
        visible[object_indices] = object_mask[yy[object_indices], xx[object_indices]]
    controller_indices = valid_indices[is_controller[:count][valid_indices]]
    if len(controller_indices):
        visible[controller_indices] |= controller_mask[yy[controller_indices], xx[controller_indices]]
    unlabelled_indices = valid_indices[~(is_object[:count][valid_indices] | is_controller[:count][valid_indices])]
    if len(unlabelled_indices):
        union_mask = np.logical_or(object_mask, controller_mask)
        visible[unlabelled_indices] = union_mask[yy[unlabelled_indices], xx[unlabelled_indices]]
    return visible


def _audit_marker_residual_subset(
    marker_tracks_yx: np.ndarray,
    *,
    object_residual_mask: np.ndarray,
    controller_residual_mask: np.ndarray,
    gate: str = TRACKER_MARKER_GATE_PCD_FILTER_RESIDUAL_TABLE_Z,
) -> MarkerResidualAudit:
    """Audit marker residual subset."""
    tracks = np.asarray(marker_tracks_yx, dtype=np.float32).reshape(-1, 2)
    object_mask = np.asarray(object_residual_mask, dtype=bool)
    controller_mask = np.asarray(controller_residual_mask, dtype=bool)
    if object_mask.shape != controller_mask.shape:
        raise ValueError("object/controller residual masks must share a shape")

    count = int(tracks.shape[0])
    pixels_yx = np.full((count, 2), -1, dtype=np.int64)
    valid = np.zeros((count,), dtype=bool)
    if count <= 0:
        return MarkerResidualAudit(
            pixels_yx=pixels_yx,
            valid=valid,
            violation=np.zeros((0,), dtype=bool),
            checked_count=0,
            violation_count=0,
            gate=str(gate),
        )

    finite = np.isfinite(tracks).all(axis=1)
    if np.any(finite):
        pixels_yx[finite] = np.rint(tracks[finite]).astype(np.int64)

    height, width = object_mask.shape[:2]
    yy = pixels_yx[:, 0]
    xx = pixels_yx[:, 1]
    in_bounds = finite & (yy >= 0) & (yy < int(height)) & (xx >= 0) & (xx < int(width))
    if np.any(in_bounds):
        union_mask = np.logical_or(object_mask, controller_mask)
        valid[in_bounds] = union_mask[yy[in_bounds], xx[in_bounds]]

    violation = ~valid
    return MarkerResidualAudit(
        pixels_yx=np.ascontiguousarray(pixels_yx, dtype=np.int64),
        valid=np.ascontiguousarray(valid, dtype=bool),
        violation=np.ascontiguousarray(violation, dtype=bool),
        checked_count=count,
        violation_count=int(np.count_nonzero(violation)),
        gate=str(gate),
    )


def _select_visible_spread_indices(tracks_yx: np.ndarray, visibility: np.ndarray, *, max_points: int) -> np.ndarray:
    """Select visible spread indices."""
    tracks = np.asarray(tracks_yx, dtype=np.float32).reshape(-1, 2)
    visible = np.flatnonzero(np.asarray(visibility, dtype=np.float32).reshape(-1) > 0.0)
    if len(visible) > 0:
        visible = visible[np.isfinite(tracks[visible]).all(axis=1)]
    cap = int(max_points)
    if cap <= 0 or len(visible) <= cap:
        return visible.astype(np.int64)
    pts = tracks[visible]
    if len(pts) == 0:
        return np.empty((0,), dtype=np.int64)
    selected_local = [0]
    min_dist2 = np.sum((pts - pts[0]) ** 2, axis=1)
    for _ in range(1, min(cap, len(pts))):
        next_local = int(np.argmax(min_dist2))
        selected_local.append(next_local)
        dist2 = np.sum((pts - pts[next_local]) ** 2, axis=1)
        min_dist2 = np.minimum(min_dist2, dist2)
    return visible[np.asarray(selected_local, dtype=np.int64)].astype(np.int64)


def _latest_tracker_arrays(result: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return the latest tracker arrays."""
    tracks = np.asarray(result.tracks_yx, dtype=np.float32)
    visibility = np.asarray(result.visibility, dtype=np.float32)
    if tracks.ndim == 4:
        tracks_latest = tracks[0, -1]
        visibility_latest = visibility[0, -1]
    elif tracks.ndim == 3:
        tracks_latest = tracks[-1]
        visibility_latest = visibility[-1]
    elif tracks.ndim == 2:
        tracks_latest = tracks
        visibility_latest = visibility
    else:
        raise ValueError(f"tracker tracks_yx must be 2D, 3D, or 4D; got {tracks.shape}")
    return (
        np.ascontiguousarray(np.asarray(tracks_latest, dtype=np.float32).reshape(-1, 2)),
        np.ascontiguousarray(np.asarray(visibility_latest, dtype=np.float32).reshape(-1)),
    )


# ---------------------------------------------------------------------------
# Main runtime: owns the worker threads and queues for
# capture -> segmentation -> tracker/pcd -> filter -> pairing -> headless capture
# ---------------------------------------------------------------------------
class MainDataProcessingDemo:
    def __init__(self, args: argparse.Namespace) -> None:
        """Initialize MainDataProcessingDemo."""
        self.args = args
        self.width, self.height = parse_profile(args.profile)
        self.lossless_max_backlog_frames = max(
            1,
            int(round(self._lossless_input_fps() * float(args.lossless_max_backlog_seconds))),
        )
        self.runtime: RealtimeCameraRuntime | None = None
        self.ray_x: np.ndarray | None = None
        self.ray_y: np.ndarray | None = None
        self.input_preview_slot: LatestSlot[FramePacket] = LatestSlot()
        self.capture_slot: LatestSlot[FramePacket] = LatestSlot()
        self.mask_slot: LatestSlot[MaskPacket] = LatestSlot()
        self.depth_profile_slot: LatestSlot[DepthProfilePacket] = LatestSlot()
        self.remote_quality_slot: LatestSlot[RemoteFfsQualityPacket] = LatestSlot()
        # Latest non-strict PCD packet; consumed only by the headless debug worker.
        self.pcd_slot: LatestSlot[MaskedPcdPacket] = LatestSlot()
        self.tracker_marker_slot: LatestSlot[TrackerMarkerPacket] = LatestSlot()
        self.paired_render_slot: LatestSlot[PairedRenderPacket] = LatestSlot()
        self.lossless_frame_queue: OrderedPacketQueue[FramePacket] = OrderedPacketQueue(
            name="frame",
            max_backlog_frames=self.lossless_max_backlog_frames,
        )
        self.lossless_pcd_mask_queue: OrderedPacketQueue[MaskPacket] = OrderedPacketQueue(
            name="mask-pcd",
            max_backlog_frames=self.lossless_max_backlog_frames,
        )
        self.lossless_tracker_mask_queue: OrderedPacketQueue[MaskPacket] = OrderedPacketQueue(
            name="mask-tracker",
            max_backlog_frames=self.lossless_max_backlog_frames,
        )
        self.lossless_pair_output_queue: OrderedPacketQueue[PairedBuildResult] = OrderedPacketQueue(
            name="pair-output",
            max_backlog_frames=self.lossless_max_backlog_frames,
        )
        self.same_seq_pairer = SameSeqPairer(max_backlog_frames=self.lossless_max_backlog_frames)
        self._lossless_pairer_lock = threading.Lock()
        self._lossless_publish_condition = threading.Condition()
        self._lossless_next_publish_seq = 0
        self._startup_hold_s = 0.0
        self.stop_event = threading.Event()
        self._lossless_capture_done = threading.Event()
        self._lossless_processing_done = threading.Event()
        self._lossless_first_pair_published = threading.Event()
        self._lossless_pipeline_active = False
        self._threads: list[threading.Thread] = []
        self.capture_stats = StageStats()
        self.seg_stats = StageStats()
        self.depth_stats = StageStats()
        self.remote_quality_stats = StageStats()
        self.pcd_stats = StageStats()
        self.tracker_stats = StageStats()
        self.filter_submit_stats = StageStats()
        self.filter_output_stats = StageStats()
        self.filter_worker: Any | None = None
        self._filter_submit_skip_count = 0
        self._last_filter_output_seq_recorded = -1
        controller_filter_min_cap = int(args.filter_min_cap)
        if self._lossless_enabled():
            controller_filter_min_cap = min(controller_filter_min_cap, DEFAULT_LOSSLESS_CONTROLLER_FILTER_MIN_CAP)
        self.object_filter_budget = FilterBudgetController(
            target_ms=max(0.0, float(args.filter_budget_ms)) * 0.5,
            min_cap=int(args.filter_min_cap),
            max_cap=max(int(args.filter_min_cap), int(args.object_filter_cap) if int(args.object_filter_cap) > 0 else 200_000),
            init_cap=int(args.object_filter_cap) if int(args.object_filter_cap) > 0 else 200_000,
        )
        self.controller_filter_budget = FilterBudgetController(
            target_ms=max(0.0, float(args.filter_budget_ms)) * 0.5,
            min_cap=int(controller_filter_min_cap),
            max_cap=max(int(controller_filter_min_cap), int(args.controller_filter_cap) if int(args.controller_filter_cap) > 0 else 200_000),
            init_cap=int(args.controller_filter_cap) if int(args.controller_filter_cap) > 0 else 200_000,
        )
        self._last_debug_log_s = 0.0
        self.ffs_runner: object | None = None
        self._local_ffs_lock = threading.Lock()
        self._local_ffs_depth_cache: OrderedDict[int, tuple[np.ndarray, float, float]] = OrderedDict()
        self.ir_to_color_aligner: FfsIrToColorAligner | None = None
        self._ir_to_color_aligner_key: tuple[
            tuple[int, int],
            tuple[int, int],
            tuple[float, ...],
            tuple[float, ...],
            tuple[float, ...],
        ] | None = None
        self.ffs_remote_client: Any | None = None
        self.remote_quality_client: Any | None = None
        self.recording_source: RecordedRgbdFrameSource | None = None
        self.headless_capture_writer: HeadlessCaptureWriter | None = None
        self.shape_prior_manager = self._create_shape_prior_manager()
        self._shape_prior_written = False
        self._formal_timeline_gated_frames = 0
        self._formal_timeline_metadata_written = False
        self._warmup_anchor_row_written = False
        self._formal_timeline_gate_started_s: float | None = None
        self._formal_timeline_gate_expired = False
        self.table_c2w: np.ndarray | None = None
        self.table_calibration_path: Path | None = None
        self._recording_first_frame_segmented = threading.Event()
        self._lossless_offered_frames = 0
        self._lossless_segmented_frames = 0
        self._lossless_pcd_results = 0
        self._lossless_tracker_results = 0
        self._lossless_pairs_emitted = 0
        self._tracker_query_points_yx: np.ndarray | None = None
        self._tracker_query_rgb_u8: np.ndarray | None = None
        self._tracker_query_is_object: np.ndarray | None = None
        self._tracker_query_is_controller: np.ndarray | None = None
        self._tracker_query_target_id: np.ndarray | None = None
        self._tracker_query_controller_instance_id: np.ndarray | None = None
        self._tracker_consistent_visible: np.ndarray | None = None
        self._tracker_query_alive_mask: np.ndarray | None = None
        self._tracker_query_initial_seq: int | None = None
        self._warned_remote_engine_contract = False
        self._fatal_error_lock = threading.Lock()
        self._fatal_error: FatalWorkerError | None = None

    @property
    def intrinsics(self) -> CameraIntrinsics:
        """Return the intrinsics."""
        if self.runtime is None:
            raise RuntimeError("camera runtime is not initialized")
        return self.runtime.intrinsics

    @property
    def serial(self) -> str:
        """Return the serial."""
        if self.runtime is None:
            return "<not-started>"
        return self.runtime.serial

    def _table_world_enabled(self) -> bool:
        """Return whether table world is enabled."""
        return self.table_c2w is not None

    def _pcd_coordinate_frame(self) -> str:
        """Return the PCD coordinate frame."""
        return TABLE_WORLD_FRAME_KIND if self._table_world_enabled() else COORDINATE_FRAME

    def _create_shape_prior_manager(self) -> shape_prior_warmup.ShapePriorWarmupManager:
        """Create the shape-prior warmup manager for the runtime."""
        enabled = bool(getattr(self.args, "shape_prior_warmup", False))
        client = None
        if enabled:
            client = shape_prior_warmup.ShapePriorLocalClient(
                case_root=Path(self.args.shape_prior_case_root),
                cuda_visible_devices=str(
                    getattr(
                        self.args,
                        "shape_prior_warmup_cuda_visible_devices",
                        shape_prior_warmup.DEFAULT_SHAPE_PRIOR_WARMUP_CUDA_VISIBLE_DEVICES,
                    )
                ),
                object_name=str(self.args.object_prompt),
                controller_name=str(self.args.shape_prior_controller_name),
                points_npz=Path(self.args.shape_prior_points_npz),
                sam3d_root=getattr(self.args, "shape_prior_sam3d_root", None),
                sam3d_config=getattr(self.args, "shape_prior_config", None),
                sam31_device=str(self.args.device),
                reuse_sam31_model=True,
            )
            if bool(getattr(self.args, "shape_prior_prewarm_stage_workers", False)):
                client.prewarm()
        return shape_prior_warmup.ShapePriorWarmupManager(
            enabled=enabled,
            client=client,
        )

    def _shape_prior_profile(self) -> dict[str, Any]:
        """Return the shape prior profile."""
        manager = getattr(self, "shape_prior_manager", None)
        if manager is None:
            return shape_prior_warmup.default_profile(enabled=False)
        return manager.profile()

    def _shape_prior_profile_payload(self) -> dict[str, Any]:
        """Return the shape prior profile payload."""
        profile = self._shape_prior_profile()
        payload = dict(profile)
        if payload.get("input_source") is None:
            payload["input_source"] = str(getattr(self.args, "input_source", ""))
        if payload.get("depth_backend") is None:
            payload["depth_backend"] = depth_backend_label(self.args)
        if payload.get("depth_source_internal") is None:
            payload["depth_source_internal"] = str(getattr(self.args, "depth_source", ""))
        return payload

    def _write_shape_prior_profile_json(self, profile: dict[str, Any] | None = None) -> None:
        """Write shape prior profile JSON."""
        path = getattr(self.args, "shape_prior_profile_json", None)
        if path is None:
            return
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = self._shape_prior_profile_payload() if profile is None else dict(profile)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _initialize_table_calibration(self) -> None:
        """Initialize table calibration."""
        if self.args.table_calibrate is None:
            return
        if self.runtime is None:
            raise RuntimeError("camera runtime is not initialized")
        path = Path(self.args.table_calibrate)
        try:
            transforms = load_table_calibration_transforms(path, serial_numbers=[str(self.runtime.serial)])
        except TableCalibrationLoadError as exc:
            raise RuntimeError(f"Invalid table calibration for active camera {self.runtime.serial}: {exc}") from exc
        self.table_c2w = np.ascontiguousarray(transforms[0], dtype=np.float32)
        self.table_calibration_path = path
        print(
            "[table-calibrate] "
            f"path={path} serial={self.runtime.serial} pcd_coordinate_frame={TABLE_WORLD_FRAME_KIND}",
            flush=True,
        )

    def _lossless_enabled(self) -> bool:
        """Return whether lossless is enabled."""
        return bool(tracker_enabled(self.args) and self.args.pcd_mode == "masked")

    def _lossless_input_fps(self) -> float:
        """Return the lossless input FPS."""
        return float(getattr(self.args, "lossless_input_fps", DEFAULT_LOSSLESS_INPUT_FPS))

    def _reset_lossless_state(self) -> None:
        """Reset lossless state."""
        self.lossless_frame_queue.reset()
        self.lossless_pcd_mask_queue.reset()
        self.lossless_tracker_mask_queue.reset()
        self.lossless_pair_output_queue.reset()
        self.same_seq_pairer.reset()
        with self._lossless_publish_condition:
            self._lossless_next_publish_seq = 0
            self._lossless_publish_condition.notify_all()
        self._lossless_capture_done.clear()
        self._lossless_processing_done.clear()
        self._lossless_first_pair_published.clear()
        self._recording_first_frame_segmented.clear()
        self._lossless_pipeline_active = True
        self._lossless_offered_frames = 0
        self._lossless_segmented_frames = 0
        self._lossless_pcd_results = 0
        self._lossless_tracker_results = 0
        self._lossless_pairs_emitted = 0

    def _close_lossless_queues(self) -> None:
        """Close lossless queues."""
        self.lossless_frame_queue.close()
        self.lossless_pcd_mask_queue.close()
        self.lossless_tracker_mask_queue.close()
        self.lossless_pair_output_queue.close()
        self._lossless_pipeline_active = False

    def _wait_for_lossless_replay_startup_pair(self, on_wait_tick: Callable[[], None] | None = None) -> bool:
        """Wait for for lossless replay startup pair."""
        if not (
            self._lossless_enabled()
            and self.args.track_mode != "none"
            and _is_replay_input_source(str(self.args.input_source))
        ):
            return True
        while not self.stop_event.is_set():
            if self._lossless_first_pair_published.wait(timeout=0.01):
                return True
            if on_wait_tick is not None:
                on_wait_tick()
        return False

    def _build_headless_capture_metadata(self) -> dict[str, Any]:
        """Build headless capture metadata."""
        if self.runtime is None:
            raise RuntimeError("camera runtime is not initialized")
        shape_profile = self._shape_prior_profile_payload()
        replay_fps = None
        recording_fps = None
        frame_count = None
        recording_case = None
        if self.recording_source is not None:
            replay_fps = float(self.recording_source.effective_fps)
            recording_fps = float(self.recording_source.recording_fps)
            frame_count = int(self.recording_source.frame_count)
            recording_case = _repo_relative_path_text(self.recording_source.case_path)
        frame_selection_policy = (
            FAKE_LIVE_FRAME_SELECTION_POLICY if str(self.args.input_source) == INPUT_SOURCE_FAKE_LIVE else None
        )
        return {
            **runtime_metadata_identity(self.args),
            "input_source": str(self.args.input_source),
            "recording_case": recording_case,
            "replay_fps": replay_fps,
            "recording_fps": recording_fps,
            "fake_live_frame_selection_policy": frame_selection_policy,
            "recording_frame_count": frame_count,
            "color_exposure": (
                None
                if getattr(self.args, "color_exposure", None) is None
                else float(getattr(self.args, "color_exposure"))
            ),
            "color_gain": (
                None
                if getattr(self.args, "color_gain", None) is None
                else float(getattr(self.args, "color_gain"))
            ),
            "depth_source": str(self.args.depth_source),
            "depth_source_internal": str(self.args.depth_source),
            "depth_units": "meters",
            "depth_coordinate_frame": COORDINATE_FRAME,
            "depth_alignment_target": "color",
            "track_mode": str(self.args.track_mode),
            "edgetam_tracking_identities": list(active_object_id_labels(self.args).values()),
            "demo_visual_mode": str(self.args.demo_visual_mode),
            "tracker_backend": str(self.args.tracker_backend),
            "tracking_product_backend": str(
                normalize_tracking_product_backend(getattr(self.args, "tracking_product_backend", DEFAULT_TRACKING_PRODUCT_BACKEND))
            ),
            "headless_prepared_only": bool(getattr(self.args, "headless_prepared_only", False)),
            "write_input_rgb_timeline": bool(getattr(self.args, "write_input_rgb_timeline", False)),
            "phystwin_strict_output_dir": (
                None
                if getattr(self.args, "phystwin_strict_output_dir", None) is None
                else _repo_relative_path_text(self.args.phystwin_strict_output_dir)
            ),
            "compatibility_target": (
                COMPATIBILITY_TARGET_PHYSTWIN
                if tracking_product_backend_is_strict(getattr(self.args, "tracking_product_backend", DEFAULT_TRACKING_PRODUCT_BACKEND))
                else None
            ),
            "mask_backend": "edgetam",
            "depth_backend": depth_backend_label(self.args),
            "shape_prior_enabled": bool(shape_profile.get("shape_prior_enabled", False)),
            "shape_prior_status": str(
                shape_profile.get(
                    "shape_prior_status",
                    shape_prior_warmup.STATUS_DISABLED,
                )
            ),
            "shape_prior_timeout_ms": int(
                getattr(
                    self.args,
                    "shape_prior_timeout_ms",
                    shape_prior_warmup.DEFAULT_SHAPE_PRIOR_TIMEOUT_MS,
                )
            ),
            "shape_prior_warmup_cuda_visible_devices": str(
                getattr(
                    self.args,
                    "shape_prior_warmup_cuda_visible_devices",
                    shape_prior_warmup.DEFAULT_SHAPE_PRIOR_WARMUP_CUDA_VISIBLE_DEVICES,
                )
            ),
            "shape_prior_controller_name": str(
                getattr(self.args, "shape_prior_controller_name", "")
            ),
            "shape_prior_case_root": _repo_relative_path_text(
                getattr(self.args, "shape_prior_case_root", None)
            ),
            "shape_prior_points_npz": _repo_relative_path_text(
                getattr(self.args, "shape_prior_points_npz", None)
            ),
            "shape_prior_skip_route_visualizations": bool(
                getattr(self.args, "shape_prior_skip_route_visualizations", True)
            ),
            "shape_prior_source_seq": shape_profile.get("shape_prior_source_seq"),
            "shape_prior_source_time_s": shape_profile.get("shape_prior_source_time_s"),
            "shape_prior_submit_ms": float(shape_profile.get("shape_prior_submit_ms", 0.0) or 0.0),
            "first_mask_depth_pair_ms": float(shape_profile.get("first_mask_depth_pair_ms", 0.0) or 0.0),
            "first_strict_pair_ms": float(shape_profile.get("first_strict_pair_ms", 0.0) or 0.0),
            "shape_prior_depth_backend": depth_backend_label(self.args),
            "shape_prior_depth_source_internal": str(self.args.depth_source),
            "execution_mode": (
                PHYSTWIN_STRICT_EXECUTION_MODE
                if tracking_product_backend_is_strict(getattr(self.args, "tracking_product_backend", DEFAULT_TRACKING_PRODUCT_BACKEND))
                else TRACKING_PRODUCT_BACKEND_REALTIME_OVERLAY
            ),
            "tracker_query_count": int(self.args.tracker_query_count),
            "tracker_query_source": tracker_query_source(self.args) if tracker_enabled(self.args) else None,
            "tracker_marker_gate": tracker_marker_gate(self.args) if tracker_enabled(self.args) else None,
            "tracker_retire_filtered_markers": (
                tracker_retire_filtered_markers(self.args) if tracker_enabled(self.args) else None
            ),
            "tracker_marker_retirement_policy": (
                tracker_marker_retirement_policy(self.args) if tracker_enabled(self.args) else None
            ),
            "tracker_display_scope": str(self.args.tracker_display_scope),
            "tracker_sync_policy": (
                "strict_same_seq_lossless_5fps" if self._lossless_enabled() else "none"
            ),
            "lossless_input_fps": float(self._lossless_input_fps()) if self._lossless_enabled() else None,
            "lossless_max_backlog_frames": int(self.lossless_max_backlog_frames) if self._lossless_enabled() else None,
            "pcd_filter_enabled": pcd_filter_enabled(self.args),
            "pcd_filter_mode": str(self.args.pcd_filter_mode if pcd_filter_enabled(self.args) else PCD_FILTER_NONE),
            "pcd_filter_preset": getattr(self.args, "pcd_filter_preset", None),
            "saved_pcd_source": (
                headless_capture_saved_pcd_source(self.args) if headless_capture_enabled(self.args) else None
            ),
            "object_filter": str(self.args.object_filter),
            "controller_filter": str(self.args.controller_filter),
            "object_filter_keep_components": int(self.args.object_filter_keep_components),
            "controller_filter_keep_components": int(self.args.controller_filter_keep_components),
            "filter_radius_m": float(self.args.filter_radius_m),
            "filter_nb_points": int(self.args.filter_nb_points),
            "filter_min_cap": int(self.args.filter_min_cap),
            "lossless_controller_filter_min_cap": (
                int(self.controller_filter_budget.min_cap) if self._lossless_enabled() else None
            ),
            "enhanced_component_voxel_size_m": float(self.args.enhanced_component_voxel_size_m),
            "pcd_max_points": int(self.args.pcd_max_points),
            "pcd_stride": int(self.args.pcd_stride),
            "pcd_mask_erode_pixels": int(self.args.pcd_mask_erode_pixels),
            "object_pcd_mask_erode_pixels": object_pcd_mask_erode_pixels(self.args),
            "controller_pcd_mask_erode_pixels": controller_pcd_mask_erode_pixels(self.args),
            "depth_min_m": float(self.args.depth_min_m),
            "depth_max_m": float(self.args.depth_max_m),
            "serial": str(self.runtime.serial),
            "width": int(self.width),
            "height": int(self.height),
            "coordinate_frame": self._pcd_coordinate_frame(),
            "pcd_coordinate_frame": self._pcd_coordinate_frame(),
            "camera_coordinate_frame": COORDINATE_FRAME,
            "table_calibration_path": _repo_relative_path_text(self.table_calibration_path),
            "table_world_frame_kind": TABLE_WORLD_FRAME_KIND if self._table_world_enabled() else None,
            "table_z_m": TABLE_Z_M if self._table_world_enabled() else None,
            "table_z_above_direction": TABLE_Z_ABOVE_DIRECTION,
            "camera_to_world_c2w": (
                None
                if self.table_c2w is None
                else np.asarray(self.table_c2w, dtype=np.float32).reshape(4, 4).tolist()
            ),
            "world_z_diagnostic_thresholds_m": [
                float(value) for value in DEFAULT_TABLE_Z_DIAGNOSTIC_THRESHOLDS_M
            ],
            "table_z_filter_enabled": bool(self.args.enable_table_z_filter),
            "table_z_filter_threshold_m": float(self.args.table_z_filter_threshold_m),
            "table_z_filter_classes": str(self.args.table_z_filter_classes),
            "intrinsics": {
                "fx": float(self.runtime.intrinsics.fx),
                "fy": float(self.runtime.intrinsics.fy),
                "cx": float(self.runtime.intrinsics.cx),
                "cy": float(self.runtime.intrinsics.cy),
            },
            "k_color": np.asarray(self.runtime.k_color, dtype=np.float32).tolist(),
        }

    def _fatal_error_snapshot(self) -> FatalWorkerError | None:
        """Return the fatal error snapshot."""
        with self._fatal_error_lock:
            return self._fatal_error

    def _record_fatal_worker_error(self, stage: str, exc: BaseException) -> FatalWorkerError:
        """Record fatal worker error."""
        fatal = FatalWorkerError(stage=str(stage), exc_type=type(exc).__name__, message=str(exc))
        should_notify = False
        with self._fatal_error_lock:
            if self._fatal_error is None:
                self._fatal_error = fatal
                should_notify = True
            else:
                fatal = self._fatal_error
        if should_notify:
            print(f"[FATAL] {fatal.log_message()}", flush=True)
            self.stop_event.set()
        return fatal

    # ------------------------------------------------------------------
    # Lifecycle: run / stop, worker startup, headless loop
    # ------------------------------------------------------------------
    def run(self) -> int:
        """Run MainDataProcessingDemo."""
        main_warmup.prepare_runtime_services_and_source(
            self,
            pcd_filter_enabled=pcd_filter_enabled,
            is_replay_input_source=_is_replay_input_source,
            recording_source_cls=RecordedRgbdFrameSource,
            start_realsense_pipeline=_start_realsense_pipeline,
            fake_live_input_source=INPUT_SOURCE_FAKE_LIVE,
            fake_live_frame_selection_policy=FAKE_LIVE_FRAME_SELECTION_POLICY,
        )
        try:
            main_warmup.prepare_runtime_projection_and_capture(
                self,
                headless_capture_enabled=headless_capture_enabled,
                headless_capture_writer_cls=HeadlessCaptureWriter,
            )
            self._run_headless()
            self._finalize_headless_tracking_product()
        finally:
            self.stop()
        return 2 if self._fatal_error_snapshot() is not None else 0

    def _finalize_headless_tracking_product(self) -> None:
        """Finalize headless tracking product."""
        if not tracking_product_backend_is_strict(getattr(self.args, "tracking_product_backend", DEFAULT_TRACKING_PRODUCT_BACKEND)):
            return
        if self._fatal_error_snapshot() is not None:
            return
        if self.headless_capture_writer is None:
            raise RuntimeError("phystwin-strict-tracking requires an initialized headless capture writer")
        output_dir = (
            Path(self.args.phystwin_strict_output_dir)
            if getattr(self.args, "phystwin_strict_output_dir", None) is not None
            else self.headless_capture_writer.output_dir / "phystwin_like"
        )
        print(f"[phystwin-strict] finalizing output_dir={output_dir}", flush=True)
        manifest = finalize_headless_capture(self.headless_capture_writer.output_dir, output_dir=output_dir)
        self.headless_capture_writer.update_metadata(
            {
                "phystwin_strict_output_dir": _repo_relative_path_text(output_dir),
                "phystwin_strict_manifest": _repo_relative_path_text(output_dir / "manifest.json"),
                "phystwin_strict_frame_count": int(manifest.get("frame_count", 0)),
                "phystwin_strict_query_count": int(manifest.get("query_count", 0)),
            }
        )
        print(
            "[phystwin-strict] "
            f"frames={manifest.get('frame_count')} queries={manifest.get('query_count')} "
            f"manifest={output_dir / 'manifest.json'}",
            flush=True,
        )

    def stop(self) -> None:
        """Stop MainDataProcessingDemo."""
        self.stop_event.set()
        self._close_lossless_queues()
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
        self.recording_source = None
        if self.ffs_remote_client is not None:
            self.ffs_remote_client.close()
            self.ffs_remote_client = None
        if self.remote_quality_client is not None:
            self.remote_quality_client.close()
        self.remote_quality_client = None
        self._run_deferred_shape_prior_after_teardown()
        self._write_shape_prior_profile_json()
        if (
            self.headless_capture_writer is not None
            and self._formal_timeline_gated_frames > 0
            and not self._formal_timeline_metadata_written
        ):
            # The run ended while formal rows were still gated on the shape
            # prior: frames.jsonl holds only the warmup row and can never be
            # chunked. Mark the capture and route the failure through the
            # existing fatal-error path so the process exits nonzero.
            error_message = (
                "run ended while formal chunk rows were still gated on "
                f"the shape prior ({self._formal_timeline_gated_frames} frames "
                "withheld); the capture has no formal timeline and cannot be "
                "chunked."
            )
            self.headless_capture_writer.update_metadata(
                {
                    "formal_timeline_incomplete": True,
                    "formal_timeline_gated_frame_count": int(
                        self._formal_timeline_gated_frames
                    ),
                }
            )
            self._record_fatal_worker_error(
                "formal chunk timeline",
                RuntimeError(error_message),
            )
        self.headless_capture_writer = None
        if self.filter_worker is not None:
            self.filter_worker.stop()
            self.filter_worker = None
        with self._local_ffs_lock:
            self._local_ffs_depth_cache.clear()

    def _create_ffs_runner(self) -> object:
        """Create the configured FFS runner."""
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
        """Return the get IR to color aligner."""
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
        """Start threads."""
        if self._lossless_enabled():
            self._reset_lossless_state()
        workers: list[tuple[str, Callable[[], None]]] = [("capture", self._capture_worker)]
        if self.args.track_mode != "none":
            workers.append(("seg", self._seg_worker))
        if self._lossless_enabled():
            workers.append(("pcd", self._lossless_pcd_worker))
            workers.append(("tracker", self._lossless_tracker_worker))
            workers.append(("pair-output", self._lossless_pair_output_worker))
        elif tracker_enabled(self.args):
            workers.append(("tracker", self._tracker_worker))
        if self.args.pcd_mode == "masked":
            if not tracker_enabled(self.args):
                workers.append(("pcd", self._pcd_worker))
        elif self.args.depth_source in {"ffs", "ffs_remote"}:
            workers.append(("depth", self._depth_profile_worker))
        if self.args.enable_remote_ffs_quality:
            workers.append(("remote-quality", self._remote_ffs_quality_worker))
        if self.args.debug:
            workers.append(("debug", self._headless_debug_worker))

        def worker_runner(worker_name: str, worker_target: Callable[[], None]) -> Callable[[], None]:
            """Return the worker runner."""
            def run_worker() -> None:
                """Run worker."""
                try:
                    worker_target()
                except Exception as exc:
                    if not self.stop_event.is_set():
                        self._record_fatal_worker_error(f"{worker_name} worker", exc)

            return run_worker

        for name, target in workers:
            thread = threading.Thread(target=worker_runner(name, target), name=f"masked-edgetam-{name}", daemon=True)
            thread.start()
            self._threads.append(thread)

    def _run_headless(self) -> None:
        """Run headless."""
        self._start_threads()
        started_s: float | None = None
        try:
            while not self.stop_event.is_set():
                if self._lossless_enabled():
                    if self._lossless_processing_done.is_set():
                        self.stop_event.set()
                        break
                    time.sleep(0.05)
                    continue
                if self.args.duration_s > 0:
                    now_s = time.perf_counter()
                    if self.headless_capture_writer is not None:
                        # --duration-s budgets the FORMAL capture: don't start
                        # the clock before the first row, nor while rows are
                        # gated on the shape-prior wait.
                        if (
                            self.headless_capture_writer.saved_pcd_count <= 0
                            or self._headless_product_rows_gated()
                        ):
                            time.sleep(0.05)
                            continue
                        if started_s is None:
                            started_s = now_s
                    elif started_s is None:
                        started_s = now_s
                    if started_s is not None and now_s - started_s >= float(self.args.duration_s):
                        self.stop_event.set()
                        break
                time.sleep(0.05)
        except KeyboardInterrupt:
            self.stop_event.set()

    # ------------------------------------------------------------------
    # Capture workers (live RealSense / fake-live recording replay)
    # ------------------------------------------------------------------
    def _publish_input_preview_packet(self, packet: FramePacket, *, record_s: float | None = None) -> None:
        """Publish input preview packet."""
        self.input_preview_slot.put(packet)
        should_write_timeline = _is_replay_input_source(str(self.args.input_source)) or bool(
            getattr(self.args, "write_input_rgb_timeline", False)
        )
        if self.headless_capture_writer is not None and should_write_timeline:
            self.headless_capture_writer.write_input_frame(packet)

    def _publish_capture_packet(
        self,
        packet: FramePacket,
        *,
        record_s: float | None = None,
        write_input_timeline: bool = True,
    ) -> None:
        """Publish capture packet."""
        if bool(write_input_timeline):
            self._publish_input_preview_packet(packet, record_s=record_s)
        self.capture_slot.put(packet)
        if self._lossless_enabled():
            if self.lossless_frame_queue.put_wait(packet, stop_event=self.stop_event) <= 0:
                return
            self._lossless_offered_frames += 1
        self.capture_stats.record(packet.receive_perf_s if record_s is None else float(record_s))

    def _capture_recording_worker(self) -> None:
        """Return the capture recording worker."""
        assert self.recording_source is not None
        source = self.recording_source
        fake_live_clock = str(self.args.input_source) == INPUT_SOURCE_FAKE_LIVE
        if self._lossless_enabled():
            frame_period_s = 1.0 / self._lossless_input_fps()
        else:
            frame_period_s = 1.0 / float(source.effective_fps)
        try:
            first_packet = source.read_packet(seq=0)
        except Exception as exc:
            if not self.stop_event.is_set():
                self._record_fatal_worker_error("recording replay", exc)
            return
        camera_start_s = float(first_packet.receive_perf_s)
        preview_seq = 0
        preview_tick = 1
        last_preview_source_index = -1

        def preview_from_packet(packet: FramePacket, *, seq: int) -> FramePacket:
            """Return the preview from packet."""
            return replace(
                packet,
                seq=int(seq),
                depth_u16=None,
                ir_left_u8=None,
                ir_right_u8=None,
                k_ir_left=None,
                t_ir_left_to_color=None,
                ir_baseline_m=0.0,
            )

        def read_preview_packet(
            *,
            seq: int,
            source_index: int,
            wait_ms: float = 0.0,
        ) -> FramePacket:
            """Read preview packet."""
            reader = getattr(source, "read_preview_packet", None)
            if callable(reader):
                return reader(seq=int(seq), frame_index=int(source_index), wait_ms=float(wait_ms))
            packet = source.read_packet(seq=int(seq), frame_index=int(source_index), wait_ms=float(wait_ms))
            return preview_from_packet(packet, seq=int(seq))

        def publish_preview_packet(packet: FramePacket) -> None:
            """Publish preview packet."""
            nonlocal preview_seq, last_preview_source_index
            self._publish_input_preview_packet(packet, record_s=packet.receive_perf_s)
            preview_seq += 1
            if packet.source_frame_index is not None:
                last_preview_source_index = max(last_preview_source_index, int(packet.source_frame_index))

        def publish_preview_source_index(*, source_index: int, wait_ms: float = 0.0) -> None:
            """Publish preview source index."""
            nonlocal preview_seq, last_preview_source_index
            if int(source_index) <= int(last_preview_source_index):
                return
            packet = read_preview_packet(seq=preview_seq, source_index=int(source_index), wait_ms=float(wait_ms))
            publish_preview_packet(packet)

        def publish_due_fake_live_previews() -> bool:
            """Publish due fake live previews."""
            nonlocal preview_tick
            if not fake_live_clock:
                return True
            now_s = time.perf_counter()
            while not self.stop_event.is_set():
                source_elapsed_s = float(preview_tick) * frame_period_s
                target_s = camera_start_s + source_elapsed_s
                if target_s > now_s:
                    break
                source_index = source.source_index_for_recording_elapsed_s(source_elapsed_s)
                preview_tick += 1
                if source_index <= last_preview_source_index:
                    if last_preview_source_index >= source.frame_count - 1:
                        break
                    continue
                try:
                    publish_preview_source_index(source_index=source_index)
                except Exception as exc:
                    if not self.stop_event.is_set():
                        self._record_fatal_worker_error("recording replay preview", exc)
                    return False
                if source_index >= source.frame_count - 1:
                    break
            return True

        if fake_live_clock:
            publish_preview_packet(preview_from_packet(first_packet, seq=preview_seq))
            self._publish_capture_packet(
                first_packet,
                record_s=first_packet.receive_perf_s,
                write_input_timeline=False,
            )
        else:
            self._publish_capture_packet(first_packet, record_s=first_packet.receive_perf_s)
        if source.frame_count <= 1:
            if self._lossless_enabled():
                self._lossless_capture_done.set()
                self.lossless_frame_queue.close()
            else:
                self.stop_event.set()
            return
        if self.args.track_mode != "none":
            while not self.stop_event.is_set():
                if self._recording_first_frame_segmented.wait(timeout=0.01):
                    break
                if not publish_due_fake_live_previews():
                    return
            if self.stop_event.is_set():
                return
        if not self._wait_for_lossless_replay_startup_pair(on_wait_tick=publish_due_fake_live_previews):
            return
        gate_done_s = time.perf_counter()
        if not publish_due_fake_live_previews():
            return
        self._startup_hold_s = max(0.0, float(gate_done_s - camera_start_s))
        if self.headless_capture_writer is not None:
            self.headless_capture_writer.update_metadata({"startup_hold_s": float(self._startup_hold_s)})
        replay_start_s = gate_done_s
        runtime_seq = 1
        if fake_live_clock:
            output_tick = max(1, int(preview_tick))
            last_source_index = max(0, int(last_preview_source_index))
            while not self.stop_event.is_set():
                source_elapsed_s = float(output_tick) * frame_period_s
                if (
                    self._lossless_enabled()
                    and float(self.args.duration_s) > 0.0
                    and source_elapsed_s >= float(self.args.duration_s)
                ):
                    break
                source_index = source.source_index_for_recording_elapsed_s(source_elapsed_s)
                output_tick += 1
                if source_index <= last_source_index:
                    if last_source_index >= source.frame_count - 1:
                        break
                    continue
                wait_start_s = time.perf_counter()
                target_s = camera_start_s + source_elapsed_s
                wait_s = target_s - wait_start_s
                if wait_s > 0.0 and self.stop_event.wait(wait_s):
                    break
                wait_done_s = time.perf_counter()
                try:
                    packet = source.read_packet(
                        seq=runtime_seq,
                        frame_index=source_index,
                        wait_ms=_elapsed_ms(wait_start_s, wait_done_s),
                    )
                except Exception as exc:
                    if not self.stop_event.is_set():
                        self._record_fatal_worker_error("recording replay", exc)
                    break
                publish_preview_packet(preview_from_packet(packet, seq=preview_seq))
                self._publish_capture_packet(
                    packet,
                    record_s=packet.receive_perf_s,
                    write_input_timeline=False,
                )
                runtime_seq += 1
                last_source_index = source_index
                if last_source_index >= source.frame_count - 1:
                    break
        else:
            for source_index in range(1, source.frame_count):
                if self.stop_event.is_set():
                    break
                if (
                    self._lossless_enabled()
                    and float(self.args.duration_s) > 0.0
                    and float(runtime_seq) * frame_period_s >= float(self.args.duration_s)
                ):
                    break
                wait_start_s = time.perf_counter()
                target_s = replay_start_s + (float(runtime_seq) * frame_period_s)
                wait_s = target_s - wait_start_s
                if wait_s > 0.0 and self.stop_event.wait(wait_s):
                    break
                wait_done_s = time.perf_counter()
                try:
                    packet = source.read_packet(
                        seq=runtime_seq,
                        frame_index=source_index,
                        wait_ms=_elapsed_ms(wait_start_s, wait_done_s),
                    )
                except Exception as exc:
                    if not self.stop_event.is_set():
                        self._record_fatal_worker_error("recording replay", exc)
                    break
                self._publish_capture_packet(packet, record_s=packet.receive_perf_s)
                runtime_seq += 1
        if self._lossless_enabled():
            self._lossless_capture_done.set()
            self.lossless_frame_queue.close()
        else:
            self.stop_event.set()

    def _capture_worker(self) -> None:
        """Return the capture worker."""
        assert self.runtime is not None
        if _is_replay_input_source(str(self.args.input_source)):
            self._capture_recording_worker()
            return
        raw_seq = 0
        output_seq = 0
        live_sampler = (
            LiveLatestFrameSampler(self._lossless_input_fps())
            if self._lossless_enabled()
            else None
        )
        pipeline = self.runtime.pipeline
        align = self.runtime.align

        def publish_output_packet(packet: FramePacket, *, record_s: float) -> None:
            """Publish one live output packet with contiguous demo sequencing."""
            nonlocal output_seq
            output_packet = replace(packet, seq=output_seq)
            self._publish_capture_packet(output_packet, record_s=float(record_s))
            output_seq += 1

        while not self.stop_event.is_set():
            wait_start_s = time.perf_counter()
            try:
                frames = pipeline.wait_for_frames()
            except Exception as exc:
                if not self.stop_event.is_set():
                    self._record_fatal_worker_error("RealSense capture", exc)
                break
            receive_perf_s = time.perf_counter()
            published_sample_before_current = False
            if live_sampler is not None:
                due_sample = live_sampler.pop_due(now_s=receive_perf_s)
                if due_sample is not None:
                    due_packet, sample_s = due_sample
                    publish_output_packet(due_packet, record_s=sample_s)
                    published_sample_before_current = True
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
                seq=raw_seq,
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
            raw_seq += 1
            if live_sampler is None:
                publish_output_packet(packet, record_s=copy_done_s)
                continue
            if output_seq == 0:
                publish_output_packet(packet, record_s=copy_done_s)
                if self.args.track_mode != "none":
                    while not self.stop_event.is_set():
                        if self._recording_first_frame_segmented.wait(timeout=0.01):
                            break
                    if self.stop_event.is_set():
                        break
                live_sampler.start(first_publish_s=time.perf_counter())
                continue
            live_sampler.put_latest(packet)
            if not published_sample_before_current:
                due_sample = live_sampler.pop_due(now_s=copy_done_s)
                if due_sample is not None:
                    due_packet, sample_s = due_sample
                    publish_output_packet(due_packet, record_s=sample_s)
        if self._lossless_enabled():
            self._lossless_capture_done.set()
            self.lossless_frame_queue.close()

    # ------------------------------------------------------------------
    # Segmentation worker: EdgeTAM model init + streaming mask loop
    # ------------------------------------------------------------------
    def _init_hf_model(self) -> tuple[Any, Any, Any, Any, Any]:
        """Return the init HF model."""
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
            **runtime_metadata_identity(self.args),
            "edge_model": self.args.model_id,
            "demo_preset": self.args.demo_preset,
            "compile_mode": self.args.compile_mode,
            "applied_targets": compile_metadata.get("applied_targets", []),
            "dtype": self.args.dtype,
            "inference_device": self.args.device,
            "inference_state_device": self.args.device,
            "video_storage_device": self.args.device,
            "frame_by_frame_streaming": True,
            "edgetam_live_session_keep_frames": int(self.args.edgetam_live_session_keep_frames),
            "offline_video_input_used": _is_replay_input_source(str(self.args.input_source)),
            "input_source": self.args.input_source,
            "demo_visual_mode": str(self.args.demo_visual_mode),
            "recording_case": (
                _repo_relative_path_text(self.args.recording_case) if _is_replay_input_source(str(self.args.input_source)) else None
            ),
            "replay_fps": (
                self.recording_source.effective_fps
                if _is_replay_input_source(str(self.args.input_source)) and self.recording_source is not None
                else None
            ),
            "recording_fps": (
                self.recording_source.recording_fps
                if _is_replay_input_source(str(self.args.input_source)) and self.recording_source is not None
                else None
            ),
            "fake_live_frame_selection_policy": (
                FAKE_LIVE_FRAME_SELECTION_POLICY if str(self.args.input_source) == INPUT_SOURCE_FAKE_LIVE else None
            ),
            "track_mode": self.args.track_mode,
            "edgetam_tracking_identities": list(active_object_id_labels(self.args).values()),
            "depth_source": self.args.depth_source,
            "depth_source_internal": str(self.args.depth_source),
            "depth_units": "meters",
            "depth_coordinate_frame": COORDINATE_FRAME,
            "depth_alignment_target": "color",
            "local_ffs_depth_cache_frames": (
                DEFAULT_LOCAL_FFS_DEPTH_CACHE_FRAMES if self.args.depth_source == "ffs" else None
            ),
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
            "pcd_coordinate_frame": self._pcd_coordinate_frame(),
            "camera_coordinate_frame": COORDINATE_FRAME,
            "table_calibration_path": _repo_relative_path_text(self.table_calibration_path),
            "table_world_frame_kind": TABLE_WORLD_FRAME_KIND if self._table_world_enabled() else None,
            "table_z_m": TABLE_Z_M if self._table_world_enabled() else None,
            "table_z_above_direction": TABLE_Z_ABOVE_DIRECTION,
            "camera_to_world_c2w": (
                None
                if self.table_c2w is None
                else np.asarray(self.table_c2w, dtype=np.float32).reshape(4, 4).tolist()
            ),
            "pcd_max_points": int(self.args.pcd_max_points),
            "pcd_stride": int(self.args.pcd_stride),
            "pcd_mask_erode_pixels": int(self.args.pcd_mask_erode_pixels),
            "object_pcd_mask_erode_pixels": object_pcd_mask_erode_pixels(self.args),
            "controller_pcd_mask_erode_pixels": controller_pcd_mask_erode_pixels(self.args),
            "pcd_filter_enabled": pcd_filter_enabled(self.args),
            "pcd_filter_mode": self.args.pcd_filter_mode if pcd_filter_enabled(self.args) else PCD_FILTER_NONE,
            "pcd_filter_preset": getattr(self.args, "pcd_filter_preset", None),
            "world_z_diagnostic_thresholds_m": [
                float(value) for value in DEFAULT_TABLE_Z_DIAGNOSTIC_THRESHOLDS_M
            ],
            "table_z_filter_enabled": bool(self.args.enable_table_z_filter),
            "table_z_filter_threshold_m": float(self.args.table_z_filter_threshold_m),
            "table_z_filter_classes": str(self.args.table_z_filter_classes),
            "headless_capture_enabled": headless_capture_enabled(self.args),
            "headless_prepared_only": bool(getattr(self.args, "headless_prepared_only", False)),
            "headless_capture_dir": (
                _repo_relative_path_text(self.args.headless_capture_dir) if headless_capture_enabled(self.args) else None
            ),
            "saved_pcd_source": (
                headless_capture_saved_pcd_source(self.args) if headless_capture_enabled(self.args) else None
            ),
            "object_filter": self.args.object_filter,
            "controller_filter": self.args.controller_filter,
            "object_filter_cap": int(self.args.object_filter_cap),
            "controller_filter_cap": int(self.args.controller_filter_cap),
            "object_filter_keep_components": int(self.args.object_filter_keep_components),
            "controller_filter_keep_components": int(self.args.controller_filter_keep_components),
            "object_filter_min_retain_ratio": float(DEFAULT_OBJECT_FILTER_MIN_RETAIN_RATIO),
            "controller_filter_min_retain_ratio": float(DEFAULT_CONTROLLER_FILTER_MIN_RETAIN_RATIO),
            "object_filter_min_raw_retain_ratio": float(DEFAULT_OBJECT_FILTER_MIN_RAW_RETAIN_RATIO),
            "controller_filter_min_raw_retain_ratio": float(DEFAULT_CONTROLLER_FILTER_MIN_RAW_RETAIN_RATIO),
            "filter_every_n": int(self.args.filter_every_n),
            "filter_max_age_frames": int(self.args.filter_max_age_frames),
            "filter_budget_ms": float(self.args.filter_budget_ms),
            "filter_min_cap": int(self.args.filter_min_cap),
            "lossless_controller_filter_min_cap": (
                int(self.controller_filter_budget.min_cap) if self._lossless_enabled() else None
            ),
            "tracker_backend": str(self.args.tracker_backend),
            "tracking_product_backend": str(
                normalize_tracking_product_backend(getattr(self.args, "tracking_product_backend", DEFAULT_TRACKING_PRODUCT_BACKEND))
            ),
            "phystwin_strict_output_dir": (
                None
                if getattr(self.args, "phystwin_strict_output_dir", None) is None
                else _repo_relative_path_text(self.args.phystwin_strict_output_dir)
            ),
            "compatibility_target": (
                COMPATIBILITY_TARGET_PHYSTWIN
                if tracking_product_backend_is_strict(getattr(self.args, "tracking_product_backend", DEFAULT_TRACKING_PRODUCT_BACKEND))
                else None
            ),
            "mask_backend": "edgetam",
            "depth_backend": depth_backend_label(self.args),
            "execution_mode": (
                PHYSTWIN_STRICT_EXECUTION_MODE
                if tracking_product_backend_is_strict(getattr(self.args, "tracking_product_backend", DEFAULT_TRACKING_PRODUCT_BACKEND))
                else TRACKING_PRODUCT_BACKEND_REALTIME_OVERLAY
            ),
            "tracker_device": str(self.args.tracker_device),
            "tracker_query_count": int(self.args.tracker_query_count),
            "tracker_query_source": tracker_query_source(self.args) if tracker_enabled(self.args) else None,
            "tracker_marker_gate": tracker_marker_gate(self.args) if tracker_enabled(self.args) else None,
            "tracker_retire_filtered_markers": (
                tracker_retire_filtered_markers(self.args) if tracker_enabled(self.args) else None
            ),
            "tracker_marker_retirement_policy": (
                tracker_marker_retirement_policy(self.args) if tracker_enabled(self.args) else None
            ),
            "tracker_display_scope": str(self.args.tracker_display_scope),
            "tracker_overlay_max_points": int(self.args.tracker_overlay_max_points),
            "tracker_marker_point_size": float(self.args.tracker_marker_point_size),
            "tracker_strict_same_seq_render": bool(tracker_enabled(self.args) and self.args.pcd_mode == "masked"),
            "tracker_visualization_mode": (
                "phystwin_rainbow_identity_3d_lift" if tracker_enabled(self.args) else "none"
            ),
            "tracker_sync_policy": (
                "strict_same_seq_lossless_5fps" if tracker_enabled(self.args) and self.args.pcd_mode == "masked" else "none"
            ),
            "lossless_input_fps": (
                float(self._lossless_input_fps()) if tracker_enabled(self.args) and self.args.pcd_mode == "masked" else None
            ),
            "lossless_max_backlog_frames": (
                int(self.lossless_max_backlog_frames) if tracker_enabled(self.args) and self.args.pcd_mode == "masked" else None
            ),
            "query_display_policy": "visible_3d_lifted_all" if tracker_enabled(self.args) else "none",
            "query_color_mode": "phystwin_rainbow_identity" if tracker_enabled(self.args) else "none",
            "tracker_lift_mask_erode_pixels": min(
                object_pcd_mask_erode_pixels(self.args),
                controller_pcd_mask_erode_pixels(self.args),
            ),
            "tapnet_repo_dir": str(self.args.tapnet_repo_dir),
            "tapnextpp_checkpoint": str(self.args.tapnextpp_checkpoint),
            "tapnextpp_image_size": str(self.args.tapnextpp_image_size),
            "tapnextpp_autocast_dtype": str(self.args.tapnextpp_autocast_dtype),
            "tapnextpp_compile": bool(self.args.tapnextpp_compile),
            "tapnextpp_fast_postprocess": bool(self.args.tapnextpp_fast_postprocess),
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
        """Return the seg worker."""
        try:
            warmup = main_warmup.prepare_segmentation_warmup(
                self,
                repo_root=REPO_ROOT,
            )
            first_frame = warmup.first_frame
            if first_frame is None:
                return
            initial_masks = warmup.initial_masks
            if initial_masks is None:
                raise RuntimeError("segmentation warmup did not produce frame-0 masks")
            session = warmup.hf_stream.EdgeTamVideoInferenceSession(
                video=None,
                video_height=int(first_frame.color_bgr.shape[0]),
                video_width=int(first_frame.color_bgr.shape[1]),
                inference_device=self.args.device,
                inference_state_device=self.args.device,
                video_storage_device=self.args.device,
                dtype=warmup.dtype,
            )
            with warmup.torch_module.inference_mode():
                first_packet = self._run_segmentation_frame(
                    hf_stream=warmup.hf_stream,
                    torch_module=warmup.torch_module,
                    dtype=warmup.dtype,
                    model=warmup.model,
                    processor=warmup.processor,
                    session=session,
                    frame=first_frame,
                    initial_masks=initial_masks,
                    add_prompt=True,
                )
                self._publish_mask_packet(first_packet)
                self.seg_stats.record(first_packet.process_done_perf_s)
                if self._lossless_enabled() or _is_replay_input_source(str(self.args.input_source)):
                    self._recording_first_frame_segmented.set()
                last_seq = first_frame.seq
                while not self.stop_event.is_set():
                    if self._lossless_enabled():
                        frame = self.lossless_frame_queue.get(stop_event=self.stop_event)
                        if frame is None:
                            break
                    else:
                        frame = self.capture_slot.get_latest_after(last_seq)
                        if frame is None:
                            time.sleep(0.001)
                            continue
                    last_seq = frame.seq
                    try:
                        packet = self._run_segmentation_frame(
                            hf_stream=warmup.hf_stream,
                            torch_module=warmup.torch_module,
                            dtype=warmup.dtype,
                            model=warmup.model,
                            processor=warmup.processor,
                            session=session,
                            frame=frame,
                            initial_masks=initial_masks,
                            add_prompt=False,
                        )
                    except Exception as exc:
                        self._record_fatal_worker_error("EdgeTAM segmentation", exc)
                        break
                    self._publish_mask_packet(packet)
                    self.seg_stats.record(packet.process_done_perf_s)
                if self._lossless_enabled():
                    self.lossless_pcd_mask_queue.close()
                    self.lossless_tracker_mask_queue.close()
        except Exception as exc:
            if not self.stop_event.is_set():
                self._record_fatal_worker_error("segmentation worker", exc)
            if self._lossless_enabled():
                self.lossless_pcd_mask_queue.close()
                self.lossless_tracker_mask_queue.close()

    # ------------------------------------------------------------------
    # Tracker: query seeding, alive masks, marker packets, worker loop
    # ------------------------------------------------------------------
    def _build_tracker_adapter(self) -> Any:
        """Build tracker adapter."""
        config = PointTrackerAdapterConfig(
            backend=str(self.args.tracker_backend),
            device=str(self.args.tracker_device),
            tapnet_repo_dir=str(self.args.tapnet_repo_dir),
            tapnextpp_checkpoint=str(self.args.tapnextpp_checkpoint),
            tapnextpp_image_size=str(self.args.tapnextpp_image_size),
            tapnextpp_autocast_dtype=str(self.args.tapnextpp_autocast_dtype),
            tapnextpp_compile=bool(self.args.tapnextpp_compile),
            tapnextpp_fast_postprocess=bool(self.args.tapnextpp_fast_postprocess),
        )
        adapter = build_point_tracker_adapter_factory(config)(0)
        availability = adapter.availability()
        if not availability.available:
            raise RuntimeError(availability.reason)
        return adapter

    def _ensure_tracker_queries(self, mask_packet: MaskPacket, adapter: Any) -> np.ndarray | None:
        """Return the ensure tracker queries."""
        if self._tracker_query_points_yx is not None:
            return self._tracker_query_points_yx
        query_source = tracker_query_source(self.args)
        if query_source == TRACKER_QUERY_SOURCE_PCD_FILTER_RESIDUAL:
            object_query_mask, controller_query_mask = self._tracker_pcd_filter_residual_masks(mask_packet)
            union_mask = np.logical_or(object_query_mask, controller_query_mask)
        else:
            object_query_mask = np.asarray(mask_packet.object_mask, dtype=bool)
            controller_query_mask = np.asarray(mask_packet.controller_mask, dtype=bool)
            union_mask = _tracker_union_mask(mask_packet)
        object_pixels = int(np.count_nonzero(object_query_mask))
        controller_pixels = int(np.count_nonzero(controller_query_mask))
        union_pixels = int(np.count_nonzero(union_mask))
        requested = int(self.args.tracker_query_count)
        if query_source == TRACKER_QUERY_SOURCE_PCD_FILTER_RESIDUAL:
            if union_pixels <= 0:
                raise RuntimeError(
                    "pcd_filter_residual query source produced no residual query candidates "
                    f"seq={mask_packet.seq} object={object_pixels} controller={controller_pixels}"
                )
            if requested > 0 and union_pixels < requested:
                raise RuntimeError(
                    "not enough residual query candidates for TAPNext++ initialization: "
                    f"requested={requested} residual={union_pixels} object={object_pixels} controller={controller_pixels}"
                )
        elif object_pixels <= 0 or controller_pixels <= 0 or union_pixels <= 0:
            return None
        query_points = sample_phystwin_dense(
            union_mask,
            seed=int(self.args.tracker_seed),
            camera_idx=0,
            torch_device="cpu",
        )
        if requested > 0 and len(query_points) > requested:
            query_points = np.ascontiguousarray(query_points[:requested], dtype=np.float32)
        if len(query_points) == 0:
            if query_source == TRACKER_QUERY_SOURCE_PCD_FILTER_RESIDUAL:
                raise RuntimeError("pcd_filter_residual query source produced no sampled query points")
            return None
        hand_a_query_mask = _mask_packet_hand_a_mask(mask_packet) & controller_query_mask
        hand_b_query_mask = _mask_packet_hand_b_mask(mask_packet) & controller_query_mask
        query_is_object, query_is_controller, query_target_id, query_controller_instance_id = _classify_query_targets_yx(
            query_points,
            object_mask=object_query_mask,
            hand_a_mask=hand_a_query_mask,
            hand_b_mask=hand_b_query_mask,
            controller_mask=controller_query_mask,
        )
        adapter.initialize([], query_points)
        self._tracker_query_points_yx = np.ascontiguousarray(query_points, dtype=np.float32)
        self._tracker_query_rgb_u8 = query_rainbow_colors_from_points_yx_rgb_u8(query_points)
        self._tracker_query_is_object = np.ascontiguousarray(query_is_object, dtype=bool)
        self._tracker_query_is_controller = np.ascontiguousarray(query_is_controller, dtype=bool)
        self._tracker_query_target_id = np.ascontiguousarray(query_target_id, dtype=np.int64)
        self._tracker_query_controller_instance_id = np.ascontiguousarray(query_controller_instance_id, dtype=np.int64)
        self._tracker_consistent_visible = np.ones((len(query_points),), dtype=bool)
        self._tracker_query_alive_mask = np.ones((len(query_points),), dtype=bool)
        self._tracker_query_initial_seq = int(mask_packet.seq)
        print(
            "[tapnextpp-tracker] "
            f"initialized query_count={len(query_points)} requested={requested or 'phystwin_dense'} "
            f"union_pixels={union_pixels} object_pixels={object_pixels} controller_pixels={controller_pixels} "
            f"hand_a_queries={int(np.count_nonzero(query_controller_instance_id == QUERY_CONTROLLER_INSTANCE_HAND_A))} "
            f"hand_b_queries={int(np.count_nonzero(query_controller_instance_id == QUERY_CONTROLLER_INSTANCE_HAND_B))} "
            f"query_source={query_source} display_scope={self.args.tracker_display_scope} device={self.args.tracker_device}",
            flush=True,
        )
        return self._tracker_query_points_yx

    def _tracker_depth_for_lift(self, mask_packet: MaskPacket) -> tuple[np.ndarray, float]:
        """Return the tracker depth for lift."""
        if mask_packet.depth_u16 is not None:
            return mask_packet.depth_u16, float(mask_packet.depth_scale_m_per_unit)
        if mask_packet.depth_source in {"ffs", "ffs_remote"}:
            depth_m, _ffs_ms, _ffs_align_ms, _remote_rtt_ms, _server_total_ms, _request_kb, _response_kb = (
                self._compute_external_ffs_depth_color_m(mask_packet)
            )
            return np.ascontiguousarray(depth_m, dtype=np.float32), 1.0
        raise RuntimeError("tracker lift requires RGB-D depth")

    def _tracker_lift_mask(self, mask_packet: MaskPacket) -> np.ndarray | None:
        """Return the tracker lift mask."""
        scope = str(self.args.tracker_display_scope)
        if scope == TRACKER_DISPLAY_SCOPE_CONTROLLER:
            mask = np.asarray(mask_packet.controller_mask, dtype=bool)
            erode_pixels = controller_pcd_mask_erode_pixels(self.args)
        elif scope == TRACKER_DISPLAY_SCOPE_OBJECT:
            mask = np.asarray(mask_packet.object_mask, dtype=bool)
            erode_pixels = object_pcd_mask_erode_pixels(self.args)
        else:
            mask = _tracker_union_mask(mask_packet)
            erode_pixels = min(object_pcd_mask_erode_pixels(self.args), controller_pcd_mask_erode_pixels(self.args))
        if erode_pixels > 0:
            return erode_binary_mask(mask, erode_pixels=erode_pixels)
        return np.ascontiguousarray(mask)

    def _tracker_pcd_filter_residual_masks(self, mask_packet: MaskPacket) -> tuple[np.ndarray, np.ndarray]:
        """Return the tracker PCD filter residual masks."""
        if not pcd_filter_enabled(self.args):
            raise RuntimeError("pcd_filter_residual query source requires enabled sync PCD filtering")
        if str(self.args.pcd_filter_mode) != "sync":
            raise RuntimeError("pcd_filter_residual query source requires --pcd-filter-mode sync")
        if self.ray_x is None or self.ray_y is None:
            raise RuntimeError("pcd_filter_residual query source requires initialized projection grids")

        if mask_packet.depth_source in {"ffs", "ffs_remote"}:
            depth_m, _ffs_ms, _ffs_align_ms, _remote_rtt_ms, _server_total_ms, _request_kb, _response_kb = (
                self._compute_external_ffs_depth_color_m(mask_packet)
            )
        else:
            if mask_packet.depth_u16 is None:
                raise RuntimeError("pcd_filter_residual query source requires RGB-D depth")
            depth_m = np.ascontiguousarray(
                mask_packet.depth_u16.astype(np.float32) * np.float32(mask_packet.depth_scale_m_per_unit)
            )

        stride = int(self.args.pcd_stride)
        if stride > 1:
            color_bgr = mask_packet.color_bgr[::stride, ::stride]
            depth_for_pcd = depth_m[::stride, ::stride]
            controller_mask = mask_packet.controller_mask[::stride, ::stride]
            object_mask = mask_packet.object_mask[::stride, ::stride]
            ray_x_for_pcd = self.ray_x[::stride, ::stride]
            ray_y_for_pcd = self.ray_y[::stride, ::stride]
        else:
            color_bgr = mask_packet.color_bgr
            depth_for_pcd = depth_m
            controller_mask = mask_packet.controller_mask
            object_mask = mask_packet.object_mask
            ray_x_for_pcd = self.ray_x
            ray_y_for_pcd = self.ray_y

        controller_erode_pixels = controller_pcd_mask_erode_pixels(self.args)
        object_erode_pixels = object_pcd_mask_erode_pixels(self.args)
        if controller_erode_pixels > 0:
            controller_mask = erode_binary_mask(controller_mask, erode_pixels=controller_erode_pixels)
        if object_erode_pixels > 0:
            object_mask = erode_binary_mask(object_mask, erode_pixels=object_erode_pixels)

        controller_xyz, controller_colors, controller_yx, _controller_timing = backproject_masked_rgbd_profiled(
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
            rng=np.random.default_rng(int(mask_packet.seq) * 2 + 31),
            return_yx=True,
        )
        object_xyz, object_colors, object_yx, _object_timing = backproject_masked_rgbd_profiled(
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
            rng=np.random.default_rng(int(mask_packet.seq) * 2 + 29),
            return_yx=True,
        )
        if stride > 1:
            controller_yx = np.ascontiguousarray(controller_yx * int(stride), dtype=np.int64)
            object_yx = np.ascontiguousarray(object_yx * int(stride), dtype=np.int64)

        filter_input = self._make_filter_input(
            seq=int(mask_packet.seq),
            object_xyz=object_xyz,
            object_colors=object_colors,
            object_yx=object_yx,
            controller_xyz=controller_xyz,
            controller_colors=controller_colors,
            controller_yx=controller_yx,
        )
        filter_output = self._filter_pcd_input(filter_input)
        object_xyz_world = _transform_points_c2w(filter_output.object_xyz, self.table_c2w)
        controller_xyz_world = _transform_points_c2w(filter_output.controller_xyz, self.table_c2w)
        object_yx = filter_output.object_yx
        controller_yx = filter_output.controller_yx
        if bool(self.args.enable_table_z_filter):
            classes = str(self.args.table_z_filter_classes)
            if classes in {TABLE_Z_FILTER_CLASS_OBJECT, TABLE_Z_FILTER_CLASS_BOTH}:
                (
                    _object_xyz,
                    _object_colors,
                    object_yx,
                    _object_table_z_stats,
                ) = apply_table_z_filter_with_yx(
                    object_xyz_world,
                    filter_output.object_rgb,
                    object_yx,
                    enabled=True,
                    threshold_m=float(self.args.table_z_filter_threshold_m),
                    table_z_m=TABLE_Z_M,
                )
            if classes in {TABLE_Z_FILTER_CLASS_CONTROLLER, TABLE_Z_FILTER_CLASS_BOTH}:
                (
                    _controller_xyz,
                    _controller_colors,
                    controller_yx,
                    _controller_table_z_stats,
                ) = apply_table_z_filter_with_yx(
                    controller_xyz_world,
                    filter_output.controller_rgb,
                    controller_yx,
                    enabled=True,
                    threshold_m=float(self.args.table_z_filter_threshold_m),
                    table_z_m=TABLE_Z_M,
                )
        shape = tuple(mask_packet.object_mask.shape[:2])
        object_residual = _mask_from_yx(shape, object_yx)
        controller_residual = _mask_from_yx(shape, controller_yx)
        return object_residual, controller_residual

    def _ensure_tracker_query_alive_mask(self, query_count: int) -> np.ndarray:
        """Return the ensure tracker query alive mask."""
        count = max(0, int(query_count))
        if self._tracker_query_alive_mask is None or len(self._tracker_query_alive_mask) != count:
            self._tracker_query_alive_mask = np.ones((count,), dtype=bool)
            self._tracker_query_initial_seq = None
        return self._tracker_query_alive_mask

    def _current_tracker_query_alive_mask(
        self,
        *,
        current_seq: int,
        query_count: int,
        residual_visibility: np.ndarray | None,
    ) -> np.ndarray:
        """Return the current tracker query alive mask."""
        alive = self._ensure_tracker_query_alive_mask(query_count)
        if self._tracker_query_initial_seq is None:
            self._tracker_query_initial_seq = int(current_seq)
        retirement_frame = int(current_seq) > int(self._tracker_query_initial_seq)
        if (
            retirement_frame
            and residual_visibility is not None
            and tracker_marker_retirement_policy(self.args)
            == TRACKER_MARKER_RETIREMENT_POLICY_PCD_FILTER_RESIDUAL_TABLE_Z_ONCE_FALSE
        ):
            residual = np.asarray(residual_visibility, dtype=bool).reshape(-1)
            count = min(len(alive), len(residual))
            if count:
                alive[:count] &= residual[:count]
        return np.ascontiguousarray(alive.copy(), dtype=bool)

    def _build_tracker_marker_packet(self, mask_packet: MaskPacket, adapter: Any) -> TrackerMarkerPacket | None:
        """Build tracker marker packet."""
        query_points = self._ensure_tracker_queries(mask_packet, adapter)
        if query_points is None:
            if self.args.debug:
                print(
                    "[tapnextpp-tracker] waiting_for_non_empty_object_and_controller_masks "
                    f"seq={mask_packet.seq}",
                    flush=True,
                )
            return None
        assert self._tracker_query_is_object is not None
        assert self._tracker_query_is_controller is not None
        assert self._tracker_query_rgb_u8 is not None
        assert self._tracker_query_target_id is not None
        assert self._tracker_query_controller_instance_id is not None
        started_s = time.perf_counter()
        rgb = np.ascontiguousarray(mask_packet.color_bgr[:, :, ::-1], dtype=np.uint8)
        result = adapter.update(rgb)
        tracks_latest, visibility_latest = _latest_tracker_arrays(result)
        query_is_object_all = np.asarray(self._tracker_query_is_object, dtype=bool).reshape(-1)
        query_is_controller_all = np.asarray(self._tracker_query_is_controller, dtype=bool).reshape(-1)
        query_target_id_all = np.asarray(self._tracker_query_target_id, dtype=np.int64).reshape(-1)
        query_controller_instance_id_all = np.asarray(
            self._tracker_query_controller_instance_id,
            dtype=np.int64,
        ).reshape(-1)
        query_is_object = query_is_object_all
        query_is_controller = query_is_controller_all
        query_target_id = query_target_id_all
        query_controller_instance_id = query_controller_instance_id_all
        common_count = min(
            int(len(tracks_latest)),
            int(len(visibility_latest)),
            int(len(query_is_object)),
            int(len(query_is_controller)),
            int(len(query_target_id)),
            int(len(query_controller_instance_id)),
        )
        tracks_latest = tracks_latest[:common_count]
        visibility_latest = visibility_latest[:common_count]
        query_is_object = query_is_object[:common_count]
        query_is_controller = query_is_controller[:common_count]
        query_target_id = query_target_id[:common_count]
        query_controller_instance_id = query_controller_instance_id[:common_count]
        target_visibility = _tracker_per_target_visibility(
            tracks_latest,
            visibility_latest,
            mask_packet=mask_packet,
            query_target_id=query_target_id,
        )
        display_visibility = _tracker_display_visibility(
            target_visibility,
            query_is_object=query_is_object,
            query_is_controller=query_is_controller,
            display_scope=str(self.args.tracker_display_scope),
        )
        lift_mask = self._tracker_lift_mask(mask_packet)
        object_residual_mask: np.ndarray | None = None
        controller_residual_mask: np.ndarray | None = None
        residual_visibility: np.ndarray | None = None
        if tracker_query_source(self.args) == TRACKER_QUERY_SOURCE_PCD_FILTER_RESIDUAL:
            object_residual_mask, controller_residual_mask = self._tracker_pcd_filter_residual_masks(mask_packet)
            residual_visibility = _query_current_residual_visibility(
                tracks_latest,
                query_is_object=query_is_object,
                query_is_controller=query_is_controller,
                object_residual_mask=object_residual_mask,
                controller_residual_mask=controller_residual_mask,
            )
            display_visibility = np.where(residual_visibility, display_visibility, 0.0).astype(np.float32, copy=False)
            lift_mask = np.logical_or(object_residual_mask, controller_residual_mask)
        query_alive_mask = self._current_tracker_query_alive_mask(
            current_seq=int(mask_packet.seq),
            query_count=len(query_points),
            residual_visibility=residual_visibility,
        )
        alive_for_display = _fit_bool_array(query_alive_mask, len(display_visibility))
        display_visibility = np.where(alive_for_display, display_visibility, 0.0).astype(np.float32, copy=False)
        selected = _select_visible_spread_indices(
            tracks_latest,
            display_visibility,
            max_points=int(self.args.tracker_overlay_max_points),
        )
        selected_tracks = tracks_latest[selected]
        selected_visibility = display_visibility[selected]
        selected_query_is_object = query_is_object[selected]
        selected_query_is_controller = query_is_controller[selected]
        selected_query_target_id = query_target_id[selected]
        selected_query_controller_instance_id = query_controller_instance_id[selected]

        lift_start_s = time.perf_counter()
        depth_for_lift, depth_scale = self._tracker_depth_for_lift(mask_packet)
        depth_max_m = float("inf") if float(self.args.depth_max_m) <= 0.0 else float(self.args.depth_max_m)
        current_lift_valid = _tracker_lift_valid_mask(
            tracks_yx=tracks_latest,
            visibility=display_visibility,
            depth=depth_for_lift,
            depth_scale_m_per_unit=float(depth_scale),
            mask=lift_mask,
            depth_min_m=float(self.args.depth_min_m),
            depth_max_m=depth_max_m,
        )
        if self._tracker_consistent_visible is None or len(self._tracker_consistent_visible) != len(query_points):
            self._tracker_consistent_visible = np.ones((len(query_points),), dtype=bool)
        current_lift_valid_full = np.zeros_like(self._tracker_consistent_visible, dtype=bool)
        fitted_count = min(len(current_lift_valid), len(current_lift_valid_full))
        current_lift_valid_full[:fitted_count] = current_lift_valid[:fitted_count]
        self._tracker_consistent_visible &= current_lift_valid_full
        consistent_visible_count = int(np.count_nonzero(self._tracker_consistent_visible))
        lifted = lift_tracks_yx_to_world(
            tracks_yx=selected_tracks,
            visibility=selected_visibility,
            depth=depth_for_lift,
            intrinsics=mask_packet.intrinsics,
            c2w=self.table_c2w if self.table_c2w is not None else np.eye(4, dtype=np.float32),
            depth_scale_m_per_unit=float(depth_scale),
            mask=lift_mask,
            depth_min_m=float(self.args.depth_min_m),
            depth_max_m=depth_max_m,
        )
        lift_ms = _elapsed_ms(lift_start_s, time.perf_counter())
        source_indices = lifted.source_indices
        if len(source_indices):
            lifted_query_indices = selected[source_indices].astype(np.int64, copy=False)
            lifted_query_is_object = selected_query_is_object[source_indices]
            lifted_query_is_controller = selected_query_is_controller[source_indices]
            lifted_query_target_id = selected_query_target_id[source_indices]
            lifted_query_controller_instance_id = selected_query_controller_instance_id[source_indices]
            lifted_marker_colors = self._tracker_query_rgb_u8[lifted_query_indices]
        else:
            lifted_query_indices = np.empty((0,), dtype=np.int64)
            lifted_query_is_object = np.empty((0,), dtype=bool)
            lifted_query_is_controller = np.empty((0,), dtype=bool)
            lifted_query_target_id = np.empty((0,), dtype=np.int64)
            lifted_query_controller_instance_id = np.empty((0,), dtype=np.int64)
            lifted_marker_colors = np.empty((0, 3), dtype=np.uint8)
        if object_residual_mask is not None and controller_residual_mask is not None:
            marker_residual_audit = _audit_marker_residual_subset(
                lifted.tracks_yx,
                object_residual_mask=object_residual_mask,
                controller_residual_mask=controller_residual_mask,
            )
        else:
            marker_residual_audit = MarkerResidualAudit(
                pixels_yx=np.empty((0, 2), dtype=np.int64),
                valid=np.empty((0,), dtype=bool),
                violation=np.empty((0,), dtype=bool),
                checked_count=0,
                violation_count=0,
                gate=tracker_marker_gate(self.args),
            )
        hand_a_query_count = int(np.count_nonzero(lifted_query_controller_instance_id == QUERY_CONTROLLER_INSTANCE_HAND_A))
        hand_b_query_count = int(np.count_nonzero(lifted_query_controller_instance_id == QUERY_CONTROLLER_INSTANCE_HAND_B))
        object_query_count = int(np.count_nonzero(lifted_query_target_id == OBJECT_ID))
        remaining_object_query_count, remaining_controller_query_count, remaining_hand_a_query_count, remaining_hand_b_query_count = (
            _remaining_query_class_counts(
                query_alive_mask,
                query_is_object=query_is_object_all,
                query_is_controller=query_is_controller_all,
                query_controller_instance_id=query_controller_instance_id_all,
            )
        )
        remaining_query_count = int(np.count_nonzero(query_alive_mask))
        retired_query_count = max(0, int(len(query_points)) - remaining_query_count)
        done_s = time.perf_counter()
        stats = getattr(result, "stats", {}) or {}
        packet = TrackerMarkerPacket(
            seq=mask_packet.seq,
            marker_xyz_m=np.ascontiguousarray(lifted.points_world, dtype=np.float32).reshape(-1, 3),
            marker_colors_rgb_u8=np.ascontiguousarray(lifted_marker_colors, dtype=np.uint8).reshape(-1, 3),
            query_rgb_u8=np.ascontiguousarray(self._tracker_query_rgb_u8, dtype=np.uint8).reshape(-1, 3),
            query_points_yx=query_points,
            tracks_yx=np.ascontiguousarray(lifted.tracks_yx, dtype=np.float32).reshape(-1, 2),
            visibility=np.ascontiguousarray(selected_visibility[source_indices], dtype=np.float32),
            query_is_object=np.ascontiguousarray(lifted_query_is_object, dtype=bool),
            query_is_controller=np.ascontiguousarray(lifted_query_is_controller, dtype=bool),
            receive_perf_s=mask_packet.receive_perf_s,
            process_done_perf_s=done_s,
            query_count=int(len(query_points)),
            consistent_visible_count=consistent_visible_count,
            model_ms=float(stats.get("model_run_ms", stats.get("cuda_event_ms", 0.0)) or 0.0),
            lift_ms=float(lift_ms),
            e2e_ms=_elapsed_ms(started_s, done_s),
            backend=str(getattr(result, "backend", None) or adapter.name),
            display_scope=str(self.args.tracker_display_scope),
            query_indices=np.ascontiguousarray(lifted_query_indices, dtype=np.int64),
            query_target_id=np.ascontiguousarray(lifted_query_target_id, dtype=np.int64),
            query_controller_instance_id=np.ascontiguousarray(lifted_query_controller_instance_id, dtype=np.int64),
            query_all_target_id=np.ascontiguousarray(query_target_id_all, dtype=np.int64),
            query_all_controller_instance_id=np.ascontiguousarray(query_controller_instance_id_all, dtype=np.int64),
            hand_a_query_count=hand_a_query_count,
            hand_b_query_count=hand_b_query_count,
            object_query_count=object_query_count,
            marker_pixels_yx=np.ascontiguousarray(marker_residual_audit.pixels_yx, dtype=np.int64).reshape(-1, 2),
            marker_residual_valid=np.ascontiguousarray(marker_residual_audit.valid, dtype=bool),
            marker_residual_violation=np.ascontiguousarray(marker_residual_audit.violation, dtype=bool),
            marker_residual_checked_count=int(marker_residual_audit.checked_count),
            marker_residual_violation_count=int(marker_residual_audit.violation_count),
            marker_residual_gate=str(marker_residual_audit.gate),
            query_alive_mask=np.ascontiguousarray(query_alive_mask, dtype=bool),
            remaining_query_count=remaining_query_count,
            remaining_object_query_count=remaining_object_query_count,
            remaining_controller_query_count=remaining_controller_query_count,
            remaining_hand_a_query_count=remaining_hand_a_query_count,
            remaining_hand_b_query_count=remaining_hand_b_query_count,
            retired_query_count=retired_query_count,
            all_tracks_yx=np.ascontiguousarray(tracks_latest, dtype=np.float32).reshape(-1, 2),
            all_tracker_visibility=np.ascontiguousarray(visibility_latest, dtype=np.float32).reshape(-1),
            coordinate_frame=self._pcd_coordinate_frame(),
        )
        if self.args.debug:
            print(
                "[tapnextpp-tracker] "
                f"seq={packet.seq} markers={packet.marker_count}/{len(selected_tracks)} "
                f"residual_violations={packet.marker_residual_violation_count}/{packet.marker_residual_checked_count} "
                f"consistent={packet.consistent_visible_count}/{packet.query_count} "
                f"remaining={packet.remaining_query_count}/{packet.query_count} retired={packet.retired_query_count} "
                f"hand_a={packet.hand_a_query_count} hand_b={packet.hand_b_query_count} object={packet.object_query_count} "
                f"queries={packet.query_count} model_ms={packet.model_ms:.1f} "
                f"lift_ms={packet.lift_ms:.1f} e2e_ms={packet.e2e_ms:.1f} "
                f"fps={self.tracker_stats.fps:.1f}",
                flush=True,
            )
        return packet

    def _tracker_worker(self) -> None:
        """Return the tracker worker."""
        try:
            adapter = self._build_tracker_adapter()
            print(
                "[tapnextpp-tracker] "
                f"backend={adapter.name} device={self.args.tracker_device} "
                f"repo={self.args.tapnet_repo_dir} checkpoint={self.args.tapnextpp_checkpoint} "
                f"image_size={self.args.tapnextpp_image_size} overlay_max={int(self.args.tracker_overlay_max_points)}",
                flush=True,
            )
            last_seq = -1
            while not self.stop_event.is_set():
                mask_packet = self.mask_slot.get_latest_after(last_seq)
                if mask_packet is None:
                    time.sleep(0.001)
                    continue
                last_seq = mask_packet.seq
                packet = self._build_tracker_marker_packet(mask_packet, adapter)
                if packet is None:
                    continue
                self.tracker_marker_slot.put(packet)
                if self.headless_capture_writer is not None and not self._headless_product_rows_gated():
                    self.headless_capture_writer.write_tracker(packet)
                self.tracker_stats.record(packet.process_done_perf_s)
        except Exception as exc:
            if not self.stop_event.is_set():
                self._record_fatal_worker_error("TAPNext++ tracker worker", exc)

    # ------------------------------------------------------------------
    # Shape-prior warmup integration
    # ------------------------------------------------------------------
    def _shape_prior_frame0_request_from_pcd_result(
        self,
        result: PcdBuildResult,
    ) -> shape_prior_warmup.ShapePriorFrame0Request | None:
        """Return the shape prior frame0 request from PCD result."""
        if not bool(getattr(self.args, "shape_prior_warmup", False)):
            return None
        if self.table_c2w is None:
            return None
        if result.depth_m is None:
            return None
        mask_packet = result.mask_packet
        if int(result.packet.seq) != int(mask_packet.seq):
            return None
        k_color = mask_packet.k_color
        if k_color is None and self.runtime is not None:
            k_color = np.asarray(self.runtime.k_color, dtype=np.float32)
        if k_color is None:
            return None
        object_observation_mask = self._shape_prior_observation_mask_from_pcd_result(result)
        return shape_prior_warmup.ShapePriorFrame0Request(
            seq=int(mask_packet.seq),
            source_timestamp_s=mask_packet.source_timestamp_s,
            input_source=str(self.args.input_source),
            depth_backend=depth_backend_label(self.args),
            depth_source_internal=str(self.args.depth_source),
            rgb_u8=mask_packet.color_bgr[:, :, ::-1],
            object_mask=mask_packet.object_mask,
            object_observation_mask=object_observation_mask,
            controller_mask=mask_packet.controller_mask,
            depth_color_m=result.depth_m,
            k_color=k_color,
            camera_to_world_c2w=self.table_c2w,
            table_z_m=TABLE_Z_M,
        )

    def _shape_prior_observation_mask_from_pcd_result(self, result: PcdBuildResult) -> np.ndarray:
        """Return the shape prior observation mask from PCD result."""
        raw = np.asarray(result.mask_packet.object_mask, dtype=bool)
        candidate = result.object_observation_mask
        if candidate is None:
            candidate = result.object_pcd_mask
        if candidate is None:
            return np.ascontiguousarray(raw, dtype=bool)

        mask = np.asarray(candidate, dtype=bool)
        if mask.shape == raw.shape:
            return np.ascontiguousarray(mask, dtype=bool)

        stride = max(1, int(result.pcd_stride))
        if stride > 1 and mask.shape == raw[::stride, ::stride].shape:
            expanded = np.zeros_like(raw, dtype=bool)
            expanded[::stride, ::stride] = mask
            return np.ascontiguousarray(expanded, dtype=bool)

        return np.ascontiguousarray(mask, dtype=bool)

    def _packet_with_shape_prior_state(self, packet: MaskedPcdPacket) -> MaskedPcdPacket:
        """Return the packet with shape prior state."""
        profile = self._shape_prior_profile()
        result = self.shape_prior_manager.ready_result()
        if result is not None and result.ready:
            return replace(
                packet,
                shape_prior_points_m=np.ascontiguousarray(result.points_m, dtype=np.float32).reshape(-1, 3),
                shape_prior_colors_rgb_u8=np.ascontiguousarray(result.colors_rgb_u8, dtype=np.uint8).reshape(-1, 3),
                shape_prior_status=shape_prior_warmup.STATUS_READY,
                shape_prior_profile=profile,
            )
        return replace(
            packet,
            shape_prior_points_m=np.empty((0, 3), dtype=np.float32),
            shape_prior_colors_rgb_u8=np.empty((0, 3), dtype=np.uint8),
            shape_prior_status=str(
                profile.get("shape_prior_status", shape_prior_warmup.STATUS_DISABLED)
            ),
            shape_prior_profile=profile,
        )

    def _maybe_start_shape_prior_from_pcd_result(
        self,
        result: PcdBuildResult,
        *,
        from_strict_pair: bool = False,
    ) -> bool:
        """Maybe start or update start shape prior from PCD result."""
        frame0_request = self._shape_prior_frame0_request_from_pcd_result(result)
        if frame0_request is None:
            return False
        submitted = self.shape_prior_manager.maybe_submit(frame0_request)
        if submitted:
            self._write_shape_prior_profile_json()
        return bool(submitted)

    def _maybe_write_shape_prior_headless_result(self) -> None:
        """Maybe start or update write shape prior headless result."""
        profile = self._shape_prior_profile_payload()
        result = self.shape_prior_manager.ready_result()
        if self.headless_capture_writer is not None and result is not None and result.ready and not self._shape_prior_written:
            self.headless_capture_writer.write_shape_prior_result(result)
            self._shape_prior_written = True
            self._write_shape_prior_profile_json(profile)
            return
        if self.headless_capture_writer is not None:
            self.headless_capture_writer.update_metadata(profile)
        self._write_shape_prior_profile_json(profile)

    def _run_deferred_shape_prior_after_teardown(self) -> None:
        """Run deferred shape prior after teardown."""
        return

    # ------------------------------------------------------------------
    # Lossless pairing & publishing (pcd+tracker pairs -> render/headless)
    # ------------------------------------------------------------------
    def _publish_strict_render_pair(
        self,
        pcd_result: PcdBuildResult,
        tracker_packet: TrackerMarkerPacket,
    ) -> PairedRenderPacket:
        """Publish strict render pair."""
        self._maybe_start_shape_prior_from_pcd_result(pcd_result, from_strict_pair=True)
        pcd_result = replace(
            pcd_result,
            packet=self._packet_with_shape_prior_state(pcd_result.packet),
        )
        pair = PairedRenderPacket(
            seq=int(pcd_result.packet.seq),
            pcd_packet=pcd_result.packet,
            tracker_packet=tracker_packet,
            mask_packet=pcd_result.mask_packet,
        )
        self.paired_render_slot.put(pair)
        self.pcd_stats.record(pcd_result.packet.process_done_perf_s)
        self.tracker_stats.record(tracker_packet.process_done_perf_s)
        self._lossless_pairs_emitted += 1
        if pair.seq == 0:
            self._lossless_first_pair_published.set()
        if self.headless_capture_writer is not None:
            self._maybe_write_shape_prior_headless_result()
            # One gate decision per frame: the row and its query_trajectory
            # sidecar must agree even if the prior flips ready mid-frame.
            rows_gated = self._headless_product_rows_gated()
            if not rows_gated:
                self.headless_capture_writer.write_tracker(tracker_packet)
            self._write_headless_pcd_result(
                pcd_result, tracker_packet=tracker_packet, gated=rows_gated
            )
        return pair

    def _publish_pairer_outputs(self, pairs: list[PairedBuildResult]) -> None:
        """Publish pairer outputs."""
        for pair in pairs:
            self.lossless_pair_output_queue.put(pair)

    def _publish_ordered_lossless_pair(self, pair: PairedBuildResult) -> PairedRenderPacket | None:
        """Publish ordered lossless pair."""
        seq = int(pair.seq)
        with self._lossless_publish_condition:
            while seq != self._lossless_next_publish_seq:
                if seq < self._lossless_next_publish_seq:
                    raise LosslessPipelineError(
                        f"lossless publish received stale seq {seq}, expected {self._lossless_next_publish_seq}"
                    )
                if self.stop_event.is_set():
                    return None
                self._lossless_publish_condition.wait(timeout=0.05)
        published = self._publish_strict_render_pair(pair.pcd_result, pair.tracker_packet)
        with self._lossless_publish_condition:
            expected = self._lossless_next_publish_seq
            if seq != expected:
                raise LosslessPipelineError(f"lossless publish expected seq {expected}, got {seq}")
            self._lossless_next_publish_seq += 1
            self._lossless_publish_condition.notify_all()
        return published

    def _maybe_finish_lossless_processing(self) -> None:
        """Maybe start or update finish lossless processing."""
        if not self._lossless_enabled():
            return
        if self.same_seq_pairer.done and not self._lossless_processing_done.is_set():
            self.lossless_pair_output_queue.close()

    def _finish_lossless_output(self) -> None:
        """Finish lossless output."""
        if not self._lossless_enabled():
            return
        if not self._lossless_processing_done.is_set():
            self._lossless_processing_done.set()

    def _lossless_pair_output_worker(self) -> None:
        """Return the lossless pair output worker."""
        try:
            while not self.stop_event.is_set():
                pair = self.lossless_pair_output_queue.get(stop_event=self.stop_event)
                if pair is None:
                    break
                self._publish_ordered_lossless_pair(pair)
            self._finish_lossless_output()
        except Exception as exc:
            if not self.stop_event.is_set():
                self._record_fatal_worker_error("lossless pair output worker", exc)

    def _lossless_pcd_worker(self) -> None:
        """Return the lossless PCD worker."""
        rng = np.random.default_rng()
        try:
            while not self.stop_event.is_set():
                mask_packet = self.lossless_pcd_mask_queue.get(stop_event=self.stop_event)
                if mask_packet is None:
                    break
                result = self._build_pcd_packet_from_mask(
                    mask_packet,
                    rng=rng,
                    require_filter_seq=True,
                )
                self._maybe_start_shape_prior_from_pcd_result(result)
                self._lossless_pcd_results += 1
                if not self.same_seq_pairer.wait_for_side_capacity("pcd", stop_event=self.stop_event):
                    break
                with self._lossless_pairer_lock:
                    pairs = self.same_seq_pairer.add_pcd_result(result)
                    self._publish_pairer_outputs(pairs)
            with self._lossless_pairer_lock:
                pairs = self.same_seq_pairer.close_pcd()
                self._publish_pairer_outputs(pairs)
                self._maybe_finish_lossless_processing()
        except Exception as exc:
            if not self.stop_event.is_set():
                self._record_fatal_worker_error("lossless PCD worker", exc)

    def _lossless_tracker_worker(self) -> None:
        """Return the lossless tracker worker."""
        try:
            adapter = self._build_tracker_adapter()
            print(
                "[tapnextpp-tracker] "
                f"backend={adapter.name} device={self.args.tracker_device} "
                f"repo={self.args.tapnet_repo_dir} checkpoint={self.args.tapnextpp_checkpoint} "
                f"image_size={self.args.tapnextpp_image_size} overlay_max={int(self.args.tracker_overlay_max_points)} "
                "strict_sync=1 lossless=1",
                flush=True,
            )
            while not self.stop_event.is_set():
                mask_packet = self.lossless_tracker_mask_queue.get(stop_event=self.stop_event)
                if mask_packet is None:
                    break
                packet = self._build_tracker_marker_packet(mask_packet, adapter)
                if packet is None:
                    raise LosslessPipelineError(f"tracker did not produce packet for seq {mask_packet.seq}")
                self._lossless_tracker_results += 1
                if not self.same_seq_pairer.wait_for_side_capacity("tracker", stop_event=self.stop_event):
                    break
                with self._lossless_pairer_lock:
                    pairs = self.same_seq_pairer.add_tracker_packet(packet)
                    self._publish_pairer_outputs(pairs)
            with self._lossless_pairer_lock:
                pairs = self.same_seq_pairer.close_tracker()
                self._publish_pairer_outputs(pairs)
                self._maybe_finish_lossless_processing()
        except Exception as exc:
            if not self.stop_event.is_set():
                self._record_fatal_worker_error("lossless TAPNext++ tracker worker", exc)

    def _strict_paired_worker(self) -> None:
        """Return the strict paired worker."""
        try:
            adapter = self._build_tracker_adapter()
            print(
                "[tapnextpp-tracker] "
                f"backend={adapter.name} device={self.args.tracker_device} "
                f"repo={self.args.tapnet_repo_dir} checkpoint={self.args.tapnextpp_checkpoint} "
                f"image_size={self.args.tapnextpp_image_size} overlay_max={int(self.args.tracker_overlay_max_points)} "
                "strict_sync=1",
                flush=True,
            )
            last_seq = -1
            rng = np.random.default_rng()
            while not self.stop_event.is_set():
                mask_packet = self.mask_slot.get_latest_after(last_seq)
                if mask_packet is None:
                    time.sleep(0.001)
                    continue
                last_seq = mask_packet.seq
                try:
                    pcd_result = self._build_pcd_packet_from_mask(
                        mask_packet,
                        rng=rng,
                        require_filter_seq=True,
                    )
                except Exception as exc:
                    if not self.stop_event.is_set():
                        print(f"[WARN] strict PCD frame {mask_packet.seq} failed: {type(exc).__name__}: {exc}", flush=True)
                    continue
                self._maybe_start_shape_prior_from_pcd_result(pcd_result)
                tracker_packet = self._build_tracker_marker_packet(mask_packet, adapter)
                if tracker_packet is None:
                    continue
                self._publish_strict_render_pair(pcd_result, tracker_packet)
        except Exception as exc:
            if not self.stop_event.is_set():
                self._record_fatal_worker_error("strict same-seq tracker/PCD", exc)

    # ------------------------------------------------------------------
    # Segmentation frame execution (EdgeTAM forward per frame)
    # ------------------------------------------------------------------
    def _wait_for_first_frame(self) -> FramePacket | None:
        """Wait for for first frame."""
        if self._lossless_enabled():
            return self.lossless_frame_queue.get(stop_event=self.stop_event)
        while not self.stop_event.is_set():
            frame = self.capture_slot.get_latest_after(-1)
            if frame is not None:
                return frame
            time.sleep(0.005)
        return None

    def _publish_mask_packet(self, packet: MaskPacket) -> None:
        """Publish mask packet."""
        self.mask_slot.put(packet)
        if self._lossless_enabled():
            if not self.lossless_pcd_mask_queue.wait_for_capacity(stop_event=self.stop_event):
                return
            if not self.lossless_tracker_mask_queue.wait_for_capacity(stop_event=self.stop_event):
                return
            self.lossless_pcd_mask_queue.put(packet)
            self.lossless_tracker_mask_queue.put(packet)
            self._lossless_segmented_frames += 1

    def _autocast_context(self, torch_module: Any) -> Any:
        """Return the autocast context."""
        if not str(self.args.device).startswith("cuda") or self.args.dtype == "float32":
            return nullcontext()
        dtype = torch_module.bfloat16 if self.args.dtype == "bfloat16" else torch_module.float16
        return torch_module.autocast("cuda", dtype=dtype)

    def _prune_edgetam_live_session(self, session: Any, *, current_frame_idx: int) -> None:
        """Prune edgetam live session."""
        keep_frames = int(self.args.edgetam_live_session_keep_frames)
        if keep_frames <= 0:
            return
        min_frame_idx = int(current_frame_idx) - keep_frames + 1

        processed_frames = getattr(session, "processed_frames", None)
        if isinstance(processed_frames, dict):
            for frame_idx in list(processed_frames):
                if int(frame_idx) < min_frame_idx:
                    processed_frames.pop(frame_idx, None)

        output_dict_per_obj = getattr(session, "output_dict_per_obj", None)
        if isinstance(output_dict_per_obj, dict):
            for output_dict in output_dict_per_obj.values():
                if not isinstance(output_dict, dict):
                    continue
                non_cond_outputs = output_dict.get("non_cond_frame_outputs")
                if isinstance(non_cond_outputs, dict):
                    for frame_idx in list(non_cond_outputs):
                        if int(frame_idx) < min_frame_idx:
                            non_cond_outputs.pop(frame_idx, None)

        frames_tracked_per_obj = getattr(session, "frames_tracked_per_obj", None)
        if isinstance(frames_tracked_per_obj, dict):
            for tracked_frames in frames_tracked_per_obj.values():
                if not isinstance(tracked_frames, dict):
                    continue
                for frame_idx in list(tracked_frames):
                    if int(frame_idx) < min_frame_idx:
                        tracked_frames.pop(frame_idx, None)

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
        initial_masks: InitialMaskBundle,
        add_prompt: bool,
    ) -> MaskPacket:
        """Run segmentation frame."""
        image = main_warmup.bgr_to_pil_rgb(frame.color_bgr)
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
                    prompt_obj_ids.append(HAND_A_ID)
                    prompt_masks.append(
                        np.asarray(initial_masks.hand_a_mask, dtype=bool)
                    )
                if object_tracking_enabled(self.args):
                    prompt_obj_ids.append(OBJECT_ID)
                    prompt_masks.append(
                        np.asarray(initial_masks.object_mask, dtype=bool)
                    )
                if controller_tracking_enabled(self.args):
                    prompt_obj_ids.append(HAND_B_ID)
                    prompt_masks.append(
                        np.asarray(initial_masks.hand_b_mask, dtype=bool)
                    )
                _unused, prompt_ms, prompt_pre_sync_ms, prompt_post_sync_ms = _time_runtime_ms(
                    torch_module,
                    self.args.device,
                    lambda: processor.add_inputs_to_inference_session(
                        inference_session=session,
                        frame_idx=int(frame.seq),
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
                fn=lambda: model(inference_session=session, frame=pixel_values, frame_idx=int(frame.seq)),
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
        masks_by_id = extract_object_masks_from_hf_output(
            output,
            post_masks,
            mask_logit_threshold=float(self.args.edgetam_mask_logit_threshold),
        )
        missing = [obj_id for obj_id in active_object_ids(self.args) if obj_id not in masks_by_id]
        if missing:
            raise RuntimeError(f"HF output missing tracked object ids: {missing}")
        reference_mask = next(iter(masks_by_id.values()))
        object_mask = masks_by_id.get(OBJECT_ID)
        if object_mask is None:
            object_mask = np.zeros_like(reference_mask, dtype=bool)
        hand_a_mask = masks_by_id.get(HAND_A_ID)
        if hand_a_mask is None:
            hand_a_mask = np.zeros_like(reference_mask, dtype=bool)
        hand_b_mask = masks_by_id.get(HAND_B_ID)
        if hand_b_mask is None:
            hand_b_mask = np.zeros_like(reference_mask, dtype=bool)
        controller_mask = np.logical_or(hand_a_mask, hand_b_mask)
        self._prune_edgetam_live_session(session, current_frame_idx=int(output.frame_idx))
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
            controller_mask=np.ascontiguousarray(controller_mask, dtype=bool),
            object_mask=np.ascontiguousarray(object_mask, dtype=bool),
            hand_a_mask=np.ascontiguousarray(hand_a_mask, dtype=bool),
            hand_b_mask=np.ascontiguousarray(hand_b_mask, dtype=bool),
            depth_u16=frame.depth_u16,
            ir_left_u8=frame.ir_left_u8,
            ir_right_u8=frame.ir_right_u8,
            k_ir_left=frame.k_ir_left,
            t_ir_left_to_color=frame.t_ir_left_to_color,
            k_color=frame.k_color,
            ir_baseline_m=frame.ir_baseline_m,
            source_timestamp_s=frame.source_timestamp_s,
            source_frame_index=frame.source_frame_index,
            source_step=frame.source_step,
        )

    # ------------------------------------------------------------------
    # Point-cloud filtering (per-class filters, async budget, telemetry)
    # ------------------------------------------------------------------
    def _make_filter_input(
        self,
        *,
        seq: int,
        object_xyz: np.ndarray,
        object_colors: np.ndarray,
        object_yx: np.ndarray | None = None,
        controller_xyz: np.ndarray,
        controller_colors: np.ndarray,
        controller_yx: np.ndarray | None = None,
    ) -> FilterInput:
        """Create filter input."""
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
            object_yx=np.asarray(
                object_yx if object_yx is not None else np.empty((0, 2), dtype=np.int64),
                dtype=np.int64,
            ).reshape(-1, 2),
            controller_yx=np.asarray(
                controller_yx if controller_yx is not None else np.empty((0, 2), dtype=np.int64),
                dtype=np.int64,
            ).reshape(-1, 2),
        )

    def _apply_single_pcd_filter(
        self,
        *,
        points: np.ndarray,
        colors: np.ndarray,
        yx: np.ndarray | None = None,
        mode: str,
        cap: int,
        voxel_size_m: float,
        keep_components: int,
        min_retain_ratio: float,
        min_raw_retain_ratio: float,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """Apply single PCD filter."""
        raw_points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        raw_colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
        raw_yx = (
            np.asarray(yx, dtype=np.int64).reshape(-1, 2)
            if yx is not None
            else np.empty((0, 2), dtype=np.int64)
        )
        if len(raw_yx) not in {0, len(raw_points)}:
            raise ValueError("yx must have the same first dimension as points when provided")

        def select_yx(source_yx: np.ndarray, indices: np.ndarray) -> np.ndarray:
            """Select YX."""
            if len(source_yx) == 0:
                return np.empty((0, 2), dtype=np.int64)
            return np.ascontiguousarray(source_yx[np.asarray(indices, dtype=np.int64)], dtype=np.int64).reshape(-1, 2)

        cap_start_s = time.perf_counter()
        cap_indices = voxel_cap_indices(
            raw_points,
            max_points=int(cap),
            voxel_size_m=float(voxel_size_m),
            rng=rng,
        )
        capped_points = np.ascontiguousarray(raw_points[cap_indices], dtype=np.float32).reshape(-1, 3)
        capped_colors = np.ascontiguousarray(raw_colors[cap_indices], dtype=np.uint8).reshape(-1, 3)
        capped_yx = select_yx(raw_yx, cap_indices)
        cap_ms = _elapsed_ms(cap_start_s, time.perf_counter())
        raw_point_count = int(len(raw_points))
        capped_point_count = int(len(capped_points))

        fallback_to_capped = False
        fallback_reason = ""
        fallback_source = "none"
        cap_raw_retain_ratio = float(capped_point_count / max(1, raw_point_count))
        if (
            mode != PCD_FILTER_NONE
            and float(min_raw_retain_ratio) > 0.0
            and raw_point_count > 0
            and capped_point_count < raw_point_count
            and cap_raw_retain_ratio < float(min_raw_retain_ratio)
        ):
            filtered_points = np.ascontiguousarray(raw_points, dtype=np.float32).reshape(-1, 3)
            filtered_colors = np.ascontiguousarray(raw_colors, dtype=np.uint8).reshape(-1, 3)
            filtered_yx = np.ascontiguousarray(raw_yx, dtype=np.int64).reshape(-1, 2)
            return filtered_points, filtered_colors, filtered_yx, {
                "mode": str(mode),
                "raw_points": raw_point_count,
                "cap_points": capped_point_count,
                "output_points": int(len(filtered_points)),
                "filter_output_points": capped_point_count,
                "filter_retain_ratio": 1.0 if capped_point_count > 0 else 0.0,
                "raw_retain_ratio": cap_raw_retain_ratio,
                "min_retain_ratio": float(min_retain_ratio),
                "min_raw_retain_ratio": float(min_raw_retain_ratio),
                "fallback_to_capped": True,
                "fallback_reason": "skip_filter_low_cap_raw_retain_ratio",
                "fallback_source": "raw",
                "cap": int(cap),
                "voxel_size_m": float(voxel_size_m),
                "keep_components": int(keep_components),
                "cap_ms": float(cap_ms),
                "filter_ms": 0.0,
            }

        filter_start_s = time.perf_counter()
        if mode == PCD_FILTER_NONE:
            filtered_points = np.asarray(capped_points, dtype=np.float32).reshape(-1, 3)
            filtered_colors = capped_colors
            filtered_yx = capped_yx
        elif mode == PCD_FILTER_VOXEL_DENSITY:
            density_indices = voxel_density_indices(
                capped_points,
                voxel_size_m=float(voxel_size_m),
                min_points_per_voxel=int(self.args.voxel_density_min_points),
            )
            filtered_points = np.asarray(capped_points[density_indices], dtype=np.float32).reshape(-1, 3)
            filtered_colors = np.asarray(capped_colors[density_indices], dtype=np.uint8).reshape(-1, 3)
            filtered_yx = select_yx(capped_yx, density_indices)
        elif mode == PCD_FILTER_PT_FILTER:
            from demo_v6_1.utils.pcd_postprocess import (
                apply_phystwin_like_radius_postprocess_with_trace,
            )

            filtered_points, filtered_colors, _unused_stats, trace = apply_phystwin_like_radius_postprocess_with_trace(
                points=capped_points,
                colors=capped_colors,
                enabled=True,
                radius_m=float(self.args.filter_radius_m),
                nb_points=int(self.args.filter_nb_points),
            )
            kept_indices = np.flatnonzero(np.asarray(trace["kept_mask"], dtype=bool).reshape(-1))
            filtered_yx = select_yx(capped_yx, kept_indices)
        elif mode == PCD_FILTER_ENHANCED_PT:
            from demo_v6_1.utils.pcd_postprocess import (
                apply_enhanced_phystwin_like_postprocess_with_trace,
            )

            filtered_points, filtered_colors, _unused_stats, trace = apply_enhanced_phystwin_like_postprocess_with_trace(
                points=capped_points,
                colors=capped_colors,
                enabled=True,
                radius_m=float(self.args.filter_radius_m),
                nb_points=int(self.args.filter_nb_points),
                component_voxel_size_m=float(self.args.enhanced_component_voxel_size_m),
                keep_near_main_gap_m=float(self.args.enhanced_keep_near_main_gap_m),
                keep_top_n_components=int(keep_components),
            )
            kept_indices = np.flatnonzero(np.asarray(trace["kept_mask"], dtype=bool).reshape(-1))
            filtered_yx = select_yx(capped_yx, kept_indices)
        else:
            raise ValueError(f"unsupported PCD filter mode: {mode}")

        filter_ms = _elapsed_ms(filter_start_s, time.perf_counter())
        filtered_points = np.ascontiguousarray(filtered_points, dtype=np.float32).reshape(-1, 3)
        filtered_colors = np.ascontiguousarray(filtered_colors, dtype=np.uint8).reshape(-1, 3)
        filtered_yx = np.ascontiguousarray(filtered_yx, dtype=np.int64).reshape(-1, 2)
        filter_output_points = int(len(filtered_points))
        retain_ratio = float(filter_output_points / max(1, capped_point_count))
        raw_retain_ratio = float(filter_output_points / max(1, raw_point_count))
        if filter_output_points == 0 and int(len(capped_points)) > 0:
            if float(min_raw_retain_ratio) > 0.0:
                filtered_points = np.ascontiguousarray(raw_points, dtype=np.float32).reshape(-1, 3)
                filtered_colors = np.ascontiguousarray(raw_colors, dtype=np.uint8).reshape(-1, 3)
                filtered_yx = np.ascontiguousarray(raw_yx, dtype=np.int64).reshape(-1, 2)
                fallback_reason = "empty_filter_output_raw"
                fallback_source = "raw"
            else:
                filtered_points = np.ascontiguousarray(capped_points, dtype=np.float32).reshape(-1, 3)
                filtered_colors = np.ascontiguousarray(capped_colors, dtype=np.uint8).reshape(-1, 3)
                filtered_yx = np.ascontiguousarray(capped_yx, dtype=np.int64).reshape(-1, 2)
                fallback_reason = "empty_filter_output"
                fallback_source = "capped"
            fallback_to_capped = True
        elif (
            float(min_raw_retain_ratio) > 0.0
            and raw_point_count > 0
            and raw_retain_ratio < float(min_raw_retain_ratio)
        ):
            filtered_points = np.ascontiguousarray(raw_points, dtype=np.float32).reshape(-1, 3)
            filtered_colors = np.ascontiguousarray(raw_colors, dtype=np.uint8).reshape(-1, 3)
            filtered_yx = np.ascontiguousarray(raw_yx, dtype=np.int64).reshape(-1, 2)
            fallback_to_capped = True
            fallback_reason = "low_filter_raw_retain_ratio"
            fallback_source = "raw"
        elif (
            float(min_retain_ratio) > 0.0
            and capped_point_count > 0
            and retain_ratio < float(min_retain_ratio)
        ):
            filtered_points = np.ascontiguousarray(capped_points, dtype=np.float32).reshape(-1, 3)
            filtered_colors = np.ascontiguousarray(capped_colors, dtype=np.uint8).reshape(-1, 3)
            filtered_yx = np.ascontiguousarray(capped_yx, dtype=np.int64).reshape(-1, 2)
            fallback_to_capped = True
            fallback_reason = "low_filter_retain_ratio"
            fallback_source = "capped"
        return filtered_points, filtered_colors, filtered_yx, {
            "mode": str(mode),
            "raw_points": raw_point_count,
            "cap_points": capped_point_count,
            "output_points": int(len(filtered_points)),
            "filter_output_points": filter_output_points,
            "filter_retain_ratio": retain_ratio,
            "raw_retain_ratio": raw_retain_ratio,
            "min_retain_ratio": float(min_retain_ratio),
            "min_raw_retain_ratio": float(min_raw_retain_ratio),
            "fallback_to_capped": bool(fallback_to_capped),
            "fallback_reason": fallback_reason,
            "fallback_source": fallback_source,
            "cap": int(cap),
            "voxel_size_m": float(voxel_size_m),
            "keep_components": int(keep_components),
            "cap_ms": float(cap_ms),
            "filter_ms": float(filter_ms),
        }

    def _filter_pcd_input(self, item: FilterInput) -> FilterOutput:
        """Return the filter PCD input."""
        started_s = time.perf_counter()
        object_points, object_colors, object_yx, object_stats = self._apply_single_pcd_filter(
            points=item.object_xyz,
            colors=item.object_rgb,
            yx=item.object_yx,
            mode=str(self.args.object_filter),
            cap=int(item.object_cap),
            voxel_size_m=float(item.object_voxel_size_m),
            keep_components=int(self.args.object_filter_keep_components),
            min_retain_ratio=float(DEFAULT_OBJECT_FILTER_MIN_RETAIN_RATIO),
            min_raw_retain_ratio=float(DEFAULT_OBJECT_FILTER_MIN_RAW_RETAIN_RATIO),
            rng=np.random.default_rng(int(item.seq) * 2 + 17),
        )
        controller_points, controller_colors, controller_yx, controller_stats = self._apply_single_pcd_filter(
            points=item.controller_xyz,
            colors=item.controller_rgb,
            yx=item.controller_yx,
            mode=str(self.args.controller_filter),
            cap=int(item.controller_cap),
            voxel_size_m=float(item.controller_voxel_size_m),
            keep_components=int(self.args.controller_filter_keep_components),
            min_retain_ratio=float(DEFAULT_CONTROLLER_FILTER_MIN_RETAIN_RATIO),
            min_raw_retain_ratio=float(DEFAULT_CONTROLLER_FILTER_MIN_RAW_RETAIN_RATIO),
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
            object_yx=object_yx,
            controller_yx=controller_yx,
            stats={
                "object": object_stats,
                "controller": controller_stats,
                "object_filter": str(self.args.object_filter),
                "controller_filter": str(self.args.controller_filter),
            },
        )

    def _filter_worker_stats(self) -> dict[str, Any]:
        """Return the filter worker stats."""
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

    def _filter_output_is_fresh(self, *, packet_seq: int, output: FilterOutput) -> bool:
        """Return the filter output is fresh."""
        age_frames = max(0, int(packet_seq) - int(output.seq))
        return age_frames <= int(self.args.filter_max_age_frames)

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
        """Return the filter telemetry from output."""
        worker_stats = self._filter_worker_stats()
        if output is None:
            return PcdFilterTelemetry(
                enabled=pcd_filter_enabled(self.args),
                mode=str(self.args.pcd_filter_mode if pcd_filter_enabled(self.args) else PCD_FILTER_NONE),
                object_raw_points=int(object_raw_points),
                object_cap_points=int(object_cap_points),
                object_output_points=int(object_cap_points),
                object_prefallback_points=int(object_cap_points),
                object_raw_retain_ratio=1.0 if int(object_raw_points) > 0 else 0.0,
                controller_raw_points=int(controller_raw_points),
                controller_cap_points=int(controller_cap_points),
                controller_output_points=int(controller_cap_points),
                controller_prefallback_points=int(controller_cap_points),
                controller_raw_retain_ratio=1.0 if int(controller_raw_points) > 0 else 0.0,
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
            object_prefallback_points=int(object_stats.get("filter_output_points", object_cap_points)),
            object_raw_retain_ratio=float(object_stats.get("raw_retain_ratio", 0.0)),
            object_fallback_reason=str(object_stats.get("fallback_reason", "")),
            controller_raw_points=int(controller_stats.get("raw_points", controller_raw_points)),
            controller_cap_points=int(controller_stats.get("cap_points", controller_cap_points)),
            controller_output_points=int(controller_stats.get("output_points", controller_cap_points)),
            controller_prefallback_points=int(controller_stats.get("filter_output_points", controller_cap_points)),
            controller_raw_retain_ratio=float(controller_stats.get("raw_retain_ratio", 0.0)),
            controller_fallback_reason=str(controller_stats.get("fallback_reason", "")),
            object_filter_cap=int(object_stats.get("cap", self.object_filter_budget.cap)),
            controller_filter_cap=int(controller_stats.get("cap", self.controller_filter_budget.cap)),
            filter_submit_fps=float(worker_stats["submit_fps"]),
            filter_output_fps=float(worker_stats["output_fps"]),
            filter_queue_drop=int(worker_stats["pending_replace_count"]),
            filter_busy=bool(worker_stats["busy"]),
        )

    # ------------------------------------------------------------------
    # PCD build: masked backprojection -> MaskedPcdPacket (+ headless writes)
    # ------------------------------------------------------------------
    def _headless_product_rows_gated(self) -> bool:
        """True while post-warmup frames must stay out of the chunk timeline.

        The gate carries its own deadline: --shape-prior-timeout-ms bounds how
        long formal rows may be withheld. On expiry rows resume so the chunk
        bridge's shape-prior wait/failure path reports loudly, instead of the
        row stream stalling silently on a hung prior.
        """
        writer = self.headless_capture_writer
        if writer is None or self._formal_timeline_gate_expired:
            return False
        profile = self._shape_prior_profile()
        gated = _formal_chunk_rows_gated(
            warmup_anchor_written=self._warmup_anchor_row_written,
            shape_prior_status=str(
                profile.get("shape_prior_status", shape_prior_warmup.STATUS_DISABLED)
            ),
        )
        if not gated:
            return False
        now_s = time.perf_counter()
        if self._formal_timeline_gate_started_s is None:
            self._formal_timeline_gate_started_s = now_s
        timeout_ms = int(
            getattr(
                self.args,
                "shape_prior_timeout_ms",
                shape_prior_warmup.DEFAULT_SHAPE_PRIOR_TIMEOUT_MS,
            )
        )
        if timeout_ms > 0 and (now_s - self._formal_timeline_gate_started_s) * 1000.0 >= float(timeout_ms):
            self._formal_timeline_gate_expired = True
            print(
                "[WARN] shape prior still not ready after --shape-prior-timeout-ms="
                f"{timeout_ms}; resuming formal chunk rows so the chunk bridge can "
                "surface the shape-prior wait/failure loudly.",
                flush=True,
            )
            return False
        return True

    def _write_headless_pcd_result(
        self,
        result: PcdBuildResult,
        tracker_packet: TrackerMarkerPacket | None = None,
        *,
        gated: bool | None = None,
    ) -> None:
        """Write headless PCD result.

        ``gated`` lets callers that already consulted
        :meth:`_headless_product_rows_gated` for this frame (to skip the
        tracker sidecar) reuse that single decision — the shape-prior worker
        flips the status asynchronously, so evaluating twice per frame could
        write a row whose query_trajectory sidecar was skipped.
        """
        if self.headless_capture_writer is None or result.depth_m is None:
            return
        if result.controller_pcd_mask is None or result.object_pcd_mask is None:
            return
        if gated is None:
            gated = self._headless_product_rows_gated()
        if gated:
            self._formal_timeline_gated_frames += 1
            return
        if self._formal_timeline_gated_frames and not self._formal_timeline_metadata_written:
            # First formal frame after the shape-prior wait: record the seam so
            # downstream tools can tell warmup frame 0 from output frame 1.
            self.headless_capture_writer.update_metadata(
                {
                    "formal_timeline_gated_frame_count": int(self._formal_timeline_gated_frames),
                    "formal_timeline_start_seq": int(result.packet.seq),
                }
            )
            self._formal_timeline_metadata_written = True
        if not self._warmup_anchor_row_written:
            # Mirror chunk_data_stream._row_ready_for_realtime_chunk_start: only
            # a chunk-ready row may claim the warmup frame-0 slot; invalid
            # startup rows keep writing and are trimmed by the bridge.
            self._warmup_anchor_row_written = (
                int(result.packet.controller_point_count) >= CONTROLLER_FINAL_COUNT
                and int(result.packet.object_point_count) > 0
            )
        self.headless_capture_writer.write_pcd(
            result.packet,
            depth_m=result.depth_m,
            mask_packet=result.mask_packet,
            controller_pcd_mask=result.controller_pcd_mask,
            object_pcd_mask=result.object_pcd_mask,
            pcd_stride=int(result.pcd_stride),
            pcd_mask_erode_pixels=int(result.pcd_mask_erode_pixels),
            object_pcd_mask_erode_pixels=int(result.object_pcd_mask_erode_pixels),
            controller_pcd_mask_erode_pixels=int(result.controller_pcd_mask_erode_pixels),
            tracker_packet=tracker_packet,
            stage_fps={
                "capture_fps": float(self.capture_stats.fps),
                "seg_fps": float(self.seg_stats.fps),
                "depth_fps": float(self.depth_stats.fps),
                "pcd_fps": float(self.pcd_stats.fps),
                "tracker_fps": float(self.tracker_stats.fps),
            },
            world_z_diagnostics=result.world_z_diagnostics,
            startup_hold_s=float(getattr(self, "_startup_hold_s", 0.0)),
        )

    def _build_pcd_packet_from_mask(
        self,
        mask_packet: MaskPacket,
        *,
        rng: np.random.Generator,
        require_filter_seq: bool = False,
    ) -> PcdBuildResult:
        """Build a masked point-cloud packet from a mask/depth pair."""
        start_s = time.perf_counter()
        assert self.ray_x is not None and self.ray_y is not None
        ray_x = self.ray_x
        ray_y = self.ray_y
        if mask_packet.depth_source in {"ffs", "ffs_remote"}:
            if mask_packet.depth_source == "ffs_remote" and self.args.ffs_remote_return in SPARSE_RETURN_TYPES:
                packet = self._compute_remote_sparse_pcd_packet(
                    mask_packet=mask_packet,
                    start_s=start_s,
                    rng=rng,
                    require_filter_seq=require_filter_seq,
                )
                return PcdBuildResult(packet=packet, depth_m=None, mask_packet=mask_packet)
            ffs_ms = 0.0
            ffs_align_ms = 0.0
            remote_rtt_ms = 0.0
            remote_server_total_ms = 0.0
            remote_request_kb = 0.0
            remote_response_kb = 0.0
            depth_convert_ms = 0.0
            (
                depth_m,
                ffs_ms,
                ffs_align_ms,
                remote_rtt_ms,
                remote_server_total_ms,
                remote_request_kb,
                remote_response_kb,
            ) = self._compute_external_ffs_depth_color_m(mask_packet)
        else:
            ffs_ms = 0.0
            ffs_align_ms = 0.0
            remote_rtt_ms = 0.0
            remote_server_total_ms = 0.0
            remote_request_kb = 0.0
            remote_response_kb = 0.0
            if mask_packet.depth_u16 is None:
                raise RuntimeError("PCD packet requires RGB-D depth")
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
        pcd_mask_erode_pixels = int(self.args.pcd_mask_erode_pixels)
        controller_erode_pixels = controller_pcd_mask_erode_pixels(self.args)
        object_erode_pixels = object_pcd_mask_erode_pixels(self.args)
        if controller_erode_pixels > 0:
            controller_mask = erode_binary_mask(controller_mask, erode_pixels=controller_erode_pixels)
        if object_erode_pixels > 0:
            object_mask = erode_binary_mask(object_mask, erode_pixels=object_erode_pixels)
        empty_pcd_timing = {
            "pcd_mask_intersection_ms": 0.0,
            "pcd_select_ms": 0.0,
            "pcd_point_cap_ms": 0.0,
            "pcd_backproject_ms": 0.0,
            "pcd_color_gather_ms": 0.0,
            "pcd_raw_points": 0.0,
            "pcd_cap_points": 0.0,
        }
        if controller_tracking_enabled(self.args):
            controller_xyz, controller_colors, controller_yx, controller_pcd_timing = backproject_masked_rgbd_profiled(
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
                return_yx=True,
            )
            if stride > 1:
                controller_yx = np.ascontiguousarray(controller_yx * int(stride), dtype=np.int64)
        else:
            controller_xyz = np.empty((0, 3), dtype=np.float32)
            controller_colors = np.empty((0, 3), dtype=np.uint8)
            controller_yx = np.empty((0, 2), dtype=np.int64)
            controller_pcd_timing = dict(empty_pcd_timing)
        if object_tracking_enabled(self.args):
            object_xyz, object_colors, object_yx, object_pcd_timing = backproject_masked_rgbd_profiled(
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
                return_yx=True,
            )
            if stride > 1:
                object_yx = np.ascontiguousarray(object_yx * int(stride), dtype=np.int64)
        else:
            object_xyz = np.empty((0, 3), dtype=np.float32)
            object_colors = np.empty((0, 3), dtype=np.uint8)
            object_yx = np.empty((0, 2), dtype=np.int64)
            object_pcd_timing = dict(empty_pcd_timing)
        controller_raw_points = int(controller_pcd_timing.get("pcd_raw_points", len(controller_xyz)))
        controller_cap_points = int(controller_pcd_timing.get("pcd_cap_points", len(controller_xyz)))
        object_raw_points = int(object_pcd_timing.get("pcd_raw_points", len(object_xyz)))
        object_cap_points = int(object_pcd_timing.get("pcd_cap_points", len(object_xyz)))
        render_controller_xyz = controller_xyz
        render_controller_colors = controller_colors
        render_controller_yx = controller_yx
        render_object_xyz = object_xyz
        render_object_colors = object_colors
        render_object_yx = object_yx
        filter_output: FilterOutput | None = None
        using_filtered = False

        if pcd_filter_enabled(self.args):
            if str(self.args.pcd_filter_mode) == "sync":
                filter_input = self._make_filter_input(
                    seq=mask_packet.seq,
                    object_xyz=object_xyz,
                    object_colors=object_colors,
                    object_yx=object_yx,
                    controller_xyz=controller_xyz,
                    controller_colors=controller_colors,
                    controller_yx=controller_yx,
                )
                self.filter_submit_stats.record()
                filter_output = self._filter_pcd_input(filter_input)
                self.filter_output_stats.record(filter_output.output_perf_s)
                render_controller_xyz = filter_output.controller_xyz
                render_controller_colors = filter_output.controller_rgb
                render_controller_yx = filter_output.controller_yx
                render_object_xyz = filter_output.object_xyz
                render_object_colors = filter_output.object_rgb
                # Keep XYZ/colors/YX aligned; downstream observation masks are
                # rebuilt from render_object_yx.
                render_object_yx = filter_output.object_yx
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
                        filter_matches = int(latest.seq) == int(mask_packet.seq)
                        if filter_matches or (
                            not bool(require_filter_seq)
                            and self._filter_output_is_fresh(packet_seq=mask_packet.seq, output=latest)
                        ):
                            render_controller_xyz = latest.controller_xyz
                            render_controller_colors = latest.controller_rgb
                            render_controller_yx = latest.controller_yx
                            render_object_xyz = latest.object_xyz
                            render_object_colors = latest.object_rgb
                            # Keep XYZ/colors/YX aligned; downstream observation
                            # masks are rebuilt from render_object_yx.
                            render_object_yx = latest.object_yx
                            using_filtered = True
                    if mask_packet.seq % int(self.args.filter_every_n) == 0:
                        if not worker.is_busy():
                            worker.submit_latest(
                                self._make_filter_input(
                                    seq=mask_packet.seq,
                                    object_xyz=object_xyz,
                                    object_colors=object_colors,
                                    object_yx=object_yx,
                                    controller_xyz=controller_xyz,
                                    controller_colors=controller_colors,
                                    controller_yx=controller_yx,
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
        render_controller_xyz = _transform_points_c2w(render_controller_xyz, self.table_c2w)
        render_object_xyz = _transform_points_c2w(render_object_xyz, self.table_c2w)
        hand_a_xyz = None
        hand_b_xyz = None
        if mask_packet.hand_a_mask is not None:
            hand_a_xyz = _select_points_by_yx_mask(
                render_controller_xyz,
                render_controller_yx,
                mask_packet.hand_a_mask,
            )
        if mask_packet.hand_b_mask is not None:
            hand_b_xyz = _select_points_by_yx_mask(
                render_controller_xyz,
                render_controller_yx,
                mask_packet.hand_b_mask,
            )
        world_z_diagnostics = build_world_z_diagnostics(
            object_xyz_m=render_object_xyz,
            controller_xyz_m=render_controller_xyz,
            hand_a_xyz_m=hand_a_xyz,
            hand_b_xyz_m=hand_b_xyz,
            table_z_m=TABLE_Z_M,
            thresholds_m=DEFAULT_TABLE_Z_DIAGNOSTIC_THRESHOLDS_M,
        )
        table_z_filter_stats: dict[str, Any] = {
            "enabled": bool(self.args.enable_table_z_filter),
            "threshold_m": float(self.args.table_z_filter_threshold_m),
            "table_z_above_direction": TABLE_Z_ABOVE_DIRECTION,
            "classes": str(self.args.table_z_filter_classes),
            "object": None,
            "controller": None,
        }
        if bool(self.args.enable_table_z_filter):
            classes = str(self.args.table_z_filter_classes)
            if classes in {TABLE_Z_FILTER_CLASS_OBJECT, TABLE_Z_FILTER_CLASS_BOTH}:
                (
                    render_object_xyz,
                    render_object_colors,
                    render_object_yx,
                    object_table_z_stats,
                ) = apply_table_z_filter_with_yx(
                    render_object_xyz,
                    render_object_colors,
                    render_object_yx,
                    enabled=True,
                    threshold_m=float(self.args.table_z_filter_threshold_m),
                    table_z_m=TABLE_Z_M,
                )
                table_z_filter_stats["object"] = object_table_z_stats
            if classes in {TABLE_Z_FILTER_CLASS_CONTROLLER, TABLE_Z_FILTER_CLASS_BOTH}:
                (
                    render_controller_xyz,
                    render_controller_colors,
                    render_controller_yx,
                    controller_table_z_stats,
                ) = apply_table_z_filter_with_yx(
                    render_controller_xyz,
                    render_controller_colors,
                    render_controller_yx,
                    enabled=True,
                    threshold_m=float(self.args.table_z_filter_threshold_m),
                    table_z_m=TABLE_Z_M,
                )
                table_z_filter_stats["controller"] = controller_table_z_stats
        world_z_diagnostics["runtime_table_z_filter"] = table_z_filter_stats
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
            coordinate_frame=self._pcd_coordinate_frame(),
            source_timestamp_s=mask_packet.source_timestamp_s,
            source_frame_index=mask_packet.source_frame_index,
            source_step=mask_packet.source_step,
        )
        return PcdBuildResult(
            packet=packet,
            depth_m=depth_m,
            mask_packet=mask_packet,
            controller_pcd_mask=controller_mask,
            object_pcd_mask=object_mask,
            object_observation_mask=_mask_from_yx(
                tuple(mask_packet.object_mask.shape[:2]),
                render_object_yx,
            ),
            pcd_stride=stride,
            pcd_mask_erode_pixels=pcd_mask_erode_pixels,
            object_pcd_mask_erode_pixels=object_erode_pixels,
            controller_pcd_mask_erode_pixels=controller_erode_pixels,
            world_z_diagnostics=world_z_diagnostics,
        )

    def _pcd_worker(self) -> None:
        """Return the PCD worker."""
        last_seq = -1
        rng = np.random.default_rng()
        while not self.stop_event.is_set():
            mask_packet = self.mask_slot.get_latest_after(last_seq)
            if mask_packet is None:
                time.sleep(0.001)
                continue
            last_seq = mask_packet.seq
            try:
                result = self._build_pcd_packet_from_mask(mask_packet, rng=rng)
            except Exception as exc:
                if not self.stop_event.is_set():
                    print(f"[WARN] PCD frame {mask_packet.seq} failed: {type(exc).__name__}: {exc}", flush=True)
                continue
            self._maybe_start_shape_prior_from_pcd_result(result)
            result = replace(result, packet=self._packet_with_shape_prior_state(result.packet))
            self.pcd_slot.put(result.packet)
            self._maybe_write_shape_prior_headless_result()
            self._write_headless_pcd_result(result)
            self.pcd_stats.record(result.packet.process_done_perf_s)

    # ------------------------------------------------------------------
    # Depth backends: profiling, local FFS, remote FFS, remote sparse quality
    # ------------------------------------------------------------------
    def _depth_profile_worker(self) -> None:
        """Return the depth profile worker."""
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
        """Compute external FFS depth color m."""
        if packet.depth_source == "ffs_remote":
            return self._compute_remote_ffs_depth_color_m(packet)
        depth_color_m, ffs_ms, ffs_align_ms = self._compute_ffs_depth_color_m(packet)
        return depth_color_m, ffs_ms, ffs_align_ms, 0.0, 0.0, 0.0, 0.0

    def _get_cached_local_ffs_depth(self, seq: int) -> tuple[np.ndarray, float, float] | None:
        """Return the get cached local FFS depth."""
        cached = self._local_ffs_depth_cache.get(int(seq))
        if cached is None:
            return None
        self._local_ffs_depth_cache.move_to_end(int(seq))
        return cached

    def _put_cached_local_ffs_depth(self, seq: int, value: tuple[np.ndarray, float, float]) -> None:
        """Return the put cached local FFS depth."""
        self._local_ffs_depth_cache[int(seq)] = value
        self._local_ffs_depth_cache.move_to_end(int(seq))
        while len(self._local_ffs_depth_cache) > DEFAULT_LOCAL_FFS_DEPTH_CACHE_FRAMES:
            self._local_ffs_depth_cache.popitem(last=False)

    def _compute_ffs_depth_color_m(self, packet: MaskPacket | FramePacket) -> tuple[np.ndarray, float, float]:
        """Compute FFS depth color m."""
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

        with self._local_ffs_lock:
            cached = self._get_cached_local_ffs_depth(int(packet.seq))
            if cached is not None:
                return cached

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
            result = (
                depth_color_m,
                _elapsed_ms(ffs_start_s, ffs_done_s),
                _elapsed_ms(align_start_s, align_done_s),
            )
            self._put_cached_local_ffs_depth(int(packet.seq), result)
            return result

    def _compute_remote_ffs_depth_color_m(
        self,
        packet: MaskPacket | FramePacket,
    ) -> tuple[np.ndarray, float, float, float, float, float, float]:
        """Compute remote FFS depth color m."""
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
        """Request remote FFS result."""
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
        """Return the warn if remote engine contract missing."""
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
        """Return the split sparse remote PCD."""
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
            """Build for label."""
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
        require_filter_seq: bool = False,
    ) -> MaskedPcdPacket:
        """Compute remote sparse PCD packet."""
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
                    filter_matches = int(latest.seq) == int(mask_packet.seq)
                    if filter_matches or (
                        not bool(require_filter_seq)
                        and self._filter_output_is_fresh(packet_seq=mask_packet.seq, output=latest)
                    ):
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
        render_controller_xyz = _transform_points_c2w(render_controller_xyz, self.table_c2w)
        render_object_xyz = _transform_points_c2w(render_object_xyz, self.table_c2w)
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
            coordinate_frame=self._pcd_coordinate_frame(),
            source_timestamp_s=mask_packet.source_timestamp_s,
            source_frame_index=mask_packet.source_frame_index,
            source_step=mask_packet.source_step,
        )

    def _remote_quality_mask_u8(self, packet: MaskPacket) -> np.ndarray:
        """Convert a boolean quality mask to uint8 image form."""
        mask = np.zeros(tuple(packet.object_mask.shape), dtype=np.uint8)
        if controller_tracking_enabled(self.args):
            mask[np.asarray(packet.controller_mask, dtype=bool)] = CONTROLLER_ID
        if object_tracking_enabled(self.args):
            mask[np.asarray(packet.object_mask, dtype=bool)] = OBJECT_ID
        return np.ascontiguousarray(mask)

    def _request_remote_quality(self, packet: MaskPacket | FramePacket, *, mask_u8: np.ndarray | None) -> RemoteFfsQualityPacket:
        """Request remote quality."""
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
        """Return the remote FFS quality worker."""
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

    # ------------------------------------------------------------------
    # Debug logging
    # ------------------------------------------------------------------
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
        tracker_packet: TrackerMarkerPacket | None = None,
        strict_sync: bool = False,
        waiting_for_pair: bool = False,
    ) -> None:
        """Return the emit debug line."""
        filter_info = filter_telemetry or PcdFilterTelemetry()
        if bool(strict_sync):
            paired_seq = str(int(seq)) if tracker_packet is not None else "none"
            tracker_seq = str(int(tracker_packet.seq)) if tracker_packet is not None else "none"
        else:
            paired_seq = "-1"
            tracker_seq = str(int(tracker_packet.seq)) if tracker_packet is not None else "-1"
        tracker_model_ms = float(tracker_packet.model_ms) if tracker_packet is not None else 0.0
        tracker_e2e_ms = float(tracker_packet.e2e_ms) if tracker_packet is not None else 0.0
        tracker_hand_a = int(tracker_packet.hand_a_query_count) if tracker_packet is not None else 0
        tracker_hand_b = int(tracker_packet.hand_b_query_count) if tracker_packet is not None else 0
        tracker_object = int(tracker_packet.object_query_count) if tracker_packet is not None else 0
        lossless_enabled = bool(strict_sync) and self._lossless_enabled()
        if lossless_enabled:
            frame_queue = self.lossless_frame_queue.stats
            pcd_queue = self.lossless_pcd_mask_queue.stats
            tracker_queue = self.lossless_tracker_mask_queue.stats
            pair_output_queue = self.lossless_pair_output_queue.stats
            pairer = self.same_seq_pairer.stats
            lossless_debug = (
                f"lossless=1 "
                f"lossless_input_fps={self._lossless_input_fps():.1f} "
                f"lossless_max_backlog={self.lossless_max_backlog_frames} "
                f"lossless_frame_q={frame_queue.size} "
                f"lossless_mask_pcd_q={pcd_queue.size} "
                f"lossless_mask_tracker_q={tracker_queue.size} "
                f"lossless_pair_output_q={pair_output_queue.size} "
                f"lossless_pairer_expected={pairer.expected_seq} "
                f"lossless_pairer_pending_pcd={pairer.pending_pcd} "
                f"lossless_pairer_pending_tracker={pairer.pending_tracker} "
                f"lossless_offered={self._lossless_offered_frames} "
                f"lossless_segmented={self._lossless_segmented_frames} "
                f"lossless_pcd_results={self._lossless_pcd_results} "
                f"lossless_tracker_results={self._lossless_tracker_results} "
                f"lossless_pairs={self._lossless_pairs_emitted} "
            )
        else:
            lossless_debug = "lossless=0 "
        print(
            "[masked-edgetam-debug] "
            f"seq={int(seq)} "
            f"strict_sync={int(bool(strict_sync))} "
            f"paired_seq={paired_seq} "
            f"tracker_seq={tracker_seq} "
            f"waiting_for_pair={int(bool(waiting_for_pair))} "
            f"{lossless_debug}"
            f"capture_fps={self.capture_stats.fps:.1f} "
            f"seg_fps={self.seg_stats.fps:.1f} "
            f"depth_fps={self.depth_stats.fps:.1f} "
            f"remote_quality_fps={self.remote_quality_stats.fps:.1f} "
            f"pcd_fps={self.pcd_stats.fps:.1f} "
            f"tracker_fps={self.tracker_stats.fps:.1f} "
            f"tracker_hand_a_queries={tracker_hand_a} "
            f"tracker_hand_b_queries={tracker_hand_b} "
            f"tracker_object_queries={tracker_object} "
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
            f"e2e_latency_ms={timing.receive_to_render_ms:.2f} "
            f"tracker_model_ms={tracker_model_ms:.2f} "
            f"tracker_e2e_ms={tracker_e2e_ms:.2f} "
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
            f"object_filter_cap_limit={int(filter_info.object_filter_cap)} "
            f"object_filter_cap_points={int(filter_info.object_cap_points)} "
            f"object_filter_prefallback_points={int(filter_info.object_prefallback_points)} "
            f"object_filter_output_points={int(filter_info.object_output_points)} "
            f"object_filter_raw_retain_ratio={filter_info.object_raw_retain_ratio:.3f} "
            f"object_filter_fallback_reason={filter_info.object_fallback_reason or 'none'} "
            f"controller_filter_input_points={int(filter_info.controller_raw_points)} "
            f"controller_filter_cap_limit={int(filter_info.controller_filter_cap)} "
            f"controller_filter_cap_points={int(filter_info.controller_cap_points)} "
            f"controller_filter_prefallback_points={int(filter_info.controller_prefallback_points)} "
            f"controller_filter_output_points={int(filter_info.controller_output_points)} "
            f"controller_filter_raw_retain_ratio={filter_info.controller_raw_retain_ratio:.3f} "
            f"controller_filter_fallback_reason={filter_info.controller_fallback_reason or 'none'} "
            f"controller_points={int(controller_points)} "
            f"object_points={int(object_points)} "
            f"dropped_capture={int(dropped_capture_frames)} "
            f"dropped_seg={int(dropped_seg_frames)} "
            f"dropped_pcd={self.paired_render_slot.dropped_count if bool(strict_sync) else self.pcd_slot.dropped_count}",
            flush=True,
        )

    def _headless_debug_worker(self) -> None:
        """Return the headless debug worker."""
        last_logged_seq = -1
        last_logged_pair_seq = -1
        last_logged_waiting_seq = -1
        while not self.stop_event.is_set():
            now_s = time.perf_counter()
            if now_s - self._last_debug_log_s < DEBUG_LOG_INTERVAL_S:
                time.sleep(0.05)
                continue
            self._last_debug_log_s = now_s
            if tracker_enabled(self.args):
                pair = self.paired_render_slot.get_latest_after(last_logged_pair_seq)
                if pair is not None:
                    last_logged_pair_seq = pair.seq
                    pcd_packet = pair.pcd_packet
                    timing = replace(
                        pcd_packet.timing,
                        receive_to_render_ms=_elapsed_ms(pcd_packet.receive_perf_s, pair.tracker_packet.process_done_perf_s),
                    )
                    self._emit_debug_line(
                        seq=pcd_packet.seq,
                        timing=timing,
                        controller_points=pcd_packet.controller_point_count,
                        object_points=pcd_packet.object_point_count,
                        dropped_capture_frames=pcd_packet.dropped_capture_frames,
                        dropped_seg_frames=pcd_packet.dropped_seg_frames,
                        filter_telemetry=pcd_packet.filter_telemetry,
                        tracker_packet=pair.tracker_packet,
                        strict_sync=True,
                        waiting_for_pair=True,
                    )
                    continue
                mask_packet = self.mask_slot.get_latest_after(last_logged_waiting_seq)
                if mask_packet is not None:
                    last_logged_waiting_seq = mask_packet.seq
                    timing = replace(
                        mask_packet.timing,
                        receive_to_render_ms=_elapsed_ms(mask_packet.receive_perf_s, now_s),
                    )
                    self._emit_debug_line(
                        seq=mask_packet.seq,
                        timing=timing,
                        dropped_capture_frames=mask_packet.dropped_capture_frames,
                        dropped_seg_frames=self.mask_slot.dropped_count,
                        strict_sync=True,
                        waiting_for_pair=True,
                    )
                    continue
                frame = self.capture_slot.get_latest_after(last_logged_waiting_seq)
                if frame is not None:
                    last_logged_waiting_seq = frame.seq
                    timing = replace(frame.timing, receive_to_render_ms=_elapsed_ms(frame.receive_perf_s, now_s))
                    self._emit_debug_line(
                        seq=frame.seq,
                        timing=timing,
                        dropped_capture_frames=self.capture_slot.dropped_count,
                        strict_sync=True,
                        waiting_for_pair=True,
                    )
                continue
            pcd_packet = self.pcd_slot.get_latest_after(last_logged_seq)
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

    def _maybe_log_debug(
        self,
        *,
        packet: MaskedPcdPacket,
        timing: PipelineTiming,
        now_s: float,
        tracker_packet: TrackerMarkerPacket | None = None,
        strict_sync: bool = False,
        waiting_for_pair: bool = False,
    ) -> None:
        """Maybe start or update log debug."""
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
            tracker_packet=tracker_packet,
            strict_sync=strict_sync,
            waiting_for_pair=waiting_for_pair,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    """Run the command-line entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        apply_demo_preset(args)
        validate_args(args)
        return MainDataProcessingDemo(args).run()
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"{parser.prog}: error: {exc}\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
