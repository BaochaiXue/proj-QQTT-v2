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
import threading
import time
from typing import Any, Callable

import numpy as np


def _resolve_repo_root() -> Path:
    candidates: list[Path] = []
    env_root = os.environ.get("QQTT_REPO_ROOT")
    if env_root:
        candidates.append(Path(env_root))
    candidates.extend([Path(__file__).resolve().parents[2], Path(__file__).resolve().parents[1], Path.cwd()])
    for candidate in candidates:
        root = candidate.expanduser().resolve()
        if (root / "data_process").is_dir() and (root / "demo_v2").is_dir():
            return root
    return Path(__file__).resolve().parents[2]


REPO_ROOT = _resolve_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qqtt.demo.realtime_single_camera_pointcloud import (  # noqa: E402
    CameraIntrinsics,
    CoalescedPostGate,
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
    parse_profile,
    resolve_serial,
    rs_extrinsics_to_matrix,
    rs_intrinsics_to_matrix,
    rs_translation_norm,
    validate_ffs_paths,
    warm_up_numba_ffs_align,
)
from qqtt.demo.render_fastpath import Open3DSceneTensorLayer  # noqa: E402
from qqtt.demo.pcd_filter_fast import (  # noqa: E402
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
from qqtt.demo.tracking_overlay_render import lift_tracks_yx_to_world  # noqa: E402
from qqtt.tracking.backends.point_tracker_adapter import (  # noqa: E402
    TRACKER_BACKEND_NONE,
    TRACKER_BACKEND_TAPNEXTPP,
    TRACKER_BACKENDS,
    PointTrackerAdapterConfig,
    build_point_tracker_adapter_factory,
    normalize_tracker_backend,
)
from qqtt.tracking.sampling import PHYSTWIN_DENSE_QUERY_POINTS, sample_phystwin_dense  # noqa: E402


DEFAULT_MODEL_ID = "yonigozlan/EdgeTAM-hf"
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
DEFAULT_FAKE_LIVE_CASE = Path("data_collect/sloth_both_eval_2min_e45_g35_20260614_155543")
PCD_MODES = ("masked", "none")
DEFAULT_PCD_MODE = "masked"
RENDER_MODES = ("pointcloud", "none")
DEFAULT_RENDER_MODE = "pointcloud"
DEFAULT_RENDER_MAX_POINTS_PER_LAYER = 5000
VIEW_MODES = ("orbit", "camera")
DEFAULT_VIEW_MODE = "orbit"
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
DEFAULT_OBJECT_FILTER = PCD_FILTER_ENHANCED_PT
DEFAULT_CONTROLLER_FILTER = PCD_FILTER_ENHANCED_PT
DEFAULT_OBJECT_FILTER_KEEP_COMPONENTS = 1
DEFAULT_CONTROLLER_FILTER_KEEP_COMPONENTS = 2
DEFAULT_OBJECT_FILTER_MIN_RETAIN_RATIO = 0.0
DEFAULT_CONTROLLER_FILTER_MIN_RETAIN_RATIO = 0.5
DEFAULT_OBJECT_FILTER_MIN_RAW_RETAIN_RATIO = 0.0
DEFAULT_CONTROLLER_FILTER_MIN_RAW_RETAIN_RATIO = 0.5
DEFAULT_FILTER_MAX_AGE_FRAMES = 3
DEFAULT_EDGETAM_LIVE_SESSION_KEEP_FRAMES = 64
CONTROLLER_ID = 1
OBJECT_ID = 2
OBJECT_LABELS = {CONTROLLER_ID: "controller", OBJECT_ID: "object"}
CONTROLLER_COLOR_RGB = (255, 96, 32)
OBJECT_COLOR_RGB = (64, 180, 255)
GEOMETRY_CONTROLLER = "masked_edgetam_controller"
GEOMETRY_OBJECT = "masked_edgetam_object"
GEOMETRY_TRACKER = "tapnextpp_tracker_markers"
COORDINATE_FRAME = "camera_color_frame"
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
DEFAULT_TRACKER_QUERY_COUNT = 4096
DEFAULT_TRACKER_SEED = 42
DEFAULT_TRACKER_MARKER_COLOR_RGB = (255, 0, 0)
DEFAULT_TRACKER_MARKER_POINT_SIZE = 8.0
DEBUG_LOG_INTERVAL_S = 1.0
FATAL_HUD_PREFIX = "FATAL WORKER ERROR"
WARMUP_HUD_TEXT = (
    "System warming up. Keep one steady pose.\n"
    "SAM3.1 first-frame initialization and compiled EdgeTAM startup are running..."
)


def _first_existing_path(candidates: list[Path]) -> Path:
    for candidate in candidates:
        path = candidate.expanduser()
        if path.exists():
            return path
    return candidates[0]


DEFAULT_TAPNET_REPO_DIR = _first_existing_path(
    [
        REPO_ROOT / "external" / "tapnet",
        REPO_ROOT.parent / "proj-QQTT-v2" / "external" / "tapnet",
    ]
)
DEFAULT_TAPNEXTPP_CHECKPOINT = _first_existing_path(
    [
        REPO_ROOT / "checkpoints" / "tapnextpp" / "tapnextpp_ckpt.pt",
        REPO_ROOT.parent / "proj-QQTT-v2" / "checkpoints" / "tapnextpp" / "tapnextpp_ckpt.pt",
    ]
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
class FatalWorkerError:
    stage: str
    exc_type: str
    message: str

    def log_message(self) -> str:
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
        self.effective_fps = self._resolve_replay_fps(float(replay_fps), metadata)
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

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def steps(self) -> list[int]:
        return [frame.step for frame in self.frames]

    def make_runtime(self) -> RealtimeCameraRuntime:
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
        wait_ms: float = 0.0,
        receive_perf_s: float | None = None,
        frame_copy_ms: float | None = None,
    ) -> FramePacket:
        ref = self.frames[int(seq)]
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
            seq=int(seq),
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
        )

    def _camera_matrix(self, metadata: dict[str, Any], key: str, *, fallback_key: str | None = None) -> np.ndarray:
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
        values = metadata.get(key)
        if not isinstance(values, list) or self.camera_index >= len(values) or values[self.camera_index] is None:
            raise ValueError(f"recording metadata missing {key} for camera {self.camera_index}")
        matrix = np.asarray(values[self.camera_index], dtype=np.float32)
        if matrix.shape != (4, 4):
            raise ValueError(f"recording metadata {key}[{self.camera_index}] must be 4x4")
        return np.ascontiguousarray(matrix, dtype=np.float32)

    def _camera_baseline(self, metadata: dict[str, Any]) -> float:
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
        values = metadata.get(key)
        if not isinstance(values, list) or self.camera_index >= len(values) or values[self.camera_index] is None:
            raise ValueError(f"recording metadata missing {key} for camera {self.camera_index}")
        value = float(values[self.camera_index])
        if value <= 0.0:
            raise ValueError(f"recording metadata {key}[{self.camera_index}] must be positive")
        return value

    def _camera_string(self, metadata: dict[str, Any], key: str, *, default: str) -> str:
        values = metadata.get(key)
        if isinstance(values, list) and self.camera_index < len(values) and values[self.camera_index] is not None:
            return str(values[self.camera_index])
        return default

    def _resolve_dimensions(self, metadata: dict[str, Any]) -> tuple[int, int]:
        wh = metadata.get("WH")
        if not isinstance(wh, list) or len(wh) != 2:
            raise ValueError("recording metadata missing WH")
        width = int(wh[0])
        height = int(wh[1])
        if width <= 0 or height <= 0:
            raise ValueError("recording metadata WH must be positive")
        return width, height

    def _resolve_replay_fps(self, replay_fps: float, metadata: dict[str, Any]) -> float:
        fps = float(metadata.get("fps", 0.0)) if replay_fps <= 0.0 else float(replay_fps)
        if fps <= 0.0:
            return 30.0
        return fps

    def _build_frame_refs(self, camera_recording: dict[str, Any]) -> list[RecordedRgbdFrameRef]:
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
        try:
            from PIL import Image

            with Image.open(path) as image:
                rgb = np.asarray(image.convert("RGB"))
        except Exception as exc:
            raise ValueError(f"failed to load recording color frame {path}: {exc}") from exc
        return np.ascontiguousarray(rgb[:, :, ::-1], dtype=np.uint8)

    def _load_depth_u16(self, path: Path) -> np.ndarray:
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
class TrackerMarkerPacket:
    seq: int
    marker_xyz_m: np.ndarray
    marker_colors_rgb_u8: np.ndarray
    query_points_yx: np.ndarray
    tracks_yx: np.ndarray
    visibility: np.ndarray
    query_is_object: np.ndarray
    query_is_controller: np.ndarray
    receive_perf_s: float
    process_done_perf_s: float
    query_count: int
    model_ms: float = 0.0
    lift_ms: float = 0.0
    e2e_ms: float = 0.0
    backend: str = TRACKER_BACKEND_TAPNEXTPP
    display_scope: str = DEFAULT_TRACKER_DISPLAY_SCOPE

    @property
    def marker_count(self) -> int:
        return int(self.marker_xyz_m.shape[0])


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


def _is_replay_input_source(input_source: str) -> bool:
    return str(input_source) in {INPUT_SOURCE_FAKE_LIVE, INPUT_SOURCE_RECORDING}


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
        "--input-source",
        choices=INPUT_SOURCES,
        default=INPUT_SOURCE_LIVE,
        help=(
            "Frame source. fake-live replays a raw single-camera data_collect case at camera cadence; "
            "recording is kept as a compatibility alias."
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
        help="Replay FPS for --input-source recording or fake-live. Use 0 to read metadata fps.",
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
        "--view-mode",
        choices=VIEW_MODES,
        default=DEFAULT_VIEW_MODE,
        help="Initial Open3D view. orbit starts from a third-person view; camera uses RealSense color intrinsics.",
    )
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
        help="Open3D point size for TAPNext++ marker layer.",
    )
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
        "--edgetam-live-session-keep-frames",
        type=int,
        default=DEFAULT_EDGETAM_LIVE_SESSION_KEEP_FRAMES,
        help="Keep this many recent streamed EdgeTAM frames/outputs in live session state; 0 disables pruning.",
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
        "--render-max-points-per-layer",
        type=int,
        default=DEFAULT_RENDER_MAX_POINTS_PER_LAYER,
        help="Final Open3D display cap per semantic PCD layer. 0 renders every PCD point.",
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
    parser.add_argument("--object-filter", choices=PCD_FILTERS, default=DEFAULT_OBJECT_FILTER)
    parser.add_argument("--controller-filter", choices=PCD_FILTERS, default=DEFAULT_CONTROLLER_FILTER)
    parser.add_argument("--object-filter-cap", type=int, default=20_000)
    parser.add_argument("--controller-filter-cap", type=int, default=20_000)
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
    parser.add_argument("--point-size", type=float, default=2.0, help="Open3D point size.")
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
    if args.input_source not in INPUT_SOURCES:
        raise ValueError(f"--input-source must be one of {', '.join(INPUT_SOURCES)}")
    if args.depth_source not in DEPTH_SOURCES:
        raise ValueError(f"--depth-source must be one of {', '.join(DEPTH_SOURCES)}")
    if str(args.view_mode) not in VIEW_MODES:
        raise ValueError(f"--view-mode must be one of {', '.join(VIEW_MODES)}")
    if float(args.replay_fps) < 0.0:
        raise ValueError("--replay-fps must be >= 0")
    if args.input_source == INPUT_SOURCE_FAKE_LIVE and args.recording_case is None:
        args.recording_case = DEFAULT_FAKE_LIVE_CASE
    if _is_replay_input_source(str(args.input_source)):
        if args.recording_case is None:
            raise ValueError(f"--input-source {args.input_source} requires --recording-case or --fake-live-case")
    elif args.recording_case is not None:
        raise ValueError("--recording-case/--fake-live-case requires --input-source recording or fake-live")
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
    if int(args.edgetam_live_session_keep_frames) < 0:
        raise ValueError("--edgetam-live-session-keep-frames must be >= 0")
    if int(args.render_max_points_per_layer) < 0:
        raise ValueError("--render-max-points-per-layer must be >= 0")
    if args.point_size <= 0:
        raise ValueError("--point-size must be positive")
    if args.pcd_filter_mode not in PCD_FILTER_MODES:
        raise ValueError(f"--pcd-filter-mode must be one of {', '.join(PCD_FILTER_MODES)}")
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
    if args.render_mode == "pointcloud" and args.pcd_mode == "none":
        raise ValueError("--render-mode pointcloud requires --pcd-mode masked")
    args.tracker_backend = normalize_tracker_backend(str(args.tracker_backend))
    if int(args.tracker_query_count) < 0:
        raise ValueError("--tracker-query-count must be >= 0")
    if int(args.tracker_overlay_max_points) < 0:
        raise ValueError("--tracker-overlay-max-points must be >= 0")
    if float(args.tracker_marker_point_size) <= 0:
        raise ValueError("--tracker-marker-point-size must be positive")
    if tracker_enabled(args):
        if args.tracker_backend != TRACKER_BACKEND_TAPNEXTPP:
            raise ValueError("single-camera tracker overlay currently supports only tapnextpp")
        if args.track_mode != TRACK_MODE_CONTROLLER_OBJECT:
            raise ValueError("--tracker-backend tapnextpp requires --track-mode controller-object")
        if args.render_mode != "pointcloud":
            raise ValueError("--tracker-backend tapnextpp requires --render-mode pointcloud")
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
    return str(track_mode) in {TRACK_MODE_CONTROLLER_OBJECT, TRACK_MODE_CONTROLLER_ONLY}


def object_tracking_enabled(args_or_track_mode: argparse.Namespace | str) -> bool:
    track_mode = args_or_track_mode if isinstance(args_or_track_mode, str) else args_or_track_mode.track_mode
    return str(track_mode) in {TRACK_MODE_CONTROLLER_OBJECT, TRACK_MODE_OBJECT_ONLY}


def object_id_labels(track_mode: str = DEFAULT_TRACK_MODE) -> dict[int, str]:
    if track_mode == TRACK_MODE_NONE:
        return {}
    if track_mode == TRACK_MODE_OBJECT_ONLY:
        return {OBJECT_ID: OBJECT_LABELS[OBJECT_ID]}
    if track_mode == TRACK_MODE_CONTROLLER_ONLY:
        return {CONTROLLER_ID: OBJECT_LABELS[CONTROLLER_ID]}
    if track_mode == TRACK_MODE_CONTROLLER_OBJECT:
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


def _union_masks(masks: list[np.ndarray], *, label: str) -> np.ndarray:
    if not masks:
        raise RuntimeError(f"SAM3.1 did not produce a mask for label {label!r}")
    output = np.zeros_like(masks[0], dtype=bool)
    for mask in masks:
        if mask.shape != output.shape:
            raise RuntimeError("SAM3.1 masks for one label have inconsistent shapes")
        output |= mask.astype(bool)
    return np.ascontiguousarray(output)


def release_sam31_runtime_resources(device: str = DEFAULT_DEVICE) -> float:
    started_s = time.perf_counter()
    helper = sys.modules.get("scripts.harness.sam31_mask_helper")
    clear_cache = getattr(helper, "clear_sam31_image_processor_cache", None) if helper is not None else None
    if clear_cache is not None:
        clear_cache()
    autocast_context = getattr(helper, "_CUDA_AUTOCAST_CONTEXT", None) if helper is not None else None
    if autocast_context is not None:
        try:
            autocast_context.__exit__(None, None, None)
        except Exception as exc:
            print(f"[WARN] SAM3.1 autocast cleanup failed: {type(exc).__name__}: {exc}", flush=True)
        if helper is not None:
            setattr(helper, "_CUDA_AUTOCAST_CONTEXT", None)
            contexts = getattr(helper, "_CUDA_AUTOCAST_CONTEXTS_BY_THREAD", None)
            if isinstance(contexts, dict):
                contexts.clear()

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
    return _elapsed_ms(started_s, time.perf_counter())


def trim_sam31_cuda_allocator(device: str = DEFAULT_DEVICE) -> float:
    started_s = time.perf_counter()
    gc.collect()
    try:
        import torch  # noqa: PLC0415

        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception as exc:
        print(f"[WARN] SAM3.1 CUDA trim failed: {type(exc).__name__}: {exc}", flush=True)
    return _elapsed_ms(started_s, time.perf_counter())


def run_sam31_first_frame_masks(color_bgr: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    from scripts.harness.sam31_mask_helper import parse_text_prompts, run_image_segmentation

    prompt_labels = []
    if object_tracking_enabled(args):
        prompt_labels.append(str(args.object_prompt))
    if controller_tracking_enabled(args):
        prompt_labels.append(str(args.controller_prompt))
    if not prompt_labels:
        empty = np.zeros(tuple(color_bgr.shape[:2]), dtype=bool)
        return empty, empty
    text_prompt = ",".join(prompt_labels)
    keep_runtime_until_all_cameras_init = bool(
        getattr(args, "sam31_keep_runtime_until_all_cameras_init", False)
    )
    try:
        result = run_image_segmentation(
            image=_bgr_to_pil_rgb(color_bgr),
            text_prompt=text_prompt,
            checkpoint_path=None,
            compile_model=False,
            max_num_objects=16,
            device=str(args.device),
            reuse_model=bool(getattr(args, "sam31_cache_init_model", False)),
        )
        setattr(args, "_sam31_last_timing_ms", result.get("timing_ms", {}))
    finally:
        if keep_runtime_until_all_cameras_init:
            trim_ms = trim_sam31_cuda_allocator(str(args.device))
            setattr(args, "_sam31_last_trim_cleanup_ms", float(trim_ms))
        else:
            release_ms = release_sam31_runtime_resources(str(args.device))
            setattr(args, "_sam31_last_release_cleanup_ms", float(release_ms))

    masks_by_label = result["masks_by_label"]
    object_mask: np.ndarray | None = None
    controller_mask: np.ndarray | None = None
    if object_tracking_enabled(args):
        object_label = parse_text_prompts(str(args.object_prompt))[0]
        object_mask = _union_masks(
            list(masks_by_label.get(object_label, [])),
            label=args.object_prompt,
        )
    if controller_tracking_enabled(args):
        controller_label = parse_text_prompts(str(args.controller_prompt))[0]
        controller_masks = list(masks_by_label.get(controller_label, []))
        controller_mask = _union_masks(controller_masks, label=args.controller_prompt)
    if object_mask is None and controller_mask is None:
        empty = np.zeros(tuple(color_bgr.shape[:2]), dtype=bool)
        return empty, empty
    if object_mask is None:
        object_mask = np.zeros_like(controller_mask, dtype=bool)
    if controller_mask is None:
        return np.zeros_like(object_mask, dtype=bool), object_mask
    return controller_mask, object_mask


def resolve_initial_masks(frame: FramePacket, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    expected_shape = tuple(frame.color_bgr.shape[:2])
    if args.init_mode == "saved-masks":
        object_mask = (
            load_binary_mask(args.object_init_mask, expected_shape=expected_shape)
            if object_tracking_enabled(args)
            else None
        )
        controller_mask = (
            load_binary_mask(args.controller_init_mask, expected_shape=expected_shape)
            if controller_tracking_enabled(args)
            else None
        )
        if object_mask is None and controller_mask is None:
            empty = np.zeros(expected_shape, dtype=bool)
            return empty, empty
        if object_mask is None:
            object_mask = np.zeros_like(controller_mask, dtype=bool)
        if controller_mask is None:
            controller_mask = np.zeros_like(object_mask, dtype=bool)
        return controller_mask, object_mask
    if args.init_mode == "sam31-first-frame":
        controller_mask, object_mask = run_sam31_first_frame_masks(frame.color_bgr, args)
        if controller_mask.shape != expected_shape or object_mask.shape != expected_shape:
            raise RuntimeError("SAM3.1 frame-0 masks do not match captured frame shape")
        return controller_mask, object_mask
    raise ValueError(f"unsupported init mode: {args.init_mode}")


def tracker_enabled(args: argparse.Namespace) -> bool:
    return normalize_tracker_backend(str(getattr(args, "tracker_backend", TRACKER_BACKEND_NONE))) != TRACKER_BACKEND_NONE


def _camera_intrinsics_matrix(intrinsics: CameraIntrinsics) -> np.ndarray:
    return np.array(
        [
            [float(intrinsics.fx), 0.0, float(intrinsics.cx)],
            [0.0, float(intrinsics.fy), float(intrinsics.cy)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _tracker_union_mask(mask_packet: MaskPacket) -> np.ndarray:
    controller = np.asarray(mask_packet.controller_mask, dtype=bool)
    obj = np.asarray(mask_packet.object_mask, dtype=bool)
    if controller.shape != obj.shape:
        raise ValueError("controller/object masks must share a shape")
    return np.logical_or(controller, obj)


def _classify_query_points_yx(
    query_points_yx: np.ndarray,
    *,
    object_mask: np.ndarray,
    controller_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
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


def _tracker_display_visibility(
    visibility: np.ndarray,
    *,
    query_is_object: np.ndarray,
    query_is_controller: np.ndarray,
    display_scope: str,
) -> np.ndarray:
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


def _select_visible_spread_indices(tracks_yx: np.ndarray, visibility: np.ndarray, *, max_points: int) -> np.ndarray:
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


def cap_render_points(
    points_xyz_m: np.ndarray,
    colors_rgb_u8: np.ndarray,
    *,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    if int(max_points) < 0:
        raise ValueError("max_points must be >= 0")
    point_count = int(points_xyz_m.shape[0])
    if int(colors_rgb_u8.shape[0]) != point_count:
        raise ValueError("points and colors must have the same length")
    if int(max_points) == 0 or point_count <= int(max_points):
        return points_xyz_m, colors_rgb_u8
    points = np.asarray(points_xyz_m, dtype=np.float32).reshape(-1, 3)
    finite = np.isfinite(points).all(axis=1)
    finite_indices = np.flatnonzero(finite)
    if len(finite_indices) > 0:
        finite_points = points[finite_indices]
        mins = finite_points.min(axis=0)
        spans = finite_points.max(axis=0) - mins
        safe_spans = np.where(spans > np.float32(1e-6), spans, np.float32(1.0))
        quantized = np.floor((finite_points - mins) / safe_spans * np.float32(1023.0)).astype(np.int32)
        ordered = finite_indices[np.lexsort((quantized[:, 2], quantized[:, 1], quantized[:, 0]))]
        if len(ordered) >= int(max_points):
            indices = ordered[np.linspace(0, len(ordered) - 1, int(max_points), dtype=np.int64)]
        else:
            extras = np.flatnonzero(~finite)
            fill_count = int(max_points) - len(ordered)
            indices = np.concatenate([ordered, extras[:fill_count]]).astype(np.int64, copy=False)
    else:
        indices = np.linspace(0, point_count - 1, int(max_points), dtype=np.int64)
    return (
        np.ascontiguousarray(points_xyz_m[indices], dtype=np.float32),
        np.ascontiguousarray(colors_rgb_u8[indices], dtype=np.uint8),
    )


def _latest_tracker_arrays(result: Any) -> tuple[np.ndarray, np.ndarray]:
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


def _solid_tracker_colors(point_count: int) -> np.ndarray:
    if int(point_count) <= 0:
        return np.empty((0, 3), dtype=np.uint8)
    rgb = np.asarray(DEFAULT_TRACKER_MARKER_COLOR_RGB, dtype=np.uint8).reshape(1, 3)
    return np.repeat(rgb, int(point_count), axis=0)


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
        self.tracker_marker_slot: LatestSlot[TrackerMarkerPacket] = LatestSlot()
        self.stop_event = threading.Event()
        self._threads: list[threading.Thread] = []
        self._request_render_update: Callable[[], None] = lambda: None
        self.capture_stats = StageStats()
        self.seg_stats = StageStats()
        self.depth_stats = StageStats()
        self.remote_quality_stats = StageStats()
        self.pcd_stats = StageStats()
        self.tracker_stats = StageStats()
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
        self.recording_source: RecordedRgbdFrameSource | None = None
        self._recording_first_frame_segmented = threading.Event()
        self._tracker_query_points_yx: np.ndarray | None = None
        self._tracker_query_is_object: np.ndarray | None = None
        self._tracker_query_is_controller: np.ndarray | None = None
        self._warned_remote_engine_contract = False
        self._fatal_error_lock = threading.Lock()
        self._fatal_error: FatalWorkerError | None = None

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

    def _fatal_error_snapshot(self) -> FatalWorkerError | None:
        with self._fatal_error_lock:
            return self._fatal_error

    def _record_fatal_worker_error(self, stage: str, exc: BaseException) -> FatalWorkerError:
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
            self._request_render_update()
        return fatal

    def _format_fatal_hud(self, fatal: FatalWorkerError) -> str:
        return (
            f"{FATAL_HUD_PREFIX}\n"
            f"{fatal.stage} failed\n"
            f"{fatal.exc_type}: {fatal.message}\n"
            "Closing Open3D viewer; check the terminal log for details."
        )

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
        if _is_replay_input_source(str(self.args.input_source)):
            self.recording_source = RecordedRgbdFrameSource(
                self.args.recording_case,
                replay_fps=float(self.args.replay_fps),
                depth_source=str(self.args.depth_source),
            )
            self.width = self.recording_source.width
            self.height = self.recording_source.height
            self.runtime = self.recording_source.make_runtime()
            replay_label = "fake-live" if self.args.input_source == INPUT_SOURCE_FAKE_LIVE else "recording-replay"
            print(
                f"[{replay_label}] "
                f"case={self.recording_source.case_path} frames={self.recording_source.frame_count} "
                f"fps={self.recording_source.effective_fps:g} first_step={self.recording_source.steps[0]} "
                f"serial={self.recording_source.serial} depth_source={self.recording_source.depth_source} "
                f"ir_stereo={str(self.recording_source.has_ir_stereo).lower()}",
                flush=True,
            )
        else:
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
        return 2 if self._fatal_error_snapshot() is not None else 0

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
        self.recording_source = None
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
        if tracker_enabled(self.args):
            workers.append(("tracker", self._tracker_worker))
        if self.args.pcd_mode == "masked":
            workers.append(("pcd", self._pcd_worker))
        elif self.args.depth_source in {"ffs", "ffs_remote"}:
            workers.append(("depth", self._depth_profile_worker))
        if self.args.enable_remote_ffs_quality:
            workers.append(("remote-quality", self._remote_ffs_quality_worker))
        if self.args.debug and self.args.render_mode == "none":
            workers.append(("debug", self._headless_debug_worker))

        def worker_runner(worker_name: str, worker_target: Callable[[], None]) -> Callable[[], None]:
            def run_worker() -> None:
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

    def _capture_recording_worker(self) -> None:
        assert self.recording_source is not None
        source = self.recording_source
        frame_period_s = 1.0 / float(source.effective_fps)
        try:
            first_packet = source.read_packet(seq=0)
        except Exception as exc:
            if not self.stop_event.is_set():
                self._record_fatal_worker_error("recording replay", exc)
            return
        self.capture_slot.put(first_packet)
        self.capture_stats.record(first_packet.receive_perf_s)
        if source.frame_count <= 1:
            self.stop_event.set()
            self._request_render_update()
            return
        if self.args.track_mode != "none":
            while not self.stop_event.is_set():
                if self._recording_first_frame_segmented.wait(timeout=0.01):
                    break
            if self.stop_event.is_set():
                return
        replay_start_s = time.perf_counter()
        for seq in range(1, source.frame_count):
            if self.stop_event.is_set():
                break
            wait_start_s = time.perf_counter()
            target_s = replay_start_s + (float(seq) * frame_period_s)
            wait_s = target_s - wait_start_s
            if wait_s > 0.0 and self.stop_event.wait(wait_s):
                break
            wait_done_s = time.perf_counter()
            try:
                packet = source.read_packet(
                    seq=seq,
                    wait_ms=_elapsed_ms(wait_start_s, wait_done_s),
                )
            except Exception as exc:
                if not self.stop_event.is_set():
                    self._record_fatal_worker_error("recording replay", exc)
                break
            self.capture_slot.put(packet)
            self.capture_stats.record(packet.receive_perf_s)
        self.stop_event.set()
        self._request_render_update()

    def _capture_worker(self) -> None:
        assert self.runtime is not None
        if _is_replay_input_source(str(self.args.input_source)):
            self._capture_recording_worker()
            return
        seq = 0
        pipeline = self.runtime.pipeline
        align = self.runtime.align
        while not self.stop_event.is_set():
            wait_start_s = time.perf_counter()
            try:
                frames = pipeline.wait_for_frames()
            except Exception as exc:
                if not self.stop_event.is_set():
                    self._record_fatal_worker_error("RealSense capture", exc)
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
            "edgetam_live_session_keep_frames": int(self.args.edgetam_live_session_keep_frames),
            "offline_video_input_used": _is_replay_input_source(str(self.args.input_source)),
            "input_source": self.args.input_source,
            "recording_case": (
                str(self.args.recording_case) if _is_replay_input_source(str(self.args.input_source)) else None
            ),
            "replay_fps": (
                self.recording_source.effective_fps
                if _is_replay_input_source(str(self.args.input_source)) and self.recording_source is not None
                else None
            ),
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
            "render_max_points_per_layer": int(self.args.render_max_points_per_layer),
            "pcd_filter_enabled": pcd_filter_enabled(self.args),
            "pcd_filter_mode": self.args.pcd_filter_mode if pcd_filter_enabled(self.args) else PCD_FILTER_NONE,
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
            "render_mode": self.args.render_mode,
            "view_mode": str(self.args.view_mode),
            "tracker_backend": str(self.args.tracker_backend),
            "tracker_device": str(self.args.tracker_device),
            "tracker_query_count": int(self.args.tracker_query_count),
            "tracker_display_scope": str(self.args.tracker_display_scope),
            "tracker_overlay_max_points": int(self.args.tracker_overlay_max_points),
            "tracker_marker_point_size": float(self.args.tracker_marker_point_size),
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
                if _is_replay_input_source(str(self.args.input_source)):
                    self._recording_first_frame_segmented.set()
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
                        self._record_fatal_worker_error("EdgeTAM segmentation", exc)
                        break
                    self.mask_slot.put(packet)
                    self.seg_stats.record(packet.process_done_perf_s)
        except Exception as exc:
            if not self.stop_event.is_set():
                self._record_fatal_worker_error("segmentation worker", exc)

    def _build_tracker_adapter(self) -> Any:
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
        if self._tracker_query_points_yx is not None:
            return self._tracker_query_points_yx
        union_mask = _tracker_union_mask(mask_packet)
        object_pixels = int(np.count_nonzero(mask_packet.object_mask))
        controller_pixels = int(np.count_nonzero(mask_packet.controller_mask))
        union_pixels = int(np.count_nonzero(union_mask))
        if object_pixels <= 0 or controller_pixels <= 0 or union_pixels <= 0:
            return None
        query_points = sample_phystwin_dense(
            union_mask,
            seed=int(self.args.tracker_seed),
            camera_idx=0,
            torch_device="cpu",
        )
        requested = int(self.args.tracker_query_count)
        if requested > 0 and len(query_points) > requested:
            query_points = np.ascontiguousarray(query_points[:requested], dtype=np.float32)
        if len(query_points) == 0:
            return None
        query_is_object, query_is_controller = _classify_query_points_yx(
            query_points,
            object_mask=mask_packet.object_mask,
            controller_mask=mask_packet.controller_mask,
        )
        adapter.initialize([], query_points)
        self._tracker_query_points_yx = np.ascontiguousarray(query_points, dtype=np.float32)
        self._tracker_query_is_object = np.ascontiguousarray(query_is_object, dtype=bool)
        self._tracker_query_is_controller = np.ascontiguousarray(query_is_controller, dtype=bool)
        print(
            "[tapnextpp-tracker] "
            f"initialized query_count={len(query_points)} requested={requested or 'phystwin_dense'} "
            f"union_pixels={union_pixels} object_pixels={object_pixels} controller_pixels={controller_pixels} "
            f"display_scope={self.args.tracker_display_scope} device={self.args.tracker_device}",
            flush=True,
        )
        return self._tracker_query_points_yx

    def _tracker_depth_for_lift(self, mask_packet: MaskPacket) -> tuple[np.ndarray, float]:
        if mask_packet.depth_u16 is not None:
            return mask_packet.depth_u16, float(mask_packet.depth_scale_m_per_unit)
        if mask_packet.depth_source in {"ffs", "ffs_remote"}:
            depth_m, _ffs_ms, _ffs_align_ms, _remote_rtt_ms, _server_total_ms, _request_kb, _response_kb = (
                self._compute_external_ffs_depth_color_m(mask_packet)
            )
            return np.ascontiguousarray(depth_m, dtype=np.float32), 1.0
        raise RuntimeError("tracker lift requires RGB-D depth")

    def _tracker_lift_mask(self, mask_packet: MaskPacket) -> np.ndarray | None:
        scope = str(self.args.tracker_display_scope)
        if scope == TRACKER_DISPLAY_SCOPE_CONTROLLER:
            return np.asarray(mask_packet.controller_mask, dtype=bool)
        if scope == TRACKER_DISPLAY_SCOPE_OBJECT:
            return np.asarray(mask_packet.object_mask, dtype=bool)
        return None

    def _tracker_worker(self) -> None:
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
                query_points = self._ensure_tracker_queries(mask_packet, adapter)
                if query_points is None:
                    if self.args.debug:
                        print(
                            "[tapnextpp-tracker] waiting_for_non_empty_object_and_controller_masks "
                            f"seq={mask_packet.seq}",
                            flush=True,
                        )
                    continue
                assert self._tracker_query_is_object is not None
                assert self._tracker_query_is_controller is not None
                started_s = time.perf_counter()
                rgb = np.ascontiguousarray(mask_packet.color_bgr[:, :, ::-1], dtype=np.uint8)
                result = adapter.update(rgb)
                tracks_latest, visibility_latest = _latest_tracker_arrays(result)
                query_is_object = self._tracker_query_is_object
                query_is_controller = self._tracker_query_is_controller
                common_count = min(
                    int(len(tracks_latest)),
                    int(len(visibility_latest)),
                    int(len(query_is_object)),
                    int(len(query_is_controller)),
                )
                tracks_latest = tracks_latest[:common_count]
                visibility_latest = visibility_latest[:common_count]
                query_is_object = query_is_object[:common_count]
                query_is_controller = query_is_controller[:common_count]
                display_visibility = _tracker_display_visibility(
                    visibility_latest,
                    query_is_object=query_is_object,
                    query_is_controller=query_is_controller,
                    display_scope=str(self.args.tracker_display_scope),
                )
                selected = _select_visible_spread_indices(
                    tracks_latest,
                    display_visibility,
                    max_points=int(self.args.tracker_overlay_max_points),
                )
                selected_tracks = tracks_latest[selected]
                selected_visibility = display_visibility[selected]
                selected_query_is_object = query_is_object[selected]
                selected_query_is_controller = query_is_controller[selected]

                lift_start_s = time.perf_counter()
                depth_for_lift, depth_scale = self._tracker_depth_for_lift(mask_packet)
                depth_max_m = float("inf") if float(self.args.depth_max_m) <= 0.0 else float(self.args.depth_max_m)
                lifted = lift_tracks_yx_to_world(
                    tracks_yx=selected_tracks,
                    visibility=selected_visibility,
                    depth=depth_for_lift,
                    intrinsics=mask_packet.intrinsics,
                    c2w=np.eye(4, dtype=np.float32),
                    depth_scale_m_per_unit=float(depth_scale),
                    mask=self._tracker_lift_mask(mask_packet),
                    depth_min_m=float(self.args.depth_min_m),
                    depth_max_m=depth_max_m,
                )
                lift_ms = _elapsed_ms(lift_start_s, time.perf_counter())
                source_indices = lifted.source_indices
                if len(source_indices):
                    lifted_query_is_object = selected_query_is_object[source_indices]
                    lifted_query_is_controller = selected_query_is_controller[source_indices]
                else:
                    lifted_query_is_object = np.empty((0,), dtype=bool)
                    lifted_query_is_controller = np.empty((0,), dtype=bool)
                done_s = time.perf_counter()
                stats = getattr(result, "stats", {}) or {}
                packet = TrackerMarkerPacket(
                    seq=mask_packet.seq,
                    marker_xyz_m=np.ascontiguousarray(lifted.points_world, dtype=np.float32).reshape(-1, 3),
                    marker_colors_rgb_u8=_solid_tracker_colors(len(lifted.points_world)),
                    query_points_yx=query_points,
                    tracks_yx=np.ascontiguousarray(lifted.tracks_yx, dtype=np.float32).reshape(-1, 2),
                    visibility=np.ascontiguousarray(selected_visibility[source_indices], dtype=np.float32),
                    query_is_object=np.ascontiguousarray(lifted_query_is_object, dtype=bool),
                    query_is_controller=np.ascontiguousarray(lifted_query_is_controller, dtype=bool),
                    receive_perf_s=mask_packet.receive_perf_s,
                    process_done_perf_s=done_s,
                    query_count=int(len(query_points)),
                    model_ms=float(stats.get("model_run_ms", stats.get("cuda_event_ms", 0.0)) or 0.0),
                    lift_ms=float(lift_ms),
                    e2e_ms=_elapsed_ms(started_s, done_s),
                    backend=str(getattr(result, "backend", None) or adapter.name),
                    display_scope=str(self.args.tracker_display_scope),
                )
                self.tracker_marker_slot.put(packet)
                self.tracker_stats.record(packet.process_done_perf_s)
                if self.args.debug:
                    print(
                        "[tapnextpp-tracker] "
                        f"seq={packet.seq} markers={packet.marker_count}/{len(selected_tracks)} "
                        f"queries={packet.query_count} model_ms={packet.model_ms:.1f} "
                        f"lift_ms={packet.lift_ms:.1f} e2e_ms={packet.e2e_ms:.1f} "
                        f"fps={self.tracker_stats.fps:.1f}",
                        flush=True,
                    )
                self._request_render_update()
        except Exception as exc:
            if not self.stop_event.is_set():
                self._record_fatal_worker_error("TAPNext++ tracker worker", exc)

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

    def _prune_edgetam_live_session(self, session: Any, *, current_frame_idx: int) -> None:
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
                if object_tracking_enabled(self.args):
                    prompt_obj_ids.append(OBJECT_ID)
                    prompt_masks.append(np.asarray(initial_object_mask, dtype=bool))
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
        masks_by_id = extract_object_masks_from_hf_output(output, post_masks)
        missing = [obj_id for obj_id in active_object_ids(self.args) if obj_id not in masks_by_id]
        if missing:
            raise RuntimeError(f"HF output missing tracked object ids: {missing}")
        reference_mask = next(iter(masks_by_id.values()))
        object_mask = masks_by_id.get(OBJECT_ID)
        if object_mask is None:
            object_mask = np.zeros_like(reference_mask, dtype=bool)
        controller_mask = masks_by_id.get(CONTROLLER_ID)
        if controller_mask is None:
            controller_mask = np.zeros_like(reference_mask, dtype=bool)
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
        keep_components: int,
        min_retain_ratio: float,
        min_raw_retain_ratio: float,
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

        fallback_to_capped = False
        fallback_reason = ""
        fallback_source = "none"
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
            from qqtt.demo.pcd_postprocess import (
                apply_phystwin_like_radius_postprocess,
            )

            filtered_points, filtered_colors, _unused_stats = apply_phystwin_like_radius_postprocess(
                points=capped_points,
                colors=capped_colors,
                enabled=True,
                radius_m=float(self.args.filter_radius_m),
                nb_points=int(self.args.filter_nb_points),
            )
        elif mode == PCD_FILTER_ENHANCED_PT:
            from qqtt.demo.pcd_postprocess import (
                apply_enhanced_phystwin_like_postprocess,
            )

            filtered_points, filtered_colors, _unused_stats = apply_enhanced_phystwin_like_postprocess(
                points=capped_points,
                colors=capped_colors,
                enabled=True,
                radius_m=float(self.args.filter_radius_m),
                nb_points=int(self.args.filter_nb_points),
                component_voxel_size_m=float(self.args.enhanced_component_voxel_size_m),
                keep_near_main_gap_m=float(self.args.enhanced_keep_near_main_gap_m),
                keep_top_n_components=int(keep_components),
            )
        else:
            raise ValueError(f"unsupported PCD filter mode: {mode}")

        filter_ms = _elapsed_ms(filter_start_s, time.perf_counter())
        filtered_points = np.ascontiguousarray(filtered_points, dtype=np.float32).reshape(-1, 3)
        filtered_colors = np.ascontiguousarray(filtered_colors, dtype=np.uint8).reshape(-1, 3)
        filter_output_points = int(len(filtered_points))
        raw_point_count = int(len(raw_points))
        capped_point_count = int(len(capped_points))
        retain_ratio = float(filter_output_points / max(1, capped_point_count))
        raw_retain_ratio = float(filter_output_points / max(1, raw_point_count))
        if filter_output_points == 0 and int(len(capped_points)) > 0:
            if float(min_raw_retain_ratio) > 0.0:
                filtered_points = np.ascontiguousarray(raw_points, dtype=np.float32).reshape(-1, 3)
                filtered_colors = np.ascontiguousarray(raw_colors, dtype=np.uint8).reshape(-1, 3)
                fallback_reason = "empty_filter_output_raw"
                fallback_source = "raw"
            else:
                filtered_points = np.ascontiguousarray(capped_points, dtype=np.float32).reshape(-1, 3)
                filtered_colors = np.ascontiguousarray(capped_colors, dtype=np.uint8).reshape(-1, 3)
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
            fallback_to_capped = True
            fallback_reason = "low_filter_retain_ratio"
            fallback_source = "capped"
        return filtered_points, filtered_colors, {
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
        started_s = time.perf_counter()
        object_points, object_colors, object_stats = self._apply_single_pcd_filter(
            points=item.object_xyz,
            colors=item.object_rgb,
            mode=str(self.args.object_filter),
            cap=int(item.object_cap),
            voxel_size_m=float(item.object_voxel_size_m),
            keep_components=int(self.args.object_filter_keep_components),
            min_retain_ratio=float(DEFAULT_OBJECT_FILTER_MIN_RETAIN_RATIO),
            min_raw_retain_ratio=float(DEFAULT_OBJECT_FILTER_MIN_RAW_RETAIN_RATIO),
            rng=np.random.default_rng(int(item.seq) * 2 + 17),
        )
        controller_points, controller_colors, controller_stats = self._apply_single_pcd_filter(
            points=item.controller_xyz,
            colors=item.controller_rgb,
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

    def _filter_output_is_fresh(self, *, packet_seq: int, output: FilterOutput) -> bool:
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
                controller_pcd_timing = dict(empty_pcd_timing)
            if object_tracking_enabled(self.args):
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
            else:
                object_xyz = np.empty((0, 3), dtype=np.float32)
                object_colors = np.empty((0, 3), dtype=np.uint8)
                object_pcd_timing = dict(empty_pcd_timing)
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
                            if self._filter_output_is_fresh(packet_seq=mask_packet.seq, output=latest):
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
                    if self._filter_output_is_fresh(packet_seq=mask_packet.seq, output=latest):
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
        if object_tracking_enabled(self.args):
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
                max(preferred.height, (16.0 if self.args.debug else 12.0) * em),
            )

        window.set_on_layout(on_layout)
        pcd_material = rendering.MaterialRecord()
        pcd_material.shader = "defaultUnlit"
        pcd_material.point_size = float(self.args.point_size)
        tracker_material = rendering.MaterialRecord()
        tracker_material.shader = "defaultUnlit"
        tracker_material.point_size = float(self.args.tracker_marker_point_size)

        def make_geometry_layer(name: str, material: object, *, min_capacity: int = 0) -> Open3DSceneTensorLayer:
            return Open3DSceneTensorLayer(
                name=name,
                o3d_module=o3d,
                o3c_module=o3c,
                rendering_module=rendering,
                scene=scene_widget.scene,
                material=material,
                device=device,
                min_capacity=max(0, int(min_capacity)),
            )

        def update_layer(
            layer: Open3DSceneTensorLayer,
            points_xyz_m: np.ndarray,
            colors_rgb_u8: np.ndarray,
            *,
            max_points: int = 0,
        ) -> tuple[float, float]:
            cap_start_s = time.perf_counter()
            display_points, display_colors = cap_render_points(
                points_xyz_m,
                colors_rgb_u8,
                max_points=int(max_points),
            )
            cap_ms = _elapsed_ms(cap_start_s, time.perf_counter())
            update = layer.update(display_points, display_colors)
            open3d_ms = float(update.open3d_update_geometry_ms)
            if update.points_count == 0:
                open3d_ms += float(update.open3d_remove_geometry_ms)
            return float(update.cpu_format_ms) + cap_ms, open3d_ms

        pcd_caps = [
            int(value)
            for value in (self.args.pcd_max_points, self.args.render_max_points_per_layer)
            if int(value) > 0
        ]
        pcd_layer_capacity = min(pcd_caps) if pcd_caps else 0
        tracker_layer_capacity = int(self.args.tracker_overlay_max_points)
        if tracker_layer_capacity <= 0:
            tracker_layer_capacity = int(self.args.tracker_query_count) or PHYSTWIN_DENSE_QUERY_POINTS
        controller_state = make_geometry_layer(GEOMETRY_CONTROLLER, pcd_material, min_capacity=pcd_layer_capacity)
        object_state = make_geometry_layer(GEOMETRY_OBJECT, pcd_material, min_capacity=pcd_layer_capacity)
        tracker_state = make_geometry_layer(GEOMETRY_TRACKER, tracker_material, min_capacity=tracker_layer_capacity)
        camera_initialized = {"value": False}
        render_post_gate = CoalescedPostGate()
        last_render_seq = {"value": -1}
        last_marker_seq = {"value": -1}
        latest_render_packet: dict[str, MaskedPcdPacket | None] = {"value": None}
        latest_marker_packet: dict[str, TrackerMarkerPacket | None] = {"value": None}
        fatal_exit_posted = {"value": False}

        def reset_camera() -> None:
            if str(self.args.view_mode) == "camera":
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
                return
            try:
                scene_widget.look_at([0.0, 0.0, 0.8], [0.0, 0.0, -1.0], [0.0, -1.0, 0.0])
            except Exception:
                bounds = o3d.geometry.AxisAlignedBoundingBox([-0.5, -0.35, 0.1], [0.5, 0.35, 1.5])
                scene_widget.setup_camera(60.0, bounds, [0.0, 0.0, 0.8])

        def render_latest() -> bool:
            packet = self.render_slot.get_latest_after(last_render_seq["value"])
            marker_packet = self.tracker_marker_slot.get_latest_after(last_marker_seq["value"])
            if packet is None and marker_packet is None:
                return False
            controller_convert_ms = controller_update_ms = 0.0
            object_convert_ms = object_update_ms = 0.0
            tracker_convert_ms = tracker_update_ms = 0.0
            if packet is not None:
                last_render_seq["value"] = packet.seq
                latest_render_packet["value"] = packet
                controller_convert_ms, controller_update_ms = update_layer(
                    controller_state,
                    packet.controller_xyz_m,
                    packet.controller_colors_rgb_u8,
                    max_points=int(self.args.render_max_points_per_layer),
                )
                object_convert_ms, object_update_ms = update_layer(
                    object_state,
                    packet.object_xyz_m,
                    packet.object_colors_rgb_u8,
                    max_points=int(self.args.render_max_points_per_layer),
                )
            if marker_packet is not None:
                last_marker_seq["value"] = marker_packet.seq
                latest_marker_packet["value"] = marker_packet
                tracker_convert_ms, tracker_update_ms = update_layer(
                    tracker_state,
                    marker_packet.marker_xyz_m,
                    marker_packet.marker_colors_rgb_u8,
                )
            active_packet = latest_render_packet["value"]
            active_marker = latest_marker_packet["value"]
            if active_packet is None:
                return True
            if not camera_initialized["value"] and active_packet.point_count > 0:
                reset_camera()
                camera_initialized["value"] = True
            render_time_s = time.perf_counter()
            latency_start_s = active_packet.receive_perf_s
            if packet is None and marker_packet is not None:
                latency_start_s = marker_packet.receive_perf_s
            latency_ms = _elapsed_ms(latency_start_s, render_time_s)
            timing = replace(
                active_packet.timing,
                open3d_convert_ms=float(controller_convert_ms + object_convert_ms + tracker_convert_ms),
                open3d_update_ms=float(controller_update_ms + object_update_ms + tracker_update_ms),
                receive_to_render_ms=latency_ms,
            )
            self.render_stats.record_render(render_time_s=render_time_s, latency_ms=latency_ms)
            hud_label.text = self._format_hud(packet=active_packet, timing=timing, tracker_packet=active_marker)
            self._maybe_log_debug(packet=active_packet, timing=timing, now_s=render_time_s)
            return True

        def render_latest_on_main_thread() -> None:
            try:
                fatal = self._fatal_error_snapshot()
                if fatal is not None:
                    hud_label.text = self._format_fatal_hud(fatal)
                    hud_label.text_color = gui.Color(1.0, 0.25, 0.20)
                    fatal_exit_posted["value"] = True
                    print(f"[FATAL] closing Open3D viewer after {fatal.log_message()}", flush=True)
                    if hasattr(window, "post_redraw"):
                        try:
                            window.post_redraw()
                        except Exception:
                            pass
                    try:
                        app.quit()
                    except Exception:
                        pass
                    return
                if self.stop_event.is_set():
                    try:
                        app.quit()
                    except Exception:
                        pass
                    return
                rendered = render_latest()
                if rendered and hasattr(window, "post_redraw"):
                    try:
                        window.post_redraw()
                    except Exception:
                        pass
            finally:
                render_post_gate.mark_done()
                if self._fatal_error_snapshot() is not None and not fatal_exit_posted["value"]:
                    request_render_update()
                elif (
                    not self.stop_event.is_set()
                    and (
                        self.render_slot.latest_seq() > last_render_seq["value"]
                        or self.tracker_marker_slot.latest_seq() > last_marker_seq["value"]
                    )
                ):
                    request_render_update()

        def request_render_update() -> None:
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

    def _format_hud(
        self,
        *,
        packet: MaskedPcdPacket,
        timing: PipelineTiming,
        tracker_packet: TrackerMarkerPacket | None = None,
    ) -> str:
        status = "late" if timing.receive_to_render_ms > self.args.latency_target_ms else "ok"
        max_points = "uncapped" if self.args.pcd_max_points == 0 else str(self.args.pcd_max_points)
        render_cap = (
            "uncapped"
            if int(self.args.render_max_points_per_layer) == 0
            else str(int(self.args.render_max_points_per_layer))
        )
        depth_line = f"depth: {self.args.depth_source}  color={self.args.pcd_color_mode}"
        preset_text = "" if self.args.demo_preset == "none" else f"  preset={self.args.demo_preset}"
        if tracker_enabled(self.args):
            if tracker_packet is None:
                tracker_line = (
                    f"tracker: {self.args.tracker_backend} waiting  "
                    f"queries={int(self.args.tracker_query_count) or PHYSTWIN_DENSE_QUERY_POINTS}  "
                    f"scope={self.args.tracker_display_scope}  device={self.args.tracker_device}"
                )
            else:
                marker_age_ms = _elapsed_ms(tracker_packet.process_done_perf_s, time.perf_counter())
                tracker_line = (
                    f"tracker: {tracker_packet.backend}  fps={self.tracker_stats.fps:.1f}  "
                    f"markers={tracker_packet.marker_count}/{tracker_packet.query_count}  "
                    f"scope={tracker_packet.display_scope}  age={marker_age_ms:.0f} ms  "
                    f"model={tracker_packet.model_ms:.0f} ms  lift={tracker_packet.lift_ms:.1f} ms  "
                    f"e2e={tracker_packet.e2e_ms:.0f} ms"
                )
        else:
            tracker_line = "tracker: off"
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
                f"pre={filter_info.controller_prefallback_points} raw-r={filter_info.controller_raw_retain_ratio:.2f}  "
                "object raw/cap/out="
                f"{filter_info.object_raw_points}/{filter_info.object_cap_points}/{filter_info.object_output_points}  "
                f"pre={filter_info.object_prefallback_points} raw-r={filter_info.object_raw_retain_ratio:.2f}"
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
            f"capture/seg/pcd/tracker/render FPS: {self.capture_stats.fps:.1f} / {self.seg_stats.fps:.1f} / "
            f"{self.pcd_stats.fps:.1f} / {self.tracker_stats.fps:.1f} / {self.render_stats.render_fps:.1f}\n"
            f"latency: {timing.receive_to_render_ms:.1f} ms ({status}, target {self.args.latency_target_ms:.1f} ms)\n"
            f"points controller/object: {packet.controller_point_count} / {packet.object_point_count}  "
            f"pcd max/object: {max_points}  render max/layer: {render_cap}\n"
            f"{tracker_line}\n"
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
            f"tracker_fps={self.tracker_stats.fps:.1f} "
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
