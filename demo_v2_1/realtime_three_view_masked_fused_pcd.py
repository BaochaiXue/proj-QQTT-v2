#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import deque
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_process.depth_backends.ffs_defaults import (  # noqa: E402
    DEFAULT_FFS_MAX_DISP,
    DEFAULT_FFS_MODEL_NAME,
    DEFAULT_FFS_TRT_BUILDER_OPTIMIZATION_LEVEL,
    DEFAULT_FFS_TRT_ENGINE_SIZE,
    DEFAULT_FFS_VALID_ITERS,
)
from data_process.depth_backends.geometry import transform_points  # noqa: E402
from demo_v2.realtime_masked_edgetam_pcd import (  # noqa: E402
    _bgr_to_pil_rgb,
    _elapsed_ms,
    _load_hf_streaming_runtime,
    _time_model_forward,
    _time_runtime_ms,
    active_object_ids,
    backproject_masked_rgbd_profiled,
    controller_tracking_enabled,
    extract_object_masks_from_hf_output,
    load_binary_mask,
    make_solid_colors,
    release_sam31_runtime_resources,
    resolve_initial_masks,
)
from demo_v2.pcd_filter_fast import voxel_cap_points  # noqa: E402
from demo_v2.realtime_single_camera_pointcloud import (  # noqa: E402
    CameraIntrinsics,
    ColorFloat32Buffer,
    DEFAULT_FFS_REPO,
    DEFAULT_FFS_TRT_TWO_STAGE_MODEL_DIR,
    FfsIrToColorAligner,
    LatestSlot,
    RenderStats,
    _load_open3d_modules,
    apply_wslg_open3d_env_defaults,
    build_projection_grid,
    ensure_float32_c_contiguous,
    pointcloud_update_requires_readd,
    validate_ffs_paths,
    warm_up_numba_ffs_align,
)


TRACK_MODE_OBJECT_ONLY = "object-only"
TRACK_MODE_CONTROLLER_OBJECT = "controller-object"
TRACK_MODE_NONE = "none"
TRACK_MODES = (TRACK_MODE_OBJECT_ONLY, TRACK_MODE_CONTROLLER_OBJECT, TRACK_MODE_NONE)

DEPTH_SOURCE_FFS = "ffs"
DEPTH_SOURCE_FFS_REMOTE = "ffs_remote"
DEPTH_SOURCE_REALSENSE = "realsense"
DEPTH_SOURCE_NONE = "none"
DEPTH_SOURCES = (DEPTH_SOURCE_FFS, DEPTH_SOURCE_FFS_REMOTE, DEPTH_SOURCE_REALSENSE, DEPTH_SOURCE_NONE)
OFFICIAL_DEPTH_SOURCES = (DEPTH_SOURCE_FFS,)
RENDER_MODES = ("pointcloud", "none")
FFS_WORKER_MODES = ("shared",)
FFS_SCHEDULES = ("strict3-latest",)
EDGETAM_WORKER_MODES = ("per-camera",)
EDGETAM_MODEL_TOPOLOGIES = ("replicated",)
INIT_MODES = ("sam31-first-frame", "saved-masks")

POSTPROCESS_NONE = "none"
POSTPROCESS_PT_FILTER = "pt-filter"
POSTPROCESS_ENHANCED_PT = "enhanced-pt"
POSTPROCESS_MODES = (POSTPROCESS_NONE, POSTPROCESS_PT_FILTER, POSTPROCESS_ENHANCED_PT)
PCD_FILTER_SCHEDULE_MODES = ("async", "sync", "none")

DEFAULT_CAMERA_IDS = (0, 1, 2)
DEFAULT_OBJECT_LABEL = "object"
DEFAULT_CONTROLLER_LABEL = "controller"
DEFAULT_MODEL_ID = "yonigozlan/EdgeTAM-hf"
DEFAULT_PROFILE = "848x480"
DEFAULT_FPS = 60
DEFAULT_DEVICE = "cuda"
DEFAULT_DTYPE = "bfloat16"
DEFAULT_COMPILE_MODE = "vision-reduce-overhead"
DEFAULT_OUTPUT_ROOT = ROOT / "result" / "demo2_1_three_view_fused_pcd"
DEFAULT_OBJECT_FILTER_CAP = 20_000
DEFAULT_CONTROLLER_FILTER_CAP = 20_000
DEFAULT_OBJECT_FILTER_VOXEL_M = 0.004
DEFAULT_CONTROLLER_FILTER_VOXEL_M = 0.003
DEFAULT_FILTER_EVERY_N = 3
DEFAULT_FILTER_BUDGET_MS = 12.0
OBJECT_ID = 2
CONTROLLER_ID = 1
OBJECT_COLOR_RGB = (64, 180, 255)
CONTROLLER_COLOR_RGB = (255, 96, 32)
DEBUG_LOG_INTERVAL_S = 1.0


@dataclass(frozen=True)
class SemanticLayerSpec:
    obj_id: int
    label: str
    default_postprocess: str


@dataclass(frozen=True)
class CameraLayerCloud:
    camera_idx: int
    label: str
    points_m: np.ndarray
    colors_rgb: np.ndarray


@dataclass(frozen=True)
class FusedLayerCloud:
    label: str
    postprocess_mode: str
    points_m: np.ndarray
    colors_rgb: np.ndarray
    per_camera: tuple[dict[str, int], ...]

    @property
    def point_count(self) -> int:
        return int(self.points_m.shape[0])


@dataclass(frozen=True)
class CameraFramePacket:
    group_id: int
    camera_idx: int
    frame_seq: int
    timestamp_ns: int
    color_bgr: np.ndarray
    ir_left_u8: np.ndarray | None
    ir_right_u8: np.ndarray | None
    k_color: np.ndarray
    k_ir_left: np.ndarray | None
    t_ir_left_to_color: np.ndarray | None
    baseline_m: float
    intrinsics: CameraIntrinsics
    c2w: np.ndarray

    @property
    def seq(self) -> int:
        return int(self.group_id)


@dataclass(frozen=True)
class CaptureGroup:
    group_id: int
    created_perf_s: float
    frames: dict[int, CameraFramePacket]

    @property
    def seq(self) -> int:
        return int(self.group_id)


@dataclass(frozen=True)
class DepthPacket:
    group_id: int
    camera_idx: int
    depth_m: np.ndarray
    ffs_ms: float
    align_ms: float


@dataclass(frozen=True)
class DepthGroup:
    group_id: int
    depths: dict[int, DepthPacket]
    total_ms: float
    per_camera_ms: dict[int, dict[str, float]]

    @property
    def seq(self) -> int:
        return int(self.group_id)


@dataclass(frozen=True)
class CameraMaskPacket:
    group_id: int
    camera_idx: int
    color_bgr: np.ndarray
    controller_mask: np.ndarray
    object_mask: np.ndarray
    model_ms: float
    cuda_event_model_ms: float
    mask_ms: float

    @property
    def seq(self) -> int:
        return int(self.group_id)


@dataclass(frozen=True)
class FusedPcdPacket:
    group_id: int
    created_perf_s: float
    object_points_m: np.ndarray
    object_colors_rgb: np.ndarray
    controller_points_m: np.ndarray
    controller_colors_rgb: np.ndarray
    fusion_ms: float
    filter_ms: float
    object_raw_points: int
    controller_raw_points: int
    ffs_cycle_ms: float
    edgetam_ms_by_camera: dict[int, float]

    @property
    def seq(self) -> int:
        return int(self.group_id)

    @property
    def object_point_count(self) -> int:
        return int(self.object_points_m.shape[0])

    @property
    def controller_point_count(self) -> int:
        return int(self.controller_points_m.shape[0])


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


def _normalize_label(label: str) -> str:
    return str(label).strip().lower().replace("_", " ").replace("-", " ")


def is_controller_label(label: str) -> bool:
    normalized = _normalize_label(label)
    return normalized in {"controller", "hand", "hands", "left hand", "right hand", "hand a", "hand b"}


def resolve_postprocess_mode(
    label: str,
    *,
    object_postprocess: str = POSTPROCESS_ENHANCED_PT,
    controller_postprocess: str = POSTPROCESS_PT_FILTER,
) -> str:
    if object_postprocess not in POSTPROCESS_MODES:
        raise ValueError(f"Unsupported object postprocess mode: {object_postprocess}")
    if controller_postprocess not in POSTPROCESS_MODES:
        raise ValueError(f"Unsupported controller postprocess mode: {controller_postprocess}")
    if is_controller_label(label):
        return controller_postprocess
    return object_postprocess


def semantic_layers_for_track_mode(
    track_mode: str,
    *,
    object_label: str = DEFAULT_OBJECT_LABEL,
    controller_label: str = DEFAULT_CONTROLLER_LABEL,
    object_postprocess: str = POSTPROCESS_ENHANCED_PT,
    controller_postprocess: str = POSTPROCESS_PT_FILTER,
) -> tuple[SemanticLayerSpec, ...]:
    if track_mode not in TRACK_MODES:
        raise ValueError(f"Unsupported track mode: {track_mode}")
    if track_mode == TRACK_MODE_NONE:
        return ()
    layers: list[SemanticLayerSpec] = []
    if track_mode == TRACK_MODE_CONTROLLER_OBJECT:
        layers.append(
            SemanticLayerSpec(
                obj_id=CONTROLLER_ID,
                label=str(controller_label),
                default_postprocess=resolve_postprocess_mode(
                    controller_label,
                    object_postprocess=object_postprocess,
                    controller_postprocess=controller_postprocess,
                ),
            )
        )
    layers.append(
        SemanticLayerSpec(
            obj_id=OBJECT_ID,
            label=str(object_label),
            default_postprocess=resolve_postprocess_mode(
                object_label,
                object_postprocess=object_postprocess,
                controller_postprocess=controller_postprocess,
            ),
        )
    )
    return tuple(layers)


def _as_points(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float32)
    if arr.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    return arr.reshape(-1, 3)


def _as_colors(colors: np.ndarray) -> np.ndarray:
    arr = np.asarray(colors, dtype=np.uint8)
    if arr.size == 0:
        return np.empty((0, 3), dtype=np.uint8)
    return arr.reshape(-1, 3)


def fuse_semantic_camera_clouds(
    camera_clouds: Sequence[CameraLayerCloud],
    layers: Sequence[SemanticLayerSpec],
) -> dict[str, FusedLayerCloud]:
    """Fuse cam0/cam1/cam2 clouds per semantic label without mixing labels."""

    clouds_by_label: dict[str, list[CameraLayerCloud]] = {layer.label: [] for layer in layers}
    postprocess_by_label = {layer.label: layer.default_postprocess for layer in layers}
    for cloud in camera_clouds:
        if cloud.label not in clouds_by_label:
            continue
        clouds_by_label[cloud.label].append(cloud)

    fused: dict[str, FusedLayerCloud] = {}
    for label, clouds in clouds_by_label.items():
        point_sets: list[np.ndarray] = []
        color_sets: list[np.ndarray] = []
        per_camera: list[dict[str, int]] = []
        for cloud in clouds:
            points = _as_points(cloud.points_m)
            colors = _as_colors(cloud.colors_rgb)
            if len(colors) != len(points):
                raise ValueError(
                    f"Point/color count mismatch for {label} cam{cloud.camera_idx}: "
                    f"{len(points)} points vs {len(colors)} colors"
                )
            point_sets.append(points)
            color_sets.append(colors)
            per_camera.append(
                {
                    "camera_idx": int(cloud.camera_idx),
                    "point_count": int(len(points)),
                }
            )

        if point_sets:
            fused_points = np.concatenate(point_sets, axis=0)
            fused_colors = np.concatenate(color_sets, axis=0)
        else:
            fused_points = np.empty((0, 3), dtype=np.float32)
            fused_colors = np.empty((0, 3), dtype=np.uint8)

        fused[label] = FusedLayerCloud(
            label=label,
            postprocess_mode=postprocess_by_label[label],
            points_m=fused_points,
            colors_rgb=fused_colors,
            per_camera=tuple(per_camera),
        )
    return fused


def apply_semantic_postprocess(
    layer: FusedLayerCloud,
    *,
    filter_cap: int = 0,
    filter_voxel_size_m: float = 0.004,
    phystwin_radius_m: float,
    phystwin_nb_points: int,
    enhanced_component_voxel_size_m: float,
    enhanced_keep_near_main_gap_m: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Apply the configured semantic PCD cleanup to one fused layer."""

    points = _as_points(layer.points_m)
    colors = _as_colors(layer.colors_rgb)
    input_count = int(len(points))
    if int(filter_cap) > 0:
        points, capped_colors_or_none = voxel_cap_points(
            points,
            colors,
            max_points=int(filter_cap),
            voxel_size_m=float(filter_voxel_size_m),
            rng=np.random.default_rng(0),
        )
        colors = _as_colors(np.empty((0, 3), dtype=np.uint8) if capped_colors_or_none is None else capped_colors_or_none)
    capped_count = int(len(points))
    if layer.postprocess_mode == POSTPROCESS_NONE:
        return points, colors, {
            "enabled": False,
            "mode": POSTPROCESS_NONE,
            "input_point_count": input_count,
            "capped_point_count": capped_count,
            "output_point_count": int(len(points)),
        }
    if layer.postprocess_mode == POSTPROCESS_PT_FILTER:
        from data_process.visualization.experiments.ffs_confidence_filter_pcd_compare import (
            _apply_phystwin_like_radius_postprocess,
        )

        filtered_points, filtered_colors, stats = _apply_phystwin_like_radius_postprocess(
            points=points,
            colors=colors,
            enabled=True,
            radius_m=float(phystwin_radius_m),
            nb_points=int(phystwin_nb_points),
        )
        stats["mode"] = POSTPROCESS_PT_FILTER
        stats["input_point_count"] = input_count
        stats["capped_point_count"] = capped_count
        return filtered_points, filtered_colors, stats
    if layer.postprocess_mode == POSTPROCESS_ENHANCED_PT:
        from data_process.visualization.experiments.ffs_confidence_filter_pcd_compare import (
            _apply_enhanced_phystwin_like_postprocess,
        )

        filtered_points, filtered_colors, stats = _apply_enhanced_phystwin_like_postprocess(
            points=points,
            colors=colors,
            enabled=True,
            radius_m=float(phystwin_radius_m),
            nb_points=int(phystwin_nb_points),
            component_voxel_size_m=float(enhanced_component_voxel_size_m),
            keep_near_main_gap_m=float(enhanced_keep_near_main_gap_m),
        )
        stats["mode"] = POSTPROCESS_ENHANCED_PT
        stats["input_point_count"] = input_count
        stats["capped_point_count"] = capped_count
        return filtered_points, filtered_colors, stats
    raise ValueError(f"Unsupported postprocess mode: {layer.postprocess_mode}")


def parse_camera_ids(value: str) -> tuple[int, ...]:
    ids = tuple(int(part.strip()) for part in str(value).split(",") if part.strip())
    if len(ids) != 3:
        raise argparse.ArgumentTypeError("Demo 2.1 expects exactly three camera ids, e.g. 0,1,2")
    if len(set(ids)) != len(ids):
        raise argparse.ArgumentTypeError(f"Camera ids must be unique: {ids}")
    return ids


def parse_profile(value: str) -> tuple[int, int]:
    try:
        width_s, height_s = str(value).lower().split("x", maxsplit=1)
        width = int(width_s)
        height = int(height_s)
    except Exception as exc:
        raise argparse.ArgumentTypeError(f"profile must look like 848x480, got {value!r}") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError(f"profile must be positive, got {value!r}")
    return width, height


def _camera_intrinsics_from_k(k_color: np.ndarray, *, width: int, height: int) -> CameraIntrinsics:
    k = np.asarray(k_color, dtype=np.float32).reshape(3, 3)
    return CameraIntrinsics(
        fx=float(k[0, 0]),
        fy=float(k[1, 1]),
        cx=float(k[0, 2]),
        cy=float(k[1, 2]),
    )


def _as_timestamp_ns(value: Any) -> int:
    try:
        # RealSense timestamps are usually milliseconds.
        return int(float(value) * 1_000_000)
    except Exception:
        return int(time.time_ns())


def _load_saved_mask_from_root(mask_root: str | Path | None, *, camera_idx: int, expected_shape: tuple[int, int]) -> np.ndarray:
    if mask_root is None:
        raise RuntimeError("saved-masks mode requires a mask root")
    root = Path(mask_root)
    candidates = [
        root / f"cam{int(camera_idx)}.png",
        root / f"{int(camera_idx)}.png",
        root / str(int(camera_idx)) / "0.png",
        root / str(int(camera_idx)) / "000000.png",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return load_binary_mask(candidate, expected_shape=expected_shape)
    raise FileNotFoundError(f"no saved mask for cam{camera_idx} under {root}")


def resolve_initial_masks_for_camera(
    frame: CameraFramePacket,
    args: argparse.Namespace,
    *,
    sam31_lock: threading.Lock,
) -> tuple[np.ndarray, np.ndarray]:
    expected_shape = tuple(frame.color_bgr.shape[:2])
    if args.init_mode == "saved-masks":
        object_mask = _load_saved_mask_from_root(
            args.object_init_mask_root,
            camera_idx=frame.camera_idx,
            expected_shape=expected_shape,
        )
        if not controller_tracking_enabled(args.track_mode):
            return np.zeros_like(object_mask, dtype=bool), object_mask
        controller_mask = _load_saved_mask_from_root(
            args.controller_init_mask_root,
            camera_idx=frame.camera_idx,
            expected_shape=expected_shape,
        )
        return controller_mask, object_mask
    if args.init_mode == "sam31-first-frame":
        # SAM3.1 initialization is intentionally serialized across cameras to
        # avoid three first-frame segmentation jobs fighting for the same GPU.
        with sam31_lock:
            return resolve_initial_masks(frame, args)
    raise ValueError(f"unsupported init mode: {args.init_mode}")


def build_contract(args: argparse.Namespace) -> dict[str, Any]:
    layers = semantic_layers_for_track_mode(
        args.track_mode,
        object_label=args.object_prompt,
        controller_label=args.controller_prompt,
        object_postprocess=args.object_postprocess,
        controller_postprocess=args.controller_postprocess,
    )
    return {
        "demo": "demo_2_1_three_view_fused_masked_pcd",
        "camera_ids": list(args.camera_ids),
        "track_mode": args.track_mode,
        "frame_by_frame_streaming": True,
        "offline_video_input_used": False,
        "edge_backend": "HF EdgeTAMVideo",
        "compile_mode": args.compile_mode,
        "dtype": args.dtype,
        "depth_source": args.depth_source,
        "render_mode": args.render_mode,
        "fusion_target_fps": float(args.fusion_target_fps),
        "official_quality_depth": args.depth_source in set(OFFICIAL_DEPTH_SOURCES),
        "native_realsense_depth_role": "fallback/debug only",
        "ffs_contract": {
            "checkpoint": DEFAULT_FFS_MODEL_NAME,
            "valid_iters": DEFAULT_FFS_VALID_ITERS,
            "capture_resolution": "848x480",
            "engine_input": f"{DEFAULT_FFS_TRT_ENGINE_SIZE[1]}x{DEFAULT_FFS_TRT_ENGINE_SIZE[0]}",
            "padding_policy": "pad_width_848_to_864",
            "builderOptimizationLevel": DEFAULT_FFS_TRT_BUILDER_OPTIMIZATION_LEVEL,
            "max_disp": DEFAULT_FFS_MAX_DISP,
            "worker_mode": args.ffs_worker_mode,
            "schedule": args.ffs_schedule,
        },
        "edgetam": {
            "worker_mode": args.edgetam_worker_mode,
            "model_topology": args.edgetam_model_topology,
            "one_streaming_session_per_camera": True,
        },
        "fusion": {
            "mode": "semantic_layers",
            "labels_are_filtered_separately": True,
            "do_not_filter_object_controller_union": True,
            "object_controller_union_before_filter": False,
        },
        "filter_scheduler": {
            "enabled": bool(args.enable_pcd_filter) and args.pcd_filter_mode != "none",
            "mode": args.pcd_filter_mode,
            "hot_path": "raw_or_capped_pcd_every_frame",
            "filtered_path": "latest_wins_async_every_n",
            "render_blocks_on_filter": args.pcd_filter_mode == "sync",
            "filter_every_n": int(args.filter_every_n),
            "filter_budget_ms": float(args.filter_budget_ms),
            "object": {
                "postprocess": args.object_postprocess,
                "cap": int(args.object_filter_cap),
                "voxel_size_m": float(args.object_filter_voxel_m),
            },
            "controller": {
                "postprocess": args.controller_postprocess,
                "cap": int(args.controller_filter_cap),
                "voxel_size_m": float(args.controller_filter_voxel_m),
            },
        },
        "semantic_layers": [
            {
                "obj_id": layer.obj_id,
                "label": layer.label,
                "postprocess": layer.default_postprocess,
            }
            for layer in layers
        ],
    }


class Demo21Runtime:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.width, self.height = parse_profile(args.profile)
        self.camera_system: Any | None = None
        self.stop_event = threading.Event()
        self.capture_group_slot: LatestSlot[CaptureGroup] = LatestSlot()
        self.depth_group_slot: LatestSlot[DepthGroup] = LatestSlot()
        self.mask_slots: dict[int, LatestSlot[CameraMaskPacket]] = {
            int(camera_idx): LatestSlot() for camera_idx in args.camera_ids
        }
        self.render_slot: LatestSlot[FusedPcdPacket] = LatestSlot()
        self.capture_group_stats = StageStats()
        self.ffs_stats = StageStats()
        self.edge_stats = {int(camera_idx): StageStats() for camera_idx in args.camera_ids}
        self.fusion_stats = StageStats()
        self.render_stats = RenderStats()
        self._threads: list[threading.Thread] = []
        self._sam31_lock = threading.Lock()
        self._summary: dict[str, Any] = {"contract": build_contract(args), "events": []}
        self._latest_depth_group: DepthGroup | None = None
        self._latest_fused: FusedPcdPacket | None = None
        self._last_debug_s = 0.0
        self._render_request: Callable[[], None] = lambda: None

    def run(self) -> int:
        if self.args.depth_source not in {DEPTH_SOURCE_FFS, DEPTH_SOURCE_NONE}:
            raise RuntimeError(
                "Demo 2.1 live runtime currently supports --depth-source ffs for official output "
                "and --depth-source none for capture/EdgeTAM isolation."
            )
        apply_wslg_open3d_env_defaults()
        self._validate_live_contract()
        self._start_camera_system()
        try:
            if self.args.render_mode == "pointcloud":
                self._run_open3d()
            else:
                self._run_headless()
        finally:
            self.stop()
            self._write_summary()
        return 0

    def _validate_live_contract(self) -> None:
        if tuple(self.args.camera_ids) != DEFAULT_CAMERA_IDS:
            raise RuntimeError("Demo 2.1 first live slice expects --camera-ids 0,1,2")
        if self.args.compile_mode != DEFAULT_COMPILE_MODE:
            raise RuntimeError("Demo 2.1 requires --compile-mode vision-reduce-overhead")
        if self.args.ffs_worker_mode != "shared":
            raise RuntimeError("Demo 2.1 requires --ffs-worker-mode shared")
        if self.args.edgetam_worker_mode != "per-camera":
            raise RuntimeError("Demo 2.1 requires --edgetam-worker-mode per-camera")
        if self.args.edgetam_model_topology != "replicated":
            raise RuntimeError("Demo 2.1 first live slice requires --edgetam-model-topology replicated")
        if int(self.args.object_filter_cap) < 0 or int(self.args.controller_filter_cap) < 0:
            raise RuntimeError("Demo 2.1 filter caps must be >= 0")
        if float(self.args.object_filter_voxel_m) <= 0 or float(self.args.controller_filter_voxel_m) <= 0:
            raise RuntimeError("Demo 2.1 filter voxel sizes must be positive")
        if int(self.args.filter_every_n) < 1:
            raise RuntimeError("Demo 2.1 --filter-every-n must be >= 1")
        if float(self.args.filter_budget_ms) < 0:
            raise RuntimeError("Demo 2.1 --filter-budget-ms must be >= 0")
        if self.args.depth_source == DEPTH_SOURCE_FFS:
            validate_ffs_paths(ffs_repo=Path(self.args.ffs_repo), model_dir=Path(self.args.ffs_trt_model_dir))
        if self._needs_world_fusion() and not Path(self.args.calibrate_path).is_file():
            raise FileNotFoundError(f"Demo 2.1 requires calibrate.pkl for world fusion: {self.args.calibrate_path}")

    def _needs_world_fusion(self) -> bool:
        return self.args.depth_source == DEPTH_SOURCE_FFS and self.args.track_mode != TRACK_MODE_NONE

    def _start_camera_system(self) -> None:
        from data_process.visualization.calibration_io import load_calibration_transforms
        from qqtt.env.camera import CameraSystem

        self.camera_system = CameraSystem(
            WH=(self.width, self.height),
            fps=int(self.args.fps),
            num_cam=3,
            serial_numbers=self.args.serials,
            capture_mode="stereo_ir",
            emitter="off",
            calibration_reference_serials=self.args.calibration_reference_serials,
            enable_keyboard_listener=False,
        )
        if self._needs_world_fusion():
            c2w_list = load_calibration_transforms(
                self.args.calibrate_path,
                serial_numbers=list(self.camera_system.serial_numbers),
                calibration_reference_serials=list(self.camera_system.calibration_reference_serials),
            )
        else:
            c2w_list = [np.eye(4, dtype=np.float32) for _ in self.args.camera_ids]
        self._c2w_by_camera = {
            int(camera_idx): np.asarray(c2w_list[int(camera_idx)], dtype=np.float32).reshape(4, 4)
            for camera_idx in self.args.camera_ids
        }
        self._stream_metadata = list(self.camera_system.stream_metadata)
        print(
            "[demo2.1] "
            f"serials={self.camera_system.serial_numbers} profile={self.width}x{self.height}@{self.args.fps} "
            f"depth={self.args.depth_source} ffs_worker=shared edgetam_workers=per-camera",
            flush=True,
        )
        print(f"[demo2.1-contract] {json.dumps(build_contract(self.args), sort_keys=True)}", flush=True)

    def stop(self) -> None:
        self.stop_event.set()
        for thread in list(self._threads):
            if thread.is_alive():
                thread.join(timeout=1.0)
        self._threads.clear()
        if self.camera_system is not None:
            try:
                if getattr(self.camera_system, "listener", None) is not None:
                    self.camera_system.listener.stop()
            except Exception:
                pass
            try:
                self.camera_system.realsense.stop()
            except Exception:
                pass
            self.camera_system = None

    def _write_summary(self) -> None:
        output_root = Path(self.args.output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        summary_path = output_root / f"session_{time.strftime('%Y%m%d_%H%M%S')}_summary.json"
        latest = self._latest_fused
        self._summary["final"] = {
            "capture_group_fps": self.capture_group_stats.fps,
            "ffs_cycle_fps": self.ffs_stats.fps,
            "fusion_fps": self.fusion_stats.fps,
            "render_fps": self.render_stats.render_fps,
            "latest_group_id": None if latest is None else latest.group_id,
            "object_points": None if latest is None else latest.object_point_count,
            "controller_points": None if latest is None else latest.controller_point_count,
        }
        summary_path.write_text(json.dumps(self._summary, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[demo2.1] summary={summary_path}", flush=True)

    def _start_threads(self) -> None:
        specs: list[tuple[str, Callable[[], None]]] = [
            ("capture-group", self._capture_group_worker),
        ]
        if self.args.depth_source == DEPTH_SOURCE_FFS:
            specs.append(("shared-ffs", self._shared_ffs_worker))
        if self.args.track_mode != TRACK_MODE_NONE:
            for camera_idx in self.args.camera_ids:
                specs.append((f"edgetam-cam{camera_idx}", lambda camera_idx=int(camera_idx): self._edgetam_camera_worker(camera_idx)))
        if self.args.track_mode != TRACK_MODE_NONE and self.args.depth_source == DEPTH_SOURCE_FFS:
            specs.append(("fusion", self._fusion_worker))
        if self.args.debug and self.args.render_mode == "none":
            specs.append(("debug", self._debug_worker))
        for name, target in specs:
            thread = threading.Thread(target=target, name=f"demo2.1-{name}", daemon=True)
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

    def _metadata_frame_packet(self, *, group_id: int, camera_idx: int, obs: dict[str, Any]) -> CameraFramePacket:
        metadata = self._stream_metadata[int(camera_idx)]
        k_color = np.asarray(metadata["K_color"], dtype=np.float32).reshape(3, 3)
        intrinsics = _camera_intrinsics_from_k(k_color, width=self.width, height=self.height)
        return CameraFramePacket(
            group_id=int(group_id),
            camera_idx=int(camera_idx),
            frame_seq=int(obs.get("step_idx", group_id)),
            timestamp_ns=_as_timestamp_ns(obs.get("timestamp", time.time() * 1000.0)),
            color_bgr=np.ascontiguousarray(obs["color"].copy()),
            ir_left_u8=np.ascontiguousarray(obs["ir_left"].copy()),
            ir_right_u8=np.ascontiguousarray(obs["ir_right"].copy()),
            k_color=k_color,
            k_ir_left=np.asarray(metadata["K_ir_left"], dtype=np.float32).reshape(3, 3),
            t_ir_left_to_color=np.asarray(metadata["T_ir_left_to_color"], dtype=np.float32).reshape(4, 4),
            baseline_m=float(metadata["ir_baseline_m"]),
            intrinsics=intrinsics,
            c2w=self._c2w_by_camera[int(camera_idx)],
        )

    def _capture_group_worker(self) -> None:
        assert self.camera_system is not None
        group_id = 0
        interval_s = 1.0 / max(1e-6, float(self.args.fusion_target_fps))
        next_tick_s = time.perf_counter()
        while not self.stop_event.is_set():
            now_s = time.perf_counter()
            if now_s < next_tick_s:
                time.sleep(min(0.002, next_tick_s - now_s))
                continue
            next_tick_s = now_s + interval_s
            try:
                obs = self.camera_system.get_observation()
                frames = {
                    int(camera_idx): self._metadata_frame_packet(
                        group_id=group_id,
                        camera_idx=int(camera_idx),
                        obs=obs[int(camera_idx)],
                    )
                    for camera_idx in self.args.camera_ids
                }
            except TimeoutError as exc:
                if not self.stop_event.is_set() and self.args.debug:
                    print(f"[WARN] Demo 2.1 capture group skipped after timeout: {exc}", flush=True)
                self._summary["capture_timeout_count"] = int(self._summary.get("capture_timeout_count", 0)) + 1
                continue
            except Exception as exc:
                if not self.stop_event.is_set():
                    print(f"[ERROR] Demo 2.1 capture group failed: {type(exc).__name__}: {exc}", flush=True)
                self.stop_event.set()
                break
            packet = CaptureGroup(group_id=group_id, created_perf_s=time.perf_counter(), frames=frames)
            self.capture_group_slot.put(packet)
            self.capture_group_stats.record(packet.created_perf_s)
            group_id += 1

    def _create_ffs_runner(self) -> object:
        from data_process.depth_backends import FastFoundationStereoTensorRTRunner

        return FastFoundationStereoTensorRTRunner(
            ffs_repo=Path(self.args.ffs_repo),
            model_dir=Path(self.args.ffs_trt_model_dir),
            trt_root=None if self.args.ffs_trt_root is None else Path(self.args.ffs_trt_root),
        )

    def _compute_ffs_depth_for_frame(
        self,
        *,
        runner: object,
        frame: CameraFramePacket,
        aligners: dict[int, FfsIrToColorAligner],
    ) -> DepthPacket:
        if (
            frame.ir_left_u8 is None
            or frame.ir_right_u8 is None
            or frame.k_ir_left is None
            or frame.t_ir_left_to_color is None
            or frame.baseline_m <= 0
        ):
            raise RuntimeError(f"cam{frame.camera_idx} is missing FFS IR stereo data")
        ffs_start_s = time.perf_counter()
        output = runner.run_pair(
            frame.ir_left_u8,
            frame.ir_right_u8,
            K_ir_left=frame.k_ir_left,
            baseline_m=float(frame.baseline_m),
        )
        ffs_done_s = time.perf_counter()
        depth_ir_left_m = np.asarray(output["depth_ir_left_m"], dtype=np.float32)
        k_ir_left_used = np.asarray(output.get("K_ir_left_used", frame.k_ir_left), dtype=np.float32)
        align_start_s = time.perf_counter()
        aligner = aligners.get(int(frame.camera_idx))
        key_shape = tuple(depth_ir_left_m.shape), tuple(frame.color_bgr.shape[:2])
        if aligner is None or getattr(aligner, "_demo21_key", None) != key_shape:
            aligner = FfsIrToColorAligner(
                k_ir_left=k_ir_left_used,
                t_ir_left_to_color=frame.t_ir_left_to_color,
                k_color=frame.k_color,
                ir_shape=depth_ir_left_m.shape,
                color_shape=frame.color_bgr.shape[:2],
            )
            setattr(aligner, "_demo21_key", key_shape)
            aligners[int(frame.camera_idx)] = aligner
        depth_color_m = np.ascontiguousarray(aligner.align(depth_ir_left_m), dtype=np.float32)
        align_done_s = time.perf_counter()
        return DepthPacket(
            group_id=frame.group_id,
            camera_idx=frame.camera_idx,
            depth_m=depth_color_m,
            ffs_ms=_elapsed_ms(ffs_start_s, ffs_done_s),
            align_ms=_elapsed_ms(align_start_s, align_done_s),
        )

    def _shared_ffs_worker(self) -> None:
        try:
            warm_up_numba_ffs_align()
            runner = self._create_ffs_runner()
            aligners: dict[int, FfsIrToColorAligner] = {}
            last_group_id = -1
            while not self.stop_event.is_set():
                group = self.capture_group_slot.get_latest_after(last_group_id)
                if group is None:
                    time.sleep(0.001)
                    continue
                last_group_id = group.group_id
                cycle_start_s = time.perf_counter()
                depths: dict[int, DepthPacket] = {}
                per_camera: dict[int, dict[str, float]] = {}
                for camera_idx in self.args.camera_ids:
                    frame = group.frames[int(camera_idx)]
                    depth = self._compute_ffs_depth_for_frame(runner=runner, frame=frame, aligners=aligners)
                    depths[int(camera_idx)] = depth
                    per_camera[int(camera_idx)] = {"ffs_ms": depth.ffs_ms, "align_ms": depth.align_ms}
                packet = DepthGroup(
                    group_id=group.group_id,
                    depths=depths,
                    total_ms=_elapsed_ms(cycle_start_s, time.perf_counter()),
                    per_camera_ms=per_camera,
                )
                self.depth_group_slot.put(packet)
                self._latest_depth_group = packet
                self.ffs_stats.record()
        except Exception as exc:
            if not self.stop_event.is_set():
                print(f"[ERROR] Demo 2.1 shared FFS worker failed: {type(exc).__name__}: {exc}", flush=True)
            self.stop_event.set()

    def _autocast_context(self, torch_module: Any) -> Any:
        if not str(self.args.device).startswith("cuda") or self.args.dtype == "float32":
            return nullcontext()
        dtype = torch_module.bfloat16 if self.args.dtype == "bfloat16" else torch_module.float16
        return torch_module.autocast("cuda", dtype=dtype)

    def _init_hf_model(self, camera_idx: int) -> tuple[Any, Any, Any, Any, Any]:
        hf_stream = _load_hf_streaming_runtime()
        torch_module = hf_stream.torch
        if str(self.args.device).startswith("cuda") and not torch_module.cuda.is_available():
            raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
        dtype = hf_stream._dtype_from_name(self.args.dtype)
        model = hf_stream.EdgeTamVideoModel.from_pretrained(self.args.model_id).to(self.args.device, dtype=dtype)
        model.eval()
        model, compile_metadata = hf_stream._apply_compile_mode(model, self.args.compile_mode)
        processor = hf_stream.Sam2VideoProcessor.from_pretrained(self.args.model_id)
        print(
            "[demo2.1-edgetam] "
            f"cam={camera_idx} topology=replicated model={self.args.model_id} "
            f"compile={self.args.compile_mode} applied={compile_metadata.get('applied_targets', [])}",
            flush=True,
        )
        return hf_stream, torch_module, dtype, model, processor

    def _run_edgetam_frame(
        self,
        *,
        torch_module: Any,
        dtype: Any,
        model: Any,
        processor: Any,
        session: Any,
        frame: CameraFramePacket,
        initial_controller_mask: np.ndarray,
        initial_object_mask: np.ndarray,
        add_prompt: bool,
    ) -> CameraMaskPacket:
        image = _bgr_to_pil_rgb(frame.color_bgr)
        inputs, preprocess_ms, _, _ = _time_runtime_ms(
            torch_module,
            self.args.device,
            lambda: processor(images=image, device=self.args.device, return_tensors="pt"),
            sync_enabled=False,
        )
        pixel_values = inputs.pixel_values[0].to(device=self.args.device, dtype=dtype)
        prompt_ms = 0.0
        with self._autocast_context(torch_module):
            if add_prompt:
                prompt_obj_ids: list[int] = []
                prompt_masks: list[np.ndarray] = []
                if controller_tracking_enabled(self.args.track_mode):
                    prompt_obj_ids.append(CONTROLLER_ID)
                    prompt_masks.append(np.asarray(initial_controller_mask, dtype=bool))
                prompt_obj_ids.append(OBJECT_ID)
                prompt_masks.append(np.asarray(initial_object_mask, dtype=bool))
                _, prompt_ms, _, _ = _time_runtime_ms(
                    torch_module,
                    self.args.device,
                    lambda: processor.add_inputs_to_inference_session(
                        inference_session=session,
                        frame_idx=0,
                        obj_ids=prompt_obj_ids,
                        input_masks=prompt_masks,
                    ),
                    sync_enabled=False,
                )
            output, wall_model_ms, cuda_event_model_ms, _, _ = _time_model_forward(
                torch_module=torch_module,
                device=self.args.device,
                profile_sync=False,
                profile_cuda_events=bool(self.args.profile_cuda_events),
                fn=lambda: model(inference_session=session, frame=pixel_values),
            )
            post_masks, postprocess_ms, _, _ = _time_runtime_ms(
                torch_module,
                self.args.device,
                lambda: processor.post_process_masks(
                    [output.pred_masks],
                    original_sizes=inputs.original_sizes,
                    binarize=False,
                )[0],
                sync_enabled=False,
            )
        masks_by_id = extract_object_masks_from_hf_output(output, post_masks)
        missing = [obj_id for obj_id in active_object_ids(self.args) if obj_id not in masks_by_id]
        if missing:
            raise RuntimeError(f"HF output missing tracked object ids for cam{frame.camera_idx}: {missing}")
        object_mask = masks_by_id[OBJECT_ID]
        controller_mask = masks_by_id.get(CONTROLLER_ID)
        if controller_mask is None:
            controller_mask = np.zeros_like(object_mask, dtype=bool)
        return CameraMaskPacket(
            group_id=frame.group_id,
            camera_idx=frame.camera_idx,
            color_bgr=frame.color_bgr,
            controller_mask=controller_mask,
            object_mask=object_mask,
            model_ms=wall_model_ms,
            cuda_event_model_ms=cuda_event_model_ms,
            mask_ms=float(preprocess_ms + prompt_ms + wall_model_ms + postprocess_ms),
        )

    def _edgetam_camera_worker(self, camera_idx: int) -> None:
        try:
            hf_stream, torch_module, dtype, model, processor = self._init_hf_model(camera_idx)
            last_group_id = -1
            initialized = False
            controller_mask: np.ndarray | None = None
            object_mask: np.ndarray | None = None
            session = None
            with torch_module.inference_mode():
                while not self.stop_event.is_set():
                    group = self.capture_group_slot.get_latest_after(last_group_id)
                    if group is None:
                        time.sleep(0.001)
                        continue
                    last_group_id = group.group_id
                    frame = group.frames[int(camera_idx)]
                    if not initialized:
                        controller_mask, object_mask = resolve_initial_masks_for_camera(
                            frame,
                            self.args,
                            sam31_lock=self._sam31_lock,
                        )
                        session = hf_stream.EdgeTamVideoInferenceSession(
                            video=None,
                            video_height=int(frame.color_bgr.shape[0]),
                            video_width=int(frame.color_bgr.shape[1]),
                            inference_device=self.args.device,
                            inference_state_device=self.args.device,
                            video_storage_device=self.args.device,
                            dtype=dtype,
                        )
                        initialized = True
                        add_prompt = True
                    else:
                        add_prompt = False
                    assert session is not None and controller_mask is not None and object_mask is not None
                    packet = self._run_edgetam_frame(
                        torch_module=torch_module,
                        dtype=dtype,
                        model=model,
                        processor=processor,
                        session=session,
                        frame=frame,
                        initial_controller_mask=controller_mask,
                        initial_object_mask=object_mask,
                        add_prompt=add_prompt,
                    )
                    self.mask_slots[int(camera_idx)].put(packet)
                    self.edge_stats[int(camera_idx)].record()
        except Exception as exc:
            if not self.stop_event.is_set():
                print(f"[ERROR] Demo 2.1 EdgeTAM cam{camera_idx} failed: {type(exc).__name__}: {exc}", flush=True)
            self.stop_event.set()

    def _wait_mask_for_group(self, *, camera_idx: int, group_id: int, deadline_s: float) -> CameraMaskPacket | None:
        last_seen = group_id - 1
        while not self.stop_event.is_set() and time.perf_counter() < deadline_s:
            packet = self.mask_slots[int(camera_idx)].get_latest_after(last_seen)
            if packet is None:
                time.sleep(0.001)
                continue
            if packet.group_id == group_id:
                return packet
            if packet.group_id > group_id:
                return None
            last_seen = packet.group_id
        return None

    def _fusion_worker(self) -> None:
        last_depth_group = -1
        rng = np.random.default_rng()
        ray_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        incomplete = 0
        while not self.stop_event.is_set():
            depth_group = self.depth_group_slot.get_latest_after(last_depth_group)
            if depth_group is None:
                time.sleep(0.001)
                continue
            last_depth_group = depth_group.group_id
            deadline_s = time.perf_counter() + float(self.args.fusion_timeout_ms) / 1000.0
            mask_by_camera: dict[int, CameraMaskPacket] = {}
            for camera_idx in self.args.camera_ids:
                mask = self._wait_mask_for_group(camera_idx=int(camera_idx), group_id=depth_group.group_id, deadline_s=deadline_s)
                if mask is None:
                    incomplete += 1
                    break
                mask_by_camera[int(camera_idx)] = mask
            if len(mask_by_camera) != len(self.args.camera_ids):
                continue
            try:
                packet = self._build_fused_packet(
                    depth_group=depth_group,
                    masks=mask_by_camera,
                    ray_cache=ray_cache,
                    rng=rng,
                )
            except Exception as exc:
                if not self.stop_event.is_set():
                    print(f"[WARN] Demo 2.1 fusion group {depth_group.group_id} failed: {type(exc).__name__}: {exc}", flush=True)
                continue
            self.render_slot.put(packet)
            self._latest_fused = packet
            self.fusion_stats.record()
            if packet.group_id % int(self.args.render_every_n) == 0:
                self._render_request()
            if incomplete:
                self._summary["dropped_incomplete_fusion_groups"] = incomplete

    def _build_fused_packet(
        self,
        *,
        depth_group: DepthGroup,
        masks: dict[int, CameraMaskPacket],
        ray_cache: dict[int, tuple[np.ndarray, np.ndarray]],
        rng: np.random.Generator,
    ) -> FusedPcdPacket:
        started_s = time.perf_counter()
        object_clouds: list[CameraLayerCloud] = []
        controller_clouds: list[CameraLayerCloud] = []
        for camera_idx in self.args.camera_ids:
            depth = depth_group.depths[int(camera_idx)]
            mask = masks[int(camera_idx)]
            if depth.group_id != mask.group_id:
                raise RuntimeError("depth/mask group mismatch")
            if int(camera_idx) not in ray_cache:
                intrinsics = self._metadata_frame_packet(
                    group_id=depth_group.group_id,
                    camera_idx=int(camera_idx),
                    obs={"color": mask.color_bgr, "ir_left": np.zeros(mask.object_mask.shape, np.uint8), "ir_right": np.zeros(mask.object_mask.shape, np.uint8)},
                ).intrinsics
                ray_cache[int(camera_idx)] = build_projection_grid(
                    width=self.width,
                    height=self.height,
                    stride=1,
                    intrinsics=intrinsics,
                )
            ray_x, ray_y = ray_cache[int(camera_idx)]
            depth_m = depth.depth_m
            object_pts_cam, object_cols, _ = backproject_masked_rgbd_profiled(
                color_bgr=mask.color_bgr,
                depth_m=depth_m,
                mask=mask.object_mask,
                ray_x=ray_x,
                ray_y=ray_y,
                depth_min_m=float(self.args.depth_min_m),
                depth_max_m=float(self.args.depth_max_m),
                max_points=int(self.args.pcd_max_points_per_camera),
                color_mode=str(self.args.pcd_color_mode),
                class_rgb=tuple(self.args.object_color),
                rng=rng,
            )
            object_clouds.append(
                CameraLayerCloud(
                    camera_idx=int(camera_idx),
                    label=str(self.args.object_prompt),
                    points_m=transform_points(object_pts_cam, self._c2w_by_camera[int(camera_idx)]),
                    colors_rgb=object_cols,
                )
            )
            if controller_tracking_enabled(self.args.track_mode):
                controller_pts_cam, controller_cols, _ = backproject_masked_rgbd_profiled(
                    color_bgr=mask.color_bgr,
                    depth_m=depth_m,
                    mask=mask.controller_mask,
                    ray_x=ray_x,
                    ray_y=ray_y,
                    depth_min_m=float(self.args.depth_min_m),
                    depth_max_m=float(self.args.depth_max_m),
                    max_points=int(self.args.pcd_max_points_per_camera),
                    color_mode=str(self.args.pcd_color_mode),
                    class_rgb=tuple(self.args.controller_color),
                    rng=rng,
                )
            else:
                controller_pts_cam = np.empty((0, 3), dtype=np.float32)
                controller_cols = np.empty((0, 3), dtype=np.uint8)
            controller_clouds.append(
                CameraLayerCloud(
                    camera_idx=int(camera_idx),
                    label=str(self.args.controller_prompt),
                    points_m=transform_points(controller_pts_cam, self._c2w_by_camera[int(camera_idx)]),
                    colors_rgb=controller_cols,
                )
            )

        layers = semantic_layers_for_track_mode(
            self.args.track_mode,
            object_label=self.args.object_prompt,
            controller_label=self.args.controller_prompt,
            object_postprocess=self.args.object_postprocess,
            controller_postprocess=self.args.controller_postprocess,
        )
        assert build_contract(self.args)["fusion"]["object_controller_union_before_filter"] is False
        fused = fuse_semantic_camera_clouds([*object_clouds, *controller_clouds], layers)
        raw_object = fused.get(str(self.args.object_prompt))
        raw_controller = fused.get(str(self.args.controller_prompt))
        object_raw_count = 0 if raw_object is None else raw_object.point_count
        controller_raw_count = 0 if raw_controller is None else raw_controller.point_count
        filter_start_s = time.perf_counter()
        if raw_object is not None:
            object_points, object_colors, _ = apply_semantic_postprocess(
                raw_object,
                filter_cap=int(self.args.object_filter_cap),
                filter_voxel_size_m=float(self.args.object_filter_voxel_m),
                phystwin_radius_m=float(self.args.phystwin_radius_m),
                phystwin_nb_points=int(self.args.phystwin_nb_points),
                enhanced_component_voxel_size_m=float(self.args.enhanced_component_voxel_size_m),
                enhanced_keep_near_main_gap_m=float(self.args.enhanced_keep_near_main_gap_m),
            )
        else:
            object_points = np.empty((0, 3), dtype=np.float32)
            object_colors = np.empty((0, 3), dtype=np.uint8)
        if raw_controller is not None:
            controller_points, controller_colors, _ = apply_semantic_postprocess(
                raw_controller,
                filter_cap=int(self.args.controller_filter_cap),
                filter_voxel_size_m=float(self.args.controller_filter_voxel_m),
                phystwin_radius_m=float(self.args.phystwin_radius_m),
                phystwin_nb_points=int(self.args.phystwin_nb_points),
                enhanced_component_voxel_size_m=float(self.args.enhanced_component_voxel_size_m),
                enhanced_keep_near_main_gap_m=float(self.args.enhanced_keep_near_main_gap_m),
            )
        else:
            controller_points = np.empty((0, 3), dtype=np.float32)
            controller_colors = np.empty((0, 3), dtype=np.uint8)
        filter_ms = _elapsed_ms(filter_start_s, time.perf_counter())
        return FusedPcdPacket(
            group_id=depth_group.group_id,
            created_perf_s=time.perf_counter(),
            object_points_m=object_points,
            object_colors_rgb=object_colors,
            controller_points_m=controller_points,
            controller_colors_rgb=controller_colors,
            fusion_ms=_elapsed_ms(started_s, time.perf_counter()),
            filter_ms=filter_ms,
            object_raw_points=object_raw_count,
            controller_raw_points=controller_raw_count,
            ffs_cycle_ms=depth_group.total_ms,
            edgetam_ms_by_camera={idx: masks[idx].cuda_event_model_ms or masks[idx].model_ms for idx in masks},
        )

    def _debug_worker(self) -> None:
        while not self.stop_event.is_set():
            time.sleep(DEBUG_LOG_INTERVAL_S)
            self._print_debug()

    def _print_debug(self) -> None:
        latest = self._latest_fused
        depth = self._latest_depth_group
        edge_ms = " ".join(
            f"cam{idx}={latest.edgetam_ms_by_camera.get(idx, 0.0):.1f}ms" if latest is not None else f"cam{idx}=0.0ms"
            for idx in self.args.camera_ids
        )
        ffs_ms = " ".join(
            f"cam{idx}={depth.per_camera_ms.get(idx, {}).get('ffs_ms', 0.0):.1f}+{depth.per_camera_ms.get(idx, {}).get('align_ms', 0.0):.1f}ms"
            if depth is not None else f"cam{idx}=0.0+0.0ms"
            for idx in self.args.camera_ids
        )
        print(
            "[demo2.1-debug] "
            f"capture_group_fps={self.capture_group_stats.fps:.2f} "
            f"ffs_cycle_fps={self.ffs_stats.fps:.2f} "
            f"edge_fps_cam0={self.edge_stats[0].fps:.2f} edge_fps_cam1={self.edge_stats[1].fps:.2f} edge_fps_cam2={self.edge_stats[2].fps:.2f} "
            f"fusion_fps={self.fusion_stats.fps:.2f} render_fps={self.render_stats.render_fps:.2f} "
            f"ffs_cycle_ms={(0.0 if depth is None else depth.total_ms):.1f} "
            f"fusion_ms={(0.0 if latest is None else latest.fusion_ms):.1f} "
            f"filter_ms={(0.0 if latest is None else latest.filter_ms):.1f} "
            f"object_points={(0 if latest is None else latest.object_point_count)} "
            f"controller_points={(0 if latest is None else latest.controller_point_count)} "
            f"edgetam_ms[{edge_ms}] ffs_ms[{ffs_ms}]",
            flush=True,
        )

    def _run_open3d(self) -> None:
        o3d, gui, rendering = _load_open3d_modules()
        o3c = o3d.core
        device = o3c.Device("CPU:0")
        app = gui.Application.instance
        app.initialize()
        window = app.create_window("Demo 2.1 Three-View Fused EdgeTAM PCD", 1280, 800)
        scene_widget = gui.SceneWidget()
        scene_widget.scene = rendering.Open3DScene(window.renderer)
        scene_widget.scene.set_background([0.02, 0.02, 0.02, 1.0])
        hud_label = gui.Label("Demo 2.1 warming up: capture + shared FFS + per-camera EdgeTAM")
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
            hud_panel.frame = gui.Rect(rect.x + 0.5 * em, rect.y + 0.5 * em, max(preferred.width, 760), max(preferred.height, 9.0 * em))

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
                    flags = rendering.Scene.UPDATE_POINTS_FLAG | rendering.Scene.UPDATE_COLORS_FLAG
                    scene_widget.scene.scene.update_geometry(self.name, self.pcd, flags)
                    self.capacity = max(self.capacity, int(points.shape[0]))
                return convert_ms, _elapsed_ms(update_start_s, time.perf_counter())

        object_state = GeometryState("demo2_1_object_fused")
        controller_state = GeometryState("demo2_1_controller_fused")
        last_render_group = {"value": -1}
        camera_ready = {"value": False}

        def reset_camera(packet: FusedPcdPacket) -> None:
            points = np.concatenate([packet.object_points_m, packet.controller_points_m], axis=0)
            if len(points) == 0:
                return
            bbox = o3d.geometry.AxisAlignedBoundingBox(points.min(axis=0), points.max(axis=0))
            center = bbox.get_center()
            extent = max(float(np.linalg.norm(bbox.get_extent())), 0.2)
            bbox = o3d.geometry.AxisAlignedBoundingBox(center - extent, center + extent)
            scene_widget.setup_camera(60.0, bbox, center)

        def render_latest() -> None:
            packet = self.render_slot.get_latest_after(last_render_group["value"])
            if packet is None:
                return
            last_render_group["value"] = packet.group_id
            object_state.update(packet.object_points_m, packet.object_colors_rgb)
            controller_state.update(packet.controller_points_m, packet.controller_colors_rgb)
            if not camera_ready["value"] and (packet.object_point_count + packet.controller_point_count) > 0:
                reset_camera(packet)
                camera_ready["value"] = True
            now = time.perf_counter()
            self.render_stats.record_render(render_time_s=now, latency_ms=_elapsed_ms(packet.created_perf_s, now))
            hud_label.text = (
                f"Demo 2.1 fused PCD | group={packet.group_id} | "
                f"object={packet.object_point_count} pts | controller={packet.controller_point_count} pts | "
                f"fusion={packet.fusion_ms:.1f} ms | filter={packet.filter_ms:.1f} ms | "
                f"render_fps={self.render_stats.render_fps:.1f}"
            )
            if self.args.debug:
                self._print_debug()
            if hasattr(window, "post_redraw"):
                try:
                    window.post_redraw()
                except Exception:
                    pass

        def request_render() -> None:
            if self.stop_event.is_set():
                return
            try:
                app.post_to_main_thread(window, render_latest)
            except Exception:
                pass

        self._render_request = request_render

        def stop_and_quit() -> None:
            self.stop_event.set()
            if os.environ.get("QQTT_WSLG_OPEN3D_FAST_EXIT") == "1":
                self.stop()
                os._exit(0)
            try:
                app.quit()
            except Exception:
                pass

        window.set_on_close(lambda: (stop_and_quit(), True)[1])
        self._start_threads()
        timer: threading.Timer | None = None
        if self.args.duration_s > 0:
            timer = threading.Timer(float(self.args.duration_s), lambda: app.post_to_main_thread(window, stop_and_quit))
            timer.daemon = True
            timer.start()
        try:
            app.run()
        finally:
            if timer is not None:
                timer.cancel()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Demo 2.1 three-view masked and fused PCD contract. The first implementation "
            "slice locks semantic fusion and postprocess policy before wiring the live hardware loop."
        )
    )
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--serials", nargs="*", default=None)
    parser.add_argument("--camera-ids", type=parse_camera_ids, default=DEFAULT_CAMERA_IDS)
    parser.add_argument("--calibrate-path", default=str(ROOT / "calibrate.pkl"))
    parser.add_argument("--calibration-reference-serials", nargs="*", default=None)
    parser.add_argument("--track-mode", choices=TRACK_MODES, default=TRACK_MODE_OBJECT_ONLY)
    parser.add_argument("--init-mode", choices=INIT_MODES, default="sam31-first-frame")
    parser.add_argument("--object-prompt", default="stuffed animal")
    parser.add_argument("--controller-prompt", default=DEFAULT_CONTROLLER_LABEL)
    parser.add_argument("--depth-source", choices=DEPTH_SOURCES, default=DEPTH_SOURCE_FFS)
    parser.add_argument("--render-mode", choices=RENDER_MODES, default="none")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--compile-mode", choices=("vision-reduce-overhead",), default="vision-reduce-overhead")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--duration-s", type=float, default=0.0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--profile-cuda-events", action="store_true")
    parser.add_argument("--fusion-target-fps", type=float, default=10.0)
    parser.add_argument("--fusion-timeout-ms", type=float, default=150.0)
    parser.add_argument("--max-inflight-groups", type=int, default=2)
    parser.add_argument("--ffs-worker-mode", choices=FFS_WORKER_MODES, default="shared")
    parser.add_argument("--ffs-schedule", choices=FFS_SCHEDULES, default="strict3-latest")
    parser.add_argument("--edgetam-worker-mode", choices=EDGETAM_WORKER_MODES, default="per-camera")
    parser.add_argument("--edgetam-model-topology", choices=EDGETAM_MODEL_TOPOLOGIES, default="replicated")
    parser.add_argument("--ffs-repo", default=str(DEFAULT_FFS_REPO))
    parser.add_argument("--ffs-trt-model-dir", default=str(DEFAULT_FFS_TRT_TWO_STAGE_MODEL_DIR))
    parser.add_argument("--ffs-trt-root", default=None)
    parser.add_argument("--object-init-mask-root", default=None)
    parser.add_argument("--controller-init-mask-root", default=None)
    parser.add_argument("--depth-min-m", type=float, default=0.2)
    parser.add_argument("--depth-max-m", type=float, default=1.5)
    parser.add_argument("--pcd-max-points-per-camera", type=int, default=20000)
    parser.add_argument("--pcd-color-mode", choices=("rgb", "class"), default="rgb")
    parser.add_argument("--object-color", nargs=3, type=int, default=list(OBJECT_COLOR_RGB))
    parser.add_argument("--controller-color", nargs=3, type=int, default=list(CONTROLLER_COLOR_RGB))
    parser.add_argument("--render-every-n", type=int, default=1)
    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--object-postprocess", choices=POSTPROCESS_MODES, default=POSTPROCESS_ENHANCED_PT)
    parser.add_argument("--controller-postprocess", choices=POSTPROCESS_MODES, default=POSTPROCESS_PT_FILTER)
    parser.add_argument("--enable-pcd-filter", action="store_true")
    parser.add_argument("--pcd-filter-mode", choices=PCD_FILTER_SCHEDULE_MODES, default="async")
    parser.add_argument("--object-filter-cap", type=int, default=DEFAULT_OBJECT_FILTER_CAP)
    parser.add_argument("--controller-filter-cap", type=int, default=DEFAULT_CONTROLLER_FILTER_CAP)
    parser.add_argument("--object-filter-voxel-m", type=float, default=DEFAULT_OBJECT_FILTER_VOXEL_M)
    parser.add_argument("--controller-filter-voxel-m", type=float, default=DEFAULT_CONTROLLER_FILTER_VOXEL_M)
    parser.add_argument("--filter-every-n", type=int, default=DEFAULT_FILTER_EVERY_N)
    parser.add_argument("--filter-budget-ms", type=float, default=DEFAULT_FILTER_BUDGET_MS)
    parser.add_argument("--phystwin-radius-m", type=float, default=0.01)
    parser.add_argument("--phystwin-nb-points", type=int, default=12)
    parser.add_argument("--enhanced-component-voxel-size-m", type=float, default=0.006)
    parser.add_argument("--enhanced-keep-near-main-gap-m", type=float, default=0.035)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the Demo 2.1 runtime contract and exit without opening cameras.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    contract = build_contract(args)
    if args.dry_run:
        print(json.dumps(contract, indent=2, sort_keys=True))
        return 0
    return Demo21Runtime(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
