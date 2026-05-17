#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import deque
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, fields, is_dataclass, replace
from itertools import product
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback
from typing import Any, Callable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_process.depth_backends.ffs_defaults import (  # noqa: E402
    DEFAULT_FFS_MAX_DISP,
    DEFAULT_FFS_MODEL_NAME,
    DEFAULT_FFS_TRT_BATCH3_TWO_STAGE_MODEL_DIR,
    DEFAULT_FFS_TRT_BUILDER_OPTIMIZATION_LEVEL,
    DEFAULT_FFS_TRT_ENGINE_SIZE,
    DEFAULT_FFS_VALID_ITERS,
)
from data_process.depth_backends.geometry import transform_points  # noqa: E402
from data_process.depth_backends.fast_foundation_stereo import (  # noqa: E402
    FFS_INPUT_STAGING_MODES,
    FFS_INPUT_STAGING_PAGEABLE,
    FFS_INPUT_STAGING_PINNED,
)
from qqtt.demo.realtime_masked_edgetam_pcd import (  # noqa: E402
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
    object_tracking_enabled,
    release_sam31_runtime_resources,
    resolve_initial_masks,
)
from qqtt.demo.pcd_filter_fast import voxel_cap_points  # noqa: E402
from qqtt.demo.realtime_single_camera_pointcloud import (  # noqa: E402
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
from qqtt.demo.render_fastpath import (  # noqa: E402
    DEFAULT_RENDER_BACKEND,
    DEFAULT_RENDER_COPY_MODE,
    DEFAULT_RENDER_LAYER_MODE,
    RENDER_BACKENDS,
    RENDER_COPY_MODES,
    RENDER_LAYER_MODES,
    RENDER_LAYER_MODE_COMBINED,
    CoalescedRenderPostGate,
    LatestOnlyRenderBuffer,
    RenderLayerCombiner,
    Open3DSceneTensorLayer,
    RenderMicroProfileRecord,
    summarize_render_records,
)


TRACK_MODE_OBJECT_ONLY = "object-only"
TRACK_MODE_CONTROLLER_ONLY = "controller-only"
TRACK_MODE_CONTROLLER_OBJECT = "controller-object"
TRACK_MODE_NONE = "none"
TRACK_MODES = (TRACK_MODE_OBJECT_ONLY, TRACK_MODE_CONTROLLER_ONLY, TRACK_MODE_CONTROLLER_OBJECT, TRACK_MODE_NONE)

DEPTH_SOURCE_FFS = "ffs"
DEPTH_SOURCE_FFS_REMOTE = "ffs_remote"
DEPTH_SOURCE_REALSENSE = "realsense"
DEPTH_SOURCE_NONE = "none"
DEPTH_SOURCES = (DEPTH_SOURCE_FFS, DEPTH_SOURCE_FFS_REMOTE, DEPTH_SOURCE_REALSENSE, DEPTH_SOURCE_NONE)
OFFICIAL_DEPTH_SOURCES = (DEPTH_SOURCE_FFS,)
RENDER_MODES = ("pointcloud", "none")
FFS_WORKER_MODES = ("shared",)
FFS_SCHEDULES = ("strict3-latest",)
FFS_TRT_BATCH_SIZES = (1, 3)
EDGETAM_WORKER_MODES = ("per-camera",)
EDGETAM_MODEL_TOPOLOGY_REPLICATED = "replicated"
EDGETAM_MODEL_TOPOLOGY_SHARED = "shared-model"
EDGETAM_MODEL_TOPOLOGIES = (EDGETAM_MODEL_TOPOLOGY_REPLICATED, EDGETAM_MODEL_TOPOLOGY_SHARED)
INIT_MODES = ("sam31-first-frame", "saved-masks")

POSTPROCESS_NONE = "none"
POSTPROCESS_PT_FILTER = "pt-filter"
POSTPROCESS_ENHANCED_PT = "enhanced-pt"
POSTPROCESS_MODES = (POSTPROCESS_NONE, POSTPROCESS_PT_FILTER, POSTPROCESS_ENHANCED_PT)
PCD_FILTER_SCHEDULE_MODES = ("async", "sync", "none")

DEFAULT_CAMERA_IDS = (0, 1, 2)
DEFAULT_OBJECT_LABEL = "object"
DEFAULT_CONTROLLER_LABEL = "hand"
EXPERIMENT_MODE_CONTROLLER_OBJECT = "controller-object-exp"
EXPERIMENT_MODE_DEMO = "demo-mode"
EXPERIMENT_MODES = (EXPERIMENT_MODE_CONTROLLER_OBJECT, EXPERIMENT_MODE_DEMO)
DEFAULT_EXPERIMENT_MODE = EXPERIMENT_MODE_DEMO
DEMO_MODE_CONTROLLER_LABEL = DEFAULT_CONTROLLER_LABEL
CONTROLLER_OBJECT_EXP_CONTROLLER_LABEL = "towel"
DEFAULT_MODEL_ID = "yonigozlan/EdgeTAM-hf"
DEFAULT_PROFILE = "848x480"
DEFAULT_FPS = 60
DEFAULT_PRESET_CAPTURE_FPS = 15
PRESET_NONE = "none"
PRESET_OFFICIAL_LOWFPS = "official-lowfps"
PRESET_PERF_5FPS = "perf-5fps"
PRESET_PERF_5FPS_SINGLE_OWNER = "perf-5fps-single-owner"
PRESET_PERF_5FPS_STAGED = "perf-5fps-staged"
PRESET_PROFESSOR_SAFE = "professor-safe"
PRESET_VISUAL_5FPS = "visual-5fps"
PRESET_VISUAL_5FPS_NO_GATE = "visual-5fps-no-gate"
PRESET_VISUAL_5FPS_SINGLE_OWNER = "visual-5fps-single-owner"
PRESET_VISUAL_5FPS_STAGED = "visual-5fps-staged"
PRESET_DEMO215_ASYNC_FILTER_5FPS = "demo2.1.5-async-filter-5fps"
PRESET_DEMO215_COMPILED_PARALLEL_EDGETAM_5FPS = "demo2.1.5-compiled-parallel-edgetam-5fps"
PRESET_DEMO215_STAGED_PARALLEL_5FPS = "demo2.1.5-staged-parallel-5fps"
PRESET_DEMO215_LIVE_FAST_NATIVE = "demo2.1.5-live-fast-native"
PRESET_DEMO215_LIVE_QUALITY_FFS = "demo2.1.5-live-quality-ffs"
PRESET_DEMO215_MASK_ONLY_DEBUG = "demo2.1.5-mask-only-debug"
PRESET_DEMO22_ASYNC_FILTER_5FPS = "demo2.2-async-filter-5fps"
PRESET_DEMO22_STAGED_PARALLEL_5FPS = "demo2.2-staged-parallel-5fps"
PRESET_CLIMB_5 = "climb-5"
PRESET_CLIMB_10 = "climb-10"
PRESET_DIAGNOSTICS = "diagnostics"
PRESETS = (
    PRESET_NONE,
    PRESET_OFFICIAL_LOWFPS,
    PRESET_PERF_5FPS,
    PRESET_PERF_5FPS_SINGLE_OWNER,
    PRESET_PERF_5FPS_STAGED,
    PRESET_PROFESSOR_SAFE,
    PRESET_VISUAL_5FPS,
    PRESET_VISUAL_5FPS_NO_GATE,
    PRESET_VISUAL_5FPS_SINGLE_OWNER,
    PRESET_VISUAL_5FPS_STAGED,
    PRESET_DEMO215_ASYNC_FILTER_5FPS,
    PRESET_DEMO215_COMPILED_PARALLEL_EDGETAM_5FPS,
    PRESET_DEMO215_STAGED_PARALLEL_5FPS,
    PRESET_DEMO215_LIVE_FAST_NATIVE,
    PRESET_DEMO215_LIVE_QUALITY_FFS,
    PRESET_DEMO215_MASK_ONLY_DEBUG,
    PRESET_DEMO22_ASYNC_FILTER_5FPS,
    PRESET_DEMO22_STAGED_PARALLEL_5FPS,
    PRESET_CLIMB_5,
    PRESET_CLIMB_10,
    PRESET_DIAGNOSTICS,
)
PRESET_COMPAT_ALIASES = {
    PRESET_PROFESSOR_SAFE: PRESET_OFFICIAL_LOWFPS,
    PRESET_VISUAL_5FPS: PRESET_PERF_5FPS,
    PRESET_VISUAL_5FPS_NO_GATE: PRESET_PERF_5FPS,
    PRESET_VISUAL_5FPS_SINGLE_OWNER: PRESET_PERF_5FPS_SINGLE_OWNER,
    PRESET_VISUAL_5FPS_STAGED: PRESET_PERF_5FPS_STAGED,
    "live-fast-native": PRESET_DEMO215_LIVE_FAST_NATIVE,
    "live-quality-ffs": PRESET_DEMO215_LIVE_QUALITY_FFS,
    "mask-only-debug": PRESET_DEMO215_MASK_ONLY_DEBUG,
}
DEFAULT_DEVICE = "cuda"
DEFAULT_DTYPE = "bfloat16"
COMPILE_MODE_VISION_REDUCE_OVERHEAD = "vision-reduce-overhead"
COMPILE_MODE_VISION_DEFAULT = "vision-default"
COMPILE_MODE_VISION_MAX_AUTOTUNE_NO_CUDAGRAPHS = "vision-max-autotune-no-cudagraphs"
COMPILE_MODE_COMPONENTS_REDUCE_OVERHEAD = "components-reduce-overhead"
COMPILE_MODE_COMPONENTS_MAX_AUTOTUNE_NO_CUDAGRAPHS = "components-max-autotune-no-cudagraphs"
COMPILE_MODE_NONE = "none"
COMPILE_MODES = (
    COMPILE_MODE_VISION_REDUCE_OVERHEAD,
    COMPILE_MODE_VISION_DEFAULT,
    COMPILE_MODE_VISION_MAX_AUTOTUNE_NO_CUDAGRAPHS,
    COMPILE_MODE_COMPONENTS_REDUCE_OVERHEAD,
    COMPILE_MODE_COMPONENTS_MAX_AUTOTUNE_NO_CUDAGRAPHS,
    COMPILE_MODE_NONE,
)
DEFAULT_COMPILE_MODE = COMPILE_MODE_VISION_REDUCE_OVERHEAD
MASK_POSTPROCESS_HF = "hf"
MASK_POSTPROCESS_CUDA_INLINE = "cuda-inline"
MASK_POSTPROCESS_MODES = (MASK_POSTPROCESS_HF, MASK_POSTPROCESS_CUDA_INLINE)
EDGETAM_INPUT_PATH_PIL = "pil"
EDGETAM_INPUT_PATH_MODES = (EDGETAM_INPUT_PATH_PIL,)
DEFAULT_DEMO22_EXPERIMENT_MODE = EXPERIMENT_MODE_CONTROLLER_OBJECT
DEFAULT_DEMO22_CONTROLLER_LABEL = CONTROLLER_OBJECT_EXP_CONTROLLER_LABEL
DEFAULT_DEMO22_DEPTH_MIN_M = 0.1
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
GPU_GATE_MODE_SERIALIZED = "serialized"
GPU_GATE_MODE_LIMITED = "limited"
GPU_GATE_MODE_OFF = "off"
GPU_GATE_MODES = (GPU_GATE_MODE_SERIALIZED, GPU_GATE_MODE_LIMITED, GPU_GATE_MODE_OFF)
PIN_MEMORY_MODE_OFF = "off"
PIN_MEMORY_MODE_EDGE = "edge"
PIN_MEMORY_MODE_FFS = "ffs"
PIN_MEMORY_MODE_ALL = "all"
PIN_MEMORY_MODES = (PIN_MEMORY_MODE_OFF, PIN_MEMORY_MODE_EDGE, PIN_MEMORY_MODE_FFS, PIN_MEMORY_MODE_ALL)
H2D_STREAM_MODE_DEFAULT = "default"
H2D_STREAM_MODE_DEDICATED = "dedicated"
H2D_STREAM_MODES = (H2D_STREAM_MODE_DEFAULT, H2D_STREAM_MODE_DEDICATED)
DEMO22_PASS_THRESHOLD_RATIO = 0.96


def canonical_preset_name(preset: str) -> str:
    return PRESET_COMPAT_ALIASES.get(str(preset), str(preset))


def resolved_capture_group_target_fps(args: argparse.Namespace) -> float:
    value = getattr(args, "capture_group_target_fps", None)
    if value is None:
        return float(args.fusion_target_fps)
    return float(value)


def async_fusion_filter_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "enable_pcd_filter", False)) and str(getattr(args, "pcd_filter_mode", "none")) == "async"


def serialized_edgetam_first_compiled_forward_enabled(args: argparse.Namespace) -> bool:
    return (
        str(getattr(args, "gpu_pipeline_mode", "")) == GPU_PIPELINE_MODE_SEPARATE_WORKERS
        and str(getattr(args, "compile_mode", "")) == COMPILE_MODE_VISION_REDUCE_OVERHEAD
        and str(getattr(args, "gpu_gate_mode", "")) == GPU_GATE_MODE_OFF
    )


def mark_torch_cudagraph_step_begin(torch_module: Any) -> bool:
    """Mark a new compiled/CUDAGraph step when the installed torch exposes it."""
    compiler = getattr(torch_module, "compiler", None)
    marker = getattr(compiler, "cudagraph_mark_step_begin", None) if compiler is not None else None
    if marker is None:
        return False
    marker()
    return True


@contextmanager
def torch_nvtx_range(torch_module: Any, enabled: bool, label: str) -> Any:
    nvtx = getattr(getattr(torch_module, "cuda", None), "nvtx", None)
    push = getattr(nvtx, "range_push", None) if nvtx is not None else None
    pop = getattr(nvtx, "range_pop", None) if nvtx is not None else None
    if not enabled or push is None or pop is None:
        yield
        return
    push(str(label))
    try:
        yield
    finally:
        pop()


def _coerce_hf_object_ids(value: Any) -> list[int]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (int, np.integer)):
        return [int(value)]
    return [int(item) for item in list(value)]


def _normalize_hf_pred_masks_for_inline(torch_module: Any, masks: Any, object_count: int) -> Any:
    if not hasattr(masks, "ndim"):
        raise RuntimeError("HF output pred_masks is not a tensor-like object")
    while int(masks.ndim) > 4 and int(masks.shape[0]) == 1:
        masks = masks[0]
    if int(masks.ndim) == 2:
        masks = masks.unsqueeze(0).unsqueeze(0)
    elif int(masks.ndim) == 3:
        if int(masks.shape[0]) == int(object_count):
            masks = masks.unsqueeze(1)
        else:
            masks = masks.unsqueeze(0)
    elif int(masks.ndim) != 4:
        raise RuntimeError(f"expected HF pred_masks rank 2-4 after squeeze, got shape {tuple(masks.shape)}")
    if int(masks.shape[0]) != int(object_count):
        if int(masks.shape[1]) == int(object_count):
            masks = masks.transpose(0, 1)
        else:
            raise RuntimeError(
                f"HF pred_masks object dimension mismatch: shape={tuple(masks.shape)} object_count={object_count}"
            )
    if int(masks.shape[1]) != 1:
        masks = masks[:, :1, :, :]
    return masks


def extract_object_masks_from_hf_output_cuda_inline(
    *,
    torch_module: Any,
    output: Any,
    height: int,
    width: int,
) -> tuple[dict[int, np.ndarray], dict[str, float]]:
    object_ids = _coerce_hf_object_ids(getattr(output, "object_ids"))
    masks = _normalize_hf_pred_masks_for_inline(torch_module, getattr(output, "pred_masks"), len(object_ids))
    resize_start_s = time.perf_counter()
    resized = torch_module.nn.functional.interpolate(
        masks.float(),
        size=(int(height), int(width)),
        mode="bilinear",
        align_corners=False,
    )
    resize_ms = _elapsed_ms(resize_start_s, time.perf_counter())
    threshold_start_s = time.perf_counter()
    mask_bool_cuda = resized[:, 0, :, :] > 0.0
    threshold_ms = _elapsed_ms(threshold_start_s, time.perf_counter())
    cpu_start_s = time.perf_counter()
    mask_np = mask_bool_cuda.detach().cpu().numpy()
    mask_to_cpu_ms = _elapsed_ms(cpu_start_s, time.perf_counter())
    masks_by_id = {
        int(obj_id): np.ascontiguousarray(np.asarray(mask_np[idx], dtype=bool))
        for idx, obj_id in enumerate(object_ids)
    }
    return masks_by_id, {
        "mask_resize_ms": float(resize_ms),
        "mask_threshold_ms": float(threshold_ms),
        "mask_to_cpu_ms": float(mask_to_cpu_ms),
    }


def _clone_tensor_tree(torch_module: Any, value: Any) -> Any:
    def clone_if_tensor(item: Any) -> Any:
        return item.clone() if torch_module.is_tensor(item) else item

    if is_dataclass(value) and not isinstance(value, type):
        return type(value)(
            **{
                field.name: _clone_tensor_tree(torch_module, getattr(value, field.name))
                for field in fields(value)
                if field.init and hasattr(value, field.name)
            }
        )
    if isinstance(value, Mapping):
        return type(value)((key, _clone_tensor_tree(torch_module, item)) for key, item in value.items())

    pytree = getattr(getattr(torch_module, "utils", None), "_pytree", None)
    tree_map = getattr(pytree, "tree_map", None) if pytree is not None else None
    if tree_map is not None:
        return tree_map(clone_if_tensor, value)
    if torch_module.is_tensor(value):
        return value.clone()
    if isinstance(value, tuple):
        return tuple(_clone_tensor_tree(torch_module, item) for item in value)
    if isinstance(value, list):
        return [_clone_tensor_tree(torch_module, item) for item in value]
    if isinstance(value, dict):
        return {key: _clone_tensor_tree(torch_module, item) for key, item in value.items()}
    return value


def wrap_compiled_vision_encoder_outputs_for_parallel(model: Any, torch_module: Any) -> bool:
    """Clone compiled vision-encoder outputs before the next concurrent CUDAGraph replay."""
    if not hasattr(model, "vision_encoder"):
        return False
    vision_encoder = getattr(model, "vision_encoder")
    if bool(getattr(vision_encoder, "_qqtt_cudagraph_output_clone_wrapper", False)):
        return False

    class _OutputCloneWrapper(torch_module.nn.Module):
        def __init__(self, wrapped: Any) -> None:
            super().__init__()
            self.wrapped = wrapped
            self._qqtt_cudagraph_output_clone_wrapper = True

        def forward(self, *args: Any, **kwargs: Any) -> Any:
            mark_torch_cudagraph_step_begin(torch_module)
            return _clone_tensor_tree(torch_module, self.wrapped(*args, **kwargs))

    setattr(model, "vision_encoder", _OutputCloneWrapper(vision_encoder))
    return True
GPU_PIPELINE_MODE_SEPARATE_WORKERS = "separate-workers"
GPU_PIPELINE_MODE_SINGLE_OWNER = "single-owner"
GPU_PIPELINE_MODE_STAGED = "staged"
GPU_PIPELINE_MODE_OVERLAPPED_STAGES = "overlapped-stages"
GPU_PIPELINE_MODES = (
    GPU_PIPELINE_MODE_SEPARATE_WORKERS,
    GPU_PIPELINE_MODE_SINGLE_OWNER,
    GPU_PIPELINE_MODE_STAGED,
    GPU_PIPELINE_MODE_OVERLAPPED_STAGES,
)
SINGLE_OWNER_ORDER_FFS_THEN_EDGETAM = "ffs-then-edgetam"
SINGLE_OWNER_ORDER_EDGETAM_THEN_FFS = "edgetam-then-ffs"
SINGLE_OWNER_ORDER_INTERLEAVED = "interleaved"
SINGLE_OWNER_ORDERS = (
    SINGLE_OWNER_ORDER_FFS_THEN_EDGETAM,
    SINGLE_OWNER_ORDER_EDGETAM_THEN_FFS,
    SINGLE_OWNER_ORDER_INTERLEAVED,
)
STAGED_ORDER_FFS_THEN_PARALLEL_EDGETAM = "ffs-then-parallel-edgetam"
STAGED_ORDERS = (STAGED_ORDER_FFS_THEN_PARALLEL_EDGETAM,)
STAGE_SCHEDULER_MODE_MASK_GATED = "mask-gated"
STAGE_SCHEDULER_MODE_EDGE_START = "edge-start"
STAGE_SCHEDULER_MODE_BOUNDED_LOOKAHEAD = "bounded-lookahead"
STAGE_SCHEDULER_MODES = (
    STAGE_SCHEDULER_MODE_MASK_GATED,
    STAGE_SCHEDULER_MODE_EDGE_START,
    STAGE_SCHEDULER_MODE_BOUNDED_LOOKAHEAD,
)
EDGETAM_STREAM_MODE_DEFAULT = "default"
EDGETAM_STREAM_MODE_PER_CAMERA = "per-camera"
EDGETAM_STREAM_MODES = (EDGETAM_STREAM_MODE_DEFAULT, EDGETAM_STREAM_MODE_PER_CAMERA)
CAPTURE_GROUP_POLICY_LATEST = "latest"
CAPTURE_GROUP_POLICY_TIMESTAMP_NEAREST = "timestamp-nearest"
CAPTURE_GROUP_POLICY_TIMESTAMP_STRICT = "timestamp-strict"
CAPTURE_GROUP_POLICIES = (
    CAPTURE_GROUP_POLICY_LATEST,
    CAPTURE_GROUP_POLICY_TIMESTAMP_NEAREST,
    CAPTURE_GROUP_POLICY_TIMESTAMP_STRICT,
)
DEFAULT_MAX_CAPTURE_SKEW_MS = 33.4
DEFAULT_PRESET_MAX_CAPTURE_SKEW_MS = 66.7
DEFAULT_MAX_FRAME_AGE_MS = 150.0
DEFAULT_CAPTURE_BUFFER_SIZE = 4


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
    realsense_timestamp_ms: float | None
    realsense_frame_number: int | None
    timestamp_domain: str | None
    capture_arrival_perf_ns: int
    color_bgr: np.ndarray
    ir_left_u8: np.ndarray | None
    ir_right_u8: np.ndarray | None
    k_color: np.ndarray
    k_ir_left: np.ndarray | None
    t_ir_left_to_color: np.ndarray | None
    baseline_m: float
    intrinsics: CameraIntrinsics
    c2w: np.ndarray
    depth_u16: np.ndarray | None = None
    depth_scale_m_per_unit: float = 0.0

    @property
    def seq(self) -> int:
        return int(self.group_id)


@dataclass(frozen=True)
class CaptureGroup:
    group_id: int
    created_perf_s: float
    frames: dict[int, CameraFramePacket]
    group_timestamp_ns: int
    max_temporal_skew_ms: float
    per_camera_time_offset_ms: dict[int, float]
    per_camera_frame_seq: dict[int, int]
    timestamp_source: str

    @property
    def seq(self) -> int:
        return int(self.group_id)


@dataclass(frozen=True)
class TemporalGroupSelection:
    frames: dict[int, CameraFramePacket]
    timestamp_source: str
    group_timestamp_ns: int
    max_temporal_skew_ms: float
    per_camera_time_offset_ms: dict[int, float]
    per_camera_frame_seq: dict[int, int]
    age_ms: float


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
    gpu_gate_wait_ms: float
    max_temporal_skew_ms: float
    per_camera_time_offset_ms: dict[int, float]
    per_camera_frame_seq: dict[int, int]
    timestamp_source: str

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
    gpu_gate_wait_ms: float

    @property
    def seq(self) -> int:
        return int(self.group_id)


@dataclass(frozen=True)
class MaskGroup:
    group_id: int
    mask_packets: dict[int, CameraMaskPacket]
    edgetam_stage_wall_ms: float
    edgetam_stage_sum_model_ms: float
    edgetam_stage_mode: str

    @property
    def seq(self) -> int:
        return int(self.group_id)


@dataclass(frozen=True)
class StageTask:
    group_id: int
    group: CaptureGroup
    reason: str

    @property
    def seq(self) -> int:
        return int(self.group_id)


class StageWindowScheduler:
    """Bounded reservation scheduler for exact group_id stage joins."""

    def __init__(self, *, max_groups: int = 8, lookahead: int = 1) -> None:
        if int(max_groups) < 1:
            raise ValueError("max_groups must be >= 1")
        self.max_groups = int(max_groups)
        self.lookahead = max(0, int(lookahead))
        self._lock = threading.Lock()
        self._captures: dict[int, CaptureGroup] = {}
        self._depth_requested: set[int] = set()
        self._mask_requested: set[int] = set()
        self._depth_done: set[int] = set()
        self._mask_done: set[int] = set()
        self._last_edge_group_id = -1
        self.capture_stale_drops = 0
        self.depth_request_count = 0
        self.mask_request_count = 0
        self.depth_lookahead_request_count = 0

    def put_capture(self, group: CaptureGroup) -> None:
        with self._lock:
            self._captures[int(group.group_id)] = group
            self._prune_locked()

    def reserve_next_edge_task(self) -> StageTask | None:
        with self._lock:
            candidates = [
                group_id
                for group_id in self._captures
                if group_id > self._last_edge_group_id and group_id not in self._mask_requested
            ]
            if not candidates:
                return None
            group_id = max(candidates)
            group = self._captures[group_id]
            self._last_edge_group_id = int(group_id)
            self._mask_requested.add(int(group_id))
            self.mask_request_count += 1
            return StageTask(group_id=int(group_id), group=group, reason="edge-current")

    def reserve_next_depth_task(self, *, mode: str) -> StageTask | None:
        with self._lock:
            current_edge_groups = sorted(
                group_id
                for group_id in self._mask_requested
                if group_id in self._captures
                and group_id not in self._depth_requested
                and group_id not in self._depth_done
            )
            if current_edge_groups:
                group_id = int(current_edge_groups[0])
                self._depth_requested.add(group_id)
                self.depth_request_count += 1
                return StageTask(group_id=group_id, group=self._captures[group_id], reason="edge-current")

            if mode != STAGE_SCHEDULER_MODE_BOUNDED_LOOKAHEAD or self.lookahead <= 0:
                return None
            if self._last_edge_group_id < 0:
                return None

            outstanding_future = [
                group_id
                for group_id in self._depth_requested
                if group_id > self._last_edge_group_id and group_id not in self._depth_done
            ]
            if len(outstanding_future) >= self.lookahead:
                return None

            future_groups = [
                group_id
                for group_id in sorted(self._captures)
                if group_id > self._last_edge_group_id
                and group_id not in self._depth_requested
                and group_id not in self._depth_done
            ]
            if not future_groups:
                return None

            group_id = int(future_groups[0])
            self._depth_requested.add(group_id)
            self.depth_request_count += 1
            self.depth_lookahead_request_count += 1
            return StageTask(group_id=group_id, group=self._captures[group_id], reason="lookahead")

    def mark_depth_done(self, group_id: int) -> None:
        with self._lock:
            self._depth_done.add(int(group_id))

    def mark_mask_done(self, group_id: int) -> None:
        with self._lock:
            self._mask_done.add(int(group_id))

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "window_max_groups": int(self.max_groups),
                "window_lookahead": int(self.lookahead),
                "window_capture_pending": int(len(self._captures)),
                "window_capture_stale_drops": int(self.capture_stale_drops),
                "depth_requested": int(len(self._depth_requested)),
                "mask_requested": int(len(self._mask_requested)),
                "depth_done": int(len(self._depth_done)),
                "mask_done": int(len(self._mask_done)),
                "last_edge_group_id": int(self._last_edge_group_id),
                "depth_request_count": int(self.depth_request_count),
                "mask_request_count": int(self.mask_request_count),
                "depth_lookahead_request_count": int(self.depth_lookahead_request_count),
            }

    def _prune_locked(self) -> None:
        while len(self._captures) > self.max_groups:
            oldest = min(self._captures)
            self._captures.pop(oldest, None)
            self.capture_stale_drops += 1
            for table in (
                self._depth_requested,
                self._mask_requested,
                self._depth_done,
                self._mask_done,
            ):
                table.discard(oldest)


class SameGroupJoinBuffer:
    """Thread-safe bounded join buffer for capture, depth, and mask groups."""

    def __init__(self, *, max_groups: int = 8) -> None:
        if int(max_groups) < 1:
            raise ValueError("max_groups must be >= 1")
        self.max_groups = int(max_groups)
        self._lock = threading.Lock()
        self._captures: dict[int, CaptureGroup] = {}
        self._depths: dict[int, DepthGroup] = {}
        self._masks: dict[int, MaskGroup] = {}
        self.capture_stale_drops = 0
        self.depth_stale_drops = 0
        self.mask_stale_drops = 0
        self.ready_join_count = 0

    def put_capture(self, group: CaptureGroup) -> None:
        with self._lock:
            self._captures[int(group.group_id)] = group
            self._prune_locked()

    def put_depth(self, depth: DepthGroup) -> None:
        with self._lock:
            self._depths[int(depth.group_id)] = depth
            self._prune_locked()

    def put_mask(self, mask: MaskGroup) -> None:
        with self._lock:
            self._masks[int(mask.group_id)] = mask
            self._prune_locked()

    def pop_latest_ready(self) -> tuple[CaptureGroup, DepthGroup, MaskGroup] | None:
        with self._lock:
            ready = set(self._captures) & set(self._depths) & set(self._masks)
            if not ready:
                return None
            group_id = max(ready)
            capture = self._captures.pop(group_id)
            depth = self._depths.pop(group_id)
            mask = self._masks.pop(group_id)
            self.ready_join_count += 1
            self._drop_older_than_locked(group_id)
            return capture, depth, mask

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "max_groups": int(self.max_groups),
                "capture_pending": int(len(self._captures)),
                "depth_pending": int(len(self._depths)),
                "mask_pending": int(len(self._masks)),
                "capture_stale_drops": int(self.capture_stale_drops),
                "depth_stale_drops": int(self.depth_stale_drops),
                "mask_stale_drops": int(self.mask_stale_drops),
                "ready_join_count": int(self.ready_join_count),
            }

    def _drop_older_than_locked(self, group_id: int) -> None:
        for table, counter_name in (
            (self._captures, "capture_stale_drops"),
            (self._depths, "depth_stale_drops"),
            (self._masks, "mask_stale_drops"),
        ):
            stale = [old_group_id for old_group_id in table if old_group_id < group_id]
            for old_group_id in stale:
                table.pop(old_group_id, None)
            setattr(self, counter_name, getattr(self, counter_name) + len(stale))

    def _prune_locked(self) -> None:
        for table, counter_name in (
            (self._captures, "capture_stale_drops"),
            (self._depths, "depth_stale_drops"),
            (self._masks, "mask_stale_drops"),
        ):
            while len(table) > self.max_groups:
                oldest = min(table)
                table.pop(oldest, None)
                setattr(self, counter_name, getattr(self, counter_name) + 1)


@dataclass(frozen=True)
class PreparedEdgeTamFrame:
    pixel_values: Any
    original_sizes: Any
    frame_idx: int
    preprocess_ms: float
    edge_h2d_profile: dict[str, Any]
    batch_vision_encoder_ms: float


@dataclass(frozen=True)
class CompleteInferenceGroup:
    group_id: int
    capture_group: CaptureGroup
    depth_group: DepthGroup
    mask_packets: dict[int, CameraMaskPacket]
    ffs_cycle_ms: float
    edgetam_cycle_ms: float
    edgetam_stage_wall_ms: float
    edgetam_stage_sum_model_ms: float
    stage_barrier_ms: float
    total_gpu_owner_ms: float
    pipeline_mode: str
    internal_order: str

    @property
    def seq(self) -> int:
        return int(self.group_id)


@dataclass(frozen=True)
class RawFusedPcdPacket:
    group_id: int
    created_perf_s: float
    raw_object: FusedLayerCloud | None
    raw_controller: FusedLayerCloud | None
    raw_fusion_ms: float
    build_object_raw_ms: float
    build_controller_raw_ms: float
    object_raw_points: int
    controller_raw_points: int
    ffs_cycle_ms: float
    edgetam_ms_by_camera: dict[int, float]
    ffs_gpu_gate_wait_ms: float
    edgetam_gpu_gate_wait_ms_by_camera: dict[int, float]
    capture_temporal_skew_ms: float
    capture_time_offsets_ms_by_camera: dict[int, float]
    timestamp_source: str

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
    ffs_gpu_gate_wait_ms: float
    edgetam_gpu_gate_wait_ms_by_camera: dict[int, float]
    capture_temporal_skew_ms: float
    capture_time_offsets_ms_by_camera: dict[int, float]
    timestamp_source: str

    @property
    def seq(self) -> int:
        return int(self.group_id)

    @property
    def object_point_count(self) -> int:
        return int(self.object_points_m.shape[0])

    @property
    def controller_point_count(self) -> int:
        return int(self.controller_points_m.shape[0])


PROFILE_FLAG_ATTRS = (
    "profile_cuda_events",
    "profile_pipeline",
    "profile_filter",
    "profile_filter_detail",
    "profile_visualization",
    "profile_gpu_gate",
    "profile_h2d",
    "profile_edgetam_stages",
    "profile_sync",
    "profile_nsys_markers",
    "gpu_sampling",
)

GPU_SAMPLING_BACKENDS = ("nvml",)


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


class MsWindowStats:
    def __init__(self, maxlen: int = 128) -> None:
        self._lock = threading.Lock()
        self._values: deque[float] = deque(maxlen=int(maxlen))

    def record(self, value_ms: float) -> None:
        with self._lock:
            self._values.append(float(value_ms))

    @property
    def median(self) -> float:
        with self._lock:
            values = list(self._values)
        if not values:
            return 0.0
        return float(np.median(np.asarray(values, dtype=np.float32)))

    @property
    def latest(self) -> float:
        with self._lock:
            if not self._values:
                return 0.0
            return float(self._values[-1])

    @property
    def p95(self) -> float:
        with self._lock:
            values = list(self._values)
        if not values:
            return 0.0
        return float(np.percentile(np.asarray(values, dtype=np.float32), 95))

    @property
    def max(self) -> float:
        with self._lock:
            values = list(self._values)
        if not values:
            return 0.0
        return float(np.max(np.asarray(values, dtype=np.float32)))


class GpuInferenceGate:
    def __init__(self, *, mode: str, max_concurrent: int) -> None:
        if mode not in GPU_GATE_MODES:
            raise ValueError(f"Unsupported GPU gate mode: {mode}")
        self.mode = str(mode)
        if self.mode == GPU_GATE_MODE_OFF:
            self.max_concurrent = 0
            self._sem = None
        else:
            self.max_concurrent = max(1, int(max_concurrent))
            self._sem = threading.Semaphore(self.max_concurrent)

    @contextmanager
    def acquire(self, *, stage: str, camera_idx: int | None, group_id: int):
        del stage, camera_idx, group_id
        if self._sem is None:
            yield 0.0
            return
        wait_start_s = time.perf_counter()
        self._sem.acquire()
        wait_ms = _elapsed_ms(wait_start_s, time.perf_counter())
        try:
            yield wait_ms
        finally:
            self._sem.release()


def edge_pin_memory_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "pin_memory", False)) and str(getattr(args, "pin_memory_mode", PIN_MEMORY_MODE_OFF)) in {
        PIN_MEMORY_MODE_EDGE,
        PIN_MEMORY_MODE_ALL,
    }


def ffs_pin_memory_requested(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "pin_memory", False)) and str(getattr(args, "pin_memory_mode", PIN_MEMORY_MODE_OFF)) in {
        PIN_MEMORY_MODE_FFS,
        PIN_MEMORY_MODE_ALL,
    }


class PinnedPixelValueStager:
    """Reusable pinned CPU + CUDA staging ring for EdgeTAM processor pixel_values."""

    def __init__(
        self,
        *,
        torch_module: Any,
        device: str,
        ring_size: int,
        h2d_stream_mode: str,
        verify_copies: bool = False,
    ) -> None:
        self.torch = torch_module
        self.device = str(device)
        self.ring_size = max(1, int(ring_size))
        self.h2d_stream_mode = str(h2d_stream_mode)
        if self.h2d_stream_mode not in H2D_STREAM_MODES:
            raise ValueError(f"Unsupported H2D stream mode: {self.h2d_stream_mode}")
        self.verify_copies = bool(verify_copies)
        self._slots: list[Any] = []
        self._device_slots: list[Any] = []
        self._events: list[Any | None] = []
        self._next_slot = 0
        self._verified = 0
        self._slot_shape: tuple[int, ...] | None = None
        self._slot_device_dtype: Any | None = None
        self._h2d_stream = (
            torch_module.cuda.Stream()
            if self.h2d_stream_mode == H2D_STREAM_MODE_DEDICATED and self.device.startswith("cuda")
            else None
        )

    def _ensure_slots(self, sample_cpu: Any, *, dtype: Any) -> None:
        shape = tuple(int(item) for item in sample_cpu.shape)
        if self._slots and self._slot_shape == shape and self._slot_device_dtype == dtype:
            return
        if getattr(sample_cpu, "is_cuda", False):
            raise RuntimeError("PinnedPixelValueStager expects CPU pixel_values from the HF processor.")
        self._slots = [
            self.torch.empty(shape, dtype=sample_cpu.dtype, pin_memory=True)
            for _ in range(self.ring_size)
        ]
        self._device_slots = [
            self.torch.empty(shape, dtype=dtype, device=self.device)
            for _ in range(self.ring_size)
        ] if self.device.startswith("cuda") else []
        self._events = [None for _ in range(self.ring_size)]
        self._slot_shape = shape
        self._slot_device_dtype = dtype

    def stage(self, pixel_values_cpu: Any, *, dtype: Any, consumer_stream: Any | None = None) -> tuple[Any, dict[str, Any]]:
        self._ensure_slots(pixel_values_cpu, dtype=dtype)
        slot_idx = int(self._next_slot)
        self._next_slot = (self._next_slot + 1) % self.ring_size
        prior_event = self._events[slot_idx]
        slot_reuse_wait_ms = 0.0
        if prior_event is not None and not prior_event.query():
            wait_start_s = time.perf_counter()
            prior_event.synchronize()
            slot_reuse_wait_ms = _elapsed_ms(wait_start_s, time.perf_counter())

        slot = self._slots[slot_idx]
        pin_copy_start_s = time.perf_counter()
        slot.copy_(pixel_values_cpu, non_blocking=False)
        pin_copy_ms = _elapsed_ms(pin_copy_start_s, time.perf_counter())
        if self.verify_copies and self._verified < 5 and not self.torch.equal(slot, pixel_values_cpu):
            raise RuntimeError("Pinned EdgeTAM pixel_values staging copy mismatch.")
        self._verified += 1

        stream = self._h2d_stream
        h2d_enqueue_start_s = time.perf_counter()
        device_slot = self._device_slots[slot_idx] if self._device_slots else None
        if stream is not None:
            with self.torch.cuda.stream(stream):
                if device_slot is not None:
                    device_slot.copy_(slot, non_blocking=True)
                    pixel_values_cuda = device_slot
                else:
                    pixel_values_cuda = slot.to(device=self.device, dtype=dtype, non_blocking=True)
                event = self.torch.cuda.Event()
                event.record(stream)
            h2d_enqueue_ms = _elapsed_ms(h2d_enqueue_start_s, time.perf_counter())
            h2d_wait_start_s = time.perf_counter()
            wait_stream = consumer_stream if consumer_stream is not None else self.torch.cuda.current_stream()
            wait_stream.wait_event(event)
            h2d_wait_ms = _elapsed_ms(h2d_wait_start_s, time.perf_counter())
        else:
            if device_slot is not None:
                device_slot.copy_(slot, non_blocking=True)
                pixel_values_cuda = device_slot
            else:
                pixel_values_cuda = slot.to(device=self.device, dtype=dtype, non_blocking=True)
            event = self.torch.cuda.Event()
            event.record(self.torch.cuda.current_stream())
            if consumer_stream is not None:
                consumer_stream.wait_event(event)
            h2d_enqueue_ms = _elapsed_ms(h2d_enqueue_start_s, time.perf_counter())
            h2d_wait_ms = 0.0

        self._events[slot_idx] = event
        return pixel_values_cuda, {
            "pin_memory": True,
            "processor_device": "cpu",
            "processor_is_pinned": False,
            "pinned_slot_idx": int(slot_idx),
            "device_slot_reused": bool(device_slot is not None),
            "device_slot_idx": int(slot_idx) if device_slot is not None else -1,
            "device_slot_dtype": str(dtype),
            "pin_copy_ms": float(pin_copy_ms),
            "slot_reuse_wait_ms": float(slot_reuse_wait_ms),
            "h2d_enqueue_ms": float(h2d_enqueue_ms),
            "h2d_wait_ms": float(h2d_wait_ms),
            "h2d_stream_mode": self.h2d_stream_mode,
        }

    def mark_consumed(self, slot_idx: int, consumer_stream: Any | None = None) -> None:
        if not self.device.startswith("cuda") or int(slot_idx) < 0:
            return
        event = self.torch.cuda.Event()
        record_stream = consumer_stream if consumer_stream is not None else self.torch.cuda.current_stream()
        event.record(record_stream)
        self._events[int(slot_idx) % self.ring_size] = event


def _normalize_label(label: str) -> str:
    return str(label).strip().lower().replace("_", " ").replace("-", " ")


def controller_prompt_for_experiment_mode(experiment_mode: str) -> str:
    if experiment_mode == EXPERIMENT_MODE_CONTROLLER_OBJECT:
        return CONTROLLER_OBJECT_EXP_CONTROLLER_LABEL
    if experiment_mode == EXPERIMENT_MODE_DEMO:
        return DEMO_MODE_CONTROLLER_LABEL
    raise ValueError(f"Unsupported experiment mode: {experiment_mode}")


def resolved_experiment_mode(args: argparse.Namespace) -> str:
    mode = str(getattr(args, "experiment_mode", DEFAULT_EXPERIMENT_MODE) or DEFAULT_EXPERIMENT_MODE)
    if mode not in EXPERIMENT_MODES:
        raise ValueError(f"Unsupported experiment mode: {mode}")
    return mode


def controller_prompt_matches_experiment_mode(args: argparse.Namespace) -> bool:
    expected = controller_prompt_for_experiment_mode(resolved_experiment_mode(args))
    return str(getattr(args, "controller_prompt", "")) == expected


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
    if controller_tracking_enabled(track_mode):
        layers.append(
            SemanticLayerSpec(
                obj_id=CONTROLLER_ID,
                label=str(controller_label),
                default_postprocess=controller_postprocess,
            )
        )
    if object_tracking_enabled(track_mode):
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


def _profile_stats(values: Sequence[float]) -> dict[str, float]:
    arr = np.asarray([float(value) for value in values if np.isfinite(float(value))], dtype=np.float64)
    if arr.size == 0:
        return {"median": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def _nested_get(record: dict[str, Any], path: Sequence[str]) -> Any:
    value: Any = record
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _deep_update_dict(target: dict[str, Any], update: dict[str, Any]) -> None:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update_dict(target[key], value)
        else:
            target[key] = value


def _series_for_path(records: Sequence[dict[str, Any]], path: Sequence[str]) -> list[float]:
    values: list[float] = []
    for record in records:
        value = _nested_get(record, path)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _event_fps(records: Sequence[dict[str, Any]], path: Sequence[str]) -> float:
    times = _series_for_path(records, path)
    if len(times) < 2:
        return 0.0
    elapsed = max(times) - min(times)
    if elapsed <= 0:
        return 0.0
    return float((len(times) - 1) / elapsed)


def _event_period_stats_ms(records: Sequence[dict[str, Any]], path: Sequence[str]) -> dict[str, float]:
    times = sorted(_series_for_path(records, path))
    if len(times) < 2:
        return _profile_stats([])
    periods_ms = [
        (float(later) - float(earlier)) * 1000.0
        for earlier, later in zip(times[:-1], times[1:])
        if float(later) >= float(earlier)
    ]
    return _profile_stats(periods_ms)


GPU_SAMPLE_METRIC_NAMES: tuple[str, ...] = (
    "gpu_util_pct",
    "memory_util_pct",
    "memory_used_mb",
    "memory_total_mb",
    "power_w",
    "power_limit_w",
    "sm_clock_mhz",
    "mem_clock_mhz",
    "temperature_c",
)


def summarize_gpu_samples(samples: Sequence[dict[str, Any]], *, start_s: float = 0.0) -> dict[str, Any]:
    selected = [
        sample for sample in samples
        if isinstance(sample, dict) and float(sample.get("sample_s", 0.0) or 0.0) >= float(start_s)
    ]
    summary: dict[str, Any] = {
        "sample_count": int(len(selected)),
        "start_s": float(start_s),
        "first_sample_s": None,
        "last_sample_s": None,
        "duration_s": 0.0,
        "metrics": {},
    }
    if selected:
        first_s = float(selected[0].get("sample_s", 0.0) or 0.0)
        last_s = float(selected[-1].get("sample_s", first_s) or first_s)
        summary.update(
            {
                "first_sample_s": first_s,
                "last_sample_s": last_s,
                "duration_s": max(0.0, last_s - first_s),
            }
        )
    for name in GPU_SAMPLE_METRIC_NAMES:
        values = [
            float(sample[name])
            for sample in selected
            if isinstance(sample.get(name), (int, float)) and np.isfinite(float(sample[name]))
        ]
        summary["metrics"][name] = _profile_stats(values)
    return summary


class GpuUtilizationSampler:
    def __init__(
        self,
        *,
        enabled: bool,
        interval_s: float,
        backend: str,
        device_index: int,
        rel_time_fn: Callable[[], float],
    ) -> None:
        self.enabled = bool(enabled)
        self.interval_s = max(0.05, float(interval_s))
        self.requested_backend = str(backend)
        self.device_index = int(device_index)
        self._rel_time_fn = rel_time_fn
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._samples: list[dict[str, Any]] = []
        self._errors: list[str] = []
        self._backend_used: str | None = None
        self._nvml: Any | None = None
        self._nvml_handle: Any | None = None

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="demo2-gpu-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(3.0, self.interval_s * 2.0 + 2.0))
        self._shutdown_nvml()

    def samples_snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(sample) for sample in self._samples]

    def diagnostics(self) -> dict[str, Any]:
        with self._lock:
            errors = list(self._errors)
            sample_count = len(self._samples)
        return {
            "enabled": self.enabled,
            "requested_backend": self.requested_backend,
            "backend_used": self._backend_used,
            "device_index": self.device_index,
            "interval_s": self.interval_s,
            "sample_count": int(sample_count),
            "errors": errors[:10],
        }

    def _append_error(self, message: str) -> None:
        with self._lock:
            if len(self._errors) < 20:
                self._errors.append(str(message))

    def _append_sample(self, sample: dict[str, Any]) -> None:
        sample = dict(sample)
        if "sample_s" not in sample:
            sample["sample_s"] = float(self._rel_time_fn())
        sample.setdefault("device_index", self.device_index)
        with self._lock:
            self._samples.append(sample)

    def _run(self) -> None:
        sampler = self._make_sampler()
        if sampler is None:
            return
        while not self._stop_event.is_set():
            try:
                sample = sampler()
                sample["sample_s"] = float(self._rel_time_fn())
                sample["device_index"] = self.device_index
                sample["source"] = self._backend_used
                self._append_sample(sample)
            except Exception as exc:
                self._append_error(f"{self._backend_used or self.requested_backend}: {type(exc).__name__}: {exc}")
            self._stop_event.wait(self.interval_s)

    def _make_sampler(self) -> Callable[[], dict[str, Any]] | None:
        requested = self.requested_backend
        if requested not in GPU_SAMPLING_BACKENDS:
            self._append_error(f"unsupported backend {requested!r}")
            return None
        sampler = self._try_make_nvml_sampler()
        if sampler is not None:
            self._backend_used = "nvml"
            return sampler
        return None

    def _try_make_nvml_sampler(self) -> Callable[[], dict[str, Any]] | None:
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        except Exception as exc:
            self._append_error(f"nvml unavailable: {type(exc).__name__}: {exc}")
            self._shutdown_nvml()
            return None

        def _sample() -> dict[str, Any]:
            nvml = self._nvml
            handle = self._nvml_handle
            util = nvml.nvmlDeviceGetUtilizationRates(handle)
            memory = nvml.nvmlDeviceGetMemoryInfo(handle)
            sample = {
                "gpu_util_pct": float(util.gpu),
                "memory_util_pct": float(util.memory),
                "memory_used_mb": float(memory.used) / (1024.0 * 1024.0),
                "memory_total_mb": float(memory.total) / (1024.0 * 1024.0),
            }
            optional_calls: tuple[tuple[str, Callable[[], float]], ...] = (
                ("power_w", lambda: float(nvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0),
                ("power_limit_w", lambda: float(nvml.nvmlDeviceGetEnforcedPowerLimit(handle)) / 1000.0),
                ("sm_clock_mhz", lambda: float(nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_SM))),
                ("mem_clock_mhz", lambda: float(nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_MEM))),
                ("temperature_c", lambda: float(nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU))),
            )
            for name, call in optional_calls:
                try:
                    sample[name] = call()
                except Exception:
                    pass
            return sample

        return _sample

    def _shutdown_nvml(self) -> None:
        nvml = self._nvml
        self._nvml = None
        self._nvml_handle = None
        if nvml is not None:
            try:
                nvml.nvmlShutdown()
            except Exception:
                pass


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


def _explicit_cli_options(argv: Sequence[str] | None) -> set[str]:
    tokens = list(sys.argv[1:] if argv is None else argv)
    explicit: set[str] = set()
    for token in tokens:
        if not str(token).startswith("--"):
            continue
        flag = str(token).split("=", maxsplit=1)[0]
        explicit.add(flag)
        if flag.startswith("--no-"):
            explicit.add("--" + flag[len("--no-"):])
    return explicit


def _set_if_not_explicit(
    args: argparse.Namespace,
    explicit: set[str],
    *,
    flag: str,
    attr: str,
    value: Any,
) -> None:
    if flag not in explicit:
        setattr(args, attr, value)


def _normalize_pin_memory_options(args: argparse.Namespace, explicit: set[str]) -> None:
    explicit_mode = "--pin-memory-mode" in explicit
    if explicit_mode:
        setattr(args, "pin_memory", str(args.pin_memory_mode) != PIN_MEMORY_MODE_OFF)
        return
    if bool(getattr(args, "pin_memory", False)):
        current_mode = str(getattr(args, "pin_memory_mode", PIN_MEMORY_MODE_OFF))
        setattr(args, "pin_memory_mode", current_mode if current_mode != PIN_MEMORY_MODE_OFF else PIN_MEMORY_MODE_ALL)
    else:
        setattr(args, "pin_memory_mode", PIN_MEMORY_MODE_OFF)


def apply_preset_defaults(args: argparse.Namespace, *, explicit_options: set[str] | None = None) -> argparse.Namespace:
    explicit = set() if explicit_options is None else set(explicit_options)
    raw_preset = str(getattr(args, "preset", PRESET_NONE))
    if raw_preset not in PRESETS:
        raise ValueError(f"Unsupported Demo 2.1 preset: {raw_preset}")
    preset = canonical_preset_name(raw_preset)
    setattr(args, "preset_canonical", preset)

    if preset not in {"", PRESET_NONE}:
        common: tuple[tuple[str, str, Any], ...] = (
            ("--profile", "profile", "848x480"),
            ("--fps", "fps", DEFAULT_PRESET_CAPTURE_FPS),
            ("--depth-source", "depth_source", DEPTH_SOURCE_FFS),
            ("--ffs-worker-mode", "ffs_worker_mode", "shared"),
            ("--ffs-schedule", "ffs_schedule", "strict3-latest"),
            ("--edgetam-worker-mode", "edgetam_worker_mode", "per-camera"),
            ("--edgetam-model-topology", "edgetam_model_topology", "replicated"),
            ("--compile-mode", "compile_mode", DEFAULT_COMPILE_MODE),
            ("--dtype", "dtype", DEFAULT_DTYPE),
            ("--gpu-gate-mode", "gpu_gate_mode", GPU_GATE_MODE_OFF),
            ("--gpu-gate-max-concurrent", "gpu_gate_max_concurrent", 0),
            ("--fusion-timeout-ms", "fusion_timeout_ms", 250.0),
            ("--capture-group-policy", "capture_group_policy", CAPTURE_GROUP_POLICY_TIMESTAMP_NEAREST),
            ("--max-capture-skew-ms", "max_capture_skew_ms", DEFAULT_PRESET_MAX_CAPTURE_SKEW_MS),
            ("--max-frame-age-ms", "max_frame_age_ms", DEFAULT_MAX_FRAME_AGE_MS),
            ("--capture-buffer-size", "capture_buffer_size", DEFAULT_CAPTURE_BUFFER_SIZE),
            ("--drop-skewed-groups", "drop_skewed_groups", True),
        )
        for flag, attr, value in common:
            _set_if_not_explicit(args, explicit, flag=flag, attr=attr, value=value)

        if preset == PRESET_OFFICIAL_LOWFPS:
            _set_if_not_explicit(args, explicit, flag="--fusion-target-fps", attr="fusion_target_fps", value=2.0)
            _set_if_not_explicit(args, explicit, flag="--render-mode", attr="render_mode", value="pointcloud")
        elif preset == PRESET_PERF_5FPS:
            _set_if_not_explicit(args, explicit, flag="--fusion-target-fps", attr="fusion_target_fps", value=5.0)
            _set_if_not_explicit(args, explicit, flag="--render-mode", attr="render_mode", value="pointcloud")
            _set_if_not_explicit(args, explicit, flag="--gpu-gate-mode", attr="gpu_gate_mode", value=GPU_GATE_MODE_OFF)
            _set_if_not_explicit(args, explicit, flag="--gpu-gate-max-concurrent", attr="gpu_gate_max_concurrent", value=0)
        elif preset == PRESET_PERF_5FPS_SINGLE_OWNER:
            _set_if_not_explicit(args, explicit, flag="--fusion-target-fps", attr="fusion_target_fps", value=5.0)
            _set_if_not_explicit(args, explicit, flag="--render-mode", attr="render_mode", value="pointcloud")
            _set_if_not_explicit(args, explicit, flag="--gpu-pipeline-mode", attr="gpu_pipeline_mode", value=GPU_PIPELINE_MODE_SINGLE_OWNER)
            _set_if_not_explicit(args, explicit, flag="--single-owner-order", attr="single_owner_order", value=SINGLE_OWNER_ORDER_FFS_THEN_EDGETAM)
            _set_if_not_explicit(args, explicit, flag="--gpu-gate-mode", attr="gpu_gate_mode", value=GPU_GATE_MODE_OFF)
            _set_if_not_explicit(args, explicit, flag="--gpu-gate-max-concurrent", attr="gpu_gate_max_concurrent", value=0)
        elif preset == PRESET_PERF_5FPS_STAGED:
            _set_if_not_explicit(args, explicit, flag="--fusion-target-fps", attr="fusion_target_fps", value=5.0)
            _set_if_not_explicit(args, explicit, flag="--render-mode", attr="render_mode", value="pointcloud")
            _set_if_not_explicit(args, explicit, flag="--gpu-pipeline-mode", attr="gpu_pipeline_mode", value=GPU_PIPELINE_MODE_STAGED)
            _set_if_not_explicit(args, explicit, flag="--staged-order", attr="staged_order", value=STAGED_ORDER_FFS_THEN_PARALLEL_EDGETAM)
            _set_if_not_explicit(args, explicit, flag="--edgetam-stream-mode", attr="edgetam_stream_mode", value=EDGETAM_STREAM_MODE_PER_CAMERA)
            _set_if_not_explicit(args, explicit, flag="--edgetam-model-topology", attr="edgetam_model_topology", value=EDGETAM_MODEL_TOPOLOGY_REPLICATED)
            _set_if_not_explicit(args, explicit, flag="--gpu-gate-mode", attr="gpu_gate_mode", value=GPU_GATE_MODE_OFF)
            _set_if_not_explicit(args, explicit, flag="--gpu-gate-max-concurrent", attr="gpu_gate_max_concurrent", value=0)
        elif preset in {PRESET_DEMO22_ASYNC_FILTER_5FPS, PRESET_DEMO215_ASYNC_FILTER_5FPS}:
            if preset == PRESET_DEMO215_ASYNC_FILTER_5FPS:
                _set_if_not_explicit(args, explicit, flag="--depth-source", attr="depth_source", value=DEPTH_SOURCE_REALSENSE)
            _set_if_not_explicit(args, explicit, flag="--fps", attr="fps", value=DEFAULT_PRESET_CAPTURE_FPS)
            _set_if_not_explicit(args, explicit, flag="--fusion-target-fps", attr="fusion_target_fps", value=15.0)
            _set_if_not_explicit(args, explicit, flag="--render-mode", attr="render_mode", value="pointcloud")
            if preset == PRESET_DEMO22_ASYNC_FILTER_5FPS:
                _set_if_not_explicit(args, explicit, flag="--ffs-trt-batch-size", attr="ffs_trt_batch_size", value=3)
                _set_if_not_explicit(
                    args,
                    explicit,
                    flag="--edgetam-batch-vision-encoder",
                    attr="edgetam_batch_vision_encoder",
                    value=True,
                )
            _set_if_not_explicit(args, explicit, flag="--gpu-pipeline-mode", attr="gpu_pipeline_mode", value=GPU_PIPELINE_MODE_SINGLE_OWNER)
            _set_if_not_explicit(args, explicit, flag="--single-owner-order", attr="single_owner_order", value=SINGLE_OWNER_ORDER_FFS_THEN_EDGETAM)
            _set_if_not_explicit(args, explicit, flag="--edgetam-model-topology", attr="edgetam_model_topology", value=EDGETAM_MODEL_TOPOLOGY_SHARED)
            _set_if_not_explicit(args, explicit, flag="--edgetam-prewarm-compile", attr="edgetam_prewarm_compile", value=True)
            _set_if_not_explicit(args, explicit, flag="--edgetam-prewarm-runs", attr="edgetam_prewarm_runs", value=1)
            _set_if_not_explicit(args, explicit, flag="--parallel-init", attr="parallel_init", value=True)
            _set_if_not_explicit(args, explicit, flag="--track-mode", attr="track_mode", value=TRACK_MODE_CONTROLLER_OBJECT)
            _set_if_not_explicit(args, explicit, flag="--init-mode", attr="init_mode", value="sam31-first-frame")
            _set_if_not_explicit(args, explicit, flag="--sam31-cache-init-model", attr="sam31_cache_init_model", value=True)
            _set_if_not_explicit(args, explicit, flag="--sam31-keep-runtime-until-all-cameras-init", attr="sam31_keep_runtime_until_all_cameras_init", value=True)
            _set_if_not_explicit(args, explicit, flag="--object-prompt", attr="object_prompt", value="stuffed animal")
            _set_if_not_explicit(args, explicit, flag="--experiment-mode", attr="experiment_mode", value=DEFAULT_DEMO22_EXPERIMENT_MODE)
            _set_if_not_explicit(
                args,
                explicit,
                flag="--controller-prompt",
                attr="controller_prompt",
                value=controller_prompt_for_experiment_mode(resolved_experiment_mode(args)),
            )
            _set_if_not_explicit(args, explicit, flag="--depth-min-m", attr="depth_min_m", value=DEFAULT_DEMO22_DEPTH_MIN_M)
            _set_if_not_explicit(args, explicit, flag="--enable-pcd-filter", attr="enable_pcd_filter", value=True)
            _set_if_not_explicit(args, explicit, flag="--pcd-filter-mode", attr="pcd_filter_mode", value="async")
            _set_if_not_explicit(args, explicit, flag="--gpu-gate-mode", attr="gpu_gate_mode", value=GPU_GATE_MODE_OFF)
            _set_if_not_explicit(args, explicit, flag="--gpu-gate-max-concurrent", attr="gpu_gate_max_concurrent", value=0)
        elif preset == PRESET_DEMO215_COMPILED_PARALLEL_EDGETAM_5FPS:
            _set_if_not_explicit(args, explicit, flag="--depth-source", attr="depth_source", value=DEPTH_SOURCE_REALSENSE)
            _set_if_not_explicit(args, explicit, flag="--fps", attr="fps", value=DEFAULT_PRESET_CAPTURE_FPS)
            _set_if_not_explicit(args, explicit, flag="--fusion-target-fps", attr="fusion_target_fps", value=15.0)
            _set_if_not_explicit(args, explicit, flag="--render-mode", attr="render_mode", value="pointcloud")
            _set_if_not_explicit(args, explicit, flag="--gpu-pipeline-mode", attr="gpu_pipeline_mode", value=GPU_PIPELINE_MODE_SEPARATE_WORKERS)
            _set_if_not_explicit(args, explicit, flag="--edgetam-stream-mode", attr="edgetam_stream_mode", value=EDGETAM_STREAM_MODE_PER_CAMERA)
            _set_if_not_explicit(args, explicit, flag="--edgetam-model-topology", attr="edgetam_model_topology", value=EDGETAM_MODEL_TOPOLOGY_REPLICATED)
            _set_if_not_explicit(args, explicit, flag="--compile-mode", attr="compile_mode", value=COMPILE_MODE_VISION_REDUCE_OVERHEAD)
            _set_if_not_explicit(args, explicit, flag="--parallel-init", attr="parallel_init", value=True)
            _set_if_not_explicit(args, explicit, flag="--track-mode", attr="track_mode", value=TRACK_MODE_CONTROLLER_OBJECT)
            _set_if_not_explicit(args, explicit, flag="--init-mode", attr="init_mode", value="sam31-first-frame")
            _set_if_not_explicit(args, explicit, flag="--sam31-cache-init-model", attr="sam31_cache_init_model", value=True)
            _set_if_not_explicit(args, explicit, flag="--sam31-keep-runtime-until-all-cameras-init", attr="sam31_keep_runtime_until_all_cameras_init", value=True)
            _set_if_not_explicit(args, explicit, flag="--object-prompt", attr="object_prompt", value="stuffed animal")
            _set_if_not_explicit(args, explicit, flag="--experiment-mode", attr="experiment_mode", value=DEFAULT_DEMO22_EXPERIMENT_MODE)
            _set_if_not_explicit(
                args,
                explicit,
                flag="--controller-prompt",
                attr="controller_prompt",
                value=controller_prompt_for_experiment_mode(resolved_experiment_mode(args)),
            )
            _set_if_not_explicit(args, explicit, flag="--depth-min-m", attr="depth_min_m", value=DEFAULT_DEMO22_DEPTH_MIN_M)
            _set_if_not_explicit(args, explicit, flag="--enable-pcd-filter", attr="enable_pcd_filter", value=True)
            _set_if_not_explicit(args, explicit, flag="--pcd-filter-mode", attr="pcd_filter_mode", value="async")
            _set_if_not_explicit(args, explicit, flag="--gpu-gate-mode", attr="gpu_gate_mode", value=GPU_GATE_MODE_OFF)
            _set_if_not_explicit(args, explicit, flag="--gpu-gate-max-concurrent", attr="gpu_gate_max_concurrent", value=0)
            _set_if_not_explicit(args, explicit, flag="--pin-memory", attr="pin_memory", value=True)
            _set_if_not_explicit(args, explicit, flag="--pin-memory-mode", attr="pin_memory_mode", value=PIN_MEMORY_MODE_EDGE)
            _set_if_not_explicit(args, explicit, flag="--pinned-ring-size", attr="pinned_ring_size", value=1)
            _set_if_not_explicit(args, explicit, flag="--h2d-stream-mode", attr="h2d_stream_mode", value=H2D_STREAM_MODE_DEFAULT)
        elif preset == PRESET_DEMO215_LIVE_FAST_NATIVE:
            _set_if_not_explicit(args, explicit, flag="--depth-source", attr="depth_source", value=DEPTH_SOURCE_REALSENSE)
            _set_if_not_explicit(args, explicit, flag="--fps", attr="fps", value=DEFAULT_PRESET_CAPTURE_FPS)
            _set_if_not_explicit(args, explicit, flag="--fusion-target-fps", attr="fusion_target_fps", value=45.0)
            _set_if_not_explicit(args, explicit, flag="--render-mode", attr="render_mode", value="pointcloud")
            _set_if_not_explicit(args, explicit, flag="--gpu-pipeline-mode", attr="gpu_pipeline_mode", value=GPU_PIPELINE_MODE_SINGLE_OWNER)
            _set_if_not_explicit(args, explicit, flag="--single-owner-order", attr="single_owner_order", value=SINGLE_OWNER_ORDER_FFS_THEN_EDGETAM)
            _set_if_not_explicit(args, explicit, flag="--edgetam-model-topology", attr="edgetam_model_topology", value=EDGETAM_MODEL_TOPOLOGY_SHARED)
            _set_if_not_explicit(args, explicit, flag="--track-mode", attr="track_mode", value=TRACK_MODE_CONTROLLER_OBJECT)
            _set_if_not_explicit(args, explicit, flag="--init-mode", attr="init_mode", value="sam31-first-frame")
            _set_if_not_explicit(args, explicit, flag="--sam31-cache-init-model", attr="sam31_cache_init_model", value=True)
            _set_if_not_explicit(args, explicit, flag="--sam31-keep-runtime-until-all-cameras-init", attr="sam31_keep_runtime_until_all_cameras_init", value=True)
            _set_if_not_explicit(args, explicit, flag="--object-prompt", attr="object_prompt", value="stuffed animal")
            _set_if_not_explicit(args, explicit, flag="--experiment-mode", attr="experiment_mode", value=DEFAULT_DEMO22_EXPERIMENT_MODE)
            _set_if_not_explicit(
                args,
                explicit,
                flag="--controller-prompt",
                attr="controller_prompt",
                value=controller_prompt_for_experiment_mode(resolved_experiment_mode(args)),
            )
            _set_if_not_explicit(args, explicit, flag="--pcd-max-points-per-camera", attr="pcd_max_points_per_camera", value=8000)
            _set_if_not_explicit(args, explicit, flag="--pcd-color-mode", attr="pcd_color_mode", value="class")
            _set_if_not_explicit(args, explicit, flag="--render-every-n", attr="render_every_n", value=2)
            _set_if_not_explicit(args, explicit, flag="--enable-pcd-filter", attr="enable_pcd_filter", value=False)
            _set_if_not_explicit(args, explicit, flag="--pcd-filter-mode", attr="pcd_filter_mode", value="none")
            _set_if_not_explicit(args, explicit, flag="--parallel-init", attr="parallel_init", value=True)
            _set_if_not_explicit(args, explicit, flag="--gpu-gate-mode", attr="gpu_gate_mode", value=GPU_GATE_MODE_OFF)
        elif preset == PRESET_DEMO215_LIVE_QUALITY_FFS:
            _set_if_not_explicit(args, explicit, flag="--depth-source", attr="depth_source", value=DEPTH_SOURCE_FFS)
            _set_if_not_explicit(args, explicit, flag="--fps", attr="fps", value=DEFAULT_PRESET_CAPTURE_FPS)
            _set_if_not_explicit(args, explicit, flag="--fusion-target-fps", attr="fusion_target_fps", value=25.0)
            _set_if_not_explicit(args, explicit, flag="--render-mode", attr="render_mode", value="pointcloud")
            _set_if_not_explicit(args, explicit, flag="--ffs-trt-batch-size", attr="ffs_trt_batch_size", value=3)
            _set_if_not_explicit(args, explicit, flag="--gpu-pipeline-mode", attr="gpu_pipeline_mode", value=GPU_PIPELINE_MODE_SINGLE_OWNER)
            _set_if_not_explicit(args, explicit, flag="--single-owner-order", attr="single_owner_order", value=SINGLE_OWNER_ORDER_FFS_THEN_EDGETAM)
            _set_if_not_explicit(args, explicit, flag="--edgetam-model-topology", attr="edgetam_model_topology", value=EDGETAM_MODEL_TOPOLOGY_SHARED)
            _set_if_not_explicit(args, explicit, flag="--track-mode", attr="track_mode", value=TRACK_MODE_CONTROLLER_OBJECT)
            _set_if_not_explicit(args, explicit, flag="--init-mode", attr="init_mode", value="sam31-first-frame")
            _set_if_not_explicit(args, explicit, flag="--sam31-cache-init-model", attr="sam31_cache_init_model", value=True)
            _set_if_not_explicit(args, explicit, flag="--sam31-keep-runtime-until-all-cameras-init", attr="sam31_keep_runtime_until_all_cameras_init", value=True)
            _set_if_not_explicit(args, explicit, flag="--object-prompt", attr="object_prompt", value="stuffed animal")
            _set_if_not_explicit(args, explicit, flag="--experiment-mode", attr="experiment_mode", value=DEFAULT_DEMO22_EXPERIMENT_MODE)
            _set_if_not_explicit(
                args,
                explicit,
                flag="--controller-prompt",
                attr="controller_prompt",
                value=controller_prompt_for_experiment_mode(resolved_experiment_mode(args)),
            )
            _set_if_not_explicit(args, explicit, flag="--pcd-max-points-per-camera", attr="pcd_max_points_per_camera", value=10000)
            _set_if_not_explicit(args, explicit, flag="--pcd-color-mode", attr="pcd_color_mode", value="rgb")
            _set_if_not_explicit(args, explicit, flag="--render-every-n", attr="render_every_n", value=2)
            _set_if_not_explicit(args, explicit, flag="--enable-pcd-filter", attr="enable_pcd_filter", value=True)
            _set_if_not_explicit(args, explicit, flag="--pcd-filter-mode", attr="pcd_filter_mode", value="async")
            _set_if_not_explicit(args, explicit, flag="--parallel-init", attr="parallel_init", value=True)
            _set_if_not_explicit(args, explicit, flag="--gpu-gate-mode", attr="gpu_gate_mode", value=GPU_GATE_MODE_OFF)
        elif preset == PRESET_DEMO215_MASK_ONLY_DEBUG:
            _set_if_not_explicit(args, explicit, flag="--depth-source", attr="depth_source", value=DEPTH_SOURCE_NONE)
            _set_if_not_explicit(args, explicit, flag="--fps", attr="fps", value=DEFAULT_PRESET_CAPTURE_FPS)
            _set_if_not_explicit(args, explicit, flag="--fusion-target-fps", attr="fusion_target_fps", value=60.0)
            _set_if_not_explicit(args, explicit, flag="--render-mode", attr="render_mode", value="none")
            _set_if_not_explicit(args, explicit, flag="--gpu-pipeline-mode", attr="gpu_pipeline_mode", value=GPU_PIPELINE_MODE_SEPARATE_WORKERS)
            _set_if_not_explicit(args, explicit, flag="--edgetam-model-topology", attr="edgetam_model_topology", value=EDGETAM_MODEL_TOPOLOGY_REPLICATED)
            _set_if_not_explicit(args, explicit, flag="--track-mode", attr="track_mode", value=TRACK_MODE_CONTROLLER_OBJECT)
            _set_if_not_explicit(args, explicit, flag="--init-mode", attr="init_mode", value="sam31-first-frame")
            _set_if_not_explicit(args, explicit, flag="--parallel-init", attr="parallel_init", value=True)
            _set_if_not_explicit(args, explicit, flag="--gpu-gate-mode", attr="gpu_gate_mode", value=GPU_GATE_MODE_OFF)
        elif preset in {PRESET_DEMO22_STAGED_PARALLEL_5FPS, PRESET_DEMO215_STAGED_PARALLEL_5FPS}:
            if preset == PRESET_DEMO215_STAGED_PARALLEL_5FPS:
                _set_if_not_explicit(args, explicit, flag="--depth-source", attr="depth_source", value=DEPTH_SOURCE_REALSENSE)
            _set_if_not_explicit(args, explicit, flag="--fps", attr="fps", value=DEFAULT_PRESET_CAPTURE_FPS)
            _set_if_not_explicit(args, explicit, flag="--fusion-target-fps", attr="fusion_target_fps", value=15.0)
            _set_if_not_explicit(args, explicit, flag="--render-mode", attr="render_mode", value="pointcloud")
            _set_if_not_explicit(args, explicit, flag="--gpu-pipeline-mode", attr="gpu_pipeline_mode", value=GPU_PIPELINE_MODE_STAGED)
            _set_if_not_explicit(args, explicit, flag="--staged-order", attr="staged_order", value=STAGED_ORDER_FFS_THEN_PARALLEL_EDGETAM)
            _set_if_not_explicit(args, explicit, flag="--edgetam-stream-mode", attr="edgetam_stream_mode", value=EDGETAM_STREAM_MODE_PER_CAMERA)
            _set_if_not_explicit(args, explicit, flag="--edgetam-model-topology", attr="edgetam_model_topology", value=EDGETAM_MODEL_TOPOLOGY_REPLICATED)
            _set_if_not_explicit(args, explicit, flag="--compile-mode", attr="compile_mode", value=COMPILE_MODE_VISION_DEFAULT)
            _set_if_not_explicit(args, explicit, flag="--edgetam-prewarm-compile", attr="edgetam_prewarm_compile", value=True)
            _set_if_not_explicit(args, explicit, flag="--edgetam-prewarm-runs", attr="edgetam_prewarm_runs", value=1)
            _set_if_not_explicit(args, explicit, flag="--parallel-init", attr="parallel_init", value=True)
            _set_if_not_explicit(args, explicit, flag="--track-mode", attr="track_mode", value=TRACK_MODE_CONTROLLER_OBJECT)
            _set_if_not_explicit(args, explicit, flag="--init-mode", attr="init_mode", value="sam31-first-frame")
            _set_if_not_explicit(args, explicit, flag="--object-prompt", attr="object_prompt", value="stuffed animal")
            _set_if_not_explicit(args, explicit, flag="--experiment-mode", attr="experiment_mode", value=DEFAULT_DEMO22_EXPERIMENT_MODE)
            _set_if_not_explicit(
                args,
                explicit,
                flag="--controller-prompt",
                attr="controller_prompt",
                value=controller_prompt_for_experiment_mode(resolved_experiment_mode(args)),
            )
            _set_if_not_explicit(args, explicit, flag="--depth-min-m", attr="depth_min_m", value=DEFAULT_DEMO22_DEPTH_MIN_M)
            _set_if_not_explicit(args, explicit, flag="--enable-pcd-filter", attr="enable_pcd_filter", value=True)
            _set_if_not_explicit(args, explicit, flag="--pcd-filter-mode", attr="pcd_filter_mode", value="async")
            _set_if_not_explicit(args, explicit, flag="--gpu-gate-mode", attr="gpu_gate_mode", value=GPU_GATE_MODE_OFF)
            _set_if_not_explicit(args, explicit, flag="--gpu-gate-max-concurrent", attr="gpu_gate_max_concurrent", value=0)
            _set_if_not_explicit(args, explicit, flag="--pin-memory", attr="pin_memory", value=True)
            _set_if_not_explicit(args, explicit, flag="--pin-memory-mode", attr="pin_memory_mode", value=PIN_MEMORY_MODE_ALL)
            _set_if_not_explicit(args, explicit, flag="--h2d-stream-mode", attr="h2d_stream_mode", value=H2D_STREAM_MODE_DEDICATED)
            _set_if_not_explicit(args, explicit, flag="--static-device-buffers", attr="static_device_buffers", value=True)
        elif preset == PRESET_CLIMB_5:
            _set_if_not_explicit(args, explicit, flag="--fusion-target-fps", attr="fusion_target_fps", value=5.0)
            _set_if_not_explicit(args, explicit, flag="--render-mode", attr="render_mode", value="none")
        elif preset == PRESET_CLIMB_10:
            _set_if_not_explicit(args, explicit, flag="--fusion-target-fps", attr="fusion_target_fps", value=10.0)
            _set_if_not_explicit(args, explicit, flag="--render-mode", attr="render_mode", value="none")
        elif preset == PRESET_DIAGNOSTICS:
            _set_if_not_explicit(args, explicit, flag="--fusion-target-fps", attr="fusion_target_fps", value=2.0)
            _set_if_not_explicit(args, explicit, flag="--render-mode", attr="render_mode", value="none")
        if (
            preset in {
                PRESET_DEMO215_ASYNC_FILTER_5FPS,
                PRESET_DEMO215_COMPILED_PARALLEL_EDGETAM_5FPS,
                PRESET_DEMO215_STAGED_PARALLEL_5FPS,
                PRESET_DEMO215_LIVE_FAST_NATIVE,
                PRESET_DEMO215_LIVE_QUALITY_FFS,
                PRESET_DEMO215_MASK_ONLY_DEBUG,
                PRESET_DEMO22_ASYNC_FILTER_5FPS,
                PRESET_DEMO22_STAGED_PARALLEL_5FPS,
            }
            and "--capture-group-target-fps" not in explicit
        ):
            setattr(args, "capture_group_target_fps", float(args.fps))
    if int(getattr(args, "ffs_trt_batch_size", 1)) == 3 and "--ffs-trt-model-dir" not in explicit:
        setattr(args, "ffs_trt_model_dir", str(DEFAULT_FFS_TRT_BATCH3_TWO_STAGE_MODEL_DIR))
    _normalize_pin_memory_options(args, explicit)
    if getattr(args, "gpu_gate_mode", None) == GPU_GATE_MODE_OFF:
        setattr(args, "gpu_gate_max_concurrent", 0)
    return args


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
        scalar = float(value)
    except Exception:
        return int(time.time_ns())
    if scalar > 1.0e15:
        return int(scalar)
    if scalar > 1.0e12:
        return int(scalar * 1_000_000.0)
    if scalar > 1.0e9:
        return int(scalar * 1_000_000_000.0)
    if scalar > 1.0e6:
        return int(scalar * 1_000_000.0)
    return int(scalar * 1_000_000_000.0)


def _packet_realsense_timestamp_ns(frame: CameraFramePacket) -> int | None:
    if frame.realsense_timestamp_ms is None:
        return None
    return int(float(frame.realsense_timestamp_ms) * 1_000_000.0)


def select_capture_timestamp_source(frames: dict[int, CameraFramePacket]) -> str:
    domains = [frame.timestamp_domain for frame in frames.values()]
    has_realsense = all(frame.realsense_timestamp_ms is not None for frame in frames.values())
    if has_realsense and all(domain is not None for domain in domains) and len(set(domains)) == 1:
        return "realsense_timestamp"
    return "host_receive_timestamp"


def _packet_timestamp_for_source(frame: CameraFramePacket, timestamp_source: str) -> int:
    if timestamp_source == "realsense_timestamp":
        timestamp_ns = _packet_realsense_timestamp_ns(frame)
        if timestamp_ns is not None:
            return timestamp_ns
    return int(frame.timestamp_ns)


def _temporal_selection_from_frames(
    frames: dict[int, CameraFramePacket],
    *,
    now_perf_ns: int,
) -> TemporalGroupSelection:
    timestamp_source = select_capture_timestamp_source(frames)
    timestamps = {
        int(camera_idx): _packet_timestamp_for_source(frame, timestamp_source)
        for camera_idx, frame in frames.items()
    }
    values = list(timestamps.values())
    group_timestamp_ns = int(np.median(np.asarray(values, dtype=np.float64)))
    offsets = {
        int(camera_idx): float((timestamp_ns - group_timestamp_ns) / 1_000_000.0)
        for camera_idx, timestamp_ns in timestamps.items()
    }
    max_temporal_skew_ms = float((max(values) - min(values)) / 1_000_000.0)
    latest_arrival_ns = max(int(frame.capture_arrival_perf_ns) for frame in frames.values())
    age_ms = float(max(0, now_perf_ns - latest_arrival_ns) / 1_000_000.0)
    return TemporalGroupSelection(
        frames=dict(frames),
        timestamp_source=timestamp_source,
        group_timestamp_ns=group_timestamp_ns,
        max_temporal_skew_ms=max_temporal_skew_ms,
        per_camera_time_offset_ms=offsets,
        per_camera_frame_seq={int(camera_idx): int(frame.frame_seq) for camera_idx, frame in frames.items()},
        age_ms=age_ms,
    )


def select_temporal_capture_triplet(
    buffers: dict[int, deque[CameraFramePacket]],
    *,
    camera_ids: Sequence[int],
    policy: str,
    max_frame_age_ms: float,
    now_perf_ns: int,
) -> TemporalGroupSelection | None:
    if policy not in CAPTURE_GROUP_POLICIES:
        raise ValueError(f"Unsupported capture group policy: {policy}")
    camera_ids = tuple(int(camera_idx) for camera_idx in camera_ids)
    if any(not buffers.get(camera_idx) for camera_idx in camera_ids):
        return None
    max_frame_age_ms = float(max_frame_age_ms)
    if policy == CAPTURE_GROUP_POLICY_LATEST:
        frames = {camera_idx: buffers[camera_idx][-1] for camera_idx in camera_ids}
        selection = _temporal_selection_from_frames(frames, now_perf_ns=now_perf_ns)
        return selection if selection.age_ms <= max_frame_age_ms else None

    best: TemporalGroupSelection | None = None
    best_score = float("inf")
    for combo in product(*(tuple(buffers[camera_idx]) for camera_idx in camera_ids)):
        frames = {camera_idx: frame for camera_idx, frame in zip(camera_ids, combo)}
        selection = _temporal_selection_from_frames(frames, now_perf_ns=now_perf_ns)
        if selection.age_ms > max_frame_age_ms:
            continue
        score = float(selection.max_temporal_skew_ms + 0.1 * selection.age_ms)
        if score < best_score:
            best = selection
            best_score = score
    return best


def build_temporal_capture_group(
    *,
    group_id: int,
    created_perf_s: float,
    selection: TemporalGroupSelection,
) -> CaptureGroup:
    frames = {
        int(camera_idx): replace(frame, group_id=int(group_id))
        for camera_idx, frame in selection.frames.items()
    }
    return CaptureGroup(
        group_id=int(group_id),
        created_perf_s=float(created_perf_s),
        frames=frames,
        group_timestamp_ns=int(selection.group_timestamp_ns),
        max_temporal_skew_ms=float(selection.max_temporal_skew_ms),
        per_camera_time_offset_ms=dict(selection.per_camera_time_offset_ms),
        per_camera_frame_seq=dict(selection.per_camera_frame_seq),
        timestamp_source=str(selection.timestamp_source),
    )


def drop_selected_and_older_frames(
    buffers: dict[int, deque[CameraFramePacket]],
    selection: TemporalGroupSelection,
) -> None:
    for camera_idx, selected in selection.frames.items():
        buffer = buffers.get(int(camera_idx))
        if buffer is None:
            continue
        selected_key = (int(selected.frame_seq), int(selected.timestamp_ns), int(selected.capture_arrival_perf_ns))
        while buffer:
            candidate = buffer[0]
            candidate_key = (int(candidate.frame_seq), int(candidate.timestamp_ns), int(candidate.capture_arrival_perf_ns))
            buffer.popleft()
            if candidate_key == selected_key:
                break


def drop_oldest_capture_buffer_frame(buffers: dict[int, deque[CameraFramePacket]]) -> bool:
    oldest_camera: int | None = None
    oldest_arrival_ns: int | None = None
    for camera_idx, buffer in buffers.items():
        if not buffer:
            continue
        arrival_ns = int(buffer[0].capture_arrival_perf_ns)
        if oldest_arrival_ns is None or arrival_ns < oldest_arrival_ns:
            oldest_arrival_ns = arrival_ns
            oldest_camera = int(camera_idx)
    if oldest_camera is None:
        return False
    buffers[oldest_camera].popleft()
    return True


def prune_stale_capture_buffers(
    buffers: dict[int, deque[CameraFramePacket]],
    *,
    max_frame_age_ms: float,
    now_perf_ns: int,
) -> int:
    pruned = 0
    max_age_ns = int(float(max_frame_age_ms) * 1_000_000.0)
    for buffer in buffers.values():
        while buffer and now_perf_ns - int(buffer[0].capture_arrival_perf_ns) > max_age_ns:
            buffer.popleft()
            pruned += 1
    return pruned


def temporal_group_is_coherent(group: CaptureGroup | DepthGroup, *, max_capture_skew_ms: float) -> bool:
    return float(group.max_temporal_skew_ms) <= float(max_capture_skew_ms)


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
        object_mask = (
            _load_saved_mask_from_root(
                args.object_init_mask_root,
                camera_idx=frame.camera_idx,
                expected_shape=expected_shape,
            )
            if object_tracking_enabled(args.track_mode)
            else None
        )
        controller_mask = (
            _load_saved_mask_from_root(
                args.controller_init_mask_root,
                camera_idx=frame.camera_idx,
                expected_shape=expected_shape,
            )
            if controller_tracking_enabled(args.track_mode)
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
        # SAM3.1 initialization is intentionally serialized across cameras to
        # avoid three first-frame segmentation jobs fighting for the same GPU.
        with sam31_lock:
            return resolve_initial_masks(frame, args)
    raise ValueError(f"unsupported init mode: {args.init_mode}")


def split_hf_vision_features_for_session(image_outputs: Any, batch_idx: int) -> dict[str, Any]:
    """Return one-camera HF EdgeTAM vision feature cache from a batched encoder output."""
    idx = int(batch_idx)
    return {
        "vision_feats": [
            feature[:, idx : idx + 1, :].contiguous()
            for feature in image_outputs.fpn_hidden_states
        ],
        "vision_pos_embeds": [
            pos_embed[:, idx : idx + 1, :].contiguous()
            for pos_embed in image_outputs.fpn_position_encoding
        ],
    }


def slice_hf_original_sizes(original_sizes: Any, batch_idx: int) -> Any:
    idx = int(batch_idx)
    if hasattr(original_sizes, "dim"):
        if int(original_sizes.dim()) == 1:
            return original_sizes
        return original_sizes[idx : idx + 1]
    if isinstance(original_sizes, np.ndarray):
        if original_sizes.ndim == 1:
            return original_sizes
        return original_sizes[idx : idx + 1]
    if isinstance(original_sizes, (list, tuple)):
        if not original_sizes:
            return original_sizes
        return type(original_sizes)([original_sizes[idx]])
    return original_sizes


def build_contract(args: argparse.Namespace) -> dict[str, Any]:
    layers = semantic_layers_for_track_mode(
        args.track_mode,
        object_label=args.object_prompt,
        controller_label=args.controller_prompt,
        object_postprocess=args.object_postprocess,
        controller_postprocess=args.controller_postprocess,
    )
    preset_canonical = getattr(args, "preset_canonical", canonical_preset_name(getattr(args, "preset", PRESET_NONE)))
    is_demo22_preset = preset_canonical in {
        PRESET_DEMO22_ASYNC_FILTER_5FPS,
        PRESET_DEMO22_STAGED_PARALLEL_5FPS,
    }
    is_demo215_preset = preset_canonical in {
        PRESET_DEMO215_ASYNC_FILTER_5FPS,
        PRESET_DEMO215_COMPILED_PARALLEL_EDGETAM_5FPS,
        PRESET_DEMO215_STAGED_PARALLEL_5FPS,
        PRESET_DEMO215_LIVE_FAST_NATIVE,
        PRESET_DEMO215_LIVE_QUALITY_FFS,
        PRESET_DEMO215_MASK_ONLY_DEBUG,
    }
    experiment_mode = resolved_experiment_mode(args)
    expected_controller_prompt = controller_prompt_for_experiment_mode(experiment_mode)
    stage_scheduler_mode = str(getattr(args, "stage_scheduler_mode", STAGE_SCHEDULER_MODE_MASK_GATED))
    return {
        "demo": (
            "demo_2_2_async_filtered_fused_pcd"
            if is_demo22_preset
            else "demo_2_1_5_realsense_async_filtered_fused_pcd"
            if is_demo215_preset
            else "demo_2_1_three_view_fused_masked_pcd"
        ),
        "preset": getattr(args, "preset", PRESET_NONE),
        "preset_canonical": preset_canonical,
        "camera_ids": list(args.camera_ids),
        "profile": args.profile,
        "fps": int(args.fps),
        "track_mode": args.track_mode,
        "experiment_mode": experiment_mode,
        "controller_semantic": expected_controller_prompt,
        "controller_prompt": str(getattr(args, "controller_prompt", "")),
        "controller_prompt_expected": expected_controller_prompt,
        "controller_prompt_matches_experiment_mode": controller_prompt_matches_experiment_mode(args),
        "frame_by_frame_streaming": True,
        "offline_video_input_used": False,
        "edge_backend": "HF EdgeTAMVideo",
        "compile_mode": args.compile_mode,
        "dtype": args.dtype,
        "input_path": args.edgetam_input_path,
        "mask_postprocess": args.mask_postprocess,
        "depth_source": args.depth_source,
        "render_mode": args.render_mode,
        "renderer": {
            "backend": str(getattr(args, "render_backend", DEFAULT_RENDER_BACKEND)),
            "layer_mode": str(getattr(args, "render_layer_mode", DEFAULT_RENDER_LAYER_MODE)),
            "async_latest_only": bool(getattr(args, "render_async_latest_only", True)),
            "copy_mode": str(getattr(args, "render_copy_mode", DEFAULT_RENDER_COPY_MODE)),
            "micro_profile": bool(getattr(args, "render_micro_profile", False)),
            "display_lod": "off",
            "quality_loss_default": False,
        },
        "fusion_target_fps": float(args.fusion_target_fps),
        "capture_group_target_fps": resolved_capture_group_target_fps(args),
        "fusion_timeout_ms": float(args.fusion_timeout_ms),
        "temporal_grouping": {
            "policy": args.capture_group_policy,
            "max_capture_skew_ms": float(args.max_capture_skew_ms),
            "max_frame_age_ms": float(args.max_frame_age_ms),
            "capture_buffer_size": int(args.capture_buffer_size),
            "drop_skewed_groups": bool(args.drop_skewed_groups),
            "no_temporal_coherent_group_no_ffs": True,
        },
        "init": {
            "mode": args.init_mode,
            "parallel_init": bool(getattr(args, "parallel_init", False)),
            "object_init_mask_root": args.object_init_mask_root,
            "controller_init_mask_root": args.controller_init_mask_root,
            "sam31_retry_interval_s": float(args.sam31_init_retry_interval_s),
            "sam31_max_attempts": int(args.sam31_init_max_attempts),
            "sam31_cache_init_model": bool(getattr(args, "sam31_cache_init_model", False)),
            "sam31_keep_runtime_until_all_cameras_init": bool(
                getattr(args, "sam31_keep_runtime_until_all_cameras_init", False)
            ),
            "sam31_torchvision_ops_preimport": args.init_mode == "sam31-first-frame",
            "formal_demo_requires_live_sam31": True,
            "fallback_allowed": False,
        },
        "official_quality_depth": args.depth_source in set(OFFICIAL_DEPTH_SOURCES),
        "native_realsense_depth_role": (
            "primary" if args.depth_source == DEPTH_SOURCE_REALSENSE else "fallback/debug only"
        ),
        "gpu_gate": {
            "mode": args.gpu_gate_mode,
            "max_concurrent": int(args.gpu_gate_max_concurrent),
        },
        "gpu_sampling": {
            "enabled": bool(getattr(args, "gpu_sampling", False)),
            "interval_s": float(getattr(args, "gpu_sampling_interval_s", 0.5)),
            "backend": str(getattr(args, "gpu_sampling_backend", "nvml")),
            "device_index": int(getattr(args, "gpu_sampling_device_index", 0)),
        },
        "profiling": {
            "profile_cuda_events": bool(args.profile_cuda_events),
            "profile_sync": bool(args.profile_sync),
            "profile_edgetam_stages": bool(args.profile_edgetam_stages),
            "profile_nsys_markers": bool(args.profile_nsys_markers),
            "full_device_sync_only_when_profile_sync": True,
        },
        "gpu_pipeline": {
            "mode": args.gpu_pipeline_mode,
            "internal_order": (
                args.staged_order
                if args.gpu_pipeline_mode == GPU_PIPELINE_MODE_STAGED
                else (
                    "cross_group_overlap"
                    if args.gpu_pipeline_mode == GPU_PIPELINE_MODE_OVERLAPPED_STAGES
                    else args.single_owner_order
                )
            ),
            "staged_order": args.staged_order,
            "ffs_stage": (
                "sequential_cam0_cam1_cam2"
                if args.gpu_pipeline_mode in {GPU_PIPELINE_MODE_STAGED, GPU_PIPELINE_MODE_OVERLAPPED_STAGES}
                else None
            ),
            "edgetam_stage": (
                (
                    "batch_vision_stateful_decode"
                    if args.gpu_pipeline_mode == GPU_PIPELINE_MODE_OVERLAPPED_STAGES
                    else "parallel_cam0_cam1_cam2"
                )
                if args.gpu_pipeline_mode in {GPU_PIPELINE_MODE_STAGED, GPU_PIPELINE_MODE_OVERLAPPED_STAGES}
                else None
            ),
            "depth_and_masks_published_together": args.gpu_pipeline_mode in {
                GPU_PIPELINE_MODE_SINGLE_OWNER,
                GPU_PIPELINE_MODE_STAGED,
            },
            "same_group_join_required": args.gpu_pipeline_mode == GPU_PIPELINE_MODE_OVERLAPPED_STAGES,
            "overlap_across_groups": args.gpu_pipeline_mode == GPU_PIPELINE_MODE_OVERLAPPED_STAGES,
            "stage_scheduler_mode": stage_scheduler_mode,
            "stage_lookahead": int(getattr(args, "stage_lookahead", 1)),
            "depth_dispatch_policy": (
                "after_mask_stage"
                if args.gpu_pipeline_mode == GPU_PIPELINE_MODE_OVERLAPPED_STAGES
                and stage_scheduler_mode == STAGE_SCHEDULER_MODE_MASK_GATED
                else "edge_start_reservation"
                if args.gpu_pipeline_mode == GPU_PIPELINE_MODE_OVERLAPPED_STAGES
                and stage_scheduler_mode == STAGE_SCHEDULER_MODE_EDGE_START
                else "bounded_lookahead_reservation"
                if args.gpu_pipeline_mode == GPU_PIPELINE_MODE_OVERLAPPED_STAGES
                and stage_scheduler_mode == STAGE_SCHEDULER_MODE_BOUNDED_LOOKAHEAD
                else "capture_dispatch"
            ),
            "separate_ffs_and_edgetam_workers": args.gpu_pipeline_mode == GPU_PIPELINE_MODE_SEPARATE_WORKERS,
        },
        "memory_for_speed": {
            "static_device_buffers": bool(args.static_device_buffers),
            "preallocate_pcd_buffers": bool(args.preallocate_pcd_buffers),
            "ffs_reusable_cuda_input_buffers": args.depth_source == DEPTH_SOURCE_FFS and args.ffs_input_staging == FFS_INPUT_STAGING_PINNED,
            "edgetam_reusable_cuda_pixel_slots": edge_pin_memory_enabled(args),
            "models_loaded_once_per_worker": args.gpu_pipeline_mode in {
                GPU_PIPELINE_MODE_SEPARATE_WORKERS,
                GPU_PIPELINE_MODE_SINGLE_OWNER,
                GPU_PIPELINE_MODE_STAGED,
                GPU_PIPELINE_MODE_OVERLAPPED_STAGES,
            },
        },
        "h2d_transfer": {
            "pin_memory": bool(args.pin_memory),
            "pin_memory_mode": args.pin_memory_mode,
            "edge_pin_enabled": edge_pin_memory_enabled(args),
            "ffs_pin_requested": ffs_pin_memory_requested(args),
            "ffs_input_staging": args.ffs_input_staging,
            "pinned_ring_size": int(args.pinned_ring_size),
            "h2d_stream_mode": args.h2d_stream_mode,
            "profile_h2d": bool(args.profile_h2d),
        },
        "tracking_overlay": {
            "enabled": bool(getattr(args, "show_tracking_overlay", False)),
            "backend": str(getattr(args, "tracking_backend", "none")),
            "source": str(getattr(args, "tracking_source", "cached")),
            "num_points": int(getattr(args, "tracking_num_points", 256)),
            "max_points": int(getattr(args, "tracking_overlay_max_points", 30)),
            "trail_len": int(getattr(args, "tracking_trail_len", 8)),
            "update_hz": float(getattr(args, "tracking_update_hz", 5.0)),
            "depth_source": str(getattr(args, "tracking_depth_source", "displayed")),
            "output_root": str(getattr(args, "tracking_output_root", "./data/experiments/demo3_live_tracking")),
            "hot_path_enabled_by_default": False,
            "blocking_render": False,
        },
        "ffs_contract": {
            "checkpoint": DEFAULT_FFS_MODEL_NAME,
            "valid_iters": DEFAULT_FFS_VALID_ITERS,
            "capture_resolution": "848x480",
            "engine_input": f"{DEFAULT_FFS_TRT_ENGINE_SIZE[1]}x{DEFAULT_FFS_TRT_ENGINE_SIZE[0]}",
            "padding_policy": "pad_width_848_to_864",
            "builderOptimizationLevel": DEFAULT_FFS_TRT_BUILDER_OPTIMIZATION_LEVEL,
            "max_disp": DEFAULT_FFS_MAX_DISP,
            "trt_batch_size": int(args.ffs_trt_batch_size),
            "trt_model_dir": str(args.ffs_trt_model_dir),
            "worker_mode": args.ffs_worker_mode,
            "schedule": args.ffs_schedule,
            "input_staging": args.ffs_input_staging,
            "batch3_isolated_artifact": int(args.ffs_trt_batch_size) == 3,
        },
        "edgetam": {
            "worker_mode": args.edgetam_worker_mode,
            "model_topology": args.edgetam_model_topology,
            "stream_mode": args.edgetam_stream_mode,
            "one_streaming_session_per_camera": True,
            "input_path": args.edgetam_input_path,
            "mask_postprocess": args.mask_postprocess,
            "multi_object_single_session_per_camera": len(active_object_ids(args)) > 1,
            "active_object_ids": active_object_ids(args),
            "serialize_first_compiled_forward": serialized_edgetam_first_compiled_forward_enabled(args),
            "prewarm_compile": bool(getattr(args, "edgetam_prewarm_compile", False)),
            "prewarm_runs": int(getattr(args, "edgetam_prewarm_runs", 0)),
            "batch_vision_encoder": bool(getattr(args, "edgetam_batch_vision_encoder", False)),
            "batch_vision_batch_size": (
                len(tuple(args.camera_ids)) if bool(getattr(args, "edgetam_batch_vision_encoder", False)) else 1
            ),
        },
        "fusion": {
            "mode": "semantic_layers",
            "labels_are_filtered_separately": True,
            "do_not_filter_object_controller_union": True,
            "object_controller_union_before_filter": False,
            "raw_fusion_before_filter": async_fusion_filter_enabled(args),
        },
        "filter_scheduler": {
            "enabled": bool(args.enable_pcd_filter) and args.pcd_filter_mode != "none",
            "mode": args.pcd_filter_mode,
            "hot_path": "raw_fused_semantic_pcd" if async_fusion_filter_enabled(args) else "sync_filtered_fused_pcd",
            "filtered_path": "latest_wins_async_filtered_packet" if async_fusion_filter_enabled(args) else "inline_filter",
            "render_blocks_on_filter": args.pcd_filter_mode == "sync",
            "render_accepts_raw_fused_pcd": not async_fusion_filter_enabled(args),
            "render_filtered_only": async_fusion_filter_enabled(args),
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
        self.ffs_stage_input_slot: LatestSlot[CaptureGroup] = LatestSlot()
        self.edgetam_stage_input_slot: LatestSlot[CaptureGroup] = LatestSlot()
        self.depth_group_slot: LatestSlot[DepthGroup] = LatestSlot()
        self.complete_inference_slot: LatestSlot[CompleteInferenceGroup] = LatestSlot()
        self.mask_slots: dict[int, LatestSlot[CameraMaskPacket]] = {
            int(camera_idx): LatestSlot() for camera_idx in args.camera_ids
        }
        self.stage_join_buffer = SameGroupJoinBuffer(max_groups=8)
        self.stage_window_scheduler = StageWindowScheduler(
            max_groups=8,
            lookahead=int(getattr(args, "stage_lookahead", 1)),
        )
        self.raw_fused_slot: LatestSlot[RawFusedPcdPacket] = LatestSlot()
        self.render_buffer: LatestOnlyRenderBuffer[FusedPcdPacket] = LatestOnlyRenderBuffer()
        self.render_post_gate = CoalescedRenderPostGate()
        self.gpu_gate = GpuInferenceGate(
            mode=str(args.gpu_gate_mode),
            max_concurrent=int(args.gpu_gate_max_concurrent),
        )
        self.capture_group_stats = StageStats()
        self.ffs_stats = StageStats()
        self.gpu_owner_stats = StageStats()
        self.edge_stats = {int(camera_idx): StageStats() for camera_idx in args.camera_ids}
        self.gpu_gate_wait_stats: dict[str, MsWindowStats] = {"ffs": MsWindowStats()}
        for camera_idx in args.camera_ids:
            self.gpu_gate_wait_stats[f"edgetam_cam{int(camera_idx)}"] = MsWindowStats()
        self.fusion_stats = StageStats()
        self.raw_fusion_stats = StageStats()
        self.filter_output_stats = StageStats()
        self.render_stats = RenderStats()
        self.temporal_skew_stats = MsWindowStats()
        self._temporal_skews_lock = threading.Lock()
        self._temporal_skews_ms: list[float] = []
        self._threads: list[threading.Thread] = []
        self._sam31_lock = threading.Lock()
        self._sam31_runtime_released_after_init = False
        self._parallel_init_executor: ThreadPoolExecutor | None = None
        self._parallel_init_futures: dict[str, Any] = {}
        self._parallel_init_started_s: float | None = None
        self._init_profile_lock = threading.Lock()
        self._edgetam_first_compiled_forward_lock = threading.Lock()
        self._edgetam_first_compiled_forward_done: set[int] = set()
        self._init_profile: dict[str, Any] = {
            "process_start_s": 0.0,
            "first_complete_inference_group_s": None,
            "first_complete_fused_group_s": None,
            "first_render_s": None,
        }
        self._first_ffs_cycle_recorded = False
        self._summary: dict[str, Any] = {"contract": build_contract(args), "events": []}
        self._profile_enabled = bool(getattr(args, "profile_json_output", None)) or any(
            bool(getattr(args, attr, False)) for attr in PROFILE_FLAG_ATTRS
        )
        self._profile_lock = threading.Lock()
        self._profile_started_perf_s = time.perf_counter()
        self._profile_records: dict[int, dict[str, Any]] = {}
        self._gpu_sampler = GpuUtilizationSampler(
            enabled=bool(getattr(args, "gpu_sampling", False)),
            interval_s=float(getattr(args, "gpu_sampling_interval_s", 0.5)),
            backend=str(getattr(args, "gpu_sampling_backend", "auto")),
            device_index=int(getattr(args, "gpu_sampling_device_index", 0)),
            rel_time_fn=self._profile_rel_s,
        )
        self._latest_depth_group: DepthGroup | None = None
        self._latest_raw_fused: RawFusedPcdPacket | None = None
        self._latest_fused: FusedPcdPacket | None = None
        self._last_debug_s = 0.0
        self._render_request: Callable[[], None] = lambda: None
        self._fatal_error: str | None = None
        self._fatal_error_lock = threading.Lock()

    def _mark_fatal_error(self, stage: str, exc: BaseException) -> None:
        message = f"{stage}: {type(exc).__name__}: {exc}"
        trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        with self._fatal_error_lock:
            if self._fatal_error is None:
                self._fatal_error = message
                self._summary["fatal_error"] = message
                self._summary["fatal_error_traceback"] = trace

    def _record_gpu_gate_wait(self, key: str, wait_ms: float) -> None:
        stats = self.gpu_gate_wait_stats.get(key)
        if stats is None:
            stats = MsWindowStats()
            self.gpu_gate_wait_stats[key] = stats
        stats.record(float(wait_ms))

    def _record_temporal_skew(self, skew_ms: float) -> None:
        value = float(skew_ms)
        self.temporal_skew_stats.record(value)
        with self._temporal_skews_lock:
            self._temporal_skews_ms.append(value)

    def _temporal_grouping_summary(self) -> dict[str, Any]:
        with self._temporal_skews_lock:
            values = list(self._temporal_skews_ms)
        stats = _profile_stats(values)
        return {
            "policy": self.args.capture_group_policy,
            "max_capture_skew_ms": float(self.args.max_capture_skew_ms),
            "max_frame_age_ms": float(self.args.max_frame_age_ms),
            "capture_buffer_size": int(self.args.capture_buffer_size),
            "drop_skewed_groups": bool(self.args.drop_skewed_groups),
            "timestamp_source": self._summary.get("capture_timestamp_source", "unknown"),
            "groups_emitted": int(self._summary.get("capture_groups_emitted", 0)),
            "groups_dropped_skew": int(self._summary.get("capture_group_skew_drop", 0)),
            "groups_dropped_no_candidate": int(self._summary.get("capture_group_no_candidate", 0)),
            "stale_frames_pruned": int(self._summary.get("capture_stale_frames_pruned", 0)),
            "ffs_groups_dropped_skew": int(self._summary.get("ffs_drop_skewed_capture_group", 0)),
            "fusion_groups_dropped_skew": int(self._summary.get("fusion_drop_skewed_group", 0)),
            "skew_ms_median": float(stats["median"]),
            "skew_ms_p95": float(stats["p95"]),
            "skew_ms_max": float(stats["max"]),
        }

    def _profile_rel_s(self, perf_s: float | None = None) -> float:
        return float((time.perf_counter() if perf_s is None else perf_s) - self._profile_started_perf_s)

    def _init_profile_snapshot(self) -> dict[str, Any]:
        with self._init_profile_lock:
            return json.loads(json.dumps(self._init_profile, default=_json_default))

    def _init_profile_set(self, path: Sequence[str], value: Any) -> None:
        with self._init_profile_lock:
            cursor = self._init_profile
            for key in path[:-1]:
                next_value = cursor.get(str(key))
                if not isinstance(next_value, dict):
                    next_value = {}
                    cursor[str(key)] = next_value
                cursor = next_value
            cursor[str(path[-1])] = value

    def _init_profile_set_once(self, path: Sequence[str], value: Any) -> None:
        with self._init_profile_lock:
            cursor = self._init_profile
            for key in path[:-1]:
                next_value = cursor.get(str(key))
                if not isinstance(next_value, dict):
                    next_value = {}
                    cursor[str(key)] = next_value
                cursor = next_value
            final_key = str(path[-1])
            if cursor.get(final_key) is None:
                cursor[final_key] = value

    def _init_profile_update(self, path: Sequence[str], values: dict[str, Any]) -> None:
        with self._init_profile_lock:
            cursor = self._init_profile
            for key in path:
                next_value = cursor.get(str(key))
                if not isinstance(next_value, dict):
                    next_value = {}
                cursor[str(key)] = next_value
                cursor = next_value
            _deep_update_dict(cursor, values)

    def _init_profile_add(self, path: Sequence[str], value: float) -> None:
        with self._init_profile_lock:
            cursor = self._init_profile
            for key in path[:-1]:
                next_value = cursor.get(str(key))
                if not isinstance(next_value, dict):
                    next_value = {}
                    cursor[str(key)] = next_value
                cursor = next_value
            final_key = str(path[-1])
            cursor[final_key] = float(cursor.get(final_key, 0.0) or 0.0) + float(value)

    def _profile_group_record(self, group_id: int) -> dict[str, Any]:
        record = self._profile_records.get(int(group_id))
        if record is None:
            record = {"group_id": int(group_id), "complete": False, "drop_reason": None}
            self._profile_records[int(group_id)] = record
        return record

    def _profile_update(self, group_id: int, **sections: Any) -> None:
        if not self._profile_enabled:
            return
        with self._profile_lock:
            record = self._profile_group_record(int(group_id))
            for key, value in sections.items():
                if isinstance(value, dict) and isinstance(record.get(key), dict):
                    _deep_update_dict(record[key], value)
                else:
                    record[key] = value

    def _profile_mark_drop(self, group_id: int, reason: str) -> None:
        if not self._profile_enabled:
            return
        with self._profile_lock:
            record = self._profile_group_record(int(group_id))
            record["complete"] = False
            record["drop_reason"] = str(reason)

    def run(self) -> int:
        if self.args.depth_source not in {DEPTH_SOURCE_FFS, DEPTH_SOURCE_REALSENSE, DEPTH_SOURCE_NONE}:
            raise RuntimeError(
                "Demo 2.1 live runtime currently supports --depth-source ffs, realsense, "
                "and none for capture/EdgeTAM isolation."
        )
        apply_wslg_open3d_env_defaults()
        self._validate_live_contract()
        self._warm_torchvision_ops_imports_for_sam31()
        self._gpu_sampler.start()
        self._start_parallel_init()
        try:
            camera_start_s = time.perf_counter()
            self._start_camera_system()
            self._init_profile_set(("camera_startup_ms",), _elapsed_ms(camera_start_s, time.perf_counter()))
            if self.args.render_mode == "pointcloud":
                self._run_open3d()
            else:
                self._run_headless()
        finally:
            self.stop()
            self._gpu_sampler.stop()
            self._write_summary()
        return 1 if self._fatal_error is not None else 0

    def warm_init_caches(
        self,
        *,
        repeats: int = 1,
        include_edgetam: bool = True,
        include_sam31: bool = True,
    ) -> dict[str, Any]:
        repeats = max(1, int(repeats))
        results: list[dict[str, Any]] = []
        for repeat_idx in range(repeats):
            repeat_start_s = time.perf_counter()
            entry: dict[str, Any] = {"repeat_idx": int(repeat_idx)}
            if include_edgetam and self.args.track_mode != TRACK_MODE_NONE:
                edge_start_s = time.perf_counter()
                hf_stream, torch_module, _dtype, model, processor = self._init_hf_model(camera_idx=-1)
                entry["edgetam_total_ms"] = float(_elapsed_ms(edge_start_s, time.perf_counter()))
                entry["edgetam_loader_profile"] = self._init_profile_snapshot().get("edgetam", {}).get("loaders", {}).get("shared", {})
                try:
                    del processor
                    del model
                    if str(self.args.device).startswith("cuda") and hasattr(torch_module, "cuda"):
                        torch_module.cuda.empty_cache()
                finally:
                    del hf_stream
            if include_sam31 and self.args.init_mode == "sam31-first-frame":
                sam_start_s = time.perf_counter()
                sam31_result = self._preload_sam31_init_model()
                entry["sam31_total_ms"] = float(_elapsed_ms(sam_start_s, time.perf_counter()))
                entry["sam31_preload_profile"] = dict(sam31_result.get("timing_ms", {}) or {})
            entry["total_ms"] = float(_elapsed_ms(repeat_start_s, time.perf_counter()))
            results.append(entry)
        return {
            "contract": build_contract(self.args),
            "repeats": results,
            "init_profile": self._init_profile_snapshot(),
        }

    def _warm_torchvision_ops_imports_for_sam31(self) -> None:
        if self.args.init_mode != "sam31-first-frame":
            self._init_profile_update(
                ("sam31", "torchvision_ops_preimport"),
                {
                    "enabled": False,
                    "ok": True,
                },
            )
            return
        start_s = time.perf_counter()
        try:
            from torchvision import ops as torchvision_ops  # noqa: F401
            from torchvision.ops import StochasticDepth  # noqa: F401
        except Exception as exc:
            self._init_profile_update(
                ("sam31", "torchvision_ops_preimport"),
                {
                    "enabled": True,
                    "ok": False,
                    "error": repr(exc),
                    "wall_ms": float(_elapsed_ms(start_s, time.perf_counter())),
                },
            )
            raise RuntimeError("SAM3.1 torchvision ops preimport failed before parallel init") from exc
        self._init_profile_update(
            ("sam31", "torchvision_ops_preimport"),
            {
                "enabled": True,
                "ok": True,
                "wall_ms": float(_elapsed_ms(start_s, time.perf_counter())),
            },
        )

    def _start_parallel_init(self) -> None:
        if not bool(getattr(self.args, "parallel_init", False)):
            self._init_profile_set(("parallel_init", "enabled"), False)
            return
        tasks: dict[str, Callable[[], Any]] = {}
        if self.args.depth_source == DEPTH_SOURCE_FFS and self.args.gpu_pipeline_mode in {
            GPU_PIPELINE_MODE_SINGLE_OWNER,
            GPU_PIPELINE_MODE_STAGED,
            GPU_PIPELINE_MODE_OVERLAPPED_STAGES,
        }:
            tasks["ffs_runner"] = self._prepare_ffs_runner
        if self.args.track_mode != TRACK_MODE_NONE and self.args.gpu_pipeline_mode in {
            GPU_PIPELINE_MODE_SINGLE_OWNER,
            GPU_PIPELINE_MODE_STAGED,
            GPU_PIPELINE_MODE_OVERLAPPED_STAGES,
        }:
            tasks["edgetam_states"] = self._init_gpu_owner_edgetam_states
        if (
            self.args.init_mode == "sam31-first-frame"
            and bool(getattr(self.args, "sam31_cache_init_model", False))
        ):
            tasks["sam31_preload"] = self._preload_sam31_init_model
        self._init_profile_update(
            ("parallel_init",),
            {
                "enabled": True,
                "tasks": sorted(tasks),
            },
        )
        if not tasks:
            return
        self._parallel_init_started_s = time.perf_counter()
        self._parallel_init_executor = ThreadPoolExecutor(
            max_workers=len(tasks),
            thread_name_prefix="demo2.2-init",
        )
        self._parallel_init_futures = {
            name: self._parallel_init_executor.submit(self._run_parallel_init_task, name, task)
            for name, task in tasks.items()
        }

    def _run_parallel_init_task(self, name: str, task: Callable[[], Any]) -> Any:
        task_start_s = time.perf_counter()
        self._init_profile_update(
            ("parallel_init", str(name)),
            {
                "started_s": self._profile_rel_s(),
            },
        )
        try:
            return task()
        finally:
            self._init_profile_update(
                ("parallel_init", str(name)),
                {
                    "finished_s": self._profile_rel_s(),
                    "duration_ms": float(_elapsed_ms(task_start_s, time.perf_counter())),
                },
            )

    def _consume_parallel_init_future(self, name: str) -> Any | None:
        future = self._parallel_init_futures.pop(str(name), None)
        if future is None:
            return None
        wait_start_s = time.perf_counter()
        try:
            value = future.result()
        except Exception as exc:
            self._init_profile_update(
                ("parallel_init", name),
                {
                    "failed": True,
                    "error": f"{type(exc).__name__}: {exc}",
                    "wait_ms": float(_elapsed_ms(wait_start_s, time.perf_counter())),
                },
            )
            raise
        wait_ms = _elapsed_ms(wait_start_s, time.perf_counter())
        self._init_profile_update(
            ("parallel_init", name),
            {
                "failed": False,
                "wait_ms": float(wait_ms),
                "consumed_s": self._profile_rel_s(),
            },
        )
        self._init_profile_set(
            ("parallel_init", "max_consume_wait_ms"),
            max(
                float(_nested_get(self._init_profile_snapshot(), ("parallel_init", "max_consume_wait_ms")) or 0.0),
                float(wait_ms),
            ),
        )
        return value

    def _shutdown_parallel_init_executor(self) -> None:
        executor = self._parallel_init_executor
        self._parallel_init_executor = None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    def _validate_live_contract(self) -> None:
        if tuple(self.args.camera_ids) != DEFAULT_CAMERA_IDS:
            raise RuntimeError("Demo 2.1 first live slice expects --camera-ids 0,1,2")
        if self.args.compile_mode not in COMPILE_MODES:
            raise RuntimeError(f"Demo 2.1 unsupported --compile-mode {self.args.compile_mode}")
        if self.args.ffs_worker_mode != "shared":
            raise RuntimeError("Demo 2.1 requires --ffs-worker-mode shared")
        if self.args.edgetam_worker_mode != "per-camera":
            raise RuntimeError("Demo 2.1 requires --edgetam-worker-mode per-camera")
        if self.args.gpu_pipeline_mode not in GPU_PIPELINE_MODES:
            raise RuntimeError(f"Demo 2.1 unsupported --gpu-pipeline-mode {self.args.gpu_pipeline_mode}")
        if self.args.single_owner_order not in SINGLE_OWNER_ORDERS:
            raise RuntimeError(f"Demo 2.1 unsupported --single-owner-order {self.args.single_owner_order}")
        if self.args.staged_order not in STAGED_ORDERS:
            raise RuntimeError(f"Demo 2.1 unsupported --staged-order {self.args.staged_order}")
        if self.args.stage_scheduler_mode not in STAGE_SCHEDULER_MODES:
            raise RuntimeError(f"Demo 2.1 unsupported --stage-scheduler-mode {self.args.stage_scheduler_mode}")
        if int(self.args.stage_lookahead) < 0:
            raise RuntimeError("Demo 2.1 --stage-lookahead must be >= 0")
        if self.args.edgetam_stream_mode not in EDGETAM_STREAM_MODES:
            raise RuntimeError(f"Demo 2.1 unsupported --edgetam-stream-mode {self.args.edgetam_stream_mode}")
        if self.args.edgetam_input_path not in EDGETAM_INPUT_PATH_MODES:
            raise RuntimeError(f"Demo 2.1 unsupported --edgetam-input-path {self.args.edgetam_input_path}")
        if self.args.mask_postprocess not in MASK_POSTPROCESS_MODES:
            raise RuntimeError(f"Demo 2.1 unsupported --mask-postprocess {self.args.mask_postprocess}")
        if (
            self.args.gpu_pipeline_mode == GPU_PIPELINE_MODE_SINGLE_OWNER
            and self.args.single_owner_order == SINGLE_OWNER_ORDER_INTERLEAVED
        ):
            raise RuntimeError(
                "Demo 2.1 --single-owner-order interleaved is reserved for a later per-camera interleaving "
                "implementation; use ffs-then-edgetam or edgetam-then-ffs for current profiling"
            )
        if bool(getattr(self.args, "edgetam_batch_vision_encoder", False)):
            if self.args.gpu_pipeline_mode not in {
                GPU_PIPELINE_MODE_SINGLE_OWNER,
                GPU_PIPELINE_MODE_OVERLAPPED_STAGES,
            }:
                raise RuntimeError(
                    "Demo 2.1 --edgetam-batch-vision-encoder requires single-owner or overlapped-stages GPU pipeline"
                )
            if self.args.edgetam_model_topology != EDGETAM_MODEL_TOPOLOGY_SHARED:
                raise RuntimeError("Demo 2.1 --edgetam-batch-vision-encoder requires shared EdgeTAM model topology")
            if edge_pin_memory_enabled(self.args):
                raise RuntimeError(
                    "Demo 2.1 --edgetam-batch-vision-encoder currently bypasses per-camera EdgeTAM pinned staging; "
                    "disable EdgeTAM pinning or use --pin-memory-mode ffs"
                )
        if (
            self.args.gpu_pipeline_mode == GPU_PIPELINE_MODE_SEPARATE_WORKERS
            and self.args.edgetam_model_topology != EDGETAM_MODEL_TOPOLOGY_REPLICATED
        ):
            raise RuntimeError("Demo 2.1 separate-workers mode requires --edgetam-model-topology replicated")
        if (
            self.args.gpu_pipeline_mode == GPU_PIPELINE_MODE_STAGED
            and self.args.edgetam_model_topology != EDGETAM_MODEL_TOPOLOGY_REPLICATED
        ):
            raise RuntimeError("Demo 2.1 staged mode requires --edgetam-model-topology replicated")
        if (
            self.args.gpu_pipeline_mode == GPU_PIPELINE_MODE_OVERLAPPED_STAGES
            and self.args.edgetam_model_topology != EDGETAM_MODEL_TOPOLOGY_SHARED
        ):
            raise RuntimeError("Demo 2.1 overlapped-stages mode requires --edgetam-model-topology shared")
        if self.args.gpu_pipeline_mode == GPU_PIPELINE_MODE_STAGED and self.args.gpu_gate_mode != GPU_GATE_MODE_OFF:
            raise RuntimeError("Demo 2.1 staged mode requires --gpu-gate-mode off so EdgeTAM can run in parallel")
        depth_pipeline_sources = {DEPTH_SOURCE_FFS, DEPTH_SOURCE_REALSENSE}
        if (
            self.args.gpu_pipeline_mode
            in {GPU_PIPELINE_MODE_SINGLE_OWNER, GPU_PIPELINE_MODE_STAGED, GPU_PIPELINE_MODE_OVERLAPPED_STAGES}
            and self.args.depth_source not in depth_pipeline_sources
        ):
            raise RuntimeError("Demo 2.1 single-owner/staged/overlapped-stages modes require --depth-source ffs or realsense")
        preset_canonical = canonical_preset_name(self.args.preset)
        if preset_canonical == PRESET_DEMO215_ASYNC_FILTER_5FPS:
            if self.args.depth_source != DEPTH_SOURCE_REALSENSE:
                raise RuntimeError("Demo 2.1.5 requires native RealSense depth")
            if self.args.gpu_pipeline_mode != GPU_PIPELINE_MODE_SINGLE_OWNER:
                raise RuntimeError("Demo 2.1.5 requires single-owner GPU pipeline")
            if not async_fusion_filter_enabled(self.args):
                raise RuntimeError("Demo 2.1.5 requires async latest-wins PCD filtering")
        if preset_canonical == PRESET_DEMO215_COMPILED_PARALLEL_EDGETAM_5FPS:
            if self.args.depth_source != DEPTH_SOURCE_REALSENSE:
                raise RuntimeError("Demo 2.1.5 parallel EdgeTAM requires native RealSense depth")
            if self.args.gpu_pipeline_mode != GPU_PIPELINE_MODE_SEPARATE_WORKERS:
                raise RuntimeError("Demo 2.1.5 parallel EdgeTAM requires separate worker GPU pipeline")
            if self.args.edgetam_model_topology != EDGETAM_MODEL_TOPOLOGY_REPLICATED:
                raise RuntimeError("Demo 2.1.5 parallel EdgeTAM requires replicated EdgeTAM models")
            if self.args.compile_mode not in {COMPILE_MODE_VISION_REDUCE_OVERHEAD, COMPILE_MODE_NONE}:
                raise RuntimeError(
                    "Demo 2.1.5 parallel EdgeTAM requires vision-reduce-overhead or none compile mode"
                )
            if self.args.gpu_gate_mode != GPU_GATE_MODE_OFF:
                raise RuntimeError("Demo 2.1.5 parallel EdgeTAM requires --gpu-gate-mode off")
            if not async_fusion_filter_enabled(self.args):
                raise RuntimeError("Demo 2.1.5 parallel EdgeTAM requires async latest-wins PCD filtering")
        if preset_canonical == PRESET_DEMO215_STAGED_PARALLEL_5FPS:
            if self.args.depth_source != DEPTH_SOURCE_REALSENSE:
                raise RuntimeError("Demo 2.1.5 staged parallel requires native RealSense depth")
            if self.args.gpu_pipeline_mode != GPU_PIPELINE_MODE_STAGED:
                raise RuntimeError("Demo 2.1.5 staged parallel requires staged GPU pipeline")
            if self.args.staged_order != STAGED_ORDER_FFS_THEN_PARALLEL_EDGETAM:
                raise RuntimeError("Demo 2.1.5 staged parallel requires depth then parallel EdgeTAM")
            if self.args.edgetam_stream_mode != EDGETAM_STREAM_MODE_PER_CAMERA:
                raise RuntimeError("Demo 2.1.5 staged parallel requires per-camera EdgeTAM streams")
            if not async_fusion_filter_enabled(self.args):
                raise RuntimeError("Demo 2.1.5 staged parallel requires async latest-wins PCD filtering")
            if self.args.pin_memory_mode != PIN_MEMORY_MODE_ALL or not bool(self.args.pin_memory):
                raise RuntimeError("Demo 2.1.5 staged parallel requires --pin-memory-mode all")
        if preset_canonical == PRESET_DEMO215_LIVE_FAST_NATIVE:
            if self.args.depth_source != DEPTH_SOURCE_REALSENSE:
                raise RuntimeError("Demo 2.1.5 live-fast-native requires native RealSense depth")
            if self.args.enable_pcd_filter or self.args.pcd_filter_mode != "none":
                raise RuntimeError("Demo 2.1.5 live-fast-native keeps PCD filtering out of the hot path")
        if preset_canonical == PRESET_DEMO215_LIVE_QUALITY_FFS:
            if self.args.depth_source != DEPTH_SOURCE_FFS:
                raise RuntimeError("Demo 2.1.5 live-quality-ffs requires local FFS depth")
            if not async_fusion_filter_enabled(self.args):
                raise RuntimeError("Demo 2.1.5 live-quality-ffs requires async latest-wins PCD filtering")
        if preset_canonical == PRESET_DEMO215_MASK_ONLY_DEBUG:
            if self.args.depth_source != DEPTH_SOURCE_NONE:
                raise RuntimeError("Demo 2.1.5 mask-only-debug requires --depth-source none")
            if self.args.render_mode != "none":
                raise RuntimeError("Demo 2.1.5 mask-only-debug requires --render-mode none")
        if preset_canonical == PRESET_DEMO22_ASYNC_FILTER_5FPS:
            if self.args.depth_source != DEPTH_SOURCE_FFS:
                raise RuntimeError("Demo 2.2 requires local FFS depth")
            if self.args.gpu_pipeline_mode not in {
                GPU_PIPELINE_MODE_SINGLE_OWNER,
                GPU_PIPELINE_MODE_OVERLAPPED_STAGES,
            }:
                raise RuntimeError("Demo 2.2 requires single-owner or overlapped-stages GPU pipeline")
            if not async_fusion_filter_enabled(self.args):
                raise RuntimeError("Demo 2.2 requires async latest-wins PCD filtering")
        if preset_canonical == PRESET_DEMO22_STAGED_PARALLEL_5FPS:
            if self.args.depth_source != DEPTH_SOURCE_FFS:
                raise RuntimeError("Demo 2.2 staged parallel requires local FFS depth")
            if self.args.gpu_pipeline_mode != GPU_PIPELINE_MODE_STAGED:
                raise RuntimeError("Demo 2.2 staged parallel requires staged GPU pipeline")
            if self.args.staged_order != STAGED_ORDER_FFS_THEN_PARALLEL_EDGETAM:
                raise RuntimeError("Demo 2.2 staged parallel requires FFS then parallel EdgeTAM")
            if self.args.edgetam_stream_mode != EDGETAM_STREAM_MODE_PER_CAMERA:
                raise RuntimeError("Demo 2.2 staged parallel requires per-camera EdgeTAM streams")
            if not async_fusion_filter_enabled(self.args):
                raise RuntimeError("Demo 2.2 staged parallel requires async latest-wins PCD filtering")
            if self.args.pin_memory_mode != PIN_MEMORY_MODE_ALL or not bool(self.args.pin_memory):
                raise RuntimeError("Demo 2.2 staged parallel requires --pin-memory-mode all")
            if self.args.ffs_input_staging != FFS_INPUT_STAGING_PINNED:
                raise RuntimeError("Demo 2.2 staged parallel requires pinned FFS input staging")
        if self.args.init_mode != "sam31-first-frame":
            raise RuntimeError("Formal Demo 2.1 requires live SAM3.1 initialization; saved masks are not allowed")
        if int(self.args.object_filter_cap) < 0 or int(self.args.controller_filter_cap) < 0:
            raise RuntimeError("Demo 2.1 filter caps must be >= 0")
        if float(self.args.object_filter_voxel_m) <= 0 or float(self.args.controller_filter_voxel_m) <= 0:
            raise RuntimeError("Demo 2.1 filter voxel sizes must be positive")
        if int(self.args.filter_every_n) < 1:
            raise RuntimeError("Demo 2.1 --filter-every-n must be >= 1")
        if float(self.args.filter_budget_ms) < 0:
            raise RuntimeError("Demo 2.1 --filter-budget-ms must be >= 0")
        if float(self.args.profile_warmup_exclude_s) < 0:
            raise RuntimeError("Demo 2.1 --profile-warmup-exclude-s must be >= 0")
        if float(self.args.gpu_sampling_interval_s) <= 0:
            raise RuntimeError("Demo 2.1 --gpu-sampling-interval-s must be > 0")
        if int(self.args.gpu_sampling_device_index) < 0:
            raise RuntimeError("Demo 2.1 --gpu-sampling-device-index must be >= 0")
        if self.args.gpu_sampling_backend not in GPU_SAMPLING_BACKENDS:
            raise RuntimeError(f"Demo 2.1 unsupported --gpu-sampling-backend {self.args.gpu_sampling_backend}")
        if float(self.args.sam31_init_retry_interval_s) < 0:
            raise RuntimeError("Demo 2.1 --sam31-init-retry-interval-s must be >= 0")
        if int(self.args.sam31_init_max_attempts) < 0:
            raise RuntimeError("Demo 2.1 --sam31-init-max-attempts must be >= 0")
        if self.args.gpu_gate_mode != GPU_GATE_MODE_OFF and int(self.args.gpu_gate_max_concurrent) < 1:
            raise RuntimeError("Demo 2.1 --gpu-gate-max-concurrent must be >= 1 unless --gpu-gate-mode off")
        if self.args.gpu_gate_mode == GPU_GATE_MODE_SERIALIZED and int(self.args.gpu_gate_max_concurrent) != 1:
            raise RuntimeError("Demo 2.1 serialized GPU gate requires --gpu-gate-max-concurrent 1")
        if int(self.args.pinned_ring_size) < 1:
            raise RuntimeError("Demo 2.1 --pinned-ring-size must be >= 1")
        if self.args.pin_memory_mode not in PIN_MEMORY_MODES:
            raise RuntimeError(f"Demo 2.1 unsupported --pin-memory-mode {self.args.pin_memory_mode}")
        if self.args.h2d_stream_mode not in H2D_STREAM_MODES:
            raise RuntimeError(f"Demo 2.1 unsupported --h2d-stream-mode {self.args.h2d_stream_mode}")
        if self.args.ffs_input_staging not in FFS_INPUT_STAGING_MODES:
            raise RuntimeError(f"Demo 2.1 unsupported --ffs-input-staging {self.args.ffs_input_staging}")
        if int(self.args.ffs_trt_batch_size) not in FFS_TRT_BATCH_SIZES:
            raise RuntimeError(f"Demo 2.1 unsupported --ffs-trt-batch-size {self.args.ffs_trt_batch_size}")
        if int(self.args.ffs_trt_batch_size) > 1:
            if self.args.depth_source != DEPTH_SOURCE_FFS:
                raise RuntimeError("Demo 2.1 batch FFS TensorRT requires --depth-source ffs")
            if len(tuple(self.args.camera_ids)) != int(self.args.ffs_trt_batch_size):
                raise RuntimeError(
                    "Demo 2.1 batch FFS TensorRT requires camera count to match --ffs-trt-batch-size"
                )
        if bool(self.args.pin_memory) and not str(self.args.device).startswith("cuda"):
            raise RuntimeError("Demo 2.1 pinned-memory ablation requires a CUDA device")
        if int(self.args.capture_buffer_size) < 1:
            raise RuntimeError("Demo 2.1 --capture-buffer-size must be >= 1")
        if float(self.args.max_capture_skew_ms) < 0:
            raise RuntimeError("Demo 2.1 --max-capture-skew-ms must be >= 0")
        if float(self.args.max_frame_age_ms) <= 0:
            raise RuntimeError("Demo 2.1 --max-frame-age-ms must be > 0")
        if self.args.capture_group_policy not in CAPTURE_GROUP_POLICIES:
            raise RuntimeError(f"Demo 2.1 unsupported --capture-group-policy {self.args.capture_group_policy}")
        if self.args.depth_source == DEPTH_SOURCE_FFS:
            validate_ffs_paths(ffs_repo=Path(self.args.ffs_repo), model_dir=Path(self.args.ffs_trt_model_dir))
        if self.args.init_mode == "saved-masks":
            if object_tracking_enabled(self.args.track_mode):
                if not self.args.object_init_mask_root:
                    raise RuntimeError("Demo 2.1 saved-masks object tracking requires --object-init-mask-root")
                object_root = Path(self.args.object_init_mask_root)
                if not object_root.is_dir():
                    raise FileNotFoundError(f"Demo 2.1 object init mask root does not exist: {object_root}")
            if controller_tracking_enabled(self.args.track_mode):
                if not self.args.controller_init_mask_root:
                    raise RuntimeError(
                        "Demo 2.1 saved-masks controller tracking requires --controller-init-mask-root"
                    )
                controller_root = Path(self.args.controller_init_mask_root)
                if not controller_root.is_dir():
                    raise FileNotFoundError(
                        f"Demo 2.1 controller init mask root does not exist: {controller_root}"
                    )
        if self._needs_world_fusion() and not Path(self.args.calibrate_path).is_file():
            raise FileNotFoundError(f"Demo 2.1 requires calibrate.pkl for world fusion: {self.args.calibrate_path}")

    def _needs_world_fusion(self) -> bool:
        return self.args.depth_source in {DEPTH_SOURCE_FFS, DEPTH_SOURCE_REALSENSE} and self.args.track_mode != TRACK_MODE_NONE

    def _start_camera_system(self) -> None:
        from data_process.visualization.calibration_io import load_calibration_transforms
        from qqtt.env.camera import CameraSystem

        capture_mode = "rgbd" if self.args.depth_source == DEPTH_SOURCE_REALSENSE else "stereo_ir"
        emitter = "auto" if self.args.depth_source == DEPTH_SOURCE_REALSENSE else "off"
        self.camera_system = CameraSystem(
            WH=(self.width, self.height),
            fps=int(self.args.fps),
            num_cam=3,
            serial_numbers=self.args.serials,
            capture_mode=capture_mode,
            emitter=emitter,
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
        self._shutdown_parallel_init_executor()
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
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        summary_path = output_root / f"session_{timestamp}_summary.json"
        latest = self._latest_fused
        self._summary["final"] = {
            "capture_group_fps": self.capture_group_stats.fps,
            "ffs_cycle_fps": self.ffs_stats.fps,
            "gpu_owner_fps": self.gpu_owner_stats.fps,
            "raw_fusion_fps": self.raw_fusion_stats.fps,
            "filter_output_fps": self.filter_output_stats.fps,
            "fusion_fps": self.fusion_stats.fps,
            "render_fps": self.render_stats.render_fps,
            "latest_group_id": None if latest is None else latest.group_id,
            "object_points": None if latest is None else latest.object_point_count,
            "controller_points": None if latest is None else latest.controller_point_count,
            "raw_fused_pending_replacements": self.raw_fused_slot.dropped_count,
            "raw_fused_pending_replacements_total": self.raw_fused_slot.total_dropped_count,
            "render_buffer": self.render_buffer.snapshot(),
            "render_post_gate": self.render_post_gate.snapshot(),
            "gpu_gate_wait_ms_median": {
                key: stats.median for key, stats in sorted(self.gpu_gate_wait_stats.items())
            },
        }
        self._summary["temporal_grouping"] = self._temporal_grouping_summary()
        self._summary["init_profile"] = self._init_profile_snapshot()
        self._summary["gpu_sampling"] = self._gpu_sampler.diagnostics()
        summary_path.write_text(json.dumps(self._summary, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")
        print(f"[demo2.1] summary={summary_path}", flush=True)
        if self._profile_enabled:
            if self.args.profile_json_output:
                profile_path = Path(self.args.profile_json_output)
            else:
                profile_path = output_root / f"session_{timestamp}_profile.json"
            self._write_profile_report(profile_path)

    def _profile_summary_for_records(self, records: Sequence[dict[str, Any]]) -> dict[str, Any]:
        complete = [record for record in records if record.get("complete")]
        rendered = [record for record in complete if isinstance(record.get("render"), dict)]
        summary: dict[str, Any] = {
            "group_count": int(len(records)),
            "complete_fusion_groups": int(len(complete)),
            "fusion_timeout_groups": int(sum(1 for record in records if record.get("drop_reason"))),
            "rendered_groups": int(len(rendered)),
            "capture_group_fps": _event_fps(records, ("t_group_created",)),
            "raw_fusion_fps": _event_fps(records, ("raw_fusion", "publish_s")),
            "filter_output_fps": _event_fps(complete, ("filter", "publish_s")),
            "fusion_fps": _event_fps(complete, ("fusion", "publish_s")),
            "render_fps": _event_fps(rendered, ("render", "render_s")),
        }
        period_stats = {
            "capture_group_period_ms": _event_period_stats_ms(records, ("t_group_created",)),
            "gpu_owner_publish_period_ms": _event_period_stats_ms(records, ("gpu_owner", "publish_s")),
            "ffs_stage_publish_period_ms": _event_period_stats_ms(records, ("ffs_stage", "publish_s")),
            "edgetam_stage_publish_period_ms": _event_period_stats_ms(records, ("edgetam_stage", "publish_s")),
            "stage_join_publish_period_ms": _event_period_stats_ms(records, ("stage_join", "publish_s")),
            "raw_fusion_publish_period_ms": _event_period_stats_ms(records, ("raw_fusion", "publish_s")),
            "filter_output_publish_period_ms": _event_period_stats_ms(complete, ("filter", "publish_s")),
            "fusion_publish_period_ms": _event_period_stats_ms(complete, ("fusion", "publish_s")),
            "display_packet_publish_period_ms": _event_period_stats_ms(records, ("render_publish", "publish_s")),
            "render_period_ms": _event_period_stats_ms(rendered, ("render", "render_s")),
        }
        summary["period_ms"] = period_stats
        summary["stage_period_ms"] = (
            period_stats["stage_join_publish_period_ms"]
            if _series_for_path(records, ("stage_join", "publish_s"))
            else period_stats["gpu_owner_publish_period_ms"]
        )
        summary["display_packet_period_ms"] = period_stats["display_packet_publish_period_ms"]
        summary["complete_group_ratio"] = float(len(complete) / len(records)) if records else 0.0
        summary["stage_drop_count"] = int(sum(1 for record in records if record.get("drop_reason")))
        summary["stage_pipeline"] = {
            "mode": str(self.args.gpu_pipeline_mode),
            "overlap_attempted": bool(self.args.gpu_pipeline_mode == GPU_PIPELINE_MODE_OVERLAPPED_STAGES),
            "scheduler_mode": str(getattr(self.args, "stage_scheduler_mode", STAGE_SCHEDULER_MODE_MASK_GATED)),
            "stage_lookahead": int(getattr(self.args, "stage_lookahead", 1)),
            "effective_period_ms": summary["stage_period_ms"],
            **self.stage_join_buffer.snapshot(),
            **self.stage_window_scheduler.snapshot(),
        }
        depth_ready_flags = [
            bool(record.get("stage_join", {}).get("depth_ready_before_mask"))
            for record in complete
            if isinstance(record.get("stage_join"), dict)
            and "depth_ready_before_mask" in record.get("stage_join", {})
        ]
        summary["stage_pipeline"]["depth_ready_before_mask_ratio"] = (
            float(sum(1 for flag in depth_ready_flags if flag) / len(depth_ready_flags))
            if depth_ready_flags
            else 0.0
        )
        depth_wait_values = _series_for_path(complete, ("stage_join", "depth_wait_after_mask_ms"))
        summary["stage_pipeline"]["mean_depth_wait_after_mask_ms"] = (
            float(sum(depth_wait_values) / len(depth_wait_values)) if depth_wait_values else 0.0
        )
        mask_wait_values = _series_for_path(complete, ("stage_join", "mask_wait_after_depth_ms"))
        summary["stage_pipeline"]["mean_mask_wait_after_depth_ms"] = (
            float(sum(mask_wait_values) / len(mask_wait_values)) if mask_wait_values else 0.0
        )
        summary["raw_fused_pending_replacements_total"] = int(self.raw_fused_slot.total_dropped_count)
        summary["render_buffer_dropped_total"] = int(self.render_buffer.snapshot().get("dropped", 0))
        target_fps = float(self.args.fusion_target_fps)
        summary["target_fps_deficit"] = float(target_fps - summary["render_fps"])
        summary["target_fps_deficit_ratio"] = float((target_fps - summary["render_fps"]) / target_fps) if target_fps > 0 else 0.0
        metric_paths: dict[str, tuple[str, ...]] = {
            "capture_temporal_skew_ms": ("capture", "max_temporal_skew_ms"),
            "edgetam_cam0_model_ms": ("edgetam", "cam0", "model_ms"),
            "edgetam_cam1_model_ms": ("edgetam", "cam1", "model_ms"),
            "edgetam_cam2_model_ms": ("edgetam", "cam2", "model_ms"),
            "edgetam_cam0_preprocess_ms": ("edgetam", "cam0", "preprocess_ms"),
            "edgetam_cam1_preprocess_ms": ("edgetam", "cam1", "preprocess_ms"),
            "edgetam_cam2_preprocess_ms": ("edgetam", "cam2", "preprocess_ms"),
            "edgetam_cam0_prompt_ms": ("edgetam", "cam0", "prompt_ms"),
            "edgetam_cam1_prompt_ms": ("edgetam", "cam1", "prompt_ms"),
            "edgetam_cam2_prompt_ms": ("edgetam", "cam2", "prompt_ms"),
            "edgetam_cam0_postprocess_ms": ("edgetam", "cam0", "postprocess_ms"),
            "edgetam_cam1_postprocess_ms": ("edgetam", "cam1", "postprocess_ms"),
            "edgetam_cam2_postprocess_ms": ("edgetam", "cam2", "postprocess_ms"),
            "edgetam_cam0_mask_resize_ms": ("edgetam", "cam0", "mask_resize_ms"),
            "edgetam_cam1_mask_resize_ms": ("edgetam", "cam1", "mask_resize_ms"),
            "edgetam_cam2_mask_resize_ms": ("edgetam", "cam2", "mask_resize_ms"),
            "edgetam_cam0_mask_threshold_ms": ("edgetam", "cam0", "mask_threshold_ms"),
            "edgetam_cam1_mask_threshold_ms": ("edgetam", "cam1", "mask_threshold_ms"),
            "edgetam_cam2_mask_threshold_ms": ("edgetam", "cam2", "mask_threshold_ms"),
            "edgetam_cam0_mask_to_cpu_ms": ("edgetam", "cam0", "mask_to_cpu_ms"),
            "edgetam_cam1_mask_to_cpu_ms": ("edgetam", "cam1", "mask_to_cpu_ms"),
            "edgetam_cam2_mask_to_cpu_ms": ("edgetam", "cam2", "mask_to_cpu_ms"),
            "edgetam_cam0_total_ms": ("edgetam", "cam0", "total_ms"),
            "edgetam_cam1_total_ms": ("edgetam", "cam1", "total_ms"),
            "edgetam_cam2_total_ms": ("edgetam", "cam2", "total_ms"),
            "edgetam_cam0_model_pre_sync_ms": ("edgetam", "cam0", "model_pre_sync_ms"),
            "edgetam_cam1_model_pre_sync_ms": ("edgetam", "cam1", "model_pre_sync_ms"),
            "edgetam_cam2_model_pre_sync_ms": ("edgetam", "cam2", "model_pre_sync_ms"),
            "edgetam_cam0_model_post_sync_ms": ("edgetam", "cam0", "model_post_sync_ms"),
            "edgetam_cam1_model_post_sync_ms": ("edgetam", "cam1", "model_post_sync_ms"),
            "edgetam_cam2_model_post_sync_ms": ("edgetam", "cam2", "model_post_sync_ms"),
            "edgetam_batch_vision_model_ms": ("edgetam", "batch_vision", "model_ms"),
            "edgetam_batch_vision_total_ms": ("edgetam", "batch_vision", "total_ms"),
            "edgetam_batch_vision_preprocess_ms": ("edgetam", "batch_vision", "preprocess_ms"),
            "edgetam_cam0_gate_wait_ms": ("edgetam", "cam0", "gate_wait_ms"),
            "edgetam_cam1_gate_wait_ms": ("edgetam", "cam1", "gate_wait_ms"),
            "edgetam_cam2_gate_wait_ms": ("edgetam", "cam2", "gate_wait_ms"),
            "edge_cam0_pin_copy_ms": ("h2d", "cam0", "edge", "pin_copy_ms"),
            "edge_cam1_pin_copy_ms": ("h2d", "cam1", "edge", "pin_copy_ms"),
            "edge_cam2_pin_copy_ms": ("h2d", "cam2", "edge", "pin_copy_ms"),
            "edge_cam0_h2d_wait_ms": ("h2d", "cam0", "edge", "h2d_wait_ms"),
            "edge_cam1_h2d_wait_ms": ("h2d", "cam1", "edge", "h2d_wait_ms"),
            "edge_cam2_h2d_wait_ms": ("h2d", "cam2", "edge", "h2d_wait_ms"),
            "ffs_gate_wait_ms": ("ffs", "gate_wait_ms"),
            "ffs_cycle_ms": ("ffs", "cycle_ms"),
            "ffs_batch_ms": ("ffs", "batch_ms"),
            "ffs_cam0_ms": ("ffs", "cam0_ffs_ms"),
            "ffs_cam1_ms": ("ffs", "cam1_ffs_ms"),
            "ffs_cam2_ms": ("ffs", "cam2_ffs_ms"),
            "ffs_cam0_stage_ms": ("h2d", "cam0", "ffs", "stage_ms"),
            "ffs_cam1_stage_ms": ("h2d", "cam1", "ffs", "stage_ms"),
            "ffs_cam2_stage_ms": ("h2d", "cam2", "ffs", "stage_ms"),
            "ffs_cam0_h2d_wait_ms": ("h2d", "cam0", "ffs", "h2d_wait_ms"),
            "ffs_cam1_h2d_wait_ms": ("h2d", "cam1", "ffs", "h2d_wait_ms"),
            "ffs_cam2_h2d_wait_ms": ("h2d", "cam2", "ffs", "h2d_wait_ms"),
            "ffs_align_cam0_ms": ("ffs", "cam0_align_ms"),
            "ffs_align_cam1_ms": ("ffs", "cam1_align_ms"),
            "ffs_align_cam2_ms": ("ffs", "cam2_align_ms"),
            "gpu_owner_total_ms": ("gpu_owner", "total_ms"),
            "gpu_owner_ffs_cycle_ms": ("gpu_owner", "ffs_cycle_ms"),
            "gpu_owner_edgetam_cycle_ms": ("gpu_owner", "edgetam_cycle_ms"),
            "staged_ffs_stage_ms": ("gpu_owner", "ffs_stage_ms"),
            "staged_edgetam_stage_wall_ms": ("gpu_owner", "edgetam_stage_wall_ms"),
            "staged_edgetam_stage_sum_model_ms": ("gpu_owner", "edgetam_stage_sum_model_ms"),
            "staged_edgetam_parallel_efficiency": ("gpu_owner", "edgetam_parallel_efficiency"),
            "staged_stage_barrier_ms": ("gpu_owner", "stage_barrier_ms"),
            "ffs_stage_wall_ms": ("ffs_stage", "wall_ms"),
            "ffs_stage_request_to_start_ms": ("ffs_stage", "request_to_start_ms"),
            "ffs_stage_input_age_ms": ("ffs_stage", "input_age_ms"),
            "edgetam_stage_wall_ms": ("edgetam_stage", "wall_ms"),
            "edgetam_stage_request_to_start_ms": ("edgetam_stage", "request_to_start_ms"),
            "edgetam_stage_sum_model_ms": ("edgetam_stage", "sum_model_ms"),
            "stage_join_wall_ms": ("stage_join", "wall_ms"),
            "stage_join_depth_wait_after_mask_ms": ("stage_join", "depth_wait_after_mask_ms"),
            "stage_join_mask_wait_after_depth_ms": ("stage_join", "mask_wait_after_depth_ms"),
            "stage_join_same_group_join_latency_ms": ("stage_join", "same_group_join_latency_ms"),
            "raw_fusion_total_ms": ("raw_fusion", "total_ms"),
            "fusion_total_ms": ("fusion", "total_ms"),
            "filter_total_ms": ("filter", "total_ms"),
            "filter_input_age_ms": ("filter", "input_age_ms"),
            "object_enhanced_pt_ms": ("fusion", "object_enhanced_pt_ms"),
            "controller_pt_filter_ms": ("fusion", "controller_pt_filter_ms"),
            "render_total_ms": ("render", "total_ms"),
            "render_queue_wait_ms": ("render", "queue_wait_ms"),
            "render_gpu_to_cpu_copy_ms": ("render", "gpu_to_cpu_copy_ms"),
            "render_combine_ms": ("render", "combine_ms"),
            "render_cpu_format_ms": ("render", "cpu_format_ms"),
            "render_open3d_points_update_ms": ("render", "open3d_points_update_ms"),
            "render_open3d_colors_update_ms": ("render", "open3d_colors_update_ms"),
            "render_open3d_update_geometry_ms": ("render", "update_geometry_ms"),
            "render_poll_events_ms": ("render", "poll_events_ms"),
            "render_update_renderer_ms": ("render", "update_renderer_ms"),
            "open3d_object_update_geometry_ms": ("render", "object_update_geometry_ms"),
            "open3d_controller_update_geometry_ms": ("render", "controller_update_geometry_ms"),
        }
        summary["metrics"] = {
            name: _profile_stats(_series_for_path(records, path))
            for name, path in metric_paths.items()
        }
        aggregate_metric_paths: dict[str, tuple[tuple[str, ...], ...]] = {
            "edgetam_model_ms": (
                ("edgetam", "cam0", "model_ms"),
                ("edgetam", "cam1", "model_ms"),
                ("edgetam", "cam2", "model_ms"),
            ),
            "edgetam_preprocess_ms": (
                ("edgetam", "cam0", "preprocess_ms"),
                ("edgetam", "cam1", "preprocess_ms"),
                ("edgetam", "cam2", "preprocess_ms"),
            ),
            "edgetam_prompt_ms": (
                ("edgetam", "cam0", "prompt_ms"),
                ("edgetam", "cam1", "prompt_ms"),
                ("edgetam", "cam2", "prompt_ms"),
            ),
            "edgetam_postprocess_ms": (
                ("edgetam", "cam0", "postprocess_ms"),
                ("edgetam", "cam1", "postprocess_ms"),
                ("edgetam", "cam2", "postprocess_ms"),
            ),
            "edgetam_mask_resize_ms": (
                ("edgetam", "cam0", "mask_resize_ms"),
                ("edgetam", "cam1", "mask_resize_ms"),
                ("edgetam", "cam2", "mask_resize_ms"),
            ),
            "edgetam_mask_threshold_ms": (
                ("edgetam", "cam0", "mask_threshold_ms"),
                ("edgetam", "cam1", "mask_threshold_ms"),
                ("edgetam", "cam2", "mask_threshold_ms"),
            ),
            "edgetam_mask_to_cpu_ms": (
                ("edgetam", "cam0", "mask_to_cpu_ms"),
                ("edgetam", "cam1", "mask_to_cpu_ms"),
                ("edgetam", "cam2", "mask_to_cpu_ms"),
            ),
            "edgetam_total_ms": (
                ("edgetam", "cam0", "total_ms"),
                ("edgetam", "cam1", "total_ms"),
                ("edgetam", "cam2", "total_ms"),
            ),
            "edge_pin_copy_ms": (
                ("h2d", "cam0", "edge", "pin_copy_ms"),
                ("h2d", "cam1", "edge", "pin_copy_ms"),
                ("h2d", "cam2", "edge", "pin_copy_ms"),
            ),
            "edge_h2d_wait_ms": (
                ("h2d", "cam0", "edge", "h2d_wait_ms"),
                ("h2d", "cam1", "edge", "h2d_wait_ms"),
                ("h2d", "cam2", "edge", "h2d_wait_ms"),
            ),
            "ffs_stage_ms": (
                ("h2d", "cam0", "ffs", "stage_ms"),
                ("h2d", "cam1", "ffs", "stage_ms"),
                ("h2d", "cam2", "ffs", "stage_ms"),
            ),
            "ffs_h2d_wait_ms": (
                ("h2d", "cam0", "ffs", "h2d_wait_ms"),
                ("h2d", "cam1", "ffs", "h2d_wait_ms"),
                ("h2d", "cam2", "ffs", "h2d_wait_ms"),
            ),
        }
        for name, paths in aggregate_metric_paths.items():
            values: list[float] = []
            for path in paths:
                values.extend(_series_for_path(records, path))
            summary["metrics"][name] = _profile_stats(values)
        render_micro_records = [
            record["render"]["micro_profile"]
            for record in rendered
            if isinstance(record.get("render"), dict) and isinstance(record["render"].get("micro_profile"), dict)
        ]
        summary["render_micro_profile"] = summarize_render_records(render_micro_records)
        summary["render_backpressure_count"] = int(summary["render_micro_profile"].get("render_backpressure_count", 0))
        if summary["fusion_fps"] < target_fps:
            summary["bottleneck_class"] = "upstream_supply"
        elif summary["render_fps"] < target_fps:
            summary["bottleneck_class"] = "visualization"
        else:
            summary["bottleneck_class"] = "target_met"
        return summary

    def _build_profile_payload(self) -> dict[str, Any]:
        with self._profile_lock:
            records = [dict(record) for _, record in sorted(self._profile_records.items())]
        warmup_s = float(self.args.profile_warmup_exclude_s)
        after_warmup = [
            record for record in records
            if float(record.get("t_group_created", 0.0)) >= warmup_s
        ]
        contract = build_contract(self.args)
        slow_filter_groups = sorted(
            (
                {
                    "group_id": int(record.get("group_id", -1)),
                    "total_enhanced_pt_ms": float(_nested_get(record, ("fusion", "object_enhanced_pt_ms")) or 0.0),
                    "input_points": int(_nested_get(record, ("points", "object_raw")) or 0),
                    "kept_points": int(_nested_get(record, ("points", "object_filtered")) or 0),
                    "detail": _nested_get(record, ("fusion", "object_filter_detail")) or {},
                }
                for record in records
            ),
            key=lambda item: item["total_enhanced_pt_ms"],
            reverse=True,
        )[:10]
        gpu_samples = self._gpu_sampler.samples_snapshot()
        gpu_sampling = self._gpu_sampler.diagnostics()
        gpu_sampling["summary_full_run"] = summarize_gpu_samples(gpu_samples, start_s=0.0)
        gpu_sampling["summary_after_warmup"] = summarize_gpu_samples(gpu_samples, start_s=warmup_s)
        gpu_sampling["samples"] = gpu_samples
        return {
            "preset": self.args.preset,
            "preset_canonical": getattr(self.args, "preset_canonical", canonical_preset_name(self.args.preset)),
            "target_fps": float(self.args.fusion_target_fps),
            "capture_group_target_fps": resolved_capture_group_target_fps(self.args),
            "demo22_pass_threshold_fps": float(self.args.fusion_target_fps) * DEMO22_PASS_THRESHOLD_RATIO,
            "track_mode": self.args.track_mode,
            "experiment_mode": contract["experiment_mode"],
            "controller_semantic": contract["controller_semantic"],
            "controller_prompt": contract["controller_prompt"],
            "controller_prompt_expected": contract["controller_prompt_expected"],
            "controller_prompt_matches_experiment_mode": contract["controller_prompt_matches_experiment_mode"],
            "depth_source": self.args.depth_source,
            "compile_mode": self.args.compile_mode,
            "dtype": self.args.dtype,
            "input_path": self.args.edgetam_input_path,
            "mask_postprocess": self.args.mask_postprocess,
            "gpu_pipeline": contract["gpu_pipeline"],
            "gpu_sampling": gpu_sampling,
            "filter_scheduler": contract["filter_scheduler"],
            "renderer": contract["renderer"],
            "render_buffer": self.render_buffer.snapshot(),
            "render_post_gate": self.render_post_gate.snapshot(),
            "gpu_gate_max_concurrent": int(self.args.gpu_gate_max_concurrent),
            "object_filter": self.args.object_postprocess,
            "controller_filter": self.args.controller_postprocess,
            "object_controller_union_before_filter": bool(contract["fusion"]["object_controller_union_before_filter"]),
            "temporal_grouping": contract["temporal_grouping"],
            "pin_memory_enabled": bool(self.args.pin_memory),
            "pin_memory_mode": self.args.pin_memory_mode,
            "ffs_input_staging": self.args.ffs_input_staging,
            "h2d_stream_mode": self.args.h2d_stream_mode,
            "h2d_transfer": contract["h2d_transfer"],
            "tracking_overlay": contract["tracking_overlay"],
            "tracking_overlay_enabled": bool(getattr(self.args, "show_tracking_overlay", False)),
            "tracking_backend": str(getattr(self.args, "tracking_backend", "none")),
            "tracking_source": str(getattr(self.args, "tracking_source", "cached")),
            "tracking_update_hz": float(getattr(self.args, "tracking_update_hz", 5.0)),
            "tracking_model_ms_median": None,
            "tracking_e2e_ms_median": None,
            "track_overlay_ms_median": 0.0,
            "visible_ratio_mean": 0.0,
            "inside_mask_ratio_mean": 0.0,
            "depth_valid_ratio_mean": 0.0,
            "lifted_3d_count_mean": 0.0,
            "init_profile": self._init_profile_snapshot(),
            "warmup_exclude_s": warmup_s,
            "summary_full_run": self._profile_summary_for_records(records),
            "summary_after_warmup": self._profile_summary_for_records(after_warmup),
            "top_slowest_object_filter_groups": slow_filter_groups,
            "per_group": records,
        }

    def _write_profile_report(self, profile_path: Path) -> None:
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        payload = self._build_profile_payload()
        profile_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")
        md_path = profile_path.with_suffix(".md")
        warm = payload["summary_after_warmup"]
        metrics = warm.get("metrics", {})
        init_profile = payload.get("init_profile", {})
        is_demo22 = payload.get("preset_canonical") in {
            PRESET_DEMO22_ASYNC_FILTER_5FPS,
            PRESET_DEMO22_STAGED_PARALLEL_5FPS,
        }
        pass_threshold = float(payload.get("demo22_pass_threshold_fps", 4.8))
        pass_status = (
            "PASS"
            if is_demo22
            and float(warm.get("render_fps", 0.0)) >= pass_threshold
            and payload.get("depth_source") == DEPTH_SOURCE_FFS
            and payload.get("filter_scheduler", {}).get("render_filtered_only")
            else "FAIL"
        )
        lines = [
            "# Demo 2.2 performance profile" if is_demo22 else "# Demo 2.1 performance profile",
            "",
            f"- preset: `{payload['preset']}`",
            f"- canonical preset: `{payload.get('preset_canonical', payload['preset'])}`",
            f"- target FPS: `{payload['target_fps']:.2f}`",
            f"- capture group target FPS: `{payload.get('capture_group_target_fps', payload['target_fps']):.2f}`",
            f"- compile mode: `{payload.get('compile_mode', 'unknown')}`",
            f"- dtype: `{payload.get('dtype', 'unknown')}`",
            f"- EdgeTAM input path: `{payload.get('input_path', 'pil')}`",
            f"- mask postprocess: `{payload.get('mask_postprocess', 'hf')}`",
            f"- render backend: `{payload.get('renderer', {}).get('backend', 'legacy-inplace')}`",
            f"- render latest-only: `{payload.get('renderer', {}).get('async_latest_only', True)}`",
            f"- render copy mode: `{payload.get('renderer', {}).get('copy_mode', 'sync-cpu')}`",
            f"- render FPS after warmup: `{warm.get('render_fps', 0.0):.2f}`",
            f"- raw fusion FPS after warmup: `{warm.get('raw_fusion_fps', 0.0):.2f}`",
            f"- filter output FPS after warmup: `{warm.get('filter_output_fps', 0.0):.2f}`",
            f"- fusion FPS after warmup: `{warm.get('fusion_fps', 0.0):.2f}`",
            f"- stage period p50 after warmup: `{warm.get('stage_period_ms', {}).get('median', 0.0):.2f} ms`",
            f"- display packet period p50 after warmup: `{warm.get('display_packet_period_ms', {}).get('median', 0.0):.2f} ms`",
            f"- groups after warmup: `{warm.get('group_count', 0)}`",
            f"- complete fused groups after warmup: `{warm.get('complete_fusion_groups', 0)}`",
            f"- rendered groups after warmup: `{warm.get('rendered_groups', 0)}`",
            f"- complete group ratio after warmup: `{warm.get('complete_group_ratio', 0.0):.3f}`",
            f"- stage drop count after warmup: `{warm.get('stage_drop_count', 0)}`",
            f"- raw fused pending replacements total: `{warm.get('raw_fused_pending_replacements_total', 0)}`",
            f"- render buffer dropped total: `{warm.get('render_buffer_dropped_total', 0)}`",
            f"- target deficit: `{warm.get('target_fps_deficit', 0.0):.2f}`",
            f"- bottleneck class: `{warm.get('bottleneck_class', 'unknown')}`",
            f"- GPU pipeline: `{payload.get('gpu_pipeline', {}).get('mode', 'separate-workers')}`",
            f"- single-owner order: `{payload.get('gpu_pipeline', {}).get('internal_order', 'ffs-then-edgetam')}`",
            f"- filter scheduler: `{payload.get('filter_scheduler', {}).get('mode', 'sync')}`",
            f"- render filtered only: `{payload.get('filter_scheduler', {}).get('render_filtered_only', False)}`",
            f"- pin memory mode: `{payload.get('pin_memory_mode', 'off')}`",
            f"- FFS input staging: `{payload.get('ffs_input_staging', 'pinned')}`",
            f"- H2D stream mode: `{payload.get('h2d_stream_mode', 'default')}`",
            "",
        ]
        if is_demo22:
            lines.insert(14, f"- Demo 2.2 PASS threshold: `{pass_threshold:.2f} FPS`")
            lines.insert(15, f"- Demo 2.2 result: `{pass_status}`")
        if int(warm.get("complete_fusion_groups", 0)) == 0:
            lines.extend(
                [
                    "Warning: this profile has no complete fused groups after warmup. Treat it as an initialization or missing-packet run, not as a valid visual FPS comparison.",
                    "",
                ]
            )
        startup_rows = (
            ("parallel init max wait ms", ("parallel_init", "max_consume_wait_ms")),
            ("camera startup ms", ("camera_startup_ms",)),
            ("EdgeTAM model load ms", ("edgetam", "model_load_ms_total")),
            ("EdgeTAM compile wrap ms", ("edgetam", "compile_wrap_ms_total")),
            ("EdgeTAM compile prewarm ms", ("edgetam", "prewarm", "total_ms")),
            ("EdgeTAM warmup/first forward ms", ("edgetam", "first_forward_total_ms_sum")),
            ("SAM3.1 model load ms", ("sam31", "model_load_ms_total")),
            ("SAM3.1 cam0 segment ms", ("sam31", "cam0", "total_ms")),
            ("SAM3.1 cam1 segment ms", ("sam31", "cam1", "total_ms")),
            ("SAM3.1 cam2 segment ms", ("sam31", "cam2", "total_ms")),
            ("FFS runner init ms", ("ffs", "runner_init_ms_total")),
            ("FFS first run ms", ("ffs", "first_run_ms_sum")),
            ("session init + prompt add ms", ("edgetam", "session_init_plus_prompt_ms_total")),
            ("SAM3.1 release cleanup ms", ("sam31", "release_cleanup_ms")),
            ("time to first complete group s", ("first_complete_fused_group_s",)),
            ("time to first rendered group s", ("first_render_s",)),
        )
        lines.extend(["## Init Profile", "", "| Stage | value |", "| --- | ---: |"])
        for label, path in startup_rows:
            value = _nested_get(init_profile, path)
            if value is None:
                lines.append(f"| {label} | `n/a` |")
            else:
                lines.append(f"| {label} | `{float(value):.2f}` |")
        lines.append("")
        gpu_sampling = payload.get("gpu_sampling", {}) or {}
        gpu_summary = gpu_sampling.get("summary_after_warmup", {}) or {}
        gpu_metrics = gpu_summary.get("metrics", {}) or {}
        lines.extend(["## GPU Sampling", ""])
        if gpu_sampling.get("enabled"):
            lines.extend(
                [
                    f"- backend requested: `{gpu_sampling.get('requested_backend', 'auto')}`",
                    f"- backend used: `{gpu_sampling.get('backend_used') or 'unavailable'}`",
                    f"- device index: `{gpu_sampling.get('device_index', 0)}`",
                    f"- interval s: `{float(gpu_sampling.get('interval_s', 0.0) or 0.0):.3f}`",
                    f"- samples after warmup: `{gpu_summary.get('sample_count', 0)}`",
                    "",
                    "| Metric | median | p90 | p95 | max |",
                    "| --- | ---: | ---: | ---: | ---: |",
                ]
            )
            for name in (
                "gpu_util_pct",
                "memory_util_pct",
                "memory_used_mb",
                "power_w",
                "sm_clock_mhz",
                "mem_clock_mhz",
                "temperature_c",
            ):
                stat = gpu_metrics.get(name, {})
                lines.append(
                    f"| `{name}` | `{stat.get('median', 0.0):.2f}` | `{stat.get('p90', 0.0):.2f}` | "
                    f"`{stat.get('p95', 0.0):.2f}` | `{stat.get('max', 0.0):.2f}` |"
                )
            errors = gpu_sampling.get("errors") or []
            if errors:
                lines.extend(["", f"- sampler errors: `{'; '.join(str(error) for error in errors[:3])}`"])
        else:
            lines.append("GPU sampling disabled for this run.")
        lines.append("")
        lines.extend([
            "## Throughput periods",
            "",
            "| Event | median ms | p90 ms | p95 ms | max ms |",
            "| --- | ---: | ---: | ---: | ---: |",
        ])
        for name, stat in sorted((warm.get("period_ms") or {}).items()):
            lines.append(
                f"| `{name}` | `{stat.get('median', 0.0):.2f}` | `{stat.get('p90', 0.0):.2f}` | "
                f"`{stat.get('p95', 0.0):.2f}` | `{stat.get('max', 0.0):.2f}` |"
            )
        lines.append("")
        lines.extend([
            "| Metric | median | p90 | p95 | max |",
            "| --- | ---: | ---: | ---: | ---: |",
        ])
        for name in (
            "capture_temporal_skew_ms",
            "edgetam_model_ms",
            "edgetam_preprocess_ms",
            "edgetam_prompt_ms",
            "edgetam_postprocess_ms",
            "edgetam_mask_resize_ms",
            "edgetam_mask_threshold_ms",
            "edgetam_mask_to_cpu_ms",
            "edgetam_total_ms",
            "ffs_cycle_ms",
            "ffs_batch_ms",
            "ffs_gate_wait_ms",
            "edgetam_batch_vision_model_ms",
            "edgetam_batch_vision_total_ms",
            "edgetam_batch_vision_preprocess_ms",
            "edgetam_cam0_model_ms",
            "edgetam_cam1_model_ms",
            "edgetam_cam2_model_ms",
            "edgetam_cam0_gate_wait_ms",
            "edgetam_cam1_gate_wait_ms",
            "edgetam_cam2_gate_wait_ms",
            "edge_pin_copy_ms",
            "edge_h2d_wait_ms",
            "edge_cam0_pin_copy_ms",
            "edge_cam1_pin_copy_ms",
            "edge_cam2_pin_copy_ms",
            "edge_cam0_h2d_wait_ms",
            "edge_cam1_h2d_wait_ms",
            "edge_cam2_h2d_wait_ms",
            "ffs_stage_ms",
            "ffs_h2d_wait_ms",
            "ffs_cam0_stage_ms",
            "ffs_cam1_stage_ms",
            "ffs_cam2_stage_ms",
            "ffs_cam0_h2d_wait_ms",
            "ffs_cam1_h2d_wait_ms",
            "ffs_cam2_h2d_wait_ms",
            "gpu_owner_total_ms",
            "gpu_owner_ffs_cycle_ms",
            "gpu_owner_edgetam_cycle_ms",
            "raw_fusion_total_ms",
            "fusion_total_ms",
            "filter_total_ms",
            "filter_input_age_ms",
            "object_enhanced_pt_ms",
            "controller_pt_filter_ms",
            "render_total_ms",
            "render_queue_wait_ms",
            "render_gpu_to_cpu_copy_ms",
            "render_combine_ms",
            "render_cpu_format_ms",
            "render_open3d_points_update_ms",
            "render_open3d_colors_update_ms",
            "render_open3d_update_geometry_ms",
            "render_poll_events_ms",
            "render_update_renderer_ms",
            "open3d_object_update_geometry_ms",
            "open3d_controller_update_geometry_ms",
        ):
            stat = metrics.get(name, {})
            lines.append(
                f"| `{name}` | `{stat.get('median', 0.0):.2f}` | `{stat.get('p90', 0.0):.2f}` | "
                f"`{stat.get('p95', 0.0):.2f}` | `{stat.get('max', 0.0):.2f}` |"
            )
        lines.extend(["", "## Top slowest object enhanced-PT groups", "", "| group | ms | input points | kept points |", "| ---: | ---: | ---: | ---: |"])
        for item in payload.get("top_slowest_object_filter_groups", [])[:10]:
            lines.append(
                f"| `{item.get('group_id', -1)}` | `{item.get('total_enhanced_pt_ms', 0.0):.2f}` | "
                f"`{item.get('input_points', 0)}` | `{item.get('kept_points', 0)}` |"
            )
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[demo2.1] profile_json={profile_path}", flush=True)
        print(f"[demo2.1] profile_md={md_path}", flush=True)

    def _thread_specs(self) -> list[tuple[str, Callable[[], None]]]:
        specs: list[tuple[str, Callable[[], None]]] = [
            ("capture-group", self._capture_group_worker),
        ]
        depth_for_pcd = self.args.depth_source in {DEPTH_SOURCE_FFS, DEPTH_SOURCE_REALSENSE}
        if self.args.gpu_pipeline_mode == GPU_PIPELINE_MODE_SINGLE_OWNER:
            if self.args.track_mode != TRACK_MODE_NONE and depth_for_pcd:
                specs.append(("gpu-owner", self._gpu_owner_pipeline_worker))
                specs.append(("fusion", self._fusion_worker_single_owner))
        elif self.args.gpu_pipeline_mode == GPU_PIPELINE_MODE_STAGED:
            if self.args.track_mode != TRACK_MODE_NONE and depth_for_pcd:
                specs.append(("staged-gpu", self._staged_gpu_pipeline_worker))
                specs.append(("fusion", self._fusion_worker_single_owner))
        elif self.args.gpu_pipeline_mode == GPU_PIPELINE_MODE_OVERLAPPED_STAGES:
            if self.args.track_mode != TRACK_MODE_NONE and depth_for_pcd:
                specs.append(("stage-dispatch", self._stage_capture_dispatch_worker))
                specs.append(("ffs-stage", self._ffs_stage_worker))
                specs.append(("edgetam-stage", self._edgetam_stage_worker))
                specs.append(("stage-join", self._stage_join_fusion_worker))
        else:
            if self.args.depth_source == DEPTH_SOURCE_FFS:
                specs.append(("shared-ffs", self._shared_ffs_worker))
            if self.args.depth_source == DEPTH_SOURCE_REALSENSE:
                specs.append(("realsense-depth", self._native_realsense_depth_worker))
            if self.args.track_mode != TRACK_MODE_NONE:
                for camera_idx in self.args.camera_ids:
                    specs.append((f"edgetam-cam{camera_idx}", lambda camera_idx=int(camera_idx): self._edgetam_camera_worker(camera_idx)))
            if self.args.track_mode != TRACK_MODE_NONE and depth_for_pcd:
                specs.append(("fusion", self._fusion_worker))
        if async_fusion_filter_enabled(self.args) and self.args.track_mode != TRACK_MODE_NONE and depth_for_pcd:
            specs.append(("filter", self._async_filter_worker))
        if self.args.debug and self.args.render_mode == "none":
            specs.append(("debug", self._debug_worker))
        return specs

    def _start_threads(self) -> None:
        specs = self._thread_specs()
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

    def _metadata_frame_packet(
        self,
        *,
        group_id: int,
        camera_idx: int,
        obs: dict[str, Any],
        capture_arrival_perf_ns: int | None = None,
    ) -> CameraFramePacket:
        metadata = self._stream_metadata[int(camera_idx)]
        k_color = np.asarray(metadata["K_color"], dtype=np.float32).reshape(3, 3)
        intrinsics = _camera_intrinsics_from_k(k_color, width=self.width, height=self.height)
        host_timestamp_s = float(obs.get("timestamp", time.time()))
        realsense_timestamp_ms: float | None = None
        if "camera_capture_timestamp" in obs:
            try:
                realsense_timestamp_ms = float(obs["camera_capture_timestamp"]) * 1000.0
            except Exception:
                realsense_timestamp_ms = None
        realsense_frame_number: int | None = None
        if "realsense_frame_number" in obs:
            try:
                realsense_frame_number = int(obs["realsense_frame_number"])
            except Exception:
                realsense_frame_number = None
        return CameraFramePacket(
            group_id=int(group_id),
            camera_idx=int(camera_idx),
            frame_seq=int(obs.get("step_idx", group_id)),
            timestamp_ns=_as_timestamp_ns(host_timestamp_s),
            realsense_timestamp_ms=realsense_timestamp_ms,
            realsense_frame_number=realsense_frame_number,
            timestamp_domain=None if obs.get("timestamp_domain") is None else str(obs.get("timestamp_domain")),
            capture_arrival_perf_ns=int(time.perf_counter_ns() if capture_arrival_perf_ns is None else capture_arrival_perf_ns),
            color_bgr=np.ascontiguousarray(obs["color"].copy()),
            ir_left_u8=(
                np.ascontiguousarray(obs["ir_left"].copy()) if obs.get("ir_left") is not None else None
            ),
            ir_right_u8=(
                np.ascontiguousarray(obs["ir_right"].copy()) if obs.get("ir_right") is not None else None
            ),
            k_color=k_color,
            k_ir_left=(
                np.asarray(metadata["K_ir_left"], dtype=np.float32).reshape(3, 3)
                if metadata.get("K_ir_left") is not None
                else None
            ),
            t_ir_left_to_color=(
                np.asarray(metadata["T_ir_left_to_color"], dtype=np.float32).reshape(4, 4)
                if metadata.get("T_ir_left_to_color") is not None
                else None
            ),
            baseline_m=float(metadata.get("ir_baseline_m", 0.0) or 0.0),
            intrinsics=intrinsics,
            c2w=self._c2w_by_camera[int(camera_idx)],
            depth_u16=(
                np.ascontiguousarray(obs["depth"].copy()) if obs.get("depth") is not None else None
            ),
            depth_scale_m_per_unit=float(metadata.get("depth_scale_m_per_unit", 0.0) or 0.0),
        )

    def _capture_group_worker(self) -> None:
        assert self.camera_system is not None
        group_id = 0
        raw_capture_id = 0
        capture_group_target_fps = resolved_capture_group_target_fps(self.args)
        interval_s = 0.0 if capture_group_target_fps <= 0 else 1.0 / max(1e-6, capture_group_target_fps)
        next_tick_s = time.perf_counter()
        buffers: dict[int, deque[CameraFramePacket]] = {
            int(camera_idx): deque(maxlen=int(self.args.capture_buffer_size))
            for camera_idx in self.args.camera_ids
        }
        while not self.stop_event.is_set():
            now_s = time.perf_counter()
            if interval_s > 0 and now_s < next_tick_s:
                time.sleep(min(0.002, next_tick_s - now_s))
                continue
            next_tick_s = now_s + interval_s if interval_s > 0 else now_s
            build_start_s = time.perf_counter()
            try:
                realsense_runtime = None if self.camera_system is None else getattr(self.camera_system, "realsense", None)
                if realsense_runtime is not None and not realsense_runtime.is_ready:
                    self._summary["capture_not_ready_skip"] = int(self._summary.get("capture_not_ready_skip", 0)) + 1
                    if self.args.debug:
                        print("[WARN] Demo 2.1 capture group skipped because not all cameras are ready", flush=True)
                    continue
                obs = self.camera_system.get_observation()
                capture_arrival_perf_ns = time.perf_counter_ns()
                for camera_idx in self.args.camera_ids:
                    frame = self._metadata_frame_packet(
                        group_id=raw_capture_id,
                        camera_idx=int(camera_idx),
                        obs=obs[int(camera_idx)],
                        capture_arrival_perf_ns=capture_arrival_perf_ns,
                    )
                    buffers[int(camera_idx)].append(frame)
                raw_capture_id += 1
            except TimeoutError as exc:
                if not self.stop_event.is_set() and self.args.debug:
                    print(f"[WARN] Demo 2.1 capture group skipped after timeout: {exc}", flush=True)
                self._summary["capture_timeout_count"] = int(self._summary.get("capture_timeout_count", 0)) + 1
                continue
            except Exception as exc:
                if not self.stop_event.is_set():
                    print(f"[ERROR] Demo 2.1 capture group failed: {type(exc).__name__}: {exc}", flush=True)
                self._mark_fatal_error("capture-group", exc)
                self.stop_event.set()
                break
            now_perf_ns = time.perf_counter_ns()
            pruned = prune_stale_capture_buffers(
                buffers,
                max_frame_age_ms=float(self.args.max_frame_age_ms),
                now_perf_ns=now_perf_ns,
            )
            if pruned:
                self._summary["capture_stale_frames_pruned"] = int(self._summary.get("capture_stale_frames_pruned", 0)) + int(pruned)
            selection = select_temporal_capture_triplet(
                buffers,
                camera_ids=self.args.camera_ids,
                policy=str(self.args.capture_group_policy),
                max_frame_age_ms=float(self.args.max_frame_age_ms),
                now_perf_ns=now_perf_ns,
            )
            if selection is None:
                self._summary["capture_group_no_candidate"] = int(self._summary.get("capture_group_no_candidate", 0)) + 1
                continue
            if selection.max_temporal_skew_ms > float(self.args.max_capture_skew_ms):
                self._summary["capture_group_skew_drop"] = int(self._summary.get("capture_group_skew_drop", 0)) + 1
                if self.args.drop_skewed_groups or self.args.capture_group_policy == CAPTURE_GROUP_POLICY_TIMESTAMP_STRICT:
                    drop_oldest_capture_buffer_frame(buffers)
                    if self._profile_enabled:
                        self._profile_mark_drop(group_id, "capture_temporal_skew")
                        self._profile_update(
                            group_id,
                            t_group_created=self._profile_rel_s(),
                            capture={
                                "group_build_ms": _elapsed_ms(build_start_s, time.perf_counter()),
                                "timestamp_source": selection.timestamp_source,
                                "max_temporal_skew_ms": float(selection.max_temporal_skew_ms),
                                "skew_drop": True,
                                "age_ms": float(selection.age_ms),
                            },
                        )
                    continue
            packet = build_temporal_capture_group(
                group_id=group_id,
                created_perf_s=time.perf_counter(),
                selection=selection,
            )
            drop_selected_and_older_frames(buffers, selection)
            self._record_temporal_skew(packet.max_temporal_skew_ms)
            self._summary["capture_groups_emitted"] = int(self._summary.get("capture_groups_emitted", 0)) + 1
            self._summary["capture_timestamp_source"] = packet.timestamp_source
            self.capture_group_slot.put(packet)
            self.capture_group_stats.record(packet.created_perf_s)
            self._profile_update(
                group_id,
                t_group_created=self._profile_rel_s(packet.created_perf_s),
                capture={
                    "group_build_ms": _elapsed_ms(build_start_s, packet.created_perf_s),
                    "timestamp_source": packet.timestamp_source,
                    "group_timestamp_ns": int(packet.group_timestamp_ns),
                    "max_temporal_skew_ms": float(packet.max_temporal_skew_ms),
                    "age_ms": float(selection.age_ms),
                    **{
                        f"frame_seq_cam{int(camera_idx)}": int(packet.frames[int(camera_idx)].frame_seq)
                        for camera_idx in self.args.camera_ids
                    },
                    **{
                        f"offset_cam{int(camera_idx)}_ms": float(packet.per_camera_time_offset_ms.get(int(camera_idx), 0.0))
                        for camera_idx in self.args.camera_ids
                    },
                },
            )
            group_id += 1

    def _create_ffs_runner(self) -> object:
        from data_process.depth_backends import (
            FastFoundationStereoTensorRTRunner,
            resolve_tensorrt_engine_static_batch_size,
        )

        start_s = time.perf_counter()
        static_batch_size = resolve_tensorrt_engine_static_batch_size(
            trt_mode="two_stage",
            model_dir=Path(self.args.ffs_trt_model_dir),
            trt_root=None if self.args.ffs_trt_root is None else Path(self.args.ffs_trt_root),
        )
        if int(static_batch_size) != int(self.args.ffs_trt_batch_size):
            raise RuntimeError(
                "FFS TensorRT engine static batch size does not match --ffs-trt-batch-size. "
                f"engine={static_batch_size} requested={self.args.ffs_trt_batch_size} "
                f"model_dir={self.args.ffs_trt_model_dir}"
            )
        runner = FastFoundationStereoTensorRTRunner(
            ffs_repo=Path(self.args.ffs_repo),
            model_dir=Path(self.args.ffs_trt_model_dir),
            trt_root=None if self.args.ffs_trt_root is None else Path(self.args.ffs_trt_root),
            input_staging=str(self.args.ffs_input_staging),
        )
        init_ms = _elapsed_ms(start_s, time.perf_counter())
        self._init_profile_add(("ffs", "runner_init_ms_total"), init_ms)
        self._init_profile_set_once(("ffs", "first_runner_init_ms"), init_ms)
        self._init_profile_set(("ffs", "static_batch_size"), int(static_batch_size))
        return runner

    def _prepare_ffs_runner(self) -> object:
        start_s = time.perf_counter()
        warm_up_numba_ffs_align()
        self._init_profile_set_once(
            ("ffs", "numba_warmup_ms"),
            _elapsed_ms(start_s, time.perf_counter()),
        )
        return self._create_ffs_runner()

    def _get_or_prepare_ffs_runner(self) -> object:
        runner = self._consume_parallel_init_future("ffs_runner")
        if runner is not None:
            return runner
        return self._prepare_ffs_runner()

    def _preload_sam31_init_model(self) -> dict[str, Any]:
        from scripts.harness.sam31_mask_helper import preload_sam31_image_processor_cache

        start_s = time.perf_counter()
        with self._sam31_lock:
            result = preload_sam31_image_processor_cache(
                checkpoint_path=None,
                compile_model=False,
                device=str(self.args.device),
            )
        timing = dict(result.get("timing_ms", {}) or {})
        self._init_profile_update(
            ("sam31", "preload"),
            {
                **timing,
                "call_wall_ms": float(_elapsed_ms(start_s, time.perf_counter())),
                "checkpoint_path": result.get("checkpoint_path"),
                "bpe_path": result.get("bpe_path"),
            },
        )
        self._init_profile_add(("sam31", "model_load_ms_total"), float(timing.get("model_load_ms", 0.0) or 0.0))
        return result

    def _consume_sam31_preload_if_ready(self) -> None:
        if "sam31_preload" not in self._parallel_init_futures:
            return
        result = self._consume_parallel_init_future("sam31_preload")
        timing = dict((result or {}).get("timing_ms", {}) or {})
        self._summary["sam31_parallel_preload_consumed"] = True
        self._init_profile_update(
            ("sam31", "preload_consumed"),
            {
                **timing,
                "checkpoint_path": (result or {}).get("checkpoint_path"),
                "bpe_path": (result or {}).get("bpe_path"),
            },
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
        return self._align_ffs_depth_for_frame(
            frame=frame,
            aligners=aligners,
            depth_ir_left_m=depth_ir_left_m,
            k_ir_left_used=k_ir_left_used,
            ffs_ms=_elapsed_ms(ffs_start_s, ffs_done_s),
        )

    def _run_ffs_cycle_for_group(
        self,
        *,
        runner: object,
        group: CaptureGroup,
        aligners: dict[int, FfsIrToColorAligner],
    ) -> tuple[DepthGroup, dict[int, dict[str, Any]]]:
        if int(self.args.ffs_trt_batch_size) > 1:
            return self._run_ffs_batch_cycle_for_group(runner=runner, group=group, aligners=aligners)
        cycle_start_s = time.perf_counter()
        depths: dict[int, DepthPacket] = {}
        per_camera: dict[int, dict[str, float]] = {}
        h2d_by_camera: dict[int, dict[str, Any]] = {}
        gate_wait_ms_total = 0.0
        for camera_idx in self.args.camera_ids:
            frame = group.frames[int(camera_idx)]
            with self.gpu_gate.acquire(stage="ffs", camera_idx=int(camera_idx), group_id=group.group_id) as gate_wait_ms:
                self._record_gpu_gate_wait("ffs", gate_wait_ms)
                depth_ir_left_m, k_ir_left_used, ffs_ms, ffs_h2d_profile = self._run_ffs_pair_for_frame(
                    runner=runner,
                    frame=frame,
                )
            gate_wait_ms_total += float(gate_wait_ms)
            depth = self._align_ffs_depth_for_frame(
                frame=frame,
                aligners=aligners,
                depth_ir_left_m=depth_ir_left_m,
                k_ir_left_used=k_ir_left_used,
                ffs_ms=ffs_ms,
            )
            depths[int(camera_idx)] = depth
            per_camera[int(camera_idx)] = {
                "ffs_ms": depth.ffs_ms,
                "align_ms": depth.align_ms,
                "gate_wait_ms": float(gate_wait_ms),
            }
            h2d_by_camera[int(camera_idx)] = dict(ffs_h2d_profile)

        packet = DepthGroup(
            group_id=group.group_id,
            depths=depths,
            total_ms=_elapsed_ms(cycle_start_s, time.perf_counter()),
            per_camera_ms=per_camera,
            gpu_gate_wait_ms=gate_wait_ms_total,
            max_temporal_skew_ms=float(group.max_temporal_skew_ms),
            per_camera_time_offset_ms=dict(group.per_camera_time_offset_ms),
            per_camera_frame_seq=dict(group.per_camera_frame_seq),
            timestamp_source=str(group.timestamp_source),
        )
        if not self._first_ffs_cycle_recorded:
            self._first_ffs_cycle_recorded = True
            first_run_by_camera = {
                f"cam{int(camera_idx)}": float(per_camera[int(camera_idx)].get("ffs_ms", 0.0))
                for camera_idx in self.args.camera_ids
            }
            first_align_by_camera = {
                f"cam{int(camera_idx)}": float(per_camera[int(camera_idx)].get("align_ms", 0.0))
                for camera_idx in self.args.camera_ids
            }
            self._init_profile_update(
                ("ffs",),
                {
                    "first_group_id": int(group.group_id),
                    "first_cycle_ms": float(packet.total_ms),
                    "first_run_ms_by_camera": first_run_by_camera,
                    "first_align_ms_by_camera": first_align_by_camera,
                    "first_run_ms_sum": float(sum(first_run_by_camera.values())),
                    "first_align_ms_sum": float(sum(first_align_by_camera.values())),
                },
            )
        self._profile_update(
            group.group_id,
            ffs={
                "gate_wait_ms": float(gate_wait_ms_total),
                "cycle_ms": float(packet.total_ms),
                "publish_s": self._profile_rel_s(),
                "capture_temporal_skew_ms": float(group.max_temporal_skew_ms),
                **{
                    f"cam{int(camera_idx)}_ffs_ms": float(per_camera[int(camera_idx)].get("ffs_ms", 0.0))
                    for camera_idx in self.args.camera_ids
                },
                **{
                    f"cam{int(camera_idx)}_align_ms": float(per_camera[int(camera_idx)].get("align_ms", 0.0))
                    for camera_idx in self.args.camera_ids
                },
                **{
                    f"cam{int(camera_idx)}_gate_wait_ms": float(per_camera[int(camera_idx)].get("gate_wait_ms", 0.0))
                    for camera_idx in self.args.camera_ids
                },
            },
            h2d={
                f"cam{int(camera_idx)}": {
                    "ffs": {
                        **h2d_by_camera.get(int(camera_idx), {}),
                        "profile_enabled": bool(self.args.profile_h2d),
                    }
                }
                for camera_idx in self.args.camera_ids
            },
        )
        return packet, h2d_by_camera

    def _run_ffs_batch_cycle_for_group(
        self,
        *,
        runner: object,
        group: CaptureGroup,
        aligners: dict[int, FfsIrToColorAligner],
    ) -> tuple[DepthGroup, dict[int, dict[str, Any]]]:
        camera_ids = tuple(int(camera_idx) for camera_idx in self.args.camera_ids)
        batch_size = int(self.args.ffs_trt_batch_size)
        if len(camera_ids) != batch_size:
            raise RuntimeError(
                f"FFS batch cycle expected {batch_size} cameras, got {len(camera_ids)} camera_ids={camera_ids}"
            )

        samples: list[dict[str, Any]] = []
        frames: list[CameraFramePacket] = []
        for camera_idx in camera_ids:
            frame = group.frames[int(camera_idx)]
            if (
                frame.ir_left_u8 is None
                or frame.ir_right_u8 is None
                or frame.k_ir_left is None
                or frame.baseline_m <= 0
            ):
                raise RuntimeError(f"cam{frame.camera_idx} is missing FFS IR stereo data")
            frames.append(frame)
            samples.append(
                {
                    "left_image": frame.ir_left_u8,
                    "right_image": frame.ir_right_u8,
                    "K_ir_left": frame.k_ir_left,
                    "baseline_m": float(frame.baseline_m),
                }
            )

        cycle_start_s = time.perf_counter()
        gate_camera_idx = -1
        with self.gpu_gate.acquire(stage="ffs", camera_idx=gate_camera_idx, group_id=group.group_id) as gate_wait_ms:
            self._record_gpu_gate_wait("ffs", gate_wait_ms)
            ffs_start_s = time.perf_counter()
            outputs = runner.run_batch(samples)
            ffs_batch_ms = _elapsed_ms(ffs_start_s, time.perf_counter())
        if len(outputs) != len(frames):
            raise RuntimeError(
                f"FFS batch output count mismatch for group {group.group_id}: {len(outputs)} vs {len(frames)}"
            )

        depths: dict[int, DepthPacket] = {}
        per_camera: dict[int, dict[str, float]] = {}
        h2d_by_camera: dict[int, dict[str, Any]] = {}
        per_camera_ffs_ms = float(ffs_batch_ms / max(1, len(frames)))
        for frame, output in zip(frames, outputs):
            depth = self._align_ffs_depth_for_frame(
                frame=frame,
                aligners=aligners,
                depth_ir_left_m=np.asarray(output["depth_ir_left_m"], dtype=np.float32),
                k_ir_left_used=np.asarray(output.get("K_ir_left_used", frame.k_ir_left), dtype=np.float32),
                ffs_ms=per_camera_ffs_ms,
            )
            camera_idx = int(frame.camera_idx)
            depths[camera_idx] = depth
            per_camera[camera_idx] = {
                "ffs_ms": depth.ffs_ms,
                "align_ms": depth.align_ms,
                "gate_wait_ms": float(gate_wait_ms),
            }
            h2d_by_camera[camera_idx] = dict(output.get("h2d_profile", {}))

        packet = DepthGroup(
            group_id=group.group_id,
            depths=depths,
            total_ms=_elapsed_ms(cycle_start_s, time.perf_counter()),
            per_camera_ms=per_camera,
            gpu_gate_wait_ms=float(gate_wait_ms),
            max_temporal_skew_ms=float(group.max_temporal_skew_ms),
            per_camera_time_offset_ms=dict(group.per_camera_time_offset_ms),
            per_camera_frame_seq=dict(group.per_camera_frame_seq),
            timestamp_source=str(group.timestamp_source),
        )
        if not self._first_ffs_cycle_recorded:
            self._first_ffs_cycle_recorded = True
            first_align_by_camera = {
                f"cam{int(camera_idx)}": float(per_camera[int(camera_idx)].get("align_ms", 0.0))
                for camera_idx in camera_ids
            }
            self._init_profile_update(
                ("ffs",),
                {
                    "first_group_id": int(group.group_id),
                    "first_cycle_ms": float(packet.total_ms),
                    "first_batch_run_ms": float(ffs_batch_ms),
                    "first_batch_size": int(batch_size),
                    "first_run_ms_by_camera": {
                        f"cam{int(camera_idx)}": per_camera_ffs_ms for camera_idx in camera_ids
                    },
                    "first_align_ms_by_camera": first_align_by_camera,
                    "first_run_ms_sum": float(ffs_batch_ms),
                    "first_align_ms_sum": float(sum(first_align_by_camera.values())),
                },
            )
        self._profile_update(
            group.group_id,
            ffs={
                "gate_wait_ms": float(gate_wait_ms),
                "cycle_ms": float(packet.total_ms),
                "batch_ms": float(ffs_batch_ms),
                "batch_size": int(batch_size),
                "publish_s": self._profile_rel_s(),
                "capture_temporal_skew_ms": float(group.max_temporal_skew_ms),
                **{
                    f"cam{int(camera_idx)}_ffs_ms": float(per_camera[int(camera_idx)].get("ffs_ms", 0.0))
                    for camera_idx in camera_ids
                },
                **{
                    f"cam{int(camera_idx)}_align_ms": float(per_camera[int(camera_idx)].get("align_ms", 0.0))
                    for camera_idx in camera_ids
                },
                **{
                    f"cam{int(camera_idx)}_gate_wait_ms": float(gate_wait_ms)
                    for camera_idx in camera_ids
                },
            },
            h2d={
                f"cam{int(camera_idx)}": {
                    "ffs": {
                        **h2d_by_camera.get(int(camera_idx), {}),
                        "profile_enabled": bool(self.args.profile_h2d),
                    }
                }
                for camera_idx in camera_ids
            },
        )
        return packet, h2d_by_camera

    def _run_ffs_pair_for_frame(
        self,
        *,
        runner: object,
        frame: CameraFramePacket,
    ) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any]]:
        if (
            frame.ir_left_u8 is None
            or frame.ir_right_u8 is None
            or frame.k_ir_left is None
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
        ffs_ms = _elapsed_ms(ffs_start_s, time.perf_counter())
        h2d_profile = dict(output.get("h2d_profile", {}))
        return (
            np.asarray(output["depth_ir_left_m"], dtype=np.float32),
            np.asarray(output.get("K_ir_left_used", frame.k_ir_left), dtype=np.float32),
            ffs_ms,
            h2d_profile,
        )

    def _align_ffs_depth_for_frame(
        self,
        *,
        frame: CameraFramePacket,
        aligners: dict[int, FfsIrToColorAligner],
        depth_ir_left_m: np.ndarray,
        k_ir_left_used: np.ndarray,
        ffs_ms: float,
    ) -> DepthPacket:
        if frame.t_ir_left_to_color is None:
            raise RuntimeError(f"cam{frame.camera_idx} is missing IR-to-color transform")
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
            ffs_ms=float(ffs_ms),
            align_ms=_elapsed_ms(align_start_s, align_done_s),
        )

    def _compute_realsense_depth_for_frame(self, *, frame: CameraFramePacket) -> DepthPacket:
        if frame.depth_u16 is None:
            raise RuntimeError(f"cam{frame.camera_idx} is missing native RealSense depth data")
        if float(frame.depth_scale_m_per_unit) <= 0.0:
            raise RuntimeError(f"cam{frame.camera_idx} is missing native RealSense depth scale")
        convert_start_s = time.perf_counter()
        depth_m = np.ascontiguousarray(
            frame.depth_u16.astype(np.float32) * np.float32(frame.depth_scale_m_per_unit),
            dtype=np.float32,
        )
        convert_ms = _elapsed_ms(convert_start_s, time.perf_counter())
        return DepthPacket(
            group_id=frame.group_id,
            camera_idx=frame.camera_idx,
            depth_m=depth_m,
            ffs_ms=0.0,
            align_ms=float(convert_ms),
        )

    def _run_realsense_depth_cycle_for_group(self, *, group: CaptureGroup) -> DepthGroup:
        cycle_start_s = time.perf_counter()
        depths: dict[int, DepthPacket] = {}
        per_camera: dict[int, dict[str, float]] = {}
        for camera_idx in self.args.camera_ids:
            frame = group.frames[int(camera_idx)]
            depth = self._compute_realsense_depth_for_frame(frame=frame)
            depths[int(camera_idx)] = depth
            per_camera[int(camera_idx)] = {
                "realsense_depth_ms": depth.align_ms,
                "ffs_ms": 0.0,
                "align_ms": depth.align_ms,
                "gate_wait_ms": 0.0,
            }

        packet = DepthGroup(
            group_id=group.group_id,
            depths=depths,
            total_ms=_elapsed_ms(cycle_start_s, time.perf_counter()),
            per_camera_ms=per_camera,
            gpu_gate_wait_ms=0.0,
            max_temporal_skew_ms=float(group.max_temporal_skew_ms),
            per_camera_time_offset_ms=dict(group.per_camera_time_offset_ms),
            per_camera_frame_seq=dict(group.per_camera_frame_seq),
            timestamp_source=str(group.timestamp_source),
        )
        if not self._first_ffs_cycle_recorded:
            self._first_ffs_cycle_recorded = True
            self._init_profile_update(
                ("realsense_depth",),
                {
                    "first_group_id": int(group.group_id),
                    "first_cycle_ms": float(packet.total_ms),
                    "first_convert_ms_by_camera": {
                        f"cam{int(camera_idx)}": float(per_camera[int(camera_idx)]["realsense_depth_ms"])
                        for camera_idx in self.args.camera_ids
                    },
                },
            )
        self._profile_update(
            group.group_id,
            realsense_depth={
                "cycle_ms": float(packet.total_ms),
                "publish_s": self._profile_rel_s(),
                "capture_temporal_skew_ms": float(group.max_temporal_skew_ms),
                **{
                    f"cam{int(camera_idx)}_convert_ms": float(
                        per_camera[int(camera_idx)].get("realsense_depth_ms", 0.0)
                    )
                    for camera_idx in self.args.camera_ids
                },
            },
        )
        return packet

    def _run_depth_cycle_for_group(
        self,
        *,
        group: CaptureGroup,
        runner: object | None,
        aligners: dict[int, FfsIrToColorAligner],
    ) -> tuple[DepthGroup, dict[int, dict[str, Any]]]:
        if self.args.depth_source == DEPTH_SOURCE_REALSENSE:
            return self._run_realsense_depth_cycle_for_group(group=group), {}
        if self.args.depth_source == DEPTH_SOURCE_FFS:
            if runner is None:
                raise RuntimeError("FFS runner is required for FFS depth")
            return self._run_ffs_cycle_for_group(runner=runner, group=group, aligners=aligners)
        raise RuntimeError(f"unsupported depth source for PCD fusion: {self.args.depth_source}")

    def _shared_ffs_worker(self) -> None:
        try:
            runner = self._get_or_prepare_ffs_runner()
            aligners: dict[int, FfsIrToColorAligner] = {}
            last_group_id = -1
            while not self.stop_event.is_set():
                group = self.capture_group_slot.get_latest_after(last_group_id)
                if group is None:
                    time.sleep(0.001)
                    continue
                last_group_id = group.group_id
                if not temporal_group_is_coherent(group, max_capture_skew_ms=float(self.args.max_capture_skew_ms)):
                    self._summary["ffs_drop_skewed_capture_group"] = int(
                        self._summary.get("ffs_drop_skewed_capture_group", 0)
                    ) + 1
                    self._profile_mark_drop(group.group_id, "ffs_drop_skewed_capture_group")
                    continue
                packet, _ = self._run_ffs_cycle_for_group(runner=runner, group=group, aligners=aligners)
                self.depth_group_slot.put(packet)
                self._latest_depth_group = packet
                self.ffs_stats.record()
        except Exception as exc:
            if not self.stop_event.is_set():
                print(f"[ERROR] Demo 2.1 shared FFS worker failed: {type(exc).__name__}: {exc}", flush=True)
            self._mark_fatal_error("shared-ffs", exc)
            self.stop_event.set()

    def _native_realsense_depth_worker(self) -> None:
        try:
            last_group_id = -1
            while not self.stop_event.is_set():
                group = self.capture_group_slot.get_latest_after(last_group_id)
                if group is None:
                    time.sleep(0.001)
                    continue
                last_group_id = group.group_id
                if not temporal_group_is_coherent(group, max_capture_skew_ms=float(self.args.max_capture_skew_ms)):
                    self._summary["realsense_depth_drop_skewed_capture_group"] = int(
                        self._summary.get("realsense_depth_drop_skewed_capture_group", 0)
                    ) + 1
                    self._profile_mark_drop(group.group_id, "realsense_depth_drop_skewed_capture_group")
                    continue
                packet = self._run_realsense_depth_cycle_for_group(group=group)
                self.depth_group_slot.put(packet)
                self._latest_depth_group = packet
                self.ffs_stats.record()
        except Exception as exc:
            if not self.stop_event.is_set():
                print(f"[ERROR] Demo 2.1 native RealSense depth worker failed: {type(exc).__name__}: {exc}", flush=True)
            self._mark_fatal_error("realsense-depth", exc)
            self.stop_event.set()

    def _autocast_context(self, torch_module: Any) -> Any:
        if not str(self.args.device).startswith("cuda") or self.args.dtype == "float32":
            return nullcontext()
        dtype = torch_module.bfloat16 if self.args.dtype == "bfloat16" else torch_module.float16
        return torch_module.autocast("cuda", dtype=dtype)

    def _init_hf_model(self, camera_idx: int) -> tuple[Any, Any, Any, Any, Any]:
        total_start_s = time.perf_counter()
        runtime_start_s = time.perf_counter()
        hf_stream = _load_hf_streaming_runtime()
        runtime_deps_ms = _elapsed_ms(runtime_start_s, time.perf_counter())
        torch_module = hf_stream.torch
        if str(self.args.device).startswith("cuda") and not torch_module.cuda.is_available():
            raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
        dtype = hf_stream._dtype_from_name(self.args.dtype)
        model_load_start_s = time.perf_counter()
        model = hf_stream.EdgeTamVideoModel.from_pretrained(self.args.model_id).to(self.args.device, dtype=dtype)
        model.eval()
        model_load_ms = _elapsed_ms(model_load_start_s, time.perf_counter())
        compile_start_s = time.perf_counter()
        model, compile_metadata = self._apply_edgetam_compile_mode(
            hf_stream=hf_stream,
            torch_module=torch_module,
            model=model,
        )
        compile_wrap_ms = _elapsed_ms(compile_start_s, time.perf_counter())
        if (
            self.args.gpu_pipeline_mode in {GPU_PIPELINE_MODE_SEPARATE_WORKERS, GPU_PIPELINE_MODE_STAGED}
            and self.args.gpu_gate_mode == GPU_GATE_MODE_OFF
            and "vision_encoder" in set(compile_metadata.get("applied_targets", []))
        ):
            compile_metadata["cudagraph_output_clone_wrapper"] = wrap_compiled_vision_encoder_outputs_for_parallel(
                model,
                torch_module,
            )
        processor_start_s = time.perf_counter()
        processor = hf_stream.Sam2VideoProcessor.from_pretrained(self.args.model_id)
        processor_load_ms = _elapsed_ms(processor_start_s, time.perf_counter())
        topology_label = (
            f"{self.args.edgetam_model_topology}:shared-loader"
            if int(camera_idx) < 0
            else str(self.args.edgetam_model_topology)
        )
        loader_key = "shared" if int(camera_idx) < 0 else f"cam{int(camera_idx)}"
        self._init_profile_update(
            ("edgetam", "loaders", loader_key),
            {
                "runtime_deps_ms": float(runtime_deps_ms),
                "model_load_ms": float(model_load_ms),
                "compile_wrap_ms": float(compile_metadata.get("wrap_ms", compile_wrap_ms)),
                "compile_wrap_wall_ms": float(compile_wrap_ms),
                "processor_load_ms": float(processor_load_ms),
                "total_ms": float(_elapsed_ms(total_start_s, time.perf_counter())),
                "compile_mode": str(self.args.compile_mode),
                "compile_requested_targets": list(compile_metadata.get("requested_targets", [])),
                "compile_targets": list(compile_metadata.get("applied_targets", [])),
                "compile_missing_targets": list(compile_metadata.get("missing_targets", [])),
                "compile_failed_targets": dict(compile_metadata.get("failed_targets", {}) or {}),
                "topology": topology_label,
            },
        )
        self._init_profile_add(("edgetam", "model_load_ms_total"), float(model_load_ms))
        self._init_profile_add(
            ("edgetam", "compile_wrap_ms_total"),
            float(compile_metadata.get("wrap_ms", compile_wrap_ms)),
        )
        self._init_profile_add(("edgetam", "processor_load_ms_total"), float(processor_load_ms))
        prewarm_profile = self._prewarm_edgetam_compile(
            hf_stream=hf_stream,
            torch_module=torch_module,
            dtype=dtype,
            model=model,
            processor=processor,
            loader_key=loader_key,
        )
        if prewarm_profile:
            self._init_profile_update(("edgetam", "loaders", loader_key, "prewarm"), prewarm_profile)
            if bool(prewarm_profile.get("enabled", False)):
                self._init_profile_set(("edgetam", "prewarm", "enabled"), True)
                self._init_profile_add(("edgetam", "prewarm", "total_ms"), float(prewarm_profile.get("total_ms", 0.0)))
                self._init_profile_add(
                    ("edgetam", "prewarm", "model_total_ms"),
                    float(prewarm_profile.get("model_total_ms", 0.0)),
                )
                self._init_profile_add(("edgetam", "prewarm", "runs"), float(prewarm_profile.get("runs", 0)))
        print(
            "[demo2.1-edgetam] "
            f"cam={camera_idx} topology={topology_label} model={self.args.model_id} "
            f"compile={self.args.compile_mode} applied={compile_metadata.get('applied_targets', [])} "
            f"clone_wrap={compile_metadata.get('cudagraph_output_clone_wrapper', False)}",
            flush=True,
        )
        return hf_stream, torch_module, dtype, model, processor

    def _apply_edgetam_compile_mode(self, *, hf_stream: Any, torch_module: Any, model: Any) -> tuple[Any, dict[str, Any]]:
        compile_mode = str(self.args.compile_mode)
        if compile_mode == COMPILE_MODE_NONE:
            return model, {
                "compile_mode": compile_mode,
                "enabled": False,
                "torch_compile_available": bool(hasattr(torch_module, "compile")),
                "torch_compile_mode": None,
                "fullgraph": False,
                "dynamic": False,
                "requested_targets": [],
                "applied_targets": [],
                "missing_targets": [],
                "whole_model_compiled": False,
                "wrap_ms": 0.0,
            }
        if compile_mode == COMPILE_MODE_VISION_REDUCE_OVERHEAD:
            return hf_stream._apply_compile_mode(model, compile_mode)
        component_compile_modes = {
            COMPILE_MODE_VISION_MAX_AUTOTUNE_NO_CUDAGRAPHS: (
                "max-autotune-no-cudagraphs",
                ["vision_encoder"],
            ),
            COMPILE_MODE_COMPONENTS_REDUCE_OVERHEAD: (
                "reduce-overhead",
                ["vision_encoder", "memory_attention", "memory_encoder", "mask_decoder"],
            ),
            COMPILE_MODE_COMPONENTS_MAX_AUTOTUNE_NO_CUDAGRAPHS: (
                "max-autotune-no-cudagraphs",
                ["vision_encoder", "memory_attention", "memory_encoder", "mask_decoder"],
            ),
        }
        if compile_mode in component_compile_modes:
            torch_compile_mode, requested_targets = component_compile_modes[compile_mode]
            metadata: dict[str, Any] = {
                "compile_mode": compile_mode,
                "enabled": True,
                "torch_compile_available": bool(hasattr(torch_module, "compile")),
                "torch_compile_mode": torch_compile_mode,
                "fullgraph": False,
                "dynamic": False,
                "requested_targets": list(requested_targets),
                "applied_targets": [],
                "missing_targets": [],
                "failed_targets": {},
                "whole_model_compiled": False,
                "wrap_ms": 0.0,
            }
            if not hasattr(torch_module, "compile"):
                metadata["failed_targets"] = {
                    target: "torch.compile is not available"
                    for target in requested_targets
                }
                return model, metadata
            started_s = time.perf_counter()
            for target in requested_targets:
                if not hasattr(model, target):
                    metadata["missing_targets"].append(target)
                    continue
                try:
                    setattr(
                        model,
                        target,
                        torch_module.compile(
                            getattr(model, target),
                            mode=torch_compile_mode,
                            fullgraph=False,
                            dynamic=False,
                        ),
                    )
                except Exception as exc:
                    metadata["failed_targets"][target] = f"{type(exc).__name__}: {exc}"
                    continue
                metadata["applied_targets"].append(target)
            metadata["wrap_ms"] = _elapsed_ms(started_s, time.perf_counter())
            return model, metadata
        if compile_mode != COMPILE_MODE_VISION_DEFAULT:
            raise RuntimeError(f"Unsupported EdgeTAM compile mode: {compile_mode}")
        metadata: dict[str, Any] = {
            "compile_mode": compile_mode,
            "enabled": True,
            "torch_compile_available": bool(hasattr(torch_module, "compile")),
            "torch_compile_mode": "default",
            "fullgraph": False,
            "dynamic": False,
            "requested_targets": ["vision_encoder"],
            "applied_targets": ["vision_encoder"] if hasattr(model, "vision_encoder") else [],
            "missing_targets": [] if hasattr(model, "vision_encoder") else ["vision_encoder"],
            "whole_model_compiled": False,
            "wrap_ms": 0.0,
        }
        if not hasattr(torch_module, "compile"):
            raise RuntimeError("Requested --compile-mode vision-default but torch.compile is not available.")
        if not hasattr(model, "vision_encoder"):
            raise RuntimeError("Requested --compile-mode vision-default, but model.vision_encoder was not found.")
        started_s = time.perf_counter()
        model.vision_encoder = torch_module.compile(
            model.vision_encoder,
            mode="default",
            fullgraph=False,
            dynamic=False,
        )
        metadata["wrap_ms"] = _elapsed_ms(started_s, time.perf_counter())
        return model, metadata

    def _dummy_prewarm_prompt_masks(self) -> tuple[list[int], list[np.ndarray]]:
        height = int(self.height)
        width = int(self.width)
        object_mask = np.zeros((height, width), dtype=bool)
        y0 = max(0, int(height * 0.36))
        y1 = min(height, int(height * 0.64))
        x0 = max(0, int(width * 0.38))
        x1 = min(width, int(width * 0.62))
        object_mask[y0:y1, x0:x1] = True
        obj_ids: list[int] = []
        masks: list[np.ndarray] = []
        if controller_tracking_enabled(self.args.track_mode):
            controller_mask = np.zeros((height, width), dtype=bool)
            cy0 = max(0, int(height * 0.18))
            cy1 = min(height, int(height * 0.34))
            cx0 = max(0, int(width * 0.18))
            cx1 = min(width, int(width * 0.34))
            controller_mask[cy0:cy1, cx0:cx1] = True
            obj_ids.append(CONTROLLER_ID)
            masks.append(controller_mask)
        if object_tracking_enabled(self.args.track_mode):
            obj_ids.append(OBJECT_ID)
            masks.append(object_mask)
        return obj_ids, masks

    def _prewarm_edgetam_compile(
        self,
        *,
        hf_stream: Any,
        torch_module: Any,
        dtype: Any,
        model: Any,
        processor: Any,
        loader_key: str,
    ) -> dict[str, Any]:
        enabled = bool(getattr(self.args, "edgetam_prewarm_compile", False))
        runs = max(0, int(getattr(self.args, "edgetam_prewarm_runs", 0)))
        if not enabled or runs <= 0:
            return {"enabled": False, "runs": runs}
        total_start_s = time.perf_counter()
        image_bgr = np.zeros((int(self.height), int(self.width), 3), dtype=np.uint8)
        image = _bgr_to_pil_rgb(image_bgr)
        obj_ids, masks = self._dummy_prewarm_prompt_masks()
        session_start_s = time.perf_counter()
        session = hf_stream.EdgeTamVideoInferenceSession(
            video=None,
            video_height=int(self.height),
            video_width=int(self.width),
            inference_device=self.args.device,
            inference_state_device=self.args.device,
            video_storage_device=self.args.device,
            dtype=dtype,
        )
        session_ms = _elapsed_ms(session_start_s, time.perf_counter())
        with torch_module.inference_mode():
            inputs, preprocess_ms, preprocess_pre_sync_ms, preprocess_post_sync_ms = _time_runtime_ms(
                torch_module,
                self.args.device,
                lambda: processor(images=image, device=self.args.device, return_tensors="pt"),
                sync_enabled=True,
            )
            pixel_values = inputs.pixel_values[0].to(device=self.args.device, dtype=dtype)
            prompt_ms = 0.0
            model_runs: list[dict[str, float]] = []
            with self._autocast_context(torch_module):
                _, prompt_ms, prompt_pre_sync_ms, prompt_post_sync_ms = _time_runtime_ms(
                    torch_module,
                    self.args.device,
                    lambda: processor.add_inputs_to_inference_session(
                        inference_session=session,
                        frame_idx=0,
                        obj_ids=obj_ids,
                        input_masks=masks,
                    ),
                    sync_enabled=True,
                )
                for run_idx in range(runs):
                    mark_torch_cudagraph_step_begin(torch_module)
                    _, wall_model_ms, cuda_event_model_ms, pre_sync_ms, post_sync_ms = _time_model_forward(
                        torch_module=torch_module,
                        device=self.args.device,
                        profile_sync=True,
                        profile_cuda_events=bool(self.args.profile_cuda_events),
                        fn=lambda: model(inference_session=session, frame=pixel_values),
                    )
                    model_runs.append(
                        {
                            "run_idx": float(run_idx),
                            "wall_model_ms": float(wall_model_ms),
                            "cuda_event_model_ms": float(cuda_event_model_ms),
                            "pre_sync_ms": float(pre_sync_ms),
                            "post_sync_ms": float(post_sync_ms),
                            "total_ms": float(pre_sync_ms + wall_model_ms + post_sync_ms),
                        }
                    )
        model_total_ms = float(sum(run["total_ms"] for run in model_runs))
        batch_vision_profile: dict[str, Any] | None = None
        if bool(getattr(self.args, "edgetam_batch_vision_encoder", False)):
            batch_size = len(tuple(self.args.camera_ids))
            batch_started_s = time.perf_counter()
            batch_inputs, batch_preprocess_ms, batch_pre_sync_ms, batch_post_sync_ms = _time_runtime_ms(
                torch_module,
                self.args.device,
                lambda: processor(images=[image] * batch_size, device=self.args.device, return_tensors="pt"),
                sync_enabled=True,
            )
            batch_pixel_values = batch_inputs.pixel_values.to(device=self.args.device, dtype=dtype)
            with torch_module.inference_mode():
                with self._autocast_context(torch_module):
                    mark_torch_cudagraph_step_begin(torch_module)
                    _, batch_wall_ms, batch_cuda_event_ms, batch_model_pre_sync_ms, batch_model_post_sync_ms = (
                        _time_model_forward(
                            torch_module=torch_module,
                            device=self.args.device,
                            profile_sync=True,
                            profile_cuda_events=bool(self.args.profile_cuda_events),
                            fn=lambda: model.get_image_features(batch_pixel_values, return_dict=True),
                        )
                    )
            batch_vision_profile = {
                "enabled": True,
                "batch_size": int(batch_size),
                "preprocess_ms": float(batch_preprocess_ms),
                "preprocess_pre_sync_ms": float(batch_pre_sync_ms),
                "preprocess_post_sync_ms": float(batch_post_sync_ms),
                "wall_model_ms": float(batch_wall_ms),
                "cuda_event_model_ms": float(batch_cuda_event_ms),
                "model_pre_sync_ms": float(batch_model_pre_sync_ms),
                "model_post_sync_ms": float(batch_model_post_sync_ms),
                "total_ms": float(_elapsed_ms(batch_started_s, time.perf_counter())),
            }
        total_ms = _elapsed_ms(total_start_s, time.perf_counter())
        if self.args.debug:
            first_run = model_runs[0]["total_ms"] if model_runs else 0.0
            print(
                "[demo2.1-edgetam-prewarm] "
                f"loader={loader_key} runs={runs} total_ms={total_ms:.2f} "
                f"first_model_total_ms={first_run:.2f} "
                f"batch_vision_ms={(batch_vision_profile or {}).get('total_ms', 0.0):.2f}",
                flush=True,
            )
        result = {
            "enabled": True,
            "runs": int(runs),
            "session_ms": float(session_ms),
            "preprocess_ms": float(preprocess_ms),
            "preprocess_pre_sync_ms": float(preprocess_pre_sync_ms),
            "preprocess_post_sync_ms": float(preprocess_post_sync_ms),
            "prompt_ms": float(prompt_ms),
            "prompt_pre_sync_ms": float(prompt_pre_sync_ms),
            "prompt_post_sync_ms": float(prompt_post_sync_ms),
            "model_total_ms": float(model_total_ms),
            "model_runs": model_runs,
            "total_ms": float(total_ms),
        }
        if batch_vision_profile is not None:
            result["batch_vision"] = batch_vision_profile
        return result

    def _run_edgetam_frame(
        self,
        *,
        torch_module: Any,
        dtype: Any,
        model: Any,
        processor: Any,
        session: Any,
        pixel_stager: PinnedPixelValueStager | None,
        stream: Any | None,
        frame: CameraFramePacket,
        initial_controller_mask: np.ndarray,
        initial_object_mask: np.ndarray,
        add_prompt: bool,
        prepared_frame: PreparedEdgeTamFrame | None = None,
    ) -> CameraMaskPacket:
        frame_started_s = time.perf_counter()
        profile_sync = bool(getattr(self.args, "profile_sync", False))
        nvtx_enabled = bool(getattr(self.args, "profile_nsys_markers", False))
        mask_postprocess_mode = str(getattr(self.args, "mask_postprocess", MASK_POSTPROCESS_HF))
        if prepared_frame is not None:
            inputs_original_sizes = prepared_frame.original_sizes
            pixel_values = prepared_frame.pixel_values
            preprocess_ms = float(prepared_frame.preprocess_ms)
            edge_h2d_profile = dict(prepared_frame.edge_h2d_profile)
            frame_idx = int(prepared_frame.frame_idx)
        else:
            with torch_nvtx_range(torch_module, nvtx_enabled, f"edgetam_preprocess_cam{int(frame.camera_idx)}"):
                image = _bgr_to_pil_rgb(frame.color_bgr)
            frame_idx = -1
            if pixel_stager is not None:
                with torch_nvtx_range(torch_module, nvtx_enabled, f"edgetam_processor_cam{int(frame.camera_idx)}"):
                    inputs, preprocess_ms, _, _ = _time_runtime_ms(
                        torch_module,
                        self.args.device,
                        lambda: processor(images=image, return_tensors="pt"),
                        sync_enabled=profile_sync,
                    )
                with torch_nvtx_range(torch_module, nvtx_enabled, f"edgetam_h2d_cam{int(frame.camera_idx)}"):
                    pixel_values, edge_h2d_profile = pixel_stager.stage(
                        inputs.pixel_values[0],
                        dtype=dtype,
                        consumer_stream=stream,
                    )
            else:
                with torch_nvtx_range(torch_module, nvtx_enabled, f"edgetam_processor_cam{int(frame.camera_idx)}"):
                    inputs, preprocess_ms, _, _ = _time_runtime_ms(
                        torch_module,
                        self.args.device,
                        lambda: processor(images=image, device=self.args.device, return_tensors="pt"),
                        sync_enabled=profile_sync,
                    )
                with torch_nvtx_range(torch_module, nvtx_enabled, f"edgetam_h2d_cam{int(frame.camera_idx)}"):
                    h2d_start_s = time.perf_counter()
                    pixel_values = inputs.pixel_values[0].to(device=self.args.device, dtype=dtype)
                    h2d_enqueue_ms = _elapsed_ms(h2d_start_s, time.perf_counter())
                edge_h2d_profile = {
                    "pin_memory": False,
                    "processor_device": str(inputs.pixel_values.device),
                    "processor_is_pinned": bool(inputs.pixel_values.is_pinned()) if hasattr(inputs.pixel_values, "is_pinned") else False,
                    "pin_copy_ms": 0.0,
                    "slot_reuse_wait_ms": 0.0,
                    "h2d_enqueue_ms": float(h2d_enqueue_ms),
                    "h2d_wait_ms": 0.0,
                    "h2d_stream_mode": H2D_STREAM_MODE_DEFAULT,
                }
            inputs_original_sizes = inputs.original_sizes
        prompt_ms = 0.0
        stream_context = (
            torch_module.cuda.stream(stream)
            if stream is not None and str(self.args.device).startswith("cuda")
            else nullcontext()
        )
        first_compiled_capture_pending = (
            serialized_edgetam_first_compiled_forward_enabled(self.args)
            and len(self._edgetam_first_compiled_forward_done) < len(tuple(self.args.camera_ids))
        )
        capture_context = self._edgetam_first_compiled_forward_lock if first_compiled_capture_pending else nullcontext()
        with capture_context:
            serialized_capture_active = (
                serialized_edgetam_first_compiled_forward_enabled(self.args)
                and len(self._edgetam_first_compiled_forward_done) < len(tuple(self.args.camera_ids))
            )
            with stream_context:
                with self._autocast_context(torch_module):
                    if add_prompt:
                        prompt_obj_ids: list[int] = []
                        prompt_masks: list[np.ndarray] = []
                        if controller_tracking_enabled(self.args.track_mode):
                            prompt_obj_ids.append(CONTROLLER_ID)
                            prompt_masks.append(np.asarray(initial_controller_mask, dtype=bool))
                        if object_tracking_enabled(self.args.track_mode):
                            prompt_obj_ids.append(OBJECT_ID)
                            prompt_masks.append(np.asarray(initial_object_mask, dtype=bool))
                        prompt_frame_idx = frame_idx if frame_idx >= 0 else 0
                        with torch_nvtx_range(torch_module, nvtx_enabled, f"edgetam_prompt_cam{int(frame.camera_idx)}"):
                            _, prompt_ms, _, _ = _time_runtime_ms(
                                torch_module,
                                self.args.device,
                                lambda: processor.add_inputs_to_inference_session(
                                    inference_session=session,
                                    frame_idx=prompt_frame_idx,
                                    obj_ids=prompt_obj_ids,
                                    input_masks=prompt_masks,
                                ),
                                sync_enabled=profile_sync,
                            )
                    gate_key = f"edgetam_cam{int(frame.camera_idx)}"
                    with self.gpu_gate.acquire(stage="edgetam", camera_idx=frame.camera_idx, group_id=frame.group_id) as gate_wait_ms:
                        self._record_gpu_gate_wait(gate_key, gate_wait_ms)
                        if prepared_frame is not None:
                            forward = lambda: model(
                                inference_session=session,
                                frame_idx=frame_idx,
                                frame=pixel_values,
                            )
                        else:
                            forward = lambda: model(inference_session=session, frame=pixel_values)
                        mark_torch_cudagraph_step_begin(torch_module)
                        with torch_nvtx_range(torch_module, nvtx_enabled, f"edgetam_model_cam{int(frame.camera_idx)}"):
                            output, wall_model_ms, cuda_event_model_ms, model_pre_sync_ms, model_post_sync_ms = _time_model_forward(
                                torch_module=torch_module,
                                device=self.args.device,
                                profile_sync=profile_sync,
                                profile_cuda_events=bool(self.args.profile_cuda_events),
                                fn=forward,
                            )
                    if serialized_capture_active and add_prompt:
                        self._edgetam_first_compiled_forward_done.add(int(frame.camera_idx))
            if stream is not None and str(self.args.device).startswith("cuda"):
                done_event = torch_module.cuda.Event()
                done_event.record(stream)
                done_event.synchronize()
            if prepared_frame is None and pixel_stager is not None:
                pixel_stager.mark_consumed(int(edge_h2d_profile.get("pinned_slot_idx", -1)), stream)
        mask_resize_ms = 0.0
        mask_threshold_ms = 0.0
        mask_to_cpu_ms = 0.0
        if mask_postprocess_mode == MASK_POSTPROCESS_CUDA_INLINE:
            with torch_nvtx_range(torch_module, nvtx_enabled, f"edgetam_mask_cuda_inline_cam{int(frame.camera_idx)}"):
                masks_by_id, inline_profile = extract_object_masks_from_hf_output_cuda_inline(
                    torch_module=torch_module,
                    output=output,
                    height=int(frame.color_bgr.shape[0]),
                    width=int(frame.color_bgr.shape[1]),
                )
            mask_resize_ms = float(inline_profile.get("mask_resize_ms", 0.0))
            mask_threshold_ms = float(inline_profile.get("mask_threshold_ms", 0.0))
            mask_to_cpu_ms = float(inline_profile.get("mask_to_cpu_ms", 0.0))
            postprocess_ms = float(mask_resize_ms + mask_threshold_ms)
        else:
            with torch_nvtx_range(torch_module, nvtx_enabled, f"edgetam_mask_hf_postprocess_cam{int(frame.camera_idx)}"):
                post_masks, postprocess_ms, _, _ = _time_runtime_ms(
                    torch_module,
                    self.args.device,
                    lambda: processor.post_process_masks(
                        [output.pred_masks],
                        original_sizes=inputs_original_sizes,
                        binarize=False,
                    )[0],
                    sync_enabled=profile_sync,
                )
            with torch_nvtx_range(torch_module, nvtx_enabled, f"edgetam_mask_to_cpu_cam{int(frame.camera_idx)}"):
                masks_by_id, mask_to_cpu_ms, _, _ = _time_runtime_ms(
                    torch_module,
                    self.args.device,
                    lambda: extract_object_masks_from_hf_output(output, post_masks),
                    sync_enabled=False,
                )
        missing = [obj_id for obj_id in active_object_ids(self.args) if obj_id not in masks_by_id]
        if missing:
            raise RuntimeError(f"HF output missing tracked object ids for cam{frame.camera_idx}: {missing}")
        reference_mask = next(iter(masks_by_id.values()))
        object_mask = masks_by_id.get(OBJECT_ID)
        if object_mask is None:
            object_mask = np.zeros_like(reference_mask, dtype=bool)
        controller_mask = masks_by_id.get(CONTROLLER_ID)
        if controller_mask is None:
            controller_mask = np.zeros_like(reference_mask, dtype=bool)
        total_ms = _elapsed_ms(frame_started_s, time.perf_counter())
        if add_prompt:
            cam_key = f"cam{int(frame.camera_idx)}"
            self._init_profile_update(
                ("edgetam", "sessions", cam_key),
                {
                    "prompt_add_ms": float(prompt_ms),
                    "first_forward_wall_model_ms": float(wall_model_ms),
                    "first_forward_cuda_event_ms": float(cuda_event_model_ms),
                    "first_forward_total_ms": float(total_ms),
                    "first_forward_group_id": int(frame.group_id),
                },
            )
            self._init_profile_add(("edgetam", "prompt_add_ms_total"), float(prompt_ms))
            self._init_profile_add(("edgetam", "first_forward_total_ms_sum"), float(total_ms))
            self._init_profile_add(("edgetam", "first_forward_model_ms_sum"), float(wall_model_ms))
            session_init_ms = float(
                _nested_get(self._init_profile_snapshot(), ("edgetam", "sessions", cam_key, "session_init_ms"))
                or 0.0
            )
            self._init_profile_set(
                ("edgetam", "session_init_plus_prompt_ms_total"),
                float(_nested_get(self._init_profile_snapshot(), ("edgetam", "session_init_ms_total")) or 0.0)
                + float(_nested_get(self._init_profile_snapshot(), ("edgetam", "prompt_add_ms_total")) or 0.0),
            )
            self._init_profile_update(
                ("edgetam", "sessions", cam_key),
                {"session_init_plus_prompt_ms": float(session_init_ms + prompt_ms)},
            )
        self._profile_update(
            frame.group_id,
            edgetam={
                f"cam{int(frame.camera_idx)}": {
                    "gate_wait_ms": float(gate_wait_ms),
                    "model_ms": float(cuda_event_model_ms or wall_model_ms),
                    "wall_model_ms": float(wall_model_ms),
                    "cuda_event_model_ms": float(cuda_event_model_ms),
                    "model_pre_sync_ms": float(model_pre_sync_ms),
                    "model_post_sync_ms": float(model_post_sync_ms),
                    "preprocess_ms": float(preprocess_ms),
                    "h2d_pin_copy_ms": float(edge_h2d_profile.get("pin_copy_ms", 0.0)),
                    "h2d_enqueue_ms": float(edge_h2d_profile.get("h2d_enqueue_ms", 0.0)),
                    "h2d_wait_ms": float(edge_h2d_profile.get("h2d_wait_ms", 0.0)),
                    "prompt_ms": float(prompt_ms),
                    "postprocess_ms": float(postprocess_ms),
                    "mask_resize_ms": float(mask_resize_ms),
                    "mask_threshold_ms": float(mask_threshold_ms),
                    "mask_to_cpu_ms": float(mask_to_cpu_ms),
                    "mask_postprocess": mask_postprocess_mode,
                    "total_ms": float(total_ms),
                    "stream_mode": str(self.args.edgetam_stream_mode),
                    "batch_vision_encoder": prepared_frame is not None,
                    "batch_vision_encoder_ms": (
                        float(prepared_frame.batch_vision_encoder_ms) if prepared_frame is not None else 0.0
                    ),
                    "frame_idx": int(frame_idx),
                    "publish_s": self._profile_rel_s(),
                }
            },
            h2d={
                f"cam{int(frame.camera_idx)}": {
                    "edge": {
                        **edge_h2d_profile,
                        "profile_enabled": bool(self.args.profile_h2d),
                    }
                }
            },
        )
        return CameraMaskPacket(
            group_id=frame.group_id,
            camera_idx=frame.camera_idx,
            color_bgr=frame.color_bgr,
            controller_mask=controller_mask,
            object_mask=object_mask,
            model_ms=wall_model_ms,
            cuda_event_model_ms=cuda_event_model_ms,
            mask_ms=float(preprocess_ms + prompt_ms + wall_model_ms + postprocess_ms + mask_to_cpu_ms),
            gpu_gate_wait_ms=gate_wait_ms,
        )

    def _edgetam_camera_worker(self, camera_idx: int) -> None:
        try:
            hf_stream, torch_module, dtype, model, processor = self._init_hf_model(camera_idx)
            pixel_stager = (
                PinnedPixelValueStager(
                    torch_module=torch_module,
                    device=str(self.args.device),
                    ring_size=int(self.args.pinned_ring_size),
                    h2d_stream_mode=str(self.args.h2d_stream_mode),
                    verify_copies=bool(self.args.debug or self.args.profile_h2d),
                )
                if edge_pin_memory_enabled(self.args)
                else None
            )
            last_group_id = -1
            initialized = False
            init_attempts = 0
            last_init_failure_log_s = 0.0
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
                    if not temporal_group_is_coherent(group, max_capture_skew_ms=float(self.args.max_capture_skew_ms)):
                        self._profile_mark_drop(group.group_id, f"edgetam_drop_skewed_group_cam{int(camera_idx)}")
                        continue
                    frame = group.frames[int(camera_idx)]
                    if not initialized:
                        init_attempts += 1
                        try:
                            self._consume_sam31_preload_if_ready()
                            controller_mask, object_mask = resolve_initial_masks_for_camera(
                                frame,
                                self.args,
                                sam31_lock=self._sam31_lock,
                            )
                        except Exception as exc:
                            key = f"sam31_init_failures_cam{int(camera_idx)}"
                            self._summary[key] = int(self._summary.get(key, 0)) + 1
                            self._profile_mark_drop(group.group_id, f"sam31_init_failed_cam{int(camera_idx)}")
                            now_s = time.perf_counter()
                            max_attempts = int(self.args.sam31_init_max_attempts)
                            will_retry = max_attempts == 0 or init_attempts < max_attempts
                            if self.args.debug and now_s - last_init_failure_log_s >= 2.0:
                                action = "retrying on the latest live frame" if will_retry else "failing without fallback"
                                print(
                                    "[demo2.1-sam31-init] "
                                    f"cam={camera_idx} attempt={init_attempts} group={group.group_id} "
                                    f"failed={type(exc).__name__}: {exc}. "
                                    f"Keep the target visible and steady; {action}.",
                                    flush=True,
                                )
                                last_init_failure_log_s = now_s
                            if not will_retry:
                                raise RuntimeError(
                                    f"SAM3.1 live initialization failed for cam{camera_idx} "
                                    f"after {init_attempts} attempt(s); no fallback is allowed"
                                ) from exc
                            time.sleep(float(self.args.sam31_init_retry_interval_s))
                            continue
                        self._summary[f"sam31_init_attempts_cam{int(camera_idx)}"] = int(init_attempts)
                        if self.args.debug:
                            print(
                                "[demo2.1-sam31-init] "
                                f"cam={camera_idx} initialized from live frame group={group.group_id} "
                                f"attempts={init_attempts} object_px={int(np.count_nonzero(object_mask))} "
                                f"controller_px={int(np.count_nonzero(controller_mask))}",
                                flush=True,
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
                        pixel_stager=pixel_stager,
                        stream=None,
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
            self._mark_fatal_error(f"edgetam-cam{int(camera_idx)}", exc)
            self.stop_event.set()

    def _init_gpu_owner_edgetam_states(self) -> dict[int, dict[str, Any]]:
        states: dict[int, dict[str, Any]] = {}
        shared_bundle: tuple[Any, Any, Any, Any, Any] | None = None
        if self.args.edgetam_model_topology == EDGETAM_MODEL_TOPOLOGY_SHARED:
            shared_bundle = self._init_hf_model(-1)
        for camera_idx in self.args.camera_ids:
            hf_stream, torch_module, dtype, model, processor = (
                shared_bundle if shared_bundle is not None else self._init_hf_model(int(camera_idx))
            )
            pixel_stager = (
                PinnedPixelValueStager(
                    torch_module=torch_module,
                    device=str(self.args.device),
                    ring_size=int(self.args.pinned_ring_size),
                    h2d_stream_mode=str(self.args.h2d_stream_mode),
                    verify_copies=bool(self.args.debug or self.args.profile_h2d),
                )
                if edge_pin_memory_enabled(self.args)
                else None
            )
            stream = (
                torch_module.cuda.Stream()
                if str(self.args.edgetam_stream_mode) == EDGETAM_STREAM_MODE_PER_CAMERA
                and str(self.args.device).startswith("cuda")
                and torch_module.cuda.is_available()
                else None
            )
            states[int(camera_idx)] = {
                "hf_stream": hf_stream,
                "torch_module": torch_module,
                "dtype": dtype,
                "model": model,
                "processor": processor,
                "pixel_stager": pixel_stager,
                "initialized": False,
                "init_attempts": 0,
                "last_init_failure_log_s": 0.0,
                "controller_mask": None,
                "object_mask": None,
                "session": None,
                "stream": stream,
            }
        return states

    def _get_or_init_gpu_owner_edgetam_states(self) -> dict[int, dict[str, Any]]:
        states = self._consume_parallel_init_future("edgetam_states")
        if states is not None:
            return states
        return self._init_gpu_owner_edgetam_states()

    def _ensure_gpu_owner_edgetam_initialized(
        self,
        *,
        state: dict[str, Any],
        camera_idx: int,
        frame: CameraFramePacket,
    ) -> bool:
        if bool(state["initialized"]):
            return True
        state["init_attempts"] = int(state["init_attempts"]) + 1
        sam31_start_s = time.perf_counter()
        try:
            self._consume_sam31_preload_if_ready()
            controller_mask, object_mask = resolve_initial_masks_for_camera(
                frame,
                self.args,
                sam31_lock=self._sam31_lock,
            )
            sam31_call_ms = _elapsed_ms(sam31_start_s, time.perf_counter())
        except Exception as exc:
            key = f"sam31_init_failures_cam{int(camera_idx)}"
            self._summary[key] = int(self._summary.get(key, 0)) + 1
            self._profile_mark_drop(frame.group_id, f"sam31_init_failed_cam{int(camera_idx)}")
            now_s = time.perf_counter()
            max_attempts = int(self.args.sam31_init_max_attempts)
            will_retry = max_attempts == 0 or int(state["init_attempts"]) < max_attempts
            if self.args.debug and now_s - float(state["last_init_failure_log_s"]) >= 2.0:
                action = "retrying on the latest live frame" if will_retry else "failing without fallback"
                print(
                    "[demo2.1-sam31-init] "
                    f"single-owner cam={camera_idx} attempt={int(state['init_attempts'])} group={frame.group_id} "
                    f"failed={type(exc).__name__}: {exc}. "
                    f"Keep the target visible and steady; {action}.",
                    flush=True,
                )
                state["last_init_failure_log_s"] = now_s
            if not will_retry:
                raise RuntimeError(
                    f"SAM3.1 live initialization failed for cam{camera_idx} "
                    f"after {int(state['init_attempts'])} attempt(s); no fallback is allowed"
                ) from exc
            time.sleep(float(self.args.sam31_init_retry_interval_s))
            return False

        state["controller_mask"] = controller_mask
        state["object_mask"] = object_mask
        sam31_timing = dict(getattr(self.args, "_sam31_last_timing_ms", {}) or {})
        sam31_timing.setdefault("total_ms", float(sam31_call_ms))
        cam_key = f"cam{int(camera_idx)}"
        self._init_profile_update(
            ("sam31", cam_key),
            {
                **sam31_timing,
                "call_wall_ms": float(sam31_call_ms),
                "object_pixels": int(np.count_nonzero(object_mask)),
                "controller_pixels": int(np.count_nonzero(controller_mask)),
                "group_id": int(frame.group_id),
                "attempts": int(state["init_attempts"]),
            },
        )
        self._init_profile_add(("sam31", "model_load_ms_total"), float(sam31_timing.get("model_load_ms", 0.0) or 0.0))
        self._init_profile_add(("sam31", "segment_total_ms_total"), float(sam31_timing.get("total_ms", sam31_call_ms) or 0.0))
        self._summary[f"sam31_init_attempts_cam{int(camera_idx)}"] = int(state["init_attempts"])
        if self.args.debug:
            print(
                "[demo2.1-sam31-init] "
                f"single-owner cam={camera_idx} initialized from live frame group={frame.group_id} "
                f"attempts={int(state['init_attempts'])} object_px={int(np.count_nonzero(object_mask))} "
                f"controller_px={int(np.count_nonzero(controller_mask))}",
                flush=True,
            )
        session_start_s = time.perf_counter()
        state["session"] = state["hf_stream"].EdgeTamVideoInferenceSession(
            video=None,
            video_height=int(frame.color_bgr.shape[0]),
            video_width=int(frame.color_bgr.shape[1]),
            inference_device=self.args.device,
            inference_state_device=self.args.device,
            video_storage_device=self.args.device,
            dtype=state["dtype"],
        )
        session_init_ms = _elapsed_ms(session_start_s, time.perf_counter())
        self._init_profile_update(
            ("edgetam", "sessions", cam_key),
            {
                "session_init_ms": float(session_init_ms),
                "init_group_id": int(frame.group_id),
            },
        )
        self._init_profile_add(("edgetam", "session_init_ms_total"), float(session_init_ms))
        state["initialized"] = True
        return True

    def _prepare_edgetam_batch_vision_frames(
        self,
        *,
        states: dict[int, dict[str, Any]],
        group: CaptureGroup,
    ) -> dict[int, PreparedEdgeTamFrame]:
        camera_ids = [int(camera_idx) for camera_idx in self.args.camera_ids]
        first_state = states[camera_ids[0]]
        torch_module = first_state["torch_module"]
        dtype = first_state["dtype"]
        model = first_state["model"]
        processor = first_state["processor"]
        started_s = time.perf_counter()
        images = [_bgr_to_pil_rgb(group.frames[idx].color_bgr) for idx in camera_ids]
        inputs, preprocess_ms, _, _ = _time_runtime_ms(
            torch_module,
            self.args.device,
            lambda: processor(images=images, device=self.args.device, return_tensors="pt"),
            sync_enabled=False,
        )
        pixel_values_batch = inputs.pixel_values.to(device=self.args.device, dtype=dtype)
        frame_indices: dict[int, int] = {}
        for pos, camera_idx in enumerate(camera_ids):
            state = states[camera_idx]
            session = state["session"]
            frame_indices[camera_idx] = int(session.add_new_frame(pixel_values_batch[pos], frame_idx=None))

        with torch_module.inference_mode():
            with self._autocast_context(torch_module):
                mark_torch_cudagraph_step_begin(torch_module)
                image_outputs, wall_model_ms, cuda_event_model_ms, _, _ = _time_model_forward(
                    torch_module=torch_module,
                    device=self.args.device,
                    profile_sync=False,
                    profile_cuda_events=bool(self.args.profile_cuda_events),
                    fn=lambda: model.get_image_features(pixel_values_batch, return_dict=True),
                )

        for pos, camera_idx in enumerate(camera_ids):
            states[camera_idx]["session"].cache.cache_vision_features(
                frame_indices[camera_idx],
                split_hf_vision_features_for_session(image_outputs, pos),
            )

        total_ms = _elapsed_ms(started_s, time.perf_counter())
        per_camera_preprocess_ms = float(preprocess_ms) / float(max(1, len(camera_ids)))
        edge_h2d_profile = {
            "pin_memory": False,
            "processor_device": str(inputs.pixel_values.device),
            "processor_is_pinned": bool(inputs.pixel_values.is_pinned()) if hasattr(inputs.pixel_values, "is_pinned") else False,
            "pin_copy_ms": 0.0,
            "slot_reuse_wait_ms": 0.0,
            "h2d_enqueue_ms": 0.0,
            "h2d_wait_ms": 0.0,
            "h2d_stream_mode": H2D_STREAM_MODE_DEFAULT,
            "batch_vision_encoder": True,
            "batch_size": int(len(camera_ids)),
        }
        self._profile_update(
            group.group_id,
            edgetam={
                "batch_vision": {
                    "enabled": True,
                    "batch_size": int(len(camera_ids)),
                    "preprocess_ms": float(preprocess_ms),
                    "wall_model_ms": float(wall_model_ms),
                    "cuda_event_model_ms": float(cuda_event_model_ms),
                    "model_ms": float(cuda_event_model_ms or wall_model_ms),
                    "total_ms": float(total_ms),
                    "publish_s": self._profile_rel_s(),
                }
            },
        )
        return {
            camera_idx: PreparedEdgeTamFrame(
                pixel_values=pixel_values_batch[pos],
                original_sizes=slice_hf_original_sizes(inputs.original_sizes, pos),
                frame_idx=frame_indices[camera_idx],
                preprocess_ms=per_camera_preprocess_ms,
                edge_h2d_profile=edge_h2d_profile,
                batch_vision_encoder_ms=float(cuda_event_model_ms or wall_model_ms),
            )
            for pos, camera_idx in enumerate(camera_ids)
        }

    def _run_gpu_owner_edgetam_cycle(
        self,
        *,
        states: dict[int, dict[str, Any]],
        group: CaptureGroup,
    ) -> tuple[dict[int, CameraMaskPacket] | None, float]:
        cycle_start_s = time.perf_counter()
        packets: dict[int, CameraMaskPacket] = {}
        initialized_before: dict[int, bool] = {}
        prepared_frames: dict[int, PreparedEdgeTamFrame] = {}
        if bool(getattr(self.args, "edgetam_batch_vision_encoder", False)):
            for camera_idx in self.args.camera_ids:
                idx = int(camera_idx)
                frame = group.frames[idx]
                state = states[idx]
                initialized_before[idx] = bool(state["initialized"])
                if not self._ensure_gpu_owner_edgetam_initialized(state=state, camera_idx=idx, frame=frame):
                    return None, _elapsed_ms(cycle_start_s, time.perf_counter())
            prepared_frames = self._prepare_edgetam_batch_vision_frames(states=states, group=group)
        for camera_idx in self.args.camera_ids:
            idx = int(camera_idx)
            frame = group.frames[idx]
            state = states[idx]
            if bool(getattr(self.args, "edgetam_batch_vision_encoder", False)):
                was_initialized = initialized_before[idx]
                prepared_frame = prepared_frames[idx]
            else:
                was_initialized = bool(state["initialized"])
                if not self._ensure_gpu_owner_edgetam_initialized(state=state, camera_idx=idx, frame=frame):
                    return None, _elapsed_ms(cycle_start_s, time.perf_counter())
                prepared_frame = None
            torch_module = state["torch_module"]
            with torch_module.inference_mode():
                packet = self._run_edgetam_frame(
                    torch_module=torch_module,
                    dtype=state["dtype"],
                    model=state["model"],
                    processor=state["processor"],
                    session=state["session"],
                    pixel_stager=state["pixel_stager"],
                    stream=state.get("stream"),
                    frame=frame,
                    initial_controller_mask=state["controller_mask"],
                    initial_object_mask=state["object_mask"],
                    add_prompt=not was_initialized,
                    prepared_frame=prepared_frame,
                )
            packets[idx] = packet
            self.mask_slots[idx].put(packet)
            self.edge_stats[idx].record()
        if (
            bool(getattr(self.args, "sam31_keep_runtime_until_all_cameras_init", False))
            and not self._sam31_runtime_released_after_init
            and all(bool(state["initialized"]) for state in states.values())
        ):
            release_start_s = time.perf_counter()
            release_ms = release_sam31_runtime_resources(str(self.args.device))
            self._sam31_runtime_released_after_init = True
            self._summary["sam31_runtime_released_after_all_cameras_init"] = True
            self._init_profile_set(
                ("sam31", "release_cleanup_ms"),
                float(release_ms if release_ms is not None else _elapsed_ms(release_start_s, time.perf_counter())),
            )
        return packets, _elapsed_ms(cycle_start_s, time.perf_counter())

    def _run_staged_edgetam_cycle_parallel(
        self,
        *,
        states: dict[int, dict[str, Any]],
        group: CaptureGroup,
        executor: ThreadPoolExecutor,
    ) -> tuple[dict[int, CameraMaskPacket] | None, float, float]:
        cycle_start_s = time.perf_counter()

        def run_one(camera_idx: int) -> tuple[int, CameraMaskPacket] | None:
            frame = group.frames[int(camera_idx)]
            state = states[int(camera_idx)]
            was_initialized = bool(state["initialized"])
            if not self._ensure_gpu_owner_edgetam_initialized(state=state, camera_idx=int(camera_idx), frame=frame):
                return None
            torch_module = state["torch_module"]
            with torch_module.inference_mode():
                packet = self._run_edgetam_frame(
                    torch_module=torch_module,
                    dtype=state["dtype"],
                    model=state["model"],
                    processor=state["processor"],
                    session=state["session"],
                    pixel_stager=state["pixel_stager"],
                    stream=state.get("stream"),
                    frame=frame,
                    initial_controller_mask=state["controller_mask"],
                    initial_object_mask=state["object_mask"],
                    add_prompt=not was_initialized,
                )
            self.mask_slots[int(camera_idx)].put(packet)
            self.edge_stats[int(camera_idx)].record()
            return int(camera_idx), packet

        futures = {
            executor.submit(run_one, int(camera_idx)): int(camera_idx)
            for camera_idx in self.args.camera_ids
        }
        packets: dict[int, CameraMaskPacket] = {}
        for future in as_completed(futures):
            result = future.result()
            if result is None:
                return None, _elapsed_ms(cycle_start_s, time.perf_counter()), 0.0
            camera_idx, packet = result
            packets[int(camera_idx)] = packet
        sum_model_ms = sum(
            float(packet.cuda_event_model_ms or packet.model_ms)
            for packet in packets.values()
        )
        return packets, _elapsed_ms(cycle_start_s, time.perf_counter()), float(sum_model_ms)

    def _gpu_owner_pipeline_worker(self) -> None:
        try:
            runner = self._get_or_prepare_ffs_runner() if self.args.depth_source == DEPTH_SOURCE_FFS else None
            aligners: dict[int, FfsIrToColorAligner] = {}
            edgetam_states = self._get_or_init_gpu_owner_edgetam_states()
            last_group_id = -1
            while not self.stop_event.is_set():
                group = self.capture_group_slot.get_latest_after(last_group_id)
                if group is None:
                    time.sleep(0.001)
                    continue
                last_group_id = group.group_id
                if not temporal_group_is_coherent(group, max_capture_skew_ms=float(self.args.max_capture_skew_ms)):
                    self._summary["gpu_owner_drop_skewed_capture_group"] = int(
                        self._summary.get("gpu_owner_drop_skewed_capture_group", 0)
                    ) + 1
                    self._profile_mark_drop(group.group_id, "gpu_owner_drop_skewed_capture_group")
                    continue

                owner_start_s = time.perf_counter()
                depth_group: DepthGroup | None = None
                mask_packets: dict[int, CameraMaskPacket] | None = None
                ffs_cycle_ms = 0.0
                edgetam_cycle_ms = 0.0
                order = str(self.args.single_owner_order)
                if order == SINGLE_OWNER_ORDER_EDGETAM_THEN_FFS:
                    mask_packets, edgetam_cycle_ms = self._run_gpu_owner_edgetam_cycle(
                        states=edgetam_states,
                        group=group,
                    )
                    if mask_packets is None:
                        continue
                    depth_group, _ = self._run_depth_cycle_for_group(
                        group=group,
                        runner=runner,
                        aligners=aligners,
                    )
                    ffs_cycle_ms = depth_group.total_ms
                else:
                    depth_group, _ = self._run_depth_cycle_for_group(
                        group=group,
                        runner=runner,
                        aligners=aligners,
                    )
                    ffs_cycle_ms = depth_group.total_ms
                    mask_packets, edgetam_cycle_ms = self._run_gpu_owner_edgetam_cycle(
                        states=edgetam_states,
                        group=group,
                    )
                    if mask_packets is None:
                        continue

                total_ms = _elapsed_ms(owner_start_s, time.perf_counter())
                packet = CompleteInferenceGroup(
                    group_id=group.group_id,
                    capture_group=group,
                    depth_group=depth_group,
                    mask_packets=mask_packets,
                    ffs_cycle_ms=float(ffs_cycle_ms),
                    edgetam_cycle_ms=float(edgetam_cycle_ms),
                    edgetam_stage_wall_ms=float(edgetam_cycle_ms),
                    edgetam_stage_sum_model_ms=sum(
                        float(packet.cuda_event_model_ms or packet.model_ms)
                        for packet in mask_packets.values()
                    ),
                    stage_barrier_ms=0.0,
                    total_gpu_owner_ms=float(total_ms),
                    pipeline_mode=GPU_PIPELINE_MODE_SINGLE_OWNER,
                    internal_order=order,
                )
                self.complete_inference_slot.put(packet)
                self._init_profile_set_once(("first_complete_inference_group_s",), self._profile_rel_s())
                self._init_profile_set_once(("first_complete_inference_group_id",), int(group.group_id))
                self._latest_depth_group = depth_group
                self.ffs_stats.record()
                self.gpu_owner_stats.record()
                self._profile_update(
                    group.group_id,
                    gpu_owner={
                        "mode": GPU_PIPELINE_MODE_SINGLE_OWNER,
                        "internal_order": order,
                        "ffs_cycle_ms": float(ffs_cycle_ms),
                        "edgetam_cycle_ms": float(edgetam_cycle_ms),
                        "edgetam_stage_wall_ms": float(edgetam_cycle_ms),
                        "edgetam_stage_sum_model_ms": sum(
                            float(packet.cuda_event_model_ms or packet.model_ms)
                            for packet in mask_packets.values()
                        ),
                        "stage_barrier_ms": 0.0,
                        "total_ms": float(total_ms),
                        "publish_s": self._profile_rel_s(),
                        "complete_group_published": True,
                    },
                )
        except Exception as exc:
            if not self.stop_event.is_set():
                print(f"[ERROR] Demo 2.1 GPU-owner worker failed: {type(exc).__name__}: {exc}", flush=True)
            self._mark_fatal_error("gpu-owner", exc)
            self.stop_event.set()

    def _staged_gpu_pipeline_worker(self) -> None:
        try:
            runner = self._get_or_prepare_ffs_runner() if self.args.depth_source == DEPTH_SOURCE_FFS else None
            aligners: dict[int, FfsIrToColorAligner] = {}
            edgetam_states = self._get_or_init_gpu_owner_edgetam_states()
            last_group_id = -1
            with ThreadPoolExecutor(max_workers=len(self.args.camera_ids), thread_name_prefix="demo2.1-staged-edgetam") as edge_executor:
                while not self.stop_event.is_set():
                    group = self.capture_group_slot.get_latest_after(last_group_id)
                    if group is None:
                        time.sleep(0.001)
                        continue
                    last_group_id = group.group_id
                    if not temporal_group_is_coherent(group, max_capture_skew_ms=float(self.args.max_capture_skew_ms)):
                        self._summary["staged_gpu_drop_skewed_capture_group"] = int(
                            self._summary.get("staged_gpu_drop_skewed_capture_group", 0)
                        ) + 1
                        self._profile_mark_drop(group.group_id, "staged_gpu_drop_skewed_capture_group")
                        continue

                    owner_start_s = time.perf_counter()
                    depth_group, _ = self._run_depth_cycle_for_group(
                        group=group,
                        runner=runner,
                        aligners=aligners,
                    )
                    ffs_cycle_ms = depth_group.total_ms

                    barrier_start_s = time.perf_counter()
                    first_state = next(iter(edgetam_states.values()), None)
                    if self.args.depth_source == DEPTH_SOURCE_FFS and first_state is not None:
                        torch_module = first_state["torch_module"]
                        if str(self.args.device).startswith("cuda") and torch_module.cuda.is_available():
                            torch_module.cuda.synchronize()
                    stage_barrier_ms = _elapsed_ms(barrier_start_s, time.perf_counter())

                    mask_packets, edgetam_wall_ms, edgetam_sum_model_ms = self._run_staged_edgetam_cycle_parallel(
                        states=edgetam_states,
                        group=group,
                        executor=edge_executor,
                    )
                    if mask_packets is None:
                        continue

                    total_ms = _elapsed_ms(owner_start_s, time.perf_counter())
                    packet = CompleteInferenceGroup(
                        group_id=group.group_id,
                        capture_group=group,
                        depth_group=depth_group,
                        mask_packets=mask_packets,
                        ffs_cycle_ms=float(ffs_cycle_ms),
                        edgetam_cycle_ms=float(edgetam_wall_ms),
                        edgetam_stage_wall_ms=float(edgetam_wall_ms),
                        edgetam_stage_sum_model_ms=float(edgetam_sum_model_ms),
                        stage_barrier_ms=float(stage_barrier_ms),
                        total_gpu_owner_ms=float(total_ms),
                        pipeline_mode=GPU_PIPELINE_MODE_STAGED,
                        internal_order=str(self.args.staged_order),
                    )
                    self.complete_inference_slot.put(packet)
                    self._init_profile_set_once(("first_complete_inference_group_s",), self._profile_rel_s())
                    self._init_profile_set_once(("first_complete_inference_group_id",), int(group.group_id))
                    self._latest_depth_group = depth_group
                    self.ffs_stats.record()
                    self.gpu_owner_stats.record()
                    parallel_efficiency = (
                        float(edgetam_sum_model_ms) / float(edgetam_wall_ms)
                        if float(edgetam_wall_ms) > 0.0
                        else 0.0
                    )
                    self._profile_update(
                        group.group_id,
                        gpu_owner={
                            "mode": GPU_PIPELINE_MODE_STAGED,
                            "internal_order": str(self.args.staged_order),
                            "staged_order": str(self.args.staged_order),
                            "ffs_stage": "sequential_cam0_cam1_cam2",
                            "edgetam_stage": "parallel_cam0_cam1_cam2",
                            "ffs_stage_ms": float(ffs_cycle_ms),
                            "edgetam_stage_wall_ms": float(edgetam_wall_ms),
                            "edgetam_stage_sum_model_ms": float(edgetam_sum_model_ms),
                            "edgetam_parallel_efficiency": float(parallel_efficiency),
                            "stage_barrier_ms": float(stage_barrier_ms),
                            "total_ms": float(total_ms),
                            "publish_s": self._profile_rel_s(),
                            "complete_group_published": True,
                        },
                    )
        except Exception as exc:
            if not self.stop_event.is_set():
                print(f"[ERROR] Demo 2.1 staged GPU worker failed: {type(exc).__name__}: {exc}", flush=True)
            self._mark_fatal_error("staged-gpu", exc)
            self.stop_event.set()

    def _stage_capture_dispatch_worker(self) -> None:
        try:
            last_group_id = -1
            while not self.stop_event.is_set():
                group = self.capture_group_slot.get_latest_after(last_group_id)
                if group is None:
                    time.sleep(0.001)
                    continue
                last_group_id = group.group_id
                if not temporal_group_is_coherent(group, max_capture_skew_ms=float(self.args.max_capture_skew_ms)):
                    self._summary["overlapped_stage_drop_skewed_capture_group"] = int(
                        self._summary.get("overlapped_stage_drop_skewed_capture_group", 0)
                    ) + 1
                    self._profile_mark_drop(group.group_id, "overlapped_stage_drop_skewed_capture_group")
                    continue
                dispatch_s = self._profile_rel_s()
                scheduler_mode = str(getattr(self.args, "stage_scheduler_mode", STAGE_SCHEDULER_MODE_MASK_GATED))
                if scheduler_mode == STAGE_SCHEDULER_MODE_MASK_GATED:
                    self.edgetam_stage_input_slot.put(group)
                    depth_dispatch_policy = "after_mask_stage"
                    ffs_dispatched = False
                else:
                    self.stage_window_scheduler.put_capture(group)
                    self.stage_join_buffer.put_capture(group)
                    depth_dispatch_policy = (
                        "edge_start_reservation"
                        if scheduler_mode == STAGE_SCHEDULER_MODE_EDGE_START
                        else "bounded_lookahead_reservation"
                    )
                    ffs_dispatched = True
                self._profile_update(
                    group.group_id,
                    stage_dispatch={
                        "group_id": int(group.group_id),
                        "capture_dispatch_s": float(dispatch_s),
                        "edgetam_dispatched": scheduler_mode == STAGE_SCHEDULER_MODE_MASK_GATED,
                        "ffs_dispatched": bool(ffs_dispatched),
                        "scheduler_mode": scheduler_mode,
                        "depth_dispatch_policy": depth_dispatch_policy,
                    },
                )
        except Exception as exc:
            if not self.stop_event.is_set():
                print(f"[ERROR] Demo 2.1 overlapped stage dispatch failed: {type(exc).__name__}: {exc}", flush=True)
            self._mark_fatal_error("stage-dispatch", exc)
            self.stop_event.set()

    def _ffs_stage_worker(self) -> None:
        try:
            runner = self._get_or_prepare_ffs_runner() if self.args.depth_source == DEPTH_SOURCE_FFS else None
            aligners: dict[int, FfsIrToColorAligner] = {}
            last_group_id = -1
            while not self.stop_event.is_set():
                scheduler_mode = str(getattr(self.args, "stage_scheduler_mode", STAGE_SCHEDULER_MODE_MASK_GATED))
                request_s = self._profile_rel_s()
                if scheduler_mode == STAGE_SCHEDULER_MODE_MASK_GATED:
                    group = self.ffs_stage_input_slot.get_latest_after(last_group_id)
                    if group is None:
                        time.sleep(0.001)
                        continue
                    if int(group.group_id) <= int(last_group_id):
                        continue
                    task = StageTask(group_id=int(group.group_id), group=group, reason="after-mask")
                else:
                    task = self.stage_window_scheduler.reserve_next_depth_task(mode=scheduler_mode)
                    if task is None:
                        time.sleep(0.001)
                        continue
                    group = task.group
                if scheduler_mode == STAGE_SCHEDULER_MODE_MASK_GATED:
                    last_group_id = int(group.group_id)
                start_rel_s = self._profile_rel_s()
                start_s = time.perf_counter()
                depth_group, _ = self._run_depth_cycle_for_group(
                    group=group,
                    runner=runner,
                    aligners=aligners,
                )
                wall_ms = _elapsed_ms(start_s, time.perf_counter())
                if scheduler_mode != STAGE_SCHEDULER_MODE_MASK_GATED:
                    self.stage_window_scheduler.mark_depth_done(group.group_id)
                self.stage_join_buffer.put_depth(depth_group)
                self._latest_depth_group = depth_group
                self.ffs_stats.record()
                self._profile_update(
                    group.group_id,
                    ffs_stage={
                        "request_s": float(request_s),
                        "start_s": float(start_rel_s),
                        "publish_s": self._profile_rel_s(),
                        "reason": task.reason,
                        "scheduler_mode": scheduler_mode,
                        "request_to_start_ms": float(max(0.0, (start_rel_s - request_s) * 1000.0)),
                        "wall_ms": float(wall_ms),
                        "depth_total_ms": float(depth_group.total_ms),
                        "input_age_ms": float((time.perf_counter() - group.created_perf_s) * 1000.0),
                    },
                )
        except Exception as exc:
            if not self.stop_event.is_set():
                print(f"[ERROR] Demo 2.1 overlapped FFS stage failed: {type(exc).__name__}: {exc}", flush=True)
            self._mark_fatal_error("ffs-stage", exc)
            self.stop_event.set()

    def _edgetam_stage_worker(self) -> None:
        try:
            edgetam_states = self._get_or_init_gpu_owner_edgetam_states()
            last_group_id = -1
            while not self.stop_event.is_set():
                scheduler_mode = str(getattr(self.args, "stage_scheduler_mode", STAGE_SCHEDULER_MODE_MASK_GATED))
                request_s = self._profile_rel_s()
                if scheduler_mode == STAGE_SCHEDULER_MODE_MASK_GATED:
                    group = self.edgetam_stage_input_slot.get_latest_after(last_group_id)
                    if group is None:
                        time.sleep(0.001)
                        continue
                    if int(group.group_id) <= int(last_group_id):
                        continue
                    task = StageTask(group_id=int(group.group_id), group=group, reason="mask-gated")
                else:
                    task = self.stage_window_scheduler.reserve_next_edge_task()
                    if task is None:
                        time.sleep(0.001)
                        continue
                    group = task.group
                    if int(group.group_id) <= int(last_group_id):
                        continue
                last_group_id = int(group.group_id)
                start_rel_s = self._profile_rel_s()
                start_s = time.perf_counter()
                mask_packets, edgetam_cycle_ms = self._run_gpu_owner_edgetam_cycle(
                    states=edgetam_states,
                    group=group,
                )
                if mask_packets is None:
                    continue
                wall_ms = _elapsed_ms(start_s, time.perf_counter())
                sum_model_ms = sum(
                    float(packet.cuda_event_model_ms or packet.model_ms)
                    for packet in mask_packets.values()
                )
                mask_group = MaskGroup(
                    group_id=group.group_id,
                    mask_packets=mask_packets,
                    edgetam_stage_wall_ms=float(wall_ms),
                    edgetam_stage_sum_model_ms=float(sum_model_ms),
                    edgetam_stage_mode="batch-vision" if bool(getattr(self.args, "edgetam_batch_vision_encoder", False)) else "sequential",
                )
                if scheduler_mode == STAGE_SCHEDULER_MODE_MASK_GATED:
                    self.stage_join_buffer.put_capture(group)
                else:
                    self.stage_window_scheduler.mark_mask_done(group.group_id)
                self.stage_join_buffer.put_mask(mask_group)
                if scheduler_mode == STAGE_SCHEDULER_MODE_MASK_GATED:
                    self.ffs_stage_input_slot.put(group)
                publish_s = self._profile_rel_s()
                self._profile_update(
                    group.group_id,
                    edgetam_stage={
                        "request_s": float(request_s),
                        "start_s": float(start_rel_s),
                        "publish_s": float(publish_s),
                        "depth_dispatch_s": float(publish_s) if scheduler_mode == STAGE_SCHEDULER_MODE_MASK_GATED else None,
                        "reason": task.reason,
                        "scheduler_mode": scheduler_mode,
                        "request_to_start_ms": float(max(0.0, (start_rel_s - request_s) * 1000.0)),
                        "wall_ms": float(wall_ms),
                        "cycle_ms": float(edgetam_cycle_ms),
                        "sum_model_ms": float(sum_model_ms),
                        "mode": mask_group.edgetam_stage_mode,
                        "stateful_monotonic": True,
                        "depth_dispatched_after_mask": scheduler_mode == STAGE_SCHEDULER_MODE_MASK_GATED,
                    },
                )
        except Exception as exc:
            if not self.stop_event.is_set():
                print(f"[ERROR] Demo 2.1 overlapped EdgeTAM stage failed: {type(exc).__name__}: {exc}", flush=True)
            self._mark_fatal_error("edgetam-stage", exc)
            self.stop_event.set()

    def _publish_complete_inference_group(
        self,
        *,
        complete: CompleteInferenceGroup,
        rng: np.random.Generator,
        ray_cache: dict[int, tuple[np.ndarray, np.ndarray]],
        fusion_waits: dict[str, float] | None = None,
        warning_label: str,
    ) -> bool:
        depth_group = complete.depth_group
        if not temporal_group_is_coherent(depth_group, max_capture_skew_ms=float(self.args.max_capture_skew_ms)):
            self._summary["fusion_drop_skewed_group"] = int(self._summary.get("fusion_drop_skewed_group", 0)) + 1
            self._profile_mark_drop(depth_group.group_id, "fusion_drop_skewed_group")
            return False
        if set(int(idx) for idx in complete.mask_packets) != set(int(idx) for idx in self.args.camera_ids):
            self._summary["fusion_timeout_groups"] = int(self._summary.get("fusion_timeout_groups", 0)) + 1
            self._profile_mark_drop(depth_group.group_id, "single_owner_missing_mask")
            return False
        waits = fusion_waits or {
            "wait_depth_ms": 0.0,
            "wait_total_ms": 0.0,
            **{f"wait_mask_cam{int(camera_idx)}_ms": 0.0 for camera_idx in self.args.camera_ids},
        }
        self._profile_update(depth_group.group_id, fusion=waits)
        try:
            if async_fusion_filter_enabled(self.args):
                raw_packet = self._build_raw_fused_packet(
                    depth_group=depth_group,
                    masks=complete.mask_packets,
                    ray_cache=ray_cache,
                    rng=rng,
                )
                self._publish_raw_fused_for_async_filter(raw_packet)
                return True
            packet = self._build_fused_packet(
                depth_group=depth_group,
                masks=complete.mask_packets,
                ray_cache=ray_cache,
                rng=rng,
            )
        except Exception as exc:
            if not self.stop_event.is_set():
                print(f"[WARN] Demo 2.1 {warning_label} group {depth_group.group_id} failed: {type(exc).__name__}: {exc}", flush=True)
            return False
        self._latest_fused = packet
        self.fusion_stats.record()
        self._summary["fusion_complete_groups"] = int(self._summary.get("fusion_complete_groups", 0)) + 1
        self._publish_render_packet(packet)
        return True

    def _stage_join_fusion_worker(self) -> None:
        rng = np.random.default_rng()
        ray_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        try:
            while not self.stop_event.is_set():
                ready = self.stage_join_buffer.pop_latest_ready()
                if ready is None:
                    time.sleep(0.001)
                    continue
                capture_group, depth_group, mask_group = ready
                start_s = time.perf_counter()
                if not (
                    int(capture_group.group_id) == int(depth_group.group_id) == int(mask_group.group_id)
                ):
                    self._summary["stage_join_group_id_mismatch"] = int(
                        self._summary.get("stage_join_group_id_mismatch", 0)
                    ) + 1
                    self._profile_mark_drop(capture_group.group_id, "stage_join_group_id_mismatch")
                    continue
                complete = CompleteInferenceGroup(
                    group_id=capture_group.group_id,
                    capture_group=capture_group,
                    depth_group=depth_group,
                    mask_packets=mask_group.mask_packets,
                    ffs_cycle_ms=float(depth_group.total_ms),
                    edgetam_cycle_ms=float(mask_group.edgetam_stage_wall_ms),
                    edgetam_stage_wall_ms=float(mask_group.edgetam_stage_wall_ms),
                    edgetam_stage_sum_model_ms=float(mask_group.edgetam_stage_sum_model_ms),
                    stage_barrier_ms=0.0,
                    total_gpu_owner_ms=float(max(depth_group.total_ms, mask_group.edgetam_stage_wall_ms)),
                    pipeline_mode=GPU_PIPELINE_MODE_OVERLAPPED_STAGES,
                    internal_order="cross_group_overlap",
                )
                self.complete_inference_slot.put(complete)
                self._init_profile_set_once(("first_complete_inference_group_s",), self._profile_rel_s())
                self._init_profile_set_once(("first_complete_inference_group_id",), int(capture_group.group_id))
                self.gpu_owner_stats.record()
                published = self._publish_complete_inference_group(
                    complete=complete,
                    rng=rng,
                    ray_cache=ray_cache,
                    warning_label="overlapped-stage fusion",
                )
                join_ms = _elapsed_ms(start_s, time.perf_counter())
                counters = self.stage_join_buffer.snapshot()
                ffs_publish_s: float | None = None
                mask_publish_s: float | None = None
                with self._profile_lock:
                    record = self._profile_records.get(int(capture_group.group_id), {})
                    if isinstance(record.get("ffs_stage"), dict):
                        value = record["ffs_stage"].get("publish_s")
                        if isinstance(value, (int, float)):
                            ffs_publish_s = float(value)
                    if isinstance(record.get("edgetam_stage"), dict):
                        value = record["edgetam_stage"].get("publish_s")
                        if isinstance(value, (int, float)):
                            mask_publish_s = float(value)
                join_publish_s = self._profile_rel_s()
                if ffs_publish_s is not None and mask_publish_s is not None:
                    depth_ready_before_mask = bool(ffs_publish_s <= mask_publish_s)
                    depth_wait_after_mask_ms = max(0.0, (ffs_publish_s - mask_publish_s) * 1000.0)
                    mask_wait_after_depth_ms = max(0.0, (mask_publish_s - ffs_publish_s) * 1000.0)
                    same_group_join_latency_ms = max(0.0, (join_publish_s - max(ffs_publish_s, mask_publish_s)) * 1000.0)
                else:
                    depth_ready_before_mask = False
                    depth_wait_after_mask_ms = 0.0
                    mask_wait_after_depth_ms = 0.0
                    same_group_join_latency_ms = 0.0
                self._profile_update(
                    capture_group.group_id,
                    gpu_owner={
                        "mode": GPU_PIPELINE_MODE_OVERLAPPED_STAGES,
                        "internal_order": "cross_group_overlap",
                        "ffs_cycle_ms": float(depth_group.total_ms),
                        "edgetam_cycle_ms": float(mask_group.edgetam_stage_wall_ms),
                        "edgetam_stage_wall_ms": float(mask_group.edgetam_stage_wall_ms),
                        "edgetam_stage_sum_model_ms": float(mask_group.edgetam_stage_sum_model_ms),
                        "stage_barrier_ms": 0.0,
                        "total_ms": float(max(depth_group.total_ms, mask_group.edgetam_stage_wall_ms)),
                        "publish_s": self._profile_rel_s(),
                        "complete_group_published": bool(published),
                    },
                    stage_join={
                        "publish_s": float(join_publish_s),
                        "wall_ms": float(join_ms),
                        "depth_mask_group_id_match": True,
                        "depth_ready_before_mask": depth_ready_before_mask,
                        "depth_wait_after_mask_ms": float(depth_wait_after_mask_ms),
                        "mask_wait_after_depth_ms": float(mask_wait_after_depth_ms),
                        "same_group_join_latency_ms": float(same_group_join_latency_ms),
                        "capture_group_id": int(capture_group.group_id),
                        "depth_group_id": int(depth_group.group_id),
                        "mask_group_id": int(mask_group.group_id),
                        **counters,
                    },
                )
        except Exception as exc:
            if not self.stop_event.is_set():
                print(f"[ERROR] Demo 2.1 overlapped stage join failed: {type(exc).__name__}: {exc}", flush=True)
            self._mark_fatal_error("stage-join", exc)
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
            if not temporal_group_is_coherent(depth_group, max_capture_skew_ms=float(self.args.max_capture_skew_ms)):
                self._summary["fusion_drop_skewed_group"] = int(self._summary.get("fusion_drop_skewed_group", 0)) + 1
                self._profile_mark_drop(depth_group.group_id, "fusion_drop_skewed_group")
                continue
            fusion_wait_start_s = time.perf_counter()
            deadline_s = time.perf_counter() + float(self.args.fusion_timeout_ms) / 1000.0
            mask_by_camera: dict[int, CameraMaskPacket] = {}
            mask_waits: dict[str, float] = {"wait_depth_ms": 0.0}
            for camera_idx in self.args.camera_ids:
                wait_start_s = time.perf_counter()
                mask = self._wait_mask_for_group(camera_idx=int(camera_idx), group_id=depth_group.group_id, deadline_s=deadline_s)
                mask_waits[f"wait_mask_cam{int(camera_idx)}_ms"] = _elapsed_ms(wait_start_s, time.perf_counter())
                if mask is None:
                    incomplete += 1
                    self._summary["fusion_timeout_groups"] = int(self._summary.get("fusion_timeout_groups", 0)) + 1
                    key = f"missing_mask_cam{int(camera_idx)}"
                    self._summary[key] = int(self._summary.get(key, 0)) + 1
                    self._profile_update(depth_group.group_id, fusion=mask_waits)
                    self._profile_mark_drop(depth_group.group_id, key)
                    break
                mask_by_camera[int(camera_idx)] = mask
            if len(mask_by_camera) != len(self.args.camera_ids):
                continue
            mask_waits["wait_total_ms"] = _elapsed_ms(fusion_wait_start_s, time.perf_counter())
            self._profile_update(depth_group.group_id, fusion=mask_waits)
            try:
                if async_fusion_filter_enabled(self.args):
                    raw_packet = self._build_raw_fused_packet(
                        depth_group=depth_group,
                        masks=mask_by_camera,
                        ray_cache=ray_cache,
                        rng=rng,
                    )
                    self._publish_raw_fused_for_async_filter(raw_packet)
                    continue
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
            self._latest_fused = packet
            self.fusion_stats.record()
            self._summary["fusion_complete_groups"] = int(self._summary.get("fusion_complete_groups", 0)) + 1
            self._publish_render_packet(packet)
            if incomplete:
                self._summary["dropped_incomplete_fusion_groups"] = incomplete

    def _fusion_worker_single_owner(self) -> None:
        last_complete_group = -1
        rng = np.random.default_rng()
        ray_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        while not self.stop_event.is_set():
            complete = self.complete_inference_slot.get_latest_after(last_complete_group)
            if complete is None:
                time.sleep(0.001)
                continue
            last_complete_group = complete.group_id
            depth_group = complete.depth_group
            if not temporal_group_is_coherent(depth_group, max_capture_skew_ms=float(self.args.max_capture_skew_ms)):
                self._summary["fusion_drop_skewed_group"] = int(self._summary.get("fusion_drop_skewed_group", 0)) + 1
                self._profile_mark_drop(depth_group.group_id, "fusion_drop_skewed_group")
                continue
            if set(int(idx) for idx in complete.mask_packets) != set(int(idx) for idx in self.args.camera_ids):
                self._summary["fusion_timeout_groups"] = int(self._summary.get("fusion_timeout_groups", 0)) + 1
                self._profile_mark_drop(depth_group.group_id, "single_owner_missing_mask")
                continue
            self._profile_update(
                depth_group.group_id,
                fusion={
                    "wait_depth_ms": 0.0,
                    "wait_total_ms": 0.0,
                    **{f"wait_mask_cam{int(camera_idx)}_ms": 0.0 for camera_idx in self.args.camera_ids},
                },
            )
            try:
                if async_fusion_filter_enabled(self.args):
                    raw_packet = self._build_raw_fused_packet(
                        depth_group=depth_group,
                        masks=complete.mask_packets,
                        ray_cache=ray_cache,
                        rng=rng,
                    )
                    self._publish_raw_fused_for_async_filter(raw_packet)
                    continue
                packet = self._build_fused_packet(
                    depth_group=depth_group,
                    masks=complete.mask_packets,
                    ray_cache=ray_cache,
                    rng=rng,
                )
            except Exception as exc:
                if not self.stop_event.is_set():
                    print(f"[WARN] Demo 2.1 single-owner fusion group {depth_group.group_id} failed: {type(exc).__name__}: {exc}", flush=True)
                continue
            self._latest_fused = packet
            self.fusion_stats.record()
            self._summary["fusion_complete_groups"] = int(self._summary.get("fusion_complete_groups", 0)) + 1
            self._publish_render_packet(packet)

    def _build_raw_fused_packet(
        self,
        *,
        depth_group: DepthGroup,
        masks: dict[int, CameraMaskPacket],
        ray_cache: dict[int, tuple[np.ndarray, np.ndarray]],
        rng: np.random.Generator,
    ) -> RawFusedPcdPacket:
        started_s = time.perf_counter()
        object_clouds: list[CameraLayerCloud] = []
        controller_clouds: list[CameraLayerCloud] = []
        build_object_raw_ms = 0.0
        build_controller_raw_ms = 0.0
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
            object_build_start_s = time.perf_counter()
            if object_tracking_enabled(self.args.track_mode):
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
            else:
                object_pts_cam = np.empty((0, 3), dtype=np.float32)
                object_cols = np.empty((0, 3), dtype=np.uint8)
            build_object_raw_ms += _elapsed_ms(object_build_start_s, time.perf_counter())
            object_clouds.append(
                CameraLayerCloud(
                    camera_idx=int(camera_idx),
                    label=str(self.args.object_prompt),
                    points_m=transform_points(object_pts_cam, self._c2w_by_camera[int(camera_idx)]),
                    colors_rgb=object_cols,
                )
            )
            controller_build_start_s = time.perf_counter()
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
            build_controller_raw_ms += _elapsed_ms(controller_build_start_s, time.perf_counter())
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
        raw_fusion_ms = _elapsed_ms(started_s, time.perf_counter())
        self._profile_update(
            depth_group.group_id,
            raw_fusion={
                "build_object_raw_ms": float(build_object_raw_ms),
                "build_controller_raw_ms": float(build_controller_raw_ms),
                "capture_temporal_skew_ms": float(depth_group.max_temporal_skew_ms),
                "timestamp_source": depth_group.timestamp_source,
                "total_ms": float(raw_fusion_ms),
                "publish_s": self._profile_rel_s(),
                "raw_packet_submitted": True,
            },
            points={
                "object_raw": int(object_raw_count),
                "controller_raw": int(controller_raw_count),
            },
        )
        return RawFusedPcdPacket(
            group_id=depth_group.group_id,
            created_perf_s=time.perf_counter(),
            raw_object=raw_object,
            raw_controller=raw_controller,
            raw_fusion_ms=float(raw_fusion_ms),
            build_object_raw_ms=float(build_object_raw_ms),
            build_controller_raw_ms=float(build_controller_raw_ms),
            object_raw_points=int(object_raw_count),
            controller_raw_points=int(controller_raw_count),
            ffs_cycle_ms=depth_group.total_ms,
            edgetam_ms_by_camera={idx: masks[idx].cuda_event_model_ms or masks[idx].model_ms for idx in masks},
            ffs_gpu_gate_wait_ms=depth_group.gpu_gate_wait_ms,
            edgetam_gpu_gate_wait_ms_by_camera={idx: masks[idx].gpu_gate_wait_ms for idx in masks},
            capture_temporal_skew_ms=float(depth_group.max_temporal_skew_ms),
            capture_time_offsets_ms_by_camera=dict(depth_group.per_camera_time_offset_ms),
            timestamp_source=str(depth_group.timestamp_source),
        )

    def _filter_raw_fused_packet(self, raw: RawFusedPcdPacket) -> FusedPcdPacket:
        filter_start_s = time.perf_counter()
        object_filter_ms = 0.0
        controller_filter_ms = 0.0
        object_filter_stats: dict[str, Any] = {}
        controller_filter_stats: dict[str, Any] = {}
        if raw.raw_object is not None:
            object_filter_start_s = time.perf_counter()
            object_points, object_colors, stats = apply_semantic_postprocess(
                raw.raw_object,
                filter_cap=int(self.args.object_filter_cap),
                filter_voxel_size_m=float(self.args.object_filter_voxel_m),
                phystwin_radius_m=float(self.args.phystwin_radius_m),
                phystwin_nb_points=int(self.args.phystwin_nb_points),
                enhanced_component_voxel_size_m=float(self.args.enhanced_component_voxel_size_m),
                enhanced_keep_near_main_gap_m=float(self.args.enhanced_keep_near_main_gap_m),
            )
            object_filter_ms = _elapsed_ms(object_filter_start_s, time.perf_counter())
            object_filter_stats = stats if isinstance(stats, dict) else {}
        else:
            object_points = np.empty((0, 3), dtype=np.float32)
            object_colors = np.empty((0, 3), dtype=np.uint8)
        if raw.raw_controller is not None:
            controller_filter_start_s = time.perf_counter()
            controller_points, controller_colors, stats = apply_semantic_postprocess(
                raw.raw_controller,
                filter_cap=int(self.args.controller_filter_cap),
                filter_voxel_size_m=float(self.args.controller_filter_voxel_m),
                phystwin_radius_m=float(self.args.phystwin_radius_m),
                phystwin_nb_points=int(self.args.phystwin_nb_points),
                enhanced_component_voxel_size_m=float(self.args.enhanced_component_voxel_size_m),
                enhanced_keep_near_main_gap_m=float(self.args.enhanced_keep_near_main_gap_m),
            )
            controller_filter_ms = _elapsed_ms(controller_filter_start_s, time.perf_counter())
            controller_filter_stats = stats if isinstance(stats, dict) else {}
        else:
            controller_points = np.empty((0, 3), dtype=np.float32)
            controller_colors = np.empty((0, 3), dtype=np.uint8)
        filter_ms = _elapsed_ms(filter_start_s, time.perf_counter())
        input_age_ms = _elapsed_ms(raw.created_perf_s, time.perf_counter())
        profile_filter: dict[str, Any] = {
            "object_enhanced_pt_ms": float(object_filter_ms),
            "controller_pt_filter_ms": float(controller_filter_ms),
            "total_ms": float(filter_ms),
            "input_age_ms": float(input_age_ms),
            "publish_s": self._profile_rel_s(),
            "pending_replacements": int(self.raw_fused_slot.dropped_count),
            "pending_replacements_total": int(self.raw_fused_slot.total_dropped_count),
        }
        if self.args.profile_filter_detail:
            profile_filter["object_filter_detail"] = object_filter_stats
            profile_filter["controller_filter_detail"] = controller_filter_stats
        self._profile_update(
            raw.group_id,
            filter=profile_filter,
            fusion={
                "build_object_raw_ms": float(raw.build_object_raw_ms),
                "build_controller_raw_ms": float(raw.build_controller_raw_ms),
                "capture_temporal_skew_ms": float(raw.capture_temporal_skew_ms),
                "timestamp_source": raw.timestamp_source,
                "object_enhanced_pt_ms": float(object_filter_ms),
                "controller_pt_filter_ms": float(controller_filter_ms),
                "filter_ms": float(filter_ms),
                "raw_fusion_ms": float(raw.raw_fusion_ms),
                "total_ms": float(raw.raw_fusion_ms + filter_ms),
                "publish_s": self._profile_rel_s(),
            },
            points={
                "object_raw": int(raw.object_raw_points),
                "object_filtered": int(len(object_points)),
                "controller_raw": int(raw.controller_raw_points),
                "controller_filtered": int(len(controller_points)),
            },
            complete=True,
            drop_reason=None,
        )
        self._init_profile_set_once(("first_complete_fused_group_s",), self._profile_rel_s())
        self._init_profile_set_once(("first_complete_fused_group_id",), int(raw.group_id))
        return FusedPcdPacket(
            group_id=raw.group_id,
            created_perf_s=time.perf_counter(),
            object_points_m=object_points,
            object_colors_rgb=object_colors,
            controller_points_m=controller_points,
            controller_colors_rgb=controller_colors,
            fusion_ms=float(raw.raw_fusion_ms),
            filter_ms=float(filter_ms),
            object_raw_points=int(raw.object_raw_points),
            controller_raw_points=int(raw.controller_raw_points),
            ffs_cycle_ms=raw.ffs_cycle_ms,
            edgetam_ms_by_camera=dict(raw.edgetam_ms_by_camera),
            ffs_gpu_gate_wait_ms=raw.ffs_gpu_gate_wait_ms,
            edgetam_gpu_gate_wait_ms_by_camera=dict(raw.edgetam_gpu_gate_wait_ms_by_camera),
            capture_temporal_skew_ms=float(raw.capture_temporal_skew_ms),
            capture_time_offsets_ms_by_camera=dict(raw.capture_time_offsets_ms_by_camera),
            timestamp_source=str(raw.timestamp_source),
        )

    def _publish_raw_fused_for_async_filter(self, raw: RawFusedPcdPacket) -> None:
        self.raw_fused_slot.put(raw)
        self._latest_raw_fused = raw
        self.raw_fusion_stats.record(raw.created_perf_s)
        self._summary["raw_fusion_groups"] = int(self._summary.get("raw_fusion_groups", 0)) + 1

    def _publish_render_packet(self, packet: FusedPcdPacket) -> None:
        if packet.group_id % int(self.args.render_every_n) != 0:
            return
        publish_s = self._profile_rel_s()
        self.render_buffer.publish(packet)
        self._profile_update(
            packet.group_id,
            render_publish={
                "publish_s": float(publish_s),
                "render_every_n": int(self.args.render_every_n),
                "render_buffer": self.render_buffer.snapshot(),
            },
        )
        self._render_request()

    def _async_filter_worker(self) -> None:
        last_raw_group = -1
        while not self.stop_event.is_set():
            raw = self.raw_fused_slot.get_latest_after(last_raw_group)
            if raw is None:
                time.sleep(0.001)
                continue
            last_raw_group = raw.group_id
            try:
                packet = self._filter_raw_fused_packet(raw)
            except Exception as exc:
                if not self.stop_event.is_set():
                    print(f"[WARN] Demo 2.1 async filter group {raw.group_id} failed: {type(exc).__name__}: {exc}", flush=True)
                self._profile_mark_drop(raw.group_id, "async_filter_failed")
                continue
            self._latest_fused = packet
            self.filter_output_stats.record(packet.created_perf_s)
            self.fusion_stats.record(packet.created_perf_s)
            self._summary["filter_output_groups"] = int(self._summary.get("filter_output_groups", 0)) + 1
            self._summary["fusion_complete_groups"] = int(self._summary.get("fusion_complete_groups", 0)) + 1
            self._publish_render_packet(packet)

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
        build_object_raw_ms = 0.0
        build_controller_raw_ms = 0.0
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
            object_build_start_s = time.perf_counter()
            if object_tracking_enabled(self.args.track_mode):
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
            else:
                object_pts_cam = np.empty((0, 3), dtype=np.float32)
                object_cols = np.empty((0, 3), dtype=np.uint8)
            build_object_raw_ms += _elapsed_ms(object_build_start_s, time.perf_counter())
            object_clouds.append(
                CameraLayerCloud(
                    camera_idx=int(camera_idx),
                    label=str(self.args.object_prompt),
                    points_m=transform_points(object_pts_cam, self._c2w_by_camera[int(camera_idx)]),
                    colors_rgb=object_cols,
                )
            )
            controller_build_start_s = time.perf_counter()
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
            build_controller_raw_ms += _elapsed_ms(controller_build_start_s, time.perf_counter())
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
        object_filter_ms = 0.0
        controller_filter_ms = 0.0
        object_filter_stats: dict[str, Any] = {}
        controller_filter_stats: dict[str, Any] = {}
        if raw_object is not None:
            object_filter_start_s = time.perf_counter()
            object_points, object_colors, _ = apply_semantic_postprocess(
                raw_object,
                filter_cap=int(self.args.object_filter_cap),
                filter_voxel_size_m=float(self.args.object_filter_voxel_m),
                phystwin_radius_m=float(self.args.phystwin_radius_m),
                phystwin_nb_points=int(self.args.phystwin_nb_points),
                enhanced_component_voxel_size_m=float(self.args.enhanced_component_voxel_size_m),
                enhanced_keep_near_main_gap_m=float(self.args.enhanced_keep_near_main_gap_m),
            )
            object_filter_ms = _elapsed_ms(object_filter_start_s, time.perf_counter())
            object_filter_stats = _ if isinstance(_, dict) else {}
        else:
            object_points = np.empty((0, 3), dtype=np.float32)
            object_colors = np.empty((0, 3), dtype=np.uint8)
        if raw_controller is not None:
            controller_filter_start_s = time.perf_counter()
            controller_points, controller_colors, _ = apply_semantic_postprocess(
                raw_controller,
                filter_cap=int(self.args.controller_filter_cap),
                filter_voxel_size_m=float(self.args.controller_filter_voxel_m),
                phystwin_radius_m=float(self.args.phystwin_radius_m),
                phystwin_nb_points=int(self.args.phystwin_nb_points),
                enhanced_component_voxel_size_m=float(self.args.enhanced_component_voxel_size_m),
                enhanced_keep_near_main_gap_m=float(self.args.enhanced_keep_near_main_gap_m),
            )
            controller_filter_ms = _elapsed_ms(controller_filter_start_s, time.perf_counter())
            controller_filter_stats = _ if isinstance(_, dict) else {}
        else:
            controller_points = np.empty((0, 3), dtype=np.float32)
            controller_colors = np.empty((0, 3), dtype=np.uint8)
        filter_ms = _elapsed_ms(filter_start_s, time.perf_counter())
        fusion_total_ms = _elapsed_ms(started_s, time.perf_counter())
        profile_fusion: dict[str, Any] = {
            "build_object_raw_ms": float(build_object_raw_ms),
            "build_controller_raw_ms": float(build_controller_raw_ms),
            "capture_temporal_skew_ms": float(depth_group.max_temporal_skew_ms),
            "timestamp_source": depth_group.timestamp_source,
            "object_enhanced_pt_ms": float(object_filter_ms),
            "controller_pt_filter_ms": float(controller_filter_ms),
            "filter_ms": float(filter_ms),
            "total_ms": float(fusion_total_ms),
            "publish_s": self._profile_rel_s(),
        }
        if self.args.profile_filter_detail:
            profile_fusion["object_filter_detail"] = object_filter_stats
            profile_fusion["controller_filter_detail"] = controller_filter_stats
        self._profile_update(
            depth_group.group_id,
            fusion=profile_fusion,
            points={
                "object_raw": int(object_raw_count),
                "object_filtered": int(len(object_points)),
                "controller_raw": int(controller_raw_count),
                "controller_filtered": int(len(controller_points)),
            },
            complete=True,
            drop_reason=None,
        )
        self._init_profile_set_once(("first_complete_fused_group_s",), self._profile_rel_s())
        self._init_profile_set_once(("first_complete_fused_group_id",), int(depth_group.group_id))
        return FusedPcdPacket(
            group_id=depth_group.group_id,
            created_perf_s=time.perf_counter(),
            object_points_m=object_points,
            object_colors_rgb=object_colors,
            controller_points_m=controller_points,
            controller_colors_rgb=controller_colors,
            fusion_ms=fusion_total_ms,
            filter_ms=filter_ms,
            object_raw_points=object_raw_count,
            controller_raw_points=controller_raw_count,
            ffs_cycle_ms=depth_group.total_ms,
            edgetam_ms_by_camera={idx: masks[idx].cuda_event_model_ms or masks[idx].model_ms for idx in masks},
            ffs_gpu_gate_wait_ms=depth_group.gpu_gate_wait_ms,
            edgetam_gpu_gate_wait_ms_by_camera={idx: masks[idx].gpu_gate_wait_ms for idx in masks},
            capture_temporal_skew_ms=float(depth_group.max_temporal_skew_ms),
            capture_time_offsets_ms_by_camera=dict(depth_group.per_camera_time_offset_ms),
            timestamp_source=str(depth_group.timestamp_source),
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
        gate_wait = " ".join(
            [f"ffs={self.gpu_gate_wait_stats['ffs'].latest:.1f}ms"]
            + [
                f"edge{idx}={self.gpu_gate_wait_stats[f'edgetam_cam{idx}'].latest:.1f}ms"
                for idx in self.args.camera_ids
            ]
        )
        print(
            "[demo2.1-debug] "
            f"capture_group_fps={self.capture_group_stats.fps:.2f} "
            f"ffs_cycle_fps={self.ffs_stats.fps:.2f} "
            f"gpu_owner_fps={self.gpu_owner_stats.fps:.2f} "
            f"edge_fps_cam0={self.edge_stats[0].fps:.2f} edge_fps_cam1={self.edge_stats[1].fps:.2f} edge_fps_cam2={self.edge_stats[2].fps:.2f} "
            f"raw_fusion_fps={self.raw_fusion_stats.fps:.2f} filter_fps={self.filter_output_stats.fps:.2f} "
            f"fusion_fps={self.fusion_stats.fps:.2f} render_fps={self.render_stats.render_fps:.2f} "
            f"ffs_cycle_ms={(0.0 if depth is None else depth.total_ms):.1f} "
            f"fusion_ms={(0.0 if latest is None else latest.fusion_ms):.1f} "
            f"filter_ms={(0.0 if latest is None else latest.filter_ms):.1f} "
            f"raw_drop={self.raw_fused_slot.dropped_count} "
            f"skew_ms_med/latest={self.temporal_skew_stats.median:.1f}/{self.temporal_skew_stats.latest:.1f} "
            f"skew_drop={int(self._summary.get('capture_group_skew_drop', 0))} "
            f"no_candidate={int(self._summary.get('capture_group_no_candidate', 0))} "
            f"ffs_skew_drop={int(self._summary.get('ffs_drop_skewed_capture_group', 0))} "
            f"fusion_skew_drop={int(self._summary.get('fusion_drop_skewed_group', 0))} "
            f"object_points={(0 if latest is None else latest.object_point_count)} "
            f"controller_points={(0 if latest is None else latest.controller_point_count)} "
            f"edgetam_ms[{edge_ms}] ffs_ms[{ffs_ms}] gpu_gate_wait[{gate_wait}]",
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

        combined_state = Open3DSceneTensorLayer(
            name="demo2_1_combined_fused",
            o3d_module=o3d,
            o3c_module=o3c,
            rendering_module=rendering,
            scene=scene_widget.scene,
            material=material,
            device=device,
            backend=str(self.args.render_backend),
        )
        object_state = Open3DSceneTensorLayer(
            name="demo2_1_object_fused",
            o3d_module=o3d,
            o3c_module=o3c,
            rendering_module=rendering,
            scene=scene_widget.scene,
            material=material,
            device=device,
            backend=str(self.args.render_backend),
        )
        controller_state = Open3DSceneTensorLayer(
            name="demo2_1_controller_fused",
            o3d_module=o3d,
            o3c_module=o3c,
            rendering_module=rendering,
            scene=scene_widget.scene,
            material=material,
            device=device,
            backend=str(self.args.render_backend),
        )
        render_combiner = RenderLayerCombiner()
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
            render_started_s = time.perf_counter()
            try:
                packet = self.render_buffer.take_latest()
                if packet is None:
                    return
                wait_packet_ms = _elapsed_ms(packet.created_perf_s, render_started_s)
                combine_ms = 0.0
                if str(getattr(self.args, "render_layer_mode", DEFAULT_RENDER_LAYER_MODE)) == RENDER_LAYER_MODE_COMBINED:
                    combined_points, combined_colors, combine_ms = render_combiner.combine(
                        (
                            (packet.object_points_m, packet.object_colors_rgb),
                            (packet.controller_points_m, packet.controller_colors_rgb),
                        )
                    )
                    combined_update = combined_state.update(combined_points, combined_colors)
                    object_update = controller_update = combined_update
                    object_update_geometry_ms = 0.0
                    controller_update_geometry_ms = 0.0
                    points_update_ms = combined_update.open3d_points_update_ms
                    colors_update_ms = combined_update.open3d_colors_update_ms
                    update_geometry_ms = combined_update.open3d_update_geometry_ms
                    cpu_format_ms = combine_ms + combined_update.cpu_format_ms
                    geometry_recreated = bool(combined_update.geometry_recreated)
                    tensor_rebound = bool(combined_update.tensor_rebound)
                    set_object_points_ms = 0.0
                    set_object_colors_ms = 0.0
                    set_controller_points_ms = 0.0
                    set_controller_colors_ms = 0.0
                    object_cpu_format_ms = 0.0
                    controller_cpu_format_ms = 0.0
                else:
                    object_update = object_state.update(packet.object_points_m, packet.object_colors_rgb)
                    controller_update = controller_state.update(packet.controller_points_m, packet.controller_colors_rgb)
                    object_update_geometry_ms = object_update.open3d_update_geometry_ms
                    controller_update_geometry_ms = controller_update.open3d_update_geometry_ms
                    points_update_ms = object_update.open3d_points_update_ms + controller_update.open3d_points_update_ms
                    colors_update_ms = object_update.open3d_colors_update_ms + controller_update.open3d_colors_update_ms
                    update_geometry_ms = object_update.open3d_update_geometry_ms + controller_update.open3d_update_geometry_ms
                    cpu_format_ms = object_update.cpu_format_ms + controller_update.cpu_format_ms
                    geometry_recreated = bool(object_update.geometry_recreated or controller_update.geometry_recreated)
                    tensor_rebound = bool(object_update.tensor_rebound or controller_update.tensor_rebound)
                    set_object_points_ms = object_update.open3d_points_update_ms
                    set_object_colors_ms = object_update.open3d_colors_update_ms
                    set_controller_points_ms = controller_update.open3d_points_update_ms
                    set_controller_colors_ms = controller_update.open3d_colors_update_ms
                    object_cpu_format_ms = object_update.cpu_format_ms
                    controller_cpu_format_ms = controller_update.cpu_format_ms
                reset_camera_ms = 0.0
                if not camera_ready["value"] and (packet.object_point_count + packet.controller_point_count) > 0:
                    reset_start_s = time.perf_counter()
                    reset_camera(packet)
                    reset_camera_ms = _elapsed_ms(reset_start_s, time.perf_counter())
                    camera_ready["value"] = True
                now = time.perf_counter()
                self.render_stats.record_render(render_time_s=now, latency_ms=_elapsed_ms(packet.created_perf_s, now))
                hud_label.text = (
                    f"Demo 2.1 fused PCD | group={packet.group_id} | "
                    f"object={packet.object_point_count} pts | controller={packet.controller_point_count} pts | "
                    f"skew={packet.capture_temporal_skew_ms:.1f} ms | "
                    f"fusion={packet.fusion_ms:.1f} ms | filter={packet.filter_ms:.1f} ms | "
                    f"render_fps={self.render_stats.render_fps:.1f}"
                )
                if self.args.debug:
                    self._print_debug()
                post_redraw_ms = 0.0
                if hasattr(window, "post_redraw"):
                    try:
                        post_redraw_start_s = time.perf_counter()
                        window.post_redraw()
                        post_redraw_ms = _elapsed_ms(post_redraw_start_s, time.perf_counter())
                    except Exception:
                        pass
                total_ms = _elapsed_ms(render_started_s, time.perf_counter())
                points_count = packet.object_point_count + packet.controller_point_count
                colors_count = int(packet.object_colors_rgb.shape[0] + packet.controller_colors_rgb.shape[0])
                render_profile = RenderMicroProfileRecord(
                    render_packet_id=int(packet.group_id),
                    points_count=int(points_count),
                    colors_count=int(colors_count),
                    queue_wait_ms=float(wait_packet_ms),
                    gpu_to_cpu_copy_ms=0.0,
                    cpu_format_ms=float(cpu_format_ms),
                    open3d_points_update_ms=float(points_update_ms),
                    open3d_colors_update_ms=float(colors_update_ms),
                    open3d_update_geometry_ms=float(update_geometry_ms),
                    open3d_poll_events_ms=0.0,
                    open3d_update_renderer_ms=float(post_redraw_ms),
                    render_total_ms=float(total_ms),
                    backpressure=False,
                    backend=str(self.args.render_backend),
                    backend_effective=str(self.args.render_backend),
                    geometry_recreated=geometry_recreated,
                    tensor_rebound=tensor_rebound,
                    extra={
                        "render_layer_mode": str(getattr(self.args, "render_layer_mode", DEFAULT_RENDER_LAYER_MODE)),
                        "object_points_count": int(packet.object_point_count),
                        "controller_points_count": int(packet.controller_point_count),
                        "combine_ms": float(combine_ms),
                        "object_cpu_format_ms": float(object_cpu_format_ms),
                        "controller_cpu_format_ms": float(controller_cpu_format_ms),
                        "object_update_geometry_ms": float(object_update_geometry_ms),
                        "controller_update_geometry_ms": float(controller_update_geometry_ms),
                        "render_buffer": self.render_buffer.snapshot(),
                        "render_post_gate": self.render_post_gate.snapshot(),
                    },
                ).to_dict()
                self._profile_update(
                    packet.group_id,
                    render={
                        "wait_packet_ms": float(wait_packet_ms),
                        "queue_wait_ms": float(wait_packet_ms),
                        "gpu_to_cpu_copy_ms": 0.0,
                        "cpu_format_ms": float(cpu_format_ms),
                        "combine_ms": float(combine_ms),
                        "set_object_points_ms": float(set_object_points_ms),
                        "set_object_colors_ms": float(set_object_colors_ms),
                        "set_controller_points_ms": float(set_controller_points_ms),
                        "set_controller_colors_ms": float(set_controller_colors_ms),
                        "object_update_geometry_ms": float(object_update_geometry_ms),
                        "controller_update_geometry_ms": float(controller_update_geometry_ms),
                        "open3d_points_update_ms": float(points_update_ms),
                        "open3d_colors_update_ms": float(colors_update_ms),
                        "update_geometry_ms": float(update_geometry_ms),
                        "poll_events_ms": 0.0,
                        "update_renderer_ms": float(post_redraw_ms),
                        "reset_camera_ms": float(reset_camera_ms),
                        "total_ms": float(total_ms),
                        "render_s": self._profile_rel_s(),
                        "micro_profile": render_profile,
                    },
                )
                self._init_profile_set_once(("first_render_s",), self._profile_rel_s())
                self._init_profile_set_once(("first_render_group_id",), int(packet.group_id))
            finally:
                self.render_post_gate.mark_done()
                if not self.stop_event.is_set() and self.render_buffer.snapshot()["pending"]:
                    request_render()

        def request_render() -> None:
            if self.stop_event.is_set():
                return
            try:
                if bool(getattr(self.args, "render_async_latest_only", True)) and not self.render_post_gate.try_mark_pending():
                    return
                if not bool(getattr(self.args, "render_async_latest_only", True)):
                    self.render_post_gate.try_mark_pending()
                app.post_to_main_thread(window, render_latest)
            except Exception:
                self.render_post_gate.mark_done()
                pass

        self._render_request = request_render

        def stop_and_quit() -> None:
            self.stop_event.set()
            if os.environ.get("QQTT_WSLG_OPEN3D_FAST_EXIT") == "1":
                self.stop()
                self._write_summary()
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
    parser.add_argument("--preset", choices=PRESETS, default=PRESET_NONE)
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--serials", nargs="*", default=None)
    parser.add_argument("--camera-ids", type=parse_camera_ids, default=DEFAULT_CAMERA_IDS)
    parser.add_argument("--calibrate-path", default=str(ROOT / "calibrate.pkl"))
    parser.add_argument("--calibration-reference-serials", nargs="*", default=None)
    parser.add_argument("--track-mode", choices=TRACK_MODES, default=TRACK_MODE_CONTROLLER_OBJECT)
    parser.add_argument("--init-mode", choices=INIT_MODES, default="sam31-first-frame")
    parser.add_argument(
        "--experiment-mode",
        choices=EXPERIMENT_MODES,
        default=DEFAULT_EXPERIMENT_MODE,
        help="Controller semantic mode: demo-mode uses hand; controller-object-exp uses towel.",
    )
    parser.add_argument("--object-prompt", default="stuffed animal")
    parser.add_argument("--controller-prompt", default=DEFAULT_CONTROLLER_LABEL)
    parser.add_argument("--depth-source", choices=DEPTH_SOURCES, default=DEPTH_SOURCE_FFS)
    parser.add_argument("--render-mode", choices=RENDER_MODES, default="none")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--compile-mode", choices=COMPILE_MODES, default=DEFAULT_COMPILE_MODE)
    parser.add_argument("--edgetam-input-path", choices=EDGETAM_INPUT_PATH_MODES, default=EDGETAM_INPUT_PATH_PIL)
    parser.add_argument("--mask-postprocess", choices=MASK_POSTPROCESS_MODES, default=MASK_POSTPROCESS_HF)
    parser.add_argument(
        "--edgetam-prewarm-compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run dummy EdgeTAM video forward(s) during init to pay torch.compile lazy cost before live sessions.",
    )
    parser.add_argument(
        "--edgetam-prewarm-runs",
        type=int,
        default=1,
        help="Number of dummy EdgeTAM video forward passes for compile prewarm when enabled.",
    )
    parser.add_argument(
        "--edgetam-batch-vision-encoder",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Experiment: batch the three camera RGB frames through HF EdgeTAM get_image_features(), "
            "split the features into per-camera session caches, then keep video tracking state per camera."
        ),
    )
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--duration-s", type=float, default=0.0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--parallel-init",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Start camera, EdgeTAM load/prewarm, FFS runner init, and SAM3.1 model preload in parallel where possible.",
    )
    parser.add_argument("--profile-cuda-events", action="store_true")
    parser.add_argument("--profile-sync", action="store_true")
    parser.add_argument("--profile-edgetam-stages", action="store_true")
    parser.add_argument("--profile-nsys-markers", action="store_true")
    parser.add_argument("--profile-pipeline", action="store_true")
    parser.add_argument("--profile-filter", action="store_true")
    parser.add_argument("--profile-filter-detail", action="store_true")
    parser.add_argument("--profile-visualization", action="store_true")
    parser.add_argument("--profile-gpu-gate", action="store_true")
    parser.add_argument("--profile-h2d", action="store_true")
    parser.add_argument("--profile-json-output", default=None)
    parser.add_argument("--profile-warmup-exclude-s", type=float, default=20.0)
    parser.add_argument(
        "--gpu-sampling",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Sample GPU utilization/memory/power/clocks in a background thread and include it in profile reports.",
    )
    parser.add_argument("--gpu-sampling-interval-s", type=float, default=0.5)
    parser.add_argument("--gpu-sampling-backend", choices=GPU_SAMPLING_BACKENDS, default="nvml")
    parser.add_argument("--gpu-sampling-device-index", type=int, default=0)
    parser.add_argument("--sam31-init-retry-interval-s", type=float, default=0.5)
    parser.add_argument(
        "--sam31-init-max-attempts",
        type=int,
        default=1,
        help="Maximum SAM3.1 live init attempts per camera. Default 1 is fail-fast for the formal demo.",
    )
    parser.add_argument(
        "--sam31-cache-init-model",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reuse one SAM3.1 image model for live first-frame initialization within this process.",
    )
    parser.add_argument(
        "--sam31-keep-runtime-until-all-cameras-init",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep cached SAM3.1 init resources until all camera EdgeTAM sessions are initialized.",
    )
    parser.add_argument("--fusion-target-fps", type=float, default=10.0)
    parser.add_argument(
        "--capture-group-target-fps",
        type=float,
        default=None,
        help=(
            "Target cadence for capture-group construction. Defaults to fusion-target-fps; "
            "Demo 2.2 presets default this to camera --fps. Use 0 for no explicit throttle."
        ),
    )
    parser.add_argument("--fusion-timeout-ms", type=float, default=150.0)
    parser.add_argument("--capture-group-policy", choices=CAPTURE_GROUP_POLICIES, default=CAPTURE_GROUP_POLICY_TIMESTAMP_NEAREST)
    parser.add_argument("--max-capture-skew-ms", type=float, default=DEFAULT_MAX_CAPTURE_SKEW_MS)
    parser.add_argument("--max-frame-age-ms", type=float, default=DEFAULT_MAX_FRAME_AGE_MS)
    parser.add_argument("--capture-buffer-size", type=int, default=DEFAULT_CAPTURE_BUFFER_SIZE)
    parser.add_argument("--drop-skewed-groups", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-inflight-groups", type=int, default=2)
    parser.add_argument("--gpu-gate-mode", choices=GPU_GATE_MODES, default=GPU_GATE_MODE_OFF)
    parser.add_argument("--gpu-gate-max-concurrent", type=int, default=0)
    parser.add_argument("--gpu-pipeline-mode", choices=GPU_PIPELINE_MODES, default=GPU_PIPELINE_MODE_SEPARATE_WORKERS)
    parser.add_argument("--single-owner-order", choices=SINGLE_OWNER_ORDERS, default=SINGLE_OWNER_ORDER_FFS_THEN_EDGETAM)
    parser.add_argument("--staged-order", choices=STAGED_ORDERS, default=STAGED_ORDER_FFS_THEN_PARALLEL_EDGETAM)
    parser.add_argument("--stage-scheduler-mode", choices=STAGE_SCHEDULER_MODES, default=STAGE_SCHEDULER_MODE_MASK_GATED)
    parser.add_argument("--stage-lookahead", type=int, default=1)
    parser.add_argument("--edgetam-stream-mode", choices=EDGETAM_STREAM_MODES, default=EDGETAM_STREAM_MODE_DEFAULT)
    parser.add_argument("--static-device-buffers", action="store_true")
    parser.add_argument("--preallocate-pcd-buffers", action="store_true")
    parser.add_argument("--ffs-worker-mode", choices=FFS_WORKER_MODES, default="shared")
    parser.add_argument("--ffs-schedule", choices=FFS_SCHEDULES, default="strict3-latest")
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--pin-memory-mode", choices=PIN_MEMORY_MODES, default=PIN_MEMORY_MODE_OFF)
    parser.add_argument("--pinned-ring-size", type=int, default=3)
    parser.add_argument("--h2d-stream-mode", choices=H2D_STREAM_MODES, default=H2D_STREAM_MODE_DEFAULT)
    parser.add_argument("--ffs-input-staging", choices=FFS_INPUT_STAGING_MODES, default=FFS_INPUT_STAGING_PINNED)
    parser.add_argument("--edgetam-worker-mode", choices=EDGETAM_WORKER_MODES, default="per-camera")
    parser.add_argument("--edgetam-model-topology", choices=EDGETAM_MODEL_TOPOLOGIES, default="replicated")
    parser.add_argument("--ffs-repo", default=str(DEFAULT_FFS_REPO))
    parser.add_argument("--ffs-trt-model-dir", default=str(DEFAULT_FFS_TRT_TWO_STAGE_MODEL_DIR))
    parser.add_argument(
        "--ffs-trt-batch-size",
        type=int,
        choices=FFS_TRT_BATCH_SIZES,
        default=1,
        help=(
            "Static TensorRT FFS batch size. Base parser default batch=1 preserves the existing engine path; "
            "Demo 2.2 async-filter preset defaults to 3 with the isolated batch3 engine path."
        ),
    )
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
    parser.add_argument(
        "--render-backend",
        choices=RENDER_BACKENDS,
        default=DEFAULT_RENDER_BACKEND,
        help=(
            "Pointcloud renderer path. legacy-inplace keeps Open3D geometry alive and updates the latest packet; "
            "tensor-o3d-dlpack is experimental and falls back to the tensor scene path when packets are CPU arrays."
        ),
    )
    parser.add_argument(
        "--render-layer-mode",
        choices=RENDER_LAYER_MODES,
        default=DEFAULT_RENDER_LAYER_MODE,
        help=(
            "Display object/controller as one combined point cloud or separate Open3D geometries. "
            "Combined preserves points/colors and halves Open3D geometry update calls for the formal object+controller demo."
        ),
    )
    parser.add_argument(
        "--render-async-latest-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use a latest-only render buffer and coalesced GUI posts so rendering cannot queue stale frames.",
    )
    parser.add_argument(
        "--render-copy-mode",
        choices=RENDER_COPY_MODES,
        default=DEFAULT_RENDER_COPY_MODE,
        help="Render copy policy. Current Demo 2.2 fused PCD packets are CPU arrays, so this is recorded for profile comparison.",
    )
    parser.add_argument(
        "--render-micro-profile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Emit render copy/update/post-redraw timing breakdown in profile records.",
    )
    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--show-tracking-overlay", action="store_true")
    parser.add_argument(
        "--tracking-backend",
        choices=("none", "cotracker3_online", "nvofa", "tapnext", "locotrack", "tapir", "vpi_lk", "offline_npz", "cached"),
        default="none",
    )
    parser.add_argument("--tracking-source", choices=("live", "cached", "offline_npz"), default="cached")
    parser.add_argument("--tracking-num-points", type=int, default=256)
    parser.add_argument("--tracking-overlay-max-points", type=int, default=30)
    parser.add_argument("--tracking-trail-len", type=int, default=8)
    parser.add_argument("--tracking-update-hz", type=float, default=5.0)
    parser.add_argument("--tracking-depth-source", choices=("displayed", "native", "ffs"), default="displayed")
    parser.add_argument("--tracking-output-root", default="./data/experiments/demo3_live_tracking")
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
    args = apply_preset_defaults(args, explicit_options=_explicit_cli_options(argv))
    contract = build_contract(args)
    if args.dry_run:
        print(json.dumps(contract, indent=2, sort_keys=True))
        return 0
    return Demo21Runtime(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
