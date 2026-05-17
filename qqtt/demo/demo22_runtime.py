from __future__ import annotations

import time
from typing import Sequence

from qqtt.demo import render_fastpath as _render_fastpath
from qqtt.demo import three_view_masked_fused_pcd_runtime as _shared_runtime


# Demo 2.2 runtime facade. The implementation is shared with earlier three-view
# demos under qqtt.demo, but this module is the sanctioned Demo 2.2 import
# boundary. Versioned demo entrypoint folders should not import each other.
PRESET_DEMO22_ASYNC_FILTER_5FPS = _shared_runtime.PRESET_DEMO22_ASYNC_FILTER_5FPS
PRESET_DEMO22_STAGED_PARALLEL_5FPS = _shared_runtime.PRESET_DEMO22_STAGED_PARALLEL_5FPS

# Compatibility constants used by Demo 2.1.5 tests/tools. These are value
# exports from the shared runtime, not imports from the Demo 2.1.5 entrypoint.
PRESET_DEMO215_ASYNC_FILTER_5FPS = _shared_runtime.PRESET_DEMO215_ASYNC_FILTER_5FPS
PRESET_DEMO215_COMPILED_PARALLEL_EDGETAM_5FPS = _shared_runtime.PRESET_DEMO215_COMPILED_PARALLEL_EDGETAM_5FPS
PRESET_DEMO215_STAGED_PARALLEL_5FPS = _shared_runtime.PRESET_DEMO215_STAGED_PARALLEL_5FPS
PRESET_DEMO215_LIVE_FAST_NATIVE = _shared_runtime.PRESET_DEMO215_LIVE_FAST_NATIVE
PRESET_DEMO215_LIVE_QUALITY_FFS = _shared_runtime.PRESET_DEMO215_LIVE_QUALITY_FFS
PRESET_DEMO215_MASK_ONLY_DEBUG = _shared_runtime.PRESET_DEMO215_MASK_ONLY_DEBUG

TRACK_MODE_OBJECT_ONLY = _shared_runtime.TRACK_MODE_OBJECT_ONLY
TRACK_MODE_CONTROLLER_ONLY = _shared_runtime.TRACK_MODE_CONTROLLER_ONLY
TRACK_MODE_CONTROLLER_OBJECT = _shared_runtime.TRACK_MODE_CONTROLLER_OBJECT
EXPERIMENT_MODE_CONTROLLER_OBJECT = _shared_runtime.EXPERIMENT_MODE_CONTROLLER_OBJECT
EXPERIMENT_MODE_DEMO = _shared_runtime.EXPERIMENT_MODE_DEMO
EXPERIMENT_MODES = _shared_runtime.EXPERIMENT_MODES
DEFAULT_EXPERIMENT_MODE = _shared_runtime.DEFAULT_EXPERIMENT_MODE
DEFAULT_DEMO22_EXPERIMENT_MODE = _shared_runtime.DEFAULT_DEMO22_EXPERIMENT_MODE
DEMO_MODE_CONTROLLER_LABEL = _shared_runtime.DEMO_MODE_CONTROLLER_LABEL
CONTROLLER_OBJECT_EXP_CONTROLLER_LABEL = _shared_runtime.CONTROLLER_OBJECT_EXP_CONTROLLER_LABEL
DEFAULT_DEMO22_CONTROLLER_LABEL = _shared_runtime.DEFAULT_DEMO22_CONTROLLER_LABEL

DEPTH_SOURCE_FFS = _shared_runtime.DEPTH_SOURCE_FFS
DEPTH_SOURCE_REALSENSE = _shared_runtime.DEPTH_SOURCE_REALSENSE

GPU_PIPELINE_MODE_SEPARATE_WORKERS = _shared_runtime.GPU_PIPELINE_MODE_SEPARATE_WORKERS
GPU_PIPELINE_MODE_SINGLE_OWNER = _shared_runtime.GPU_PIPELINE_MODE_SINGLE_OWNER
GPU_PIPELINE_MODE_STAGED = _shared_runtime.GPU_PIPELINE_MODE_STAGED
GPU_PIPELINE_MODE_OVERLAPPED_STAGES = _shared_runtime.GPU_PIPELINE_MODE_OVERLAPPED_STAGES
SINGLE_OWNER_ORDER_FFS_THEN_EDGETAM = _shared_runtime.SINGLE_OWNER_ORDER_FFS_THEN_EDGETAM
STAGED_ORDER_FFS_THEN_PARALLEL_EDGETAM = _shared_runtime.STAGED_ORDER_FFS_THEN_PARALLEL_EDGETAM

EDGETAM_MODEL_TOPOLOGY_REPLICATED = _shared_runtime.EDGETAM_MODEL_TOPOLOGY_REPLICATED
EDGETAM_MODEL_TOPOLOGY_SHARED = _shared_runtime.EDGETAM_MODEL_TOPOLOGY_SHARED
EDGETAM_STREAM_MODE_PER_CAMERA = _shared_runtime.EDGETAM_STREAM_MODE_PER_CAMERA

PIN_MEMORY_MODE_ALL = _shared_runtime.PIN_MEMORY_MODE_ALL
H2D_STREAM_MODE_DEDICATED = _shared_runtime.H2D_STREAM_MODE_DEDICATED
POSTPROCESS_NONE = _shared_runtime.POSTPROCESS_NONE
POSTPROCESS_PT_FILTER = _shared_runtime.POSTPROCESS_PT_FILTER
POSTPROCESS_ENHANCED_PT = _shared_runtime.POSTPROCESS_ENHANCED_PT
CONTROLLER_ID = _shared_runtime.CONTROLLER_ID
OBJECT_ID = _shared_runtime.OBJECT_ID

DEFAULT_COMPILE_MODE = _shared_runtime.DEFAULT_COMPILE_MODE
COMPILE_MODE_NONE = _shared_runtime.COMPILE_MODE_NONE
COMPILE_MODE_VISION_REDUCE_OVERHEAD = _shared_runtime.COMPILE_MODE_VISION_REDUCE_OVERHEAD
COMPILE_MODE_VISION_MAX_AUTOTUNE_NO_CUDAGRAPHS = _shared_runtime.COMPILE_MODE_VISION_MAX_AUTOTUNE_NO_CUDAGRAPHS
COMPILE_MODE_COMPONENTS_REDUCE_OVERHEAD = _shared_runtime.COMPILE_MODE_COMPONENTS_REDUCE_OVERHEAD
COMPILE_MODE_COMPONENTS_MAX_AUTOTUNE_NO_CUDAGRAPHS = _shared_runtime.COMPILE_MODE_COMPONENTS_MAX_AUTOTUNE_NO_CUDAGRAPHS
MASK_POSTPROCESS_HF = _shared_runtime.MASK_POSTPROCESS_HF
MASK_POSTPROCESS_CUDA_INLINE = _shared_runtime.MASK_POSTPROCESS_CUDA_INLINE
MASK_POSTPROCESS_MODES = _shared_runtime.MASK_POSTPROCESS_MODES
DEFAULT_DEMO22_DEPTH_MIN_M = _shared_runtime.DEFAULT_DEMO22_DEPTH_MIN_M
DEFAULT_FFS_TRT_TWO_STAGE_MODEL_DIR = _shared_runtime.DEFAULT_FFS_TRT_TWO_STAGE_MODEL_DIR
DEFAULT_FFS_TRT_BATCH3_TWO_STAGE_MODEL_DIR = _shared_runtime.DEFAULT_FFS_TRT_BATCH3_TWO_STAGE_MODEL_DIR
FFS_TRT_BATCH_SIZES = _shared_runtime.FFS_TRT_BATCH_SIZES
GPU_SAMPLING_BACKENDS = _shared_runtime.GPU_SAMPLING_BACKENDS
COMPILE_MODE_VISION_DEFAULT = _shared_runtime.COMPILE_MODE_VISION_DEFAULT

RENDER_BACKENDS = _render_fastpath.RENDER_BACKENDS
RENDER_COPY_MODES = _render_fastpath.RENDER_COPY_MODES
RENDER_LAYER_MODES = _render_fastpath.RENDER_LAYER_MODES
DEFAULT_RENDER_BACKEND = _render_fastpath.DEFAULT_RENDER_BACKEND
DEFAULT_RENDER_COPY_MODE = _render_fastpath.DEFAULT_RENDER_COPY_MODE
DEFAULT_RENDER_LAYER_MODE = _render_fastpath.DEFAULT_RENDER_LAYER_MODE

CameraFramePacket = _shared_runtime.CameraFramePacket
CameraIntrinsics = _shared_runtime.CameraIntrinsics
CaptureGroup = _shared_runtime.CaptureGroup
DepthGroup = _shared_runtime.DepthGroup
FusedLayerCloud = _shared_runtime.FusedLayerCloud
MaskGroup = _shared_runtime.MaskGroup
RawFusedPcdPacket = _shared_runtime.RawFusedPcdPacket
SameGroupJoinBuffer = _shared_runtime.SameGroupJoinBuffer

class Demo22Runtime(_shared_runtime.Demo21Runtime):
    """Demo 2.2 named facade over the shared three-view runtime."""

elapsed_ms = _shared_runtime._elapsed_ms
explicit_cli_options = _shared_runtime._explicit_cli_options
summarize_gpu_samples = _shared_runtime.summarize_gpu_samples
controller_prompt_for_experiment_mode = _shared_runtime.controller_prompt_for_experiment_mode
resolved_experiment_mode = _shared_runtime.resolved_experiment_mode
controller_prompt_matches_experiment_mode = _shared_runtime.controller_prompt_matches_experiment_mode


def parse_camera_ids(value: str) -> tuple[int, ...]:
    return _shared_runtime.parse_camera_ids(value)


def build_arg_parser():
    return _shared_runtime.build_arg_parser()


def apply_preset_defaults(args, *, explicit_options: set[str] | None = None):
    return _shared_runtime.apply_preset_defaults(args, explicit_options=explicit_options)


def build_contract(args) -> dict:
    return _shared_runtime.build_contract(args)


def main(argv: Sequence[str] | None = None) -> int:
    return _shared_runtime.main(argv)


__all__ = [
    "CameraFramePacket",
    "CameraIntrinsics",
    "CaptureGroup",
    "COMPILE_MODE_VISION_DEFAULT",
    "COMPILE_MODE_VISION_REDUCE_OVERHEAD",
    "COMPILE_MODE_VISION_MAX_AUTOTUNE_NO_CUDAGRAPHS",
    "COMPILE_MODE_COMPONENTS_REDUCE_OVERHEAD",
    "COMPILE_MODE_COMPONENTS_MAX_AUTOTUNE_NO_CUDAGRAPHS",
    "COMPILE_MODE_NONE",
    "CONTROLLER_ID",
    "CONTROLLER_OBJECT_EXP_CONTROLLER_LABEL",
    "DEFAULT_COMPILE_MODE",
    "DEFAULT_DEMO22_DEPTH_MIN_M",
    "DEFAULT_DEMO22_CONTROLLER_LABEL",
    "DEFAULT_DEMO22_EXPERIMENT_MODE",
    "DEFAULT_EXPERIMENT_MODE",
    "DEFAULT_FFS_TRT_BATCH3_TWO_STAGE_MODEL_DIR",
    "DEFAULT_FFS_TRT_TWO_STAGE_MODEL_DIR",
    "DEPTH_SOURCE_FFS",
    "DEPTH_SOURCE_REALSENSE",
    "Demo22Runtime",
    "EDGETAM_MODEL_TOPOLOGY_REPLICATED",
    "EDGETAM_MODEL_TOPOLOGY_SHARED",
    "EDGETAM_STREAM_MODE_PER_CAMERA",
    "EXPERIMENT_MODE_CONTROLLER_OBJECT",
    "EXPERIMENT_MODE_DEMO",
    "EXPERIMENT_MODES",
    "FFS_TRT_BATCH_SIZES",
    "FusedLayerCloud",
    "DepthGroup",
    "GPU_PIPELINE_MODE_SEPARATE_WORKERS",
    "GPU_PIPELINE_MODE_SINGLE_OWNER",
    "GPU_PIPELINE_MODE_STAGED",
    "GPU_PIPELINE_MODE_OVERLAPPED_STAGES",
    "GPU_SAMPLING_BACKENDS",
    "H2D_STREAM_MODE_DEDICATED",
    "MASK_POSTPROCESS_CUDA_INLINE",
    "MASK_POSTPROCESS_HF",
    "MASK_POSTPROCESS_MODES",
    "OBJECT_ID",
    "PIN_MEMORY_MODE_ALL",
    "POSTPROCESS_ENHANCED_PT",
    "POSTPROCESS_NONE",
    "POSTPROCESS_PT_FILTER",
    "PRESET_DEMO215_ASYNC_FILTER_5FPS",
    "PRESET_DEMO215_COMPILED_PARALLEL_EDGETAM_5FPS",
    "PRESET_DEMO215_LIVE_FAST_NATIVE",
    "PRESET_DEMO215_LIVE_QUALITY_FFS",
    "PRESET_DEMO215_MASK_ONLY_DEBUG",
    "PRESET_DEMO215_STAGED_PARALLEL_5FPS",
    "PRESET_DEMO22_ASYNC_FILTER_5FPS",
    "PRESET_DEMO22_STAGED_PARALLEL_5FPS",
    "RawFusedPcdPacket",
    "MaskGroup",
    "SameGroupJoinBuffer",
    "DEFAULT_RENDER_BACKEND",
    "DEFAULT_RENDER_COPY_MODE",
    "DEFAULT_RENDER_LAYER_MODE",
    "RENDER_BACKENDS",
    "RENDER_COPY_MODES",
    "RENDER_LAYER_MODES",
    "SINGLE_OWNER_ORDER_FFS_THEN_EDGETAM",
    "STAGED_ORDER_FFS_THEN_PARALLEL_EDGETAM",
    "TRACK_MODE_CONTROLLER_OBJECT",
    "TRACK_MODE_CONTROLLER_ONLY",
    "TRACK_MODE_OBJECT_ONLY",
    "apply_preset_defaults",
    "build_arg_parser",
    "build_contract",
    "controller_prompt_for_experiment_mode",
    "controller_prompt_matches_experiment_mode",
    "elapsed_ms",
    "explicit_cli_options",
    "main",
    "parse_camera_ids",
    "resolved_experiment_mode",
    "summarize_gpu_samples",
    "time",
]
