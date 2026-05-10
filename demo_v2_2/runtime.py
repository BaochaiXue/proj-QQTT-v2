from __future__ import annotations

import time
from typing import Sequence

from demo_v2_1 import realtime_three_view_masked_fused_pcd as _legacy


# Demo 2.2 / 2.1.5 public presets. Keep this module as the import boundary for
# new demos so public wrappers do not depend on the Demo 2.1 entrypoint.
PRESET_DEMO22_ASYNC_FILTER_5FPS = _legacy.PRESET_DEMO22_ASYNC_FILTER_5FPS
PRESET_DEMO22_STAGED_PARALLEL_5FPS = _legacy.PRESET_DEMO22_STAGED_PARALLEL_5FPS
PRESET_DEMO215_ASYNC_FILTER_5FPS = _legacy.PRESET_DEMO215_ASYNC_FILTER_5FPS
PRESET_DEMO215_COMPILED_PARALLEL_EDGETAM_5FPS = _legacy.PRESET_DEMO215_COMPILED_PARALLEL_EDGETAM_5FPS
PRESET_DEMO215_STAGED_PARALLEL_5FPS = _legacy.PRESET_DEMO215_STAGED_PARALLEL_5FPS

TRACK_MODE_OBJECT_ONLY = _legacy.TRACK_MODE_OBJECT_ONLY
TRACK_MODE_CONTROLLER_OBJECT = _legacy.TRACK_MODE_CONTROLLER_OBJECT

DEPTH_SOURCE_FFS = _legacy.DEPTH_SOURCE_FFS
DEPTH_SOURCE_REALSENSE = _legacy.DEPTH_SOURCE_REALSENSE

GPU_PIPELINE_MODE_SEPARATE_WORKERS = _legacy.GPU_PIPELINE_MODE_SEPARATE_WORKERS
GPU_PIPELINE_MODE_SINGLE_OWNER = _legacy.GPU_PIPELINE_MODE_SINGLE_OWNER
GPU_PIPELINE_MODE_STAGED = _legacy.GPU_PIPELINE_MODE_STAGED
SINGLE_OWNER_ORDER_FFS_THEN_EDGETAM = _legacy.SINGLE_OWNER_ORDER_FFS_THEN_EDGETAM
STAGED_ORDER_FFS_THEN_PARALLEL_EDGETAM = _legacy.STAGED_ORDER_FFS_THEN_PARALLEL_EDGETAM

EDGETAM_MODEL_TOPOLOGY_REPLICATED = _legacy.EDGETAM_MODEL_TOPOLOGY_REPLICATED
EDGETAM_MODEL_TOPOLOGY_SHARED = _legacy.EDGETAM_MODEL_TOPOLOGY_SHARED
EDGETAM_STREAM_MODE_PER_CAMERA = _legacy.EDGETAM_STREAM_MODE_PER_CAMERA

PIN_MEMORY_MODE_ALL = _legacy.PIN_MEMORY_MODE_ALL
H2D_STREAM_MODE_DEDICATED = _legacy.H2D_STREAM_MODE_DEDICATED
POSTPROCESS_NONE = _legacy.POSTPROCESS_NONE

DEFAULT_COMPILE_MODE = _legacy.DEFAULT_COMPILE_MODE
COMPILE_MODE_NONE = _legacy.COMPILE_MODE_NONE
DEFAULT_DEMO22_DEPTH_MIN_M = _legacy.DEFAULT_DEMO22_DEPTH_MIN_M
DEFAULT_FFS_TRT_TWO_STAGE_MODEL_DIR = _legacy.DEFAULT_FFS_TRT_TWO_STAGE_MODEL_DIR
DEFAULT_FFS_TRT_BATCH3_TWO_STAGE_MODEL_DIR = _legacy.DEFAULT_FFS_TRT_BATCH3_TWO_STAGE_MODEL_DIR
FFS_TRT_BATCH_SIZES = _legacy.FFS_TRT_BATCH_SIZES
GPU_SAMPLING_BACKENDS = _legacy.GPU_SAMPLING_BACKENDS
COMPILE_MODE_VISION_DEFAULT = _legacy.COMPILE_MODE_VISION_DEFAULT

CameraFramePacket = _legacy.CameraFramePacket
CameraIntrinsics = _legacy.CameraIntrinsics
CaptureGroup = _legacy.CaptureGroup
FusedLayerCloud = _legacy.FusedLayerCloud
RawFusedPcdPacket = _legacy.RawFusedPcdPacket

Demo22Runtime = _legacy.Demo21Runtime

elapsed_ms = _legacy._elapsed_ms
explicit_cli_options = _legacy._explicit_cli_options
summarize_gpu_samples = _legacy.summarize_gpu_samples


def parse_camera_ids(value: str) -> tuple[int, ...]:
    return _legacy.parse_camera_ids(value)


def build_arg_parser():
    return _legacy.build_arg_parser()


def apply_preset_defaults(args, *, explicit_options: set[str] | None = None):
    return _legacy.apply_preset_defaults(args, explicit_options=explicit_options)


def build_contract(args) -> dict:
    return _legacy.build_contract(args)


def main(argv: Sequence[str] | None = None) -> int:
    return _legacy.main(argv)


__all__ = [
    "CameraFramePacket",
    "CameraIntrinsics",
    "CaptureGroup",
    "COMPILE_MODE_VISION_DEFAULT",
    "COMPILE_MODE_NONE",
    "DEFAULT_COMPILE_MODE",
    "DEFAULT_DEMO22_DEPTH_MIN_M",
    "DEFAULT_FFS_TRT_BATCH3_TWO_STAGE_MODEL_DIR",
    "DEFAULT_FFS_TRT_TWO_STAGE_MODEL_DIR",
    "DEPTH_SOURCE_FFS",
    "DEPTH_SOURCE_REALSENSE",
    "Demo22Runtime",
    "EDGETAM_MODEL_TOPOLOGY_REPLICATED",
    "EDGETAM_MODEL_TOPOLOGY_SHARED",
    "EDGETAM_STREAM_MODE_PER_CAMERA",
    "FFS_TRT_BATCH_SIZES",
    "FusedLayerCloud",
    "GPU_PIPELINE_MODE_SEPARATE_WORKERS",
    "GPU_PIPELINE_MODE_SINGLE_OWNER",
    "GPU_PIPELINE_MODE_STAGED",
    "GPU_SAMPLING_BACKENDS",
    "H2D_STREAM_MODE_DEDICATED",
    "PIN_MEMORY_MODE_ALL",
    "POSTPROCESS_NONE",
    "PRESET_DEMO215_ASYNC_FILTER_5FPS",
    "PRESET_DEMO215_COMPILED_PARALLEL_EDGETAM_5FPS",
    "PRESET_DEMO215_STAGED_PARALLEL_5FPS",
    "PRESET_DEMO22_ASYNC_FILTER_5FPS",
    "PRESET_DEMO22_STAGED_PARALLEL_5FPS",
    "RawFusedPcdPacket",
    "SINGLE_OWNER_ORDER_FFS_THEN_EDGETAM",
    "STAGED_ORDER_FFS_THEN_PARALLEL_EDGETAM",
    "TRACK_MODE_CONTROLLER_OBJECT",
    "TRACK_MODE_OBJECT_ONLY",
    "apply_preset_defaults",
    "build_arg_parser",
    "build_contract",
    "elapsed_ms",
    "explicit_cli_options",
    "main",
    "parse_camera_ids",
    "summarize_gpu_samples",
    "time",
]
