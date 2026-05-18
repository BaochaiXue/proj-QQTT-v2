from __future__ import annotations

from typing import Sequence

from qqtt.demo import render_fastpath as _render_fastpath
from qqtt.demo import three_view_masked_fused_pcd_runtime as _shared_runtime


# Demo 2.3 runtime facade. Versioned entrypoints import this boundary rather
# than importing Demo 2.2 files.
PRESET_DEMO23_DUAL4090_MAXFPS = _shared_runtime.PRESET_DEMO23_DUAL4090_MAXFPS
PRESET_DEMO22_ASYNC_FILTER_5FPS = _shared_runtime.PRESET_DEMO22_ASYNC_FILTER_5FPS
PRESET_DEMO22_STAGED_PARALLEL_5FPS = _shared_runtime.PRESET_DEMO22_STAGED_PARALLEL_5FPS

TRACK_MODE_OBJECT_ONLY = _shared_runtime.TRACK_MODE_OBJECT_ONLY
TRACK_MODE_CONTROLLER_ONLY = _shared_runtime.TRACK_MODE_CONTROLLER_ONLY
TRACK_MODE_CONTROLLER_OBJECT = _shared_runtime.TRACK_MODE_CONTROLLER_OBJECT
EXPERIMENT_MODE_CONTROLLER_OBJECT = _shared_runtime.EXPERIMENT_MODE_CONTROLLER_OBJECT
EXPERIMENT_MODE_DEMO = _shared_runtime.EXPERIMENT_MODE_DEMO
EXPERIMENT_MODES = _shared_runtime.EXPERIMENT_MODES
DEFAULT_DEMO22_EXPERIMENT_MODE = _shared_runtime.DEFAULT_DEMO22_EXPERIMENT_MODE
CONTROLLER_OBJECT_EXP_CONTROLLER_LABEL = _shared_runtime.CONTROLLER_OBJECT_EXP_CONTROLLER_LABEL

DEPTH_SOURCE_FFS = _shared_runtime.DEPTH_SOURCE_FFS
GPU_PIPELINE_MODE_SEPARATE_WORKERS = _shared_runtime.GPU_PIPELINE_MODE_SEPARATE_WORKERS
GPU_PIPELINE_MODE_SINGLE_OWNER = _shared_runtime.GPU_PIPELINE_MODE_SINGLE_OWNER
GPU_PIPELINE_MODE_STAGED = _shared_runtime.GPU_PIPELINE_MODE_STAGED
GPU_PIPELINE_MODE_OVERLAPPED_STAGES = _shared_runtime.GPU_PIPELINE_MODE_OVERLAPPED_STAGES
GPU_PIPELINE_MODE_DUAL_GPU_SPLIT = _shared_runtime.GPU_PIPELINE_MODE_DUAL_GPU_SPLIT
STAGE_SCHEDULER_MODES = _shared_runtime.STAGE_SCHEDULER_MODES

FFS_TRT_BATCH_SIZES = _shared_runtime.FFS_TRT_BATCH_SIZES
GPU_SAMPLING_BACKENDS = _shared_runtime.GPU_SAMPLING_BACKENDS
POSTPROCESS_NONE = _shared_runtime.POSTPROCESS_NONE
POSTPROCESS_PT_FILTER = _shared_runtime.POSTPROCESS_PT_FILTER
POSTPROCESS_ENHANCED_PT = _shared_runtime.POSTPROCESS_ENHANCED_PT

DEFAULT_FFS_TRT_BATCH3_TWO_STAGE_MODEL_DIR = _shared_runtime.DEFAULT_FFS_TRT_BATCH3_TWO_STAGE_MODEL_DIR

RENDER_BACKENDS = _render_fastpath.RENDER_BACKENDS
RENDER_COPY_MODES = _render_fastpath.RENDER_COPY_MODES
RENDER_LAYER_MODES = _render_fastpath.RENDER_LAYER_MODES

CameraFramePacket = _shared_runtime.CameraFramePacket
CameraIntrinsics = _shared_runtime.CameraIntrinsics
CameraMaskPacket = _shared_runtime.CameraMaskPacket
CaptureGroup = _shared_runtime.CaptureGroup
DepthGroup = _shared_runtime.DepthGroup
DepthPacket = _shared_runtime.DepthPacket
FusedLayerCloud = _shared_runtime.FusedLayerCloud
MaskGroup = _shared_runtime.MaskGroup
RawFusedPcdPacket = _shared_runtime.RawFusedPcdPacket
SameGroupJoinBuffer = _shared_runtime.SameGroupJoinBuffer
StageTask = _shared_runtime.StageTask
StageWindowScheduler = _shared_runtime.StageWindowScheduler


class Demo23Runtime(_shared_runtime.Demo21Runtime):
    """Demo 2.3 dual-GPU runtime facade."""


class Demo23WorkerRuntime(_shared_runtime.Demo21Runtime):
    """Worker-local runtime used inside Demo 2.3 GPU subprocesses."""


explicit_cli_options = _shared_runtime._explicit_cli_options
elapsed_ms = _shared_runtime._elapsed_ms
controller_prompt_for_experiment_mode = _shared_runtime.controller_prompt_for_experiment_mode
controller_prompt_matches_experiment_mode = _shared_runtime.controller_prompt_matches_experiment_mode
resolved_experiment_mode = _shared_runtime.resolved_experiment_mode
summarize_gpu_samples = _shared_runtime.summarize_gpu_samples
summarize_gpu_samples_by_device = _shared_runtime.summarize_gpu_samples_by_device


def parse_camera_ids(value: str) -> tuple[int, ...]:
    return _shared_runtime.parse_camera_ids(value)


def parse_gpu_sampling_device_indexes(value: str) -> tuple[int, ...]:
    return _shared_runtime.parse_gpu_sampling_device_indexes(value)


def build_arg_parser():
    return _shared_runtime.build_arg_parser()


def apply_preset_defaults(args, *, explicit_options: set[str] | None = None):
    return _shared_runtime.apply_preset_defaults(args, explicit_options=explicit_options)


def build_contract(args) -> dict:
    return _shared_runtime.build_contract(args)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    args = apply_preset_defaults(args, explicit_options=explicit_cli_options(argv))
    contract = build_contract(args)
    if args.dry_run:
        import json

        print(json.dumps(contract, indent=2, sort_keys=True))
        return 0
    return Demo23Runtime(args).run()


__all__ = [
    "CameraFramePacket",
    "CameraIntrinsics",
    "CameraMaskPacket",
    "CaptureGroup",
    "DepthGroup",
    "DepthPacket",
    "Demo23Runtime",
    "Demo23WorkerRuntime",
    "EXPERIMENT_MODE_CONTROLLER_OBJECT",
    "EXPERIMENT_MODE_DEMO",
    "EXPERIMENT_MODES",
    "FFS_TRT_BATCH_SIZES",
    "FusedLayerCloud",
    "GPU_PIPELINE_MODE_DUAL_GPU_SPLIT",
    "GPU_PIPELINE_MODE_OVERLAPPED_STAGES",
    "GPU_PIPELINE_MODE_SEPARATE_WORKERS",
    "GPU_PIPELINE_MODE_SINGLE_OWNER",
    "GPU_PIPELINE_MODE_STAGED",
    "GPU_SAMPLING_BACKENDS",
    "MaskGroup",
    "POSTPROCESS_ENHANCED_PT",
    "POSTPROCESS_NONE",
    "POSTPROCESS_PT_FILTER",
    "PRESET_DEMO22_ASYNC_FILTER_5FPS",
    "PRESET_DEMO22_STAGED_PARALLEL_5FPS",
    "PRESET_DEMO23_DUAL4090_MAXFPS",
    "RawFusedPcdPacket",
    "RENDER_BACKENDS",
    "RENDER_COPY_MODES",
    "RENDER_LAYER_MODES",
    "SameGroupJoinBuffer",
    "STAGE_SCHEDULER_MODES",
    "StageTask",
    "StageWindowScheduler",
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
    "parse_gpu_sampling_device_indexes",
    "resolved_experiment_mode",
    "summarize_gpu_samples",
    "summarize_gpu_samples_by_device",
]
