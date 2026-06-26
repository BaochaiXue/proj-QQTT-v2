#!/usr/bin/env python3
"""Demo v5 realtime orchestration entrypoint.

This runner owns process boundaries, GPU routing, and artifact publication. The
actual camera/tracker stack runs in ``demo_v5/realtime_dense_track.py``;
SAM3D shape prior work can run in a separate managed worker; the default
side-by-side point viewer starts as soon as capture starts, while optional
realtime_phystwin optimization still starts only after a committed chunk.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime
from typing import Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT_STR = str(REPO_ROOT)
if REPO_ROOT_STR in sys.path:
    sys.path.remove(REPO_ROOT_STR)
sys.path.insert(0, REPO_ROOT_STR)

from demo_v5.realtime_data_process_track import stream_chunks_from_headless_capture, write_chunks_from_headless_capture
from demo_v5.chunked_final_data_aggregate import migrate_legacy_online_static_case


DEFAULT_DATA_PROCESS_BASE_PATH = Path("result/demo_v5/data_process_sam3d_chunks")
DEFAULT_REALTIME_PHYSTWIN_ROOT = Path("realtime_phystwin")
DEFAULT_INPUT_SOURCE = "fake-live"
DEFAULT_REPLAY_FPS = 5.0
DEFAULT_CHUNK_SECONDS = 7.0
DEFAULT_CHUNK_POLL_INTERVAL_S = 0.001
DEFAULT_CAMERA_LOSSLESS_INPUT_FPS = 5.0
DEFAULT_CASE_PREFIX = "demo_v5"
DEFAULT_DEPTH_BACKEND = "native-realsense"
DEFAULT_MAX_CHUNKS: int | None = None
DEFAULT_CAPTURE_EXTRA_SECONDS = 10.0
DEFAULT_SHAPE_PRIOR_ENDPOINT = "tcp://127.0.0.1:7100"
DEFAULT_MASK_RADIUS_OUTLIER_RADIUS_M = 0.01
DEFAULT_MASK_RADIUS_OUTLIER_NB_POINTS = 40
DEFAULT_REALTIME_GPU_MODE = "single"
DEFAULT_WARMUP_GPU_MODE = "dual"
DEFAULT_GPU_MODE = DEFAULT_REALTIME_GPU_MODE
GPU_MODE_CAMERA_CUDA_VISIBLE_DEVICES = {
    "single": "0",
    "dual": "1",
}
GPU_MODE_SHAPE_PRIOR_DEVICE = {
    "single": "cuda:0",
    "dual": "cuda:1",
}
DEFAULT_CAMERA_DEVICE = "cuda"
DEFAULT_CAMERA_TRACKER_DEVICE = "cuda"
DEFAULT_CAMERA_DTYPE = "bfloat16"
DEFAULT_SHAPE_PRIOR_WORKER_MODE = "managed"
DEFAULT_SHAPE_PRIOR_WORKER_CONDA_ENV = "phystwin-max"
DEFAULT_SHAPE_PRIOR_WORKER_DEVICE = "cuda:0"
DEFAULT_SHAPE_PRIOR_WORKER_STARTUP_GRACE_S = 0.0
DEFAULT_SHAPE_PRIOR_WORKER_MAX_OBSERVATION_TO_ALIGNED_P95_M = 0.06
DEFAULT_OPTIMIZATION_MODE = "disabled"
DEFAULT_OPTIMIZATION_CUDA_VISIBLE_DEVICES = "1"
DEFAULT_OPTIMIZATION_DEVICE = "cuda:0"
DEFAULT_OPTIMIZATION_ZERO_ITERATIONS = 10
DEFAULT_OPTIMIZATION_BATCH_SIZE = 4
DEFAULT_OPTIMIZATION_SEGMENT_STRIDE = 16
DEFAULT_OPTIMIZATION_POLL_SEC = 1.0
DEFAULT_OPTIMIZATION_RECENT_WINDOW_COUNT = 8
DEFAULT_OPTIMIZATION_SEED = 42
DEFAULT_OPTIMIZATION_EXPERIMENTS_DIR = "experiments_online_v5"
DEFAULT_OPTIMIZATION_ZERO_EXPERIMENTS_DIR = "experiments_online_v5_cma"
DEFAULT_OPTIMIZATION_START_GRACE_S = 2.0
DEFAULT_POINT_VIEWER_MODE = "window"
DEFAULT_POINT_VIEWER_CONDA_ENV = "demo_2_max"
DEFAULT_POINT_VIEWER_CUDA_VISIBLE_DEVICES = "1"
DEFAULT_POINT_VIEWER_CAM_IDX = 0
DEFAULT_POINT_VIEWER_POLL_SEC = 0.1
DEFAULT_POINT_VIEWER_OBJECT_STRIDE = 1
DEFAULT_POINT_VIEWER_OBJECT_RADIUS = 3
DEFAULT_POINT_VIEWER_CONTROLLER_RADIUS = 6
DEFAULT_POINT_VIEWER_OBJECT_COLOR_MODE = "rainbow"
POINT_VIEWER_LAYOUT_SIDE_BY_SIDE = "side-by-side"
POINT_VIEWER_LAYOUT_OUTPUT_ONLY = "output-only"
POINT_VIEWER_LAYOUTS = (POINT_VIEWER_LAYOUT_SIDE_BY_SIDE, POINT_VIEWER_LAYOUT_OUTPUT_ONLY)
DEFAULT_POINT_VIEWER_LAYOUT = POINT_VIEWER_LAYOUT_SIDE_BY_SIDE
DEFAULT_POINT_VIEWER_RENDER_MODE = "sam3d-final-data"
DEFAULT_TABLE_CALIBRATE_PATH = Path("table_calibrate.pkl")
DEFAULT_SAM31_CHECKPOINT_PATH = Path("vendor/demo_runtime/checkpoints/sam31/sam3.1_multiplex.pt")
SAM31_CHECKPOINT_ENV = "QQTT_SAM31_CHECKPOINT"


def _apply_default_sam31_checkpoint_env(env: dict[str, str]) -> None:
    """Prefer the vendored SAM 3.1 checkpoint without overriding callers."""
    if env.get(SAM31_CHECKPOINT_ENV):
        return
    checkpoint = REPO_ROOT / DEFAULT_SAM31_CHECKPOINT_PATH
    if checkpoint.is_file():
        env[SAM31_CHECKPOINT_ENV] = str(checkpoint)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Demo v5 realtime data_process_sam3d runner. It turns Demo v5 "
            "single-camera fake/live capture into one online data_process_sam3d "
            "case and can launch an online point viewer or continuous "
            "realtime_phystwin optimization."
        )
    )
    parser.add_argument(
        "--input-source",
        choices=("fake-live", "live"),
        default=DEFAULT_INPUT_SOURCE,
        help="Camera source mode used when Demo v5 launches its own capture.",
    )
    parser.add_argument("--replay-fps", type=float, default=DEFAULT_REPLAY_FPS)
    parser.add_argument(
        "--camera-source-replay-fps",
        type=float,
        default=None,
        help=(
            "Optional Demo v5 fake-live pacing FPS. When omitted, Demo v5 uses "
            "--replay-fps; Demo v5 output metadata/window math still use --replay-fps."
        ),
    )
    parser.add_argument("--chunk-seconds", type=float, default=DEFAULT_CHUNK_SECONDS)
    parser.add_argument(
        "--chunk-poll-interval-s",
        type=float,
        default=DEFAULT_CHUNK_POLL_INTERVAL_S,
        help="Polling interval for realtime frames.jsonl chunk tailing.",
    )
    parser.add_argument("--depth-backend", choices=("ir-ffs", "native-realsense"), default=DEFAULT_DEPTH_BACKEND)
    parser.add_argument(
        "--chunk-frame-count",
        type=int,
        default=None,
        help="Override chunk length in frames. Defaults to round(replay_fps * chunk_seconds).",
    )
    parser.add_argument(
        "--allow-degraded-online",
        action="store_true",
        help="Append degraded track-process chunks to online_data. Invalid chunks are always diagnostic-only.",
    )
    parser.add_argument("--base-path", type=Path, default=DEFAULT_DATA_PROCESS_BASE_PATH)
    parser.add_argument("--futurephystwin-base-path", dest="base_path", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--case-prefix", default=DEFAULT_CASE_PREFIX)
    parser.add_argument(
        "--gpu-mode",
        choices=tuple(GPU_MODE_CAMERA_CUDA_VISIBLE_DEVICES),
        default=DEFAULT_REALTIME_GPU_MODE,
        help=(
            "Backward-compatible realtime GPU routing preset. Prefer "
            "--realtime-gpu-mode for new experiments."
        ),
    )
    parser.add_argument(
        "--realtime-gpu-mode",
        choices=tuple(GPU_MODE_CAMERA_CUDA_VISIBLE_DEVICES),
        default=None,
        help="GPU routing preset for the Demo v5 camera/fake-camera -> final-data realtime subprocess.",
    )
    parser.add_argument(
        "--warmup-gpu-mode",
        choices=tuple(GPU_MODE_SHAPE_PRIOR_DEVICE),
        default=DEFAULT_WARMUP_GPU_MODE,
        help="GPU routing preset for SAM3D shape-prior warmup device selection.",
    )
    parser.add_argument(
        "--camera-cuda-visible-devices",
        default=None,
        help="Explicit CUDA_VISIBLE_DEVICES override for the Demo v5 subprocess.",
    )
    parser.add_argument(
        "--camera-device",
        default=DEFAULT_CAMERA_DEVICE,
        help="Segmentation/runtime device passed to Demo v5 inside the subprocess CUDA namespace.",
    )
    parser.add_argument(
        "--camera-tracker-device",
        default=DEFAULT_CAMERA_TRACKER_DEVICE,
        help="TAPNext++ tracker device passed to Demo v5 inside the subprocess CUDA namespace.",
    )
    parser.add_argument(
        "--camera-dtype",
        choices=("bfloat16", "float16", "float32"),
        default=DEFAULT_CAMERA_DTYPE,
        help="Segmentation/runtime dtype passed to Demo v5 inside the subprocess CUDA namespace.",
    )
    parser.add_argument(
        "--camera-lossless-max-backlog-seconds",
        type=float,
        default=None,
        help=(
            "Optional strict lossless replay backlog window passed to Demo v5. "
            "Omit it to keep Demo v5 defaults."
        ),
    )
    parser.add_argument(
        "--camera-headless-prepared-only",
        dest="camera_headless_prepared_only",
        action="store_true",
        help="Ask Demo v5 to write only prepared PhysTwin frames needed by Demo v5 chunking.",
    )
    parser.add_argument(
        "--camera-legacy-headless-artifacts",
        dest="camera_headless_prepared_only",
        action="store_false",
        help="Keep Demo v5 legacy per-frame headless artifacts in addition to prepared PhysTwin frames.",
    )
    parser.set_defaults(camera_headless_prepared_only=True)
    parser.add_argument(
        "--write-input-rgb-timeline",
        dest="write_input_rgb_timeline",
        action="store_true",
        default=None,
        help="Write input_rgb/*.png and input_frames.jsonl for the Demo v5 side-by-side realtime viewer.",
    )
    parser.add_argument(
        "--no-write-input-rgb-timeline",
        dest="write_input_rgb_timeline",
        action="store_false",
        help="Disable the side-by-side input RGB timeline even when the viewer layout is side-by-side.",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=DEFAULT_MAX_CHUNKS,
        help=(
            "Optional realtime chunk cap for debug/short validation runs. "
            "Omit it to stream until the fake-live recording or live capture ends."
        ),
    )
    parser.add_argument(
        "--capture-extra-seconds",
        type=float,
        default=DEFAULT_CAPTURE_EXTRA_SECONDS,
        help="Extra Demo v5 runtime beyond max_chunks*chunk_seconds to absorb startup/warmup latency.",
    )
    parser.add_argument(
        "--camera-capture-dir",
        type=Path,
        default=None,
        help="Headless capture directory for the Demo v5 realtime subprocess.",
    )
    parser.add_argument(
        "--source-headless-capture",
        type=Path,
        default=None,
        help="Existing Demo v5 headless capture directory to chunk without launching capture.",
    )
    parser.add_argument("--surface-points-npy", type=Path, default=None)
    parser.add_argument("--interior-points-npy", type=Path, default=None)
    parser.add_argument(
        "--write-final-pcd",
        dest="write_final_pcd",
        action="store_true",
        help="Write dense per-frame pcd/*.npz into each published chunk case for diagnostics/export.",
    )
    parser.add_argument(
        "--no-write-final-pcd",
        dest="write_final_pcd",
        action="store_false",
        help="Skip dense per-frame pcd/*.npz in chunk cases; final_data/tracking/mask/color remain complete.",
    )
    parser.set_defaults(write_final_pcd=False)
    parser.add_argument(
        "--shape-prior-warmup",
        dest="shape_prior_warmup",
        action="store_true",
        help="Keep SAM3D shape-prior warmup enabled for Demo v5 capture.",
    )
    parser.add_argument(
        "--no-shape-prior-warmup",
        dest="shape_prior_warmup",
        action="store_false",
        help="Disable SAM3D shape-prior warmup.",
    )
    parser.set_defaults(shape_prior_warmup=True)
    parser.add_argument(
        "--shape-prior-start-policy",
        choices=(
            "async-after-first-mask-depth-pair",
            "async-after-first-strict-pair",
            "blocking-before-first-output",
            "after-teardown",
        ),
        default="async-after-first-mask-depth-pair",
    )
    parser.add_argument(
        "--shape-prior-execution",
        choices=("remote-worker", "local-subprocess"),
        default="remote-worker",
    )
    parser.add_argument("--shape-prior-endpoint", default=DEFAULT_SHAPE_PRIOR_ENDPOINT)
    parser.add_argument("--shape-prior-timeout-ms", type=int, default=180000)
    parser.add_argument(
        "--shape-prior-chunk-wait-timeout-s",
        type=float,
        default=300.0,
        help="How long Demo v5 waits for required shape-prior structure points before writing final_data chunks.",
    )
    parser.add_argument(
        "--shape-prior-device",
        default=None,
        help="Explicit shape-prior device override. Defaults from --warmup-gpu-mode.",
    )
    parser.add_argument("--shape-prior-profile-json", type=Path, default=None)
    parser.add_argument(
        "--mask-radius-outlier-filter",
        dest="mask_radius_outlier_filter",
        action="store_true",
        help="Apply data_process_sam3d-style 3D mask radius-outlier refinement before final_data chunking.",
    )
    parser.add_argument(
        "--no-mask-radius-outlier-filter",
        dest="mask_radius_outlier_filter",
        action="store_false",
        help="Disable 3D mask radius-outlier refinement. Intended for tiny synthetic fixtures only.",
    )
    parser.set_defaults(mask_radius_outlier_filter=True)
    parser.add_argument("--mask-radius-outlier-radius-m", type=float, default=DEFAULT_MASK_RADIUS_OUTLIER_RADIUS_M)
    parser.add_argument("--mask-radius-outlier-nb-points", type=int, default=DEFAULT_MASK_RADIUS_OUTLIER_NB_POINTS)
    parser.add_argument(
        "--shape-prior-worker-mode",
        choices=("managed", "external", "disabled"),
        default=DEFAULT_SHAPE_PRIOR_WORKER_MODE,
        help=(
            "SAM3D worker lifecycle for remote-worker warmup. managed starts "
            "services/shape_prior_remote/server.py and releases it before GPU1 "
            "optimization starts."
        ),
    )
    parser.add_argument("--shape-prior-worker-conda-env", default=DEFAULT_SHAPE_PRIOR_WORKER_CONDA_ENV)
    parser.add_argument("--shape-prior-worker-cuda-visible-devices", default=None)
    parser.add_argument("--shape-prior-worker-device", default=DEFAULT_SHAPE_PRIOR_WORKER_DEVICE)
    parser.add_argument(
        "--shape-prior-worker-startup-grace-s",
        type=float,
        default=DEFAULT_SHAPE_PRIOR_WORKER_STARTUP_GRACE_S,
    )
    parser.add_argument("--shape-prior-worker-sam3d-root", type=Path, default=None)
    parser.add_argument("--shape-prior-worker-futurephystwin-root", type=Path, default=None)
    parser.add_argument("--shape-prior-worker-config", type=Path, default=None)
    parser.add_argument(
        "--shape-prior-worker-max-observation-to-aligned-p95-m",
        type=float,
        default=DEFAULT_SHAPE_PRIOR_WORKER_MAX_OBSERVATION_TO_ALIGNED_P95_M,
        help=(
            "Managed SAM3D worker alignment coverage tolerance. Demo v5 uses "
            "0.06m for the current stuffed-animal single-view warmup path."
        ),
    )
    parser.add_argument(
        "--shape-prior-worker-preload-models",
        dest="shape_prior_worker_preload_models",
        action="store_true",
    )
    parser.add_argument(
        "--no-shape-prior-worker-preload-models",
        dest="shape_prior_worker_preload_models",
        action="store_false",
    )
    parser.set_defaults(shape_prior_worker_preload_models=True)
    parser.add_argument("--shape-prior-worker-warmup-models", action="store_true")
    parser.add_argument("--shape-prior-worker-debug", action="store_true")
    parser.add_argument(
        "--optimization-mode",
        choices=("continuous", "disabled"),
        default=DEFAULT_OPTIMIZATION_MODE,
        help="continuous starts one realtime_phystwin zero-order then first-order process for the whole online stream.",
    )
    parser.add_argument(
        "--point-viewer-mode",
        choices=("window", "disabled"),
        default=DEFAULT_POINT_VIEWER_MODE,
        help="window launches the Demo v5 point viewer.",
    )
    parser.add_argument(
        "--point-viewer-layout",
        choices=POINT_VIEWER_LAYOUTS,
        default=DEFAULT_POINT_VIEWER_LAYOUT,
        help="Viewer layout. side-by-side shows live RGB input next to final_data output chunks.",
    )
    parser.add_argument("--point-viewer-conda-env", default=DEFAULT_POINT_VIEWER_CONDA_ENV)
    parser.add_argument("--point-viewer-cuda-visible-devices", default=DEFAULT_POINT_VIEWER_CUDA_VISIBLE_DEVICES)
    parser.add_argument("--point-viewer-cam-idx", type=int, default=DEFAULT_POINT_VIEWER_CAM_IDX)
    parser.add_argument("--point-viewer-poll-sec", type=float, default=DEFAULT_POINT_VIEWER_POLL_SEC)
    parser.add_argument("--point-viewer-object-stride", type=int, default=DEFAULT_POINT_VIEWER_OBJECT_STRIDE)
    parser.add_argument("--point-viewer-object-radius", type=int, default=DEFAULT_POINT_VIEWER_OBJECT_RADIUS)
    parser.add_argument("--point-viewer-controller-radius", type=int, default=DEFAULT_POINT_VIEWER_CONTROLLER_RADIUS)
    parser.add_argument(
        "--point-viewer-render-mode",
        choices=("rgb-overlay", "sam3d-final-data"),
        default=DEFAULT_POINT_VIEWER_RENDER_MODE,
    )
    parser.add_argument(
        "--point-viewer-object-color-mode",
        choices=("rainbow", "green", "object-colors"),
        default=DEFAULT_POINT_VIEWER_OBJECT_COLOR_MODE,
    )
    parser.add_argument("--realtime-phystwin-root", type=Path, default=DEFAULT_REALTIME_PHYSTWIN_ROOT)
    parser.add_argument("--optimization-conda-env", default=None)
    parser.add_argument("--optimization-cuda-visible-devices", default=DEFAULT_OPTIMIZATION_CUDA_VISIBLE_DEVICES)
    parser.add_argument("--optimization-device", default=DEFAULT_OPTIMIZATION_DEVICE)
    parser.add_argument("--optimization-experiments-dir", default=DEFAULT_OPTIMIZATION_EXPERIMENTS_DIR)
    parser.add_argument("--optimization-zero-experiments-dir", default=DEFAULT_OPTIMIZATION_ZERO_EXPERIMENTS_DIR)
    parser.add_argument("--optimization-zero-iterations", type=int, default=DEFAULT_OPTIMIZATION_ZERO_ITERATIONS)
    parser.add_argument("--optimization-iterations", type=int, default=None)
    parser.add_argument("--optimization-batch-size", type=int, default=DEFAULT_OPTIMIZATION_BATCH_SIZE)
    parser.add_argument("--optimization-zero-batch-size", type=int, default=None)
    parser.add_argument("--optimization-segment-stride", type=int, default=DEFAULT_OPTIMIZATION_SEGMENT_STRIDE)
    parser.add_argument("--optimization-poll-sec", type=float, default=DEFAULT_OPTIMIZATION_POLL_SEC)
    parser.add_argument("--optimization-recent-window-count", type=int, default=DEFAULT_OPTIMIZATION_RECENT_WINDOW_COUNT)
    parser.add_argument("--optimization-seed", type=int, default=DEFAULT_OPTIMIZATION_SEED)
    parser.add_argument("--optimization-start-grace-s", type=float, default=DEFAULT_OPTIMIZATION_START_GRACE_S)
    parser.add_argument("--optimization-train-frame", type=int, default=None)
    parser.add_argument("--optimization-checkpoint-interval", type=int, default=None)
    parser.add_argument("--optimization-wait-timeout-s", type=float, default=0.0)
    parser.add_argument("--optimization-wandb-mode", default="offline")
    parser.add_argument("--optimization-realtime-vis", action="store_true")
    parser.add_argument("--optimization-realtime-vis-dir", default=None)
    parser.add_argument("--optimization-no-sample-recent", action="store_true")
    parser.add_argument(
        "--optimization-stop-when-finished",
        action="store_true",
        help="Ask first-order online training to stop early after the stream is finished. Off by default to preserve quality.",
    )
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
    parser.add_argument("--dry-run", action="store_true", help="Print resolved Demo v5 contract and exit.")
    return parser


def resolve_chunk_frame_count(args: argparse.Namespace) -> int:
    chunk_seconds = float(args.chunk_seconds)
    if not math.isfinite(chunk_seconds) or chunk_seconds <= 0.0:
        raise ValueError("chunk seconds must be positive")
    replay_fps = float(args.replay_fps)
    if not math.isfinite(replay_fps) or replay_fps <= 0.0:
        raise ValueError("replay fps must be positive")
    if args.chunk_frame_count is not None:
        value = int(args.chunk_frame_count)
    else:
        value = int(round(replay_fps * chunk_seconds))
    if value <= 0:
        raise ValueError("chunk frame count must be positive")
    return value


def resolve_camera_source_replay_fps(args: argparse.Namespace) -> float:
    value = args.camera_source_replay_fps
    fps = float(args.replay_fps if value is None else value)
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("Demo v5 source replay fps must be positive")
    return fps


def resolve_realtime_gpu_mode(args: argparse.Namespace) -> str:
    value = getattr(args, "realtime_gpu_mode", None)
    if value is None:
        value = getattr(args, "gpu_mode", DEFAULT_REALTIME_GPU_MODE)
    value = str(value)
    if value not in GPU_MODE_CAMERA_CUDA_VISIBLE_DEVICES:
        raise ValueError(f"unsupported realtime gpu mode: {value!r}")
    return value


def resolve_warmup_gpu_mode(args: argparse.Namespace) -> str:
    value = str(getattr(args, "warmup_gpu_mode", DEFAULT_WARMUP_GPU_MODE))
    if value not in GPU_MODE_SHAPE_PRIOR_DEVICE:
        raise ValueError(f"unsupported warmup gpu mode: {value!r}")
    return value


def resolve_camera_cuda_visible_devices(args: argparse.Namespace) -> str:
    """Resolve the GPU namespace used by the camera/fake-camera subprocess."""
    override = None if args.camera_cuda_visible_devices is None else str(args.camera_cuda_visible_devices).strip()
    if override:
        return override
    try:
        return GPU_MODE_CAMERA_CUDA_VISIBLE_DEVICES[resolve_realtime_gpu_mode(args)]
    except KeyError as exc:
        raise ValueError(f"unsupported realtime gpu mode: {resolve_realtime_gpu_mode(args)!r}") from exc


def resolve_shape_prior_device(args: argparse.Namespace) -> str:
    """Resolve the CUDA device name seen inside the shape-prior worker process."""
    override = getattr(args, "shape_prior_device", None)
    if override is not None and str(override).strip():
        return str(override).strip()
    try:
        return GPU_MODE_SHAPE_PRIOR_DEVICE[resolve_warmup_gpu_mode(args)]
    except KeyError as exc:
        raise ValueError(f"unsupported warmup gpu mode: {resolve_warmup_gpu_mode(args)!r}") from exc


def resolve_shape_prior_worker_cuda_visible_devices(args: argparse.Namespace) -> str:
    override = getattr(args, "shape_prior_worker_cuda_visible_devices", None)
    if override is not None and str(override).strip():
        return str(override).strip()
    try:
        return GPU_MODE_CAMERA_CUDA_VISIBLE_DEVICES[resolve_warmup_gpu_mode(args)]
    except KeyError as exc:
        raise ValueError(f"unsupported warmup gpu mode: {resolve_warmup_gpu_mode(args)!r}") from exc


def resolve_optimization_cuda_visible_devices(args: argparse.Namespace) -> str:
    value = str(getattr(args, "optimization_cuda_visible_devices", DEFAULT_OPTIMIZATION_CUDA_VISIBLE_DEVICES)).strip()
    if not value:
        raise ValueError("--optimization-cuda-visible-devices must be non-empty when optimization is enabled")
    return value


def resolve_optimization_device(args: argparse.Namespace) -> str:
    value = str(getattr(args, "optimization_device", DEFAULT_OPTIMIZATION_DEVICE)).strip()
    if not value:
        raise ValueError("--optimization-device must be non-empty when optimization is enabled")
    return value


def resolve_point_viewer_cuda_visible_devices(args: argparse.Namespace) -> str:
    value = str(getattr(args, "point_viewer_cuda_visible_devices", DEFAULT_POINT_VIEWER_CUDA_VISIBLE_DEVICES)).strip()
    if not value:
        raise ValueError("--point-viewer-cuda-visible-devices must be non-empty when point viewer is enabled")
    return value


def resolve_point_viewer_layout(args: argparse.Namespace) -> str:
    value = str(getattr(args, "point_viewer_layout", DEFAULT_POINT_VIEWER_LAYOUT))
    if value not in POINT_VIEWER_LAYOUTS:
        raise ValueError(f"unsupported point viewer layout: {value!r}")
    return value


def point_viewer_uses_side_by_side(args: argparse.Namespace) -> bool:
    return resolve_point_viewer_layout(args) == POINT_VIEWER_LAYOUT_SIDE_BY_SIDE


def point_viewer_start_policy(args: argparse.Namespace) -> str:
    if str(getattr(args, "point_viewer_mode", DEFAULT_POINT_VIEWER_MODE)) != "window":
        return "disabled"
    if point_viewer_uses_side_by_side(args):
        return "immediate_after_camera_start"
    return "after_first_committed_online_chunk"


def resolve_write_input_rgb_timeline(args: argparse.Namespace) -> bool:
    value = getattr(args, "write_input_rgb_timeline", None)
    if value is not None:
        return bool(value)
    return (
        str(getattr(args, "point_viewer_mode", DEFAULT_POINT_VIEWER_MODE)) == "window"
        and point_viewer_uses_side_by_side(args)
    )


def _repo_path(path: str | Path) -> Path:
    value = Path(path).expanduser()
    if value.is_absolute():
        return value
    return REPO_ROOT / value


def resolve_realtime_phystwin_root(args: argparse.Namespace) -> Path:
    return Path(args.realtime_phystwin_root)


def _resolved_realtime_phystwin_root(args: argparse.Namespace) -> Path:
    return _repo_path(args.realtime_phystwin_root).resolve()


def resolve_realtime_phystwin_base_path(args: argparse.Namespace) -> Path:
    return Path(args.base_path) / "data"


def _relative_for_realtime_phystwin(args: argparse.Namespace, path: str | Path) -> str:
    target = _repo_path(path).resolve()
    start = _resolved_realtime_phystwin_root(args)
    return os.path.relpath(target, start=start)


def _python_command_prefix(conda_env: str | None) -> list[str]:
    env_name = "" if conda_env is None else str(conda_env).strip()
    if env_name:
        return ["conda", "run", "-n", env_name, "--no-capture-output", "python"]
    return ["python"]


def _prepend_env_path(env: dict[str, str], key: str, path: Path) -> None:
    value = str(path)
    current = env.get(key, "")
    parts = [item for item in current.split(os.pathsep) if item]
    if value not in parts:
        env[key] = value if not parts else value + os.pathsep + os.pathsep.join(parts)


def _conda_env_prefix(conda_env: str | None) -> Path | None:
    env_name = "" if conda_env is None else str(conda_env).strip()
    if not env_name:
        return None
    candidates: list[Path] = []
    current_prefix = os.environ.get("CONDA_PREFIX")
    if current_prefix:
        current = Path(current_prefix)
        if current.name == env_name:
            candidates.append(current)
        if current.parent.name == "envs":
            candidates.append(current.parent / env_name)
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe:
        exe = Path(conda_exe)
        if len(exe.parents) >= 2:
            candidates.append(exe.parents[1] / "envs" / env_name)
    candidates.append(Path.home() / "miniconda3" / "envs" / env_name)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def _apply_shape_prior_worker_cuda_build_env(args: argparse.Namespace, env: dict[str, str]) -> None:
    """Expose conda CUDA headers/libs for workers that JIT-compile extensions."""
    prefix = _conda_env_prefix(getattr(args, "shape_prior_worker_conda_env", None))
    if prefix is None:
        return
    include_dir = prefix / "targets" / "x86_64-linux" / "include"
    lib_dir = prefix / "targets" / "x86_64-linux" / "lib"
    if include_dir.is_dir():
        _prepend_env_path(env, "CPATH", include_dir)
        _prepend_env_path(env, "CPLUS_INCLUDE_PATH", include_dir)
    if lib_dir.is_dir():
        _prepend_env_path(env, "LIBRARY_PATH", lib_dir)
        _prepend_env_path(env, "LD_LIBRARY_PATH", lib_dir)
    nvcc = prefix / "bin" / "nvcc"
    env.setdefault("CUDA_HOME", str(prefix))
    if nvcc.is_file():
        env.setdefault("CUDACXX", str(nvcc))
    env.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
    env.setdefault("MAX_JOBS", "8")


def build_shape_prior_worker_command(args: argparse.Namespace) -> list[str]:
    command = [
        *_python_command_prefix(getattr(args, "shape_prior_worker_conda_env", None)),
        str(Path("services") / "shape_prior_remote" / "server.py"),
        "--bind",
        str(args.shape_prior_endpoint),
        "--device",
        str(args.shape_prior_worker_device),
    ]
    if args.shape_prior_worker_sam3d_root is not None:
        command.extend(["--sam3d-root", str(args.shape_prior_worker_sam3d_root)])
    if args.shape_prior_worker_futurephystwin_root is not None:
        command.extend(["--futurephystwin-root", str(args.shape_prior_worker_futurephystwin_root)])
    if args.shape_prior_worker_config is not None:
        command.extend(["--config", str(args.shape_prior_worker_config)])
    command.extend(
        [
            "--max-observation-to-aligned-p95-m",
            str(float(args.shape_prior_worker_max_observation_to_aligned_p95_m)),
        ]
    )
    if bool(args.shape_prior_worker_preload_models):
        command.append("--preload-models")
    if bool(args.shape_prior_worker_warmup_models):
        command.append("--warmup-models")
    if bool(args.shape_prior_worker_debug):
        command.append("--debug")
    return command


def build_realtime_phystwin_optimization_command(
    args: argparse.Namespace,
    *,
    chunk_frame_count: int,
) -> list[str]:
    command = [
        *_python_command_prefix(getattr(args, "optimization_conda_env", None)),
        "train_online_zero_then_first.py",
        "--base_path",
        _relative_for_realtime_phystwin(args, resolve_realtime_phystwin_base_path(args)),
        "--online_dir",
        _relative_for_realtime_phystwin(args, resolve_online_dir(args)),
        "--case_name",
        str(args.case_prefix),
        "--experiments_dir",
        str(args.optimization_experiments_dir),
        "--zero_experiments_dir",
        str(args.optimization_zero_experiments_dir),
        "--static_data_path",
        _relative_for_realtime_phystwin(args, resolve_static_data_path(args)),
        "--device",
        resolve_optimization_device(args),
        "--zero_iterations",
        str(int(args.optimization_zero_iterations)),
        "--batch_size",
        str(int(args.optimization_batch_size)),
        "--segment_len",
        str(int(chunk_frame_count)),
        "--segment_stride",
        str(int(args.optimization_segment_stride)),
        "--poll_sec",
        str(float(args.optimization_poll_sec)),
        "--recent_window_count",
        str(int(args.optimization_recent_window_count)),
        "--seed",
        str(int(args.optimization_seed)),
    ]
    if args.optimization_zero_batch_size is not None:
        command.extend(["--zero_batch_size", str(int(args.optimization_zero_batch_size))])
    if args.optimization_iterations is not None:
        command.extend(["--iterations", str(int(args.optimization_iterations))])
    if args.optimization_train_frame is not None:
        command.extend(["--train_frame", str(int(args.optimization_train_frame))])
    if args.optimization_checkpoint_interval is not None:
        command.extend(["--checkpoint_interval", str(int(args.optimization_checkpoint_interval))])
    if bool(args.optimization_realtime_vis):
        command.append("--zero_realtime_vis")
        command.append("--realtime_vis")
    if args.optimization_realtime_vis_dir is not None:
        vis_root = Path(args.optimization_realtime_vis_dir)
        command.extend(["--zero_realtime_vis_dir", str(vis_root / "zero_order")])
        command.extend(["--realtime_vis_dir", str(vis_root / "first_order")])
    if bool(args.optimization_no_sample_recent):
        command.append("--no_sample_recent")
    if bool(args.optimization_stop_when_finished):
        command.append("--stop_when_finished")
    return command


def build_point_viewer_command(args: argparse.Namespace, *, capture_dir: Path | None = None) -> list[str]:
    layout = resolve_point_viewer_layout(args)
    capture_text = "" if capture_dir is None else str(capture_dir)
    input_timeline_text = "" if capture_dir is None else str(Path(capture_dir) / "input_frames.jsonl")
    command = [
        *_python_command_prefix(getattr(args, "point_viewer_conda_env", None)),
        str(Path("demo_v5") / "visualize_track.py"),
        "--layout",
        layout,
        "--online-dir",
        str(resolve_online_dir(args)),
        "--case-dir",
        str(Path(args.base_path) / "data" / str(args.case_prefix)),
        "--render-mode",
        str(args.point_viewer_render_mode),
        "--cam-idx",
        str(int(args.point_viewer_cam_idx)),
        "--fps",
        str(float(args.replay_fps)),
        "--poll-sec",
        str(float(args.point_viewer_poll_sec)),
        "--object-stride",
        str(int(args.point_viewer_object_stride)),
        "--object-radius",
        str(int(args.point_viewer_object_radius)),
        "--controller-radius",
        str(int(args.point_viewer_controller_radius)),
        "--object-color-mode",
        str(args.point_viewer_object_color_mode),
    ]
    if layout == POINT_VIEWER_LAYOUT_SIDE_BY_SIDE:
        command.extend(
            [
                "--capture-dir",
                capture_text,
                "--input-rgb-timeline",
                input_timeline_text,
            ]
        )
    return command


def _load_optional_points(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    arr = np.asarray(np.load(path), dtype=np.float64)
    if arr.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{path} must contain an Nx3 point array")
    return np.ascontiguousarray(arr, dtype=np.float64)


def resolve_online_dir(args: argparse.Namespace) -> Path:
    return Path(args.base_path) / "online_data" / str(args.case_prefix)


def resolve_static_data_path(args: argparse.Namespace) -> Path:
    return Path(args.base_path) / "data" / str(args.case_prefix) / "final_data.pkl"


def _contract(args: argparse.Namespace) -> dict[str, object]:
    chunk_frame_count = int(resolve_chunk_frame_count(args))
    return {
        "demo_version": "demo_v5",
        "input_source": str(args.input_source),
        "replay_fps": float(args.replay_fps),
        "camera_source_replay_fps": resolve_camera_source_replay_fps(args),
        "camera_source_replay_fps_override": (
            None if args.camera_source_replay_fps is None else float(args.camera_source_replay_fps)
        ),
        "camera_lossless_input_fps": resolve_camera_source_replay_fps(args),
        "chunk_seconds": float(args.chunk_seconds),
        "chunk_poll_interval_s": float(args.chunk_poll_interval_s),
        "chunk_frame_count": chunk_frame_count,
        "allow_degraded_online": bool(args.allow_degraded_online),
        "base_path": str(args.base_path),
        "case_prefix": str(args.case_prefix),
        "output_format": "online-primary-static-case",
        "online_dir": str(resolve_online_dir(args)),
        "static_data_path": str(resolve_static_data_path(args)),
        "realtime_phystwin_base_path": str(resolve_realtime_phystwin_base_path(args)),
        "max_chunks": args.max_chunks,
        "depth_backend": str(args.depth_backend),
        "capture_extra_seconds": float(args.capture_extra_seconds),
        "camera_capture_dir": None if args.camera_capture_dir is None else str(args.camera_capture_dir),
        "gpu_mode": resolve_realtime_gpu_mode(args),
        "realtime_gpu_mode": resolve_realtime_gpu_mode(args),
        "warmup_gpu_mode": resolve_warmup_gpu_mode(args),
        "camera_cuda_visible_devices": resolve_camera_cuda_visible_devices(args),
        "camera_cuda_visible_devices_override": (
            None if args.camera_cuda_visible_devices is None else str(args.camera_cuda_visible_devices)
        ),
        "camera_device": str(args.camera_device),
        "camera_tracker_device": str(args.camera_tracker_device),
        "camera_dtype": str(args.camera_dtype),
        "camera_lossless_max_backlog_seconds": args.camera_lossless_max_backlog_seconds,
        "camera_headless_prepared_only": bool(args.camera_headless_prepared_only),
        "write_input_rgb_timeline": resolve_write_input_rgb_timeline(args),
        "shape_prior_warmup": bool(args.shape_prior_warmup),
        "shape_prior_start_policy": str(args.shape_prior_start_policy),
        "shape_prior_execution": str(args.shape_prior_execution),
        "shape_prior_endpoint": str(args.shape_prior_endpoint),
        "shape_prior_device": resolve_shape_prior_device(args),
        "shape_prior_device_override": None if args.shape_prior_device is None else str(args.shape_prior_device),
        "shape_prior_worker_mode": str(args.shape_prior_worker_mode),
        "shape_prior_worker_command": build_shape_prior_worker_command(args),
        "shape_prior_worker_cuda_visible_devices": resolve_shape_prior_worker_cuda_visible_devices(args),
        "shape_prior_worker_device": str(args.shape_prior_worker_device),
        "shape_prior_worker_conda_env": str(args.shape_prior_worker_conda_env),
        "shape_prior_worker_max_observation_to_aligned_p95_m": float(
            args.shape_prior_worker_max_observation_to_aligned_p95_m
        ),
        "shape_prior_worker_released_before_optimization": bool(
            str(args.shape_prior_worker_mode) == "managed" and str(args.optimization_mode) == "continuous"
        ),
        "shape_prior_worker_released_before_point_viewer": bool(
            str(args.shape_prior_worker_mode) == "managed" and str(args.point_viewer_mode) == "window"
            and not point_viewer_uses_side_by_side(args)
        ),
        "shape_prior_chunk_wait_timeout_s": float(args.shape_prior_chunk_wait_timeout_s),
        "mask_radius_outlier_filter": bool(args.mask_radius_outlier_filter),
        "mask_radius_outlier_radius_m": float(args.mask_radius_outlier_radius_m),
        "mask_radius_outlier_nb_points": int(args.mask_radius_outlier_nb_points),
        "write_final_pcd": bool(args.write_final_pcd),
        "source_headless_capture": None if args.source_headless_capture is None else str(args.source_headless_capture),
        "point_viewer_mode": str(args.point_viewer_mode),
        "point_viewer_layout": resolve_point_viewer_layout(args),
        "point_viewer_command": build_point_viewer_command(args),
        "point_viewer_cuda_visible_devices": resolve_point_viewer_cuda_visible_devices(args),
        "point_viewer_start_policy": point_viewer_start_policy(args),
        "point_viewer_capture_dir": None,
        "point_viewer_fps": float(args.replay_fps),
        "point_viewer_object_color_mode": str(args.point_viewer_object_color_mode),
        "optimization_mode": str(args.optimization_mode),
        "optimization_command": build_realtime_phystwin_optimization_command(
            args,
            chunk_frame_count=chunk_frame_count,
        ),
        "optimization_cuda_visible_devices": resolve_optimization_cuda_visible_devices(args),
        "optimization_device": resolve_optimization_device(args),
        "optimization_start_policy": (
            "after_first_committed_online_chunk" if str(args.optimization_mode) == "continuous" else "disabled"
        ),
        "optimization_scope": (
            "single_continuous_online_case" if str(args.optimization_mode) == "continuous" else "disabled"
        ),
        "optimization_zero_iterations": int(args.optimization_zero_iterations),
        "optimization_batch_size": int(args.optimization_batch_size),
        "optimization_zero_batch_size": args.optimization_zero_batch_size,
        "optimization_iterations": args.optimization_iterations,
        "optimization_segment_len": chunk_frame_count,
        "optimization_segment_stride": int(args.optimization_segment_stride),
        "optimization_recent_window_count": int(args.optimization_recent_window_count),
        "optimization_poll_sec": float(args.optimization_poll_sec),
        "optimization_start_grace_s": float(args.optimization_start_grace_s),
        "optimization_no_sample_recent": bool(args.optimization_no_sample_recent),
        "optimization_stop_when_finished": bool(args.optimization_stop_when_finished),
        "realtime_phystwin_root": str(resolve_realtime_phystwin_root(args)),
    }


def validate_runtime_args(args: argparse.Namespace, *, chunk_frame_count: int) -> None:
    if float(args.chunk_poll_interval_s) <= 0.0:
        raise ValueError("--chunk-poll-interval-s must be positive")
    resolve_camera_source_replay_fps(args)
    if int(chunk_frame_count) <= 0:
        raise ValueError("chunk frame count must be positive")
    if str(args.shape_prior_worker_mode) == "disabled" and bool(args.shape_prior_warmup):
        raise ValueError("--shape-prior-worker-mode disabled requires --no-shape-prior-warmup")
    if str(args.shape_prior_worker_mode) == "managed" and str(args.shape_prior_execution) != "remote-worker":
        raise ValueError("managed shape-prior worker requires --shape-prior-execution remote-worker")
    if float(args.shape_prior_worker_startup_grace_s) < 0.0:
        raise ValueError("--shape-prior-worker-startup-grace-s must be non-negative")
    if float(args.shape_prior_worker_max_observation_to_aligned_p95_m) <= 0.0:
        raise ValueError("--shape-prior-worker-max-observation-to-aligned-p95-m must be positive")
    if str(args.point_viewer_mode) == "window":
        resolve_point_viewer_layout(args)
        if int(args.point_viewer_cam_idx) < 0:
            raise ValueError("--point-viewer-cam-idx must be non-negative")
        if float(args.point_viewer_poll_sec) <= 0.0:
            raise ValueError("--point-viewer-poll-sec must be positive")
        if int(args.point_viewer_object_stride) <= 0:
            raise ValueError("--point-viewer-object-stride must be positive")
        if int(args.point_viewer_object_radius) <= 0:
            raise ValueError("--point-viewer-object-radius must be positive")
        if int(args.point_viewer_controller_radius) <= 0:
            raise ValueError("--point-viewer-controller-radius must be positive")
        resolve_point_viewer_cuda_visible_devices(args)
    if str(args.optimization_mode) == "continuous":
        if args.source_headless_capture is not None:
            raise ValueError("continuous optimization requires fake-live or live capture; use --optimization-mode disabled for source-headless conversion")
        if int(args.optimization_zero_iterations) <= 0:
            raise ValueError("--optimization-zero-iterations must be positive")
        if int(args.optimization_batch_size) <= 0:
            raise ValueError("--optimization-batch-size must be positive")
        if int(args.optimization_segment_stride) <= 0:
            raise ValueError("--optimization-segment-stride must be positive")
        if float(args.optimization_poll_sec) < 0.0:
            raise ValueError("--optimization-poll-sec must be non-negative")
        if float(args.optimization_start_grace_s) < 0.0:
            raise ValueError("--optimization-start-grace-s must be non-negative")
        if int(args.optimization_recent_window_count) <= 0:
            raise ValueError("--optimization-recent-window-count must be positive")
        if args.optimization_zero_batch_size is not None and int(args.optimization_zero_batch_size) <= 0:
            raise ValueError("--optimization-zero-batch-size must be positive")
        if args.optimization_iterations is not None and int(args.optimization_iterations) <= 0:
            raise ValueError("--optimization-iterations must be positive")
        if args.optimization_wait_timeout_s is not None and float(args.optimization_wait_timeout_s) < 0.0:
            raise ValueError("--optimization-wait-timeout-s must be non-negative")
        root = _resolved_realtime_phystwin_root(args)
        if not (root / "train_online_zero_then_first.py").is_file():
            raise ValueError(f"realtime_phystwin root is missing train_online_zero_then_first.py: {root}")
        resolve_optimization_cuda_visible_devices(args)
        resolve_optimization_device(args)


def _camera_duration_s(args: argparse.Namespace, *, chunk_frame_count: int) -> float:
    if args.max_chunks is None:
        return 0.0
    fps = resolve_camera_source_replay_fps(args)
    if fps <= 0.0:
        fps = DEFAULT_REPLAY_FPS
    return (float(args.max_chunks) * float(chunk_frame_count) / fps) + float(args.capture_extra_seconds)


def build_camera_realtime_command(
    args: argparse.Namespace,
    *,
    capture_dir: Path,
    profile_json: Path,
    chunk_frame_count: int,
) -> list[str]:
    """Build the subprocess command that emits prepared PhysTwin frames."""
    script = Path("demo_v5") / "realtime_dense_track.py"
    camera_source_replay_fps = resolve_camera_source_replay_fps(args)
    if str(args.depth_backend) == "ir-ffs":
        depth_source = "ffs"
    elif str(args.depth_backend) == "native-realsense":
        depth_source = "realsense"
    else:
        raise ValueError(f"unsupported depth backend: {args.depth_backend!r}")
    command = [
        "python",
        str(script),
        "--input-source",
        str(args.input_source),
        "--depth-source",
        depth_source,
        "--depth-backend-label",
        str(args.depth_backend),
        "--duration-s",
        f"{_camera_duration_s(args, chunk_frame_count=chunk_frame_count):.3f}",
        "--render-mode",
        "none",
        "--headless-capture-dir",
        str(capture_dir),
        "--tracking-product-backend",
        "phystwin-strict-tracking",
        "--track-mode",
        "controller-object",
        "--pcd-mode",
        "masked",
        "--tracker-backend",
        "tapnextpp",
        "--tracker-overlay-max-points",
        "0",
        "--demo-visual-mode",
        "tracking",
        "--replay-fps",
        str(camera_source_replay_fps),
        "--device",
        str(args.camera_device),
        "--dtype",
        str(args.camera_dtype),
        "--tracker-device",
        str(args.camera_tracker_device),
        "--enable-pcd-filter",
        "--pcd-filter-mode",
        "sync",
        "--pcd-filter-preset",
        "original",
        "--table-calibrate",
        str(DEFAULT_TABLE_CALIBRATE_PATH),
        "--enable-table-z-filter",
        "--runtime-product-name",
        "demo_v5_realtime_dense_track",
        "--metadata-demo-version",
        "demo_v5",
        "--metadata-reference-pipeline",
        "data_process_sam3d",
    ]
    if args.camera_lossless_max_backlog_seconds is not None:
        command.extend(
            [
                "--lossless-max-backlog-seconds",
                str(float(args.camera_lossless_max_backlog_seconds)),
            ]
        )
    if float(camera_source_replay_fps) != float(DEFAULT_CAMERA_LOSSLESS_INPUT_FPS):
        command.extend(["--lossless-input-fps", str(float(camera_source_replay_fps))])
    if bool(args.camera_headless_prepared_only):
        command.append("--headless-prepared-only")
    if resolve_write_input_rgb_timeline(args):
        command.append("--write-input-rgb-timeline")
    if bool(args.shape_prior_warmup):
        command.extend(
            [
                "--shape-prior-warmup",
                "--shape-prior-start-policy",
                str(args.shape_prior_start_policy),
                "--shape-prior-execution",
                str(args.shape_prior_execution),
                "--shape-prior-endpoint",
                str(args.shape_prior_endpoint),
                "--shape-prior-timeout-ms",
                str(int(args.shape_prior_timeout_ms)),
                "--shape-prior-device",
                resolve_shape_prior_device(args),
                "--shape-prior-profile-json",
                str(profile_json),
            ]
        )
        if bool(args.shape_prior_skip_route_visualizations):
            command.append("--shape-prior-skip-route-visualizations")
        else:
            command.append("--shape-prior-render-route-visualizations")
    else:
        command.append("--no-shape-prior-warmup")
    return command


def _default_capture_dir(args: argparse.Namespace, base_path: Path) -> Path:
    if args.camera_capture_dir is not None:
        return Path(args.camera_capture_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base_path / f"{args.case_prefix}_camera_capture_{stamp}"


def _stop_process(process: subprocess.Popen[bytes]) -> int | None:
    if process.poll() is not None:
        return process.returncode
    used_process_group = False
    pid = getattr(process, "pid", None)
    try:
        if pid is not None:
            os.killpg(os.getpgid(int(pid)), signal.SIGTERM)
            used_process_group = True
        else:
            process.terminate()
        return process.wait(timeout=10)
    except Exception:
        try:
            if pid is not None and used_process_group:
                os.killpg(os.getpgid(int(pid)), signal.SIGKILL)
            else:
                process.kill()
            return process.wait(timeout=10)
        except Exception:
            return process.poll()


def _start_managed_shape_prior_worker(args: argparse.Namespace) -> subprocess.Popen[bytes] | None:
    """Start the optional SAM3D worker under Demo v5 lifecycle control."""
    if not bool(args.shape_prior_warmup):
        return None
    if str(args.shape_prior_worker_mode) != "managed":
        return None
    command = build_shape_prior_worker_command(args)
    env = os.environ.copy()
    _apply_shape_prior_worker_cuda_build_env(args, env)
    cuda_visible_devices = resolve_shape_prior_worker_cuda_visible_devices(args)
    if cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    process = subprocess.Popen(command, cwd=REPO_ROOT, env=env, start_new_session=True)
    grace_s = float(args.shape_prior_worker_startup_grace_s)
    if grace_s > 0.0:
        time.sleep(grace_s)
    if process.poll() is not None:
        raise RuntimeError(f"managed shape-prior worker exited during startup with code {process.returncode}")
    return process


def _optimization_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = resolve_optimization_cuda_visible_devices(args)
    wandb_mode = str(getattr(args, "optimization_wandb_mode", "") or "").strip()
    if wandb_mode:
        env["WANDB_MODE"] = wandb_mode
        if wandb_mode == "offline":
            env.setdefault("WANDB_SILENT", "true")
    return env


def _point_viewer_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = resolve_point_viewer_cuda_visible_devices(args)
    return env


def _start_continuous_optimization(
    args: argparse.Namespace,
    *,
    chunk_frame_count: int,
) -> subprocess.Popen[bytes]:
    """Launch realtime_phystwin against the already-publishing online case."""
    command = build_realtime_phystwin_optimization_command(args, chunk_frame_count=chunk_frame_count)
    return subprocess.Popen(
        command,
        cwd=_resolved_realtime_phystwin_root(args),
        env=_optimization_env(args),
        start_new_session=True,
    )


def _start_point_viewer(args: argparse.Namespace, *, capture_dir: Path | None = None) -> subprocess.Popen[bytes]:
    """Launch the lightweight online point viewer in the repo environment."""
    return subprocess.Popen(
        build_point_viewer_command(args, capture_dir=capture_dir),
        cwd=REPO_ROOT,
        env=_point_viewer_env(args),
        start_new_session=True,
    )


def _wait_for_process(process: subprocess.Popen[bytes], *, timeout_s: float) -> int | None:
    if process.poll() is not None:
        return process.returncode
    if float(timeout_s) <= 0.0:
        return process.wait()
    try:
        return process.wait(timeout=float(timeout_s))
    except subprocess.TimeoutExpired:
        return None


def select_validation_chunk_cases(manifests: Sequence[dict[str, object]]) -> list[str]:
    if len(manifests) < 5:
        raise ValueError("at least five chunks are required for second-last and fifth-last validation")
    return [
        str(manifests[-2]["case_name"]),
        str(manifests[-5]["case_name"]),
    ]


def _runtime_chunk_summary(manifests: Sequence[dict[str, object]]) -> dict[str, object]:
    publish_times = [
        float(item["publish_wall_s"])
        for item in manifests
        if item.get("publish_wall_s") is not None
    ]
    intervals = [publish_times[idx] - publish_times[idx - 1] for idx in range(1, len(publish_times))]
    backlog_values = [
        int(item["backlog_chunks"])
        for item in manifests
        if item.get("backlog_chunks") is not None
    ]
    shape_publish_times = [
        float(item["publish_wall_s"])
        for item in manifests
        if item.get("publish_wall_s") is not None
        and bool(item.get("shape_prior_complete") or item.get("shape_prior_target_counts_met"))
    ]
    quality_order = {"normal": 0, "degraded": 1, "invalid": 2}
    quality_values = [
        str(item.get("track_process_status", "normal"))
        for item in manifests
    ]
    track_process_status = "normal"
    if quality_values:
        track_process_status = max(quality_values, key=lambda value: quality_order.get(value, -1))
    quality_counts = {
        status: int(sum(1 for value in quality_values if value == status))
        for status in ("normal", "degraded", "invalid")
    }
    invalid_chunks = [
        str(item.get("case_name", ""))
        for item in manifests
        if str(item.get("track_process_status", "normal")) == "invalid"
    ]
    return {
        "first_ready_chunk_wall_s": publish_times[0] if publish_times else None,
        "first_shape_prior_ready_chunk_wall_s": shape_publish_times[0] if shape_publish_times else None,
        "steady_publish_intervals_s": intervals,
        "steady_state_publish_interval_max_s": max(intervals) if intervals else None,
        "max_backlog_chunks": max(backlog_values) if backlog_values else None,
        "track_process_status": track_process_status,
        "track_process_status_counts": quality_counts,
        "track_process_invalid_chunk_count": int(len(invalid_chunks)),
        "track_process_invalid_chunks": invalid_chunks,
        "online_publish_skipped_chunk_count": int(
            sum(1 for item in manifests if bool(item.get("online_publish_skipped", False)))
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    chunk_frame_count = resolve_chunk_frame_count(args)
    validate_runtime_args(args, chunk_frame_count=chunk_frame_count)

    if bool(args.dry_run):
        print(json.dumps(_contract(args), indent=2, sort_keys=True))
        return 0

    base_path = Path(args.base_path)
    base_path.mkdir(parents=True, exist_ok=True)
    startup_migration = migrate_legacy_online_static_case(base_path, str(args.case_prefix))
    if args.source_headless_capture is not None:
        manifests = write_chunks_from_headless_capture(
            args.source_headless_capture,
            base_path=base_path,
            case_prefix=str(args.case_prefix),
            chunk_frame_count=chunk_frame_count,
            fps=int(round(float(args.replay_fps))),
            max_chunks=args.max_chunks,
            surface_points=_load_optional_points(args.surface_points_npy),
            interior_points=_load_optional_points(args.interior_points_npy),
            mask_radius_outlier_filter=bool(args.mask_radius_outlier_filter),
            mask_radius_outlier_radius_m=float(args.mask_radius_outlier_radius_m),
            mask_radius_outlier_nb_points=int(args.mask_radius_outlier_nb_points),
            write_final_pcd=bool(args.write_final_pcd),
            allow_degraded_online=bool(args.allow_degraded_online),
        )
        final_migration = migrate_legacy_online_static_case(base_path, str(args.case_prefix))
        summary = {
            "demo_version": "demo_v5",
            "mode": "source-headless-capture",
            "source_headless_capture": str(args.source_headless_capture),
            "base_path": str(base_path),
            "case_prefix": str(args.case_prefix),
            "output_format": "online-primary-static-case",
            "online_dir": str(resolve_online_dir(args)),
            "static_data_path": str(resolve_static_data_path(args)),
            "chunk_frame_count": int(chunk_frame_count),
            "allow_degraded_online": bool(args.allow_degraded_online),
            "max_chunks": args.max_chunks,
            "chunk_count": int(len(manifests)),
            "chunks": manifests,
            "write_final_pcd": bool(args.write_final_pcd),
            "optimization_mode": str(args.optimization_mode),
            "optimization_started": False,
            "optimization_scope": "disabled_for_source_headless_conversion",
            "startup_legacy_static_case_migration": startup_migration,
            "final_legacy_static_case_migration": final_migration,
        }
        summary.update(_runtime_chunk_summary(manifests))
        summary_path = base_path / f"{args.case_prefix}_chunks_manifest.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 1 if str(summary.get("track_process_status", "normal")) == "invalid" else 0

    capture_dir = _default_capture_dir(args, base_path)
    capture_dir.mkdir(parents=True, exist_ok=True)
    profile_json = Path(args.shape_prior_profile_json) if args.shape_prior_profile_json is not None else capture_dir / "shape_prior_profile.json"
    command = build_camera_realtime_command(
        args,
        capture_dir=capture_dir,
        profile_json=profile_json,
        chunk_frame_count=chunk_frame_count,
    )
    camera_env = os.environ.copy()
    _apply_default_sam31_checkpoint_env(camera_env)
    cuda_visible_devices = resolve_camera_cuda_visible_devices(args).strip()
    if cuda_visible_devices:
        camera_env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    shape_prior_worker_process = _start_managed_shape_prior_worker(args)
    shape_prior_worker_return_code: int | None = None
    shape_prior_worker_released_before_optimization = False
    shape_prior_worker_released_before_point_viewer = False
    point_viewer_process: subprocess.Popen[bytes] | None = None
    point_viewer_started = False
    point_viewer_started_manifest: dict[str, object] | None = None
    point_viewer_start_wall_s: float | None = None
    point_viewer_return_code: int | None = None
    point_viewer_left_running = False
    optimization_process: subprocess.Popen[bytes] | None = None
    optimization_started_manifest: dict[str, object] | None = None
    optimization_start_wall_s: float | None = None
    optimization_return_code: int | None = None
    optimization_timed_out = False

    def on_chunk_written(manifest: dict[str, object]) -> None:
        nonlocal point_viewer_process
        nonlocal point_viewer_started
        nonlocal point_viewer_started_manifest
        nonlocal point_viewer_start_wall_s
        nonlocal optimization_process
        nonlocal optimization_started_manifest
        nonlocal optimization_start_wall_s
        nonlocal shape_prior_worker_process
        nonlocal shape_prior_worker_return_code
        nonlocal shape_prior_worker_released_before_optimization
        nonlocal shape_prior_worker_released_before_point_viewer
        # Output-only viewing starts after the first committed chunk. The
        # side-by-side viewer starts immediately after camera launch so warmup
        # RGB remains visible while the output side waits for chunks.
        if (
            str(args.point_viewer_mode) == "window"
            and not point_viewer_uses_side_by_side(args)
            and point_viewer_process is None
        ):
            if shape_prior_worker_process is not None:
                shape_prior_worker_return_code = _stop_process(shape_prior_worker_process)
                shape_prior_worker_process = None
                shape_prior_worker_released_before_point_viewer = True
            point_viewer_process = _start_point_viewer(args)
            point_viewer_started = True
            point_viewer_started_manifest = dict(manifest)
            point_viewer_start_wall_s = time.monotonic()
        if str(args.optimization_mode) != "continuous" or optimization_process is not None:
            return
        if shape_prior_worker_process is not None:
            shape_prior_worker_return_code = _stop_process(shape_prior_worker_process)
            shape_prior_worker_process = None
            shape_prior_worker_released_before_optimization = True
            start_grace_s = float(args.optimization_start_grace_s)
            if start_grace_s > 0.0:
                time.sleep(start_grace_s)
        elif shape_prior_worker_released_before_point_viewer:
            shape_prior_worker_released_before_optimization = True
        optimization_process = _start_continuous_optimization(
            args,
            chunk_frame_count=chunk_frame_count,
        )
        optimization_started_manifest = dict(manifest)
        optimization_start_wall_s = time.monotonic()

    process = subprocess.Popen(command, env=camera_env, start_new_session=True)
    if str(args.point_viewer_mode) == "window" and point_viewer_uses_side_by_side(args):
        point_viewer_process = _start_point_viewer(args, capture_dir=capture_dir)
        point_viewer_started = True
        point_viewer_start_wall_s = time.monotonic()
    surface_points = _load_optional_points(args.surface_points_npy)
    interior_points = _load_optional_points(args.interior_points_npy)
    try:
        # The bridge tails frames.jsonl and publishes fixed-size chunks while
        # the camera subprocess is still running, so fake-live and live share the
        # same realtime chunking path.
        manifests = stream_chunks_from_headless_capture(
            capture_dir,
            base_path=base_path,
            case_prefix=str(args.case_prefix),
            chunk_frame_count=chunk_frame_count,
            fps=int(round(float(args.replay_fps))),
            max_chunks=args.max_chunks,
            capture_finished=lambda: process.poll() is not None,
            require_shape_prior=bool(args.shape_prior_warmup),
            shape_prior_wait_timeout_s=float(args.shape_prior_chunk_wait_timeout_s),
            poll_interval_s=float(args.chunk_poll_interval_s),
            surface_points=surface_points,
            interior_points=interior_points,
            mask_radius_outlier_filter=bool(args.mask_radius_outlier_filter),
            mask_radius_outlier_radius_m=float(args.mask_radius_outlier_radius_m),
            mask_radius_outlier_nb_points=int(args.mask_radius_outlier_nb_points),
            write_final_pcd=bool(args.write_final_pcd),
            on_chunk_written=on_chunk_written,
            allow_degraded_online=bool(args.allow_degraded_online),
        )
    finally:
        return_code = _stop_process(process)
        if shape_prior_worker_process is not None:
            shape_prior_worker_return_code = _stop_process(shape_prior_worker_process)
            shape_prior_worker_process = None
        if optimization_process is not None:
            optimization_return_code = _wait_for_process(
                optimization_process,
                timeout_s=float(args.optimization_wait_timeout_s),
            )
            if optimization_return_code is None:
                optimization_timed_out = True
                optimization_return_code = _stop_process(optimization_process)
        if point_viewer_process is not None:
            point_viewer_return_code = point_viewer_process.poll()
            point_viewer_left_running = point_viewer_return_code is None
    final_migration = migrate_legacy_online_static_case(base_path, str(args.case_prefix))
    validation_cases = select_validation_chunk_cases(manifests) if len(manifests) >= 5 else []
    runtime_summary = _runtime_chunk_summary(manifests)
    track_process_invalid = str(runtime_summary.get("track_process_status", "normal")) == "invalid"
    if track_process_invalid:
        stop_reason = "track_process_invalid"
    elif args.max_chunks is not None and len(manifests) >= int(args.max_chunks):
        stop_reason = "max_chunks_reached"
    elif return_code == 0:
        stop_reason = "camera_completed"
    elif return_code is None:
        stop_reason = "camera_status_unknown"
    else:
        stop_reason = "camera_exited_before_target"
    summary = {
        "demo_version": "demo_v5",
        "mode": "full-fake-realtime-camera" if str(args.input_source) == "fake-live" else "full-live-camera",
        "gpu_mode": resolve_realtime_gpu_mode(args),
        "realtime_gpu_mode": resolve_realtime_gpu_mode(args),
        "warmup_gpu_mode": resolve_warmup_gpu_mode(args),
        "camera_command": command,
        "camera_cuda_visible_devices": cuda_visible_devices,
        "camera_cuda_visible_devices_override": (
            None if args.camera_cuda_visible_devices is None else str(args.camera_cuda_visible_devices)
        ),
        "camera_lossless_max_backlog_seconds": args.camera_lossless_max_backlog_seconds,
        "camera_headless_prepared_only": bool(args.camera_headless_prepared_only),
        "write_input_rgb_timeline": resolve_write_input_rgb_timeline(args),
        "camera_source_replay_fps": resolve_camera_source_replay_fps(args),
        "camera_source_replay_fps_override": (
            None if args.camera_source_replay_fps is None else float(args.camera_source_replay_fps)
        ),
        "camera_lossless_input_fps": resolve_camera_source_replay_fps(args),
        "shape_prior_device": resolve_shape_prior_device(args),
        "shape_prior_device_override": None if args.shape_prior_device is None else str(args.shape_prior_device),
        "shape_prior_worker_mode": str(args.shape_prior_worker_mode),
        "shape_prior_worker_command": build_shape_prior_worker_command(args),
        "shape_prior_worker_cuda_visible_devices": resolve_shape_prior_worker_cuda_visible_devices(args),
        "shape_prior_worker_device": str(args.shape_prior_worker_device),
        "shape_prior_worker_max_observation_to_aligned_p95_m": float(
            args.shape_prior_worker_max_observation_to_aligned_p95_m
        ),
        "shape_prior_worker_return_code": shape_prior_worker_return_code,
        "shape_prior_worker_released_before_optimization": shape_prior_worker_released_before_optimization,
        "shape_prior_worker_released_before_point_viewer": shape_prior_worker_released_before_point_viewer,
        "camera_return_code": return_code,
        "camera_stop_reason": stop_reason,
        "camera_capture_dir": str(capture_dir),
        "base_path": str(base_path),
        "case_prefix": str(args.case_prefix),
        "output_format": "online-primary-static-case",
        "online_dir": str(resolve_online_dir(args)),
        "static_data_path": str(resolve_static_data_path(args)),
        "realtime_phystwin_base_path": str(resolve_realtime_phystwin_base_path(args)),
        "chunk_frame_count": int(chunk_frame_count),
        "chunk_poll_interval_s": float(args.chunk_poll_interval_s),
        "allow_degraded_online": bool(args.allow_degraded_online),
        "max_chunks": args.max_chunks,
        "chunk_count": int(len(manifests)),
        "chunks": manifests,
        "validation_chunk_cases": validation_cases,
        "external_shape_prior_points": bool(surface_points is not None or interior_points is not None),
        "write_final_pcd": bool(args.write_final_pcd),
        "point_viewer_mode": str(args.point_viewer_mode),
        "point_viewer_layout": resolve_point_viewer_layout(args),
        "point_viewer_started": point_viewer_started,
        "point_viewer_start_policy": point_viewer_start_policy(args),
        "point_viewer_capture_dir": str(capture_dir) if point_viewer_uses_side_by_side(args) else None,
        "point_viewer_started_from_chunk": point_viewer_started_manifest,
        "point_viewer_start_wall_s": point_viewer_start_wall_s,
        "point_viewer_command": build_point_viewer_command(
            args,
            capture_dir=capture_dir if point_viewer_uses_side_by_side(args) else None,
        ),
        "point_viewer_cuda_visible_devices": resolve_point_viewer_cuda_visible_devices(args),
        "point_viewer_fps": float(args.replay_fps),
        "point_viewer_object_color_mode": str(args.point_viewer_object_color_mode),
        "point_viewer_return_code": point_viewer_return_code,
        "point_viewer_left_running": point_viewer_left_running,
        "optimization_mode": str(args.optimization_mode),
        "optimization_started": optimization_started_manifest is not None,
        "optimization_scope": (
            "single_continuous_online_case" if str(args.optimization_mode) == "continuous" else "disabled"
        ),
        "optimization_start_policy": (
            "after_first_committed_online_chunk" if str(args.optimization_mode) == "continuous" else "disabled"
        ),
        "optimization_started_from_chunk": optimization_started_manifest,
        "optimization_start_wall_s": optimization_start_wall_s,
        "optimization_command": build_realtime_phystwin_optimization_command(
            args,
            chunk_frame_count=chunk_frame_count,
        ),
        "optimization_cuda_visible_devices": resolve_optimization_cuda_visible_devices(args),
        "optimization_device": resolve_optimization_device(args),
        "optimization_return_code": optimization_return_code,
        "optimization_timed_out": optimization_timed_out,
        "optimization_zero_iterations": int(args.optimization_zero_iterations),
        "optimization_batch_size": int(args.optimization_batch_size),
        "optimization_zero_batch_size": args.optimization_zero_batch_size,
        "optimization_iterations": args.optimization_iterations,
        "optimization_segment_len": int(chunk_frame_count),
        "optimization_segment_stride": int(args.optimization_segment_stride),
        "optimization_recent_window_count": int(args.optimization_recent_window_count),
        "optimization_poll_sec": float(args.optimization_poll_sec),
        "optimization_start_grace_s": float(args.optimization_start_grace_s),
        "optimization_stop_when_finished": bool(args.optimization_stop_when_finished),
        "startup_legacy_static_case_migration": startup_migration,
        "final_legacy_static_case_migration": final_migration,
    }
    summary.update(runtime_summary)
    summary_path = base_path / f"{args.case_prefix}_chunks_manifest.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if track_process_invalid:
        return 1
    if return_code not in (0, None) and not manifests:
        return int(return_code)
    if args.max_chunks is not None and len(manifests) < int(args.max_chunks):
        return 1
    if str(args.point_viewer_mode) == "window" and not point_viewer_started:
        return 1
    if point_viewer_return_code not in (0, None):
        return int(point_viewer_return_code)
    if str(args.optimization_mode) == "continuous" and optimization_started_manifest is None:
        return 1
    if optimization_return_code not in (0, None):
        return int(optimization_return_code)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
