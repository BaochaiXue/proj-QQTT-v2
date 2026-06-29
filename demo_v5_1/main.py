#!/usr/bin/env python3
"""Demo v5.1 realtime orchestration entrypoint.

This runner owns process boundaries, GPU routing, and artifact publication. The
actual camera/tracker stack runs in ``demo_v5_1/main_data_processing.py``;
SAM3D shape prior warmup runs as local one-shot stages; the default
side-by-side visualizer starts as soon as capture starts.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
from datetime import datetime
from typing import Sequence

import numpy as np
import yaml


# Keep this repo at the front of the import path when the script is launched
# from another working directory. Removing the existing entry first avoids a
# duplicate path while preserving the "current checkout wins" import order.
REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT_STR = str(REPO_ROOT)
if REPO_ROOT_STR in sys.path:
    sys.path.remove(REPO_ROOT_STR)
sys.path.insert(0, REPO_ROOT_STR)

from demo_v5_1.realtime_data_process_track import (
    stream_chunks_from_headless_capture,
    write_chunks_from_headless_capture,
)


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "default.yaml"


def load_default_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, object]:
    """Load Demo v5.1 defaults from YAML."""
    text = Path(path).read_text(encoding="utf-8")
    loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise ValueError(f"default config must be a mapping: {path}")
    return dict(loaded)


_DEFAULT_CONFIG = load_default_config()


def _cfg(section: str, key: str) -> object:
    return _DEFAULT_CONFIG[section][key]


def _cfg_optional_path(section: str, key: str) -> Path | None:
    value = _cfg(section, key)
    if value is None or str(value).strip() == "":
        return None
    return Path(str(value))


# Defaults below describe the current Demo v5.1 realtime path.
DEFAULT_DATA_PROCESS_BASE_PATH = Path(str(_cfg("paths", "data_process_base_path")))
DEFAULT_INPUT_SOURCE = str(_cfg("input", "input_source"))
DEFAULT_REPLAY_FPS = float(_cfg("input", "replay_fps"))
DEFAULT_CHUNK_SECONDS = float(_cfg("chunking", "chunk_seconds"))
DEFAULT_CHUNK_POLL_INTERVAL_S = float(_cfg("chunking", "chunk_poll_interval_s"))
DEFAULT_CAMERA_SOURCE_REPLAY_FPS = float(_cfg("input", "camera_source_replay_fps"))
DEFAULT_CAMERA_FPS = int(_cfg("camera", "camera_fps"))
CAMERA_FPS_CHOICES = tuple(int(item) for item in _cfg("camera", "camera_fps_choices"))
DEFAULT_CAMERA_COLOR_EXPOSURE = float(_cfg("camera", "camera_color_exposure"))
DEFAULT_CAMERA_COLOR_GAIN = float(_cfg("camera", "camera_color_gain"))
DEFAULT_CASE_PREFIX = str(_cfg("camera", "case_prefix"))
DEFAULT_DEPTH_BACKEND = str(_cfg("camera", "depth_backend"))
DEFAULT_MAX_CHUNKS: int | None = (
    None
    if _cfg("chunking", "max_chunks") is None
    else int(_cfg("chunking", "max_chunks"))
)
DEFAULT_SHAPE_PRIOR_TIMEOUT_MS = int(_cfg("shape_prior", "shape_prior_timeout_ms"))
DEFAULT_SHAPE_PRIOR_CHUNK_WAIT_TIMEOUT_S = float(
    _cfg("shape_prior", "shape_prior_chunk_wait_timeout_s")
)
CONFIG_SHAPE_PRIOR_CONTROLLER_NAME = str(
    _cfg("shape_prior", "shape_prior_controller_name")
)
DEFAULT_SHAPE_PRIOR_SAM3D_ROOT = _cfg_optional_path("shape_prior", "shape_prior_sam3d_root")
DEFAULT_SHAPE_PRIOR_CONFIG = _cfg_optional_path("shape_prior", "shape_prior_config")
DEFAULT_MASK_RADIUS_OUTLIER_RADIUS_M = float(
    _cfg("camera", "mask_radius_outlier_radius_m")
)
DEFAULT_MASK_RADIUS_OUTLIER_NB_POINTS = int(
    _cfg("camera", "mask_radius_outlier_nb_points")
)
DEFAULT_MAIN_DATA_PROCESSING_CUDA_VISIBLE_DEVICES = str(
    _cfg("gpu", "main_data_processing_cuda_visible_devices")
)
DEFAULT_SHAPE_PRIOR_WARMUP_CUDA_VISIBLE_DEVICES = str(
    _cfg("gpu", "shape_prior_warmup_cuda_visible_devices")
)
DEFAULT_VISUALIZER_CUDA_VISIBLE_DEVICES = str(
    _cfg("gpu", "visualizer_cuda_visible_devices")
)
DEFAULT_PERCEPTION_DEVICE = str(_cfg("camera", "perception_device"))
DEFAULT_TRACKER_DEVICE = str(_cfg("camera", "tracker_device"))
DEFAULT_INFERENCE_DTYPE = str(_cfg("camera", "inference_dtype"))
DEFAULT_VISUALIZER_MODE = str(_cfg("visualizer", "visualizer_mode"))
DEFAULT_VISUALIZER_CONDA_ENV = str(
    _cfg("visualizer", "visualizer_conda_env")
)
DEFAULT_VISUALIZER_CAM_IDX = int(_cfg("visualizer", "visualizer_cam_idx"))
DEFAULT_VISUALIZER_POLL_SEC = float(
    _cfg("visualizer", "visualizer_poll_sec")
)
DEFAULT_VISUALIZER_OBJECT_STRIDE = int(
    _cfg("visualizer", "visualizer_object_stride")
)
DEFAULT_VISUALIZER_OBJECT_RADIUS = int(
    _cfg("visualizer", "visualizer_object_radius")
)
DEFAULT_VISUALIZER_CONTROLLER_RADIUS = int(
    _cfg("visualizer", "visualizer_controller_radius")
)
DEFAULT_VISUALIZER_OBJECT_COLOR_MODE = str(
    _cfg("visualizer", "visualizer_object_color_mode")
)
VISUALIZER_LAYOUT_SIDE_BY_SIDE = str(
    _cfg("visualizer", "visualizer_layout_side_by_side")
)
VISUALIZER_LAYOUT_OUTPUT_ONLY = str(
    _cfg("visualizer", "visualizer_layout_output_only")
)
VISUALIZER_LAYOUTS = tuple(
    str(item) for item in _cfg("visualizer", "visualizer_layouts")
)
DEFAULT_VISUALIZER_LAYOUT = str(_cfg("visualizer", "visualizer_layout"))
DEFAULT_VISUALIZER_RENDER_MODE = str(
    _cfg("visualizer", "visualizer_render_mode")
)
DEFAULT_TABLE_CALIBRATE_PATH = Path(str(_cfg("paths", "table_calibrate_path")))
DEFAULT_SAM31_CHECKPOINT_PATH = Path(str(_cfg("paths", "sam31_checkpoint_path")))
SAM31_CHECKPOINT_ENV = str(_cfg("paths", "sam31_checkpoint_env"))


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for Demo v5.1 realtime orchestration."""
    parser = argparse.ArgumentParser(
        description=(
            "Demo v5 realtime data_process_sam3d runner. It turns Demo v5 "
            "single-camera fake/live capture into one online data_process_sam3d "
            "case and can launch an online visualizer."
        )
    )
    # Input/chunking options define the online case cadence. The camera can run
    # longer than the requested chunk count; the chunk writer is what stops
    # publishing after max_chunks.
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
    parser.add_argument(
        "--depth-backend",
        choices=("ir-ffs", "native-realsense"),
        default=DEFAULT_DEPTH_BACKEND,
    )
    parser.add_argument(
        "--chunk-frame-count",
        type=int,
        default=None,
        help="Override chunk length in frames. Defaults to round(replay_fps * chunk_seconds).",
    )
    parser.add_argument(
        "--allow-degraded-online",
        action="store_true",
        help=(
            "Append degraded track-process chunks to online_data. Invalid chunks "
            "are always diagnostic-only."
        ),
    )
    parser.add_argument("--base-path", type=Path, default=DEFAULT_DATA_PROCESS_BASE_PATH)
    parser.add_argument("--case-prefix", default=DEFAULT_CASE_PREFIX)
    parser.add_argument(
        "--main-data-processing-cuda-visible-devices",
        default=DEFAULT_MAIN_DATA_PROCESSING_CUDA_VISIBLE_DEVICES,
        help=(
            "CUDA_VISIBLE_DEVICES for main warmup and the realtime "
            "data_process subprocess."
        ),
    )
    parser.add_argument(
        "--shape-prior-warmup-cuda-visible-devices",
        default=DEFAULT_SHAPE_PRIOR_WARMUP_CUDA_VISIBLE_DEVICES,
        help="CUDA_VISIBLE_DEVICES for the SAM3D shape-prior warmup stages.",
    )
    parser.add_argument(
        "--perception-device",
        default=DEFAULT_PERCEPTION_DEVICE,
        help="Segmentation/depth/perception device passed to the camera runtime.",
    )
    parser.add_argument(
        "--tracker-device",
        default=DEFAULT_TRACKER_DEVICE,
        help="Point-tracker device passed to the camera runtime.",
    )
    parser.add_argument(
        "--inference-dtype",
        choices=("bfloat16", "float16", "float32"),
        default=DEFAULT_INFERENCE_DTYPE,
        help="Torch autocast dtype passed to the camera runtime.",
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
        "--camera-fps",
        type=int,
        choices=CAMERA_FPS_CHOICES,
        default=DEFAULT_CAMERA_FPS,
        help="RealSense capture FPS passed to Demo v5.1 live camera. Defaults to 5 FPS.",
    )
    parser.add_argument(
        "--camera-color-exposure",
        type=float,
        default=DEFAULT_CAMERA_COLOR_EXPOSURE,
        help="Manual RealSense RGB exposure passed to Demo v5.1 live camera.",
    )
    parser.add_argument(
        "--camera-color-gain",
        type=float,
        default=DEFAULT_CAMERA_COLOR_GAIN,
        help="Manual RealSense RGB gain passed to Demo v5.1 live camera.",
    )
    # The current chunker consumes prepared frames directly. Keeping only
    # prepared artifacts keeps live runs small and avoids old per-frame outputs
    # becoming part of the v5.1 contract.
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
        help=(
            "Keep Demo v5 legacy per-frame headless artifacts in addition to "
            "prepared realtime frames."
        ),
    )
    parser.set_defaults(camera_headless_prepared_only=True)
    parser.add_argument(
        "--write-input-rgb-timeline",
        dest="write_input_rgb_timeline",
        action="store_true",
        default=None,
        help=(
            "Write input_rgb/*.png and input_frames.jsonl for the Demo v5 "
            "side-by-side visualizer."
        ),
    )
    parser.add_argument(
        "--no-write-input-rgb-timeline",
        dest="write_input_rgb_timeline",
        action="store_false",
        help=(
            "Disable the side-by-side input RGB timeline even when the "
            "visualizer layout is side-by-side."
        ),
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
        "--shape-prior-timeout-ms",
        type=int,
        default=DEFAULT_SHAPE_PRIOR_TIMEOUT_MS,
    )
    parser.add_argument(
        "--shape-prior-chunk-wait-timeout-s",
        type=float,
        default=DEFAULT_SHAPE_PRIOR_CHUNK_WAIT_TIMEOUT_S,
        help=(
            "How long Demo v5 waits for required shape-prior structure points "
            "before writing final_data chunks."
        ),
    )
    parser.add_argument(
        "--shape-prior-controller-name",
        default=CONFIG_SHAPE_PRIOR_CONTROLLER_NAME,
        help="Controller label used when writing the one-camera shape-prior case.",
    )
    parser.add_argument(
        "--shape-prior-sam3d-root",
        type=Path,
        default=DEFAULT_SHAPE_PRIOR_SAM3D_ROOT,
        help="Optional SAM3D checkout override for shape-prior generation.",
    )
    parser.add_argument(
        "--shape-prior-config",
        type=Path,
        default=DEFAULT_SHAPE_PRIOR_CONFIG,
        help="Optional SAM3D pipeline config override.",
    )
    parser.add_argument("--shape-prior-profile-json", type=Path, default=None)
    parser.add_argument(
        "--mask-radius-outlier-filter",
        dest="mask_radius_outlier_filter",
        action="store_true",
        help=(
            "Apply data_process_sam3d-style 3D mask radius-outlier refinement "
            "before final_data chunking."
        ),
    )
    parser.add_argument(
        "--no-mask-radius-outlier-filter",
        dest="mask_radius_outlier_filter",
        action="store_false",
        help=(
            "Disable 3D mask radius-outlier refinement. Intended for tiny "
            "synthetic fixtures only."
        ),
    )
    parser.set_defaults(mask_radius_outlier_filter=True)
    parser.add_argument(
        "--mask-radius-outlier-radius-m",
        type=float,
        default=DEFAULT_MASK_RADIUS_OUTLIER_RADIUS_M,
    )
    parser.add_argument(
        "--mask-radius-outlier-nb-points",
        type=int,
        default=DEFAULT_MASK_RADIUS_OUTLIER_NB_POINTS,
    )
    # Side-by-side visualization starts immediately so warmup RGB is visible;
    # output-only visualization waits for the first committed chunk.
    parser.add_argument(
        "--visualizer-mode",
        choices=("window", "disabled"),
        default=DEFAULT_VISUALIZER_MODE,
        help="window launches the Demo v5 visualizer.",
    )
    parser.add_argument(
        "--visualizer-layout",
        choices=VISUALIZER_LAYOUTS,
        default=DEFAULT_VISUALIZER_LAYOUT,
        help="Viewer layout. side-by-side shows live RGB input next to final_data output chunks.",
    )
    parser.add_argument("--visualizer-conda-env", default=DEFAULT_VISUALIZER_CONDA_ENV)
    parser.add_argument(
        "--visualizer-cuda-visible-devices",
        default=DEFAULT_VISUALIZER_CUDA_VISIBLE_DEVICES,
    )
    parser.add_argument("--visualizer-cam-idx", type=int, default=DEFAULT_VISUALIZER_CAM_IDX)
    parser.add_argument("--visualizer-poll-sec", type=float, default=DEFAULT_VISUALIZER_POLL_SEC)
    parser.add_argument(
        "--visualizer-object-stride",
        type=int,
        default=DEFAULT_VISUALIZER_OBJECT_STRIDE,
    )
    parser.add_argument(
        "--visualizer-object-radius",
        type=int,
        default=DEFAULT_VISUALIZER_OBJECT_RADIUS,
    )
    parser.add_argument(
        "--visualizer-controller-radius",
        type=int,
        default=DEFAULT_VISUALIZER_CONTROLLER_RADIUS,
    )
    parser.add_argument(
        "--visualizer-render-mode",
        choices=("rgb-overlay", "sam3d-final-data"),
        default=DEFAULT_VISUALIZER_RENDER_MODE,
    )
    parser.add_argument(
        "--visualizer-object-color-mode",
        choices=("rainbow", "green", "object-colors"),
        default=DEFAULT_VISUALIZER_OBJECT_COLOR_MODE,
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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved Demo v5 contract and exit.",
    )
    return parser


def resolve_chunk_frame_count(args: argparse.Namespace) -> int:
    """Resolve the frame count used to close each online chunk."""
    if args.chunk_frame_count is not None:
        value = int(args.chunk_frame_count)
    else:
        chunk_seconds = float(args.chunk_seconds)
        replay_fps = float(args.replay_fps)
        value = int(round(replay_fps * chunk_seconds))
    if value <= 0:
        raise ValueError("chunk frame count must be positive")
    return value


def resolve_camera_source_replay_fps(args: argparse.Namespace) -> float:
    """Resolve fake-live source pacing while preserving output replay FPS."""
    value = args.camera_source_replay_fps
    fps = float(args.replay_fps if value is None else value)
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("Demo v5 source replay fps must be positive")
    return fps


def resolve_main_data_processing_cuda_visible_devices(
    args: argparse.Namespace,
) -> str:
    """Resolve the GPU namespace for the main data processing process."""
    value = str(args.main_data_processing_cuda_visible_devices).strip()
    if not value:
        raise ValueError(
            "--main-data-processing-cuda-visible-devices must be non-empty"
        )
    return value


def resolve_shape_prior_warmup_cuda_visible_devices(args: argparse.Namespace) -> str:
    """Resolve the GPU namespace used by shape-prior warmup stages."""
    value = str(args.shape_prior_warmup_cuda_visible_devices).strip()
    if not value:
        raise ValueError("--shape-prior-warmup-cuda-visible-devices must be non-empty")
    return value


def resolve_visualizer_cuda_visible_devices(args: argparse.Namespace) -> str:
    """Resolve the CUDA namespace for the optional visualizer."""
    value = str(
        getattr(
            args,
            "visualizer_cuda_visible_devices",
            DEFAULT_VISUALIZER_CUDA_VISIBLE_DEVICES,
        )
    ).strip()
    if not value:
        raise ValueError(
            "--visualizer-cuda-visible-devices must be non-empty "
            "when visualizer is enabled"
        )
    return value


def resolve_visualizer_layout(args: argparse.Namespace) -> str:
    """Validate and return the configured visualizer layout."""
    value = str(getattr(args, "visualizer_layout", DEFAULT_VISUALIZER_LAYOUT))
    if value not in VISUALIZER_LAYOUTS:
        raise ValueError(f"unsupported visualizer layout: {value!r}")
    return value


def visualizer_uses_side_by_side(args: argparse.Namespace) -> bool:
    """Return whether the viewer should show RGB input beside final_data."""
    return resolve_visualizer_layout(args) == VISUALIZER_LAYOUT_SIDE_BY_SIDE


def visualizer_start_policy(args: argparse.Namespace) -> str:
    """Describe when the visualizer should start during a live run."""
    if str(getattr(args, "visualizer_mode", DEFAULT_VISUALIZER_MODE)) != "window":
        return "disabled"
    if visualizer_uses_side_by_side(args):
        return "immediate_after_camera_start"
    return "after_first_committed_online_chunk"


def resolve_write_input_rgb_timeline(args: argparse.Namespace) -> bool:
    """Resolve whether capture should publish the side-by-side RGB timeline."""
    value = getattr(args, "write_input_rgb_timeline", None)
    if value is not None:
        return bool(value)
    return (
        str(getattr(args, "visualizer_mode", DEFAULT_VISUALIZER_MODE)) == "window"
        and visualizer_uses_side_by_side(args)
    )


def _repo_path(path: str | Path) -> Path:
    value = Path(path).expanduser()
    if value.is_absolute():
        return value
    return REPO_ROOT / value


def _python_command_prefix(conda_env: str | None) -> list[str]:
    env_name = "" if conda_env is None else str(conda_env).strip()
    if env_name:
        return ["conda", "run", "-n", env_name, "--no-capture-output", "python"]
    return ["python"]


def build_visualizer_command(
    args: argparse.Namespace,
    *,
    capture_dir: Path | None = None,
) -> list[str]:
    """Build the viewer command; side-by-side mode also receives RGB timeline paths."""
    layout = resolve_visualizer_layout(args)
    capture_text = "" if capture_dir is None else str(capture_dir)
    input_timeline_text = (
        "" if capture_dir is None else str(Path(capture_dir) / "input_frames.jsonl")
    )
    command = [
        *_python_command_prefix(getattr(args, "visualizer_conda_env", None)),
        str(Path("demo_v5_1") / "visualize_track.py"),
        "--layout",
        layout,
        "--online-dir",
        str(resolve_online_dir(args)),
        "--case-dir",
        str(Path(args.base_path) / "data" / str(args.case_prefix)),
        "--render-mode",
        str(args.visualizer_render_mode),
        "--cam-idx",
        str(int(args.visualizer_cam_idx)),
        "--fps",
        str(float(args.replay_fps)),
        "--poll-sec",
        str(float(args.visualizer_poll_sec)),
        "--object-stride",
        str(int(args.visualizer_object_stride)),
        "--object-radius",
        str(int(args.visualizer_object_radius)),
        "--controller-radius",
        str(int(args.visualizer_controller_radius)),
        "--object-color-mode",
        str(args.visualizer_object_color_mode),
    ]
    if layout == VISUALIZER_LAYOUT_SIDE_BY_SIDE:
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
    """Return the online_data directory for the active case."""
    return Path(args.base_path) / "online_data" / str(args.case_prefix)


def resolve_static_data_path(args: argparse.Namespace) -> Path:
    """Return the aggregate final_data.pkl path for the active case."""
    return Path(args.base_path) / "data" / str(args.case_prefix) / "final_data.pkl"


def _remove_generated_path(path: Path) -> bool:
    if path.is_dir():
        shutil.rmtree(path)
        return True
    if path.exists():
        path.unlink()
        return True
    return False


def prepare_realtime_output_for_new_capture(
    base_path: str | Path,
    case_prefix: str,
) -> dict[str, object]:
    """Remove stale generated outputs for a new live/fake-live run of one case."""
    base = Path(base_path)
    case_name = str(case_prefix)
    removed_online_dir = _remove_generated_path(base / "online_data" / case_name)
    removed_data_dir = _remove_generated_path(base / "data" / case_name)
    removed_manifest = _remove_generated_path(base / f"{case_name}_chunks_manifest.json")
    return {
        "removed_online_dir": bool(removed_online_dir),
        "removed_data_dir": bool(removed_data_dir),
        "removed_manifest": bool(removed_manifest),
    }


def _contract(args: argparse.Namespace) -> dict[str, object]:
    """Return the dry-run/runtime summary without launching subprocesses."""
    chunk_frame_count = int(resolve_chunk_frame_count(args))
    return {
        "demo_version": "demo_v5_1",
        "input_source": str(args.input_source),
        "replay_fps": float(args.replay_fps),
        "camera_source_replay_fps": resolve_camera_source_replay_fps(args),
        "camera_source_replay_fps_override": (
            None if args.camera_source_replay_fps is None else float(args.camera_source_replay_fps)
        ),
        "chunk_seconds": float(args.chunk_seconds),
        "chunk_poll_interval_s": float(args.chunk_poll_interval_s),
        "chunk_frame_count": chunk_frame_count,
        "allow_degraded_online": bool(args.allow_degraded_online),
        "base_path": str(args.base_path),
        "case_prefix": str(args.case_prefix),
        "output_format": "online-primary-static-case",
        "online_dir": str(resolve_online_dir(args)),
        "static_data_path": str(resolve_static_data_path(args)),
        "max_chunks": args.max_chunks,
        "depth_backend": str(args.depth_backend),
        "main_data_processing_capture_dir": (
            None if args.camera_capture_dir is None else str(args.camera_capture_dir)
        ),
        "main_data_processing_cuda_visible_devices": (
            resolve_main_data_processing_cuda_visible_devices(args)
        ),
        "perception_device": str(args.perception_device),
        "tracker_device": str(args.tracker_device),
        "inference_dtype": str(args.inference_dtype),
        "camera_lossless_max_backlog_seconds": args.camera_lossless_max_backlog_seconds,
        "camera_headless_prepared_only": bool(args.camera_headless_prepared_only),
        "write_input_rgb_timeline": resolve_write_input_rgb_timeline(args),
        "shape_prior_warmup": bool(args.shape_prior_warmup),
        "shape_prior_warmup_cuda_visible_devices": (
            resolve_shape_prior_warmup_cuda_visible_devices(args)
        ),
        "shape_prior_controller_name": str(args.shape_prior_controller_name),
        "shape_prior_sam3d_root": (
            None if args.shape_prior_sam3d_root is None else str(args.shape_prior_sam3d_root)
        ),
        "shape_prior_config": (
            None if args.shape_prior_config is None else str(args.shape_prior_config)
        ),
        "shape_prior_chunk_wait_timeout_s": float(args.shape_prior_chunk_wait_timeout_s),
        "mask_radius_outlier_filter": bool(args.mask_radius_outlier_filter),
        "mask_radius_outlier_radius_m": float(args.mask_radius_outlier_radius_m),
        "mask_radius_outlier_nb_points": int(args.mask_radius_outlier_nb_points),
        "source_headless_capture": (
            None
            if args.source_headless_capture is None
            else str(args.source_headless_capture)
        ),
        "visualizer_mode": str(args.visualizer_mode),
        "visualizer_layout": resolve_visualizer_layout(args),
        "visualizer_command": build_visualizer_command(args),
        "visualizer_cuda_visible_devices": resolve_visualizer_cuda_visible_devices(args),
        "visualizer_start_policy": visualizer_start_policy(args),
        "visualizer_capture_dir": None,
        "visualizer_fps": float(args.replay_fps),
        "visualizer_object_color_mode": str(args.visualizer_object_color_mode),
    }


def validate_runtime_args(args: argparse.Namespace, *, chunk_frame_count: int) -> None:
    """Validate cross-option constraints before launching subprocesses."""
    if float(args.chunk_poll_interval_s) <= 0.0:
        raise ValueError("--chunk-poll-interval-s must be positive")
    resolve_camera_source_replay_fps(args)
    if int(chunk_frame_count) <= 0:
        raise ValueError("chunk frame count must be positive")
    resolve_main_data_processing_cuda_visible_devices(args)
    if bool(args.shape_prior_warmup):
        resolve_shape_prior_warmup_cuda_visible_devices(args)
    if str(args.visualizer_mode) == "window":
        resolve_visualizer_layout(args)
        if int(args.visualizer_cam_idx) < 0:
            raise ValueError("--visualizer-cam-idx must be non-negative")
        if float(args.visualizer_poll_sec) <= 0.0:
            raise ValueError("--visualizer-poll-sec must be positive")
        if int(args.visualizer_object_stride) <= 0:
            raise ValueError("--visualizer-object-stride must be positive")
        if int(args.visualizer_object_radius) <= 0:
            raise ValueError("--visualizer-object-radius must be positive")
        if int(args.visualizer_controller_radius) <= 0:
            raise ValueError("--visualizer-controller-radius must be positive")
        resolve_visualizer_cuda_visible_devices(args)


def _main_data_processing_duration_s(
    args: argparse.Namespace,
    *,
    chunk_frame_count: int,
) -> float:
    # Demo v5.1 chunks are bounded by the chunk publisher, not by the camera
    # subprocess. Keeping camera duration unbounded prevents shape-prior warmup
    # time from consuming the realtime RGB input timeline.
    return 0.0


def build_main_data_processing_command(
    args: argparse.Namespace,
    *,
    capture_dir: Path,
    profile_json: Path,
    chunk_frame_count: int,
) -> list[str]:
    """Build the subprocess command that emits prepared realtime frames."""
    script = Path("demo_v5_1") / "main_data_processing.py"
    camera_source_replay_fps = resolve_camera_source_replay_fps(args)
    if str(args.depth_backend) == "ir-ffs":
        depth_source = "ffs"
    elif str(args.depth_backend) == "native-realsense":
        depth_source = "realsense"
    else:
        raise ValueError(f"unsupported depth backend: {args.depth_backend!r}")
    duration_s = _main_data_processing_duration_s(
        args,
        chunk_frame_count=chunk_frame_count,
    )
    # This is the only v5.1 camera/tracker entrypoint. It writes prepared
    # per-frame NPZ payloads plus optional input RGB timeline data; chunk
    # materialization happens in realtime_data_process_track.py.
    # Offline parity with data_process_sam3d/data_process_pcd.py:L84-L149,
    # data_process_sam3d/data_process_mask.py:L42-L152, and
    # data_process_sam3d/data_process_track.py:L49-L55. The subprocess emits the
    # realtime equivalents of those PCD, mask, and cotracker inputs.
    command = [
        "python",
        str(script),
        "--fps",
        str(int(args.camera_fps)),
        "--color-exposure",
        str(float(args.camera_color_exposure)),
        "--color-gain",
        str(float(args.camera_color_gain)),
        "--input-source",
        str(args.input_source),
        "--depth-source",
        depth_source,
        "--depth-backend-label",
        str(args.depth_backend),
        "--duration-s",
        f"{duration_s:.3f}",
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
        str(args.perception_device),
        "--dtype",
        str(args.inference_dtype),
        "--tracker-device",
        str(args.tracker_device),
        "--enable-pcd-filter",
        "--pcd-filter-mode",
        "sync",
        "--pcd-filter-preset",
        "original",
        "--table-calibrate",
        str(DEFAULT_TABLE_CALIBRATE_PATH),
        "--enable-table-z-filter",
        "--runtime-product-name",
        "demo_v5_1_main_data_processing",
        "--metadata-demo-version",
        "demo_v5_1",
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
    if float(camera_source_replay_fps) != float(DEFAULT_CAMERA_SOURCE_REPLAY_FPS):
        command.extend(["--lossless-input-fps", str(float(camera_source_replay_fps))])
    if bool(args.camera_headless_prepared_only):
        command.append("--headless-prepared-only")
    if resolve_write_input_rgb_timeline(args):
        command.append("--write-input-rgb-timeline")
    if bool(args.shape_prior_warmup):
        command.extend(
            [
                "--shape-prior-warmup",
                "--shape-prior-timeout-ms",
                str(int(args.shape_prior_timeout_ms)),
                "--shape-prior-profile-json",
                str(profile_json),
                "--shape-prior-warmup-cuda-visible-devices",
                resolve_shape_prior_warmup_cuda_visible_devices(args),
                "--shape-prior-controller-name",
                str(args.shape_prior_controller_name),
            ]
        )
        if args.shape_prior_sam3d_root is not None:
            command.extend(["--shape-prior-sam3d-root", str(args.shape_prior_sam3d_root)])
        if args.shape_prior_config is not None:
            command.extend(["--shape-prior-config", str(args.shape_prior_config)])
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


def _visualizer_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = resolve_visualizer_cuda_visible_devices(args)
    return env


def _start_visualizer(
    args: argparse.Namespace,
    *,
    capture_dir: Path | None = None,
) -> subprocess.Popen[bytes]:
    """Launch the lightweight online visualizer in the repo environment."""
    return subprocess.Popen(
        build_visualizer_command(args, capture_dir=capture_dir),
        cwd=REPO_ROOT,
        env=_visualizer_env(args),
        start_new_session=True,
    )


def _runtime_chunk_summary(manifests: Sequence[dict[str, object]]) -> dict[str, object]:
    publish_times = [
        float(item["publish_wall_s"])
        for item in manifests
        if item.get("publish_wall_s") is not None
    ]
    intervals = [
        publish_times[idx] - publish_times[idx - 1]
        for idx in range(1, len(publish_times))
    ]
    backlog_values = [
        int(item["backlog_chunks"])
        for item in manifests
        if item.get("backlog_chunks") is not None
    ]
    shape_publish_times = [
        float(item["publish_wall_s"])
        for item in manifests
        if item.get("publish_wall_s") is not None
        and bool(item.get("shape_prior_complete"))
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
        str(item.get("chunk_name", item.get("chunk_index", "")))
        for item in manifests
        if str(item.get("track_process_status", "normal")) == "invalid"
    ]
    return {
        "first_ready_chunk_wall_s": publish_times[0] if publish_times else None,
        "first_shape_prior_ready_chunk_wall_s": (
            shape_publish_times[0] if shape_publish_times else None
        ),
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
    """Run Demo v5.1 offline conversion or live/fake-live orchestration."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    chunk_frame_count = resolve_chunk_frame_count(args)
    validate_runtime_args(args, chunk_frame_count=chunk_frame_count)

    if bool(args.dry_run):
        print(json.dumps(_contract(args), indent=2, sort_keys=True))
        return 0

    base_path = Path(args.base_path)
    base_path.mkdir(parents=True, exist_ok=True)
    startup_realtime_case_cleanup = None
    if args.source_headless_capture is None:
        startup_realtime_case_cleanup = prepare_realtime_output_for_new_capture(
            base_path,
            str(args.case_prefix),
        )
    if args.source_headless_capture is not None:
        # Offline conversion path: consume an existing capture directory and
        # write online/static final_data products without launching camera or
        # visualizer subprocesses.
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
            allow_degraded_online=bool(args.allow_degraded_online),
        )
        summary = {
            "demo_version": "demo_v5_1",
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
        }
        summary.update(_runtime_chunk_summary(manifests))
        summary_path = base_path / f"{args.case_prefix}_chunks_manifest.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 1 if str(summary.get("track_process_status", "normal")) == "invalid" else 0

    capture_dir = _default_capture_dir(args, base_path)
    capture_dir.mkdir(parents=True, exist_ok=True)
    profile_json = (
        Path(args.shape_prior_profile_json)
        if args.shape_prior_profile_json is not None
        else capture_dir / "shape_prior_profile.json"
    )
    main_data_processing_command = build_main_data_processing_command(
        args,
        capture_dir=capture_dir,
        profile_json=profile_json,
        chunk_frame_count=chunk_frame_count,
    )
    main_data_processing_env = os.environ.copy()
    if not main_data_processing_env.get(SAM31_CHECKPOINT_ENV):
        main_data_processing_env[SAM31_CHECKPOINT_ENV] = str(
            _repo_path(DEFAULT_SAM31_CHECKPOINT_PATH)
        )
    main_data_processing_cuda_visible_devices = (
        resolve_main_data_processing_cuda_visible_devices(args).strip()
    )
    if main_data_processing_cuda_visible_devices:
        main_data_processing_env["CUDA_VISIBLE_DEVICES"] = (
            main_data_processing_cuda_visible_devices
        )
    visualizer_process: subprocess.Popen[bytes] | None = None
    visualizer_started = False
    visualizer_started_manifest: dict[str, object] | None = None
    visualizer_start_wall_s: float | None = None
    visualizer_return_code: int | None = None
    visualizer_left_running = False

    def on_chunk_written(manifest: dict[str, object]) -> None:
        """Start downstream consumers exactly once when the first chunk commits."""
        nonlocal visualizer_process
        nonlocal visualizer_started
        nonlocal visualizer_started_manifest
        nonlocal visualizer_start_wall_s
        # Output-only viewing starts after the first committed chunk. The
        # side-by-side visualizer starts immediately after launch so warmup
        # RGB remains visible while the output side waits for chunks.
        if (
            str(args.visualizer_mode) == "window"
            and not visualizer_uses_side_by_side(args)
            and visualizer_process is None
        ):
            visualizer_process = _start_visualizer(args)
            visualizer_started = True
            visualizer_started_manifest = dict(manifest)
            visualizer_start_wall_s = time.monotonic()

    main_data_processing = subprocess.Popen(
        main_data_processing_command,
        env=main_data_processing_env,
        start_new_session=True,
    )
    if str(args.visualizer_mode) == "window" and visualizer_uses_side_by_side(args):
        visualizer_process = _start_visualizer(args, capture_dir=capture_dir)
        visualizer_started = True
        visualizer_start_wall_s = time.monotonic()
    surface_points = _load_optional_points(args.surface_points_npy)
    interior_points = _load_optional_points(args.interior_points_npy)
    try:
        # The bridge tails frames.jsonl and publishes fixed-size chunks while
        # the camera subprocess is still running, so fake-live and live share the
        # same realtime chunking path.
        # Offline parity with data_process_track.py:L37-L378 and
        # data_process_sample.py:L250-L352. stream_chunks_from_headless_capture
        # materializes those outputs incrementally instead of after the
        # recording has finished.
        manifests = stream_chunks_from_headless_capture(
            capture_dir,
            base_path=base_path,
            case_prefix=str(args.case_prefix),
            chunk_frame_count=chunk_frame_count,
            fps=int(round(float(args.replay_fps))),
            max_chunks=args.max_chunks,
            capture_finished=lambda: main_data_processing.poll() is not None,
            require_shape_prior=bool(args.shape_prior_warmup),
            shape_prior_wait_timeout_s=float(args.shape_prior_chunk_wait_timeout_s),
            poll_interval_s=float(args.chunk_poll_interval_s),
            surface_points=surface_points,
            interior_points=interior_points,
            mask_radius_outlier_filter=bool(args.mask_radius_outlier_filter),
            mask_radius_outlier_radius_m=float(args.mask_radius_outlier_radius_m),
            mask_radius_outlier_nb_points=int(args.mask_radius_outlier_nb_points),
            on_chunk_written=on_chunk_written,
            allow_degraded_online=bool(args.allow_degraded_online),
        )
    finally:
        main_data_processing_return_code = _stop_process(
            main_data_processing
        )
        if visualizer_process is not None:
            visualizer_return_code = visualizer_process.poll()
            visualizer_left_running = visualizer_return_code is None
    runtime_summary = _runtime_chunk_summary(manifests)
    track_process_invalid = str(runtime_summary.get("track_process_status", "normal")) == "invalid"
    if track_process_invalid:
        stop_reason = "track_process_invalid"
    elif args.max_chunks is not None and len(manifests) >= int(args.max_chunks):
        stop_reason = "max_chunks_reached"
    elif main_data_processing_return_code == 0:
        stop_reason = "main_data_processing_completed"
    elif main_data_processing_return_code is None:
        stop_reason = "main_data_processing_status_unknown"
    else:
        stop_reason = "main_data_processing_exited_before_target"
    summary = {
        "demo_version": "demo_v5_1",
        "mode": (
            "full-fake-main-data-processing"
            if str(args.input_source) == "fake-live"
            else "full-live-main-data-processing"
        ),
        "main_data_processing_command": main_data_processing_command,
        "main_data_processing_cuda_visible_devices": (
            main_data_processing_cuda_visible_devices
        ),
        "camera_lossless_max_backlog_seconds": args.camera_lossless_max_backlog_seconds,
        "camera_headless_prepared_only": bool(args.camera_headless_prepared_only),
        "write_input_rgb_timeline": resolve_write_input_rgb_timeline(args),
        "camera_source_replay_fps": resolve_camera_source_replay_fps(args),
        "camera_source_replay_fps_override": (
            None if args.camera_source_replay_fps is None else float(args.camera_source_replay_fps)
        ),
        "shape_prior_warmup_cuda_visible_devices": (
            resolve_shape_prior_warmup_cuda_visible_devices(args)
        ),
        "shape_prior_controller_name": str(args.shape_prior_controller_name),
        "shape_prior_sam3d_root": (
            None if args.shape_prior_sam3d_root is None else str(args.shape_prior_sam3d_root)
        ),
        "shape_prior_config": (
            None if args.shape_prior_config is None else str(args.shape_prior_config)
        ),
        "main_data_processing_return_code": (
            main_data_processing_return_code
        ),
        "main_data_processing_stop_reason": stop_reason,
        "main_data_processing_capture_dir": str(capture_dir),
        "base_path": str(base_path),
        "case_prefix": str(args.case_prefix),
        "output_format": "online-primary-static-case",
        "online_dir": str(resolve_online_dir(args)),
        "static_data_path": str(resolve_static_data_path(args)),
        "startup_realtime_case_cleanup": startup_realtime_case_cleanup,
        "chunk_frame_count": int(chunk_frame_count),
        "chunk_poll_interval_s": float(args.chunk_poll_interval_s),
        "allow_degraded_online": bool(args.allow_degraded_online),
        "max_chunks": args.max_chunks,
        "chunk_count": int(len(manifests)),
        "chunks": manifests,
        "external_shape_prior_points": bool(
            surface_points is not None or interior_points is not None
        ),
        "visualizer_mode": str(args.visualizer_mode),
        "visualizer_layout": resolve_visualizer_layout(args),
        "visualizer_started": visualizer_started,
        "visualizer_start_policy": visualizer_start_policy(args),
        "visualizer_capture_dir": str(capture_dir) if visualizer_uses_side_by_side(args) else None,
        "visualizer_started_from_chunk": visualizer_started_manifest,
        "visualizer_start_wall_s": visualizer_start_wall_s,
        "visualizer_command": build_visualizer_command(
            args,
            capture_dir=capture_dir if visualizer_uses_side_by_side(args) else None,
        ),
        "visualizer_cuda_visible_devices": resolve_visualizer_cuda_visible_devices(args),
        "visualizer_fps": float(args.replay_fps),
        "visualizer_object_color_mode": str(args.visualizer_object_color_mode),
        "visualizer_return_code": visualizer_return_code,
        "visualizer_left_running": visualizer_left_running,
    }
    summary.update(runtime_summary)
    summary_path = base_path / f"{args.case_prefix}_chunks_manifest.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if track_process_invalid:
        return 1
    if main_data_processing_return_code not in (0, None) and not manifests:
        return int(main_data_processing_return_code)
    if args.max_chunks is not None and len(manifests) < int(args.max_chunks):
        return 1
    if str(args.visualizer_mode) == "window" and not visualizer_started:
        return 1
    if visualizer_return_code not in (0, None):
        return int(visualizer_return_code)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
