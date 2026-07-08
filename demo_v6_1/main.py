#!/usr/bin/env python3
"""Demo v6.1 realtime orchestration entrypoint.

This runner owns process boundaries, GPU routing, and artifact publication. The
actual camera/tracker stack runs in ``demo_v6_1/main_data_processing.py``;
SAM3D shape prior warmup runs as local one-shot stages; the default
side-by-side visualizer starts as soon as capture starts.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
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

from demo_v6_1.chunk_data_stream import (
    stream_chunk_data_from_headless_capture,
    write_chunk_data_from_headless_capture,
)
from demo_v6_1.phystwin_shen_launch import (
    PhystwinShenLaunch,
    PhystwinShenSettings,
    launch_phystwin_shen,
    validate_phystwin_shen_repo,
)


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "default.yaml"


def load_default_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, object]:
    """Load Demo v6.1 defaults from YAML."""
    text = Path(path).read_text(encoding="utf-8")
    loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise ValueError(f"default config must be a mapping: {path}")
    return dict(loaded)


_DEFAULT_CONFIG = load_default_config()


def _cfg(section: str, key: str) -> object:
    """Read one default; config/default.yaml is the single source of defaults."""
    return _DEFAULT_CONFIG[section][key]


def _cfg_optional_path(section: str, key: str) -> Path | None:
    """Read an optional path default; empty/None YAML values mean "unset"."""
    value = _cfg(section, key)
    if value is None or str(value).strip() == "":
        return None
    return Path(str(value))


# Defaults below describe the current Demo v6.1 realtime path.
DEFAULT_DATA_PROCESS_BASE_PATH = Path(str(_cfg("paths", "data_process_base_path")))
DEFAULT_INPUT_SOURCE = str(_cfg("input", "input_source"))
DEFAULT_FAKE_LIVE_CASE = _cfg_optional_path("input", "fake_live_case")
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
DEFAULT_SHAPE_PRIOR_SAM3D_ROOT = _cfg_optional_path(
    "shape_prior", "shape_prior_sam3d_root"
)
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
DEFAULT_EDGETAM_MASK_LOGIT_THRESHOLD = float(
    _cfg("camera", "edgetam_mask_logit_threshold")
)
# Downstream consumers are mutually exclusive per session: the Demo v6.1
# viewer window or the Phystwin_shen online trainer + HTML viewer.
DOWNSTREAM_MODE_DISABLED = "disabled"
DOWNSTREAM_MODE_DEMO_VISUALIZER = "demo_visualizer"
DOWNSTREAM_MODE_PHYSTWIN_SHEN = "phystwin_shen"
DOWNSTREAM_MODES = (
    DOWNSTREAM_MODE_DISABLED,
    DOWNSTREAM_MODE_DEMO_VISUALIZER,
    DOWNSTREAM_MODE_PHYSTWIN_SHEN,
)
DEFAULT_DOWNSTREAM_MODE = str(_cfg("downstream", "mode"))
DEFAULT_PHYSTWIN_SHEN_REPO_PATH = Path(str(_cfg("phystwin_shen", "repo_path")))
DEFAULT_PHYSTWIN_SHEN_CONDA_ENV = str(_cfg("phystwin_shen", "conda_env"))
DEFAULT_PHYSTWIN_SHEN_CASE_NAME = str(_cfg("phystwin_shen", "case_name"))
DEFAULT_PHYSTWIN_SHEN_VIEWER_HOST = str(_cfg("phystwin_shen", "viewer_host"))
DEFAULT_PHYSTWIN_SHEN_VIEWER_PORT = int(_cfg("phystwin_shen", "viewer_port"))
PHYSTWIN_SHEN_VIEWER_CAM_IDX = int(_cfg("phystwin_shen", "viewer_cam_idx"))
PHYSTWIN_SHEN_VIEWER_POINT_MODE = str(_cfg("phystwin_shen", "viewer_point_mode"))
PHYSTWIN_SHEN_VIEWER_POINT_STRIDE = int(_cfg("phystwin_shen", "viewer_point_stride"))
PHYSTWIN_SHEN_VIEWER_IMAGE_INDEX_MODE = str(
    _cfg("phystwin_shen", "viewer_image_index_mode")
)
PHYSTWIN_SHEN_TRAIN_DEVICE = str(_cfg("phystwin_shen", "train_device"))
PHYSTWIN_SHEN_TRAIN_BATCH_SIZE = int(_cfg("phystwin_shen", "train_batch_size"))
PHYSTWIN_SHEN_TRAIN_SEGMENT_LEN = int(_cfg("phystwin_shen", "train_segment_len"))
PHYSTWIN_SHEN_TRAIN_SEGMENT_STRIDE = int(_cfg("phystwin_shen", "train_segment_stride"))
PHYSTWIN_SHEN_TRAIN_POLL_SEC = float(_cfg("phystwin_shen", "train_poll_sec"))
PHYSTWIN_SHEN_TRAIN_RECENT_WINDOW_COUNT = int(
    _cfg("phystwin_shen", "train_recent_window_count")
)
PHYSTWIN_SHEN_TRAIN_REALTIME_VIS_EVERY = int(
    _cfg("phystwin_shen", "train_realtime_vis_every")
)
PHYSTWIN_SHEN_TRAIN_STOP_WHEN_FINISHED = bool(
    _cfg("phystwin_shen", "train_stop_when_finished")
)
DEFAULT_PHYSTWIN_SHEN_CUDA_VISIBLE_DEVICES = str(
    _cfg("gpu", "phystwin_shen_cuda_visible_devices")
)
DEFAULT_VISUALIZER_CONDA_ENV = str(_cfg("visualizer", "visualizer_conda_env"))
DEFAULT_VISUALIZER_CAM_IDX = int(_cfg("visualizer", "visualizer_cam_idx"))
DEFAULT_VISUALIZER_POLL_SEC = float(_cfg("visualizer", "visualizer_poll_sec"))
DEFAULT_VISUALIZER_PLAYBACK_FPS = float(
    _cfg("visualizer", "visualizer_playback_fps")
)
DEFAULT_VISUALIZER_OBJECT_STRIDE = int(_cfg("visualizer", "visualizer_object_stride"))
DEFAULT_VISUALIZER_OBJECT_RADIUS = int(_cfg("visualizer", "visualizer_object_radius"))
DEFAULT_VISUALIZER_CONTROLLER_RADIUS = int(
    _cfg("visualizer", "visualizer_controller_radius")
)
DEFAULT_VISUALIZER_OBJECT_COLOR_MODE = str(
    _cfg("visualizer", "visualizer_object_color_mode")
)
VISUALIZER_LAYOUT_SIDE_BY_SIDE = str(
    _cfg("visualizer", "visualizer_layout_side_by_side")
)
VISUALIZER_LAYOUT_OUTPUT_ONLY = str(_cfg("visualizer", "visualizer_layout_output_only"))
VISUALIZER_LAYOUTS = tuple(
    str(item) for item in _cfg("visualizer", "visualizer_layouts")
)
DEFAULT_VISUALIZER_LAYOUT = str(_cfg("visualizer", "visualizer_layout"))
DEFAULT_VISUALIZER_RENDER_MODE = str(_cfg("visualizer", "visualizer_render_mode"))
DEFAULT_TABLE_CALIBRATE_PATH = Path(str(_cfg("paths", "table_calibrate_path")))
DEFAULT_SAM31_CHECKPOINT_PATH = Path(str(_cfg("paths", "sam31_checkpoint_path")))
SAM31_CHECKPOINT_ENV = str(_cfg("paths", "sam31_checkpoint_env"))
EDGE_TAM_TRACKING_IDENTITIES = ("hand_a", "object", "hand_b")

CAPTURE_DIR_NAME = "capture"
DATA_DIR_NAME = "data"
ONLINE_DATA_DIR_NAME = "online_data"
SHAPE_PRIOR_CASE_DIR_NAME = "shape_prior_case"
SHAPE_PRIOR_DIR_NAME = "shape_prior"
RUN_SUMMARY_NAME = "run_summary.json"


# ---------------------------------------------------------------------------
# CLI definition
# ---------------------------------------------------------------------------


class _StoreFakeLiveCase(argparse.Action):
    """Track whether --fake-live-case was explicitly provided."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: object,
        option_string: str | None = None,
    ) -> None:
        del parser, option_string
        setattr(namespace, self.dest, values)
        setattr(namespace, "fake_live_case_cli_override", True)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for Demo v6.1 realtime orchestration."""
    parser = argparse.ArgumentParser(
        description=(
            "Demo v6.1 realtime data_process_sam3d runner. It turns Demo v6.1 "
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
        help="Camera source mode used when Demo v6.1 launches its own capture.",
    )
    parser.add_argument("--replay-fps", type=float, default=DEFAULT_REPLAY_FPS)
    parser.add_argument(
        "--camera-source-replay-fps",
        type=float,
        default=None,
        help=(
            "Optional Demo v6.1 fake-live pacing FPS. When omitted, Demo v6.1 uses "
            "--replay-fps; Demo v6.1 output metadata/window math still use --replay-fps."
        ),
    )
    parser.add_argument(
        "--fake-live-case",
        action=_StoreFakeLiveCase,
        type=Path,
        default=DEFAULT_FAKE_LIVE_CASE,
        help="Raw data_collect case folder passed to Demo v6.1 fake-live replay.",
    )
    parser.set_defaults(fake_live_case_cli_override=False)
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
        "--base-path", type=Path, default=DEFAULT_DATA_PROCESS_BASE_PATH
    )
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
        "--edgetam-mask-logit-threshold",
        type=float,
        default=DEFAULT_EDGETAM_MASK_LOGIT_THRESHOLD,
        help=(
            "Logit threshold passed to EdgeTAM mask binarization. "
            "Lower values make masks more permissive."
        ),
    )
    parser.add_argument(
        "--camera-lossless-max-backlog-seconds",
        type=float,
        default=None,
        help=(
            "Optional strict lossless replay backlog window passed to Demo v6.1. "
            "Omit it to keep Demo v6.1 defaults."
        ),
    )
    parser.add_argument(
        "--camera-fps",
        type=int,
        choices=CAMERA_FPS_CHOICES,
        default=DEFAULT_CAMERA_FPS,
        help=(
            "RealSense capture FPS passed to Demo v6.1 live camera. "
            "The default 30 FPS input is sampled at replay FPS for output."
        ),
    )
    parser.add_argument(
        "--camera-color-exposure",
        type=float,
        default=DEFAULT_CAMERA_COLOR_EXPOSURE,
        help="Manual RealSense RGB exposure passed to Demo v6.1 live camera.",
    )
    parser.add_argument(
        "--camera-color-gain",
        type=float,
        default=DEFAULT_CAMERA_COLOR_GAIN,
        help="Manual RealSense RGB gain passed to Demo v6.1 live camera.",
    )
    # The current chunker consumes prepared frames directly. Keeping only
    # prepared artifacts keeps live runs small and avoids old per-frame outputs
    # becoming part of the v6.1 contract.
    parser.add_argument(
        "--camera-headless-prepared-only",
        dest="camera_headless_prepared_only",
        action="store_true",
        help="Ask Demo v6.1 to write only prepared PhysTwin frames needed by Demo v6.1 chunking.",
    )
    parser.add_argument(
        "--camera-legacy-headless-artifacts",
        dest="camera_headless_prepared_only",
        action="store_false",
        help=(
            "Keep Demo v6.1 legacy per-frame headless artifacts in addition to "
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
            "Write input_rgb/*.png and input_frames.jsonl for the Demo v6.1 "
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
        help="Headless capture directory for the Demo v6.1 realtime subprocess.",
    )
    parser.add_argument(
        "--source-headless-capture",
        type=Path,
        default=None,
        help="Existing Demo v6.1 headless capture directory to chunk without launching capture.",
    )
    parser.add_argument("--surface-points-npy", type=Path, default=None)
    parser.add_argument("--interior-points-npy", type=Path, default=None)
    parser.add_argument(
        "--shape-prior-warmup",
        dest="shape_prior_warmup",
        action="store_true",
        help="Keep SAM3D shape-prior warmup enabled for Demo v6.1 capture.",
    )
    parser.add_argument(
        "--no-shape-prior-warmup",
        dest="shape_prior_warmup",
        action="store_false",
        help="Disable SAM3D shape-prior warmup.",
    )
    parser.set_defaults(shape_prior_warmup=True)
    parser.add_argument(
        "--shape-prior-prewarm-stage-workers",
        dest="shape_prior_prewarm_stage_workers",
        action="store_true",
        help=(
            "Spawn pre-warmed one-shot upscale/generate/align workers at app "
            "boot so model loading happens before frame 0 arrives."
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
        default=DEFAULT_SHAPE_PRIOR_TIMEOUT_MS,
    )
    parser.add_argument(
        "--shape-prior-chunk-wait-timeout-s",
        type=float,
        default=DEFAULT_SHAPE_PRIOR_CHUNK_WAIT_TIMEOUT_S,
        help=(
            "How long Demo v6.1 waits for required shape-prior structure points "
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
        "--asap-augment",
        dest="asap_augment",
        action="store_true",
        help=(
            "Fill invalid object_points in place and publish deformed "
            "shape-prior trajectories as asap_surface_points/"
            "asap_interior_points via live ASAP mesh deformation "
            "(design_spec_v6_1.md)."
        ),
    )
    parser.add_argument(
        "--no-asap-augment",
        dest="asap_augment",
        action="store_false",
        help="Disable ASAP augmentation of published chunks.",
    )
    parser.set_defaults(asap_augment=True)
    parser.add_argument(
        "--asap-mesh-path",
        type=Path,
        default=None,
        help=(
            "Explicit final_mesh.glb override for ASAP augmentation. Defaults "
            "to <shape_prior_case_dir>/shape/matching/final_mesh.glb from the "
            "capture metadata; ASAP fails fast when the mesh is missing."
        ),
    )
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
    # Exactly one downstream consumer runs per session. demo_visualizer keeps
    # the historical viewer policies (side-by-side starts immediately,
    # output-only waits for the first committed chunk); phystwin_shen starts
    # Phystwin_shen training + HTML viewer when the shape prior is ready.
    parser.add_argument(
        "--downstream-mode",
        choices=DOWNSTREAM_MODES,
        default=DEFAULT_DOWNSTREAM_MODE,
        help=(
            "Downstream consumer of the online stream: disabled, the Demo "
            "v6.1 viewer window, or Phystwin_shen train + HTML viewer."
        ),
    )
    parser.add_argument(
        "--phystwin-shen-repo",
        type=Path,
        default=DEFAULT_PHYSTWIN_SHEN_REPO_PATH,
        help="Phystwin_shen checkout used by --downstream-mode phystwin_shen.",
    )
    parser.add_argument(
        "--phystwin-shen-conda-env",
        default=DEFAULT_PHYSTWIN_SHEN_CONDA_ENV,
    )
    parser.add_argument(
        "--phystwin-shen-cuda-visible-devices",
        default=DEFAULT_PHYSTWIN_SHEN_CUDA_VISIBLE_DEVICES,
    )
    parser.add_argument(
        "--phystwin-shen-viewer-host",
        default=DEFAULT_PHYSTWIN_SHEN_VIEWER_HOST,
    )
    parser.add_argument(
        "--phystwin-shen-viewer-port",
        type=int,
        default=DEFAULT_PHYSTWIN_SHEN_VIEWER_PORT,
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
    parser.add_argument(
        "--visualizer-cam-idx", type=int, default=DEFAULT_VISUALIZER_CAM_IDX
    )
    parser.add_argument(
        "--visualizer-poll-sec", type=float, default=DEFAULT_VISUALIZER_POLL_SEC
    )
    parser.add_argument(
        "--visualizer-playback-fps",
        type=float,
        default=DEFAULT_VISUALIZER_PLAYBACK_FPS,
        help="Playback FPS for the visualizer final_data timeline.",
    )
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
        help="Print resolved Demo v6.1 contract and exit.",
    )
    return parser


# ---------------------------------------------------------------------------
# Option resolution
# ---------------------------------------------------------------------------


def resolve_chunk_frame_count(args: argparse.Namespace) -> int:
    """Resolve the frame count used to close each online chunk."""
    if args.chunk_frame_count is not None:
        value = int(args.chunk_frame_count)
    else:
        # Chunks are sized on the output replay timeline (--replay-fps), not
        # the camera capture FPS, so each chunk spans chunk_seconds of output.
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
        raise ValueError("Demo v6.1 source replay fps must be positive")
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


def resolve_downstream_mode(args: argparse.Namespace) -> str:
    """Validate and return the downstream consumer mode.

    YAML-sourced defaults bypass argparse ``choices``, so the enum is
    enforced here as well.
    """
    value = str(getattr(args, "downstream_mode", DEFAULT_DOWNSTREAM_MODE))
    if value not in DOWNSTREAM_MODES:
        raise ValueError(
            f"unsupported downstream mode: {value!r}; expected one of "
            f"{DOWNSTREAM_MODES}"
        )
    return value


def demo_visualizer_enabled(args: argparse.Namespace) -> bool:
    """Return whether the Demo v6.1 viewer window is the downstream consumer."""
    return resolve_downstream_mode(args) == DOWNSTREAM_MODE_DEMO_VISUALIZER


def phystwin_shen_enabled(args: argparse.Namespace) -> bool:
    """Return whether Phystwin_shen is the downstream consumer."""
    return resolve_downstream_mode(args) == DOWNSTREAM_MODE_PHYSTWIN_SHEN


def resolve_phystwin_shen_cuda_visible_devices(args: argparse.Namespace) -> str:
    """Resolve the GPU namespace for the Phystwin_shen subprocesses."""
    value = str(
        getattr(
            args,
            "phystwin_shen_cuda_visible_devices",
            DEFAULT_PHYSTWIN_SHEN_CUDA_VISIBLE_DEVICES,
        )
    ).strip()
    if not value:
        raise ValueError(
            "--phystwin-shen-cuda-visible-devices must be non-empty when "
            "downstream mode is phystwin_shen"
        )
    return value


def resolve_phystwin_shen_settings(args: argparse.Namespace) -> PhystwinShenSettings:
    """Assemble the Phystwin_shen launch settings from config/CLI."""
    return PhystwinShenSettings(
        repo_path=Path(args.phystwin_shen_repo).expanduser().resolve(),
        conda_env=str(args.phystwin_shen_conda_env),
        case_name=DEFAULT_PHYSTWIN_SHEN_CASE_NAME,
        base_path=Path(args.base_path).expanduser().resolve(),
        cuda_visible_devices=resolve_phystwin_shen_cuda_visible_devices(args),
        viewer_host=str(args.phystwin_shen_viewer_host),
        viewer_port=int(args.phystwin_shen_viewer_port),
        viewer_cam_idx=PHYSTWIN_SHEN_VIEWER_CAM_IDX,
        viewer_point_mode=PHYSTWIN_SHEN_VIEWER_POINT_MODE,
        viewer_point_stride=PHYSTWIN_SHEN_VIEWER_POINT_STRIDE,
        viewer_image_index_mode=PHYSTWIN_SHEN_VIEWER_IMAGE_INDEX_MODE,
        train_device=PHYSTWIN_SHEN_TRAIN_DEVICE,
        train_batch_size=PHYSTWIN_SHEN_TRAIN_BATCH_SIZE,
        train_segment_len=PHYSTWIN_SHEN_TRAIN_SEGMENT_LEN,
        train_segment_stride=PHYSTWIN_SHEN_TRAIN_SEGMENT_STRIDE,
        train_poll_sec=PHYSTWIN_SHEN_TRAIN_POLL_SEC,
        train_recent_window_count=PHYSTWIN_SHEN_TRAIN_RECENT_WINDOW_COUNT,
        train_realtime_vis_every=PHYSTWIN_SHEN_TRAIN_REALTIME_VIS_EVERY,
        train_stop_when_finished=PHYSTWIN_SHEN_TRAIN_STOP_WHEN_FINISHED,
    )


def visualizer_start_policy(args: argparse.Namespace) -> str:
    """Describe when the visualizer should start during a live run."""
    if not demo_visualizer_enabled(args):
        return "disabled"
    if visualizer_uses_side_by_side(args):
        return "immediate_after_camera_start"
    return "after_first_committed_online_chunk"


def resolve_write_input_rgb_timeline(args: argparse.Namespace) -> bool:
    """Resolve whether capture should publish the side-by-side RGB timeline."""
    value = getattr(args, "write_input_rgb_timeline", None)
    if value is not None:
        return bool(value)
    # Default: the timeline only exists for the side-by-side viewer, so write
    # it exactly when that viewer will run.
    return demo_visualizer_enabled(args) and visualizer_uses_side_by_side(args)


def _python_command_prefix(conda_env: str | None) -> list[str]:
    """Return the python command prefix."""
    env_name = "" if conda_env is None else str(conda_env).strip()
    if env_name:
        active_env = os.environ.get("CONDA_DEFAULT_ENV", "").strip()
        if active_env == env_name:
            # Avoid nesting `conda run` inside the same long-running demo env:
            # the wrapper can outlive/crash separately from the real child.
            return [sys.executable]
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
        str(Path("demo_v6_1") / "visualize_track.py"),
        "--layout",
        layout,
        "--online-dir",
        str(resolve_online_dir(args)),
        "--case-dir",
        str(resolve_static_data_dir(args)),
        "--render-mode",
        str(args.visualizer_render_mode),
        "--cam-idx",
        str(int(args.visualizer_cam_idx)),
        "--fps",
        str(float(args.visualizer_playback_fps)),
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
    """Load an optional Nx3 float64 point array from an .npy file."""
    if path is None:
        return None
    arr = np.asarray(np.load(path), dtype=np.float64)
    if arr.size == 0:
        # Normalize empty inputs to (0, 3) so downstream shape checks stay
        # uniform regardless of how the empty array was saved.
        return np.empty((0, 3), dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{path} must contain an Nx3 point array")
    return np.ascontiguousarray(arr, dtype=np.float64)


# ---------------------------------------------------------------------------
# Fixed output layout under --base-path
# ---------------------------------------------------------------------------


def resolve_online_dir(args: argparse.Namespace) -> Path:
    """Return the fixed online_data directory."""
    return Path(args.base_path) / ONLINE_DATA_DIR_NAME


def resolve_static_data_dir(args: argparse.Namespace) -> Path:
    """Return the fixed aggregate data directory."""
    return Path(args.base_path) / DATA_DIR_NAME


def resolve_static_data_path(args: argparse.Namespace) -> Path:
    """Return the aggregate final_data.pkl path."""
    return resolve_static_data_dir(args) / "final_data.pkl"


def resolve_shape_prior_case_root(args: argparse.Namespace) -> Path:
    """Return the fixed shape-prior case root."""
    return Path(args.base_path) / SHAPE_PRIOR_CASE_DIR_NAME


def resolve_shape_prior_points_npz(args: argparse.Namespace) -> Path:
    """Return the fixed shape-prior points export path."""
    return Path(args.base_path) / SHAPE_PRIOR_DIR_NAME / "points.npz"


def resolve_run_summary_path(base_path: str | Path) -> Path:
    """Return the fixed run summary path."""
    return Path(base_path) / RUN_SUMMARY_NAME


def _remove_generated_path(path: Path) -> bool:
    """Delete a generated file or directory; return True when it existed."""
    if path.is_dir():
        shutil.rmtree(path)
        return True
    if path.exists():
        path.unlink()
        return True
    return False


def prepare_realtime_output_for_new_run(
    base_path: str | Path,
    *,
    clear_capture: bool,
    legacy_case_prefix: str,
) -> dict[str, object]:
    """Remove stale generated outputs before writing fixed Demo v6.1 paths."""
    base = Path(base_path)
    cleanup_paths = {
        "capture": base / CAPTURE_DIR_NAME,
        "shape_prior_case": base / SHAPE_PRIOR_CASE_DIR_NAME,
        "shape_prior": base / SHAPE_PRIOR_DIR_NAME,
        "data": base / DATA_DIR_NAME,
        "online_data": base / ONLINE_DATA_DIR_NAME,
        "run_summary": resolve_run_summary_path(base),
        "legacy_chunks_manifest": base / f"{legacy_case_prefix}_chunks_manifest.json",
    }
    if not bool(clear_capture):
        cleanup_paths.pop("capture")
    return {
        f"removed_{name}": bool(_remove_generated_path(path))
        for name, path in cleanup_paths.items()
    }


def _contract(args: argparse.Namespace) -> dict[str, object]:
    """Return the dry-run/runtime summary without launching subprocesses."""
    chunk_frame_count = int(resolve_chunk_frame_count(args))
    return {
        "demo_version": "demo_v6_1",
        "input_source": str(args.input_source),
        "replay_fps": float(args.replay_fps),
        "camera_source_replay_fps": resolve_camera_source_replay_fps(args),
        "camera_source_replay_fps_override": (
            None
            if args.camera_source_replay_fps is None
            else float(args.camera_source_replay_fps)
        ),
        "chunk_seconds": float(args.chunk_seconds),
        "chunk_poll_interval_s": float(args.chunk_poll_interval_s),
        "chunk_frame_count": chunk_frame_count,
        "base_path": str(args.base_path),
        "case_prefix": str(args.case_prefix),
        "output_format": "online-primary-static-case",
        "online_dir": str(resolve_online_dir(args)),
        "static_data_path": str(resolve_static_data_path(args)),
        "shape_prior_case_root": str(resolve_shape_prior_case_root(args)),
        "shape_prior_points_npz": str(resolve_shape_prior_points_npz(args)),
        "max_chunks": args.max_chunks,
        "depth_backend": str(args.depth_backend),
        "edgetam_tracking_identities": list(EDGE_TAM_TRACKING_IDENTITIES),
        "main_data_processing_capture_dir": str(
            _default_capture_dir(args, Path(args.base_path))
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
        "shape_prior_prewarm_stage_workers": bool(
            args.shape_prior_prewarm_stage_workers
        ),
        "shape_prior_warmup_cuda_visible_devices": (
            resolve_shape_prior_warmup_cuda_visible_devices(args)
        ),
        "shape_prior_controller_name": str(args.shape_prior_controller_name),
        "shape_prior_sam3d_root": (
            None
            if args.shape_prior_sam3d_root is None
            else str(args.shape_prior_sam3d_root)
        ),
        "shape_prior_config": (
            None if args.shape_prior_config is None else str(args.shape_prior_config)
        ),
        "shape_prior_chunk_wait_timeout_s": float(
            args.shape_prior_chunk_wait_timeout_s
        ),
        "mask_radius_outlier_filter": bool(args.mask_radius_outlier_filter),
        "mask_radius_outlier_radius_m": float(args.mask_radius_outlier_radius_m),
        "mask_radius_outlier_nb_points": int(args.mask_radius_outlier_nb_points),
        "source_headless_capture": (
            None
            if args.source_headless_capture is None
            else str(args.source_headless_capture)
        ),
        "downstream_mode": resolve_downstream_mode(args),
        "visualizer_layout": resolve_visualizer_layout(args),
        "visualizer_command": build_visualizer_command(args),
        "visualizer_cuda_visible_devices": resolve_visualizer_cuda_visible_devices(
            args
        ),
        "visualizer_start_policy": visualizer_start_policy(args),
        "visualizer_capture_dir": None,
        "visualizer_fps": float(args.visualizer_playback_fps),
        "visualizer_object_color_mode": str(args.visualizer_object_color_mode),
        "phystwin_shen_repo_path": str(args.phystwin_shen_repo),
        "phystwin_shen_conda_env": str(args.phystwin_shen_conda_env),
        "phystwin_shen_cuda_visible_devices": (
            resolve_phystwin_shen_cuda_visible_devices(args)
        ),
        "phystwin_shen_viewer_url": (
            f"http://{args.phystwin_shen_viewer_host}:"
            f"{int(args.phystwin_shen_viewer_port)}/"
        ),
    }


def validate_runtime_args(args: argparse.Namespace, *, chunk_frame_count: int) -> None:
    """Validate cross-option constraints before launching subprocesses."""
    if float(args.chunk_poll_interval_s) <= 0.0:
        raise ValueError("--chunk-poll-interval-s must be positive")
    if not np.isfinite(float(args.visualizer_playback_fps)):
        raise ValueError("--visualizer-playback-fps must be finite")
    if float(args.visualizer_playback_fps) <= 0.0:
        raise ValueError("--visualizer-playback-fps must be positive")
    resolve_camera_source_replay_fps(args)
    if (
        bool(getattr(args, "fake_live_case_cli_override", False))
        and str(args.input_source) != "fake-live"
    ):
        raise ValueError("--fake-live-case requires --input-source fake-live")
    if int(chunk_frame_count) <= 0:
        raise ValueError("chunk frame count must be positive")
    if not np.isfinite(float(args.edgetam_mask_logit_threshold)):
        raise ValueError("--edgetam-mask-logit-threshold must be finite")
    resolve_main_data_processing_cuda_visible_devices(args)
    if bool(args.shape_prior_warmup):
        resolve_shape_prior_warmup_cuda_visible_devices(args)
    resolve_downstream_mode(args)
    if phystwin_shen_enabled(args):
        # Fail fast before launching subprocesses: a bad checkout/port/GPU
        # config should not surface only at shape-prior-ready time.
        validate_phystwin_shen_repo(args.phystwin_shen_repo)
        resolve_phystwin_shen_cuda_visible_devices(args)
        if not str(args.phystwin_shen_conda_env).strip():
            raise ValueError("--phystwin-shen-conda-env must be non-empty")
        if not (0 < int(args.phystwin_shen_viewer_port) < 65536):
            raise ValueError("--phystwin-shen-viewer-port must be 1..65535")
    if demo_visualizer_enabled(args):
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


# ---------------------------------------------------------------------------
# Camera subprocess command and process lifecycle
# ---------------------------------------------------------------------------


def build_main_data_processing_command(
    args: argparse.Namespace,
    *,
    capture_dir: Path,
    profile_json: Path,
    chunk_frame_count: int,
) -> list[str]:
    """Build the subprocess command that emits prepared realtime frames."""
    script = Path("demo_v6_1") / "main_data_processing.py"
    camera_source_replay_fps = resolve_camera_source_replay_fps(args)
    if str(args.depth_backend) == "ir-ffs":
        depth_source = "ffs"
    elif str(args.depth_backend) == "native-realsense":
        depth_source = "realsense"
    else:
        raise ValueError(f"unsupported depth backend: {args.depth_backend!r}")
    # Demo v6.1 chunks are bounded by the chunk publisher, not by the camera
    # subprocess (chunk_frame_count stays in the signature for that contract).
    # Keeping camera duration unbounded (0.0 = run until stopped) prevents
    # shape-prior warmup time from consuming the realtime RGB input timeline.
    duration_s = 0.0
    # This is the only v6.1 camera/tracker entrypoint. It writes prepared
    # per-frame NPZ payloads plus optional input RGB timeline data; chunk
    # materialization happens in chunk_data_stream.py.
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
        "--edgetam-mask-logit-threshold",
        str(float(args.edgetam_mask_logit_threshold)),
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
        "demo_v6_1_main_data_processing",
        "--metadata-demo-version",
        "demo_v6_1",
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
    if str(args.input_source) == "fake-live" and args.fake_live_case is not None:
        command.extend(["--fake-live-case", str(args.fake_live_case)])
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
                "--shape-prior-case-root",
                str(resolve_shape_prior_case_root(args)),
                "--shape-prior-points-npz",
                str(resolve_shape_prior_points_npz(args)),
            ]
        )
        if bool(args.shape_prior_prewarm_stage_workers):
            command.append("--shape-prior-prewarm-stage-workers")
        else:
            command.append("--no-shape-prior-prewarm-stage-workers")
        if args.shape_prior_sam3d_root is not None:
            command.extend(
                ["--shape-prior-sam3d-root", str(args.shape_prior_sam3d_root)]
            )
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
    """Return the default capture dir."""
    if args.camera_capture_dir is not None:
        return Path(args.camera_capture_dir)
    return base_path / CAPTURE_DIR_NAME


def _stop_process(process: subprocess.Popen[bytes]) -> int | None:
    """Stop a child, escalating SIGTERM -> SIGKILL with a 10 s grace each.

    Children are launched with ``start_new_session=True``, so signalling the
    whole process group also reaps grandchildren (conda run wrappers, CUDA
    workers). When the group signal fails we fall back to plain
    terminate/kill on the direct child.
    """
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


def _start_visualizer(
    args: argparse.Namespace,
    *,
    capture_dir: Path | None = None,
) -> subprocess.Popen[bytes]:
    """Launch the lightweight online visualizer in the repo environment."""
    command = build_visualizer_command(args, capture_dir=capture_dir)
    # The viewer gets its own CUDA namespace so it never competes with the
    # capture/tracker GPUs.
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = resolve_visualizer_cuda_visible_devices(args)
    return subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=env,
        start_new_session=True,
    )


# ---------------------------------------------------------------------------
# Run summary and entrypoint
# ---------------------------------------------------------------------------


def _runtime_chunk_summary(manifests: Sequence[dict[str, object]]) -> dict[str, object]:
    """Aggregate per-chunk manifests into run-level publish/quality stats."""
    # publish_wall_s values are wall-clock seconds; consecutive differences
    # measure the steady-state publish cadence downstream consumers observed.
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
    # The worst chunk status becomes the run-level status. Statuses missing
    # from the table rank lowest (-1) so a stray label never outranks a real
    # degraded/invalid signal.
    quality_order = {"normal": 0, "degraded": 1, "invalid": 2}
    quality_values = [
        str(item.get("track_process_status", "normal")) for item in manifests
    ]
    track_process_status = "normal"
    if quality_values:
        track_process_status = max(
            quality_values, key=lambda value: quality_order.get(value, -1)
        )
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
            sum(
                1
                for item in manifests
                if bool(item.get("online_publish_skipped", False))
            )
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run Demo v6.1 offline conversion or live/fake-live orchestration."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    chunk_frame_count = resolve_chunk_frame_count(args)
    validate_runtime_args(args, chunk_frame_count=chunk_frame_count)

    if bool(args.dry_run):
        print(json.dumps(_contract(args), indent=2, sort_keys=True))
        return 0

    base_path = Path(args.base_path)
    base_path.mkdir(parents=True, exist_ok=True)
    startup_output_cleanup = prepare_realtime_output_for_new_run(
        base_path,
        clear_capture=args.source_headless_capture is None,
        legacy_case_prefix=str(args.case_prefix),
    )
    if args.source_headless_capture is not None:
        # Offline conversion path: consume an existing capture directory and
        # write online/static final_data products without launching camera or
        # visualizer subprocesses.
        manifests = write_chunk_data_from_headless_capture(
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
            asap_augment=bool(args.asap_augment),
            asap_mesh_path=args.asap_mesh_path,
        )
        summary = {
            "demo_version": "demo_v6_1",
            "mode": "source-headless-capture",
            "source_headless_capture": str(args.source_headless_capture),
            "base_path": str(base_path),
            "case_prefix": str(args.case_prefix),
            "output_format": "online-primary-static-case",
            "online_dir": str(resolve_online_dir(args)),
            "static_data_path": str(resolve_static_data_path(args)),
            "shape_prior_case_root": str(resolve_shape_prior_case_root(args)),
            "shape_prior_points_npz": str(resolve_shape_prior_points_npz(args)),
            "startup_output_cleanup": startup_output_cleanup,
            "chunk_frame_count": int(chunk_frame_count),
            "max_chunks": args.max_chunks,
            "chunk_count": int(len(manifests)),
            "chunks": manifests,
        }
        summary.update(_runtime_chunk_summary(manifests))
        summary_path = resolve_run_summary_path(base_path)
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

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
        # A caller-provided checkpoint env var wins. Otherwise anchor the
        # configured (possibly relative) YAML path to the repo root so launches
        # from other working directories still find the vendored checkpoint.
        checkpoint_path = Path(DEFAULT_SAM31_CHECKPOINT_PATH).expanduser()
        if not checkpoint_path.is_absolute():
            checkpoint_path = REPO_ROOT / checkpoint_path
        main_data_processing_env[SAM31_CHECKPOINT_ENV] = str(checkpoint_path)
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
    phystwin_launch: PhystwinShenLaunch | None = None
    shape_prior_points_npz = resolve_shape_prior_points_npz(args)

    def _maybe_start_phystwin_shen() -> None:
        """Launch Phystwin_shen exactly once when the shape prior is ready.

        The warmup completion artifact (points.npz) doubles as the "GPU 1 is
        free" signal: the SAM3D stage subprocesses have exited by the time it
        is written. train_online_warp.py then keeps waiting for the first
        committed chunk on its own. Without warmup there is nothing to wait
        for, so the launch happens on the first poll.
        """
        nonlocal phystwin_launch
        if phystwin_launch is not None or not phystwin_shen_enabled(args):
            return
        if bool(args.shape_prior_warmup):
            if not shape_prior_points_npz.is_file():
                return
            trigger = "shape_prior_points_ready"
        else:
            trigger = "warmup_disabled_immediate"
        phystwin_launch = launch_phystwin_shen(
            resolve_phystwin_shen_settings(args),
            python_prefix=_python_command_prefix(args.phystwin_shen_conda_env),
            log_dir=base_path / "phystwin_shen",
            trigger=trigger,
            wall_time_origin_s=0.0,
        )
        print(
            "[demo_v6_1] phystwin_shen started "
            f"({trigger}); viewer: {phystwin_launch.settings.viewer_url}"
        )

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
            demo_visualizer_enabled(args)
            and not visualizer_uses_side_by_side(args)
            and visualizer_process is None
        ):
            visualizer_process = _start_visualizer(args)
            visualizer_started = True
            visualizer_started_manifest = dict(manifest)
            visualizer_start_wall_s = time.monotonic()
        # Safety net: a chunk can only commit after the shape prior is ready.
        _maybe_start_phystwin_shen()

    main_data_processing = subprocess.Popen(
        main_data_processing_command,
        env=main_data_processing_env,
        start_new_session=True,
    )
    if demo_visualizer_enabled(args) and visualizer_uses_side_by_side(args):
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
        # data_process_sample.py:L250-L352. stream_chunk_data_from_headless_capture
        # materializes those outputs incrementally instead of after the
        # recording has finished.
        manifests = stream_chunk_data_from_headless_capture(
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
            before_poll=_maybe_start_phystwin_shen,
            asap_augment=bool(args.asap_augment),
            asap_mesh_path=args.asap_mesh_path,
        )
    finally:
        main_data_processing_return_code = _stop_process(main_data_processing)
        if visualizer_process is not None:
            visualizer_return_code = visualizer_process.poll()
            visualizer_left_running = visualizer_return_code is None
        # Phystwin_shen keeps serving/training after the demo run ends (same
        # policy as a viewer window); only its exit status is recorded.
    runtime_summary = _runtime_chunk_summary(manifests)
    if args.max_chunks is not None and len(manifests) >= int(args.max_chunks):
        stop_reason = "max_chunks_reached"
    elif main_data_processing_return_code == 0:
        stop_reason = "main_data_processing_completed"
    elif main_data_processing_return_code is None:
        stop_reason = "main_data_processing_status_unknown"
    else:
        stop_reason = "main_data_processing_exited_before_target"
    summary = {
        "demo_version": "demo_v6_1",
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
            None
            if args.camera_source_replay_fps is None
            else float(args.camera_source_replay_fps)
        ),
        "shape_prior_warmup_cuda_visible_devices": (
            resolve_shape_prior_warmup_cuda_visible_devices(args)
        ),
        "shape_prior_controller_name": str(args.shape_prior_controller_name),
        "shape_prior_sam3d_root": (
            None
            if args.shape_prior_sam3d_root is None
            else str(args.shape_prior_sam3d_root)
        ),
        "shape_prior_config": (
            None if args.shape_prior_config is None else str(args.shape_prior_config)
        ),
        "main_data_processing_return_code": main_data_processing_return_code,
        "main_data_processing_stop_reason": stop_reason,
        "main_data_processing_capture_dir": str(capture_dir),
        "base_path": str(base_path),
        "case_prefix": str(args.case_prefix),
        "output_format": "online-primary-static-case",
        "online_dir": str(resolve_online_dir(args)),
        "static_data_path": str(resolve_static_data_path(args)),
        "shape_prior_case_root": str(resolve_shape_prior_case_root(args)),
        "shape_prior_points_npz": str(resolve_shape_prior_points_npz(args)),
        "startup_output_cleanup": startup_output_cleanup,
        "chunk_frame_count": int(chunk_frame_count),
        "chunk_poll_interval_s": float(args.chunk_poll_interval_s),
        "max_chunks": args.max_chunks,
        "chunk_count": int(len(manifests)),
        "chunks": manifests,
        "external_shape_prior_points": bool(
            surface_points is not None or interior_points is not None
        ),
        "downstream_mode": resolve_downstream_mode(args),
        "visualizer_layout": resolve_visualizer_layout(args),
        "visualizer_started": visualizer_started,
        "visualizer_start_policy": visualizer_start_policy(args),
        "visualizer_capture_dir": str(capture_dir)
        if visualizer_uses_side_by_side(args)
        else None,
        "visualizer_started_from_chunk": visualizer_started_manifest,
        "visualizer_start_wall_s": visualizer_start_wall_s,
        "visualizer_command": build_visualizer_command(
            args,
            capture_dir=capture_dir if visualizer_uses_side_by_side(args) else None,
        ),
        "visualizer_cuda_visible_devices": resolve_visualizer_cuda_visible_devices(
            args
        ),
        "visualizer_fps": float(args.visualizer_playback_fps),
        "visualizer_object_color_mode": str(args.visualizer_object_color_mode),
        "visualizer_return_code": visualizer_return_code,
        "visualizer_left_running": visualizer_left_running,
        "phystwin_shen_started": phystwin_launch is not None,
    }
    if phystwin_launch is not None:
        summary.update(phystwin_launch.summary())
    summary.update(runtime_summary)
    summary_path = resolve_run_summary_path(base_path)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    # Exit policy: a camera failure only fails the run when nothing was
    # published (chunks already committed remain valid products). Beyond that,
    # enforce the requested chunk target and downstream health; a downstream
    # process left running (return code None) is normal.
    if main_data_processing_return_code not in (0, None) and not manifests:
        return int(main_data_processing_return_code)
    if args.max_chunks is not None and len(manifests) < int(args.max_chunks):
        return 1
    if demo_visualizer_enabled(args) and not visualizer_started:
        return 1
    if visualizer_return_code not in (0, None):
        return int(visualizer_return_code)
    if phystwin_shen_enabled(args):
        if phystwin_launch is None:
            return 1
        phystwin_summary = phystwin_launch.summary()
        for key in (
            "phystwin_shen_viewer_return_code",
            "phystwin_shen_train_return_code",
        ):
            code = phystwin_summary[key]
            if code not in (0, None):
                return int(code)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
