"""Demo v6.1 realtime orchestration option-resolution helpers."""

from __future__ import annotations

import argparse
import copy
import math
import os
from pathlib import Path
import sys

import numpy as np

from demo_v6_2.orchestration.main_config import (
    DEFAULT_CAMERA_SERIALS,
    DEFAULT_DOWNSTREAM_MODE,
    DEFAULT_PHYSTWIN_SHEN_CUDA_VISIBLE_DEVICES,
    DEFAULT_PHYSTWIN_SHEN_RUNTIME_CONFIG,
    DEFAULT_VISUALIZER_CUDA_VISIBLE_DEVICES,
    DEFAULT_VISUALIZER_LAYOUT,
    DOWNSTREAM_MODE_DEMO_VISUALIZER,
    DOWNSTREAM_MODE_PHYSTWIN_SHEN,
    DOWNSTREAM_MODES,
    VISUALIZER_LAYOUT_SIDE_BY_SIDE,
    VISUALIZER_LAYOUTS,
)
from demo_v6_2.phystwin_shen_launch import PhystwinShenSettings


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


def resolve_camera_serials(args: argparse.Namespace) -> list[str]:
    """Resolve the configured camera serial list and enforce single-camera use.

    The config schema (camera.camera_serials) and the repeatable
    ``--camera-serial`` flag are lists so a future multi-camera runtime can
    extend them, but this runtime drives exactly one RealSense: any other
    count fails fast here, before subprocesses launch.
    """
    cli_serials = args.camera_serials
    serials = [
        str(item).strip()
        for item in (DEFAULT_CAMERA_SERIALS if cli_serials is None else cli_serials)
    ]
    if len(serials) != 1 or not serials[0]:
        raise ValueError(
            f"single-camera runtime requires exactly one serial; got {serials!r}"
        )
    return serials


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
        pipeline_config=Path(args.phystwin_shen_pipeline_config).expanduser(),
        conda_env=str(args.phystwin_shen_conda_env),
        base_path=Path(args.base_path).expanduser().resolve(),
        cuda_visible_devices=resolve_phystwin_shen_cuda_visible_devices(args),
        runtime_config=copy.deepcopy(DEFAULT_PHYSTWIN_SHEN_RUNTIME_CONFIG),
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
