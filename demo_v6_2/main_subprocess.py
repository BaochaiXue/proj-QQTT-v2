"""Demo v6.1 subprocess commands, runtime contract, and process lifecycle."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import subprocess

import numpy as np

from demo_v6_2.main_config import (
    CAPTURE_DIR_NAME,
    DEFAULT_CAMERA_SOURCE_REPLAY_FPS,
    DEFAULT_TABLE_CALIBRATE_PATH,
    EDGE_TAM_TRACKING_IDENTITIES,
    REPO_ROOT,
    VISUALIZER_LAYOUT_SIDE_BY_SIDE,
)
from demo_v6_2.main_layout import (
    resolve_online_dir,
    resolve_shape_prior_case_root,
    resolve_shape_prior_points_npz,
    resolve_static_data_dir,
    resolve_static_data_path,
)
from demo_v6_2.main_options import (
    _python_command_prefix,
    demo_visualizer_enabled,
    phystwin_shen_enabled,
    resolve_camera_source_replay_fps,
    resolve_chunk_frame_count,
    resolve_downstream_mode,
    resolve_main_data_processing_cuda_visible_devices,
    resolve_phystwin_shen_cuda_visible_devices,
    resolve_shape_prior_warmup_cuda_visible_devices,
    resolve_visualizer_cuda_visible_devices,
    resolve_visualizer_layout,
    resolve_write_input_rgb_timeline,
    visualizer_start_policy,
)
from demo_v6_2.phystwin_shen_launch import validate_phystwin_shen_repo


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
        str(Path("demo_v6_2") / "visualize_track.py"),
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
    script = Path("demo_v6_2") / "main_data_processing.py"
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
