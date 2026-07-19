"""Demo v6.2 subprocess commands and process lifecycle."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import subprocess
import time

from demo_v6_2.orchestration.main_config import (
    CAPTURE_DIR_NAME,
    DEFAULT_CAMERA_SOURCE_REPLAY_FPS,
    DEFAULT_TABLE_CALIBRATE_PATH,
    REPO_ROOT,
    VISUALIZER_LAYOUT_SIDE_BY_SIDE,
)
from demo_v6_2.orchestration.main_layout import (
    resolve_online_dir,
    resolve_shape_prior_case_root,
    resolve_shape_prior_points_npz,
    resolve_static_data_dir,
)
from demo_v6_2.main_options import (
    python_command_prefix,
    resolve_camera_serials,
    resolve_camera_source_replay_fps,
    resolve_shape_prior_warmup_cuda_visible_devices,
    resolve_visualizer_cuda_visible_devices,
    resolve_visualizer_layout,
    resolve_write_input_rgb_timeline,
)


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
        *python_command_prefix(getattr(args, "visualizer_conda_env", None)),
        str(Path("demo_v6_2") / "visualization" / "visualize_track.py"),
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


# ---------------------------------------------------------------------------
# Camera subprocess command and process lifecycle
# ---------------------------------------------------------------------------


def build_main_data_processing_command(
    args: argparse.Namespace,
    *,
    capture_dir: Path,
    profile_json: Path,
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
    # Demo v6.2 chunks are bounded by the chunk publisher, not by the camera
    # subprocess; the camera runs until stopped, so shape-prior warmup time
    # never consumes the realtime RGB input timeline.
    # This is the only v6.1 camera/tracker entrypoint. It writes prepared
    # per-frame NPZ payloads plus optional input RGB timeline data; chunk
    # materialization happens in streaming/session.py.
    # Offline parity with data_process_origin/data_process_pcd.py:L84-L149,
    # data_process_origin/data_process_mask.py:L42-L152, and
    # data_process_origin/data_process_track.py:L49-L55. The subprocess emits the
    # realtime equivalents of those PCD, mask, and cotracker inputs.
    command = [
        "python",
        str(script),
        "--fps",
        str(int(args.camera_fps)),
        "--color-exposure",
        str(float(args.camera_color_exposure)),
        # resolve_camera_serials enforces the single-camera invariant; the
        # camera subprocess receives the one resolved serial.
        "--serial",
        resolve_camera_serials(args)[0],
        # Warm-up live RGB preview runs in the camera process (it owns the
        # frames and the warm-up lifecycle); forwarded for every downstream
        # mode.
        (
            "--warmup-rgb-preview"
            if bool(args.warmup_rgb_preview)
            else "--no-warmup-rgb-preview"
        ),
        "--color-gain",
        str(float(args.camera_color_gain)),
        "--input-source",
        str(args.input_source),
        "--depth-source",
        depth_source,
        "--depth-backend-label",
        str(args.depth_backend),
        "--headless-capture-dir",
        str(capture_dir),
        "--track-mode",
        "controller-object",
        "--pcd-mode",
        "masked",
        "--tracker-backend",
        "tapnextpp",
        "--tracker-overlay-max-points",
        "0",
        "--replay-fps",
        str(camera_source_replay_fps),
        "--device",
        str(args.perception_device),
        "--dtype",
        str(args.inference_dtype),
        "--shape-prior-object-prompt",
        str(args.shape_prior_object_prompt),
        "--edgetam-mask-logit-threshold",
        str(float(args.edgetam_mask_logit_threshold)),
        "--tracker-device",
        str(args.tracker_device),
        "--table-calibrate",
        str(DEFAULT_TABLE_CALIBRATE_PATH),
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
    command.extend(
        ["--volume-sample-size-m", str(float(args.volume_sample_size_m))]
    )
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
                "--shape-prior-cache-root",
                str(args.shape_prior_cache_root),
                "--shape-prior-case-root",
                str(resolve_shape_prior_case_root(args)),
                "--shape-prior-points-npz",
                str(resolve_shape_prior_points_npz(args)),
            ]
        )
        # Absence of --shape-prior-object (YAML null) disables the cache; a
        # present value is the cache identity. Never forwarded as "none".
        if args.shape_prior_object is not None:
            command.extend(["--shape-prior-object", str(args.shape_prior_object)])
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


def default_capture_dir(args: argparse.Namespace, base_path: Path) -> Path:
    """Return the default capture dir."""
    if args.camera_capture_dir is not None:
        return Path(args.camera_capture_dir)
    return base_path / CAPTURE_DIR_NAME


def _process_group_alive(process_group_id: int) -> bool:
    """Return whether a saved process group still has any live members."""
    try:
        os.killpg(int(process_group_id), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_process_group_exit(
    process_group_id: int,
    timeout_s: float,
    *,
    process: subprocess.Popen[bytes] | None = None,
) -> bool:
    """Wait until a saved process group has no members.

    Polling the direct child while waiting is essential: an exited but
    unreaped supervisor remains a zombie in its old process group, which would
    otherwise make every normal cleanup wait through both timeout windows.
    """
    deadline = time.monotonic() + float(timeout_s)
    while True:
        if process is not None:
            process.poll()
        if not _process_group_alive(process_group_id):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


def stop_process(
    process: subprocess.Popen[bytes],
    *,
    process_group_id: int | None = None,
) -> int | None:
    """Stop a saved child process group, escalating SIGTERM to SIGKILL.

    Every caller launches its child with ``start_new_session=True``, so the
    direct child's PID is also the process-group ID. The explicit group ID is
    retained because a supervisor can exit before one of its viewer/training
    descendants; cleanup must still kill that surviving group.
    """
    group_id = int(
        process_group_id if process_group_id is not None else getattr(process, "pid")
    )
    group_alive = _process_group_alive(group_id)
    if group_alive:
        try:
            os.killpg(group_id, signal.SIGTERM)
        except ProcessLookupError:
            group_alive = False
        except PermissionError:
            if process.poll() is None:
                process.terminate()
        if group_alive and not _wait_for_process_group_exit(
            group_id,
            10.0,
            process=process,
        ):
            try:
                os.killpg(group_id, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except PermissionError:
                if process.poll() is None:
                    process.kill()
            if not _wait_for_process_group_exit(
                group_id,
                10.0,
                process=process,
            ):
                raise RuntimeError(
                    f"process group is still alive after SIGKILL: pgid={group_id}"
                )
    elif process.poll() is None:
        process.terminate()
    if process.poll() is None:
        try:
            return process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            process.kill()
            try:
                return process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                return process.poll()
    return process.returncode


def start_visualizer(
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
