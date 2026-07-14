"""Interactive playback loops and online chunk timeline assembly.

Extracted from ``visualization/visualize_track.py`` as part of a behavior-preserving
file split. Depends on ``viz_camera_model``, ``viz_input_timeline``,
``viz_panels``, and ``viz_renderers``.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from demo_v6_2.visualization.viz_camera_model import (
    CameraModel,
    _require_cv2,
    infer_case_dir,
    load_camera_model,
    load_pickle,
    normalize_online_dir,
    read_json,
)
from demo_v6_2.visualization.viz_input_timeline import (
    CameraToFinalDataFpsMeter,
    OutputStreamPlaybackCursor,
    _chunk_frame_count,
    _resolve_capture_dir,
    _resolve_input_rgb_timeline,
    estimate_historical_camera_to_final_data_fps,
    list_available_chunk_paths,
    load_fake_input_frame_total,
    load_latest_input_rgb_frame,
    source_time_input_display_latency_s,
)
from demo_v6_2.pipeline_status import read_status_events
from demo_v6_2.visualization.viz_panels import (
    _draw_camera_to_final_data_fps_overlay,
    _draw_center_label,
    _draw_fake_rgb_frame_counter_overlay,
    _draw_panel_label,
    _draw_status,
    _panel_image,
    draw_pipeline_status,
    render_side_by_side_frame,
)
from demo_v6_2.visualization.viz_renderers import (
    RENDER_MODE_RGB_OVERLAY,
    RENDER_MODE_SAM3D_FINAL_DATA,
    Sam3DGuiFinalDataRenderer,
    build_frame_renderer,
)


LAYOUT_OUTPUT_ONLY = "output-only"
LAYOUT_SIDE_BY_SIDE = "side-by-side"
LAYOUTS = (LAYOUT_SIDE_BY_SIDE, LAYOUT_OUTPUT_ONLY)


# --- Interactive playback loops ----------------------------------------------


def _window_is_open(window_name: str) -> bool:
    """Return the window is open."""
    cv2 = _require_cv2()
    try:
        return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1
    except Exception:
        return True


def _key_requests_quit(key: int) -> bool:
    """Return the key requests quit."""
    return key in (27, ord("q"), ord("Q"))


def _wait_with_pause(window_name: str, *, delay_s: float) -> bool:
    """Wait one playback period while servicing space-to-pause and quit keys.

    Returns False when the viewer should stop (quit key or closed window).
    Unpausing resets the deadline so playback resumes immediately.
    """
    cv2 = _require_cv2()
    deadline = time.monotonic() + max(float(delay_s), 0.0)
    paused = False
    while True:
        if not _window_is_open(window_name):
            return False
        wait_s = 0.05 if paused else min(0.05, max(0.0, deadline - time.monotonic()))
        key = cv2.waitKey(max(1, int(wait_s * 1000))) & 0xFF
        if _key_requests_quit(key):
            return False
        if key == ord(" "):
            paused = not paused
            if not paused:
                deadline = time.monotonic()
        if paused:
            continue
        if time.monotonic() >= deadline:
            return True


def play_chunk(
    chunk: Mapping[str, Any],
    *,
    case_dir: Path,
    renderer: Any,
    args: argparse.Namespace,
    fps: float,
) -> np.ndarray | None:
    """Play one chunk frame-by-frame in an OpenCV window."""
    cv2 = _require_cv2()
    period_s = 1.0 / max(float(fps), 1e-6)
    frame_count = _chunk_frame_count(chunk)
    last_image = None
    for local_frame in range(frame_count):
        image = renderer.render_frame(
            chunk,
            local_frame=local_frame,
            case_dir=case_dir,
        )
        cv2.imshow(str(args.window_name), image)
        last_image = image
        if not _wait_with_pause(str(args.window_name), delay_s=period_s):
            return None
    return last_image


def use_interactive_side_by_side(args: argparse.Namespace) -> bool:
    """Return whether side-by-side mode should use the Open3D GUI renderer."""
    return (
        str(getattr(args, "layout", LAYOUT_OUTPUT_ONLY)) == LAYOUT_SIDE_BY_SIDE
        and str(getattr(args, "render_mode", RENDER_MODE_RGB_OVERLAY)) == RENDER_MODE_SAM3D_FINAL_DATA
        and getattr(args, "output_video", None) is None
    )


def run_interactive_side_by_side(args: argparse.Namespace) -> int:
    """Run the Open3D output window next to a live OpenCV RGB input window."""
    cv2 = _require_cv2()
    online_dir = normalize_online_dir(args.online_dir)
    case_dir = infer_case_dir(online_dir, args.case_dir)
    camera = load_camera_model(case_dir, cam_idx=int(args.cam_idx))
    fps = resolve_playback_fps(args, camera)
    capture_dir = _resolve_capture_dir(args)
    input_timeline = _resolve_input_rgb_timeline(args, capture_dir=capture_dir)
    fake_input_frame_total = load_fake_input_frame_total(capture_dir)
    width, height = camera.image_size
    left_window_name = f"{args.window_name} - RGB input"
    right_window_name = f"{args.window_name} - final_data output"
    output_renderer = Sam3DGuiFinalDataRenderer(
        image_size=camera.image_size,
        show_invisible_object_points=bool(args.show_invisible_object_points),
        window_name=right_window_name,
        window_position=(int(width) + 80, 50),
        show_latency_overlay=bool(args.latency_overlay),
    )
    output_frames: list[tuple[dict[str, Any], int]] = []
    loaded_paths: set[Path] = set()
    cursor = OutputStreamPlaybackCursor(fps=fps)
    final_data_fps_meter = CameraToFinalDataFpsMeter()
    paused = False

    cv2.namedWindow(left_window_name, cv2.WINDOW_NORMAL)
    try:
        cv2.resizeWindow(left_window_name, int(width), int(height))
        cv2.moveWindow(left_window_name, 30, 50)
    except Exception:
        pass

    try:
        while True:
            appended = _append_new_output_frames(
                online_dir,
                start_chunk=int(args.start_chunk),
                loaded_paths=loaded_paths,
                output_frames=output_frames,
            )
            latest = max(0, len(output_frames) - 1)
            now_s = time.monotonic()
            camera_to_final_data_fps = final_data_fps_meter.update(
                appended_frames=appended,
                now_s=now_s,
            )
            if camera_to_final_data_fps is None:
                camera_to_final_data_fps = final_data_fps_meter.seed(
                    estimate_historical_camera_to_final_data_fps(
                        online_dir,
                        start_chunk=int(args.start_chunk),
                    )
                )

            input_frame = None
            if capture_dir is not None and input_timeline is not None:
                input_frame = load_latest_input_rgb_frame(input_timeline, capture_dir=capture_dir)
            if output_frames and not paused:
                # The left panel follows the latest camera RGB. The right panel
                # plays only committed chunk frames at the configured 5 FPS.
                cursor.advance(latest=latest, now_s=now_s, paused=False)
            else:
                cursor.advance(latest=latest, now_s=now_s, paused=True)
            # Compose the left panel with the same overlays as the left half
            # of render_side_by_side_frame.
            input_panel = _panel_image(
                None if input_frame is None else input_frame.image_bgr,
                image_size=camera.image_size,
            )
            if input_frame is None:
                _draw_center_label(input_panel, "waiting for RGB input")
            _draw_panel_label(input_panel, "RGB input")
            _draw_fake_rgb_frame_counter_overlay(
                input_panel,
                input_frame,
                fake_input_frame_total=fake_input_frame_total,
            )
            _draw_camera_to_final_data_fps_overlay(input_panel, camera_to_final_data_fps)
            cv2.imshow(left_window_name, input_panel)

            if output_frames:
                chunk, local_frame = output_frames[int(cursor.output_index)]
                latency_s = None
                if bool(args.latency_overlay):
                    latency_s = source_time_input_display_latency_s(
                        input_frame=input_frame,
                        output_frames=output_frames,
                        output_index=int(cursor.output_index),
                        fps=fps,
                    )
                if not output_renderer.update_frame(
                    chunk,
                    local_frame=local_frame,
                    case_dir=case_dir,
                    input_to_display_latency_s=latency_s,
                ):
                    return 0
            else:
                if not output_renderer.poll():
                    return 0

            key = cv2.waitKey(max(1, int(float(args.poll_sec) * 1000))) & 0xFF
            if _key_requests_quit(key) or not _window_is_open(left_window_name):
                return 0
            if key == ord(" "):
                paused = not paused
                cursor.last_step_s = time.monotonic()
            elif key in (ord("f"), ord("F")):
                paused = False
                cursor.seek(latest, latest=latest, now_s=time.monotonic())
    finally:
        output_renderer.close()
        try:
            cv2.destroyWindow(left_window_name)
        except Exception:
            pass


def run_side_by_side(args: argparse.Namespace) -> int:
    """Run the single-window side-by-side viewer/video fallback."""
    cv2 = _require_cv2()
    online_dir = normalize_online_dir(args.online_dir)
    case_dir = infer_case_dir(online_dir, args.case_dir)
    camera = load_camera_model(case_dir, cam_idx=int(args.cam_idx))
    fps = resolve_playback_fps(args, camera)
    renderer = build_frame_renderer(args, camera=camera, fps=fps)
    capture_dir = _resolve_capture_dir(args)
    input_timeline = _resolve_input_rgb_timeline(args, capture_dir=capture_dir)
    fake_input_frame_total = load_fake_input_frame_total(capture_dir)

    window_name = str(args.window_name)
    trackbar_name = "output frame"
    output_frames: list[tuple[dict[str, Any], int]] = []
    loaded_paths: set[Path] = set()
    cursor = OutputStreamPlaybackCursor(fps=fps)
    final_data_fps_meter = CameraToFinalDataFpsMeter()
    follow_latest = bool(args.follow_latest)
    paused = False
    # Guard against feedback: syncing the trackbar position programmatically
    # fires on_trackbar, which must not be treated as a user seek.
    trackbar_guard = {"updating": False}

    def on_trackbar(value: int) -> None:
        """Return the on trackbar."""
        nonlocal follow_latest
        if trackbar_guard["updating"]:
            return
        latest = max(0, len(output_frames) - 1)
        cursor.seek(value, latest=latest, now_s=time.monotonic())
        follow_latest = int(cursor.output_index) >= latest

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.createTrackbar(trackbar_name, window_name, 0, 1, on_trackbar)
    try:
        while True:
            appended = _append_new_output_frames(
                online_dir,
                start_chunk=int(args.start_chunk),
                loaded_paths=loaded_paths,
                output_frames=output_frames,
            )
            latest = max(0, len(output_frames) - 1)
            # OpenCV rejects a trackbar max of 0, and some GUI backends do not
            # implement setTrackbarMax at all.
            try:
                cv2.setTrackbarMax(trackbar_name, window_name, max(1, latest))
            except Exception:
                pass
            now_s = time.monotonic()
            camera_to_final_data_fps = final_data_fps_meter.update(
                appended_frames=appended,
                now_s=now_s,
            )
            if camera_to_final_data_fps is None:
                camera_to_final_data_fps = final_data_fps_meter.seed(
                    estimate_historical_camera_to_final_data_fps(
                        online_dir,
                        start_chunk=int(args.start_chunk),
                    )
                )

            input_frame = None
            if capture_dir is not None and input_timeline is not None:
                input_frame = load_latest_input_rgb_frame(input_timeline, capture_dir=capture_dir)
            if output_frames and follow_latest and not paused:
                # The left panel follows the latest camera RGB. The right panel
                # plays only committed chunk frames at the configured 5 FPS.
                cursor.advance(latest=latest, now_s=now_s, paused=False)
            else:
                cursor.seek(cursor.output_index, latest=latest)
            output_frame = _render_output_timeline_frame(
                output_frames,
                output_index=int(cursor.output_index),
                renderer=renderer,
                case_dir=case_dir,
            )
            input_to_display_latency_s = None
            if bool(args.latency_overlay):
                input_to_display_latency_s = source_time_input_display_latency_s(
                    input_frame=input_frame,
                    output_frames=output_frames,
                    output_index=int(cursor.output_index),
                    fps=fps,
                )
            image = render_side_by_side_frame(
                input_frame=input_frame,
                output_frame=output_frame,
                image_size=camera.image_size,
                right_blank_label=str(args.right_blank_label),
                camera_to_final_data_fps=camera_to_final_data_fps,
                fake_input_frame_total=fake_input_frame_total,
                input_to_display_latency_s=input_to_display_latency_s,
                show_latency_overlay=bool(args.latency_overlay),
            )
            # Live pipeline-status band (design question 25): show what the
            # pipeline is doing right now / whether warm-up failed, tailed from
            # <base_path>/pipeline_status.jsonl (online_dir's parent).
            image = draw_pipeline_status(
                image, read_status_events(online_dir.parent, tail=200)
            )
            cv2.imshow(window_name, image)
            trackbar_guard["updating"] = True
            try:
                cv2.setTrackbarPos(trackbar_name, window_name, int(cursor.output_index))
            finally:
                trackbar_guard["updating"] = False
            key = cv2.waitKey(max(1, int(float(args.poll_sec) * 1000))) & 0xFF
            if _key_requests_quit(key) or not _window_is_open(window_name):
                return 0
            if key == ord(" "):
                paused = not paused
                cursor.last_step_s = time.monotonic()
            elif key in (ord("f"), ord("F")):
                follow_latest = True
                paused = False
                cursor.seek(latest, latest=latest, now_s=time.monotonic())
    finally:
        renderer.close()


def wait_for_chunk(
    online_dir: Path,
    *,
    chunk_id: int,
    poll_sec: float,
    window_name: str,
    last_image: np.ndarray | None,
) -> dict[str, Any] | None:
    """Block until the next online chunk appears or the stream finishes."""
    cv2 = _require_cv2()
    chunks_dir = online_dir / "chunks"
    chunk_path = chunks_dir / f"chunk_{int(chunk_id):06d}.pkl"
    while True:
        if chunk_path.is_file():
            return dict(load_pickle(chunk_path))
        manifest = read_json(online_dir / "manifest.json")
        latest = int(manifest.get("latest_committed_chunk", -1))
        status = str(manifest.get("status", "recording"))
        if status == "finished" and latest < int(chunk_id):
            return None
        if last_image is not None:
            waiting = last_image.copy()
            _draw_status(waiting, f"waiting for chunk {int(chunk_id):06d}")
            cv2.imshow(window_name, waiting)
        key = cv2.waitKey(max(1, int(float(poll_sec) * 1000))) & 0xFF
        if _key_requests_quit(key) or not _window_is_open(window_name):
            return None


def resolve_playback_fps(args: argparse.Namespace, camera: CameraModel) -> float:
    """Resolve playback FPS from CLI, metadata, or the default fallback."""
    fps = None if args.fps is None else float(args.fps)
    if fps is None:
        fps = camera.metadata_fps
    if fps is None:
        fps = 5.0
    if not math.isfinite(float(fps)) or float(fps) <= 0.0:
        raise ValueError("--fps must be positive")
    return float(fps)


def _append_new_output_frames(
    online_dir: Path,
    *,
    start_chunk: int,
    loaded_paths: set[Path],
    output_frames: list[tuple[dict[str, Any], int]],
) -> int:
    """Load each new chunk once and flatten it into frame-level playback rows."""
    appended = 0
    for chunk_path in list_available_chunk_paths(online_dir, start_chunk=start_chunk):
        resolved = chunk_path.resolve()
        if resolved in loaded_paths:
            continue
        try:
            chunk = dict(load_pickle(chunk_path))
        except Exception:
            # Leave the path unmarked so a partially written chunk is retried
            # on the next poll.
            continue
        loaded_paths.add(resolved)
        frame_count = _chunk_frame_count(chunk)
        for local_frame in range(frame_count):
            output_frames.append((chunk, int(local_frame)))
            appended += 1
    return appended


def _render_output_timeline_frame(
    output_frames: Sequence[tuple[dict[str, Any], int]],
    *,
    output_index: int,
    renderer: Any,
    case_dir: Path,
) -> np.ndarray | None:
    """Render output timeline frame."""
    if not output_frames:
        return None
    idx = min(max(int(output_index), 0), len(output_frames) - 1)
    chunk, local_frame = output_frames[idx]
    return renderer.render_frame(chunk, local_frame=local_frame, case_dir=case_dir)
