"""2D panel composition, overlays, marker colors, and chunk-frame rendering.

Extracted from ``visualization/visualize_track.py`` as part of a behavior-preserving
file split. Depends on ``viz_camera_model`` and ``viz_input_timeline``.
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from demo_v7.runtime.pipeline_status import STAGE_FATAL, STAGE_LABELS
from demo_v7.runtime.visualization.viz_camera_model import (
    CameraModel,
    _require_cv2,
    project_world_points_to_pixels,
)
from demo_v7.runtime.visualization.viz_input_timeline import (
    InputRgbFrame,
    _chunk_frame_count,
    _source_frame_for_chunk_frame,
    format_input_display_latency,
)


DEFAULT_RIGHT_BLANK_LABEL = "waiting for first final_data chunk"


# --- Panel composition and overlay drawing -----------------------------------


def _blank_image(image_size: tuple[int, int]) -> np.ndarray:
    """Return the blank image."""
    width, height = int(image_size[0]), int(image_size[1])
    return np.zeros((height, width, 3), dtype=np.uint8)


def _panel_image(image: np.ndarray | None, *, image_size: tuple[int, int]) -> np.ndarray:
    """Return the panel image."""
    cv2 = _require_cv2()
    width, height = int(image_size[0]), int(image_size[1])
    if image is None:
        return _blank_image((width, height))
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return _blank_image((width, height))
    bgr = np.ascontiguousarray(arr[:, :, :3], dtype=np.uint8)
    if bgr.shape[1] == width and bgr.shape[0] == height:
        return bgr.copy()
    return cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)


def _draw_panel_label(image: np.ndarray, text: str, *, right: bool = False) -> None:
    """Draw panel label."""
    if image.shape[0] < 40 or image.shape[1] < 160:
        return
    cv2 = _require_cv2()
    origin = (12, 28)
    if right:
        (text_width, _text_height), _baseline = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            1,
        )
        origin = (max(12, int(image.shape[1]) - int(text_width) - 12), 28)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_camera_to_final_data_fps_overlay(image: np.ndarray, fps: float | None) -> None:
    """Draw camera to final data FPS overlay."""
    if image.shape[0] < 40 or image.shape[1] < 280:
        return
    cv2 = _require_cv2()
    if fps is None or not math.isfinite(float(fps)):
        text = "camera->final_data -- FPS"
    else:
        text = f"camera->final_data {float(fps):.1f} FPS"
    (text_width, _text_height), _baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        1,
    )
    origin = (max(12, int(image.shape[1]) - int(text_width) - 12), 28)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_fake_rgb_frame_counter_overlay(
    image: np.ndarray,
    input_frame: InputRgbFrame | None,
    *,
    fake_input_frame_total: int | None,
) -> None:
    """Draw fake RGB frame counter overlay."""
    if image.shape[0] < 70 or image.shape[1] < 180:
        return
    # The counter only applies to fake-live replays, where the recording's
    # total frame count is known ahead of time (1-based for display).
    if input_frame is None or input_frame.source_frame_index is None or fake_input_frame_total is None:
        return
    try:
        current = int(input_frame.source_frame_index) + 1
        total = int(fake_input_frame_total)
    except (TypeError, ValueError):
        return
    if current <= 0 or total <= 0:
        return
    text = f"RGB frame {current}/{total}"
    cv2 = _require_cv2()
    origin = (12, 54)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_center_label(image: np.ndarray, text: str) -> None:
    """Draw center label."""
    if image.shape[0] < 60 or image.shape[1] < 160:
        return
    cv2 = _require_cv2()
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
    origin = (
        max(12, (int(image.shape[1]) - int(text_width)) // 2),
        max(32, (int(image.shape[0]) + int(text_height)) // 2),
    )
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1, cv2.LINE_AA)


def render_side_by_side_frame(
    *,
    input_frame: InputRgbFrame | None,
    output_frame: np.ndarray | None,
    image_size: tuple[int, int],
    right_blank_label: str = DEFAULT_RIGHT_BLANK_LABEL,
    camera_to_final_data_fps: float | None = None,
    fake_input_frame_total: int | None = None,
    input_to_display_latency_s: float | None = None,
    show_latency_overlay: bool = True,
) -> np.ndarray:
    """Compose one RGB-input/final_data-output frame for display or video."""
    left = _panel_image(None if input_frame is None else input_frame.image_bgr, image_size=image_size)
    right = _panel_image(output_frame, image_size=image_size)
    if input_frame is None:
        _draw_center_label(left, "waiting for RGB input")
    if output_frame is None:
        _draw_center_label(right, str(right_blank_label))
    _draw_panel_label(left, "RGB input")
    _draw_fake_rgb_frame_counter_overlay(left, input_frame, fake_input_frame_total=fake_input_frame_total)
    _draw_camera_to_final_data_fps_overlay(left, camera_to_final_data_fps)
    _draw_panel_label(right, "final_data output", right=True)
    if show_latency_overlay:
        # Latency HUD: right-aligned, one 26px line below the panel label.
        text = format_input_display_latency(input_to_display_latency_s)
        if right.shape[0] >= 40 and right.shape[1] >= 220:
            cv2 = _require_cv2()
            (text_width, _text_height), _baseline = cv2.getTextSize(
                text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                1,
            )
            origin = (max(12, int(right.shape[1]) - int(text_width) - 12), 54)
            cv2.putText(right, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(right, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    return np.ascontiguousarray(np.concatenate([left, right], axis=1), dtype=np.uint8)


def read_background(
    case_dir: Path,
    *,
    cam_idx: int,
    source_frame: int,
    image_size: tuple[int, int],
    use_background: bool,
    frame_path: Path | None = None,
) -> np.ndarray:
    """Return the source RGB frame when present, otherwise a black canvas."""
    if not use_background:
        return _blank_image(image_size)
    cv2 = _require_cv2()
    if frame_path is not None and frame_path.is_file():
        image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if image is not None:
            return image
    # Probe the aligned-case color/<cam>/ layout before the fake-live
    # input_rgb/ layout, trying plain and zero-padded names, png before jpg.
    color_dir = case_dir / "color" / str(int(cam_idx))
    input_rgb_dir = case_dir / "input_rgb"
    for path in (
        color_dir / f"{int(source_frame)}.png",
        color_dir / f"{int(source_frame):06d}.png",
        color_dir / f"{int(source_frame)}.jpg",
        color_dir / f"{int(source_frame):06d}.jpg",
        input_rgb_dir / f"{int(source_frame)}.png",
        input_rgb_dir / f"{int(source_frame):06d}.png",
        input_rgb_dir / f"{int(source_frame)}.jpg",
        input_rgb_dir / f"{int(source_frame):06d}.jpg",
    ):
        if not path.is_file():
            continue
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is not None:
            return image
    return _blank_image(image_size)


# --- Marker colors -----------------------------------------------------------


def parse_bgr_color(text: str) -> tuple[int, int, int]:
    """Parse a comma-separated B,G,R color triplet."""
    parts = [part.strip() for part in str(text).split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("color must be B,G,R")
    try:
        values = [int(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("color must be B,G,R integers") from exc
    return tuple(max(0, min(255, value)) for value in values)


def _sam3d_rainbow_colors_bgr(chunk: Mapping[str, Any], point_indices: np.ndarray) -> np.ndarray:
    # Colors are keyed to each point's Y height in the chunk's FIRST frame so
    # a point keeps one stable color for the whole chunk. Falls back to a
    # piecewise-linear rainbow when matplotlib is unavailable.
    """Return the SAM3D rainbow colors BGR."""
    indices = np.asarray(point_indices, dtype=np.int64).reshape(-1)
    if indices.size == 0:
        return np.empty((0, 3), dtype=np.uint8)
    object_points = chunk.get("object_points")
    if object_points is None:
        normalized = np.zeros((indices.shape[0],), dtype=np.float32)
    else:
        arr = np.asarray(object_points, dtype=np.float64)
        if arr.ndim != 3 or arr.shape[0] == 0 or arr.shape[2] < 2:
            normalized = np.zeros((indices.shape[0],), dtype=np.float32)
        else:
            y_values = np.asarray(arr[0, :, 1], dtype=np.float64).reshape(-1)
            if y_values.size == 0:
                normalized = np.zeros((indices.shape[0],), dtype=np.float32)
            else:
                finite = np.isfinite(y_values)
                if np.any(finite):
                    y_min = float(np.nanmin(y_values[finite]))
                    y_max = float(np.nanmax(y_values[finite]))
                    span = y_max - y_min
                    if math.isfinite(span) and span > 1e-9:
                        selected_y = y_values[np.clip(indices, 0, y_values.shape[0] - 1)]
                        normalized = np.clip((selected_y - y_min) / span, 0.0, 1.0).astype(np.float32)
                    else:
                        normalized = np.zeros((indices.shape[0],), dtype=np.float32)
                else:
                    normalized = np.zeros((indices.shape[0],), dtype=np.float32)
    try:
        import matplotlib.pyplot as plt

        rgb = np.asarray(plt.cm.rainbow(normalized)[:, :3], dtype=np.float32) * 255.0
    except Exception:
        rgb = np.stack(
            [
                255.0 * normalized,
                255.0 * (1.0 - np.abs(normalized - 0.5) * 2.0),
                255.0 * (1.0 - normalized),
            ],
            axis=1,
        )
    return np.ascontiguousarray(np.clip(rgb, 0, 255).astype(np.uint8)[:, ::-1], dtype=np.uint8)


def _sam3d_rainbow_colors_rgb_float(object_points: np.ndarray, point_count: int) -> np.ndarray:
    # Open3D variant of _sam3d_rainbow_colors_bgr: same first-frame Y-height
    # keying, but returns float RGB in [0, 1] for point-cloud colors.
    """Return the SAM3D rainbow colors RGB float."""
    count = max(0, int(point_count))
    if count == 0:
        return np.empty((0, 3), dtype=np.float64)
    arr = np.asarray(object_points, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[0] == 0 or arr.shape[1] < count or arr.shape[2] < 2:
        normalized = np.zeros((count,), dtype=np.float64)
    else:
        y_values = np.asarray(arr[0, :count, 1], dtype=np.float64)
        finite = np.isfinite(y_values)
        if np.any(finite):
            y_min = float(np.nanmin(y_values[finite]))
            y_max = float(np.nanmax(y_values[finite]))
            span = y_max - y_min
            if math.isfinite(span) and span > 1e-9:
                normalized = np.clip((y_values - y_min) / span, 0.0, 1.0)
            else:
                normalized = np.zeros((count,), dtype=np.float64)
        else:
            normalized = np.zeros((count,), dtype=np.float64)
    try:
        import matplotlib.pyplot as plt

        return np.ascontiguousarray(np.asarray(plt.cm.rainbow(normalized)[:, :3], dtype=np.float64))
    except Exception:
        rgb = np.stack(
            [
                normalized,
                1.0 - np.abs(normalized - 0.5) * 2.0,
                1.0 - normalized,
            ],
            axis=1,
        )
        return np.ascontiguousarray(np.clip(rgb, 0.0, 1.0), dtype=np.float64)


def object_point_colors(
    chunk: Mapping[str, Any],
    *,
    local_frame: int,
    point_indices: np.ndarray,
    mode: str,
) -> np.ndarray:
    """Resolve BGR colors for projected object point markers."""
    mode_value = str(mode)
    if mode_value == "green":
        return np.tile(np.array([[0, 255, 0]], dtype=np.uint8), (point_indices.shape[0], 1))
    if mode_value == "object-colors":
        colors = chunk.get("object_colors")
        if colors is not None:
            arr = np.asarray(colors)
            if arr.ndim == 3 and local_frame < arr.shape[0] and arr.shape[2] >= 3:
                selected = np.asarray(arr[int(local_frame), point_indices, :3], dtype=np.float64)
                if selected.size:
                    if float(np.nanmax(selected)) <= 1.0:
                        selected = selected * 255.0
                    selected = np.clip(selected, 0.0, 255.0).astype(np.uint8)
                    return selected[:, ::-1]
    return _sam3d_rainbow_colors_bgr(chunk, point_indices)


def controller_point_colors(
    chunk: Mapping[str, Any],
    *,
    local_frame: int,
    point_indices: np.ndarray,
    fallback_color: tuple[int, int, int],
) -> np.ndarray:
    """Resolve BGR colors for projected controller point markers."""
    color = np.asarray(fallback_color, dtype=np.uint8).reshape(1, 3)
    return np.tile(color, (point_indices.shape[0], 1))


# --- Chunk-frame rendering ---------------------------------------------------


def _draw_sam3d_markers(
    image: np.ndarray,
    pixels: np.ndarray,
    colors: np.ndarray,
    *,
    radius: int,
) -> None:
    """Draw SAM3D markers."""
    if pixels.size == 0:
        return
    cv2 = _require_cv2()
    draw_radius = max(int(radius), 1)
    for (x_value, y_value), color in zip(pixels, colors, strict=False):
        center = (int(x_value), int(y_value))
        color_bgr = tuple(int(value) for value in color)
        cv2.circle(
            image,
            center,
            draw_radius,
            color_bgr,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )


def _draw_status(image: np.ndarray, text: str) -> None:
    """Draw status text on an output image."""
    cv2 = _require_cv2()
    origin = (12, 28)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)


def draw_pipeline_status(
    image: np.ndarray,
    events: list[Mapping[str, Any]],
    *,
    now_s: float | None = None,
) -> np.ndarray:
    """Draw the live pipeline-status band along the bottom of ``image``.

    Answers design question 25: shows what the pipeline is doing right now
    (capturing / shape-prior / warm-up ready / streaming chunks / finished) from
    the ``pipeline_status.jsonl`` events the orchestrator, camera, and shape-prior
    stages emit, and turns the band red when a fatal warm-up/shape-prior error was
    reported. Drawing in place; returns ``image`` for convenience.
    """
    cv2 = _require_cv2()
    if image is None or getattr(image, "size", 0) == 0:
        return image
    height, width = image.shape[:2]
    band_h = 34
    y0 = max(0, height - band_h)
    # The most recent event is the headline; a fatal anywhere flags the failure.
    latest = events[-1] if events else None
    fatal = next(
        (e for e in reversed(events)
         if not e.get("ok", True) or e.get("stage") == STAGE_FATAL),
        None,
    )
    headline = fatal if fatal is not None else latest
    if headline is None:
        bg, stage_key, detail, source, event_t = (
            (60, 50, 40), "", "waiting for pipeline...", "", None,
        )
    else:
        bg = (40, 40, 200) if fatal is not None else (60, 50, 40)  # BGR red vs slate
        stage_key = str(headline.get("stage", ""))
        detail = str(headline.get("detail", ""))
        source = str(headline.get("source", ""))
        event_t = headline.get("t")
    label = STAGE_LABELS.get(stage_key, stage_key or "—")
    ago = ""
    if event_t is not None:
        now = time.time() if now_s is None else float(now_s)
        ago = f"  ({source} {max(0.0, now - float(event_t)):.0f}s ago)"
    text = f"PIPELINE: {label}" + (f" — {detail}" if detail else "") + ago
    overlay = image.copy()
    cv2.rectangle(overlay, (0, y0), (width, height), bg, -1)
    cv2.addWeighted(overlay, 0.65, image, 0.35, 0, image)
    origin = (12, y0 + 23)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def render_chunk_frame(
    chunk: Mapping[str, Any],
    *,
    local_frame: int,
    case_dir: Path,
    camera: CameraModel,
    cam_idx: int,
    use_background: bool,
    show_invisible_object_points: bool,
    object_stride: int,
    object_radius: int,
    controller_radius: int,
    object_color_mode: str,
    controller_color: tuple[int, int, int],
    fps: float,
    background_frame_paths: Mapping[int, Path] | None = None,
) -> np.ndarray:
    """Draw one online chunk frame as colored object/controller pixels."""
    source_frame = _source_frame_for_chunk_frame(chunk, local_frame)
    frame_path = None
    if background_frame_paths is not None:
        # Fake-live chunks keep the original recording source ids, while saved
        # RGB files are named by receive sequence; the timeline bridges them.
        frame_path = background_frame_paths.get(int(source_frame))
    image = read_background(
        case_dir,
        cam_idx=cam_idx,
        source_frame=source_frame,
        image_size=camera.image_size,
        use_background=use_background,
        frame_path=frame_path,
    )
    image_size = (int(image.shape[1]), int(image.shape[0]))
    object_points = chunk.get("object_points")
    if object_points is not None:
        object_arr = np.asarray(object_points)
        if object_arr.ndim == 3 and int(local_frame) < int(object_arr.shape[0]):
            visibility = None
            if not show_invisible_object_points and chunk.get("object_visibilities") is not None:
                vis_arr = np.asarray(chunk["object_visibilities"])
                if vis_arr.ndim == 2 and int(local_frame) < int(vis_arr.shape[0]):
                    visibility = vis_arr[int(local_frame)]
            object_pixels, object_indices = project_world_points_to_pixels(
                object_arr[int(local_frame)],
                intrinsic=camera.intrinsic,
                camera_to_world=camera.camera_to_world,
                image_size=image_size,
                visibility=visibility,
                stride=object_stride,
            )
            _draw_sam3d_markers(
                image,
                object_pixels,
                object_point_colors(
                    chunk,
                    local_frame=int(local_frame),
                    point_indices=object_indices,
                    mode=object_color_mode,
                ),
                radius=object_radius,
            )
    controller_points = chunk.get("controller_points")
    if controller_points is not None:
        controller_arr = np.asarray(controller_points)
        if controller_arr.ndim == 3 and int(local_frame) < int(controller_arr.shape[0]):
            controller_pixels, controller_indices = project_world_points_to_pixels(
                controller_arr[int(local_frame)],
                intrinsic=camera.intrinsic,
                camera_to_world=camera.camera_to_world,
                image_size=image_size,
                stride=1,
            )
            _draw_sam3d_markers(
                image,
                controller_pixels,
                controller_point_colors(
                    chunk,
                    local_frame=int(local_frame),
                    point_indices=controller_indices,
                    fallback_color=controller_color,
                ),
                radius=controller_radius,
            )
    frame_count = _chunk_frame_count(chunk)
    chunk_id = int(chunk.get("chunk_id", -1))
    _draw_status(
        image,
        f"chunk {chunk_id:06d}  frame {int(local_frame) + 1}/{frame_count}  source {source_frame}  {fps:g} FPS",
    )
    return image
