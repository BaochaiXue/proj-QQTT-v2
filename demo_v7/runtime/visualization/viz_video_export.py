"""Offline MP4 export for output-only and side-by-side layouts.

Extracted from ``visualization/visualize_track.py`` as part of a behavior-preserving
file split. Depends on the other ``viz_*`` modules.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from demo_v7.runtime.utils.render import open_video_writer
from demo_v7.runtime.visualization.viz_camera_model import (
    _require_cv2,
    infer_case_dir,
    load_camera_model,
    load_pickle,
    normalize_online_dir,
)
from demo_v7.runtime.visualization.viz_input_timeline import (
    _chunk_frame_count,
    list_available_chunk_paths,
    load_input_rgb_frames,
)
from demo_v7.runtime.visualization.viz_panels import render_side_by_side_frame
from demo_v7.runtime.visualization.viz_playback import (
    LAYOUT_OUTPUT_ONLY,
    LAYOUT_SIDE_BY_SIDE,
    _append_new_output_frames,
    _playback_context,
    _render_output_timeline_frame,
    resolve_playback_fps,
)
from demo_v7.runtime.visualization.viz_renderers import build_frame_renderer


def export_side_by_side_output_video(args: argparse.Namespace) -> int:
    """Export existing side-by-side frames to an MP4 file."""
    _require_cv2()
    (
        online_dir,
        case_dir,
        camera,
        fps,
        capture_dir,
        input_timeline,
        fake_input_frame_total,
    ) = _playback_context(args)
    renderer = build_frame_renderer(args, camera=camera, fps=fps)
    input_frames = (
        []
        if capture_dir is None or input_timeline is None
        else load_input_rgb_frames(input_timeline, capture_dir=capture_dir)
    )
    loaded_paths: set[Path] = set()
    output_frames: list[tuple[dict[str, Any], int]] = []
    _append_new_output_frames(
        online_dir,
        start_chunk=int(args.start_chunk),
        loaded_paths=loaded_paths,
        output_frames=output_frames,
    )
    total_frames = max(len(input_frames), len(output_frames), 1)
    output_path = Path(args.output_video).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    try:
        for index in range(total_frames):
            input_frame = input_frames[min(index, len(input_frames) - 1)] if input_frames else None
            output_frame = None
            if index < len(output_frames):
                output_frame = _render_output_timeline_frame(
                    output_frames,
                    output_index=index,
                    renderer=renderer,
                    case_dir=case_dir,
                )
            image = render_side_by_side_frame(
                input_frame=input_frame,
                output_frame=output_frame,
                image_size=camera.image_size,
                right_blank_label=str(args.right_blank_label),
                fake_input_frame_total=fake_input_frame_total,
                show_latency_overlay=False,
            )
            if writer is None:
                height, width = image.shape[:2]
                writer = open_video_writer(output_path, size=(int(width), int(height)), fps=fps)
            writer.write(image)
    finally:
        if writer is not None:
            writer.release()
        renderer.close()
    return 0


def export_output_video(args: argparse.Namespace) -> int:
    """Export existing output chunks to an MP4 file."""
    _require_cv2()
    if str(getattr(args, "layout", LAYOUT_OUTPUT_ONLY)) == LAYOUT_SIDE_BY_SIDE:
        return export_side_by_side_output_video(args)
    online_dir = normalize_online_dir(args.online_dir)
    case_dir = infer_case_dir(online_dir, args.case_dir)
    camera = load_camera_model(case_dir, cam_idx=int(args.cam_idx))
    fps = resolve_playback_fps(args, camera)
    chunk_paths = list_available_chunk_paths(online_dir, start_chunk=int(args.start_chunk))
    if not chunk_paths:
        raise ValueError(f"no chunk_*.pkl files found under {online_dir / 'chunks'}")
    output_path = Path(args.output_video).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    renderer = build_frame_renderer(args, camera=camera, fps=fps)
    writer = None
    try:
        for chunk_path in chunk_paths:
            chunk = dict(load_pickle(chunk_path))
            frame_count = _chunk_frame_count(chunk)
            for local_frame in range(frame_count):
                image = renderer.render_frame(chunk, local_frame=local_frame, case_dir=case_dir)
                if writer is None:
                    height, width = image.shape[:2]
                    writer = open_video_writer(output_path, size=(int(width), int(height)), fps=fps)
                writer.write(image)
    finally:
        if writer is not None:
            writer.release()
        renderer.close()
    return 0
