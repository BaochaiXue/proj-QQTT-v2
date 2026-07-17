#!/usr/bin/env python3
"""Track visualization for Demo v6.2 object/controller point chunks.

The viewer can render historical chunk files or follow a live run. In
side-by-side mode the left panel follows camera RGB input while the right panel
chooses the final_data frame whose source timestamp best matches the desired
camera-to-output latency.

This module is the thin CLI entry point. The implementation lives in the
``demo_v6_2.visualization`` package. This file keeps the module constants it
owns and the CLI surface.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT_STR = str(REPO_ROOT)
if REPO_ROOT_STR in sys.path:
    sys.path.remove(REPO_ROOT_STR)
sys.path.insert(0, REPO_ROOT_STR)

from demo_v6_2.visualization.viz_camera_model import (
    _require_cv2,
    infer_case_dir,
    load_camera_model,
    normalize_online_dir,
)
from demo_v6_2.visualization.viz_panels import DEFAULT_RIGHT_BLANK_LABEL, parse_bgr_color
from demo_v6_2.visualization.viz_playback import (
    LAYOUT_OUTPUT_ONLY,
    LAYOUT_SIDE_BY_SIDE,
    LAYOUTS,
    play_chunk,
    resolve_playback_fps,
    run_interactive_side_by_side,
    run_side_by_side,
    use_interactive_side_by_side,
    wait_for_chunk,
)
from demo_v6_2.visualization.viz_renderers import (
    RENDER_MODE_RGB_OVERLAY,
    RENDER_MODES,
    build_frame_renderer,
)
from demo_v6_2.visualization.viz_video_export import export_output_video


DEFAULT_WINDOW_NAME = "Demo v6.2 visualize track"
DEFAULT_OBJECT_RADIUS = 3
DEFAULT_CONTROLLER_RADIUS = 6


# --- CLI entry points --------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the Demo v6.2 chunk viewer CLI parser."""
    parser = argparse.ArgumentParser(
        description="Play Demo v6.2 online object/controller points chunk by chunk."
    )
    parser.add_argument("--layout", choices=LAYOUTS, default=LAYOUT_OUTPUT_ONLY)
    parser.add_argument(
        "--online-dir",
        type=Path,
        required=True,
        help="Path to online_data or its chunks directory.",
    )
    parser.add_argument(
        "--case-dir",
        type=Path,
        default=None,
        help="Path to data. Inferred from --online-dir when omitted.",
    )
    parser.add_argument("--render-mode", choices=RENDER_MODES, default=RENDER_MODE_RGB_OVERLAY)
    parser.add_argument("--output-video", type=Path, default=None, help="Write existing chunks to MP4 and exit instead of opening a live window.")
    parser.add_argument("--capture-dir", type=Path, default=None, help="Headless capture dir containing input_frames.jsonl and input_rgb/*.png.")
    parser.add_argument("--input-rgb-timeline", type=Path, default=None, help="Path to input_frames.jsonl. Defaults to --capture-dir/input_frames.jsonl.")
    parser.add_argument("--right-blank-label", default=DEFAULT_RIGHT_BLANK_LABEL)
    parser.add_argument("--follow-latest", dest="follow_latest", action="store_true", default=True)
    parser.add_argument("--no-follow-latest", dest="follow_latest", action="store_false")
    parser.add_argument("--cam-idx", type=int, default=0)
    parser.add_argument("--fps", type=float, default=None, help="Playback FPS. Defaults to metadata fps, then 5.")
    parser.add_argument("--latency-overlay", dest="latency_overlay", action="store_true", default=True)
    parser.add_argument("--no-latency-overlay", dest="latency_overlay", action="store_false")
    parser.add_argument("--poll-sec", type=float, default=0.1)
    parser.add_argument("--start-chunk", type=int, default=0)
    parser.add_argument("--object-stride", type=int, default=1)
    parser.add_argument("--object-radius", type=int, default=DEFAULT_OBJECT_RADIUS)
    parser.add_argument("--controller-radius", type=int, default=DEFAULT_CONTROLLER_RADIUS)
    parser.add_argument("--object-color-mode", choices=("rainbow", "green", "object-colors"), default="rainbow")
    parser.add_argument("--controller-color", type=parse_bgr_color, default=parse_bgr_color("0,0,255"))
    parser.add_argument("--show-invisible-object-points", action="store_true")
    parser.add_argument("--no-background", action="store_true")
    parser.add_argument("--window-name", default=DEFAULT_WINDOW_NAME)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Validate viewer CLI arguments before opening windows or videos."""
    if str(args.layout) not in LAYOUTS:
        raise ValueError(f"--layout must be one of {', '.join(LAYOUTS)}")
    if int(args.cam_idx) < 0:
        raise ValueError("--cam-idx must be non-negative")
    if float(args.poll_sec) <= 0.0:
        raise ValueError("--poll-sec must be positive")
    if int(args.start_chunk) < 0:
        raise ValueError("--start-chunk must be non-negative")
    if int(args.object_stride) <= 0:
        raise ValueError("--object-stride must be positive")
    if int(args.object_radius) <= 0:
        raise ValueError("--object-radius must be positive")
    if int(args.controller_radius) <= 0:
        raise ValueError("--controller-radius must be positive")


def run(args: argparse.Namespace) -> int:
    """Play committed chunks in order, tailing the online directory live."""
    validate_args(args)
    if args.output_video is not None:
        return export_output_video(args)
    if str(args.layout) == LAYOUT_SIDE_BY_SIDE:
        if use_interactive_side_by_side(args):
            return run_interactive_side_by_side(args)
        return run_side_by_side(args)
    cv2 = _require_cv2()
    online_dir = normalize_online_dir(args.online_dir)
    case_dir = infer_case_dir(online_dir, args.case_dir)
    camera = load_camera_model(case_dir, cam_idx=int(args.cam_idx))
    fps = resolve_playback_fps(args, camera)
    renderer = build_frame_renderer(args, camera=camera, fps=fps)
    cv2.namedWindow(str(args.window_name), cv2.WINDOW_NORMAL)
    chunk_id = int(args.start_chunk)
    last_image: np.ndarray | None = None
    try:
        while True:
            chunk = wait_for_chunk(
                online_dir,
                chunk_id=chunk_id,
                poll_sec=float(args.poll_sec),
                window_name=str(args.window_name),
                last_image=last_image,
            )
            if chunk is None:
                return 0
            last_image = play_chunk(
                chunk,
                case_dir=case_dir,
                renderer=renderer,
                args=args,
                fps=fps,
            )
            if last_image is None:
                return 0
            chunk_id += 1
    finally:
        renderer.close()


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments and run the Demo v6.2 viewer."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
