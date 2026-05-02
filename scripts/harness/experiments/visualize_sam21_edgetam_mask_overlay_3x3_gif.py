#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_process.visualization.experiments.sam21_mask_overlay_panel import (
    DEFAULT_CASE_DIR,
    DEFAULT_OUTPUT_DIR,
    run_dynamics_round1_overlay_workflow,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a 3x3 GIF comparing SAM2.1 Small/Tiny and compiled EdgeTAM "
            "masks against SAM3.1 masks."
        )
    )
    parser.add_argument("--case-dir", type=Path, default=DEFAULT_CASE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--all-frames", action="store_true")
    parser.add_argument("--gif-fps", type=int, default=6)
    parser.add_argument("--tile-width", type=int, default=320)
    parser.add_argument("--tile-height", type=int, default=180)
    parser.add_argument("--row-label-width", type=int, default=92)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_dynamics_round1_overlay_workflow(
        root=ROOT,
        case_dir=Path(args.case_dir),
        output_dir=Path(args.output_dir),
        frames=None if bool(args.all_frames) else args.frames,
        gif_fps=int(args.gif_fps),
        tile_width=int(args.tile_width),
        tile_height=int(args.tile_height),
        row_label_width=int(args.row_label_width),
    )
    print(f"[mask-overlay] gif: {summary['gif_path']}", flush=True)
    print(f"[mask-overlay] first frame: {summary['first_frame_path']}", flush=True)
    docs = summary.get("docs", {})
    if docs:
        print(f"[mask-overlay] report: {docs.get('benchmark_md')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
