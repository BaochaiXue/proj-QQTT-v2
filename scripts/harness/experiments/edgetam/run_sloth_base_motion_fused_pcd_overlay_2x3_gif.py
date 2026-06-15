#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_process.visualization.experiments.sam21_pcd_overlay_panel import (
    DEFAULT_DEPTH_SCALE_OVERRIDE_M_PER_UNIT,
    DEFAULT_DOC_JSON,
    DEFAULT_DOC_MD,
    DEFAULT_OUTPUT_DIR,
    SLOTH_BASE_MOTION_CASE_DIR,
    SLOTH_BASE_MOTION_DOC_JSON,
    run_sloth_base_motion_fused_pcd_overlay_workflow,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a 2x3 fused PCD overlay GIF for sloth_base_motion_ffs using "
            "existing SAM3.1, SAM2.1 Small, and compiled EdgeTAM masks."
        )
    )
    parser.add_argument("--case-dir", type=Path, default=SLOTH_BASE_MOTION_CASE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--existing-results-json", type=Path, default=SLOTH_BASE_MOTION_DOC_JSON)
    parser.add_argument("--frames", type=int, help="Limit frames for smoke/debug runs. Omit for all frames.")
    parser.add_argument("--gif-fps", type=int, default=6)
    parser.add_argument("--tile-width", type=int, default=320)
    parser.add_argument("--tile-height", type=int, default=180)
    parser.add_argument("--row-label-width", type=int, default=140)
    parser.add_argument("--depth-scale-override-m-per-unit", type=float, default=DEFAULT_DEPTH_SCALE_OVERRIDE_M_PER_UNIT)
    parser.add_argument(
        "--max-points-per-camera",
        type=int,
        help="Optional deterministic sampling cap before mask fusion.",
    )
    parser.add_argument(
        "--max-points-per-render",
        type=int,
        default=100_000,
        help="Optional deterministic sampling cap per rendered fused overlay cloud.",
    )
    parser.add_argument("--doc-md", type=Path, default=DEFAULT_DOC_MD)
    parser.add_argument("--doc-json", type=Path, default=DEFAULT_DOC_JSON)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_sloth_base_motion_fused_pcd_overlay_workflow(
        root=ROOT,
        case_dir=args.case_dir,
        output_dir=args.output_dir,
        existing_results_json=args.existing_results_json,
        frames=args.frames,
        gif_fps=int(args.gif_fps),
        tile_width=int(args.tile_width),
        tile_height=int(args.tile_height),
        row_label_width=int(args.row_label_width),
        depth_scale_override_m_per_unit=float(args.depth_scale_override_m_per_unit),
        max_points_per_camera=args.max_points_per_camera,
        max_points_per_render=args.max_points_per_render,
        doc_md=args.doc_md,
        doc_json=args.doc_json,
    )
    print(f"[pcd-overlay] gif: {summary['gif_path']}", flush=True)
    print(f"[pcd-overlay] first frame: {summary['first_frame_path']}", flush=True)
    print(f"[pcd-overlay] report: {summary['docs']['benchmark_md']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
