from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render per-camera native-vs-FFS depth diagnostic panels.")
    parser.add_argument("--case_name", type=str, default=None)
    parser.add_argument("--aligned_root", type=Path, default=ROOT / "data")
    parser.add_argument("--realsense_case", type=str, default=None)
    parser.add_argument("--ffs_case", type=str, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--frame_start", type=int, default=None)
    parser.add_argument("--frame_end", type=int, default=None)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--camera_ids", nargs="*", type=int, default=None)
    parser.add_argument("--depth_min_m", type=float, default=0.1)
    parser.add_argument("--depth_max_m", type=float, default=3.0)
    parser.add_argument("--roi", action="append", default=None, help="Repeatable ROI in x0,y0,x1,y1 or name:x0,y0,x1,y1 format.")
    parser.add_argument("--preset", choices=("review_quality",), default=None)
    parser.add_argument("--show_edge_overlay", dest="show_edge_overlay", action="store_true")
    parser.add_argument("--no_show_edge_overlay", dest="show_edge_overlay", action="store_false")
    parser.add_argument("--write_mp4", action="store_true")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--use_float_ffs_depth_when_available", action="store_true")
    parser.set_defaults(show_edge_overlay=None)
    return parser.parse_args()


def apply_preset(args: argparse.Namespace) -> argparse.Namespace:
    if args.preset == "review_quality":
        args.show_edge_overlay = True if args.show_edge_overlay is None else args.show_edge_overlay
        args.depth_min_m = 0.2 if args.depth_min_m == 0.1 else args.depth_min_m
        args.depth_max_m = 1.5 if args.depth_max_m == 3.0 else args.depth_max_m
    elif args.show_edge_overlay is None:
        args.show_edge_overlay = False
    return args


def main() -> int:
    args = apply_preset(parse_args())
    from data_process.visualization.depth_diagnostics import parse_named_roi_spec
    from data_process.visualization.workflows.depth_panels import run_depth_panels_workflow

    rois = [parse_named_roi_spec(spec) for spec in (args.roi or [])]
    output_dir = args.output_dir
    if output_dir is None:
        if args.case_name:
            output_dir = args.aligned_root / args.case_name / "depth_panels"
        else:
            output_dir = args.aligned_root / f"depth_panels_{args.realsense_case}_vs_{args.ffs_case}"

    result = run_depth_panels_workflow(
        aligned_root=args.aligned_root,
        output_dir=output_dir,
        case_name=args.case_name,
        realsense_case=args.realsense_case,
        ffs_case=args.ffs_case,
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        frame_stride=args.frame_stride,
        camera_ids=args.camera_ids,
        depth_min_m=args.depth_min_m,
        depth_max_m=args.depth_max_m,
        rois=rois or None,
        write_mp4=args.write_mp4,
        fps=args.fps,
        use_float_ffs_depth_when_available=args.use_float_ffs_depth_when_available,
        preset=args.preset,
        show_edge_overlay=bool(args.show_edge_overlay),
    )
    print(f"Depth panel outputs written to {result['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
