from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render native-vs-FFS cross-view reprojection diagnostics.")
    parser.add_argument("--case_name", type=str, default=None)
    parser.add_argument("--aligned_root", type=Path, default=ROOT / "data")
    parser.add_argument("--realsense_case", type=str, default=None)
    parser.add_argument("--ffs_case", type=str, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--frame_start", type=int, default=None)
    parser.add_argument("--frame_end", type=int, default=None)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--camera_ids", nargs="*", type=int, default=None)
    parser.add_argument("--camera_pair", action="append", default=None, help="Repeatable camera pair src,dst.")
    parser.add_argument("--write_mp4", action="store_true")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--use_float_ffs_depth_when_available", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from data_process.visualization.reprojection_compare import parse_camera_pair, run_reprojection_compare_workflow

    camera_pairs = [parse_camera_pair(spec) for spec in (args.camera_pair or [])]
    output_dir = args.output_dir
    if output_dir is None:
        if args.case_name:
            output_dir = args.aligned_root / args.case_name / "reprojection_compare"
        else:
            output_dir = args.aligned_root / f"reprojection_{args.realsense_case}_vs_{args.ffs_case}"

    result = run_reprojection_compare_workflow(
        aligned_root=args.aligned_root,
        output_dir=output_dir,
        case_name=args.case_name,
        realsense_case=args.realsense_case,
        ffs_case=args.ffs_case,
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        frame_stride=args.frame_stride,
        camera_ids=args.camera_ids,
        camera_pairs=camera_pairs or None,
        write_mp4=args.write_mp4,
        fps=args.fps,
        use_float_ffs_depth_when_available=args.use_float_ffs_depth_when_available,
    )
    print(f"Reprojection outputs written to {result['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
