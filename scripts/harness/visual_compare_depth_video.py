from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render native-vs-FFS fused point-cloud comparison frames and videos.")
    parser.add_argument("--case_name", type=str, default=None)
    parser.add_argument("--aligned_root", type=Path, default=ROOT / "data")
    parser.add_argument("--realsense_case", type=str, default=None)
    parser.add_argument("--ffs_case", type=str, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--frame_start", type=int, default=None)
    parser.add_argument("--frame_end", type=int, default=None)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--voxel_size", type=float, default=None)
    parser.add_argument("--max_points_per_camera", type=int, default=50000)
    parser.add_argument("--depth_min_m", type=float, default=0.1)
    parser.add_argument("--depth_max_m", type=float, default=3.0)
    parser.add_argument("--renderer", choices=("auto", "open3d", "fallback"), default="auto")
    parser.add_argument("--write_ply", action="store_true")
    parser.add_argument("--write_mp4", action="store_true")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--panel_layout", choices=("side_by_side", "stacked"), default="side_by_side")
    parser.add_argument("--use_float_ffs_depth_when_available", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from data_process.visualization.pointcloud_compare import run_depth_comparison_workflow
    output_dir = args.output_dir
    if output_dir is None:
        if args.case_name:
            output_dir = args.aligned_root / args.case_name / "comparison"
        else:
            output_dir = args.aligned_root / f"comparison_{args.realsense_case}_vs_{args.ffs_case}"

    result = run_depth_comparison_workflow(
        aligned_root=args.aligned_root,
        case_name=args.case_name,
        realsense_case=args.realsense_case,
        ffs_case=args.ffs_case,
        output_dir=output_dir,
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        frame_stride=args.frame_stride,
        voxel_size=args.voxel_size,
        max_points_per_camera=args.max_points_per_camera,
        depth_min_m=args.depth_min_m,
        depth_max_m=args.depth_max_m,
        renderer=args.renderer,
        write_ply=args.write_ply,
        write_mp4=args.write_mp4,
        fps=args.fps,
        panel_layout=args.panel_layout,
        use_float_ffs_depth_when_available=args.use_float_ffs_depth_when_available,
    )
    print(f"Comparison outputs written to {result['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
