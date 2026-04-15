from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_process.visualization.pointcloud_defaults import (
    DEFAULT_POINTCLOUD_DEPTH_MAX_M,
    DEFAULT_POINTCLOUD_DEPTH_MIN_M,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write single-frame fused PLY compare outputs for Native, FFS raw, and FFS postprocess."
    )
    parser.add_argument("--case_name", type=str, default=None)
    parser.add_argument("--aligned_root", type=Path, default=ROOT / "data")
    parser.add_argument("--realsense_case", type=str, default=None)
    parser.add_argument("--ffs_case", type=str, default=None)
    parser.add_argument("--frame_idx", type=int, default=0)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--voxel_size", type=float, default=None)
    parser.add_argument("--max_points_per_camera", type=int, default=None)
    parser.add_argument("--depth_min_m", type=float, default=DEFAULT_POINTCLOUD_DEPTH_MIN_M)
    parser.add_argument("--depth_max_m", type=float, default=DEFAULT_POINTCLOUD_DEPTH_MAX_M)
    parser.add_argument(
        "--use_float_ffs_depth_when_available",
        dest="use_float_ffs_depth_when_available",
        action="store_true",
    )
    parser.add_argument(
        "--no_use_float_ffs_depth_when_available",
        dest="use_float_ffs_depth_when_available",
        action="store_false",
    )
    parser.set_defaults(use_float_ffs_depth_when_available=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from data_process.visualization.workflows.triplet_ply_compare import run_triplet_ply_compare_workflow

    output_dir = args.output_dir
    if output_dir is None:
        if args.case_name:
            output_dir = args.aligned_root / args.case_name / f"ply_compare_triplet_frame_{args.frame_idx:04d}"
        else:
            output_dir = (
                args.aligned_root
                / f"ply_compare_triplet_{args.realsense_case}_vs_{args.ffs_case}_frame_{args.frame_idx:04d}"
            )

    result = run_triplet_ply_compare_workflow(
        aligned_root=args.aligned_root,
        output_dir=output_dir,
        case_name=args.case_name,
        realsense_case=args.realsense_case,
        ffs_case=args.ffs_case,
        frame_idx=args.frame_idx,
        voxel_size=args.voxel_size,
        max_points_per_camera=args.max_points_per_camera,
        depth_min_m=args.depth_min_m,
        depth_max_m=args.depth_max_m,
        use_float_ffs_depth_when_available=bool(args.use_float_ffs_depth_when_available),
    )
    print(f"Triplet fused PLY compare outputs written to {result['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
