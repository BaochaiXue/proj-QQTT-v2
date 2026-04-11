from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream multi-frame native-vs-FFS remove-invisible point-cloud variants to Rerun and export fused PLYs."
    )
    parser.add_argument("--aligned_root", type=Path, default=ROOT / "data")
    parser.add_argument("--realsense_case", type=str, required=True)
    parser.add_argument("--ffs_case", type=str, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--frame_start", type=int, default=None)
    parser.add_argument("--frame_end", type=int, default=None)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--ffs_repo", type=Path, default=None)
    parser.add_argument("--ffs_model_path", type=Path, default=None)
    parser.add_argument(
        "--rerun_output",
        choices=("viewer_and_rrd", "viewer_only", "rrd_only"),
        default="viewer_and_rrd",
    )
    parser.add_argument(
        "--viewer_layout",
        choices=("default", "horizontal_triple"),
        default="default",
    )
    parser.add_argument("--voxel_size", type=float, default=None)
    parser.add_argument("--max_points_per_camera", type=int, default=None)
    parser.add_argument("--depth_min_m", type=float, default=0.1)
    parser.add_argument("--depth_max_m", type=float, default=3.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from data_process.visualization.rerun_compare import run_rerun_compare_workflow

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = args.aligned_root / f"rerun_compare_{args.realsense_case}_vs_{args.ffs_case}"

    summary = run_rerun_compare_workflow(
        aligned_root=args.aligned_root,
        realsense_case=args.realsense_case,
        ffs_case=args.ffs_case,
        output_dir=output_dir,
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        frame_stride=args.frame_stride,
        ffs_repo=args.ffs_repo,
        ffs_model_path=args.ffs_model_path,
        rerun_output=args.rerun_output,
        viewer_layout=args.viewer_layout,
        voxel_size=args.voxel_size,
        max_points_per_camera=args.max_points_per_camera,
        depth_min_m=args.depth_min_m,
        depth_max_m=args.depth_max_m,
    )
    print(f"Wrote rerun compare outputs to {output_dir}")
    if summary.get("rrd_path"):
        print(f"Saved RRD: {summary['rrd_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
