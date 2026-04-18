from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose Native-vs-FFS floating-point source causes on aligned cases.")
    parser.add_argument("--case_name", type=str, default=None)
    parser.add_argument("--aligned_root", type=Path, default=ROOT / "data")
    parser.add_argument("--realsense_case", type=str, default=None)
    parser.add_argument("--ffs_case", type=str, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--frame_start", type=int, default=None)
    parser.add_argument("--frame_end", type=int, default=None)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--camera_ids", nargs="*", type=int, default=None)
    parser.add_argument("--use_float_ffs_depth_when_available", action="store_true")
    parser.add_argument("--ffs_native_like_postprocess", action="store_true")
    parser.add_argument("--radius_m", type=float, default=0.01)
    parser.add_argument("--nb_points", type=int, default=40)
    parser.add_argument("--edge_band_px", type=int, default=8)
    parser.add_argument("--dark_threshold", type=float, default=40.0)
    parser.add_argument("--occlusion_depth_tol_m", type=float, default=0.02)
    parser.add_argument("--occlusion_depth_tol_ratio", type=float, default=0.03)
    parser.add_argument("--write_mp4", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from data_process.visualization.floating_point_diagnostics import run_floating_point_source_diagnostics_workflow

    output_dir = args.output_dir
    if output_dir is None:
        if args.case_name:
            output_dir = args.aligned_root / args.case_name / "floating_point_diagnostics"
        else:
            output_dir = args.aligned_root / f"floating_point_diagnostics_{args.realsense_case}_vs_{args.ffs_case}"

    result = run_floating_point_source_diagnostics_workflow(
        aligned_root=args.aligned_root,
        output_dir=output_dir,
        case_name=args.case_name,
        realsense_case=args.realsense_case,
        ffs_case=args.ffs_case,
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        frame_stride=args.frame_stride,
        camera_ids=args.camera_ids,
        use_float_ffs_depth_when_available=args.use_float_ffs_depth_when_available,
        ffs_native_like_postprocess=args.ffs_native_like_postprocess,
        radius_m=args.radius_m,
        nb_points=args.nb_points,
        edge_band_px=args.edge_band_px,
        dark_threshold=args.dark_threshold,
        occlusion_depth_tol_m=args.occlusion_depth_tol_m,
        occlusion_depth_tol_ratio=args.occlusion_depth_tol_ratio,
        write_mp4=args.write_mp4,
    )
    print(f"Floating-point diagnostics written to {result['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
