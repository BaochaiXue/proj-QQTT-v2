from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a single-frame masked RGB reference board plus a Native-vs-FFS masked pointcloud board from the 3 original calibrated camera views."
    )
    parser.add_argument("--case_name", type=str, default=None)
    parser.add_argument("--aligned_root", type=Path, default=ROOT / "data")
    parser.add_argument("--realsense_case", type=str, default=None)
    parser.add_argument("--ffs_case", type=str, default=None)
    parser.add_argument("--frame_idx", type=int, default=0)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--text_prompt", type=str, required=True)
    parser.add_argument(
        "--mask_source_mode",
        choices=("reuse_or_generate", "generate", "require_existing"),
        default="reuse_or_generate",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--source_mode", choices=("auto", "mp4", "frames"), default="auto")
    parser.add_argument("--ann_frame_index", type=int, default=None)
    parser.add_argument("--camera_ids", nargs="+", type=int, default=None)
    parser.add_argument("--voxel_size", type=float, default=None)
    parser.add_argument("--max_points_per_camera", type=int, default=None)
    parser.add_argument("--depth_min_m", type=float, default=0.0)
    parser.add_argument("--depth_max_m", type=float, default=1.5)
    parser.add_argument("--look_distance", type=float, default=1.0)
    parser.add_argument("--native_depth_postprocess", action="store_true")
    parser.add_argument("--ffs_native_like_postprocess", action="store_true")
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


def _default_output_dir(
    *,
    aligned_root: Path,
    case_name: str | None,
    realsense_case: str | None,
    ffs_case: str | None,
    frame_idx: int,
) -> Path:
    if case_name:
        return aligned_root / case_name / f"masked_camera_view_compare_frame_{int(frame_idx):04d}"
    return aligned_root / f"masked_camera_view_compare_{realsense_case}_vs_{ffs_case}_frame_{int(frame_idx):04d}"


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = _default_output_dir(
            aligned_root=Path(args.aligned_root).resolve(),
            case_name=args.case_name,
            realsense_case=args.realsense_case,
            ffs_case=args.ffs_case,
            frame_idx=int(args.frame_idx),
        )
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    from data_process.visualization.io_case import resolve_case_dirs
    from data_process.visualization.workflows.masked_camera_view_compare import (
        run_masked_camera_view_compare_workflow,
    )
    from scripts.harness.visual_compare_masked_pointcloud import resolve_mask_roots_for_compare

    native_case_dir, _, _ = resolve_case_dirs(
        aligned_root=Path(args.aligned_root).resolve(),
        case_name=args.case_name,
        realsense_case=args.realsense_case,
        ffs_case=args.ffs_case,
    )
    if args.camera_ids is None:
        inferred_camera_ids = sorted(int(path.name) for path in (native_case_dir / "color").iterdir() if path.is_dir())
    else:
        inferred_camera_ids = [int(item) for item in args.camera_ids]

    ann_frame_index = int(args.frame_idx) if args.ann_frame_index is None else int(args.ann_frame_index)
    mask_roots = resolve_mask_roots_for_compare(
        aligned_root=Path(args.aligned_root).resolve(),
        case_name=args.case_name,
        realsense_case=args.realsense_case,
        ffs_case=args.ffs_case,
        output_dir=output_dir,
        text_prompt=args.text_prompt,
        camera_ids=inferred_camera_ids,
        frame_idx=ann_frame_index,
        mask_source_mode=str(args.mask_source_mode),
        checkpoint=args.checkpoint,
        source_mode=str(args.source_mode),
    )

    result = run_masked_camera_view_compare_workflow(
        aligned_root=Path(args.aligned_root).resolve(),
        output_dir=output_dir,
        case_name=args.case_name,
        realsense_case=args.realsense_case,
        ffs_case=args.ffs_case,
        frame_idx=int(args.frame_idx),
        camera_ids=inferred_camera_ids,
        native_mask_root=mask_roots["native"]["mask_root"],
        ffs_mask_root=mask_roots["ffs"]["mask_root"],
        native_mask_source=str(mask_roots["native"]["mask_source"]),
        ffs_mask_source=str(mask_roots["ffs"]["mask_source"]),
        mask_source_mode=str(args.mask_source_mode),
        text_prompt=args.text_prompt,
        voxel_size=args.voxel_size,
        max_points_per_camera=args.max_points_per_camera,
        depth_min_m=float(args.depth_min_m),
        depth_max_m=float(args.depth_max_m),
        use_float_ffs_depth_when_available=bool(args.use_float_ffs_depth_when_available),
        native_depth_postprocess=bool(args.native_depth_postprocess),
        ffs_native_like_postprocess=bool(args.ffs_native_like_postprocess),
        look_distance=float(args.look_distance),
    )
    print(f"Masked camera-view compare outputs written to {result['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
