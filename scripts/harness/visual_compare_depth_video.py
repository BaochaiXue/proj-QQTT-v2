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
    parser.add_argument("--depth_min_m", type=float, default=DEFAULT_POINTCLOUD_DEPTH_MIN_M)
    parser.add_argument("--depth_max_m", type=float, default=DEFAULT_POINTCLOUD_DEPTH_MAX_M)
    parser.add_argument("--renderer", choices=("auto", "open3d", "fallback"), default="auto")
    parser.add_argument("--preset", choices=("tabletop_compare_2x3",), default=None)
    parser.add_argument("--render_mode", choices=("color_by_rgb", "color_by_depth", "color_by_height", "color_by_normals", "neutral_gray_shaded"), default="neutral_gray_shaded")
    parser.add_argument("--views", nargs="+", choices=("oblique", "top", "side"), default=["oblique"])
    parser.add_argument("--view_mode", choices=("fixed", "camera_poses_table_focus"), default="fixed")
    parser.add_argument("--focus_mode", choices=("none", "table"), default="none")
    parser.add_argument("--layout_mode", choices=("pair", "grid_2x3"), default="pair")
    parser.add_argument("--scene_crop_mode", choices=("none", "auto_table_bbox", "auto_object_bbox", "manual_xyz_roi"), default="none")
    parser.add_argument("--crop_margin_xy", type=float, default=0.12)
    parser.add_argument("--crop_min_z", type=float, default=-0.15)
    parser.add_argument("--crop_max_z", type=float, default=0.35)
    parser.add_argument("--roi_x_min", type=float, default=None)
    parser.add_argument("--roi_x_max", type=float, default=None)
    parser.add_argument("--roi_y_min", type=float, default=None)
    parser.add_argument("--roi_y_max", type=float, default=None)
    parser.add_argument("--roi_z_min", type=float, default=None)
    parser.add_argument("--roi_z_max", type=float, default=None)
    parser.add_argument("--write_ply", action="store_true")
    parser.add_argument("--write_mp4", action="store_true")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--panel_layout", choices=("side_by_side", "stacked"), default="side_by_side")
    parser.add_argument("--use_float_ffs_depth_when_available", action="store_true")
    parser.add_argument("--zoom_scale", type=float, default=1.0)
    parser.add_argument("--view_distance_scale", type=float, default=1.0)
    parser.add_argument("--projection_mode", choices=("perspective", "orthographic"), default="perspective")
    parser.add_argument("--ortho_scale", type=float, default=None)
    parser.add_argument("--point_radius_px", type=int, default=2)
    parser.add_argument("--supersample_scale", type=int, default=2)
    parser.add_argument("--image_flip", choices=("none", "vertical", "horizontal", "both"), default="none")
    return parser.parse_args()


def apply_preset(args: argparse.Namespace) -> argparse.Namespace:
    if args.preset == "tabletop_compare_2x3":
        args.render_mode = "color_by_height"
        args.view_mode = "camera_poses_table_focus"
        args.focus_mode = "table"
        args.layout_mode = "grid_2x3"
        args.scene_crop_mode = "auto_table_bbox"
        args.projection_mode = "orthographic"
        args.view_distance_scale = 1.0
        args.point_radius_px = 3
        args.supersample_scale = 2
        args.depth_min_m = 0.2
        args.depth_max_m = 1.5
        args.zoom_scale = 1.0
        args.views = ["oblique"]
    return args


def main() -> int:
    args = apply_preset(parse_args())
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
        render_mode=args.render_mode,
        views=args.views,
        view_mode=args.view_mode,
        focus_mode=args.focus_mode,
        layout_mode=args.layout_mode,
        scene_crop_mode=args.scene_crop_mode,
        crop_margin_xy=args.crop_margin_xy,
        crop_min_z=args.crop_min_z,
        crop_max_z=args.crop_max_z,
        roi_x_min=args.roi_x_min,
        roi_x_max=args.roi_x_max,
        roi_y_min=args.roi_y_min,
        roi_y_max=args.roi_y_max,
        roi_z_min=args.roi_z_min,
        roi_z_max=args.roi_z_max,
        write_ply=args.write_ply,
        write_mp4=args.write_mp4,
        fps=args.fps,
        panel_layout=args.panel_layout,
        use_float_ffs_depth_when_available=args.use_float_ffs_depth_when_available,
        zoom_scale=args.zoom_scale,
        view_distance_scale=args.view_distance_scale,
        projection_mode=args.projection_mode,
        ortho_scale=args.ortho_scale,
        point_radius_px=args.point_radius_px,
        supersample_scale=args.supersample_scale,
        image_flip=args.image_flip,
    )
    print(f"Comparison outputs written to {result['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
