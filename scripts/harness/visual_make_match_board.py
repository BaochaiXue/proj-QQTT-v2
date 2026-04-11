from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a single professor-facing 3-view point-cloud match board.")
    parser.add_argument("--case_name", type=str, default=None)
    parser.add_argument("--aligned_root", type=Path, default=ROOT / "data")
    parser.add_argument("--realsense_case", type=str, default=None)
    parser.add_argument("--ffs_case", type=str, default=None)
    parser.add_argument("--frame_idx", type=int, default=0)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--renderer", choices=("auto", "open3d", "fallback"), default="fallback")
    parser.add_argument("--projection_mode", choices=("perspective", "orthographic"), default="perspective")
    parser.add_argument("--display_frame", choices=("calibration_world", "semantic_world"), default="semantic_world")
    parser.add_argument("--scene_crop_mode", choices=("none", "auto_table_bbox", "auto_object_bbox", "manual_xyz_roi"), default="auto_object_bbox")
    parser.add_argument("--crop_margin_xy", type=float, default=0.12)
    parser.add_argument("--crop_min_z", type=float, default=-0.15)
    parser.add_argument("--crop_max_z", type=float, default=0.35)
    parser.add_argument("--object_height_min", type=float, default=0.02)
    parser.add_argument("--object_height_max", type=float, default=0.60)
    parser.add_argument("--object_component_mode", choices=("graph_union", "union", "largest", "topk"), default="graph_union")
    parser.add_argument("--object_component_topk", type=int, default=2)
    parser.add_argument("--roi_x_min", type=float, default=None)
    parser.add_argument("--roi_x_max", type=float, default=None)
    parser.add_argument("--roi_y_min", type=float, default=None)
    parser.add_argument("--roi_y_max", type=float, default=None)
    parser.add_argument("--roi_z_min", type=float, default=None)
    parser.add_argument("--roi_z_max", type=float, default=None)
    parser.add_argument("--manual_image_roi_json", type=Path, default=None)
    parser.add_argument("--camera_ids", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--orbit_mode", choices=("observed_hemisphere", "full_360"), default="observed_hemisphere")
    parser.add_argument("--num_orbit_steps", type=int, default=24)
    parser.add_argument("--orbit_degrees", type=float, default=360.0)
    parser.add_argument("--orbit_radius_scale", type=float, default=1.9)
    parser.add_argument("--view_height_offset", type=float, default=0.0)
    parser.add_argument("--coverage_margin_deg", type=float, default=18.0)
    parser.add_argument("--point_radius_px", type=int, default=4)
    parser.add_argument("--supersample_scale", type=int, default=3)
    parser.add_argument("--voxel_size", type=float, default=None)
    parser.add_argument("--max_points_per_camera", type=int, default=50000)
    parser.add_argument("--depth_min_m", type=float, default=0.2)
    parser.add_argument("--depth_max_m", type=float, default=1.5)
    parser.add_argument("--use_float_ffs_depth_when_available", dest="use_float_ffs_depth_when_available", action="store_true")
    parser.add_argument("--no_use_float_ffs_depth_when_available", dest="use_float_ffs_depth_when_available", action="store_false")
    parser.add_argument("--angle_mode", choices=("auto", "explicit"), default="auto")
    parser.add_argument("--angle_deg", type=float, default=None)
    parser.add_argument("--write_debug", dest="write_debug", action="store_true")
    parser.add_argument("--no_write_debug", dest="write_debug", action="store_false")
    parser.set_defaults(
        use_float_ffs_depth_when_available=True,
        write_debug=False,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from data_process.visualization.workflows.match_board import run_match_board_workflow

    output_dir = args.output_dir
    if output_dir is None:
        if args.case_name:
            output_dir = args.aligned_root / args.case_name / f"match_board_frame_{args.frame_idx:04d}"
        else:
            output_dir = args.aligned_root / f"match_board_{args.realsense_case}_vs_{args.ffs_case}_frame_{args.frame_idx:04d}"

    result = run_match_board_workflow(
        aligned_root=args.aligned_root,
        output_dir=output_dir,
        case_name=args.case_name,
        realsense_case=args.realsense_case,
        ffs_case=args.ffs_case,
        frame_idx=args.frame_idx,
        renderer=args.renderer,
        projection_mode=args.projection_mode,
        display_frame=args.display_frame,
        scene_crop_mode=args.scene_crop_mode,
        crop_margin_xy=args.crop_margin_xy,
        crop_min_z=args.crop_min_z,
        crop_max_z=args.crop_max_z,
        object_height_min=args.object_height_min,
        object_height_max=args.object_height_max,
        object_component_mode=args.object_component_mode,
        object_component_topk=args.object_component_topk,
        roi_x_min=args.roi_x_min,
        roi_x_max=args.roi_x_max,
        roi_y_min=args.roi_y_min,
        roi_y_max=args.roi_y_max,
        roi_z_min=args.roi_z_min,
        roi_z_max=args.roi_z_max,
        manual_image_roi_json=args.manual_image_roi_json,
        camera_ids=args.camera_ids,
        orbit_mode=args.orbit_mode,
        num_orbit_steps=args.num_orbit_steps,
        orbit_degrees=args.orbit_degrees,
        orbit_radius_scale=args.orbit_radius_scale,
        view_height_offset=args.view_height_offset,
        coverage_margin_deg=args.coverage_margin_deg,
        point_radius_px=args.point_radius_px,
        supersample_scale=args.supersample_scale,
        voxel_size=args.voxel_size,
        max_points_per_camera=args.max_points_per_camera,
        depth_min_m=args.depth_min_m,
        depth_max_m=args.depth_max_m,
        use_float_ffs_depth_when_available=args.use_float_ffs_depth_when_available,
        angle_mode=args.angle_mode,
        angle_deg=args.angle_deg,
        write_debug=args.write_debug,
    )
    print(f"Match board written to {result['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
