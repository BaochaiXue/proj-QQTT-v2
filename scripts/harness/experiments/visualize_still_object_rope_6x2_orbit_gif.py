from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_process.visualization.experiments.still_object_orbit_gif import (  # noqa: E402
    DEFAULT_6X2_OUTPUT_DIR,
    default_still_object_rope_6x2_case_specs,
    run_still_object_rope_6x2_orbit_gif_workflow,
)


def _positive_int_or_none(value: str) -> int | None:
    parsed = int(value)
    return None if parsed <= 0 else parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a 6x2 Native Depth vs FFS raw-RGB masked-object orbit GIF "
            "for still-object rounds 1-4 and still-rope rounds 1-2."
        )
    )
    parser.add_argument("--output_dir", type=Path, default=ROOT / DEFAULT_6X2_OUTPUT_DIR)
    parser.add_argument("--frame_idx", type=int, default=0)
    parser.add_argument("--start_camera_idx", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=360)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--tile_width", type=int, default=360)
    parser.add_argument("--tile_height", type=int, default=220)
    parser.add_argument("--row_label_width", type=int, default=180)
    parser.add_argument("--depth_min_m", type=float, default=0.2)
    parser.add_argument("--depth_max_m", type=float, default=1.5)
    parser.add_argument("--max_points_per_camera", type=_positive_int_or_none, default=None)
    parser.add_argument("--max_points_per_variant", type=_positive_int_or_none, default=120000)
    parser.add_argument("--robust_bounds_percentile", type=float, default=1.0)
    parser.add_argument(
        "--crop_to_robust_bounds",
        dest="crop_to_robust_bounds",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no_crop_to_robust_bounds",
        dest="crop_to_robust_bounds",
        action="store_false",
    )
    parser.add_argument("--crop_margin_ratio", type=float, default=0.25)
    parser.add_argument(
        "--render_mode",
        choices=("color_by_rgb", "color_by_height", "color_by_depth", "color_by_normals", "neutral_gray_shaded"),
        default="color_by_rgb",
    )
    parser.add_argument(
        "--projection_mode",
        choices=("orthographic", "perspective"),
        default="orthographic",
    )
    parser.add_argument("--ortho_margin", type=float, default=1.28)
    parser.add_argument("--point_radius_px", type=int, default=1)
    parser.add_argument("--supersample_scale", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_still_object_rope_6x2_orbit_gif_workflow(
        case_specs=default_still_object_rope_6x2_case_specs(root=ROOT),
        output_dir=args.output_dir,
        frame_idx=int(args.frame_idx),
        start_camera_idx=int(args.start_camera_idx),
        num_frames=int(args.num_frames),
        fps=int(args.fps),
        tile_width=int(args.tile_width),
        tile_height=int(args.tile_height),
        row_label_width=int(args.row_label_width),
        depth_min_m=float(args.depth_min_m),
        depth_max_m=float(args.depth_max_m),
        max_points_per_camera=args.max_points_per_camera,
        max_points_per_variant=args.max_points_per_variant,
        robust_bounds_percentile=float(args.robust_bounds_percentile),
        crop_to_robust_bounds=bool(args.crop_to_robust_bounds),
        crop_margin_ratio=float(args.crop_margin_ratio),
        render_mode=str(args.render_mode),
        projection_mode=str(args.projection_mode),
        ortho_margin=float(args.ortho_margin),
        point_radius_px=int(args.point_radius_px),
        supersample_scale=int(args.supersample_scale),
    )
    print(f"GIF written to {summary['gif_path']}")
    print(f"First frame written to {summary['first_frame_path']}")
    print(f"Summary written to {Path(summary['output_dir']) / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
