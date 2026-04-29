from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_OUTPUT_ROOT = ROOT / "data" / "experiments" / "ffs_mask_erode_multipage_object_pcd_10x3_frame_0000_phystwin"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render static round1-3 frame-0 object-only Open3D PCD boards as one multipage "
            "mask-erode experiment: page1 is native/original/1..8px, page2 is 9..18px, "
            "and page3 is 19..28px."
        )
    )
    parser.add_argument("--aligned_root", type=Path, default=ROOT / "data")
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--frame_idx", type=int, default=0)
    parser.add_argument("--depth_min_m", type=float, default=0.2)
    parser.add_argument("--depth_max_m", type=float, default=1.5)
    parser.add_argument("--point_size", type=float, default=2.0)
    parser.add_argument("--look_distance", type=float, default=1.0)
    parser.add_argument("--tile_width", type=int, default=480)
    parser.add_argument("--tile_height", type=int, default=360)
    parser.add_argument(
        "--row_label_width",
        type=int,
        default=500,
        help="Left-side row-label band width in pixels. Increase this if row labels are clipped.",
    )
    parser.add_argument("--text_prompt", type=str, default="stuffed animal")
    parser.add_argument(
        "--no_object_mask",
        action="store_true",
        help="Disable object masks. This is mainly for debugging; the default experiment uses masks.",
    )
    parser.add_argument(
        "--max_points_per_camera",
        type=int,
        default=80_000,
        help="Deterministic per-camera sampling cap before fusing each row; use <=0 to disable.",
    )
    parser.add_argument(
        "--no_float_ffs_depth",
        action="store_true",
        help="Do not prefer depth_ffs_float_m when it is available.",
    )
    parser.add_argument(
        "--no_phystwin_like_postprocess",
        action="store_true",
        help="Disable the default display-only PhysTwin-like radius-neighbor cleanup.",
    )
    parser.add_argument("--phystwin_radius_m", type=float, default=0.01)
    parser.add_argument("--phystwin_nb_points", type=int, default=40)
    parser.add_argument(
        "--rounds",
        type=str,
        default="round1,round2,round3",
        help="Comma-separated subset of round1,round2,round3, or 'all'.",
    )
    return parser.parse_args()


def _select_round_specs(rounds_spec: str, all_specs: list[dict]) -> list[dict]:
    normalized = str(rounds_spec).strip().lower()
    if normalized in {"", "all"}:
        return list(all_specs)
    selected = {item.strip() for item in normalized.split(",") if item.strip()}
    specs = [item for item in all_specs if str(item["round_id"]) in selected]
    missing = sorted(selected - {str(item["round_id"]) for item in specs})
    if missing:
        raise ValueError(f"Unknown static round selection: {', '.join(missing)}")
    return specs


def main() -> int:
    args = parse_args()

    from data_process.visualization.experiments.ffs_mask_erode_multipage_sweep_pcd_compare import (
        run_ffs_mask_erode_multipage_sweep_pcd_workflow,
    )
    from data_process.visualization.experiments.native_ffs_fused_pcd_compare import (
        build_static_native_ffs_fused_pcd_round_specs,
    )

    all_specs = build_static_native_ffs_fused_pcd_round_specs(aligned_root=Path(args.aligned_root).resolve())
    round_specs = _select_round_specs(str(args.rounds), all_specs)
    max_points_per_camera = None if int(args.max_points_per_camera) <= 0 else int(args.max_points_per_camera)
    summary = run_ffs_mask_erode_multipage_sweep_pcd_workflow(
        aligned_root=Path(args.aligned_root).resolve(),
        output_root=Path(args.output_root).resolve(),
        frame_idx=int(args.frame_idx),
        depth_min_m=float(args.depth_min_m),
        depth_max_m=float(args.depth_max_m),
        point_size=float(args.point_size),
        look_distance=float(args.look_distance),
        tile_width=int(args.tile_width),
        tile_height=int(args.tile_height),
        row_label_width=int(args.row_label_width),
        max_points_per_camera=max_points_per_camera,
        text_prompt=str(args.text_prompt),
        use_object_mask=not bool(args.no_object_mask),
        phystwin_like_postprocess=not bool(args.no_phystwin_like_postprocess),
        phystwin_radius_m=float(args.phystwin_radius_m),
        phystwin_nb_points=int(args.phystwin_nb_points),
        use_float_ffs_depth_when_available=not bool(args.no_float_ffs_depth),
        round_specs=round_specs,
    )
    print(f"FFS multipage mask erode object PCD experiment written to {summary['output_dir']}")
    for round_summary in summary["rounds"]:
        print(f"{round_summary['round_id']}:")
        for page in round_summary["pages"]:
            print(f"  {page['page_id']}: {page['board_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
