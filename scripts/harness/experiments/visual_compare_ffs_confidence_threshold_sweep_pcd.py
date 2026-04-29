from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from data_process.depth_backends.ffs_defaults import (
    DEFAULT_FFS_MAX_DISP,
    DEFAULT_FFS_MODEL_PATH,
    DEFAULT_FFS_REPO,
    DEFAULT_FFS_SCALE,
    DEFAULT_FFS_VALID_ITERS,
)
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "experiments" / "ffs_confidence_threshold_sweep_object_pcd_frame_0000_erode1_phystwin"
DEFAULT_THRESHOLDS = "0.01,0.05,0.10,0.15,0.20,0.25,0.50"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render one experiment folder for static round1-3 frame-0 Open3D 6x3 PCD confidence "
            "threshold sweep boards. Rows compare native depth, original FFS, and four "
            "confidence-filtered FFS variants after object masking, 1px mask erosion, and "
            "PhysTwin-like radius-neighbor postprocess."
        )
    )
    parser.add_argument("--aligned_root", type=Path, default=ROOT / "data")
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--ffs_repo", type=Path, default=DEFAULT_FFS_REPO)
    parser.add_argument("--model_path", type=Path, default=DEFAULT_FFS_MODEL_PATH)
    parser.add_argument("--scale", type=float, default=DEFAULT_FFS_SCALE)
    parser.add_argument("--valid_iters", type=int, default=DEFAULT_FFS_VALID_ITERS)
    parser.add_argument("--max_disp", type=int, default=DEFAULT_FFS_MAX_DISP)
    parser.add_argument("--frame_idx", type=int, default=0)
    parser.add_argument("--thresholds", type=str, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--depth_min_m", type=float, default=0.2)
    parser.add_argument("--depth_max_m", type=float, default=1.5)
    parser.add_argument("--point_size", type=float, default=2.0)
    parser.add_argument("--look_distance", type=float, default=1.0)
    parser.add_argument("--tile_width", type=int, default=480)
    parser.add_argument("--tile_height", type=int, default=360)
    parser.add_argument("--text_prompt", type=str, default="stuffed animal")
    parser.add_argument(
        "--mask_erode_pixels",
        type=int,
        default=1,
        help="Erode the object mask inward by this many pixels before point-cloud generation.",
    )
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
        "--phystwin_radius_m",
        type=float,
        default=0.01,
        help="Radius in meters for the mandatory PhysTwin-like display postprocess.",
    )
    parser.add_argument(
        "--phystwin_nb_points",
        type=int,
        default=40,
        help="Minimum neighbors required inside --phystwin_radius_m for display postprocess.",
    )
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

    from data_process.visualization.experiments.ffs_confidence_filter_pcd_compare import (
        build_static_confidence_filter_round_specs,
    )
    from data_process.visualization.experiments.ffs_confidence_threshold_sweep_pcd_compare import (
        run_ffs_confidence_threshold_sweep_pcd_workflow,
    )

    all_specs = build_static_confidence_filter_round_specs(aligned_root=Path(args.aligned_root).resolve())
    round_specs = _select_round_specs(str(args.rounds), all_specs)
    max_points_per_camera = None if int(args.max_points_per_camera) <= 0 else int(args.max_points_per_camera)
    summary = run_ffs_confidence_threshold_sweep_pcd_workflow(
        aligned_root=Path(args.aligned_root).resolve(),
        output_root=Path(args.output_root).resolve(),
        ffs_repo=Path(args.ffs_repo).resolve(),
        model_path=Path(args.model_path).resolve(),
        scale=float(args.scale),
        valid_iters=int(args.valid_iters),
        max_disp=int(args.max_disp),
        frame_idx=int(args.frame_idx),
        thresholds=str(args.thresholds),
        depth_min_m=float(args.depth_min_m),
        depth_max_m=float(args.depth_max_m),
        point_size=float(args.point_size),
        look_distance=float(args.look_distance),
        tile_width=int(args.tile_width),
        tile_height=int(args.tile_height),
        max_points_per_camera=max_points_per_camera,
        text_prompt=str(args.text_prompt),
        use_object_mask=not bool(args.no_object_mask),
        mask_erode_pixels=int(args.mask_erode_pixels),
        phystwin_like_postprocess=True,
        phystwin_radius_m=float(args.phystwin_radius_m),
        phystwin_nb_points=int(args.phystwin_nb_points),
        round_specs=round_specs,
    )
    print(f"FFS confidence threshold sweep PCD experiment written to {summary['output_dir']}")
    for round_summary in summary["rounds"]:
        board_count = len(round_summary["threshold_summaries"])
        print(f"{round_summary['round_id']}: {board_count} boards under {round_summary['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
