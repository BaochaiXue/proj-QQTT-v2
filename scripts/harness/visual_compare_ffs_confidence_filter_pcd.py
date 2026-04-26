from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_FFS_REPO = Path("/home/zhangxinjie/Fast-FoundationStereo")
DEFAULT_MODEL_PATH = DEFAULT_FFS_REPO / "weights" / "23-36-37" / "model_best_bp2_serialize.pth"
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "static" / "ffs_confidence_filter_object_pcd_6x3_frame_0000_threshold_0_10"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render static round1-3 frame-0 Open3D 6x3 PCD boards comparing native depth, "
            "raw PyTorch FFS, and four confidence-filtered FFS variants after object masking."
        )
    )
    parser.add_argument("--aligned_root", type=Path, default=ROOT / "data")
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--ffs_repo", type=Path, default=DEFAULT_FFS_REPO)
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--valid_iters", type=int, default=4)
    parser.add_argument("--max_disp", type=int, default=192)
    parser.add_argument("--frame_idx", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--depth_min_m", type=float, default=0.2)
    parser.add_argument("--depth_max_m", type=float, default=1.5)
    parser.add_argument("--point_size", type=float, default=2.0)
    parser.add_argument("--look_distance", type=float, default=1.0)
    parser.add_argument("--tile_width", type=int, default=480)
    parser.add_argument("--tile_height", type=int, default=360)
    parser.add_argument("--text_prompt", type=str, default="stuffed animal")
    parser.add_argument(
        "--no_object_mask",
        action="store_true",
        help="Disable the default static object mask and render the full scene.",
    )
    parser.add_argument(
        "--max_points_per_camera",
        type=int,
        default=80_000,
        help="Deterministic per-camera sampling cap before fusing each row; use <=0 to disable.",
    )
    parser.add_argument(
        "--phystwin_like_postprocess",
        action="store_true",
        help=(
            "After object masking and row fusion, render only points that pass the "
            "PhysTwin-like radius-neighbor filter."
        ),
    )
    parser.add_argument(
        "--phystwin_radius_m",
        type=float,
        default=0.01,
        help="Radius in meters for --phystwin_like_postprocess.",
    )
    parser.add_argument(
        "--phystwin_nb_points",
        type=int,
        default=40,
        help="Minimum neighbors required inside --phystwin_radius_m for --phystwin_like_postprocess.",
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

    from data_process.visualization.workflows.ffs_confidence_filter_pcd_compare import (
        build_static_confidence_filter_round_specs,
        run_ffs_confidence_filter_pcd_compare_workflow,
    )

    all_specs = build_static_confidence_filter_round_specs(aligned_root=Path(args.aligned_root).resolve())
    round_specs = _select_round_specs(str(args.rounds), all_specs)
    max_points_per_camera = None if int(args.max_points_per_camera) <= 0 else int(args.max_points_per_camera)
    summary = run_ffs_confidence_filter_pcd_compare_workflow(
        aligned_root=Path(args.aligned_root).resolve(),
        output_root=Path(args.output_root).resolve(),
        ffs_repo=Path(args.ffs_repo).resolve(),
        model_path=Path(args.model_path).resolve(),
        scale=float(args.scale),
        valid_iters=int(args.valid_iters),
        max_disp=int(args.max_disp),
        frame_idx=int(args.frame_idx),
        confidence_threshold=float(args.threshold),
        depth_min_m=float(args.depth_min_m),
        depth_max_m=float(args.depth_max_m),
        point_size=float(args.point_size),
        look_distance=float(args.look_distance),
        tile_width=int(args.tile_width),
        tile_height=int(args.tile_height),
        max_points_per_camera=max_points_per_camera,
        text_prompt=str(args.text_prompt),
        use_object_mask=not bool(args.no_object_mask),
        phystwin_like_postprocess=bool(args.phystwin_like_postprocess),
        phystwin_radius_m=float(args.phystwin_radius_m),
        phystwin_nb_points=int(args.phystwin_nb_points),
        round_specs=round_specs,
    )
    print(f"FFS confidence filter PCD comparison outputs written to {summary['output_dir']}")
    for round_summary in summary["rounds"]:
        print(f"{round_summary['round_id']}: {round_summary['board_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
