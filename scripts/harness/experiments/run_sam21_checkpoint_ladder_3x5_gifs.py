#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_process.visualization.experiments.sam21_checkpoint_ladder_panel import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SAM2_CHECKPOINT_CACHE,
    run_ladder_workflow,
    run_sam21_worker,
    run_sam21_stable_worker,
)


def _comma_list(value: str | None) -> list[str] | None:
    if value is None or not str(value).strip():
        return None
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate SAM3.1 vs SAM2.1 large/base+/small/tiny 3x5 time GIF panels "
            "for still-object and still-rope aligned cases."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint-cache", type=Path, default=DEFAULT_SAM2_CHECKPOINT_CACHE)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--gif-fps", type=int, default=6)
    parser.add_argument("--tile-width", type=int, default=260)
    parser.add_argument("--tile-height", type=int, default=180)
    parser.add_argument("--row-label-width", type=int, default=92)
    parser.add_argument("--depth-min-m", type=float, default=0.2)
    parser.add_argument("--depth-max-m", type=float, default=1.5)
    parser.add_argument("--max-points-per-camera", type=int)
    parser.add_argument("--max-points-per-render", type=int, default=80_000)
    parser.add_argument("--bbox-padding-px", type=int, default=0)
    parser.add_argument("--case-keys", help="Comma-separated subset of case keys for debugging.")
    parser.add_argument("--checkpoint-keys", help="Comma-separated subset of checkpoint keys for debugging.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-sam2", action="store_true", help="Reuse existing SAM2 masks/timings and only render/report.")
    parser.add_argument("--skip-render", action="store_true", help="Run SAM2 workers and reports without rendering GIFs.")
    parser.add_argument(
        "--stable-throughput",
        action="store_true",
        help=(
            "Benchmark one long-lived worker per checkpoint. Each case/camera job gets "
            "five no-output warmup propagations, then a no-output speed pass; mask "
            "collection is performed separately for panel rendering."
        ),
    )
    parser.add_argument("--stable-warmup-runs", type=int, default=5)
    parser.add_argument(
        "--stable-no-speed-step-marker",
        action="store_true",
        help=(
            "Do not call torch.compiler.cudagraph_mark_step_begin() during stable "
            "warmup/speed propagation. This is closer to upstream benchmark.py but "
            "can fail on the local CUDA Graph path."
        ),
    )
    parser.add_argument(
        "--no-download-missing-checkpoints",
        action="store_true",
        help="Fail if a SAM2.1 checkpoint is missing instead of downloading it.",
    )
    parser.add_argument("--no-first-frame-ply", action="store_true")
    parser.add_argument("--phystwin-radius-m", type=float, default=0.01)
    parser.add_argument("--phystwin-nb-points", type=int, default=40)
    parser.add_argument("--enhanced-component-voxel-size-m", type=float, default=0.01)
    parser.add_argument("--enhanced-keep-near-main-gap-m", type=float, default=0.0)

    worker = parser.add_argument_group("worker mode")
    worker.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    worker.add_argument("--case-key", help=argparse.SUPPRESS)
    worker.add_argument("--case-dir", type=Path, help=argparse.SUPPRESS)
    worker.add_argument("--text-prompt", help=argparse.SUPPRESS)
    worker.add_argument("--camera-idx", type=int, help=argparse.SUPPRESS)
    worker.add_argument("--checkpoint-key", help=argparse.SUPPRESS)
    worker.add_argument("--checkpoint-label", help=argparse.SUPPRESS)
    worker.add_argument("--checkpoint", type=Path, help=argparse.SUPPRESS)
    worker.add_argument("--config", help=argparse.SUPPRESS)
    worker.add_argument("--output-mask-root", type=Path, help=argparse.SUPPRESS)
    worker.add_argument("--result-json", type=Path, help=argparse.SUPPRESS)
    worker.add_argument("--stable-worker", action="store_true", help=argparse.SUPPRESS)
    worker.add_argument("--job-manifest", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.stable_worker:
        missing = [
            name
            for name in (
                "checkpoint_key",
                "checkpoint_label",
                "checkpoint",
                "config",
                "job_manifest",
                "result_json",
            )
            if getattr(args, name) is None
        ]
        if missing:
            raise ValueError(f"Missing stable worker arguments: {', '.join(missing)}")
        summary = run_sam21_stable_worker(
            checkpoint_key=str(args.checkpoint_key),
            checkpoint_label=str(args.checkpoint_label),
            checkpoint_path=Path(args.checkpoint),
            config=str(args.config),
            job_manifest=Path(args.job_manifest),
            result_json=Path(args.result_json),
            warmup_runs=int(args.stable_warmup_runs),
            speed_use_step_marker=not bool(args.stable_no_speed_step_marker),
            overwrite=bool(args.overwrite),
        )
        print(
            f"[sam21-stable-worker] {summary['checkpoint_key']}: "
            f"{summary['aggregate_ms_per_frame']:.2f} ms/frame "
            f"{summary['aggregate_fps']:.2f} FPS over {summary['total_timed_frames']} frames",
            flush=True,
        )
        return 0

    if args.worker:
        missing = [
            name
            for name in (
                "case_key",
                "case_dir",
                "text_prompt",
                "camera_idx",
                "checkpoint_key",
                "checkpoint_label",
                "checkpoint",
                "config",
                "output_mask_root",
                "result_json",
            )
            if getattr(args, name) is None
        ]
        if missing:
            raise ValueError(f"Missing worker arguments: {', '.join(missing)}")
        result = run_sam21_worker(
            case_key=str(args.case_key),
            case_dir=Path(args.case_dir),
            text_prompt=str(args.text_prompt),
            camera_idx=int(args.camera_idx),
            checkpoint_key=str(args.checkpoint_key),
            checkpoint_label=str(args.checkpoint_label),
            checkpoint_path=Path(args.checkpoint),
            config=str(args.config),
            output_mask_root=Path(args.output_mask_root),
            result_json=Path(args.result_json),
            frames=int(args.frames),
            bbox_padding_px=int(args.bbox_padding_px),
            overwrite=bool(args.overwrite),
        )
        print(
            f"[sam21-worker] {result['case_key']} cam{result['camera_idx']} "
            f"{result['checkpoint_key']}: {result['inference_ms_per_frame']:.2f} ms/frame",
            flush=True,
        )
        return 0

    summary = run_ladder_workflow(
        script_path=Path(__file__),
        root=ROOT,
        output_dir=Path(args.output_dir),
        checkpoint_cache=Path(args.checkpoint_cache),
        case_keys=_comma_list(args.case_keys),
        checkpoint_keys=_comma_list(args.checkpoint_keys),
        frames=int(args.frames),
        gif_fps=int(args.gif_fps),
        tile_width=int(args.tile_width),
        tile_height=int(args.tile_height),
        row_label_width=int(args.row_label_width),
        depth_min_m=float(args.depth_min_m),
        depth_max_m=float(args.depth_max_m),
        max_points_per_camera=args.max_points_per_camera,
        max_points_per_render=args.max_points_per_render,
        bbox_padding_px=int(args.bbox_padding_px),
        download_missing_checkpoints=not bool(args.no_download_missing_checkpoints),
        run_sam2=not bool(args.skip_sam2),
        render_gifs=not bool(args.skip_render),
        overwrite=bool(args.overwrite),
        stable_throughput=bool(args.stable_throughput),
        stable_warmup_runs=int(args.stable_warmup_runs),
        stable_speed_use_step_marker=not bool(args.stable_no_speed_step_marker),
        save_first_frame_ply=not bool(args.no_first_frame_ply),
        phystwin_radius_m=float(args.phystwin_radius_m),
        phystwin_nb_points=int(args.phystwin_nb_points),
        enhanced_component_voxel_size_m=float(args.enhanced_component_voxel_size_m),
        enhanced_keep_near_main_gap_m=float(args.enhanced_keep_near_main_gap_m),
    )
    print(f"[sam21] summary: {summary['output_dir']}", flush=True)
    docs = summary.get("docs", {})
    if docs:
        print(f"[sam21] report: {docs.get('benchmark_md')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
