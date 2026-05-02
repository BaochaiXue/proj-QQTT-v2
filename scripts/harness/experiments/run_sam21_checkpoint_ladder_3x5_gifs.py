#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_process.visualization.experiments.sam21_checkpoint_ladder_panel import (
    CASE_SET_DYNAMICS,
    CASE_SET_STILL_OBJECT_ROPE,
    DEFAULT_DYNAMICS_OUTPUT_DIR,
    DEFAULT_EDGETAM_CHECKPOINT,
    EDGETAM_COMPILE_EAGER,
    EDGETAM_COMPILE_IMAGE_ENCODER,
    EDGETAM_COMPILE_NO_POS_CACHE,
    DEFAULT_EDGETAM_ENV_NAME,
    DEFAULT_EDGETAM_MODEL_CFG,
    DEFAULT_EDGETAM_REPO,
    DEFAULT_FFS_ENV_NAME,
    DEFAULT_FFS_REPO,
    DEFAULT_FFS_TRT_TWO_STAGE_MODEL_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SAM2_CHECKPOINT_CACHE,
    SAM21_INIT_BOX,
    SAM21_INIT_MASK,
    run_ladder_workflow,
    run_edgetam_dynamics_round1_3x6_workflow,
    run_ffs_depth_cache_worker,
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
            "for aligned diagnostic cases."
        )
    )
    parser.add_argument(
        "--case-set",
        choices=(CASE_SET_STILL_OBJECT_ROPE, CASE_SET_DYNAMICS),
        default=CASE_SET_STILL_OBJECT_ROPE,
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--checkpoint-cache", type=Path, default=DEFAULT_SAM2_CHECKPOINT_CACHE)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--all-frames", action="store_true")
    parser.add_argument("--gif-fps", type=int, default=6)
    parser.add_argument("--tile-width", type=int, default=260)
    parser.add_argument("--tile-height", type=int, default=180)
    parser.add_argument("--row-label-width", type=int, default=92)
    parser.add_argument("--depth-min-m", type=float, default=0.2)
    parser.add_argument("--depth-max-m", type=float, default=1.5)
    parser.add_argument("--max-points-per-camera", type=int)
    parser.add_argument("--max-points-per-render", type=int, default=80_000)
    parser.add_argument("--bbox-padding-px", type=int, default=0)
    parser.add_argument("--sam21-init-mode", choices=(SAM21_INIT_BOX, SAM21_INIT_MASK))
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
    parser.add_argument("--skip-sam31-preflight", action="store_true")
    parser.add_argument("--sam31-env-name", default=DEFAULT_FFS_ENV_NAME)
    parser.add_argument("--skip-ffs-depth-cache", action="store_true")
    parser.add_argument("--ffs-env-name", default=DEFAULT_FFS_ENV_NAME)
    parser.add_argument("--ffs-repo", type=Path, default=DEFAULT_FFS_REPO)
    parser.add_argument("--ffs-trt-model-dir", type=Path, default=DEFAULT_FFS_TRT_TWO_STAGE_MODEL_DIR)
    parser.add_argument(
        "--edgetam-round1-3x6",
        action="store_true",
        help="Add EdgeTAM as a sixth column and render the ffs_dynamics_round1 3x6 panel.",
    )
    parser.add_argument("--skip-edgetam", action="store_true")
    parser.add_argument("--edgetam-env-name", default=DEFAULT_EDGETAM_ENV_NAME)
    parser.add_argument("--edgetam-repo", type=Path, default=DEFAULT_EDGETAM_REPO)
    parser.add_argument("--edgetam-checkpoint", default=str(DEFAULT_EDGETAM_CHECKPOINT))
    parser.add_argument("--edgetam-model-cfg", default=DEFAULT_EDGETAM_MODEL_CFG)
    parser.add_argument(
        "--edgetam-compile-mode",
        choices=(EDGETAM_COMPILE_EAGER, EDGETAM_COMPILE_IMAGE_ENCODER, EDGETAM_COMPILE_NO_POS_CACHE),
        default=EDGETAM_COMPILE_NO_POS_CACHE,
    )
    parser.add_argument("--edgetam-warmup-runs", type=int, default=5)
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
    worker.add_argument("--sam31-mask-root", type=Path, help=argparse.SUPPRESS)
    worker.add_argument("--result-json", type=Path, help=argparse.SUPPRESS)
    worker.add_argument("--stable-worker", action="store_true", help=argparse.SUPPRESS)
    worker.add_argument("--job-manifest", type=Path, help=argparse.SUPPRESS)
    worker.add_argument("--ffs-depth-worker", action="store_true", help=argparse.SUPPRESS)
    worker.add_argument("--depth-cache-root", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    selected_frames = None if bool(args.all_frames) else int(args.frames)
    selected_output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else ROOT
        / (
            DEFAULT_DYNAMICS_OUTPUT_DIR
            if (args.case_set == CASE_SET_DYNAMICS or bool(args.edgetam_round1_3x6))
            else DEFAULT_OUTPUT_DIR
        )
    )
    selected_init_mode = (
        str(args.sam21_init_mode)
        if args.sam21_init_mode is not None
        else (SAM21_INIT_MASK if args.case_set == CASE_SET_DYNAMICS else SAM21_INIT_BOX)
    )

    if args.ffs_depth_worker:
        missing = [
            name
            for name in ("case_key", "case_dir", "depth_cache_root")
            if getattr(args, name) is None
        ]
        if missing:
            raise ValueError(f"Missing FFS depth worker arguments: {', '.join(missing)}")
        summary = run_ffs_depth_cache_worker(
            case_key=str(args.case_key),
            case_dir=Path(args.case_dir),
            depth_cache_root=Path(args.depth_cache_root),
            frames=selected_frames,
            ffs_repo=Path(args.ffs_repo),
            ffs_trt_model_dir=Path(args.ffs_trt_model_dir),
            overwrite=bool(args.overwrite),
        )
        print(
            f"[ffs-depth-worker] {summary['case_key']}: {summary['depth_dir']}",
            flush=True,
        )
        return 0

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
            frames=selected_frames,
            sam31_mask_root=Path(args.sam31_mask_root) if args.sam31_mask_root is not None else None,
            sam21_init_mode=selected_init_mode,
            bbox_padding_px=int(args.bbox_padding_px),
            overwrite=bool(args.overwrite),
        )
        print(
            f"[sam21-worker] {result['case_key']} cam{result['camera_idx']} "
            f"{result['checkpoint_key']}: {result['inference_ms_per_frame']:.2f} ms/frame",
            flush=True,
        )
        return 0

    if args.edgetam_round1_3x6:
        summary = run_edgetam_dynamics_round1_3x6_workflow(
            root=ROOT,
            output_dir=selected_output_dir,
            edgetam_script_path=ROOT / "scripts/harness/experiments/run_edgetam_video_masks.py",
            frames=None if (bool(args.all_frames) or args.case_set == CASE_SET_DYNAMICS or bool(args.edgetam_round1_3x6)) else int(args.frames),
            gif_fps=int(args.gif_fps),
            tile_width=int(args.tile_width),
            tile_height=int(args.tile_height),
            row_label_width=int(args.row_label_width),
            depth_min_m=float(args.depth_min_m),
            depth_max_m=float(args.depth_max_m),
            max_points_per_camera=args.max_points_per_camera,
            max_points_per_render=args.max_points_per_render,
            ensure_sam31_masks=not bool(args.skip_sam31_preflight),
            sam31_env_name=str(args.sam31_env_name),
            ensure_ffs_depth_cache=not bool(args.skip_ffs_depth_cache),
            ffs_script_path=Path(__file__),
            ffs_env_name=str(args.ffs_env_name),
            ffs_repo=Path(args.ffs_repo),
            ffs_trt_model_dir=Path(args.ffs_trt_model_dir),
            run_edgetam=not bool(args.skip_edgetam),
            edgetam_env_name=str(args.edgetam_env_name),
            edgetam_repo=Path(args.edgetam_repo),
            edgetam_checkpoint=str(args.edgetam_checkpoint),
            edgetam_model_cfg=str(args.edgetam_model_cfg),
            edgetam_compile_mode=str(args.edgetam_compile_mode),
            edgetam_warmup_runs=int(args.edgetam_warmup_runs),
            overwrite_edgetam=bool(args.overwrite),
            save_first_frame_ply=not bool(args.no_first_frame_ply),
            phystwin_radius_m=float(args.phystwin_radius_m),
            phystwin_nb_points=int(args.phystwin_nb_points),
            enhanced_component_voxel_size_m=float(args.enhanced_component_voxel_size_m),
            enhanced_keep_near_main_gap_m=float(args.enhanced_keep_near_main_gap_m),
        )
        print(f"[edgetam-3x6] gif: {summary['gif_summary']['gif_path']}", flush=True)
        docs = summary.get("docs", {})
        if docs:
            print(f"[edgetam-3x6] report: {docs.get('benchmark_md')}", flush=True)
        return 0

    summary = run_ladder_workflow(
        script_path=Path(__file__),
        root=ROOT,
        output_dir=selected_output_dir,
        case_set=str(args.case_set),
        checkpoint_cache=Path(args.checkpoint_cache),
        case_keys=_comma_list(args.case_keys),
        checkpoint_keys=_comma_list(args.checkpoint_keys),
        frames=None if (bool(args.all_frames) or args.case_set == CASE_SET_DYNAMICS) else int(args.frames),
        gif_fps=int(args.gif_fps),
        tile_width=int(args.tile_width),
        tile_height=int(args.tile_height),
        row_label_width=int(args.row_label_width),
        depth_min_m=float(args.depth_min_m),
        depth_max_m=float(args.depth_max_m),
        max_points_per_camera=args.max_points_per_camera,
        max_points_per_render=args.max_points_per_render,
        bbox_padding_px=int(args.bbox_padding_px),
        sam21_init_mode=selected_init_mode,
        ensure_sam31_masks=(args.case_set == CASE_SET_DYNAMICS and not bool(args.skip_sam31_preflight)),
        sam31_env_name=str(args.sam31_env_name),
        ensure_ffs_depth_cache=(args.case_set == CASE_SET_DYNAMICS and not bool(args.skip_ffs_depth_cache)),
        ffs_env_name=str(args.ffs_env_name),
        ffs_repo=Path(args.ffs_repo),
        ffs_trt_model_dir=Path(args.ffs_trt_model_dir),
        download_missing_checkpoints=not bool(args.no_download_missing_checkpoints),
        run_sam2=not bool(args.skip_sam2),
        render_gifs=not bool(args.skip_render),
        overwrite=bool(args.overwrite),
        stable_throughput=bool(args.stable_throughput or args.case_set == CASE_SET_DYNAMICS),
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
