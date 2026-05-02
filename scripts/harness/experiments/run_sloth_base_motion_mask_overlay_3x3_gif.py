#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_process.visualization.experiments.sam21_checkpoint_ladder_panel import (
    DEFAULT_EDGETAM_CHECKPOINT,
    DEFAULT_EDGETAM_ENV_NAME,
    DEFAULT_EDGETAM_MODEL_CFG,
    DEFAULT_EDGETAM_REPO,
    DEFAULT_SAM2_CHECKPOINT_CACHE,
)
from data_process.visualization.experiments.sam21_mask_overlay_panel import (
    SLOTH_BASE_MOTION_CASE_DIR,
    SLOTH_BASE_MOTION_CASE_KEY,
    SLOTH_BASE_MOTION_CASE_LABEL,
    SLOTH_BASE_MOTION_DOC_JSON,
    SLOTH_BASE_MOTION_DOC_MD,
    SLOTH_BASE_MOTION_OUTPUT_DIR,
    render_mask_overlay_3x3_gif,
    write_overlay_report,
)
from data_process.visualization.experiments.sam21_checkpoint_ladder_panel import write_json


SAM21_SPECS = (
    ("small", "SAM2.1 Small", "sam2.1_hiera_small.pt", "configs/sam2.1/sam2.1_hiera_s.yaml"),
    ("tiny", "SAM2.1 Tiny", "sam2.1_hiera_tiny.pt", "configs/sam2.1/sam2.1_hiera_t.yaml"),
)


def _run_logged(command: list[str], *, log_path: Path, cwd: Path = ROOT) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("TQDM_DISABLE", "1")
    with log_path.open("w", encoding="utf-8") as log_handle:
        try:
            subprocess.run(
                command,
                cwd=str(cwd),
                env=env,
                check=True,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError:
            log_handle.flush()
            tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:])
            print(f"[sloth-overlay] command failed; log tail from {log_path}:\n{tail}", flush=True)
            raise


def _load_records(paths: list[Path]) -> list[dict[str, Any]]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate SAM3.1, SAM2.1 Small/Tiny, and compiled EdgeTAM masks for "
            "sloth_base_motion_ffs, then render a 3x3 XOR overlay GIF."
        )
    )
    parser.add_argument("--case-dir", type=Path, default=SLOTH_BASE_MOTION_CASE_DIR)
    parser.add_argument("--output-dir", type=Path, default=SLOTH_BASE_MOTION_OUTPUT_DIR)
    parser.add_argument("--text-prompt", default="sloth")
    parser.add_argument(
        "--sam31-text-prompt",
        default="sloth,stuffed animal",
        help="Prompt aliases used for SAM3.1 generation and downstream mask-union loading.",
    )
    parser.add_argument("--frames", type=int)
    parser.add_argument("--all-frames", action="store_true", default=True)
    parser.add_argument("--gif-fps", type=int, default=6)
    parser.add_argument("--tile-width", type=int, default=320)
    parser.add_argument("--tile-height", type=int, default=180)
    parser.add_argument("--row-label-width", type=int, default=92)
    parser.add_argument("--sam31-env-name", default="FFS-SAM-RS")
    parser.add_argument("--sam21-env-name", default="SAM21-max")
    parser.add_argument("--sam21-checkpoint-cache", type=Path, default=DEFAULT_SAM2_CHECKPOINT_CACHE)
    parser.add_argument("--edgetam-env-name", default=DEFAULT_EDGETAM_ENV_NAME)
    parser.add_argument("--edgetam-repo", type=Path, default=DEFAULT_EDGETAM_REPO)
    parser.add_argument("--edgetam-checkpoint", type=Path, default=DEFAULT_EDGETAM_CHECKPOINT)
    parser.add_argument("--edgetam-model-cfg", default=DEFAULT_EDGETAM_MODEL_CFG)
    parser.add_argument("--edgetam-warmup-runs", type=int, default=5)
    parser.add_argument("--reuse-existing", action="store_true", help="Skip mask workers and only render/report.")
    parser.add_argument("--skip-sam31", action="store_true")
    parser.add_argument("--skip-sam21", action="store_true")
    parser.add_argument("--skip-edgetam", action="store_true")
    parser.add_argument("--overwrite", action="store_true", default=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    case_dir = (ROOT / args.case_dir).resolve() if not Path(args.case_dir).is_absolute() else Path(args.case_dir).resolve()
    output_dir = (ROOT / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir).resolve()
    sam31_mask_root = output_dir / "sam31_masks"
    timing_dir = output_dir / "timings"
    log_dir = output_dir / "logs"
    frames = None if bool(args.all_frames) else args.frames
    output_dir.mkdir(parents=True, exist_ok=True)

    commands: list[dict[str, Any]] = []
    if not bool(args.reuse_existing) and not bool(args.skip_sam31):
        command = [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            str(args.sam31_env_name),
            "python",
            str(ROOT / "scripts/harness/generate_sam31_masks.py"),
            "--case_root",
            str(case_dir),
            "--output_dir",
            str(sam31_mask_root),
            "--text_prompt",
            str(args.sam31_text_prompt),
            "--camera_ids",
            "0",
            "1",
            "2",
            "--source_mode",
            "frames",
            "--overwrite",
        ]
        log_path = log_dir / "sam31_generate.log"
        print("[sloth-overlay] generating SAM3.1 masks", flush=True)
        _run_logged(command, log_path=log_path)
        commands.append({"name": "sam31", "command": command, "log_path": str(log_path)})

    sam21_record_paths: list[Path] = []
    if not bool(args.reuse_existing) and not bool(args.skip_sam21):
        for checkpoint_key, checkpoint_label, checkpoint_file, config in SAM21_SPECS:
            checkpoint_path = Path(args.sam21_checkpoint_cache).expanduser().resolve() / checkpoint_file
            for camera_idx in (0, 1, 2):
                result_json = timing_dir / f"{SLOTH_BASE_MOTION_CASE_KEY}_cam{camera_idx}_{checkpoint_key}.json"
                command = [
                    "conda",
                    "run",
                    "--no-capture-output",
                    "-n",
                    str(args.sam21_env_name),
                    "python",
                    str(ROOT / "scripts/harness/experiments/run_sam21_checkpoint_ladder_3x5_gifs.py"),
                    "--worker",
                    "--case-key",
                    SLOTH_BASE_MOTION_CASE_KEY,
                    "--case-dir",
                    str(case_dir),
                    "--text-prompt",
                    str(args.sam31_text_prompt),
                    "--camera-idx",
                    str(camera_idx),
                    "--checkpoint-key",
                    checkpoint_key,
                    "--checkpoint-label",
                    checkpoint_label,
                    "--checkpoint",
                    str(checkpoint_path),
                    "--config",
                    config,
                    "--output-mask-root",
                    str(output_dir / "masks" / SLOTH_BASE_MOTION_CASE_KEY / checkpoint_key),
                    "--sam31-mask-root",
                    str(sam31_mask_root),
                    "--result-json",
                    str(result_json),
                    "--sam21-init-mode",
                    "mask",
                    "--all-frames",
                    "--overwrite",
                ]
                log_path = log_dir / f"sam21_{checkpoint_key}_cam{camera_idx}.log"
                print(f"[sloth-overlay] SAM2.1 {checkpoint_key} cam{camera_idx}", flush=True)
                _run_logged(command, log_path=log_path)
                commands.append({"name": f"sam21_{checkpoint_key}_cam{camera_idx}", "command": command, "log_path": str(log_path)})
                sam21_record_paths.append(result_json)
    else:
        for checkpoint_key, _label, _checkpoint_file, _config in SAM21_SPECS:
            for camera_idx in (0, 1, 2):
                sam21_record_paths.append(timing_dir / f"{SLOTH_BASE_MOTION_CASE_KEY}_cam{camera_idx}_{checkpoint_key}.json")

    edgetam_record_paths: list[Path] = []
    if not bool(args.reuse_existing) and not bool(args.skip_edgetam):
        for camera_idx in (0, 1, 2):
            result_json = timing_dir / f"{SLOTH_BASE_MOTION_CASE_KEY}_cam{camera_idx}_edgetam.json"
            command = [
                "conda",
                "run",
                "--no-capture-output",
                "-n",
                str(args.edgetam_env_name),
                "python",
                str(ROOT / "scripts/harness/experiments/run_edgetam_video_masks.py"),
                "--case-key",
                SLOTH_BASE_MOTION_CASE_KEY,
                "--case-dir",
                str(case_dir),
                "--text-prompt",
                str(args.sam31_text_prompt),
                "--camera-idx",
                str(camera_idx),
                "--output-mask-root",
                str(output_dir / "masks" / SLOTH_BASE_MOTION_CASE_KEY / "edgetam"),
                "--sam31-mask-root",
                str(sam31_mask_root),
                "--result-json",
                str(result_json),
                "--checkpoint",
                str(Path(args.edgetam_checkpoint).expanduser().resolve()),
                "--model-cfg",
                str(args.edgetam_model_cfg),
                "--compile-mode",
                "compile_image_encoder_no_pos_cache_patch",
                "--warmup-runs",
                str(int(args.edgetam_warmup_runs)),
                "--all-frames",
                "--overwrite",
            ]
            log_path = log_dir / f"edgetam_compiled_cam{camera_idx}.log"
            print(f"[sloth-overlay] EdgeTAM compiled cam{camera_idx}", flush=True)
            _run_logged(command, log_path=log_path, cwd=Path(args.edgetam_repo).expanduser().resolve())
            commands.append({"name": f"edgetam_cam{camera_idx}", "command": command, "log_path": str(log_path)})
            edgetam_record_paths.append(result_json)
    else:
        for camera_idx in (0, 1, 2):
            edgetam_record_paths.append(timing_dir / f"{SLOTH_BASE_MOTION_CASE_KEY}_cam{camera_idx}_edgetam.json")

    print("[sloth-overlay] rendering 3x3 GIF", flush=True)
    summary = render_mask_overlay_3x3_gif(
        root=ROOT,
        case_dir=case_dir,
        output_dir=output_dir,
        case_key=SLOTH_BASE_MOTION_CASE_KEY,
        case_label=SLOTH_BASE_MOTION_CASE_LABEL,
        text_prompt=str(args.sam31_text_prompt),
        frames=frames,
        gif_fps=int(args.gif_fps),
        tile_width=int(args.tile_width),
        tile_height=int(args.tile_height),
        row_label_width=int(args.row_label_width),
        sam31_mask_root=sam31_mask_root,
        output_name="sloth_base_motion_ffs_mask_overlay_3x3_small_tiny_edgetam_compiled",
        background_mode="black_union_rgb",
        color_overlap=False,
    )
    summary["sam21_timing_records"] = _load_records(sam21_record_paths)
    summary["edgetam_timing_records"] = _load_records(edgetam_record_paths)
    summary["commands"] = commands
    summary["output_dir"] = str(output_dir)
    summary["target_prompt"] = str(args.text_prompt)
    summary["sam31_text_prompt"] = str(args.sam31_text_prompt)

    json_path = ROOT / SLOTH_BASE_MOTION_DOC_JSON
    md_path = ROOT / SLOTH_BASE_MOTION_DOC_MD
    write_json(json_path, summary)
    write_overlay_report(md_path, summary)
    write_json(output_dir / "summary.json", {**summary, "docs": {"benchmark_md": str(md_path), "benchmark_json": str(json_path)}})

    print(f"[sloth-overlay] gif: {summary['gif_path']}", flush=True)
    print(f"[sloth-overlay] first frame: {summary['first_frame_path']}", flush=True)
    print(f"[sloth-overlay] report: {md_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
