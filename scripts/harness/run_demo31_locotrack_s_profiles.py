#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "docs/generated/demo31_locotrack_s_rendered_profile"
DEMO31_ENTRYPOINT = ROOT / "demo_v3_1/realtime_three_view_cotracker3_realsense_overlay_dual4090.py"


def _csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in str(value).split(",") if part.strip())


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run rendered Demo 3.1 LocoTrack-S serial vs batch-views profiles.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--duration-s", type=float, default=60.0)
    parser.add_argument("--camera-ids", default="0,1,2")
    parser.add_argument("--mask-gpu", default="0")
    parser.add_argument("--cotracker-gpu", default="1")
    parser.add_argument("--calibrate-path", default="calibrate.pkl")
    parser.add_argument("--conda-env", default="demo_3_1_max")
    parser.add_argument("--python", default="python")
    parser.add_argument("--no-conda", action="store_true")
    parser.add_argument("--locotrack-repo-dir", default="external/locotrack/locotrack_pytorch")
    parser.add_argument("--locotrack-checkpoint", default="checkpoints/locotrack/locotrack_small.ckpt")
    parser.add_argument("--locotrack-model-size", default="small", choices=("small", "base"))
    parser.add_argument("--locotrack-query-chunk-size", type=int, default=256)
    parser.add_argument("--execution-modes", type=_csv_ints_or_modes, default=("serial", "batch-views"))
    parser.add_argument("--query-counts", type=_csv_ints, default=(512, 1024, 2048, 4096))
    parser.add_argument("--window-frames", type=_csv_ints, default=(4, 8, 12))
    parser.add_argument("--profile-limit", type=int, default=0, help="Run only the first N matrix entries; 0 means all.")
    parser.add_argument("--print-commands", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print commands and write the manifest without executing.")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser


def _csv_ints_or_modes(value: str) -> tuple[str, ...]:
    modes = tuple(part.strip() for part in str(value).split(",") if part.strip())
    valid = {"serial", "batch-views"}
    invalid = [mode for mode in modes if mode not in valid]
    if invalid:
        raise argparse.ArgumentTypeError(f"Unsupported execution mode(s): {invalid}")
    return modes


def _base_python_command(args: argparse.Namespace) -> list[str]:
    if bool(args.no_conda):
        return [str(args.python)]
    return ["conda", "run", "--no-capture-output", "-n", str(args.conda_env), str(args.python)]


def _profile_path(output_dir: Path, *, mode: str, query_count: int, window_frames: int, duration_s: float) -> Path:
    duration_tag = str(int(duration_s)) if float(duration_s).is_integer() else str(duration_s).replace(".", "p")
    return output_dir / f"{mode.replace('-', '_')}_q{int(query_count)}_w{int(window_frames)}_{duration_tag}s.json"


def build_profile_commands(args: argparse.Namespace) -> list[dict[str, object]]:
    commands: list[dict[str, object]] = []
    for mode in tuple(args.execution_modes):
        for query_count in tuple(args.query_counts):
            for window_frames in tuple(args.window_frames):
                profile_path = _profile_path(
                    Path(args.output_dir),
                    mode=str(mode),
                    query_count=int(query_count),
                    window_frames=int(window_frames),
                    duration_s=float(args.duration_s),
                )
                cmd = [
                    *_base_python_command(args),
                    str(DEMO31_ENTRYPOINT),
                    "--duration-s",
                    str(float(args.duration_s)),
                    "--camera-ids",
                    str(args.camera_ids),
                    "--mask-gpu",
                    str(args.mask_gpu),
                    "--cotracker-gpu",
                    str(args.cotracker_gpu),
                    "--require-two-cuda",
                    "--calibrate-path",
                    str(args.calibrate_path),
                    "--render-mode",
                    "pointcloud",
                    "--render-micro-profile",
                    "--gpu-sampling",
                    "--gpu-sampling-device-indexes",
                    "0,1",
                    "--cotracker-backend",
                    "locotrack",
                    "--tracking-backend-execution-mode",
                    str(mode),
                    "--tracker-batch-query-count-policy",
                    "fixed",
                    "--cotracker-query-count",
                    str(int(query_count)),
                    "--overlay-display-scope",
                    "controller",
                    "--overlay-max-points-per-camera",
                    "0",
                    "--wait-for-tracking-overlay",
                    "--locotrack-repo-dir",
                    str(args.locotrack_repo_dir),
                    "--locotrack-checkpoint",
                    str(args.locotrack_checkpoint),
                    "--locotrack-model-size",
                    str(args.locotrack_model_size),
                    "--locotrack-window-frames",
                    str(int(window_frames)),
                    "--locotrack-query-chunk-size",
                    str(int(args.locotrack_query_chunk_size)),
                    "--profile-json-output",
                    str(profile_path),
                ]
                commands.append(
                    {
                        "execution_mode": str(mode),
                        "query_count_per_camera": int(query_count),
                        "window_frames": int(window_frames),
                        "profile_json": str(profile_path),
                        "command": cmd,
                    }
                )
    if int(args.profile_limit) > 0:
        commands = commands[: int(args.profile_limit)]
    return commands


def _quote_command(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    commands = build_profile_commands(args)
    manifest = {
        "demo": "demo3.1",
        "backend": "locotrack",
        "model_size": str(args.locotrack_model_size),
        "output_dir": str(output_dir),
        "commands": commands,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.print_commands or args.dry_run:
        for entry in commands:
            print(_quote_command(entry["command"]))  # type: ignore[arg-type]
    if args.dry_run or args.print_commands:
        return 0

    env = dict(os.environ)
    env.setdefault("QQTT_WSLG_OPEN3D_FAST_EXIT", "1")
    failures: list[dict[str, object]] = []
    for entry in commands:
        profile_path = Path(str(entry["profile_json"]))
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        command = [str(part) for part in entry["command"]]  # type: ignore[index]
        print(f"[demo31-locotrack-profile] {profile_path}")
        completed = subprocess.run(command, cwd=str(ROOT), env=env, check=False)
        if completed.returncode != 0:
            failure = {
                "profile_json": str(profile_path),
                "returncode": int(completed.returncode),
                "command": command,
            }
            failures.append(failure)
            if not args.continue_on_error:
                (output_dir / "failures.json").write_text(
                    json.dumps(failures, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                return int(completed.returncode)
    if failures:
        (output_dir / "failures.json").write_text(json.dumps(failures, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
