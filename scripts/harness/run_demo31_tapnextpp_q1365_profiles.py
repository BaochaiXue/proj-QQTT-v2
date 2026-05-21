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
DEFAULT_OUTPUT_DIR = ROOT / "docs/generated/demo31_tapnextpp_rendered_profile"
DEMO31_ENTRYPOINT = ROOT / "demo_v3_1/realtime_three_view_cotracker3_realsense_overlay_dual4090.py"
SUMMARY_SCRIPT = ROOT / "scripts/harness/summarize_demo31_tapnextpp_profiles.py"


def _csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in str(value).split(",") if part.strip())


def _csv_modes(value: str) -> tuple[str, ...]:
    modes = tuple(part.strip() for part in str(value).split(",") if part.strip())
    valid = {"serial", "batch-views"}
    invalid = [mode for mode in modes if mode not in valid]
    if invalid:
        raise argparse.ArgumentTypeError(f"Unsupported execution mode(s): {invalid}")
    return modes


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run rendered Demo 3.1 TAPNext++ q1365/view serial vs batch-views profiles.",
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
    parser.add_argument("--tapnet-repo-dir", default="external/tapnet")
    parser.add_argument("--tapnextpp-checkpoint", default="checkpoints/tapnextpp/tapnextpp_ckpt.pt")
    parser.add_argument("--tapnextpp-image-size", default="256,256")
    parser.add_argument("--tapnextpp-autocast-dtype", default="fp16", choices=("fp16", "bf16", "fp32"))
    parser.add_argument("--execution-modes", type=_csv_modes, default=("serial", "batch-views"))
    parser.add_argument("--query-counts", type=_csv_ints, default=(1365,))
    parser.add_argument("--include-q4096-stress", action="store_true")
    parser.add_argument("--profile-limit", type=int, default=0, help="Run only the first N matrix entries; 0 means all.")
    parser.add_argument("--print-commands", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print commands and write the manifest without executing.")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--skip-summary", action="store_true")
    return parser


def _base_python_command(args: argparse.Namespace) -> list[str]:
    if bool(args.no_conda):
        return [str(args.python)]
    return ["conda", "run", "--no-capture-output", "-n", str(args.conda_env), str(args.python)]


def _duration_tag(duration_s: float) -> str:
    return str(int(duration_s)) if float(duration_s).is_integer() else str(duration_s).replace(".", "p")


def _profile_path(output_dir: Path, *, mode: str, query_count: int, duration_s: float) -> Path:
    return output_dir / f"{mode.replace('-', '_')}_q{int(query_count)}_live_{_duration_tag(duration_s)}s.json"


def build_profile_commands(args: argparse.Namespace) -> list[dict[str, object]]:
    query_counts = list(tuple(args.query_counts))
    if bool(args.include_q4096_stress) and 4096 not in query_counts:
        query_counts.append(4096)
    commands: list[dict[str, object]] = []
    for mode in tuple(args.execution_modes):
        for query_count in query_counts:
            profile_path = _profile_path(
                Path(args.output_dir),
                mode=str(mode),
                query_count=int(query_count),
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
                "tapnextpp",
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
                "--tapnet-repo-dir",
                str(args.tapnet_repo_dir),
                "--tapnextpp-checkpoint",
                str(args.tapnextpp_checkpoint),
                "--tapnextpp-image-size",
                str(args.tapnextpp_image_size),
                "--tapnextpp-autocast-dtype",
                str(args.tapnextpp_autocast_dtype),
                "--profile-json-output",
                str(profile_path),
            ]
            commands.append(
                {
                    "execution_mode": str(mode),
                    "query_count_per_camera": int(query_count),
                    "total_query_count_across_views": int(query_count) * 3,
                    "profile_json": str(profile_path),
                    "command": cmd,
                }
            )
    if int(args.profile_limit) > 0:
        commands = commands[: int(args.profile_limit)]
    return commands


def _quote_command(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def _summary_outputs(output_dir: Path, query_counts: Sequence[int]) -> tuple[Path, Path]:
    counts = {int(item) for item in query_counts}
    if counts == {1365}:
        return output_dir / "summary_q1365_live.json", output_dir / "summary_q1365_live.md"
    return output_dir / "summary_tapnextpp_live.json", output_dir / "summary_tapnextpp_live.md"


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    commands = build_profile_commands(args)
    manifest = {
        "demo": "demo3.1",
        "backend": "tapnextpp",
        "output_dir": str(output_dir),
        "q1365_view_note": "1365 points/view is approximately 4095 total points across three views.",
        "q4096_view_note": "4096 points/view is 12288 total points and is treated as a stress test.",
        "commands": commands,
    }
    (output_dir / "manifest_q1365.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.print_commands or args.dry_run:
        for entry in commands:
            print(_quote_command(entry["command"]))  # type: ignore[arg-type]
    if args.dry_run or args.print_commands:
        return 0

    env = dict(os.environ)
    env.setdefault("QQTT_WSLG_OPEN3D_FAST_EXIT", "1")
    failures: list[dict[str, object]] = []
    completed_profiles: list[Path] = []
    for entry in commands:
        profile_path = Path(str(entry["profile_json"]))
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        command = [str(part) for part in entry["command"]]  # type: ignore[index]
        print(f"[demo31-tapnextpp-profile] {profile_path}")
        completed = subprocess.run(command, cwd=str(ROOT), env=env, check=False)
        if completed.returncode != 0:
            failure = {
                "profile_json": str(profile_path),
                "returncode": int(completed.returncode),
                "command": command,
            }
            failures.append(failure)
            if not args.continue_on_error:
                (output_dir / "failures_q1365.json").write_text(
                    json.dumps(failures, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                return int(completed.returncode)
        elif profile_path.is_file():
            completed_profiles.append(profile_path)
    if failures:
        (output_dir / "failures_q1365.json").write_text(
            json.dumps(failures, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return 1
    if completed_profiles and not bool(args.skip_summary):
        output_json, output_md = _summary_outputs(
            output_dir,
            [int(entry["query_count_per_camera"]) for entry in commands],
        )
        summary_cmd = [
            *_base_python_command(args),
            str(SUMMARY_SCRIPT),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ]
        for profile in completed_profiles:
            summary_cmd.extend(["--profile-json", str(profile)])
        print(f"[demo31-tapnextpp-summary] {output_md}")
        completed = subprocess.run(summary_cmd, cwd=str(ROOT), env=env, check=False)
        return int(completed.returncode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
