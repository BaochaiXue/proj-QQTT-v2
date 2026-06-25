#!/usr/bin/env python3
"""Run online CMA zero-order initialization, then online first-order training."""

from argparse import ArgumentParser
from pathlib import Path
import subprocess
import sys


def add_optional_arg(command, name, value):
    if value is not None:
        command.extend([name, str(value)])


def main():
    parser = ArgumentParser()
    parser.add_argument("--base_path", type=str, default="data/different_types")
    parser.add_argument("--online_base_path", type=str, default="online_data")
    parser.add_argument("--online_dir", type=str, default=None)
    parser.add_argument("--case_name", type=str, required=True)
    parser.add_argument("--experiments_dir", type=str, default="experiments_online")
    parser.add_argument(
        "--zero_experiments_dir",
        type=str,
        default="experiments_online_cma",
    )
    parser.add_argument("--static_data_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--train_frame", type=int, default=None)
    parser.add_argument("--zero_iterations", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--zero_batch_size", type=int, default=None)
    parser.add_argument("--segment_len", type=int, default=35)
    parser.add_argument("--segment_stride", type=int, default=16)
    parser.add_argument("--poll_sec", type=float, default=1.0)
    parser.add_argument("--recent_window_count", type=int, default=8)
    parser.add_argument("--checkpoint_interval", type=int, default=None)
    parser.add_argument("--stop_when_finished", action="store_true")
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--zero_realtime_vis", action="store_true")
    parser.add_argument("--zero_realtime_vis_dir", type=str, default=None)
    parser.add_argument("--zero_realtime_vis_every", type=int, default=1)
    parser.add_argument("--no_zero_realtime_iteration_history", action="store_true")
    parser.add_argument("--realtime_vis", action="store_true")
    parser.add_argument("--realtime_vis_dir", type=str, default=None)
    parser.add_argument("--realtime_vis_every", type=int, default=1)
    parser.add_argument("--no_realtime_iteration_history", action="store_true")
    parser.add_argument("--no_sample_recent", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.zero_iterations <= 0:
        raise ValueError("--zero_iterations must be positive")

    zero_batch_size = (
        args.batch_size
        if args.zero_batch_size is None
        else int(args.zero_batch_size)
    )
    python = sys.executable
    repo_dir = Path(__file__).resolve().parent

    zero_command = [
        python,
        str(repo_dir / "optimize_online_cma.py"),
        "--base_path",
        args.base_path,
        "--online_base_path",
        args.online_base_path,
        "--case_name",
        args.case_name,
        "--experiments_dir",
        args.zero_experiments_dir,
        "--device",
        args.device,
        "--max_iter",
        str(args.zero_iterations),
        "--batch_size",
        str(zero_batch_size),
        "--segment_len",
        str(args.segment_len),
        "--segment_stride",
        str(args.segment_stride),
        "--poll_sec",
        str(args.poll_sec),
        "--recent_window_count",
        str(args.recent_window_count),
        "--seed",
        str(args.seed),
    ]
    add_optional_arg(zero_command, "--online_dir", args.online_dir)
    add_optional_arg(zero_command, "--static_data_path", args.static_data_path)
    add_optional_arg(zero_command, "--train_frame", args.train_frame)
    add_optional_arg(zero_command, "--realtime_vis_dir", args.zero_realtime_vis_dir)
    zero_command.extend(["--realtime_vis_every", str(args.zero_realtime_vis_every)])
    if args.zero_realtime_vis:
        zero_command.append("--realtime_vis")
    if args.no_zero_realtime_iteration_history:
        zero_command.append("--no_realtime_iteration_history")
    if args.no_sample_recent:
        zero_command.append("--no_sample_recent")

    print("[Zero-to-First] Starting online zero-order optimization")
    subprocess.run(zero_command, check=True, cwd=repo_dir)

    zero_experiments_dir = Path(args.zero_experiments_dir)
    if not zero_experiments_dir.is_absolute():
        zero_experiments_dir = repo_dir / zero_experiments_dir
    optimal_params_path = (
        zero_experiments_dir
        / args.case_name
        / "optimal_params.pkl"
    )
    if not optimal_params_path.exists():
        raise FileNotFoundError(
            "Online zero-order optimization completed without producing "
            f"{optimal_params_path}"
        )

    first_command = [
        python,
        str(repo_dir / "train_online_warp.py"),
        "--base_path",
        args.base_path,
        "--online_base_path",
        args.online_base_path,
        "--case_name",
        args.case_name,
        "--experiments_dir",
        args.experiments_dir,
        "--device",
        args.device,
        "--batch_size",
        str(args.batch_size),
        "--segment_len",
        str(args.segment_len),
        "--segment_stride",
        str(args.segment_stride),
        "--poll_sec",
        str(args.poll_sec),
        "--recent_window_count",
        str(args.recent_window_count),
        "--realtime_vis_every",
        str(args.realtime_vis_every),
        "--seed",
        str(args.seed),
        "--optimal_params_path",
        str(optimal_params_path),
    ]
    add_optional_arg(first_command, "--online_dir", args.online_dir)
    add_optional_arg(first_command, "--static_data_path", args.static_data_path)
    add_optional_arg(first_command, "--train_frame", args.train_frame)
    add_optional_arg(first_command, "--iterations", args.iterations)
    add_optional_arg(
        first_command, "--checkpoint_interval", args.checkpoint_interval
    )
    add_optional_arg(
        first_command, "--realtime_vis_dir", args.realtime_vis_dir
    )
    if args.stop_when_finished:
        first_command.append("--stop_when_finished")
    if args.save_video:
        first_command.append("--save_video")
    if args.realtime_vis:
        first_command.append("--realtime_vis")
    if args.no_realtime_iteration_history:
        first_command.append("--no_realtime_iteration_history")
    if args.no_sample_recent:
        first_command.append("--no_sample_recent")

    print(
        "[Zero-to-First] Starting online first-order training with "
        f"{optimal_params_path}"
    )
    subprocess.run(first_command, check=True, cwd=repo_dir)


if __name__ == "__main__":
    main()
