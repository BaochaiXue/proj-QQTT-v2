from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from shutil import copy2

from qqtt.env.camera.defaults import (
    DEFAULT_FPS,
    DEFAULT_HEIGHT,
    DEFAULT_NUM_CAM,
    DEFAULT_WIDTH,
)

_PROJECT_ROOT = next(
    (p for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents] if (p / ".git").exists()),
    Path(__file__).resolve().parent,
)


def _resolve_path(path: str) -> Path:
    return (_PROJECT_ROOT / path).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record synchronized multi-camera RealSense RGB-D data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--case_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=str(_resolve_path("./data_collect")))
    parser.add_argument(
        "--calibrate_path",
        type=str,
        default=str(_resolve_path("./calibrate.pkl")),
        help="Calibration file to copy into the recorded case if it exists.",
    )
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--num-cam", type=int, default=DEFAULT_NUM_CAM)
    parser.add_argument(
        "--disable-keyboard-listener",
        action="store_true",
        help="Disable the keyboard listener used for spacebar start/stop.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    from qqtt.env import CameraSystem

    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    case_name = args.case_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_root / case_name

    camera_system = CameraSystem(
        WH=[args.width, args.height],
        fps=args.fps,
        num_cam=args.num_cam,
        enable_keyboard_listener=not args.disable_keyboard_listener,
    )
    camera_system.record(output_path=str(output_path))

    calibrate_path = Path(args.calibrate_path).resolve()
    if calibrate_path.exists():
        copy2(calibrate_path, output_path / "calibrate.pkl")
    else:
        print(f"[record] warning: calibrate file not found, skipping copy: {calibrate_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
