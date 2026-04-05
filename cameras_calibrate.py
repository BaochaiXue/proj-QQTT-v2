from __future__ import annotations

import argparse

from qqtt.env.camera.defaults import (
    DEFAULT_FPS,
    DEFAULT_HEIGHT,
    DEFAULT_NUM_CAM,
    DEFAULT_WIDTH,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate a 3-camera RealSense setup with a ChArUco board.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--num-cam", type=int, default=DEFAULT_NUM_CAM)
    parser.add_argument(
        "--disable-keyboard-listener",
        action="store_true",
        help="Disable the optional keyboard listener.",
    )
    parser.add_argument(
        "--enable-keyboard-listener",
        action="store_true",
        help="Enable the keyboard listener during calibration.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    from qqtt.env import CameraSystem

    enable_keyboard_listener = args.enable_keyboard_listener
    if args.disable_keyboard_listener:
        enable_keyboard_listener = False

    camera_system = CameraSystem(
        WH=[args.width, args.height],
        fps=args.fps,
        num_cam=args.num_cam,
        enable_keyboard_listener=enable_keyboard_listener,
    )
    camera_system.calibrate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
