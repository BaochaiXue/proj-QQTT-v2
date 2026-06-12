from __future__ import annotations

import argparse

from qqtt.env.camera.calibration_boards import (
    DEFAULT_CALIBRATION_BOARD,
    available_calibration_boards,
    get_calibration_board_config,
)
from qqtt.env.camera.defaults import DEFAULT_EXPOSURE, DEFAULT_GAIN, DEFAULT_NUM_CAM

CALIBRATE_DEFAULT_WIDTH = 1280
CALIBRATE_DEFAULT_HEIGHT = 720
CALIBRATE_DEFAULT_FPS = 5
CALIBRATION_WORLD_FRAME_CHOICES = ("opencv-board-native", "robopil-rx180")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate the single-camera RealSense branch with a ChArUco board. "
            "Pass --num-cam or --serials for explicit multi-camera calibration."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--width", type=int, default=CALIBRATE_DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=CALIBRATE_DEFAULT_HEIGHT)
    parser.add_argument("--fps", type=int, default=CALIBRATE_DEFAULT_FPS)
    parser.add_argument("--num-cam", type=int, default=DEFAULT_NUM_CAM)
    parser.add_argument(
        "--exposure",
        type=float,
        default=DEFAULT_EXPOSURE,
        help="Base manual RGB exposure. Known lab-rig serials use shared per-camera overrides.",
    )
    parser.add_argument(
        "--gain",
        type=float,
        default=DEFAULT_GAIN,
        help="Base manual RGB gain. Known lab-rig serials use shared per-camera overrides.",
    )
    board_group = parser.add_argument_group("Calibration board")
    board_group.add_argument(
        "--calibration-board",
        choices=available_calibration_boards(),
        default=DEFAULT_CALIBRATION_BOARD,
        help=(
            "Named ChArUco board profile. The legacy 4x5 board remains "
            "available for old rigs but is deprecated."
        ),
    )
    board_group.add_argument(
        "--board-squares-x",
        type=int,
        default=None,
        help="Override the ChArUco square count in the board X direction.",
    )
    board_group.add_argument(
        "--board-squares-y",
        type=int,
        default=None,
        help="Override the ChArUco square count in the board Y direction.",
    )
    board_group.add_argument(
        "--board-square-size-mm",
        type=float,
        default=None,
        help="Override the checker/square size in millimeters.",
    )
    board_group.add_argument(
        "--board-marker-size-mm",
        type=float,
        default=None,
        help="Override the ArUco marker size in millimeters.",
    )
    board_group.add_argument(
        "--board-dictionary",
        default=None,
        help="Override the cv2.aruco dictionary name, e.g. DICT_5X5_250.",
    )
    parser.add_argument(
        "--calibration-world-frame",
        choices=CALIBRATION_WORLD_FRAME_CHOICES,
        default="opencv-board-native",
        help=(
            "World-frame convention written into calibrate.pkl. "
            "The default keeps QQTT's OpenCV ChArUco board-native frame. "
            "robopil-rx180 matches the yfang/Robopil converted board frame."
        ),
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=1,
        help=(
            "Number of accepted calibration samples to collect. "
            "The best single sample is still written to preserve calibrate.pkl semantics."
        ),
    )
    parser.add_argument(
        "--serials",
        nargs="*",
        default=None,
        help="Optional explicit logical camera order for calibration.",
    )
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


def resolve_board_config_from_args(args: argparse.Namespace):
    return get_calibration_board_config(args.calibration_board).with_overrides(
        squares_x=args.board_squares_x,
        squares_y=args.board_squares_y,
        square_size_mm=args.board_square_size_mm,
        marker_size_mm=args.board_marker_size_mm,
        dictionary_name=args.board_dictionary,
    )


def build_camera_system_kwargs(args: argparse.Namespace) -> dict[str, object]:
    return {
        "WH": [args.width, args.height],
        "fps": args.fps,
        "num_cam": args.num_cam,
        "serial_numbers": args.serials if args.serials else None,
        "capture_mode": "color",
        "exposure": args.exposure,
        "gain": args.gain,
    }


def main() -> int:
    args = build_parser().parse_args()
    from qqtt.env import CameraSystem

    board_config = resolve_board_config_from_args(args)
    enable_keyboard_listener = args.enable_keyboard_listener
    if args.disable_keyboard_listener:
        enable_keyboard_listener = False

    camera_system = CameraSystem(
        **build_camera_system_kwargs(args),
        enable_keyboard_listener=enable_keyboard_listener,
    )
    camera_system.calibrate(
        board_config=board_config,
        world_frame_convention=args.calibration_world_frame,
        calibration_samples=args.calibration_samples,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
