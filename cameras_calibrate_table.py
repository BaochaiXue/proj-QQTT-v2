from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from qqtt.env.camera.calibration_boards import (
    DEFAULT_CALIBRATION_BOARD,
    available_calibration_boards,
    charuco_board_config_to_metadata,
    get_calibration_board_config,
)
from qqtt.env.camera.defaults import DEFAULT_EXPOSURE, DEFAULT_GAIN
from qqtt.env.camera.table_calibration import (
    _dist_coeffs_to_metadata,
    build_table_calibration_metadata,
    estimate_table_c2w_from_charuco_image,
    table_calibration_metadata_path_for,
    write_table_calibration_files,
)


TABLE_CALIBRATE_DEFAULT_WIDTH = 1280
TABLE_CALIBRATE_DEFAULT_HEIGHT = 720
TABLE_CALIBRATE_DEFAULT_FPS = 5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Strict one-shot single-camera table Z=0 calibration from one "
            "RealSense color frame and a ChArUco board."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--width", type=int, default=TABLE_CALIBRATE_DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=TABLE_CALIBRATE_DEFAULT_HEIGHT)
    parser.add_argument("--fps", type=int, default=TABLE_CALIBRATE_DEFAULT_FPS)
    parser.add_argument(
        "--serial",
        default=None,
        help="Optional RealSense serial number. Defaults to the first connected camera.",
    )
    parser.add_argument(
        "--exposure",
        type=float,
        default=DEFAULT_EXPOSURE,
        help="Manual RGB exposure.",
    )
    parser.add_argument(
        "--gain",
        type=float,
        default=DEFAULT_GAIN,
        help="Manual RGB gain.",
    )

    board_group = parser.add_argument_group("Calibration board")
    board_group.add_argument(
        "--calibration-board",
        choices=available_calibration_boards(),
        default=DEFAULT_CALIBRATION_BOARD,
        help="Named ChArUco board profile.",
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
        "--max-reprojection-error-px",
        type=float,
        default=0.20,
        help="Strict maximum accepted ChArUco reprojection error in pixels.",
    )
    parser.add_argument(
        "--min-corner-fraction",
        type=float,
        default=0.60,
        help="Strict minimum fraction of ChArUco chessboard corners to accept.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("table_calibrate.pkl"),
        help="Output table calibration pickle path.",
    )
    parser.add_argument(
        "--diagnostic-image",
        type=Path,
        default=Path("table_calibrate_diagnostic.png"),
        help="Output diagnostic image path.",
    )
    parser.add_argument(
        "--no-diagnostic-image",
        dest="diagnostic_image",
        action="store_const",
        const=None,
        help="Do not write a diagnostic image.",
    )
    parser.set_defaults(disable_keyboard_listener=True)
    parser.add_argument(
        "--disable-keyboard-listener",
        dest="disable_keyboard_listener",
        action="store_true",
        help="Disable the optional keyboard listener.",
    )
    parser.add_argument(
        "--enable-keyboard-listener",
        dest="disable_keyboard_listener",
        action="store_false",
        help="Enable the optional keyboard listener.",
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


def resolve_output_paths(
    *,
    output: Path,
    diagnostic_image: Path | None,
) -> tuple[Path, Path, Path | None]:
    output_path = Path(output)
    sidecar_path = table_calibration_metadata_path_for(output_path)
    diagnostic_path = None if diagnostic_image is None else Path(diagnostic_image)
    return output_path, sidecar_path, diagnostic_path


def validate_cli_args(args: argparse.Namespace) -> None:
    output = Path(args.output)
    if output.name == "calibrate.pkl":
        raise ValueError(
            "Refusing to overwrite calibrate.pkl; table calibration output must use "
            "table_calibrate.pkl or another non-legacy name."
        )
    if int(args.width) <= 0:
        raise ValueError("--width must be > 0")
    if int(args.height) <= 0:
        raise ValueError("--height must be > 0")
    if int(args.fps) <= 0:
        raise ValueError("--fps must be > 0")
    if (
        not math.isfinite(float(args.max_reprojection_error_px))
        or float(args.max_reprojection_error_px) <= 0.0
    ):
        raise ValueError("--max-reprojection-error-px must be > 0")
    if (
        not math.isfinite(float(args.min_corner_fraction))
        or float(args.min_corner_fraction) <= 0.0
        or float(args.min_corner_fraction) > 1.0
    ):
        raise ValueError("--min-corner-fraction must be in (0, 1]")


def _dist_coeffs_from_metadata(metadata: dict, key: str):
    coeffs = metadata.get(key)
    if coeffs is None:
        return None
    coeffs_array = np.asarray(coeffs, dtype=np.float64).reshape(-1, 1)
    if coeffs_array.size == 0:
        return None
    return coeffs_array


def run_table_calibration(args: argparse.Namespace) -> tuple[Path, Path]:
    validate_cli_args(args)
    board_config = resolve_board_config_from_args(args)
    output_path, _sidecar_path, diagnostic_path = resolve_output_paths(
        output=args.output,
        diagnostic_image=args.diagnostic_image,
    )

    camera_system = None
    try:
        from qqtt.env import CameraSystem

        serial_numbers = [args.serial] if args.serial else None
        camera_system = CameraSystem(
            WH=[args.width, args.height],
            fps=args.fps,
            num_cam=1,
            serial_numbers=serial_numbers,
            capture_mode="color",
            exposure=args.exposure,
            gain=args.gain,
            enable_keyboard_listener=not args.disable_keyboard_listener,
        )
        obs = camera_system.get_observation()
        color_bgr = obs[0]["color"]
        intrinsic = camera_system.realsense.get_intrinsics()[0]
        stream_metadata = list(getattr(camera_system, "stream_metadata", []))
        camera_metadata = stream_metadata[0] if stream_metadata else {}
        dist_coeffs = _dist_coeffs_from_metadata(
            camera_metadata,
            "color_distortion_coeffs",
        )
        distortion_model = camera_metadata.get("color_distortion_model")

        (
            c2w,
            reprojection_error_px,
            corner_count,
            corner_fraction,
            min_charuco_corners,
            diagnostic_bgr,
        ) = estimate_table_c2w_from_charuco_image(
            image_bgr=color_bgr,
            board_config=board_config,
            camera_matrix=intrinsic,
            dist_coeffs=dist_coeffs,
            max_reprojection_error_px=args.max_reprojection_error_px,
            min_corner_fraction=args.min_corner_fraction,
        )

        if diagnostic_path is not None:
            import cv2

            diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(diagnostic_path), diagnostic_bgr):
                raise OSError(f"Failed to write diagnostic image: {diagnostic_path}")

        selected_serial_numbers = list(getattr(camera_system, "serial_numbers", []))
        if not selected_serial_numbers:
            selected_serial_numbers = [args.serial] if args.serial else ["cam0"]
        metadata = build_table_calibration_metadata(
            serial_numbers=selected_serial_numbers,
            WH=[args.width, args.height],
            fps=args.fps,
            transform_count=1,
            calibration_board=charuco_board_config_to_metadata(board_config),
            max_reprojection_error_px=args.max_reprojection_error_px,
            min_corner_fraction=args.min_corner_fraction,
            min_charuco_corners=min_charuco_corners,
            per_camera_reprojection_error=[reprojection_error_px],
            per_camera_corner_count=[corner_count],
            per_camera_corner_fraction=[corner_fraction],
            distortion_used=dist_coeffs is not None,
            distortion_model_by_camera=[distortion_model],
            distortion_coeffs_by_camera=[_dist_coeffs_to_metadata(dist_coeffs)],
            diagnostic_image_path=(
                None if diagnostic_path is None else str(diagnostic_path)
            ),
        )
        sidecar_path = write_table_calibration_files(output_path, [c2w], metadata)
        print(f"[table_calibrate] Wrote {output_path}")
        print(f"[table_calibrate] Wrote {sidecar_path}")
        if diagnostic_path is not None:
            print(f"[table_calibrate] Wrote {diagnostic_path}")
        return output_path, sidecar_path
    finally:
        if camera_system is not None:
            try:
                camera_system.stop(wait=True)
            except TypeError:
                camera_system.stop()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run_table_calibration(args)
    except ValueError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
