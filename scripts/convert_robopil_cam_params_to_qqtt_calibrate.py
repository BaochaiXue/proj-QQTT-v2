from __future__ import annotations

import argparse
import pickle
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qqtt.env.camera.calibration_boards import (
    DEFAULT_CALIBRATION_BOARD,
    available_calibration_boards,
    charuco_board_config_to_metadata,
    get_calibration_board_config,
)
from qqtt.env.camera.calibration_metadata import (
    build_calibration_metadata,
    write_calibration_metadata,
)


CALIBRATION_WORLD_FRAME_OPENCV_BOARD_NATIVE = "opencv-board-native"
CALIBRATION_WORLD_FRAME_ROBOPIL_RX180 = "robopil-rx180"
WORLD_FRAME_CHOICES = (
    CALIBRATION_WORLD_FRAME_OPENCV_BOARD_NATIVE,
    CALIBRATION_WORLD_FRAME_ROBOPIL_RX180,
)


def load_robopil_cam_params(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        params = pickle.load(handle)
    if not isinstance(params, dict):
        raise ValueError("Robopil cam_params.pkl must contain a dict keyed by serial.")
    return {str(serial): value for serial, value in params.items()}


def convert_cam_params_to_c2ws(
    cam_params: dict[str, Any],
    serials: list[str],
) -> list[np.ndarray]:
    c2ws: list[np.ndarray] = []
    for serial in serials:
        if serial not in cam_params:
            raise ValueError(f"Serial {serial!r} is missing from Robopil cam_params.")
        entry = cam_params[serial]
        if not isinstance(entry, dict) or "extrinsic" not in entry:
            raise ValueError(f"Robopil entry for serial {serial!r} must contain 'extrinsic'.")
        board_to_camera = np.asarray(entry["extrinsic"], dtype=np.float64)
        if board_to_camera.shape != (4, 4):
            raise ValueError(
                f"Robopil extrinsic for serial {serial!r} must be 4x4, "
                f"got {board_to_camera.shape}."
            )
        c2ws.append(np.linalg.inv(board_to_camera))
    return c2ws


def write_qqtt_calibration_from_robopil(
    *,
    cam_params_path: str | Path,
    output_calibrate_path: str | Path,
    serials: list[str] | None = None,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    calibration_board: str = DEFAULT_CALIBRATION_BOARD,
    world_frame_convention: str = CALIBRATION_WORLD_FRAME_ROBOPIL_RX180,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    input_path = Path(cam_params_path)
    output_path = Path(output_calibrate_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")

    cam_params = load_robopil_cam_params(input_path)
    if serials is None:
        serials = sorted(str(serial) for serial in cam_params)
    serials = list(serials)
    c2ws = convert_cam_params_to_c2ws(cam_params, serials)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(c2ws, handle)

    board_config = get_calibration_board_config(calibration_board)
    metadata = build_calibration_metadata(
        serial_numbers=serials,
        WH=(int(width), int(height)),
        fps=int(fps),
        transform_count=len(c2ws),
        calibration_board=charuco_board_config_to_metadata(board_config),
        world_frame_convention=world_frame_convention,
        distortion_used=False,
        source_format="robopil_cam_params",
        source_path=str(input_path),
    )
    metadata["input_transform_convention"] = "board_to_camera_w2c_extrinsic"
    metadata["output_transform_convention"] = "camera_to_world_c2w"
    metadata["compatibility_contract"] = "qqtt_calibrate_pkl_c2w_list_v1"
    sidecar_path = write_calibration_metadata(output_path, metadata)
    return output_path, sidecar_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert yfang/Robopil cam_params.pkl into QQTT-compatible "
            "calibrate.pkl plus calibrate_metadata.json."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Path to Robopil cam_params.pkl.")
    parser.add_argument("--output", default="calibrate.pkl", help="Output QQTT calibrate.pkl path.")
    parser.add_argument(
        "--serials",
        nargs="+",
        default=None,
        help="Serial order for the QQTT c2w list. Defaults to sorted Robopil dict keys.",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--calibration-board",
        choices=available_calibration_boards(),
        default=DEFAULT_CALIBRATION_BOARD,
    )
    parser.add_argument(
        "--world-frame-convention",
        choices=WORLD_FRAME_CHOICES,
        default=CALIBRATION_WORLD_FRAME_ROBOPIL_RX180,
        help=(
            "World frame represented by the input Robopil extrinsics. "
            "The yfang script applies the Robopil Rx180 convention."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_path, sidecar_path = write_qqtt_calibration_from_robopil(
        cam_params_path=args.input,
        output_calibrate_path=args.output,
        serials=args.serials,
        width=args.width,
        height=args.height,
        fps=args.fps,
        calibration_board=args.calibration_board,
        world_frame_convention=args.world_frame_convention,
        overwrite=args.overwrite,
    )
    print(f"Wrote {output_path} and {sidecar_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
