"""Qt-free half of the visual table calibration (estimate + save).

Kept import-light and PySide6-free so the .venv test suite can exercise the
success path with a synthetically rendered ChArUco board — the dialog in
``calibrate_dialog.py`` is only the camera loop + rendering around this.
Mirrors ``cameras_calibrate_table.py``: same strict thresholds, same qqtt
estimation and writer functions, same output file set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Mirror cameras_calibrate_table.py defaults exactly: same stream profile,
# same strict acceptance thresholds.
CALIBRATE_WIDTH = 1280
CALIBRATE_HEIGHT = 720
CALIBRATE_FPS = 5
MAX_REPROJECTION_ERROR_PX = 0.20
MIN_CORNER_FRACTION = 0.60


@dataclass
class FrameEstimate:
    """One frame's strict calibration estimate (ok or the failure reason)."""

    display_bgr: np.ndarray
    ok: bool
    message: str = ""
    c2w: np.ndarray | None = None
    reprojection_error_px: float = 0.0
    corner_count: int = 0
    corner_fraction: float = 0.0
    min_charuco_corners: int = 0
    diagnostic_bgr: np.ndarray | None = None
    camera_matrix: np.ndarray | None = None
    dist_coeffs: Any = None
    distortion_model: Any = None
    serial_numbers: list[str] = field(default_factory=list)


def estimate_frame(
    color_bgr: np.ndarray,
    *,
    board_config,
    camera_matrix: np.ndarray,
    dist_coeffs=None,
) -> FrameEstimate:
    """Run the CLI tool's strict estimation on one frame, never raising."""
    from qqtt.env.camera.table_calibration import (  # noqa: PLC0415
        estimate_table_c2w_from_charuco_image,
    )

    try:
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
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            max_reprojection_error_px=MAX_REPROJECTION_ERROR_PX,
            min_corner_fraction=MIN_CORNER_FRACTION,
        )
    except Exception as exc:  # noqa: BLE001 — strict tool raises many kinds
        return FrameEstimate(
            display_bgr=color_bgr, ok=False, message=str(exc)
        )
    return FrameEstimate(
        display_bgr=diagnostic_bgr,
        ok=True,
        c2w=c2w,
        reprojection_error_px=float(reprojection_error_px),
        corner_count=int(corner_count),
        corner_fraction=float(corner_fraction),
        min_charuco_corners=int(min_charuco_corners),
        diagnostic_bgr=diagnostic_bgr,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
    )


def save_estimate(
    estimate: FrameEstimate,
    *,
    board_config,
    output_path: Path,
    diagnostic_path: Path | None,
) -> Path:
    """Write the CLI-identical file set from a successful FrameEstimate."""
    from qqtt.env.camera.calibration_boards import (  # noqa: PLC0415
        charuco_board_config_to_metadata,
    )
    from qqtt.env.camera.table_calibration import (  # noqa: PLC0415
        _dist_coeffs_to_metadata,
        build_table_calibration_metadata,
        write_table_calibration_files,
    )

    assert estimate.ok and estimate.c2w is not None
    if diagnostic_path is not None and estimate.diagnostic_bgr is not None:
        import cv2  # noqa: PLC0415

        diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(diagnostic_path), estimate.diagnostic_bgr):
            raise OSError(f"Failed to write diagnostic image: {diagnostic_path}")
    metadata = build_table_calibration_metadata(
        serial_numbers=estimate.serial_numbers or ["cam0"],
        WH=[CALIBRATE_WIDTH, CALIBRATE_HEIGHT],
        fps=CALIBRATE_FPS,
        transform_count=1,
        calibration_board=charuco_board_config_to_metadata(board_config),
        max_reprojection_error_px=MAX_REPROJECTION_ERROR_PX,
        min_corner_fraction=MIN_CORNER_FRACTION,
        min_charuco_corners=estimate.min_charuco_corners,
        per_camera_reprojection_error=[estimate.reprojection_error_px],
        per_camera_corner_count=[estimate.corner_count],
        per_camera_corner_fraction=[estimate.corner_fraction],
        distortion_used=estimate.dist_coeffs is not None,
        distortion_model_by_camera=[estimate.distortion_model],
        distortion_coeffs_by_camera=[
            _dist_coeffs_to_metadata(estimate.dist_coeffs)
        ],
        diagnostic_image_path=(
            None if diagnostic_path is None else str(diagnostic_path)
        ),
    )
    write_table_calibration_files(output_path, [estimate.c2w], metadata)
    return output_path
