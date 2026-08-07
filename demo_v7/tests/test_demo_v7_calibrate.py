"""Hardware-free tests for the visual calibration core (calibrate_core).

A ChArUco board rendered by OpenCV stands in for the camera frame, so the
strict estimation success path and the CLI-identical save path both run in
the plain .venv suite (the Qt dialog around them is smoke-tested offscreen
in the GUI env).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from demo_v7.gui.calibrate_core import (
    CALIBRATE_HEIGHT,
    CALIBRATE_WIDTH,
    estimate_frame,
    save_estimate,
)

cv2 = pytest.importorskip("cv2")


@pytest.fixture(scope="module")
def board_config():
    from qqtt.env.camera.calibration_boards import (
        DEFAULT_CALIBRATION_BOARD,
        get_calibration_board_config,
    )

    return get_calibration_board_config(DEFAULT_CALIBRATION_BOARD)


@pytest.fixture(scope="module")
def board_frame(board_config) -> np.ndarray:
    """A frontal synthetic camera frame showing the full board."""
    from qqtt.env.camera.calibration_boards import create_charuco_board

    _dictionary, board = create_charuco_board(board_config)
    # 3x supersample + area downscale + slight optical blur: the crisp
    # binary rendering alone biases subpixel corner refinement to ~0.47px,
    # over the strict 0.2px gate; camera-like soft edges land at ~0.14px.
    scale = 3
    board_h = (CALIBRATE_HEIGHT - 40) * scale
    board_w = board_h * board_config.squares_x // board_config.squares_y
    board_img = board.generateImage((board_w, board_h))
    canvas = np.full(
        (CALIBRATE_HEIGHT * scale, CALIBRATE_WIDTH * scale), 180, np.uint8
    )
    y0 = (CALIBRATE_HEIGHT * scale - board_h) // 2
    x0 = (CALIBRATE_WIDTH * scale - board_w) // 2
    canvas[y0 : y0 + board_h, x0 : x0 + board_w] = board_img
    frame = cv2.resize(
        canvas,
        (CALIBRATE_WIDTH, CALIBRATE_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )
    frame = cv2.GaussianBlur(frame, (0, 0), 1.2)
    return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)


K = np.array(
    [
        [900.0, 0.0, CALIBRATE_WIDTH / 2],
        [0.0, 900.0, CALIBRATE_HEIGHT / 2],
        [0.0, 0.0, 1.0],
    ]
)


class TestEstimateFrame:
    def test_synthetic_board_passes_strict_estimate(
        self, board_config, board_frame
    ) -> None:
        estimate = estimate_frame(
            board_frame, board_config=board_config, camera_matrix=K
        )
        assert estimate.ok, estimate.message
        assert estimate.c2w is not None and estimate.c2w.shape == (4, 4)
        assert estimate.reprojection_error_px < 0.2
        assert estimate.corner_fraction >= 0.6
        assert estimate.diagnostic_bgr is not None  # overlay for the GUI view

    def test_empty_frame_fails_without_raising(self, board_config) -> None:
        blank = np.full((CALIBRATE_HEIGHT, CALIBRATE_WIDTH, 3), 128, np.uint8)
        estimate = estimate_frame(
            blank, board_config=board_config, camera_matrix=K
        )
        assert not estimate.ok
        assert estimate.message  # the strict tool's reason, shown in the GUI
        assert estimate.c2w is None


class TestSaveEstimate:
    def test_saved_files_load_via_runtime_loader(
        self, board_config, board_frame, tmp_path
    ) -> None:
        from qqtt.env.camera.table_calibration import (
            load_table_calibration_transforms,
        )

        estimate = estimate_frame(
            board_frame, board_config=board_config, camera_matrix=K
        )
        assert estimate.ok
        estimate.serial_numbers = ["synthetic-cam"]
        out = tmp_path / "table_calibrate.pkl"
        diag = tmp_path / "table_calibrate_diagnostic.png"
        save_estimate(
            estimate,
            board_config=board_config,
            output_path=out,
            diagnostic_path=diag,
        )
        assert diag.is_file()
        sidecar = json.loads(
            (tmp_path / "table_calibrate_metadata.json").read_text()
        )
        assert sidecar["table_calibration_reference_serials"] == [
            "synthetic-cam"
        ]
        # The exact loader the v6.2 runtime uses accepts the file set,
        # including the per-serial lookup path.
        transforms = load_table_calibration_transforms(
            out, serial_numbers=["synthetic-cam"]
        )
        assert np.allclose(transforms[0], estimate.c2w)
