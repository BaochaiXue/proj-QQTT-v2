from __future__ import annotations

import unittest

import cameras_calibrate
from qqtt.env.camera.calibration_boards import (
    DEFAULT_CALIBRATION_BOARD,
    LEGACY_CALIBRATION_BOARD,
    charuco_board_config_to_metadata,
    create_charuco_board,
    get_calibration_board_config,
)
from qqtt.env.camera.calibration_metadata import build_calibration_metadata


class CalibrationBoardProfilesTest(unittest.TestCase):
    def test_default_board_matches_calibio_lab_reference(self) -> None:
        config = get_calibration_board_config(DEFAULT_CALIBRATION_BOARD)

        self.assertEqual(config.name, "calibio-12x9-30mm")
        self.assertEqual(config.squares_x, 12)
        self.assertEqual(config.squares_y, 9)
        self.assertAlmostEqual(config.square_length_m, 0.030)
        self.assertAlmostEqual(config.marker_length_m, 0.022)
        self.assertEqual(config.dictionary_name, "DICT_5X5_250")
        self.assertFalse(config.deprecated)
        self.assertEqual(config.chessboard_corner_count, 88)

    def test_legacy_board_profile_remains_available_but_deprecated(self) -> None:
        config = get_calibration_board_config(LEGACY_CALIBRATION_BOARD)

        self.assertEqual(config.squares_x, 4)
        self.assertEqual(config.squares_y, 5)
        self.assertAlmostEqual(config.square_length_m, 0.050)
        self.assertAlmostEqual(config.marker_length_m, 0.037)
        self.assertEqual(config.dictionary_name, "DICT_4X4_50")
        self.assertTrue(config.deprecated)
        self.assertEqual(config.chessboard_corner_count, 12)

    def test_cli_defaults_to_new_board_and_supports_overrides(self) -> None:
        parser = cameras_calibrate.build_parser()
        default_args = parser.parse_args([])

        self.assertEqual(default_args.calibration_board, DEFAULT_CALIBRATION_BOARD)

        override_args = parser.parse_args(
            [
                "--calibration-board",
                LEGACY_CALIBRATION_BOARD,
                "--board-squares-x",
                "6",
                "--board-squares-y",
                "7",
                "--board-square-size-mm",
                "40",
                "--board-marker-size-mm",
                "30",
                "--board-dictionary",
                "DICT_5X5_250",
            ]
        )
        config = cameras_calibrate.resolve_board_config_from_args(override_args)

        self.assertEqual(config.name, f"{LEGACY_CALIBRATION_BOARD}+overrides")
        self.assertEqual(config.squares_x, 6)
        self.assertEqual(config.squares_y, 7)
        self.assertAlmostEqual(config.square_length_m, 0.040)
        self.assertAlmostEqual(config.marker_length_m, 0.030)
        self.assertEqual(config.dictionary_name, "DICT_5X5_250")
        self.assertFalse(config.deprecated)

    def test_cv2_board_builds_expected_corner_count(self) -> None:
        config = get_calibration_board_config(DEFAULT_CALIBRATION_BOARD)
        _, board = create_charuco_board(config)

        self.assertEqual(len(board.getChessboardCorners()), 88)

    def test_metadata_records_board_profile(self) -> None:
        config = get_calibration_board_config(DEFAULT_CALIBRATION_BOARD)
        metadata = build_calibration_metadata(
            serial_numbers=["cam0", "cam1", "cam2"],
            WH=(1280, 720),
            fps=5,
            transform_count=3,
            calibration_board=charuco_board_config_to_metadata(config),
        )

        self.assertEqual(
            metadata["calibration_board"]["name"],
            DEFAULT_CALIBRATION_BOARD,
        )
        self.assertEqual(metadata["calibration_board"]["dictionary_name"], "DICT_5X5_250")
        self.assertEqual(metadata["calibration_board"]["square_length_mm"], 30.0)
        self.assertEqual(metadata["calibration_board"]["marker_length_mm"], 22.0)


if __name__ == "__main__":
    unittest.main()
