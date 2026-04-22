from __future__ import annotations

import unittest

import numpy as np

from data_process.visualization.layouts import compose_registration_matrix_board
from data_process.visualization.source_compare import build_source_legend_image


class StereoOrderRegistrationLayoutSmokeTest(unittest.TestCase):
    def test_compose_registration_matrix_board_builds_clean_3x4_layout(self) -> None:
        panel = np.full((180, 240, 3), 96, dtype=np.uint8)
        board = compose_registration_matrix_board(
            title_lines=["demo | frame_idx=0 | stereo-order registration", "shared object ROI | colors encode source camera"],
            row_headers=["Native", "FFS-current", "FFS-swapped"],
            column_headers=["Oblique", "Top", "Front", "Side"],
            image_rows=[
                [panel, panel, panel, panel],
                [panel, panel, panel, panel],
                [panel, panel, panel, panel],
            ],
            legend_image=build_source_legend_image(),
        )
        self.assertEqual(board.ndim, 3)
        self.assertGreater(board.shape[0], 180 * 3)
        self.assertGreater(board.shape[1], 176 + 240 * 4)

    def test_compose_registration_matrix_board_builds_clean_4x3_layout(self) -> None:
        panel = np.full((160, 220, 3), 112, dtype=np.uint8)
        board = compose_registration_matrix_board(
            title_lines=["demo | frame_idx=0 | masked camera-view compare", "fixed original camera views | 4x3 compare"],
            row_headers=["Native", "Native + PS", "FFS", "FFS + PS"],
            column_headers=["Cam0", "Cam1", "Cam2"],
            image_rows=[
                [panel, panel, panel],
                [panel, panel, panel],
                [panel, panel, panel],
                [panel, panel, panel],
            ],
        )
        self.assertEqual(board.ndim, 3)
        self.assertGreater(board.shape[0], 160 * 4)
        self.assertGreater(board.shape[1], 176 + 220 * 3)


if __name__ == "__main__":
    unittest.main()
