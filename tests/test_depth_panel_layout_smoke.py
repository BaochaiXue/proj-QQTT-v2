from __future__ import annotations

import unittest

import numpy as np

from data_process.visualization.layouts import compose_depth_review_board, overlay_scalar_colorbar


class DepthPanelLayoutSmokeTest(unittest.TestCase):
    def test_review_board_builds_title_metrics_and_rows(self) -> None:
        image = np.full((180, 260, 3), 90, dtype=np.uint8)
        image = overlay_scalar_colorbar(
            image,
            label="m",
            min_text="0.20",
            max_text="1.50",
            colormap=2,
        )
        board = compose_depth_review_board(
            title_lines=["Depth Review", "Cam0 | frame 0"],
            metric_lines=["Valid 80%", "Median diff 4.2 mm"],
            rows=[[image, image], [image, image]],
        )
        self.assertEqual(board.ndim, 3)
        self.assertGreater(board.shape[0], image.shape[0] * 2)
        self.assertGreater(board.shape[1], image.shape[1] * 2 - 40)
