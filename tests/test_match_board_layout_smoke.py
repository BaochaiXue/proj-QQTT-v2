from __future__ import annotations

import unittest

import numpy as np

from data_process.visualization.layouts import compose_turntable_board


class MatchBoardLayoutSmokeTest(unittest.TestCase):
    def test_match_board_layout_builds_clean_2x3_matrix(self) -> None:
        panel = np.full((180, 260, 3), 90, dtype=np.uint8)
        board = compose_turntable_board(
            title_lines=["demo case | frame_idx=0 | 3-view point-cloud match", "angle=+12.0 deg | proj=perspective | crop=auto_object_bbox"],
            column_headers=["Source attribution", "Support count", "Mismatch residual"],
            row_headers=["Native", "FFS"],
            native_images=[panel, panel, panel],
            ffs_images=[panel, panel, panel],
            overview_inset=None,
        )
        self.assertEqual(board.ndim, 3)
        self.assertGreater(board.shape[0], 180 * 2)
        self.assertEqual(board.shape[1], 170 + 260 * 3)


if __name__ == "__main__":
    unittest.main()
