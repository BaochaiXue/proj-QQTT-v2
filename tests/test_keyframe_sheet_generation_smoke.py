from __future__ import annotations

import unittest

import numpy as np

from data_process.visualization.turntable_compare import compose_keyframe_sheet, compose_side_by_side_large


class KeyframeSheetGenerationSmokeTest(unittest.TestCase):
    def test_keyframe_sheet_builds_from_large_side_by_side_frames(self) -> None:
        native = np.full((180, 240, 3), 90, dtype=np.uint8)
        ffs = np.full((180, 240, 3), 130, dtype=np.uint8)
        overview = np.full((160, 300, 3), 50, dtype=np.uint8)

        board = compose_side_by_side_large(
            title_lines=["demo case", "geom | orbit=0.0 deg"],
            native_image=native,
            ffs_image=ffs,
            overview_inset=overview,
        )
        sheet = compose_keyframe_sheet([board, board, board, board], max_width=2400, max_height=2400)

        self.assertGreater(board.shape[1], native.shape[1] * 2 - 1)
        self.assertGreater(sheet.shape[0], board.shape[0] // 2)
        self.assertGreater(sheet.shape[1], board.shape[1] // 2)


if __name__ == "__main__":
    unittest.main()
