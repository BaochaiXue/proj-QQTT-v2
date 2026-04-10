from __future__ import annotations

import unittest

import numpy as np

from data_process.visualization.face_quality import build_face_metric_tile, compose_face_quality_board


class FaceQualityBoardLayoutTest(unittest.TestCase):
    def test_face_quality_board_builds_readable_rows(self) -> None:
        panel = np.full((120, 160, 3), 90, dtype=np.uint8)
        row = [
            build_face_metric_tile(panel, label="RGB | box_face", metrics=None),
            build_face_metric_tile(panel, label="Native residual", metrics={"valid_depth_ratio": 0.9, "plane_fit_rmse_mm": 1.2, "mad_mm": 0.8, "p90_abs_residual_mm": 2.1}),
            build_face_metric_tile(panel, label="FFS residual", metrics={"valid_depth_ratio": 0.95, "plane_fit_rmse_mm": 0.9, "mad_mm": 0.6, "p90_abs_residual_mm": 1.8}),
            build_face_metric_tile(panel, label="FFS-swapped residual", metrics={"valid_depth_ratio": 0.7, "plane_fit_rmse_mm": 3.0, "mad_mm": 2.2, "p90_abs_residual_mm": 5.2}),
        ]
        board = compose_face_quality_board(
            title_lines=["demo | frame_idx=0 | face smoothness", "Rows = patches | Columns = RGB / Native / FFS / FFS-swapped"],
            metric_lines=["Lower plane-fit RMSE / MAD / p90 means smoother face patches."],
            patch_rows=[row, row],
        )
        self.assertEqual(board.ndim, 3)
        self.assertGreater(board.shape[0], 0)
        self.assertGreater(board.shape[1], 0)


if __name__ == "__main__":
    unittest.main()
