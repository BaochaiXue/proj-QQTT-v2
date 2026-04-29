from __future__ import annotations

import unittest

import numpy as np

from data_process.visualization.experiments.still_object_projection_panel import (
    ROW_HEADERS_13,
    build_projected_pcd_rgb_tile,
    build_still_object_projection_board,
)


class StillObjectProjectionPanelSmokeTest(unittest.TestCase):
    def test_build_board_accepts_requested_13x3_matrix(self) -> None:
        image_rows = [[np.full((24, 32, 3), 80, dtype=np.uint8) for _ in range(3)] for _ in range(13)]
        board = build_still_object_projection_board(
            round_label="Still Object Round 1",
            frame_idx=0,
            model_config={
                "ffs_model_name": "20-30-48",
                "ffs_valid_iters": 4,
                "trt_builder_optimization_level": 5,
                "row_label_width": 160,
            },
            column_headers=["Cam0", "Cam1", "Cam2"],
            image_rows=image_rows,
        )
        self.assertEqual(len(ROW_HEADERS_13), 13)
        self.assertEqual(board.ndim, 3)
        self.assertGreater(board.shape[0], 13 * 24)
        self.assertGreater(board.shape[1], 3 * 32)

    def test_projected_pcd_rgb_tile_draws_visible_points_and_removed_highlight(self) -> None:
        color_image = np.full((32, 32, 3), 180, dtype=np.uint8)
        points = np.asarray([[0.0, 0.0, 1.0], [0.01, 0.0, 1.0]], dtype=np.float32)
        colors = np.asarray([[20, 40, 220], [20, 220, 40]], dtype=np.uint8)
        sources = np.asarray([0, 1], dtype=np.int32)
        K = np.asarray([[100.0, 0.0, 16.0], [0.0, 100.0, 16.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        tile = build_projected_pcd_rgb_tile(
            color_image=color_image,
            points_world=points,
            colors_bgr=colors,
            source_camera_idx=sources,
            K_color=K,
            c2w=np.eye(4, dtype=np.float32),
            label="projection",
            tile_size=(64, 64),
            removed_mask=np.asarray([False, True], dtype=bool),
            point_radius_px=1,
            removed_radius_px=2,
        )
        self.assertEqual(tile.shape, (64, 64, 3))
        self.assertGreater(int(np.max(tile[:, :, 0])), 200)


if __name__ == "__main__":
    unittest.main()

