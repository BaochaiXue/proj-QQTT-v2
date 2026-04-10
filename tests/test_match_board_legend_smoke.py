from __future__ import annotations

import unittest

import numpy as np

from data_process.visualization.source_compare import render_mismatch_residual
from data_process.visualization.support_compare import overlay_support_legend, render_support_count_map


class MatchBoardLegendSmokeTest(unittest.TestCase):
    def test_support_legend_overlay_changes_canvas(self) -> None:
        raw = render_support_count_map(
            np.array([[1, 2], [3, 0]], dtype=np.uint8),
            np.array([[True, True], [True, False]], dtype=bool),
        )
        canvas = np.tile(raw, (80, 120, 1))
        with_legend = overlay_support_legend(canvas)
        self.assertEqual(with_legend.shape, canvas.shape)
        self.assertGreater(int(np.sum(np.abs(with_legend.astype(np.int32) - canvas.astype(np.int32)))), 0)

    def test_mismatch_render_includes_visible_overlay(self) -> None:
        camera_clouds = [
            {
                "camera_idx": 0,
                "serial": "cam0",
                "points": np.array([[0.0, 0.0, 1.0], [0.02, 0.0, 1.0]], dtype=np.float32),
                "colors": np.full((2, 3), 160, dtype=np.uint8),
            },
            {
                "camera_idx": 1,
                "serial": "cam1",
                "points": np.array([[0.0, 0.0, 1.02], [0.02, 0.0, 1.01]], dtype=np.float32),
                "colors": np.full((2, 3), 180, dtype=np.uint8),
            },
        ]
        image, metrics = render_mismatch_residual(
            camera_clouds,
            view_config={
                "camera_position": np.array([0.0, 0.0, 0.0], dtype=np.float32),
                "center": np.array([0.0, 0.0, 1.0], dtype=np.float32),
                "up": np.array([0.0, 1.0, 0.0], dtype=np.float32),
            },
            width=320,
            height=240,
            projection_mode="perspective",
            ortho_scale=None,
        )
        self.assertGreater(int(image.sum()), 0)
        self.assertIn("summary", metrics)


if __name__ == "__main__":
    unittest.main()
