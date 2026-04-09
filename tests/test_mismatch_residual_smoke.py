from __future__ import annotations

import unittest

import numpy as np

from data_process.visualization.source_compare import render_mismatch_residual


class MismatchResidualSmokeTest(unittest.TestCase):
    def test_mismatch_render_reports_nonzero_overlap_residual(self) -> None:
        camera_clouds = [
            {
                "camera_idx": 0,
                "serial": "serial-0",
                "points": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
                "colors": np.zeros((1, 3), dtype=np.uint8),
            },
            {
                "camera_idx": 1,
                "serial": "serial-1",
                "points": np.array([[0.0, 0.04, 0.0]], dtype=np.float32),
                "colors": np.zeros((1, 3), dtype=np.uint8),
            },
        ]
        image, residual = render_mismatch_residual(
            camera_clouds,
            view_config={
                "camera_position": np.array([0.0, -1.0, 0.0], dtype=np.float32),
                "center": np.array([0.0, 0.0, 0.0], dtype=np.float32),
                "up": np.array([0.0, 0.0, 1.0], dtype=np.float32),
            },
            width=240,
            height=180,
            projection_mode="perspective",
            ortho_scale=None,
        )

        self.assertEqual(image.shape, (180, 240, 3))
        self.assertGreater(residual["summary"]["overlap_pixel_count"], 0)
        self.assertGreater(residual["summary"]["residual_max_m"], 0.0)


if __name__ == "__main__":
    unittest.main()
