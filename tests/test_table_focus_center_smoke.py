from __future__ import annotations

import unittest

import numpy as np

from data_process.visualization.pointcloud_compare import estimate_focus_point


class TableFocusCenterSmokeTest(unittest.TestCase):
    def test_estimates_focus_from_dense_table_band(self) -> None:
        table_points = np.array(
            [
                [-0.4, -0.3, 0.02],
                [-0.3, 0.2, 0.01],
                [0.1, -0.1, 0.03],
                [0.3, 0.3, 0.02],
                [0.0, 0.0, 0.015],
            ],
            dtype=np.float32,
        )
        background = np.array(
            [
                [1.5, 1.5, 1.2],
                [-1.2, 1.3, 1.0],
            ],
            dtype=np.float32,
        )
        focus = estimate_focus_point(
            [table_points, background],
            bounds_min=np.array([-1.2, -0.3, 0.01], dtype=np.float32),
            bounds_max=np.array([1.5, 1.5, 1.2], dtype=np.float32),
            focus_mode="table",
        )
        self.assertLess(abs(float(focus[0])), 0.25)
        self.assertLess(abs(float(focus[1])), 0.25)
        self.assertLess(abs(float(focus[2]) - 0.02), 0.05)


if __name__ == "__main__":
    unittest.main()
