from __future__ import annotations

import unittest

import numpy as np

from data_process.visualization.depth_colormap import colorize_depth_meters, colorize_depth_units
from data_process.visualization.depth_diagnostics import colorize_depth_map


class DepthColormapConsistencySmokeTest(unittest.TestCase):
    def test_shared_helper_matches_panel_colorizer(self) -> None:
        depth_m = np.array(
            [
                [0.0, 0.25, 0.50],
                [0.75, 1.00, np.nan],
                [1.25, 1.50, 2.00],
            ],
            dtype=np.float32,
        )
        expected = colorize_depth_map(depth_m, depth_min_m=0.1, depth_max_m=2.0)
        actual = colorize_depth_meters(depth_m, depth_min_m=0.1, depth_max_m=2.0)
        self.assertTrue(np.array_equal(actual, expected))

    def test_uint16_depth_units_path_matches_metric_path(self) -> None:
        depth_raw = np.array(
            [
                [0, 250, 500],
                [750, 1000, 0],
                [1250, 1500, 2000],
            ],
            dtype=np.uint16,
        )
        expected = colorize_depth_meters(depth_raw.astype(np.float32) * 0.001, depth_min_m=0.1, depth_max_m=2.0)
        actual = colorize_depth_units(
            depth_raw,
            depth_scale_m_per_unit=0.001,
            depth_min_m=0.1,
            depth_max_m=2.0,
        )
        self.assertTrue(np.array_equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
