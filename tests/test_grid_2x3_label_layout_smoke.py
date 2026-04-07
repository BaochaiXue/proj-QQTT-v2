from __future__ import annotations

import unittest

import numpy as np

from data_process.visualization.pointcloud_compare import compose_grid_2x3


class Grid2x3LabelLayoutSmokeTest(unittest.TestCase):
    def test_grid_2x3_includes_title_and_header_bands(self) -> None:
        panel = np.full((120, 160, 3), 80, dtype=np.uint8)
        grid = compose_grid_2x3(
            title="demo title",
            column_headers=["View from Cam0", "View from Cam1", "View from Cam2"],
            row_headers=["Native depth", "FFS depth"],
            native_images=[panel, panel, panel],
            ffs_images=[panel, panel, panel],
        )
        self.assertEqual(grid.shape[0], 42 + 38 + 120 * 2)
        self.assertEqual(grid.shape[1], 170 + 160 * 3)
        self.assertGreater(int(grid[:42].sum()), 0)
        self.assertGreater(int(grid[42:80].sum()), 0)
        self.assertGreater(int(grid[80:, :170].sum()), 0)


if __name__ == "__main__":
    unittest.main()
