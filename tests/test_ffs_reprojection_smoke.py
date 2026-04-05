from __future__ import annotations

import unittest

import numpy as np

from scripts.harness.ffs_geometry import (
    project_to_color,
    rasterize_nearest_depth,
    transform_points,
    unproject_ir_depth,
)


class FfsReprojectionSmokeTest(unittest.TestCase):
    def test_reprojects_point_into_expected_color_pixel(self) -> None:
        depth = np.zeros((2, 2), dtype=np.float32)
        depth[0, 0] = 2.0
        K = np.array(
            [
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        points_ir, valid_mask = unproject_ir_depth(depth, K)
        self.assertEqual(int(valid_mask.sum()), 1)

        T = np.eye(4, dtype=np.float32)
        T[0, 3] = 1.0
        points_color = transform_points(points_ir, T)
        uv, z = project_to_color(points_color, K)
        self.assertAlmostEqual(float(uv[0, 0]), 1.0)
        self.assertAlmostEqual(float(uv[0, 1]), 0.0)
        self.assertAlmostEqual(float(z[0]), 2.0)

    def test_z_buffer_keeps_nearest_depth(self) -> None:
        uv = np.array([[0.0, 0.0], [0.0, 0.0], [2.0, 1.0]], dtype=np.float32)
        z = np.array([2.0, 1.0, 3.0], dtype=np.float32)
        rasterized = rasterize_nearest_depth(uv, z, output_shape=(3, 3))

        self.assertEqual(float(rasterized[0, 0]), 1.0)
        self.assertEqual(float(rasterized[1, 2]), 3.0)


if __name__ == "__main__":
    unittest.main()
