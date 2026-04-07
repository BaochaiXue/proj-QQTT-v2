from __future__ import annotations

import unittest

import numpy as np

from data_process.visualization.pointcloud_compare import render_point_cloud_fallback


class ProjectionModeSmokeTest(unittest.TestCase):
    def test_orthographic_and_perspective_both_render(self) -> None:
        points = np.array(
            [
                [-0.2, -0.1, 0.8],
                [0.0, 0.0, 1.0],
                [0.2, 0.1, 1.2],
            ],
            dtype=np.float32,
        )
        colors = np.array(
            [
                [255, 0, 0],
                [0, 255, 0],
                [0, 0, 255],
            ],
            dtype=np.uint8,
        )
        view_config = {
            "center": np.array([0.0, 0.0, 1.0], dtype=np.float32),
            "camera_position": np.array([0.0, -2.0, 1.0], dtype=np.float32),
            "up": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        }
        scalar_bounds = {"height": (0.0, 1.5), "depth": (0.0, 2.0)}
        perspective = render_point_cloud_fallback(
            points,
            colors,
            view_config=view_config,
            render_mode="color_by_rgb",
            scalar_bounds=scalar_bounds,
            projection_mode="perspective",
            point_radius_px=2,
            supersample_scale=1,
        )
        orthographic = render_point_cloud_fallback(
            points,
            colors,
            view_config=view_config,
            render_mode="color_by_rgb",
            scalar_bounds=scalar_bounds,
            projection_mode="orthographic",
            ortho_scale=1.0,
            point_radius_px=2,
            supersample_scale=1,
        )
        self.assertEqual(perspective.shape, orthographic.shape)
        self.assertGreater(int(perspective.sum()), 0)
        self.assertGreater(int(orthographic.sum()), 0)
        self.assertFalse(np.array_equal(perspective, orthographic))


if __name__ == "__main__":
    unittest.main()
