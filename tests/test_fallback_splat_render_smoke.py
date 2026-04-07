from __future__ import annotations

import unittest

import numpy as np

from data_process.visualization.pointcloud_compare import render_point_cloud_fallback


class FallbackSplatRenderSmokeTest(unittest.TestCase):
    def test_splat_radius_makes_render_denser_than_single_pixels(self) -> None:
        points = np.array(
            [
                [-0.1, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.1, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        colors = np.full((3, 3), 255, dtype=np.uint8)
        view_config = {
            "center": np.array([0.0, 0.0, 1.0], dtype=np.float32),
            "camera_position": np.array([0.0, -2.0, 1.0], dtype=np.float32),
            "up": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        }
        scalar_bounds = {"height": (0.0, 2.0), "depth": (0.0, 2.0)}
        sparse = render_point_cloud_fallback(
            points,
            colors,
            view_config=view_config,
            render_mode="color_by_rgb",
            scalar_bounds=scalar_bounds,
            point_radius_px=1,
            supersample_scale=1,
        )
        dense = render_point_cloud_fallback(
            points,
            colors,
            view_config=view_config,
            render_mode="color_by_rgb",
            scalar_bounds=scalar_bounds,
            point_radius_px=4,
            supersample_scale=2,
        )
        self.assertGreater(int(np.count_nonzero(dense)), int(np.count_nonzero(sparse)))


if __name__ == "__main__":
    unittest.main()
