from __future__ import annotations

import unittest

import numpy as np

from data_process.visualization.source_compare import render_source_split_images


class SourceSplitRenderSmokeTest(unittest.TestCase):
    def test_split_render_returns_three_camera_panels(self) -> None:
        camera_clouds = []
        for camera_idx in (0, 1, 2):
            camera_clouds.append(
                {
                    "camera_idx": camera_idx,
                    "serial": f"serial-{camera_idx}",
                    "points": np.array([[camera_idx * 0.04, 0.0, 0.0]], dtype=np.float32),
                    "colors": np.zeros((1, 3), dtype=np.uint8),
                }
            )
        images, metrics = render_source_split_images(
            camera_clouds,
            view_config={
                "camera_position": np.array([0.0, -1.0, 0.0], dtype=np.float32),
                "center": np.array([0.0, 0.0, 0.0], dtype=np.float32),
                "up": np.array([0.0, 0.0, 1.0], dtype=np.float32),
            },
            scalar_bounds={"height": (-0.1, 0.1), "depth": (0.0, 2.0)},
            renderer="fallback",
            width=200,
            height=160,
            point_radius_px=2,
            supersample_scale=1,
            projection_mode="perspective",
            ortho_scale=None,
        )

        self.assertEqual(len(images), 3)
        self.assertEqual(len(metrics), 3)
        self.assertTrue(all(image.shape == (160, 200, 3) for image in images))


if __name__ == "__main__":
    unittest.main()
