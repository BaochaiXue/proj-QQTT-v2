from __future__ import annotations

import unittest

import numpy as np

from data_process.visualization.source_compare import render_source_attribution_overlay, source_color_bgr


class SourceAttributionOverlaySmokeTest(unittest.TestCase):
    def test_overlay_uses_stable_source_colors(self) -> None:
        camera_clouds = []
        for camera_idx, x_value in zip((0, 1, 2), (-0.08, 0.0, 0.08), strict=False):
            points = np.array([[x_value, 0.0, 0.0]], dtype=np.float32)
            camera_clouds.append(
                {
                    "camera_idx": camera_idx,
                    "serial": f"serial-{camera_idx}",
                    "points": points,
                    "colors": np.zeros((1, 3), dtype=np.uint8),
                }
            )
        image, metrics = render_source_attribution_overlay(
            camera_clouds,
            view_config={
                "camera_position": np.array([0.0, -1.0, 0.0], dtype=np.float32),
                "center": np.array([0.0, 0.0, 0.0], dtype=np.float32),
                "up": np.array([0.0, 0.0, 1.0], dtype=np.float32),
            },
            width=320,
            height=240,
            projection_mode="perspective",
            ortho_scale=None,
            alpha=1.0,
        )

        unique_colors = {tuple(color.tolist()) for color in image.reshape(-1, 3)}
        for camera_idx in (0, 1, 2):
            self.assertIn(source_color_bgr(camera_idx), unique_colors)
        self.assertEqual(len(metrics["per_camera"]), 3)


if __name__ == "__main__":
    unittest.main()
