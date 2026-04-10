from __future__ import annotations

import unittest

import numpy as np

from data_process.visualization.source_compare import render_source_attribution_overlay


class StereoOrderRegistrationSourceColorSmokeTest(unittest.TestCase):
    def test_source_overlay_uses_fixed_camera_colors_without_embedded_legend_when_disabled(self) -> None:
        camera_clouds = []
        for camera_idx, x_offset in enumerate((-0.05, 0.0, 0.05)):
            points = np.array(
                [
                    [x_offset, 0.0, 0.0],
                    [x_offset, 0.0, 0.05],
                    [x_offset, 0.02, 0.03],
                ],
                dtype=np.float32,
            )
            camera_clouds.append(
                {
                    "camera_idx": camera_idx,
                    "serial": f"serial-{camera_idx}",
                    "points": points,
                    "colors": np.full((len(points), 3), 180, dtype=np.uint8),
                }
            )
        view_config = {
            "view_name": "top",
            "label": "Top",
            "center": np.array([0.0, 0.0, 0.02], dtype=np.float32),
            "camera_position": np.array([0.0, 0.0, 0.40], dtype=np.float32),
            "up": np.array([0.0, -1.0, 0.0], dtype=np.float32),
        }
        image, metrics = render_source_attribution_overlay(
            camera_clouds,
            view_config=view_config,
            width=220,
            height=180,
            projection_mode="orthographic",
            ortho_scale=0.18,
            alpha=1.0,
            show_legend=False,
        )
        self.assertFalse(metrics["show_legend"])
        pixels = image.reshape(-1, 3)
        self.assertTrue(np.any((pixels[:, 2] > 220) & (pixels[:, 1] < 40) & (pixels[:, 0] < 40)))
        self.assertTrue(np.any((pixels[:, 1] > 220) & (pixels[:, 2] < 40) & (pixels[:, 0] < 40)))
        self.assertTrue(np.any((pixels[:, 0] > 220) & (pixels[:, 1] < 40) & (pixels[:, 2] < 40)))


if __name__ == "__main__":
    unittest.main()
