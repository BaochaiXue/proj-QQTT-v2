from __future__ import annotations

import unittest

import numpy as np

from data_process.visualization.support_compare import compute_support_count_map, render_support_count_map, summarize_support_counts


class SupportRenderSmokeTest(unittest.TestCase):
    def test_support_render_counts_camera_agreement(self) -> None:
        shared_points = np.array(
            [
                [-0.05, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.05, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        clouds = []
        for camera_idx in range(3):
            clouds.append(
                {
                    "camera_idx": camera_idx,
                    "serial": str(camera_idx),
                    "points": shared_points.copy(),
                    "colors": np.full((len(shared_points), 3), 180, dtype=np.uint8),
                }
            )
        view_config = {
            "center": np.array([0.0, 0.0, 1.0], dtype=np.float32),
            "camera_position": np.array([0.0, -2.0, 1.0], dtype=np.float32),
            "up": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        }
        support = compute_support_count_map(
            clouds,
            view_config=view_config,
            width=320,
            height=240,
            projection_mode="perspective",
            ortho_scale=None,
        )
        image = render_support_count_map(support["support_count"], support["valid"])
        summary = summarize_support_counts(support["support_count"], support["valid"])

        self.assertEqual(image.shape, (240, 320, 3))
        self.assertGreater(summary["support_ratio_3"], 0.0)
        self.assertGreater(int(image.sum()), 0)


if __name__ == "__main__":
    unittest.main()
