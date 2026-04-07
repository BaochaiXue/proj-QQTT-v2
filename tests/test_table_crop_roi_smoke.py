from __future__ import annotations

import unittest

import numpy as np

from data_process.visualization.pointcloud_compare import compute_scene_crop_bounds, crop_points_to_bounds


class TableCropRoiSmokeTest(unittest.TestCase):
    def test_auto_table_bbox_excludes_far_background(self) -> None:
        table_points = np.array(
            [
                [-0.4, -0.3, 0.02],
                [-0.3, 0.2, 0.03],
                [0.1, -0.1, 0.01],
                [0.3, 0.3, 0.02],
                [0.0, 0.0, 0.015],
            ],
            dtype=np.float32,
        )
        background = np.array(
            [
                [1.8, 1.4, 1.2],
                [-1.7, 1.3, 1.1],
            ],
            dtype=np.float32,
        )
        crop_bounds = compute_scene_crop_bounds(
            [table_points, background],
            focus_point=np.array([0.0, 0.0, 0.02], dtype=np.float32),
            scene_crop_mode="auto_table_bbox",
            crop_margin_xy=0.1,
            crop_min_z=-0.1,
            crop_max_z=0.2,
        )
        cropped_points, _ = crop_points_to_bounds(
            np.concatenate([table_points, background], axis=0),
            np.zeros((len(table_points) + len(background), 3), dtype=np.uint8),
            crop_bounds,
        )
        self.assertGreaterEqual(len(cropped_points), len(table_points))
        self.assertFalse(np.any(cropped_points[:, 2] > 0.5))


if __name__ == "__main__":
    unittest.main()
