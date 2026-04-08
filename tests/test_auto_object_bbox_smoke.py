from __future__ import annotations

import unittest

import numpy as np

from data_process.visualization.pointcloud_compare import compute_scene_crop_bounds


class AutoObjectBboxSmokeTest(unittest.TestCase):
    def test_auto_object_bbox_focuses_on_object_above_table(self) -> None:
        grid_x, grid_y = np.meshgrid(np.linspace(-0.6, 0.6, 20), np.linspace(-0.5, 0.5, 16))
        table_points = np.stack(
            [grid_x.reshape(-1), grid_y.reshape(-1), np.zeros(grid_x.size, dtype=np.float32)],
            axis=1,
        ).astype(np.float32)
        object_grid_x, object_grid_y, object_grid_z = np.meshgrid(
            np.linspace(-0.08, 0.08, 4),
            np.linspace(-0.07, 0.07, 4),
            np.linspace(0.05, 0.12, 3),
        )
        object_points = np.stack(
            [
                object_grid_x.reshape(-1),
                object_grid_y.reshape(-1),
                object_grid_z.reshape(-1),
            ],
            axis=1,
        ).astype(np.float32)
        stacked = np.concatenate([table_points, object_points], axis=0)
        bounds = compute_scene_crop_bounds(
            [stacked],
            focus_point=np.array([0.0, 0.0, 0.02], dtype=np.float32),
            scene_crop_mode="auto_object_bbox",
            crop_margin_xy=0.10,
            crop_min_z=-0.15,
            crop_max_z=0.35,
            object_height_min=0.02,
            object_height_max=0.20,
            object_component_mode="largest",
            object_component_topk=1,
        )
        self.assertLess(float(bounds["max"][0] - bounds["min"][0]), 0.5)
        self.assertLess(float(bounds["max"][1] - bounds["min"][1]), 0.5)
        self.assertGreater(float(bounds["object_roi_max"][2]), 0.07)
        self.assertFalse(bool(bounds["fallback_used"]))


if __name__ == "__main__":
    unittest.main()
