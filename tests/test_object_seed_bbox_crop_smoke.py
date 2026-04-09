from __future__ import annotations

import unittest

import numpy as np

from data_process.visualization.pointcloud_compare import compute_scene_crop_bounds


class ObjectSeedBboxCropSmokeTest(unittest.TestCase):
    def test_object_seed_points_drive_auto_object_bbox_when_available(self) -> None:
        xx, yy = np.meshgrid(np.linspace(-0.4, 0.4, 40), np.linspace(-0.4, 0.4, 40), indexing="xy")
        table = np.stack([xx.reshape(-1), yy.reshape(-1), np.zeros(xx.size)], axis=1).astype(np.float32)
        torso = np.stack(
            [
                np.linspace(-0.05, 0.05, 40),
                np.linspace(-0.02, 0.02, 40),
                np.linspace(0.04, 0.10, 40),
            ],
            axis=1,
        ).astype(np.float32)
        head = np.stack(
            [
                np.linspace(-0.03, 0.03, 16),
                np.linspace(-0.01, 0.01, 16),
                np.linspace(0.22, 0.28, 16),
            ],
            axis=1,
        ).astype(np.float32)
        full_scene = np.concatenate([table, torso, head], axis=0)

        crop_bounds = compute_scene_crop_bounds(
            [full_scene],
            focus_point=np.array([0.0, 0.0, 0.05], dtype=np.float32),
            scene_crop_mode="auto_object_bbox",
            crop_margin_xy=0.10,
            crop_min_z=-0.10,
            crop_max_z=0.30,
            object_seed_point_sets=[np.concatenate([torso, head], axis=0)],
            object_height_min=0.02,
            object_height_max=0.30,
            object_component_mode="union",
            object_component_topk=3,
        )

        self.assertTrue(bool(crop_bounds.get("seed_bbox_used", False)))
        self.assertGreater(float(crop_bounds["object_roi_max"][2]), 0.24)
        self.assertLess(float(crop_bounds["object_roi_min"][2]), 0.05)


if __name__ == "__main__":
    unittest.main()
