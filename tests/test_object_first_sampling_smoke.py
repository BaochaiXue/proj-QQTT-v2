from __future__ import annotations

import unittest

import numpy as np

from data_process.visualization.object_compare import build_object_first_layers


class ObjectFirstSamplingSmokeTest(unittest.TestCase):
    def test_object_first_sampling_preserves_sparse_head_points(self) -> None:
        rng = np.random.default_rng(0)
        body = np.stack(
            [
                rng.uniform(-0.05, 0.05, 1000),
                rng.uniform(-0.05, 0.05, 1000),
                rng.uniform(0.05, 0.15, 1000),
            ],
            axis=1,
        ).astype(np.float32)
        head = np.stack(
            [
                rng.uniform(-0.03, 0.03, 25),
                rng.uniform(-0.03, 0.03, 25),
                rng.uniform(0.25, 0.32, 25),
            ],
            axis=1,
        ).astype(np.float32)
        context = np.stack(
            [
                rng.uniform(-0.4, 0.4, 6000),
                rng.uniform(-0.4, 0.4, 6000),
                rng.uniform(-0.02, 0.02, 6000),
            ],
            axis=1,
        ).astype(np.float32)
        points = np.concatenate([body, head, context], axis=0)
        colors = np.full((len(points), 3), 120, dtype=np.uint8)
        camera_cloud = {
            "camera_idx": 0,
            "serial": "serial-0",
            "points": points,
            "colors": colors,
            "K_color": np.eye(3, dtype=np.float32),
            "c2w": np.eye(4, dtype=np.float32),
            "color_path": "",
        }
        layers = build_object_first_layers(
            [camera_cloud],
            object_roi_min=np.array([-0.08, -0.08, 0.02], dtype=np.float32),
            object_roi_max=np.array([0.08, 0.08, 0.34], dtype=np.float32),
            plane_point=np.zeros((3,), dtype=np.float32),
            plane_normal=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            table_color_bgr=None,
            object_height_min=0.02,
            object_height_max=0.40,
            context_max_points_per_camera=200,
        )

        object_points = layers["object_camera_clouds"][0]["points"]
        context_points = layers["context_camera_clouds"][0]["points"]
        self.assertEqual(len(object_points), len(body) + len(head))
        self.assertLessEqual(len(context_points), 200)
        self.assertEqual(int(np.count_nonzero(object_points[:, 2] >= 0.24)), len(head))


if __name__ == "__main__":
    unittest.main()
