from __future__ import annotations

import unittest

import numpy as np

from data_process.visualization.object_compare import build_object_first_layers


class ObjectCompareContractSmokeTest(unittest.TestCase):
    def test_object_context_contract_keeps_aligned_lengths(self) -> None:
        points = np.array(
            [
                [0.0, 0.0, 0.08],
                [0.02, 0.0, 0.12],
                [0.05, 0.02, 0.02],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        colors = np.tile(np.array([[100, 150, 200]], dtype=np.uint8), (len(points), 1))
        camera_cloud = {
            "camera_idx": 0,
            "serial": "cam0",
            "points": points,
            "colors": colors,
            "source_camera_idx": np.zeros((len(points),), dtype=np.int16),
            "source_serial": np.full((len(points),), "cam0", dtype=object),
            "c2w": np.eye(4, dtype=np.float32),
            "K_color": np.eye(3, dtype=np.float32),
        }
        layers = build_object_first_layers(
            [camera_cloud],
            object_roi_min=np.array([-0.01, -0.01, 0.01], dtype=np.float32),
            object_roi_max=np.array([0.06, 0.03, 0.15], dtype=np.float32),
            plane_point=np.zeros((3,), dtype=np.float32),
            plane_normal=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            table_color_bgr=None,
            object_height_min=0.01,
            object_height_max=0.20,
            context_max_points_per_camera=None,
        )
        self.assertEqual(len(layers["object_points"]), len(layers["object_colors"]))
        self.assertEqual(len(layers["context_points"]), len(layers["context_colors"]))
        self.assertEqual(len(layers["combined_points"]), len(layers["combined_colors"]))
        self.assertEqual(len(layers["combined_points"]), len(layers["combined_source_camera_idx"]))
        self.assertEqual(
            len(layers["combined_points"]),
            len(layers["object_points"]) + len(layers["context_points"]),
        )
