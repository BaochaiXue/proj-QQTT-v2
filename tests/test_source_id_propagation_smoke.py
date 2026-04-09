from __future__ import annotations

import unittest

import numpy as np

from data_process.visualization.object_compare import build_object_first_layers


class SourceIdPropagationSmokeTest(unittest.TestCase):
    def test_source_camera_idx_survives_object_context_and_fused_cloud_build(self) -> None:
        camera_clouds = []
        for camera_idx in (0, 1):
            points = np.array(
                [
                    [camera_idx * 0.1, 0.0, 0.06],
                    [camera_idx * 0.1, 0.02, 0.10],
                    [camera_idx * 0.1, 0.04, 0.00],
                ],
                dtype=np.float32,
            )
            camera_clouds.append(
                {
                    "camera_idx": camera_idx,
                    "serial": f"serial-{camera_idx}",
                    "points": points,
                    "colors": np.full((len(points), 3), 120 + camera_idx * 20, dtype=np.uint8),
                    "source_camera_idx": np.full((len(points),), camera_idx, dtype=np.int16),
                    "source_serial": np.full((len(points),), f"serial-{camera_idx}", dtype=object),
                    "K_color": np.eye(3, dtype=np.float32),
                    "c2w": np.eye(4, dtype=np.float32),
                    "color_path": "",
                }
            )

        layers = build_object_first_layers(
            camera_clouds,
            object_roi_min=np.array([-0.1, -0.1, 0.02], dtype=np.float32),
            object_roi_max=np.array([0.2, 0.2, 0.12], dtype=np.float32),
            plane_point=np.zeros((3,), dtype=np.float32),
            plane_normal=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            table_color_bgr=None,
            object_height_min=0.02,
            object_height_max=0.12,
            context_max_points_per_camera=10,
        )

        self.assertEqual(len(layers["object_points"]), len(layers["object_source_camera_idx"]))
        self.assertEqual(len(layers["context_points"]), len(layers["context_source_camera_idx"]))
        self.assertEqual(len(layers["combined_points"]), len(layers["combined_source_camera_idx"]))
        self.assertEqual(set(layers["combined_source_camera_idx"].tolist()), {0, 1})


if __name__ == "__main__":
    unittest.main()
