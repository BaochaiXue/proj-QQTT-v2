from __future__ import annotations

import unittest

import numpy as np

from data_process.visualization.semantic_world import infer_semantic_world_transform, transform_points_to_semantic


class SemanticWorldInferenceSmokeTest(unittest.TestCase):
    def test_inferred_semantic_z_points_toward_camera_side_and_frame_is_right_handed(self) -> None:
        xx, yy = np.meshgrid(np.linspace(-0.3, 0.3, 16), np.linspace(-0.2, 0.2, 14), indexing="xy")
        table = np.stack([xx.reshape(-1), yy.reshape(-1), np.zeros(xx.size)], axis=1).astype(np.float32)
        object_points = np.array(
            [
                [-0.05, -0.04, 0.06],
                [0.04, 0.03, 0.08],
                [0.00, 0.00, 0.14],
            ],
            dtype=np.float32,
        )
        scene_points = np.concatenate([table, object_points], axis=0)
        camera_centers = np.array(
            [
                [0.0, -0.55, 0.22],
                [0.52, 0.02, 0.24],
                [-0.46, 0.12, 0.21],
            ],
            dtype=np.float32,
        )
        frame = infer_semantic_world_transform(
            scene_points=scene_points,
            camera_centers=camera_centers,
            plane_point=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            plane_normal=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        )
        semantic_z = np.asarray(frame["semantic_axes"]["z"], dtype=np.float32)
        table_point = np.asarray(frame["plane_point"], dtype=np.float32)
        c_mean = np.asarray(frame["mean_camera_center"], dtype=np.float32)
        self.assertGreater(float((c_mean - table_point) @ semantic_z), 0.0)

        rotation = np.asarray(frame["transform"], dtype=np.float32)[:3, :3]
        self.assertGreater(float(np.linalg.det(rotation)), 0.0)

    def test_transformed_table_is_flat_in_semantic_z(self) -> None:
        table = np.array(
            [
                [-0.2, -0.2, 0.0],
                [0.2, -0.2, 0.0],
                [-0.2, 0.2, 0.0],
                [0.2, 0.2, 0.0],
            ],
            dtype=np.float32,
        )
        camera_centers = np.array([[0.0, 0.0, 0.5]], dtype=np.float32)
        frame = infer_semantic_world_transform(
            scene_points=table,
            camera_centers=camera_centers,
            plane_point=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            plane_normal=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        )
        transformed = transform_points_to_semantic(table, frame)
        self.assertLess(float(np.max(np.abs(transformed[:, 2]))), 1e-4)


if __name__ == "__main__":
    unittest.main()
