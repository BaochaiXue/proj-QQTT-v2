from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from data_process.visualization.object_compare import build_geometry_constrained_foreground_mask


class GeometryConstrainedMaskSmokeTest(unittest.TestCase):
    def test_geometry_constraint_reduces_table_leakage_inside_manual_roi(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            image_path = tmp_root / "rgb.png"
            image = np.full((120, 160, 3), 180, dtype=np.uint8)
            cv2.imwrite(str(image_path), image)

            table_u = np.linspace(25, 130, 80, dtype=np.float32)
            table_v = np.linspace(35, 80, 32, dtype=np.float32)
            uu_table, vv_table = np.meshgrid(table_u, table_v, indexing="xy")
            table_points = np.stack(
                [
                    (uu_table.reshape(-1) - 80.0) / 120.0,
                    np.zeros(uu_table.size, dtype=np.float32),
                    np.full(uu_table.size, 1.25, dtype=np.float32),
                ],
                axis=1,
            ).astype(np.float32)

            object_u = np.linspace(55, 105, 38, dtype=np.float32)
            object_v = np.linspace(30, 78, 48, dtype=np.float32)
            uu_obj, vv_obj = np.meshgrid(object_u, object_v, indexing="xy")
            object_points = np.stack(
                [
                    (uu_obj.reshape(-1) - 80.0) / 120.0,
                    np.full(uu_obj.size, 0.12, dtype=np.float32),
                    np.full(uu_obj.size, 0.95, dtype=np.float32),
                ],
                axis=1,
            ).astype(np.float32)

            camera_cloud = {
                "camera_idx": 0,
                "serial": "serial-0",
                "points": np.concatenate([table_points, object_points], axis=0),
                "colors": np.full((len(table_points) + len(object_points), 3), 160, dtype=np.uint8),
                "K_color": np.array([[120.0, 0.0, 80.0], [0.0, 120.0, 60.0], [0.0, 0.0, 1.0]], dtype=np.float32),
                "c2w": np.eye(4, dtype=np.float32),
                "color_path": str(image_path),
            }

            mask, metrics = build_geometry_constrained_foreground_mask(
                camera_cloud,
                roi=(20, 20, 135, 95),
                plane_point=np.zeros((3,), dtype=np.float32),
                plane_normal=np.array([0.0, 1.0, 0.0], dtype=np.float32),
                object_height_min=0.02,
                object_height_max=0.30,
            )

            self.assertGreater(metrics["geometry_mask_pixels"], 0)
            self.assertGreater(metrics["refined_mask_pixels"], 0)
            self.assertLess(metrics["refined_mask_pixels"], (135 - 20) * (95 - 20))
            self.assertTrue(mask[74, 80])
            self.assertFalse(mask[60, 30])


if __name__ == "__main__":
    unittest.main()
