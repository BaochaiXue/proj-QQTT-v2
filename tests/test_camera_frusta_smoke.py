from __future__ import annotations

import unittest

import numpy as np

from data_process.visualization.camera_frusta import (
    build_camera_frustum_geometry,
    collect_camera_geometry_points,
    extract_camera_pose,
)


class CameraFrustaSmokeTest(unittest.TestCase):
    def test_extracts_camera_pose_and_frustum_geometry_from_c2w(self) -> None:
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, 3] = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        pose = extract_camera_pose(c2w, camera_idx=0, serial="serial-0")
        geometry = build_camera_frustum_geometry(pose, frustum_scale=0.25)

        np.testing.assert_allclose(pose["position"], np.array([1.0, 2.0, 3.0], dtype=np.float32))
        np.testing.assert_allclose(pose["forward"], np.array([0.0, 0.0, 1.0], dtype=np.float32))
        self.assertEqual(pose["label"], "Cam0 | serial-0")
        self.assertEqual(geometry["frustum_corners"].shape, (4, 3))
        self.assertGreater(float(geometry["forward_tip"][2]), 3.25)

        points = collect_camera_geometry_points([geometry])
        self.assertGreaterEqual(len(points), 7)


if __name__ == "__main__":
    unittest.main()
