from __future__ import annotations

import unittest

import numpy as np

from data_process.visualization.pointcloud_compare import build_camera_pose_view_configs


class CameraPoseViewConfigSmokeTest(unittest.TestCase):
    def test_builds_camera_pose_views_from_c2w(self) -> None:
        c2w_list = [
            np.array([[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]], dtype=np.float32),
            np.array([[0, -1, 0, 4], [1, 0, 0, 5], [0, 0, 1, 6], [0, 0, 0, 1]], dtype=np.float32),
            np.array([[1, 0, 0, 7], [0, 1, 0, 8], [0, 0, 1, 9], [0, 0, 0, 1]], dtype=np.float32),
        ]
        focus = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        configs = build_camera_pose_view_configs(
            c2w_list=c2w_list,
            serial_numbers=["a", "b", "c"],
            focus_point=focus,
        )
        self.assertEqual([cfg["view_name"] for cfg in configs], ["cam0", "cam1", "cam2"])
        np.testing.assert_allclose(configs[0]["camera_position"], np.array([1.0, 2.0, 3.0], dtype=np.float32))
        np.testing.assert_allclose(configs[0]["center"], focus)
        np.testing.assert_allclose(configs[0]["up"], np.array([0.0, -1.0, 0.0], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
