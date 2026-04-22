from __future__ import annotations

import unittest

import numpy as np

from data_process.visualization.views import build_original_camera_view_configs


class OriginalCameraViewConfigSmokeTest(unittest.TestCase):
    def test_builds_strict_original_camera_views_from_c2w(self) -> None:
        c2w_list = [
            np.eye(4, dtype=np.float32),
            np.array([[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]], dtype=np.float32),
            np.array([[1, 0, 0, 4], [0, 0, -1, 5], [0, 1, 0, 6], [0, 0, 0, 1]], dtype=np.float32),
        ]
        configs = build_original_camera_view_configs(
            c2w_list=c2w_list,
            serial_numbers=["a", "b", "c"],
            look_distance=1.25,
        )

        self.assertEqual([cfg["view_name"] for cfg in configs], ["cam0", "cam1", "cam2"])
        np.testing.assert_allclose(configs[0]["camera_position"], np.array([0.0, 0.0, 0.0], dtype=np.float32))
        np.testing.assert_allclose(configs[0]["forward"], np.array([0.0, 0.0, 1.0], dtype=np.float32))
        np.testing.assert_allclose(configs[0]["up"], np.array([0.0, -1.0, 0.0], dtype=np.float32))
        np.testing.assert_allclose(configs[0]["center"], np.array([0.0, 0.0, 1.25], dtype=np.float32))

        np.testing.assert_allclose(configs[1]["camera_position"], np.array([1.0, 2.0, 3.0], dtype=np.float32))
        np.testing.assert_allclose(configs[1]["center"], np.array([1.0, 2.0, 4.25], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
