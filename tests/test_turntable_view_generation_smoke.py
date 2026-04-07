from __future__ import annotations

import unittest

import numpy as np

from data_process.visualization.camera_frusta import extract_camera_poses
from data_process.visualization.turntable_compare import (
    build_camera_anchored_orbit_views,
    estimate_orbit_axis,
)


class TurntableViewGenerationSmokeTest(unittest.TestCase):
    def test_orbit_views_stay_near_real_camera_anchors(self) -> None:
        c2w_list = []
        for position in (
            np.array([1.0, 0.0, 0.6], dtype=np.float32),
            np.array([0.0, 1.0, 0.6], dtype=np.float32),
            np.array([-1.0, 0.0, 0.6], dtype=np.float32),
        ):
            transform = np.eye(4, dtype=np.float32)
            transform[:3, 3] = position
            c2w_list.append(transform)

        poses = extract_camera_poses(
            c2w_list,
            serial_numbers=["a", "b", "c"],
            camera_ids=[0, 1, 2],
        )
        focus_point = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        orbit_axis = estimate_orbit_axis(poses)
        steps = build_camera_anchored_orbit_views(
            camera_poses=poses,
            focus_point=focus_point,
            orbit_axis=orbit_axis,
            num_orbit_steps=3,
            orbit_degrees=30.0,
        )

        self.assertEqual(len(steps), 3)
        self.assertAlmostEqual(steps[1]["angle_deg"], 0.0)
        np.testing.assert_allclose(steps[1]["view_configs"][0]["camera_position"], poses[0]["position"])

        anchor_radius = float(np.linalg.norm(poses[0]["position"] - focus_point))
        for step in steps:
            view = step["view_configs"][0]
            self.assertAlmostEqual(float(np.linalg.norm(view["camera_position"] - focus_point)), anchor_radius, places=5)

        self.assertFalse(
            np.allclose(
                steps[0]["view_configs"][0]["camera_position"],
                steps[2]["view_configs"][0]["camera_position"],
            )
        )


if __name__ == "__main__":
    unittest.main()
