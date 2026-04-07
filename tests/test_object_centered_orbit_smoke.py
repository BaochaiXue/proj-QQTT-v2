from __future__ import annotations

import unittest

import numpy as np

from data_process.visualization.camera_frusta import extract_camera_poses
from data_process.visualization.turntable_compare import build_object_centered_orbit_views, estimate_orbit_axis


class ObjectCenteredOrbitSmokeTest(unittest.TestCase):
    def test_object_centered_orbit_uses_shared_radius_and_full_turn(self) -> None:
        c2w_list = []
        for position in (
            np.array([1.0, 0.0, 0.8], dtype=np.float32),
            np.array([0.0, 1.0, 0.85], dtype=np.float32),
            np.array([-1.0, 0.0, 0.82], dtype=np.float32),
        ):
            transform = np.eye(4, dtype=np.float32)
            transform[:3, 3] = position
            c2w_list.append(transform)

        poses = extract_camera_poses(c2w_list, serial_numbers=["a", "b", "c"], camera_ids=[0, 1, 2])
        orbit = build_object_centered_orbit_views(
            camera_poses=poses,
            focus_point=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            bounds_min=np.array([-0.3, -0.25, -0.02], dtype=np.float32),
            bounds_max=np.array([0.3, 0.25, 0.18], dtype=np.float32),
            orbit_axis=estimate_orbit_axis(poses),
            num_orbit_steps=4,
            orbit_degrees=360.0,
            orbit_radius_scale=1.8,
            view_height_offset=0.0,
        )

        self.assertEqual(len(orbit["orbit_steps"]), 4)
        self.assertEqual(orbit["orbit_path"].shape, (4, 3))
        self.assertIn(0, orbit["camera_reference_azimuths_deg"])

        radii = []
        for step in orbit["orbit_steps"]:
            position = step["view_config"]["camera_position"]
            radii.append(float(np.linalg.norm(position)))
        self.assertLess(max(radii) - min(radii), 1e-4)
        self.assertGreater(orbit["orbit_height"], 0.05)


if __name__ == "__main__":
    unittest.main()
