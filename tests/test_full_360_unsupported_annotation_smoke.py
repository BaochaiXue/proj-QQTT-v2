from __future__ import annotations

import unittest

import numpy as np

from data_process.visualization.camera_frusta import extract_camera_poses
from data_process.visualization.turntable_compare import build_object_centered_orbit_views, estimate_orbit_axis


class Full360UnsupportedAnnotationSmokeTest(unittest.TestCase):
    def test_full_360_marks_backside_views_as_unsupported(self) -> None:
        c2w_list = []
        for position in (
            np.array([1.0, -0.2, 0.7], dtype=np.float32),
            np.array([0.7, 0.3, 0.72], dtype=np.float32),
            np.array([0.2, 0.7, 0.68], dtype=np.float32),
        ):
            transform = np.eye(4, dtype=np.float32)
            transform[:3, 3] = position
            c2w_list.append(transform)

        poses = extract_camera_poses(c2w_list, serial_numbers=["a", "b", "c"], camera_ids=[0, 1, 2])
        orbit = build_object_centered_orbit_views(
            camera_poses=poses,
            focus_point=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            bounds_min=np.array([-0.2, -0.2, -0.02], dtype=np.float32),
            bounds_max=np.array([0.2, 0.2, 0.18], dtype=np.float32),
            orbit_axis=estimate_orbit_axis(poses),
            num_orbit_steps=12,
            orbit_degrees=360.0,
            orbit_radius_scale=1.8,
            view_height_offset=0.0,
            orbit_mode="full_360",
            coverage_margin_deg=10.0,
            show_unsupported_warning=True,
        )

        unsupported_steps = [step for step in orbit["orbit_steps"] if not step["view_config"]["is_supported"]]
        self.assertGreater(len(unsupported_steps), 0)
        self.assertTrue(any(step["view_config"]["warning_text"] for step in unsupported_steps))


if __name__ == "__main__":
    unittest.main()
