from __future__ import annotations

import unittest

import numpy as np

from data_process.visualization.camera_frusta import extract_camera_poses
from data_process.visualization.turntable_compare import (
    build_object_centered_orbit_views,
    compute_camera_azimuths_deg,
    estimate_orbit_axis,
    estimate_supported_coverage_arc,
)


class ObservedHemisphereOrbitSmokeTest(unittest.TestCase):
    def test_observed_hemisphere_orbit_stays_within_supported_arc(self) -> None:
        c2w_list = []
        for position in (
            np.array([1.0, -0.2, 0.7], dtype=np.float32),
            np.array([0.6, 0.5, 0.72], dtype=np.float32),
            np.array([0.1, 0.8, 0.68], dtype=np.float32),
        ):
            transform = np.eye(4, dtype=np.float32)
            transform[:3, 3] = position
            c2w_list.append(transform)

        poses = extract_camera_poses(c2w_list, serial_numbers=["a", "b", "c"], camera_ids=[0, 1, 2])
        focus = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        orbit_axis = estimate_orbit_axis(poses)
        azimuths = compute_camera_azimuths_deg(camera_poses=poses, focus_point=focus, orbit_axis=orbit_axis)
        coverage = estimate_supported_coverage_arc(azimuths, coverage_margin_deg=12.0)
        orbit = build_object_centered_orbit_views(
            camera_poses=poses,
            focus_point=focus,
            bounds_min=np.array([-0.2, -0.2, -0.02], dtype=np.float32),
            bounds_max=np.array([0.2, 0.2, 0.18], dtype=np.float32),
            orbit_axis=orbit_axis,
            num_orbit_steps=8,
            orbit_degrees=360.0,
            orbit_radius_scale=1.8,
            view_height_offset=0.0,
            orbit_mode="observed_hemisphere",
            coverage_margin_deg=12.0,
            show_unsupported_warning=True,
        )

        self.assertLess(coverage["span_deg"], 360.0)
        self.assertTrue(all(orbit["orbit_supported_mask"]))
        self.assertEqual(len(orbit["orbit_steps"]), 8)


if __name__ == "__main__":
    unittest.main()
