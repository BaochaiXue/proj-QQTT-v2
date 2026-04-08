from __future__ import annotations

import unittest

import numpy as np

from data_process.visualization.camera_frusta import build_camera_frustum_geometry, extract_camera_poses
from data_process.visualization.turntable_compare import build_scene_overview_state, render_overview_inset


class CameraOverviewInsetSmokeTest(unittest.TestCase):
    def test_overview_inset_draws_cameras_crop_and_orbit_path(self) -> None:
        scene_points = np.array(
            [
                [-0.2, -0.2, 0.02],
                [0.2, -0.2, 0.03],
                [0.2, 0.2, 0.01],
                [-0.2, 0.2, 0.02],
            ],
            dtype=np.float32,
        )
        scene_colors = np.full((len(scene_points), 3), 140, dtype=np.uint8)
        c2w_list = []
        for position in (
            np.array([0.8, -0.8, 0.7], dtype=np.float32),
            np.array([0.0, -1.0, 0.75], dtype=np.float32),
            np.array([-0.8, -0.8, 0.7], dtype=np.float32),
        ):
            transform = np.eye(4, dtype=np.float32)
            transform[:3, 3] = position
            c2w_list.append(transform)
        poses = extract_camera_poses(c2w_list, serial_numbers=["a", "b", "c"], camera_ids=[0, 1, 2])
        geometries = [build_camera_frustum_geometry(pose, frustum_scale=0.16) for pose in poses]

        overview = build_scene_overview_state(
            scene_points=scene_points,
            scene_colors=scene_colors,
            camera_geometries=geometries,
            focus_point=np.array([0.0, 0.0, 0.02], dtype=np.float32),
            render_mode="color_by_height",
            renderer="fallback",
            scalar_bounds={"height": (0.0, 0.1), "depth": (0.0, 2.0)},
            point_radius_px=2,
            supersample_scale=1,
            orbit_path_points=np.array(
                [
                    [0.6, 0.0, 0.55],
                    [0.0, 0.6, 0.55],
                    [-0.6, 0.0, 0.55],
                    [0.0, -0.6, 0.55],
                ],
                dtype=np.float32,
            ),
            crop_bounds={
                "min": np.array([-0.25, -0.25, -0.02], dtype=np.float32),
                "max": np.array([0.25, 0.25, 0.12], dtype=np.float32),
            },
        )
        self.assertEqual(overview["label"], "Top")
        inset = render_overview_inset(
            overview,
            current_views=[
                {
                    "camera_position": np.array([0.6, 0.0, 0.55], dtype=np.float32),
                    "anchor_camera_position": np.array([0.0, 0.0, 0.02], dtype=np.float32),
                    "center": np.array([0.0, 0.0, 0.02], dtype=np.float32),
                    "color_bgr": (255, 255, 255),
                    "nearest_camera_idx": 0,
                }
            ],
        )
        self.assertEqual(inset.shape[:2], (320, 560))
        self.assertGreater(int(inset.sum()), 0)


if __name__ == "__main__":
    unittest.main()
