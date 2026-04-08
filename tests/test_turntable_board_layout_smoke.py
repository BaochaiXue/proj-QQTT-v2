from __future__ import annotations

import unittest

import numpy as np

from data_process.visualization.camera_frusta import build_camera_frustum_geometry, extract_camera_poses
from data_process.visualization.turntable_compare import (
    build_camera_anchored_orbit_views,
    build_object_centered_orbit_views,
    build_scene_overview_state,
    compose_keyframe_sheet,
    compose_side_by_side_large,
    compose_turntable_board,
    estimate_orbit_axis,
    render_overview_inset,
)


class TurntableBoardLayoutSmokeTest(unittest.TestCase):
    def test_board_and_keyframe_sheet_include_overview_inset(self) -> None:
        scene_points = np.array(
            [
                [-0.2, -0.1, 0.8],
                [0.0, 0.0, 1.0],
                [0.2, 0.1, 1.2],
                [0.1, -0.2, 0.9],
            ],
            dtype=np.float32,
        )
        scene_colors = np.full((len(scene_points), 3), 180, dtype=np.uint8)
        focus_point = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        c2w_list = []
        for position in (
            np.array([0.7, -0.9, 1.1], dtype=np.float32),
            np.array([0.0, -1.1, 1.0], dtype=np.float32),
            np.array([-0.7, -0.9, 1.1], dtype=np.float32),
        ):
            transform = np.eye(4, dtype=np.float32)
            transform[:3, 3] = position
            c2w_list.append(transform)
        poses = extract_camera_poses(c2w_list, serial_numbers=["a", "b", "c"], camera_ids=[0, 1, 2])
        geometries = [build_camera_frustum_geometry(pose, frustum_scale=0.18) for pose in poses]

        overview_state = build_scene_overview_state(
            scene_points=scene_points,
            scene_colors=scene_colors,
            camera_geometries=geometries,
            focus_point=focus_point,
            render_mode="color_by_height",
            renderer="fallback",
            scalar_bounds={"height": (0.5, 1.3), "depth": (0.0, 2.0)},
            point_radius_px=2,
            supersample_scale=1,
            orbit_path_points=scene_points,
            crop_bounds={
                "min": np.array([-0.25, -0.25, 0.75], dtype=np.float32),
                "max": np.array([0.25, 0.25, 1.25], dtype=np.float32),
            },
        )
        orbit_steps = build_camera_anchored_orbit_views(
            camera_poses=poses,
            focus_point=focus_point,
            orbit_axis=estimate_orbit_axis(poses),
            num_orbit_steps=1,
            orbit_degrees=0.0,
        )
        inset = render_overview_inset(
            overview_state,
            current_views=orbit_steps[0]["view_configs"],
        )
        self.assertEqual(inset.shape[:2], (320, 560))
        self.assertGreater(int(inset.sum()), 0)

        panel = np.full((120, 160, 3), 70, dtype=np.uint8)
        board = compose_turntable_board(
            title_lines=["demo case", "frame=0"],
            column_headers=["Near Cam0", "Near Cam1", "Near Cam2"],
            row_headers=["Native", "FFS"],
            native_images=[panel, panel, panel],
            ffs_images=[panel, panel, panel],
            overview_inset=inset,
        )
        self.assertEqual(board.shape[1], 170 + 160 * 3)
        self.assertGreater(board.shape[0], 40 + 120 * 2)

        object_orbit = build_object_centered_orbit_views(
            camera_poses=poses,
            focus_point=focus_point,
            bounds_min=np.array([-0.25, -0.25, 0.75], dtype=np.float32),
            bounds_max=np.array([0.25, 0.25, 1.25], dtype=np.float32),
            orbit_axis=estimate_orbit_axis(poses),
            num_orbit_steps=1,
            orbit_degrees=360.0,
            orbit_radius_scale=1.8,
            view_height_offset=0.0,
            orbit_mode="full_360",
            coverage_margin_deg=12.0,
            show_unsupported_warning=True,
        )
        side = compose_side_by_side_large(
            title_lines=["demo case", "frame=0"],
            native_image=panel,
            ffs_image=panel,
            overview_inset=render_overview_inset(
                overview_state,
                current_views=object_orbit["orbit_steps"][0]["view_configs"],
            ),
        )
        self.assertGreaterEqual(side.shape[1], 160 * 2)
        self.assertGreater(side.shape[0], 120)

        sheet = compose_keyframe_sheet([board, board])
        self.assertGreater(sheet.shape[0], board.shape[0] // 2)
        self.assertGreater(int(sheet.sum()), 0)


if __name__ == "__main__":
    unittest.main()
