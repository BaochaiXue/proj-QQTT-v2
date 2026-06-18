from __future__ import annotations

import unittest

import numpy as np

from qqtt.demo.demo32_side_by_side_panel import (
    CAMERA_COLOR_FRAME,
    SideBySidePanelHud,
    SideBySidePanelInputs,
    compute_rgb_ahead_frames,
    render_projected_pcd_panel,
    render_side_by_side_panel,
    render_tracking_overlay_panel,
)


class Demo32SideBySidePanelTest(unittest.TestCase):
    def test_compute_rgb_ahead_frames_clamps_negative_values(self) -> None:
        self.assertEqual(compute_rgb_ahead_frames(rgb_seq=8, paired_seq=5), 3)
        self.assertEqual(compute_rgb_ahead_frames(rgb_seq=5, paired_seq=8), 0)

    def test_render_side_by_side_panel_stacks_three_columns_and_draws_hud(self) -> None:
        left = np.full((4, 5, 3), (10, 20, 30), dtype=np.uint8)
        middle = np.full((4, 5, 3), (40, 50, 60), dtype=np.uint8)
        right = np.full((4, 5, 3), (70, 80, 90), dtype=np.uint8)
        hud = SideBySidePanelHud(
            rgb_seq=9,
            paired_seq=7,
            input_time_s=1.4,
            pipeline_latency_ms=230.0,
            display_latency_ms=245.0,
            startup_hold_s=2.5,
            filter_preset="enhanced-pt",
            marker_count=12,
            tracking_background="target-union",
            object_point_count=3,
            controller_point_count=4,
        )

        panel = render_side_by_side_panel(
            SideBySidePanelInputs(
                rgb_image_bgr=left,
                pcd_panel_bgr=middle,
                tracking_panel_bgr=right,
                hud=hud,
            )
        )

        self.assertEqual(panel.shape, (4, 15, 3))
        self.assertGreater(int(panel.sum()), int(left.sum() + middle.sum() + right.sum()))

    def test_render_side_by_side_panel_resizes_inputs_to_output_cell(self) -> None:
        left = np.full((4, 5, 3), 20, dtype=np.uint8)
        middle = np.full((8, 10, 3), 40, dtype=np.uint8)
        right = np.full((2, 3, 3), 60, dtype=np.uint8)
        hud = SideBySidePanelHud(
            rgb_seq=1,
            paired_seq=1,
            input_time_s=0.0,
            pipeline_latency_ms=1.0,
            display_latency_ms=2.0,
            startup_hold_s=0.0,
            filter_preset="pt",
            marker_count=0,
        )

        panel = render_side_by_side_panel(
            SideBySidePanelInputs(left, middle, right, hud),
            cell_size=(6, 4),
        )

        self.assertEqual(panel.shape, (4, 18, 3))

    def test_render_projected_pcd_panel_draws_camera_frame_points(self) -> None:
        points = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=np.float32)
        colors = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
        panel, count = render_projected_pcd_panel(
            width=8,
            height=6,
            intrinsics={"fx": 4.0, "fy": 4.0, "cx": 4.0, "cy": 3.0},
            controller_xyz_m=points[:1],
            controller_rgb_u8=colors[:1],
            object_xyz_m=points[1:],
            object_rgb_u8=colors[1:],
            point_size=1,
            max_render_points=0,
            coordinate_frame=CAMERA_COLOR_FRAME,
            camera_to_world_c2w=None,
        )

        self.assertEqual(panel.shape, (6, 8, 3))
        self.assertEqual(count["controller_points"], 1)
        self.assertEqual(count["object_points"], 1)
        self.assertGreater(int(panel.sum()), 0)

    def test_render_tracking_overlay_panel_draws_visible_query_points(self) -> None:
        image = np.zeros((6, 8, 3), dtype=np.uint8)
        tracks_yx = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
        visibility = np.array([1.0, 0.0], dtype=np.float32)
        marker_rgb = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)

        panel, counts = render_tracking_overlay_panel(
            image_bgr=image,
            tracks_yx=tracks_yx,
            visibility=visibility,
            marker_rgb_u8=marker_rgb,
            query_is_object=np.array([True, False], dtype=bool),
            query_is_controller=np.array([False, True], dtype=bool),
            query_controller_instance_id=np.array([0, 1], dtype=np.int64),
            marker_radius=1,
        )

        self.assertEqual(panel.shape, (6, 8, 3))
        self.assertEqual(counts["query_points"], 1)
        self.assertEqual(counts["query_object_points"], 1)
        self.assertEqual(counts["query_controller_points"], 0)
        self.assertGreater(int(panel.sum()), 0)


if __name__ == "__main__":
    unittest.main()
