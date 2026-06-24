from __future__ import annotations

import unittest
import warnings
from types import SimpleNamespace

import numpy as np

from qqtt.demo.demo32_side_by_side_panel import (
    CAMERA_COLOR_FRAME,
    SideBySidePanelHud,
    SideBySidePanelInputs,
    TABLE_WORLD_FRAME_KIND,
    compute_rgb_ahead_frames,
    _hud_lines,
    _remaining_query_legend_lines,
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

    def test_remaining_query_legend_reports_total_and_class_breakdown(self) -> None:
        hud = SideBySidePanelHud(
            rgb_seq=1,
            paired_seq=1,
            input_time_s=0.0,
            pipeline_latency_ms=1.0,
            display_latency_ms=2.0,
            startup_hold_s=0.0,
            filter_preset="original",
            marker_count=4,
            query_count=10,
            remaining_query_count=7,
            remaining_object_query_count=2,
            remaining_controller_query_count=5,
            remaining_hand_a_query_count=3,
            remaining_hand_b_query_count=2,
        )

        lines = _remaining_query_legend_lines(hud)

        self.assertEqual(lines[0], "remaining 7/10")
        self.assertIn("obj=2", lines[1])
        self.assertIn("ctrl=5", lines[1])
        self.assertIn("hand_a=3", lines[2])
        self.assertIn("hand_b=2", lines[2])

    def test_hud_lines_report_stage_fps(self) -> None:
        hud = SideBySidePanelHud(
            rgb_seq=1,
            paired_seq=1,
            input_time_s=0.0,
            pipeline_latency_ms=1.0,
            display_latency_ms=2.0,
            startup_hold_s=0.0,
            filter_preset="original",
            marker_count=4,
            capture_fps=5.0,
            seg_fps=4.5,
            depth_fps=4.0,
            pcd_fps=3.5,
            tracker_fps=3.0,
            render_fps=2.5,
        )

        lines = _hud_lines(hud)

        self.assertIn("FPS cap/seg/depth/pcd/tracker/render", lines[-1])
        self.assertIn("5.0/4.5/4.0/3.5/3.0/2.5", lines[-1])

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

    def test_render_projected_pcd_panel_draws_shape_prior_reference_points(self) -> None:
        panel, count = render_projected_pcd_panel(
            width=8,
            height=6,
            intrinsics={"fx": 4.0, "fy": 4.0, "cx": 4.0, "cy": 3.0},
            controller_xyz_m=np.empty((0, 3), dtype=np.float32),
            controller_rgb_u8=np.empty((0, 3), dtype=np.uint8),
            object_xyz_m=np.empty((0, 3), dtype=np.float32),
            object_rgb_u8=np.empty((0, 3), dtype=np.uint8),
            shape_prior_xyz_m=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
            shape_prior_rgb_u8=np.array([[150, 150, 150]], dtype=np.uint8),
            point_size=1,
            max_render_points=0,
            coordinate_frame=CAMERA_COLOR_FRAME,
            camera_to_world_c2w=None,
        )

        self.assertEqual(count["shape_prior_points"], 1)
        self.assertGreater(int(panel.sum()), 0)

    def test_render_projected_pcd_panel_accepts_intrinsics_object(self) -> None:
        panel, count = render_projected_pcd_panel(
            width=8,
            height=6,
            intrinsics=SimpleNamespace(fx=4.0, fy=4.0, cx=4.0, cy=3.0),
            controller_xyz_m=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
            controller_rgb_u8=np.array([[255, 0, 0]], dtype=np.uint8),
            object_xyz_m=np.empty((0, 3), dtype=np.float32),
            object_rgb_u8=np.empty((0, 3), dtype=np.uint8),
            point_size=1,
            max_render_points=0,
            coordinate_frame=CAMERA_COLOR_FRAME,
            camera_to_world_c2w=None,
        )

        self.assertEqual(panel.shape, (6, 8, 3))
        self.assertEqual(count["controller_points"], 1)
        self.assertGreater(int(panel.sum()), 0)

    def test_render_projected_pcd_panel_accepts_intrinsics_matrix(self) -> None:
        intrinsics = np.array(
            [
                [4.0, 0.0, 4.0],
                [0.0, 4.0, 3.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        panel, count = render_projected_pcd_panel(
            width=8,
            height=6,
            intrinsics=intrinsics,
            controller_xyz_m=np.empty((0, 3), dtype=np.float32),
            controller_rgb_u8=np.empty((0, 3), dtype=np.uint8),
            object_xyz_m=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
            object_rgb_u8=np.array([[0, 255, 0]], dtype=np.uint8),
            point_size=1,
            max_render_points=0,
            coordinate_frame=CAMERA_COLOR_FRAME,
            camera_to_world_c2w=None,
        )

        self.assertEqual(panel.shape, (6, 8, 3))
        self.assertEqual(count["object_points"], 1)
        self.assertGreater(int(panel.sum()), 0)

    def test_render_projected_pcd_panel_projects_table_world_points(self) -> None:
        camera_to_world = np.eye(4, dtype=np.float32)
        camera_to_world[0, 3] = 1.0
        world_point = np.array([[1.0, 0.0, 1.0]], dtype=np.float32)

        panel, count = render_projected_pcd_panel(
            width=8,
            height=6,
            intrinsics={"fx": 4.0, "fy": 4.0, "cx": 4.0, "cy": 3.0},
            controller_xyz_m=world_point,
            controller_rgb_u8=np.array([[255, 0, 0]], dtype=np.uint8),
            object_xyz_m=np.empty((0, 3), dtype=np.float32),
            object_rgb_u8=np.empty((0, 3), dtype=np.uint8),
            point_size=1,
            max_render_points=0,
            coordinate_frame=TABLE_WORLD_FRAME_KIND,
            camera_to_world_c2w=camera_to_world,
        )

        self.assertEqual(count["controller_points"], 1)
        self.assertGreater(int(panel.sum()), 0)

    def test_render_projected_pcd_panel_requires_table_world_transform(self) -> None:
        with self.assertRaisesRegex(ValueError, "camera_to_world_c2w"):
            render_projected_pcd_panel(
                width=8,
                height=6,
                intrinsics={"fx": 4.0, "fy": 4.0, "cx": 4.0, "cy": 3.0},
                controller_xyz_m=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
                controller_rgb_u8=np.array([[255, 0, 0]], dtype=np.uint8),
                object_xyz_m=np.empty((0, 3), dtype=np.float32),
                object_rgb_u8=np.empty((0, 3), dtype=np.uint8),
                point_size=1,
                max_render_points=0,
                coordinate_frame=TABLE_WORLD_FRAME_KIND,
                camera_to_world_c2w=None,
            )

    def test_render_projected_pcd_panel_filters_invalid_points_without_warnings(self) -> None:
        points = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
                [0.0, 0.0, 0.0],
                [np.nan, 0.0, 1.0],
                [10.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        colors = np.full((5, 3), (255, 0, 0), dtype=np.uint8)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            panel, count = render_projected_pcd_panel(
                width=8,
                height=6,
                intrinsics={"fx": 4.0, "fy": 4.0, "cx": 4.0, "cy": 3.0},
                controller_xyz_m=points,
                controller_rgb_u8=colors,
                object_xyz_m=np.empty((0, 3), dtype=np.float32),
                object_rgb_u8=np.empty((0, 3), dtype=np.uint8),
                point_size=1,
                max_render_points=0,
                coordinate_frame=CAMERA_COLOR_FRAME,
                camera_to_world_c2w=None,
            )

        self.assertEqual(caught, [])
        self.assertEqual(count["controller_points"], 1)
        self.assertGreater(int(panel.sum()), 0)

    def test_render_projected_pcd_panel_rejects_mismatched_point_colors(self) -> None:
        with self.assertRaisesRegex(ValueError, "controller_xyz_m.*controller_rgb_u8"):
            render_projected_pcd_panel(
                width=8,
                height=6,
                intrinsics={"fx": 4.0, "fy": 4.0, "cx": 4.0, "cy": 3.0},
                controller_xyz_m=np.zeros((2, 3), dtype=np.float32),
                controller_rgb_u8=np.zeros((1, 3), dtype=np.uint8),
                object_xyz_m=np.empty((0, 3), dtype=np.float32),
                object_rgb_u8=np.empty((0, 3), dtype=np.uint8),
                point_size=1,
                max_render_points=0,
                coordinate_frame=CAMERA_COLOR_FRAME,
                camera_to_world_c2w=None,
            )

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

    def test_render_tracking_overlay_panel_rejects_mismatched_arrays(self) -> None:
        with self.assertRaisesRegex(ValueError, "tracks_yx.*visibility"):
            render_tracking_overlay_panel(
                image_bgr=np.zeros((6, 8, 3), dtype=np.uint8),
                tracks_yx=np.zeros((2, 2), dtype=np.float32),
                visibility=np.ones((1,), dtype=np.float32),
                marker_rgb_u8=np.zeros((2, 3), dtype=np.uint8),
                query_is_object=np.ones((2,), dtype=bool),
                query_is_controller=np.zeros((2,), dtype=bool),
                query_controller_instance_id=np.zeros((2,), dtype=np.int64),
                marker_radius=1,
            )


if __name__ == "__main__":
    unittest.main()
