from __future__ import annotations

import unittest

import numpy as np

from data_process.visualization.pointcloud_compare import project_world_points_to_image, render_point_cloud_fallback


class ProjectionModeSmokeTest(unittest.TestCase):
    def test_point_above_center_projects_to_upper_half(self) -> None:
        view_config = {
            "center": np.array([0.0, 0.0, 1.0], dtype=np.float32),
            "camera_position": np.array([0.0, -2.0, 1.0], dtype=np.float32),
            "up": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        }
        point_above = np.array([[0.0, 0.0, 1.25]], dtype=np.float32)
        width = 320
        height = 240
        for projection_mode, ortho_scale in (("perspective", None), ("orthographic", 1.0)):
            projected = project_world_points_to_image(
                point_above,
                view_config=view_config,
                width=width,
                height=height,
                projection_mode=projection_mode,
                ortho_scale=ortho_scale,
            )
            self.assertTrue(bool(projected["valid"][0]))
            self.assertLess(float(projected["uv"][0, 1]), height * 0.5)

    def test_orthographic_and_perspective_both_render(self) -> None:
        points = np.array(
            [
                [-0.2, -0.1, 0.8],
                [0.0, 0.0, 1.0],
                [0.2, 0.1, 1.2],
            ],
            dtype=np.float32,
        )
        colors = np.array(
            [
                [255, 0, 0],
                [0, 255, 0],
                [0, 0, 255],
            ],
            dtype=np.uint8,
        )
        view_config = {
            "center": np.array([0.0, 0.0, 1.0], dtype=np.float32),
            "camera_position": np.array([0.0, -2.0, 1.0], dtype=np.float32),
            "up": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        }
        scalar_bounds = {"height": (0.0, 1.5), "depth": (0.0, 2.0)}
        perspective = render_point_cloud_fallback(
            points,
            colors,
            view_config=view_config,
            render_mode="color_by_rgb",
            scalar_bounds=scalar_bounds,
            projection_mode="perspective",
            point_radius_px=2,
            supersample_scale=1,
        )
        orthographic = render_point_cloud_fallback(
            points,
            colors,
            view_config=view_config,
            render_mode="color_by_rgb",
            scalar_bounds=scalar_bounds,
            projection_mode="orthographic",
            ortho_scale=1.0,
            point_radius_px=2,
            supersample_scale=1,
        )
        self.assertEqual(perspective.shape, orthographic.shape)
        self.assertGreater(int(perspective.sum()), 0)
        self.assertGreater(int(orthographic.sum()), 0)
        self.assertFalse(np.array_equal(perspective, orthographic))

    def test_fallback_rgb_render_keeps_upright_vertical_order(self) -> None:
        points = np.array(
            [
                [0.0, 0.0, 1.25],
                [0.0, 0.0, 0.75],
            ],
            dtype=np.float32,
        )
        colors = np.array(
            [
                [0, 0, 255],
                [0, 255, 0],
            ],
            dtype=np.uint8,
        )
        view_config = {
            "center": np.array([0.0, 0.0, 1.0], dtype=np.float32),
            "camera_position": np.array([0.0, -2.0, 1.0], dtype=np.float32),
            "up": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        }
        image = render_point_cloud_fallback(
            points,
            colors,
            view_config=view_config,
            render_mode="color_by_rgb",
            scalar_bounds={"height": (0.0, 2.0), "depth": (0.0, 3.0)},
            width=240,
            height=180,
            point_radius_px=1,
            supersample_scale=1,
            projection_mode="orthographic",
            ortho_scale=1.0,
        )
        red_rows = np.where(np.all(image == np.array([0, 0, 255], dtype=np.uint8), axis=2))[0]
        green_rows = np.where(np.all(image == np.array([0, 255, 0], dtype=np.uint8), axis=2))[0]
        self.assertGreater(len(red_rows), 0)
        self.assertGreater(len(green_rows), 0)
        self.assertLess(float(np.mean(red_rows)), float(np.mean(green_rows)))


if __name__ == "__main__":
    unittest.main()
