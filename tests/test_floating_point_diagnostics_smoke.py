from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from data_process.visualization.floating_point_diagnostics import (
    assign_primary_cause,
    classify_cross_view_relation,
    detect_radius_outlier_indices,
    load_frame_camera_clouds_with_metadata,
)
from data_process.visualization.io_case import load_case_metadata
from tests.visualization_test_utils import make_visualization_case


class FloatingPointDiagnosticsSmokeTest(unittest.TestCase):
    def test_radius_outlier_filter_removes_isolated_point(self) -> None:
        rng = np.random.default_rng(0)
        dense_cluster = rng.normal(loc=0.0, scale=0.001, size=(64, 3)).astype(np.float32)
        isolated_point = np.array([[0.10, 0.10, 0.10]], dtype=np.float32)
        points = np.concatenate([dense_cluster, isolated_point], axis=0)

        result = detect_radius_outlier_indices(points, radius_m=0.01, nb_points=40)

        self.assertEqual(result["outlier_indices"].tolist(), [64])
        self.assertEqual(len(result["inlier_indices"]), 64)

    def test_loader_preserves_source_pixels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            case_dir = Path(tmp_dir) / "case"
            make_visualization_case(case_dir, frame_num=1)
            metadata = load_case_metadata(case_dir)

            camera_clouds = load_frame_camera_clouds_with_metadata(
                case_dir=case_dir,
                metadata=metadata,
                frame_idx=0,
                depth_source="realsense",
                use_float_ffs_depth_when_available=False,
            )

            cam0 = camera_clouds[0]
            depth = np.load(case_dir / "depth" / "0" / "0.npy")
            valid = depth > 0
            expected_pixels = {
                (int(x), int(y))
                for y, x in np.argwhere(valid)
            }
            observed_pixels = {
                (int(pixel[0]), int(pixel[1]))
                for pixel in np.asarray(cam0["source_pixel_uv"], dtype=np.int32)
            }

            self.assertEqual(observed_pixels, expected_pixels)
            self.assertEqual(len(cam0["source_depth_m"]), len(expected_pixels))

    def test_primary_cause_priority_prefers_occlusion_then_edge_then_dark(self) -> None:
        self.assertEqual(
            assign_primary_cause(
                occluded_in_other_views=True,
                near_edge=True,
                dark_region=True,
            ),
            "occlusion",
        )
        self.assertEqual(
            assign_primary_cause(
                occluded_in_other_views=False,
                near_edge=True,
                dark_region=True,
            ),
            "edge",
        )
        self.assertEqual(
            assign_primary_cause(
                occluded_in_other_views=False,
                near_edge=False,
                dark_region=True,
            ),
            "dark",
        )
        self.assertEqual(
            assign_primary_cause(
                occluded_in_other_views=False,
                near_edge=False,
                dark_region=False,
            ),
            "other",
        )

    def test_cross_view_occlusion_marks_closer_depth_as_occluding(self) -> None:
        k_color = np.array(
            [[1.0, 0.0, 2.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        camera_clouds = [
            {
                "camera_idx": 0,
                "c2w": np.eye(4, dtype=np.float32),
                "K_color": k_color,
                "depth_m": np.ones((5, 5), dtype=np.float32),
            },
            {
                "camera_idx": 1,
                "c2w": np.eye(4, dtype=np.float32),
                "K_color": k_color,
                "depth_m": np.full((5, 5), 0.0, dtype=np.float32),
            },
        ]
        camera_clouds[1]["depth_m"][2, 2] = 0.8

        result = classify_cross_view_relation(
            np.array([0.0, 0.0, 1.0], dtype=np.float32),
            source_camera_idx=0,
            camera_clouds=camera_clouds,
            occlusion_depth_tol_m=0.02,
            occlusion_depth_tol_ratio=0.03,
        )

        self.assertFalse(result["cross_view_supported"])
        self.assertTrue(result["occluded_in_other_views"])
        self.assertEqual(result["support_count_other_views"], 0)


if __name__ == "__main__":
    unittest.main()
