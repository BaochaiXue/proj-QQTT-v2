from __future__ import annotations

import unittest

import numpy as np

from data_process.depth_backends.radius_outlier_filter import (
    FFS_RADIUS_OUTLIER_FILTER_ARCHIVE_POLICY,
    FFS_RADIUS_OUTLIER_FILTER_MODE,
    apply_ffs_radius_outlier_filter_float_m,
    apply_ffs_radius_outlier_filter_u16,
    build_ffs_radius_outlier_filter_contract,
)


class FfsRadiusOutlierFilterSmokeTest(unittest.TestCase):
    def test_float_filter_removes_single_isolated_depth_pixel(self) -> None:
        depth_m = np.ones((11, 11), dtype=np.float32)
        depth_m[0, 0] = 0.1
        k_color = np.array([[4000.0, 0.0, 5.0], [0.0, 4000.0, 5.0], [0.0, 0.0, 1.0]], dtype=np.float32)

        filtered_depth_m, stats = apply_ffs_radius_outlier_filter_float_m(
            depth_m,
            K_color=k_color,
            radius_m=0.01,
            nb_points=40,
        )

        self.assertEqual(filtered_depth_m.shape, depth_m.shape)
        self.assertEqual(filtered_depth_m.dtype, np.float32)
        self.assertEqual(float(filtered_depth_m[0, 0]), 0.0)
        self.assertAlmostEqual(float(filtered_depth_m[5, 5]), 1.0, places=6)
        self.assertGreaterEqual(int(stats["outlier_pixel_count"]), 1)
        self.assertEqual(stats["inlier_pixel_count"], 120)
        self.assertEqual(stats["mode"], FFS_RADIUS_OUTLIER_FILTER_MODE)
        self.assertEqual(stats["archive_policy"], FFS_RADIUS_OUTLIER_FILTER_ARCHIVE_POLICY)

    def test_u16_wrapper_preserves_invalid_zero_semantics(self) -> None:
        depth_m = np.ones((11, 11), dtype=np.float32)
        depth_m[0, 0] = 0.0
        depth_m[0, 1] = 0.1
        k_color = np.array([[4000.0, 0.0, 5.0], [0.0, 4000.0, 5.0], [0.0, 0.0, 1.0]], dtype=np.float32)

        filtered_u16, filtered_depth_m, stats = apply_ffs_radius_outlier_filter_u16(
            depth_m,
            K_color=k_color,
            depth_scale_m_per_unit=0.001,
            radius_m=0.01,
            nb_points=40,
        )

        self.assertEqual(filtered_u16.shape, depth_m.shape)
        self.assertEqual(filtered_u16.dtype, np.uint16)
        self.assertEqual(filtered_depth_m.shape, depth_m.shape)
        self.assertEqual(float(filtered_depth_m[0, 0]), 0.0)
        self.assertEqual(int(filtered_u16[0, 0]), 0)
        self.assertEqual(float(filtered_depth_m[0, 1]), 0.0)
        self.assertEqual(int(filtered_u16[0, 1]), 0)
        self.assertGreater(int(filtered_u16[5, 5]), 0)
        self.assertEqual(stats["outlier_pixel_count"], 1)

    def test_contract_builder_matches_new_metadata_shape(self) -> None:
        contract = build_ffs_radius_outlier_filter_contract(radius_m=0.01, nb_points=40)

        self.assertEqual(
            contract,
            {
                "mode": FFS_RADIUS_OUTLIER_FILTER_MODE,
                "radius_m": 0.01,
                "nb_points": 40,
                "archive_policy": FFS_RADIUS_OUTLIER_FILTER_ARCHIVE_POLICY,
            },
        )


if __name__ == "__main__":
    unittest.main()
