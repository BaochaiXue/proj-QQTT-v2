from __future__ import annotations

import unittest

import numpy as np

from qqtt.env.camera.realsense.depth_postprocess import (
    FFS_NATIVE_LIKE_DEPTH_POSTPROCESS_DIR,
    FFS_NATIVE_LIKE_DEPTH_POSTPROCESS_FLOAT_DIR,
    FFS_NATIVE_LIKE_DEPTH_POSTPROCESS_ON_THE_FLY_SUFFIX,
    NATIVE_DEPTH_POSTPROCESS_CONTRACT,
    apply_ffs_native_like_depth_postprocess_float_m,
    apply_ffs_native_like_depth_postprocess_u16,
    native_depth_postprocess_contract,
)


class FfsNativeLikeDepthPostprocessSmokeTest(unittest.TestCase):
    def test_contract_matches_current_realsense_filter_chain(self) -> None:
        contract = native_depth_postprocess_contract()
        self.assertEqual(contract["mode"], "native_depth_postprocess")
        self.assertEqual(
            contract["chain"],
            ("depth_to_disparity", "spatial_filter", "temporal_filter", "disparity_to_depth"),
        )
        self.assertEqual(contract["spatial_filter"]["filter_magnitude"], 5)
        self.assertEqual(contract["spatial_filter"]["filter_smooth_alpha"], 0.75)
        self.assertEqual(contract["spatial_filter"]["filter_smooth_delta"], 1)
        self.assertEqual(contract["spatial_filter"]["holes_fill"], 1)
        self.assertEqual(contract["temporal_filter"]["filter_smooth_alpha"], 0.75)
        self.assertEqual(contract["temporal_filter"]["filter_smooth_delta"], 1)
        self.assertEqual(contract, NATIVE_DEPTH_POSTPROCESS_CONTRACT)
        self.assertEqual(FFS_NATIVE_LIKE_DEPTH_POSTPROCESS_DIR, "depth_ffs_native_like_postprocess")
        self.assertEqual(FFS_NATIVE_LIKE_DEPTH_POSTPROCESS_FLOAT_DIR, "depth_ffs_native_like_postprocess_float_m")
        self.assertEqual(FFS_NATIVE_LIKE_DEPTH_POSTPROCESS_ON_THE_FLY_SUFFIX, "ffs_native_like_postprocess")

    def test_numpy_paths_preserve_shape_dtype_and_invalid_zero_semantics(self) -> None:
        zero_depth = np.zeros((3, 4), dtype=np.uint16)
        filtered_zero = apply_ffs_native_like_depth_postprocess_u16(
            zero_depth,
            depth_scale_m_per_unit=0.001,
            fps=30,
            frame_number=1,
        )
        self.assertEqual(filtered_zero.shape, zero_depth.shape)
        self.assertEqual(filtered_zero.dtype, np.uint16)
        self.assertTrue(np.array_equal(filtered_zero, zero_depth))

        depth_m = np.full((3, 4), 1.25, dtype=np.float32)
        depth_m[0, 0] = 0.0
        filtered_u16, filtered_m = apply_ffs_native_like_depth_postprocess_float_m(
            depth_m,
            depth_scale_m_per_unit=0.001,
            fps=30,
            frame_number=2,
        )
        self.assertEqual(filtered_u16.shape, depth_m.shape)
        self.assertEqual(filtered_u16.dtype, np.uint16)
        self.assertEqual(filtered_m.shape, depth_m.shape)
        self.assertEqual(filtered_m.dtype, np.float32)
        self.assertGreater(int((filtered_u16 > 0).sum()), 0)
        self.assertGreater(float(filtered_m[1, 1]), 0.0)


if __name__ == "__main__":
    unittest.main()
