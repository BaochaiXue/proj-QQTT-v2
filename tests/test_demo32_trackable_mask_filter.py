from __future__ import annotations

import unittest

import numpy as np

from qqtt.demo.trackable_mask_filter import (
    TrackableMaskFilterConfig,
    build_standard_filter_trackable_masks_for_camera,
)
from qqtt.demo.three_view_masked_fused_pcd_runtime import (
    POSTPROCESS_ENHANCED_PT,
    POSTPROCESS_NONE,
    POSTPROCESS_PT_FILTER,
)


class Demo32TrackableMaskFilterTest(unittest.TestCase):
    def _config(self, **overrides):
        values = {
            "depth_min_m": 0.1,
            "depth_max_m": 3.0,
            "object_point_control": "none",
            "object_postprocess": POSTPROCESS_NONE,
            "controller_postprocess": POSTPROCESS_NONE,
            "controller_trackable_max_points_per_camera": 4999,
            "seed": 42,
        }
        values.update(overrides)
        return TrackableMaskFilterConfig(**values)

    def test_standard_filter_rejects_invalid_depth_pixels(self) -> None:
        depth = np.array([[1.0, np.nan], [0.0, 2.0]], dtype=np.float32)
        object_mask = np.array([[True, True], [True, False]])
        controller_mask = np.array([[False, True], [False, True]])

        result = build_standard_filter_trackable_masks_for_camera(
            camera_idx=0,
            depth_m=depth,
            object_mask=object_mask,
            controller_mask=controller_mask,
            intrinsics=np.eye(3, dtype=np.float32),
            c2w=np.eye(4, dtype=np.float32),
            config=self._config(),
        )

        expected_object = np.array([[True, False], [False, False]])
        expected_controller = np.array([[False, False], [False, True]])
        np.testing.assert_array_equal(result.object_mask, expected_object)
        np.testing.assert_array_equal(result.controller_mask, expected_controller)
        self.assertEqual(result.stats["depth_valid_object_pixels"], 1)
        self.assertEqual(result.stats["depth_valid_controller_pixels"], 1)

    def test_controller_cap_is_applied_after_filtering(self) -> None:
        depth = np.ones((4, 4), dtype=np.float32)
        object_mask = np.zeros((4, 4), dtype=bool)
        object_mask[0, 0] = True
        controller_mask = np.ones((4, 4), dtype=bool)

        result = build_standard_filter_trackable_masks_for_camera(
            camera_idx=2,
            depth_m=depth,
            object_mask=object_mask,
            controller_mask=controller_mask,
            intrinsics=np.eye(3, dtype=np.float32),
            c2w=np.eye(4, dtype=np.float32),
            config=self._config(controller_trackable_max_points_per_camera=5),
        )

        self.assertEqual(result.stats["controller_trackable_before_cap"], 16)
        self.assertEqual(result.stats["controller_trackable_after_cap"], 5)
        self.assertTrue(result.stats["controller_trackable_cap_applied"])
        self.assertEqual(int(np.count_nonzero(result.controller_mask)), 5)

    def test_object_enhanced_pt_removes_disconnected_floating_island_before_query(self) -> None:
        depth = np.ones((12, 12), dtype=np.float32)
        object_mask = np.zeros((12, 12), dtype=bool)
        object_mask[1:4, 1:4] = True
        object_mask[8:10, 8:10] = True
        controller_mask = np.zeros_like(object_mask)
        intrinsics = np.array([[1000.0, 0.0, 0.0], [0.0, 1000.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

        result = build_standard_filter_trackable_masks_for_camera(
            camera_idx=0,
            depth_m=depth,
            object_mask=object_mask,
            controller_mask=controller_mask,
            intrinsics=intrinsics,
            c2w=np.eye(4, dtype=np.float32),
            config=self._config(
                object_point_control="fixed-cap",
                object_postprocess=POSTPROCESS_ENHANCED_PT,
                controller_postprocess=POSTPROCESS_NONE,
                phystwin_radius_m=0.003,
                phystwin_nb_points=1,
                enhanced_component_voxel_size_m=0.003,
                enhanced_keep_near_main_gap_m=0.001,
            ),
        )

        expected_object = np.zeros_like(object_mask)
        expected_object[1:4, 1:4] = True
        np.testing.assert_array_equal(result.object_mask, expected_object)
        np.testing.assert_array_equal(result.union_mask, expected_object)
        self.assertEqual(result.stats["object_filter"]["mode"], POSTPROCESS_ENHANCED_PT)
        self.assertEqual(result.stats["object_trackable_pixels"], 9)

    def test_controller_pt_filter_keeps_two_separated_valid_components_before_query(self) -> None:
        depth = np.ones((12, 12), dtype=np.float32)
        object_mask = np.zeros((12, 12), dtype=bool)
        controller_mask = np.zeros_like(object_mask)
        controller_mask[1:3, 1:3] = True
        controller_mask[8:10, 8:10] = True
        intrinsics = np.array([[1000.0, 0.0, 0.0], [0.0, 1000.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

        result = build_standard_filter_trackable_masks_for_camera(
            camera_idx=0,
            depth_m=depth,
            object_mask=object_mask,
            controller_mask=controller_mask,
            intrinsics=intrinsics,
            c2w=np.eye(4, dtype=np.float32),
            config=self._config(
                object_point_control="fixed-cap",
                object_postprocess=POSTPROCESS_NONE,
                controller_postprocess=POSTPROCESS_PT_FILTER,
                phystwin_radius_m=0.003,
                phystwin_nb_points=1,
                controller_trackable_max_points_per_camera=16,
            ),
        )

        np.testing.assert_array_equal(result.object_mask, object_mask)
        np.testing.assert_array_equal(result.controller_mask, controller_mask)
        np.testing.assert_array_equal(result.union_mask, controller_mask)
        self.assertEqual(result.stats["controller_filter"]["mode"], POSTPROCESS_PT_FILTER)
        self.assertEqual(result.stats["controller_trackable_after_cap"], 8)


if __name__ == "__main__":
    unittest.main()
