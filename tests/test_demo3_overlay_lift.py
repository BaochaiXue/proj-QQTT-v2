from __future__ import annotations

import unittest

import numpy as np

from qqtt.demo.tracking_overlay_render import lift_tracks_yx_to_world


class Demo3OverlayLiftTest(unittest.TestCase):
    def test_yx_tracks_index_depth_and_apply_c2w(self) -> None:
        depth = np.zeros((4, 5), dtype=np.uint16)
        depth[2, 3] = 1000
        tracks_yx = np.array([[2.0, 3.0]], dtype=np.float32)
        visibility = np.array([1.0], dtype=np.float32)
        K = np.eye(3, dtype=np.float32)
        c2w = np.eye(4, dtype=np.float32)
        c2w[0, 3] = 10.0

        lifted = lift_tracks_yx_to_world(
            tracks_yx=tracks_yx,
            visibility=visibility,
            depth=depth,
            intrinsics=K,
            c2w=c2w,
            depth_scale_m_per_unit=0.001,
        )

        np.testing.assert_allclose(lifted.points_world, np.array([[13.0, 2.0, 1.0]], dtype=np.float32))
        np.testing.assert_array_equal(lifted.source_indices, np.array([0]))

    def test_fractional_tracks_use_semantic_projection_grid_pixel(self) -> None:
        depth = np.full((2, 2), 2000, dtype=np.uint16)
        lifted = lift_tracks_yx_to_world(
            tracks_yx=np.array([[0.49, 1.49]], dtype=np.float32),
            visibility=np.array([1.0], dtype=np.float32),
            depth=depth,
            intrinsics=np.eye(3, dtype=np.float32),
            c2w=np.eye(4, dtype=np.float32),
            depth_scale_m_per_unit=0.001,
        )

        np.testing.assert_allclose(lifted.points_world, np.array([[2.0, 0.0, 2.0]], dtype=np.float32))

    def test_invisible_tracks_are_skipped(self) -> None:
        depth = np.full((3, 3), 1000, dtype=np.uint16)
        lifted = lift_tracks_yx_to_world(
            tracks_yx=np.array([[1.0, 1.0]], dtype=np.float32),
            visibility=np.array([0.0], dtype=np.float32),
            depth=depth,
            intrinsics=np.eye(3, dtype=np.float32),
            c2w=np.eye(4, dtype=np.float32),
        )

        self.assertEqual(lifted.points_world.shape, (0, 3))
        self.assertFalse(lifted.valid_mask[0])

    def test_invalid_depth_is_skipped(self) -> None:
        depth = np.zeros((3, 3), dtype=np.uint16)
        lifted = lift_tracks_yx_to_world(
            tracks_yx=np.array([[1.0, 1.0]], dtype=np.float32),
            visibility=np.array([1.0], dtype=np.float32),
            depth=depth,
            intrinsics=np.eye(3, dtype=np.float32),
            c2w=np.eye(4, dtype=np.float32),
        )

        self.assertEqual(lifted.points_world.shape, (0, 3))
        self.assertFalse(lifted.valid_mask[0])

    def test_mask_rejects_tracks_outside_semantic_region(self) -> None:
        depth = np.full((3, 3), 1000, dtype=np.uint16)
        mask = np.zeros((3, 3), dtype=np.uint8)
        lifted = lift_tracks_yx_to_world(
            tracks_yx=np.array([[1.0, 1.0]], dtype=np.float32),
            visibility=np.array([1.0], dtype=np.float32),
            depth=depth,
            intrinsics=np.eye(3, dtype=np.float32),
            c2w=np.eye(4, dtype=np.float32),
            mask=mask,
        )

        self.assertEqual(lifted.points_world.shape, (0, 3))
        self.assertFalse(lifted.valid_mask[0])


if __name__ == "__main__":
    unittest.main()
