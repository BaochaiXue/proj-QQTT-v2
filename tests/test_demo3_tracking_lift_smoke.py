from __future__ import annotations

import unittest

import numpy as np

from qqtt.tracking.lift import lift_tracks_to_world


class Demo3TrackingLiftSmokeTest(unittest.TestCase):
    def test_lift_tracks_uses_yx_depth_indexing(self) -> None:
        tracks_yx = np.array([[10.0, 20.0]], dtype=np.float32)
        visibility = np.array([1.0], dtype=np.float32)
        depth = np.zeros((30, 40), dtype=np.uint16)
        depth[10, 20] = 500
        mask = np.ones((30, 40), dtype=np.uint8)
        K = np.array([[10.0, 0.0, 20.0], [0.0, 10.0, 10.0], [0.0, 0.0, 1.0]], dtype=np.float32)

        lifted = lift_tracks_to_world(
            tracks_yx_t=tracks_yx,
            visibility_t=visibility,
            depth_uint16=depth,
            depth_scale_m_per_unit=0.001,
            mask=mask,
            K=K,
            c2w=np.eye(4, dtype=np.float32),
        )

        self.assertEqual(lifted.points_world.shape, (1, 3))
        np.testing.assert_allclose(lifted.points_world[0], [0.0, 0.0, 0.5], atol=1e-6)

    def test_lift_tracks_rejects_outside_mask_invalid_depth_and_invisible(self) -> None:
        tracks_yx = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [8.0, 8.0]], dtype=np.float32)
        visibility = np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float32)
        depth = np.zeros((5, 5), dtype=np.uint16)
        depth[1, 1] = 1000
        depth[2, 2] = 0
        depth[3, 3] = 1000
        mask = np.zeros((5, 5), dtype=np.uint8)
        mask[1, 1] = 255
        mask[2, 2] = 255
        mask[3, 3] = 255

        lifted = lift_tracks_to_world(
            tracks_yx_t=tracks_yx,
            visibility_t=visibility,
            depth_uint16=depth,
            depth_scale_m_per_unit=0.001,
            mask=mask,
            K=np.eye(3, dtype=np.float32),
            c2w=np.eye(4, dtype=np.float32),
        )

        self.assertEqual(lifted.stats["num_lifted"], 1)
        self.assertEqual(lifted.stats["num_depth_valid"], 1)
        self.assertEqual(lifted.valid_track_mask.tolist(), [True, False, False, False])


if __name__ == "__main__":
    unittest.main()
