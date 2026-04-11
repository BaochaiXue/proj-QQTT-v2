from __future__ import annotations

import unittest

import numpy as np

from data_process.depth_backends import apply_remove_invisible_mask


class FfsRemoveInvisibleMaskSmokeTest(unittest.TestCase):
    def test_matches_upstream_overlap_invalidation_semantics(self) -> None:
        disparity_raw = np.array(
            [
                [2.0, 2.0, 2.0, 2.0, 2.0, -1.0],
                [0.0, 1.0, 5.0, 0.0, np.inf, 2.0],
            ],
            dtype=np.float32,
        )

        masked, stats = apply_remove_invisible_mask(disparity_raw)

        self.assertTrue(np.isinf(masked[0, 0]))
        self.assertTrue(np.isinf(masked[0, 1]))
        self.assertEqual(float(masked[0, 2]), 2.0)
        self.assertEqual(float(masked[1, 1]), 1.0)
        self.assertTrue(np.isinf(masked[1, 2]))
        self.assertEqual(int(stats["remove_invisible_pixel_count"]), 3)
        self.assertAlmostEqual(float(stats["remove_invisible_ratio"]), 0.25)


if __name__ == "__main__":
    unittest.main()
