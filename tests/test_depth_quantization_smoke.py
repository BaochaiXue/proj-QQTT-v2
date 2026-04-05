from __future__ import annotations

import unittest

import numpy as np

from scripts.harness.ffs_geometry import quantize_depth_with_invalid_zero


class DepthQuantizationSmokeTest(unittest.TestCase):
    def test_invalid_stays_zero_and_valid_quantizes(self) -> None:
        depth = np.array(
            [
                [0.0, 0.5],
                [np.nan, 1.234],
            ],
            dtype=np.float32,
        )
        encoded = quantize_depth_with_invalid_zero(depth, 0.001)

        self.assertEqual(encoded.dtype, np.uint16)
        self.assertEqual(int(encoded[0, 0]), 0)
        self.assertEqual(int(encoded[1, 0]), 0)
        self.assertEqual(int(encoded[0, 1]), 500)
        self.assertEqual(int(encoded[1, 1]), 1234)


if __name__ == "__main__":
    unittest.main()
