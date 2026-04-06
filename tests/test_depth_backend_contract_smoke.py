from __future__ import annotations

import unittest

import numpy as np

from data_process.depth_backends.geometry import align_depth_to_color, quantize_depth_with_invalid_zero


class DepthBackendContractSmokeTest(unittest.TestCase):
    def test_color_aligned_quantized_depth_preserves_invalid_zero(self) -> None:
        depth_ir = np.array([[1.0, 0.0], [1.5, 2.0]], dtype=np.float32)
        K = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        T = np.eye(4, dtype=np.float32)
        depth_color = align_depth_to_color(depth_ir, K, T, K, output_shape=(2, 2))
        depth_quantized = quantize_depth_with_invalid_zero(depth_color, 0.001)

        self.assertEqual(depth_quantized.dtype, np.uint16)
        self.assertEqual(int(depth_quantized[0, 1]), 0)
        self.assertGreater(int(depth_quantized[0, 0]), 0)


if __name__ == "__main__":
    unittest.main()
