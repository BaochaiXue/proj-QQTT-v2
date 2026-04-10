from __future__ import annotations

import unittest

import numpy as np

from data_process.depth_backends.fast_foundation_stereo import build_disparity_products


class RawClippedDisparityAuditPathTest(unittest.TestCase):
    def test_audit_mode_keeps_raw_disparity_while_production_disparity_stays_clipped(self) -> None:
        result = build_disparity_products(
            np.array([[2.0, -3.0], [0.0, 4.0]], dtype=np.float32),
            K_ir_left=np.array([[400.0, 0.0, 100.0], [0.0, 400.0, 80.0], [0.0, 0.0, 1.0]], dtype=np.float32),
            baseline_m=0.095,
            scale=1.0,
            valid_iters=8,
            max_disp=192,
            audit_mode=True,
        )
        self.assertIn("disparity_raw", result)
        self.assertIn("audit_stats", result)
        self.assertEqual(float(result["disparity_raw"][0, 1]), -3.0)
        self.assertEqual(float(result["disparity"][0, 1]), 0.0)
        self.assertGreater(float(result["depth_ir_left_m"][0, 0]), 0.0)
        self.assertEqual(float(result["depth_ir_left_m"][0, 1]), 0.0)


if __name__ == "__main__":
    unittest.main()
