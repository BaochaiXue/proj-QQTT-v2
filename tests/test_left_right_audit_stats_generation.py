from __future__ import annotations

import unittest

import numpy as np

from data_process.depth_backends.ffs_audit import summarize_left_right_audit
from data_process.depth_backends.fast_foundation_stereo import compute_disparity_audit_stats


class LeftRightAuditStatsGenerationTest(unittest.TestCase):
    def test_disparity_stats_capture_positive_and_nonpositive_fractions(self) -> None:
        stats = compute_disparity_audit_stats(
            np.array([[2.0, -1.0], [0.0, 4.0]], dtype=np.float32)
        )
        self.assertAlmostEqual(stats["positive_fraction_of_finite"], 0.5)
        self.assertAlmostEqual(stats["nonpositive_fraction_of_finite"], 0.5)
        self.assertEqual(stats["max_disparity"], 4.0)

    def test_left_right_summary_prefers_higher_positive_and_valid_depth(self) -> None:
        normal = {
            "disparity": np.array([[2.0, 1.0], [0.0, 0.0]], dtype=np.float32),
            "depth_ir_left_m": np.array([[1.0, 2.0], [0.0, 0.0]], dtype=np.float32),
            "audit_stats": compute_disparity_audit_stats(np.array([[2.0, 1.0], [0.0, 0.0]], dtype=np.float32)),
        }
        swapped = {
            "disparity": np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32),
            "depth_ir_left_m": np.zeros((2, 2), dtype=np.float32),
            "audit_stats": compute_disparity_audit_stats(np.array([[-1.0, -2.0], [0.0, 0.0]], dtype=np.float32)),
        }
        summary = summarize_left_right_audit(normal_run=normal, swapped_run=swapped)
        self.assertEqual(summary["plausible_ordering"], "normal")
        self.assertGreater(summary["normal"]["plausibility_score"], summary["swapped"]["plausibility_score"])


if __name__ == "__main__":
    unittest.main()
