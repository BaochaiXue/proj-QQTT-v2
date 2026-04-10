from __future__ import annotations

import unittest

import numpy as np

from data_process.visualization.face_quality import compute_patch_plane_metrics


class PlaneFitMetricComputationTest(unittest.TestCase):
    def test_plane_metrics_are_small_on_clean_plane(self) -> None:
        depth = np.full((32, 32), 1.2, dtype=np.float32)
        K = np.array([[420.0, 0.0, 16.0], [0.0, 420.0, 16.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        metrics = compute_patch_plane_metrics(depth, K, (4, 4, 28, 28))
        self.assertGreater(metrics["valid_depth_ratio"], 0.9)
        self.assertLess(metrics["plane_fit_rmse_mm"], 0.02)
        self.assertLess(metrics["mad_mm"], 0.02)
        self.assertLess(metrics["p90_abs_residual_mm"], 0.02)

    def test_plane_metrics_increase_with_noise(self) -> None:
        depth = np.full((32, 32), 1.2, dtype=np.float32)
        rng = np.random.default_rng(0)
        depth[4:28, 4:28] += rng.normal(0.0, 0.01, size=(24, 24)).astype(np.float32)
        K = np.array([[420.0, 0.0, 16.0], [0.0, 420.0, 16.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        metrics = compute_patch_plane_metrics(depth, K, (4, 4, 28, 28))
        self.assertGreater(metrics["plane_fit_rmse_mm"], 1.0)
        self.assertGreater(metrics["p90_abs_residual_mm"], 1.0)


if __name__ == "__main__":
    unittest.main()
