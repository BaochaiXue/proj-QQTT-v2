from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from data_process.depth_backends.benchmarking import (
    build_tradeoff_summary,
    compute_reference_depth_metrics,
    expand_benchmark_configs,
    summarize_latency_samples_ms,
)


class FfsBenchmarkingSmokeTest(unittest.TestCase):
    def test_expand_benchmark_configs_preserves_grid_order(self) -> None:
        configs = expand_benchmark_configs(
            model_paths=[
                Path("C:/models/23-36-37/model_best_bp2_serialize.pth"),
                Path("C:/models/20-30-48/model_best_bp2_serialize.pth"),
            ],
            scales=[1.0, 0.5],
            valid_iters_values=[8, 4],
            max_disp_values=[192],
        )

        self.assertEqual(len(configs), 8)
        self.assertEqual(configs[0].config_id, "23-36-37_scale1_iters8_disp192")
        self.assertEqual(configs[1].config_id, "23-36-37_scale1_iters4_disp192")
        self.assertEqual(configs[2].config_id, "23-36-37_scale0.5_iters8_disp192")
        self.assertEqual(configs[-1].config_id, "20-30-48_scale0.5_iters4_disp192")

    def test_summarize_latency_samples_ms_reports_fps(self) -> None:
        summary = summarize_latency_samples_ms([20.0, 25.0, 30.0])

        self.assertEqual(summary["sample_count"], 3)
        self.assertAlmostEqual(summary["latency_mean_ms"], 25.0, places=6)
        self.assertAlmostEqual(summary["fps_from_mean"], 40.0, places=6)
        self.assertAlmostEqual(summary["latency_p90_ms"], 29.0, places=6)

    def test_reference_depth_metrics_use_overlap_only(self) -> None:
        reference = np.array([[1.0, 0.0], [2.0, 3.0]], dtype=np.float32)
        candidate = np.array([[1.1, 4.0], [0.0, 2.7]], dtype=np.float32)

        metrics = compute_reference_depth_metrics(reference, candidate)

        self.assertAlmostEqual(metrics["reference_valid_ratio"], 0.75, places=6)
        self.assertAlmostEqual(metrics["candidate_valid_ratio"], 0.75, places=6)
        self.assertAlmostEqual(metrics["overlap_valid_ratio"], 0.5, places=6)
        self.assertAlmostEqual(metrics["median_abs_depth_diff_m"], 0.2, places=6)
        self.assertAlmostEqual(metrics["p90_abs_depth_diff_m"], 0.28, places=6)

    def test_reference_depth_metrics_resize_candidate_when_scale_changes(self) -> None:
        reference = np.array(
            [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]],
            dtype=np.float32,
        )
        candidate = np.array([[1.0, 3.0]], dtype=np.float32)

        metrics = compute_reference_depth_metrics(reference, candidate)

        self.assertAlmostEqual(metrics["overlap_valid_ratio"], 1.0, places=6)
        self.assertAlmostEqual(metrics["median_abs_depth_diff_m"], 0.5, places=6)

    def test_build_tradeoff_summary_prefers_reference_like_config_within_target(self) -> None:
        results = [
            {
                "config": {"config_id": "slow_best"},
                "latency_summary": {"latency_mean_ms": 80.0, "latency_p90_ms": 82.0, "fps_from_mean": 12.5},
                "reference_metrics": {"median_abs_depth_diff_m": 0.0, "p90_abs_depth_diff_m": 0.0, "overlap_valid_ratio": 1.0},
            },
            {
                "config": {"config_id": "fast_noisy"},
                "latency_summary": {"latency_mean_ms": 30.0, "latency_p90_ms": 32.0, "fps_from_mean": 33.3},
                "reference_metrics": {"median_abs_depth_diff_m": 0.010, "p90_abs_depth_diff_m": 0.050, "overlap_valid_ratio": 0.95},
            },
            {
                "config": {"config_id": "fast_cleaner"},
                "latency_summary": {"latency_mean_ms": 35.0, "latency_p90_ms": 36.0, "fps_from_mean": 28.6},
                "reference_metrics": {"median_abs_depth_diff_m": 0.002, "p90_abs_depth_diff_m": 0.010, "overlap_valid_ratio": 0.98},
            },
        ]

        summary = build_tradeoff_summary(results, target_fps_values=[25.0, 35.0])

        self.assertEqual(summary["fastest_overall"]["config_id"], "fast_noisy")
        self.assertEqual(summary["most_reference_like"]["config_id"], "slow_best")
        self.assertEqual(summary["targets"]["25.0"]["config_id"], "fast_cleaner")
        self.assertIsNone(summary["targets"]["35.0"])


if __name__ == "__main__":
    unittest.main()
