from __future__ import annotations

import unittest

import numpy as np

from data_process.depth_backends.ffs_defaults import DEFAULT_FFS_TRT_BATCH3_TWO_STAGE_MODEL_DIR
from scripts.ffs_trt import build_batch3_4090_engine, build_batch3_5090_engine
from scripts.ffs_trt.validate_batch3_4090_engine import (
    camera_order_diagonal_pass,
    measured_window_indices,
    summarize_values,
)


class FfsTrtBatch3ScriptsTest(unittest.TestCase):
    def test_summarize_values_returns_expected_percentiles(self) -> None:
        summary = summarize_values([1, 2, 3, 4, 100])
        self.assertEqual(summary["min"], 1.0)
        self.assertEqual(summary["max"], 100.0)
        self.assertAlmostEqual(summary["avg"], 22.0)
        self.assertAlmostEqual(summary["p50"], 3.0)
        self.assertGreater(summary["p90"], summary["p50"])
        self.assertGreater(summary["p99"], summary["p95"])

    def test_camera_order_diff_matrix_diagonal_pass(self) -> None:
        matrix = np.array(
            [
                [0.1, 2.0, 3.0],
                [2.0, 0.2, 3.0],
                [3.0, 2.0, 0.3],
            ],
            dtype=np.float32,
        )
        self.assertTrue(camera_order_diagonal_pass(matrix))

    def test_camera_order_diff_matrix_swapped_fail(self) -> None:
        matrix = np.array(
            [
                [2.0, 0.1, 3.0],
                [0.2, 2.0, 3.0],
                [3.0, 2.0, 0.3],
            ],
            dtype=np.float32,
        )
        self.assertFalse(camera_order_diagonal_pass(matrix))

    def test_measured_window_excludes_warmup(self) -> None:
        self.assertEqual(measured_window_indices(total_count=10, warmup_kits=2, measure_kits=4), [2, 3, 4, 5])
        with self.assertRaisesRegex(ValueError, "warmup\\+measure"):
            measured_window_indices(total_count=5, warmup_kits=2, measure_kits=4)

    def test_build_script_parser_defaults_batch_size_3(self) -> None:
        args = build_batch3_4090_engine.build_parser().parse_args([])
        self.assertEqual(args.batch_size, 3)

    def test_5090_build_script_uses_isolated_batch3_output_dir(self) -> None:
        args = build_batch3_5090_engine.build_parser().parse_args([])
        self.assertEqual(args.batch_size, 3)
        self.assertEqual(args.out_dir, DEFAULT_FFS_TRT_BATCH3_TWO_STAGE_MODEL_DIR)
        self.assertTrue(str(args.out_dir).endswith("_batch3"))

    def test_existing_wsl_verify_script_default_batch_size_1(self) -> None:
        from pathlib import Path

        script = Path("scripts/harness/verify_ffs_tensorrt_wsl.py").read_text(encoding="utf-8")
        self.assertIn('parser.add_argument("--batch_size", type=int, default=1)', script)


if __name__ == "__main__":
    unittest.main()
