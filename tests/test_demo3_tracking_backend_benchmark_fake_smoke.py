from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.visualization_test_utils import make_visualization_case


class Demo3TrackingBackendBenchmarkFakeSmokeTest(unittest.TestCase):
    def test_fake_backend_benchmark_still_writes_results_csv(self) -> None:
        from scripts.harness.experiments.run_demo3_tracking_backend_benchmark import parse_args, run_benchmark

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            case_root = root / "case"
            make_visualization_case(case_root, frame_num=2)
            args = parse_args(
                [
                    "--case-root",
                    str(case_root),
                    "--output-root",
                    str(root / "out"),
                    "--backends",
                    "fake",
                    "--cameras",
                    "0,1,2",
                    "--num-query-points",
                    "3",
                    "--frames",
                    "2",
                ]
            )
            summary = run_benchmark(args)
            output_dir = Path(summary["output_dir"])
            self.assertTrue((output_dir / "results.csv").is_file())
            self.assertTrue((output_dir / "profile.json").is_file())
            self.assertTrue((output_dir / "profile.md").is_file())
            self.assertTrue((output_dir / "fake" / "points_3" / "cam0.npz").is_file())
            self.assertIn("total_wall_ms", summary["profile"])

    def test_explicit_unavailable_backend_can_be_required(self) -> None:
        from scripts.harness.experiments.run_demo3_tracking_backend_benchmark import parse_args, run_benchmark

        with tempfile.TemporaryDirectory() as tmp_dir:
            availability_json = Path(tmp_dir) / "availability.json"
            availability_json.write_text(
                '{"tapnext": {"backend": "tapnext", "available": false, "reason": "forced unavailable"}}\n',
                encoding="utf-8",
            )
            args = parse_args(
                [
                    "--case-root",
                    tmp_dir,
                    "--output-root",
                    str(Path(tmp_dir) / "out"),
                    "--backends",
                    "tapnext",
                    "--backend-availability-json",
                    str(availability_json),
                    "--install-probe-only",
                    "--require-available",
                ]
            )
            with self.assertRaises(RuntimeError):
                run_benchmark(args)


if __name__ == "__main__":
    unittest.main()
