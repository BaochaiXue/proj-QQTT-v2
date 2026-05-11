from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


class Demo3TrackingBackendAvailabilitySmokeTest(unittest.TestCase):
    def test_stack_probe_writes_json_and_markdown(self) -> None:
        from scripts.harness.experiments.check_demo3_tracking_backend_stack import parse_args, build_report, main

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            args = parse_args(["--output-json", str(root / "stack.json"), "--output-md", str(root / "stack.md")])
            report = build_report(args)
            self.assertIn("system", report)
            self.assertIn("backends", report)
            self.assertIn("nvofa", report["backends"])
            self.assertEqual(main(["--output-json", str(root / "stack.json"), "--output-md", str(root / "stack.md")]), 0)
            self.assertTrue((root / "stack.json").is_file())
            self.assertTrue((root / "stack.md").is_file())

    def test_auto_highperf_probe_only_does_not_require_optional_backends(self) -> None:
        from scripts.harness.experiments.run_demo3_tracking_backend_benchmark import parse_args, run_benchmark

        with tempfile.TemporaryDirectory() as tmp_dir:
            args = parse_args(
                [
                    "--case-root",
                    tmp_dir,
                    "--output-root",
                    str(Path(tmp_dir) / "out"),
                    "--backends",
                    "auto_highperf",
                    "--install-probe-only",
                ]
            )
            summary = run_benchmark(args)
            self.assertEqual(summary["rows"], [])
            self.assertIn("cotracker3_online", summary["availability"])


if __name__ == "__main__":
    unittest.main()
