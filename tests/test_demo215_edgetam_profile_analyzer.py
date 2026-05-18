from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.demo2_1_5 import analyze_edgetam80_profiles as analyzer


def _profile(stage_values: list[float], *, compiled: bool = True) -> dict:
    return {
        "compile_mode": "vision-reduce-overhead" if compiled else "none",
        "edgetam_graph_output_policy": {"requested": "auto", "effective": "clone" if compiled else "none"},
        "init_profile": {
            "edgetam": {
                "loaders": {
                    "cam0": {
                        "compiled_module_count": 1 if compiled else 0,
                        "compiled_module_names": ["vision_encoder"] if compiled else [],
                        "compiled_module_types": {"vision_encoder": "OptimizedModule"} if compiled else {},
                    },
                    "cam1": {
                        "compiled_module_count": 1 if compiled else 0,
                        "compiled_module_names": ["vision_encoder"] if compiled else [],
                    },
                    "cam2": {
                        "compiled_module_count": 1 if compiled else 0,
                        "compiled_module_names": ["vision_encoder"] if compiled else [],
                    },
                }
            }
        },
        "summary_after_warmup": {
            "complete_mask_group_fps": 12.0,
            "complete_mask_groups": len(stage_values),
            "metrics": {
                "edgetam_stage_wall_ms": analyzer.stats(stage_values),
                "edgetam_cam0_model_ms": analyzer.stats([20.0, 22.0]),
                "edgetam_cam1_model_ms": analyzer.stats([21.0, 23.0]),
                "edgetam_cam2_model_ms": analyzer.stats([19.0, 24.0]),
            },
        },
        "gpu_sampling": {
            "summary_after_warmup": {
                "metrics": {
                    "gpu_util_pct": analyzer.stats([40.0, 50.0, 60.0]),
                }
            }
        },
        "per_group": [],
    }


class Demo215EdgeTamProfileAnalyzerTest(unittest.TestCase):
    def test_stats_computes_percentiles(self) -> None:
        result = analyzer.stats([1, 2, 3, 4, 5])

        self.assertEqual(result["median"], 3.0)
        self.assertAlmostEqual(result["p90"], 4.6)
        self.assertAlmostEqual(result["p95"], 4.8)
        self.assertAlmostEqual(result["p99"], 4.96)

    def test_analyzer_fails_when_no_mode_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "slow.json"
            path.write_text(json.dumps(_profile([100.0, 110.0, 120.0])), encoding="utf-8")

            rows = [analyzer.summarize_profile(path, target_stage_wall_p50_ms=80.0)]

        self.assertFalse(rows[0]["pass_80ms"])

    def test_analyzer_passes_when_one_mode_is_under_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fast.json"
            path.write_text(json.dumps(_profile([60.0, 70.0, 75.0])), encoding="utf-8")

            row = analyzer.summarize_profile(path, target_stage_wall_p50_ms=80.0)

        self.assertTrue(row["pass_80ms"])
        self.assertEqual(row["compiled_module_count"], 3)
        self.assertTrue(row["per_camera_compiled"]["cam0"])

    def test_batch_vision_variant_is_labeled(self) -> None:
        payload = _profile([70.0, 75.0, 80.0])
        payload["init_profile"]["edgetam"]["loaders"] = {
            "shared": {
                "compiled_module_count": 1,
                "compiled_module_names": ["vision_encoder"],
            }
        }
        payload["summary_after_warmup"]["metrics"]["edgetam_batch_vision_total_ms"] = analyzer.stats([12.0, 13.0])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "demo215_edgetam80_batchvision_vision_reduce_overhead_towel_profile.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            row = analyzer.summarize_profile(path, target_stage_wall_p50_ms=80.0)
            md = analyzer.markdown_report([row], target_stage_wall_p50_ms=80.0)

        self.assertEqual(row["variant"], "batch-vision-shared-model")
        self.assertEqual(row["compiled_module_count"], 1)
        self.assertIn("batch-vision-shared-model", md)

    def test_analyzer_marks_empty_stage_samples_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.json"
            path.write_text(json.dumps(_profile([])), encoding="utf-8")

            row = analyzer.summarize_profile(path, target_stage_wall_p50_ms=80.0)
            md = analyzer.markdown_report([row], target_stage_wall_p50_ms=80.0)

        self.assertFalse(row["valid_stage_samples"])
        self.assertFalse(row["pass_80ms"])
        self.assertIn("n/a", md)
        self.assertIn("no valid stage samples", md)

    def test_main_returns_nonzero_when_fail_if_no_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "slow.json"
            path.write_text(json.dumps(_profile([90.0, 100.0, 110.0])), encoding="utf-8")

            code = analyzer.main(["--profiles", str(path), "--fail-if-no-pass"])

        self.assertEqual(code, 1)

    def test_main_writes_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profile = Path(tmp) / "fast.json"
            out_json = Path(tmp) / "report.json"
            out_md = Path(tmp) / "report.md"
            profile.write_text(json.dumps(_profile([50.0, 60.0, 70.0])), encoding="utf-8")

            code = analyzer.main(
                [
                    "--profiles",
                    str(profile),
                    "--output-json",
                    str(out_json),
                    "--output-md",
                    str(out_md),
                    "--fail-if-no-pass",
                ]
            )

            self.assertEqual(code, 0)
            self.assertTrue(out_json.exists())
            self.assertIn("pass", out_md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
