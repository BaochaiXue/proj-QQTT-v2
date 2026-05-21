from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.harness.summarize_demo31_tapnextpp_profiles import render_markdown, summarize_profile


class Demo31TapNextPPProfileSummaryTest(unittest.TestCase):
    def _write_profile(
        self,
        directory: Path,
        *,
        filename: str,
        query_count: int,
        execution_mode: str,
        rendered_groups: int = 12,
    ) -> Path:
        profile = directory / filename
        profile.write_text(
            json.dumps(
                {
                    "contract": {
                        "tracking_backend_execution_mode": execution_mode,
                        "tracking_query_count_requested": str(query_count),
                    },
                    "summary": {
                        "tracker_publish_fps": 20.0,
                        "tracker_group_wall_ms_p50": 21.0,
                        "tracker_group_wall_ms_p95": 25.0,
                        "tracker_model_ms_sum_per_group_p50": 18.0,
                        "tracker_model_ms_sum_per_group_p95": 23.0,
                        "tracker_model_ms_max_per_group_p50": 7.0,
                        "tracker_model_ms_max_per_group_p95": 9.0,
                        "per_camera_model_ms_p50_by_camera": {"0": 6.0, "1": 7.0, "2": 5.0},
                        "model_calls_per_group": 1 if execution_mode == "batch-views" else 3,
                        "model_instances_expected": 1 if execution_mode == "batch-views" else 3,
                        "model_instances_actual": 1 if execution_mode == "batch-views" else 3,
                        "query_count_per_camera": query_count,
                        "total_query_count_across_views": query_count * 3,
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        profile.with_name(f"{profile.stem}_shared_runtime.json").write_text(
            json.dumps({"summary_after_warmup": {"render_fps": 30.0, "rendered_groups": rendered_groups}}) + "\n",
            encoding="utf-8",
        )
        return profile

    def test_summary_marks_q1365_as_four_thousand_total_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profile = self._write_profile(
                Path(tmp),
                filename="batch_views_q1365_live_45s.json",
                query_count=1365,
                execution_mode="batch-views",
            )

            row = summarize_profile(profile)
            markdown = render_markdown([row])

        self.assertEqual(row["query_count_per_camera"], 1365)
        self.assertEqual(row["total_query_count_across_views"], 4095)
        self.assertEqual(row["target_class"], "~4000_total_target")
        self.assertTrue(row["valid_rendered_profile"])
        self.assertIn("q1365/view", markdown)
        self.assertIn("4095", markdown)

    def test_summary_marks_q4096_as_stress_test(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profile = self._write_profile(
                Path(tmp),
                filename="serial_q4096_live_45s.json",
                query_count=4096,
                execution_mode="serial",
            )

            row = summarize_profile(profile)
            markdown = render_markdown([row])

        self.assertEqual(row["query_count_per_camera"], 4096)
        self.assertEqual(row["total_query_count_across_views"], 12288)
        self.assertEqual(row["target_class"], "stress_12288_total")
        self.assertIn("q4096/view", markdown)
        self.assertIn("12288", markdown)

    def test_summary_reports_invalid_rendered_profile_when_no_groups_rendered(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profile = self._write_profile(
                Path(tmp),
                filename="serial_q1365_live_45s.json",
                query_count=1365,
                execution_mode="serial",
                rendered_groups=0,
            )

            row = summarize_profile(profile)

        self.assertFalse(row["valid_rendered_profile"])
        self.assertEqual(row["tracker_group_wall_ms_p50"], 21.0)
        self.assertEqual(row["model_calls_per_group"], 3)


if __name__ == "__main__":
    unittest.main()
