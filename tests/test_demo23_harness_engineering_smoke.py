from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.harness import check_harness_engineering
from scripts.harness import summarize_demo23_failure_packet as packet


class Demo23HarnessEngineeringSmokeTest(unittest.TestCase):
    def test_failure_packet_flags_weak_calibration_and_no_render_metric(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            profile = root / "profile.json"
            summary = root / "summary.json"
            calibration = root / "calibration.json"
            preflight = root / "preflight.json"
            profile.write_text(
                json.dumps(
                    {
                        "demo_version": "demo2.3",
                        "pipeline": "dual-gpu-split",
                        "target_fps": 30.0,
                        "render_mode": "none",
                        "ffs_contract": {
                            "trt_batch_size": 3,
                            "builderOptimizationLevel": 5,
                            "trt_model_dir": "data/experiments/ffs_trt_4090_848x480_pad864_builderopt5_batch3/engines/model_20-30-48_iters_4_res_480x864_batch3",
                            "batch3_isolated_artifact": True,
                        },
                        "summary_after_warmup": {
                            "capture_group_fps": 28.0,
                            "raw_fusion_fps": 13.0,
                            "filter_output_fps": 13.0,
                            "fusion_fps": 13.0,
                            "render_fps": 0.0,
                            "complete_group_ratio": 0.95,
                            "dual_gpu": {
                                "ffs_queue_drops": 1,
                                "edgetam_queue_drops": 2,
                                "stale_depth_drops": 3,
                                "stale_mask_drops": 4,
                                "ffs_worker_period_ms": {"median": 39.0},
                                "edgetam_worker_period_ms": {"median": 54.0},
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            summary.write_text(
                json.dumps(
                    {
                        "final": {"latest_group_id": 7, "object_points": 10, "controller_points": 20},
                        "temporal_grouping": {"skew_ms_p95": 40.0, "skew_ms_max": 60.0, "max_capture_skew_ms": 66.7},
                    }
                ),
                encoding="utf-8",
            )
            calibration.write_text(
                json.dumps(
                    {
                        "mapping_mode": "calibrate-c2w",
                        "debug_identity_c2w": False,
                        "debug_invert_c2w": False,
                        "runtime_serial_numbers": ["a", "b", "c"],
                        "calibration_reference_serials": ["a", "b", "c"],
                    }
                ),
                encoding="utf-8",
            )
            preflight.write_text(
                json.dumps(
                    {
                        "min_charuco_corners": 35,
                        "frames": [
                            {
                                "camera_idx": 0,
                                "serial": "a",
                                "charuco_corner_count": 10,
                                "passes_corner_threshold": False,
                                "passes_error_threshold": False,
                                "reprojection_error": 0.31,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result = packet.build_failure_packet(
                profile_json=profile,
                summary_json=summary,
                calibration_report=calibration,
                calibration_preflight=preflight,
            )

        codes = {item["code"] for item in result["risk_flags"]}
        self.assertNotIn("ffs_contract_not_batch3_opt5", codes)
        self.assertIn("weak_calibration_preflight", codes)
        self.assertIn("temporal_skew_pressure", codes)
        self.assertIn("latest_only_drop_pressure", codes)
        self.assertIn("no_render_deficit_metric", codes)

    def test_markdown_renderer_mentions_contract_and_risks(self) -> None:
        markdown = packet.render_markdown(
            {
                "inputs": {"profile_json": "profile.json"},
                "profile": {
                    "pipeline": "dual-gpu-split",
                    "render_mode": "none",
                    "target_fps": 30,
                    "fps": {"fusion": 13.0},
                    "ffs_contract": {
                        "trt_batch_size": 3,
                        "builderOptimizationLevel": 5,
                        "trt_model_dir": "batch3",
                    },
                },
                "runtime_summary": {"fatal": None, "latest_group_id": 7, "object_points": 10, "controller_points": 20},
                "risk_flags": [{"severity": "high", "code": "weak_calibration_preflight", "message": "weak"}],
            }
        )
        self.assertIn("Demo 2.3 Failure Packet", markdown)
        self.assertIn("dual-gpu-split", markdown)
        self.assertIn("weak_calibration_preflight", markdown)

    def test_harness_engineering_guard_passes(self) -> None:
        self.assertEqual(check_harness_engineering.main(), 0)


if __name__ == "__main__":
    unittest.main()
