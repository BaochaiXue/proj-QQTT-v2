from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from qqtt.env.camera.preflight import evaluate_capture_preflight


class RecordPreflightPolicySmokeTest(unittest.TestCase):
    def _write_probe_results(self, root: Path) -> Path:
        payload = {
            "cases": [
                {
                    "topology_type": "three_camera",
                    "stream_set": "rgb_ir_pair",
                    "serials": ["a", "b", "c"],
                    "width": 848,
                    "height": 480,
                    "fps": 30,
                    "emitter_request": "on",
                    "success": False,
                },
                {
                    "topology_type": "three_camera",
                    "stream_set": "rgbd_ir_pair",
                    "serials": ["a", "b", "c"],
                    "width": 848,
                    "height": 480,
                    "fps": 30,
                    "emitter_request": "on",
                    "success": False,
                },
            ]
        }
        path = root / "probe.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_rgbd_is_supported_without_probe_gate(self) -> None:
        decision = evaluate_capture_preflight(
            capture_mode="rgbd",
            serials=["a", "b", "c"],
            width=848,
            height=480,
            fps=30,
            emitter="on",
        )
        self.assertTrue(decision.allowed_to_record)
        self.assertEqual(decision.operator_status, "supported")

    def test_both_eval_is_blocked_when_probe_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            probe_path = self._write_probe_results(Path(tmp_dir))
            decision = evaluate_capture_preflight(
                capture_mode="both_eval",
                serials=["a", "b", "c"],
                width=848,
                height=480,
                fps=30,
                emitter="on",
                probe_results_path=probe_path,
                probe_results_md_path=probe_path.with_suffix(".md"),
            )
        self.assertFalse(decision.allowed_to_record)
        self.assertEqual(decision.operator_status, "blocked")

    def test_stereo_ir_warns_but_is_allowed_when_probe_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            probe_path = self._write_probe_results(Path(tmp_dir))
            decision = evaluate_capture_preflight(
                capture_mode="stereo_ir",
                serials=["a", "b", "c"],
                width=848,
                height=480,
                fps=30,
                emitter="on",
                probe_results_path=probe_path,
                probe_results_md_path=probe_path.with_suffix(".md"),
            )
        self.assertTrue(decision.allowed_to_record)
        self.assertEqual(decision.operator_status, "experimental_warning")

    def test_pending_serial_resolution_is_explicit(self) -> None:
        decision = evaluate_capture_preflight(
            capture_mode="both_eval",
            serials=None,
            width=848,
            height=480,
            fps=30,
            emitter="on",
        )
        self.assertEqual(decision.operator_status, "pending_serial_resolution")

    def test_missing_probe_file_reports_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            probe_path = Path(tmp_dir) / "missing_probe.json"
            decision = evaluate_capture_preflight(
                capture_mode="stereo_ir",
                serials=["a", "b", "c"],
                width=848,
                height=480,
                fps=30,
                emitter="on",
                probe_results_path=probe_path,
                probe_results_md_path=probe_path.with_suffix(".md"),
            )
        self.assertEqual(decision.operator_status, "unknown")
        self.assertIn("No D455 stream probe results file", decision.reason)
