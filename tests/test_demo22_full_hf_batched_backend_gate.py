from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from demo_v2_2 import runtime as demo


def _valid_report() -> dict:
    return {
        "source": {"commit": "4a41c4a45f"},
        "decision": {
            "hf_batched_multisession_usable": True,
            "recommended_precision_mode": demo.EDGETAM_PRECISION_MODE_MEMORY_PATH_FP32,
            "recommended_compile_mode": demo.COMPILE_MODE_REDUCE_OVERHEAD,
        },
        "backend_contract": {
            "batch_memory_attention": True,
            "batch_mask_decoder": True,
            "batch_memory_encoder": True,
            "batched_state_scatter": True,
            "used_public_session_step_in_hot_path": False,
            "partial_fallback_used": False,
            "contract_pass": True,
        },
    }


class Demo22FullHfBatchedBackendGateTests(unittest.TestCase):
    def test_refuses_missing_external_report(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "not found"):
            demo.load_full_batched_edgetam_report("/tmp/does-not-exist-edgetam-report.json")

    def test_accepts_synthetic_valid_report(self) -> None:
        validation = demo.validate_full_batched_edgetam_report(_valid_report())

        self.assertTrue(validation["hf_batched_multisession_usable"])
        self.assertTrue(validation["batch_memory_attention"])
        self.assertTrue(validation["batch_mask_decoder"])
        self.assertTrue(validation["batch_memory_encoder"])
        self.assertTrue(validation["batched_state_scatter"])
        self.assertFalse(validation["used_public_session_step_in_hot_path"])
        self.assertFalse(validation["partial_fallback_used"])

    def test_refuses_unusable_report(self) -> None:
        report = _valid_report()
        report["decision"]["hf_batched_multisession_usable"] = False

        with self.assertRaisesRegex(RuntimeError, "usable=true"):
            demo.validate_full_batched_edgetam_report(report)

    def test_refuses_partial_fallback(self) -> None:
        report = _valid_report()
        report["backend_contract"]["partial_fallback_used"] = True

        with self.assertRaisesRegex(RuntimeError, "partial fallback"):
            demo.validate_full_batched_edgetam_report(report)

    def test_refuses_public_session_hot_path(self) -> None:
        report = _valid_report()
        report["backend_contract"]["used_public_session_step_in_hot_path"] = True

        with self.assertRaisesRegex(RuntimeError, "public session step"):
            demo.validate_full_batched_edgetam_report(report)

    def test_loads_json_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            path.write_text(json.dumps(_valid_report()), encoding="utf-8")

            loaded = demo.load_full_batched_edgetam_report(path)

        self.assertEqual(loaded["_path"], str(path))
        self.assertTrue(loaded["decision"]["hf_batched_multisession_usable"])


if __name__ == "__main__":
    unittest.main()
