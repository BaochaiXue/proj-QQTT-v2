from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from demo_v2_2 import runtime as demo


def _valid_trt_report() -> dict:
    return {
        "batchtam_component_engines_usable": True,
        "batchtam_closed_loop_usable": True,
        "demo22_trt_integration_allowed": True,
        "recommended_trt_scope": demo.EDGETAM_TRT_SCOPE_MEMORY_PATH_ALL,
        "memory_attention_shape_strategy": "bucketed_static_engines",
        "memory_attention_bucket_count": 16,
        "memory_attention_buckets_exported": 16,
        "memory_attention_buckets_built": 16,
        "memory_attention_buckets_validated": 16,
        "memory_attention_bucketed_closed_loop_pass": True,
        "mask_decoder_trt_closed_loop_pass": True,
        "memory_encoder_trt_closed_loop_pass": True,
        "memory_path_all_trt_closed_loop_pass": True,
    }


class Demo22BatchTamTrtGateTests(unittest.TestCase):
    def test_refuses_missing_batchtam_report(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "not found"):
            demo.load_batchtam_trt_report("/tmp/missing-batchtam-trt-report.json")

    def test_accepts_valid_batchtam_report(self) -> None:
        validation = demo.validate_batchtam_trt_report(_valid_trt_report())

        self.assertTrue(validation["batchtam_component_engines_usable"])
        self.assertTrue(validation["batchtam_closed_loop_usable"])
        self.assertTrue(validation["demo22_trt_integration_allowed"])
        self.assertEqual(validation["recommended_trt_scope"], demo.EDGETAM_TRT_SCOPE_MEMORY_PATH_ALL)
        self.assertEqual(validation["memory_attention_buckets_validated"], 16)

    def test_refuses_unusable_batchtam_report(self) -> None:
        report = _valid_trt_report()
        report["batchtam_closed_loop_usable"] = False

        with self.assertRaisesRegex(RuntimeError, "missing true integration fields"):
            demo.validate_batchtam_trt_report(report)

    def test_refuses_partial_memory_path_scope(self) -> None:
        report = _valid_trt_report()
        report["memory_encoder_trt_closed_loop_pass"] = False

        with self.assertRaisesRegex(RuntimeError, "memory_path_all gate failed"):
            demo.validate_batchtam_trt_report(report)

    def test_loads_json_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "batchtam_report.json"
            path.write_text(json.dumps(_valid_trt_report()), encoding="utf-8")

            loaded = demo.load_batchtam_trt_report(path)

        self.assertEqual(loaded["_path"], str(path))
        self.assertTrue(loaded["batchtam_closed_loop_usable"])

    def test_validates_required_engine_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            engine_dir = Path(tmp) / "engines"
            bucket_dir = engine_dir / "memory_attention_buckets"
            bucket_dir.mkdir(parents=True)
            for name in ("mask_decoder_b3.engine", "memory_encoder_b3.engine"):
                (engine_dir / name).write_bytes(b"engine")
            (bucket_dir / "memory_attention_objptr4_spmem1.engine").write_bytes(b"engine")
            args = argparse.Namespace(
                edgetam_backend=demo.EDGETAM_BACKEND_HF_BATCHED_MULTISESSION,
                edgetam_component_runtime=demo.EDGETAM_COMPONENT_RUNTIME_TRT,
                edgetam_trt_engine_dir=str(engine_dir),
                edgetam_trt_memory_attention_bucket_dir=None,
            )

            demo.validate_batchtam_trt_artifacts(args)

    def test_missing_engine_artifact_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(
                edgetam_backend=demo.EDGETAM_BACKEND_HF_BATCHED_MULTISESSION,
                edgetam_component_runtime=demo.EDGETAM_COMPONENT_RUNTIME_TRT,
                edgetam_trt_engine_dir=str(Path(tmp) / "engines"),
                edgetam_trt_memory_attention_bucket_dir=None,
            )

            with self.assertRaisesRegex(RuntimeError, "engine dir not found"):
                demo.validate_batchtam_trt_artifacts(args)


if __name__ == "__main__":
    unittest.main()
