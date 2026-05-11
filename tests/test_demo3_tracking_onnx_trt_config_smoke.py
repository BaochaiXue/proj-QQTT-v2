from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from qqtt.tracking.onnx_trt.export_probe import run_export_probe
from qqtt.tracking.onnx_trt.ort_sessions import build_ort_providers, tensorrt_provider_options


class Demo3TrackingOnnxTrtConfigSmokeTest(unittest.TestCase):
    def test_tensorrt_provider_config_has_cache_and_fp16(self) -> None:
        options = tensorrt_provider_options(engine_cache_path="data/cache/demo3_tracking_trt", fp16=True)
        self.assertTrue(options["trt_fp16_enable"])
        self.assertTrue(options["trt_engine_cache_enable"])
        self.assertEqual(options["trt_engine_cache_path"], "data/cache/demo3_tracking_trt")
        providers = build_ort_providers(engine_cache_path="data/cache/demo3_tracking_trt", trt_fp16=True)
        self.assertEqual(providers[0][0], "TensorrtExecutionProvider")
        self.assertIn("CUDAExecutionProvider", providers)

    def test_fake_export_probe_records_missing_onnx_without_crashing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = run_export_probe(model_name="locotrack_s", onnx_path=None, engine_cache_path=Path(tmp_dir) / "cache")
        self.assertEqual(result["export_onnx"], "fail")
        self.assertIn("No exportable model", result["quality_notes"])


if __name__ == "__main__":
    unittest.main()
