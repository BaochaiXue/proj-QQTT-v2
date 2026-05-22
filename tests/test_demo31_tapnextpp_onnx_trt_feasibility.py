from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


class Demo31TapNextPPOnnxTrtFeasibilityTest(unittest.TestCase):
    def test_state_byte_estimates_label_target_and_stress_scale(self) -> None:
        from scripts.harness.experiments.probe_demo31_tapnextpp_onnx_trt_feasibility import (
            estimate_tapnext_state_bytes,
        )

        q1365 = estimate_tapnext_state_bytes(batch_size=3, query_count=1365, image_size=(256, 256))
        q4096 = estimate_tapnext_state_bytes(batch_size=3, query_count=4096, image_size=(256, 256))

        self.assertEqual(q1365["total_query_count"], 4095)
        self.assertEqual(q4096["total_query_count"], 12288)
        self.assertGreater(q4096["hidden_state_bytes"], q1365["hidden_state_bytes"])
        self.assertEqual(q1365["state_tensor_count"], 24)

    def test_skip_model_probe_writes_json_and_markdown_without_heavy_export(self) -> None:
        from scripts.harness.experiments.probe_demo31_tapnextpp_onnx_trt_feasibility import (
            build_arg_parser,
            run_probe,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            args = build_arg_parser().parse_args(
                [
                    "--output-dir",
                    str(root / "out"),
                    "--artifact-dir",
                    str(root / "artifacts"),
                    "--skip-model-load",
                ]
            )
            payload = run_probe(args)
            self.assertFalse(payload["live_runtime_changed"])
            self.assertEqual(payload["actual_probe_cases"], [])
            self.assertEqual(payload["conclusion"]["status"], "stack_only")
            self.assertTrue((root / "out" / "summary.json").is_file())
            self.assertTrue((root / "out" / "summary.md").is_file())
            self.assertFalse((root / "artifacts").exists())


if __name__ == "__main__":
    unittest.main()
