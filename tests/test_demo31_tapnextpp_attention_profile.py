from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


class Demo31TapNextPPAttentionProfileTest(unittest.TestCase):
    def test_skip_model_load_writes_stack_only_report(self) -> None:
        from scripts.harness.experiments.profile_demo31_tapnextpp_attention_kernels import (
            build_arg_parser,
            main,
            run_profile,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            args = build_arg_parser().parse_args(["--skip-model-load", "--output-dir", str(root)])
            payload = run_profile(args)
            self.assertFalse(payload["live_runtime_changed"])
            self.assertEqual(payload["status"], "stack_only")
            self.assertIn("torch_sdp_flash_enabled", payload["stack"])
            self.assertEqual(main(["--skip-model-load", "--output-dir", str(root)]), 0)
            self.assertTrue((root / "summary.json").is_file())
            self.assertTrue((root / "summary.md").is_file())

    def test_interpretation_detects_flash_sdpa_without_marking_primary_bottleneck(self) -> None:
        from scripts.harness.experiments.profile_demo31_tapnextpp_attention_kernels import _interpret

        summary = {
            "total_self_device_ms": 100.0,
            "scaled_dot_product_attention_detected": True,
            "flash_attention_detected": True,
            "mem_efficient_attention_detected": False,
            "math_attention_likely": False,
            "tag_device_time_total_ms": {
                "scaled_dot_product_attention": 8.0,
                "flash_attention": 8.0,
                "linear": 35.0,
                "einsum": 20.0,
            },
        }
        interp = _interpret(summary)
        self.assertTrue(interp["uses_scaled_dot_product_attention"])
        self.assertTrue(interp["uses_flash_attention_kernel"])
        self.assertFalse(interp["math_attention_fallback_likely"])
        self.assertFalse(interp["attention_is_primary_bottleneck"])


if __name__ == "__main__":
    unittest.main()
