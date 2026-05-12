from __future__ import annotations

import unittest

from demo_v2_2 import runtime as demo


class Demo22FinalProfileSourceTests(unittest.TestCase):
    def test_profile_json_must_be_full_hf_batched_backend(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "not from hf_batched_multisession"):
            demo.final_fps_from_demo22_profile(
                {
                    "edgetam_backend": demo.EDGETAM_BACKEND_HF_BATCH_VISION_SEQ_SESSION,
                    "summary_after_warmup": {"filter_fps": 20.0},
                }
            )

    def test_mask_only_profile_cannot_satisfy_final_fps_source(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "cannot define Demo 2.2 final FPS"):
            demo.final_fps_from_demo22_profile(
                {
                    "edgetam_backend": demo.EDGETAM_BACKEND_HF_BATCHED_MULTISESSION,
                    "profile_kind": "mask-only",
                    "summary_after_warmup": {"filter_fps": 20.0},
                }
            )

    def test_external_fork_profile_cannot_satisfy_final_fps_source(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "cannot define Demo 2.2 final FPS"):
            demo.final_fps_from_demo22_profile(
                {
                    "edgetam_backend": demo.EDGETAM_BACKEND_HF_BATCHED_MULTISESSION,
                    "profile_kind": "external-fork",
                    "summary_after_warmup": {"filter_fps": 20.0},
                }
            )

    def test_final_fps_parser_uses_filter_fps_primary(self) -> None:
        result = demo.final_fps_from_demo22_profile(
            {
                "edgetam_backend": demo.EDGETAM_BACKEND_HF_BATCHED_MULTISESSION,
                "summary_after_warmup": {
                    "capture_group_fps": 18.0,
                    "raw_fusion_fps": 16.0,
                    "filter_fps": 15.5,
                    "render_fps": 10.0,
                },
            }
        )

        self.assertEqual(result["final_fps"], 15.5)
        self.assertEqual(result["final_fps_source"], "filter_fps")
        self.assertEqual(result["raw_fusion_fps"], 16.0)
        self.assertEqual(result["render_fps"], 10.0)

    def test_final_fps_parser_accepts_profile_filter_output_key(self) -> None:
        result = demo.final_fps_from_demo22_profile(
            {
                "edgetam_backend": demo.EDGETAM_BACKEND_HF_BATCHED_MULTISESSION,
                "summary_after_warmup": {
                    "capture_group_fps": 18.0,
                    "raw_fusion_fps": 16.0,
                    "filter_output_fps": 14.5,
                    "render_fps": 8.0,
                },
            }
        )

        self.assertEqual(result["final_fps"], 14.5)
        self.assertEqual(result["final_fps_source"], "filter_fps")


if __name__ == "__main__":
    unittest.main()
