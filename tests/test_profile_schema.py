from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from qqtt.demo.services.profile_schema import (
    DEMO23_REQUIRED_PROFILE_KEYS,
    DEMO31_REQUIRED_PROFILE_KEYS,
    ProfileKeys,
    RuntimeProfile,
    build_empty_dual_gpu_profile_summary,
    classify_bottleneck,
    event_fps,
    write_profile_json,
    write_profile_markdown,
)


class ProfileSchemaTests(unittest.TestCase):
    def test_demo31_empty_profile_contains_required_keys(self) -> None:
        summary = build_empty_dual_gpu_profile_summary(
            {
                "mask_gpu_physical": 0,
                "cotracker_gpu_physical": 1,
                "semantic_mode": "exp",
                "tracking_query_mode": "phystwin_dense",
            }
        )

        for key in DEMO31_REQUIRED_PROFILE_KEYS:
            self.assertIn(key, summary)
        self.assertFalse(summary["cross_gpu_cuda_tensor_transfer"])
        self.assertEqual(summary["ipc_payload"], "cpu_numpy_latest_wins")

    def test_demo23_required_keys_include_object_volume_fields(self) -> None:
        self.assertIn(ProfileKeys.OBJECT_VOLUME_MS, DEMO23_REQUIRED_PROFILE_KEYS)
        self.assertIn(ProfileKeys.OBJECT_VOLUME_EXACT, DEMO23_REQUIRED_PROFILE_KEYS)
        self.assertIn(ProfileKeys.OBJECT_VOLUME_OCCUPIED_VOXELS, DEMO23_REQUIRED_PROFILE_KEYS)
        self.assertIn(ProfileKeys.OBJECT_VOLUME_OUTPUT_POINTS, DEMO23_REQUIRED_PROFILE_KEYS)

    def test_classify_bottleneck(self) -> None:
        self.assertEqual(classify_bottleneck({"render_total_ms_p50": 25.0}), "renderer")
        self.assertEqual(
            classify_bottleneck({ProfileKeys.OBJECT_VOLUME_MS_P50: 18.0}),
            "object_volume_filter",
        )
        self.assertEqual(
            classify_bottleneck({"render_waited_for_object_volume_filter": True}),
            "object_volume_filter_blocking_render",
        )

    def test_event_fps(self) -> None:
        self.assertEqual(event_fps([]), 0.0)
        self.assertEqual(event_fps([1.0]), 0.0)
        self.assertEqual(event_fps([0.0, 0.5, 1.0]), 2.0)

    def test_profile_writers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profile = RuntimeProfile(
                demo="demo-test",
                preset="unit",
                summary={
                    ProfileKeys.RENDERED_FPS: 12.3,
                    ProfileKeys.OBJECT_VOLUME_MS: 4.5,
                },
            )
            json_path = Path(tmp) / "profile.json"
            md_path = Path(tmp) / "profile.md"
            write_profile_json(json_path, profile)
            write_profile_markdown(md_path, profile)

            self.assertTrue(json_path.is_file())
            self.assertIn("rendered FPS", md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
