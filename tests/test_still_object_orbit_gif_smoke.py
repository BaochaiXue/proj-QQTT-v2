from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

import data_process.visualization.experiments.still_object_orbit_gif as orbit_gif
from tests.visualization_test_utils import make_sam31_masks, make_visualization_case


class StillObjectOrbitGifSmokeTest(unittest.TestCase):
    def test_parse_mask_erode_pixels(self) -> None:
        self.assertEqual(orbit_gif.parse_mask_erode_pixels(orbit_gif.DEFAULT_6X2_ERODE_SWEEP_PIXELS), [1, 3, 5, 10])
        self.assertEqual(orbit_gif.parse_mask_erode_pixels("1,3"), [1, 3])
        with self.assertRaises(ValueError):
            orbit_gif.parse_mask_erode_pixels("0,1")
        with self.assertRaises(ValueError):
            orbit_gif.parse_mask_erode_pixels("1,1")

    def test_erode_sweep_writes_variant_gifs_and_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            case_specs: list[dict[str, object]] = []
            for idx in range(6):
                case_dir = tmp_root / "data" / f"case_{idx}"
                make_visualization_case(case_dir, include_depth_ffs_float_m=True, frame_num=1)
                mask_root = make_sam31_masks(case_dir, prompt_labels_by_object={1: "object"})
                case_specs.append(
                    {
                        "label": f"Case {idx}",
                        "case_dir": case_dir,
                        "mask_root": mask_root,
                        "text_prompt": "object",
                    }
                )

            original_render = orbit_gif._render_variant_tile

            def fake_render_variant_tile(*, points: np.ndarray, tile_width: int, tile_height: int, **kwargs) -> np.ndarray:
                value = min(240, 30 + int(len(points)))
                return np.full((int(tile_height), int(tile_width), 3), value, dtype=np.uint8)

            try:
                orbit_gif._render_variant_tile = fake_render_variant_tile  # type: ignore[assignment]
                summary = orbit_gif.run_still_object_rope_6x2_orbit_gif_erode_sweep_workflow(
                    case_specs=case_specs,
                    output_root=tmp_root / "output",
                    erode_pixels=[1, 3],
                    frame_idx=0,
                    num_frames=2,
                    fps=2,
                    tile_width=48,
                    tile_height=32,
                    row_label_width=80,
                    max_points_per_camera=None,
                    max_points_per_variant=None,
                )
            finally:
                orbit_gif._render_variant_tile = original_render  # type: ignore[assignment]

            self.assertTrue((tmp_root / "output" / "summary.json").is_file())
            self.assertEqual(summary["erode_pixels"], [1, 3])
            self.assertEqual(summary["panel_layout"], "3x4")
            self.assertFalse(summary["pt_like_postprocess_enabled"])
            self.assertTrue(summary["enhanced_pt_like_removed_highlight_enabled"])
            self.assertEqual(len(summary["variants"]), 2)
            for variant in summary["variants"]:
                self.assertTrue(Path(variant["gif_path"]).is_file())
                self.assertTrue(Path(variant["first_frame_path"]).is_file())
                self.assertTrue(Path(variant["summary_path"]).is_file())
                self.assertEqual(variant["panel_layout"], "3x4")
                self.assertIn("_3x4_", Path(variant["gif_path"]).name)
                self.assertFalse(variant["pt_like_postprocess_enabled"])
                self.assertTrue(variant["enhanced_pt_like_removed_highlight_enabled"])
                first_case = variant["case_summaries"][0]
                erode_debug = first_case["mask_erode_debug"]["0"]
                self.assertLessEqual(
                    erode_debug["mask_pixel_count_after_erode"],
                    erode_debug["mask_pixel_count_before_erode"],
                )
                native_highlight = first_case["native_pt_like_removed_highlight_stats"]
                ffs_highlight = first_case["ffs_pt_like_removed_highlight_stats"]
                self.assertTrue(native_highlight["enabled"])
                self.assertTrue(ffs_highlight["enabled"])
                self.assertFalse(native_highlight["deletes_points"])
                self.assertFalse(ffs_highlight["deletes_points"])
                self.assertEqual(
                    native_highlight["input_point_count"],
                    native_highlight["render_point_count"],
                )
                self.assertEqual(
                    ffs_highlight["input_point_count"],
                    ffs_highlight["render_point_count"],
                )


if __name__ == "__main__":
    unittest.main()
