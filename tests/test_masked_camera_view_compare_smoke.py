from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_process.visualization.workflows.masked_camera_view_compare import (
    run_masked_camera_view_compare_workflow,
)
from tests.visualization_test_utils import make_sam31_masks, make_visualization_case


class MaskedCameraViewCompareSmokeTest(unittest.TestCase):
    def test_workflow_writes_2x3_board_and_uses_fixed_camera_views(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            aligned_root = tmp_root / "data"
            native_case = aligned_root / "native_case"
            ffs_case = aligned_root / "ffs_case"
            make_visualization_case(native_case, frame_num=1)
            make_visualization_case(ffs_case, include_depth_ffs=True, include_depth_ffs_float_m=True, frame_num=1)
            native_mask_root = make_sam31_masks(native_case)
            ffs_mask_root = make_sam31_masks(ffs_case)
            output_dir = tmp_root / "masked_camera_view_output"

            summary = run_masked_camera_view_compare_workflow(
                aligned_root=aligned_root,
                output_dir=output_dir,
                realsense_case="native_case",
                ffs_case="ffs_case",
                frame_idx=0,
                text_prompt="sloth",
                native_mask_root=native_mask_root,
                ffs_mask_root=ffs_mask_root,
                native_mask_source="reused_existing",
                ffs_mask_source="reused_existing",
                mask_source_mode="reuse_or_generate",
                look_distance=1.25,
                render_frame_fn=lambda points, colors, **kwargs: np.full((96, 128, 3), 120, dtype=np.uint8),
            )

            self.assertTrue((output_dir / "00_masked_rgb_board.png").is_file())
            self.assertTrue((output_dir / "01_masked_camera_view_board.png").is_file())
            self.assertTrue((output_dir / "summary.json").is_file())
            self.assertTrue((output_dir / "debug" / "native_masked_fused.ply").is_file())
            self.assertTrue((output_dir / "debug" / "ffs_masked_fused.ply").is_file())
            self.assertEqual(summary["board_mode"], "2x3")
            self.assertEqual(summary["board_row_headers"], ["Native", "FFS"])
            self.assertEqual(summary["render_contract"]["view_mode"], "original_camera_extrinsics")
            self.assertEqual(summary["render_contract"]["projection_mode"], "original_camera_pinhole")
            self.assertEqual(len(summary["column_views"]), 3)
            self.assertEqual(
                [item["camera_idx"] for item in summary["column_views"]],
                [0, 1, 2],
            )
            self.assertEqual(
                [item["label"] for item in summary["column_views"]],
                [
                    "Cam0 | 239222300433",
                    "Cam1 | 239222300781",
                    "Cam2 | 239222303506",
                ],
            )
            np.testing.assert_allclose(summary["column_views"][0]["camera_position"], [0.0, 0.0, 0.0], atol=1e-6)
            np.testing.assert_allclose(summary["column_views"][0]["center"], [0.0, 0.0, 1.25], atol=1e-6)
            np.testing.assert_allclose(summary["column_views"][1]["camera_position"], [0.25, 0.0, 0.0], atol=1e-6)
            np.testing.assert_allclose(summary["column_views"][2]["camera_position"], [0.0, 0.25, 0.0], atol=1e-6)
            self.assertEqual(summary["column_views"][0]["image_size"], [10, 8])
            self.assertEqual(len(summary["column_views"][0]["intrinsic_matrix"]), 9)
            self.assertEqual(len(summary["column_views"][0]["extrinsic_matrix"]), 16)
            self.assertFalse(summary["empty_mask_fallback_used"])
            self.assertIn("rgb_board_path", summary)
            self.assertEqual(summary["variants"]["masked_rgb_reference"]["panel_count"], 3)
            self.assertEqual(len(summary["debug_artifacts"]["masked_rgb_paths"]), 3)
            for source_name in ("native", "ffs"):
                self.assertEqual(len(summary["debug_artifacts"][f"{source_name}_render_paths"]), 3)
                for camera_entry in summary["mask_sources"][source_name]["per_camera"]:
                    self.assertLessEqual(
                        int(camera_entry["post_mask_point_count"]),
                        int(camera_entry["pre_mask_point_count"]),
                    )

    def test_workflow_can_apply_native_and_ffs_postprocess(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            aligned_root = tmp_root / "data"
            native_case = aligned_root / "native_case"
            ffs_case = aligned_root / "ffs_case"
            make_visualization_case(native_case, frame_num=1)
            make_visualization_case(
                ffs_case,
                include_depth_ffs=True,
                include_depth_ffs_float_m=True,
                include_depth_ffs_native_like_postprocess=True,
                include_depth_ffs_native_like_postprocess_float_m=True,
                frame_num=1,
            )
            native_mask_root = make_sam31_masks(native_case)
            ffs_mask_root = make_sam31_masks(ffs_case)
            output_dir = tmp_root / "masked_camera_view_postprocess_output"

            summary = run_masked_camera_view_compare_workflow(
                aligned_root=aligned_root,
                output_dir=output_dir,
                realsense_case="native_case",
                ffs_case="ffs_case",
                frame_idx=0,
                text_prompt="sloth",
                native_mask_root=native_mask_root,
                ffs_mask_root=ffs_mask_root,
                native_mask_source="reused_existing",
                ffs_mask_source="reused_existing",
                mask_source_mode="reuse_or_generate",
                native_depth_postprocess=True,
                ffs_native_like_postprocess=True,
                render_frame_fn=lambda points, colors, **kwargs: np.full((96, 128, 3), 90, dtype=np.uint8),
            )

            self.assertTrue(summary["native_depth_postprocess"])
            self.assertTrue(summary["ffs_native_like_postprocess"])
            self.assertEqual(summary["board_mode"], "4x3")
            self.assertEqual(summary["board_row_headers"], ["Native", "Native + PS", "FFS", "FFS + PS"])
            self.assertEqual(summary["postprocess_contract"]["mode"], "phystwin_data_process_mask")
            self.assertEqual(summary["postprocess"]["native"]["origin"], "on_the_fly")
            self.assertEqual(summary["postprocess"]["ffs"]["origin"], "on_the_fly")
            self.assertEqual(summary["postprocess"]["native"]["radius_m"], 0.01)
            self.assertEqual(summary["postprocess"]["native"]["nb_points"], 40)
            self.assertEqual(
                sorted(summary["variants"].keys()),
                ["ffs_postprocess", "ffs_raw", "masked_rgb_reference", "native_postprocess", "native_raw"],
            )
            self.assertEqual(len(summary["variants"]["native_raw"]["render_paths"]), 3)
            self.assertEqual(len(summary["variants"]["native_postprocess"]["render_paths"]), 3)
            self.assertEqual(len(summary["variants"]["ffs_raw"]["render_paths"]), 3)
            self.assertEqual(len(summary["variants"]["ffs_postprocess"]["render_paths"]), 3)
            for camera_stats in summary["source_stats"]["native"]["per_camera"]:
                self.assertFalse(camera_stats["native_depth_postprocess_enabled"])
                self.assertFalse(camera_stats["native_depth_postprocess_applied"])
                self.assertEqual(camera_stats["native_depth_postprocess_origin"], "none")
            for camera_stats in summary["source_stats"]["ffs"]["per_camera"]:
                self.assertFalse(camera_stats["ffs_native_like_postprocess_enabled"])
                self.assertFalse(camera_stats["ffs_native_like_postprocess_applied"])
                self.assertEqual(camera_stats["ffs_native_like_postprocess_origin"], "none")
            for source_name in ("native", "ffs"):
                self.assertTrue(summary["postprocess"][source_name]["applied"])
                self.assertEqual(len(summary["debug_artifacts"][f"{source_name}_postprocess_render_paths"]), 3)
                for camera_entry in summary["mask_sources"][source_name]["per_camera"]:
                    self.assertIn("mask_postprocess", camera_entry)
                    self.assertIn("raw_mask_point_count", camera_entry)
                    self.assertIn("postprocess_mask_point_count", camera_entry)
                    self.assertGreaterEqual(
                        int(camera_entry["mask_postprocess"]["pre_mask_pixel_count"]),
                        int(camera_entry["mask_postprocess"]["post_mask_pixel_count"]),
                    )
                    self.assertGreaterEqual(
                        int(camera_entry["pre_mask_point_count"]),
                        int(camera_entry["post_mask_point_count"]),
                    )


if __name__ == "__main__":
    unittest.main()
