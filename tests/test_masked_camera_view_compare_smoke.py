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

            self.assertTrue((output_dir / "01_masked_camera_view_board.png").is_file())
            self.assertTrue((output_dir / "summary.json").is_file())
            self.assertTrue((output_dir / "debug" / "native_masked_fused.ply").is_file())
            self.assertTrue((output_dir / "debug" / "ffs_masked_fused.ply").is_file())
            self.assertEqual(summary["render_contract"]["view_mode"], "original_camera_extrinsics")
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
            self.assertFalse(summary["empty_mask_fallback_used"])
            for source_name in ("native", "ffs"):
                self.assertEqual(len(summary["debug_artifacts"][f"{source_name}_render_paths"]), 3)
                for camera_entry in summary["mask_sources"][source_name]["per_camera"]:
                    self.assertLessEqual(
                        int(camera_entry["post_mask_point_count"]),
                        int(camera_entry["pre_mask_point_count"]),
                    )


if __name__ == "__main__":
    unittest.main()
