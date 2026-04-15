from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from data_process.visualization.triplet_ply_compare import run_triplet_ply_compare_workflow
from tests.visualization_test_utils import make_visualization_case


class TripletPlyCompareWorkflowSmokeTest(unittest.TestCase):
    def test_two_case_workflow_writes_three_fused_plys_and_on_the_fly_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            aligned_root = tmp_root / "data"
            native_case = aligned_root / "native_case"
            ffs_case = aligned_root / "ffs_case"
            make_visualization_case(native_case, frame_num=1)
            make_visualization_case(ffs_case, include_depth_ffs=True, include_depth_ffs_float_m=True, frame_num=1)

            output_dir = tmp_root / "triplet_output"
            summary = run_triplet_ply_compare_workflow(
                aligned_root=aligned_root,
                realsense_case="native_case",
                ffs_case="ffs_case",
                output_dir=output_dir,
                frame_idx=0,
            )

            self.assertTrue((output_dir / "ply_fullscene" / "native_frame_0000_fused_fullscene.ply").is_file())
            self.assertTrue((output_dir / "ply_fullscene" / "ffs_raw_frame_0000_fused_fullscene.ply").is_file())
            self.assertTrue((output_dir / "ply_fullscene" / "ffs_postprocess_frame_0000_fused_fullscene.ply").is_file())
            self.assertTrue((output_dir / "summary.json").is_file())

            on_disk_summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertFalse(summary["same_case_mode"])
            self.assertEqual(summary["depth_min_m"], 0.0)
            self.assertEqual(summary["depth_max_m"], 1.5)
            self.assertEqual(on_disk_summary["depth_min_m"], 0.0)
            self.assertEqual(on_disk_summary["depth_max_m"], 1.5)
            self.assertEqual(on_disk_summary["variants"]["ffs_postprocess"]["ffs_native_like_postprocess_origin"], "on_the_fly")
            self.assertEqual(on_disk_summary["variants"]["ffs_raw"]["depth_dirs_used"], ["depth_ffs_float_m"])
            self.assertEqual(
                sorted(on_disk_summary["pointcloud_contract"].keys()),
                ["ffs_postprocess", "ffs_raw", "native"],
            )

    def test_same_case_workflow_prefers_aligned_postprocess_auxiliary_stream(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            aligned_root = tmp_root / "data"
            case_dir = aligned_root / "both_case"
            make_visualization_case(
                case_dir,
                include_depth_ffs=True,
                include_depth_ffs_float_m=True,
                include_depth_ffs_native_like_postprocess=True,
                include_depth_ffs_native_like_postprocess_float_m=True,
                frame_num=1,
            )

            output_dir = tmp_root / "triplet_output"
            summary = run_triplet_ply_compare_workflow(
                aligned_root=aligned_root,
                case_name="both_case",
                output_dir=output_dir,
                frame_idx=0,
            )

            self.assertTrue(summary["same_case_mode"])
            self.assertEqual(summary["depth_min_m"], 0.0)
            self.assertEqual(summary["depth_max_m"], 1.5)
            self.assertEqual(summary["variants"]["ffs_postprocess"]["ffs_native_like_postprocess_origin"], "aligned_auxiliary")
            self.assertEqual(
                summary["variants"]["ffs_postprocess"]["depth_dirs_used"],
                ["depth_ffs_native_like_postprocess_float_m"],
            )
            self.assertEqual(summary["variants"]["ffs_raw"]["depth_dirs_used"], ["depth_ffs_float_m"])


if __name__ == "__main__":
    unittest.main()
