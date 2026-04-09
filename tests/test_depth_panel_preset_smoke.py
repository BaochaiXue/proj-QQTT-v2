from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
import unittest

from scripts.harness.visual_compare_depth_panels import apply_preset
from data_process.visualization.panel_compare import run_depth_panel_workflow
from tests.visualization_test_utils import make_visualization_case


class DepthPanelPresetSmokeTest(unittest.TestCase):
    def test_review_quality_preset_updates_defaults(self) -> None:
        parser_args = argparse.Namespace(
            preset="review_quality",
            show_edge_overlay=None,
            depth_min_m=0.1,
            depth_max_m=3.0,
        )
        applied = apply_preset(parser_args)
        self.assertTrue(applied.show_edge_overlay)
        self.assertEqual(applied.depth_min_m, 0.2)
        self.assertEqual(applied.depth_max_m, 1.5)

    def test_workflow_summary_records_common_depth_range_and_preset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            aligned_root = root / "data"
            native_case = aligned_root / "native_case"
            ffs_case = aligned_root / "ffs_case"
            make_visualization_case(native_case, include_depth_ffs=False, frame_num=1)
            make_visualization_case(ffs_case, include_depth_ffs=True, include_depth_ffs_float_m=True, frame_num=1)

            output_dir = root / "panel_output"
            result = run_depth_panel_workflow(
                aligned_root=aligned_root,
                output_dir=output_dir,
                realsense_case="native_case",
                ffs_case="ffs_case",
                camera_ids=[0],
                depth_min_m=0.2,
                depth_max_m=1.5,
                rois=[{"name": "Head", "bbox": (1, 1, 6, 6)}],
                preset="review_quality",
                show_edge_overlay=True,
            )
            summary = result["summary"]
            camera_summary = summary["per_camera"]["0"]
            self.assertEqual(summary["preset"], "review_quality")
            self.assertTrue(summary["show_edge_overlay"])
            self.assertEqual(camera_summary["frame_metrics"][0]["depth_range_m"], [0.2, 1.5])
            self.assertEqual(camera_summary["rois"][0]["name"], "Head")
            self.assertTrue((output_dir / "camera_0" / "frames" / "000000.png").is_file())
