from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tests.visualization_test_utils import make_visualization_case


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "harness" / "visual_compare_depth_panels.py"


class VisualCompareDepthPanelsSmokeTest(unittest.TestCase):
    def test_two_case_panel_outputs_are_created(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            aligned_root = tmp_root / "data"
            native_case = aligned_root / "native_case"
            ffs_case = aligned_root / "ffs_case"
            make_visualization_case(native_case, include_depth_ffs=False, frame_num=2)
            make_visualization_case(ffs_case, include_depth_ffs=True, include_depth_ffs_float_m=True, frame_num=2)

            output_dir = tmp_root / "panel_output"
            cmd = [
                sys.executable,
                str(SCRIPT),
                "--aligned_root",
                str(aligned_root),
                "--realsense_case",
                "native_case",
                "--ffs_case",
                "ffs_case",
                "--output_dir",
                str(output_dir),
                "--camera_ids",
                "0",
                "2",
                "--roi",
                "1,1,6,6",
            ]
            subprocess.run(cmd, check=True, cwd=ROOT)

            self.assertTrue((output_dir / "camera_0" / "frames" / "000000.png").is_file())
            self.assertTrue((output_dir / "camera_2" / "frames" / "000001.png").is_file())
            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["camera_ids"], [0, 2])
            self.assertFalse(summary["same_case_mode"])

    def test_ffs_native_like_prefers_aligned_auxiliary_ffs_depth_stream(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            aligned_root = tmp_root / "data"
            native_case = aligned_root / "native_case"
            ffs_case = aligned_root / "ffs_case"
            make_visualization_case(native_case, include_depth_ffs=False, frame_num=1)
            make_visualization_case(
                ffs_case,
                include_depth_ffs=True,
                include_depth_ffs_float_m=True,
                include_depth_ffs_native_like_postprocess=True,
                include_depth_ffs_native_like_postprocess_float_m=True,
                frame_num=1,
            )

            output_dir = tmp_root / "panel_output"
            cmd = [
                sys.executable,
                str(SCRIPT),
                "--aligned_root",
                str(aligned_root),
                "--realsense_case",
                "native_case",
                "--ffs_case",
                "ffs_case",
                "--output_dir",
                str(output_dir),
                "--camera_ids",
                "0",
                "--ffs_native_like_postprocess",
                "--use_float_ffs_depth_when_available",
            ]
            subprocess.run(cmd, check=True, cwd=ROOT)

            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            frame_metric = summary["per_camera"]["0"]["frame_metrics"][0]
            self.assertTrue(summary["ffs_native_like_postprocess_enabled"])
            self.assertEqual(frame_metric["ffs_depth_dir_used"], "depth_ffs_native_like_postprocess_float_m")
            self.assertTrue(frame_metric["ffs_native_like_postprocess_applied"])
            self.assertEqual(frame_metric["ffs_native_like_postprocess_origin"], "aligned_auxiliary")

    def test_ffs_native_like_can_run_on_the_fly_without_auxiliary_stream(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            aligned_root = tmp_root / "data"
            native_case = aligned_root / "native_case"
            ffs_case = aligned_root / "ffs_case"
            make_visualization_case(native_case, include_depth_ffs=False, frame_num=1)
            make_visualization_case(ffs_case, include_depth_ffs=True, include_depth_ffs_float_m=True, frame_num=1)

            output_dir = tmp_root / "panel_output"
            cmd = [
                sys.executable,
                str(SCRIPT),
                "--aligned_root",
                str(aligned_root),
                "--realsense_case",
                "native_case",
                "--ffs_case",
                "ffs_case",
                "--output_dir",
                str(output_dir),
                "--camera_ids",
                "0",
                "--ffs_native_like_postprocess",
                "--use_float_ffs_depth_when_available",
            ]
            subprocess.run(cmd, check=True, cwd=ROOT)

            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            frame_metric = summary["per_camera"]["0"]["frame_metrics"][0]
            self.assertTrue(summary["ffs_native_like_postprocess_enabled"])
            self.assertEqual(frame_metric["ffs_depth_dir_used"], "depth_ffs_float_m+ffs_native_like_postprocess")
            self.assertTrue(frame_metric["ffs_native_like_postprocess_applied"])
            self.assertEqual(frame_metric["ffs_native_like_postprocess_origin"], "on_the_fly")


if __name__ == "__main__":
    unittest.main()
