from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tests.visualization_test_utils import make_visualization_case


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "harness" / "visual_compare_depth_video.py"


class VisualCompareDepthVideoGridSmokeTest(unittest.TestCase):
    def test_grid_2x3_outputs_are_created(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            aligned_root = tmp_root / "data"
            native_case = aligned_root / "native_case"
            ffs_case = aligned_root / "ffs_case"
            make_visualization_case(native_case, include_depth_ffs=False, frame_num=1)
            make_visualization_case(ffs_case, include_depth_ffs=True, include_depth_ffs_float_m=True, frame_num=1)

            output_dir = tmp_root / "grid_output"
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
                "--renderer",
                "fallback",
                "--render_mode",
                "color_by_rgb",
                "--view_mode",
                "camera_poses_table_focus",
                "--focus_mode",
                "table",
                "--layout_mode",
                "grid_2x3",
                "--zoom_scale",
                "2.2",
                "--image_flip",
                "vertical",
            ]
            subprocess.run(cmd, check=True, cwd=ROOT)

            self.assertTrue((output_dir / "grid_2x3_frames" / "000000.png").is_file())
            self.assertTrue((output_dir / "view_cam0" / "native_frames" / "000000.png").is_file())
            metadata = json.loads((output_dir / "comparison_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["layout_mode"], "grid_2x3")
            self.assertEqual(metadata["view_mode"], "camera_poses_table_focus")
            self.assertEqual(metadata["views"], ["cam0", "cam1", "cam2"])
            self.assertEqual(metadata["image_flip"], "vertical")


if __name__ == "__main__":
    unittest.main()
