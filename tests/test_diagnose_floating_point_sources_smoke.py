from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tests.visualization_test_utils import make_sam31_masks, make_visualization_case


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "harness" / "diagnose_floating_point_sources.py"


class DiagnoseFloatingPointSourcesSmokeTest(unittest.TestCase):
    def test_same_case_outputs_are_created(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            aligned_root = tmp_root / "data"
            case_dir = aligned_root / "sample_case"
            make_visualization_case(
                case_dir,
                include_depth_ffs=True,
                include_depth_ffs_float_m=True,
                frame_num=1,
                include_sparse_outlier=True,
            )
            make_sam31_masks(case_dir, prompt_labels_by_object={1: "sloth", 2: "sloth"})

            output_dir = tmp_root / "floating_point_output"
            cmd = [
                sys.executable,
                str(SCRIPT),
                "--case_name",
                "sample_case",
                "--aligned_root",
                str(aligned_root),
                "--output_dir",
                str(output_dir),
                "--text_prompt",
                "sloth",
                "--use_float_ffs_depth_when_available",
            ]
            subprocess.run(cmd, check=True, cwd=ROOT)

            self.assertTrue((output_dir / "native" / "frames" / "000000.png").is_file())
            self.assertTrue((output_dir / "ffs" / "frames" / "000000.png").is_file())
            self.assertTrue((output_dir / "comparison_frames" / "000000.png").is_file())
            self.assertTrue((output_dir / "00_outlier_projection_board.png").is_file())
            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["same_case_mode"])
            self.assertTrue(summary["masked_mode"])
            self.assertIn("aggregate", summary["native"])
            self.assertIn("aggregate", summary["ffs"])
            self.assertEqual(summary["native"]["aggregate"]["frame_count"], 1)
            self.assertEqual(summary["ffs"]["aggregate"]["frame_count"], 1)
            self.assertIsNotNone(summary["comparison_board_path"])
            self.assertEqual(summary["render_contract"]["comparison_board_layout"], "4x3")


if __name__ == "__main__":
    unittest.main()
