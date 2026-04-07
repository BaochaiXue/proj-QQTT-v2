from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tests.visualization_test_utils import make_visualization_case


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "harness" / "visual_compare_reprojection.py"


class VisualCompareReprojectionSmokeTest(unittest.TestCase):
    def test_same_case_reprojection_outputs_are_created(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            aligned_root = tmp_root / "data"
            case_dir = aligned_root / "sample_case"
            make_visualization_case(case_dir, include_depth_ffs=True, include_depth_ffs_float_m=True, frame_num=2)

            output_dir = tmp_root / "reprojection_output"
            cmd = [
                sys.executable,
                str(SCRIPT),
                "--case_name",
                "sample_case",
                "--aligned_root",
                str(aligned_root),
                "--output_dir",
                str(output_dir),
                "--camera_pair",
                "0,1",
            ]
            subprocess.run(cmd, check=True, cwd=ROOT)

            self.assertTrue((output_dir / "pair_0_to_1" / "frames" / "000000.png").is_file())
            summary = json.loads((output_dir / "summary_metrics.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["same_case_mode"])
            self.assertIn("0_to_1", summary["per_pair"])


if __name__ == "__main__":
    unittest.main()
