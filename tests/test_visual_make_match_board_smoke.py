from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tests.visualization_test_utils import make_visualization_case


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "harness" / "visual_make_match_board.py"


class VisualMakeMatchBoardSmokeTest(unittest.TestCase):
    def test_cli_writes_match_board(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            aligned_root = tmp_root / "data"
            case_dir = aligned_root / "sample_case"
            make_visualization_case(case_dir, include_depth_ffs=True, include_depth_ffs_float_m=True, frame_num=1)

            output_dir = tmp_root / "match_board"
            cmd = [
                sys.executable,
                str(SCRIPT),
                "--case_name",
                "sample_case",
                "--aligned_root",
                str(aligned_root),
                "--output_dir",
                str(output_dir),
                "--renderer",
                "fallback",
                "--num_orbit_steps",
                "6",
            ]
            subprocess.run(cmd, check=True, cwd=ROOT)

            self.assertTrue((output_dir / "01_pointcloud_match_board.png").is_file())
            summary = json.loads((output_dir / "match_board_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["top_level_output"], str((output_dir / "01_pointcloud_match_board.png").resolve()))


if __name__ == "__main__":
    unittest.main()
