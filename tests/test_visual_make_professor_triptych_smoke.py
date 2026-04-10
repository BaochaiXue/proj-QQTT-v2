from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tests.visualization_test_utils import make_visualization_case


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "harness" / "visual_make_professor_triptych.py"


class VisualMakeProfessorTriptychSmokeTest(unittest.TestCase):
    def test_cli_writes_three_figure_pack(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            aligned_root = tmp_root / "data"
            case_dir = aligned_root / "sample_case"
            make_visualization_case(case_dir, include_depth_ffs=True, include_depth_ffs_float_m=True, frame_num=1)

            output_dir = tmp_root / "professor_pack"
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
                "5",
            ]
            subprocess.run(cmd, check=True, cwd=ROOT)

            self.assertTrue((output_dir / "01_hero_compare.png").is_file())
            self.assertTrue((output_dir / "02_merge_evidence.png").is_file())
            self.assertTrue((output_dir / "03_truth_board.png").is_file())
            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["top_level_outputs"]["hero_compare"], str((output_dir / "01_hero_compare.png").resolve()))


if __name__ == "__main__":
    unittest.main()
