from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tests.test_pointcloud_fusion_smoke import make_aligned_case


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "harness" / "visual_compare_depth_video.py"


class VisualCompareDepthVideoSmokeTest(unittest.TestCase):
    def test_same_case_comparison_writes_expected_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            aligned_root = tmp_root / "data"
            case_dir = aligned_root / "sample_case"
            make_aligned_case(case_dir, include_depth_ffs=True)

            output_dir = tmp_root / "comparison_output"
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
            ]
            subprocess.run(cmd, check=True, cwd=ROOT)

            self.assertTrue((output_dir / "native_frames" / "000000.png").is_file())
            self.assertTrue((output_dir / "ffs_frames" / "000000.png").is_file())
            self.assertTrue((output_dir / "side_by_side_frames" / "000000.png").is_file())
            self.assertTrue((output_dir / "comparison_metadata.json").is_file())
            self.assertTrue((output_dir / "metrics.json").is_file())

            comparison_metadata = json.loads((output_dir / "comparison_metadata.json").read_text(encoding="utf-8"))
            self.assertTrue(comparison_metadata["same_case_mode"])


if __name__ == "__main__":
    unittest.main()
