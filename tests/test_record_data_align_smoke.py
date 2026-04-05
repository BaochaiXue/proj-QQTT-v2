from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "record_data_align_minimal"
SCRIPT = ROOT / "data_process" / "record_data_align.py"


class RecordDataAlignSmokeTest(unittest.TestCase):
    def test_aligns_minimal_case(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            base_path = tmp_root / "data_collect"
            case_dir = base_path / "sample_case"
            shutil.copytree(FIXTURE, case_dir)

            output_path = tmp_root / "data"
            cmd = [
                sys.executable,
                str(SCRIPT),
                "--base_path",
                str(base_path),
                "--output_path",
                str(output_path),
                "--case_name",
                "sample_case",
                "--start",
                "10",
                "--end",
                "11",
            ]
            subprocess.run(cmd, check=True, cwd=ROOT)

            aligned_case = output_path / "sample_case"
            self.assertTrue((aligned_case / "calibrate.pkl").is_file())
            self.assertTrue((aligned_case / "metadata.json").is_file())

            metadata = json.loads((aligned_case / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["frame_num"], 2)
            self.assertEqual(metadata["start_step"], 10)
            self.assertEqual(metadata["end_step"], 11)
            self.assertEqual(len(metadata["serial_numbers"]), 3)

            for camera_idx in range(3):
                color_dir = aligned_case / "color" / str(camera_idx)
                depth_dir = aligned_case / "depth" / str(camera_idx)
                self.assertTrue((color_dir / "0.png").is_file())
                self.assertTrue((color_dir / "1.png").is_file())
                self.assertTrue((depth_dir / "0.npy").is_file())
                self.assertTrue((depth_dir / "1.npy").is_file())


if __name__ == "__main__":
    unittest.main()
