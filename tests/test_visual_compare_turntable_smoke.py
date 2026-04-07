from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from data_process.visualization.turntable_compare import run_turntable_compare_workflow
from tests.visualization_test_utils import make_visualization_case


class VisualCompareTurntableSmokeTest(unittest.TestCase):
    def test_turntable_workflow_writes_primary_single_frame_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            aligned_root = tmp_root / "data"
            case_dir = aligned_root / "sample_case"
            make_visualization_case(case_dir, include_depth_ffs=True, include_depth_ffs_float_m=True, frame_num=1)

            output_dir = tmp_root / "turntable_output"
            result = run_turntable_compare_workflow(
                aligned_root=aligned_root,
                case_name="sample_case",
                output_dir=output_dir,
                renderer="fallback",
                num_orbit_steps=3,
                write_mp4=False,
                write_keyframe_sheet=True,
            )

            self.assertEqual(result["output_dir"], str(output_dir.resolve()))
            self.assertTrue((output_dir / "scene_overview_with_cameras.png").is_file())
            self.assertTrue((output_dir / "turntable_keyframes_sheet.png").is_file())
            self.assertTrue((output_dir / "turntable_metadata.json").is_file())
            self.assertEqual(len(list((output_dir / "boards").glob("*.png"))), 3)

            metadata = json.loads((output_dir / "turntable_metadata.json").read_text(encoding="utf-8"))
            self.assertTrue(metadata["same_case_mode"])
            self.assertEqual(metadata["num_orbit_steps"], 3)


if __name__ == "__main__":
    unittest.main()
