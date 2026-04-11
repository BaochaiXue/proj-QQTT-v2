from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from data_process.visualization.match_board import run_match_board_workflow
from tests.visualization_test_utils import make_visualization_case


class MatchBoardOutputContractSmokeTest(unittest.TestCase):
    def test_default_output_writes_only_board_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            aligned_root = root / "data"
            case_dir = aligned_root / "sample_case"
            make_visualization_case(case_dir, include_depth_ffs=True, include_depth_ffs_float_m=True, frame_num=1)

            output_dir = root / "match_board"
            result = run_match_board_workflow(
                aligned_root=aligned_root,
                case_name="sample_case",
                output_dir=output_dir,
                renderer="fallback",
                num_orbit_steps=6,
            )

            self.assertEqual(result["output_dir"], str(output_dir.resolve()))
            self.assertEqual(
                sorted(path.name for path in output_dir.iterdir()),
                ["01_pointcloud_match_board.png", "match_board_summary.json"],
            )
            summary = json.loads((output_dir / "match_board_summary.json").read_text(encoding="utf-8"))
            self.assertIn("match_angle_selection", summary)
            self.assertIn("top_level_output", summary)
            self.assertEqual(summary["display_frame"], "semantic_world")
            self.assertIn("product_artifacts", summary)
            self.assertIn("debug_artifacts", summary)
            self.assertFalse(summary["debug_artifacts"]["enabled"])
            self.assertFalse(summary["debug_written"])

    def test_debug_gating_writes_debug_dir_only_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            aligned_root = root / "data"
            case_dir = aligned_root / "sample_case"
            make_visualization_case(case_dir, include_depth_ffs=True, include_depth_ffs_float_m=True, frame_num=1)

            output_dir = root / "match_board"
            run_match_board_workflow(
                aligned_root=aligned_root,
                case_name="sample_case",
                output_dir=output_dir,
                renderer="fallback",
                num_orbit_steps=5,
                write_debug=True,
            )

            self.assertTrue((output_dir / "debug").is_dir())
            self.assertTrue((output_dir / "debug" / "match_angle_candidates.json").is_file())
            top_level_names = {path.name for path in output_dir.iterdir()}
            self.assertIn("01_pointcloud_match_board.png", top_level_names)
            self.assertIn("match_board_summary.json", top_level_names)
            self.assertIn("debug", top_level_names)
            summary = json.loads((output_dir / "match_board_summary.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["debug_artifacts"]["enabled"])


if __name__ == "__main__":
    unittest.main()
