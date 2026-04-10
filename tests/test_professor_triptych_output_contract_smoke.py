from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from data_process.visualization.professor_triptych import run_professor_triptych_workflow
from tests.visualization_test_utils import make_visualization_case


class ProfessorTriptychOutputContractSmokeTest(unittest.TestCase):
    def test_default_output_writes_only_three_figures_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            aligned_root = root / "data"
            case_dir = aligned_root / "sample_case"
            make_visualization_case(case_dir, include_depth_ffs=True, include_depth_ffs_float_m=True, frame_num=1)

            output_dir = root / "professor_pack"
            result = run_professor_triptych_workflow(
                aligned_root=aligned_root,
                case_name="sample_case",
                output_dir=output_dir,
                renderer="fallback",
                num_orbit_steps=6,
            )

            self.assertEqual(result["output_dir"], str(output_dir.resolve()))
            self.assertEqual(
                sorted(path.name for path in output_dir.iterdir()),
                ["01_hero_compare.png", "02_merge_evidence.png", "03_truth_board.png", "summary.json"],
            )
            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["frame_idx"], 0)
            self.assertFalse(summary["debug_written"])
            self.assertIn("hero_angle_selection", summary)
            self.assertIn("truth_camera_pair", summary)
            self.assertIn("object_projected_area_ratio", summary["hero_angle_selection"])
            self.assertIn("object_bbox_fill_ratio", summary["hero_angle_selection"])
            self.assertIn("object_multi_camera_support_ratio", summary["hero_angle_selection"])
            self.assertIn("object_mismatch_residual_m", summary["hero_angle_selection"])
            self.assertIn("context_dominance_penalty", summary["hero_angle_selection"])
            self.assertIn("silhouette_penalty", summary["hero_angle_selection"])
            self.assertIn("final_score", summary["hero_angle_selection"])
            self.assertIn("object_warp_valid_ratio_native", summary["truth_camera_pair"])
            self.assertIn("object_warp_valid_ratio_ffs", summary["truth_camera_pair"])
            self.assertIn("object_residual_mean_native", summary["truth_camera_pair"])
            self.assertIn("object_residual_mean_ffs", summary["truth_camera_pair"])
            self.assertIn("object_overlap_area", summary["truth_camera_pair"])
            self.assertIn("pair_object_visibility_score", summary["truth_camera_pair"])

    def test_debug_gating_keeps_top_level_clean_and_writes_debug_dir_only_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            aligned_root = root / "data"
            case_dir = aligned_root / "sample_case"
            make_visualization_case(case_dir, include_depth_ffs=True, include_depth_ffs_float_m=True, frame_num=1)

            output_dir = root / "professor_pack"
            run_professor_triptych_workflow(
                aligned_root=aligned_root,
                case_name="sample_case",
                output_dir=output_dir,
                renderer="fallback",
                num_orbit_steps=5,
                write_debug=True,
            )

            self.assertTrue((output_dir / "debug").is_dir())
            self.assertTrue((output_dir / "debug" / "hero_angle_candidates.json").is_file())
            self.assertTrue((output_dir / "debug" / "truth_pair_candidates.json").is_file())
            self.assertTrue((output_dir / "debug" / "scene_overview_with_cameras.png").is_file())
            top_level_names = {path.name for path in output_dir.iterdir()}
            self.assertIn("01_hero_compare.png", top_level_names)
            self.assertIn("02_merge_evidence.png", top_level_names)
            self.assertIn("03_truth_board.png", top_level_names)
            self.assertIn("summary.json", top_level_names)
            self.assertIn("debug", top_level_names)


if __name__ == "__main__":
    unittest.main()
