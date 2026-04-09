from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from data_process.visualization.turntable_compare import run_turntable_compare_workflow
from tests.visualization_test_utils import make_visualization_case


class CompareDebugMetricsRefinementSmokeTest(unittest.TestCase):
    def test_turntable_writes_pass1_pass2_refinement_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            aligned_root = tmp_root / "data"
            case_dir = aligned_root / "sample_case"
            make_visualization_case(case_dir, include_depth_ffs=True, include_depth_ffs_float_m=True, frame_num=1)

            output_dir = tmp_root / "turntable_output"
            run_turntable_compare_workflow(
                aligned_root=aligned_root,
                case_name="sample_case",
                output_dir=output_dir,
                renderer="fallback",
                num_orbit_steps=1,
                write_mp4=False,
                write_gif=False,
                write_keyframe_sheet=False,
            )

            self.assertTrue((output_dir / "object_roi_pass1_world.json").is_file())
            self.assertTrue((output_dir / "object_roi_pass2_world.json").is_file())
            self.assertTrue((output_dir / "per_camera_auto_bbox" / "cam0.json").is_file())
            self.assertTrue((output_dir / "compare_debug_metrics.json").is_file())

            debug_metrics = json.loads((output_dir / "compare_debug_metrics.json").read_text(encoding="utf-8"))
            self.assertIn("world_roi_pass1", debug_metrics)
            self.assertIn("world_roi_pass2", debug_metrics)
            self.assertIn("head_like_protrusion_recovered_heuristic", debug_metrics)
            self.assertIn("pass1_seed_mask_metrics", debug_metrics["native"])
            self.assertIn("pass2_seed_mask_metrics", debug_metrics["native"])
            self.assertIn("pass1_mask_debug", debug_metrics["native"])
            self.assertIn("pass2_mask_debug", debug_metrics["native"])


if __name__ == "__main__":
    unittest.main()
