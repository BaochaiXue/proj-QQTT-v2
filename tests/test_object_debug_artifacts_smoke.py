from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from data_process.visualization.turntable_compare import run_turntable_compare_workflow
from tests.visualization_test_utils import make_visualization_case


class ObjectDebugArtifactsSmokeTest(unittest.TestCase):
    def test_turntable_writes_object_debug_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            aligned_root = tmp_root / "data"
            case_dir = aligned_root / "sample_case"
            make_visualization_case(case_dir, include_depth_ffs=True, include_depth_ffs_float_m=True, frame_num=1)

            roi_json = tmp_root / "roi.json"
            roi_json.write_text(json.dumps({"0": [1, 1, 8, 6], "1": [1, 1, 8, 6], "2": [1, 1, 8, 6]}), encoding="utf-8")
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
                manual_image_roi_json=roi_json,
            )

            debug_dir = output_dir / "debug"
            self.assertTrue((debug_dir / "compare_debug_metrics.json").is_file())
            debug_metrics = json.loads((debug_dir / "compare_debug_metrics.json").read_text(encoding="utf-8"))
            self.assertIn("native", debug_metrics)
            self.assertIn("ffs", debug_metrics)
            self.assertIn("mask_debug", debug_metrics["native"])
            self.assertIn("0", debug_metrics["native"]["mask_debug"])
            self.assertIn("refined_mask_pixels", debug_metrics["native"]["mask_debug"]["0"])
            self.assertTrue((debug_dir / "per_camera_object_mask_overlay" / "native_cam0.png").is_file())
            self.assertTrue((debug_dir / "per_camera_object_cloud" / "native_cam0.ply").is_file())
            self.assertTrue((debug_dir / "fused_object_only" / "native_object_only.ply").is_file())
            self.assertTrue((debug_dir / "fused_object_context" / "native_object_context.ply").is_file())


if __name__ == "__main__":
    unittest.main()
