from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from data_process.visualization.turntable_compare import prepare_object_roi_refinement
from tests.visualization_test_utils import make_object_refinement_raw_scene


class AutoObjectRefinementFromProjectedBboxSmokeTest(unittest.TestCase):
    def test_pass2_world_roi_recovers_protrusion_from_projected_bboxes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_scene = make_object_refinement_raw_scene(Path(tmp_dir))
            refinement = prepare_object_roi_refinement(
                raw_scene=raw_scene,
                focus_mode="table",
                scene_crop_mode="auto_object_bbox",
                crop_margin_xy=0.10,
                crop_min_z=-0.08,
                crop_max_z=0.35,
                manual_xyz_roi=None,
                manual_image_roi_by_camera=None,
                object_height_min=0.02,
                object_height_max=0.22,
                object_component_mode="largest",
                object_component_topk=2,
            )

            self.assertIsNotNone(refinement["pass1_crop"])
            self.assertIsNotNone(refinement["pass2_crop"])
            self.assertGreater(
                float(refinement["pass2_crop"]["object_roi_max"][2]),
                float(refinement["pass1_crop"]["object_roi_max"][2]) + 0.02,
            )
            self.assertTrue(bool(refinement["pass2_crop"].get("seed_bbox_used", False)))


if __name__ == "__main__":
    unittest.main()
