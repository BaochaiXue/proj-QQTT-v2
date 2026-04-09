from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from data_process.visualization.turntable_compare import prepare_object_roi_refinement
from tests.visualization_test_utils import make_object_refinement_raw_scene


class PixelMaskExpandsWorldRoiSmokeTest(unittest.TestCase):
    def test_pixel_mask_evidence_expands_world_roi_between_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_scene = make_object_refinement_raw_scene(Path(tmp_dir))
            refinement = prepare_object_roi_refinement(
                raw_scene=raw_scene,
                focus_mode="table",
                scene_crop_mode="auto_object_bbox",
                crop_margin_xy=0.08,
                crop_min_z=-0.08,
                crop_max_z=0.30,
                manual_xyz_roi=None,
                manual_image_roi_by_camera=None,
                object_height_min=0.02,
                object_height_max=0.20,
                object_component_mode="largest",
                object_component_topk=1,
            )

            pass1_max_z = float(refinement["pass1_crop"]["object_roi_max"][2])
            pass2_max_z = float(refinement["pass2_crop"]["object_roi_max"][2])
            self.assertGreater(pass2_max_z, pass1_max_z + 0.015)
            self.assertTrue(bool(refinement["pass2_crop"].get("seed_bbox_used", False)))


if __name__ == "__main__":
    unittest.main()
