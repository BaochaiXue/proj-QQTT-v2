from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

from data_process.visualization.turntable_compare import run_turntable_compare_workflow
from tests.visualization_test_utils import make_visualization_case


class TurntableFrameContractSmokeTest(unittest.TestCase):
    def test_turntable_metadata_defaults_to_semantic_world_and_writes_both_overviews(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            make_visualization_case(root / "case", include_depth_ffs=True)
            output_dir = root / "turntable"
            run_turntable_compare_workflow(
                aligned_root=root,
                output_dir=output_dir,
                case_name="case",
                frame_idx=0,
                renderer="fallback",
                write_mp4=False,
                write_gif=False,
                write_keyframe_sheet=False,
                num_orbit_steps=2,
                orbit_degrees=30.0,
                point_radius_px=2,
                supersample_scale=1,
            )
            metadata = json.loads((output_dir / "turntable_metadata.json").read_text(encoding="utf-8"))
            self.assertTrue(metadata["frame_contract"]["uses_semantic_world"])
            self.assertEqual(metadata["display_frame"], "semantic_world")
            self.assertEqual(metadata["calibration_contract"]["transform_convention"], "camera_to_world_c2w")
            self.assertTrue((output_dir / "scene_overview_calibration_frame.png").is_file())
            self.assertTrue((output_dir / "scene_overview_semantic_frame.png").is_file())

    def test_turntable_can_render_in_raw_calibration_world_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            make_visualization_case(root / "case", include_depth_ffs=True)
            output_dir = root / "turntable"
            run_turntable_compare_workflow(
                aligned_root=root,
                output_dir=output_dir,
                case_name="case",
                frame_idx=0,
                renderer="fallback",
                write_mp4=False,
                write_gif=False,
                write_keyframe_sheet=False,
                num_orbit_steps=2,
                orbit_degrees=30.0,
                point_radius_px=2,
                supersample_scale=1,
                display_frame="calibration_world",
            )
            metadata = json.loads((output_dir / "turntable_metadata.json").read_text(encoding="utf-8"))
            self.assertFalse(metadata["frame_contract"]["uses_semantic_world"])
            self.assertEqual(metadata["display_frame"], "calibration_world")
