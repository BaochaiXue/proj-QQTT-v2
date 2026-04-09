from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

from data_process.visualization.turntable_compare import run_turntable_compare_workflow
from tests.visualization_test_utils import make_visualization_case


class TurntableFrameContractSmokeTest(unittest.TestCase):
    def test_turntable_metadata_writes_explicit_frame_contract(self) -> None:
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
            self.assertFalse(metadata["frame_contract"]["uses_semantic_world"])
            self.assertEqual(metadata["calibration_contract"]["transform_convention"], "camera_to_world_c2w")
            self.assertTrue((output_dir / "scene_overview_calibration_frame.png").is_file())
