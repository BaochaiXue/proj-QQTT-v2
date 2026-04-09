from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from data_process.visualization.workflows.turntable_compare import run_turntable_workflow
from tests.visualization_test_utils import make_visualization_case


class TurntableWorkflowSmokeTest(unittest.TestCase):
    def test_workflow_wrapper_writes_expected_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            make_visualization_case(root / "native_case", include_depth_ffs=True)
            output_dir = root / "turntable_outputs"
            result = run_turntable_workflow(
                aligned_root=root,
                output_dir=output_dir,
                case_name="native_case",
                frame_idx=0,
                renderer="fallback",
                write_mp4=False,
                write_gif=False,
                write_keyframe_sheet=True,
                num_orbit_steps=4,
                orbit_degrees=90.0,
                point_radius_px=2,
                supersample_scale=1,
            )
            self.assertEqual(Path(result["output_dir"]), output_dir.resolve())
            self.assertTrue((output_dir / "turntable_keyframes_geom.png").exists())
            self.assertTrue((output_dir / "scene_overview_with_cameras.png").exists())
