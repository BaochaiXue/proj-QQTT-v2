from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from data_process.visualization.pointcloud_compare import run_depth_comparison_workflow
from tests.visualization_test_utils import make_visualization_case


class FusedCloudRenderConfigSmokeTest(unittest.TestCase):
    def test_multi_view_render_outputs_are_created(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            aligned_root = tmp_root / "data"
            case_dir = aligned_root / "sample_case"
            make_visualization_case(case_dir, include_depth_ffs=True, include_depth_ffs_float_m=True, frame_num=1)

            output_dir = tmp_root / "comparison_output"
            result = run_depth_comparison_workflow(
                aligned_root=aligned_root,
                case_name="sample_case",
                output_dir=output_dir,
                renderer="fallback",
                render_mode="color_by_height",
                views=["oblique", "top"],
                zoom_scale=1.2,
            )

            self.assertTrue((output_dir / "view_oblique" / "side_by_side_frames" / "000000.png").is_file())
            self.assertTrue((output_dir / "view_top" / "native_frames" / "000000.png").is_file())
            self.assertEqual(result["comparison_metadata"]["views"], ["oblique", "top"])
            metadata = json.loads((output_dir / "comparison_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["render_mode"], "color_by_height")


if __name__ == "__main__":
    unittest.main()
