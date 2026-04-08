from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from data_process.visualization.turntable_compare import (
    load_single_frame_compare_clouds,
    resolve_single_frame_case_selection,
)
from tests.visualization_test_utils import make_visualization_case


class ManualImageRoiFilterSmokeTest(unittest.TestCase):
    def test_manual_image_roi_reduces_loaded_compare_points(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            aligned_root = tmp_root / "data"
            case_dir = aligned_root / "sample_case"
            make_visualization_case(case_dir, include_depth_ffs=True, include_depth_ffs_float_m=True, frame_num=1)

            selection = resolve_single_frame_case_selection(
                aligned_root=aligned_root,
                case_name="sample_case",
                realsense_case=None,
                ffs_case=None,
                frame_idx=0,
                camera_ids=[0, 1, 2],
            )
            full_scene = load_single_frame_compare_clouds(
                selection,
                voxel_size=None,
                max_points_per_camera=None,
                depth_min_m=0.2,
                depth_max_m=1.5,
                use_float_ffs_depth_when_available=True,
                pixel_roi_by_camera=None,
            )
            roi_scene = load_single_frame_compare_clouds(
                selection,
                voxel_size=None,
                max_points_per_camera=None,
                depth_min_m=0.2,
                depth_max_m=1.5,
                use_float_ffs_depth_when_available=True,
                pixel_roi_by_camera={
                    0: (3, 2, 6, 5),
                    1: (3, 2, 6, 5),
                    2: (3, 2, 6, 5),
                },
            )

            self.assertGreater(len(full_scene["native_points"]), len(roi_scene["native_points"]))
            self.assertGreater(len(full_scene["ffs_points"]), len(roi_scene["ffs_points"]))
            self.assertGreater(len(roi_scene["native_points"]), 0)
            self.assertGreater(len(roi_scene["ffs_points"]), 0)


if __name__ == "__main__":
    unittest.main()
