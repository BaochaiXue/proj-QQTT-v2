from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from data_process.visualization.io_case import load_case_frame_camera_clouds, load_case_metadata
from tests.visualization_test_utils import make_visualization_case


class IoCaseFfsNativeLikeLoaderSmokeTest(unittest.TestCase):
    def test_prefers_aligned_auxiliary_postprocess_depth_stream(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            case_dir = Path(tmp_dir) / "ffs_case"
            make_visualization_case(
                case_dir,
                include_depth_ffs=True,
                include_depth_ffs_float_m=True,
                include_depth_ffs_native_like_postprocess=True,
                include_depth_ffs_native_like_postprocess_float_m=True,
                frame_num=1,
            )

            metadata = load_case_metadata(case_dir)
            _, stats = load_case_frame_camera_clouds(
                case_dir=case_dir,
                metadata=metadata,
                frame_idx=0,
                depth_source="ffs",
                use_float_ffs_depth_when_available=True,
                max_points_per_camera=None,
                depth_min_m=0.1,
                depth_max_m=3.0,
                ffs_native_like_postprocess=True,
            )

            camera_stats = stats["per_camera"][0]
            self.assertEqual(camera_stats["depth_dir_used"], "depth_ffs_native_like_postprocess_float_m")
            self.assertEqual(camera_stats["source_depth_dir_used"], "depth_ffs_native_like_postprocess_float_m")
            self.assertTrue(camera_stats["ffs_native_like_postprocess_applied"])
            self.assertEqual(camera_stats["ffs_native_like_postprocess_origin"], "aligned_auxiliary")

    def test_can_apply_postprocess_on_the_fly_when_auxiliary_stream_is_absent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            case_dir = Path(tmp_dir) / "ffs_case"
            make_visualization_case(
                case_dir,
                include_depth_ffs=True,
                include_depth_ffs_float_m=True,
                frame_num=1,
            )

            metadata = load_case_metadata(case_dir)
            _, stats = load_case_frame_camera_clouds(
                case_dir=case_dir,
                metadata=metadata,
                frame_idx=0,
                depth_source="ffs",
                use_float_ffs_depth_when_available=True,
                max_points_per_camera=None,
                depth_min_m=0.1,
                depth_max_m=3.0,
                ffs_native_like_postprocess=True,
            )

            camera_stats = stats["per_camera"][0]
            self.assertEqual(camera_stats["depth_dir_used"], "depth_ffs_float_m+ffs_native_like_postprocess")
            self.assertEqual(camera_stats["source_depth_dir_used"], "depth_ffs_float_m")
            self.assertTrue(camera_stats["ffs_native_like_postprocess_applied"])
            self.assertEqual(camera_stats["ffs_native_like_postprocess_origin"], "on_the_fly")


if __name__ == "__main__":
    unittest.main()
