from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from data_process.visualization.io_case import load_case_frame_cloud_with_sources, load_case_metadata
from tests.visualization_test_utils import make_visualization_case


class IoCaseFfsRawLoaderSmokeTest(unittest.TestCase):
    def test_prefers_archived_float_raw_depth_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            case_dir = Path(tmp_dir) / "ffs_case"
            make_visualization_case(
                case_dir,
                include_depth_ffs=True,
                include_depth_ffs_float_m=True,
                include_depth_ffs_original=True,
                include_depth_ffs_float_m_original=True,
                frame_num=1,
                depth_backend_used="both",
                depth_source_for_depth_dir="realsense",
            )

            metadata = load_case_metadata(case_dir)
            _, _, stats, _ = load_case_frame_cloud_with_sources(
                case_dir=case_dir,
                metadata=metadata,
                frame_idx=0,
                depth_source="ffs_raw",
                use_float_ffs_depth_when_available=True,
                voxel_size=None,
                max_points_per_camera=None,
                depth_min_m=0.1,
                depth_max_m=3.0,
            )

            camera_stats = stats["per_camera"][0]
            self.assertEqual(camera_stats["depth_dir_used"], "depth_ffs_float_m_original")
            self.assertEqual(camera_stats["source_depth_dir_used"], "depth_ffs_float_m_original")

    def test_falls_back_to_legacy_raw_ffs_depth_when_archive_is_absent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            case_dir = Path(tmp_dir) / "ffs_case"
            make_visualization_case(
                case_dir,
                include_depth_ffs=True,
                include_depth_ffs_float_m=True,
                frame_num=1,
                depth_backend_used="both",
                depth_source_for_depth_dir="realsense",
            )

            metadata = load_case_metadata(case_dir)
            _, _, stats, _ = load_case_frame_cloud_with_sources(
                case_dir=case_dir,
                metadata=metadata,
                frame_idx=0,
                depth_source="ffs_raw",
                use_float_ffs_depth_when_available=True,
                voxel_size=None,
                max_points_per_camera=None,
                depth_min_m=0.1,
                depth_max_m=3.0,
            )

            camera_stats = stats["per_camera"][0]
            self.assertEqual(camera_stats["depth_dir_used"], "depth_ffs_float_m")
            self.assertEqual(camera_stats["source_depth_dir_used"], "depth_ffs_float_m")

    def test_ffs_backend_raw_loader_prefers_depth_original_uint_archive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            case_dir = Path(tmp_dir) / "ffs_case"
            make_visualization_case(
                case_dir,
                include_depth_original=True,
                frame_num=1,
                depth_backend_used="ffs",
                depth_source_for_depth_dir="ffs",
            )

            metadata = load_case_metadata(case_dir)
            _, _, stats, _ = load_case_frame_cloud_with_sources(
                case_dir=case_dir,
                metadata=metadata,
                frame_idx=0,
                depth_source="ffs_raw",
                use_float_ffs_depth_when_available=False,
                voxel_size=None,
                max_points_per_camera=None,
                depth_min_m=0.1,
                depth_max_m=3.0,
            )

            camera_stats = stats["per_camera"][0]
            self.assertEqual(camera_stats["depth_dir_used"], "depth_original")
            self.assertEqual(camera_stats["source_depth_dir_used"], "depth_original")


if __name__ == "__main__":
    unittest.main()
