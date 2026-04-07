from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from data_process.visualization.turntable_compare import resolve_single_frame_case_selection
from tests.visualization_test_utils import make_visualization_case


class SingleFrameCaseSelectionSmokeTest(unittest.TestCase):
    def test_same_case_selection_uses_one_case_with_depth_ffs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            aligned_root = Path(tmp_dir) / "data"
            case_dir = aligned_root / "sample_case"
            make_visualization_case(case_dir, include_depth_ffs=True, frame_num=2)

            selection = resolve_single_frame_case_selection(
                aligned_root=aligned_root,
                case_name="sample_case",
                realsense_case=None,
                ffs_case=None,
                frame_idx=1,
            )

            self.assertTrue(selection["same_case_mode"])
            self.assertEqual(selection["native_case_dir"], selection["ffs_case_dir"])
            self.assertEqual(selection["native_frame_idx"], 1)
            self.assertEqual(selection["ffs_frame_idx"], 1)

    def test_two_case_selection_keeps_native_and_ffs_dirs_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            aligned_root = Path(tmp_dir) / "data"
            native_case = aligned_root / "native_case"
            ffs_case = aligned_root / "ffs_case"
            make_visualization_case(native_case, include_depth_ffs=False, frame_num=2)
            make_visualization_case(ffs_case, include_depth_ffs=True, frame_num=2)

            selection = resolve_single_frame_case_selection(
                aligned_root=aligned_root,
                case_name=None,
                realsense_case="native_case",
                ffs_case="ffs_case",
                frame_idx=0,
            )

            self.assertFalse(selection["same_case_mode"])
            self.assertNotEqual(selection["native_case_dir"], selection["ffs_case_dir"])
            self.assertEqual(selection["camera_ids"], [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
