from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from data_process.visualization.io_case import resolve_case_dir, resolve_case_dirs
from tests.visualization_test_utils import make_visualization_case


class GroupedAlignedCaseResolutionSmokeTest(unittest.TestCase):
    def test_resolves_unique_basename_inside_grouped_aligned_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            aligned_root = Path(tmp_dir) / "data"
            case_dir = aligned_root / "static" / "sample_case"
            make_visualization_case(case_dir, include_depth_ffs=True, frame_num=1)

            resolved = resolve_case_dir(aligned_root=aligned_root, case_ref="sample_case")
            self.assertEqual(resolved, case_dir.resolve())

    def test_resolve_case_dirs_accepts_group_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            aligned_root = Path(tmp_dir) / "data"
            native_case = aligned_root / "static" / "native_case"
            ffs_case = aligned_root / "static" / "ffs_case"
            make_visualization_case(native_case, include_depth_ffs=False, frame_num=1)
            make_visualization_case(ffs_case, include_depth_ffs=True, frame_num=1)

            native_case_dir, ffs_case_dir, same_case_mode = resolve_case_dirs(
                aligned_root=aligned_root,
                case_name=None,
                realsense_case="static/native_case",
                ffs_case="static/ffs_case",
            )

            self.assertFalse(same_case_mode)
            self.assertEqual(native_case_dir, native_case.resolve())
            self.assertEqual(ffs_case_dir, ffs_case.resolve())

    def test_ambiguous_basename_requires_relative_subpath(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            aligned_root = Path(tmp_dir) / "data"
            make_visualization_case(aligned_root / "static" / "sample_case", include_depth_ffs=True, frame_num=1)
            make_visualization_case(aligned_root / "dynamic" / "sample_case", include_depth_ffs=True, frame_num=1)

            with self.assertRaisesRegex(ValueError, "Ambiguous case reference"):
                resolve_case_dir(aligned_root=aligned_root, case_ref="sample_case")


if __name__ == "__main__":
    unittest.main()
