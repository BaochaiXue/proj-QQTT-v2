from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from data_process.aligned_case_metadata import (
    ALIGNED_METADATA_EXT_FILENAME,
    LEGACY_ALIGNED_METADATA_KEYS,
)
from data_process.record_data_align import align_case
from tests.test_record_data_align_ffs_smoke import FIXTURE, FakeOutlierRunner, FakeRunner, make_v2_case


class RecordDataAlignBothSmokeTest(unittest.TestCase):
    def test_aligns_case_with_both_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            base_path = tmp_root / "data_collect"
            case_dir = base_path / "sample_case"
            make_v2_case(case_dir)

            output_path = tmp_root / "data"
            args = Namespace(
                base_path=base_path,
                case_name="sample_case",
                output_path=output_path,
                start=10,
                end=11,
                fps=None,
                write_mp4=False,
                depth_backend="both",
                ffs_repo="C:/external/fake",
                ffs_model_path="C:/external/fake/model.pth",
                ffs_scale=1.0,
                ffs_valid_iters=4,
                ffs_max_disp=64,
                ffs_radius_outlier_filter=False,
                ffs_radius_outlier_radius_m=0.01,
                ffs_radius_outlier_nb_points=40,
                ffs_native_like_postprocess=False,
                write_ffs_float_m=True,
                fail_if_no_ir_stereo=True,
            )
            align_case(args, runner_factory=FakeRunner)

            aligned_case = output_path / "sample_case"
            self.assertTrue((aligned_case / "depth" / "0" / "0.npy").is_file())
            self.assertTrue((aligned_case / "depth_ffs" / "0" / "0.npy").is_file())
            self.assertTrue((aligned_case / "depth_ffs_float_m" / "0" / "0.npy").is_file())
            self.assertTrue((aligned_case / ALIGNED_METADATA_EXT_FILENAME).is_file())

            copied_depth = np.load(aligned_case / "depth" / "0" / "0.npy")
            original_depth = np.load(case_dir / "depth" / "0" / "10.npy")
            self.assertTrue(np.array_equal(copied_depth, original_depth))

            ffs_depth = np.load(aligned_case / "depth_ffs" / "0" / "0.npy")
            self.assertEqual(ffs_depth.dtype, np.uint16)
            self.assertGreater(int((ffs_depth > 0).sum()), 0)

            metadata = json.loads((aligned_case / "metadata.json").read_text(encoding="utf-8"))
            metadata_ext = json.loads((aligned_case / ALIGNED_METADATA_EXT_FILENAME).read_text(encoding="utf-8"))
            self.assertEqual(set(metadata.keys()), set(LEGACY_ALIGNED_METADATA_KEYS))
            self.assertEqual(metadata_ext["depth_backend_used"], "both")
            self.assertEqual(metadata_ext["depth_source_for_depth_dir"], "realsense")
            self.assertFalse(metadata_ext["ffs_native_like_postprocess_enabled"])
            self.assertFalse(metadata_ext["ffs_radius_outlier_filter_enabled"])
            self.assertFalse((aligned_case / "depth_ffs_native_like_postprocess").exists())

    def test_aligns_case_with_both_backend_and_ffs_native_like_aux_streams(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            base_path = tmp_root / "data_collect"
            case_dir = base_path / "sample_case"
            make_v2_case(case_dir)

            output_path = tmp_root / "data"
            args = Namespace(
                base_path=base_path,
                case_name="sample_case",
                output_path=output_path,
                start=10,
                end=11,
                fps=None,
                write_mp4=False,
                depth_backend="both",
                ffs_repo="C:/external/fake",
                ffs_model_path="C:/external/fake/model.pth",
                ffs_scale=1.0,
                ffs_valid_iters=4,
                ffs_max_disp=64,
                ffs_radius_outlier_filter=False,
                ffs_radius_outlier_radius_m=0.01,
                ffs_radius_outlier_nb_points=40,
                ffs_native_like_postprocess=True,
                write_ffs_float_m=False,
                fail_if_no_ir_stereo=True,
            )
            align_case(args, runner_factory=FakeRunner)

            aligned_case = output_path / "sample_case"
            self.assertTrue((aligned_case / "depth" / "0" / "0.npy").is_file())
            self.assertTrue((aligned_case / "depth_ffs" / "0" / "0.npy").is_file())
            self.assertTrue((aligned_case / "depth_ffs_native_like_postprocess" / "0" / "0.npy").is_file())
            self.assertTrue((aligned_case / "depth_ffs_native_like_postprocess_float_m" / "0" / "0.npy").is_file())
            self.assertTrue((aligned_case / ALIGNED_METADATA_EXT_FILENAME).is_file())

            metadata = json.loads((aligned_case / "metadata.json").read_text(encoding="utf-8"))
            metadata_ext = json.loads((aligned_case / ALIGNED_METADATA_EXT_FILENAME).read_text(encoding="utf-8"))
            self.assertEqual(set(metadata.keys()), set(LEGACY_ALIGNED_METADATA_KEYS))
            self.assertTrue(metadata_ext["ffs_native_like_postprocess_enabled"])

    def test_aligns_case_with_both_backend_and_radius_filter_preserving_native_depth(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            base_path = tmp_root / "data_collect"
            case_dir = base_path / "sample_case"
            make_v2_case(case_dir)

            output_path = tmp_root / "data"
            args = Namespace(
                base_path=base_path,
                case_name="sample_case",
                output_path=output_path,
                start=10,
                end=11,
                fps=None,
                write_mp4=False,
                depth_backend="both",
                ffs_repo="C:/external/fake",
                ffs_model_path="C:/external/fake/model.pth",
                ffs_scale=1.0,
                ffs_valid_iters=4,
                ffs_max_disp=64,
                ffs_radius_outlier_filter=True,
                ffs_radius_outlier_radius_m=0.01,
                ffs_radius_outlier_nb_points=3,
                ffs_native_like_postprocess=False,
                write_ffs_float_m=True,
                fail_if_no_ir_stereo=True,
            )
            align_case(args, runner_factory=FakeOutlierRunner)

            aligned_case = output_path / "sample_case"
            copied_depth = np.load(aligned_case / "depth" / "0" / "0.npy")
            original_depth = np.load(case_dir / "depth" / "0" / "10.npy")
            self.assertTrue(np.array_equal(copied_depth, original_depth))
            self.assertTrue((aligned_case / "depth_ffs_original" / "0" / "0.npy").is_file())
            self.assertTrue((aligned_case / "depth_ffs_float_m_original" / "0" / "0.npy").is_file())

            filtered_depth = np.load(aligned_case / "depth_ffs" / "0" / "0.npy")
            raw_depth = np.load(aligned_case / "depth_ffs_original" / "0" / "0.npy")
            self.assertEqual(int(filtered_depth[0, 0]), 0)
            self.assertGreater(int(raw_depth[0, 0]), 0)

            metadata_ext = json.loads((aligned_case / ALIGNED_METADATA_EXT_FILENAME).read_text(encoding="utf-8"))
            self.assertTrue(metadata_ext["ffs_radius_outlier_filter_enabled"])
            self.assertEqual(
                metadata_ext["streams_present"],
                [
                    "color",
                    "ir_left",
                    "ir_right",
                    "depth",
                    "depth_ffs",
                    "depth_ffs_float_m",
                    "depth_ffs_original",
                    "depth_ffs_float_m_original",
                ],
            )


if __name__ == "__main__":
    unittest.main()
