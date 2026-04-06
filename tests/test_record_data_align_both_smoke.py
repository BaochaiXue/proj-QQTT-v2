from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from data_process.record_data_align import align_case
from tests.test_record_data_align_ffs_smoke import FIXTURE, FakeRunner, make_v2_case


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
                write_ffs_float_m=True,
                fail_if_no_ir_stereo=True,
            )
            align_case(args, runner_factory=FakeRunner)

            aligned_case = output_path / "sample_case"
            self.assertTrue((aligned_case / "depth" / "0" / "0.npy").is_file())
            self.assertTrue((aligned_case / "depth_ffs" / "0" / "0.npy").is_file())
            self.assertTrue((aligned_case / "depth_ffs_float_m" / "0" / "0.npy").is_file())

            copied_depth = np.load(aligned_case / "depth" / "0" / "0.npy")
            original_depth = np.load(case_dir / "depth" / "0" / "10.npy")
            self.assertTrue(np.array_equal(copied_depth, original_depth))

            ffs_depth = np.load(aligned_case / "depth_ffs" / "0" / "0.npy")
            self.assertEqual(ffs_depth.dtype, np.uint16)
            self.assertGreater(int((ffs_depth > 0).sum()), 0)

            metadata = json.loads((aligned_case / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["depth_backend_used"], "both")
            self.assertEqual(metadata["depth_source_for_depth_dir"], "realsense")


if __name__ == "__main__":
    unittest.main()
