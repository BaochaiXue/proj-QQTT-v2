from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from data_process.record_data_align import align_case


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "record_data_align_minimal"


class FakeRunner:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def run_pair(self, left_image, right_image, *, K_ir_left, baseline_m):
        height, width = left_image.shape[:2]
        disparity = np.full((height, width), 10.0, dtype=np.float32)
        depth_ir = np.full((height, width), 1.0, dtype=np.float32)
        return {
            "disparity": disparity,
            "depth_ir_left_m": depth_ir,
            "K_ir_left_used": np.asarray(K_ir_left, dtype=np.float32),
            "baseline_m": float(baseline_m),
            "scale": 1.0,
            "valid_iters": 4,
            "max_disp": 64,
        }


def make_v2_case(case_dir: Path) -> None:
    shutil.copytree(FIXTURE, case_dir)
    shutil.copytree(case_dir / "color", case_dir / "ir_left")
    shutil.copytree(case_dir / "color", case_dir / "ir_right")

    metadata_path = case_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    intrinsics = metadata["intrinsics"]
    identity = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    metadata.update(
        {
            "schema_version": "qqtt_recording_v2",
            "logical_camera_names": ["cam0", "cam1", "cam2"],
            "capture_mode": "stereo_ir",
            "streams_present": ["color", "ir_left", "ir_right"],
            "K_color": intrinsics,
            "K_ir_left": intrinsics,
            "K_ir_right": intrinsics,
            "T_ir_left_to_right": [
                [[1.0, 0.0, 0.0, -0.095], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
                for _ in intrinsics
            ],
            "T_ir_left_to_color": [identity for _ in intrinsics],
            "ir_baseline_m": [0.095 for _ in intrinsics],
            "depth_scale_m_per_unit": [0.001 for _ in intrinsics],
            "depth_encoding": "uint16_meters_scaled_invalid_zero",
        }
    )
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")


class RecordDataAlignFfsSmokeTest(unittest.TestCase):
    def test_aligns_case_with_ffs_backend(self) -> None:
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
                depth_backend="ffs",
                ffs_repo="C:/external/fake",
                ffs_model_path="C:/external/fake/model.pth",
                ffs_scale=1.0,
                ffs_valid_iters=4,
                ffs_max_disp=64,
                ffs_native_like_postprocess=False,
                write_ffs_float_m=True,
                fail_if_no_ir_stereo=True,
            )
            align_case(args, runner_factory=FakeRunner)

            aligned_case = output_path / "sample_case"
            self.assertTrue((aligned_case / "depth" / "0" / "0.npy").is_file())
            self.assertTrue((aligned_case / "depth_ffs_float_m" / "0" / "0.npy").is_file())
            self.assertTrue((aligned_case / "ir_left" / "0" / "0.png").is_file())
            self.assertTrue((aligned_case / "ir_right" / "0" / "0.png").is_file())

            depth = np.load(aligned_case / "depth" / "0" / "0.npy")
            self.assertEqual(depth.dtype, np.uint16)
            self.assertGreater(int((depth > 0).sum()), 0)
            self.assertGreaterEqual(int((depth == 0).sum()), 0)

            metadata = json.loads((aligned_case / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["depth_backend_used"], "ffs")
            self.assertEqual(metadata["depth_source_for_depth_dir"], "ffs")
            self.assertIn("ffs_config", metadata)
            self.assertFalse(metadata["ffs_native_like_postprocess_enabled"])

    def test_aligns_case_with_ffs_native_like_aux_streams(self) -> None:
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
                depth_backend="ffs",
                ffs_repo="C:/external/fake",
                ffs_model_path="C:/external/fake/model.pth",
                ffs_scale=1.0,
                ffs_valid_iters=4,
                ffs_max_disp=64,
                ffs_native_like_postprocess=True,
                write_ffs_float_m=False,
                fail_if_no_ir_stereo=True,
            )
            align_case(args, runner_factory=FakeRunner)

            aligned_case = output_path / "sample_case"
            self.assertTrue((aligned_case / "depth" / "0" / "0.npy").is_file())
            self.assertTrue((aligned_case / "depth_ffs_native_like_postprocess" / "0" / "0.npy").is_file())
            self.assertTrue((aligned_case / "depth_ffs_native_like_postprocess_float_m" / "0" / "0.npy").is_file())

            metadata = json.loads((aligned_case / "metadata.json").read_text(encoding="utf-8"))
            self.assertTrue(metadata["ffs_native_like_postprocess_enabled"])


if __name__ == "__main__":
    unittest.main()
