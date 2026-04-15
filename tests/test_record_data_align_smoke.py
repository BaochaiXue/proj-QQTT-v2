from __future__ import annotations

import json
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from data_process.aligned_case_metadata import (
    ALIGNED_METADATA_EXT_FILENAME,
    LEGACY_ALIGNED_METADATA_KEYS,
)
from data_process.record_data_align import align_case


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "record_data_align_minimal"
SCRIPT = ROOT / "data_process" / "record_data_align.py"


class RecordDataAlignSmokeTest(unittest.TestCase):
    def test_aligns_minimal_case(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            base_path = tmp_root / "data_collect"
            case_dir = base_path / "sample_case"
            shutil.copytree(FIXTURE, case_dir)

            output_path = tmp_root / "data"
            cmd = [
                sys.executable,
                str(SCRIPT),
                "--base_path",
                str(base_path),
                "--output_path",
                str(output_path),
                "--case_name",
                "sample_case",
                "--start",
                "10",
                "--end",
                "11",
            ]
            subprocess.run(cmd, check=True, cwd=ROOT)

            aligned_case = output_path / "sample_case"
            self.assertTrue((aligned_case / "calibrate.pkl").is_file())
            self.assertTrue((aligned_case / "metadata.json").is_file())
            self.assertTrue((aligned_case / ALIGNED_METADATA_EXT_FILENAME).is_file())

            metadata = json.loads((aligned_case / "metadata.json").read_text(encoding="utf-8"))
            metadata_ext = json.loads((aligned_case / ALIGNED_METADATA_EXT_FILENAME).read_text(encoding="utf-8"))
            self.assertEqual(set(metadata.keys()), set(LEGACY_ALIGNED_METADATA_KEYS))
            self.assertEqual(metadata["frame_num"], 2)
            self.assertEqual(metadata["start_step"], 10)
            self.assertEqual(metadata["end_step"], 11)
            self.assertEqual(len(metadata["serial_numbers"]), 3)
            self.assertEqual(metadata_ext["depth_backend_used"], "realsense")
            self.assertEqual(metadata_ext["depth_source_for_depth_dir"], "realsense")
            self.assertEqual(metadata_ext["streams_present"], ["color", "depth"])

            for camera_idx in range(3):
                color_dir = aligned_case / "color" / str(camera_idx)
                depth_dir = aligned_case / "depth" / str(camera_idx)
                self.assertTrue((color_dir / "0.png").is_file())
                self.assertTrue((color_dir / "1.png").is_file())
                self.assertTrue((depth_dir / "0.npy").is_file())
                self.assertTrue((depth_dir / "1.npy").is_file())

    def test_formal_different_types_export_reorders_calibrate_to_case_serial_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            base_path = tmp_root / "data_collect"
            case_dir = base_path / "sample_case"
            shutil.copytree(FIXTURE, case_dir)

            metadata_path = case_dir / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["serial_numbers"] = ["cam0", "cam2", "cam1"]
            metadata["calibration_reference_serials"] = ["cam0", "cam1", "cam2"]
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            output_path = tmp_root / "data" / "different_types"
            args = mock.Mock(
                base_path=base_path,
                case_name="sample_case",
                output_path=output_path,
                start=10,
                end=11,
                fps=None,
                write_mp4=False,
                depth_backend="realsense",
                ffs_repo=None,
                ffs_model_path=None,
                ffs_scale=1.0,
                ffs_valid_iters=8,
                ffs_max_disp=192,
                ffs_native_like_postprocess=False,
                write_ffs_float_m=False,
                fail_if_no_ir_stereo=False,
            )

            def fake_write_mp4s(ffmpeg_bin: str, output_case_dir: Path, num_cameras: int, fps: int) -> None:
                for camera_idx in range(num_cameras):
                    (output_case_dir / "color" / f"{camera_idx}.mp4").write_bytes(b"mp4")

            with (
                mock.patch("data_process.record_data_align.find_ffmpeg", return_value="fake_ffmpeg"),
                mock.patch("data_process.record_data_align.write_mp4s", side_effect=fake_write_mp4s),
            ):
                align_case(args)

            aligned_case = output_path / "sample_case"
            with (aligned_case / "calibrate.pkl").open("rb") as handle:
                c2ws = pickle.load(handle)
            self.assertEqual([float(item[0][3]) for item in c2ws], [0.0, 2.0, 1.0])

    def test_formal_different_types_export_writes_color_mp4_sidecars_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            base_path = tmp_root / "data_collect"
            case_dir = base_path / "sample_case"
            shutil.copytree(FIXTURE, case_dir)

            output_path = tmp_root / "data" / "different_types"

            args = mock.Mock(
                base_path=base_path,
                case_name="sample_case",
                output_path=output_path,
                start=10,
                end=11,
                fps=None,
                write_mp4=False,
                depth_backend="realsense",
                ffs_repo=None,
                ffs_model_path=None,
                ffs_scale=1.0,
                ffs_valid_iters=8,
                ffs_max_disp=192,
                ffs_native_like_postprocess=False,
                write_ffs_float_m=False,
                fail_if_no_ir_stereo=False,
            )

            def fake_write_mp4s(ffmpeg_bin: str, output_case_dir: Path, num_cameras: int, fps: int) -> None:
                for camera_idx in range(num_cameras):
                    (output_case_dir / "color" / f"{camera_idx}.mp4").write_bytes(b"mp4")

            with (
                mock.patch("data_process.record_data_align.find_ffmpeg", return_value="fake_ffmpeg"),
                mock.patch("data_process.record_data_align.write_mp4s", side_effect=fake_write_mp4s),
            ):
                align_case(args)

            aligned_case = output_path / "sample_case"
            for camera_idx in range(3):
                self.assertTrue((aligned_case / "color" / f"{camera_idx}.mp4").is_file())


if __name__ == "__main__":
    unittest.main()
