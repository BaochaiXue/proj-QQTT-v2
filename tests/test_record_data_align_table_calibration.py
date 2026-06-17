from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from data_process.record_data_align import write_aligned_table_calibration_file
from qqtt.env.camera.table_calibration import (
    build_table_calibration_metadata,
    load_table_calibration_metadata,
    load_table_calibration_transforms,
    write_table_calibration_files,
)


def _transform_with_translation(x: float) -> np.ndarray:
    transform = np.eye(4, dtype=np.float32)
    transform[:3, 3] = np.array([x, 0.0, 0.0], dtype=np.float32)
    return transform


def _metadata_for(serial_numbers: list[str]) -> dict[str, object]:
    return build_table_calibration_metadata(
        serial_numbers=serial_numbers,
        WH=[640, 480],
        fps=30,
        transform_count=len(serial_numbers),
        calibration_board={"name": "calibio-12x9-30mm"},
        max_reprojection_error_px=0.5,
        min_corner_fraction=60 / 88,
        min_charuco_corners=60,
        per_camera_reprojection_error=[0.10 + index * 0.01 for index in range(len(serial_numbers))],
        per_camera_corner_count=[60 + index for index in range(len(serial_numbers))],
        per_camera_corner_fraction=[(60 + index) / 88 for index in range(len(serial_numbers))],
        distortion_used=True,
        distortion_model_by_camera=["brown_conrady" for _ in serial_numbers],
        distortion_coeffs_by_camera=[
            [0.1 + index, 0.01, 0.001, 0.0001, 0.0]
            for index in range(len(serial_numbers))
        ],
    )


class RecordDataAlignTableCalibrationTest(unittest.TestCase):
    def test_write_aligned_table_calibration_file_preserves_raw_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            case_dir = root / "raw_case"
            output_case_dir = root / "aligned_case"
            case_dir.mkdir()

            transform = _transform_with_translation(1.0)
            write_table_calibration_files(
                case_dir / "table_calibrate.pkl",
                [transform],
                _metadata_for(["a"]),
            )

            written = write_aligned_table_calibration_file(
                case_dir=case_dir,
                output_case_dir=output_case_dir,
                metadata={"serial_numbers": ["a"], "logical_camera_names": ["front"]},
            )

            self.assertTrue(written)
            self.assertTrue((output_case_dir / "table_calibrate.pkl").is_file())
            self.assertTrue((output_case_dir / "table_calibrate_metadata.json").is_file())
            loaded = load_table_calibration_transforms(
                output_case_dir / "table_calibrate.pkl",
                serial_numbers=["a"],
                table_calibration_reference_serials=["a"],
            )
            np.testing.assert_allclose(loaded[0], transform, atol=1e-6)
            aligned_metadata = load_table_calibration_metadata(output_case_dir / "table_calibrate.pkl")
            self.assertEqual(aligned_metadata["serial_numbers"], ["a"])
            self.assertEqual(aligned_metadata["table_calibration_reference_serials"], ["a"])
            self.assertEqual(aligned_metadata["logical_camera_names"], ["front"])

    def test_write_aligned_table_calibration_file_reorders_and_subsets_to_aligned_serials(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            case_dir = root / "raw_case"
            output_case_dir = root / "aligned_case"
            case_dir.mkdir()

            first = _transform_with_translation(1.0)
            second = _transform_with_translation(2.0)
            write_table_calibration_files(
                case_dir / "table_calibrate.pkl",
                [first, second],
                _metadata_for(["a", "b"]),
            )

            written = write_aligned_table_calibration_file(
                case_dir=case_dir,
                output_case_dir=output_case_dir,
                metadata={
                    "serial_numbers": ["b"],
                    "table_calibration_reference_serials": ["a", "b"],
                    "logical_camera_names": ["right"],
                },
            )

            self.assertTrue(written)
            loaded = load_table_calibration_transforms(
                output_case_dir / "table_calibrate.pkl",
                serial_numbers=["b"],
                table_calibration_reference_serials=["b"],
            )
            self.assertEqual(len(loaded), 1)
            np.testing.assert_allclose(loaded[0], second, atol=1e-6)
            aligned_metadata = load_table_calibration_metadata(output_case_dir / "table_calibrate.pkl")
            self.assertEqual(aligned_metadata["serial_numbers"], ["b"])
            self.assertEqual(aligned_metadata["table_calibration_reference_serials"], ["b"])
            self.assertEqual(aligned_metadata["logical_camera_names"], ["right"])
            self.assertEqual(aligned_metadata["transform_count"], 1)
            self.assertEqual(aligned_metadata["per_camera_reprojection_error"], [0.11])
            self.assertEqual(aligned_metadata["per_camera_corner_count"], [61])
            self.assertEqual(aligned_metadata["per_camera_corner_fraction"], [61 / 88])
            self.assertEqual(aligned_metadata["distortion_model_by_camera"], ["brown_conrady"])
            self.assertEqual(aligned_metadata["distortion_coeffs_by_camera"], [[1.1, 0.01, 0.001, 0.0001, 0.0]])

    def test_write_aligned_table_calibration_file_is_noop_when_raw_file_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            case_dir = root / "raw_case"
            output_case_dir = root / "aligned_case"
            case_dir.mkdir()

            written = write_aligned_table_calibration_file(
                case_dir=case_dir,
                output_case_dir=output_case_dir,
                metadata={"serial_numbers": ["a"]},
            )

            self.assertFalse(written)
            self.assertFalse((output_case_dir / "table_calibrate.pkl").exists())
            self.assertFalse((output_case_dir / "table_calibrate_metadata.json").exists())


if __name__ == "__main__":
    unittest.main()
