from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from data_process.aligned_case_metadata import (
    ALIGNED_METADATA_EXT_FILENAME,
    load_aligned_metadata,
    write_split_aligned_metadata,
)
from data_process.record_data_align import write_aligned_table_calibration_file
from qqtt.env.camera.table_calibration import (
    TABLE_WORLD_FRAME_KIND,
    build_table_calibration_metadata,
    load_table_calibration_metadata,
    load_table_calibration_transforms,
    table_calibration_metadata_path_for,
    write_table_calibration_files,
)


def _transform_with_translation(x: float) -> np.ndarray:
    transform = np.eye(4, dtype=np.float32)
    transform[:3, 3] = np.array([x, 0.0, 0.0], dtype=np.float32)
    return transform


def _identity_intrinsics(count: int) -> list[list[list[float]]]:
    return [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]] * count


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

    def test_write_aligned_table_calibration_file_rejects_declared_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            case_dir = root / "raw_case"
            output_case_dir = root / "aligned_case"
            case_dir.mkdir()

            with self.assertRaisesRegex(FileNotFoundError, "declared table calibration file"):
                write_aligned_table_calibration_file(
                    case_dir=case_dir,
                    output_case_dir=output_case_dir,
                    metadata={
                        "serial_numbers": ["a"],
                        "table_calibration_path": "table/noncanonical.pkl",
                        "table_calibration_metadata_path": "table/noncanonical_metadata.json",
                    },
                )

            self.assertFalse((output_case_dir / "table_calibrate.pkl").exists())

    def test_write_aligned_table_calibration_file_uses_declared_noncanonical_source_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            case_dir = root / "raw_case"
            output_case_dir = root / "aligned_case"
            source_table = case_dir / "calibration" / "table_source.pkl"
            source_metadata = case_dir / "metadata" / "table_source.json"
            case_dir.mkdir()

            transform = _transform_with_translation(3.0)
            write_table_calibration_files(
                source_table,
                [transform],
                _metadata_for(["a"]),
            )
            source_metadata.parent.mkdir()
            table_calibration_metadata_path_for(source_table).rename(source_metadata)

            written = write_aligned_table_calibration_file(
                case_dir=case_dir,
                output_case_dir=output_case_dir,
                metadata={
                    "serial_numbers": ["a"],
                    "table_calibration_path": "calibration/table_source.pkl",
                    "table_calibration_metadata_path": "metadata/table_source.json",
                },
            )

            self.assertTrue(written)
            self.assertTrue((output_case_dir / "table_calibrate.pkl").is_file())
            self.assertTrue((output_case_dir / "table_calibrate_metadata.json").is_file())
            self.assertFalse((output_case_dir / "calibration" / "table_source.pkl").exists())
            loaded = load_table_calibration_transforms(
                output_case_dir / "table_calibrate.pkl",
                serial_numbers=["a"],
                table_calibration_reference_serials=["a"],
            )
            np.testing.assert_allclose(loaded[0], transform, atol=1e-6)

    def test_split_aligned_metadata_writes_table_fields_through_strict_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            case_dir = Path(tmpdir) / "case"
            case_dir.mkdir()

            write_split_aligned_metadata(
                case_dir,
                {
                    "schema_version": "qqtt_aligned_case_v2",
                    "source_case_name": "raw_case",
                    "serial_numbers": ["a"],
                    "calibration_reference_serials": ["a"],
                    "source_calibration_reference_serials": ["a"],
                    "logical_camera_names": ["front"],
                    "fps": 30,
                    "WH": [640, 480],
                    "frame_num": 1,
                    "start_step": 0,
                    "end_step": 0,
                    "capture_mode": "rgbd",
                    "streams_present": ["color", "depth"],
                    "depth_backend_used": "realsense",
                    "depth_source_for_depth_dir": "realsense",
                    "ffs_native_like_postprocess_enabled": False,
                    "depth_scale_m_per_unit": [0.001],
                    "depth_encoding": "uint16_meters_scaled_invalid_zero",
                    "intrinsics": _identity_intrinsics(1),
                    "K_color": _identity_intrinsics(1),
                    "K_ir_left": [None],
                    "K_ir_right": [None],
                    "T_ir_left_to_right": [None],
                    "T_ir_left_to_color": [None],
                    "ir_baseline_m": [None],
                    "source_streams_present": ["color", "depth"],
                    "ffs_confidence_filter": {"enabled": False},
                    "ffs_radius_outlier_filter_enabled": False,
                    "ffs_radius_outlier_filter": {"mode": "disabled"},
                    "table_calibration_path": "table_calibrate.pkl",
                    "table_calibration_metadata_path": "table_calibrate_metadata.json",
                    "table_world_frame_kind": TABLE_WORLD_FRAME_KIND,
                },
            )

            metadata_ext = load_aligned_metadata(case_dir)[1]
            self.assertEqual(metadata_ext["table_calibration_path"], "table_calibrate.pkl")
            self.assertEqual(
                metadata_ext["table_calibration_metadata_path"],
                "table_calibrate_metadata.json",
            )
            self.assertEqual(metadata_ext["table_world_frame_kind"], TABLE_WORLD_FRAME_KIND)
            raw_metadata_ext = (case_dir / ALIGNED_METADATA_EXT_FILENAME).read_text(encoding="utf-8")
            self.assertIn("table_calibration_path", raw_metadata_ext)


if __name__ == "__main__":
    unittest.main()
