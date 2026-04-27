from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from data_process.aligned_case_metadata import ALIGNED_METADATA_EXT_FILENAME, write_split_aligned_metadata
from data_process.visualization.io_case import load_case_metadata


def _identity_intrinsics(count: int) -> list[list[list[float]]]:
    return [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]] * count


def _write_legacy_metadata(case_dir: Path, *, serial_numbers: list[str], intrinsics: list) -> None:
    (case_dir / "metadata.json").write_text(
        json.dumps(
            {
                "intrinsics": intrinsics,
                "serial_numbers": serial_numbers,
                "fps": 30,
                "WH": [848, 480],
                "frame_num": 1,
                "start_step": 0,
                "end_step": 0,
            }
        ),
        encoding="utf-8",
    )


class AlignedMetadataLoaderSmokeTest(unittest.TestCase):
    def test_loader_merges_metadata_ext_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            case_dir = Path(tmp_dir) / "case"
            case_dir.mkdir(parents=True, exist_ok=True)
            write_split_aligned_metadata(
                case_dir,
                {
                    "schema_version": "qqtt_aligned_case_v2",
                    "serial_numbers": ["a", "b", "c"],
                    "calibration_reference_serials": ["a", "b", "c"],
                    "fps": 30,
                    "WH": [848, 480],
                    "frame_num": 2,
                    "start_step": 10,
                    "end_step": 11,
                    "intrinsics": [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]] * 3,
                    "K_color": [[[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]]] * 3,
                    "depth_backend_used": "realsense",
                    "depth_scale_m_per_unit": [0.001, 0.001, 0.001],
                },
            )

            metadata = load_case_metadata(case_dir)
            self.assertEqual(metadata["frame_num"], 2)
            self.assertEqual(metadata["depth_backend_used"], "realsense")
            self.assertIn("K_color", metadata)

    def test_loader_supports_legacy_only_metadata_without_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            case_dir = Path(tmp_dir) / "case"
            case_dir.mkdir(parents=True, exist_ok=True)
            (case_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "intrinsics": _identity_intrinsics(3),
                        "serial_numbers": ["a", "b", "c"],
                        "fps": 30,
                        "WH": [848, 480],
                        "frame_num": 1,
                        "start_step": 0,
                        "end_step": 0,
                    }
                ),
                encoding="utf-8",
            )

            metadata = load_case_metadata(case_dir)
            self.assertEqual(metadata["frame_num"], 1)
            self.assertFalse((case_dir / ALIGNED_METADATA_EXT_FILENAME).exists())

    def test_loader_keeps_legacy_fields_from_metadata_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            case_dir = Path(tmp_dir) / "case"
            case_dir.mkdir(parents=True, exist_ok=True)
            (case_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "intrinsics": _identity_intrinsics(3),
                        "serial_numbers": ["a", "b", "c"],
                        "fps": 30,
                        "WH": [848, 480],
                        "frame_num": 1,
                        "start_step": 0,
                        "end_step": 0,
                    }
                ),
                encoding="utf-8",
            )
            (case_dir / ALIGNED_METADATA_EXT_FILENAME).write_text(
                json.dumps(
                    {
                        "fps": 99,
                        "depth_backend_used": "ffs",
                    }
                ),
                encoding="utf-8",
            )

            metadata = load_case_metadata(case_dir)
            self.assertEqual(metadata["fps"], 30)
            self.assertEqual(metadata["depth_backend_used"], "ffs")

    def test_loader_supports_existing_unsplit_metadata_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            case_dir = Path(tmp_dir) / "case"
            case_dir.mkdir(parents=True, exist_ok=True)
            (case_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "intrinsics": _identity_intrinsics(3),
                        "serial_numbers": ["a", "b", "c"],
                        "fps": 30,
                        "WH": [848, 480],
                        "frame_num": 1,
                        "start_step": 0,
                        "end_step": 0,
                        "K_color": [[[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]]] * 3,
                    }
                ),
                encoding="utf-8",
            )

            metadata = load_case_metadata(case_dir)
            self.assertEqual(metadata["frame_num"], 1)
            self.assertIn("K_color", metadata)

    def test_loader_rejects_duplicate_serial_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            case_dir = Path(tmp_dir) / "case"
            case_dir.mkdir(parents=True, exist_ok=True)
            _write_legacy_metadata(
                case_dir,
                serial_numbers=["a", "b", "b"],
                intrinsics=_identity_intrinsics(3),
            )

            with self.assertRaisesRegex(ValueError, "duplicate serials"):
                load_case_metadata(case_dir)

    def test_loader_rejects_per_camera_length_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            case_dir = Path(tmp_dir) / "case"
            case_dir.mkdir(parents=True, exist_ok=True)
            _write_legacy_metadata(
                case_dir,
                serial_numbers=["a", "b", "c"],
                intrinsics=_identity_intrinsics(2),
            )

            with self.assertRaisesRegex(ValueError, "does not match serial_numbers"):
                load_case_metadata(case_dir)

    def test_loader_rejects_missing_calibration_reference_serial(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            case_dir = Path(tmp_dir) / "case"
            case_dir.mkdir(parents=True, exist_ok=True)
            _write_legacy_metadata(
                case_dir,
                serial_numbers=["a", "b", "c"],
                intrinsics=_identity_intrinsics(3),
            )
            (case_dir / ALIGNED_METADATA_EXT_FILENAME).write_text(
                json.dumps({"calibration_reference_serials": ["a", "b"]}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "does not cover case serial_numbers"):
                load_case_metadata(case_dir)

    def test_loader_accepts_reordered_calibration_reference_serials(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            case_dir = Path(tmp_dir) / "case"
            case_dir.mkdir(parents=True, exist_ok=True)
            write_split_aligned_metadata(
                case_dir,
                {
                    "schema_version": "qqtt_aligned_case_v2",
                    "serial_numbers": ["a", "c", "b"],
                    "calibration_reference_serials": ["a", "b", "c"],
                    "fps": 30,
                    "WH": [848, 480],
                    "frame_num": 1,
                    "start_step": 0,
                    "end_step": 0,
                    "intrinsics": _identity_intrinsics(3),
                    "K_color": _identity_intrinsics(3),
                },
            )

            metadata = load_case_metadata(case_dir)
            self.assertEqual(metadata["serial_numbers"], ["a", "c", "b"])
            self.assertEqual(metadata["calibration_reference_serials"], ["a", "b", "c"])


if __name__ == "__main__":
    unittest.main()
