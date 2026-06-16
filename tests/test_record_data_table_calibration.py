from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from qqtt.env.camera.calibration_boards import (
    charuco_board_config_to_metadata,
    get_calibration_board_config,
)
from qqtt.env.camera.table_calibration import (
    TABLE_WORLD_FRAME_KIND,
    build_table_calibration_metadata,
    table_calibration_metadata_path_for,
    write_table_calibration_files,
)
from record_data import copy_table_calibration_into_case


def _write_case_metadata(case_dir: Path, serial_numbers: list[str]) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "serial_numbers": list(serial_numbers),
        "existing_field": "preserved",
    }
    (case_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def _write_sample_table_calibration(root: Path, serial_numbers: list[str]) -> Path:
    table_path = root / "source_table_calibrate.pkl"
    board = get_calibration_board_config("calibio-12x9-30mm")
    metadata = build_table_calibration_metadata(
        serial_numbers=list(serial_numbers),
        WH=[640, 480],
        fps=30,
        transform_count=len(serial_numbers),
        calibration_board=charuco_board_config_to_metadata(board),
        max_reprojection_error_px=1.0,
        min_corner_fraction=0.6,
        min_charuco_corners=53,
        per_camera_reprojection_error=[0.2 for _ in serial_numbers],
        per_camera_corner_count=[60 for _ in serial_numbers],
        per_camera_corner_fraction=[60 / 88 for _ in serial_numbers],
    )
    transforms = [np.eye(4, dtype=np.float32) for _ in serial_numbers]
    write_table_calibration_files(table_path, transforms, metadata)
    return table_path


class RecordDataTableCalibrationTest(unittest.TestCase):
    def test_copies_table_calibration_and_updates_case_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            case_dir = root / "case"
            _write_case_metadata(case_dir, ["cam-a"])
            table_path = _write_sample_table_calibration(root, ["cam-a"])
            source_sidecar = table_calibration_metadata_path_for(table_path)

            copy_table_calibration_into_case(
                table_calibrate_path=table_path,
                output_path=case_dir,
                serial_numbers=["cam-a"],
            )

            copied_table = case_dir / "table_calibrate.pkl"
            copied_sidecar = case_dir / "table_calibrate_metadata.json"
            self.assertEqual(copied_table.read_bytes(), table_path.read_bytes())
            self.assertEqual(copied_sidecar.read_bytes(), source_sidecar.read_bytes())

            metadata = json.loads(
                (case_dir / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["existing_field"], "preserved")
            self.assertEqual(metadata["table_calibration_path"], "table_calibrate.pkl")
            self.assertEqual(
                metadata["table_calibration_metadata_path"],
                "table_calibrate_metadata.json",
            )
            self.assertEqual(metadata["table_world_frame_kind"], TABLE_WORLD_FRAME_KIND)
            self.assertEqual(
                metadata["table_calibration_reference_serials"],
                ["cam-a"],
            )

    def test_rejects_table_calibration_that_does_not_cover_case_serials(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            case_dir = root / "case"
            _write_case_metadata(case_dir, ["cam-b"])
            table_path = _write_sample_table_calibration(root, ["cam-a"])

            with self.assertRaisesRegex(Exception, "does not cover serials"):
                copy_table_calibration_into_case(
                    table_calibrate_path=table_path,
                    output_path=case_dir,
                    serial_numbers=["cam-b"],
                )

            self.assertFalse((case_dir / "table_calibrate.pkl").exists())
            self.assertFalse((case_dir / "table_calibrate_metadata.json").exists())


if __name__ == "__main__":
    unittest.main()
