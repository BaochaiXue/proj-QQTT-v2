from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import record_data
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
from record_data import (
    copy_table_calibration_into_case,
    validate_table_calibration_for_case,
)


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

    def test_copy_uses_validated_snapshot_when_source_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            case_dir = root / "case"
            _write_case_metadata(case_dir, ["cam-a"])
            table_path = _write_sample_table_calibration(root, ["cam-a"])
            validated = validate_table_calibration_for_case(
                table_calibrate_path=table_path,
                serial_numbers=["cam-a"],
            )
            original_table_bytes = table_path.read_bytes()
            original_sidecar_bytes = table_calibration_metadata_path_for(
                table_path
            ).read_bytes()

            _write_sample_table_calibration(root, ["cam-a", "cam-c"])

            copy_table_calibration_into_case(
                table_calibrate_path=table_path,
                output_path=case_dir,
                serial_numbers=["cam-a"],
                validated_table_calibration=validated,
            )

            copied_sidecar = case_dir / "table_calibrate_metadata.json"
            self.assertEqual(
                (case_dir / "table_calibrate.pkl").read_bytes(),
                original_table_bytes,
            )
            self.assertEqual(copied_sidecar.read_bytes(), original_sidecar_bytes)
            copied_metadata = json.loads(copied_sidecar.read_text(encoding="utf-8"))
            case_metadata = json.loads(
                (case_dir / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                copied_metadata["table_calibration_reference_serials"],
                ["cam-a"],
            )
            self.assertEqual(
                case_metadata["table_calibration_reference_serials"],
                ["cam-a"],
            )

    def test_main_validates_table_calibration_before_recording(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            table_path = _write_sample_table_calibration(root, ["cam-a"])

            class FakeCameraSystem:
                instances = []

                def __init__(self, **_kwargs) -> None:
                    self.serial_numbers = ["cam-b"]
                    self.record_called = False
                    self.stop_called = False
                    self.realsense = SimpleNamespace(stop=lambda: None)
                    self.__class__.instances.append(self)

                def stop(self) -> None:
                    self.stop_called = True

                def record(self, *, output_path: str, max_frames) -> None:
                    self.record_called = True

            allowed_preflight = SimpleNamespace(
                allowed_to_record=True,
                operator_status="supported",
                reason="",
                probe_results_md="",
            )
            argv = [
                "record_data.py",
                "--output_dir",
                str(root / "recordings"),
                "--case_name",
                "case",
                "--calibrate_path",
                str(root / "missing_calibrate.pkl"),
                "--serials",
                "cam-b",
                "--table-calibrate",
                str(table_path),
            ]
            with (
                patch.object(sys, "argv", argv),
                patch("qqtt.env.CameraSystem", FakeCameraSystem),
                patch.object(
                    record_data,
                    "evaluate_capture_preflight",
                    return_value=allowed_preflight,
                ),
                patch.object(
                    record_data,
                    "format_capture_preflight_summary",
                    return_value="preflight ok",
                ),
            ):
                with self.assertRaisesRegex(Exception, "does not cover serials"):
                    record_data.main()

            self.assertEqual(len(FakeCameraSystem.instances), 1)
            self.assertFalse(FakeCameraSystem.instances[0].record_called)
            self.assertTrue(FakeCameraSystem.instances[0].stop_called)


if __name__ == "__main__":
    unittest.main()
