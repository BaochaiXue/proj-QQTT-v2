from __future__ import annotations

import json
import pickle
import tempfile
from pathlib import Path
import unittest

import numpy as np

from qqtt.env.camera.calibration_boards import get_calibration_board_config
from qqtt.env.camera.table_calibration import (
    TABLE_CALIBRATE_COMPATIBILITY_CONTRACT,
    TABLE_CALIBRATION_METADATA_SCHEMA_VERSION,
    TABLE_WORLD_FRAME_KIND,
    TableCalibrationLoadError,
    build_table_calibration_metadata,
    load_table_calibration_metadata,
    load_table_calibration_transforms,
    table_calibration_metadata_path_for,
    validate_table_calibration_acceptance,
    write_table_calibration_files,
)


class TableCalibrationContractTest(unittest.TestCase):
    def test_metadata_path_uses_output_stem(self) -> None:
        self.assertEqual(
            table_calibration_metadata_path_for(Path("table_calibrate.pkl")),
            Path("table_calibrate_metadata.json"),
        )
        self.assertEqual(
            table_calibration_metadata_path_for(Path("custom.pkl")),
            Path("custom_metadata.json"),
        )

    def test_writer_and_loader_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output = root / "table_calibrate.pkl"
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, 3] = np.array([0.1, -0.2, 0.3], dtype=np.float32)
            metadata = build_table_calibration_metadata(
                serial_numbers=["239222300781"],
                WH=[1280, 720],
                fps=5,
                transform_count=1,
                calibration_board={"name": "calibio-12x9-30mm"},
                max_reprojection_error_px=0.20,
                min_corner_fraction=0.60,
                min_charuco_corners=53,
                per_camera_reprojection_error=[0.12],
                per_camera_corner_count=[58],
                per_camera_corner_fraction=[0.659],
                distortion_used=True,
                distortion_model_by_camera=["inverse_brown_conrady"],
                distortion_coeffs_by_camera=[[0.0, 0.0, 0.0, 0.0, 0.0]],
                diagnostic_image_path="table_calibrate_diagnostic.png",
            )

            sidecar = write_table_calibration_files(output, [c2w], metadata)

            self.assertTrue(output.is_file())
            self.assertEqual(sidecar, root / "table_calibrate_metadata.json")
            loaded = load_table_calibration_transforms(
                output,
                serial_numbers=["239222300781"],
                table_calibration_reference_serials=["239222300781"],
            )
            np.testing.assert_allclose(loaded[0], c2w, atol=1e-6)
            loaded_metadata = load_table_calibration_metadata(output)
            self.assertEqual(
                loaded_metadata["schema_version"],
                TABLE_CALIBRATION_METADATA_SCHEMA_VERSION,
            )
            self.assertEqual(
                loaded_metadata["compatibility_contract"],
                TABLE_CALIBRATE_COMPATIBILITY_CONTRACT,
            )
            self.assertEqual(loaded_metadata["world_frame_kind"], TABLE_WORLD_FRAME_KIND)
            self.assertEqual(loaded_metadata["diagnostic_image_path"], "table_calibrate_diagnostic.png")

    def test_loader_rejects_missing_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "table_calibrate.pkl"
            with output.open("wb") as handle:
                pickle.dump([np.eye(4, dtype=np.float32)], handle)

            with self.assertRaisesRegex(TableCalibrationLoadError, "Missing table calibration metadata"):
                load_table_calibration_transforms(output)

    def test_loader_rejects_wrong_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "table_calibrate.pkl"
            with output.open("wb") as handle:
                pickle.dump([np.eye(4, dtype=np.float32)], handle)
            table_calibration_metadata_path_for(output).write_text(
                json.dumps({"schema_version": "wrong"}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(TableCalibrationLoadError, "Unsupported table calibration metadata schema"):
                load_table_calibration_transforms(output)

    def test_loader_rejects_invalid_transform_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "table_calibrate.pkl"
            metadata = build_table_calibration_metadata(
                serial_numbers=["cam0"],
                WH=[1280, 720],
                fps=5,
                transform_count=1,
                calibration_board={"name": "calibio-12x9-30mm"},
                max_reprojection_error_px=0.20,
                min_corner_fraction=0.60,
                min_charuco_corners=53,
                per_camera_reprojection_error=[0.10],
                per_camera_corner_count=[60],
                per_camera_corner_fraction=[0.68],
            )
            table_calibration_metadata_path_for(output).write_text(json.dumps(metadata), encoding="utf-8")
            with output.open("wb") as handle:
                pickle.dump([np.eye(3, dtype=np.float32)], handle)

            with self.assertRaisesRegex(TableCalibrationLoadError, "Unsupported table calibration transform shape"):
                load_table_calibration_transforms(output)

    def test_acceptance_requires_strict_corner_fraction_and_error(self) -> None:
        board_config = get_calibration_board_config("calibio-12x9-30mm")
        accepted_min = validate_table_calibration_acceptance(
            board_config=board_config,
            corner_count=53,
            reprojection_error_px=0.20,
            max_reprojection_error_px=0.20,
            min_corner_fraction=0.60,
        )
        self.assertEqual(accepted_min["min_charuco_corners"], 53)
        self.assertAlmostEqual(accepted_min["corner_fraction"], 53 / 88)

        with self.assertRaisesRegex(ValueError, "ChArUco corner count"):
            validate_table_calibration_acceptance(
                board_config=board_config,
                corner_count=52,
                reprojection_error_px=0.10,
                max_reprojection_error_px=0.20,
                min_corner_fraction=0.60,
            )

        with self.assertRaisesRegex(ValueError, "reprojection error"):
            validate_table_calibration_acceptance(
                board_config=board_config,
                corner_count=60,
                reprojection_error_px=0.21,
                max_reprojection_error_px=0.20,
                min_corner_fraction=0.60,
            )


if __name__ == "__main__":
    unittest.main()
