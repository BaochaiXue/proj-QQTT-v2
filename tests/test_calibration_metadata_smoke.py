from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from qqtt.env.camera.calibration_metadata import (
    build_calibration_metadata,
    calibration_metadata_path_for,
    load_calibration_reference_serials,
    write_calibration_metadata,
)


class CalibrationMetadataSmokeTest(unittest.TestCase):
    def test_writes_and_loads_reference_serials(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            calibrate_path = Path(tmp_dir) / "calibrate.pkl"
            metadata = build_calibration_metadata(
                serial_numbers=["cam_b", "cam_a", "cam_c"],
                WH=(848, 480),
                fps=30,
                transform_count=3,
                per_camera_reprojection_error=[0.01, 0.02, 0.03],
            )
            sidecar_path = write_calibration_metadata(calibrate_path, metadata)

            self.assertEqual(sidecar_path, Path(tmp_dir) / "calibrate_metadata.json")
            self.assertEqual(load_calibration_reference_serials(calibrate_path), ["cam_b", "cam_a", "cam_c"])

    def test_rejects_duplicate_serials(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate serials"):
            build_calibration_metadata(
                serial_numbers=["cam_a", "cam_a"],
                WH=(848, 480),
                fps=30,
                transform_count=2,
            )

    def test_rejects_sidecar_transform_count_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            calibrate_path = Path(tmp_dir) / "calibrate.pkl"
            sidecar_path = calibration_metadata_path_for(calibrate_path)
            sidecar_path.write_text(
                json.dumps(
                    {
                        "schema_version": "qqtt_calibration_v1",
                        "serial_numbers": ["cam_a", "cam_b", "cam_c"],
                        "calibration_reference_serials": ["cam_a", "cam_b", "cam_c"],
                        "transform_count": 2,
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "transform_count does not match"):
                load_calibration_reference_serials(calibrate_path)


if __name__ == "__main__":
    unittest.main()
