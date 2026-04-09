from __future__ import annotations

import pickle
from pathlib import Path
import tempfile
import unittest

import numpy as np

from data_process.visualization.calibration_frame import CALIBRATION_WORLD_FRAME_KIND
from data_process.visualization.calibration_io import (
    CalibrationLoadError,
    build_calibration_contract_summary,
    infer_calibration_mapping_mode,
    load_calibration_transforms,
)


class CalibrationContractHardeningTest(unittest.TestCase):
    def test_rejects_invalid_bottom_row(self) -> None:
        transforms = [np.eye(4, dtype=np.float32)]
        transforms[0][3, 3] = 0.0
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "calibrate.pkl"
            with path.open("wb") as handle:
                pickle.dump(transforms, handle)
            with self.assertRaises(CalibrationLoadError):
                load_calibration_transforms(path)

    def test_rejects_duplicate_calibration_reference_serials(self) -> None:
        transforms = [np.eye(4, dtype=np.float32) for _ in range(2)]
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "calibrate.pkl"
            with path.open("wb") as handle:
                pickle.dump(transforms, handle)
            with self.assertRaises(CalibrationLoadError):
                load_calibration_transforms(
                    path,
                    serial_numbers=["a"],
                    calibration_reference_serials=["a", "a"],
                )

    def test_builds_explicit_calibration_contract_summary(self) -> None:
        summary = build_calibration_contract_summary(
            calibrate_path="calibrate.pkl",
            transform_count=3,
            serial_numbers=["a", "b", "c"],
            calibration_reference_serials=["a", "b", "c"],
            mapping_mode=infer_calibration_mapping_mode(
                serial_numbers=["a", "b", "c"],
                calibration_reference_serials=["a", "b", "c"],
            ),
        )
        self.assertEqual(summary["world_frame_kind"], CALIBRATION_WORLD_FRAME_KIND)
        self.assertEqual(summary["transform_convention"], "camera_to_world_c2w")
