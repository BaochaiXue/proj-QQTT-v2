from __future__ import annotations

import pickle
from pathlib import Path
import tempfile
import unittest

import numpy as np

from data_process.visualization.calibration_io import CalibrationLoadError, load_calibration_transforms


class CalibrateLoaderSmokeTest(unittest.TestCase):
    def test_reorders_same_length_case_by_calibration_reference_serials(self) -> None:
        transforms = [
            np.array([[1, 0, 0, 10], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32),
            np.array([[1, 0, 0, 20], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32),
            np.array([[1, 0, 0, 30], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32),
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "calibrate.pkl"
            with path.open("wb") as handle:
                pickle.dump(transforms, handle)

            selected = load_calibration_transforms(
                path,
                serial_numbers=["cam_a", "cam_c", "cam_b"],
                calibration_reference_serials=["cam_a", "cam_b", "cam_c"],
            )

            self.assertEqual([float(item[0, 3]) for item in selected], [10.0, 30.0, 20.0])

    def test_loads_subset_by_serial_mapping(self) -> None:
        transforms = [
            np.eye(4, dtype=np.float32),
            np.array([[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32),
            np.array([[1, 0, 0, 2], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32),
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "calibrate.pkl"
            with path.open("wb") as handle:
                pickle.dump(transforms, handle)

            selected = load_calibration_transforms(
                path,
                serial_numbers=["239222300781"],
                calibration_reference_serials=["239222300433", "239222300781", "239222303506"],
            )
            self.assertEqual(len(selected), 1)
            self.assertAlmostEqual(float(selected[0][0, 3]), 1.0)

    def test_raises_on_ambiguous_subset_without_mapping(self) -> None:
        transforms = [np.eye(4, dtype=np.float32) for _ in range(3)]
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "calibrate.pkl"
            with path.open("wb") as handle:
                pickle.dump(transforms, handle)
            with self.assertRaises(CalibrationLoadError):
                load_calibration_transforms(path, serial_numbers=["239222300781"])


if __name__ == "__main__":
    unittest.main()
