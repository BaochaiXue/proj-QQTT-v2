from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


class CalibrationLoadError(RuntimeError):
    pass


def describe_supported_calibration_schema() -> str:
    return (
        "Supported calibrate.pkl schema: a Python list/tuple or numpy array of shape (N, 4, 4), "
        "where each 4x4 transform is camera-to-world (c2w). This matches the current producer in "
        "qqtt.env.camera.camera_system.CameraSystem.calibrate(), which appends c2w matrices in the "
        "camera order used during calibration."
    )


def _coerce_transform_list(obj) -> list[np.ndarray]:
    if isinstance(obj, np.ndarray):
        if obj.ndim == 3 and obj.shape[1:] == (4, 4):
            return [np.asarray(item, dtype=np.float32) for item in obj]
        raise CalibrationLoadError(f"Unsupported calibration ndarray shape: {obj.shape}")
    if isinstance(obj, (list, tuple)):
        transforms = [np.asarray(item, dtype=np.float32) for item in obj]
        if not transforms:
            raise CalibrationLoadError("Calibration transform list is empty.")
        for item in transforms:
            if item.shape != (4, 4):
                raise CalibrationLoadError(f"Unsupported calibration transform shape: {item.shape}")
        return transforms
    raise CalibrationLoadError(
        f"Unsupported calibrate.pkl object type: {type(obj).__name__}. {describe_supported_calibration_schema()}"
    )


def load_calibration_transforms(
    calibrate_path: str | Path,
    *,
    serial_numbers: list[str] | None = None,
    calibration_reference_serials: list[str] | None = None,
) -> list[np.ndarray]:
    calibrate_path = Path(calibrate_path).resolve()
    if not calibrate_path.exists():
        raise CalibrationLoadError(f"Missing calibrate.pkl: {calibrate_path}")

    with calibrate_path.open("rb") as handle:
        raw = pickle.load(handle)
    transforms = _coerce_transform_list(raw)

    if serial_numbers is None:
        return transforms

    if len(transforms) == len(serial_numbers):
        return transforms

    if calibration_reference_serials is not None:
        index_by_serial = {serial: idx for idx, serial in enumerate(calibration_reference_serials)}
        missing = [serial for serial in serial_numbers if serial not in index_by_serial]
        if missing:
            raise CalibrationLoadError(
                "Calibration serial mapping is incomplete. "
                f"Missing serials: {missing}. Supported schema: {describe_supported_calibration_schema()}"
            )
        selected = []
        for serial in serial_numbers:
            idx = index_by_serial[serial]
            if idx >= len(transforms):
                raise CalibrationLoadError(
                    "Calibration transform count does not match calibration_reference_serials. "
                    f"idx={idx}, transform_count={len(transforms)}"
                )
            selected.append(transforms[idx])
        return selected

    raise CalibrationLoadError(
        "Cannot map calibrate.pkl transforms to this case because the file does not embed serials and "
        "metadata lacks calibration_reference_serials. "
        + describe_supported_calibration_schema()
    )
