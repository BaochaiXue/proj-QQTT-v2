from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from .calibration_frame import CALIBRATION_WORLD_FRAME_KIND


class CalibrationLoadError(RuntimeError):
    pass


def describe_supported_calibration_schema() -> str:
    return (
        "Supported calibrate.pkl schema: a Python list/tuple or numpy array of shape (N, 4, 4), "
        "where each 4x4 transform is camera-to-world (c2w). This matches the current producer in "
        "qqtt.env.camera.camera_system.CameraSystem.calibrate(), which appends c2w matrices in the "
        "camera order used during calibration. The 'world' frame here is the raw ChArUco/board "
        f"calibration frame, exposed in visualization metadata as '{CALIBRATION_WORLD_FRAME_KIND}'."
    )


def _validate_serial_list(name: str, serials: list[str] | None) -> None:
    if serials is None:
        return
    if len(serials) == 0:
        raise CalibrationLoadError(f"{name} must not be empty when provided.")
    if len(serials) != len(set(serials)):
        raise CalibrationLoadError(f"{name} contains duplicate serials: {serials}")


def _validate_transform_matrix(matrix: np.ndarray, *, index: int) -> np.ndarray:
    item = np.asarray(matrix, dtype=np.float32)
    if item.shape != (4, 4):
        raise CalibrationLoadError(f"Unsupported calibration transform shape at index {index}: {item.shape}")
    if not np.all(np.isfinite(item)):
        raise CalibrationLoadError(f"Calibration transform at index {index} contains non-finite values.")
    expected_bottom = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    if not np.allclose(item[3], expected_bottom, atol=1e-4):
        raise CalibrationLoadError(
            f"Calibration transform at index {index} has invalid homogeneous bottom row {item[3].tolist()}."
        )
    rotation = item[:3, :3]
    if abs(float(np.linalg.det(rotation))) <= 1e-6:
        raise CalibrationLoadError(f"Calibration transform at index {index} is singular or degenerate.")
    return item


def _coerce_transform_list(obj) -> list[np.ndarray]:
    if isinstance(obj, np.ndarray):
        if obj.ndim == 3 and obj.shape[1:] == (4, 4):
            return [_validate_transform_matrix(item, index=idx) for idx, item in enumerate(obj)]
        raise CalibrationLoadError(f"Unsupported calibration ndarray shape: {obj.shape}")
    if isinstance(obj, (list, tuple)):
        transforms = [_validate_transform_matrix(item, index=idx) for idx, item in enumerate(obj)]
        if not transforms:
            raise CalibrationLoadError("Calibration transform list is empty.")
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

    _validate_serial_list("serial_numbers", serial_numbers)
    _validate_serial_list("calibration_reference_serials", calibration_reference_serials)

    with calibrate_path.open("rb") as handle:
        raw = pickle.load(handle)
    transforms = _coerce_transform_list(raw)

    if serial_numbers is None:
        return transforms

    if calibration_reference_serials is not None:
        if len(calibration_reference_serials) != len(transforms):
            raise CalibrationLoadError(
                "Calibration transform count does not match calibration_reference_serials length. "
                f"transform_count={len(transforms)}, calibration_reference_serials={len(calibration_reference_serials)}"
            )
        if list(serial_numbers) == list(calibration_reference_serials):
            return transforms
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

    if len(transforms) == len(serial_numbers):
        return transforms

    raise CalibrationLoadError(
        "Cannot map calibrate.pkl transforms to this case because the file does not embed serials and "
        "metadata lacks calibration_reference_serials. "
        + describe_supported_calibration_schema()
    )


def build_calibration_contract_summary(
    *,
    calibrate_path: str | Path,
    transform_count: int,
    serial_numbers: list[str] | None = None,
    calibration_reference_serials: list[str] | None = None,
    mapping_mode: str,
) -> dict[str, object]:
    return {
        "calibrate_path": str(Path(calibrate_path).resolve()),
        "schema_version": "qqtt_calibrate_c2w_v1",
        "world_frame_kind": CALIBRATION_WORLD_FRAME_KIND,
        "transform_convention": "camera_to_world_c2w",
        "transform_count": int(transform_count),
        "serial_numbers": None if serial_numbers is None else list(serial_numbers),
        "calibration_reference_serials": None if calibration_reference_serials is None else list(calibration_reference_serials),
        "mapping_mode": mapping_mode,
    }


def infer_calibration_mapping_mode(
    *,
    serial_numbers: list[str] | None,
    calibration_reference_serials: list[str] | None,
) -> str:
    if serial_numbers is None:
        return "raw_transform_order"
    if calibration_reference_serials is None:
        return "direct_length_match"
    if len(serial_numbers) == len(calibration_reference_serials) and list(serial_numbers) == list(calibration_reference_serials):
        return "direct_length_match"
    if len(serial_numbers) == len(calibration_reference_serials):
        return "calibration_reference_serials_reordered"
    return "calibration_reference_serials_subset"
