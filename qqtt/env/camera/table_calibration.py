from __future__ import annotations

from datetime import datetime, timezone
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np


TABLE_CALIBRATION_METADATA_SCHEMA_VERSION = "qqtt_table_calibration_v1"
TABLE_CALIBRATE_COMPATIBILITY_CONTRACT = "qqtt_table_calibrate_c2w_v1"
TABLE_WORLD_FRAME_KIND = "table_world_z0"


class TableCalibrationLoadError(RuntimeError):
    pass


def table_calibration_metadata_path_for(table_calibrate_path: str | Path) -> Path:
    path = Path(table_calibrate_path)
    return path.with_name(f"{path.stem}_metadata.json")


def _validate_serials(name: str, serials: Any) -> list[str]:
    if not isinstance(serials, list) or not serials:
        raise TableCalibrationLoadError(f"{name} must be a non-empty serial list.")
    if not all(isinstance(item, str) and item for item in serials):
        raise TableCalibrationLoadError(f"{name} must contain non-empty string serials.")
    duplicates = sorted({item for item in serials if serials.count(item) > 1})
    if duplicates:
        raise TableCalibrationLoadError(f"{name} contains duplicate serials: {duplicates}")
    return list(serials)


def _validate_per_camera_length(name: str, values: list[Any], expected_count: int) -> None:
    if len(values) != expected_count:
        raise ValueError(
            f"{name} length must match transform_count. "
            f"{name}={len(values)}, transform_count={expected_count}"
        )


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def build_table_calibration_metadata(
    *,
    serial_numbers: list[str],
    WH,
    fps: int,
    transform_count: int,
    calibration_board: dict[str, Any],
    max_reprojection_error_px: float,
    min_corner_fraction: float,
    min_charuco_corners: int,
    per_camera_reprojection_error: list[float],
    per_camera_corner_count: list[int],
    per_camera_corner_fraction: list[float],
    distortion_used: bool | None = None,
    distortion_model_by_camera: list[str | None] | None = None,
    distortion_coeffs_by_camera: list[list[float] | None] | None = None,
    diagnostic_image_path: str | None = None,
) -> dict[str, Any]:
    serials = _validate_serials("serial_numbers", list(serial_numbers))
    expected_count = int(transform_count)
    if expected_count != len(serials):
        raise ValueError(
            "transform_count must match serial_numbers length. "
            f"transform_count={expected_count}, serial_numbers={len(serials)}"
        )
    _validate_per_camera_length(
        "per_camera_reprojection_error",
        per_camera_reprojection_error,
        expected_count,
    )
    _validate_per_camera_length(
        "per_camera_corner_count",
        per_camera_corner_count,
        expected_count,
    )
    _validate_per_camera_length(
        "per_camera_corner_fraction",
        per_camera_corner_fraction,
        expected_count,
    )

    metadata: dict[str, Any] = {
        "schema_version": TABLE_CALIBRATION_METADATA_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "serial_numbers": serials,
        "table_calibration_reference_serials": serials,
        "logical_camera_names": [f"cam{i}" for i in range(len(serials))],
        "WH": list(WH),
        "fps": int(fps),
        "transform_count": expected_count,
        "transform_convention": "camera_to_world_c2w",
        "world_frame_kind": TABLE_WORLD_FRAME_KIND,
        "compatibility_contract": TABLE_CALIBRATE_COMPATIBILITY_CONTRACT,
        "calibration_board": dict(calibration_board),
        "max_reprojection_error_px": float(max_reprojection_error_px),
        "min_corner_fraction": float(min_corner_fraction),
        "min_charuco_corners": int(min_charuco_corners),
        "per_camera_reprojection_error": [
            float(item) for item in per_camera_reprojection_error
        ],
        "per_camera_corner_count": [int(item) for item in per_camera_corner_count],
        "per_camera_corner_fraction": [
            float(item) for item in per_camera_corner_fraction
        ],
    }
    if distortion_used is not None:
        metadata["distortion_used"] = bool(distortion_used)
    if distortion_model_by_camera is not None:
        _validate_per_camera_length(
            "distortion_model_by_camera",
            distortion_model_by_camera,
            expected_count,
        )
        metadata["distortion_model_by_camera"] = list(distortion_model_by_camera)
    if distortion_coeffs_by_camera is not None:
        _validate_per_camera_length(
            "distortion_coeffs_by_camera",
            distortion_coeffs_by_camera,
            expected_count,
        )
        metadata["distortion_coeffs_by_camera"] = [
            None if item is None else [float(value) for value in item]
            for item in distortion_coeffs_by_camera
        ]
    if diagnostic_image_path is not None:
        metadata["diagnostic_image_path"] = str(diagnostic_image_path)
    return metadata


def _validate_transform_matrix(matrix: Any, *, index: int) -> np.ndarray:
    try:
        item = np.asarray(matrix, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise TableCalibrationLoadError(
            f"Table calibration transform at index {index} is not numeric."
        ) from exc
    if item.shape != (4, 4):
        raise TableCalibrationLoadError(
            f"Unsupported table calibration transform shape at index {index}: {item.shape}"
        )
    if not np.all(np.isfinite(item)):
        raise TableCalibrationLoadError(
            f"Table calibration transform at index {index} contains non-finite values."
        )
    expected_bottom = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    if not np.allclose(item[3], expected_bottom, atol=1e-4):
        raise TableCalibrationLoadError(
            f"Table calibration transform at index {index} has invalid homogeneous bottom row."
        )
    if abs(float(np.linalg.det(item[:3, :3]))) <= 1e-6:
        raise TableCalibrationLoadError(
            f"Table calibration transform at index {index} is singular or degenerate."
        )
    return item


def _coerce_transform_list(raw: Any) -> list[np.ndarray]:
    if isinstance(raw, np.ndarray):
        if raw.ndim == 3 and raw.shape[1:] == (4, 4):
            return [
                _validate_transform_matrix(item, index=idx)
                for idx, item in enumerate(raw)
            ]
        raise TableCalibrationLoadError(
            f"Unsupported table calibration ndarray shape: {raw.shape}"
        )
    if isinstance(raw, (list, tuple)):
        transforms = [
            _validate_transform_matrix(item, index=idx) for idx, item in enumerate(raw)
        ]
        if not transforms:
            raise TableCalibrationLoadError("Table calibration transform list is empty.")
        return transforms
    raise TableCalibrationLoadError(
        f"Unsupported table calibration object type: {type(raw).__name__}"
    )


def _require_metadata_field(metadata: dict[str, Any], name: str) -> Any:
    if name not in metadata:
        raise TableCalibrationLoadError(
            f"table calibration metadata missing required field: {name}"
        )
    return metadata[name]


def _validate_metadata_float(
    metadata: dict[str, Any],
    name: str,
    *,
    greater_than: float | None = None,
    greater_equal: float | None = None,
    less_equal: float | None = None,
) -> float:
    raw = _require_metadata_field(metadata, name)
    if not _is_finite_number(raw):
        raise TableCalibrationLoadError(f"{name} must be finite.")
    value = float(raw)
    if greater_than is not None and value <= greater_than:
        raise TableCalibrationLoadError(f"{name} must be > {greater_than}.")
    if greater_equal is not None and value < greater_equal:
        raise TableCalibrationLoadError(f"{name} must be >= {greater_equal}.")
    if less_equal is not None and value > less_equal:
        raise TableCalibrationLoadError(f"{name} must be <= {less_equal}.")
    return value


def _validate_metadata_int(
    metadata: dict[str, Any],
    name: str,
    *,
    greater_equal: int | None = None,
) -> int:
    raw = _require_metadata_field(metadata, name)
    if not _is_finite_number(raw):
        raise TableCalibrationLoadError(f"{name} must be finite.")
    value = float(raw)
    if not value.is_integer():
        raise TableCalibrationLoadError(f"{name} must be an integer.")
    result = int(value)
    if greater_equal is not None and result < greater_equal:
        raise TableCalibrationLoadError(f"{name} must be >= {greater_equal}.")
    return result


def _validate_metadata_string_list(
    metadata: dict[str, Any],
    name: str,
    expected_count: int,
) -> list[str]:
    raw = _require_metadata_field(metadata, name)
    if not isinstance(raw, list):
        raise TableCalibrationLoadError(f"{name} must be a list.")
    if len(raw) != expected_count:
        raise TableCalibrationLoadError(
            f"{name} length must match transform_count. "
            f"{name}={len(raw)}, transform_count={expected_count}"
        )
    if not all(isinstance(item, str) and item for item in raw):
        raise TableCalibrationLoadError(f"{name} must contain non-empty strings.")
    return list(raw)


def _validate_metadata_float_list(
    metadata: dict[str, Any],
    name: str,
    expected_count: int,
    *,
    greater_equal: float | None = None,
    less_equal: float | None = None,
) -> list[float]:
    raw = _require_metadata_field(metadata, name)
    if not isinstance(raw, list):
        raise TableCalibrationLoadError(f"{name} must be a list.")
    if len(raw) != expected_count:
        raise TableCalibrationLoadError(
            f"{name} length must match transform_count. "
            f"{name}={len(raw)}, transform_count={expected_count}"
        )
    values = []
    for index, item in enumerate(raw):
        if not _is_finite_number(item):
            raise TableCalibrationLoadError(f"{name}[{index}] must be finite.")
        value = float(item)
        if greater_equal is not None and value < greater_equal:
            raise TableCalibrationLoadError(
                f"{name}[{index}] must be >= {greater_equal}."
            )
        if less_equal is not None and value > less_equal:
            raise TableCalibrationLoadError(
                f"{name}[{index}] must be <= {less_equal}."
            )
        values.append(value)
    return values


def _validate_metadata_int_list(
    metadata: dict[str, Any],
    name: str,
    expected_count: int,
    *,
    greater_equal: int | None = None,
) -> list[int]:
    raw = _require_metadata_field(metadata, name)
    if not isinstance(raw, list):
        raise TableCalibrationLoadError(f"{name} must be a list.")
    if len(raw) != expected_count:
        raise TableCalibrationLoadError(
            f"{name} length must match transform_count. "
            f"{name}={len(raw)}, transform_count={expected_count}"
        )
    values = []
    for index, item in enumerate(raw):
        if not _is_finite_number(item):
            raise TableCalibrationLoadError(f"{name}[{index}] must be finite.")
        value = float(item)
        if not value.is_integer():
            raise TableCalibrationLoadError(f"{name}[{index}] must be an integer.")
        result = int(value)
        if greater_equal is not None and result < greater_equal:
            raise TableCalibrationLoadError(
                f"{name}[{index}] must be >= {greater_equal}."
            )
        values.append(result)
    return values


def _reject_nonfinite_metadata_numbers(value: Any, path: str) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, float) and not np.isfinite(value):
        raise TableCalibrationLoadError(f"{path} must be finite.")
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_nonfinite_metadata_numbers(item, f"{path}.{key}")
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_nonfinite_metadata_numbers(item, f"{path}[{index}]")


def _validate_table_metadata_object(
    metadata: Any,
    *,
    sidecar_path: Path | None,
) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        raise TableCalibrationLoadError("Table calibration metadata must be a JSON object.")
    if metadata.get("schema_version") != TABLE_CALIBRATION_METADATA_SCHEMA_VERSION:
        location = "" if sidecar_path is None else f" in {sidecar_path}"
        raise TableCalibrationLoadError(
            "Unsupported table calibration metadata schema"
            f"{location}: {metadata.get('schema_version')!r}"
        )
    if metadata.get("compatibility_contract") != TABLE_CALIBRATE_COMPATIBILITY_CONTRACT:
        raise TableCalibrationLoadError(
            "Unsupported table calibration compatibility contract: "
            f"{metadata.get('compatibility_contract')!r}"
        )
    if metadata.get("world_frame_kind") != TABLE_WORLD_FRAME_KIND:
        raise TableCalibrationLoadError(
            f"Unsupported table world frame kind: {metadata.get('world_frame_kind')!r}"
        )
    if metadata.get("transform_convention") != "camera_to_world_c2w":
        raise TableCalibrationLoadError(
            "Unsupported table calibration transform_convention: "
            f"{metadata.get('transform_convention')!r}"
        )

    serials = _validate_serials(
        "table calibration serial_numbers",
        metadata.get("serial_numbers"),
    )
    reference_serials = _validate_serials(
        "table calibration reference serials",
        metadata.get("table_calibration_reference_serials", serials),
    )
    if len(serials) != len(reference_serials):
        raise TableCalibrationLoadError(
            "table calibration serial_numbers and reference serials length mismatch"
        )
    try:
        transform_count = int(metadata.get("transform_count", len(reference_serials)))
    except (TypeError, ValueError) as exc:
        raise TableCalibrationLoadError(
            "table calibration transform_count must be an int"
        ) from exc
    if transform_count != len(reference_serials):
        raise TableCalibrationLoadError(
            "table calibration transform_count does not match reference serials"
        )
    _validate_metadata_string_list(metadata, "logical_camera_names", transform_count)
    calibration_board = _require_metadata_field(metadata, "calibration_board")
    if not isinstance(calibration_board, dict) or not calibration_board:
        raise TableCalibrationLoadError("calibration_board must be a non-empty object.")
    _validate_metadata_float(
        metadata,
        "max_reprojection_error_px",
        greater_than=0.0,
    )
    _validate_metadata_float(
        metadata,
        "min_corner_fraction",
        greater_than=0.0,
        less_equal=1.0,
    )
    _validate_metadata_int(metadata, "min_charuco_corners", greater_equal=0)
    _validate_metadata_float_list(
        metadata,
        "per_camera_reprojection_error",
        transform_count,
        greater_equal=0.0,
    )
    _validate_metadata_int_list(
        metadata,
        "per_camera_corner_count",
        transform_count,
        greater_equal=0,
    )
    _validate_metadata_float_list(
        metadata,
        "per_camera_corner_fraction",
        transform_count,
        greater_equal=0.0,
        less_equal=1.0,
    )
    _reject_nonfinite_metadata_numbers(metadata, "metadata")
    return metadata


def write_table_calibration_files(
    table_calibrate_path: str | Path,
    c2w_list: list[np.ndarray],
    metadata: dict[str, Any],
) -> Path:
    output_path = Path(table_calibrate_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    transforms = [
        _validate_transform_matrix(item, index=idx) for idx, item in enumerate(c2w_list)
    ]
    if not transforms:
        raise ValueError("table calibration transforms must not be empty")
    if int(metadata.get("transform_count", -1)) != len(transforms):
        raise ValueError(
            "table calibration metadata transform_count does not match transforms"
        )
    encoded_metadata = json.dumps(metadata, allow_nan=False)
    _validate_table_metadata_object(metadata, sidecar_path=None)

    with output_path.open("wb") as handle:
        pickle.dump(transforms, handle)
    sidecar_path = table_calibration_metadata_path_for(output_path)
    sidecar_path.write_text(encoded_metadata, encoding="utf-8")
    return sidecar_path


def load_table_calibration_metadata(table_calibrate_path: str | Path) -> dict[str, Any]:
    sidecar_path = table_calibration_metadata_path_for(table_calibrate_path)
    if not sidecar_path.is_file():
        raise TableCalibrationLoadError(
            f"Missing table calibration metadata: {sidecar_path}"
        )
    try:
        metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TableCalibrationLoadError(
            f"Invalid table calibration metadata JSON: {sidecar_path}"
        ) from exc
    return _validate_table_metadata_object(metadata, sidecar_path=sidecar_path)


def load_table_calibration_transforms(
    table_calibrate_path: str | Path,
    *,
    serial_numbers: list[str] | None = None,
    table_calibration_reference_serials: list[str] | None = None,
) -> list[np.ndarray]:
    path = Path(table_calibrate_path)
    if not path.is_file():
        raise TableCalibrationLoadError(f"Missing table calibration file: {path}")
    metadata = load_table_calibration_metadata(path)
    if table_calibration_reference_serials is None:
        reference_serials = list(metadata["table_calibration_reference_serials"])
    else:
        reference_serials = _validate_serials(
            "table_calibration_reference_serials",
            table_calibration_reference_serials,
        )

    try:
        with path.open("rb") as handle:
            raw = pickle.load(handle)
    except (
        pickle.PickleError,
        EOFError,
        AttributeError,
        ImportError,
        IndexError,
        TypeError,
        ValueError,
    ) as exc:
        raise TableCalibrationLoadError(
            f"Invalid table calibration pickle: {path}"
        ) from exc
    transforms = _coerce_transform_list(raw)
    if len(transforms) != int(metadata["transform_count"]):
        raise TableCalibrationLoadError(
            "table calibration transform count does not match metadata transform_count"
        )
    if len(transforms) != len(reference_serials):
        raise TableCalibrationLoadError(
            "table calibration transform count does not match reference serials"
        )
    if serial_numbers is None:
        return transforms

    requested_serials = _validate_serials("serial_numbers", serial_numbers)
    index_by_serial = {serial: idx for idx, serial in enumerate(reference_serials)}
    missing = [serial for serial in requested_serials if serial not in index_by_serial]
    if missing:
        raise TableCalibrationLoadError(
            f"Table calibration does not cover serials: {missing}"
        )
    return [transforms[index_by_serial[serial]] for serial in requested_serials]


def validate_table_calibration_acceptance(
    *,
    board_config,
    corner_count: int,
    reprojection_error_px: float,
    max_reprojection_error_px: float,
    min_corner_fraction: float,
) -> dict[str, float | int]:
    # For calibio-12x9-30mm, chessboard_corner_count is 88, so 0.60 yields 53.
    if not _is_finite_number(corner_count) or float(corner_count) < 0.0:
        raise ValueError("corner_count must be finite and >= 0.")
    corner_count_value = float(corner_count)
    if not _is_finite_number(reprojection_error_px) or float(reprojection_error_px) < 0.0:
        raise ValueError("reprojection_error_px must be finite and >= 0.")
    if (
        not _is_finite_number(max_reprojection_error_px)
        or float(max_reprojection_error_px) <= 0.0
    ):
        raise ValueError("max_reprojection_error_px must be finite and > 0.")
    if (
        not _is_finite_number(min_corner_fraction)
        or float(min_corner_fraction) <= 0.0
        or float(min_corner_fraction) > 1.0
    ):
        raise ValueError("min_corner_fraction must be finite and in (0, 1].")

    board_corners = int(board_config.chessboard_corner_count)
    if board_corners <= 0:
        raise ValueError("ChArUco board must expose at least one chessboard corner.")
    min_charuco_corners = int(
        np.ceil(float(min_corner_fraction) * float(board_corners))
    )
    corner_fraction = corner_count_value / float(board_corners)
    if corner_count_value < min_charuco_corners:
        raise ValueError(
            f"ChArUco corner count {corner_count_value:g} is below strict table calibration "
            f"minimum {min_charuco_corners}."
        )
    if float(reprojection_error_px) > float(max_reprojection_error_px):
        raise ValueError(
            f"ChArUco reprojection error {float(reprojection_error_px):.6f}px exceeds "
            f"strict table calibration maximum {float(max_reprojection_error_px):.6f}px."
        )
    return {
        "min_charuco_corners": min_charuco_corners,
        "corner_fraction": corner_fraction,
    }
