from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np


TABLE_CALIBRATION_METADATA_SCHEMA_VERSION = "qqtt_table_calibration_v1"
TABLE_CALIBRATE_COMPATIBILITY_CONTRACT = "qqtt_table_calibrate_c2w_v1"
TABLE_WORLD_FRAME_KIND = "table_world_z0"
_TABLE_CALIBRATION_CORNER_FRACTION_TOLERANCE = 5e-3
_TABLE_CALIBRATION_METADATA_ALLOWED_KEYS = frozenset(
    {
        "calibration_board",
        "compatibility_contract",
        "created_at_utc",
        "diagnostic_image_path",
        "distortion_coeffs_by_camera",
        "distortion_model_by_camera",
        "distortion_used",
        "fps",
        "logical_camera_names",
        "max_reprojection_error_px",
        "min_charuco_corners",
        "min_corner_fraction",
        "per_camera_corner_count",
        "per_camera_corner_fraction",
        "per_camera_reprojection_error",
        "schema_version",
        "serial_numbers",
        "table_calibration_reference_serials",
        "transform_convention",
        "transform_count",
        "WH",
        "world_frame_kind",
    }
)


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
    if not isinstance(values, list):
        raise ValueError(f"{name} must be a list.")
    if len(values) != expected_count:
        raise ValueError(
            f"{name} length must match transform_count. "
            f"{name}={len(values)}, transform_count={expected_count}"
        )


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        return False
    return bool(np.isfinite(float(value)))


def _normalize_json_scalars(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _normalize_json_scalars(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_json_scalars(item) for item in value]
    return value


def _dist_coeffs_to_metadata(coeffs) -> list[float] | None:
    if coeffs is None:
        return None
    coeffs_array = np.asarray(coeffs, dtype=np.float64).reshape(-1)
    if coeffs_array.size == 0:
        return None
    return [float(value) for value in coeffs_array]


def _numeric_string_path(value: Any, path: str) -> str | None:
    if isinstance(value, np.ndarray):
        return _numeric_string_path(value.tolist(), path)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            float(text)
        except ValueError:
            return None
        return path
    if isinstance(value, dict):
        for key, item in value.items():
            found = _numeric_string_path(item, f"{path}.{key}")
            if found is not None:
                return found
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found = _numeric_string_path(item, f"{path}[{index}]")
            if found is not None:
                return found
    return None


def _validate_build_int(name: str, value: Any, *, greater_equal: int | None = None) -> int:
    try:
        return _validate_metadata_int({name: value}, name, greater_equal=greater_equal)
    except TableCalibrationLoadError as exc:
        raise ValueError(str(exc)) from exc


def _validate_build_float(name: str, value: Any) -> float:
    if not _is_finite_number(value):
        raise ValueError(f"{name} must be a finite JSON number.")
    return float(value)


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
    try:
        serials = _validate_serials("serial_numbers", serial_numbers)
    except TableCalibrationLoadError as exc:
        raise ValueError(str(exc)) from exc
    if not isinstance(calibration_board, dict):
        raise ValueError("calibration_board must be a dict.")
    expected_count = _validate_build_int(
        "transform_count",
        transform_count,
        greater_equal=1,
    )
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
        "serial_numbers": list(serials),
        "table_calibration_reference_serials": list(serials),
        "logical_camera_names": [f"cam{i}" for i in range(len(serials))],
        "WH": list(WH),
        "fps": fps,
        "transform_count": expected_count,
        "transform_convention": "camera_to_world_c2w",
        "world_frame_kind": TABLE_WORLD_FRAME_KIND,
        "compatibility_contract": TABLE_CALIBRATE_COMPATIBILITY_CONTRACT,
        "calibration_board": dict(calibration_board),
        "max_reprojection_error_px": max_reprojection_error_px,
        "min_corner_fraction": min_corner_fraction,
        "min_charuco_corners": min_charuco_corners,
        "per_camera_reprojection_error": list(per_camera_reprojection_error),
        "per_camera_corner_count": list(per_camera_corner_count),
        "per_camera_corner_fraction": list(per_camera_corner_fraction),
    }
    if distortion_used is not None:
        if not isinstance(distortion_used, bool):
            raise ValueError("distortion_used must be a bool.")
        metadata["distortion_used"] = distortion_used
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
        coeffs_by_camera = []
        for idx, item in enumerate(distortion_coeffs_by_camera):
            if item is None:
                coeffs_by_camera.append(None)
                continue
            if not isinstance(item, list):
                raise ValueError(
                    f"distortion_coeffs_by_camera[{idx}] must be a coefficient list or null."
                )
            coeffs_by_camera.append(
                [
                    _validate_build_float(
                        f"distortion_coeffs_by_camera[{idx}][{value_idx}]",
                        value,
                    )
                    for value_idx, value in enumerate(item)
                ]
            )
        metadata["distortion_coeffs_by_camera"] = coeffs_by_camera
    if diagnostic_image_path is not None:
        if not isinstance(diagnostic_image_path, str) or not diagnostic_image_path:
            raise ValueError("diagnostic_image_path must be a non-empty string.")
        metadata["diagnostic_image_path"] = diagnostic_image_path

    try:
        _validate_table_metadata_object(metadata, sidecar_path=None)
    except TableCalibrationLoadError as exc:
        raise ValueError(str(exc)) from exc

    metadata["WH"] = [int(value) for value in metadata["WH"]]
    metadata["fps"] = int(metadata["fps"])
    metadata["max_reprojection_error_px"] = float(metadata["max_reprojection_error_px"])
    metadata["min_corner_fraction"] = float(metadata["min_corner_fraction"])
    metadata["min_charuco_corners"] = int(metadata["min_charuco_corners"])
    metadata["per_camera_reprojection_error"] = [
        float(item) for item in metadata["per_camera_reprojection_error"]
    ]
    metadata["per_camera_corner_count"] = [
        int(item) for item in metadata["per_camera_corner_count"]
    ]
    metadata["per_camera_corner_fraction"] = [
        float(item) for item in metadata["per_camera_corner_fraction"]
    ]
    metadata["calibration_board"] = _normalize_json_scalars(
        metadata["calibration_board"]
    )
    return metadata


def _validate_transform_matrix(matrix: Any, *, index: int) -> np.ndarray:
    numeric_string_path = _numeric_string_path(matrix, f"transform[{index}]")
    if numeric_string_path is not None:
        raise TableCalibrationLoadError(
            "Table calibration transform at index "
            f"{index} contains numeric string at {numeric_string_path}."
        )
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
    rotation = item[:3, :3]
    determinant = float(np.linalg.det(rotation))
    if abs(determinant) <= 1e-6:
        raise TableCalibrationLoadError(
            f"Table calibration transform at index {index} is singular or degenerate."
        )
    if not np.allclose(
        rotation.T @ rotation,
        np.eye(3, dtype=np.float32),
        atol=1e-3,
        rtol=1e-3,
    ):
        raise TableCalibrationLoadError(
            f"Table calibration transform at index {index} rotation block is not orthonormal."
        )
    if not np.isclose(determinant, 1.0, atol=1e-3, rtol=1e-3):
        raise TableCalibrationLoadError(
            f"Table calibration transform at index {index} rotation determinant is not close to +1."
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


def _validate_metadata_nonempty_string(metadata: dict[str, Any], name: str) -> str:
    raw = _require_metadata_field(metadata, name)
    if not isinstance(raw, str) or not raw:
        raise TableCalibrationLoadError(f"{name} must be a non-empty string.")
    return raw


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


def _validate_metadata_positive_int_pair(
    metadata: dict[str, Any],
    name: str,
) -> list[int]:
    raw = _require_metadata_field(metadata, name)
    if not isinstance(raw, list) or len(raw) != 2:
        raise TableCalibrationLoadError(f"{name} must be a length-2 list.")
    values = []
    for index, item in enumerate(raw):
        if not _is_finite_number(item):
            raise TableCalibrationLoadError(f"{name}[{index}] must be finite.")
        value = float(item)
        if not value.is_integer():
            raise TableCalibrationLoadError(f"{name}[{index}] must be an integer.")
        result = int(value)
        if result <= 0:
            raise TableCalibrationLoadError(f"{name}[{index}] must be > 0.")
        values.append(result)
    return values


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


def _validate_optional_metadata_string_or_none_list(
    metadata: dict[str, Any],
    name: str,
    expected_count: int,
) -> list[str | None] | None:
    if name not in metadata:
        return None
    raw = metadata[name]
    if not isinstance(raw, list):
        raise TableCalibrationLoadError(f"{name} must be a list.")
    if len(raw) != expected_count:
        raise TableCalibrationLoadError(
            f"{name} length must match transform_count. "
            f"{name}={len(raw)}, transform_count={expected_count}"
        )
    values = []
    for index, item in enumerate(raw):
        if item is None:
            values.append(None)
            continue
        if not isinstance(item, str) or not item:
            raise TableCalibrationLoadError(
                f"{name}[{index}] must be a non-empty string or null."
            )
        values.append(item)
    return values


def _validate_optional_metadata_coeffs_by_camera(
    metadata: dict[str, Any],
    name: str,
    expected_count: int,
) -> list[list[float] | None] | None:
    if name not in metadata:
        return None
    raw = metadata[name]
    if not isinstance(raw, list):
        raise TableCalibrationLoadError(f"{name} must be a list.")
    if len(raw) != expected_count:
        raise TableCalibrationLoadError(
            f"{name} length must match transform_count. "
            f"{name}={len(raw)}, transform_count={expected_count}"
        )
    values = []
    for camera_index, item in enumerate(raw):
        if item is None:
            values.append(None)
            continue
        if not isinstance(item, list):
            raise TableCalibrationLoadError(
                f"{name}[{camera_index}] must be a coefficient list or null."
            )
        coeffs = []
        for coeff_index, coeff in enumerate(item):
            if not _is_finite_number(coeff):
                raise TableCalibrationLoadError(
                    f"{name}[{camera_index}][{coeff_index}] must be finite."
                )
            coeffs.append(float(coeff))
        values.append(coeffs)
    return values


def _validate_optional_metadata_bool(metadata: dict[str, Any], name: str) -> bool | None:
    if name not in metadata:
        return None
    raw = metadata[name]
    if not isinstance(raw, bool):
        raise TableCalibrationLoadError(f"{name} must be a bool.")
    return raw


def _validate_optional_metadata_nonempty_string(
    metadata: dict[str, Any],
    name: str,
) -> str | None:
    if name not in metadata:
        return None
    raw = metadata[name]
    if not isinstance(raw, str) or not raw:
        raise TableCalibrationLoadError(f"{name} must be a non-empty string.")
    return raw


def _validate_optional_board_corner_count(
    calibration_board: dict[str, Any],
) -> int | None:
    board_name = calibration_board.get("name")
    registered_corner_count: int | None = None
    if isinstance(board_name, str) and board_name:
        from qqtt.env.camera.calibration_boards import get_calibration_board_config

        try:
            board_config = get_calibration_board_config(board_name)
        except ValueError as exc:
            if "chessboard_corner_count" not in calibration_board:
                raise TableCalibrationLoadError(
                    f"Unknown calibration_board.name: {board_name!r}"
                ) from exc
        else:
            registered_corner_count = int(board_config.chessboard_corner_count)

    raw = calibration_board.get("chessboard_corner_count")
    if raw is None:
        if registered_corner_count is not None:
            return registered_corner_count
        raise TableCalibrationLoadError(
            "calibration_board must include a known non-empty name or explicit "
            "chessboard_corner_count."
        )
    if not _is_finite_number(raw):
        raise TableCalibrationLoadError(
            "calibration_board.chessboard_corner_count must be finite."
        )
    value = float(raw)
    if not value.is_integer():
        raise TableCalibrationLoadError(
            "calibration_board.chessboard_corner_count must be an integer."
        )
    result = int(value)
    if result <= 0:
        raise TableCalibrationLoadError(
            "calibration_board.chessboard_corner_count must be > 0."
        )
    if registered_corner_count is not None and result != registered_corner_count:
        raise TableCalibrationLoadError(
            "calibration_board.chessboard_corner_count must match registered "
            f"board profile {board_name!r}: expected {registered_corner_count}, "
            f"got {result}."
        )
    return result


def _validate_metadata_acceptance_fields(
    *,
    max_reprojection_error_px: float,
    min_corner_fraction: float,
    min_charuco_corners: int,
    per_camera_reprojection_error: list[float],
    per_camera_corner_count: list[int],
    per_camera_corner_fraction: list[float],
    chessboard_corner_count: int | None,
) -> None:
    for index, value in enumerate(per_camera_reprojection_error):
        if value > max_reprojection_error_px:
            raise TableCalibrationLoadError(
                "per_camera_reprojection_error"
                f"[{index}] must be <= max_reprojection_error_px."
            )
    for index, value in enumerate(per_camera_corner_count):
        if value < min_charuco_corners:
            raise TableCalibrationLoadError(
                f"per_camera_corner_count[{index}] must be >= min_charuco_corners."
            )
        if chessboard_corner_count is not None and value > chessboard_corner_count:
            raise TableCalibrationLoadError(
                "per_camera_corner_count"
                f"[{index}] must be <= calibration_board.chessboard_corner_count."
            )
    for index, value in enumerate(per_camera_corner_fraction):
        if value < min_corner_fraction:
            raise TableCalibrationLoadError(
                f"per_camera_corner_fraction[{index}] must be >= min_corner_fraction."
            )
    min_required_corners = 11
    if chessboard_corner_count is not None:
        min_required_corners = max(
            min_required_corners,
            int(math.ceil(min_corner_fraction * chessboard_corner_count)),
        )
    if min_charuco_corners < min_required_corners:
        raise TableCalibrationLoadError(
            "min_charuco_corners must be >= "
            "max(11, ceil(min_corner_fraction * chessboard_corner_count)). "
            f"min_charuco_corners={min_charuco_corners}, "
            f"required={min_required_corners}"
        )
    if (
        chessboard_corner_count is not None
        and min_charuco_corners > chessboard_corner_count
    ):
        raise TableCalibrationLoadError(
            "min_charuco_corners must be <= "
            "calibration_board.chessboard_corner_count."
        )
    if chessboard_corner_count is not None:
        for index, value in enumerate(per_camera_corner_fraction):
            expected_fraction = per_camera_corner_count[index] / chessboard_corner_count
            if (
                abs(value - expected_fraction)
                > _TABLE_CALIBRATION_CORNER_FRACTION_TOLERANCE
            ):
                raise TableCalibrationLoadError(
                    f"per_camera_corner_fraction[{index}] must match "
                    "per_camera_corner_count / "
                    "calibration_board.chessboard_corner_count."
                )


def _reject_nonfinite_metadata_numbers(value: Any, path: str) -> None:
    if isinstance(value, (bool, np.bool_)):
        return
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
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
    unknown_keys = sorted(set(metadata) - _TABLE_CALIBRATION_METADATA_ALLOWED_KEYS)
    if unknown_keys:
        raise TableCalibrationLoadError(
            f"table calibration metadata contains unknown fields: {unknown_keys}"
        )
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
    transform_convention = _require_metadata_field(metadata, "transform_convention")
    if transform_convention != "camera_to_world_c2w":
        raise TableCalibrationLoadError(
            "Unsupported table calibration transform_convention: "
            f"{transform_convention!r}"
        )

    _validate_metadata_nonempty_string(metadata, "created_at_utc")
    serials = _validate_serials(
        "table calibration serial_numbers",
        _require_metadata_field(metadata, "serial_numbers"),
    )
    reference_serials = _validate_serials(
        "table_calibration_reference_serials",
        _require_metadata_field(metadata, "table_calibration_reference_serials"),
    )
    if serials != reference_serials:
        raise TableCalibrationLoadError(
            "table calibration serial_numbers must exactly match "
            "table_calibration_reference_serials"
        )
    _validate_metadata_positive_int_pair(metadata, "WH")
    _validate_metadata_int(metadata, "fps", greater_equal=1)
    transform_count = _validate_metadata_int(
        metadata,
        "transform_count",
        greater_equal=1,
    )
    if transform_count != len(reference_serials):
        raise TableCalibrationLoadError(
            "table calibration transform_count does not match reference serials"
        )
    _validate_metadata_string_list(metadata, "logical_camera_names", transform_count)
    calibration_board = _require_metadata_field(metadata, "calibration_board")
    if not isinstance(calibration_board, dict) or not calibration_board:
        raise TableCalibrationLoadError("calibration_board must be a non-empty object.")
    numeric_string_path = _numeric_string_path(calibration_board, "calibration_board")
    if numeric_string_path is not None:
        raise TableCalibrationLoadError(
            f"{numeric_string_path} must be a JSON number, not a numeric string."
        )
    chessboard_corner_count = _validate_optional_board_corner_count(calibration_board)
    max_reprojection_error_px = _validate_metadata_float(
        metadata,
        "max_reprojection_error_px",
        greater_than=0.0,
    )
    min_corner_fraction = _validate_metadata_float(
        metadata,
        "min_corner_fraction",
        greater_than=0.0,
        less_equal=1.0,
    )
    min_charuco_corners = _validate_metadata_int(
        metadata,
        "min_charuco_corners",
        greater_equal=0,
    )
    per_camera_reprojection_error = _validate_metadata_float_list(
        metadata,
        "per_camera_reprojection_error",
        transform_count,
        greater_equal=0.0,
    )
    per_camera_corner_count = _validate_metadata_int_list(
        metadata,
        "per_camera_corner_count",
        transform_count,
        greater_equal=0,
    )
    per_camera_corner_fraction = _validate_metadata_float_list(
        metadata,
        "per_camera_corner_fraction",
        transform_count,
        greater_equal=0.0,
        less_equal=1.0,
    )
    _validate_metadata_acceptance_fields(
        max_reprojection_error_px=max_reprojection_error_px,
        min_corner_fraction=min_corner_fraction,
        min_charuco_corners=min_charuco_corners,
        per_camera_reprojection_error=per_camera_reprojection_error,
        per_camera_corner_count=per_camera_corner_count,
        per_camera_corner_fraction=per_camera_corner_fraction,
        chessboard_corner_count=chessboard_corner_count,
    )
    _validate_optional_metadata_bool(metadata, "distortion_used")
    _validate_optional_metadata_string_or_none_list(
        metadata,
        "distortion_model_by_camera",
        transform_count,
    )
    _validate_optional_metadata_coeffs_by_camera(
        metadata,
        "distortion_coeffs_by_camera",
        transform_count,
    )
    _validate_optional_metadata_nonempty_string(metadata, "diagnostic_image_path")
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
    try:
        _validate_table_metadata_object(metadata, sidecar_path=None)
    except TableCalibrationLoadError as exc:
        raise ValueError(str(exc)) from exc
    normalized_metadata = _normalize_json_scalars(metadata)
    if int(normalized_metadata["transform_count"]) != len(transforms):
        raise ValueError(
            "table calibration metadata transform_count does not match transforms"
        )
    try:
        encoded_metadata = json.dumps(normalized_metadata, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("table calibration metadata must be JSON serializable.") from exc

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
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
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
    reference_serials = list(metadata["table_calibration_reference_serials"])
    if table_calibration_reference_serials is not None:
        expected_reference_serials = _validate_serials(
            "table_calibration_reference_serials",
            table_calibration_reference_serials,
        )
        if expected_reference_serials != reference_serials:
            raise TableCalibrationLoadError(
                "table_calibration_reference_serials does not match metadata "
                "table_calibration_reference_serials."
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
    if not corner_count_value.is_integer():
        raise ValueError("corner_count must be an integer.")
    if corner_count_value > float(board_corners):
        raise ValueError("corner_count must be <= chessboard_corner_count.")
    min_charuco_corners = max(
        11,
        int(np.ceil(float(min_corner_fraction) * float(board_corners))),
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


def estimate_table_c2w_from_charuco_image(
    *,
    image_bgr: np.ndarray,
    board_config,
    camera_matrix: np.ndarray,
    dist_coeffs=None,
    max_reprojection_error_px: float,
    min_corner_fraction: float,
) -> tuple[np.ndarray, float, int, float, int, np.ndarray]:
    import cv2

    from qqtt.env.camera.calibration_boards import (
        create_charuco_board,
        detect_charuco_board,
        estimate_charuco_board_pose,
        get_charuco_chessboard_corners,
    )

    image = np.asarray(image_bgr)
    if image.ndim not in (2, 3):
        raise ValueError(f"image_bgr must be a grayscale or BGR image, got {image.shape}.")

    intrinsic = np.asarray(camera_matrix, dtype=np.float64)
    if intrinsic.shape != (3, 3) or not np.all(np.isfinite(intrinsic)):
        raise ValueError("camera_matrix must be a finite 3x3 matrix.")

    dist_coeffs_array = None
    if dist_coeffs is not None:
        dist_coeffs_array = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1)
        if dist_coeffs_array.size == 0:
            dist_coeffs_array = None

    if image.ndim == 2:
        diagnostic_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        diagnostic_bgr = image.copy()

    _dictionary, board = create_charuco_board(board_config)
    (
        charuco_corners,
        charuco_ids,
        marker_corners,
        marker_ids,
    ) = detect_charuco_board(
        image,
        board,
    )
    if marker_ids is None or len(marker_corners) == 0:
        raise ValueError("No ArUco markers detected for table calibration.")

    cv2.aruco.drawDetectedMarkers(diagnostic_bgr, marker_corners, marker_ids)
    if charuco_corners is None or charuco_ids is None or len(charuco_corners) == 0:
        raise ValueError("No ChArUco corners detected for table calibration.")

    pose_ok, rvec, tvec = estimate_charuco_board_pose(
        charuco_corners=charuco_corners,
        charuco_ids=charuco_ids,
        board=board,
        camera_matrix=intrinsic,
        dist_coeffs=dist_coeffs_array,
    )
    if (not pose_ok) or rvec is None or tvec is None:
        raise ValueError("Failed to estimate ChArUco pose for table calibration.")

    charuco_id_values = np.asarray(charuco_ids, dtype=np.int64).reshape(-1)
    chessboard_corners = np.asarray(
        get_charuco_chessboard_corners(board),
        dtype=np.float64,
    )
    object_points = chessboard_corners[charuco_id_values, :]
    reprojected_points, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        intrinsic,
        dist_coeffs_array,
    )
    reprojected_points = reprojected_points.reshape(-1, 2)
    observed_corners = np.asarray(charuco_corners, dtype=np.float64).reshape(-1, 2)
    reprojection_error_px = float(
        np.sqrt(np.sum((reprojected_points - observed_corners) ** 2, axis=1)).mean()
    )
    corner_count = int(observed_corners.shape[0])
    acceptance = validate_table_calibration_acceptance(
        board_config=board_config,
        corner_count=corner_count,
        reprojection_error_px=reprojection_error_px,
        max_reprojection_error_px=max_reprojection_error_px,
        min_corner_fraction=min_corner_fraction,
    )

    R_board_to_camera = cv2.Rodrigues(rvec)[0]
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = R_board_to_camera
    w2c[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    c2w = np.linalg.inv(w2c).astype(np.float32)

    cv2.aruco.drawDetectedCornersCharuco(
        image=diagnostic_bgr,
        charucoCorners=observed_corners.reshape(-1, 1, 2).astype(np.float32),
        charucoIds=charuco_id_values.reshape(-1, 1).astype(np.int32),
    )
    cv2.drawFrameAxes(
        diagnostic_bgr,
        intrinsic,
        dist_coeffs_array,
        rvec,
        tvec,
        0.1,
    )

    return (
        c2w,
        reprojection_error_px,
        corner_count,
        float(acceptance["corner_fraction"]),
        int(acceptance["min_charuco_corners"]),
        diagnostic_bgr,
    )
