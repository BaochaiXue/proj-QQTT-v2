from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ALIGNED_METADATA_FILENAME = "metadata.json"
ALIGNED_METADATA_EXT_FILENAME = "metadata_ext.json"

LEGACY_ALIGNED_METADATA_KEYS = (
    "intrinsics",
    "serial_numbers",
    "fps",
    "WH",
    "frame_num",
    "start_step",
    "end_step",
)

ALIGNED_METADATA_EXT_KEYS = (
    "schema_version",
    "source_case_name",
    "calibration_reference_serials",
    "logical_camera_names",
    "capture_mode",
    "streams_present",
    "depth_backend_used",
    "depth_source_for_depth_dir",
    "ffs_native_like_postprocess_enabled",
    "depth_scale_m_per_unit",
    "depth_encoding",
    "K_color",
    "K_ir_left",
    "K_ir_right",
    "T_ir_left_to_right",
    "T_ir_left_to_color",
    "ir_baseline_m",
    "source_streams_present",
    "ffs_config",
    "ffs_confidence_filter",
    "ffs_radius_outlier_filter_enabled",
    "ffs_radius_outlier_filter",
)

_KNOWN_ALIGNED_METADATA_KEYS = set(LEGACY_ALIGNED_METADATA_KEYS) | set(ALIGNED_METADATA_EXT_KEYS)

_PER_CAMERA_LIST_KEYS = (
    "intrinsics",
    "logical_camera_names",
    "K_color",
    "K_ir_left",
    "K_ir_right",
    "T_ir_left_to_right",
    "T_ir_left_to_color",
    "ir_baseline_m",
)


def _validate_unique_string_list(metadata: dict[str, Any], key: str, *, required: bool) -> list[str] | None:
    values = metadata.get(key)
    if values is None:
        if required:
            raise ValueError(f"Aligned metadata missing {key!r}.")
        return None
    if not isinstance(values, list) or not values:
        raise ValueError(f"Aligned metadata {key!r} must be a non-empty list.")
    if not all(isinstance(item, str) and item for item in values):
        raise ValueError(f"Aligned metadata {key!r} must contain non-empty string serials.")

    duplicates = sorted({item for item in values if values.count(item) > 1})
    if duplicates:
        raise ValueError(f"Aligned metadata {key!r} contains duplicate serials: {duplicates}")
    return values


def _validate_per_camera_list_length(
    metadata: dict[str, Any],
    key: str,
    camera_count: int,
    *,
    allow_scalar: bool = False,
) -> None:
    if key not in metadata:
        return
    value = metadata[key]
    if allow_scalar and not isinstance(value, list):
        return
    if not isinstance(value, list):
        raise ValueError(f"Aligned metadata {key!r} must be a per-camera list.")
    if len(value) != camera_count:
        raise ValueError(
            f"Aligned metadata {key!r} length {len(value)} does not match "
            f"serial_numbers length {camera_count}."
        )


def validate_aligned_metadata_camera_contract(metadata: dict[str, Any]) -> None:
    """Validate the serial-index contract used by aligned-case consumers."""

    serial_numbers = _validate_unique_string_list(metadata, "serial_numbers", required=True)
    assert serial_numbers is not None
    camera_count = len(serial_numbers)

    calibration_reference_serials = _validate_unique_string_list(
        metadata,
        "calibration_reference_serials",
        required=False,
    )
    if calibration_reference_serials is not None:
        missing = [serial for serial in serial_numbers if serial not in calibration_reference_serials]
        if missing:
            raise ValueError(
                "Aligned metadata calibration_reference_serials does not cover case serial_numbers: "
                f"{missing}"
            )

    for key in _PER_CAMERA_LIST_KEYS:
        _validate_per_camera_list_length(metadata, key, camera_count)
    _validate_per_camera_list_length(
        metadata,
        "depth_scale_m_per_unit",
        camera_count,
        allow_scalar=True,
    )


def split_aligned_metadata(metadata: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    missing_legacy = [key for key in LEGACY_ALIGNED_METADATA_KEYS if key not in metadata]
    if missing_legacy:
        raise ValueError(f"Aligned metadata missing legacy keys: {missing_legacy}")

    unexpected = sorted(set(metadata) - _KNOWN_ALIGNED_METADATA_KEYS)
    if unexpected:
        raise ValueError(f"Aligned metadata contains unexpected keys: {unexpected}")

    validate_aligned_metadata_camera_contract(metadata)

    legacy_metadata = {
        key: metadata[key]
        for key in LEGACY_ALIGNED_METADATA_KEYS
    }
    ext_metadata = {
        key: metadata[key]
        for key in ALIGNED_METADATA_EXT_KEYS
        if key in metadata
    }
    return legacy_metadata, ext_metadata


def load_aligned_metadata(case_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    metadata_path = case_dir / ALIGNED_METADATA_FILENAME
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing aligned metadata: {metadata_path}")
    legacy_or_merged = json.loads(metadata_path.read_text(encoding="utf-8"))

    ext_path = case_dir / ALIGNED_METADATA_EXT_FILENAME
    ext_metadata = json.loads(ext_path.read_text(encoding="utf-8")) if ext_path.exists() else {}

    merged_metadata = dict(legacy_or_merged)
    for key, value in ext_metadata.items():
        if key in LEGACY_ALIGNED_METADATA_KEYS:
            continue
        merged_metadata[key] = value
    validate_aligned_metadata_camera_contract(merged_metadata)
    return legacy_or_merged, ext_metadata, merged_metadata


def write_split_aligned_metadata(case_dir: Path, metadata: dict[str, Any]) -> tuple[Path, Path]:
    legacy_metadata, ext_metadata = split_aligned_metadata(metadata)
    metadata_path = case_dir / ALIGNED_METADATA_FILENAME
    metadata_ext_path = case_dir / ALIGNED_METADATA_EXT_FILENAME
    metadata_path.write_text(json.dumps(legacy_metadata), encoding="utf-8")
    metadata_ext_path.write_text(json.dumps(ext_metadata), encoding="utf-8")
    return metadata_path, metadata_ext_path
