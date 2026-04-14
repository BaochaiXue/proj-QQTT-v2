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
)

_KNOWN_ALIGNED_METADATA_KEYS = set(LEGACY_ALIGNED_METADATA_KEYS) | set(ALIGNED_METADATA_EXT_KEYS)


def split_aligned_metadata(metadata: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    missing_legacy = [key for key in LEGACY_ALIGNED_METADATA_KEYS if key not in metadata]
    if missing_legacy:
        raise ValueError(f"Aligned metadata missing legacy keys: {missing_legacy}")

    unexpected = sorted(set(metadata) - _KNOWN_ALIGNED_METADATA_KEYS)
    if unexpected:
        raise ValueError(f"Aligned metadata contains unexpected keys: {unexpected}")

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
    return legacy_or_merged, ext_metadata, merged_metadata


def write_split_aligned_metadata(case_dir: Path, metadata: dict[str, Any]) -> tuple[Path, Path]:
    legacy_metadata, ext_metadata = split_aligned_metadata(metadata)
    metadata_path = case_dir / ALIGNED_METADATA_FILENAME
    metadata_ext_path = case_dir / ALIGNED_METADATA_EXT_FILENAME
    metadata_path.write_text(json.dumps(legacy_metadata), encoding="utf-8")
    metadata_ext_path.write_text(json.dumps(ext_metadata), encoding="utf-8")
    return metadata_path, metadata_ext_path
