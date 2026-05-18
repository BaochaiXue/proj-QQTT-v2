from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


CALIBRATION_METADATA_SCHEMA_VERSION = "qqtt_calibration_v1"


def calibration_metadata_path_for(calibrate_path: str | Path) -> Path:
    path = Path(calibrate_path)
    return path.with_name(f"{path.stem}_metadata.json")


def _validate_serials(name: str, serials: Any) -> list[str]:
    if not isinstance(serials, list) or not serials:
        raise ValueError(f"{name} must be a non-empty serial list.")
    if not all(isinstance(item, str) and item for item in serials):
        raise ValueError(f"{name} must contain non-empty string serials.")
    duplicates = sorted({item for item in serials if serials.count(item) > 1})
    if duplicates:
        raise ValueError(f"{name} contains duplicate serials: {duplicates}")
    return list(serials)


def build_calibration_metadata(
    *,
    serial_numbers: list[str],
    WH,
    fps: int,
    transform_count: int,
    per_camera_reprojection_error: list[float] | None = None,
    calibration_board: dict[str, Any] | None = None,
    world_frame_convention: str | None = None,
    distortion_used: bool | None = None,
    distortion_model_by_camera: list[str | None] | None = None,
    distortion_coeffs_by_camera: list[list[float] | None] | None = None,
    per_camera_corner_count: list[int] | None = None,
    per_camera_pose_stability: dict[str, Any] | None = None,
    calibration_samples_requested: int | None = None,
    calibration_samples_used: int | None = None,
    selected_sample_index: int | None = None,
    sample_mean_reprojection_error: list[float] | None = None,
    source_format: str | None = None,
    source_path: str | None = None,
) -> dict[str, Any]:
    serials = _validate_serials("serial_numbers", list(serial_numbers))
    if int(transform_count) != len(serials):
        raise ValueError(
            "Calibration transform count must match serial_numbers length. "
            f"transform_count={transform_count}, serial_numbers={len(serials)}"
        )
    metadata = {
        "schema_version": CALIBRATION_METADATA_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "serial_numbers": serials,
        "calibration_reference_serials": serials,
        "logical_camera_names": [f"cam{i}" for i in range(len(serials))],
        "WH": list(WH),
        "fps": int(fps),
        "transform_count": int(transform_count),
        "transform_convention": "camera_to_world_c2w",
        "compatibility_contract": "qqtt_calibrate_pkl_c2w_list_v1",
        "per_camera_reprojection_error": None
        if per_camera_reprojection_error is None
        else [float(item) for item in per_camera_reprojection_error],
    }
    if calibration_board is not None:
        metadata["calibration_board"] = dict(calibration_board)
    if world_frame_convention is not None:
        metadata["world_frame_convention"] = str(world_frame_convention)
    if distortion_used is not None:
        metadata["distortion_used"] = bool(distortion_used)
    if distortion_model_by_camera is not None:
        metadata["distortion_model_by_camera"] = list(distortion_model_by_camera)
    if distortion_coeffs_by_camera is not None:
        metadata["distortion_coeffs_by_camera"] = [
            None if item is None else [float(value) for value in item]
            for item in distortion_coeffs_by_camera
        ]
    if per_camera_corner_count is not None:
        metadata["per_camera_corner_count"] = [int(item) for item in per_camera_corner_count]
    if per_camera_pose_stability is not None:
        metadata["per_camera_pose_stability"] = dict(per_camera_pose_stability)
    if calibration_samples_requested is not None:
        metadata["calibration_samples_requested"] = int(calibration_samples_requested)
    if calibration_samples_used is not None:
        metadata["calibration_samples_used"] = int(calibration_samples_used)
    if selected_sample_index is not None:
        metadata["selected_sample_index"] = int(selected_sample_index)
    if sample_mean_reprojection_error is not None:
        metadata["sample_mean_reprojection_error"] = [
            float(item) for item in sample_mean_reprojection_error
        ]
    if source_format is not None:
        metadata["source_format"] = str(source_format)
    if source_path is not None:
        metadata["source_path"] = str(source_path)
    return metadata


def write_calibration_metadata(calibrate_path: str | Path, metadata: dict[str, Any]) -> Path:
    sidecar_path = calibration_metadata_path_for(calibrate_path)
    sidecar_path.write_text(json.dumps(metadata), encoding="utf-8")
    return sidecar_path


def load_calibration_metadata(calibrate_path: str | Path) -> dict[str, Any] | None:
    sidecar_path = calibration_metadata_path_for(calibrate_path)
    if not sidecar_path.exists():
        return None
    metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != CALIBRATION_METADATA_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported calibration metadata schema in {sidecar_path}: "
            f"{metadata.get('schema_version')!r}"
        )
    serials = _validate_serials("calibration metadata serial_numbers", metadata.get("serial_numbers"))
    reference_serials = _validate_serials(
        "calibration metadata calibration_reference_serials",
        metadata.get("calibration_reference_serials", serials),
    )
    if reference_serials != serials:
        raise ValueError(
            "Calibration metadata serial_numbers and calibration_reference_serials must match "
            "for raw calibrate.pkl producers."
        )
    transform_count = int(metadata.get("transform_count", len(serials)))
    if transform_count != len(serials):
        raise ValueError(
            "Calibration metadata transform_count does not match serial_numbers length. "
            f"transform_count={transform_count}, serial_numbers={len(serials)}"
        )
    return metadata


def load_calibration_reference_serials(calibrate_path: str | Path) -> list[str] | None:
    metadata = load_calibration_metadata(calibrate_path)
    if metadata is None:
        return None
    return list(metadata["calibration_reference_serials"])
