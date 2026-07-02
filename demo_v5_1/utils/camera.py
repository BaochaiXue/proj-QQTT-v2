"""RealSense camera helpers: profiles, intrinsics, device selection."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

SUPPORTED_CAPTURE_FPS = (5, 15, 30, 60)
SUPPORTED_PROFILES = ("848x480", "640x480")


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


def parse_profile(profile: str) -> tuple[int, int]:
    """Parse profile."""
    if profile not in SUPPORTED_PROFILES:
        raise ValueError(f"unsupported profile {profile!r}; expected one of {', '.join(SUPPORTED_PROFILES)}")
    width_text, height_text = profile.split("x", 1)
    return int(width_text), int(height_text)


def camera_intrinsics_from_rs(intrinsics: object) -> CameraIntrinsics:
    """Return the camera intrinsics from rs."""
    return CameraIntrinsics(
        fx=float(getattr(intrinsics, "fx")),
        fy=float(getattr(intrinsics, "fy")),
        cx=float(getattr(intrinsics, "ppx")),
        cy=float(getattr(intrinsics, "ppy")),
    )


def camera_intrinsics_to_matrix(intrinsics: CameraIntrinsics) -> np.ndarray:
    """Return the camera intrinsics to matrix."""
    return np.array(
        [
            [float(intrinsics.fx), 0.0, float(intrinsics.cx)],
            [0.0, float(intrinsics.fy), float(intrinsics.cy)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def rs_intrinsics_to_matrix(intrinsics: object) -> np.ndarray:
    """Return the rs intrinsics to matrix."""
    return camera_intrinsics_to_matrix(camera_intrinsics_from_rs(intrinsics))


def rs_extrinsics_to_matrix(extrinsics: object) -> np.ndarray:
    # librealsense stores rotation column-major; return row-major [R|t]
    # for standard to_point = R @ from_point + t consumers.
    """Return the rs extrinsics to matrix."""
    rotation = list(map(float, getattr(extrinsics, "rotation")))
    translation = list(map(float, getattr(extrinsics, "translation")))
    return np.array(
        [
            [rotation[0], rotation[3], rotation[6], translation[0]],
            [rotation[1], rotation[4], rotation[7], translation[1]],
            [rotation[2], rotation[5], rotation[8], translation[2]],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def rs_translation_norm(extrinsics: object) -> float:
    """Translation magnitude in meters (e.g. IR stereo baseline)."""
    tx, ty, tz = map(float, getattr(extrinsics, "translation"))
    return float(math.sqrt(tx * tx + ty * ty + tz * tz))


def load_realsense_module():
    """Load realsense module."""
    try:
        import pyrealsense2 as rs  # type: ignore
    except ImportError as exc:
        raise RuntimeError("pyrealsense2 is required for camera capture") from exc
    return rs


def _device_info(device: object, info_key: object) -> str:
    """Read a camera_info field, returning "" when the device does not expose it."""
    if hasattr(device, "supports") and device.supports(info_key):
        return str(device.get_info(info_key))
    return ""


def list_d400_serials(rs: object) -> list[str]:
    """Return the list d400 serials."""
    context = rs.context()
    serials: list[str] = []
    for device in context.query_devices():
        product_line = _device_info(device, rs.camera_info.product_line)
        serial = _device_info(device, rs.camera_info.serial_number)
        if serial and product_line.upper() == "D400":
            serials.append(serial)
    return sorted(serials)


def resolve_serial(rs: object, requested_serial: str | None) -> str:
    """Resolve serial."""
    serials = list_d400_serials(rs)
    if requested_serial:
        if serials and requested_serial not in serials:
            available = ", ".join(serials)
            raise RuntimeError(f"requested serial {requested_serial!r} is not a detected D400 device; available: {available}")
        return requested_serial
    if not serials:
        raise RuntimeError("no D400 RealSense device detected")
    return serials[0]


def apply_emitter(profile: object, emitter: str, rs: object) -> None:
    """Apply emitter."""
    if emitter == "auto":
        return
    depth_sensor = profile.get_device().first_depth_sensor()
    if depth_sensor.supports(rs.option.emitter_enabled):
        depth_sensor.set_option(rs.option.emitter_enabled, 1.0 if emitter == "on" else 0.0)
