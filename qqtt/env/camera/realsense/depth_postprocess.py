from __future__ import annotations

from typing import Any

import numpy as np
import pyrealsense2 as rs


FFS_NATIVE_LIKE_DEPTH_POSTPROCESS_DIR = "depth_ffs_native_like_postprocess"
FFS_NATIVE_LIKE_DEPTH_POSTPROCESS_FLOAT_DIR = "depth_ffs_native_like_postprocess_float_m"
FFS_NATIVE_LIKE_DEPTH_POSTPROCESS_ON_THE_FLY_SUFFIX = "ffs_native_like_postprocess"
NATIVE_DEPTH_POSTPROCESS_CONTRACT = {
    "mode": "native_depth_postprocess",
    "chain": (
        "depth_to_disparity",
        "spatial_filter",
        "temporal_filter",
        "disparity_to_depth",
    ),
    "spatial_filter": {
        "filter_magnitude": 5,
        "filter_smooth_alpha": 0.75,
        "filter_smooth_delta": 1,
        "holes_fill": 1,
    },
    "temporal_filter": {
        "filter_smooth_alpha": 0.75,
        "filter_smooth_delta": 1,
    },
}


def native_depth_postprocess_contract() -> dict[str, Any]:
    return {
        "mode": str(NATIVE_DEPTH_POSTPROCESS_CONTRACT["mode"]),
        "chain": tuple(NATIVE_DEPTH_POSTPROCESS_CONTRACT["chain"]),
        "spatial_filter": dict(NATIVE_DEPTH_POSTPROCESS_CONTRACT["spatial_filter"]),
        "temporal_filter": dict(NATIVE_DEPTH_POSTPROCESS_CONTRACT["temporal_filter"]),
    }


def _configure_native_depth_filters(*, rs_module=rs):
    depth_to_disparity = rs_module.disparity_transform(True)
    disparity_to_depth = rs_module.disparity_transform(False)

    spatial = rs_module.spatial_filter()
    for option_name, option_value in NATIVE_DEPTH_POSTPROCESS_CONTRACT["spatial_filter"].items():
        spatial.set_option(getattr(rs_module.option, option_name), option_value)

    temporal = rs_module.temporal_filter()
    for option_name, option_value in NATIVE_DEPTH_POSTPROCESS_CONTRACT["temporal_filter"].items():
        temporal.set_option(getattr(rs_module.option, option_name), option_value)

    return depth_to_disparity, spatial, temporal, disparity_to_depth


def apply_native_depth_postprocess_frame(depth_frame, *, rs_module=rs):
    depth_to_disparity, spatial, temporal, disparity_to_depth = _configure_native_depth_filters(rs_module=rs_module)
    filtered_depth = depth_to_disparity.process(depth_frame)
    filtered_depth = spatial.process(filtered_depth)
    filtered_depth = temporal.process(filtered_depth)
    filtered_depth = disparity_to_depth.process(filtered_depth)
    return filtered_depth


def _decode_depth_units_to_meters(depth_u16: np.ndarray, depth_scale_m_per_unit: float) -> np.ndarray:
    depth = np.asarray(depth_u16, dtype=np.uint16)
    depth_m = depth.astype(np.float32) * float(depth_scale_m_per_unit)
    depth_m[depth == 0] = 0.0
    return depth_m


def _quantize_depth_with_invalid_zero(depth_m: np.ndarray, depth_scale_m_per_unit: float) -> np.ndarray:
    depth = np.asarray(depth_m, dtype=np.float32)
    scale = float(depth_scale_m_per_unit)
    encoded = np.zeros(depth.shape, dtype=np.uint16)
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        return encoded
    scaled = np.rint(depth[valid] / scale)
    scaled = np.clip(scaled, 1, np.iinfo(np.uint16).max)
    encoded[valid] = scaled.astype(np.uint16)
    return encoded


def apply_ffs_native_like_depth_postprocess_u16(
    depth_u16: np.ndarray,
    *,
    depth_scale_m_per_unit: float,
    fps: int = 30,
    frame_number: int = 1,
    timestamp_ms: float | None = None,
    rs_module=rs,
) -> np.ndarray:
    depth = np.ascontiguousarray(np.asarray(depth_u16, dtype=np.uint16))
    if depth.ndim != 2:
        raise ValueError(f"Expected a 2D uint16 depth image, got shape={depth.shape}.")

    height, width = depth.shape
    timestamp_value = float(timestamp_ms) if timestamp_ms is not None else float(frame_number) * (1000.0 / max(1, int(fps)))

    device = rs_module.software_device()
    sensor = device.add_sensor("Depth")
    stream = rs_module.video_stream()
    stream.type = rs_module.stream.depth
    stream.index = 0
    stream.uid = 1
    stream.width = int(width)
    stream.height = int(height)
    stream.fps = int(fps)
    stream.bpp = 2
    stream.fmt = rs_module.format.z16
    sensor.add_video_stream(stream)
    sensor.add_read_only_option(rs_module.option.depth_units, float(depth_scale_m_per_unit))

    queue = rs_module.frame_queue()
    sensor.open(sensor.get_stream_profiles())
    sensor.start(queue)
    try:
        frame = rs_module.software_video_frame()
        frame.stride = int(depth.strides[0])
        frame.bpp = 2
        frame.pixels = np.asarray(depth, dtype="ushort")
        frame.timestamp = timestamp_value
        frame.frame_number = int(frame_number)
        frame.profile = sensor.get_stream_profiles()[0].as_video_stream_profile()
        frame.depth_units = float(depth_scale_m_per_unit)
        sensor.on_video_frame(frame)

        input_frame = queue.wait_for_frame(1000)
        output_frame = apply_native_depth_postprocess_frame(input_frame, rs_module=rs_module)
        return np.ascontiguousarray(np.asanyarray(output_frame.get_data()), dtype=np.uint16)
    finally:
        sensor.stop()
        sensor.close()


def apply_ffs_native_like_depth_postprocess_float_m(
    depth_m: np.ndarray,
    *,
    depth_scale_m_per_unit: float,
    fps: int = 30,
    frame_number: int = 1,
    timestamp_ms: float | None = None,
    rs_module=rs,
) -> tuple[np.ndarray, np.ndarray]:
    quantized_depth = _quantize_depth_with_invalid_zero(depth_m, depth_scale_m_per_unit)
    filtered_u16 = apply_ffs_native_like_depth_postprocess_u16(
        quantized_depth,
        depth_scale_m_per_unit=depth_scale_m_per_unit,
        fps=fps,
        frame_number=frame_number,
        timestamp_ms=timestamp_ms,
        rs_module=rs_module,
    )
    filtered_m = _decode_depth_units_to_meters(filtered_u16, depth_scale_m_per_unit)
    return filtered_u16, filtered_m
