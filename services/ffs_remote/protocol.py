from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

import numpy as np


PROTOCOL_VERSION = 1
RETURN_TYPES = ("depth_u16", "depth_float_m")


class FfsRemoteProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class FfsDepthRequest:
    metadata: dict[str, Any]
    ir_left_u8: np.ndarray
    ir_right_u8: np.ndarray


@dataclass(frozen=True)
class FfsDepthResponse:
    metadata: dict[str, Any]
    depth: np.ndarray


def _json_dumps(data: dict[str, Any]) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _json_loads(data: bytes) -> dict[str, Any]:
    value = json.loads(data.decode("utf-8"))
    if not isinstance(value, dict):
        raise FfsRemoteProtocolError("metadata JSON must decode to an object")
    return value


def matrix_to_list(value: np.ndarray, *, shape: tuple[int, int]) -> list[float]:
    matrix = np.asarray(value, dtype=np.float32).reshape(shape)
    return [float(item) for item in matrix.ravel()]


def matrix_from_metadata(metadata: dict[str, Any], key: str, *, shape: tuple[int, int]) -> np.ndarray:
    if key not in metadata:
        raise FfsRemoteProtocolError(f"request metadata missing {key!r}")
    array = np.asarray(metadata[key], dtype=np.float32)
    expected = int(np.prod(shape))
    if array.size != expected:
        raise FfsRemoteProtocolError(f"metadata {key!r} expected {expected} values, got {array.size}")
    return np.ascontiguousarray(array.reshape(shape), dtype=np.float32)


def build_depth_request_parts(
    *,
    frame_id: int,
    ir_left_u8: np.ndarray,
    ir_right_u8: np.ndarray,
    color_shape: tuple[int, int],
    k_ir_left: np.ndarray,
    k_color: np.ndarray,
    t_ir_left_to_color: np.ndarray,
    baseline_m: float,
    depth_scale_m_per_unit: float,
    return_type: str,
    camera_idx: int = 0,
) -> list[bytes]:
    if return_type not in RETURN_TYPES:
        raise FfsRemoteProtocolError(f"unsupported return_type: {return_type}")
    left = np.ascontiguousarray(ir_left_u8, dtype=np.uint8)
    right = np.ascontiguousarray(ir_right_u8, dtype=np.uint8)
    if left.ndim != 2 or right.ndim != 2:
        raise FfsRemoteProtocolError("IR images must be 2D uint8 arrays")
    if left.shape != right.shape:
        raise FfsRemoteProtocolError(f"IR image shapes differ: left={left.shape} right={right.shape}")
    color_h, color_w = [int(item) for item in color_shape]
    if color_h <= 0 or color_w <= 0:
        raise FfsRemoteProtocolError(f"invalid color_shape: {color_shape!r}")
    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "frame_id": int(frame_id),
        "camera_idx": int(camera_idx),
        "ir_shape": [int(left.shape[0]), int(left.shape[1])],
        "color_shape": [color_h, color_w],
        "k_ir_left": matrix_to_list(k_ir_left, shape=(3, 3)),
        "k_color": matrix_to_list(k_color, shape=(3, 3)),
        "t_ir_left_to_color": matrix_to_list(t_ir_left_to_color, shape=(4, 4)),
        "baseline_m": float(baseline_m),
        "depth_scale_m_per_unit": float(depth_scale_m_per_unit),
        "return_type": str(return_type),
        "ir_dtype": "uint8",
    }
    return [_json_dumps(metadata), left.tobytes(order="C"), right.tobytes(order="C")]


def parse_depth_request_parts(parts: list[bytes] | tuple[bytes, ...]) -> FfsDepthRequest:
    if len(parts) != 3:
        raise FfsRemoteProtocolError(f"request expected 3 parts, got {len(parts)}")
    metadata = _json_loads(parts[0])
    if int(metadata.get("protocol_version", -1)) != PROTOCOL_VERSION:
        raise FfsRemoteProtocolError(f"unsupported protocol_version={metadata.get('protocol_version')!r}")
    if metadata.get("return_type") not in RETURN_TYPES:
        raise FfsRemoteProtocolError(f"unsupported return_type={metadata.get('return_type')!r}")
    shape_value = metadata.get("ir_shape")
    if not isinstance(shape_value, list | tuple) or len(shape_value) != 2:
        raise FfsRemoteProtocolError(f"invalid ir_shape={shape_value!r}")
    ir_shape = (int(shape_value[0]), int(shape_value[1]))
    expected_bytes = int(np.prod(ir_shape))
    if len(parts[1]) != expected_bytes or len(parts[2]) != expected_bytes:
        raise FfsRemoteProtocolError(
            f"IR byte payload mismatch: expected={expected_bytes} left={len(parts[1])} right={len(parts[2])}"
        )
    left = np.frombuffer(parts[1], dtype=np.uint8).reshape(ir_shape)
    right = np.frombuffer(parts[2], dtype=np.uint8).reshape(ir_shape)
    return FfsDepthRequest(
        metadata=metadata,
        ir_left_u8=np.ascontiguousarray(left),
        ir_right_u8=np.ascontiguousarray(right),
    )


def build_depth_response_parts(
    *,
    frame_id: int,
    depth: np.ndarray,
    depth_dtype: str,
    status: str = "ok",
    error: str = "",
    server_ffs_ms: float = 0.0,
    server_align_ms: float = 0.0,
    server_total_ms: float = 0.0,
    depth_scale_m_per_unit: float = 0.001,
) -> list[bytes]:
    dtype = np.dtype(depth_dtype)
    depth_array = np.ascontiguousarray(depth, dtype=dtype)
    if depth_array.ndim != 2:
        raise FfsRemoteProtocolError(f"depth response must be 2D, got {depth_array.shape}")
    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "frame_id": int(frame_id),
        "status": str(status),
        "error": str(error),
        "depth_shape": [int(depth_array.shape[0]), int(depth_array.shape[1])],
        "depth_dtype": str(depth_array.dtype),
        "depth_scale_m_per_unit": float(depth_scale_m_per_unit),
        "server_ffs_ms": float(server_ffs_ms),
        "server_align_ms": float(server_align_ms),
        "server_total_ms": float(server_total_ms),
    }
    return [_json_dumps(metadata), depth_array.tobytes(order="C")]


def parse_depth_response_parts(parts: list[bytes] | tuple[bytes, ...]) -> FfsDepthResponse:
    if len(parts) != 2:
        raise FfsRemoteProtocolError(f"response expected 2 parts, got {len(parts)}")
    metadata = _json_loads(parts[0])
    if int(metadata.get("protocol_version", -1)) != PROTOCOL_VERSION:
        raise FfsRemoteProtocolError(f"unsupported protocol_version={metadata.get('protocol_version')!r}")
    shape_value = metadata.get("depth_shape")
    if not isinstance(shape_value, list | tuple) or len(shape_value) != 2:
        raise FfsRemoteProtocolError(f"invalid depth_shape={shape_value!r}")
    depth_shape = (int(shape_value[0]), int(shape_value[1]))
    dtype = np.dtype(str(metadata.get("depth_dtype", "")))
    expected_bytes = int(np.prod(depth_shape)) * int(dtype.itemsize)
    if len(parts[1]) != expected_bytes:
        raise FfsRemoteProtocolError(f"depth byte payload mismatch: expected={expected_bytes} got={len(parts[1])}")
    depth = np.frombuffer(parts[1], dtype=dtype).reshape(depth_shape)
    return FfsDepthResponse(metadata=metadata, depth=np.ascontiguousarray(depth))
