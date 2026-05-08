from __future__ import annotations

from dataclasses import dataclass
import json
import time
from typing import Any

import numpy as np


PROTOCOL_NAME = "qqtt_demo_v0_2_async_remote_ffs"
PROTOCOL_VERSION = 1
COMPRESSION_MODES = ("lz4",)
RETURN_TYPES = ("depth_u16",)
MODES = ("single", "triplet")


class AsyncFfsProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class AsyncCameraRequest:
    camera_idx: int
    serial: str
    width: int
    height: int
    k_ir_left: np.ndarray
    k_color: np.ndarray
    t_ir_left_to_color: np.ndarray
    baseline_m: float
    ir_left_u8: np.ndarray
    ir_right_u8: np.ndarray


@dataclass(frozen=True)
class AsyncFfsRequest:
    header: dict[str, Any]
    cameras: list[AsyncCameraRequest]


@dataclass(frozen=True)
class AsyncCameraDepth:
    camera_idx: int
    serial: str
    depth_u16: np.ndarray


@dataclass(frozen=True)
class AsyncFfsReply:
    header: dict[str, Any]
    depths: list[AsyncCameraDepth]


def now_perf_ns() -> int:
    return int(time.perf_counter_ns())


def _json_dumps(data: dict[str, Any]) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _json_loads(data: bytes) -> dict[str, Any]:
    value = json.loads(data.decode("utf-8"))
    if not isinstance(value, dict):
        raise AsyncFfsProtocolError("JSON frame must decode to an object")
    return value


def _validate_compression(compression: str) -> str:
    value = str(compression)
    if value not in COMPRESSION_MODES:
        raise AsyncFfsProtocolError(f"unsupported compression={value!r}")
    return value


def _validate_return_type(return_type: str) -> str:
    value = str(return_type)
    if value not in RETURN_TYPES:
        raise AsyncFfsProtocolError(f"unsupported return_type={value!r}")
    return value


def _validate_mode(mode: str) -> str:
    value = str(mode)
    if value not in MODES:
        raise AsyncFfsProtocolError(f"unsupported mode={value!r}")
    return value


def _compress_lz4(array: np.ndarray) -> tuple[bytes, dict[str, Any]]:
    try:
        import lz4.frame  # type: ignore[import-not-found]
    except Exception as exc:
        raise AsyncFfsProtocolError("lz4 compression requires the lz4 package") from exc
    contiguous = np.ascontiguousarray(array)
    raw = contiguous.tobytes(order="C")
    encoded = lz4.frame.compress(raw, compression_level=0)
    return encoded, {
        "compression": "lz4",
        "dtype": str(contiguous.dtype),
        "shape": [int(item) for item in contiguous.shape],
        "uncompressed_bytes": int(len(raw)),
        "encoded_bytes": int(len(encoded)),
    }


def _decompress_lz4(payload: bytes, *, metadata: dict[str, Any], prefix: str) -> np.ndarray:
    try:
        import lz4.frame  # type: ignore[import-not-found]
    except Exception as exc:
        raise AsyncFfsProtocolError("lz4 compression requires the lz4 package") from exc
    compression = str(metadata.get(f"{prefix}_compression", metadata.get("compression", "")))
    _validate_compression(compression)
    dtype = np.dtype(str(metadata.get(f"{prefix}_dtype", "")))
    shape_value = metadata.get(f"{prefix}_shape")
    if not isinstance(shape_value, list | tuple) or not shape_value:
        raise AsyncFfsProtocolError(f"invalid {prefix}_shape={shape_value!r}")
    shape = tuple(int(item) for item in shape_value)
    raw = lz4.frame.decompress(payload)
    expected_bytes = int(np.prod(shape)) * int(dtype.itemsize)
    if len(raw) != expected_bytes:
        raise AsyncFfsProtocolError(
            f"{prefix} byte payload mismatch: expected={expected_bytes} got={len(raw)}"
        )
    return np.ascontiguousarray(np.frombuffer(raw, dtype=dtype).reshape(shape))


def matrix_to_nested_list(value: np.ndarray, *, shape: tuple[int, int]) -> list[list[float]]:
    matrix = np.asarray(value, dtype=np.float32).reshape(shape)
    return [[float(item) for item in row] for row in matrix]


def matrix_from_camera(camera: dict[str, Any], key: str, *, shape: tuple[int, int]) -> np.ndarray:
    if key not in camera:
        raise AsyncFfsProtocolError(f"camera metadata missing {key!r}")
    array = np.asarray(camera[key], dtype=np.float32)
    expected = int(np.prod(shape))
    if array.size != expected:
        raise AsyncFfsProtocolError(f"camera {key!r} expected {expected} values, got {array.size}")
    return np.ascontiguousarray(array.reshape(shape), dtype=np.float32)


def _add_payload_metadata(metadata: dict[str, Any], prefix: str, payload_meta: dict[str, Any]) -> None:
    metadata[f"{prefix}_compression"] = payload_meta["compression"]
    metadata[f"{prefix}_dtype"] = payload_meta["dtype"]
    metadata[f"{prefix}_shape"] = payload_meta["shape"]
    metadata[f"{prefix}_uncompressed_bytes"] = payload_meta["uncompressed_bytes"]
    metadata[f"{prefix}_encoded_bytes"] = payload_meta["encoded_bytes"]


def camera_header_from_arrays(
    *,
    camera_idx: int,
    serial: str,
    ir_left_u8: np.ndarray,
    k_ir_left: np.ndarray,
    k_color: np.ndarray,
    t_ir_left_to_color: np.ndarray,
    baseline_m: float,
) -> dict[str, Any]:
    left = np.asarray(ir_left_u8, dtype=np.uint8)
    if left.ndim != 2:
        raise AsyncFfsProtocolError("IR image must be a 2D uint8 array")
    height, width = [int(item) for item in left.shape]
    return {
        "camera_idx": int(camera_idx),
        "serial": str(serial),
        "width": width,
        "height": height,
        "format": "Y8",
        "dtype": "uint8",
        "K_ir_left": matrix_to_nested_list(k_ir_left, shape=(3, 3)),
        "K_color": matrix_to_nested_list(k_color, shape=(3, 3)),
        "T_ir_left_to_color": matrix_to_nested_list(t_ir_left_to_color, shape=(4, 4)),
        "baseline_m": float(baseline_m),
    }


def build_request_parts(
    *,
    request_id: str,
    mode: str,
    camera_payloads: list[dict[str, Any]],
    target_kit_fps: float,
    compression: str = "lz4",
    return_type: str = "depth_u16",
    created_perf_ns: int | None = None,
) -> list[bytes]:
    mode = _validate_mode(mode)
    compression = _validate_compression(compression)
    return_type = _validate_return_type(return_type)
    expected_count = 1 if mode == "single" else 3
    if len(camera_payloads) != expected_count:
        raise AsyncFfsProtocolError(f"mode={mode!r} expected {expected_count} camera payloads, got {len(camera_payloads)}")
    header = {
        "protocol": PROTOCOL_NAME,
        "version": PROTOCOL_VERSION,
        "request_id": str(request_id),
        "mode": mode,
        "compression": compression,
        "return_type": return_type,
        "created_perf_ns": int(now_perf_ns() if created_perf_ns is None else created_perf_ns),
        "target_kit_fps": float(target_kit_fps),
        "cameras": [],
        "request_uncompressed_bytes": 0,
        "request_encoded_bytes": 0,
    }
    parts: list[bytes] = [b""]
    for payload in camera_payloads:
        left = np.ascontiguousarray(payload["ir_left_u8"], dtype=np.uint8)
        right = np.ascontiguousarray(payload["ir_right_u8"], dtype=np.uint8)
        if left.ndim != 2 or right.ndim != 2:
            raise AsyncFfsProtocolError("IR images must be 2D uint8 arrays")
        if left.shape != right.shape:
            raise AsyncFfsProtocolError(f"IR shape mismatch: left={left.shape} right={right.shape}")
        camera = camera_header_from_arrays(
            camera_idx=int(payload["camera_idx"]),
            serial=str(payload["serial"]),
            ir_left_u8=left,
            k_ir_left=np.asarray(payload["K_ir_left"], dtype=np.float32),
            k_color=np.asarray(payload["K_color"], dtype=np.float32),
            t_ir_left_to_color=np.asarray(payload["T_ir_left_to_color"], dtype=np.float32),
            baseline_m=float(payload["baseline_m"]),
        )
        left_payload, left_meta = _compress_lz4(left)
        right_payload, right_meta = _compress_lz4(right)
        prefix = f"cam{int(camera['camera_idx'])}"
        _add_payload_metadata(camera, "ir_left", left_meta)
        _add_payload_metadata(camera, "ir_right", right_meta)
        header["request_uncompressed_bytes"] += int(left_meta["uncompressed_bytes"]) + int(right_meta["uncompressed_bytes"])
        header["request_encoded_bytes"] += int(left_meta["encoded_bytes"]) + int(right_meta["encoded_bytes"])
        header["cameras"].append(camera)
        parts.extend([left_payload, right_payload])
        header[f"{prefix}_payload_order"] = ["left", "right"]
    parts[0] = _json_dumps(header)
    return parts


def parse_request_parts(parts: list[bytes] | tuple[bytes, ...]) -> AsyncFfsRequest:
    if len(parts) < 3 or len(parts) % 2 != 1:
        raise AsyncFfsProtocolError(f"request expected header plus left/right pairs, got {len(parts)} frames")
    header = _json_loads(parts[0])
    if header.get("protocol") != PROTOCOL_NAME:
        raise AsyncFfsProtocolError(f"unsupported protocol={header.get('protocol')!r}")
    if int(header.get("version", -1)) != PROTOCOL_VERSION:
        raise AsyncFfsProtocolError(f"unsupported version={header.get('version')!r}")
    mode = _validate_mode(str(header.get("mode", "")))
    _validate_compression(str(header.get("compression", "")))
    _validate_return_type(str(header.get("return_type", "")))
    cameras_meta = header.get("cameras")
    if not isinstance(cameras_meta, list):
        raise AsyncFfsProtocolError("request header cameras must be a list")
    expected_count = 1 if mode == "single" else 3
    if len(cameras_meta) != expected_count:
        raise AsyncFfsProtocolError(f"mode={mode!r} expected {expected_count} cameras, got {len(cameras_meta)}")
    if len(parts) != 1 + 2 * len(cameras_meta):
        raise AsyncFfsProtocolError(
            f"request frame count mismatch: expected={1 + 2 * len(cameras_meta)} got={len(parts)}"
        )
    cameras: list[AsyncCameraRequest] = []
    part_idx = 1
    for camera in cameras_meta:
        if not isinstance(camera, dict):
            raise AsyncFfsProtocolError("camera metadata must be an object")
        left = _decompress_lz4(parts[part_idx], metadata=camera, prefix="ir_left")
        right = _decompress_lz4(parts[part_idx + 1], metadata=camera, prefix="ir_right")
        part_idx += 2
        if left.dtype != np.uint8 or right.dtype != np.uint8:
            raise AsyncFfsProtocolError("decoded IR payloads must be uint8")
        if left.shape != right.shape:
            raise AsyncFfsProtocolError(f"decoded IR shape mismatch: left={left.shape} right={right.shape}")
        height = int(camera.get("height", left.shape[0]))
        width = int(camera.get("width", left.shape[1]))
        if left.shape != (height, width):
            raise AsyncFfsProtocolError(f"decoded IR shape {left.shape} does not match camera metadata {(height, width)}")
        cameras.append(
            AsyncCameraRequest(
                camera_idx=int(camera["camera_idx"]),
                serial=str(camera.get("serial", "")),
                width=width,
                height=height,
                k_ir_left=matrix_from_camera(camera, "K_ir_left", shape=(3, 3)),
                k_color=matrix_from_camera(camera, "K_color", shape=(3, 3)),
                t_ir_left_to_color=matrix_from_camera(camera, "T_ir_left_to_color", shape=(4, 4)),
                baseline_m=float(camera["baseline_m"]),
                ir_left_u8=left,
                ir_right_u8=right,
            )
        )
    return AsyncFfsRequest(header=header, cameras=cameras)


def build_reply_parts(
    *,
    request: AsyncFfsRequest | None,
    request_header: dict[str, Any] | None = None,
    depths: list[np.ndarray] | None = None,
    status: str = "ok",
    error: str = "",
    per_camera_stats: list[dict[str, Any]] | None = None,
    server_total_ms: float = 0.0,
    compression: str = "lz4",
    return_type: str = "depth_u16",
) -> list[bytes]:
    compression = _validate_compression(compression)
    return_type = _validate_return_type(return_type)
    header_source = request.header if request is not None else (request_header or {})
    camera_source = request.cameras if request is not None else []
    depth_values = [] if depths is None else list(depths)
    if status == "ok" and len(depth_values) != len(camera_source):
        raise AsyncFfsProtocolError(f"reply expected {len(camera_source)} depths, got {len(depth_values)}")
    reply_header: dict[str, Any] = {
        "protocol": PROTOCOL_NAME,
        "version": PROTOCOL_VERSION,
        "request_id": str(header_source.get("request_id", "")),
        "mode": str(header_source.get("mode", "")),
        "compression": compression,
        "return_type": return_type,
        "status": str(status),
        "error": str(error),
        "created_perf_ns": int(header_source.get("created_perf_ns", 0) or 0),
        "server_total_ms": float(server_total_ms),
        "cameras": [],
        "per_camera_stats": [] if per_camera_stats is None else list(per_camera_stats),
        "response_uncompressed_bytes": 0,
        "response_encoded_bytes": 0,
    }
    parts: list[bytes] = [b""]
    if status != "ok":
        parts[0] = _json_dumps(reply_header)
        return parts
    for camera, depth in zip(camera_source, depth_values, strict=True):
        depth_u16 = np.ascontiguousarray(depth, dtype=np.uint16)
        if depth_u16.ndim != 2:
            raise AsyncFfsProtocolError(f"depth_u16 must be 2D, got {depth_u16.shape}")
        payload, meta = _compress_lz4(depth_u16)
        camera_meta = {
            "camera_idx": int(camera.camera_idx),
            "serial": str(camera.serial),
            "width": int(depth_u16.shape[1]),
            "height": int(depth_u16.shape[0]),
            "dtype": "uint16",
        }
        _add_payload_metadata(camera_meta, "depth", meta)
        reply_header["response_uncompressed_bytes"] += int(meta["uncompressed_bytes"])
        reply_header["response_encoded_bytes"] += int(meta["encoded_bytes"])
        reply_header["cameras"].append(camera_meta)
        parts.append(payload)
    parts[0] = _json_dumps(reply_header)
    return parts


def build_error_reply_parts(
    *,
    request_id: str = "",
    mode: str = "",
    error: str,
    created_perf_ns: int = 0,
    compression: str = "lz4",
    return_type: str = "depth_u16",
    server_total_ms: float = 0.0,
) -> list[bytes]:
    return build_reply_parts(
        request=None,
        request_header={
            "request_id": str(request_id),
            "mode": str(mode),
            "created_perf_ns": int(created_perf_ns),
        },
        depths=[],
        status="error",
        error=str(error),
        server_total_ms=float(server_total_ms),
        compression=compression,
        return_type=return_type,
    )


def parse_reply_parts(parts: list[bytes] | tuple[bytes, ...]) -> AsyncFfsReply:
    if len(parts) < 1:
        raise AsyncFfsProtocolError("reply must contain at least a header frame")
    header = _json_loads(parts[0])
    if header.get("protocol") != PROTOCOL_NAME:
        raise AsyncFfsProtocolError(f"unsupported protocol={header.get('protocol')!r}")
    if int(header.get("version", -1)) != PROTOCOL_VERSION:
        raise AsyncFfsProtocolError(f"unsupported version={header.get('version')!r}")
    _validate_compression(str(header.get("compression", "")))
    _validate_return_type(str(header.get("return_type", "")))
    status = str(header.get("status", ""))
    if status != "ok":
        return AsyncFfsReply(header=header, depths=[])
    cameras_meta = header.get("cameras")
    if not isinstance(cameras_meta, list):
        raise AsyncFfsProtocolError("reply header cameras must be a list")
    if len(parts) != 1 + len(cameras_meta):
        raise AsyncFfsProtocolError(f"reply frame count mismatch: expected={1 + len(cameras_meta)} got={len(parts)}")
    depths: list[AsyncCameraDepth] = []
    for idx, camera in enumerate(cameras_meta):
        if not isinstance(camera, dict):
            raise AsyncFfsProtocolError("reply camera metadata must be an object")
        depth = _decompress_lz4(parts[idx + 1], metadata=camera, prefix="depth")
        if depth.dtype != np.uint16:
            raise AsyncFfsProtocolError(f"reply depth must be uint16, got {depth.dtype}")
        depths.append(
            AsyncCameraDepth(
                camera_idx=int(camera["camera_idx"]),
                serial=str(camera.get("serial", "")),
                depth_u16=np.ascontiguousarray(depth, dtype=np.uint16),
            )
        )
    return AsyncFfsReply(header=header, depths=depths)
