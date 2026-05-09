from __future__ import annotations

from dataclasses import dataclass
import json
import time
from typing import Any

import numpy as np


PROTOCOL_NAME = "qqtt_demo_v0_3_staged_remote_ffs"
PROTOCOL_VERSION = 1
CAMERA_COUNT = 3
COMPRESSION_MODES = ("lz4",)
RETURN_TYPES = ("depth_u16",)
MODES = ("triplet-replay",)
METRIC_KEYS = (
    "server_decode_ms",
    "server_ffs_cam0_ms",
    "server_ffs_cam1_ms",
    "server_ffs_cam2_ms",
    "server_ffs_triplet_ms",
    "server_ffs_batch3_ms",
    "server_postprocess_encode_ms",
    "server_total_ms",
    "raw_queue_size",
    "decoded_queue_size",
    "postprocess_queue_size",
    "send_queue_size",
    "depth_nonzero_cam0",
    "depth_nonzero_cam1",
    "depth_nonzero_cam2",
    "request_kb",
    "reply_kb",
)


class StagedFfsProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class StagedCameraRequest:
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
class StagedFfsRequest:
    header: dict[str, Any]
    cameras: list[StagedCameraRequest]


@dataclass(frozen=True)
class StagedCameraDepth:
    camera_idx: int
    serial: str
    depth_u16: np.ndarray


@dataclass(frozen=True)
class StagedFfsReply:
    header: dict[str, Any]
    depths: list[StagedCameraDepth]


def now_perf_ns() -> int:
    return int(time.perf_counter_ns())


def _json_dumps(data: dict[str, Any]) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _json_loads(data: bytes) -> dict[str, Any]:
    value = json.loads(data.decode("utf-8"))
    if not isinstance(value, dict):
        raise StagedFfsProtocolError("JSON frame must decode to an object")
    return value


def _validate_compression(compression: str) -> str:
    value = str(compression)
    if value not in COMPRESSION_MODES:
        raise StagedFfsProtocolError(f"unsupported compression={value!r}")
    return value


def _validate_return_type(return_type: str) -> str:
    value = str(return_type)
    if value not in RETURN_TYPES:
        raise StagedFfsProtocolError(f"unsupported return_type={value!r}")
    return value


def _validate_mode(mode: str) -> str:
    value = str(mode)
    if value not in MODES:
        raise StagedFfsProtocolError(f"unsupported mode={value!r}")
    return value


def _compress_lz4(array: np.ndarray) -> tuple[bytes, dict[str, Any]]:
    try:
        import lz4.frame  # type: ignore[import-not-found]
    except Exception as exc:
        raise StagedFfsProtocolError("lz4 compression requires the lz4 package") from exc
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
        raise StagedFfsProtocolError("lz4 compression requires the lz4 package") from exc
    compression = str(metadata.get(f"{prefix}_compression", metadata.get("compression", "")))
    _validate_compression(compression)
    dtype = np.dtype(str(metadata.get(f"{prefix}_dtype", "")))
    shape_value = metadata.get(f"{prefix}_shape")
    if not isinstance(shape_value, list | tuple) or not shape_value:
        raise StagedFfsProtocolError(f"invalid {prefix}_shape={shape_value!r}")
    shape = tuple(int(item) for item in shape_value)
    raw = lz4.frame.decompress(payload)
    expected_bytes = int(np.prod(shape)) * int(dtype.itemsize)
    if len(raw) != expected_bytes:
        raise StagedFfsProtocolError(
            f"{prefix} byte payload mismatch: expected={expected_bytes} got={len(raw)}"
        )
    return np.ascontiguousarray(np.frombuffer(raw, dtype=dtype).reshape(shape))


def _add_payload_metadata(metadata: dict[str, Any], prefix: str, payload_meta: dict[str, Any]) -> None:
    metadata[f"{prefix}_compression"] = payload_meta["compression"]
    metadata[f"{prefix}_dtype"] = payload_meta["dtype"]
    metadata[f"{prefix}_shape"] = payload_meta["shape"]
    metadata[f"{prefix}_uncompressed_bytes"] = payload_meta["uncompressed_bytes"]
    metadata[f"{prefix}_encoded_bytes"] = payload_meta["encoded_bytes"]


def matrix_to_nested_list(value: np.ndarray, *, shape: tuple[int, int]) -> list[list[float]]:
    matrix = np.asarray(value, dtype=np.float32).reshape(shape)
    return [[float(item) for item in row] for row in matrix]


def matrix_from_camera(camera: dict[str, Any], key: str, *, shape: tuple[int, int]) -> np.ndarray:
    if key not in camera:
        raise StagedFfsProtocolError(f"camera metadata missing {key!r}")
    array = np.asarray(camera[key], dtype=np.float32)
    expected = int(np.prod(shape))
    if array.size != expected:
        raise StagedFfsProtocolError(f"camera {key!r} expected {expected} values, got {array.size}")
    return np.ascontiguousarray(array.reshape(shape), dtype=np.float32)


def _camera_header_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    left = np.asarray(payload["ir_left_u8"], dtype=np.uint8)
    if left.ndim != 2:
        raise StagedFfsProtocolError("IR image must be a 2D uint8 array")
    height, width = [int(item) for item in left.shape]
    return {
        "camera_idx": int(payload["camera_idx"]),
        "serial": str(payload.get("serial", "")),
        "width": width,
        "height": height,
        "format": "Y8",
        "dtype": "uint8",
        "K_ir_left": matrix_to_nested_list(np.asarray(payload["K_ir_left"], dtype=np.float32), shape=(3, 3)),
        "K_color": matrix_to_nested_list(np.asarray(payload["K_color"], dtype=np.float32), shape=(3, 3)),
        "T_ir_left_to_color": matrix_to_nested_list(
            np.asarray(payload["T_ir_left_to_color"], dtype=np.float32),
            shape=(4, 4),
        ),
        "baseline_m": float(payload["baseline_m"]),
    }


def build_request_parts(
    *,
    request_id: str,
    kit_idx: int,
    camera_payloads: list[dict[str, Any]],
    capture_kit_fps: float,
    mode: str = "triplet-replay",
    phase: str = "measured",
    compression: str = "lz4",
    return_type: str = "depth_u16",
    created_perf_ns: int | None = None,
) -> list[bytes]:
    mode = _validate_mode(mode)
    compression = _validate_compression(compression)
    return_type = _validate_return_type(return_type)
    if len(camera_payloads) != CAMERA_COUNT:
        raise StagedFfsProtocolError(f"v0.3 request expected {CAMERA_COUNT} camera payloads, got {len(camera_payloads)}")
    if float(capture_kit_fps) <= 0:
        raise StagedFfsProtocolError("capture_kit_fps must be positive")
    header: dict[str, Any] = {
        "protocol": PROTOCOL_NAME,
        "version": PROTOCOL_VERSION,
        "mode": mode,
        "request_id": str(request_id),
        "kit_idx": int(kit_idx),
        "phase": str(phase),
        "compression": compression,
        "return_type": return_type,
        "created_perf_ns": int(now_perf_ns() if created_perf_ns is None else created_perf_ns),
        "capture_kit_fps": float(capture_kit_fps),
        "camera_count": CAMERA_COUNT,
        "cameras": [],
        "request_uncompressed_bytes": 0,
        "request_encoded_bytes": 0,
        "request_kb": 0.0,
    }
    parts: list[bytes] = [b""]
    for payload in camera_payloads:
        left = np.ascontiguousarray(payload["ir_left_u8"], dtype=np.uint8)
        right = np.ascontiguousarray(payload["ir_right_u8"], dtype=np.uint8)
        if left.ndim != 2 or right.ndim != 2:
            raise StagedFfsProtocolError("IR images must be 2D uint8 arrays")
        if left.shape != right.shape:
            raise StagedFfsProtocolError(f"IR shape mismatch: left={left.shape} right={right.shape}")
        camera = _camera_header_from_payload({**payload, "ir_left_u8": left})
        left_payload, left_meta = _compress_lz4(left)
        right_payload, right_meta = _compress_lz4(right)
        _add_payload_metadata(camera, "ir_left", left_meta)
        _add_payload_metadata(camera, "ir_right", right_meta)
        header["request_uncompressed_bytes"] += int(left_meta["uncompressed_bytes"]) + int(right_meta["uncompressed_bytes"])
        header["request_encoded_bytes"] += int(left_meta["encoded_bytes"]) + int(right_meta["encoded_bytes"])
        header["cameras"].append(camera)
        parts.extend([left_payload, right_payload])
    camera_order = [int(camera["camera_idx"]) for camera in header["cameras"]]
    if camera_order != [0, 1, 2]:
        raise StagedFfsProtocolError(f"v0.3 camera order must be [0, 1, 2], got {camera_order}")
    parts[0] = _json_dumps(header)
    header["request_kb"] = float(sum(len(part) for part in parts) / 1024.0)
    parts[0] = _json_dumps(header)
    return parts


def parse_request_parts(parts: list[bytes] | tuple[bytes, ...]) -> StagedFfsRequest:
    if len(parts) != 1 + (2 * CAMERA_COUNT):
        raise StagedFfsProtocolError(f"v0.3 request expected 7 frames, got {len(parts)}")
    header = _json_loads(parts[0])
    if header.get("protocol") != PROTOCOL_NAME:
        raise StagedFfsProtocolError(f"unsupported protocol={header.get('protocol')!r}")
    if int(header.get("version", -1)) != PROTOCOL_VERSION:
        raise StagedFfsProtocolError(f"unsupported version={header.get('version')!r}")
    _validate_mode(str(header.get("mode", "")))
    _validate_compression(str(header.get("compression", "")))
    _validate_return_type(str(header.get("return_type", "")))
    cameras_meta = header.get("cameras")
    if not isinstance(cameras_meta, list) or len(cameras_meta) != CAMERA_COUNT:
        raise StagedFfsProtocolError("v0.3 request header must contain exactly three cameras")
    cameras: list[StagedCameraRequest] = []
    part_idx = 1
    for camera in cameras_meta:
        if not isinstance(camera, dict):
            raise StagedFfsProtocolError("camera metadata must be an object")
        left = _decompress_lz4(parts[part_idx], metadata=camera, prefix="ir_left")
        right = _decompress_lz4(parts[part_idx + 1], metadata=camera, prefix="ir_right")
        part_idx += 2
        if left.dtype != np.uint8 or right.dtype != np.uint8:
            raise StagedFfsProtocolError("decoded IR payloads must be uint8")
        if left.shape != right.shape:
            raise StagedFfsProtocolError(f"decoded IR shape mismatch: left={left.shape} right={right.shape}")
        height = int(camera.get("height", left.shape[0]))
        width = int(camera.get("width", left.shape[1]))
        if left.shape != (height, width):
            raise StagedFfsProtocolError(f"decoded IR shape {left.shape} does not match metadata {(height, width)}")
        cameras.append(
            StagedCameraRequest(
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
    camera_order = [camera.camera_idx for camera in cameras]
    if camera_order != [0, 1, 2]:
        raise StagedFfsProtocolError(f"v0.3 camera order must be [0, 1, 2], got {camera_order}")
    return StagedFfsRequest(header=header, cameras=cameras)


def empty_metrics() -> dict[str, float]:
    return {key: 0.0 for key in METRIC_KEYS}


def _complete_reply_header(header: dict[str, Any], parts: list[bytes]) -> list[bytes]:
    for _ in range(2):
        parts[0] = _json_dumps(header)
        header["reply_kb"] = float(sum(len(part) for part in parts) / 1024.0)
    parts[0] = _json_dumps(header)
    return parts


def build_reply_parts(
    *,
    request: StagedFfsRequest | None,
    request_header: dict[str, Any] | None = None,
    depths: list[np.ndarray] | None = None,
    status: str = "ok",
    error: str = "",
    metrics: dict[str, Any] | None = None,
    compression: str = "lz4",
    return_type: str = "depth_u16",
) -> list[bytes]:
    compression = _validate_compression(compression)
    return_type = _validate_return_type(return_type)
    header_source = request.header if request is not None else (request_header or {})
    camera_source = request.cameras if request is not None else []
    depth_values = [] if depths is None else list(depths)
    if status == "ok" and len(depth_values) != CAMERA_COUNT:
        raise StagedFfsProtocolError(f"v0.3 reply expected {CAMERA_COUNT} depths, got {len(depth_values)}")
    reply_metrics = empty_metrics()
    if metrics:
        reply_metrics.update({key: float(value) for key, value in metrics.items() if key in reply_metrics})
    reply_header: dict[str, Any] = {
        "protocol": PROTOCOL_NAME,
        "version": PROTOCOL_VERSION,
        "mode": str(header_source.get("mode", "")),
        "request_id": str(header_source.get("request_id", "")),
        "kit_idx": int(header_source.get("kit_idx", -1)),
        "phase": str(header_source.get("phase", "")),
        "compression": compression,
        "return_type": return_type,
        "status": str(status),
        "error": str(error),
        "created_perf_ns": int(header_source.get("created_perf_ns", 0) or 0),
        "camera_count": CAMERA_COUNT,
        "cameras": [],
        "response_uncompressed_bytes": 0,
        "response_encoded_bytes": 0,
        **reply_metrics,
    }
    parts: list[bytes] = [b""]
    if status != "ok":
        return _complete_reply_header(reply_header, parts)
    for camera, depth in zip(camera_source, depth_values, strict=True):
        depth_u16 = np.ascontiguousarray(depth, dtype=np.uint16)
        if depth_u16.ndim != 2:
            raise StagedFfsProtocolError(f"depth_u16 must be 2D, got {depth_u16.shape}")
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
    return _complete_reply_header(reply_header, parts)


def build_error_reply_parts(
    *,
    request_id: str = "",
    kit_idx: int = -1,
    phase: str = "",
    error: str,
    created_perf_ns: int = 0,
    metrics: dict[str, Any] | None = None,
    compression: str = "lz4",
    return_type: str = "depth_u16",
) -> list[bytes]:
    return build_reply_parts(
        request=None,
        request_header={
            "request_id": str(request_id),
            "mode": "triplet-replay",
            "kit_idx": int(kit_idx),
            "phase": str(phase),
            "created_perf_ns": int(created_perf_ns),
        },
        depths=[],
        status="error",
        error=str(error),
        metrics=metrics,
        compression=compression,
        return_type=return_type,
    )


def parse_reply_parts(parts: list[bytes] | tuple[bytes, ...]) -> StagedFfsReply:
    if len(parts) < 1:
        raise StagedFfsProtocolError("reply must contain at least a header frame")
    header = _json_loads(parts[0])
    if header.get("protocol") != PROTOCOL_NAME:
        raise StagedFfsProtocolError(f"unsupported protocol={header.get('protocol')!r}")
    if int(header.get("version", -1)) != PROTOCOL_VERSION:
        raise StagedFfsProtocolError(f"unsupported version={header.get('version')!r}")
    _validate_compression(str(header.get("compression", "")))
    _validate_return_type(str(header.get("return_type", "")))
    status = str(header.get("status", ""))
    if status != "ok":
        return StagedFfsReply(header=header, depths=[])
    cameras_meta = header.get("cameras")
    if not isinstance(cameras_meta, list) or len(cameras_meta) != CAMERA_COUNT:
        raise StagedFfsProtocolError("v0.3 reply header must contain exactly three cameras")
    if len(parts) != 1 + CAMERA_COUNT:
        raise StagedFfsProtocolError(f"v0.3 reply expected 4 frames, got {len(parts)}")
    depths: list[StagedCameraDepth] = []
    for idx, camera in enumerate(cameras_meta):
        if not isinstance(camera, dict):
            raise StagedFfsProtocolError("reply camera metadata must be an object")
        depth = _decompress_lz4(parts[idx + 1], metadata=camera, prefix="depth")
        if depth.dtype != np.uint16:
            raise StagedFfsProtocolError(f"reply depth must be uint16, got {depth.dtype}")
        depths.append(
            StagedCameraDepth(
                camera_idx=int(camera["camera_idx"]),
                serial=str(camera.get("serial", "")),
                depth_u16=np.ascontiguousarray(depth, dtype=np.uint16),
            )
        )
    return StagedFfsReply(header=header, depths=depths)
