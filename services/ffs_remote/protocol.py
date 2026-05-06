from __future__ import annotations

from dataclasses import dataclass
import io
import json
from typing import Any

import numpy as np


PROTOCOL_VERSION = 1
RETURN_TYPES = ("depth_u16", "depth_float_m", "masked_uv_depth", "masked_xyz")
FULL_DEPTH_RETURN_TYPES = ("depth_u16", "depth_float_m")
SPARSE_RETURN_TYPES = ("masked_uv_depth", "masked_xyz")
COMPRESSION_MODES = ("none", "zstd", "lz4", "png")


class FfsRemoteProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class FfsDepthRequest:
    metadata: dict[str, Any]
    ir_left_u8: np.ndarray
    ir_right_u8: np.ndarray
    mask_u8: np.ndarray | None = None


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


def _validate_compression(compression: str) -> str:
    value = str(compression)
    if value not in COMPRESSION_MODES:
        raise FfsRemoteProtocolError(f"unsupported compression: {value}")
    return value


def _encode_array_payload(array: np.ndarray, *, compression: str) -> tuple[bytes, dict[str, Any]]:
    codec = _validate_compression(compression)
    payload_array = np.ascontiguousarray(array)
    raw = payload_array.tobytes(order="C")
    if codec == "none":
        encoded = raw
    elif codec == "zstd":
        try:
            import zstandard as zstd  # type: ignore[import-not-found]
        except Exception as exc:
            raise FfsRemoteProtocolError("zstd compression requires the zstandard package") from exc
        encoded = zstd.ZstdCompressor(level=1).compress(raw)
    elif codec == "lz4":
        try:
            import lz4.frame  # type: ignore[import-not-found]
        except Exception as exc:
            raise FfsRemoteProtocolError("lz4 compression requires the lz4 package") from exc
        encoded = lz4.frame.compress(raw, compression_level=0)
    elif codec == "png":
        if payload_array.ndim != 2 or payload_array.dtype not in (np.dtype("uint8"), np.dtype("uint16")):
            raise FfsRemoteProtocolError("png compression supports only 2D uint8/uint16 arrays")
        try:
            from PIL import Image
        except Exception as exc:
            raise FfsRemoteProtocolError("png compression requires Pillow") from exc
        buffer = io.BytesIO()
        Image.fromarray(payload_array).save(buffer, format="PNG", optimize=False)
        encoded = buffer.getvalue()
    else:  # pragma: no cover - guarded above
        raise FfsRemoteProtocolError(f"unsupported compression: {codec}")
    return encoded, {
        "compression": codec,
        "dtype": str(payload_array.dtype),
        "shape": [int(item) for item in payload_array.shape],
        "uncompressed_bytes": int(len(raw)),
        "encoded_bytes": int(len(encoded)),
    }


def _decode_array_payload(payload: bytes, *, metadata: dict[str, Any], prefix: str) -> np.ndarray:
    codec = str(metadata.get(f"{prefix}_compression", metadata.get("compression", "none")))
    _validate_compression(codec)
    dtype = np.dtype(str(metadata.get(f"{prefix}_dtype", metadata.get("ir_dtype", ""))))
    shape_value = metadata.get(f"{prefix}_shape")
    if not isinstance(shape_value, list | tuple) or not shape_value:
        raise FfsRemoteProtocolError(f"invalid {prefix}_shape={shape_value!r}")
    shape = tuple(int(item) for item in shape_value)
    expected_items = int(np.prod(shape))
    if codec == "none":
        raw = payload
    elif codec == "zstd":
        try:
            import zstandard as zstd  # type: ignore[import-not-found]
        except Exception as exc:
            raise FfsRemoteProtocolError("zstd compression requires the zstandard package") from exc
        raw = zstd.ZstdDecompressor().decompress(payload)
    elif codec == "lz4":
        try:
            import lz4.frame  # type: ignore[import-not-found]
        except Exception as exc:
            raise FfsRemoteProtocolError("lz4 compression requires the lz4 package") from exc
        raw = lz4.frame.decompress(payload)
    elif codec == "png":
        try:
            from PIL import Image
        except Exception as exc:
            raise FfsRemoteProtocolError("png compression requires Pillow") from exc
        array = np.asarray(Image.open(io.BytesIO(payload)))
        if array.shape != shape or np.dtype(array.dtype) != dtype:
            raise FfsRemoteProtocolError(
                f"png payload mismatch for {prefix}: expected shape={shape} dtype={dtype}, "
                f"got shape={array.shape} dtype={array.dtype}"
            )
        return np.ascontiguousarray(array, dtype=dtype)
    else:  # pragma: no cover - guarded above
        raise FfsRemoteProtocolError(f"unsupported compression: {codec}")
    expected_bytes = expected_items * int(dtype.itemsize)
    if len(raw) != expected_bytes:
        raise FfsRemoteProtocolError(f"{prefix} byte payload mismatch: expected={expected_bytes} got={len(raw)}")
    return np.ascontiguousarray(np.frombuffer(raw, dtype=dtype).reshape(shape))


def _add_payload_metadata(metadata: dict[str, Any], prefix: str, payload_meta: dict[str, Any]) -> None:
    metadata[f"{prefix}_compression"] = payload_meta["compression"]
    metadata[f"{prefix}_dtype"] = payload_meta["dtype"]
    metadata[f"{prefix}_shape"] = payload_meta["shape"]
    metadata[f"{prefix}_uncompressed_bytes"] = payload_meta["uncompressed_bytes"]
    metadata[f"{prefix}_encoded_bytes"] = payload_meta["encoded_bytes"]


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
    mask_u8: np.ndarray | None = None,
    compression: str = "none",
) -> list[bytes]:
    if return_type not in RETURN_TYPES:
        raise FfsRemoteProtocolError(f"unsupported return_type: {return_type}")
    left = np.ascontiguousarray(ir_left_u8, dtype=np.uint8)
    right = np.ascontiguousarray(ir_right_u8, dtype=np.uint8)
    if left.ndim != 2 or right.ndim != 2:
        raise FfsRemoteProtocolError("IR images must be 2D uint8 arrays")
    if left.shape != right.shape:
        raise FfsRemoteProtocolError(f"IR image shapes differ: left={left.shape} right={right.shape}")
    mask = None
    if mask_u8 is not None:
        mask = np.ascontiguousarray(mask_u8, dtype=np.uint8)
        if mask.ndim != 2:
            raise FfsRemoteProtocolError("mask_u8 must be a 2D uint8 array")
        if mask.shape != tuple(int(item) for item in color_shape):
            raise FfsRemoteProtocolError(f"mask_u8 shape {mask.shape} does not match color_shape {color_shape}")
    color_h, color_w = [int(item) for item in color_shape]
    if color_h <= 0 or color_w <= 0:
        raise FfsRemoteProtocolError(f"invalid color_shape: {color_shape!r}")
    left_payload, left_meta = _encode_array_payload(left, compression=compression)
    right_payload, right_meta = _encode_array_payload(right, compression=compression)
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
        "compression": _validate_compression(compression),
        "has_mask": bool(mask is not None),
    }
    _add_payload_metadata(metadata, "ir_left", left_meta)
    _add_payload_metadata(metadata, "ir_right", right_meta)
    parts = [_json_dumps(metadata), left_payload, right_payload]
    if mask is not None:
        mask_payload, mask_meta = _encode_array_payload(mask, compression=compression)
        _add_payload_metadata(metadata, "mask", mask_meta)
        parts[0] = _json_dumps(metadata)
        parts.append(mask_payload)
    return parts


def parse_depth_request_parts(parts: list[bytes] | tuple[bytes, ...]) -> FfsDepthRequest:
    if len(parts) not in {3, 4}:
        raise FfsRemoteProtocolError(f"request expected 3 or 4 parts, got {len(parts)}")
    metadata = _json_loads(parts[0])
    if int(metadata.get("protocol_version", -1)) != PROTOCOL_VERSION:
        raise FfsRemoteProtocolError(f"unsupported protocol_version={metadata.get('protocol_version')!r}")
    if metadata.get("return_type") not in RETURN_TYPES:
        raise FfsRemoteProtocolError(f"unsupported return_type={metadata.get('return_type')!r}")
    shape_value = metadata.get("ir_shape")
    if not isinstance(shape_value, list | tuple) or len(shape_value) != 2:
        raise FfsRemoteProtocolError(f"invalid ir_shape={shape_value!r}")
    ir_shape = (int(shape_value[0]), int(shape_value[1]))
    if "ir_left_shape" not in metadata:
        expected_bytes = int(np.prod(ir_shape))
        if len(parts[1]) != expected_bytes or len(parts[2]) != expected_bytes:
            raise FfsRemoteProtocolError(
                f"IR byte payload mismatch: expected={expected_bytes} left={len(parts[1])} right={len(parts[2])}"
            )
        left = np.frombuffer(parts[1], dtype=np.uint8).reshape(ir_shape)
        right = np.frombuffer(parts[2], dtype=np.uint8).reshape(ir_shape)
    else:
        left = _decode_array_payload(parts[1], metadata=metadata, prefix="ir_left")
        right = _decode_array_payload(parts[2], metadata=metadata, prefix="ir_right")
    if left.shape != ir_shape or right.shape != ir_shape:
        raise FfsRemoteProtocolError(f"decoded IR shape mismatch: left={left.shape} right={right.shape} expected={ir_shape}")
    mask = None
    if bool(metadata.get("has_mask", False)):
        if len(parts) != 4:
            raise FfsRemoteProtocolError("metadata has_mask=true but request has no mask payload")
        mask = _decode_array_payload(parts[3], metadata=metadata, prefix="mask")
    return FfsDepthRequest(
        metadata=metadata,
        ir_left_u8=np.ascontiguousarray(left),
        ir_right_u8=np.ascontiguousarray(right),
        mask_u8=None if mask is None else np.ascontiguousarray(mask, dtype=np.uint8),
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
    return_type: str | None = None,
    compression: str = "none",
    extra_metadata: dict[str, Any] | None = None,
) -> list[bytes]:
    dtype = np.dtype(depth_dtype)
    depth_array = np.ascontiguousarray(depth, dtype=dtype)
    if depth_array.ndim != 2:
        raise FfsRemoteProtocolError(f"depth response must be 2D, got {depth_array.shape}")
    depth_payload, depth_meta = _encode_array_payload(depth_array, compression=compression)
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
        "return_type": "" if return_type is None else str(return_type),
        "compression": _validate_compression(compression),
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    _add_payload_metadata(metadata, "depth", depth_meta)
    return [_json_dumps(metadata), depth_payload]


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
    if "depth_compression" not in metadata:
        dtype = np.dtype(str(metadata.get("depth_dtype", "")))
        expected_bytes = int(np.prod(depth_shape)) * int(dtype.itemsize)
        if len(parts[1]) != expected_bytes:
            raise FfsRemoteProtocolError(f"depth byte payload mismatch: expected={expected_bytes} got={len(parts[1])}")
        depth = np.frombuffer(parts[1], dtype=dtype).reshape(depth_shape)
    else:
        depth = _decode_array_payload(parts[1], metadata=metadata, prefix="depth")
    return FfsDepthResponse(metadata=metadata, depth=np.ascontiguousarray(depth))
