from __future__ import annotations

from dataclasses import dataclass
import io
import json
from typing import Any

import numpy as np


PROTOCOL_NAME = "qqtt_demo32_shape_prior_remote"
PROTOCOL_VERSION = 1


class ShapePriorProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class ShapePriorRequest:
    metadata: dict[str, Any]
    rgb_u8: np.ndarray
    object_mask: np.ndarray
    controller_mask: np.ndarray
    depth_color_m: np.ndarray
    k_color: np.ndarray
    camera_to_world_c2w: np.ndarray


@dataclass(frozen=True)
class ShapePriorResponse:
    metadata: dict[str, Any]
    points_m: np.ndarray
    colors_rgb_u8: np.ndarray
    surface_points_m: np.ndarray
    interior_points_m: np.ndarray


def _json_dumps(data: dict[str, Any]) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _json_loads(data: bytes) -> dict[str, Any]:
    value = json.loads(data.decode("utf-8"))
    if not isinstance(value, dict):
        raise ShapePriorProtocolError("metadata JSON must decode to an object")
    return value


def _array_to_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(array), allow_pickle=False)
    return buffer.getvalue()


def _array_from_bytes(payload: bytes) -> np.ndarray:
    with io.BytesIO(payload) as buffer:
        return np.load(buffer, allow_pickle=False)


def _require_protocol(metadata: dict[str, Any]) -> None:
    if str(metadata.get("protocol")) != PROTOCOL_NAME:
        raise ShapePriorProtocolError(f"unexpected protocol={metadata.get('protocol')!r}")
    if int(metadata.get("version", -1)) != PROTOCOL_VERSION:
        raise ShapePriorProtocolError(f"unsupported version={metadata.get('version')!r}")


def build_shape_prior_request_parts(*, snapshot: Any, request_id: str) -> list[bytes]:
    rgb = np.ascontiguousarray(snapshot.rgb_u8, dtype=np.uint8)
    object_mask = np.ascontiguousarray(snapshot.object_mask, dtype=bool)
    controller_mask = np.ascontiguousarray(snapshot.controller_mask, dtype=bool)
    depth = np.ascontiguousarray(snapshot.depth_color_m, dtype=np.float32)
    k_color = np.ascontiguousarray(snapshot.k_color, dtype=np.float32).reshape(3, 3)
    c2w = np.ascontiguousarray(snapshot.camera_to_world_c2w, dtype=np.float32).reshape(4, 4)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ShapePriorProtocolError(f"rgb_u8 must be HxWx3, got {rgb.shape}")
    if object_mask.shape != rgb.shape[:2] or controller_mask.shape != rgb.shape[:2] or depth.shape != rgb.shape[:2]:
        raise ShapePriorProtocolError("mask/depth shapes must match RGB height and width")
    metadata = {
        "protocol": PROTOCOL_NAME,
        "version": PROTOCOL_VERSION,
        "request_id": str(request_id),
        "seq": int(snapshot.seq),
        "source_timestamp_s": snapshot.source_timestamp_s,
        "input_source": str(snapshot.input_source),
        "depth_backend": str(snapshot.depth_backend),
        "depth_source_internal": str(snapshot.depth_source_internal),
        "table_z_m": float(getattr(snapshot, "table_z_m", 0.0)),
        "table_z_above_direction": str(getattr(snapshot, "table_z_above_direction", "negative")),
        "shape_backend": "sam3d-objects",
        "rgb_shape": [int(v) for v in rgb.shape],
        "depth_shape": [int(v) for v in depth.shape],
        "object_mask_pixels": int(np.count_nonzero(object_mask)),
        "controller_mask_pixels": int(np.count_nonzero(controller_mask)),
    }
    return [
        _json_dumps(metadata),
        _array_to_bytes(rgb),
        _array_to_bytes(object_mask),
        _array_to_bytes(controller_mask),
        _array_to_bytes(depth),
        _array_to_bytes(k_color),
        _array_to_bytes(c2w),
    ]


def parse_shape_prior_request_parts(parts: list[bytes]) -> ShapePriorRequest:
    if len(parts) != 7:
        raise ShapePriorProtocolError(f"shape-prior request expected 7 frames, got {len(parts)}")
    metadata = _json_loads(parts[0])
    _require_protocol(metadata)
    rgb = np.ascontiguousarray(_array_from_bytes(parts[1]), dtype=np.uint8)
    object_mask = np.ascontiguousarray(_array_from_bytes(parts[2]), dtype=bool)
    controller_mask = np.ascontiguousarray(_array_from_bytes(parts[3]), dtype=bool)
    depth = np.ascontiguousarray(_array_from_bytes(parts[4]), dtype=np.float32)
    k_color = np.ascontiguousarray(_array_from_bytes(parts[5]), dtype=np.float32).reshape(3, 3)
    c2w = np.ascontiguousarray(_array_from_bytes(parts[6]), dtype=np.float32).reshape(4, 4)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ShapePriorProtocolError(f"rgb payload must be HxWx3, got {rgb.shape}")
    if object_mask.shape != rgb.shape[:2] or controller_mask.shape != rgb.shape[:2] or depth.shape != rgb.shape[:2]:
        raise ShapePriorProtocolError("request mask/depth payload shapes must match RGB")
    return ShapePriorRequest(
        metadata=metadata,
        rgb_u8=rgb,
        object_mask=object_mask,
        controller_mask=controller_mask,
        depth_color_m=depth,
        k_color=k_color,
        camera_to_world_c2w=c2w,
    )


def build_shape_prior_response_parts(
    *,
    request_id: str,
    seq: int,
    status: str,
    points_m: np.ndarray,
    colors_rgb_u8: np.ndarray,
    surface_points_m: np.ndarray | None = None,
    interior_points_m: np.ndarray | None = None,
    metadata: dict[str, Any] | None = None,
) -> list[bytes]:
    points = np.asarray(points_m, dtype=np.float32).reshape(-1, 3)
    colors = np.asarray(colors_rgb_u8, dtype=np.uint8).reshape(-1, 3)
    if len(points) != len(colors):
        raise ShapePriorProtocolError("points_m and colors_rgb_u8 must have the same length")
    surface = (
        np.empty((0, 3), dtype=np.float32)
        if surface_points_m is None
        else np.asarray(surface_points_m, dtype=np.float32).reshape(-1, 3)
    )
    interior = (
        np.empty((0, 3), dtype=np.float32)
        if interior_points_m is None
        else np.asarray(interior_points_m, dtype=np.float32).reshape(-1, 3)
    )
    payload = dict(metadata or {})
    payload.update(
        {
            "protocol": PROTOCOL_NAME,
            "version": PROTOCOL_VERSION,
            "request_id": str(request_id),
            "seq": int(seq),
            "status": str(status),
            "point_count": int(len(points)),
            "surface_point_count": int(len(surface)),
            "interior_point_count": int(len(interior)),
        }
    )
    return [
        _json_dumps(payload),
        _array_to_bytes(points),
        _array_to_bytes(colors),
        _array_to_bytes(surface),
        _array_to_bytes(interior),
    ]


def build_error_response_parts(*, request_id: str, seq: int, error: str) -> list[bytes]:
    return [
        _json_dumps(
            {
                "protocol": PROTOCOL_NAME,
                "version": PROTOCOL_VERSION,
                "request_id": str(request_id),
                "seq": int(seq),
                "status": "error",
                "error": str(error),
                "point_count": 0,
            }
        )
    ]


def parse_shape_prior_response_parts(parts: list[bytes]) -> ShapePriorResponse:
    if len(parts) not in {1, 3, 5}:
        raise ShapePriorProtocolError(f"shape-prior response expected 1, 3, or 5 frames, got {len(parts)}")
    metadata = _json_loads(parts[0])
    _require_protocol(metadata)
    if len(parts) == 1:
        return ShapePriorResponse(
            metadata=metadata,
            points_m=np.empty((0, 3), dtype=np.float32),
            colors_rgb_u8=np.empty((0, 3), dtype=np.uint8),
            surface_points_m=np.empty((0, 3), dtype=np.float32),
            interior_points_m=np.empty((0, 3), dtype=np.float32),
        )
    points = np.ascontiguousarray(_array_from_bytes(parts[1]), dtype=np.float32).reshape(-1, 3)
    colors = np.ascontiguousarray(_array_from_bytes(parts[2]), dtype=np.uint8).reshape(-1, 3)
    if len(points) != len(colors):
        raise ShapePriorProtocolError("response points and colors have different lengths")
    if len(parts) == 5:
        surface = np.ascontiguousarray(_array_from_bytes(parts[3]), dtype=np.float32).reshape(-1, 3)
        interior = np.ascontiguousarray(_array_from_bytes(parts[4]), dtype=np.float32).reshape(-1, 3)
    else:
        surface = np.empty((0, 3), dtype=np.float32)
        interior = np.empty((0, 3), dtype=np.float32)
    return ShapePriorResponse(
        metadata=metadata,
        points_m=points,
        colors_rgb_u8=colors,
        surface_points_m=surface,
        interior_points_m=interior,
    )
