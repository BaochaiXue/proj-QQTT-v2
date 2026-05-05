from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np

from .protocol import build_depth_request_parts, parse_depth_response_parts


@dataclass(frozen=True)
class FfsRemoteDepthResult:
    frame_id: int
    depth_color_m: np.ndarray
    rtt_ms: float
    server_ffs_ms: float
    server_align_ms: float
    server_total_ms: float
    request_bytes: int
    response_bytes: int


class FfsRemoteDepthClient:
    def __init__(
        self,
        *,
        endpoint: str,
        timeout_ms: int,
        return_type: str = "depth_u16",
        max_inflight: int = 1,
        zmq_context: Any | None = None,
        zmq_socket: Any | None = None,
    ) -> None:
        if not endpoint:
            raise ValueError("endpoint must be non-empty")
        if int(timeout_ms) <= 0:
            raise ValueError("timeout_ms must be positive")
        if int(max_inflight) != 1:
            raise ValueError("first ffs_remote implementation only supports max_inflight=1")
        if return_type not in {"depth_u16", "depth_float_m"}:
            raise ValueError("return_type must be depth_u16 or depth_float_m")
        self.endpoint = str(endpoint)
        self.timeout_ms = int(timeout_ms)
        self.return_type = str(return_type)
        self.max_inflight = int(max_inflight)
        self._external_socket = zmq_socket is not None
        self._context = zmq_context
        self._socket = zmq_socket

    def close(self) -> None:
        if self._socket is not None and not self._external_socket:
            try:
                self._socket.close(linger=0)
            except Exception:
                pass
        self._socket = None

    def _connect(self) -> Any:
        if self._socket is not None:
            return self._socket
        import zmq

        if self._context is None:
            self._context = zmq.Context.instance()
        socket = self._context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        socket.connect(self.endpoint)
        self._socket = socket
        return socket

    def _reset_socket_after_error(self) -> None:
        if self._external_socket:
            return
        self.close()

    def request_depth_color_m(
        self,
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
    ) -> FfsRemoteDepthResult:
        request_parts = build_depth_request_parts(
            frame_id=int(frame_id),
            ir_left_u8=ir_left_u8,
            ir_right_u8=ir_right_u8,
            color_shape=color_shape,
            k_ir_left=k_ir_left,
            k_color=k_color,
            t_ir_left_to_color=t_ir_left_to_color,
            baseline_m=float(baseline_m),
            depth_scale_m_per_unit=float(depth_scale_m_per_unit),
            return_type=self.return_type,
        )
        request_bytes = sum(len(part) for part in request_parts)
        socket = self._connect()
        start_s = time.perf_counter()
        try:
            socket.send_multipart(request_parts)
            response_parts = socket.recv_multipart()
        except Exception:
            self._reset_socket_after_error()
            raise
        rtt_ms = (time.perf_counter() - start_s) * 1000.0
        response = parse_depth_response_parts(response_parts)
        metadata = response.metadata
        status = str(metadata.get("status", ""))
        if status != "ok":
            raise RuntimeError(f"remote FFS server returned status={status!r}: {metadata.get('error', '')}")
        response_frame_id = int(metadata.get("frame_id", -1))
        if response_frame_id != int(frame_id):
            raise RuntimeError(f"remote FFS frame_id mismatch: request={frame_id} response={response_frame_id}")
        depth = response.depth
        if depth.dtype == np.uint16:
            scale = float(metadata.get("depth_scale_m_per_unit", depth_scale_m_per_unit))
            depth_m = np.ascontiguousarray(depth.astype(np.float32) * np.float32(scale), dtype=np.float32)
        else:
            depth_m = np.ascontiguousarray(depth.astype(np.float32), dtype=np.float32)
        return FfsRemoteDepthResult(
            frame_id=response_frame_id,
            depth_color_m=depth_m,
            rtt_ms=float(rtt_ms),
            server_ffs_ms=float(metadata.get("server_ffs_ms", 0.0)),
            server_align_ms=float(metadata.get("server_align_ms", 0.0)),
            server_total_ms=float(metadata.get("server_total_ms", 0.0)),
            request_bytes=int(request_bytes),
            response_bytes=int(sum(len(part) for part in response_parts)),
        )
