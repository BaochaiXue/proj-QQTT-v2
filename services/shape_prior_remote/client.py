from __future__ import annotations

import time
from typing import Any
from uuid import uuid4

import numpy as np

from qqtt.demo.shape_prior_warmup import (
    SHAPE_PRIOR_STATUS_FAILED,
    SHAPE_PRIOR_STATUS_READY,
    ShapePriorResult,
    ShapePriorSnapshot,
)

from .protocol import (
    build_shape_prior_request_parts,
    parse_shape_prior_response_parts,
)


class ShapePriorRemoteClient:
    def __init__(
        self,
        *,
        endpoint: str,
        timeout_ms: int = 5000,
        zmq_context: Any | None = None,
        zmq_socket: Any | None = None,
    ) -> None:
        if not endpoint:
            raise ValueError("endpoint must be non-empty")
        if int(timeout_ms) <= 0:
            raise ValueError("timeout_ms must be positive")
        self.endpoint = str(endpoint)
        self.timeout_ms = int(timeout_ms)
        self._context = zmq_context
        self._socket = zmq_socket
        self._external_socket = zmq_socket is not None

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

    def request_shape_prior(self, snapshot: ShapePriorSnapshot) -> ShapePriorResult:
        request_id = f"shape-prior-{int(snapshot.seq)}-{uuid4().hex[:8]}"
        request_parts = build_shape_prior_request_parts(snapshot=snapshot, request_id=request_id)
        request_bytes = int(sum(len(part) for part in request_parts))
        socket = self._connect()
        start_s = time.perf_counter()
        try:
            socket.send_multipart(request_parts)
            response_parts = socket.recv_multipart()
        except Exception:
            self._reset_socket_after_error()
            raise
        rtt_ms = (time.perf_counter() - start_s) * 1000.0
        response = parse_shape_prior_response_parts(response_parts)
        metadata = dict(response.metadata)
        status = str(metadata.get("status", "error"))
        if str(metadata.get("request_id")) != request_id:
            raise RuntimeError(
                f"shape-prior request_id mismatch: request={request_id} response={metadata.get('request_id')}"
            )
        metadata.setdefault("request_upload_ms", 0.0)
        metadata.setdefault("response_download_ms", float(rtt_ms))
        metadata.setdefault("request_bytes", request_bytes)
        metadata.setdefault("response_bytes", int(sum(len(part) for part in response_parts)))
        if status != SHAPE_PRIOR_STATUS_READY:
            return ShapePriorResult(
                seq=int(metadata.get("seq", snapshot.seq)),
                source_seq=int(snapshot.seq),
                source_timestamp_s=snapshot.source_timestamp_s,
                status=SHAPE_PRIOR_STATUS_FAILED,
                points_m=np.empty((0, 3), dtype=np.float32),
                colors_rgb_u8=np.empty((0, 3), dtype=np.uint8),
                surface_points_m=np.empty((0, 3), dtype=np.float32),
                interior_points_m=np.empty((0, 3), dtype=np.float32),
                metadata=metadata,
                error=str(metadata.get("error", f"shape-prior worker returned status={status!r}")),
            )
        return ShapePriorResult(
            seq=int(metadata.get("seq", snapshot.seq)),
            source_seq=int(snapshot.seq),
            source_timestamp_s=snapshot.source_timestamp_s,
            status=SHAPE_PRIOR_STATUS_READY,
            points_m=np.ascontiguousarray(response.points_m, dtype=np.float32),
            colors_rgb_u8=np.ascontiguousarray(response.colors_rgb_u8, dtype=np.uint8),
            surface_points_m=np.ascontiguousarray(response.surface_points_m, dtype=np.float32),
            interior_points_m=np.ascontiguousarray(response.interior_points_m, dtype=np.float32),
            metadata=metadata,
        )
