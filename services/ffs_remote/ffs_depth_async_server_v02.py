#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import queue
import sys
import threading
import time
from typing import Any

import numpy as np


def _resolve_repo_root() -> Path:
    candidates = [Path(__file__).resolve().parents[2], Path.cwd()]
    env_root = os.environ.get("QQTT_REPO_ROOT")
    if env_root:
        candidates.insert(0, Path(env_root))
    for candidate in candidates:
        root = candidate.expanduser().resolve()
        if (root / "data_process").is_dir() and (root / "services").is_dir():
            return root
    return Path(__file__).resolve().parents[2]


REPO_ROOT = _resolve_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_process.depth_backends.ffs_defaults import (  # noqa: E402
    DEFAULT_FFS_REPO,
    DEFAULT_FFS_MODEL_NAME,
    DEFAULT_FFS_MAX_DISP,
    DEFAULT_FFS_TRT_BUILDER_OPTIMIZATION_LEVEL,
    DEFAULT_FFS_TRT_ENGINE_SIZE,
    DEFAULT_FFS_TRT_INPUT_SIZE,
    DEFAULT_FFS_TRT_TWO_STAGE_MODEL_DIR,
    DEFAULT_FFS_VALID_ITERS,
)
from data_process.depth_backends.geometry import quantize_depth_with_invalid_zero  # noqa: E402
from qqtt.demo.realtime_single_camera_pointcloud import warm_up_numba_ffs_align  # noqa: E402
from services.ffs_remote.async_protocol_v02 import (  # noqa: E402
    AsyncFfsProtocolError,
    AsyncFfsRequest,
    PROTOCOL_NAME,
    build_error_reply_parts,
    build_reply_parts,
    parse_request_parts,
)
from services.ffs_remote.ffs_depth_server import (  # noqa: E402
    _CachedAligner,
    _engine_contract_metadata,
    _validate_engine_contract,
)


def _elapsed_ms(start_s: float, end_s: float | None = None) -> float:
    stop_s = time.perf_counter() if end_s is None else end_s
    return (stop_s - start_s) * 1000.0


@dataclass(frozen=True)
class _QueuedRequest:
    identity: bytes
    parts: list[bytes]
    received_s: float


@dataclass(frozen=True)
class _QueuedResponse:
    identity: bytes
    parts: list[bytes]
    request_id: str
    status: str
    elapsed_ms: float


@dataclass(frozen=True)
class _DecodedRequest:
    identity: bytes
    request: AsyncFfsRequest
    received_s: float
    decode_started_s: float
    decode_done_s: float


@dataclass(frozen=True)
class _InferenceResult:
    identity: bytes
    request: AsyncFfsRequest
    depths: list[np.ndarray]
    per_camera_stats: list[dict[str, Any]]
    received_s: float
    decode_started_s: float
    decode_done_s: float
    inference_started_s: float
    inference_done_s: float


def _patch_reply_header(
    parts: list[bytes],
    *,
    server_total_ms: float,
    server_stage_ms: dict[str, Any],
) -> list[bytes]:
    if not parts:
        return parts
    header = json.loads(parts[0].decode("utf-8"))
    if not isinstance(header, dict):
        return parts
    header["server_total_ms"] = float(server_total_ms)
    header["server_stage_ms"] = dict(server_stage_ms)
    parts[0] = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return parts


def _run_ffs_request(
    *,
    request: AsyncFfsRequest,
    runner: Any,
    aligner: _CachedAligner,
    warmup_remaining: int,
    args: argparse.Namespace,
    worker_idx: int,
) -> tuple[list[np.ndarray], list[dict[str, Any]], int]:
    depths: list[np.ndarray] = []
    stats: list[dict[str, Any]] = []
    for camera in request.cameras:
        if warmup_remaining > 0:
            warmup_start_s = time.perf_counter()
            for _ in range(warmup_remaining):
                runner.run_pair(
                    camera.ir_left_u8,
                    camera.ir_right_u8,
                    K_ir_left=camera.k_ir_left,
                    baseline_m=float(camera.baseline_m),
                )
            if bool(args.debug):
                print(
                    "[demo-v0.2-async-server] "
                    f"worker={worker_idx} lazy_warmup count={warmup_remaining} "
                    f"elapsed_ms={_elapsed_ms(warmup_start_s):.2f}",
                    flush=True,
                )
            warmup_remaining = 0
        ffs_start_s = time.perf_counter()
        output = runner.run_pair(
            camera.ir_left_u8,
            camera.ir_right_u8,
            K_ir_left=camera.k_ir_left,
            baseline_m=float(camera.baseline_m),
        )
        ffs_done_s = time.perf_counter()
        depth_ir_left_m = np.asarray(output["depth_ir_left_m"], dtype=np.float32)
        k_ir_left_used = np.asarray(output.get("K_ir_left_used", camera.k_ir_left), dtype=np.float32)
        align_start_s = time.perf_counter()
        depth_color_m = aligner.align(
            depth_ir_left_m=depth_ir_left_m,
            color_shape=(int(camera.height), int(camera.width)),
            k_ir_left=k_ir_left_used,
            t_ir_left_to_color=camera.t_ir_left_to_color,
            k_color=camera.k_color,
        )
        align_done_s = time.perf_counter()
        depth_u16 = quantize_depth_with_invalid_zero(
            depth_color_m,
            float(args.depth_scale_m_per_unit),
        )
        depths.append(depth_u16)
        stats.append(
            {
                "camera_idx": int(camera.camera_idx),
                "serial": str(camera.serial),
                "server_ffs_ms": _elapsed_ms(ffs_start_s, ffs_done_s),
                "server_align_ms": _elapsed_ms(align_start_s, align_done_s),
                "depth_nonzero": int(np.count_nonzero(depth_u16)),
            }
        )
    return depths, stats, warmup_remaining


class _AsyncFfsWorker(threading.Thread):
    def __init__(
        self,
        *,
        worker_idx: int,
        request_queue: "queue.Queue[_QueuedRequest | None]",
        response_queue: "queue.Queue[_QueuedResponse]",
        args: argparse.Namespace,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(name=f"demo-v0.2-async-ffs-worker-{worker_idx}", daemon=True)
        self.worker_idx = int(worker_idx)
        self._request_queue = request_queue
        self._response_queue = response_queue
        self._args = args
        self._stop_event = stop_event

    def run(self) -> None:
        from data_process.depth_backends import FastFoundationStereoTensorRTRunner

        runner = FastFoundationStereoTensorRTRunner(
            ffs_repo=Path(self._args.ffs_repo),
            model_dir=Path(self._args.ffs_trt_model_dir),
            trt_root=None if self._args.ffs_trt_root is None else Path(self._args.ffs_trt_root),
        )
        aligner = _CachedAligner()
        warmup_remaining = int(self._args.warmup)
        while not self._stop_event.is_set():
            item = self._request_queue.get()
            if item is None:
                return
            decode_started_s = time.perf_counter()
            request_id = ""
            status = "ok"
            try:
                request = parse_request_parts(item.parts)
                decode_done_s = time.perf_counter()
                request_id = str(request.header.get("request_id", ""))
                inference_started_s = time.perf_counter()
                depths, per_camera_stats, warmup_remaining = _run_ffs_request(
                    request=request,
                    runner=runner,
                    aligner=aligner,
                    warmup_remaining=warmup_remaining,
                    args=self._args,
                    worker_idx=self.worker_idx,
                )
                inference_done_s = time.perf_counter()
                encode_started_s = time.perf_counter()
                parts = build_reply_parts(
                    request=request,
                    depths=depths,
                    status="ok",
                    per_camera_stats=per_camera_stats,
                    server_total_ms=0.0,
                    compression=str(self._args.compress),
                    return_type=str(self._args.return_type),
                )
                encode_done_s = time.perf_counter()
                server_stage_ms = {
                    "pipeline_mode": "fused-worker",
                    "router_queue_ms": _elapsed_ms(item.received_s, decode_started_s),
                    "decode_ms": _elapsed_ms(decode_started_s, decode_done_s),
                    "ffs_queue_ms": 0.0,
                    "ffs_stage_ms": _elapsed_ms(inference_started_s, inference_done_s),
                    "encode_queue_ms": 0.0,
                    "encode_ms": _elapsed_ms(encode_started_s, encode_done_s),
                }
                parts = _patch_reply_header(
                    parts,
                    server_total_ms=_elapsed_ms(item.received_s, encode_done_s),
                    server_stage_ms=server_stage_ms,
                )
            except Exception as exc:
                error_done_s = time.perf_counter()
                status = "error"
                error = f"{type(exc).__name__}: {exc}"
                request_id, mode, created_perf_ns = _best_effort_request_identity(item.parts)
                server_stage_ms = {
                    "pipeline_mode": "fused-worker",
                    "router_queue_ms": _elapsed_ms(item.received_s, decode_started_s),
                    "decode_ms": _elapsed_ms(decode_started_s, error_done_s),
                    "ffs_queue_ms": 0.0,
                    "ffs_stage_ms": 0.0,
                    "encode_queue_ms": 0.0,
                    "encode_ms": 0.0,
                }
                parts = build_error_reply_parts(
                    request_id=request_id,
                    mode=mode,
                    created_perf_ns=created_perf_ns,
                    error=error,
                    server_total_ms=_elapsed_ms(item.received_s, error_done_s),
                    server_stage_ms=server_stage_ms,
                    compression=str(self._args.compress),
                    return_type=str(self._args.return_type),
                )
                if bool(self._args.debug):
                    print(
                        "[demo-v0.2-async-server] "
                        f"worker={self.worker_idx} request_id={request_id} status=error error={error}",
                        flush=True,
                    )
            elapsed_ms = _elapsed_ms(item.received_s)
            self._response_queue.put(
                _QueuedResponse(
                    identity=item.identity,
                    parts=parts,
                    request_id=request_id,
                    status=status,
                    elapsed_ms=elapsed_ms,
                )
            )

    def _run_request(
        self,
        *,
        request: AsyncFfsRequest,
        runner: Any,
        aligner: _CachedAligner,
        warmup_remaining: int,
    ) -> tuple[list[np.ndarray], list[dict[str, Any]], int]:
        return _run_ffs_request(
            request=request,
            runner=runner,
            aligner=aligner,
            warmup_remaining=warmup_remaining,
            args=self._args,
            worker_idx=self.worker_idx,
        )


class _DecodeWorker(threading.Thread):
    def __init__(
        self,
        *,
        worker_idx: int,
        request_queue: "queue.Queue[_QueuedRequest | None]",
        decoded_queue: "queue.Queue[_DecodedRequest]",
        response_queue: "queue.Queue[_QueuedResponse]",
        args: argparse.Namespace,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(name=f"demo-v0.2-decode-worker-{worker_idx}", daemon=True)
        self.worker_idx = int(worker_idx)
        self._request_queue = request_queue
        self._decoded_queue = decoded_queue
        self._response_queue = response_queue
        self._args = args
        self._stop_event = stop_event

    def run(self) -> None:
        while not self._stop_event.is_set():
            item = self._request_queue.get()
            if item is None:
                return
            decode_started_s = time.perf_counter()
            try:
                request = parse_request_parts(item.parts)
                decode_done_s = time.perf_counter()
                self._decoded_queue.put(
                    _DecodedRequest(
                        identity=item.identity,
                        request=request,
                        received_s=item.received_s,
                        decode_started_s=decode_started_s,
                        decode_done_s=decode_done_s,
                    )
                )
            except Exception as exc:
                decode_done_s = time.perf_counter()
                request_id, mode, created_perf_ns = _best_effort_request_identity(item.parts)
                server_stage_ms = {
                    "pipeline_mode": "staged",
                    "router_queue_ms": _elapsed_ms(item.received_s, decode_started_s),
                    "decode_ms": _elapsed_ms(decode_started_s, decode_done_s),
                    "ffs_queue_ms": 0.0,
                    "ffs_stage_ms": 0.0,
                    "encode_queue_ms": 0.0,
                    "encode_ms": 0.0,
                }
                parts = build_error_reply_parts(
                    request_id=request_id,
                    mode=mode,
                    created_perf_ns=created_perf_ns,
                    error=f"{type(exc).__name__}: {exc}",
                    server_total_ms=_elapsed_ms(item.received_s, decode_done_s),
                    server_stage_ms=server_stage_ms,
                    compression=str(self._args.compress),
                    return_type=str(self._args.return_type),
                )
                self._response_queue.put(
                    _QueuedResponse(
                        identity=item.identity,
                        parts=parts,
                        request_id=request_id,
                        status="error",
                        elapsed_ms=_elapsed_ms(item.received_s, decode_done_s),
                    )
                )


class _StagedFfsWorker(threading.Thread):
    def __init__(
        self,
        *,
        worker_idx: int,
        decoded_queue: "queue.Queue[_DecodedRequest | None]",
        encode_queue: "queue.Queue[_InferenceResult]",
        response_queue: "queue.Queue[_QueuedResponse]",
        args: argparse.Namespace,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(name=f"demo-v0.2-staged-ffs-worker-{worker_idx}", daemon=True)
        self.worker_idx = int(worker_idx)
        self._decoded_queue = decoded_queue
        self._encode_queue = encode_queue
        self._response_queue = response_queue
        self._args = args
        self._stop_event = stop_event

    def run(self) -> None:
        from data_process.depth_backends import FastFoundationStereoTensorRTRunner

        runner = FastFoundationStereoTensorRTRunner(
            ffs_repo=Path(self._args.ffs_repo),
            model_dir=Path(self._args.ffs_trt_model_dir),
            trt_root=None if self._args.ffs_trt_root is None else Path(self._args.ffs_trt_root),
        )
        aligner = _CachedAligner()
        warmup_remaining = int(self._args.warmup)
        while not self._stop_event.is_set():
            item = self._decoded_queue.get()
            if item is None:
                return
            inference_started_s = time.perf_counter()
            request_id = str(item.request.header.get("request_id", ""))
            try:
                depths, per_camera_stats, warmup_remaining = _run_ffs_request(
                    request=item.request,
                    runner=runner,
                    aligner=aligner,
                    warmup_remaining=warmup_remaining,
                    args=self._args,
                    worker_idx=self.worker_idx,
                )
                inference_done_s = time.perf_counter()
                self._encode_queue.put(
                    _InferenceResult(
                        identity=item.identity,
                        request=item.request,
                        depths=depths,
                        per_camera_stats=per_camera_stats,
                        received_s=item.received_s,
                        decode_started_s=item.decode_started_s,
                        decode_done_s=item.decode_done_s,
                        inference_started_s=inference_started_s,
                        inference_done_s=inference_done_s,
                    )
                )
            except Exception as exc:
                inference_done_s = time.perf_counter()
                server_stage_ms = {
                    "pipeline_mode": "staged",
                    "router_queue_ms": _elapsed_ms(item.received_s, item.decode_started_s),
                    "decode_ms": _elapsed_ms(item.decode_started_s, item.decode_done_s),
                    "ffs_queue_ms": _elapsed_ms(item.decode_done_s, inference_started_s),
                    "ffs_stage_ms": _elapsed_ms(inference_started_s, inference_done_s),
                    "encode_queue_ms": 0.0,
                    "encode_ms": 0.0,
                }
                parts = build_error_reply_parts(
                    request_id=request_id,
                    mode=str(item.request.header.get("mode", "")),
                    created_perf_ns=int(item.request.header.get("created_perf_ns", 0) or 0),
                    error=f"{type(exc).__name__}: {exc}",
                    server_total_ms=_elapsed_ms(item.received_s, inference_done_s),
                    server_stage_ms=server_stage_ms,
                    compression=str(self._args.compress),
                    return_type=str(self._args.return_type),
                )
                self._response_queue.put(
                    _QueuedResponse(
                        identity=item.identity,
                        parts=parts,
                        request_id=request_id,
                        status="error",
                        elapsed_ms=_elapsed_ms(item.received_s, inference_done_s),
                    )
                )


class _EncodeWorker(threading.Thread):
    def __init__(
        self,
        *,
        worker_idx: int,
        encode_queue: "queue.Queue[_InferenceResult | None]",
        response_queue: "queue.Queue[_QueuedResponse]",
        args: argparse.Namespace,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(name=f"demo-v0.2-encode-worker-{worker_idx}", daemon=True)
        self.worker_idx = int(worker_idx)
        self._encode_queue = encode_queue
        self._response_queue = response_queue
        self._args = args
        self._stop_event = stop_event

    def run(self) -> None:
        while not self._stop_event.is_set():
            item = self._encode_queue.get()
            if item is None:
                return
            request_id = str(item.request.header.get("request_id", ""))
            encode_started_s = time.perf_counter()
            status = "ok"
            try:
                parts = build_reply_parts(
                    request=item.request,
                    depths=item.depths,
                    status="ok",
                    per_camera_stats=item.per_camera_stats,
                    server_total_ms=0.0,
                    compression=str(self._args.compress),
                    return_type=str(self._args.return_type),
                )
            except Exception as exc:
                status = "error"
                parts = build_error_reply_parts(
                    request_id=request_id,
                    mode=str(item.request.header.get("mode", "")),
                    created_perf_ns=int(item.request.header.get("created_perf_ns", 0) or 0),
                    error=f"{type(exc).__name__}: {exc}",
                    compression=str(self._args.compress),
                    return_type=str(self._args.return_type),
                )
            encode_done_s = time.perf_counter()
            server_stage_ms = {
                "pipeline_mode": "staged",
                "router_queue_ms": _elapsed_ms(item.received_s, item.decode_started_s),
                "decode_ms": _elapsed_ms(item.decode_started_s, item.decode_done_s),
                "ffs_queue_ms": _elapsed_ms(item.decode_done_s, item.inference_started_s),
                "ffs_stage_ms": _elapsed_ms(item.inference_started_s, item.inference_done_s),
                "encode_queue_ms": _elapsed_ms(item.inference_done_s, encode_started_s),
                "encode_ms": _elapsed_ms(encode_started_s, encode_done_s),
            }
            parts = _patch_reply_header(
                parts,
                server_total_ms=_elapsed_ms(item.received_s, encode_done_s),
                server_stage_ms=server_stage_ms,
            )
            self._response_queue.put(
                _QueuedResponse(
                    identity=item.identity,
                    parts=parts,
                    request_id=request_id,
                    status=status,
                    elapsed_ms=_elapsed_ms(item.received_s, encode_done_s),
                )
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Demo v0.2 async ROUTER remote FFS depth server.")
    parser.add_argument("--bind", default="tcp://0.0.0.0:7002", help="ZeroMQ ROUTER bind endpoint.")
    parser.add_argument("--ffs-repo", type=Path, default=DEFAULT_FFS_REPO, help="Fast-FoundationStereo repo path.")
    parser.add_argument(
        "--ffs-trt-model-dir",
        type=Path,
        default=DEFAULT_FFS_TRT_TWO_STAGE_MODEL_DIR,
        help="Two-stage TensorRT FFS engine directory.",
    )
    parser.add_argument("--ffs-trt-root", type=Path, default=None, help="Optional TensorRT runtime root.")
    parser.add_argument("--return", dest="return_type", choices=["depth_u16"], default="depth_u16")
    parser.add_argument("--compress", choices=["lz4"], default="lz4")
    parser.add_argument(
        "--server-pipeline-mode",
        choices=["fused-worker", "staged"],
        default="fused-worker",
        help="fused-worker keeps decode+FFS+encode in one worker; staged splits decode, FFS, and encode queues.",
    )
    parser.add_argument("--gpu-workers", type=int, default=1, help="FFS worker count. Default 1 keeps one TRT context owner.")
    parser.add_argument("--decode-workers", type=int, default=1, help="Decode/decompress workers used by --server-pipeline-mode staged.")
    parser.add_argument("--encode-workers", type=int, default=1, help="Reply compression workers used by --server-pipeline-mode staged.")
    parser.add_argument("--max-queue", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--depth-scale-m-per-unit", type=float, default=0.001)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--strict-engine-contract", action="store_true")
    parser.add_argument("--required-model", default=DEFAULT_FFS_MODEL_NAME)
    parser.add_argument("--required-valid-iters", type=int, default=DEFAULT_FFS_VALID_ITERS)
    parser.add_argument("--required-height", type=int, default=DEFAULT_FFS_TRT_ENGINE_SIZE[0])
    parser.add_argument("--required-width", type=int, default=DEFAULT_FFS_TRT_ENGINE_SIZE[1])
    parser.add_argument("--required-capture-height", type=int, default=DEFAULT_FFS_TRT_INPUT_SIZE[0])
    parser.add_argument("--required-capture-width", type=int, default=DEFAULT_FFS_TRT_INPUT_SIZE[1])
    parser.add_argument("--required-builder-optimization-level", type=int, default=DEFAULT_FFS_TRT_BUILDER_OPTIMIZATION_LEVEL)
    parser.add_argument("--required-max-disp", type=int, default=DEFAULT_FFS_MAX_DISP)
    return parser


def _best_effort_request_identity(parts: list[bytes]) -> tuple[str, str, int]:
    try:
        header = json.loads(parts[0].decode("utf-8"))
        if not isinstance(header, dict):
            return "", "", 0
        return (
            str(header.get("request_id", "")),
            str(header.get("mode", "")),
            int(header.get("created_perf_ns", 0) or 0),
        )
    except Exception:
        return "", "", 0


def _split_router_message(parts: list[bytes]) -> tuple[bytes, list[bytes]]:
    if len(parts) < 2:
        raise AsyncFfsProtocolError("ROUTER message missing identity or payload")
    identity = parts[0]
    payload = parts[1:]
    if payload and payload[0] == b"":
        payload = payload[1:]
    if not payload:
        raise AsyncFfsProtocolError("ROUTER message missing payload")
    return identity, payload


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if int(args.gpu_workers) <= 0:
        parser.exit(2, "ffs_depth_async_server_v02.py: error: --gpu-workers must be positive\n")
    if int(args.decode_workers) <= 0:
        parser.exit(2, "ffs_depth_async_server_v02.py: error: --decode-workers must be positive\n")
    if int(args.encode_workers) <= 0:
        parser.exit(2, "ffs_depth_async_server_v02.py: error: --encode-workers must be positive\n")
    if int(args.max_queue) <= 0:
        parser.exit(2, "ffs_depth_async_server_v02.py: error: --max-queue must be positive\n")
    try:
        _validate_engine_contract(args)
    except ValueError as exc:
        parser.exit(2, f"ffs_depth_async_server_v02.py: error: {exc}\n")

    import zmq

    warm_up_numba_ffs_align()
    context = zmq.Context.instance()
    socket = context.socket(zmq.ROUTER)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(str(args.bind))
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)
    request_queue: "queue.Queue[_QueuedRequest | None]" = queue.Queue(maxsize=int(args.max_queue))
    response_queue: "queue.Queue[_QueuedResponse]" = queue.Queue()
    stop_event = threading.Event()
    decoded_queue: "queue.Queue[_DecodedRequest | None] | None" = None
    encode_queue: "queue.Queue[_InferenceResult | None] | None" = None
    workers: list[threading.Thread] = []
    if str(args.server_pipeline_mode) == "fused-worker":
        workers.extend(
            _AsyncFfsWorker(
                worker_idx=idx,
                request_queue=request_queue,
                response_queue=response_queue,
                args=args,
                stop_event=stop_event,
            )
            for idx in range(int(args.gpu_workers))
        )
    else:
        decoded_queue = queue.Queue(maxsize=int(args.max_queue))
        encode_queue = queue.Queue(maxsize=int(args.max_queue))
        workers.extend(
            _DecodeWorker(
                worker_idx=idx,
                request_queue=request_queue,
                decoded_queue=decoded_queue,
                response_queue=response_queue,
                args=args,
                stop_event=stop_event,
            )
            for idx in range(int(args.decode_workers))
        )
        workers.extend(
            _StagedFfsWorker(
                worker_idx=idx,
                decoded_queue=decoded_queue,
                encode_queue=encode_queue,
                response_queue=response_queue,
                args=args,
                stop_event=stop_event,
            )
            for idx in range(int(args.gpu_workers))
        )
        workers.extend(
            _EncodeWorker(
                worker_idx=idx,
                encode_queue=encode_queue,
                response_queue=response_queue,
                args=args,
                stop_event=stop_event,
            )
            for idx in range(int(args.encode_workers))
        )
    for worker in workers:
        worker.start()
    print(
        "[demo-v0.2-async-server] "
        + json.dumps(
            {
                "bind": str(args.bind),
                "protocol": PROTOCOL_NAME,
                "return_type": str(args.return_type),
                "compress": str(args.compress),
                "server_pipeline_mode": str(args.server_pipeline_mode),
                "gpu_workers": int(args.gpu_workers),
                "decode_workers": int(args.decode_workers),
                "encode_workers": int(args.encode_workers),
                "max_queue": int(args.max_queue),
                "ffs_repo": str(args.ffs_repo),
                "ffs_trt_model_dir": str(args.ffs_trt_model_dir),
                "engine_contract": _engine_contract_metadata(args),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        while True:
            while True:
                try:
                    response = response_queue.get_nowait()
                except queue.Empty:
                    break
                socket.send_multipart([response.identity, *response.parts])
                if bool(args.debug):
                    print(
                        "[demo-v0.2-async-server] "
                        f"request_id={response.request_id} status={response.status} "
                        f"elapsed_ms={response.elapsed_ms:.2f} queued={request_queue.qsize()}",
                        flush=True,
                    )
            events = dict(poller.poll(timeout=2))
            if socket not in events:
                continue
            raw_parts = socket.recv_multipart()
            received_s = time.perf_counter()
            try:
                identity, payload = _split_router_message(raw_parts)
                item = _QueuedRequest(identity=identity, parts=payload, received_s=received_s)
                try:
                    request_queue.put_nowait(item)
                except queue.Full:
                    request_id, mode, created_perf_ns = _best_effort_request_identity(payload)
                    error_parts = build_error_reply_parts(
                        request_id=request_id,
                        mode=mode,
                        created_perf_ns=created_perf_ns,
                        error="server request queue full",
                        compression=str(args.compress),
                        return_type=str(args.return_type),
                    )
                    socket.send_multipart([identity, *error_parts])
            except Exception as exc:
                try:
                    identity = raw_parts[0] if raw_parts else b""
                    error_parts = build_error_reply_parts(
                        error=f"{type(exc).__name__}: {exc}",
                        compression=str(args.compress),
                        return_type=str(args.return_type),
                    )
                    if identity:
                        socket.send_multipart([identity, *error_parts])
                except Exception:
                    pass
    except KeyboardInterrupt:
        return 0
    finally:
        stop_event.set()
        for _ in range(int(args.decode_workers) if str(args.server_pipeline_mode) == "staged" else int(args.gpu_workers)):
            try:
                request_queue.put_nowait(None)
            except queue.Full:
                pass
        if decoded_queue is not None:
            for _ in range(int(args.gpu_workers)):
                try:
                    decoded_queue.put_nowait(None)
                except queue.Full:
                    pass
        if encode_queue is not None:
            for _ in range(int(args.encode_workers)):
                try:
                    encode_queue.put_nowait(None)
                except queue.Full:
                    pass
        for worker in workers:
            worker.join(timeout=2.0)
        socket.close(linger=0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
