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
from demo_v2.realtime_single_camera_pointcloud import warm_up_numba_ffs_align  # noqa: E402
from services.ffs_remote.async_protocol_v03 import (  # noqa: E402
    PROTOCOL_NAME,
    StagedFfsProtocolError,
    StagedFfsRequest,
    build_error_reply_parts,
    build_reply_parts,
    empty_metrics,
    parse_request_parts,
)
from services.ffs_remote.ffs_depth_server import (  # noqa: E402
    _CachedAligner,
    _engine_contract_metadata,
    _validate_engine_contract,
)


FFS_MODES = ("sequential_batch1", "batch3")
DEFAULT_BATCH3_MODEL_DIR = (
    REPO_ROOT
    / "data"
    / "experiments"
    / "ffs_trt_4090_848x480_pad864_builderopt5_batch3"
    / "engines"
    / "model_20-30-48_iters_4_res_480x864_batch3"
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
class _DecodedRequest:
    identity: bytes
    request: StagedFfsRequest
    request_bytes: int
    received_s: float
    decode_started_s: float
    decode_done_s: float


@dataclass(frozen=True)
class _FfsCameraOutput:
    camera_idx: int
    serial: str
    depth_ir_left_m: np.ndarray
    k_ir_left_used: np.ndarray
    ffs_ms: float


@dataclass(frozen=True)
class _InferenceResult:
    identity: bytes
    request: StagedFfsRequest
    request_bytes: int
    camera_outputs: list[_FfsCameraOutput]
    ffs_metrics: dict[str, float]
    received_s: float
    decode_started_s: float
    decode_done_s: float
    inference_started_s: float
    inference_done_s: float


@dataclass(frozen=True)
class _QueuedResponse:
    identity: bytes
    parts: list[bytes]
    request_id: str
    status: str
    elapsed_ms: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Demo v0.3 staged ROUTER remote FFS 100-kit triplet server.")
    parser.add_argument("--bind", default="tcp://0.0.0.0:7003", help="ZeroMQ ROUTER bind endpoint.")
    parser.add_argument("--ffs-repo", type=Path, default=DEFAULT_FFS_REPO, help="Fast-FoundationStereo repo path.")
    parser.add_argument(
        "--ffs-trt-model-dir",
        type=Path,
        default=DEFAULT_FFS_TRT_TWO_STAGE_MODEL_DIR,
        help="Batch=1 two-stage TensorRT FFS engine directory.",
    )
    parser.add_argument(
        "--ffs-trt-batch3-model-dir",
        type=Path,
        default=DEFAULT_BATCH3_MODEL_DIR,
        help="Batch=3 two-stage TensorRT FFS engine directory.",
    )
    parser.add_argument("--ffs-trt-root", type=Path, default=None, help="Optional TensorRT runtime root.")
    parser.add_argument("--return", dest="return_type", choices=["depth_u16"], default="depth_u16")
    parser.add_argument("--compression", "--compress", dest="compression", choices=["lz4"], default="lz4")
    parser.add_argument("--ffs-mode", choices=FFS_MODES, default="sequential_batch1")
    parser.add_argument("--decode-workers", type=int, default=2)
    parser.add_argument("--postprocess-workers", type=int, default=2)
    parser.add_argument("--ffs-workers", type=int, default=1)
    parser.add_argument("--max-raw-queue", type=int, default=64)
    parser.add_argument("--max-decoded-queue", type=int, default=64)
    parser.add_argument("--max-postprocess-queue", type=int, default=64)
    parser.add_argument("--max-send-queue", type=int, default=64)
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


def validate_args(args: argparse.Namespace) -> None:
    if int(args.ffs_workers) != 1:
        raise ValueError("--ffs-workers must equal 1 for v0.3; one TensorRT context owner is required")
    if int(args.decode_workers) <= 0:
        raise ValueError("--decode-workers must be positive")
    if int(args.postprocess_workers) <= 0:
        raise ValueError("--postprocess-workers must be positive")
    for key in ("max_raw_queue", "max_decoded_queue", "max_postprocess_queue", "max_send_queue"):
        if int(getattr(args, key)) <= 0:
            raise ValueError(f"--{key.replace('_', '-')} must be positive")
    if float(args.depth_scale_m_per_unit) <= 0:
        raise ValueError("--depth-scale-m-per-unit must be positive")
    _validate_engine_contract(args)
    if str(args.ffs_mode) == "batch3" and not Path(args.ffs_trt_batch3_model_dir).exists():
        raise ValueError(f"batch3 model dir does not exist: {Path(args.ffs_trt_batch3_model_dir)}")


def _best_effort_request_identity(parts: list[bytes]) -> tuple[str, int, str, int]:
    try:
        header = json.loads(parts[0].decode("utf-8"))
        if not isinstance(header, dict):
            return "", -1, "", 0
        return (
            str(header.get("request_id", "")),
            int(header.get("kit_idx", -1)),
            str(header.get("phase", "")),
            int(header.get("created_perf_ns", 0) or 0),
        )
    except Exception:
        return "", -1, "", 0


def _split_router_message(parts: list[bytes]) -> tuple[bytes, list[bytes]]:
    if len(parts) < 2:
        raise StagedFfsProtocolError("ROUTER message missing identity or payload")
    identity = parts[0]
    payload = parts[1:]
    if payload and payload[0] == b"":
        payload = payload[1:]
    if not payload:
        raise StagedFfsProtocolError("ROUTER message missing payload")
    return identity, payload


def _make_runner(args: argparse.Namespace):
    from data_process.depth_backends import FastFoundationStereoTensorRTRunner

    model_dir = Path(args.ffs_trt_batch3_model_dir) if str(args.ffs_mode) == "batch3" else Path(args.ffs_trt_model_dir)
    return FastFoundationStereoTensorRTRunner(
        ffs_repo=Path(args.ffs_repo),
        model_dir=model_dir,
        trt_root=None if args.ffs_trt_root is None else Path(args.ffs_trt_root),
    )


def _batch_samples(request: StagedFfsRequest) -> list[dict[str, Any]]:
    return [
        {
            "left_image": camera.ir_left_u8,
            "right_image": camera.ir_right_u8,
            "K_ir_left": camera.k_ir_left,
            "baseline_m": float(camera.baseline_m),
        }
        for camera in request.cameras
    ]


def _run_warmup_if_needed(
    *,
    runner: Any,
    request: StagedFfsRequest,
    args: argparse.Namespace,
    warmup_remaining: int,
) -> int:
    count = int(warmup_remaining)
    if count <= 0:
        return 0
    started_s = time.perf_counter()
    if str(args.ffs_mode) == "batch3":
        samples = _batch_samples(request)
        for _ in range(count):
            runner.run_batch(samples)
    else:
        for _ in range(count):
            for camera in request.cameras:
                runner.run_pair(
                    camera.ir_left_u8,
                    camera.ir_right_u8,
                    K_ir_left=camera.k_ir_left,
                    baseline_m=float(camera.baseline_m),
                )
    if bool(args.debug):
        print(
            "[demo-v0.3-staged-server] "
            f"lazy_warmup count={count} mode={args.ffs_mode} elapsed_ms={_elapsed_ms(started_s):.2f}",
            flush=True,
        )
    return 0


def run_ffs_models(
    *,
    request: StagedFfsRequest,
    runner: Any,
    args: argparse.Namespace,
) -> tuple[list[_FfsCameraOutput], dict[str, float]]:
    metrics = {
        "server_ffs_cam0_ms": 0.0,
        "server_ffs_cam1_ms": 0.0,
        "server_ffs_cam2_ms": 0.0,
        "server_ffs_triplet_ms": 0.0,
        "server_ffs_batch3_ms": 0.0,
    }
    triplet_started_s = time.perf_counter()
    outputs: list[_FfsCameraOutput] = []
    if str(args.ffs_mode) == "batch3":
        samples = _batch_samples(request)
        batch_started_s = time.perf_counter()
        raw_outputs = runner.run_batch(samples)
        batch_done_s = time.perf_counter()
        if len(raw_outputs) != 3:
            raise RuntimeError(f"batch3 runner returned {len(raw_outputs)} outputs, expected 3")
        batch_ms = _elapsed_ms(batch_started_s, batch_done_s)
        metrics["server_ffs_batch3_ms"] = batch_ms
        for camera, output in zip(request.cameras, raw_outputs, strict=True):
            outputs.append(
                _FfsCameraOutput(
                    camera_idx=int(camera.camera_idx),
                    serial=str(camera.serial),
                    depth_ir_left_m=np.asarray(output["depth_ir_left_m"], dtype=np.float32),
                    k_ir_left_used=np.asarray(output.get("K_ir_left_used", camera.k_ir_left), dtype=np.float32),
                    ffs_ms=0.0,
                )
            )
    else:
        for camera in request.cameras:
            cam_started_s = time.perf_counter()
            output = runner.run_pair(
                camera.ir_left_u8,
                camera.ir_right_u8,
                K_ir_left=camera.k_ir_left,
                baseline_m=float(camera.baseline_m),
            )
            cam_done_s = time.perf_counter()
            cam_ms = _elapsed_ms(cam_started_s, cam_done_s)
            metrics[f"server_ffs_cam{int(camera.camera_idx)}_ms"] = cam_ms
            outputs.append(
                _FfsCameraOutput(
                    camera_idx=int(camera.camera_idx),
                    serial=str(camera.serial),
                    depth_ir_left_m=np.asarray(output["depth_ir_left_m"], dtype=np.float32),
                    k_ir_left_used=np.asarray(output.get("K_ir_left_used", camera.k_ir_left), dtype=np.float32),
                    ffs_ms=cam_ms,
                )
            )
    metrics["server_ffs_triplet_ms"] = _elapsed_ms(triplet_started_s)
    return outputs, metrics


class _DecodeWorker(threading.Thread):
    def __init__(
        self,
        *,
        worker_idx: int,
        raw_queue: "queue.Queue[_QueuedRequest | None]",
        decoded_queue: "queue.Queue[_DecodedRequest]",
        send_queue: "queue.Queue[_QueuedResponse]",
        args: argparse.Namespace,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(name=f"demo-v0.3-decode-{worker_idx}", daemon=True)
        self._raw_queue = raw_queue
        self._decoded_queue = decoded_queue
        self._send_queue = send_queue
        self._args = args
        self._stop_event = stop_event

    def run(self) -> None:
        while not self._stop_event.is_set():
            item = self._raw_queue.get()
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
                        request_bytes=sum(len(part) for part in item.parts),
                        received_s=item.received_s,
                        decode_started_s=decode_started_s,
                        decode_done_s=decode_done_s,
                    )
                )
            except Exception as exc:
                decode_done_s = time.perf_counter()
                request_id, kit_idx, phase, created_perf_ns = _best_effort_request_identity(item.parts)
                metrics = empty_metrics()
                metrics.update(
                    {
                        "server_decode_ms": _elapsed_ms(decode_started_s, decode_done_s),
                        "server_total_ms": _elapsed_ms(item.received_s, decode_done_s),
                        "raw_queue_size": float(self._raw_queue.qsize()),
                        "decoded_queue_size": float(self._decoded_queue.qsize()),
                        "send_queue_size": float(self._send_queue.qsize()),
                        "request_kb": float(sum(len(part) for part in item.parts) / 1024.0),
                    }
                )
                parts = build_error_reply_parts(
                    request_id=request_id,
                    kit_idx=kit_idx,
                    phase=phase,
                    created_perf_ns=created_perf_ns,
                    error=f"{type(exc).__name__}: {exc}",
                    metrics=metrics,
                    compression=str(self._args.compression),
                    return_type=str(self._args.return_type),
                )
                self._send_queue.put(
                    _QueuedResponse(
                        identity=item.identity,
                        parts=parts,
                        request_id=request_id,
                        status="error",
                        elapsed_ms=_elapsed_ms(item.received_s, decode_done_s),
                    )
                )


class _SingleFfsWorker(threading.Thread):
    def __init__(
        self,
        *,
        decoded_queue: "queue.Queue[_DecodedRequest | None]",
        postprocess_queue: "queue.Queue[_InferenceResult]",
        send_queue: "queue.Queue[_QueuedResponse]",
        args: argparse.Namespace,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(name="demo-v0.3-single-ffs", daemon=True)
        self._decoded_queue = decoded_queue
        self._postprocess_queue = postprocess_queue
        self._send_queue = send_queue
        self._args = args
        self._stop_event = stop_event

    def run(self) -> None:
        runner = _make_runner(self._args)
        warmup_remaining = int(self._args.warmup)
        while not self._stop_event.is_set():
            item = self._decoded_queue.get()
            if item is None:
                return
            inference_started_s = time.perf_counter()
            request_id = str(item.request.header.get("request_id", ""))
            try:
                warmup_remaining = _run_warmup_if_needed(
                    runner=runner,
                    request=item.request,
                    args=self._args,
                    warmup_remaining=warmup_remaining,
                )
                camera_outputs, ffs_metrics = run_ffs_models(request=item.request, runner=runner, args=self._args)
                inference_done_s = time.perf_counter()
                self._postprocess_queue.put(
                    _InferenceResult(
                        identity=item.identity,
                        request=item.request,
                        request_bytes=item.request_bytes,
                        camera_outputs=camera_outputs,
                        ffs_metrics=ffs_metrics,
                        received_s=item.received_s,
                        decode_started_s=item.decode_started_s,
                        decode_done_s=item.decode_done_s,
                        inference_started_s=inference_started_s,
                        inference_done_s=inference_done_s,
                    )
                )
            except Exception as exc:
                inference_done_s = time.perf_counter()
                metrics = empty_metrics()
                metrics.update(
                    {
                        "server_decode_ms": _elapsed_ms(item.decode_started_s, item.decode_done_s),
                        "server_total_ms": _elapsed_ms(item.received_s, inference_done_s),
                        "request_kb": float(item.request_bytes / 1024.0),
                    }
                )
                parts = build_error_reply_parts(
                    request_id=request_id,
                    kit_idx=int(item.request.header.get("kit_idx", -1)),
                    phase=str(item.request.header.get("phase", "")),
                    created_perf_ns=int(item.request.header.get("created_perf_ns", 0) or 0),
                    error=f"{type(exc).__name__}: {exc}",
                    metrics=metrics,
                    compression=str(self._args.compression),
                    return_type=str(self._args.return_type),
                )
                self._send_queue.put(
                    _QueuedResponse(
                        identity=item.identity,
                        parts=parts,
                        request_id=request_id,
                        status="error",
                        elapsed_ms=_elapsed_ms(item.received_s, inference_done_s),
                    )
                )


class _PostprocessEncodeWorker(threading.Thread):
    def __init__(
        self,
        *,
        worker_idx: int,
        raw_queue: "queue.Queue[_QueuedRequest | None]",
        decoded_queue: "queue.Queue[_DecodedRequest | None]",
        postprocess_queue: "queue.Queue[_InferenceResult | None]",
        send_queue: "queue.Queue[_QueuedResponse]",
        args: argparse.Namespace,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(name=f"demo-v0.3-postprocess-encode-{worker_idx}", daemon=True)
        self._raw_queue = raw_queue
        self._decoded_queue = decoded_queue
        self._postprocess_queue = postprocess_queue
        self._send_queue = send_queue
        self._args = args
        self._stop_event = stop_event
        self._aligner = _CachedAligner()

    def run(self) -> None:
        while not self._stop_event.is_set():
            item = self._postprocess_queue.get()
            if item is None:
                return
            request_id = str(item.request.header.get("request_id", ""))
            encode_started_s = time.perf_counter()
            status = "ok"
            try:
                depths: list[np.ndarray] = []
                metrics = empty_metrics()
                metrics.update(item.ffs_metrics)
                metrics["server_decode_ms"] = _elapsed_ms(item.decode_started_s, item.decode_done_s)
                metrics["request_kb"] = float(item.request_bytes / 1024.0)
                for camera, output in zip(item.request.cameras, item.camera_outputs, strict=True):
                    depth_color_m = self._aligner.align(
                        depth_ir_left_m=output.depth_ir_left_m,
                        color_shape=(int(camera.height), int(camera.width)),
                        k_ir_left=output.k_ir_left_used,
                        t_ir_left_to_color=camera.t_ir_left_to_color,
                        k_color=camera.k_color,
                    )
                    depth_u16 = quantize_depth_with_invalid_zero(
                        depth_color_m,
                        float(self._args.depth_scale_m_per_unit),
                    )
                    depths.append(depth_u16)
                    metrics[f"depth_nonzero_cam{int(camera.camera_idx)}"] = float(np.count_nonzero(depth_u16))
                metrics["raw_queue_size"] = float(self._raw_queue.qsize())
                metrics["decoded_queue_size"] = float(self._decoded_queue.qsize())
                metrics["postprocess_queue_size"] = float(self._postprocess_queue.qsize())
                metrics["send_queue_size"] = float(self._send_queue.qsize())
                parts = build_reply_parts(
                    request=item.request,
                    depths=depths,
                    status="ok",
                    metrics=metrics,
                    compression=str(self._args.compression),
                    return_type=str(self._args.return_type),
                )
            except Exception as exc:
                status = "error"
                metrics = empty_metrics()
                metrics.update(
                    {
                        "server_decode_ms": _elapsed_ms(item.decode_started_s, item.decode_done_s),
                        "request_kb": float(item.request_bytes / 1024.0),
                    }
                )
                parts = build_error_reply_parts(
                    request_id=request_id,
                    kit_idx=int(item.request.header.get("kit_idx", -1)),
                    phase=str(item.request.header.get("phase", "")),
                    created_perf_ns=int(item.request.header.get("created_perf_ns", 0) or 0),
                    error=f"{type(exc).__name__}: {exc}",
                    metrics=metrics,
                    compression=str(self._args.compression),
                    return_type=str(self._args.return_type),
                )
            encode_done_s = time.perf_counter()
            header = json.loads(parts[0].decode("utf-8"))
            header["server_postprocess_encode_ms"] = _elapsed_ms(encode_started_s, encode_done_s)
            header["server_total_ms"] = _elapsed_ms(item.received_s, encode_done_s)
            parts[0] = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
            header["reply_kb"] = float(sum(len(part) for part in parts) / 1024.0)
            parts[0] = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
            if bool(self._args.debug):
                print(
                    "[demo-v0.3-staged-server] "
                    f"request_id={request_id} status={status} ffs_mode={self._args.ffs_mode} "
                    f"server_total_ms={header.get('server_total_ms', 0.0):.2f} "
                    f"ffs_triplet_ms={header.get('server_ffs_triplet_ms', 0.0):.2f} "
                    f"reply_kb={header.get('reply_kb', 0.0):.2f}",
                    flush=True,
                )
            self._send_queue.put(
                _QueuedResponse(
                    identity=item.identity,
                    parts=parts,
                    request_id=request_id,
                    status=status,
                    elapsed_ms=_elapsed_ms(item.received_s, encode_done_s),
                )
            )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_args(args)
    except ValueError as exc:
        parser.exit(2, f"ffs_depth_staged_server_v03.py: error: {exc}\n")

    import zmq

    warm_up_numba_ffs_align()
    context = zmq.Context.instance()
    socket = context.socket(zmq.ROUTER)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(str(args.bind))
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)

    raw_queue: "queue.Queue[_QueuedRequest | None]" = queue.Queue(maxsize=int(args.max_raw_queue))
    decoded_queue: "queue.Queue[_DecodedRequest | None]" = queue.Queue(maxsize=int(args.max_decoded_queue))
    postprocess_queue: "queue.Queue[_InferenceResult | None]" = queue.Queue(maxsize=int(args.max_postprocess_queue))
    send_queue: "queue.Queue[_QueuedResponse]" = queue.Queue(maxsize=int(args.max_send_queue))
    stop_event = threading.Event()

    workers: list[threading.Thread] = []
    workers.extend(
        _DecodeWorker(
            worker_idx=idx,
            raw_queue=raw_queue,
            decoded_queue=decoded_queue,  # type: ignore[arg-type]
            send_queue=send_queue,
            args=args,
            stop_event=stop_event,
        )
        for idx in range(int(args.decode_workers))
    )
    workers.append(
        _SingleFfsWorker(
            decoded_queue=decoded_queue,
            postprocess_queue=postprocess_queue,  # type: ignore[arg-type]
            send_queue=send_queue,
            args=args,
            stop_event=stop_event,
        )
    )
    workers.extend(
        _PostprocessEncodeWorker(
            worker_idx=idx,
            raw_queue=raw_queue,
            decoded_queue=decoded_queue,
            postprocess_queue=postprocess_queue,
            send_queue=send_queue,
            args=args,
            stop_event=stop_event,
        )
        for idx in range(int(args.postprocess_workers))
    )
    for worker in workers:
        worker.start()

    print(
        "[demo-v0.3-staged-server] "
        + json.dumps(
            {
                "bind": str(args.bind),
                "protocol": PROTOCOL_NAME,
                "return_type": str(args.return_type),
                "compression": str(args.compression),
                "ffs_mode": str(args.ffs_mode),
                "decode_workers": int(args.decode_workers),
                "ffs_workers": int(args.ffs_workers),
                "postprocess_workers": int(args.postprocess_workers),
                "max_raw_queue": int(args.max_raw_queue),
                "max_decoded_queue": int(args.max_decoded_queue),
                "max_postprocess_queue": int(args.max_postprocess_queue),
                "max_send_queue": int(args.max_send_queue),
                "ffs_repo": str(args.ffs_repo),
                "ffs_trt_model_dir": str(args.ffs_trt_model_dir),
                "ffs_trt_batch3_model_dir": str(args.ffs_trt_batch3_model_dir),
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
                    response = send_queue.get_nowait()
                except queue.Empty:
                    break
                socket.send_multipart([response.identity, *response.parts])
            events = dict(poller.poll(timeout=2))
            if socket not in events:
                continue
            raw_parts = socket.recv_multipart()
            received_s = time.perf_counter()
            try:
                identity, payload = _split_router_message(raw_parts)
                item = _QueuedRequest(identity=identity, parts=payload, received_s=received_s)
                try:
                    raw_queue.put_nowait(item)
                except queue.Full:
                    request_id, kit_idx, phase, created_perf_ns = _best_effort_request_identity(payload)
                    metrics = empty_metrics()
                    metrics.update(
                        {
                            "raw_queue_size": float(raw_queue.qsize()),
                            "decoded_queue_size": float(decoded_queue.qsize()),
                            "postprocess_queue_size": float(postprocess_queue.qsize()),
                            "send_queue_size": float(send_queue.qsize()),
                            "request_kb": float(sum(len(part) for part in payload) / 1024.0),
                        }
                    )
                    error_parts = build_error_reply_parts(
                        request_id=request_id,
                        kit_idx=kit_idx,
                        phase=phase,
                        created_perf_ns=created_perf_ns,
                        error="server raw request queue full",
                        metrics=metrics,
                        compression=str(args.compression),
                        return_type=str(args.return_type),
                    )
                    socket.send_multipart([identity, *error_parts])
            except Exception as exc:
                try:
                    identity = raw_parts[0] if raw_parts else b""
                    error_parts = build_error_reply_parts(
                        error=f"{type(exc).__name__}: {exc}",
                        compression=str(args.compression),
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
        for _ in range(int(args.decode_workers)):
            try:
                raw_queue.put_nowait(None)
            except queue.Full:
                pass
        try:
            decoded_queue.put_nowait(None)
        except queue.Full:
            pass
        for _ in range(int(args.postprocess_workers)):
            try:
                postprocess_queue.put_nowait(None)
            except queue.Full:
                pass
        for worker in workers:
            worker.join(timeout=2.0)
        socket.close(linger=0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
