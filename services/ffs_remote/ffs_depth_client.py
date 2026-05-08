from __future__ import annotations

import argparse
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from services.ffs_remote.protocol import (
        COMPRESSION_MODES,
        FULL_DEPTH_RETURN_TYPES,
        RETURN_TYPES,
        SPARSE_RETURN_TYPES,
        build_depth_request_parts,
        parse_depth_response_parts,
    )
else:
    from .protocol import (
        COMPRESSION_MODES,
        FULL_DEPTH_RETURN_TYPES,
        RETURN_TYPES,
        SPARSE_RETURN_TYPES,
        build_depth_request_parts,
        parse_depth_response_parts,
    )


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
    return_type: str = "depth_u16"
    compression: str = "none"
    sparse_payload: np.ndarray | None = None
    metadata: dict[str, Any] | None = None


class FfsRemoteDepthClient:
    def __init__(
        self,
        *,
        endpoint: str,
        timeout_ms: int,
        return_type: str = "depth_u16",
        compression: str = "none",
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
        if return_type not in RETURN_TYPES:
            raise ValueError(f"return_type must be one of {', '.join(RETURN_TYPES)}")
        if compression not in COMPRESSION_MODES:
            raise ValueError(f"compression must be one of {', '.join(COMPRESSION_MODES)}")
        self.endpoint = str(endpoint)
        self.timeout_ms = int(timeout_ms)
        self.return_type = str(return_type)
        self.compression = str(compression)
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
        mask_u8: np.ndarray | None = None,
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
            mask_u8=mask_u8,
            compression=self.compression,
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
        return_type = str(metadata.get("return_type") or self.return_type)
        sparse_payload = None
        if return_type in SPARSE_RETURN_TYPES:
            depth_m = np.empty((0, 0), dtype=np.float32)
            sparse_payload = np.ascontiguousarray(depth.astype(np.float32), dtype=np.float32)
        elif depth.dtype == np.uint16:
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
            return_type=return_type,
            compression=str(metadata.get("compression", self.compression)),
            sparse_payload=sparse_payload,
            metadata=dict(metadata),
        )


def _parse_profile(value: str) -> tuple[int, int]:
    try:
        width_s, height_s = str(value).lower().split("x", maxsplit=1)
        width = int(width_s)
        height = int(height_s)
    except Exception as exc:
        raise argparse.ArgumentTypeError("expected WIDTHxHEIGHT, for example 848x480") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("profile dimensions must be positive")
    return width, height


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _load_realsense_module() -> Any:
    try:
        import pyrealsense2 as rs  # type: ignore[import-not-found]
    except Exception as exc:
        raise RuntimeError("pyrealsense2 is required for --real-ir-depth-benchmark") from exc
    return rs


def _list_d400_serials(rs: Any) -> list[str]:
    serials: list[str] = []
    context = rs.context()
    for device in context.query_devices():
        try:
            product_line = device.get_info(rs.camera_info.product_line)
            serial = device.get_info(rs.camera_info.serial_number)
        except Exception:
            continue
        if serial and str(product_line).upper() == "D400":
            serials.append(str(serial))
    return sorted(serials)


def _resolve_serial(rs: Any, requested_serial: str | None) -> str:
    serials = _list_d400_serials(rs)
    if requested_serial:
        if serials and str(requested_serial) not in serials:
            raise RuntimeError(
                f"requested serial {requested_serial!r} is not a detected D400 device; available: {', '.join(serials)}"
            )
        return str(requested_serial)
    if not serials:
        raise RuntimeError("no D400 RealSense device detected")
    return serials[0]


def _rs_intrinsics_to_matrix(intrinsics: Any) -> np.ndarray:
    return np.array(
        [
            [float(getattr(intrinsics, "fx")), 0.0, float(getattr(intrinsics, "ppx"))],
            [0.0, float(getattr(intrinsics, "fy")), float(getattr(intrinsics, "ppy"))],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _rs_extrinsics_to_matrix(extrinsics: Any) -> np.ndarray:
    rotation = list(map(float, getattr(extrinsics, "rotation")))
    translation = list(map(float, getattr(extrinsics, "translation")))
    return np.array(
        [
            [rotation[0], rotation[1], rotation[2], translation[0]],
            [rotation[3], rotation[4], rotation[5], translation[1]],
            [rotation[6], rotation[7], rotation[8], translation[2]],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _rs_translation_norm(extrinsics: Any) -> float:
    tx, ty, tz = map(float, getattr(extrinsics, "translation"))
    return float((tx * tx + ty * ty + tz * tz) ** 0.5)


def _save_depth_artifacts(depth_m: np.ndarray, *, output_dir: Path, prefix: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    depth = np.ascontiguousarray(depth_m, dtype=np.float32)
    npy_path = output_dir / f"{prefix}_depth_m.npy"
    preview_path = output_dir / f"{prefix}_depth_preview.png"
    np.save(npy_path, depth)

    valid = np.isfinite(depth) & (depth > 0.0)
    if np.any(valid):
        valid_values = depth[valid]
        lo = float(np.percentile(valid_values, 2.0))
        hi = float(np.percentile(valid_values, 98.0))
        if hi <= lo:
            hi = float(np.max(valid_values))
        if hi <= lo:
            hi = lo + 1e-6
        preview = np.zeros(depth.shape, dtype=np.uint8)
        scaled = np.clip((depth[valid] - lo) / (hi - lo), 0.0, 1.0)
        preview[valid] = np.asarray(scaled * 255.0, dtype=np.uint8)
    else:
        preview = np.zeros(depth.shape, dtype=np.uint8)
    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("Pillow is required to save --save-first-depth-preview PNG output") from exc
    Image.fromarray(preview).save(preview_path)
    return npy_path, preview_path


@dataclass(frozen=True)
class _RealIrDepthRequest:
    frame_id: int
    ir_left_u8: np.ndarray
    ir_right_u8: np.ndarray
    color_shape: tuple[int, int]
    k_ir_left: np.ndarray
    k_color: np.ndarray
    t_ir_left_to_color: np.ndarray
    baseline_m: float
    depth_scale_m_per_unit: float
    submitted_s: float


@dataclass(frozen=True)
class _RealIrDepthReply:
    worker_idx: int
    frame_id: int
    submitted_s: float
    completed_s: float
    result: FfsRemoteDepthResult | None = None
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.result is not None


@dataclass(frozen=True)
class _RealIrCameraRuntime:
    serial: str
    pipeline: Any
    k_ir_left: np.ndarray
    k_color: np.ndarray
    t_ir_left_to_color: np.ndarray
    baseline_m: float
    depth_scale_m_per_unit: float


def _is_timeout_error_text(error: str) -> bool:
    lowered = error.lower()
    return "timeout" in lowered or "timed out" in lowered or "again" in lowered


class _RealIrDepthWorker(threading.Thread):
    def __init__(
        self,
        *,
        worker_idx: int,
        endpoint: str,
        timeout_ms: int,
        return_type: str,
        compression: str,
        result_queue: "queue.Queue[_RealIrDepthReply]",
    ) -> None:
        super().__init__(name=f"ffs-remote-real-ir-worker-{worker_idx}", daemon=True)
        self.worker_idx = int(worker_idx)
        self._endpoint = str(endpoint)
        self._timeout_ms = int(timeout_ms)
        self._return_type = str(return_type)
        self._compression = str(compression)
        self._result_queue = result_queue
        self._tasks: "queue.Queue[_RealIrDepthRequest | None]" = queue.Queue(maxsize=1)

    def try_submit(self, request: _RealIrDepthRequest) -> bool:
        try:
            self._tasks.put_nowait(request)
            return True
        except queue.Full:
            return False

    def stop(self) -> None:
        try:
            self._tasks.put_nowait(None)
        except queue.Full:
            pass

    def run(self) -> None:
        client = FfsRemoteDepthClient(
            endpoint=self._endpoint,
            timeout_ms=self._timeout_ms,
            return_type=self._return_type,
            compression=self._compression,
        )
        try:
            while True:
                request = self._tasks.get()
                if request is None:
                    return
                try:
                    result = client.request_depth_color_m(
                        frame_id=request.frame_id,
                        ir_left_u8=request.ir_left_u8,
                        ir_right_u8=request.ir_right_u8,
                        color_shape=request.color_shape,
                        k_ir_left=request.k_ir_left,
                        k_color=request.k_color,
                        t_ir_left_to_color=request.t_ir_left_to_color,
                        baseline_m=request.baseline_m,
                        depth_scale_m_per_unit=request.depth_scale_m_per_unit,
                    )
                    reply = _RealIrDepthReply(
                        worker_idx=self.worker_idx,
                        frame_id=request.frame_id,
                        submitted_s=request.submitted_s,
                        completed_s=time.perf_counter(),
                        result=result,
                    )
                except Exception as exc:
                    reply = _RealIrDepthReply(
                        worker_idx=self.worker_idx,
                        frame_id=request.frame_id,
                        submitted_s=request.submitted_s,
                        completed_s=time.perf_counter(),
                        error=f"{type(exc).__name__}: {exc}",
                    )
                self._result_queue.put(reply)
        finally:
            client.close()


def _submit_to_real_ir_worker(
    workers: list[_RealIrDepthWorker],
    request: _RealIrDepthRequest,
    *,
    start_idx: int,
) -> int | None:
    for offset in range(len(workers)):
        idx = (start_idx + offset) % len(workers)
        if workers[idx].try_submit(request):
            return idx
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Remote FFS depth client utilities.")
    parser.add_argument("--endpoint", required=True, help="ZeroMQ server endpoint, for example tcp://100.x.y.z:7001.")
    parser.add_argument("--timeout-ms", type=int, default=80, help="Send/receive timeout in milliseconds.")
    parser.add_argument("--return-type", choices=RETURN_TYPES, default="depth_u16")
    parser.add_argument(
        "--compress",
        choices=COMPRESSION_MODES,
        default="none",
        help="Compress IR request payloads. Server response compression is controlled by the server.",
    )
    parser.add_argument(
        "--mask-fraction",
        type=float,
        default=0.0,
        help="Synthetic mask occupancy for sparse return echo benchmarks. Use 0 to omit the mask payload.",
    )
    parser.add_argument("--echo-benchmark", action="store_true", help="Send synthetic IR pairs and report RTT/throughput.")
    parser.add_argument(
        "--real-ir-depth-benchmark",
        action="store_true",
        help="Capture real RealSense IR left/right frames and request real remote FFS depth.",
    )
    parser.add_argument(
        "--three-camera-real-ir-depth-benchmark",
        action="store_true",
        help=(
            "Demo v0.1 benchmark: capture up to three local RealSense IR pairs, "
            "send one full-depth remote FFS request per camera per group, and "
            "measure async client-side aggregate camera-FPS."
        ),
    )
    parser.add_argument("--serial", default=None, help="RealSense D400 serial for --real-ir-depth-benchmark.")
    parser.add_argument(
        "--serials",
        nargs="*",
        default=None,
        help="RealSense D400 serials for --three-camera-real-ir-depth-benchmark. Defaults to the first --max-cams devices.",
    )
    parser.add_argument("--max-cams", type=int, default=3, help="Maximum cameras for --three-camera-real-ir-depth-benchmark.")
    parser.add_argument("--profile", default="848x480", help="Synthetic IR payload profile for --echo-benchmark.")
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help=(
            "Target FPS. For --three-camera-real-ir-depth-benchmark this is the group/per-camera rate; "
            "--fps 15 with three cameras targets 45 aggregate camera-FPS."
        ),
    )
    parser.add_argument("--duration-s", type=float, default=20.0, help="Benchmark duration in seconds.")
    parser.add_argument("--warmup-frames", type=int, default=15, help="RealSense warmup frames before real-IR benchmark timing.")
    parser.add_argument("--depth-scale-m-per-unit", type=float, default=0.001)
    parser.add_argument("--baseline-m", type=float, default=0.055)
    parser.add_argument(
        "--save-first-depth-preview",
        action="store_true",
        help="Save the first successful returned depth as .npy plus a PNG preview.",
    )
    parser.add_argument(
        "--inflight",
        type=int,
        default=1,
        help="Benchmark-only number of independent REQ sockets for real-IR depth requests.",
    )
    parser.add_argument(
        "--drop-stale-replies",
        action="store_true",
        help="For multi-inflight real-IR benchmarks, count out-of-order older replies as stale latest-wins drops.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/generated"),
        help="Directory for --save-first-depth-preview artifacts.",
    )
    parser.add_argument("--debug", action="store_true", help="Print once-per-second echo benchmark progress.")
    return parser


def run_echo_benchmark(args: argparse.Namespace, *, client: FfsRemoteDepthClient | None = None) -> dict[str, float]:
    width, height = _parse_profile(str(args.profile))
    if float(args.duration_s) <= 0:
        raise ValueError("--duration-s must be positive")
    if float(args.fps) <= 0:
        raise ValueError("--fps must be positive")
    if int(args.timeout_ms) <= 0:
        raise ValueError("--timeout-ms must be positive")
    owned_client = client is None
    if client is None:
        client = FfsRemoteDepthClient(
            endpoint=str(args.endpoint),
            timeout_ms=int(args.timeout_ms),
            return_type=str(args.return_type),
            compression=str(args.compress),
        )
    left = np.zeros((height, width), dtype=np.uint8)
    right = np.zeros((height, width), dtype=np.uint8)
    k_ir = np.array([[600.0, 0.0, width / 2.0], [0.0, 600.0, height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    k_color = k_ir.copy()
    transform = np.eye(4, dtype=np.float32)
    mask_u8: np.ndarray | None = None
    if float(args.mask_fraction) < 0.0 or float(args.mask_fraction) > 1.0:
        raise ValueError("--mask-fraction must be in [0, 1]")
    if float(args.mask_fraction) > 0.0:
        rng = np.random.default_rng(20260505)
        mask_u8 = np.ascontiguousarray((rng.random((height, width)) < float(args.mask_fraction)).astype(np.uint8))

    started_s = time.perf_counter()
    deadline_s = started_s + float(args.duration_s)
    next_send_s = started_s
    frame_id = 0
    successes = 0
    failures = 0
    rtts: list[float] = []
    server_totals: list[float] = []
    request_bytes = 0
    response_bytes = 0
    sparse_points: list[float] = []
    last_log_s = started_s
    try:
        while time.perf_counter() < deadline_s:
            now_s = time.perf_counter()
            if now_s < next_send_s:
                time.sleep(min(0.002, next_send_s - now_s))
                continue
            left.fill(frame_id % 251)
            right.fill((frame_id * 7) % 251)
            try:
                result = client.request_depth_color_m(
                    frame_id=frame_id,
                    ir_left_u8=left,
                    ir_right_u8=right,
                    color_shape=(height, width),
                    k_ir_left=k_ir,
                    k_color=k_color,
                    t_ir_left_to_color=transform,
                    baseline_m=float(args.baseline_m),
                    depth_scale_m_per_unit=float(args.depth_scale_m_per_unit),
                    mask_u8=mask_u8,
                )
                successes += 1
                rtts.append(float(result.rtt_ms))
                server_totals.append(float(result.server_total_ms))
                request_bytes += int(result.request_bytes)
                response_bytes += int(result.response_bytes)
                if result.sparse_payload is not None:
                    sparse_points.append(float(result.sparse_payload.shape[0]))
            except Exception as exc:
                failures += 1
                if bool(args.debug):
                    print(f"[ffs-remote-client] frame_id={frame_id} status=error error={type(exc).__name__}: {exc}", flush=True)
            frame_id += 1
            next_send_s += 1.0 / float(args.fps)
            now_s = time.perf_counter()
            if bool(args.debug) and now_s - last_log_s >= 1.0:
                elapsed_s = max(1e-9, now_s - started_s)
                print(
                    "[ffs-remote-client] "
                    f"sent={frame_id} ok={successes} failed={failures} "
                    f"reply_fps={successes / elapsed_s:.2f} "
                    f"rtt_ms_p50={_percentile(rtts, 50):.2f} "
                    f"rtt_ms_p95={_percentile(rtts, 95):.2f}",
                    flush=True,
                )
                last_log_s = now_s
    finally:
        if owned_client:
            client.close()

    elapsed_s = max(1e-9, time.perf_counter() - started_s)
    summary = {
        "duration_s": float(elapsed_s),
        "sent": float(frame_id),
        "ok": float(successes),
        "failed": float(failures),
        "reply_fps": float(successes / elapsed_s),
        "rtt_ms_p50": _percentile(rtts, 50),
        "rtt_ms_p90": _percentile(rtts, 90),
        "rtt_ms_p95": _percentile(rtts, 95),
        "server_total_ms_p50": _percentile(server_totals, 50),
        "request_kb_mean": float((request_bytes / max(1, successes)) / 1024.0),
        "response_kb_mean": float((response_bytes / max(1, successes)) / 1024.0),
        "mbps_payload": float(((request_bytes + response_bytes) * 8.0) / (elapsed_s * 1_000_000.0)),
        "sparse_points_mean": float(np.mean(np.asarray(sparse_points, dtype=np.float64))) if sparse_points else 0.0,
    }
    print(
        "[ffs-remote-client-summary] "
        + " ".join(f"{key}={value:.2f}" for key, value in summary.items()),
        flush=True,
    )
    return summary


def _validate_real_ir_depth_args(args: argparse.Namespace) -> None:
    if str(args.return_type) not in FULL_DEPTH_RETURN_TYPES:
        raise ValueError(
            "--real-ir-depth-benchmark requires a full-frame return type: "
            + ", ".join(FULL_DEPTH_RETURN_TYPES)
        )
    if float(args.duration_s) <= 0:
        raise ValueError("--duration-s must be positive")
    if float(args.fps) <= 0:
        raise ValueError("--fps must be positive")
    if int(args.timeout_ms) <= 0:
        raise ValueError("--timeout-ms must be positive")
    if int(args.warmup_frames) < 0:
        raise ValueError("--warmup-frames must be non-negative")
    if int(args.inflight) <= 0:
        raise ValueError("--inflight must be positive")


def _validate_three_camera_real_ir_depth_args(args: argparse.Namespace) -> None:
    if str(args.return_type) not in FULL_DEPTH_RETURN_TYPES:
        raise ValueError(
            "--three-camera-real-ir-depth-benchmark requires a full-frame return type: "
            + ", ".join(FULL_DEPTH_RETURN_TYPES)
        )
    if float(args.duration_s) <= 0:
        raise ValueError("--duration-s must be positive")
    if float(args.fps) <= 0:
        raise ValueError("--fps must be positive")
    if int(args.timeout_ms) <= 0:
        raise ValueError("--timeout-ms must be positive")
    if int(args.warmup_frames) < 0:
        raise ValueError("--warmup-frames must be non-negative")
    if int(args.inflight) <= 0:
        raise ValueError("--inflight must be positive")
    if int(args.max_cams) <= 0:
        raise ValueError("--max-cams must be positive")
    if args.serials is not None and len(args.serials) != len(set(args.serials)):
        raise ValueError("--serials contains duplicate serial numbers")


def _resolve_serials(rs: Any, requested_serials: list[str] | None, *, max_cams: int) -> list[str]:
    serials = _list_d400_serials(rs)
    if requested_serials:
        missing = [str(serial) for serial in requested_serials if str(serial) not in serials]
        if serials and missing:
            raise RuntimeError(
                f"requested serials are not detected D400 devices: {', '.join(missing)}; "
                f"available: {', '.join(serials)}"
            )
        return [str(serial) for serial in requested_serials]
    if not serials:
        raise RuntimeError("no D400 RealSense device detected")
    return serials[: int(max_cams)]


def _start_real_ir_camera_runtime(
    *,
    rs: Any,
    serial: str,
    width: int,
    height: int,
    fps: int,
    fallback_baseline_m: float,
) -> _RealIrCameraRuntime:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(str(serial))
    config.enable_stream(rs.stream.infrared, 1, int(width), int(height), rs.format.y8, int(fps))
    config.enable_stream(rs.stream.infrared, 2, int(width), int(height), rs.format.y8, int(fps))
    config.enable_stream(rs.stream.color, int(width), int(height), rs.format.bgr8, int(fps))
    profile = pipeline.start(config)
    try:
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = float(depth_sensor.get_depth_scale())
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        ir_left_stream = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
        ir_right_stream = profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()
        baseline_m = _rs_translation_norm(ir_left_stream.get_extrinsics_to(ir_right_stream))
        if baseline_m <= 0:
            baseline_m = float(fallback_baseline_m)
        return _RealIrCameraRuntime(
            serial=str(serial),
            pipeline=pipeline,
            k_ir_left=_rs_intrinsics_to_matrix(ir_left_stream.get_intrinsics()),
            k_color=_rs_intrinsics_to_matrix(color_stream.get_intrinsics()),
            t_ir_left_to_color=_rs_extrinsics_to_matrix(ir_left_stream.get_extrinsics_to(color_stream)),
            baseline_m=baseline_m,
            depth_scale_m_per_unit=depth_scale,
        )
    except Exception:
        try:
            pipeline.stop()
        except Exception:
            pass
        raise


def run_real_ir_depth_benchmark(
    args: argparse.Namespace,
    *,
    client: FfsRemoteDepthClient | None = None,
) -> dict[str, float | str]:
    _validate_real_ir_depth_args(args)
    width, height = _parse_profile(str(args.profile))
    rs = _load_realsense_module()
    serial = _resolve_serial(rs, None if args.serial is None else str(args.serial))
    owned_client = client is None
    if client is not None and int(args.inflight) != 1:
        raise ValueError("injected real-IR benchmark client only supports --inflight 1")
    if client is None and int(args.inflight) == 1:
        client = FfsRemoteDepthClient(
            endpoint=str(args.endpoint),
            timeout_ms=int(args.timeout_ms),
            return_type=str(args.return_type),
            compression=str(args.compress),
        )

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.infrared, 1, width, height, rs.format.y8, int(args.fps))
    config.enable_stream(rs.stream.infrared, 2, width, height, rs.format.y8, int(args.fps))
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, int(args.fps))

    profile = pipeline.start(config)
    first_depth_npy = ""
    first_depth_preview = ""
    try:
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = float(depth_sensor.get_depth_scale())
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        ir_left_stream = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
        ir_right_stream = profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()
        k_ir_left = _rs_intrinsics_to_matrix(ir_left_stream.get_intrinsics())
        k_color = _rs_intrinsics_to_matrix(color_stream.get_intrinsics())
        t_ir_left_to_color = _rs_extrinsics_to_matrix(ir_left_stream.get_extrinsics_to(color_stream))
        baseline_m = _rs_translation_norm(ir_left_stream.get_extrinsics_to(ir_right_stream))
        if baseline_m <= 0:
            baseline_m = float(args.baseline_m)

        for _ in range(int(args.warmup_frames)):
            pipeline.wait_for_frames(int(args.timeout_ms))

        started_s = time.perf_counter()
        deadline_s = started_s + float(args.duration_s)
        next_send_s = started_s
        frame_id = 0
        submitted = 0
        successes = 0
        accepted_replies = 0
        failures = 0
        capture_misses = 0
        submit_skips = 0
        stale_replies = 0
        timeout_count = 0
        latest_accepted_frame_id = -1
        rtts: list[float] = []
        server_ffs: list[float] = []
        server_align: list[float] = []
        server_totals: list[float] = []
        request_bytes = 0
        response_bytes = 0
        depth_nonzero_counts: list[float] = []
        depth_shapes: set[tuple[int, int]] = set()
        response_compressions: set[str] = set()
        last_log_s = started_s

        inflight = int(args.inflight)
        result_queue: "queue.Queue[_RealIrDepthReply]" = queue.Queue()
        workers: list[_RealIrDepthWorker] = []
        pending_frame_ids: set[int] = set()
        next_worker_idx = 0
        max_pending_observed = 0
        if client is None:
            workers = [
                _RealIrDepthWorker(
                    worker_idx=idx,
                    endpoint=str(args.endpoint),
                    timeout_ms=int(args.timeout_ms),
                    return_type=str(args.return_type),
                    compression=str(args.compress),
                    result_queue=result_queue,
                )
                for idx in range(inflight)
            ]
            for worker in workers:
                worker.start()

        def process_reply(reply: _RealIrDepthReply) -> None:
            nonlocal successes
            nonlocal accepted_replies
            nonlocal failures
            nonlocal stale_replies
            nonlocal timeout_count
            nonlocal request_bytes
            nonlocal response_bytes
            nonlocal first_depth_npy
            nonlocal first_depth_preview
            nonlocal latest_accepted_frame_id
            pending_frame_ids.discard(int(reply.frame_id))
            if not reply.ok:
                failures += 1
                if _is_timeout_error_text(reply.error):
                    timeout_count += 1
                if bool(args.debug):
                    print(
                        "[ffs-remote-real-ir] "
                        f"frame_id={reply.frame_id} worker={reply.worker_idx} status=error error={reply.error}",
                        flush=True,
                    )
                return
            result = reply.result
            assert result is not None
            successes += 1
            rtts.append(float(result.rtt_ms))
            server_ffs.append(float(result.server_ffs_ms))
            server_align.append(float(result.server_align_ms))
            server_totals.append(float(result.server_total_ms))
            request_bytes += int(result.request_bytes)
            response_bytes += int(result.response_bytes)
            depth = result.depth_color_m
            depth_shapes.add(tuple(int(item) for item in depth.shape))
            response_compressions.add(str(result.compression))
            depth_nonzero_counts.append(float(np.count_nonzero(np.isfinite(depth) & (depth > 0.0))))
            if bool(args.drop_stale_replies) and int(result.frame_id) < latest_accepted_frame_id:
                stale_replies += 1
                return
            accepted_replies += 1
            latest_accepted_frame_id = max(latest_accepted_frame_id, int(result.frame_id))
            if bool(args.save_first_depth_preview) and not first_depth_npy:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                prefix = f"demo2_real_ir_remote_depth_{timestamp}_frame{int(result.frame_id):06d}"
                npy_path, preview_path = _save_depth_artifacts(
                    depth,
                    output_dir=Path(args.output_dir),
                    prefix=prefix,
                )
                first_depth_npy = str(npy_path)
                first_depth_preview = str(preview_path)

        def drain_replies(*, block: bool = False) -> None:
            while True:
                try:
                    reply = result_queue.get(timeout=0.05 if block else 0.0)
                except queue.Empty:
                    return
                process_reply(reply)

        try:
            while time.perf_counter() < deadline_s:
                if workers:
                    drain_replies()
                now_s = time.perf_counter()
                if now_s < next_send_s:
                    time.sleep(min(0.002, next_send_s - now_s))
                    continue
                if len(pending_frame_ids) >= inflight:
                    submit_skips += 1
                    next_send_s += 1.0 / float(args.fps)
                    continue
                try:
                    frames = pipeline.wait_for_frames(int(args.timeout_ms))
                    left_frame = frames.get_infrared_frame(1)
                    right_frame = frames.get_infrared_frame(2)
                    color_frame = frames.get_color_frame()
                    if not left_frame or not right_frame or not color_frame:
                        capture_misses += 1
                        next_send_s += 1.0 / float(args.fps)
                        continue
                    left = np.array(np.asanyarray(left_frame.get_data()), dtype=np.uint8, copy=True)
                    right = np.array(np.asanyarray(right_frame.get_data()), dtype=np.uint8, copy=True)
                    color = np.asanyarray(color_frame.get_data())
                    color_shape = (int(color.shape[0]), int(color.shape[1]))
                    request = _RealIrDepthRequest(
                        frame_id=frame_id,
                        ir_left_u8=left,
                        ir_right_u8=right,
                        color_shape=color_shape,
                        k_ir_left=k_ir_left,
                        k_color=k_color,
                        t_ir_left_to_color=t_ir_left_to_color,
                        baseline_m=baseline_m,
                        depth_scale_m_per_unit=depth_scale,
                        submitted_s=time.perf_counter(),
                    )
                    if workers:
                        submitted_worker_idx = _submit_to_real_ir_worker(
                            workers,
                            request,
                            start_idx=next_worker_idx,
                        )
                        if submitted_worker_idx is None:
                            submit_skips += 1
                            next_send_s += 1.0 / float(args.fps)
                            continue
                        next_worker_idx = (submitted_worker_idx + 1) % len(workers)
                        pending_frame_ids.add(frame_id)
                        max_pending_observed = max(max_pending_observed, len(pending_frame_ids))
                    else:
                        assert client is not None
                        try:
                            result = client.request_depth_color_m(
                                frame_id=request.frame_id,
                                ir_left_u8=request.ir_left_u8,
                                ir_right_u8=request.ir_right_u8,
                                color_shape=request.color_shape,
                                k_ir_left=request.k_ir_left,
                                k_color=request.k_color,
                                t_ir_left_to_color=request.t_ir_left_to_color,
                                baseline_m=request.baseline_m,
                                depth_scale_m_per_unit=request.depth_scale_m_per_unit,
                            )
                            process_reply(
                                _RealIrDepthReply(
                                    worker_idx=0,
                                    frame_id=request.frame_id,
                                    submitted_s=request.submitted_s,
                                    completed_s=time.perf_counter(),
                                    result=result,
                                )
                            )
                        except Exception as exc:
                            process_reply(
                                _RealIrDepthReply(
                                    worker_idx=0,
                                    frame_id=request.frame_id,
                                    submitted_s=request.submitted_s,
                                    completed_s=time.perf_counter(),
                                    error=f"{type(exc).__name__}: {exc}",
                                )
                            )
                    submitted += 1
                    frame_id += 1
                except Exception as exc:
                    failures += 1
                    if bool(args.debug):
                        print(f"[ffs-remote-real-ir] frame_id={frame_id} status=error error={type(exc).__name__}: {exc}", flush=True)
                next_send_s += 1.0 / float(args.fps)
                now_s = time.perf_counter()
                if bool(args.debug) and now_s - last_log_s >= 1.0:
                    elapsed_s = max(1e-9, now_s - started_s)
                    print(
                        "[ffs-remote-real-ir] "
                        f"submitted={submitted} ok={successes} accepted={accepted_replies} "
                        f"failed={failures} stale={stale_replies} capture_miss={capture_misses} "
                        f"submitted_fps={submitted / elapsed_s:.2f} "
                        f"completed_fps={successes / elapsed_s:.2f} "
                        f"reply_fps={accepted_replies / elapsed_s:.2f} "
                        f"inflight={len(pending_frame_ids)} "
                        f"rtt_ms_p50={_percentile(rtts, 50):.2f} "
                        f"server_total_ms_p50={_percentile(server_totals, 50):.2f} "
                        f"depth_nonzero_mean={_mean(depth_nonzero_counts):.0f}",
                        flush=True,
                    )
                    last_log_s = now_s

            while pending_frame_ids:
                drain_replies(block=True)
        finally:
            for worker in workers:
                worker.stop()
            for worker in workers:
                worker.join(timeout=1.0)
    finally:
        try:
            pipeline.stop()
        finally:
            if owned_client and client is not None:
                client.close()

    elapsed_s = max(1e-9, time.perf_counter() - started_s)
    summary: dict[str, float | str] = {
        "duration_s": float(elapsed_s),
        "sent": float(submitted),
        "submitted": float(submitted),
        "ok": float(successes),
        "accepted": float(accepted_replies),
        "failed": float(failures),
        "capture_miss": float(capture_misses),
        "submit_skip": float(submit_skips),
        "stale_replies": float(stale_replies),
        "timeout_count": float(timeout_count),
        "inflight": float(inflight),
        "max_pending_observed": float(max_pending_observed),
        "submitted_fps": float(submitted / elapsed_s),
        "completed_fps": float(successes / elapsed_s),
        "reply_fps": float(accepted_replies / elapsed_s),
        "rtt_ms_p50": _percentile(rtts, 50),
        "rtt_ms_p90": _percentile(rtts, 90),
        "rtt_ms_p95": _percentile(rtts, 95),
        "server_ffs_ms_p50": _percentile(server_ffs, 50),
        "server_align_ms_p50": _percentile(server_align, 50),
        "server_total_ms_p50": _percentile(server_totals, 50),
        "request_kb_mean": float((request_bytes / max(1, successes)) / 1024.0),
        "response_kb_mean": float((response_bytes / max(1, successes)) / 1024.0),
        "mbps_payload": float(((request_bytes + response_bytes) * 8.0) / (elapsed_s * 1_000_000.0)),
        "depth_nonzero_count_mean": _mean(depth_nonzero_counts),
        "serial": serial,
        "request_compression": str(args.compress),
        "response_compression": ",".join(sorted(response_compressions)) if response_compressions else "",
        "return_type": str(args.return_type),
        "depth_shapes": ",".join(f"{h}x{w}" for h, w in sorted(depth_shapes)) if depth_shapes else "",
        "first_depth_npy_path": first_depth_npy,
        "first_depth_preview_path": first_depth_preview,
    }
    print(
        "[ffs-remote-real-ir-summary] "
        + " ".join(
            f"{key}={value:.2f}" if isinstance(value, float) else f"{key}={value}"
            for key, value in summary.items()
        ),
        flush=True,
    )
    return summary


def run_three_camera_real_ir_depth_benchmark(args: argparse.Namespace) -> dict[str, float | str]:
    _validate_three_camera_real_ir_depth_args(args)
    width, height = _parse_profile(str(args.profile))
    rs = _load_realsense_module()
    serials = _resolve_serials(
        rs,
        None if args.serials is None else [str(serial) for serial in args.serials],
        max_cams=int(args.max_cams),
    )
    if not serials:
        raise RuntimeError("no RealSense cameras selected")
    capture_fps = int(round(float(args.fps)))
    if capture_fps <= 0:
        raise ValueError("--fps must round to a positive integer for RealSense capture")

    runtimes: list[_RealIrCameraRuntime] = []
    result_queue: "queue.Queue[_RealIrDepthReply]" = queue.Queue()
    workers: list[_RealIrDepthWorker] = [
        _RealIrDepthWorker(
            worker_idx=idx,
            endpoint=str(args.endpoint),
            timeout_ms=int(args.timeout_ms),
            return_type=str(args.return_type),
            compression=str(args.compress),
            result_queue=result_queue,
        )
        for idx in range(int(args.inflight))
    ]
    for worker in workers:
        worker.start()

    first_depth_npy = ""
    first_depth_preview = ""
    started_s = time.perf_counter()
    try:
        for serial in serials:
            runtimes.append(
                _start_real_ir_camera_runtime(
                    rs=rs,
                    serial=str(serial),
                    width=width,
                    height=height,
                    fps=capture_fps,
                    fallback_baseline_m=float(args.baseline_m),
                )
            )
        for _ in range(int(args.warmup_frames)):
            for runtime in runtimes:
                runtime.pipeline.wait_for_frames(int(args.timeout_ms))

        cam_count = len(runtimes)
        deadline_s = time.perf_counter() + float(args.duration_s)
        next_group_s = time.perf_counter()
        group_id = 0
        next_worker_idx = 0
        pending_frame_ids: set[int] = set()
        pending_info: dict[int, tuple[int, int, str]] = {}
        group_expected_counts: dict[int, int] = {}
        group_success_counts: dict[int, int] = {}
        per_camera_submitted = {runtime.serial: 0 for runtime in runtimes}
        per_camera_success = {runtime.serial: 0 for runtime in runtimes}
        per_camera_failure = {runtime.serial: 0 for runtime in runtimes}
        per_camera_latest_group = {runtime.serial: -1 for runtime in runtimes}
        groups_submitted = 0
        full_groups_submitted = 0
        submitted = 0
        successes = 0
        accepted_replies = 0
        failures = 0
        capture_misses = 0
        submit_skips = 0
        stale_replies = 0
        timeout_count = 0
        max_pending_observed = 0
        rtts: list[float] = []
        server_ffs: list[float] = []
        server_align: list[float] = []
        server_totals: list[float] = []
        request_bytes = 0
        response_bytes = 0
        depth_nonzero_counts: list[float] = []
        depth_shapes: set[tuple[int, int]] = set()
        response_compressions: set[str] = set()
        response_return_types: set[str] = set()
        last_log_s = time.perf_counter()

        def process_reply(reply: _RealIrDepthReply) -> None:
            nonlocal successes
            nonlocal accepted_replies
            nonlocal failures
            nonlocal stale_replies
            nonlocal timeout_count
            nonlocal request_bytes
            nonlocal response_bytes
            nonlocal first_depth_npy
            nonlocal first_depth_preview
            info = pending_info.pop(int(reply.frame_id), (-1, -1, "unknown"))
            pending_frame_ids.discard(int(reply.frame_id))
            reply_group_id, _reply_cam_idx, reply_serial = info
            if not reply.ok:
                failures += 1
                if reply_serial in per_camera_failure:
                    per_camera_failure[reply_serial] += 1
                if _is_timeout_error_text(reply.error):
                    timeout_count += 1
                if bool(args.debug):
                    print(
                        "[ffs-remote-demo-v0.1] "
                        f"group={reply_group_id} serial={reply_serial} frame_id={reply.frame_id} "
                        f"worker={reply.worker_idx} status=error error={reply.error}",
                        flush=True,
                    )
                return
            result = reply.result
            assert result is not None
            successes += 1
            rtts.append(float(result.rtt_ms))
            server_ffs.append(float(result.server_ffs_ms))
            server_align.append(float(result.server_align_ms))
            server_totals.append(float(result.server_total_ms))
            request_bytes += int(result.request_bytes)
            response_bytes += int(result.response_bytes)
            response_compressions.add(str(result.compression))
            response_return_types.add(str(result.return_type))
            depth = result.depth_color_m
            depth_shapes.add(tuple(int(item) for item in depth.shape))
            depth_nonzero_counts.append(float(np.count_nonzero(np.isfinite(depth) & (depth > 0.0))))
            if bool(args.drop_stale_replies) and reply_group_id < per_camera_latest_group.get(reply_serial, -1):
                stale_replies += 1
                return
            accepted_replies += 1
            if reply_serial in per_camera_success:
                per_camera_success[reply_serial] += 1
                per_camera_latest_group[reply_serial] = max(per_camera_latest_group[reply_serial], int(reply_group_id))
            if reply_group_id >= 0:
                group_success_counts[reply_group_id] = group_success_counts.get(reply_group_id, 0) + 1
            if bool(args.save_first_depth_preview) and not first_depth_npy and depth.size:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                prefix = f"demo_v0_1_threecam_remote_depth_{timestamp}_frame{int(result.frame_id):06d}"
                npy_path, preview_path = _save_depth_artifacts(
                    depth,
                    output_dir=Path(args.output_dir),
                    prefix=prefix,
                )
                first_depth_npy = str(npy_path)
                first_depth_preview = str(preview_path)

        def drain_replies(*, block: bool = False) -> None:
            while True:
                try:
                    reply = result_queue.get(timeout=0.05 if block else 0.0)
                except queue.Empty:
                    return
                process_reply(reply)

        try:
            while time.perf_counter() < deadline_s:
                drain_replies()
                now_s = time.perf_counter()
                if now_s < next_group_s:
                    time.sleep(min(0.002, next_group_s - now_s))
                    continue
                if len(pending_frame_ids) + cam_count > int(args.inflight):
                    submit_skips += cam_count
                    next_group_s += 1.0 / float(args.fps)
                    continue

                group_requests: list[tuple[int, _RealIrCameraRuntime, _RealIrDepthRequest]] = []
                group_capture_ok = True
                for cam_idx, runtime in enumerate(runtimes):
                    try:
                        frames = runtime.pipeline.wait_for_frames(int(args.timeout_ms))
                        left_frame = frames.get_infrared_frame(1)
                        right_frame = frames.get_infrared_frame(2)
                        color_frame = frames.get_color_frame()
                        if not left_frame or not right_frame or not color_frame:
                            raise RuntimeError("missing IR or color frame")
                        left = np.array(np.asanyarray(left_frame.get_data()), dtype=np.uint8, copy=True)
                        right = np.array(np.asanyarray(right_frame.get_data()), dtype=np.uint8, copy=True)
                        color = np.asanyarray(color_frame.get_data())
                        frame_id = int(group_id * cam_count + cam_idx)
                        request = _RealIrDepthRequest(
                            frame_id=frame_id,
                            ir_left_u8=left,
                            ir_right_u8=right,
                            color_shape=(int(color.shape[0]), int(color.shape[1])),
                            k_ir_left=runtime.k_ir_left,
                            k_color=runtime.k_color,
                            t_ir_left_to_color=runtime.t_ir_left_to_color,
                            baseline_m=runtime.baseline_m,
                            depth_scale_m_per_unit=runtime.depth_scale_m_per_unit,
                            submitted_s=time.perf_counter(),
                        )
                        group_requests.append((cam_idx, runtime, request))
                    except Exception as exc:
                        capture_misses += 1
                        group_capture_ok = False
                        if bool(args.debug):
                            print(
                                "[ffs-remote-demo-v0.1] "
                                f"group={group_id} serial={runtime.serial} status=capture_error "
                                f"error={type(exc).__name__}: {exc}",
                                flush=True,
                            )
                if not group_capture_ok or len(group_requests) != cam_count:
                    group_id += 1
                    next_group_s += 1.0 / float(args.fps)
                    continue

                submitted_this_group = 0
                for cam_idx, runtime, request in group_requests:
                    submitted_worker_idx = _submit_to_real_ir_worker(
                        workers,
                        request,
                        start_idx=next_worker_idx,
                    )
                    if submitted_worker_idx is None:
                        submit_skips += 1
                        continue
                    next_worker_idx = (submitted_worker_idx + 1) % len(workers)
                    pending_frame_ids.add(request.frame_id)
                    pending_info[request.frame_id] = (int(group_id), int(cam_idx), runtime.serial)
                    per_camera_submitted[runtime.serial] += 1
                    submitted += 1
                    submitted_this_group += 1
                    max_pending_observed = max(max_pending_observed, len(pending_frame_ids))
                if submitted_this_group:
                    groups_submitted += 1
                    group_expected_counts[int(group_id)] = submitted_this_group
                    if submitted_this_group == cam_count:
                        full_groups_submitted += 1
                group_id += 1
                next_group_s += 1.0 / float(args.fps)

                now_s = time.perf_counter()
                if bool(args.debug) and now_s - last_log_s >= 1.0:
                    elapsed_s = max(1e-9, now_s - started_s)
                    complete_groups = sum(
                        1
                        for gid, expected in group_expected_counts.items()
                        if expected == cam_count and group_success_counts.get(gid, 0) >= cam_count
                    )
                    print(
                        "[ffs-remote-demo-v0.1] "
                        f"groups={groups_submitted} full_groups={full_groups_submitted} "
                        f"complete_groups={complete_groups} submitted={submitted} ok={successes} "
                        f"accepted={accepted_replies} failed={failures} stale={stale_replies} "
                        f"capture_miss={capture_misses} submit_skip={submit_skips} "
                        f"aggregate_completed_fps={successes / elapsed_s:.2f} "
                        f"complete_group_fps={complete_groups / elapsed_s:.2f} "
                        f"inflight={len(pending_frame_ids)} "
                        f"rtt_ms_p50={_percentile(rtts, 50):.2f} "
                        f"server_total_ms_p50={_percentile(server_totals, 50):.2f}",
                        flush=True,
                    )
                    last_log_s = now_s

            while pending_frame_ids:
                drain_replies(block=True)
        finally:
            for worker in workers:
                worker.stop()
            for worker in workers:
                worker.join(timeout=1.0)
    finally:
        for runtime in runtimes:
            try:
                runtime.pipeline.stop()
            except Exception:
                pass

    elapsed_s = max(1e-9, time.perf_counter() - started_s)
    cam_count = max(1, len(runtimes))
    complete_groups = sum(
        1
        for gid, expected in group_expected_counts.items()
        if expected == cam_count and group_success_counts.get(gid, 0) >= cam_count
    )
    per_camera_completed_fps = {
        serial: float(count / elapsed_s)
        for serial, count in per_camera_success.items()
    }
    summary: dict[str, float | str] = {
        "duration_s": float(elapsed_s),
        "camera_count": float(len(runtimes)),
        "target_per_camera_fps": float(args.fps),
        "target_aggregate_camera_fps": float(args.fps) * float(len(runtimes)),
        "groups_submitted": float(groups_submitted),
        "full_groups_submitted": float(full_groups_submitted),
        "complete_groups": float(complete_groups),
        "group_submit_fps": float(groups_submitted / elapsed_s),
        "complete_group_fps": float(complete_groups / elapsed_s),
        "submitted": float(submitted),
        "ok": float(successes),
        "accepted": float(accepted_replies),
        "failed": float(failures),
        "capture_miss": float(capture_misses),
        "submit_skip": float(submit_skips),
        "stale_replies": float(stale_replies),
        "timeout_count": float(timeout_count),
        "inflight": float(args.inflight),
        "max_pending_observed": float(max_pending_observed),
        "aggregate_submitted_fps": float(submitted / elapsed_s),
        "aggregate_completed_fps": float(successes / elapsed_s),
        "aggregate_reply_fps": float(accepted_replies / elapsed_s),
        "rtt_ms_p50": _percentile(rtts, 50),
        "rtt_ms_p90": _percentile(rtts, 90),
        "rtt_ms_p95": _percentile(rtts, 95),
        "server_ffs_ms_p50": _percentile(server_ffs, 50),
        "server_align_ms_p50": _percentile(server_align, 50),
        "server_total_ms_p50": _percentile(server_totals, 50),
        "request_kb_mean": float((request_bytes / max(1, successes)) / 1024.0),
        "response_kb_mean": float((response_bytes / max(1, successes)) / 1024.0),
        "mbps_payload": float(((request_bytes + response_bytes) * 8.0) / (elapsed_s * 1_000_000.0)),
        "depth_nonzero_count_mean": _mean(depth_nonzero_counts),
        "serials": ",".join(runtime.serial for runtime in runtimes),
        "per_camera_completed_fps": ",".join(
            f"{serial}:{per_camera_completed_fps.get(serial, 0.0):.2f}"
            for serial in per_camera_success
        ),
        "request_compression": str(args.compress),
        "response_compression": ",".join(sorted(response_compressions)) if response_compressions else "",
        "return_type": ",".join(sorted(response_return_types)) if response_return_types else str(args.return_type),
        "depth_shapes": ",".join(f"{h}x{w}" for h, w in sorted(depth_shapes)) if depth_shapes else "",
        "first_depth_npy_path": first_depth_npy,
        "first_depth_preview_path": first_depth_preview,
    }
    print(
        "[ffs-remote-demo-v0.1-summary] "
        + " ".join(
            f"{key}={value:.2f}" if isinstance(value, float) else f"{key}={value}"
            for key, value in summary.items()
        ),
        flush=True,
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        selected_modes = [
            bool(args.echo_benchmark),
            bool(args.real_ir_depth_benchmark),
            bool(args.three_camera_real_ir_depth_benchmark),
        ]
        if sum(1 for selected in selected_modes if selected) != 1:
            raise ValueError(
                "choose exactly one mode: --echo-benchmark, --real-ir-depth-benchmark, "
                "or --three-camera-real-ir-depth-benchmark"
            )
        if args.echo_benchmark:
            run_echo_benchmark(args)
        elif args.real_ir_depth_benchmark:
            run_real_ir_depth_benchmark(args)
        else:
            run_three_camera_real_ir_depth_benchmark(args)
    except (RuntimeError, ValueError, OSError) as exc:
        build_parser().exit(2, f"ffs_depth_client.py: error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
