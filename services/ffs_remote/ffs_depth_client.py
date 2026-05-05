from __future__ import annotations

import argparse
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
    from services.ffs_remote.protocol import build_depth_request_parts, parse_depth_response_parts
else:
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Remote FFS depth client utilities.")
    parser.add_argument("--endpoint", required=True, help="ZeroMQ server endpoint, for example tcp://100.x.y.z:7001.")
    parser.add_argument("--timeout-ms", type=int, default=80, help="Send/receive timeout in milliseconds.")
    parser.add_argument("--return-type", choices=("depth_u16", "depth_float_m"), default="depth_u16")
    parser.add_argument("--echo-benchmark", action="store_true", help="Send synthetic IR pairs and report RTT/throughput.")
    parser.add_argument("--profile", default="848x480", help="Synthetic IR payload profile for --echo-benchmark.")
    parser.add_argument("--fps", type=float, default=30.0, help="Target request rate for --echo-benchmark.")
    parser.add_argument("--duration-s", type=float, default=20.0, help="Benchmark duration in seconds.")
    parser.add_argument("--depth-scale-m-per-unit", type=float, default=0.001)
    parser.add_argument("--baseline-m", type=float, default=0.055)
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
        )
    left = np.zeros((height, width), dtype=np.uint8)
    right = np.zeros((height, width), dtype=np.uint8)
    k_ir = np.array([[600.0, 0.0, width / 2.0], [0.0, 600.0, height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    k_color = k_ir.copy()
    transform = np.eye(4, dtype=np.float32)

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
                )
                successes += 1
                rtts.append(float(result.rtt_ms))
                server_totals.append(float(result.server_total_ms))
                request_bytes += int(result.request_bytes)
                response_bytes += int(result.response_bytes)
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
    }
    print(
        "[ffs-remote-client-summary] "
        + " ".join(f"{key}={value:.2f}" for key, value in summary.items()),
        flush=True,
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if not args.echo_benchmark:
            raise ValueError("currently supported CLI mode is --echo-benchmark")
        run_echo_benchmark(args)
    except (RuntimeError, ValueError, OSError) as exc:
        build_parser().exit(2, f"ffs_depth_client.py: error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
