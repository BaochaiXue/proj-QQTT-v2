#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import queue
import sys
import threading
import time
from typing import Any

import numpy as np


if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from services.ffs_remote.async_protocol_v02 import (  # noqa: E402
    AsyncFfsProtocolError,
    build_request_parts,
    now_perf_ns,
    parse_reply_parts,
)
from services.ffs_remote.ffs_depth_client import (  # noqa: E402
    _list_d400_serials,
    _rs_extrinsics_to_matrix,
    _rs_intrinsics_to_matrix,
    _rs_translation_norm,
)


@dataclass(frozen=True)
class CameraCalibration:
    camera_idx: int
    serial: str
    width: int
    height: int
    k_ir_left: np.ndarray
    k_color: np.ndarray
    t_ir_left_to_color: np.ndarray
    baseline_m: float


@dataclass(frozen=True)
class CameraIrPacket:
    calibration: CameraCalibration
    left_u8: np.ndarray
    right_u8: np.ndarray
    captured_perf_ns: int
    frame_seq: int


class LatestPacketSlot:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._packet: CameraIrPacket | None = None

    def put(self, packet: CameraIrPacket) -> None:
        with self._lock:
            self._packet = packet

    def get(self) -> CameraIrPacket | None:
        with self._lock:
            return self._packet


class CaptureWorker(threading.Thread):
    def __init__(
        self,
        *,
        rs: Any,
        camera_idx: int,
        serial: str,
        width: int,
        height: int,
        fps: int,
        slot: LatestPacketSlot,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(name=f"demo-v0.2-capture-{serial}", daemon=True)
        self._rs = rs
        self._camera_idx = int(camera_idx)
        self._serial = str(serial)
        self._width = int(width)
        self._height = int(height)
        self._fps = int(fps)
        self._slot = slot
        self._stop_event = stop_event
        self.error = ""
        self.capture_count = 0
        self.calibration: CameraCalibration | None = None
        self._pipeline: Any | None = None

    def run(self) -> None:
        rs = self._rs
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self._serial)
        config.enable_stream(rs.stream.infrared, 1, self._width, self._height, rs.format.y8, self._fps)
        config.enable_stream(rs.stream.infrared, 2, self._width, self._height, rs.format.y8, self._fps)
        config.enable_stream(rs.stream.color, self._width, self._height, rs.format.bgr8, self._fps)
        try:
            profile = pipeline.start(config)
            self._pipeline = pipeline
            color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
            ir_left_stream = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
            ir_right_stream = profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()
            baseline_m = _rs_translation_norm(ir_left_stream.get_extrinsics_to(ir_right_stream))
            if baseline_m <= 0:
                baseline_m = 0.055
            self.calibration = CameraCalibration(
                camera_idx=self._camera_idx,
                serial=self._serial,
                width=self._width,
                height=self._height,
                k_ir_left=_rs_intrinsics_to_matrix(ir_left_stream.get_intrinsics()),
                k_color=_rs_intrinsics_to_matrix(color_stream.get_intrinsics()),
                t_ir_left_to_color=_rs_extrinsics_to_matrix(ir_left_stream.get_extrinsics_to(color_stream)),
                baseline_m=float(baseline_m),
            )
            frame_seq = 0
            while not self._stop_event.is_set():
                frames = pipeline.wait_for_frames(5000)
                left_frame = frames.get_infrared_frame(1)
                right_frame = frames.get_infrared_frame(2)
                if not left_frame or not right_frame:
                    continue
                assert self.calibration is not None
                self._slot.put(
                    CameraIrPacket(
                        calibration=self.calibration,
                        left_u8=np.array(np.asanyarray(left_frame.get_data()), dtype=np.uint8, copy=True),
                        right_u8=np.array(np.asanyarray(right_frame.get_data()), dtype=np.uint8, copy=True),
                        captured_perf_ns=now_perf_ns(),
                        frame_seq=frame_seq,
                    )
                )
                frame_seq += 1
                self.capture_count += 1
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self._stop_event.set()
        finally:
            try:
                pipeline.stop()
            except Exception:
                pass


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


def _parse_serials(value: str) -> list[str]:
    serials = [item.strip() for item in str(value).replace(";", ",").split(",") if item.strip()]
    if len(serials) != len(set(serials)):
        raise argparse.ArgumentTypeError("--serials contains duplicates")
    return serials


def _load_realsense_module() -> Any:
    try:
        import pyrealsense2 as rs  # type: ignore[import-not-found]
    except Exception as exc:
        raise RuntimeError("pyrealsense2 is required for live RealSense modes") from exc
    return rs


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _camera_to_payload(packet: CameraIrPacket) -> dict[str, Any]:
    calibration = packet.calibration
    return {
        "camera_idx": calibration.camera_idx,
        "serial": calibration.serial,
        "ir_left_u8": packet.left_u8,
        "ir_right_u8": packet.right_u8,
        "K_ir_left": calibration.k_ir_left,
        "K_color": calibration.k_color,
        "T_ir_left_to_color": calibration.t_ir_left_to_color,
        "baseline_m": calibration.baseline_m,
    }


def _save_depth_preview(depth_u16: np.ndarray, *, output_dir: Path, prefix: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    npy_path = output_dir / f"{prefix}.npy"
    png_path = output_dir / f"{prefix}.png"
    np.save(npy_path, depth_u16)
    valid = depth_u16 > 0
    preview = np.zeros(depth_u16.shape, dtype=np.uint8)
    if np.any(valid):
        values = depth_u16[valid].astype(np.float32)
        lo = float(np.percentile(values, 2.0))
        hi = float(np.percentile(values, 98.0))
        if hi <= lo:
            hi = lo + 1.0
        preview[valid] = np.asarray(np.clip((values - lo) / (hi - lo), 0.0, 1.0) * 255.0, dtype=np.uint8)
    from PIL import Image

    Image.fromarray(preview).save(png_path)
    return npy_path, png_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Demo v0.2 async remote FFS triplet throughput client.")
    parser.add_argument("--mode", choices=("triplet-live", "triplet-replay", "single-live", "single-replay"), required=True)
    parser.add_argument("--endpoint", default="tcp://192.168.0.162:7002")
    parser.add_argument("--serials", default="239222300412,239222300781,239222303506")
    parser.add_argument("--camera-id", default="cam0")
    parser.add_argument("--profile", default="848x480")
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--target-kit-fps", type=float, default=15.0)
    parser.add_argument("--target-camera-fps", type=float, default=45.0)
    parser.add_argument("--duration-s", type=float, default=60.0)
    parser.add_argument("--record-duration-s", type=float, default=30.0)
    parser.add_argument("--max-inflight", type=int, default=6)
    parser.add_argument("--compression", choices=("lz4",), default="lz4")
    parser.add_argument("--return-type", choices=("depth_u16",), default="depth_u16")
    parser.add_argument("--drop-stale-replies", action="store_true")
    parser.add_argument("--record-dir", type=Path, default=None)
    parser.add_argument("--replay-dir", type=Path, default=None)
    parser.add_argument("--no-send", action="store_true")
    parser.add_argument("--save-first-depth-preview", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("docs/generated"))
    parser.add_argument("--debug", action="store_true")
    return parser


def _select_serials(rs: Any, requested: list[str], *, single: bool) -> list[str]:
    available = _list_d400_serials(rs)
    if not requested:
        requested = available[:1 if single else 3]
    missing = [serial for serial in requested if available and serial not in available]
    if missing:
        raise RuntimeError(f"requested serials not detected: {missing}; available={available}")
    return requested[:1] if single else requested[:3]


def _start_capture_workers(args: argparse.Namespace, *, single: bool) -> tuple[list[CaptureWorker], list[LatestPacketSlot], threading.Event]:
    width, height = _parse_profile(str(args.profile))
    rs = _load_realsense_module()
    serials = _select_serials(rs, _parse_serials(str(args.serials)), single=single)
    stop_event = threading.Event()
    slots = [LatestPacketSlot() for _ in serials]
    workers = [
        CaptureWorker(
            rs=rs,
            camera_idx=idx,
            serial=serial,
            width=width,
            height=height,
            fps=int(args.camera_fps),
            slot=slots[idx],
            stop_event=stop_event,
        )
        for idx, serial in enumerate(serials)
    ]
    for worker in workers:
        worker.start()
    deadline_s = time.perf_counter() + 15.0
    while time.perf_counter() < deadline_s:
        if any(worker.error for worker in workers):
            raise RuntimeError("; ".join(worker.error for worker in workers if worker.error))
        if all(slot.get() is not None for slot in slots):
            return workers, slots, stop_event
        time.sleep(0.05)
    raise RuntimeError("timed out waiting for initial camera IR packets")


def _metadata_camera(calibration: CameraCalibration) -> dict[str, Any]:
    return {
        "camera_idx": int(calibration.camera_idx),
        "serial": str(calibration.serial),
        "width": int(calibration.width),
        "height": int(calibration.height),
        "K_ir_left": calibration.k_ir_left.tolist(),
        "K_color": calibration.k_color.tolist(),
        "T_ir_left_to_color": calibration.t_ir_left_to_color.tolist(),
        "baseline_m": float(calibration.baseline_m),
    }


def _calibration_from_metadata(item: dict[str, Any]) -> CameraCalibration:
    return CameraCalibration(
        camera_idx=int(item["camera_idx"]),
        serial=str(item["serial"]),
        width=int(item["width"]),
        height=int(item["height"]),
        k_ir_left=np.asarray(item["K_ir_left"], dtype=np.float32).reshape(3, 3),
        k_color=np.asarray(item["K_color"], dtype=np.float32).reshape(3, 3),
        t_ir_left_to_color=np.asarray(item["T_ir_left_to_color"], dtype=np.float32).reshape(4, 4),
        baseline_m=float(item["baseline_m"]),
    )


def record_triplet_live(args: argparse.Namespace) -> dict[str, float | str]:
    if args.record_dir is None:
        raise ValueError("--record-dir is required for recording")
    workers, slots, stop_event = _start_capture_workers(args, single=False)
    record_dir = Path(args.record_dir)
    started_s = time.perf_counter()
    saved = 0
    try:
        for cam_idx in range(len(slots)):
            (record_dir / f"cam{cam_idx}" / "left").mkdir(parents=True, exist_ok=True)
            (record_dir / f"cam{cam_idx}" / "right").mkdir(parents=True, exist_ok=True)
        from PIL import Image

        next_save_s = time.perf_counter()
        deadline_s = started_s + float(args.record_duration_s)
        while time.perf_counter() < deadline_s:
            now_s = time.perf_counter()
            if now_s < next_save_s:
                time.sleep(min(0.002, next_save_s - now_s))
                continue
            packets = [slot.get() for slot in slots]
            if all(packet is not None for packet in packets):
                for cam_idx, packet in enumerate(packets):
                    assert packet is not None
                    Image.fromarray(packet.left_u8).save(record_dir / f"cam{cam_idx}" / "left" / f"{saved:06d}.png")
                    Image.fromarray(packet.right_u8).save(record_dir / f"cam{cam_idx}" / "right" / f"{saved:06d}.png")
                saved += 1
            next_save_s += 1.0 / float(args.camera_fps)
        calibrations = [slot.get().calibration for slot in slots if slot.get() is not None]
        metadata = {
            "mode": "triplet-record",
            "profile": str(args.profile),
            "camera_fps": int(args.camera_fps),
            "frame_count": int(saved),
            "cameras": [_metadata_camera(calibration) for calibration in calibrations],
        }
        record_dir.mkdir(parents=True, exist_ok=True)
        (record_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    finally:
        stop_event.set()
        for worker in workers:
            worker.join(timeout=2.0)
    elapsed_s = max(1e-9, time.perf_counter() - started_s)
    summary = {"saved_frames": float(saved), "record_fps": float(saved / elapsed_s), "record_dir": str(record_dir)}
    print(
        "[demo-v0.2-record-summary] "
        + " ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in summary.items()),
        flush=True,
    )
    return summary


def _load_replay_metadata(replay_dir: Path) -> tuple[list[CameraCalibration], int]:
    metadata_path = replay_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    cameras = [_calibration_from_metadata(item) for item in metadata["cameras"]]
    return cameras, int(metadata["frame_count"])


def _load_replay_packet(replay_dir: Path, calibration: CameraCalibration, frame_idx: int) -> CameraIrPacket:
    from PIL import Image

    cam_dir = replay_dir / f"cam{calibration.camera_idx}"
    left = np.asarray(Image.open(cam_dir / "left" / f"{frame_idx:06d}.png"), dtype=np.uint8)
    right = np.asarray(Image.open(cam_dir / "right" / f"{frame_idx:06d}.png"), dtype=np.uint8)
    return CameraIrPacket(
        calibration=calibration,
        left_u8=np.ascontiguousarray(left),
        right_u8=np.ascontiguousarray(right),
        captured_perf_ns=now_perf_ns(),
        frame_seq=int(frame_idx),
    )


def _drain_replies(
    *,
    socket: Any,
    poller: Any,
    pending: dict[str, dict[str, Any]],
    stats: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    import zmq

    while True:
        events = dict(poller.poll(timeout=0))
        if socket not in events:
            return
        try:
            parts = socket.recv_multipart(flags=zmq.NOBLOCK)
        except zmq.Again:
            return
        completed_perf_ns = now_perf_ns()
        try:
            reply = parse_reply_parts(parts)
        except Exception as exc:
            stats["failed"] += 1
            if bool(args.debug):
                print(f"[demo-v0.2-client] status=decode_error error={type(exc).__name__}: {exc}", flush=True)
            continue
        request_id = str(reply.header.get("request_id", ""))
        sent = pending.pop(request_id, None)
        if sent is None:
            stats["stale_replies"] += 1
            continue
        if str(reply.header.get("status", "")) != "ok":
            stats["failed"] += 1
            continue
        group_id = int(sent["group_id"])
        if bool(args.drop_stale_replies) and group_id < int(stats["latest_accepted_group_id"]):
            stats["stale_replies"] += 1
            continue
        stats["latest_accepted_group_id"] = max(int(stats["latest_accepted_group_id"]), group_id)
        latency_ms = (completed_perf_ns - int(sent["sent_perf_ns"])) / 1_000_000.0
        stats["kit_latencies_ms"].append(float(latency_ms))
        stats["completed_kits"] += 1
        stats["completed_depths"] += len(reply.depths)
        stats["response_bytes"] += sum(len(part) for part in parts)
        stats["server_total_ms"].append(float(reply.header.get("server_total_ms", 0.0)))
        stage_ms = reply.header.get("server_stage_ms", {})
        if isinstance(stage_ms, dict):
            for key, stats_key in (
                ("decode_ms", "server_decode_ms"),
                ("ffs_stage_ms", "server_ffs_stage_ms"),
                ("encode_ms", "server_encode_ms"),
                ("router_queue_ms", "server_router_queue_ms"),
                ("ffs_queue_ms", "server_ffs_queue_ms"),
                ("encode_queue_ms", "server_encode_queue_ms"),
            ):
                stats[stats_key].append(float(stage_ms.get(key, 0.0)))
        per_camera = reply.header.get("per_camera_stats", [])
        if isinstance(per_camera, list):
            for item in per_camera:
                if not isinstance(item, dict):
                    continue
                stats["server_ffs_ms"].append(float(item.get("server_ffs_ms", 0.0)))
                stats["server_align_ms"].append(float(item.get("server_align_ms", 0.0)))
        for depth in reply.depths:
            serial = depth.serial or f"cam{depth.camera_idx}"
            stats["per_camera_depths"][serial] = stats["per_camera_depths"].get(serial, 0) + 1
            stats["depth_nonzero"][serial] = stats["depth_nonzero"].get(serial, 0.0) + float(np.count_nonzero(depth.depth_u16))
            if bool(args.save_first_depth_preview) and not stats["first_depth_preview"]:
                prefix = f"demo_v02_{request_id}_cam{depth.camera_idx}"
                npy_path, png_path = _save_depth_preview(depth.depth_u16, output_dir=Path(args.output_dir), prefix=prefix)
                stats["first_depth_npy"] = str(npy_path)
                stats["first_depth_preview"] = str(png_path)


def run_network_benchmark(args: argparse.Namespace, *, replay: bool, single: bool) -> dict[str, float | str]:
    if int(args.max_inflight) <= 0:
        raise ValueError("--max-inflight must be positive")
    import zmq

    if replay:
        if args.replay_dir is None:
            raise ValueError("--replay-dir is required for replay modes")
        calibrations, frame_count = _load_replay_metadata(Path(args.replay_dir))
        if single:
            camera_idx = int(str(args.camera_id).replace("cam", ""))
            calibrations = [item for item in calibrations if int(item.camera_idx) == camera_idx]
            if not calibrations:
                raise ValueError(f"camera-id {args.camera_id!r} not found in replay metadata")
        else:
            calibrations = calibrations[:3]
        slots: list[LatestPacketSlot] = []
        stop_event = None
        workers: list[CaptureWorker] = []
    else:
        workers, slots, stop_event = _start_capture_workers(args, single=single)
        calibrations = [slot.get().calibration for slot in slots if slot.get() is not None]
        frame_count = 0

    target_fps = float(args.target_camera_fps if single else args.target_kit_fps)
    context = zmq.Context.instance()
    socket = context.socket(zmq.DEALER)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.SNDHWM, max(1000, int(args.max_inflight) * 4))
    socket.setsockopt(zmq.RCVHWM, max(1000, int(args.max_inflight) * 4))
    socket.connect(str(args.endpoint))
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)
    pending: dict[str, dict[str, Any]] = {}
    stats: dict[str, Any] = {
        "submitted_kits": 0,
        "submitted_depths": 0,
        "completed_kits": 0,
        "completed_depths": 0,
        "failed": 0,
        "submit_skips": 0,
        "stale_replies": 0,
        "request_bytes": 0,
        "response_bytes": 0,
        "kit_latencies_ms": [],
        "server_total_ms": [],
        "server_decode_ms": [],
        "server_ffs_stage_ms": [],
        "server_encode_ms": [],
        "server_router_queue_ms": [],
        "server_ffs_queue_ms": [],
        "server_encode_queue_ms": [],
        "server_ffs_ms": [],
        "server_align_ms": [],
        "per_camera_depths": {},
        "depth_nonzero": {},
        "latest_accepted_group_id": -1,
        "inflight_samples": [],
        "first_depth_npy": "",
        "first_depth_preview": "",
    }
    started_s = time.perf_counter()
    deadline_s = started_s + float(args.duration_s)
    next_send_s = started_s
    group_id = 0
    try:
        while time.perf_counter() < deadline_s:
            _drain_replies(socket=socket, poller=poller, pending=pending, stats=stats, args=args)
            now_s = time.perf_counter()
            stats["inflight_samples"].append(float(len(pending)))
            if now_s < next_send_s:
                time.sleep(min(0.002, next_send_s - now_s))
                continue
            if len(pending) >= int(args.max_inflight):
                stats["submit_skips"] += 1
                next_send_s += 1.0 / target_fps
                continue
            if replay:
                frame_idx = group_id % max(1, frame_count)
                packets = [_load_replay_packet(Path(args.replay_dir), calibration, frame_idx) for calibration in calibrations]
            else:
                packets = [slot.get() for slot in slots]
                if any(packet is None for packet in packets):
                    next_send_s += 1.0 / target_fps
                    continue
                packets = [packet for packet in packets if packet is not None]
            mode = "single" if single else "triplet"
            request_id = f"{mode}-{group_id:08d}"
            parts = build_request_parts(
                request_id=request_id,
                mode=mode,
                camera_payloads=[_camera_to_payload(packet) for packet in packets],
                target_kit_fps=target_fps,
                compression=str(args.compression),
                return_type=str(args.return_type),
                created_perf_ns=now_perf_ns(),
            )
            try:
                socket.send_multipart(parts, flags=zmq.NOBLOCK)
            except zmq.Again:
                stats["submit_skips"] += 1
                next_send_s += 1.0 / target_fps
                continue
            pending[request_id] = {"sent_perf_ns": now_perf_ns(), "group_id": group_id}
            stats["submitted_kits"] += 1
            stats["submitted_depths"] += len(packets)
            stats["request_bytes"] += sum(len(part) for part in parts)
            if bool(args.debug) and int(stats["submitted_kits"]) % max(1, int(math.ceil(target_fps))) == 0:
                elapsed_s = max(1e-9, time.perf_counter() - started_s)
                print(
                    "[demo-v0.2-client] "
                    f"submitted_kits={stats['submitted_kits']} completed_kits={stats['completed_kits']} "
                    f"completed_camera_depth_fps={stats['completed_depths'] / elapsed_s:.2f} "
                    f"inflight={len(pending)} latency_p50={_percentile(stats['kit_latencies_ms'], 50):.2f} "
                    f"server_total_p50={_percentile(stats['server_total_ms'], 50):.2f}",
                    flush=True,
                )
            group_id += 1
            next_send_s += 1.0 / target_fps
        drain_deadline_s = time.perf_counter() + 5.0
        while pending and time.perf_counter() < drain_deadline_s:
            _drain_replies(socket=socket, poller=poller, pending=pending, stats=stats, args=args)
            time.sleep(0.002)
    finally:
        socket.close(linger=0)
        if not replay and stop_event is not None:
            stop_event.set()
            for worker in workers:
                worker.join(timeout=2.0)

    elapsed_s = max(1e-9, time.perf_counter() - started_s)
    per_camera_fps = {
        serial: float(count / elapsed_s)
        for serial, count in stats["per_camera_depths"].items()
    }
    summary: dict[str, float | str] = {
        "duration_s": float(elapsed_s),
        "mode": str(args.mode),
        "target_kit_fps": float(args.target_kit_fps),
        "target_camera_fps": float(args.target_camera_fps),
        "max_inflight": float(args.max_inflight),
        "submitted_kit_fps": float(stats["submitted_kits"] / elapsed_s),
        "submitted_camera_fps": float(stats["submitted_depths"] / elapsed_s),
        "completed_kit_fps": float(stats["completed_kits"] / elapsed_s),
        "completed_camera_depth_fps": float(stats["completed_depths"] / elapsed_s),
        "kit_e2e_ms_p50": _percentile(stats["kit_latencies_ms"], 50),
        "kit_e2e_ms_p90": _percentile(stats["kit_latencies_ms"], 90),
        "kit_e2e_ms_p95": _percentile(stats["kit_latencies_ms"], 95),
        "server_total_ms_p50": _percentile(stats["server_total_ms"], 50),
        "server_total_ms_p95": _percentile(stats["server_total_ms"], 95),
        "server_decode_ms_p50": _percentile(stats["server_decode_ms"], 50),
        "server_decode_ms_p95": _percentile(stats["server_decode_ms"], 95),
        "server_ffs_stage_ms_p50": _percentile(stats["server_ffs_stage_ms"], 50),
        "server_ffs_stage_ms_p95": _percentile(stats["server_ffs_stage_ms"], 95),
        "server_encode_ms_p50": _percentile(stats["server_encode_ms"], 50),
        "server_encode_ms_p95": _percentile(stats["server_encode_ms"], 95),
        "server_router_queue_ms_p50": _percentile(stats["server_router_queue_ms"], 50),
        "server_ffs_queue_ms_p50": _percentile(stats["server_ffs_queue_ms"], 50),
        "server_encode_queue_ms_p50": _percentile(stats["server_encode_queue_ms"], 50),
        "server_ffs_ms_per_camera_p50": _percentile(stats["server_ffs_ms"], 50),
        "server_ffs_ms_per_camera_p95": _percentile(stats["server_ffs_ms"], 95),
        "server_align_ms_per_camera_p50": _percentile(stats["server_align_ms"], 50),
        "server_align_ms_per_camera_p95": _percentile(stats["server_align_ms"], 95),
        "request_kb_mean": float((stats["request_bytes"] / max(1, stats["submitted_kits"])) / 1024.0),
        "response_kb_mean": float((stats["response_bytes"] / max(1, stats["completed_kits"])) / 1024.0),
        "inflight_mean": _mean(stats["inflight_samples"]),
        "inflight_max": float(max(stats["inflight_samples"]) if stats["inflight_samples"] else 0.0),
        "send_queue_depth_mean": 0.0,
        "send_queue_depth_max": 0.0,
        "stale_replies": float(stats["stale_replies"]),
        "timeouts": float(len(pending)),
        "failed": float(stats["failed"]),
        "submit_skips": float(stats["submit_skips"]),
        "per_camera_completed_fps": ",".join(f"{serial}:{fps:.2f}" for serial, fps in sorted(per_camera_fps.items())),
        "first_depth_npy": str(stats["first_depth_npy"]),
        "first_depth_preview": str(stats["first_depth_preview"]),
    }
    print(
        "[demo-v0.2-summary] "
        + " ".join(
            f"{key}={value:.2f}" if isinstance(value, float) else f"{key}={value}"
            for key, value in summary.items()
        ),
        flush=True,
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    single = str(args.mode).startswith("single")
    replay = str(args.mode).endswith("replay")
    try:
        if bool(args.no_send):
            if replay:
                raise ValueError("--no-send is only valid with live record mode")
            record_triplet_live(args)
        else:
            run_network_benchmark(args, replay=replay, single=single)
    except (RuntimeError, ValueError, OSError, AsyncFfsProtocolError) as exc:
        build_parser().exit(2, f"async_remote_ffs_triplet_client.py: error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
