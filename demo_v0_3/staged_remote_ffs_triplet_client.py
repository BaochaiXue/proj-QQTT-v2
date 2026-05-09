#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from services.ffs_remote.async_protocol_v03 import (  # noqa: E402
    CAMERA_COUNT,
    StagedFfsProtocolError,
    build_request_parts,
    now_perf_ns,
    parse_reply_parts,
)


DEFAULT_REPLAY_DIR = Path("result/demo_v0_3_ir_triplet_100kits_848x480")
DEFAULT_OUTPUT_DIR = Path("docs/generated")


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
class ReplayKitCamera:
    camera_idx: int
    left_ir_path: Path
    right_ir_path: Path


@dataclass(frozen=True)
class ReplayKit:
    kit_idx: int
    source_kit_idx: int
    capture_time_s: float
    cameras: list[ReplayKitCamera]


@dataclass(frozen=True)
class ReplayDataset:
    replay_dir: Path
    calibrations: dict[int, CameraCalibration]
    kits: list[ReplayKit]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RequestTask:
    phase: str
    ordinal: int
    kit: ReplayKit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Demo v0.3 100-kit staged remote FFS triplet replay client.")
    parser.add_argument("--mode", choices=("triplet-replay",), required=True)
    parser.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY_DIR)
    parser.add_argument("--endpoint", default="tcp://192.168.0.162:7003")
    parser.add_argument("--capture-kit-fps", type=float, default=15.0)
    parser.add_argument("--warmup-kits", type=int, default=20)
    parser.add_argument("--measure-kits", type=int, default=100)
    parser.add_argument("--max-inflight", type=int, default=6)
    parser.add_argument("--compression", choices=("lz4",), default="lz4")
    parser.add_argument("--return-type", choices=("depth_u16",), default="depth_u16")
    parser.add_argument("--replay-once-measured", action="store_true")
    parser.add_argument("--drop-stale-replies", action="store_true")
    parser.add_argument("--timeout-ms", type=int, default=5000)
    parser.add_argument("--save-first-depth-preview", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--per-kit-jsonl", type=Path, default=None)
    parser.add_argument("--debug", action="store_true")
    return parser


def _lookup(item: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in item:
            return item[key]
    if default is not None:
        return default
    raise KeyError(keys[0])


def _calibration_from_metadata(item: dict[str, Any]) -> CameraCalibration:
    width = int(_lookup(item, "width", default=848))
    height = int(_lookup(item, "height", default=480))
    return CameraCalibration(
        camera_idx=int(_lookup(item, "camera_idx")),
        serial=str(_lookup(item, "serial", default=f"cam{int(_lookup(item, 'camera_idx'))}")),
        width=width,
        height=height,
        k_ir_left=np.asarray(_lookup(item, "K_ir_left", "k_ir_left"), dtype=np.float32).reshape(3, 3),
        k_color=np.asarray(_lookup(item, "K_color", "k_color"), dtype=np.float32).reshape(3, 3),
        t_ir_left_to_color=np.asarray(_lookup(item, "T_ir_left_to_color", "t_ir_left_to_color"), dtype=np.float32).reshape(4, 4),
        baseline_m=float(_lookup(item, "baseline_m", "ir_baseline_m")),
    )


def load_replay_dataset(replay_dir: Path) -> ReplayDataset:
    replay_dir = Path(replay_dir)
    metadata_path = replay_dir / "metadata.json"
    kits_path = replay_dir / "kits.jsonl"
    if not metadata_path.is_file():
        raise ValueError(f"missing replay metadata: {metadata_path}")
    if not kits_path.is_file():
        raise ValueError(f"missing replay kits jsonl: {kits_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"{metadata_path} must decode to a JSON object")
    raw_cameras = metadata.get("cameras")
    if not isinstance(raw_cameras, list) or len(raw_cameras) < CAMERA_COUNT:
        raise ValueError(f"{metadata_path} must contain at least three cameras")
    calibrations = {
        int(calibration.camera_idx): calibration
        for calibration in (_calibration_from_metadata(dict(item)) for item in raw_cameras[:CAMERA_COUNT])
    }
    if sorted(calibrations) != [0, 1, 2]:
        raise ValueError(f"v0.3 replay camera order must be cam0/cam1/cam2, got {sorted(calibrations)}")

    kits: list[ReplayKit] = []
    with kits_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"invalid kit row in {kits_path}")
            raw_kit_cameras = item.get("cameras")
            if not isinstance(raw_kit_cameras, list) or len(raw_kit_cameras) != CAMERA_COUNT:
                raise ValueError("each v0.3 kit must contain exactly three cameras")
            kit_cameras = [
                ReplayKitCamera(
                    camera_idx=int(camera["camera_idx"]),
                    left_ir_path=Path(str(camera["left_ir_path"])),
                    right_ir_path=Path(str(camera["right_ir_path"])),
                )
                for camera in raw_kit_cameras
            ]
            kit_cameras = sorted(kit_cameras, key=lambda value: value.camera_idx)
            if [camera.camera_idx for camera in kit_cameras] != [0, 1, 2]:
                raise ValueError(f"kit camera order must be cam0/cam1/cam2, got {[camera.camera_idx for camera in kit_cameras]}")
            kits.append(
                ReplayKit(
                    kit_idx=int(item["kit_idx"]),
                    source_kit_idx=int(item.get("source_kit_idx", item["kit_idx"])),
                    capture_time_s=float(item.get("capture_time_s", 0.0)),
                    cameras=kit_cameras,
                )
            )
    if not kits:
        raise ValueError(f"no replay kits found in {kits_path}")
    return ReplayDataset(replay_dir=replay_dir, calibrations=calibrations, kits=kits, metadata=metadata)


def _load_ir_image(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        image = np.load(path)
    else:
        from PIL import Image

        image = np.asarray(Image.open(path).convert("L"))
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image, dtype=np.uint8)


def _camera_payload(dataset: ReplayDataset, kit_camera: ReplayKitCamera) -> dict[str, Any]:
    calibration = dataset.calibrations[int(kit_camera.camera_idx)]
    left = _load_ir_image(dataset.replay_dir / kit_camera.left_ir_path)
    right = _load_ir_image(dataset.replay_dir / kit_camera.right_ir_path)
    if left.shape != right.shape:
        raise ValueError(f"left/right IR shape mismatch for cam{kit_camera.camera_idx}: {left.shape} vs {right.shape}")
    if left.shape != (calibration.height, calibration.width):
        raise ValueError(
            f"cam{kit_camera.camera_idx} IR shape {left.shape} does not match calibration "
            f"{(calibration.height, calibration.width)}"
        )
    return {
        "camera_idx": calibration.camera_idx,
        "serial": calibration.serial,
        "ir_left_u8": left,
        "ir_right_u8": right,
        "K_ir_left": calibration.k_ir_left,
        "K_color": calibration.k_color,
        "T_ir_left_to_color": calibration.t_ir_left_to_color,
        "baseline_m": calibration.baseline_m,
    }


def build_request_schedule(
    *,
    kits: list[ReplayKit],
    warmup_kits: int,
    measure_kits: int,
    replay_once_measured: bool,
) -> list[RequestTask]:
    if int(warmup_kits) < 0:
        raise ValueError("warmup_kits must be non-negative")
    if int(measure_kits) <= 0:
        raise ValueError("measure_kits must be positive")
    if not kits:
        raise ValueError("at least one replay kit is required")
    if bool(replay_once_measured) and int(measure_kits) > len(kits):
        raise ValueError(
            f"--replay-once-measured requested {measure_kits} measured kits, "
            f"but replay contains only {len(kits)} kits"
        )
    tasks: list[RequestTask] = []
    for idx in range(int(warmup_kits)):
        tasks.append(RequestTask(phase="warmup", ordinal=idx, kit=kits[idx % len(kits)]))
    for idx in range(int(measure_kits)):
        kit = kits[idx] if bool(replay_once_measured) else kits[idx % len(kits)]
        tasks.append(RequestTask(phase="measured", ordinal=idx, kit=kit))
    return tasks


def summarize_values(values: list[float] | np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {key: 0.0 for key in ("avg", "min", "max", "p50", "p90", "p95", "p99")}
    return {
        "avg": float(np.mean(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
    }


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


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


def _metric_float(header: dict[str, Any], key: str) -> float:
    return float(header.get(key, 0.0) or 0.0)


def _complete_summary(
    *,
    args: argparse.Namespace,
    dataset: ReplayDataset,
    rows: list[dict[str, Any]],
    measured_failed_kits: int,
    measured_stale_kits: int,
    first_measured_send_s: float | None,
    last_measured_done_s: float | None,
    first_depth_npy: str,
    first_depth_preview: str,
) -> dict[str, Any]:
    elapsed_s = 0.0
    if first_measured_send_s is not None and last_measured_done_s is not None:
        elapsed_s = max(1e-9, last_measured_done_s - first_measured_send_s)
    measured_completed = len(rows)
    completed_depths = 3 * measured_completed
    summary: dict[str, Any] = {
        "mode": str(args.mode),
        "endpoint": str(args.endpoint),
        "replay_dir": str(dataset.replay_dir),
        "warmup_kits": int(args.warmup_kits),
        "measure_kits": int(args.measure_kits),
        "warmup_included_in_stats": False,
        "capture_kit_fps": float(args.capture_kit_fps),
        "max_inflight": int(args.max_inflight),
        "compression": str(args.compression),
        "return_type": str(args.return_type),
        "measured_completed_kits": int(measured_completed),
        "measured_failed_kits": int(measured_failed_kits),
        "measured_stale_kits": int(measured_stale_kits),
        "completed_kit_fps_mean": float(measured_completed / elapsed_s) if elapsed_s > 0 else 0.0,
        "completed_camera_depth_fps_mean": float(completed_depths / elapsed_s) if elapsed_s > 0 else 0.0,
        "depth_nonzero_cam0_min": int(min((row["depth_nonzero_cam0"] for row in rows), default=0)),
        "depth_nonzero_cam1_min": int(min((row["depth_nonzero_cam1"] for row in rows), default=0)),
        "depth_nonzero_cam2_min": int(min((row["depth_nonzero_cam2"] for row in rows), default=0)),
        "first_depth_npy": first_depth_npy,
        "first_depth_preview": first_depth_preview,
    }
    for key in (
        "kit_e2e_ms",
        "server_decode_ms",
        "server_ffs_triplet_ms",
        "server_ffs_batch3_ms",
        "server_postprocess_encode_ms",
        "server_total_ms",
        "request_kb",
        "reply_kb",
    ):
        summary[key] = summarize_values([float(row[key]) for row in rows])
    return summary


def _write_outputs(args: argparse.Namespace, summary: dict[str, Any], rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = _timestamp()
    summary_path = Path(args.summary_json) if args.summary_json is not None else output_dir / f"demo_v03_100kit_remote_{stamp}.summary.json"
    per_kit_path = Path(args.per_kit_jsonl) if args.per_kit_jsonl is not None else output_dir / f"demo_v03_100kit_remote_{stamp}.per_kit.jsonl"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    per_kit_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with per_kit_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return summary_path, per_kit_path


def run_triplet_replay(args: argparse.Namespace) -> dict[str, Any]:
    if int(args.max_inflight) <= 0:
        raise ValueError("--max-inflight must be positive")
    if float(args.capture_kit_fps) <= 0:
        raise ValueError("--capture-kit-fps must be positive")
    if int(args.timeout_ms) <= 0:
        raise ValueError("--timeout-ms must be positive")
    dataset = load_replay_dataset(Path(args.replay_dir))
    tasks = build_request_schedule(
        kits=dataset.kits,
        warmup_kits=int(args.warmup_kits),
        measure_kits=int(args.measure_kits),
        replay_once_measured=bool(args.replay_once_measured),
    )

    import zmq

    context = zmq.Context.instance()
    socket = context.socket(zmq.DEALER)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.SNDHWM, max(1000, int(args.max_inflight) * 4))
    socket.setsockopt(zmq.RCVHWM, max(1000, int(args.max_inflight) * 4))
    socket.connect(str(args.endpoint))
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)

    pending: dict[str, dict[str, Any]] = {}
    measured_rows: list[dict[str, Any]] = []
    measured_failed_kits = 0
    measured_stale_kits = 0
    first_measured_send_s: float | None = None
    last_measured_done_s: float | None = None
    first_depth_npy = ""
    first_depth_preview = ""
    next_submit_s = time.perf_counter()
    period_s = 1.0 / float(args.capture_kit_fps)
    task_idx = 0

    def drain_replies() -> None:
        nonlocal measured_failed_kits, measured_stale_kits, last_measured_done_s, first_depth_npy, first_depth_preview
        while True:
            events = dict(poller.poll(timeout=0))
            if socket not in events:
                return
            try:
                parts = socket.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                return
            done_s = time.perf_counter()
            try:
                reply = parse_reply_parts(parts)
            except Exception:
                measured_stale_kits += 1
                continue
            request_id = str(reply.header.get("request_id", ""))
            sent = pending.pop(request_id, None)
            if sent is None:
                if bool(args.drop_stale_replies):
                    measured_stale_kits += 1
                continue
            phase = str(sent["phase"])
            if str(reply.header.get("status", "")) != "ok":
                if phase == "measured":
                    measured_failed_kits += 1
                continue
            if phase != "measured":
                continue
            kit_e2e_ms = (done_s - float(sent["sent_s"])) * 1000.0
            row: dict[str, Any] = {
                "measure_idx": int(sent["ordinal"]),
                "kit_idx": int(sent["kit_idx"]),
                "source_kit_idx": int(sent["source_kit_idx"]),
                "request_id": request_id,
                "kit_e2e_ms": float(kit_e2e_ms),
                "request_kb": float(sent["request_kb"]),
                "reply_kb": float(sum(len(part) for part in parts) / 1024.0),
                "server_decode_ms": _metric_float(reply.header, "server_decode_ms"),
                "server_ffs_cam0_ms": _metric_float(reply.header, "server_ffs_cam0_ms"),
                "server_ffs_cam1_ms": _metric_float(reply.header, "server_ffs_cam1_ms"),
                "server_ffs_cam2_ms": _metric_float(reply.header, "server_ffs_cam2_ms"),
                "server_ffs_triplet_ms": _metric_float(reply.header, "server_ffs_triplet_ms"),
                "server_ffs_batch3_ms": _metric_float(reply.header, "server_ffs_batch3_ms"),
                "server_postprocess_encode_ms": _metric_float(reply.header, "server_postprocess_encode_ms"),
                "server_total_ms": _metric_float(reply.header, "server_total_ms"),
                "raw_queue_size": int(_metric_float(reply.header, "raw_queue_size")),
                "decoded_queue_size": int(_metric_float(reply.header, "decoded_queue_size")),
                "postprocess_queue_size": int(_metric_float(reply.header, "postprocess_queue_size")),
                "send_queue_size": int(_metric_float(reply.header, "send_queue_size")),
                "depth_nonzero_cam0": int(_metric_float(reply.header, "depth_nonzero_cam0")),
                "depth_nonzero_cam1": int(_metric_float(reply.header, "depth_nonzero_cam1")),
                "depth_nonzero_cam2": int(_metric_float(reply.header, "depth_nonzero_cam2")),
            }
            measured_rows.append(row)
            last_measured_done_s = done_s
            if bool(args.save_first_depth_preview) and not first_depth_preview and reply.depths:
                npy_path, png_path = _save_depth_preview(
                    reply.depths[0].depth_u16,
                    output_dir=Path(args.output_dir),
                    prefix=f"demo_v03_{request_id}_cam{reply.depths[0].camera_idx}",
                )
                first_depth_npy = str(npy_path)
                first_depth_preview = str(png_path)

    def expire_timeouts() -> None:
        nonlocal measured_failed_kits
        now_s = time.perf_counter()
        expired = [
            request_id
            for request_id, sent in pending.items()
            if (now_s - float(sent["sent_s"])) * 1000.0 > int(args.timeout_ms)
        ]
        for request_id in expired:
            sent = pending.pop(request_id)
            if str(sent["phase"]) == "measured":
                measured_failed_kits += 1

    try:
        while task_idx < len(tasks) or pending:
            drain_replies()
            expire_timeouts()
            now_s = time.perf_counter()
            if task_idx >= len(tasks):
                time.sleep(0.002)
                continue
            if len(pending) >= int(args.max_inflight):
                time.sleep(0.001)
                continue
            if now_s < next_submit_s:
                time.sleep(min(0.002, next_submit_s - now_s))
                continue

            task = tasks[task_idx]
            request_id = f"{task.phase}-{task.ordinal:06d}"
            payloads = [_camera_payload(dataset, camera) for camera in task.kit.cameras]
            parts = build_request_parts(
                request_id=request_id,
                kit_idx=int(task.kit.kit_idx),
                camera_payloads=payloads,
                capture_kit_fps=float(args.capture_kit_fps),
                phase=task.phase,
                compression=str(args.compression),
                return_type=str(args.return_type),
                created_perf_ns=now_perf_ns(),
            )
            try:
                socket.send_multipart(parts, flags=zmq.NOBLOCK)
            except zmq.Again:
                time.sleep(0.001)
                continue
            sent_s = time.perf_counter()
            if task.phase == "measured" and first_measured_send_s is None:
                first_measured_send_s = sent_s
            pending[request_id] = {
                "phase": task.phase,
                "ordinal": int(task.ordinal),
                "kit_idx": int(task.kit.kit_idx),
                "source_kit_idx": int(task.kit.source_kit_idx),
                "sent_s": sent_s,
                "request_kb": float(sum(len(part) for part in parts) / 1024.0),
            }
            if bool(args.debug):
                print(
                    "[demo-v0.3-client] "
                    f"submitted request_id={request_id} phase={task.phase} "
                    f"inflight={len(pending)} completed_measured={len(measured_rows)}",
                    flush=True,
                )
            task_idx += 1
            next_submit_s += period_s
    finally:
        socket.close(linger=0)

    summary = _complete_summary(
        args=args,
        dataset=dataset,
        rows=measured_rows,
        measured_failed_kits=measured_failed_kits,
        measured_stale_kits=measured_stale_kits,
        first_measured_send_s=first_measured_send_s,
        last_measured_done_s=last_measured_done_s,
        first_depth_npy=first_depth_npy,
        first_depth_preview=first_depth_preview,
    )
    summary_path, per_kit_path = _write_outputs(args, summary, measured_rows)
    summary["summary_json"] = str(summary_path)
    summary["per_kit_jsonl"] = str(per_kit_path)
    print("[demo-v0.3-summary] " + json.dumps(summary, sort_keys=True), flush=True)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if str(args.mode) != "triplet-replay":
            raise ValueError("--mode must be triplet-replay")
        run_triplet_replay(args)
    except (RuntimeError, ValueError, OSError, StagedFfsProtocolError) as exc:
        parser.exit(2, f"staged_remote_ffs_triplet_client.py: error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
