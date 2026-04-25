from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime
import json
import pickle
from pathlib import Path
import shutil
import time
from typing import Any, Callable

import numpy as np

from data_process.aligned_case_metadata import LEGACY_ALIGNED_METADATA_KEYS
from qqtt.env.camera.defaults import (
    DEFAULT_FPS,
    DEFAULT_HEIGHT,
    DEFAULT_NUM_CAM,
    DEFAULT_WIDTH,
)


PROJECT_ROOT = next(
    (p for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents] if (p / ".git").exists()),
    Path(__file__).resolve().parent,
)


def _resolve_path(path: str) -> Path:
    return (PROJECT_ROOT / path).resolve()


@dataclass
class RealtimeExportState:
    start_time_s: float
    last_stats_time_s: float
    last_stats_frame_count: int = 0
    frame_count: int = 0
    sync_reject_count: int = 0
    duplicate_skip_count: int = 0
    write_ms_samples: list[float] = field(default_factory=list)
    last_step_tuple: tuple[int, ...] | None = None
    last_accepted_step_tuple: tuple[int, ...] | None = None
    last_accepted_timestamps: list[float] = field(default_factory=list)
    last_sync_delta_ms: float = 0.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Realtime native RealSense RGB-D export to a PhysTwin-compatible aligned case.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--case_name", type=str, default=None)
    parser.add_argument("--output_root", type=Path, default=_resolve_path("./data/different_types_real_time"))
    parser.add_argument("--calibrate_path", type=Path, default=_resolve_path("./calibrate.pkl"))
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--num-cam", type=int, default=DEFAULT_NUM_CAM)
    parser.add_argument("--serials", nargs="*", default=None)
    parser.add_argument("--duration-s", type=float, default=0.0)
    parser.add_argument("--stats-log-interval-s", type=float, default=1.0)
    parser.add_argument("--sync-tolerance-ms", type=float, default=50.0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove an existing output case before starting. Refuses to reuse a non-empty case by default.",
    )
    return parser


def _make_case_name(case_name: str | None) -> str:
    return case_name or datetime.now().strftime("%Y%m%d_%H%M%S")


def _prepare_case_dir(case_dir: Path, *, num_cameras: int, overwrite: bool) -> None:
    if case_dir.exists():
        if overwrite:
            shutil.rmtree(case_dir)
        elif any(case_dir.iterdir()):
            raise FileExistsError(f"Realtime output case already exists and is not empty: {case_dir}")
    for stream_name in ("color", "depth"):
        for camera_idx in range(num_cameras):
            (case_dir / stream_name / str(camera_idx)).mkdir(parents=True, exist_ok=True)


def _legacy_metadata(*, recording_metadata: dict[str, Any], frame_count: int) -> dict[str, Any]:
    end_step = int(frame_count) - 1 if int(frame_count) > 0 else -1
    metadata = {
        "intrinsics": recording_metadata["intrinsics"],
        "serial_numbers": recording_metadata["serial_numbers"],
        "fps": int(recording_metadata["fps"]),
        "WH": recording_metadata["WH"],
        "frame_num": int(frame_count),
        "start_step": 0,
        "end_step": end_step,
    }
    missing = [key for key in LEGACY_ALIGNED_METADATA_KEYS if key not in metadata]
    if missing:
        raise ValueError(f"Realtime metadata missing legacy keys: {missing}")
    return {key: metadata[key] for key in LEGACY_ALIGNED_METADATA_KEYS}


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload), encoding="utf-8")
    tmp_path.replace(path)


def _atomic_write_npy(path: Path, array: np.ndarray) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("wb") as handle:
        np.save(handle, array)
    tmp_path.replace(path)


def _atomic_write_png(path: Path, image: np.ndarray) -> None:
    import cv2

    tmp_path = path.with_name(f".{path.stem}.tmp{path.suffix}")
    ok = cv2.imwrite(str(tmp_path), image)
    if not ok:
        raise RuntimeError(f"Failed to write PNG frame: {path}")
    tmp_path.replace(path)


def _write_metadata(case_dir: Path, recording_metadata: dict[str, Any], frame_count: int) -> None:
    _atomic_write_json(case_dir / "metadata.json", _legacy_metadata(recording_metadata=recording_metadata, frame_count=frame_count))


def _write_normalized_calibration(case_dir: Path, calibrate_path: Path, recording_metadata: dict[str, Any]) -> None:
    transforms = _load_calibration_transforms(
        calibrate_path,
        serial_numbers=list(recording_metadata["serial_numbers"]),
        calibration_reference_serials=recording_metadata.get(
            "calibration_reference_serials",
            recording_metadata["serial_numbers"],
        ),
    )
    tmp_path = case_dir / ".calibrate.pkl.tmp"
    with tmp_path.open("wb") as handle:
        pickle.dump(transforms, handle)
    tmp_path.replace(case_dir / "calibrate.pkl")


def _validate_transform_matrix(matrix: Any, *, index: int) -> np.ndarray:
    item = np.asarray(matrix, dtype=np.float32)
    if item.shape != (4, 4):
        raise ValueError(f"Unsupported calibration transform shape at index {index}: {item.shape}")
    if not np.all(np.isfinite(item)):
        raise ValueError(f"Calibration transform at index {index} contains non-finite values.")
    expected_bottom = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    if not np.allclose(item[3], expected_bottom, atol=1e-4):
        raise ValueError(f"Calibration transform at index {index} has invalid homogeneous bottom row.")
    return item


def _coerce_transform_list(raw: Any) -> list[np.ndarray]:
    if isinstance(raw, np.ndarray):
        if raw.ndim != 3 or raw.shape[1:] != (4, 4):
            raise ValueError(f"Unsupported calibration ndarray shape: {raw.shape}")
        return [_validate_transform_matrix(item, index=idx) for idx, item in enumerate(raw)]
    if isinstance(raw, (list, tuple)):
        if not raw:
            raise ValueError("Calibration transform list is empty.")
        return [_validate_transform_matrix(item, index=idx) for idx, item in enumerate(raw)]
    raise ValueError(f"Unsupported calibrate.pkl object type: {type(raw).__name__}")


def _load_calibration_transforms(
    calibrate_path: Path,
    *,
    serial_numbers: list[str],
    calibration_reference_serials: list[str],
) -> list[np.ndarray]:
    with Path(calibrate_path).open("rb") as handle:
        transforms = _coerce_transform_list(pickle.load(handle))

    if len(calibration_reference_serials) != len(transforms):
        raise ValueError(
            "Calibration transform count does not match calibration_reference_serials length. "
            f"transform_count={len(transforms)}, calibration_reference_serials={len(calibration_reference_serials)}"
        )
    index_by_serial = {serial: idx for idx, serial in enumerate(calibration_reference_serials)}
    missing = [serial for serial in serial_numbers if serial not in index_by_serial]
    if missing:
        raise ValueError(f"Calibration serial mapping is incomplete. Missing serials: {missing}")
    return [transforms[index_by_serial[serial]] for serial in serial_numbers]


def _frame_step_tuple(observation: dict[int, dict[str, Any]], *, num_cameras: int) -> tuple[int, ...]:
    return tuple(int(observation[camera_idx]["step_idx"]) for camera_idx in range(num_cameras))


def _frame_timestamps(observation: dict[int, dict[str, Any]], *, num_cameras: int) -> list[float]:
    return [float(observation[camera_idx]["timestamp"]) for camera_idx in range(num_cameras)]


def _validate_observation(observation: dict[int, dict[str, Any]], *, num_cameras: int) -> None:
    for camera_idx in range(num_cameras):
        if camera_idx not in observation:
            raise RuntimeError(f"Missing camera {camera_idx} in realtime observation")
        for key in ("color", "depth", "timestamp", "step_idx"):
            if key not in observation[camera_idx]:
                raise RuntimeError(f"Missing {key!r} for camera {camera_idx} in realtime observation")


def _write_frame_set(case_dir: Path, observation: dict[int, dict[str, Any]], *, frame_idx: int, num_cameras: int) -> float:
    start_s = time.monotonic()
    for camera_idx in range(num_cameras):
        color = np.asarray(observation[camera_idx]["color"], dtype=np.uint8)
        depth = np.asarray(observation[camera_idx]["depth"], dtype=np.uint16)
        _atomic_write_png(case_dir / "color" / str(camera_idx) / f"{frame_idx}.png", color)
        _atomic_write_npy(case_dir / "depth" / str(camera_idx) / f"{frame_idx}.npy", depth)
    return (time.monotonic() - start_s) * 1000.0


def _build_stats_row(*, state: RealtimeExportState, now_s: float) -> dict[str, Any]:
    elapsed_s = max(now_s - state.start_time_s, 1e-9)
    window_elapsed_s = max(now_s - state.last_stats_time_s, 1e-9)
    window_frames = int(state.frame_count) - int(state.last_stats_frame_count)
    return {
        "aligned_frame_set_fps_window": float(window_frames / window_elapsed_s),
        "aligned_frame_set_fps_total": float(state.frame_count / elapsed_s),
        "frame_num": int(state.frame_count),
        "per_camera_step_idx": [] if state.last_accepted_step_tuple is None else list(state.last_accepted_step_tuple),
        "per_camera_timestamp": list(state.last_accepted_timestamps),
        "sync_delta_ms": float(state.last_sync_delta_ms),
        "sync_reject_count": int(state.sync_reject_count),
        "duplicate_skip_count": int(state.duplicate_skip_count),
        "write_ms_avg": float(np.mean(state.write_ms_samples)) if state.write_ms_samples else 0.0,
        "elapsed_s": float(elapsed_s),
    }


def _append_stats_row(stats_jsonl_path: Path, row: dict[str, Any]) -> None:
    with stats_jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def _print_stats(row: dict[str, Any]) -> None:
    print(
        "[realtime-align] "
        f"frames={row['frame_num']} "
        f"fps_window={row['aligned_frame_set_fps_window']:.2f} "
        f"fps_total={row['aligned_frame_set_fps_total']:.2f} "
        f"sync_ms={row['sync_delta_ms']:.2f} "
        f"rejects={row['sync_reject_count']} "
        f"duplicates={row['duplicate_skip_count']} "
        f"write_ms_avg={row['write_ms_avg']:.2f}",
        flush=True,
    )


def _finalize_summary(summary_path: Path, row: dict[str, Any]) -> None:
    _atomic_write_json(summary_path, row)


def run_realtime_export(
    args: argparse.Namespace,
    *,
    camera_system_factory: Callable[..., Any] | None = None,
    monotonic_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    calibrate_path = Path(args.calibrate_path).resolve()
    if not calibrate_path.is_file():
        raise FileNotFoundError(f"Missing calibrate.pkl required for realtime aligned export: {calibrate_path}")

    case_name = _make_case_name(args.case_name)
    output_root = Path(args.output_root).resolve()
    case_dir = output_root / case_name
    log_dir = output_root / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stats_jsonl_path = log_dir / f"{case_name}_stats.jsonl"
    summary_path = log_dir / f"{case_name}_summary.json"
    if bool(args.overwrite):
        stats_jsonl_path.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)

    if case_dir.exists() and any(case_dir.iterdir()) and not bool(args.overwrite):
        raise FileExistsError(f"Realtime output case already exists and is not empty: {case_dir}")

    if camera_system_factory is None:
        from qqtt.env import CameraSystem

        camera_system_factory = CameraSystem

    camera_system = camera_system_factory(
        WH=[int(args.width), int(args.height)],
        fps=int(args.fps),
        num_cam=int(args.num_cam),
        serial_numbers=args.serials if args.serials else None,
        capture_mode="rgbd",
        enable_keyboard_listener=False,
    )
    state = RealtimeExportState(start_time_s=monotonic_fn(), last_stats_time_s=monotonic_fn())
    final_row: dict[str, Any] | None = None

    try:
        recording_metadata = camera_system.build_recording_metadata()
        num_cameras = len(recording_metadata["serial_numbers"])
        if num_cameras != int(args.num_cam):
            raise RuntimeError(f"Camera count mismatch: expected {args.num_cam}, got {num_cameras}")

        _prepare_case_dir(case_dir, num_cameras=num_cameras, overwrite=bool(args.overwrite))
        _write_normalized_calibration(case_dir, calibrate_path, recording_metadata)
        _write_metadata(case_dir, recording_metadata, frame_count=0)

        duration_s = float(args.duration_s)
        stats_interval_s = max(float(args.stats_log_interval_s), 0.001)
        sync_tolerance_s = float(args.sync_tolerance_ms) / 1000.0
        deadline_s = state.start_time_s + duration_s if duration_s > 0 else None

        while True:
            now_s = monotonic_fn()
            if deadline_s is not None and now_s >= deadline_s:
                break

            observation = camera_system.get_observation()
            _validate_observation(observation, num_cameras=num_cameras)
            step_tuple = _frame_step_tuple(observation, num_cameras=num_cameras)
            if step_tuple == state.last_step_tuple:
                state.duplicate_skip_count += 1
                sleep_fn(0.001)
            else:
                state.last_step_tuple = step_tuple
                timestamps = _frame_timestamps(observation, num_cameras=num_cameras)
                sync_delta_s = max(timestamps) - min(timestamps)
                if sync_delta_s > sync_tolerance_s:
                    state.sync_reject_count += 1
                    state.last_sync_delta_ms = sync_delta_s * 1000.0
                    sleep_fn(0.001)
                else:
                    frame_idx = state.frame_count
                    write_ms = _write_frame_set(case_dir, observation, frame_idx=frame_idx, num_cameras=num_cameras)
                    state.write_ms_samples.append(write_ms)
                    state.frame_count += 1
                    state.last_accepted_step_tuple = step_tuple
                    state.last_accepted_timestamps = timestamps
                    state.last_sync_delta_ms = sync_delta_s * 1000.0
                    _write_metadata(case_dir, recording_metadata, frame_count=state.frame_count)

            now_s = monotonic_fn()
            if now_s - state.last_stats_time_s >= stats_interval_s:
                final_row = _build_stats_row(state=state, now_s=now_s)
                _append_stats_row(stats_jsonl_path, final_row)
                _print_stats(final_row)
                state.last_stats_time_s = now_s
                state.last_stats_frame_count = state.frame_count

        final_row = _build_stats_row(state=state, now_s=monotonic_fn())
        _append_stats_row(stats_jsonl_path, final_row)
        _finalize_summary(summary_path, final_row)
        _write_metadata(case_dir, recording_metadata, frame_count=state.frame_count)
        return {
            "case_dir": str(case_dir),
            "stats_jsonl_path": str(stats_jsonl_path),
            "summary_path": str(summary_path),
            "stats": final_row,
        }
    except KeyboardInterrupt:
        final_row = _build_stats_row(state=state, now_s=monotonic_fn())
        _append_stats_row(stats_jsonl_path, final_row)
        _finalize_summary(summary_path, final_row)
        raise
    finally:
        if final_row is None:
            final_row = _build_stats_row(state=state, now_s=monotonic_fn())
            _finalize_summary(summary_path, final_row)
        realsense = getattr(camera_system, "realsense", None)
        if realsense is not None:
            realsense.stop()
        shm_manager = getattr(camera_system, "shm_manager", None)
        if shm_manager is not None:
            try:
                shm_manager.shutdown()
            except Exception:
                pass


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_realtime_export(args)
    except KeyboardInterrupt:
        print("[realtime-align] interrupted; summary was written before shutdown.", flush=True)
        return 130
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
