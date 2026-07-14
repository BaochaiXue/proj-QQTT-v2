"""Input RGB timeline, playback state, chunk timing/latency, and throughput.

Extracted from ``visualization/visualize_track.py`` as part of a behavior-preserving
file split. Low-level module: depends only on ``viz_camera_model``.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from demo_v6_2.visualization.viz_camera_model import (
    _require_cv2,
    load_pickle,
    normalize_online_dir,
    read_json,
)


@dataclass(frozen=True)
class InputRgbFrame:
    """One RGB input frame plus source-frame timing metadata."""

    seq: int
    image_bgr: np.ndarray
    path: Path | None
    source_frame_index: int | None
    source_timestamp_s: float | None


@dataclass
class OutputStreamPlaybackCursor:
    """Small playback state machine for live output-frame progression."""

    fps: float
    output_index: int = 0
    last_step_s: float | None = None

    def advance(self, *, latest: int, now_s: float, paused: bool) -> int:
        """Advance playback by elapsed time while staying within loaded output.

        Advances at most one frame per call, so the caller's poll cadence must
        be at least as fast as the playback FPS to keep up.
        """
        latest_index = max(0, int(latest))
        self.output_index = min(max(int(self.output_index), 0), latest_index)
        now = float(now_s)
        if self.last_step_s is None:
            self.last_step_s = now
            return int(self.output_index)
        if paused or self.output_index >= latest_index:
            self.last_step_s = now
            return int(self.output_index)
        period_s = 1.0 / max(float(self.fps), 1e-6)
        elapsed_s = max(0.0, now - float(self.last_step_s))
        if elapsed_s + 1e-9 < period_s:
            return int(self.output_index)
        self.output_index = min(latest_index, int(self.output_index) + 1)
        self.last_step_s = now
        return int(self.output_index)

    def seek(self, index: int, *, latest: int, now_s: float | None = None) -> int:
        """Move playback to a bounded output-frame index."""
        self.output_index = min(max(int(index), 0), max(0, int(latest)))
        if now_s is not None:
            self.last_step_s = float(now_s)
        return int(self.output_index)


@dataclass
class CameraToFinalDataFpsMeter:
    """Estimate online publish throughput from newly appended output frames."""

    _last_update_s: float | None = None
    _fps: float | None = None

    def seed(self, fps: float | None) -> float | None:
        """Seed the meter with a historical FPS estimate when available."""
        if fps is None:
            return self._fps
        value = float(fps)
        if not math.isfinite(value) or value <= 0.0:
            return self._fps
        self._fps = value
        return self._fps

    def update(self, *, appended_frames: int, now_s: float) -> float | None:
        """Update the FPS estimate from the number of newly appended frames."""
        count = int(appended_frames)
        if count <= 0:
            return self._fps
        now = float(now_s)
        if self._last_update_s is None:
            self._last_update_s = now
            return None
        elapsed = now - float(self._last_update_s)
        self._last_update_s = now
        if elapsed <= 1e-9:
            return self._fps
        self._fps = float(count) / elapsed
        return self._fps


def load_fake_input_frame_total(capture_dir: str | Path | None) -> int | None:
    """Return the expected fake-live RGB frame count when metadata provides it."""
    if capture_dir is None:
        return None
    metadata = read_json(Path(capture_dir).expanduser() / "metadata.json")
    if str(metadata.get("input_source") or "") != "fake-live":
        return None
    try:
        total = int(metadata.get("recording_frame_count"))
    except (TypeError, ValueError):
        return None
    return total if total > 0 else None


# --- Input RGB timeline loading ----------------------------------------------


def _input_rgb_path_from_row(row: Mapping[str, Any], *, capture_dir: Path) -> Path | None:
    """Return the input RGB path from row."""
    value = row.get("input_rgb_path")
    if value is not None and str(value).strip():
        path = Path(str(value))
        return path if path.is_absolute() else capture_dir / path
    seq = row.get("seq")
    if seq is not None:
        try:
            seq_int = int(seq)
        except (TypeError, ValueError):
            seq_int = -1
        if seq_int >= 0:
            for directory in ("input_rgb", "rgb"):
                path = capture_dir / directory / f"{seq_int:06d}.png"
                if path.is_file():
                    return path
    return None


def _input_rgb_frame_from_row(row: Mapping[str, Any], *, capture_dir: Path) -> InputRgbFrame | None:
    """Return the input RGB frame from row."""
    cv2 = _require_cv2()
    path = _input_rgb_path_from_row(row, capture_dir=capture_dir)
    if path is None or not path.is_file():
        return None
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    try:
        seq = int(row.get("seq", 0))
    except (TypeError, ValueError):
        seq = 0
    source_frame_index = row.get("source_frame_index")
    try:
        source_frame_index = None if source_frame_index is None else int(source_frame_index)
    except (TypeError, ValueError):
        source_frame_index = None
    source_timestamp_s = row.get("source_timestamp_s")
    try:
        source_timestamp_s = None if source_timestamp_s is None else float(source_timestamp_s)
    except (TypeError, ValueError):
        source_timestamp_s = None
    return InputRgbFrame(
        seq=seq,
        image_bgr=np.ascontiguousarray(image, dtype=np.uint8),
        path=path,
        source_frame_index=source_frame_index,
        source_timestamp_s=source_timestamp_s,
    )


def _read_jsonl_rows(path: str | Path) -> list[dict[str, Any]]:
    """Read every non-blank row of a .jsonl file as a dict.

    Blank lines are skipped and each remaining line is parsed with ``json.loads``;
    rows that fail to parse (or are not dict-convertible) are skipped. A missing
    file yields an empty list.
    """
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    rows: list[dict[str, Any]] = []
    for line in lines:
        text = line.strip()
        if not text:
            continue
        try:
            rows.append(dict(json.loads(text)))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    return rows


def load_latest_input_rgb_frame(timeline_path: str | Path, *, capture_dir: str | Path) -> InputRgbFrame | None:
    """Load the newest RGB input frame referenced by an input timeline."""
    capture_path = Path(capture_dir).expanduser()
    for row in reversed(_read_jsonl_rows(Path(timeline_path).expanduser())):
        frame = _input_rgb_frame_from_row(row, capture_dir=capture_path)
        if frame is not None:
            return frame
    return None


def load_input_rgb_frames(timeline_path: str | Path, *, capture_dir: str | Path) -> list[InputRgbFrame]:
    """Load all RGB input frames referenced by an input timeline."""
    capture_path = Path(capture_dir).expanduser()
    frames: list[InputRgbFrame] = []
    for row in _read_jsonl_rows(Path(timeline_path).expanduser()):
        frame = _input_rgb_frame_from_row(row, capture_dir=capture_path)
        if frame is not None:
            frames.append(frame)
    return frames


def load_input_rgb_background_paths(
    timeline_path: str | Path,
    *,
    capture_dir: str | Path,
) -> dict[int, Path]:
    """Map original source-frame ids to fake-live RGB images for output export."""
    capture_path = Path(capture_dir).expanduser()
    paths: dict[int, Path] = {}
    for row in _read_jsonl_rows(Path(timeline_path).expanduser()):
        try:
            source_frame_index = int(row["source_frame_index"])
        except (KeyError, TypeError, ValueError):
            continue
        path = _input_rgb_path_from_row(row, capture_dir=capture_path)
        if path is None or not path.is_file():
            continue
        paths[source_frame_index] = path
    return paths


# --- Chunk frame timing and latency ------------------------------------------


def _chunk_frame_count(chunk: Mapping[str, Any]) -> int:
    """Return the chunk frame count."""
    for key in ("object_points", "controller_points"):
        value = chunk.get(key)
        if value is not None:
            return int(np.asarray(value).shape[0])
    return 0


def _source_frame_for_chunk_frame(chunk: Mapping[str, Any], local_frame: int) -> int:
    """Return the source frame for chunk frame."""
    source_indices = chunk.get("source_frame_indices")
    if source_indices is not None:
        try:
            return int(source_indices[int(local_frame)])
        except (IndexError, TypeError, ValueError):
            pass
    return int(chunk.get("start_frame", 0)) + int(local_frame)


def format_input_display_latency(latency_s: float | None) -> str:
    """Format an input-display latency value for the viewer overlay."""
    if latency_s is None:
        return "input->display --"
    try:
        value = float(latency_s)
    except (TypeError, ValueError):
        return "input->display --"
    if not math.isfinite(value) or value < 0.0:
        return "input->display --"
    rounded = Decimal(str(value + 1e-9)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return f"input->display {rounded:.2f}s"


def _source_time_for_chunk_frame(
    chunk: Mapping[str, Any],
    local_frame: int,
    *,
    fps: float,
    allow_frame_index_fallback: bool = False,
) -> float | None:
    """Return the source time for chunk frame."""
    source_timestamps = chunk.get("source_timestamps_s")
    if source_timestamps is not None:
        try:
            value = float(source_timestamps[int(local_frame)])
            if math.isfinite(value):
                return value
        except (IndexError, TypeError, ValueError):
            pass
    source_indices = chunk.get("source_frame_indices")
    if allow_frame_index_fallback and source_indices is not None and math.isfinite(float(fps)) and float(fps) > 0.0:
        try:
            return float(source_indices[int(local_frame)]) / float(fps)
        except (IndexError, TypeError, ValueError):
            pass
    return None


def source_time_input_display_latency_s(
    *,
    input_frame: InputRgbFrame | None,
    output_frames: Sequence[tuple[Mapping[str, Any], int]],
    output_index: int,
    fps: float,
) -> float | None:
    """Estimate latency between current input source time and displayed output."""
    if input_frame is None or input_frame.source_timestamp_s is None or not output_frames:
        return None
    idx = min(max(int(output_index), 0), len(output_frames) - 1)
    chunk, local_frame = output_frames[idx]
    output_source_time = _source_time_for_chunk_frame(chunk, int(local_frame), fps=float(fps))
    if output_source_time is None:
        return None
    origin_chunk, origin_local_frame = output_frames[0]
    origin_source_time = _source_time_for_chunk_frame(origin_chunk, int(origin_local_frame), fps=float(fps))
    if origin_source_time is None:
        return None
    try:
        input_source_time = float(input_frame.source_timestamp_s)
        output_time = float(output_source_time)
        origin_time = float(origin_source_time)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (input_source_time, output_time, origin_time)):
        return None
    input_elapsed_s = input_source_time - origin_time
    output_elapsed_s = output_time - origin_time
    latency_s = input_elapsed_s - output_elapsed_s
    if latency_s < 0.0 or not math.isfinite(latency_s):
        return None
    return float(latency_s)


# --- Online chunk discovery and throughput estimation ------------------------


def _chunk_sort_key(path: Path) -> tuple[int, str]:
    # Chunk files are named chunk_<id>.pkl; sort numerically by id.
    """Return the chunk sort key."""
    stem = path.stem
    try:
        return (int(stem.rsplit("_", 1)[1]), path.name)
    except (IndexError, ValueError):
        return (0, path.name)


def list_available_chunk_paths(online_dir: Path, *, start_chunk: int) -> list[Path]:
    """List committed chunk pickle files at or after the requested chunk id."""
    chunks_dir = normalize_online_dir(online_dir) / "chunks"
    paths = sorted(chunks_dir.glob("chunk_*.pkl"), key=_chunk_sort_key)
    start = int(start_chunk)
    return [path for path in paths if _chunk_sort_key(path)[0] >= start]


def _run_root_for_online_dir(online_dir: Path) -> Path | None:
    """Return the run root that owns an online data directory."""
    path = normalize_online_dir(online_dir)
    if path.name != "online_data":
        return None
    return path.parent


def _camera_to_final_data_fps_from_run_manifest(online_dir: Path) -> float | None:
    """Recover publish FPS from the run-level summary when available."""
    run_root = _run_root_for_online_dir(online_dir)
    if run_root is None or not run_root.is_dir():
        return None
    candidates = [run_root / "run_summary.json"]
    for manifest_path in candidates:
        if not manifest_path.is_file():
            continue
        manifest = read_json(manifest_path)
        online_dir_value = str(manifest.get("online_dir", ""))
        if online_dir_value and Path(online_dir_value).name != "online_data":
            continue
        intervals = []
        for value in manifest.get("steady_publish_intervals_s", []) or []:
            try:
                interval = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(interval) and interval > 1e-9:
                intervals.append(interval)
        try:
            chunk_frame_count = int(manifest.get("chunk_frame_count", 0))
        except (TypeError, ValueError):
            chunk_frame_count = 0
        if intervals and chunk_frame_count > 0:
            return float(len(intervals) * chunk_frame_count) / float(sum(intervals))
    return None


def _camera_to_final_data_fps_from_chunk_mtimes(online_dir: Path, *, start_chunk: int) -> float | None:
    """Fallback throughput estimate based on chunk file mtimes.

    The first chunk's frames accumulated before its commit time, so only
    frames committed after the first mtime are divided by the mtime span.
    """
    chunk_infos: list[tuple[float, int]] = []
    for chunk_path in list_available_chunk_paths(online_dir, start_chunk=start_chunk):
        try:
            mtime_s = float(chunk_path.stat().st_mtime)
        except OSError:
            continue
        try:
            chunk = dict(load_pickle(chunk_path))
        except Exception:
            continue
        frame_count = _chunk_frame_count(chunk)
        if frame_count > 0 and math.isfinite(mtime_s):
            chunk_infos.append((mtime_s, int(frame_count)))
    if len(chunk_infos) < 2:
        return None
    chunk_infos.sort(key=lambda item: item[0])
    elapsed_s = float(chunk_infos[-1][0] - chunk_infos[0][0])
    if elapsed_s <= 1e-9:
        return None
    frames_after_first_commit = sum(frame_count for _mtime_s, frame_count in chunk_infos[1:])
    if frames_after_first_commit <= 0:
        return None
    return float(frames_after_first_commit) / elapsed_s


def estimate_historical_camera_to_final_data_fps(online_dir: Path, *, start_chunk: int) -> float | None:
    """Estimate camera->final_data throughput.

    This is used when reopening already committed chunks.
    """
    manifest_fps = _camera_to_final_data_fps_from_run_manifest(online_dir)
    if manifest_fps is not None:
        return manifest_fps
    return _camera_to_final_data_fps_from_chunk_mtimes(online_dir, start_chunk=start_chunk)


def _resolve_capture_dir(args: argparse.Namespace) -> Path | None:
    """Resolve the capture directory for RGB timeline lookup."""
    value = getattr(args, "capture_dir", None)
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(value).expanduser()


def _resolve_input_rgb_timeline(args: argparse.Namespace, *, capture_dir: Path | None) -> Path | None:
    """Resolve input RGB timeline."""
    value = getattr(args, "input_rgb_timeline", None)
    if value is not None and str(value).strip():
        return Path(value).expanduser()
    if capture_dir is None:
        return None
    return capture_dir / "input_frames.jsonl"
