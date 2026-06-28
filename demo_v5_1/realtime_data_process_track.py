"""Convert Demo v5 headless capture rows into realtime data_process_sam3d chunks.

The camera process appends ``frames.jsonl`` and prepared per-frame NPZ payloads.
This bridge tails that stream, waits for the shape prior only at chunk materialize
time, and publishes both per-window cases and the online aggregate output used
by realtime_phystwin.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np
from PIL import Image

from demo_v5_1.data_process_chunk_writer import (
    DataProcessChunk,
    write_data_process_chunk_case,
)
from demo_v5_1.chunked_final_data_output import ChunkedFinalDataWriter
from demo_v5_1.data_process_schema import normalize_data_process_keys
from qqtt.demo.pcd_postprocess import _detect_radius_outlier_indices
from qqtt.demo import phystwin_strict_product as strict


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_jsonl_from_offset(path: Path, offset: int) -> tuple[list[dict[str, Any]], int]:
    """Read only newly appended complete JSONL rows.

    ``frames.jsonl`` is written by another process. If the writer is in the
    middle of a row, return the rows that were already complete and leave the
    offset at the start of the partial line for the next poll.
    """
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows, int(offset)
    with path.open("r", encoding="utf-8") as handle:
        handle.seek(max(0, int(offset)))
        while True:
            row_offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            stripped = line.strip()
            if stripped:
                try:
                    rows.append(json.loads(stripped))
                except json.JSONDecodeError:
                    return rows, int(row_offset)
        return rows, int(handle.tell())


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _count_jsonl_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _online_total_frame_count(
    frames_path: Path,
    *,
    chunk_size: int,
    max_chunks: int | None,
) -> int | None:
    if int(chunk_size) <= 0:
        return None
    row_count = _count_jsonl_rows(frames_path)
    full_chunks = row_count // int(chunk_size)
    if max_chunks is not None:
        full_chunks = min(full_chunks, int(max_chunks))
    return int(full_chunks * int(chunk_size))


def _relative_wall_s(origin_s: float) -> float:
    return float(time.monotonic() - float(origin_s))


def _complete_chunk_backlog(frames_path: Path, *, chunk_size: int, published_chunk_count: int) -> int:
    if int(chunk_size) <= 0:
        return 0
    complete_chunks = _count_jsonl_rows(frames_path) // int(chunk_size)
    return max(0, int(complete_chunks) - int(published_chunk_count))


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _row_source_frame_index(row: Mapping[str, Any], fallback: int) -> int:
    value = _optional_int(row.get("source_frame_index"))
    if value is not None:
        return int(value)
    value = _optional_int(row.get("seq"))
    if value is not None:
        return int(value)
    return int(fallback)


def _rows_source_frame_indices(rows: Sequence[Mapping[str, Any]], *, fallback_start: int) -> list[int]:
    return [
        _row_source_frame_index(row, int(fallback_start) + offset)
        for offset, row in enumerate(rows)
    ]


def _rows_source_timestamps(rows: Sequence[Mapping[str, Any]]) -> list[float] | None:
    values: list[float] = []
    for row in rows:
        value = _optional_float(row.get("source_timestamp_s"))
        if value is None:
            return None
        values.append(float(value))
    return values


def _row_ready_for_realtime_chunk_start(row: Mapping[str, Any]) -> bool:
    controller_points = _optional_int(row.get("controller_point_count"))
    if controller_points is not None and int(controller_points) < 30:
        return False
    object_points = _optional_int(row.get("object_point_count"))
    if object_points is not None and int(object_points) <= 0:
        return False
    return True


def _trim_warmup_delayed_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Drop leading warmup rows before chunking realtime output.

    Demo v5 writes ``input_frames.jsonl`` from camera start, but ``frames.jsonl``
    is the data_process output stream. With shape-prior warmup, the first strict
    pair can be source frame 0 processed many seconds late; the next row then
    jumps to the realtime source frame after warmup. That delayed row is useful
    for shape-prior diagnostics but must not define chunk_000000. Live RealSense
    can also emit one strict pair before color-aligned PCD is ready; that row has
    masks but zero controller/object points and must not anchor controller FPS.
    """
    trimmed = [dict(row) for row in rows]
    skipped = 0
    while trimmed and not _row_ready_for_realtime_chunk_start(trimmed[0]):
        skipped += 1
        trimmed.pop(0)
    while len(trimmed) >= 2:
        first = trimmed[0]
        second = trimmed[1]
        first_source = _optional_int(first.get("source_frame_index"))
        second_source = _optional_int(second.get("source_frame_index"))
        if first_source is None or second_source is None:
            break
        source_gap = int(second_source) - int(first_source)
        if source_gap <= 1:
            break

        following_gap = None
        if len(trimmed) >= 3:
            third_source = _optional_int(trimmed[2].get("source_frame_index"))
            if third_source is not None:
                following_gap = int(third_source) - int(second_source)
        expected_gap = max(2, int(following_gap) if following_gap is not None and following_gap > 0 else 2)

        startup_hold_s = _optional_float(first.get("startup_hold_s")) or _optional_float(second.get("startup_hold_s")) or 0.0
        latency_ms = _optional_float(first.get("pipeline_latency_ms")) or 0.0
        first_time = _optional_float(first.get("source_timestamp_s"))
        second_time = _optional_float(second.get("source_timestamp_s"))
        timestamp_gap_s = 0.0 if first_time is None or second_time is None else float(second_time - first_time)

        looks_like_warmup_hold = (
            startup_hold_s > 0.0
            and latency_ms >= max(1000.0, startup_hold_s * 500.0)
            and timestamp_gap_s >= max(1.0, startup_hold_s * 0.5)
        )
        looks_like_stream_jump = source_gap > max(10, expected_gap * 3)
        if not (looks_like_warmup_hold and looks_like_stream_jump):
            break
        skipped += 1
        trimmed.pop(0)
        while trimmed and not _row_ready_for_realtime_chunk_start(trimmed[0]):
            skipped += 1
            trimmed.pop(0)
    return trimmed, int(skipped)


@dataclass
class _WarmupStartFilterState:
    pending_rows: list[dict[str, Any]] = field(default_factory=list)
    resolved: bool = False
    skipped_rows: int = 0


def _filter_warmup_start_rows(
    state: _WarmupStartFilterState,
    rows: Sequence[Mapping[str, Any]],
    *,
    capture_finished: bool,
) -> list[dict[str, Any]]:
    """Hold the first live rows until warmup-delayed source rows can be trimmed."""
    if state.resolved:
        return [dict(row) for row in rows]
    state.pending_rows.extend(dict(row) for row in rows)
    trimmed, skipped = _trim_warmup_delayed_rows(state.pending_rows)
    if skipped:
        state.skipped_rows += int(skipped)
        state.pending_rows = trimmed
    if len(state.pending_rows) < 2 and not bool(capture_finished):
        return []
    state.resolved = True
    output = list(state.pending_rows)
    state.pending_rows = []
    return output


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _load_mask_frame(path: Path) -> dict[str, np.ndarray]:
    payload = np.load(path, allow_pickle=False)
    frame = {
        "object": np.asarray(payload["object_mask"], dtype=bool),
        "controller": np.asarray(payload["controller_mask"], dtype=bool),
    }
    if "hand_a_mask" in payload:
        frame["hand_a"] = np.asarray(payload["hand_a_mask"], dtype=bool)
    if "hand_b_mask" in payload:
        frame["hand_b"] = np.asarray(payload["hand_b_mask"], dtype=bool)
    return strict.normalize_processed_mask_frame(frame)


def _apply_depth_validity_to_mask_frame(
    frame: Mapping[str, np.ndarray],
    depth_m: np.ndarray,
) -> dict[str, np.ndarray]:
    depth = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.0)
    normalized = strict.normalize_processed_mask_frame(frame)
    filtered: dict[str, np.ndarray] = {}
    for key, mask in normalized.items():
        arr = np.asarray(mask, dtype=bool)
        if arr.shape != valid.shape:
            raise ValueError(f"mask {key!r} shape {arr.shape} does not match depth shape {valid.shape}")
        filtered[key] = np.ascontiguousarray(arr & valid, dtype=bool)
    return strict.normalize_processed_mask_frame(filtered)


def _apply_radius_outlier_to_mask_frame(
    frame: Mapping[str, np.ndarray],
    points_grid: np.ndarray,
    *,
    enabled: bool,
    radius_m: float,
    nb_points: int,
) -> dict[str, np.ndarray]:
    normalized = strict.normalize_processed_mask_frame(frame)
    if not bool(enabled):
        return normalized
    grid = np.asarray(points_grid, dtype=np.float32)
    if grid.ndim == 4:
        grid = grid[0]
    if grid.ndim != 3 or grid.shape[-1] != 3:
        raise ValueError(f"points_grid must have shape H,W,3 or 1,H,W,3; got {grid.shape}")

    filtered = {key: np.asarray(value, dtype=bool).copy() for key, value in normalized.items()}
    for key in ("object", "controller"):
        mask = filtered[key]
        if mask.shape != grid.shape[:2]:
            raise ValueError(f"mask {key!r} shape {mask.shape} does not match points grid {grid.shape[:2]}")
        yy, xx = np.nonzero(mask)
        if len(yy) == 0:
            continue
        points = grid[yy, xx]
        point_valid = np.isfinite(points).all(axis=1) & (np.linalg.norm(points, axis=1) > 1e-9)
        if not np.all(point_valid):
            mask[yy[~point_valid], xx[~point_valid]] = False
        if not np.any(point_valid):
            continue
        valid_indices = np.flatnonzero(point_valid)
        result = _detect_radius_outlier_indices(
            points[point_valid],
            radius_m=float(radius_m),
            nb_points=int(nb_points),
        )
        keep_sub = np.zeros((len(valid_indices),), dtype=bool)
        keep_sub[np.asarray(result["inlier_indices"], dtype=np.int64)] = True
        remove_indices = valid_indices[~keep_sub]
        if len(remove_indices):
            mask[yy[remove_indices], xx[remove_indices]] = False
        filtered[key] = mask
    return strict.normalize_processed_mask_frame(filtered)


def _depth_path(capture_dir: Path, row: Mapping[str, Any]) -> Path:
    if "depth_color_m_path" in row:
        return capture_dir / str(row["depth_color_m_path"])
    if "ffs_depth_path" in row:
        return capture_dir / str(row["ffs_depth_path"])
    raise ValueError("headless capture frame is missing depth_color_m_path or legacy ffs_depth_path")


def _intrinsics_matrix(metadata: Mapping[str, Any]) -> np.ndarray:
    intrinsics = metadata.get("intrinsics")
    if isinstance(intrinsics, Mapping):
        return np.array(
            [
                [float(intrinsics["fx"]), 0.0, float(intrinsics["cx"])],
                [0.0, float(intrinsics["fy"]), float(intrinsics["cy"])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
    arr = np.asarray(intrinsics, dtype=np.float32)
    if arr.shape == (1, 3, 3):
        return np.ascontiguousarray(arr[0], dtype=np.float32)
    return np.ascontiguousarray(arr.reshape(3, 3), dtype=np.float32)


def _camera_to_world(metadata: Mapping[str, Any]) -> np.ndarray:
    value = metadata.get("camera_to_world_c2w")
    if value is None:
        return np.eye(4, dtype=np.float32)
    return np.ascontiguousarray(np.asarray(value, dtype=np.float32).reshape(4, 4), dtype=np.float32)


def _full_tracks_and_visibility(trajectory: np.lib.npyio.NpzFile, query_count: int) -> tuple[np.ndarray, np.ndarray]:
    if "all_tracks_yx" in trajectory.files:
        tracks = np.asarray(trajectory["all_tracks_yx"], dtype=np.float32).reshape(-1, 2)
        vis_key = "all_tracker_visibility" if "all_tracker_visibility" in trajectory.files else "visibility"
        visibility = np.asarray(trajectory[vis_key], dtype=bool).reshape(-1)
        if tracks.shape[0] != int(query_count) or visibility.shape[0] != int(query_count):
            raise ValueError("all_tracks_yx/all_tracker_visibility must match query_points_yx length")
        return np.ascontiguousarray(tracks, dtype=np.float32), np.ascontiguousarray(visibility, dtype=bool)

    track_key = "tracks_yx"
    vis_key = "visibility"
    if track_key not in trajectory.files or vis_key not in trajectory.files:
        raise ValueError("trajectory payload requires all_tracks_yx or tracks_yx plus visibility")
    active_tracks = np.asarray(trajectory[track_key], dtype=np.float32).reshape(-1, 2)
    active_visibility = np.asarray(trajectory[vis_key], dtype=bool).reshape(-1)
    if active_tracks.shape[0] != active_visibility.shape[0]:
        raise ValueError("tracks_yx and visibility must have matching active query count")
    if active_tracks.shape[0] == int(query_count):
        return (
            np.ascontiguousarray(active_tracks, dtype=np.float32),
            np.ascontiguousarray(active_visibility, dtype=bool),
        )
    if "query_indices" not in trajectory.files:
        raise ValueError("sparse tracks_yx payload requires query_indices to expand to full query count")
    indices = np.asarray(trajectory["query_indices"], dtype=np.int64).reshape(-1)
    if indices.shape[0] != active_tracks.shape[0]:
        raise ValueError("query_indices length must match active tracks_yx length")
    if np.any(indices < 0) or np.any(indices >= int(query_count)):
        raise ValueError("query_indices contains out-of-range values")
    tracks = np.zeros((int(query_count), 2), dtype=np.float32)
    visibility = np.zeros((int(query_count),), dtype=bool)
    tracks[indices] = active_tracks
    visibility[indices] = active_visibility
    return np.ascontiguousarray(tracks, dtype=np.float32), np.ascontiguousarray(visibility, dtype=bool)


def _shape_points_from_capture(
    capture_dir: Path,
    metadata: Mapping[str, Any],
    *,
    surface_points: np.ndarray | None,
    interior_points: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve surface/interior structure points for a chunk.

    Explicit NPY overrides are treated as authoritative. Otherwise the latest
    capture metadata points at the async SAM3D result produced by the camera
    process.
    """
    if surface_points is not None or interior_points is not None:
        return (
            np.empty((0, 3), dtype=np.float64)
            if surface_points is None
            else np.ascontiguousarray(np.asarray(surface_points, dtype=np.float64).reshape(-1, 3)),
            np.empty((0, 3), dtype=np.float64)
            if interior_points is None
            else np.ascontiguousarray(np.asarray(interior_points, dtype=np.float64).reshape(-1, 3)),
        )
    shape_path = metadata.get("shape_prior_path")
    if shape_path:
        payload = np.load(capture_dir / str(shape_path), allow_pickle=False)
        if "surface_points_m" in payload.files or "interior_points_m" in payload.files:
            return (
                np.empty((0, 3), dtype=np.float64)
                if "surface_points_m" not in payload.files
                else np.ascontiguousarray(np.asarray(payload["surface_points_m"], dtype=np.float64).reshape(-1, 3)),
                np.empty((0, 3), dtype=np.float64)
                if "interior_points_m" not in payload.files
                else np.ascontiguousarray(np.asarray(payload["interior_points_m"], dtype=np.float64).reshape(-1, 3)),
            )
        points = np.ascontiguousarray(np.asarray(payload["points_m"], dtype=np.float64).reshape(-1, 3))
        return points, np.empty((0, 3), dtype=np.float64)
    return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)


def _has_shape_points(surface_points: np.ndarray, interior_points: np.ndarray) -> bool:
    return int(np.asarray(surface_points).reshape(-1, 3).shape[0]) + int(np.asarray(interior_points).reshape(-1, 3).shape[0]) > 0


def _shape_prior_terminal_error(metadata: Mapping[str, Any]) -> str | None:
    status = str(metadata.get("shape_prior_status") or "").strip().lower()
    if status not in {"failed", "unavailable", "disabled"}:
        return None
    detail = metadata.get("shape_prior_error") or metadata.get("error") or "no surface/interior points became ready"
    return f"shape prior {status}: {detail}"


def _read_json_file_stable(
    path: Path,
    *,
    deadline_s: float,
    poll_interval_s: float,
) -> Mapping[str, Any]:
    last_error: Exception | None = None
    while True:
        try:
            text = path.read_text(encoding="utf-8")
            if not text.strip():
                raise json.JSONDecodeError("empty JSON file", text, 0)
            payload = json.loads(text)
            if not isinstance(payload, Mapping):
                raise ValueError(f"{path} must contain a JSON object")
            return payload
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            if time.monotonic() >= float(deadline_s):
                raise RuntimeError(f"timed out waiting for stable JSON metadata at {path}") from last_error
            time.sleep(max(0.0, float(poll_interval_s)))


def _shape_points_for_chunk(
    capture: Path,
    *,
    surface_points: np.ndarray | None,
    interior_points: np.ndarray | None,
    require_shape_prior: bool,
    shape_prior_wait_timeout_s: float,
    capture_finished: Callable[[], bool] | None,
    before_poll: Callable[[], None] | None,
    poll_interval_s: float,
) -> tuple[Mapping[str, Any], np.ndarray, np.ndarray]:
    """Wait until required shape-prior points are available for final_data.

    Demo v5 keeps capture realtime, but ``final_data.pkl`` must contain
    structure points when shape-prior warmup is enabled. Waiting happens here,
    after a source window has closed, not inside the camera/tracker loop.
    """
    explicit_points = surface_points is not None or interior_points is not None
    deadline = time.monotonic() + max(0.0, float(shape_prior_wait_timeout_s))
    while True:
        metadata = _read_json_file_stable(
            capture / "metadata.json",
            deadline_s=deadline,
            poll_interval_s=float(poll_interval_s),
        )
        shape_surface, shape_interior = _shape_points_from_capture(
            capture,
            metadata,
            surface_points=surface_points,
            interior_points=interior_points,
        )
        if explicit_points or not bool(require_shape_prior) or _has_shape_points(shape_surface, shape_interior):
            return metadata, shape_surface, shape_interior
        terminal_error = _shape_prior_terminal_error(metadata)
        if terminal_error is not None:
            raise RuntimeError(terminal_error)
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "shape prior is required for Demo v5 final_data chunks, but no surface/interior points became ready"
            )
        if before_poll is not None:
            before_poll()
        if capture_finished is not None and capture_finished() and time.monotonic() >= deadline:
            raise RuntimeError("capture finished before required shape prior became ready")
        time.sleep(max(0.0, float(poll_interval_s)))


def _controller_track_manifest_fields(track_process_data: Mapping[str, Any]) -> dict[str, Any]:
    track_process_data = normalize_data_process_keys(track_process_data)
    if "controller_track_query_indices" not in track_process_data:
        controller_points = np.asarray(track_process_data.get("controller_points", np.empty((0, 0, 3))))
        return {
            "controller_track_selection_mode": "independent_fps",
            "controller_track_count": int(controller_points.shape[1] if controller_points.ndim >= 2 else 0),
        }
    query_indices = np.asarray(track_process_data["controller_track_query_indices"], dtype=np.int64).reshape(-1)
    active_indices = np.asarray(
        track_process_data.get("controller_track_active_query_indices", query_indices),
        dtype=np.int64,
    ).reshape(-1)
    statuses = np.asarray(track_process_data.get("controller_track_status", []), dtype=str).reshape(-1)
    payload = {
        "controller_track_selection_mode": "streaming_stable",
        "controller_track_count": int(len(query_indices)),
        "controller_track_query_indices": [int(value) for value in query_indices.tolist()],
        "controller_track_active_query_indices": [int(value) for value in active_indices.tolist()],
        "controller_track_direct_count": int(np.count_nonzero(statuses == "direct")),
        "controller_track_recovered_count": int(np.count_nonzero(statuses == "recovered")),
        "controller_track_revived_count": int(np.count_nonzero((statuses == "revived") | (statuses == "recovered"))),
        "controller_track_fallback_count": int(np.count_nonzero(statuses == "fallback")),
        "controller_track_missing_count": int(np.count_nonzero(statuses == "missing")),
        "controller_track_status": [str(value) for value in statuses.tolist()],
    }
    if "controller_track_confidence" in track_process_data:
        confidence = np.asarray(track_process_data["controller_track_confidence"], dtype=np.float32)
        payload["controller_track_mean_confidence"] = float(np.mean(confidence)) if confidence.size else 1.0
        payload["controller_track_low_confidence_ratio"] = (
            float(np.count_nonzero(confidence < 0.25) / confidence.size) if confidence.size else 0.0
        )
    if "controller_track_mode" in track_process_data:
        modes = np.asarray(track_process_data["controller_track_mode"], dtype=str)
        bundle_recovered = np.char.find(modes.astype(str), "bundle_recovered") >= 0
        unrecoverable = np.char.find(modes.astype(str), "unrecoverable") >= 0
        payload["controller_track_direct_frame_count"] = int(np.count_nonzero(modes == "direct_valid"))
        payload["controller_track_neighbor_recovered_frame_count"] = int(np.count_nonzero(bundle_recovered))
        payload["controller_track_unrecoverable_frame_count"] = int(np.count_nonzero(unrecoverable))
        payload["controller_track_mode_summary"] = {
            str(mode): int(np.count_nonzero(modes == mode))
            for mode in sorted(set(str(value) for value in modes.reshape(-1).tolist()))
        }
    if "track_process_status" in track_process_data:
        payload["track_process_status"] = str(np.asarray(track_process_data["track_process_status"]).item())
    return payload


def _track_process_invalid(manifest: Mapping[str, Any]) -> bool:
    return str(normalize_data_process_keys(manifest, validate_aliases=False).get("track_process_status", "normal")) == "invalid"


def _track_process_online_publish_skip_reason(
    manifest: Mapping[str, Any],
    *,
    allow_degraded_online: bool = False,
) -> str | None:
    status = str(normalize_data_process_keys(manifest, validate_aliases=False).get("track_process_status", "normal"))
    if status == "invalid":
        return "track_process_invalid"
    if status == "degraded" and not bool(allow_degraded_online):
        return "track_process_degraded"
    return None


def _object_track_manifest_fields(track_process_data: Mapping[str, Any]) -> dict[str, Any]:
    track_process_data = normalize_data_process_keys(track_process_data)
    if "object_track_query_indices" not in track_process_data:
        object_points = np.asarray(track_process_data.get("object_points", np.empty((0, 0, 3))))
        return {
            "object_track_selection_mode": "per_chunk_volume_sample",
            "object_track_count": int(object_points.shape[1] if object_points.ndim >= 2 else 0),
        }
    query_indices = np.asarray(track_process_data["object_track_query_indices"], dtype=np.int64).reshape(-1)
    active_indices = np.asarray(
        track_process_data.get("object_track_active_query_indices", query_indices),
        dtype=np.int64,
    ).reshape(-1)
    statuses = np.asarray(track_process_data.get("object_track_status", []), dtype=str).reshape(-1)
    return {
        "object_track_selection_mode": "streaming_stable",
        "object_track_count": int(len(query_indices)),
        "object_track_query_indices": [int(value) for value in query_indices.tolist()],
        "object_track_active_query_indices": [int(value) for value in active_indices.tolist()],
        "object_track_direct_count": int(np.count_nonzero(statuses == "direct")),
        "object_track_revived_count": int(np.count_nonzero(statuses == "revived")),
        "object_track_fallback_count": int(np.count_nonzero(statuses == "fallback")),
        "object_track_missing_count": int(np.count_nonzero(statuses == "missing")),
        "object_track_status_summary": {
            "direct": int(np.count_nonzero(statuses == "direct")),
            "revived": int(np.count_nonzero(statuses == "revived")),
            "fallback": int(np.count_nonzero(statuses == "fallback")),
            "missing": int(np.count_nonzero(statuses == "missing")),
        },
    }


def _track_input_with_session_query_schema(
    *,
    tracks_yx: np.ndarray,
    visibility: np.ndarray,
    processed_masks: Sequence[Sequence[Mapping[str, Any]]],
    pcd_points: np.ndarray,
    pcd_colors: np.ndarray,
    session_query_schema: dict[str, np.ndarray] | None,
) -> dict[str, np.ndarray]:
    """Build strict track input while preserving session-wide query identity."""
    query_ids = None
    query_semantic_labels = None
    if session_query_schema is not None and "query_ids" in session_query_schema:
        query_ids = session_query_schema["query_ids"]
        query_semantic_labels = session_query_schema["query_semantic_labels"]
    track_input = strict.build_track_process_input(
        tracks_yx=tracks_yx,
        visibility=visibility,
        processed_masks=processed_masks,
        pcd_points=pcd_points,
        pcd_colors=pcd_colors,
        query_ids=query_ids,
        query_semantic_labels=query_semantic_labels,
    )
    if session_query_schema is None:
        return track_input
    if "query_ids" not in session_query_schema:
        session_query_schema["query_ids"] = np.ascontiguousarray(track_input["query_ids"], dtype=np.int64)
        session_query_schema["query_semantic_labels"] = np.ascontiguousarray(
            track_input["query_semantic_labels"],
            dtype=np.int8,
        )
        return track_input
    if not np.array_equal(session_query_schema["query_ids"], track_input["query_ids"]):
        raise ValueError("Demo v5 session query_ids changed across chunks")
    if not np.array_equal(session_query_schema["query_semantic_labels"], track_input["query_semantic_labels"]):
        raise ValueError("Demo v5 session query_semantic_labels changed across chunks")
    return track_input


def _chunk_payload_from_rows(
    capture_dir: Path,
    metadata: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    surface_points: np.ndarray,
    interior_points: np.ndarray,
    fps: int,
    serial_number: str,
    chunk_index: int,
    mask_radius_outlier_filter: bool,
    mask_radius_outlier_radius_m: float,
    mask_radius_outlier_nb_points: int,
    write_final_pcd: bool = True,
    object_track_selector: strict.StreamingObjectTrackSelector | None = None,
    controller_track_selector: strict.StreamingControllerTrackSelector | None = None,
    session_query_schema: dict[str, np.ndarray] | None = None,
) -> DataProcessChunk:
    """Materialize a chunk from legacy row artifacts.

    This path is kept for older captures that did not write prepared data_process
    frame NPZs. New realtime runs usually use ``_chunk_payload_from_prepared_frames``
    to avoid rebuilding dense PCD and masks from separate files.
    """
    c2w = _camera_to_world(metadata)
    intrinsics = _intrinsics_matrix(metadata)

    rgb_frames: list[np.ndarray] = []
    processed_masks: list[list[dict[str, np.ndarray]]] = []
    pcd_points: list[np.ndarray] = []
    pcd_colors: list[np.ndarray] = []
    tracks: list[np.ndarray] = []
    visibility: list[np.ndarray] = []
    query_points_yx: np.ndarray | None = None

    for row in rows:
        rgb = _load_rgb(capture_dir / str(row["rgb_path"]))
        depth = np.load(_depth_path(capture_dir, row))
        points, colors = strict.dense_world_pcd_grid(
            depth_m=depth,
            color_rgb_u8=rgb,
            intrinsics=intrinsics,
            c2w=c2w,
        )
        rgb_frames.append(rgb)
        pcd_points.append(points)
        pcd_colors.append(colors)
        mask_frame = _load_mask_frame(capture_dir / str(row["mask_path"]))
        depth_valid_mask_frame = _apply_depth_validity_to_mask_frame(mask_frame, depth)
        processed_masks.append(
            [
                _apply_radius_outlier_to_mask_frame(
                    depth_valid_mask_frame,
                    points,
                    enabled=bool(mask_radius_outlier_filter),
                    radius_m=float(mask_radius_outlier_radius_m),
                    nb_points=int(mask_radius_outlier_nb_points),
                )
            ]
        )

        trajectory = np.load(capture_dir / str(row["query_trajectory_path"]), allow_pickle=False)
        if query_points_yx is None:
            query_points_yx = np.asarray(trajectory["query_points_yx"], dtype=np.float32)
        full_tracks, full_visibility = _full_tracks_and_visibility(trajectory, int(len(query_points_yx)))
        tracks.append(full_tracks)
        visibility.append(full_visibility)

    tracks_yx = np.stack(tracks, axis=0)
    tracker_visibility = np.stack(visibility, axis=0)
    if query_points_yx is None:
        query_points_yx = np.empty((0, 2), dtype=np.float32)
    queries_txy = np.zeros((len(query_points_yx), 3), dtype=np.float32)
    if len(query_points_yx):
        queries_txy[:, 1] = query_points_yx[:, 1]
        queries_txy[:, 2] = query_points_yx[:, 0]

    pcd_points_arr = np.stack(pcd_points, axis=0)
    pcd_colors_arr = np.stack(pcd_colors, axis=0)
    track_input = _track_input_with_session_query_schema(
        tracks_yx=tracks_yx,
        visibility=tracker_visibility,
        processed_masks=processed_masks,
        pcd_points=pcd_points_arr,
        pcd_colors=pcd_colors_arr,
        session_query_schema=session_query_schema,
    )
    filtered = strict.apply_phystwin_motion_filters(track_input)
    if object_track_selector is not None:
        filtered = object_track_selector.select(
            filtered,
            surface_points=surface_points,
            interior_points=interior_points,
        )
    if controller_track_selector is None:
        track_process = strict.select_final_controller_points(filtered, count=30)
    else:
        track_process = controller_track_selector.select(filtered)

    return DataProcessChunk(
        rgb_frames=rgb_frames,
        processed_masks=processed_masks,
        track_process_data=track_process,
        intrinsics=intrinsics,
        camera_to_world_c2w=c2w,
        tracks_yx=tracks_yx,
        tracker_visibility=tracker_visibility,
        queries_txy=queries_txy,
        surface_points=surface_points,
        interior_points=interior_points,
        fps=int(fps),
        serial_number=serial_number,
        depth_backend=str(metadata.get("depth_backend") or metadata.get("depth_source", "")),
        depth_source_internal=str(
            metadata.get("depth_source_internal")
            or metadata.get("depth_source")
            or metadata.get("depth_backend", "")
        ),
        chunk_index=int(chunk_index),
        source_frame_indices=[int(row.get("seq", idx)) for idx, row in enumerate(rows)],
        pcd_points=pcd_points_arr if bool(write_final_pcd) else None,
        pcd_colors=pcd_colors_arr if bool(write_final_pcd) else None,
    )


def _queries_txy_from_yx(query_points_yx: np.ndarray) -> np.ndarray:
    queries_yx = np.asarray(query_points_yx, dtype=np.float32).reshape(-1, 2)
    queries_txy = np.zeros((len(queries_yx), 3), dtype=np.float32)
    if len(queries_yx):
        queries_txy[:, 1] = queries_yx[:, 1]
        queries_txy[:, 2] = queries_yx[:, 0]
    return np.ascontiguousarray(queries_txy, dtype=np.float32)


def _prepared_frame_from_row(
    capture_dir: Path,
    row: Mapping[str, Any],
) -> strict.PreparedPhysTwinFrame | None:
    path_value = row.get("prepared_data_process_frame_path", row.get("prepared_phystwin_frame_path"))
    if path_value is None:
        return None
    return strict.load_prepared_phystwin_frame(capture_dir / str(path_value))


def _chunk_payload_from_prepared_frames(
    metadata: Mapping[str, Any],
    frames: Sequence[strict.PreparedPhysTwinFrame],
    *,
    surface_points: np.ndarray,
    interior_points: np.ndarray,
    fps: int,
    serial_number: str,
    chunk_index: int,
    write_final_pcd: bool = True,
    object_track_selector: strict.StreamingObjectTrackSelector | None = None,
    controller_track_selector: strict.StreamingControllerTrackSelector | None = None,
    session_query_schema: dict[str, np.ndarray] | None = None,
) -> DataProcessChunk:
    """Materialize a chunk from prepared per-frame NPZ payloads."""
    if not frames:
        raise ValueError("prepared data_process chunk requires at least one frame")
    c2w = _camera_to_world(metadata)
    intrinsics = _intrinsics_matrix(metadata)
    first_queries = np.asarray(frames[0].query_points_yx, dtype=np.float32).reshape(-1, 2)

    rgb_frames: list[np.ndarray] = []
    processed_masks: list[list[dict[str, np.ndarray]]] = []
    tracks: list[np.ndarray] = []
    visibility: list[np.ndarray] = []
    pcd_points: list[np.ndarray] = []
    pcd_colors: list[np.ndarray] = []
    source_frame_indices: list[int] = []

    for frame in frames:
        queries = np.asarray(frame.query_points_yx, dtype=np.float32).reshape(-1, 2)
        if queries.shape != first_queries.shape or not np.allclose(queries, first_queries):
            raise ValueError("prepared data_process frames in one chunk must share query_points_yx")
        rgb_frames.append(np.ascontiguousarray(frame.rgb_frame, dtype=np.uint8))
        processed_masks.append([strict.normalize_processed_mask_frame(frame.processed_mask_frame)])
        tracks.append(np.ascontiguousarray(frame.tracks_yx, dtype=np.float32).reshape(-1, 2))
        visibility.append(np.ascontiguousarray(frame.visibility, dtype=bool).reshape(-1))
        pcd_points.append(np.ascontiguousarray(frame.pcd_points, dtype=np.float32))
        pcd_colors.append(np.ascontiguousarray(frame.pcd_colors, dtype=np.uint8))
        source_frame_indices.append(int(frame.source_frame_index if frame.source_frame_index is not None else frame.seq))

    tracks_yx = np.stack(tracks, axis=0)
    tracker_visibility = np.stack(visibility, axis=0)
    pcd_points_arr = np.stack(pcd_points, axis=0)
    pcd_colors_arr = np.stack(pcd_colors, axis=0)
    track_input = _track_input_with_session_query_schema(
        tracks_yx=tracks_yx,
        visibility=tracker_visibility,
        processed_masks=processed_masks,
        pcd_points=pcd_points_arr,
        pcd_colors=pcd_colors_arr,
        session_query_schema=session_query_schema,
    )
    filtered = strict.apply_phystwin_motion_filters(track_input)
    if object_track_selector is not None:
        filtered = object_track_selector.select(
            filtered,
            surface_points=surface_points,
            interior_points=interior_points,
        )
    if controller_track_selector is None:
        track_process = strict.select_final_controller_points(filtered, count=30)
    else:
        track_process = controller_track_selector.select(filtered)

    return DataProcessChunk(
        rgb_frames=rgb_frames,
        processed_masks=processed_masks,
        track_process_data=track_process,
        intrinsics=intrinsics,
        camera_to_world_c2w=c2w,
        tracks_yx=tracks_yx,
        tracker_visibility=tracker_visibility,
        queries_txy=_queries_txy_from_yx(first_queries),
        surface_points=surface_points,
        interior_points=interior_points,
        fps=int(fps),
        serial_number=serial_number,
        depth_backend=str(metadata.get("depth_backend") or metadata.get("depth_source", "")),
        depth_source_internal=str(
            metadata.get("depth_source_internal")
            or metadata.get("depth_source")
            or metadata.get("depth_backend", "")
        ),
        chunk_index=int(chunk_index),
        source_frame_indices=source_frame_indices,
        pcd_points=pcd_points_arr if bool(write_final_pcd) else None,
        pcd_colors=pcd_colors_arr if bool(write_final_pcd) else None,
    )


def _write_chunk_from_rows(
    *,
    capture: Path,
    metadata: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    base_path: str | Path,
    case_prefix: str,
    chunk_index: int,
    row_start: int,
    row_end: int,
    fps: int,
    serial_number: str,
    surface_points: np.ndarray,
    interior_points: np.ndarray,
    mask_radius_outlier_filter: bool,
    mask_radius_outlier_radius_m: float,
    mask_radius_outlier_nb_points: int,
    wall_time_origin_s: float,
    window_closed_wall_s: float,
    prepared_frames: Sequence[strict.PreparedPhysTwinFrame | None] | None = None,
    backlog_chunks: Callable[[], int] | None = None,
    write_final_pcd: bool = True,
    online_writer: ChunkedFinalDataWriter | None = None,
    allow_degraded_online: bool = False,
    object_track_selector: strict.StreamingObjectTrackSelector | None = None,
    controller_track_selector: strict.StreamingControllerTrackSelector | None = None,
    session_query_schema: dict[str, np.ndarray] | None = None,
    warmup_skipped_rows: int = 0,
) -> dict[str, Any]:
    """Write one data_process_sam3d case and optionally commit it to online output."""
    case_name = f"{case_prefix}_chunk_{chunk_index:04d}"
    source_window_start_s = float(row_start) / float(fps)
    source_window_end_s = float(row_end) / float(fps)
    materialize_start_wall_s = _relative_wall_s(float(wall_time_origin_s))
    prepared = list(prepared_frames or [])
    prepared_count = sum(1 for frame in prepared if frame is not None)
    if prepared and len(prepared) == len(rows) and prepared_count == len(rows):
        # Prepared frames are the realtime path: RGB, masks, dense world PCD,
        # full tracks, visibility, and query points are already synchronized by
        # the camera process for the same source seq.
        chunk = _chunk_payload_from_prepared_frames(
            metadata,
            [frame for frame in prepared if frame is not None],
            surface_points=surface_points,
            interior_points=interior_points,
            fps=int(fps),
            serial_number=serial_number,
            chunk_index=chunk_index,
            write_final_pcd=bool(write_final_pcd),
            object_track_selector=object_track_selector,
            controller_track_selector=controller_track_selector,
            session_query_schema=session_query_schema,
        )
        materialization_source = "prepared_data_process_frame"
        legacy_reprocess_count = 0
    else:
        # Fallback for historical captures where Demo v5 has to reconstruct the
        # strict product from image/depth/mask/trajectory sidecar files.
        chunk = _chunk_payload_from_rows(
            capture,
            metadata,
            rows,
            surface_points=surface_points,
            interior_points=interior_points,
            fps=int(fps),
            serial_number=serial_number,
            chunk_index=chunk_index,
            mask_radius_outlier_filter=bool(mask_radius_outlier_filter),
            mask_radius_outlier_radius_m=float(mask_radius_outlier_radius_m),
            mask_radius_outlier_nb_points=int(mask_radius_outlier_nb_points),
            write_final_pcd=bool(write_final_pcd),
            object_track_selector=object_track_selector,
            controller_track_selector=controller_track_selector,
            session_query_schema=session_query_schema,
        )
        materialization_source = "legacy_reprocess"
        legacy_reprocess_count = len(rows)
    track_finalize_done_wall_s = max(_relative_wall_s(float(wall_time_origin_s)), window_closed_wall_s)
    track_fields = _object_track_manifest_fields(chunk.track_process_data)
    track_fields.update(_controller_track_manifest_fields(chunk.track_process_data))
    chunk_source_frame_indices = (
        [int(value) for value in chunk.source_frame_indices]
        if chunk.source_frame_indices is not None
        else _rows_source_frame_indices(rows, fallback_start=row_start)
    )
    chunk_source_timestamps_s = _rows_source_timestamps(rows)

    def manifest_extras() -> dict[str, Any]:
        backlog_count = 0 if backlog_chunks is None else int(backlog_chunks())
        payload = {
            "source_capture_dir": str(capture),
            "source_row_start": int(row_start),
            "source_row_end": int(row_end),
            "warmup_skipped_rows": int(warmup_skipped_rows),
            "source_window_start_s": source_window_start_s,
            "source_window_end_s": source_window_end_s,
            "chunk_ready_source_seq": int(rows[-1].get("seq", row_end - 1)),
            "chunk_ready_source_frame_index": int(chunk_source_frame_indices[-1]),
            "chunk_ready_source_time_s": (
                None
                if rows[-1].get("source_timestamp_s") is None
                else float(rows[-1]["source_timestamp_s"])
            ),
            "window_closed_wall_s": float(window_closed_wall_s),
            "track_finalize_done_wall_s": float(track_finalize_done_wall_s),
            "materialize_start_wall_s": materialize_start_wall_s,
            "materialize_end_wall_s": track_finalize_done_wall_s,
            "materialize_latency_ms": float((track_finalize_done_wall_s - materialize_start_wall_s) * 1000.0),
            "backlog_chunks": backlog_count,
            "chunk_materialization_source": materialization_source,
            "prepared_frame_count": int(prepared_count),
            "legacy_reprocess_frame_count": int(legacy_reprocess_count),
        }
        payload.update(track_fields)
        return payload

    manifest = write_data_process_chunk_case(
        base_path,
        case_name,
        chunk,
        manifest_extras=manifest_extras,
        relative_wall_time_s=lambda: _relative_wall_s(float(wall_time_origin_s)),
    )
    skip_reason = _track_process_online_publish_skip_reason(
        manifest,
        allow_degraded_online=bool(allow_degraded_online),
    )
    if skip_reason is not None:
        manifest.update(
            {
                "online_publish_skipped": True,
                "online_publish_skip_reason": str(skip_reason),
            }
        )
        return manifest
    if online_writer is not None:
        online_result = online_writer.commit_case_chunk(
            Path(manifest["data_process_case_root"]),
            source_frame_indices=chunk_source_frame_indices,
            source_timestamps_s=chunk_source_timestamps_s,
            status="recording",
        )
        manifest.update(
            {
                "online_dir": online_result["online_dir"],
                "online_manifest_path": online_result["online_manifest_path"],
                "online_chunk_path": online_result["online_chunk_path"],
                "online_chunk_id": online_result["online_chunk_id"],
                "online_latest_committed_frame": online_result["online_latest_committed_frame"],
                "static_data_path": online_result["static_data_path"],
            }
        )
    manifest.setdefault("online_publish_skipped", False)
    return manifest


def write_chunks_from_headless_capture(
    capture_dir: str | Path,
    *,
    base_path: str | Path,
    case_prefix: str = "demo_v5_1",
    chunk_frame_count: int = 25,
    fps: int = 5,
    max_chunks: int | None = None,
    surface_points: np.ndarray | None = None,
    interior_points: np.ndarray | None = None,
    mask_radius_outlier_filter: bool = True,
    mask_radius_outlier_radius_m: float = 0.01,
    mask_radius_outlier_nb_points: int = 40,
    on_chunk_written: Callable[[dict[str, Any]], None] | None = None,
    write_final_pcd: bool = True,
    write_online_output: bool = True,
    online_case_name: str | None = None,
    allow_degraded_online: bool = False,
) -> list[dict[str, Any]]:
    """Convert a completed headless capture into chunk cases."""
    capture = Path(capture_dir)
    if int(chunk_frame_count) <= 0:
        raise ValueError("chunk_frame_count must be positive")
    metadata = _read_json_file_stable(
        capture / "metadata.json",
        deadline_s=time.monotonic() + 5.0,
        poll_interval_s=0.05,
    )
    shape_surface, shape_interior = _shape_points_from_capture(
        capture,
        metadata,
        surface_points=surface_points,
        interior_points=interior_points,
    )
    serials = metadata.get("serial_numbers") or ["demo-v5-single-camera"]
    serial_number = str(serials[0])

    manifests: list[dict[str, Any]] = []
    chunk_size = int(chunk_frame_count)
    chunk_index = 1
    row_buffer: list[dict[str, Any]] = []
    prepared_buffer: list[strict.PreparedPhysTwinFrame | None] = []
    row_start = 0
    wall_time_origin_s = time.monotonic()
    frames_path = capture / "frames.jsonl"
    rows_to_process, warmup_skipped_rows = _trim_warmup_delayed_rows(list(_iter_jsonl(frames_path)))
    online_writer = None
    if bool(write_online_output):
        online_writer = ChunkedFinalDataWriter(
            base_path=base_path,
            case_name=str(online_case_name or case_prefix),
            chunk_size=chunk_size,
            num_frames_total=(
                min(len(rows_to_process) // chunk_size, int(max_chunks)) * chunk_size
                if max_chunks is not None
                else (len(rows_to_process) // chunk_size) * chunk_size
            ),
            allow_degraded=bool(allow_degraded_online),
        )
    object_track_selector = strict.StreamingObjectTrackSelector(volume_sample_size=0.005)
    controller_track_selector = strict.StreamingControllerTrackSelector(count=30)
    session_query_schema: dict[str, np.ndarray] = {}
    for row_idx, row in enumerate(rows_to_process):
        if max_chunks is not None and len(manifests) >= int(max_chunks):
            break
        row_buffer.append(row)
        prepared_buffer.append(_prepared_frame_from_row(capture, row))
        if len(row_buffer) < chunk_size:
            continue
        window_closed_wall_s = _relative_wall_s(float(wall_time_origin_s))
        chunk_rows = row_buffer
        chunk_prepared = prepared_buffer
        manifest = _write_chunk_from_rows(
            capture=capture,
            metadata=metadata,
            rows=chunk_rows,
            base_path=base_path,
            case_prefix=case_prefix,
            chunk_index=chunk_index,
            row_start=row_start,
            row_end=row_idx + 1,
            fps=int(fps),
            serial_number=serial_number,
            surface_points=shape_surface,
            interior_points=shape_interior,
            mask_radius_outlier_filter=bool(mask_radius_outlier_filter),
            mask_radius_outlier_radius_m=float(mask_radius_outlier_radius_m),
            mask_radius_outlier_nb_points=int(mask_radius_outlier_nb_points),
            wall_time_origin_s=wall_time_origin_s,
            window_closed_wall_s=window_closed_wall_s,
            prepared_frames=chunk_prepared,
            backlog_chunks=lambda path=frames_path, size=chunk_size, published=chunk_index: _complete_chunk_backlog(
                path,
                chunk_size=size,
                published_chunk_count=published,
            ),
            write_final_pcd=bool(write_final_pcd),
            online_writer=online_writer,
            allow_degraded_online=bool(allow_degraded_online),
            object_track_selector=object_track_selector,
            controller_track_selector=controller_track_selector,
            session_query_schema=session_query_schema,
            warmup_skipped_rows=warmup_skipped_rows,
        )
        manifests.append(manifest)
        if not bool(manifest.get("online_publish_skipped", False)) and on_chunk_written is not None:
            on_chunk_written(manifest)
        if _track_process_invalid(manifest):
            break
        chunk_index += 1
        row_start = row_idx + 1
        row_buffer = []
        prepared_buffer = []
    if online_writer is not None:
        online_writer.finish()
    return manifests


def _wait_for_metadata(capture: Path, *, capture_finished: Callable[[], bool], poll_interval_s: float) -> Mapping[str, Any]:
    metadata_path = capture / "metadata.json"
    while True:
        if metadata_path.is_file():
            try:
                return _read_json_file_stable(
                    metadata_path,
                    deadline_s=time.monotonic() + max(0.5, float(poll_interval_s) * 4.0),
                    poll_interval_s=float(poll_interval_s),
                )
            except RuntimeError:
                if capture_finished():
                    raise RuntimeError(f"capture finished before stable metadata appeared: {metadata_path}")
        elif capture_finished():
            raise RuntimeError(f"capture finished before metadata appeared: {metadata_path}")
        time.sleep(max(0.0, float(poll_interval_s)))


def stream_chunks_from_headless_capture(
    capture_dir: str | Path,
    *,
    base_path: str | Path,
    case_prefix: str = "demo_v5_1",
    chunk_frame_count: int = 25,
    fps: int = 5,
    max_chunks: int | None = None,
    capture_finished: Callable[[], bool],
    before_poll: Callable[[], None] | None = None,
    poll_interval_s: float = 0.05,
    surface_points: np.ndarray | None = None,
    interior_points: np.ndarray | None = None,
    require_shape_prior: bool = False,
    shape_prior_wait_timeout_s: float = 300.0,
    mask_radius_outlier_filter: bool = True,
    mask_radius_outlier_radius_m: float = 0.01,
    mask_radius_outlier_nb_points: int = 40,
    on_chunk_written: Callable[[dict[str, Any]], None] | None = None,
    write_final_pcd: bool = True,
    write_online_output: bool = True,
    online_case_name: str | None = None,
    allow_degraded_online: bool = False,
) -> list[dict[str, Any]]:
    """Tail a live headless capture and publish chunks as windows close."""
    capture = Path(capture_dir)
    if int(chunk_frame_count) <= 0:
        raise ValueError("chunk_frame_count must be positive")
    metadata = _wait_for_metadata(
        capture,
        capture_finished=capture_finished,
        poll_interval_s=float(poll_interval_s),
    )
    serials = metadata.get("serial_numbers") or ["demo-v5-single-camera"]
    serial_number = str(serials[0])
    frames_path = capture / "frames.jsonl"
    manifests: list[dict[str, Any]] = []
    next_row_idx = 0
    frames_offset = 0
    row_start = 0
    row_buffer: list[dict[str, Any]] = []
    prepared_buffer: list[strict.PreparedPhysTwinFrame | None] = []
    chunk_index = 1
    chunk_size = int(chunk_frame_count)
    wall_time_origin_s = time.monotonic()
    warmup_start_filter = _WarmupStartFilterState()
    online_writer = None
    if bool(write_online_output):
        online_writer = ChunkedFinalDataWriter(
            base_path=base_path,
            case_name=str(online_case_name or case_prefix),
            chunk_size=chunk_size,
            num_frames_total=(
                None if max_chunks is None else int(max_chunks) * int(chunk_size)
            ),
            allow_degraded=bool(allow_degraded_online),
        )
    object_track_selector = strict.StreamingObjectTrackSelector(volume_sample_size=0.005)
    controller_track_selector = strict.StreamingControllerTrackSelector(count=30)
    session_query_schema: dict[str, np.ndarray] = {}

    while True:
        if max_chunks is not None and len(manifests) >= int(max_chunks):
            break
        if before_poll is not None:
            before_poll()
        rows, frames_offset = _read_jsonl_from_offset(frames_path, frames_offset)
        saw_new_rows = bool(rows)
        rows = _filter_warmup_start_rows(
            warmup_start_filter,
            rows,
            capture_finished=bool(capture_finished()),
        )
        for row in rows:
            # A chunk closes strictly by row count. Shape-prior readiness and
            # final_data materialization are handled after this window boundary
            # so source pacing remains tied to the camera/fake-camera stream.
            row_buffer.append(row)
            prepared_buffer.append(_prepared_frame_from_row(capture, row))
            next_row_idx += 1
            if len(row_buffer) < chunk_size:
                continue
            window_closed_wall_s = _relative_wall_s(float(wall_time_origin_s))
            chunk_prepared = prepared_buffer
            latest_metadata, shape_surface, shape_interior = _shape_points_for_chunk(
                capture,
                surface_points=surface_points,
                interior_points=interior_points,
                require_shape_prior=bool(require_shape_prior),
                shape_prior_wait_timeout_s=float(shape_prior_wait_timeout_s),
                capture_finished=capture_finished,
                before_poll=before_poll,
                poll_interval_s=float(poll_interval_s),
            )
            manifest = _write_chunk_from_rows(
                capture=capture,
                metadata=latest_metadata,
                rows=row_buffer,
                base_path=base_path,
                case_prefix=case_prefix,
                chunk_index=chunk_index,
                row_start=row_start,
                row_end=next_row_idx,
                fps=int(fps),
                serial_number=serial_number,
                surface_points=shape_surface,
                interior_points=shape_interior,
                mask_radius_outlier_filter=bool(mask_radius_outlier_filter),
                mask_radius_outlier_radius_m=float(mask_radius_outlier_radius_m),
                mask_radius_outlier_nb_points=int(mask_radius_outlier_nb_points),
                wall_time_origin_s=wall_time_origin_s,
                window_closed_wall_s=window_closed_wall_s,
                prepared_frames=chunk_prepared,
                backlog_chunks=lambda path=frames_path, size=chunk_size, published=chunk_index: _complete_chunk_backlog(
                    path,
                    chunk_size=size,
                    published_chunk_count=published,
                ),
                write_final_pcd=bool(write_final_pcd),
                online_writer=online_writer,
                allow_degraded_online=bool(allow_degraded_online),
                object_track_selector=object_track_selector,
                controller_track_selector=controller_track_selector,
                session_query_schema=session_query_schema,
                warmup_skipped_rows=warmup_start_filter.skipped_rows,
            )
            manifests.append(manifest)
            if not bool(manifest.get("online_publish_skipped", False)) and on_chunk_written is not None:
                on_chunk_written(manifest)
            if _track_process_invalid(manifest):
                break
            chunk_index += 1
            row_start = next_row_idx
            row_buffer = []
            prepared_buffer = []
            if max_chunks is not None and len(manifests) >= int(max_chunks):
                break
        if manifests and _track_process_invalid(manifests[-1]):
            break
        if max_chunks is not None and len(manifests) >= int(max_chunks):
            break
        if capture_finished() and not saw_new_rows:
            break
        time.sleep(max(0.0, float(poll_interval_s)))
    if online_writer is not None:
        online_writer.finish()
    return manifests


__all__ = ["stream_chunks_from_headless_capture", "write_chunks_from_headless_capture"]
