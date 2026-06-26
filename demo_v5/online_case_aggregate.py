"""Build the static data/<case> view from published Demo v5 chunk cases."""
from __future__ import annotations

import json
import os
from pathlib import Path
import pickle
import shutil
from typing import Any, Mapping, Sequence

import numpy as np

from demo_v5.atomic_io import (
    atomic_json_dump as _atomic_json_dump,
    atomic_pickle_dump as _atomic_pickle_dump,
)
from demo_v5.futurephystwin_chunk_writer import (
    FUTUREPHYSTWIN_FINAL_DATA_KEYS,
    FUTUREPHYSTWIN_TOPOLOGY_KEYS,
    FUTUREPHYSTWIN_TRACK_PROCESS_KEYS,
    validate_futurephystwin_case,
)


FINAL_TIME_KEYS = (
    "controller_points",
    "object_colors",
    "object_motions_valid",
    "object_points",
    "object_visibilities",
)
FINAL_STATIC_KEYS = (
    "controller_fps_indices",
    "controller_selected_query_ids",
    "controller_sample_query_ids",
    "object_sample_indices",
    "object_selected_query_ids",
    "object_sample_query_ids",
    *FUTUREPHYSTWIN_TOPOLOGY_KEYS,
    "surface_points",
    "interior_points",
)
FINAL_FIRST_STATIC_KEYS: tuple[str, ...] = ()
TRACK_TIME_KEYS = (
    "controller_points",
    "object_colors",
    "object_motions_valid",
    "object_points",
    "object_visibilities",
)
TRACK_STATIC_KEYS = FUTUREPHYSTWIN_TOPOLOGY_KEYS
TRACK_FIRST_STATIC_KEYS = ("controller_mask",)
METADATA_INVARIANT_KEYS = (
    "fps",
    "WH",
    "intrinsics",
    "serial_numbers",
    "camera_count",
    "demo_version",
    "runtime_product_name",
    "runtime_contract",
    "reference_pipeline",
    "depth_backend",
    "depth_source_internal",
)
GENERATED_FILES = (
    "final_data.pkl",
    "track_process_data.pkl",
    "calibrate.pkl",
    "metadata.json",
    "split.json",
    "manifest.json",
    "READY",
)
GENERATED_DIRS = (
    "color",
    "mask",
    "tracking",
    "cotracker",
    "pcd",
    "depth",
)


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _frame_count_from_payload(payload: Mapping[str, Any]) -> int:
    return int(np.asarray(payload["object_points"]).shape[0])


def _split_payload(frame_count: int) -> dict[str, Any]:
    train_end = max(1, int(int(frame_count) * 0.7))
    train_end = min(train_end, int(frame_count))
    return {
        "frame_len": int(frame_count),
        "train": [0, int(train_end)],
        "test": [int(train_end), int(frame_count)],
    }


def _arrays_match(left: Any, right: Any) -> bool:
    left_arr = np.asarray(left)
    right_arr = np.asarray(right)
    if left_arr.shape != right_arr.shape and left_arr.size == 1 and right_arr.size == 1:
        if not (
            np.issubdtype(left_arr.dtype, np.number)
            and np.issubdtype(right_arr.dtype, np.number)
        ):
            return str(left_arr.item()) == str(right_arr.item())
    if left_arr.shape != right_arr.shape:
        return False
    if np.issubdtype(left_arr.dtype, np.number) and np.issubdtype(right_arr.dtype, np.number):
        return bool(np.allclose(left_arr, right_arr, rtol=1e-6, atol=1e-6, equal_nan=True))
    return bool(np.array_equal(left_arr, right_arr))


def _require_matching_value(name: str, expected: Any, actual: Any) -> None:
    if not _arrays_match(expected, actual):
        raise ValueError(f"aggregate chunk invariant mismatch for {name}")


def _static_invariant_value(key: str, value: Any) -> Any:
    if key in {"topology_version", "topology_hash"}:
        return str(np.asarray(value).item() if isinstance(value, np.ndarray) else value)
    return np.ascontiguousarray(np.asarray(value))


def _require_payload_keys(payload: Mapping[str, Any], keys: Sequence[str], *, label: str) -> None:
    missing = [key for key in keys if key not in payload]
    if missing:
        raise ValueError(f"{label} missing required keys: {missing}")


def _concatenate_payloads(
    payloads: Sequence[Mapping[str, Any]],
    *,
    time_keys: Sequence[str],
    static_keys: Sequence[str],
    label: str,
) -> dict[str, Any]:
    """Concatenate time-varying arrays while enforcing static invariants."""
    if not payloads:
        raise ValueError(f"cannot aggregate empty {label} payload list")
    combined: dict[str, Any] = {}
    for key in time_keys:
        arrays = [np.ascontiguousarray(np.asarray(payload[key])) for payload in payloads]
        tail_shape = arrays[0].shape[1:]
        for chunk_idx, arr in enumerate(arrays):
            if arr.ndim < 1:
                raise ValueError(f"{label}.{key} must have a frame axis")
            if arr.shape[1:] != tail_shape:
                raise ValueError(
                    f"cannot concatenate {label}.{key}: chunk {chunk_idx} tail shape "
                    f"{arr.shape[1:]} != {tail_shape}"
                )
        combined[key] = np.ascontiguousarray(np.concatenate(arrays, axis=0))
    for key in static_keys:
        first = _static_invariant_value(key, payloads[0][key])
        for chunk_idx, payload in enumerate(payloads[1:], start=1):
            value = _static_invariant_value(key, payload[key])
            if not _arrays_match(first, value):
                raise ValueError(f"aggregate static invariant mismatch for {label}.{key} at chunk {chunk_idx}")
        combined[key] = first
    return combined


def _load_calibrate_matrix(case_dir: Path) -> np.ndarray:
    c2ws = _load_pickle(case_dir / "calibrate.pkl")
    if len(c2ws) != 1:
        raise ValueError(f"calibrate.pkl must contain one camera matrix: {case_dir}")
    c2w = np.asarray(c2ws[0], dtype=np.float32)
    if c2w.shape != (4, 4):
        raise ValueError(f"calibrate.pkl camera matrix must be 4x4: {case_dir}")
    return np.ascontiguousarray(c2w)


def _load_tracking_payload(case_dir: Path, name: str) -> dict[str, np.ndarray]:
    with np.load(case_dir / name / "0.npz") as payload:
        return {
            "tracks": np.ascontiguousarray(np.asarray(payload["tracks"], dtype=np.float32)),
            "visibility": np.ascontiguousarray(np.asarray(payload["visibility"], dtype=bool)),
            "queries_txy": np.ascontiguousarray(np.asarray(payload["queries_txy"], dtype=np.float32)),
        }


def _validate_chunk_cases(chunk_cases: Sequence[Path]) -> None:
    """Ensure chunks can be concatenated without changing camera/query identity."""
    if not chunk_cases:
        raise ValueError("aggregate requires at least one READY chunk case")
    first_metadata: dict[str, Any] | None = None
    first_calibrate: np.ndarray | None = None
    first_queries: dict[str, np.ndarray] = {}
    first_final_static: dict[str, np.ndarray] = {}
    first_track_static: dict[str, np.ndarray] = {}
    for chunk_idx, chunk_case in enumerate(chunk_cases):
        validate_futurephystwin_case(chunk_case, require_ready=True)
        final_data = _load_pickle(chunk_case / "final_data.pkl")
        track_process = _load_pickle(chunk_case / "track_process_data.pkl")
        _require_payload_keys(final_data, FUTUREPHYSTWIN_FINAL_DATA_KEYS, label="final_data.pkl")
        _require_payload_keys(track_process, FUTUREPHYSTWIN_TRACK_PROCESS_KEYS, label="track_process_data.pkl")

        metadata = _load_json(chunk_case / "metadata.json")
        missing_metadata = [key for key in METADATA_INVARIANT_KEYS if key not in metadata]
        if missing_metadata:
            raise ValueError(f"metadata.json missing aggregate invariant keys: {missing_metadata}")
        calibrate = _load_calibrate_matrix(chunk_case)
        if first_metadata is None:
            first_metadata = metadata
            first_calibrate = calibrate
            first_final_static = {
                key: _static_invariant_value(key, final_data[key])
                for key in FINAL_STATIC_KEYS
            }
            first_track_static = {
                key: _static_invariant_value(key, track_process[key])
                for key in TRACK_STATIC_KEYS
            }
        else:
            assert first_calibrate is not None
            _require_matching_value("calibrate.pkl", first_calibrate, calibrate)
            for key in METADATA_INVARIANT_KEYS:
                _require_matching_value(f"metadata.json {key}", first_metadata.get(key), metadata.get(key))
            for key, expected in first_final_static.items():
                _require_matching_value(f"final_data.pkl {key}", expected, final_data[key])
            for key, expected in first_track_static.items():
                _require_matching_value(f"track_process_data.pkl {key}", expected, track_process[key])

        for name in ("tracking", "cotracker"):
            tracking = _load_tracking_payload(chunk_case, name)
            queries = tracking["queries_txy"]
            if name not in first_queries:
                first_queries[name] = queries
            else:
                _require_matching_value(f"{name}/0.npz queries_txy", first_queries[name], queries)
            if tracking["tracks"].shape[0] != _frame_count_from_payload(final_data):
                raise ValueError(f"{name}/0.npz tracks frame count mismatch at chunk {chunk_idx}")
            if tracking["visibility"].shape != tracking["tracks"].shape[:2]:
                raise ValueError(f"{name}/0.npz visibility shape mismatch at chunk {chunk_idx}")


def _remove_generated_contents(case_dir: Path) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    for relative in GENERATED_FILES:
        path = case_dir / relative
        if path.exists():
            path.unlink()
    for relative in GENERATED_DIRS:
        path = case_dir / relative
        if path.exists():
            shutil.rmtree(path)


def _copy_indexed_files(
    source_dir: Path,
    target_dir: Path,
    *,
    start_frame: int,
    frame_count: int,
    required: bool,
) -> int:
    if not source_dir.is_dir():
        if required:
            raise ValueError(f"missing required frame directory: {source_dir}")
        return 0
    copied = 0
    target_dir.mkdir(parents=True, exist_ok=True)
    for local_frame_idx in range(int(frame_count)):
        matches = sorted(
            path
            for path in source_dir.glob(f"{local_frame_idx}.*")
            if path.is_file()
        )
        if required and not matches:
            raise ValueError(f"missing required frame {local_frame_idx} in {source_dir}")
        for source_path in matches:
            target_name = f"{int(start_frame) + local_frame_idx}{source_path.suffix}"
            shutil.copy2(source_path, target_dir / target_name)
            copied += 1
    return copied


def _copy_optional_depth_tree(source_case: Path, aggregate_case: Path, *, start_frame: int, frame_count: int) -> int:
    source_depth = source_case / "depth"
    if not source_depth.is_dir():
        return 0
    copied = 0
    camera_dirs = sorted(path for path in source_depth.iterdir() if path.is_dir())
    if camera_dirs:
        for camera_dir in camera_dirs:
            copied += _copy_indexed_files(
                camera_dir,
                aggregate_case / "depth" / camera_dir.name,
                start_frame=start_frame,
                frame_count=frame_count,
                required=False,
            )
    else:
        copied += _copy_indexed_files(
            source_depth,
            aggregate_case / "depth",
            start_frame=start_frame,
            frame_count=frame_count,
            required=False,
        )
    return copied


def _write_processed_masks(chunk_cases: Sequence[Path], aggregate_case: Path) -> None:
    masks: list[Any] = []
    for chunk_case in chunk_cases:
        source = _load_pickle(chunk_case / "mask" / "processed_masks.pkl")
        masks.extend(list(source))
    _atomic_pickle_dump(masks, aggregate_case / "mask" / "processed_masks.pkl")


def _write_tracking(chunk_cases: Sequence[Path], aggregate_case: Path, *, name: str) -> None:
    payloads = [_load_tracking_payload(chunk_case, name) for chunk_case in chunk_cases]
    first_queries = payloads[0]["queries_txy"]
    for chunk_idx, payload in enumerate(payloads[1:], start=1):
        _require_matching_value(f"{name}/0.npz queries_txy", first_queries, payload["queries_txy"])
        if payload["tracks"].shape[1:] != payloads[0]["tracks"].shape[1:]:
            raise ValueError(f"cannot concatenate {name}/0.npz tracks at chunk {chunk_idx}")
    tracks = np.concatenate([payload["tracks"] for payload in payloads], axis=0)
    visibility = np.concatenate([payload["visibility"] for payload in payloads], axis=0)
    target_dir = aggregate_case / name
    target_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = target_dir / "0.tmp.npz"
    np.savez(
        tmp_path,
        tracks=np.ascontiguousarray(tracks, dtype=np.float32),
        visibility=np.ascontiguousarray(visibility, dtype=bool),
        queries_txy=np.ascontiguousarray(first_queries, dtype=np.float32),
    )
    os.replace(tmp_path, target_dir / "0.npz")


def _write_frame_artifacts(chunk_cases: Sequence[Path], aggregate_case: Path) -> dict[str, int]:
    start_frame = 0
    copied_counts = {"color": 0, "pcd": 0, "depth": 0}
    for chunk_case in chunk_cases:
        frame_count = _frame_count_from_payload(_load_pickle(chunk_case / "final_data.pkl"))
        copied_counts["color"] += _copy_indexed_files(
            chunk_case / "color" / "0",
            aggregate_case / "color" / "0",
            start_frame=start_frame,
            frame_count=frame_count,
            required=True,
        )
        copied_counts["pcd"] += _copy_indexed_files(
            chunk_case / "pcd",
            aggregate_case / "pcd",
            start_frame=start_frame,
            frame_count=frame_count,
            required=False,
        )
        copied_counts["depth"] += _copy_optional_depth_tree(
            chunk_case,
            aggregate_case,
            start_frame=start_frame,
            frame_count=frame_count,
        )
        start_frame += frame_count
    return copied_counts


def _aggregate_metadata(first_metadata: Mapping[str, Any], frame_count: int) -> dict[str, Any]:
    metadata = dict(first_metadata)
    metadata["start_step"] = 0
    metadata["frame_num"] = int(frame_count)
    metadata["end_step"] = int(frame_count)
    return metadata


def _aggregate_manifest(
    *,
    aggregate_case: Path,
    chunk_cases: Sequence[Path],
    frame_count: int,
    ready: bool,
    copied_counts: Mapping[str, int],
    final_data: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "case_dir": str(aggregate_case),
        "frame_count": int(frame_count),
        "chunk_case_count": int(len(chunk_cases)),
        "source_chunk_cases": [str(path) for path in chunk_cases],
        "ready": bool(ready),
        "frame_index_semantics": "received_demo_v5_stream_index",
        "copied_color_frame_files": int(copied_counts.get("color", 0)),
        "copied_pcd_frame_files": int(copied_counts.get("pcd", 0)),
        "copied_depth_frame_files": int(copied_counts.get("depth", 0)),
        "topology_version": str(
            np.asarray(final_data["topology_version"]).item()
            if isinstance(final_data["topology_version"], np.ndarray)
            else final_data["topology_version"]
        ),
        "topology_hash": str(
            np.asarray(final_data["topology_hash"]).item()
            if isinstance(final_data["topology_hash"], np.ndarray)
            else final_data["topology_hash"]
        ),
    }


def build_aggregate_case_from_chunk_cases(
    chunk_cases: Sequence[str | Path],
    aggregate_case: str | Path,
    *,
    ready: bool = False,
) -> dict[str, Any]:
    """Rewrite data/<case> as the prefix aggregate of READY chunk cases."""
    cases = [Path(path) for path in chunk_cases]
    target = Path(aggregate_case)
    _validate_chunk_cases(cases)

    final_payloads = [_load_pickle(case / "final_data.pkl") for case in cases]
    track_payloads = [_load_pickle(case / "track_process_data.pkl") for case in cases]
    final_data = _concatenate_payloads(
        final_payloads,
        time_keys=FINAL_TIME_KEYS,
        static_keys=FINAL_STATIC_KEYS,
        label="final_data.pkl",
    )
    for key in FINAL_FIRST_STATIC_KEYS:
        final_data[key] = np.ascontiguousarray(np.asarray(final_payloads[0][key]))
    track_process = _concatenate_payloads(
        track_payloads,
        time_keys=TRACK_TIME_KEYS,
        static_keys=TRACK_STATIC_KEYS,
        label="track_process_data.pkl",
    )
    for key in TRACK_FIRST_STATIC_KEYS:
        track_process[key] = np.ascontiguousarray(np.asarray(track_payloads[0][key]))
    frame_count = _frame_count_from_payload(final_data)

    _remove_generated_contents(target)
    copied_counts = _write_frame_artifacts(cases, target)
    _write_processed_masks(cases, target)
    _write_tracking(cases, target, name="tracking")
    _write_tracking(cases, target, name="cotracker")
    _atomic_pickle_dump(final_data, target / "final_data.pkl")
    _atomic_pickle_dump(track_process, target / "track_process_data.pkl")
    shutil.copy2(cases[0] / "calibrate.pkl", target / "calibrate.pkl")
    _atomic_json_dump(_aggregate_metadata(_load_json(cases[0] / "metadata.json"), frame_count), target / "metadata.json")
    _atomic_json_dump(_split_payload(frame_count), target / "split.json")

    manifest = _aggregate_manifest(
        aggregate_case=target,
        chunk_cases=cases,
        frame_count=frame_count,
        ready=ready,
        copied_counts=copied_counts,
        final_data=final_data,
    )
    _atomic_json_dump(manifest, target / "manifest.json")
    if ready:
        (target / "READY").write_text("ready\n", encoding="utf-8")
    else:
        ready_path = target / "READY"
        if ready_path.exists():
            ready_path.unlink()
    validate_futurephystwin_case(target, require_ready=ready)
    return manifest


class OnlineAggregateCaseWriter:
    """Incrementally rebuild the aggregate case after each committed chunk."""

    def __init__(self, case_dir: str | Path) -> None:
        self.case_dir = Path(case_dir)
        self.chunk_cases: list[Path] = []

    def validate_next_chunk_case(self, chunk_case_dir: str | Path) -> None:
        _validate_chunk_cases([*self.chunk_cases, Path(chunk_case_dir)])

    def add_chunk_case(self, chunk_case_dir: str | Path) -> dict[str, Any]:
        source = Path(chunk_case_dir)
        candidate_cases = [*self.chunk_cases, source]
        manifest = build_aggregate_case_from_chunk_cases(candidate_cases, self.case_dir, ready=False)
        self.chunk_cases = candidate_cases
        return manifest

    def finish(self) -> dict[str, Any] | None:
        if not self.chunk_cases:
            return None
        return build_aggregate_case_from_chunk_cases(self.chunk_cases, self.case_dir, ready=True)


def _complete_ready_case(case_dir: Path) -> bool:
    try:
        validate_futurephystwin_case(case_dir, require_ready=True)
    except (FileNotFoundError, ValueError, OSError, pickle.PickleError, json.JSONDecodeError):
        return False
    return True


def migrate_legacy_online_static_case(base_path: str | Path, case_name: str) -> dict[str, Any]:
    """Recover data/<case> when chunks exist but the aggregate is missing."""
    base = Path(base_path)
    aggregate_dir = base / "data" / str(case_name)
    chunk_cases = [
        path
        for path in sorted(base.glob(f"{case_name}_chunk_*"))
        if path.is_dir() and (path / "READY").is_file()
    ]
    if not chunk_cases:
        return {
            "migrated": False,
            "case_dir": str(aggregate_dir),
            "chunk_case_count": 0,
            "reason": "no_ready_chunk_cases",
        }
    if _complete_ready_case(aggregate_dir):
        return {
            "migrated": False,
            "case_dir": str(aggregate_dir),
            "chunk_case_count": int(len(chunk_cases)),
            "reason": "already_complete",
        }
    manifest = build_aggregate_case_from_chunk_cases(chunk_cases, aggregate_dir, ready=True)
    return {
        "migrated": True,
        "case_dir": str(aggregate_dir),
        "chunk_case_count": int(len(chunk_cases)),
        "manifest": manifest,
    }


__all__ = [
    "OnlineAggregateCaseWriter",
    "build_aggregate_case_from_chunk_cases",
    "migrate_legacy_online_static_case",
]
