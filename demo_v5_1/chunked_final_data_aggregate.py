"""Build the static data/<case> view from published Demo v5.1 chunk cases.

Chunk cases are individually valid data_process_sam3d cases. Aggregation
concatenates frame-major arrays, copies per-frame artifacts into a single index
space, and rejects any chunk whose static camera/query identity differs.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import pickle
import shutil
from typing import Any, Mapping, Sequence

import numpy as np

from demo_v5_1.data_process_chunk_writer import (
    DATA_PROCESS_FINAL_DATA_KEYS,
    DATA_PROCESS_QUERY_SCHEMA_KEYS,
    DATA_PROCESS_TRACK_PROCESS_KEYS,
    validate_data_process_case,
)
from demo_v5_1.tools.atomic_io import (
    atomic_json_dump as _atomic_json_dump,
    atomic_pickle_dump as _atomic_pickle_dump,
)
from demo_v5_1.tools.io import load_json as _load_json
from demo_v5_1.tools.io import load_pickle as _load_pickle


FINAL_TIME_KEYS = (
    "controller_points",
    "object_colors",
    "object_motions_valid",
    "object_points",
    "object_visibilities",
)
FINAL_STATIC_KEYS = (
    "controller_final_indices",
    "controller_selected_query_ids",
    "controller_sample_query_ids",
    "object_sample_indices",
    "object_selected_query_ids",
    "object_sample_query_ids",
    *DATA_PROCESS_QUERY_SCHEMA_KEYS,
    "surface_points",
    "interior_points",
)
FINAL_FIRST_STATIC_KEYS: tuple[str, ...] = ()
# Track-process payloads mirror final_data for realtime readers but keep extra
# controller diagnostic arrays. Time keys concatenate; static keys must match
# exactly across all chunks.
TRACK_TIME_KEYS = (
    "controller_points",
    "object_colors",
    "object_motions_valid",
    "object_points",
    "object_visibilities",
)
TRACK_OPTIONAL_TIME_KEYS = (
    "controller_source_query_ids",
    "controller_track_mode",
    "controller_track_confidence",
    "controller_filter_reason",
    "controller_neighbor_support_count",
    "controller_neighbor_raw_visible_count",
    "controller_neighbor_depth_valid_count",
    "controller_neighbor_processed_mask_valid_count",
    "controller_neighbor_motion_valid_count",
    "controller_neighbor_fit_residual",
)
TRACK_STATIC_KEYS = DATA_PROCESS_QUERY_SCHEMA_KEYS
TRACK_FIRST_STATIC_KEYS = ("controller_mask",)
TRACK_OPTIONAL_FIRST_STATIC_KEYS = (
    "controller_neighbor_query_ids",
    "track_process_status",
)
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
SCALAR_STATIC_KEYS = (
    "query_schema_version",
    "query_schema_hash",
)
GENERATED_FILES = (
    "final_data.pkl",
    "track_process_data.pkl",
    "calibrate.pkl",
    "metadata.json",
    "split.json",
    "manifest.json",
    "READY",
    "DEGRADED",
    "INVALID",
)
GENERATED_DIRS = (
    "color",
    "mask",
    "tracking",
    "cotracker",
    "pcd",
    "depth",
)


def _frame_count_from_payload(payload: Mapping[str, Any]) -> int:
    return int(np.asarray(payload["object_points"]).shape[0])


def _require_matching_array_invariant(
    name: str,
    expected: np.ndarray,
    actual: Any,
) -> None:
    if not isinstance(actual, np.ndarray):
        raise ValueError(f"aggregate invariant mismatch for {name}")
    if (
        expected.shape != actual.shape
        or expected.dtype != actual.dtype
        or not np.array_equal(expected, actual)
    ):
        raise ValueError(f"aggregate invariant mismatch for {name}")


def _require_matching_scalar_invariant(name: str, expected: Any, actual: Any) -> None:
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"aggregate invariant mismatch for {name}")


def _require_matching_json_invariant(name: str, expected: Any, actual: Any) -> None:
    if actual != expected:
        raise ValueError(f"aggregate invariant mismatch for {name}")


def _normalize_static_invariant_for_compare(
    name: str,
    key: str,
    value: Any,
) -> Any:
    if key in SCALAR_STATIC_KEYS:
        if not isinstance(value, str):
            raise ValueError(f"aggregate invariant mismatch for {name}")
        return value
    if not isinstance(value, np.ndarray):
        raise ValueError(f"aggregate invariant mismatch for {name}")
    return np.ascontiguousarray(value)


def _require_matching_static_invariant(
    key: str,
    name: str,
    expected: Any,
    actual: Any,
) -> None:
    if key in SCALAR_STATIC_KEYS:
        _require_matching_scalar_invariant(name, expected, actual)
        return
    _require_matching_array_invariant(name, expected, actual)


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
    # Offline parity with data_process_sam3d/data_process_sample.py:L335-L352.
    # That path writes one whole-case final_data.pkl. Demo v5.1 concatenates the
    # same time-varying keys from realtime chunks while keeping static keys
    # invariant.
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
        first = _normalize_static_invariant_for_compare(
            f"{label}.{key}",
            key,
            payloads[0][key],
        )
        for chunk_idx, payload in enumerate(payloads[1:], start=1):
            value = _normalize_static_invariant_for_compare(
                f"{label}.{key} at chunk {chunk_idx}",
                key,
                payload[key],
            )
            _require_matching_static_invariant(
                key,
                f"{label}.{key} at chunk {chunk_idx}",
                first,
                value,
            )
        combined[key] = first
    return combined


def _concatenate_optional_time_keys(
    combined: dict[str, Any],
    payloads: Sequence[Mapping[str, Any]],
    *,
    keys: Sequence[str],
    label: str,
) -> None:
    """Concatenate optional diagnostics only when every chunk provides them."""
    for key in keys:
        if not all(key in payload for payload in payloads):
            continue
        arrays = [np.ascontiguousarray(np.asarray(payload[key])) for payload in payloads]
        tail_shape = arrays[0].shape[1:]
        for chunk_idx, arr in enumerate(arrays):
            if arr.ndim < 1 or arr.shape[1:] != tail_shape:
                raise ValueError(f"cannot concatenate optional {label}.{key} at chunk {chunk_idx}")
        combined[key] = np.ascontiguousarray(np.concatenate(arrays, axis=0))


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


def _degraded_chunk_case_allowed(chunk_case: Path) -> bool:
    if not (chunk_case / "DEGRADED").is_file():
        return False
    try:
        manifest = _load_json(chunk_case / "manifest.json")
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return False
    return str(manifest.get("track_process_status", "normal")) == "degraded"


def _validate_chunk_cases(chunk_cases: Sequence[Path], *, allow_degraded: bool = False) -> None:
    """Ensure chunks can be concatenated without changing camera/query identity."""
    # Offline parity with data_process_sam3d/data_process_track.py:L462-L463
    # and data_process_sam3d/data_process_sample.py:L437-L440. Those paths
    # assume one stable case. Demo v5.1 validates that all chunks share those
    # stable case fields.
    if not chunk_cases:
        raise ValueError("aggregate requires at least one READY chunk case")
    first_metadata: dict[str, Any] | None = None
    first_calibrate: np.ndarray | None = None
    first_queries: dict[str, np.ndarray] = {}
    first_final_static: dict[str, np.ndarray] = {}
    first_track_static: dict[str, np.ndarray] = {}
    for chunk_idx, chunk_case in enumerate(chunk_cases):
        require_ready = not (bool(allow_degraded) and _degraded_chunk_case_allowed(chunk_case))
        validate_data_process_case(chunk_case, require_ready=require_ready)
        final_data = dict(_load_pickle(chunk_case / "final_data.pkl"))
        track_process = dict(_load_pickle(chunk_case / "track_process_data.pkl"))
        _require_payload_keys(final_data, DATA_PROCESS_FINAL_DATA_KEYS, label="final_data.pkl")
        _require_payload_keys(track_process, DATA_PROCESS_TRACK_PROCESS_KEYS, label="track_process_data.pkl")

        metadata = _load_json(chunk_case / "metadata.json")
        missing_metadata = [key for key in METADATA_INVARIANT_KEYS if key not in metadata]
        if missing_metadata:
            raise ValueError(f"metadata.json missing aggregate invariant keys: {missing_metadata}")
        calibrate = _load_calibrate_matrix(chunk_case)
        if first_metadata is None:
            first_metadata = metadata
            first_calibrate = calibrate
            # The first chunk defines aggregate invariants. Later chunks can
            # extend time, but cannot reinterpret cameras, calibration, or
            # query/sample ids.
            first_final_static = {
                key: _normalize_static_invariant_for_compare(
                    f"final_data.pkl {key}",
                    key,
                    final_data[key],
                )
                for key in FINAL_STATIC_KEYS
            }
            first_track_static = {
                key: _normalize_static_invariant_for_compare(
                    f"track_process_data.pkl {key}",
                    key,
                    track_process[key],
                )
                for key in TRACK_STATIC_KEYS
            }
        else:
            assert first_calibrate is not None
            _require_matching_array_invariant("calibrate.pkl", first_calibrate, calibrate)
            for key in METADATA_INVARIANT_KEYS:
                _require_matching_json_invariant(
                    f"metadata.json {key}",
                    first_metadata.get(key),
                    metadata.get(key),
                )
            for key, expected in first_final_static.items():
                _require_matching_static_invariant(
                    key,
                    f"final_data.pkl {key}",
                    expected,
                    _normalize_static_invariant_for_compare(
                        f"final_data.pkl {key}",
                        key,
                        final_data[key],
                    ),
                )
            for key, expected in first_track_static.items():
                _require_matching_static_invariant(
                    key,
                    f"track_process_data.pkl {key}",
                    expected,
                    _normalize_static_invariant_for_compare(
                        f"track_process_data.pkl {key}",
                        key,
                        track_process[key],
                    ),
                )

        for name in ("tracking", "cotracker"):
            tracking = _load_tracking_payload(chunk_case, name)
            queries = tracking["queries_txy"]
            if name not in first_queries:
                first_queries[name] = queries
            else:
                _require_matching_array_invariant(
                    f"{name}/0.npz queries_txy",
                    first_queries[name],
                    queries,
                )
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
    """Copy per-frame files while renumbering local chunk frames globally."""
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
        _require_matching_array_invariant(
            f"{name}/0.npz queries_txy at chunk {chunk_idx}",
            first_queries,
            payload["queries_txy"],
        )
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
        "frame_index_semantics": "received_demo_v5_1_stream_index",
        "copied_color_frame_files": int(copied_counts.get("color", 0)),
        "copied_pcd_frame_files": int(copied_counts.get("pcd", 0)),
        "copied_depth_frame_files": int(copied_counts.get("depth", 0)),
        "query_schema_version": str(
            np.asarray(final_data["query_schema_version"]).item()
            if isinstance(final_data["query_schema_version"], np.ndarray)
            else final_data["query_schema_version"]
        ),
        "query_schema_hash": str(
            np.asarray(final_data["query_schema_hash"]).item()
            if isinstance(final_data["query_schema_hash"], np.ndarray)
            else final_data["query_schema_hash"]
        ),
    }


def build_aggregate_case_from_chunk_cases(
    chunk_cases: Sequence[str | Path],
    aggregate_case: str | Path,
    *,
    ready: bool = False,
    allow_degraded: bool = False,
) -> dict[str, Any]:
    """Rewrite data/<case> as the prefix aggregate of READY chunk cases."""
    cases = [Path(path) for path in chunk_cases]
    target = Path(aggregate_case)
    _validate_chunk_cases(cases, allow_degraded=bool(allow_degraded))

    # final_data.pkl and track_process_data.pkl share the frame axis but have
    # different static/diagnostic fields, so aggregate them in parallel and
    # write the two contract files separately.
    final_payloads = [dict(_load_pickle(case / "final_data.pkl")) for case in cases]
    track_payloads = [dict(_load_pickle(case / "track_process_data.pkl")) for case in cases]
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
    _concatenate_optional_time_keys(
        track_process,
        track_payloads,
        keys=TRACK_OPTIONAL_TIME_KEYS,
        label="track_process_data.pkl",
    )
    for key in TRACK_FIRST_STATIC_KEYS:
        track_process[key] = np.ascontiguousarray(np.asarray(track_payloads[0][key]))
    for key in TRACK_OPTIONAL_FIRST_STATIC_KEYS:
        if key in track_payloads[0]:
            value = track_payloads[0][key]
            track_process[key] = str(value) if key == "track_process_status" else np.ascontiguousarray(np.asarray(value))
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
    _atomic_json_dump(
        {
            "frame_len": int(frame_count),
            "train": [0, int(frame_count)],
            "test": [int(frame_count), int(frame_count)],
        },
        target / "split.json",
    )

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
    validate_data_process_case(target, require_ready=ready)
    return manifest


class FinalDataAggregateWriter:
    """Incrementally rebuild the aggregate case after each committed chunk."""

    def __init__(self, case_dir: str | Path, *, allow_degraded: bool = False) -> None:
        self.case_dir = Path(case_dir)
        self.chunk_cases: list[Path] = []
        self.allow_degraded = bool(allow_degraded)

    def validate_next_chunk_case(self, chunk_case_dir: str | Path) -> None:
        """Validate that a candidate chunk preserves aggregate invariants."""
        _validate_chunk_cases([*self.chunk_cases, Path(chunk_case_dir)], allow_degraded=self.allow_degraded)

    def add_chunk_case(self, chunk_case_dir: str | Path) -> dict[str, Any]:
        """Append one chunk case and rewrite the aggregate as a non-ready prefix."""
        source = Path(chunk_case_dir)
        candidate_cases = [*self.chunk_cases, source]
        manifest = build_aggregate_case_from_chunk_cases(
            candidate_cases,
            self.case_dir,
            ready=False,
            allow_degraded=self.allow_degraded,
        )
        self.chunk_cases = candidate_cases
        return manifest

    def finish(self) -> dict[str, Any] | None:
        """Mark the aggregate ready after all committed chunks are present."""
        if not self.chunk_cases:
            return None
        return build_aggregate_case_from_chunk_cases(
            self.chunk_cases,
            self.case_dir,
            ready=True,
            allow_degraded=self.allow_degraded,
        )


def _complete_ready_case(case_dir: Path) -> bool:
    try:
        validate_data_process_case(case_dir, require_ready=True)
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
    "FinalDataAggregateWriter",
    "build_aggregate_case_from_chunk_cases",
    "migrate_legacy_online_static_case",
]
