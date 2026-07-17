"""Publish Demo v6.2 chunks in the online final_data format.

The online stream has two views of the same data: small per-window chunk pickle
files for low-latency readers, and a continuously rewritten ``data`` directory
with the current aggregate ``final_data.pkl``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from demo_v6_2.streaming.data_keys import TIME_KEYS
from demo_v6_2.streaming.data_payload import (
    DATA_PROCESS_SAM3D_REALTIME_CONTRACT_VERSION,
    DATA_PROCESS_QUERY_SCHEMA_KEYS,
)
from demo_v6_2.utils.atomic_io import atomic_json_dump, atomic_pickle_dump

# Shape-prior point clouds do not vary per frame; the latest committed value is
# republished with every aggregate rewrite.
STATIC_KEYS = (
    "surface_points",
    "interior_points",
)

# These arrays describe identity or sampling choices, so they are copied as
# static metadata instead of sliced per frame.
FINAL_DATA_STATIC_KEYS = (
    "controller_final_indices",
    "controller_selected_query_ids",
    "controller_sample_query_ids",
    "object_sample_indices",
    "object_selected_query_ids",
    "object_sample_query_ids",
    *DATA_PROCESS_QUERY_SCHEMA_KEYS,
)
OPTIONAL_STATIC_KEYS = ("controller_neighbor_query_ids",)
# Small track_process subset that rides along with each committed final_data
# window so online readers get recovery diagnostics without the full payload.
TRACK_DIAGNOSTIC_KEYS = (
    "controller_proxied",
    "controller_neighbor_query_ids",
    "track_process_status",
)


def _static_mapping_vectors(data: Mapping[str, Any]) -> dict[str, Any]:
    """Copy the strict static point/query mapping vectors for an online chunk."""
    vectors: dict[str, Any] = {}
    for key in DATA_PROCESS_QUERY_SCHEMA_KEYS:
        value = data[key]
        vectors[key] = (
            value
            if isinstance(value, str)
            else np.ascontiguousarray(np.asarray(value))
        )
    for key in (
        "controller_final_indices",
        "controller_selected_query_ids",
        "object_sample_indices",
        "object_selected_query_ids",
    ):
        vectors[key] = np.ascontiguousarray(
            np.asarray(data[key], dtype=np.int64).reshape(-1)
        )
    return vectors


def build_online_chunk_record(
    data: Mapping[str, Any],
    *,
    case_name: str,
    chunk_id: int,
    start_frame: int,
    end_frame: int,
    source_frame_indices: Sequence[int],
    source_timestamps_s: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Create the small per-window payload stored under online_data/chunks."""
    # Offline parity with data_process_origin/data_process_sample.py::process_unique_points.
    # That path produces one final_data.pkl for a completed case. Demo v6.2
    # slices that same final_data contract into per-window online chunks.
    data = dict(data)
    indices = [int(idx) for idx in source_frame_indices]
    chunk: dict[str, Any] = {
        "case_name": str(case_name),
        "chunk_id": int(chunk_id),
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "source_frame_indices": indices,
    }
    if source_timestamps_s is not None:
        timestamps = [float(value) for value in source_timestamps_s]
        if len(timestamps) != len(indices):
            raise ValueError(
                "source_timestamps_s length must match source_frame_indices"
            )
        chunk["source_timestamps_s"] = timestamps
    # TIME_KEYS are indexed by local frame inside this chunk. Static query and
    # sampling vectors are attached below unchanged so every chunk can be read
    # independently.
    local_frames = list(range(0, int(end_frame) - int(start_frame)))
    for key in TIME_KEYS:
        value = data.get(key)
        if value is None:
            continue
        try:
            chunk[key] = value[local_frames]
        except TypeError:
            # Plain Python sequences reject list indices; copy frame by frame.
            chunk[key] = [value[int(idx)] for idx in local_frames]
    chunk.update(_static_mapping_vectors(data))
    for key in OPTIONAL_STATIC_KEYS:
        if key in data:
            chunk[key] = np.ascontiguousarray(np.asarray(data[key]))
    if "track_process_status" in data:
        chunk["track_process_status"] = str(data["track_process_status"])
    return chunk


class ChunkDataWriter:
    """Maintain online chunks and the continuously growing static case.

    The writer is append-only from the perspective of online readers: each
    chunk pickle gets a new monotonic id, while manifest/static-case files are
    atomically rewritten to point at the latest committed prefix.
    """

    def __init__(
        self,
        *,
        base_path: str | Path,
        case_name: str,
        chunk_size: int,
        num_frames_total: int | None = None,
    ) -> None:
        """Initialize ChunkDataWriter."""
        if int(chunk_size) <= 0:
            raise ValueError("chunk_size must be positive")
        self.base_path = Path(base_path)
        self.case_name = str(case_name)
        self.chunk_size = int(chunk_size)
        self.num_frames_total = (
            None if num_frames_total is None else int(num_frames_total)
        )
        self.online_dir = self.base_path / "online_data"
        self.chunks_dir = self.online_dir / "chunks"
        self.static_case_dir = self.base_path / "data"
        self.static_data_path = self.static_case_dir / "final_data.pkl"
        self.latest_committed_chunk = -1
        self.latest_committed_frame = 0
        self.version = 0
        self._time_arrays: dict[str, list[np.ndarray]] = {key: [] for key in TIME_KEYS}
        self._static_arrays: dict[str, Any] = {}
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self.static_case_dir.mkdir(parents=True, exist_ok=True)
        self._write_manifest(status="recording")

    def commit_chunk_data(
        self,
        final_data: Mapping[str, Any],
        track_process: Mapping[str, Any],
        *,
        source_frame_indices: Sequence[int],
        source_timestamps_s: Sequence[float] | None = None,
        status: str = "recording",
    ) -> dict[str, Any]:
        """Commit one in-memory final_data window plus track diagnostics.

        Appends one final_data chunk and rewrites online metadata atomically.
        """
        data = dict(final_data)
        # Only the small TRACK_DIAGNOSTIC_KEYS subset of track_process rides
        # along; the full track payload is not republished per chunk.
        for key in TRACK_DIAGNOSTIC_KEYS:
            if key not in track_process:
                continue
            value = track_process[key]
            data[key] = (
                str(value)
                if key == "track_process_status"
                else np.ascontiguousarray(np.asarray(value))
            )
        # Chunk length is the leading (frame) axis of whichever per-frame
        # tensor is present.
        for key in ("object_points", "controller_points"):
            if key in data and data[key] is not None:
                frame_count = int(np.asarray(data[key]).shape[0])
                break
        else:
            raise KeyError(
                "Cannot infer online frame count: missing object_points/controller_points"
            )
        start_frame = int(self.latest_committed_frame)
        end_frame = start_frame + int(frame_count)
        if len(source_frame_indices) != frame_count:
            raise ValueError("source_frame_indices length must match chunk frame count")
        if source_timestamps_s is not None and len(source_timestamps_s) != frame_count:
            raise ValueError("source_timestamps_s length must match chunk frame count")
        chunk_id = int(self.latest_committed_chunk + 1)
        chunk = build_online_chunk_record(
            data,
            case_name=self.case_name,
            chunk_id=chunk_id,
            start_frame=start_frame,
            end_frame=end_frame,
            source_frame_indices=source_frame_indices,
            source_timestamps_s=source_timestamps_s,
        )
        atomic_pickle_dump(chunk, self.chunks_dir / f"chunk_{chunk_id:06d}.pkl")
        self._append_static_data(data, frame_count=frame_count)
        self.latest_committed_chunk = chunk_id
        self.latest_committed_frame = end_frame
        self.version += 1
        manifest = self._write_manifest(status=status)
        return {
            "online_dir": str(self.online_dir),
            "online_manifest_path": str(self.online_dir / "manifest.json"),
            "online_chunk_path": str(self.chunks_dir / f"chunk_{chunk_id:06d}.pkl"),
            "static_data_path": str(self.static_data_path),
            "online_chunk_id": chunk_id,
            "online_latest_committed_frame": int(self.latest_committed_frame),
            "online_manifest": manifest,
        }

    def finish(self, *, status: str = "finished") -> dict[str, Any]:
        """Publish a terminal online manifest (``finished`` or ``failed``)."""
        self.version += 1
        return self._write_manifest(status=str(status))

    def _append_static_data(self, data: Mapping[str, Any], *, frame_count: int) -> None:
        """Update data/final_data.pkl as a prefix aggregate."""
        # Offline parity with data_process_origin/data_process_sample.py::process_unique_points.
        # That path writes one static final_data.pkl. Demo v6.2 continuously
        # rewrites the same schema as a prefix aggregate for realtime consumers.
        data = dict(data)
        # Time arrays grow by concatenation. Static arrays are overwritten with
        # the latest value, but upstream validation requires those values to be
        # stable for all committed chunks.
        for key in TIME_KEYS:
            value = data.get(key)
            if value is None:
                continue
            arr = np.asarray(value)
            if int(arr.shape[0]) != int(frame_count):
                raise ValueError(
                    f"{key} has {arr.shape[0]} frames, expected {frame_count}"
                )
            self._time_arrays[key].append(np.ascontiguousarray(arr))
        for key in STATIC_KEYS:
            value = data.get(key)
            if value is not None:
                self._static_arrays[key] = np.ascontiguousarray(np.asarray(value))
        for key in OPTIONAL_STATIC_KEYS:
            value = data.get(key)
            if value is not None:
                self._static_arrays[key] = np.ascontiguousarray(np.asarray(value))
        if "track_process_status" in data:
            self._static_arrays["track_process_status"] = str(
                data["track_process_status"]
            )
        self._static_arrays.update(_static_mapping_vectors(data))
        payload: dict[str, Any] = {}
        for key, values in self._time_arrays.items():
            if values:
                payload[key] = np.ascontiguousarray(np.concatenate(values, axis=0))
        for key in FINAL_DATA_STATIC_KEYS:
            value = self._static_arrays.get(key)
            if value is not None:
                payload[key] = value
        for key in OPTIONAL_STATIC_KEYS:
            value = self._static_arrays.get(key)
            if value is not None:
                payload[key] = value
        if "track_process_status" in self._static_arrays:
            payload["track_process_status"] = str(
                self._static_arrays["track_process_status"]
            )
        for key in STATIC_KEYS:
            payload[key] = self._static_arrays.get(
                key,
                np.empty((0, 3), dtype=np.float32),
            )
        atomic_pickle_dump(payload, self.static_data_path)
        metadata = {
            "case_name": self.case_name,
            "runtime_contract": DATA_PROCESS_SAM3D_REALTIME_CONTRACT_VERSION,
            "online_dir": str(self.online_dir),
            "chunk_size": int(self.chunk_size),
            "latest_committed_frame": int(self.latest_committed_frame + frame_count),
        }
        if "query_schema_version" in self._static_arrays:
            metadata["query_schema_version"] = str(
                self._static_arrays["query_schema_version"]
            )
        if "query_schema_hash" in self._static_arrays:
            metadata["query_schema_hash"] = str(
                self._static_arrays["query_schema_hash"]
            )
        atomic_json_dump(metadata, self.static_case_dir / "metadata.json")

    def _write_manifest(self, *, status: str) -> dict[str, Any]:
        """Atomically rewrite online_data/manifest.json for the current state.

        ``version`` increases monotonically with every commit/finish, so
        readers can detect any rewrite even when frame counters are unchanged.
        """
        latest_frame = int(self.latest_committed_frame)
        total = (
            latest_frame
            if self.num_frames_total is None
            else int(self.num_frames_total)
        )
        manifest = {
            "case_name": self.case_name,
            "runtime_contract": DATA_PROCESS_SAM3D_REALTIME_CONTRACT_VERSION,
            "status": str(status),
            "chunk_size": int(self.chunk_size),
            "num_frames_total": int(total),
            "latest_committed_chunk": int(self.latest_committed_chunk),
            "latest_committed_frame": latest_frame,
            "version": int(self.version),
            "source_num_frames_total": int(total),
            "source_start_frame": 0,
            "source_end_frame": int(total),
            "source_frame_step": 1,
            "online_num_frames_total": int(total),
        }
        if "query_schema_version" in self._static_arrays:
            value = self._static_arrays["query_schema_version"]
            manifest["query_schema_version"] = str(
                np.asarray(value).item() if isinstance(value, np.ndarray) else value
            )
        if "query_schema_hash" in self._static_arrays:
            value = self._static_arrays["query_schema_hash"]
            manifest["query_schema_hash"] = str(
                np.asarray(value).item() if isinstance(value, np.ndarray) else value
            )
        atomic_json_dump(manifest, self.online_dir / "manifest.json")
        return manifest


__all__ = [
    "ChunkDataWriter",
]
