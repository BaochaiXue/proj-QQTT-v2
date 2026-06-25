from __future__ import annotations

import json
import os
from pathlib import Path
import pickle
from typing import Any, Mapping, Sequence

import numpy as np

from demo_v5.futurephystwin_chunk_writer import (
    FUTUREPHYSTWIN_TOPOLOGY_KEYS,
    build_topology_payload,
)
from demo_v5.online_case_aggregate import OnlineAggregateCaseWriter
from demo_v5.pickle_compat import dump_pickle_legacy_numpy


TIME_KEYS = (
    "object_points",
    "object_colors",
    "object_visibilities",
    "object_motions_valid",
    "controller_points",
    "asap_object_points_filled",
    "asap_surface_points",
    "asap_interior_points",
)

STATIC_KEYS = (
    "surface_points",
    "interior_points",
)

FINAL_DATA_STATIC_KEYS = (
    "controller_fps_indices",
    "controller_selected_query_ids",
    "controller_sample_query_ids",
    "object_sample_indices",
    "object_selected_query_ids",
    "object_sample_query_ids",
    *FUTUREPHYSTWIN_TOPOLOGY_KEYS,
)


def atomic_pickle_dump(obj: Any, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(target.name + ".tmp")
    with tmp_path.open("wb") as handle:
        dump_pickle_legacy_numpy(obj, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, target)


def atomic_json_dump(obj: Mapping[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(target.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(dict(obj), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, target)


def _infer_frame_count(data: Mapping[str, Any]) -> int:
    for key in ("object_points", "controller_points"):
        if key in data and data[key] is not None:
            return int(np.asarray(data[key]).shape[0])
    raise KeyError("Cannot infer online frame count: missing object_points/controller_points")


def _take_source_frames(value: Any, source_frame_indices: Sequence[int]) -> Any:
    try:
        return value[source_frame_indices]
    except TypeError:
        return [value[int(idx)] for idx in source_frame_indices]


def _as_static_vector(data: Mapping[str, Any], key: str) -> np.ndarray | None:
    value = data.get(key)
    if value is None:
        return None
    return np.ascontiguousarray(np.asarray(value, dtype=np.int64).reshape(-1))


def _static_mapping_vectors(data: Mapping[str, Any]) -> dict[str, Any]:
    vectors: dict[str, Any] = {}
    has_topology = all(key in data for key in FUTUREPHYSTWIN_TOPOLOGY_KEYS)
    if has_topology:
        for key in FUTUREPHYSTWIN_TOPOLOGY_KEYS:
            value = data[key]
            vectors[key] = value if isinstance(value, str) else np.ascontiguousarray(np.asarray(value))
    controller_points = np.asarray(data["controller_points"])
    controller_count = int(controller_points.shape[1])
    controller_fps = _as_static_vector(data, "controller_fps_indices")
    if controller_fps is None or controller_fps.shape[0] != controller_count:
        controller_fps = np.arange(controller_count, dtype=np.int64)
    vectors["controller_fps_indices"] = np.ascontiguousarray(controller_fps, dtype=np.int64)
    selected_controller_ids = _as_static_vector(data, "controller_selected_query_ids")
    if selected_controller_ids is None or selected_controller_ids.shape[0] != controller_count:
        query_ids = _as_static_vector(data, "controller_query_indices")
        selected_controller_ids = np.full((controller_count,), -1, dtype=np.int64)
        if query_ids is not None:
            valid = (controller_fps >= 0) & (controller_fps < query_ids.shape[0])
            selected_controller_ids[valid] = query_ids[controller_fps[valid]]
        else:
            selected_controller_ids = controller_fps.copy()
    vectors["controller_selected_query_ids"] = np.ascontiguousarray(selected_controller_ids, dtype=np.int64)
    controller_sample_query_ids = _as_static_vector(data, "controller_sample_query_ids")
    if controller_sample_query_ids is None or controller_sample_query_ids.shape[0] != controller_count:
        controller_sample_query_ids = selected_controller_ids
    vectors["controller_sample_query_ids"] = np.ascontiguousarray(controller_sample_query_ids, dtype=np.int64)

    object_points = np.asarray(data["object_points"])
    object_count = int(object_points.shape[1])
    object_sample = _as_static_vector(data, "object_sample_indices")
    if object_sample is None or object_sample.shape[0] != object_count:
        object_sample = _as_static_vector(data, "object_volume_sample_indices")
    if object_sample is None or object_sample.shape[0] != object_count:
        object_sample = np.arange(object_count, dtype=np.int64)
    vectors["object_sample_indices"] = np.ascontiguousarray(object_sample, dtype=np.int64)
    selected_object_ids = _as_static_vector(data, "object_selected_query_ids")
    if selected_object_ids is None or selected_object_ids.shape[0] != object_count:
        query_ids = _as_static_vector(data, "object_anchor_query_indices")
        if query_ids is None or query_ids.shape[0] != object_count:
            query_ids = _as_static_vector(data, "object_query_indices")
        if query_ids is not None and query_ids.shape[0] == object_count:
            selected_object_ids = query_ids
        elif query_ids is not None and object_sample.size and int(np.max(object_sample)) < query_ids.shape[0]:
            selected_object_ids = query_ids[object_sample]
        else:
            selected_object_ids = object_sample.copy()
    vectors["object_selected_query_ids"] = np.ascontiguousarray(selected_object_ids, dtype=np.int64)
    object_sample_query_ids = _as_static_vector(data, "object_sample_query_ids")
    if object_sample_query_ids is None or object_sample_query_ids.shape[0] != object_count:
        object_sample_query_ids = selected_object_ids
    vectors["object_sample_query_ids"] = np.ascontiguousarray(object_sample_query_ids, dtype=np.int64)
    if not has_topology:
        vectors.update(
            build_topology_payload(
                {**data, **vectors},
                object_sample_query_ids=vectors["object_sample_query_ids"],
                controller_sample_query_ids=vectors["controller_sample_query_ids"],
            )
        )
    return vectors


def build_online_chunk(
    data: Mapping[str, Any],
    *,
    case_name: str,
    chunk_id: int,
    start_frame: int,
    end_frame: int,
    source_frame_indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    if source_frame_indices is None:
        source_frame_indices = list(range(int(start_frame), int(end_frame)))
    indices = [int(idx) for idx in source_frame_indices]
    chunk: dict[str, Any] = {
        "case_name": str(case_name),
        "chunk_id": int(chunk_id),
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "source_frame_indices": indices,
    }
    for key in TIME_KEYS:
        value = data.get(key)
        if value is not None:
            chunk[key] = _take_source_frames(value, list(range(0, int(end_frame) - int(start_frame))))
    chunk.update(_static_mapping_vectors(data))
    return chunk


class DemoV5OnlineOutputWriter:
    def __init__(
        self,
        *,
        base_path: str | Path,
        case_name: str,
        chunk_size: int,
        num_frames_total: int | None = None,
        source_start_frame: int = 0,
        source_frame_step: int = 1,
    ) -> None:
        if int(chunk_size) <= 0:
            raise ValueError("chunk_size must be positive")
        if int(source_frame_step) <= 0:
            raise ValueError("source_frame_step must be positive")
        self.base_path = Path(base_path)
        self.case_name = str(case_name)
        self.chunk_size = int(chunk_size)
        self.num_frames_total = None if num_frames_total is None else int(num_frames_total)
        self.source_start_frame = int(source_start_frame)
        self.source_frame_step = int(source_frame_step)
        self.online_dir = self.base_path / "online_data" / self.case_name
        self.chunks_dir = self.online_dir / "chunks"
        self.static_case_dir = self.base_path / "data" / self.case_name
        self.static_data_path = self.static_case_dir / "final_data.pkl"
        self.latest_committed_chunk = -1
        self.latest_committed_frame = 0
        self.version = 0
        self._time_arrays: dict[str, list[np.ndarray]] = {key: [] for key in TIME_KEYS}
        self._static_arrays: dict[str, Any] = {}
        self._aggregate_writer = OnlineAggregateCaseWriter(self.static_case_dir)
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self.static_case_dir.mkdir(parents=True, exist_ok=True)
        self._write_manifest(status="recording")

    def commit_final_data_chunk(
        self,
        data: Mapping[str, Any],
        *,
        source_frame_indices: Sequence[int] | None = None,
        status: str = "recording",
    ) -> dict[str, Any]:
        frame_count = _infer_frame_count(data)
        start_frame = int(self.latest_committed_frame)
        end_frame = start_frame + int(frame_count)
        if source_frame_indices is None:
            source_frame_indices = list(
                range(
                    int(self.source_start_frame) + start_frame * int(self.source_frame_step),
                    int(self.source_start_frame) + end_frame * int(self.source_frame_step),
                    int(self.source_frame_step),
                )
            )
        if len(source_frame_indices) != frame_count:
            raise ValueError("source_frame_indices length must match chunk frame count")
        chunk_id = int(self.latest_committed_chunk + 1)
        chunk = build_online_chunk(
            data,
            case_name=self.case_name,
            chunk_id=chunk_id,
            start_frame=start_frame,
            end_frame=end_frame,
            source_frame_indices=source_frame_indices,
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

    def commit_case_chunk(
        self,
        case_dir: str | Path,
        *,
        source_frame_indices: Sequence[int] | None = None,
        status: str = "recording",
    ) -> dict[str, Any]:
        chunk_case_dir = Path(case_dir)
        self._aggregate_writer.validate_next_chunk_case(chunk_case_dir)
        with (chunk_case_dir / "final_data.pkl").open("rb") as handle:
            final_data = pickle.load(handle)
        result = self.commit_final_data_chunk(
            final_data,
            source_frame_indices=source_frame_indices,
            status=status,
        )
        aggregate_manifest = self._aggregate_writer.add_chunk_case(chunk_case_dir)
        result["aggregate_case_dir"] = str(self.static_case_dir)
        result["aggregate_manifest"] = aggregate_manifest
        return result

    def finish(self) -> dict[str, Any]:
        aggregate_manifest = self._aggregate_writer.finish()
        self.version += 1
        manifest = self._write_manifest(status="finished")
        if aggregate_manifest is not None:
            manifest["aggregate_case_dir"] = str(self.static_case_dir)
        return manifest

    def _append_static_data(self, data: Mapping[str, Any], *, frame_count: int) -> None:
        for key in TIME_KEYS:
            value = data.get(key)
            if value is None:
                continue
            arr = np.asarray(value)
            if int(arr.shape[0]) != int(frame_count):
                raise ValueError(f"{key} has {arr.shape[0]} frames, expected {frame_count}")
            self._time_arrays[key].append(np.ascontiguousarray(arr))
        for key in STATIC_KEYS:
            value = data.get(key)
            if value is not None:
                self._static_arrays[key] = np.ascontiguousarray(np.asarray(value))
        self._static_arrays.update(_static_mapping_vectors(data))
        payload: dict[str, Any] = {}
        for key, values in self._time_arrays.items():
            if values:
                payload[key] = np.ascontiguousarray(np.concatenate(values, axis=0))
        for key in FINAL_DATA_STATIC_KEYS:
            value = self._static_arrays.get(key)
            if value is not None:
                payload[key] = value
        for key in STATIC_KEYS:
            payload[key] = self._static_arrays.get(
                key,
                np.empty((0, 3), dtype=np.float32),
            )
        atomic_pickle_dump(payload, self.static_data_path)
        metadata = {
            "case_name": self.case_name,
            "demo_version": "demo_v5",
            "runtime_product_name": "demo_v5_realtime_camera_final_data",
            "reference_pipeline": "data_process_sam3d",
            "online_dir": str(self.online_dir),
            "chunk_size": int(self.chunk_size),
            "latest_committed_frame": int(self.latest_committed_frame + frame_count),
        }
        if "topology_version" in self._static_arrays:
            metadata["topology_version"] = str(self._static_arrays["topology_version"])
        if "topology_hash" in self._static_arrays:
            metadata["topology_hash"] = str(self._static_arrays["topology_hash"])
        atomic_json_dump(metadata, self.static_case_dir / "metadata.json")

    def _write_manifest(self, *, status: str) -> dict[str, Any]:
        latest_frame = int(self.latest_committed_frame)
        total = latest_frame if self.num_frames_total is None else int(self.num_frames_total)
        source_end = int(self.source_start_frame) + int(total) * int(self.source_frame_step)
        manifest = {
            "case_name": self.case_name,
            "demo_version": "demo_v5",
            "runtime_product_name": "demo_v5_realtime_camera_final_data",
            "reference_pipeline": "data_process_sam3d",
            "status": str(status),
            "chunk_size": int(self.chunk_size),
            "num_frames_total": int(total),
            "latest_committed_chunk": int(self.latest_committed_chunk),
            "latest_committed_frame": latest_frame,
            "version": int(self.version),
            "source_num_frames_total": int(total),
            "source_start_frame": int(self.source_start_frame),
            "source_end_frame": int(source_end),
            "source_frame_step": int(self.source_frame_step),
            "online_num_frames_total": int(total),
        }
        if "topology_version" in self._static_arrays:
            value = self._static_arrays["topology_version"]
            manifest["topology_version"] = str(np.asarray(value).item() if isinstance(value, np.ndarray) else value)
        if "topology_hash" in self._static_arrays:
            value = self._static_arrays["topology_hash"]
            manifest["topology_hash"] = str(np.asarray(value).item() if isinstance(value, np.ndarray) else value)
        atomic_json_dump(manifest, self.online_dir / "manifest.json")
        return manifest


__all__ = [
    "TIME_KEYS",
    "STATIC_KEYS",
    "DemoV4OnlineOutputWriter",
    "DemoV5OnlineOutputWriter",
    "atomic_json_dump",
    "atomic_pickle_dump",
    "build_online_chunk",
]


DemoV4OnlineOutputWriter = DemoV5OnlineOutputWriter
