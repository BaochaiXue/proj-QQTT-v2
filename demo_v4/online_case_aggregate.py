from __future__ import annotations

import json
import os
from pathlib import Path
import pickle
import shutil
from typing import Any, Mapping

import numpy as np


TRACK_PROCESS_KEYS = (
    "controller_mask",
    "controller_points",
    "object_colors",
    "object_motions_valid",
    "object_points",
    "object_visibilities",
)


def _atomic_pickle_dump(obj: Any, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(target.name + ".tmp")
    with tmp_path.open("wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, target)


def _atomic_json_dump(obj: Mapping[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(target.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(dict(obj), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, target)


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _frame_count_from_final_data(case_dir: Path) -> int:
    payload = _load_pickle(case_dir / "final_data.pkl")
    return int(np.asarray(payload["object_points"]).shape[0])


def _split_payload(frame_count: int) -> dict[str, Any]:
    train_end = max(1, int(int(frame_count) * 0.7))
    train_end = min(train_end, int(frame_count))
    return {
        "frame_len": int(frame_count),
        "train": [0, int(train_end)],
        "test": [int(train_end), int(frame_count)],
    }


def _write_track_process_from_final_data(case_dir: Path) -> None:
    final_data = _load_pickle(case_dir / "final_data.pkl")
    payload = {
        key: np.ascontiguousarray(np.asarray(final_data[key]))
        for key in TRACK_PROCESS_KEYS
    }
    _atomic_pickle_dump(payload, case_dir / "track_process_data.pkl")


def _copy_color_frames(source_case: Path, aggregate_case: Path, *, start_frame: int, frame_count: int) -> None:
    target_dir = aggregate_case / "color" / "0"
    target_dir.mkdir(parents=True, exist_ok=True)
    for frame_idx in range(int(frame_count)):
        shutil.copy2(
            source_case / "color" / "0" / f"{frame_idx}.png",
            target_dir / f"{int(start_frame) + frame_idx}.png",
        )


def _append_processed_masks(source_case: Path, aggregate_case: Path) -> None:
    target_path = aggregate_case / "mask" / "processed_masks.pkl"
    existing = _load_pickle(target_path) if target_path.is_file() else []
    source = _load_pickle(source_case / "mask" / "processed_masks.pkl")
    _atomic_pickle_dump(list(existing) + list(source), target_path)


def _append_tracking_npz(source_case: Path, aggregate_case: Path, *, name: str) -> None:
    target_dir = aggregate_case / name
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / "0.npz"
    with np.load(source_case / name / "0.npz") as source:
        source_tracks = np.ascontiguousarray(np.asarray(source["tracks"], dtype=np.float32))
        source_visibility = np.ascontiguousarray(np.asarray(source["visibility"], dtype=bool))
        source_queries = np.ascontiguousarray(np.asarray(source["queries_txy"], dtype=np.float32))
    if target_path.is_file():
        with np.load(target_path) as existing:
            tracks = np.concatenate([np.asarray(existing["tracks"], dtype=np.float32), source_tracks], axis=0)
            visibility = np.concatenate([np.asarray(existing["visibility"], dtype=bool), source_visibility], axis=0)
            queries = np.ascontiguousarray(np.asarray(existing["queries_txy"], dtype=np.float32))
    else:
        tracks = source_tracks
        visibility = source_visibility
        queries = source_queries
    tmp_path = target_path.with_name(target_path.name + ".tmp.npz")
    np.savez(
        tmp_path,
        tracks=np.ascontiguousarray(tracks),
        visibility=np.ascontiguousarray(visibility),
        queries_txy=queries,
    )
    os.replace(tmp_path, target_path)


def _copy_optional_pcd_frames(source_case: Path, aggregate_case: Path, *, start_frame: int, frame_count: int) -> None:
    source_dir = source_case / "pcd"
    if not source_dir.is_dir():
        return
    target_dir = aggregate_case / "pcd"
    target_dir.mkdir(parents=True, exist_ok=True)
    for frame_idx in range(int(frame_count)):
        source_path = source_dir / f"{frame_idx}.npz"
        if source_path.is_file():
            shutil.copy2(source_path, target_dir / f"{int(start_frame) + frame_idx}.npz")


class OnlineAggregateCaseWriter:
    def __init__(self, case_dir: str | Path) -> None:
        self.case_dir = Path(case_dir)
        self.chunk_cases: list[Path] = []
        self._next_frame = 0

    def add_chunk_case(self, chunk_case_dir: str | Path) -> dict[str, Any]:
        source = Path(chunk_case_dir)
        if not (source / "READY").is_file():
            raise ValueError(f"source chunk case is not READY: {source}")
        if not (self.case_dir / "final_data.pkl").is_file():
            raise ValueError(f"aggregate final_data.pkl is missing: {self.case_dir / 'final_data.pkl'}")

        ready = self.case_dir / "READY"
        if ready.exists():
            ready.unlink()
        self.case_dir.mkdir(parents=True, exist_ok=True)
        frame_count = _frame_count_from_final_data(source)
        start_frame = int(self._next_frame)

        if not (self.case_dir / "calibrate.pkl").is_file():
            shutil.copy2(source / "calibrate.pkl", self.case_dir / "calibrate.pkl")
        _copy_color_frames(source, self.case_dir, start_frame=start_frame, frame_count=frame_count)
        _append_processed_masks(source, self.case_dir)
        _append_tracking_npz(source, self.case_dir, name="tracking")
        _append_tracking_npz(source, self.case_dir, name="cotracker")
        _copy_optional_pcd_frames(source, self.case_dir, start_frame=start_frame, frame_count=frame_count)
        _write_track_process_from_final_data(self.case_dir)

        source_metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
        total_frames = _frame_count_from_final_data(self.case_dir)
        source_metadata["frame_num"] = int(total_frames)
        source_metadata["end_step"] = int(total_frames)
        _atomic_json_dump(source_metadata, self.case_dir / "metadata.json")
        _atomic_json_dump(_split_payload(total_frames), self.case_dir / "split.json")

        self._next_frame += frame_count
        self.chunk_cases.append(source)
        ready.write_text("ready\n", encoding="utf-8")
        return self._manifest(total_frames)

    def finish(self) -> dict[str, Any] | None:
        if not self.chunk_cases:
            return None
        frame_count = _frame_count_from_final_data(self.case_dir)
        ready = self.case_dir / "READY"
        if not ready.is_file():
            ready.write_text("ready\n", encoding="utf-8")
        return self._manifest(frame_count)

    def _manifest(self, frame_count: int) -> dict[str, Any]:
        manifest = {
            "case_dir": str(self.case_dir),
            "frame_count": int(frame_count),
            "chunk_case_count": int(len(self.chunk_cases)),
            "source_chunk_cases": [str(path) for path in self.chunk_cases],
        }
        _atomic_json_dump(manifest, self.case_dir / "manifest.json")
        return manifest


def migrate_legacy_online_static_case(base_path: str | Path, case_name: str) -> dict[str, Any]:
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
        }
    writer = OnlineAggregateCaseWriter(aggregate_dir)
    for chunk_case in chunk_cases:
        writer.add_chunk_case(chunk_case)
    manifest = writer.finish() or {}
    return {
        "migrated": True,
        "case_dir": str(aggregate_dir),
        "chunk_case_count": int(len(chunk_cases)),
        "manifest": manifest,
    }


__all__ = [
    "OnlineAggregateCaseWriter",
    "migrate_legacy_online_static_case",
]
