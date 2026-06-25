from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


DEMO_V5_SCHEMA_NAME = "demo_v5_realtime_phystwin"
DEMO_V5_SCHEMA_VERSION = 1
SEMANTIC_UNKNOWN = np.int8(0)
SEMANTIC_OBJECT = np.int8(1)
SEMANTIC_CONTROLLER = np.int8(2)


def as_points(value: Any) -> np.ndarray:
    return np.ascontiguousarray(
        np.asarray(value, dtype=np.float32).reshape(-1, 3), dtype=np.float32
    )


def hash_topology(parts: Sequence[np.ndarray], metadata: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(dict(metadata), sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    for part in parts:
        arr = np.ascontiguousarray(part)
        digest.update(str(arr.dtype).encode("ascii"))
        digest.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
        digest.update(arr.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class DemoV5SessionTopology:
    fps: float
    frame_dt_s: float
    warmup_frame_count: int
    coordinate_frame: str
    query_points_yx: np.ndarray
    query_semantics: np.ndarray
    query_rest_points: np.ndarray
    object_candidate_query_ids: np.ndarray
    controller_candidate_query_ids: np.ndarray
    object_query_ids: np.ndarray
    controller_query_ids: np.ndarray
    topology_hash: str
    schema_name: str = DEMO_V5_SCHEMA_NAME
    schema_version: int = DEMO_V5_SCHEMA_VERSION

    @property
    def object_count(self) -> int:
        return int(self.object_query_ids.shape[0])

    @property
    def controller_count(self) -> int:
        return int(self.controller_query_ids.shape[0])

    def to_metadata(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "schema_version": int(self.schema_version),
            "topology_hash": self.topology_hash,
            "topology_version": 0,
            "fps": float(self.fps),
            "frame_dt_s": float(self.frame_dt_s),
            "warmup_frame_count": int(self.warmup_frame_count),
            "coordinate_frame": self.coordinate_frame,
            "position_unit": "meter",
            "query_count": int(self.query_points_yx.shape[0]),
            "object_candidate_count": int(self.object_candidate_query_ids.shape[0]),
            "controller_candidate_count": int(
                self.controller_candidate_query_ids.shape[0]
            ),
            "object_point_count": self.object_count,
            "controller_point_count": self.controller_count,
        }

    def save(self, root: str | Path) -> tuple[Path, Path]:
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        npz_path = root / "session_topology.npz"
        json_path = root / "session_topology.json"
        tmp_npz = npz_path.with_name(npz_path.name + ".tmp")
        with tmp_npz.open("wb") as handle:
            np.savez_compressed(
                handle,
                query_points_yx=np.ascontiguousarray(
                    self.query_points_yx, dtype=np.float32
                ),
                query_semantics=np.ascontiguousarray(
                    self.query_semantics, dtype=np.int8
                ),
                query_rest_points=np.ascontiguousarray(
                    self.query_rest_points, dtype=np.float32
                ),
                object_candidate_query_ids=np.ascontiguousarray(
                    self.object_candidate_query_ids, dtype=np.int64
                ),
                controller_candidate_query_ids=np.ascontiguousarray(
                    self.controller_candidate_query_ids, dtype=np.int64
                ),
                object_query_ids=np.ascontiguousarray(
                    self.object_query_ids, dtype=np.int64
                ),
                controller_query_ids=np.ascontiguousarray(
                    self.controller_query_ids, dtype=np.int64
                ),
            )
        tmp_npz.replace(npz_path)
        tmp_json = json_path.with_name(json_path.name + ".tmp")
        tmp_json.write_text(
            json.dumps(self.to_metadata(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tmp_json.replace(json_path)
        return npz_path, json_path
