from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .base import TrackingResult


def _validate_phystwin_arrays(tracks_yx: np.ndarray, visibility: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tracks = np.asarray(tracks_yx, dtype=np.float32)
    vis = np.asarray(visibility, dtype=np.float32)
    if tracks.ndim != 3 or tracks.shape[-1] != 2:
        raise ValueError(f"tracks_yx must have shape (T,N,2); got {tracks.shape}")
    if vis.shape != tracks.shape[:2] and vis.shape != (*tracks.shape[:2], 1):
        raise ValueError(f"visibility must have shape (T,N) or (T,N,1); got {vis.shape} for tracks {tracks.shape}")
    return tracks, vis


def save_phystwin_tracking_npz(
    path: str | Path,
    *,
    tracks_yx: np.ndarray,
    visibility: np.ndarray,
    confidence: np.ndarray | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tracks, vis = _validate_phystwin_arrays(tracks_yx, visibility)
    payload_metadata = dict(metadata or {})
    coordinate_order = str(payload_metadata.get("coordinate_order", "yx")).lower()
    if coordinate_order != "yx":
        raise ValueError(f"PhysTwin tracking metadata coordinate_order must be yx; got {coordinate_order!r}")
    payload_metadata["coordinate_order"] = "yx"
    payload_metadata.setdefault("query_frame_idx", 0)
    payload_metadata.setdefault("num_query_points", int(tracks.shape[1]))
    payload: dict[str, Any] = {
        "tracks": tracks.astype(np.float32),
        "visibility": vis.astype(np.float32),
        "metadata_json": np.asarray(json.dumps(payload_metadata, indent=2, sort_keys=True)),
    }
    if confidence is not None:
        conf = np.asarray(confidence, dtype=np.float32)
        if conf.shape != tracks.shape[:2] and conf.shape != (*tracks.shape[:2], 1):
            raise ValueError(f"confidence must have shape (T,N) or (T,N,1); got {conf.shape}")
        payload["confidence"] = conf.astype(np.float32)
    np.savez_compressed(output_path, **payload)
    return output_path


def load_phystwin_tracking_npz(path: str | Path) -> TrackingResult:
    with np.load(Path(path), allow_pickle=False) as data:
        tracks = data["tracks"].astype(np.float32)
        visibility = data["visibility"].astype(np.float32)
        confidence = data["confidence"].astype(np.float32) if "confidence" in data.files else None
        track_ids = data["track_ids"] if "track_ids" in data.files else None
        query_points_yx = data["query_points_yx"].astype(np.float32) if "query_points_yx" in data.files else None
        raw_metadata = str(data["metadata_json"].item()) if "metadata_json" in data.files else str(data["metadata"].item()) if "metadata" in data.files else "{}"
    metadata = json.loads(raw_metadata)
    if str(metadata.get("coordinate_order", "yx")).lower() != "yx":
        raise ValueError(f"Tracking NPZ coordinate_order must be yx; got {metadata.get('coordinate_order')!r}")
    _validate_phystwin_arrays(tracks, visibility)
    return TrackingResult(
        tracks_yx=tracks,
        visibility=visibility,
        confidence=confidence,
        backend=str(metadata.get("backend", "")),
        camera_idx=metadata.get("camera_idx"),
        coordinate_order="yx",
        stats={"metadata": metadata},
        track_ids=track_ids,
        query_points_yx=query_points_yx,
    )


def save_cotracker_like_npz(result: TrackingResult, path: str | Path, *, camera_idx: int | None = None, metadata: dict[str, Any] | None = None) -> Path:
    cam_idx = result.camera_idx if camera_idx is None else camera_idx
    payload_metadata: dict[str, Any] = {
        "backend": result.backend,
        "coordinate_order": "yx",
        "camera_idx": None if cam_idx is None else int(cam_idx),
        "query_frame_idx": 0,
        "num_query_points": int(result.tracks_yx.shape[1]),
    }
    if metadata:
        payload_metadata.update(metadata)
    output_path = save_phystwin_tracking_npz(
        path,
        tracks_yx=result.tracks_yx,
        visibility=result.visibility,
        confidence=result.confidence,
        metadata=payload_metadata,
    )
    if result.track_ids is None and result.query_points_yx is None:
        return output_path
    with np.load(output_path, allow_pickle=False) as data:
        payload = {name: data[name] for name in data.files}
    if result.track_ids is not None:
        payload["track_ids"] = np.asarray(result.track_ids)
    if result.query_points_yx is not None:
        payload["query_points_yx"] = result.query_points_yx.astype(np.float32)
    np.savez_compressed(output_path, **payload)
    return output_path


def load_cotracker_like_npz(path: str | Path) -> tuple[TrackingResult, dict[str, Any]]:
    result = load_phystwin_tracking_npz(path)
    return result, dict(result.stats.get("metadata", {}))
