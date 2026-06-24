from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import pickle
import shutil
from typing import Any, Callable, Mapping, Sequence
import uuid

import numpy as np
from PIL import Image


FUTUREPHYSTWIN_FINAL_DATA_KEYS = (
    "controller_mask",
    "controller_points",
    "object_colors",
    "object_motions_valid",
    "object_points",
    "object_visibilities",
    "surface_points",
    "interior_points",
)

FUTUREPHYSTWIN_TRACK_PROCESS_KEYS = (
    "controller_mask",
    "controller_points",
    "object_colors",
    "object_motions_valid",
    "object_points",
    "object_visibilities",
)

DATA_PROCESS_SAM3D_METRICS = {
    "mask_radius_outlier_filter_source": "data_process_sam3d/data_process_mask.py::process_pcd_mask",
    "mask_radius_outlier_radius_m": 0.01,
    "mask_radius_outlier_nb_points": 40,
    "semantic_filter": "first_frame_object_controller_labels_then_per_frame_mask_visibility",
    "motion_filter_source": "data_process_sam3d/data_process_track.py::filter_motion",
    "motion_neighbor_dist_m": 0.01,
    "motion_min_neighbors": 5,
    "motion_similarity_m": 0.005,
    "controller_visibility_policy": "visible_for_whole_chunk_then_motion_consistent",
    "controller_fps_count": 30,
    "object_sampling_source": "data_process_sam3d/data_process_sample.py::process_unique_points",
    "object_volume_sample_size_m": 0.005,
    "shape_prior_sampling_backend": "sam3d-single-view",
    "shape_prior_sampling_source": "data_process_sam3d/data_process_sample.py",
    "shape_prior_target_surface_points": 700,
    "shape_prior_target_interior_points": 1000,
    "shape_prior_volume_sample_size_m": 0.005,
    "shape_prior_effective_max_dist_m": 0.035,
    "shape_prior_uses_mvsam3d": False,
    "shape_prior_ground_policy": "preserve",
}


@dataclass(frozen=True)
class FuturePhysTwinChunk:
    rgb_frames: Sequence[np.ndarray]
    processed_masks: Sequence[Sequence[Mapping[str, np.ndarray]]]
    track_process_data: Mapping[str, np.ndarray]
    intrinsics: np.ndarray
    camera_to_world_c2w: np.ndarray
    tracks_yx: np.ndarray | None = None
    tracker_visibility: np.ndarray | None = None
    queries_txy: np.ndarray | None = None
    surface_points: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float64))
    interior_points: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float64))
    pcd_points: np.ndarray | None = None
    pcd_colors: np.ndarray | None = None
    fps: int = 5
    serial_number: str = "demo-v4-single-camera"
    depth_backend: str = ""
    depth_source_internal: str = ""
    chunk_index: int | None = None
    source_frame_indices: Sequence[int] | None = None


def _array(name: str, value: Any, shape_tail: tuple[int, ...] | None = None) -> np.ndarray:
    arr = np.asarray(value)
    if shape_tail is not None and tuple(arr.shape[-len(shape_tail):]) != tuple(shape_tail):
        raise ValueError(f"{name} must end with shape {shape_tail}, got {arr.shape}")
    return np.ascontiguousarray(arr)


def _ensure_frame_count(chunk: FuturePhysTwinChunk) -> int:
    frame_count = len(chunk.rgb_frames)
    if frame_count <= 0:
        raise ValueError("FuturePhysTwin chunk requires at least one RGB frame")
    track_points = np.asarray(chunk.track_process_data["object_points"])
    if track_points.ndim != 3 or track_points.shape[0] != frame_count or track_points.shape[-1] != 3:
        raise ValueError("track_process_data['object_points'] must have shape T,N,3 matching RGB frame count")
    if len(chunk.processed_masks) != frame_count:
        raise ValueError("processed_masks must have one entry per RGB frame")
    return int(frame_count)


def _write_rgb_frames(case_dir: Path, rgb_frames: Sequence[np.ndarray]) -> None:
    color_dir = case_dir / "color" / "0"
    color_dir.mkdir(parents=True, exist_ok=True)
    for frame_idx, frame in enumerate(rgb_frames):
        rgb = np.asarray(frame, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"rgb frame {frame_idx} must be HxWx3, got {rgb.shape}")
        Image.fromarray(np.ascontiguousarray(rgb, dtype=np.uint8), mode="RGB").save(
            color_dir / f"{frame_idx}.png",
            compress_level=1,
        )


def _write_processed_masks(case_dir: Path, processed_masks: Sequence[Sequence[Mapping[str, np.ndarray]]]) -> None:
    mask_dir = case_dir / "mask"
    mask_dir.mkdir(parents=True, exist_ok=True)
    normalized: list[list[dict[str, np.ndarray]]] = []
    for frame in processed_masks:
        camera_entries: list[dict[str, np.ndarray]] = []
        for camera_frame in frame:
            if "object" not in camera_frame or "controller" not in camera_frame:
                raise ValueError("processed mask entries require object and controller masks")
            camera_entries.append(
                {
                    key: np.ascontiguousarray(np.asarray(value, dtype=bool))
                    for key, value in camera_frame.items()
                }
            )
        normalized.append(camera_entries)
    with (mask_dir / "processed_masks.pkl").open("wb") as handle:
        pickle.dump(normalized, handle)


def _write_tracking(case_dir: Path, chunk: FuturePhysTwinChunk, frame_count: int) -> None:
    if chunk.tracks_yx is None:
        tracks = np.zeros((frame_count, 0, 2), dtype=np.float32)
    else:
        tracks = np.ascontiguousarray(np.asarray(chunk.tracks_yx, dtype=np.float32))
    if tracks.ndim != 3 or tracks.shape[0] != frame_count or tracks.shape[-1] != 2:
        raise ValueError("tracks_yx must have shape T,N,2")
    if chunk.tracker_visibility is None:
        visibility = np.zeros(tracks.shape[:2], dtype=bool)
    else:
        visibility = np.ascontiguousarray(np.asarray(chunk.tracker_visibility, dtype=bool))
    if visibility.shape != tracks.shape[:2]:
        raise ValueError("tracker_visibility must have shape T,N")
    if chunk.queries_txy is None:
        queries = np.zeros((tracks.shape[1], 3), dtype=np.float32)
    else:
        queries = np.ascontiguousarray(np.asarray(chunk.queries_txy, dtype=np.float32))
    if queries.ndim != 2 or queries.shape[1] != 3 or queries.shape[0] != tracks.shape[1]:
        raise ValueError("queries_txy must have shape N,3 matching tracks")

    for name in ("tracking", "cotracker"):
        directory = case_dir / name
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(
            directory / "0.npz",
            tracks=tracks,
            visibility=visibility,
            queries_txy=queries,
        )


def _write_optional_pcd(case_dir: Path, chunk: FuturePhysTwinChunk, frame_count: int) -> None:
    if chunk.pcd_points is None and chunk.pcd_colors is None:
        return
    if chunk.pcd_points is None or chunk.pcd_colors is None:
        raise ValueError("pcd_points and pcd_colors must be provided together")
    points = np.asarray(chunk.pcd_points, dtype=np.float32)
    colors = np.asarray(chunk.pcd_colors)
    if points.shape != colors.shape or points.ndim != 5 or points.shape[0] != frame_count or points.shape[-1] != 3:
        raise ValueError("pcd_points and pcd_colors must have shape T,C,H,W,3")
    pcd_dir = case_dir / "pcd"
    pcd_dir.mkdir(parents=True, exist_ok=True)
    for frame_idx in range(frame_count):
        np.savez(
            pcd_dir / f"{frame_idx}.npz",
            points=np.ascontiguousarray(points[frame_idx]),
            colors=np.ascontiguousarray(colors[frame_idx]),
        )


def _track_process_payload(track_process_data: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    payload: dict[str, np.ndarray] = {}
    for key in FUTUREPHYSTWIN_TRACK_PROCESS_KEYS:
        if key not in track_process_data:
            raise ValueError(f"track_process_data missing required key: {key}")
        payload[key] = np.ascontiguousarray(np.asarray(track_process_data[key]))
    _validate_track_shapes(payload)
    return payload


def _sample_object_volume_indices(
    object_points: np.ndarray,
    *,
    surface_points: np.ndarray | None = None,
    interior_points: np.ndarray | None = None,
    volume_sample_size: float = 0.005,
) -> np.ndarray:
    pts = np.asarray(object_points, dtype=np.float64)
    if pts.ndim != 3 or pts.shape[-1] != 3:
        raise ValueError("object_points must have shape T,N,3")
    if pts.shape[1] == 0:
        return np.empty((0,), dtype=np.int64)
    voxel = float(volume_sample_size)
    if voxel <= 0.0:
        raise ValueError("volume_sample_size must be positive")
    bound_inputs = [pts[0]]
    for prior_points in (surface_points, interior_points):
        if prior_points is None:
            continue
        prior = np.asarray(prior_points, dtype=np.float64).reshape(-1, 3)
        if prior.size:
            bound_inputs.append(prior)
    min_bound = np.min(np.concatenate(bound_inputs, axis=0), axis=0)
    seen: set[tuple[int, int, int]] = set()
    keep: list[int] = []
    for idx, point in enumerate(pts[0]):
        grid = tuple(np.floor((point - min_bound) / voxel).astype(np.int64).tolist())
        if grid in seen:
            continue
        seen.add(grid)
        keep.append(int(idx))
    return np.asarray(keep, dtype=np.int64)


def _final_data_payload(
    track_process: Mapping[str, np.ndarray],
    *,
    surface_points: np.ndarray,
    interior_points: np.ndarray,
) -> dict[str, np.ndarray]:
    indices = _sample_object_volume_indices(
        track_process["object_points"],
        surface_points=surface_points,
        interior_points=interior_points,
        volume_sample_size=0.005,
    )
    final = {
        "controller_mask": np.ascontiguousarray(np.asarray(track_process["controller_mask"], dtype=bool)),
        "controller_points": np.ascontiguousarray(np.asarray(track_process["controller_points"], dtype=np.float64)),
        "object_points": np.ascontiguousarray(np.asarray(track_process["object_points"], dtype=np.float64)[:, indices, :]),
        "object_colors": np.ascontiguousarray(np.asarray(track_process["object_colors"], dtype=np.float64)[:, indices, :]),
        "object_visibilities": np.ascontiguousarray(np.asarray(track_process["object_visibilities"], dtype=bool)[:, indices]),
        "object_motions_valid": np.ascontiguousarray(np.asarray(track_process["object_motions_valid"], dtype=bool)[:, indices]),
        "surface_points": np.ascontiguousarray(np.asarray(surface_points, dtype=np.float64).reshape(-1, 3)),
        "interior_points": np.ascontiguousarray(np.asarray(interior_points, dtype=np.float64).reshape(-1, 3)),
    }
    _validate_final_shapes(final)
    return final


def _metadata_payload(chunk: FuturePhysTwinChunk, frame_count: int, width_height: tuple[int, int]) -> dict[str, Any]:
    intrinsics = np.asarray(chunk.intrinsics, dtype=np.float32)
    if intrinsics.shape == (3, 3):
        intrinsics = intrinsics.reshape(1, 3, 3)
    if intrinsics.shape != (1, 3, 3):
        raise ValueError(f"intrinsics must have shape 3,3 or 1,3,3 for single-camera Demo v4; got {intrinsics.shape}")
    width, height = width_height
    return {
        "fps": int(chunk.fps),
        "WH": [int(width), int(height)],
        "frame_num": int(frame_count),
        "start_step": 0,
        "end_step": int(frame_count),
        "intrinsics": intrinsics.astype(float).tolist(),
        "serial_numbers": [str(chunk.serial_number)],
        "camera_count": 1,
        "demo_version": "demo_v4",
        "depth_backend": str(chunk.depth_backend),
        "depth_source_internal": str(chunk.depth_source_internal),
    }


def _split_payload(frame_count: int) -> dict[str, Any]:
    train_end = max(1, int(int(frame_count) * 0.7))
    train_end = min(train_end, int(frame_count))
    return {
        "frame_len": int(frame_count),
        "train": [0, int(train_end)],
        "test": [int(train_end), int(frame_count)],
    }


def write_futurephystwin_chunk_case(
    base_path: str | Path,
    case_name: str,
    chunk: FuturePhysTwinChunk,
    manifest_extras: Mapping[str, Any] | Callable[[], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    base = Path(base_path)
    base.mkdir(parents=True, exist_ok=True)
    case = base / str(case_name)
    if case.exists():
        raise FileExistsError(f"FuturePhysTwin chunk case already exists: {case}")
    staging_root = base / ".publishing"
    staging_root.mkdir(parents=True, exist_ok=True)
    staging = staging_root / f"{case.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    staging.mkdir(parents=True)
    frame_count = _ensure_frame_count(chunk)
    first_rgb = np.asarray(chunk.rgb_frames[0], dtype=np.uint8)
    height, width = first_rgb.shape[:2]

    try:
        _write_rgb_frames(staging, chunk.rgb_frames)
        _write_processed_masks(staging, chunk.processed_masks)
        _write_tracking(staging, chunk, frame_count)
        _write_optional_pcd(staging, chunk, frame_count)

        c2w = np.asarray(chunk.camera_to_world_c2w, dtype=np.float32)
        if c2w.shape != (4, 4):
            raise ValueError(f"camera_to_world_c2w must be 4x4, got {c2w.shape}")
        with (staging / "calibrate.pkl").open("wb") as handle:
            pickle.dump([np.ascontiguousarray(c2w, dtype=np.float32)], handle)

        metadata = _metadata_payload(chunk, frame_count, (width, height))
        (staging / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        split = _split_payload(frame_count)
        (staging / "split.json").write_text(json.dumps(split, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        track_process = _track_process_payload(chunk.track_process_data)
        with (staging / "track_process_data.pkl").open("wb") as handle:
            pickle.dump(track_process, handle)

        final_data = _final_data_payload(
            track_process,
            surface_points=chunk.surface_points,
            interior_points=chunk.interior_points,
        )
        with (staging / "final_data.pkl").open("wb") as handle:
            pickle.dump(final_data, handle)

        manifest = {
            "case_name": str(case_name),
            "frame_count": int(frame_count),
            "chunk_index": None if chunk.chunk_index is None else int(chunk.chunk_index),
            "camera_count": 1,
            "futurephystwin_case_root": str(case),
            "final_data_path": "final_data.pkl",
            "track_process_data_path": "track_process_data.pkl",
            "surface_point_count": int(final_data["surface_points"].shape[0]),
            "interior_point_count": int(final_data["interior_points"].shape[0]),
            "depth_backend": str(chunk.depth_backend),
            "depth_source_internal": str(chunk.depth_source_internal),
            "data_process_sam3d_metrics": dict(DATA_PROCESS_SAM3D_METRICS),
            "publish_contract": "ready_marker_atomic_rename",
        }
        if manifest_extras is not None:
            extras = manifest_extras() if callable(manifest_extras) else manifest_extras
            manifest.update(dict(extras))
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        validate_futurephystwin_case(staging)
        (staging / "READY").write_text("ready\n", encoding="utf-8")
        os.replace(staging, case)
        try:
            staging_root.rmdir()
        except OSError:
            pass
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        try:
            staging_root.rmdir()
        except OSError:
            pass
        raise


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _validate_track_shapes(payload: Mapping[str, np.ndarray]) -> None:
    object_points = _array("object_points", payload["object_points"], (3,))
    if object_points.ndim != 3:
        raise ValueError("object_points must have shape T,N,3")
    frame_count, object_count, _ = object_points.shape
    object_colors = _array("object_colors", payload["object_colors"], (3,))
    if object_colors.shape != object_points.shape:
        raise ValueError("object_colors must match object_points shape")
    for key in ("object_visibilities", "object_motions_valid"):
        arr = np.asarray(payload[key])
        if arr.shape != (frame_count, object_count):
            raise ValueError(f"{key} must have shape T,N matching object_points")
    controller_points = _array("controller_points", payload["controller_points"], (3,))
    if controller_points.ndim != 3 or controller_points.shape[0] != frame_count:
        raise ValueError("controller_points must have shape T,M,3 matching object frame count")
    controller_mask = np.asarray(payload["controller_mask"])
    if controller_mask.ndim != 1:
        raise ValueError("controller_mask must be a 1D mask over candidate controller points")


def _validate_final_shapes(payload: Mapping[str, np.ndarray]) -> None:
    for key in FUTUREPHYSTWIN_FINAL_DATA_KEYS:
        if key not in payload:
            raise ValueError(f"final_data.pkl missing required key: {key}")
    _validate_track_shapes(payload)
    object_points = np.asarray(payload["object_points"], dtype=np.float64)
    controller_points = np.asarray(payload["controller_points"], dtype=np.float64)
    if not np.isfinite(object_points).all() or not np.isfinite(controller_points).all():
        raise ValueError("object/controller points must be finite")
    if object_points.shape[1] and np.any(np.linalg.norm(object_points[0], axis=1) <= 1e-9):
        raise ValueError("first-frame object points contain zero-depth placeholders")
    if controller_points.shape[1] and np.any(np.linalg.norm(controller_points[0], axis=1) <= 1e-9):
        raise ValueError("first-frame controller points contain zero-depth placeholders")
    if object_points.shape[1] and controller_points.shape[1]:
        distances = np.linalg.norm(
            object_points[0, :, None, :] - controller_points[0, None, :, :],
            axis=-1,
        )
        if np.any(distances <= 1e-8):
            raise ValueError("first-frame controller points overlap object points")
    for key in ("surface_points", "interior_points"):
        arr = np.asarray(payload[key])
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"{key} must have shape N,3")


def validate_futurephystwin_case(case_dir: str | Path, *, require_ready: bool = False) -> dict[str, Any]:
    case = Path(case_dir)
    if require_ready and not (case / "READY").is_file():
        raise ValueError(f"missing READY marker for FuturePhysTwin case: {case / 'READY'}")
    final_path = case / "final_data.pkl"
    if not final_path.is_file():
        raise ValueError(f"missing final_data.pkl: {final_path}")
    final_data = _load_pickle(final_path)
    if not isinstance(final_data, Mapping):
        raise ValueError("final_data.pkl must contain a mapping")
    _validate_final_shapes(final_data)

    required_files = (
        "track_process_data.pkl",
        "calibrate.pkl",
        "metadata.json",
        "split.json",
        "color/0/0.png",
        "mask/processed_masks.pkl",
        "tracking/0.npz",
        "cotracker/0.npz",
    )
    for relative in required_files:
        path = case / relative
        if not path.is_file():
            raise ValueError(f"missing required FuturePhysTwin case file: {relative}")

    track_process = _load_pickle(case / "track_process_data.pkl")
    if not isinstance(track_process, Mapping):
        raise ValueError("track_process_data.pkl must contain a mapping")
    for key in FUTUREPHYSTWIN_TRACK_PROCESS_KEYS:
        if key not in track_process:
            raise ValueError(f"track_process_data.pkl missing required key: {key}")
    _validate_track_shapes(track_process)

    c2ws = _load_pickle(case / "calibrate.pkl")
    if len(c2ws) != 1 or np.asarray(c2ws[0]).shape != (4, 4):
        raise ValueError("calibrate.pkl must contain one 4x4 camera-to-world matrix")

    metadata = json.loads((case / "metadata.json").read_text(encoding="utf-8"))
    intrinsics = np.asarray(metadata.get("intrinsics"), dtype=np.float32)
    if intrinsics.shape != (1, 3, 3):
        raise ValueError("metadata.json intrinsics must have shape 1,3,3")
    frame_count = int(np.asarray(final_data["object_points"]).shape[0])
    if int(metadata.get("frame_num", -1)) != frame_count:
        raise ValueError("metadata.json frame_num must match final_data frame count")

    split = json.loads((case / "split.json").read_text(encoding="utf-8"))
    if int(split.get("frame_len", -1)) != frame_count:
        raise ValueError("split.json frame_len must match final_data frame count")

    return {
        "valid": True,
        "case_dir": str(case),
        "frame_count": frame_count,
        "object_point_count": int(np.asarray(final_data["object_points"]).shape[1]),
        "controller_point_count": int(np.asarray(final_data["controller_points"]).shape[1]),
        "surface_point_count": int(np.asarray(final_data["surface_points"]).shape[0]),
        "interior_point_count": int(np.asarray(final_data["interior_points"]).shape[0]),
    }


__all__ = [
    "FUTUREPHYSTWIN_FINAL_DATA_KEYS",
    "FUTUREPHYSTWIN_TRACK_PROCESS_KEYS",
    "DATA_PROCESS_SAM3D_METRICS",
    "FuturePhysTwinChunk",
    "validate_futurephystwin_case",
    "write_futurephystwin_chunk_case",
]
