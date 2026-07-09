"""Capture metadata, trajectories, shape prior, and track manifest telemetry."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Callable, Mapping

import numpy as np

from demo_v6_2 import asap


# ---------------------------------------------------------------------------
# Capture metadata, trajectories, and shape prior
# ---------------------------------------------------------------------------
def _intrinsics_matrix(metadata: Mapping[str, Any]) -> np.ndarray:
    """Accept the fx/fy/cx/cy mapping or matrix metadata forms as float32 3x3."""
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
    """Return the camera to world."""
    value = metadata.get("camera_to_world_c2w")
    if value is None:
        return np.eye(4, dtype=np.float32)
    return np.ascontiguousarray(
        np.asarray(value, dtype=np.float32).reshape(4, 4), dtype=np.float32
    )


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
    # Offline parity with data_process_sam3d/data_process_sample.py:L183-L241.
    # That offline stage samples SAM3D surface/interior prior points. The
    # realtime path receives those points from the async shape-prior worker
    # instead of sampling here.
    if surface_points is not None or interior_points is not None:
        return (
            np.empty((0, 3), dtype=np.float64)
            if surface_points is None
            else np.ascontiguousarray(
                np.asarray(surface_points, dtype=np.float64).reshape(-1, 3)
            ),
            np.empty((0, 3), dtype=np.float64)
            if interior_points is None
            else np.ascontiguousarray(
                np.asarray(interior_points, dtype=np.float64).reshape(-1, 3)
            ),
        )
    shape_path = metadata.get("shape_prior_path")
    if shape_path:
        payload = np.load(capture_dir / str(shape_path), allow_pickle=False)
        if "surface_points_m" in payload.files or "interior_points_m" in payload.files:
            return (
                np.empty((0, 3), dtype=np.float64)
                if "surface_points_m" not in payload.files
                else np.ascontiguousarray(
                    np.asarray(payload["surface_points_m"], dtype=np.float64).reshape(
                        -1, 3
                    )
                ),
                np.empty((0, 3), dtype=np.float64)
                if "interior_points_m" not in payload.files
                else np.ascontiguousarray(
                    np.asarray(payload["interior_points_m"], dtype=np.float64).reshape(
                        -1, 3
                    )
                ),
            )
        points = np.ascontiguousarray(
            np.asarray(payload["points_m"], dtype=np.float64).reshape(-1, 3)
        )
        return points, np.empty((0, 3), dtype=np.float64)
    return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)


def _read_json_file_stable(
    path: Path,
    *,
    deadline_s: float,
    poll_interval_s: float,
) -> Mapping[str, Any]:
    """Read JSON file stable."""
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
                raise RuntimeError(
                    f"timed out waiting for stable JSON metadata at {path}"
                ) from last_error
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

    Demo v6.1 keeps capture realtime, but ``final_data.pkl`` must contain
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
        if (
            explicit_points
            or not bool(require_shape_prior)
            # "Ready" means at least one surface or interior structure point.
            or int(np.asarray(shape_surface).reshape(-1, 3).shape[0])
            + int(np.asarray(shape_interior).reshape(-1, 3).shape[0])
            > 0
        ):
            return metadata, shape_surface, shape_interior
        # A terminal shape-prior status can never produce points later; fail
        # fast instead of waiting out the full deadline.
        status = str(metadata.get("shape_prior_status") or "").strip().lower()
        if status in {"failed", "unavailable", "disabled"}:
            detail = (
                metadata.get("shape_prior_error")
                or metadata.get("error")
                or "no surface/interior points became ready"
            )
            raise RuntimeError(f"shape prior {status}: {detail}")
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "shape prior is required for Demo v6.1 final_data chunks, "
                "but no surface/interior points became ready"
            )
        if before_poll is not None:
            before_poll()
        if (
            capture_finished is not None
            and capture_finished()
            and time.monotonic() >= deadline
        ):
            raise RuntimeError(
                "capture finished before required shape prior became ready"
            )
        time.sleep(max(0.0, float(poll_interval_s)))


def _wait_for_asap_case_dir(
    capture: Path,
    metadata: Mapping[str, Any],
    *,
    shape_prior_wait_timeout_s: float,
    before_poll: Callable[[], None] | None,
    poll_interval_s: float,
) -> Mapping[str, Any]:
    """Wait for warmup to record shape_prior_case_dir (ASAP needs the mesh).

    Explicit surface/interior overrides skip the shape-point wait, so the
    first window can materialize while warmup is still writing its result;
    that is "not ready yet", not "mesh missing", and must not fail fast. A
    terminal shape-prior status or the timeout still fails fast.
    """
    if metadata.get("shape_prior_case_dir"):
        return metadata
    deadline = time.monotonic() + max(0.0, float(shape_prior_wait_timeout_s))
    while True:
        latest = _read_json_file_stable(
            capture / "metadata.json",
            deadline_s=deadline,
            poll_interval_s=float(poll_interval_s),
        )
        if latest.get("shape_prior_case_dir"):
            return latest
        status = str(latest.get("shape_prior_status") or "").strip().lower()
        if status in {"failed", "unavailable", "disabled"}:
            detail = (
                latest.get("shape_prior_error")
                or latest.get("error")
                or "shape-prior warmup recorded no case directory"
            )
            raise asap.AsapMeshError(
                f"ASAP augmentation requires final_mesh.glb but shape prior is "
                f"{status}: {detail}"
            )
        if time.monotonic() >= deadline:
            raise asap.AsapMeshError(
                "ASAP augmentation is enabled but shape-prior warmup did not "
                "record shape_prior_case_dir before the timeout"
            )
        if before_poll is not None:
            before_poll()
        time.sleep(max(0.0, float(poll_interval_s)))


# ---------------------------------------------------------------------------
# Track manifest telemetry
# ---------------------------------------------------------------------------
def _controller_track_manifest_fields(
    track_process_data: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the controller track manifest fields."""
    track_process_data = dict(track_process_data)
    if "controller_track_query_indices" not in track_process_data:
        controller_points = np.asarray(
            track_process_data.get("controller_points", np.empty((0, 0, 3)))
        )
        return {
            "controller_track_selection_mode": "independent_fps",
            "controller_track_count": int(
                controller_points.shape[1] if controller_points.ndim >= 2 else 0
            ),
        }
    query_indices = np.asarray(
        track_process_data["controller_track_query_indices"], dtype=np.int64
    ).reshape(-1)
    active_indices = np.asarray(
        track_process_data.get("controller_track_active_query_indices", query_indices),
        dtype=np.int64,
    ).reshape(-1)
    statuses = np.asarray(
        track_process_data.get("controller_track_status", []), dtype=str
    ).reshape(-1)
    payload = {
        "controller_track_selection_mode": "streaming_stable",
        "controller_track_count": int(len(query_indices)),
        "controller_track_query_indices": [
            int(value) for value in query_indices.tolist()
        ],
        "controller_track_active_query_indices": [
            int(value) for value in active_indices.tolist()
        ],
        "controller_track_direct_count": int(np.count_nonzero(statuses == "direct")),
        "controller_track_proxied_count": int(np.count_nonzero(statuses == "proxied")),
        "controller_track_status": [str(value) for value in statuses.tolist()],
    }
    if "controller_proxied" in track_process_data:
        proxied = np.asarray(track_process_data["controller_proxied"], dtype=bool)
        payload["controller_track_direct_frame_count"] = int(
            proxied.size - np.count_nonzero(proxied)
        )
        payload["controller_track_proxied_frame_count"] = int(
            np.count_nonzero(proxied)
        )
        payload["controller_track_proxied_ratio"] = (
            float(np.count_nonzero(proxied) / proxied.size) if proxied.size else 0.0
        )
    if "track_process_status" in track_process_data:
        payload["track_process_status"] = str(
            np.asarray(track_process_data["track_process_status"]).item()
        )
    return payload


def _object_track_manifest_fields(
    track_process_data: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the object track manifest fields."""
    track_process_data = dict(track_process_data)
    if "object_track_query_indices" not in track_process_data:
        object_points = np.asarray(
            track_process_data.get("object_points", np.empty((0, 0, 3)))
        )
        return {
            "object_track_selection_mode": "per_chunk_volume_sample",
            "object_track_count": int(
                object_points.shape[1] if object_points.ndim >= 2 else 0
            ),
        }
    query_indices = np.asarray(
        track_process_data["object_track_query_indices"], dtype=np.int64
    ).reshape(-1)
    active_indices = np.asarray(
        track_process_data.get("object_track_active_query_indices", query_indices),
        dtype=np.int64,
    ).reshape(-1)
    statuses = np.asarray(
        track_process_data.get("object_track_status", []), dtype=str
    ).reshape(-1)
    return {
        "object_track_selection_mode": "streaming_stable",
        "object_track_count": int(len(query_indices)),
        "object_track_query_indices": [int(value) for value in query_indices.tolist()],
        "object_track_active_query_indices": [
            int(value) for value in active_indices.tolist()
        ],
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
