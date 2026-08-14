"""Capture metadata, trajectories, and shape-prior waits for chunk streaming."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

from demo_v7.runtime.streaming import asap
from demo_v7.runtime.utils.projection import intrinsics_to_matrix

if TYPE_CHECKING:
    from demo_v7.runtime.streaming.session import ChunkStreamSession


# ---------------------------------------------------------------------------
# Capture metadata, trajectories, and shape prior
# ---------------------------------------------------------------------------
def _intrinsics_matrix(metadata: Mapping[str, Any]) -> np.ndarray:
    """Accept the fx/fy/cx/cy mapping or matrix metadata forms as float32 3x3."""
    return np.ascontiguousarray(intrinsics_to_matrix(metadata.get("intrinsics")))


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


def _wait_for_shape_candidates(
    session: ChunkStreamSession,
) -> tuple[Mapping[str, Any], np.ndarray, np.ndarray]:
    """Wait until the RAW shape-prior candidate pools are available.

    The camera's warm-up publishes raw surface/interior CANDIDATES (origin
    1024/10000 pools); the final origin-parity selection happens once, at
    chunk-0 identity freeze, with the final tracked object claiming the
    shared voxel occupancy first. Waiting happens here, after a source window
    has closed, not inside the camera/tracker loop. Explicit NPY overrides
    are also treated as candidates and go through the unified sampling.
    """
    explicit_points = (
        session.surface_points is not None or session.interior_points is not None
    )
    deadline = time.monotonic() + max(0.0, session.shape_prior_wait_timeout_s)
    capture_already_finished = False
    while True:
        metadata = _read_json_file_stable(
            session.capture / "metadata.json",
            deadline_s=deadline,
            poll_interval_s=session.poll_interval_s,
        )
        shape_surface, shape_interior = _shape_points_from_capture(
            session.capture,
            metadata,
            surface_points=session.surface_points,
            interior_points=session.interior_points,
        )
        if (
            explicit_points
            or not session.require_shape_prior
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
                "shape prior is required for Demo v6.2 final_data chunks, "
                "but no surface/interior points became ready"
            )
        if session.before_poll is not None:
            session.before_poll()
        # A finished capture can never produce candidates later; allow one
        # extra poll so the camera's final metadata flush is not raced, then
        # fail fast instead of waiting out the full timeout.
        if capture_already_finished:
            raise RuntimeError(
                "capture finished before required shape prior became ready"
            )
        capture_already_finished = session.capture_finished()
        time.sleep(max(0.0, session.poll_interval_s))


def _wait_for_asap_case_dir(
    session: ChunkStreamSession,
    metadata: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Wait for warmup to record shape_prior_case_dir (ASAP needs the mesh).

    Explicit surface/interior overrides skip the shape-point wait, so the
    first window can materialize while warmup is still writing its result;
    that is "not ready yet", not "mesh missing", and must not fail fast. A
    terminal shape-prior status or the timeout still fails fast. The
    ``ChunkStreamSession`` supplies the capture location and wait/poll policy.
    """
    if metadata.get("shape_prior_case_dir"):
        return metadata
    deadline = time.monotonic() + max(0.0, session.shape_prior_wait_timeout_s)
    while True:
        latest = _read_json_file_stable(
            session.capture / "metadata.json",
            deadline_s=deadline,
            poll_interval_s=session.poll_interval_s,
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
        if session.before_poll is not None:
            session.before_poll()
        time.sleep(max(0.0, session.poll_interval_s))
