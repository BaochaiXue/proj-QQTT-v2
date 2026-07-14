"""Build Demo v6.1 realtime final_data chunk payloads.

Demo v6.1 publishes online chunks directly. It no longer materializes each
window as a data_process-style case directory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any, Mapping, Sequence

import numpy as np

from demo_v6_2.tracking import NEIGHBOR_TABLE_SIZE, RECOVERY_NEIGHBOR_COUNT

DATA_PROCESS_QUERY_SCHEMA_VERSION = "data_process_sam3d_realtime_query_schema_v1"
DATA_PROCESS_SAM3D_REALTIME_CONTRACT_VERSION = (
    "data_process_sam3d_realtime_final_data_v1"
)
# Query schema fields are static over an online run. The hash is intentionally
# part of both final_data and track_process_data so readers can identify the
# query topology without scanning large arrays.
DATA_PROCESS_QUERY_SCHEMA_KEYS = (
    "query_schema_version",
    "query_schema_hash",
    "query_ids",
    "query_semantic_labels",
    "object_sample_query_ids",
    "controller_sample_query_ids",
)

DATA_PROCESS_TRACK_PROCESS_KEYS = (
    "controller_mask",
    "controller_points",
    "object_colors",
    "object_motions_valid",
    "object_points",
    "object_visibilities",
)

DATA_PROCESS_TRACK_PROCESS_TRACE_KEYS = (
    "query_ids",
    "query_semantic_labels",
    "controller_final_indices",
    "controller_query_indices",
    "controller_candidate_query_ids",
    "controller_candidate_mask",
    "controller_sample_query_ids",
    "controller_track_query_indices",
    "controller_track_active_query_indices",
    "controller_track_status",
    "object_query_indices",
    "object_candidate_query_ids",
    "object_volume_sample_indices",
    "object_sample_indices",
    "object_selected_query_ids",
    "object_sample_query_ids",
    "object_track_query_indices",
    "object_track_active_query_indices",
    "object_track_status",
)

# Candidate keys describe every controller query before final 30-point
# selection. Track keys describe the stable selected controller points after
# strict filtering and recovery.
CONTROLLER_CANDIDATE_TIME_KEYS = (
    "controller_raw_visible",
    "controller_processed_mask_valid",
    "controller_depth_valid",
    "controller_measurement_valid",
    "controller_motions_valid",
)
CONTROLLER_CANDIDATE_POINT_KEYS = ("controller_raw_points",)
CONTROLLER_TRACK_TIME_KEYS = ("controller_proxied",)
CONTROLLER_TRACK_STATIC_KEYS = ("controller_neighbor_query_ids",)

DATA_PROCESS_SAM3D_METRICS = {
    "runtime_contract": DATA_PROCESS_SAM3D_REALTIME_CONTRACT_VERSION,
    "mask_radius_outlier_filter_source": (
        "data_process_sam3d/data_process_mask.py::process_pcd_mask"
    ),
    "mask_radius_outlier_radius_m": 0.01,
    "mask_radius_outlier_nb_points": 40,
    "semantic_filter": "first_frame_object_controller_labels_then_per_frame_mask_visibility",
    "motion_filter_source": "data_process_sam3d/data_process_track.py::filter_motion",
    "motion_neighbor_dist_m": 0.01,
    "motion_min_neighbors": 5,
    "motion_similarity_m": 0.005,
    "controller_visibility_policy": (
        "chunk0_whole_window_valid_then_per_frame_temporary_invalid"
    ),
    "controller_final_count": 30,
    "controller_recovery_source": "design_spec.md::local_rigid_registration",
    "controller_recovery_neighbor_table_size": NEIGHBOR_TABLE_SIZE,
    "controller_recovery_neighbor_count": RECOVERY_NEIGHBOR_COUNT,
    "object_sampling_source": "data_process_sam3d/data_process_sample.py::process_unique_points",
    "object_volume_sample_size_m": 0.005,
    "shape_prior_sampling_source": "data_process_sam3d/data_process_sample.py",
    "shape_prior_target_surface_points": 1024,
    "shape_prior_interior_candidate_points": 10000,
    "shape_prior_volume_sample_size_m": 0.005,
    "shape_prior_configured_max_dist_m": 0.05,
    "shape_prior_effective_max_dist_m": 0.05,
    "shape_prior_distance_policy": "canonical_single_view_configured",
    "offline_single_view_parity": True,
    "shape_prior_uses_mvsam3d": False,
    "shape_prior_ground_policy": "preserve",
}


@dataclass(frozen=True)
class ChunkDataWindow:
    """In-memory representation of one fixed-size realtime final_data window."""

    track_process_data: Mapping[str, np.ndarray]
    surface_points: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float64)
    )
    interior_points: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float64)
    )
    fps: int = 5
    serial_number: str = "demo-v6-1-single-camera"
    depth_backend: str = ""
    depth_source_internal: str = ""
    chunk_index: int | None = None
    source_frame_indices: Sequence[int] | None = None


# ---- Section: query identity schema ----------------------------------------


def _query_schema_hash(payload: Mapping[str, Any]) -> str:
    """Hash query identity fields that must stay stable across online chunks."""
    digest = hashlib.sha256()
    digest.update(
        _scalar_str(
            payload.get("query_schema_version", DATA_PROCESS_QUERY_SCHEMA_VERSION)
        ).encode("utf-8")
    )
    for key in (
        "query_ids",
        "query_semantic_labels",
        "object_sample_query_ids",
        "controller_sample_query_ids",
    ):
        arr = np.ascontiguousarray(np.asarray(payload[key]))
        digest.update(str(arr.dtype).encode("utf-8"))
        digest.update(str(arr.shape).encode("utf-8"))
        digest.update(arr.tobytes())
    return digest.hexdigest()


def _scalar_str(value: Any) -> str:
    """Render a scalar for JSON manifests, unwrapping 0-d/1-element arrays."""
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return str(value.item())
        if value.size == 1:
            return str(value.reshape(-1)[0].item())
    return str(value)


def build_query_schema_payload(
    track_process_data: Mapping[str, Any],
    *,
    object_sample_query_ids: np.ndarray,
    controller_sample_query_ids: np.ndarray,
) -> dict[str, Any]:
    """Build query-id metadata shared by final_data and track_process_data."""
    query_ids = np.ascontiguousarray(
        np.asarray(track_process_data["query_ids"], dtype=np.int64).reshape(-1)
    )
    query_semantic_labels = np.ascontiguousarray(
        np.asarray(
            track_process_data["query_semantic_labels"], dtype=np.int8
        ).reshape(-1)
    )
    object_sample_query_ids = np.ascontiguousarray(
        np.asarray(object_sample_query_ids, dtype=np.int64).reshape(-1)
    )
    controller_sample_query_ids = np.ascontiguousarray(
        np.asarray(controller_sample_query_ids, dtype=np.int64).reshape(-1)
    )
    payload: dict[str, Any] = {
        "query_schema_version": DATA_PROCESS_QUERY_SCHEMA_VERSION,
        "query_ids": query_ids,
        "query_semantic_labels": query_semantic_labels,
        "object_sample_query_ids": object_sample_query_ids,
        "controller_sample_query_ids": controller_sample_query_ids,
    }
    payload["query_schema_hash"] = _query_schema_hash(payload)
    return payload


# ---- Section: track_process_data payload -----------------------------------


def _track_process_payload(
    track_process_data: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Normalize strict tracking output into data_process_sam3d track payload."""
    # Offline parity with data_process_sam3d/data_process_track.py:L127-L135
    # and L321-L322. That path assembles object/controller point, color, and
    # visibility arrays, then adds the controller candidate mask after motion
    # filtering.
    track_process_data = dict(track_process_data)
    payload: dict[str, Any] = {}
    for key in DATA_PROCESS_TRACK_PROCESS_KEYS:
        payload[key] = np.ascontiguousarray(np.asarray(track_process_data[key]))
    for key in DATA_PROCESS_TRACK_PROCESS_TRACE_KEYS:
        if key not in track_process_data:
            continue
        arr = np.asarray(track_process_data[key])
        if key.endswith("_status"):
            payload[key] = np.asarray(arr, dtype="<U16").reshape(-1)
        elif key.endswith("_mask"):
            payload[key] = np.ascontiguousarray(arr.astype(bool).reshape(-1))
        elif key == "query_semantic_labels":
            payload[key] = np.ascontiguousarray(arr.astype(np.int8).reshape(-1))
        else:
            payload[key] = np.ascontiguousarray(arr.astype(np.int64).reshape(-1))
    for key in CONTROLLER_CANDIDATE_TIME_KEYS:
        if key == "controller_motions_valid":
            continue
        if key in track_process_data:
            payload[key] = np.ascontiguousarray(
                np.asarray(track_process_data[key], dtype=bool)
            )
    controller_motion_source = track_process_data.get(
        "controller_candidate_motions_valid",
        track_process_data.get("controller_motions_valid"),
    )
    if controller_motion_source is not None:
        payload["controller_motions_valid"] = np.ascontiguousarray(
            np.asarray(controller_motion_source, dtype=bool)
        )
    for key in CONTROLLER_CANDIDATE_POINT_KEYS:
        if key in track_process_data:
            payload[key] = np.ascontiguousarray(
                np.asarray(track_process_data[key], dtype=np.float32)
            )
    for key in CONTROLLER_TRACK_TIME_KEYS:
        if key not in track_process_data:
            continue
        payload[key] = np.ascontiguousarray(
            np.asarray(track_process_data[key], dtype=bool)
        )
    for key in CONTROLLER_TRACK_STATIC_KEYS:
        if key in track_process_data:
            payload[key] = np.ascontiguousarray(
                np.asarray(track_process_data[key], dtype=np.int64)
            )
    if "track_process_status" in track_process_data:
        payload["track_process_status"] = str(
            np.asarray(track_process_data["track_process_status"]).item()
        )
    return payload


# ---- Section: final_data payload --------------------------------------------


def _final_data_payload(
    track_process: Mapping[str, np.ndarray],
    *,
    surface_points: np.ndarray,
    interior_points: np.ndarray,
) -> dict[str, np.ndarray]:
    """Assemble the realtime final_data.pkl contract."""
    # Offline parity with data_process_sam3d/data_process_sample.py:L250-L352.
    # That path reduces track_process_data into final_data.pkl, carrying object
    # samples, controller handles, and optional surface/interior shape-prior
    # points.
    track_process = dict(track_process)
    object_points = np.asarray(track_process["object_points"])
    # Strict tracking already selected object tracks; preserve that selection.
    indices = np.arange(object_points.shape[1], dtype=np.int64)
    object_sample_indices = indices
    object_selected_query_ids = np.asarray(
        track_process["object_track_query_indices"], dtype=np.int64
    ).reshape(-1)

    controller_points = np.asarray(track_process["controller_points"])
    controller_count = int(controller_points.shape[1])
    # Controller points are always the selected control anchors. The *_query_ids
    # arrays record which original query each anchor came from.
    controller_final_indices = np.arange(controller_count, dtype=np.int64)
    controller_selected_query_ids = np.asarray(
        track_process["controller_track_query_indices"],
        dtype=np.int64,
    ).reshape(-1)
    final = {
        "controller_points": np.ascontiguousarray(controller_points.astype(np.float64)),
        "controller_final_indices": np.ascontiguousarray(
            controller_final_indices.astype(np.int64)
        ),
        "controller_selected_query_ids": np.ascontiguousarray(
            controller_selected_query_ids.astype(np.int64)
        ),
        "controller_sample_query_ids": np.ascontiguousarray(
            controller_selected_query_ids.astype(np.int64)
        ),
        "object_points": np.ascontiguousarray(
            np.asarray(track_process["object_points"], dtype=np.float64)[:, indices, :]
        ),
        "object_colors": np.ascontiguousarray(
            np.asarray(track_process["object_colors"], dtype=np.float64)[:, indices, :]
        ),
        "object_visibilities": np.ascontiguousarray(
            np.asarray(track_process["object_visibilities"], dtype=bool)[:, indices]
        ),
        "object_motions_valid": np.ascontiguousarray(
            np.asarray(track_process["object_motions_valid"], dtype=bool)[:, indices]
        ),
        "object_sample_indices": np.ascontiguousarray(
            object_sample_indices.astype(np.int64)
        ),
        "object_selected_query_ids": np.ascontiguousarray(
            object_selected_query_ids.astype(np.int64)
        ),
        "object_sample_query_ids": np.ascontiguousarray(
            object_selected_query_ids.astype(np.int64)
        ),
        "surface_points": np.ascontiguousarray(
            np.asarray(surface_points, dtype=np.float64).reshape(-1, 3)
        ),
        "interior_points": np.ascontiguousarray(
            np.asarray(interior_points, dtype=np.float64).reshape(-1, 3)
        ),
    }
    final.update(
        build_query_schema_payload(
            track_process,
            object_sample_query_ids=final["object_sample_query_ids"],
            controller_sample_query_ids=final["controller_sample_query_ids"],
        )
    )
    return final


# ---- Section: quality manifest metrics --------------------------------------


def _zero_point_count(points: np.ndarray) -> int:
    """Count first-frame points sitting at the origin (norm <= 1e-9 m).

    Origin-pinned points are a proxy for depth dropouts that were never
    backfilled, so the manifest surfaces them as a quality counter.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim < 2 or pts.shape[-1] != 3 or pts.shape[0] == 0:
        return 0
    return int(np.count_nonzero(np.linalg.norm(pts[0], axis=-1) <= 1e-9))


def _quality_manifest_fields(
    final_data: Mapping[str, np.ndarray],
    track_process: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Compute per-chunk quality counters published in the online manifest."""
    object_points = np.asarray(final_data["object_points"], dtype=np.float64)
    controller_points = np.asarray(final_data["controller_points"], dtype=np.float64)
    surface_points = np.asarray(final_data["surface_points"], dtype=np.float64).reshape(
        -1, 3
    )
    interior_points = np.asarray(
        final_data["interior_points"], dtype=np.float64
    ).reshape(-1, 3)
    controller_mask = np.asarray(
        track_process.get(
            "controller_mask", np.ones((controller_points.shape[1],), dtype=bool)
        ),
        dtype=bool,
    )
    shape_prior_fields_present = bool(
        surface_points.shape[0] > 0 and interior_points.shape[0] > 0
    )
    payload = {
        "object_point_count": int(object_points.shape[1]),
        "controller_point_count": int(controller_points.shape[1]),
        "controller_candidate_count": int(controller_mask.shape[0]),
        "controller_valid_candidate_count": int(np.count_nonzero(controller_mask)),
        "surface_point_count": int(surface_points.shape[0]),
        "interior_point_count": int(interior_points.shape[0]),
        "shape_prior_fields_present": shape_prior_fields_present,
        "shape_prior_complete": shape_prior_fields_present,
        "object_points_finite": bool(np.isfinite(object_points).all()),
        "controller_points_finite": bool(np.isfinite(controller_points).all()),
        "shape_prior_points_finite": bool(
            np.isfinite(surface_points).all() and np.isfinite(interior_points).all()
        ),
        "first_frame_zero_object_points": _zero_point_count(object_points),
        "first_frame_zero_controller_points": _zero_point_count(controller_points),
    }
    return payload


# ---- Section: chunk assembly -------------------------------------------------


def build_window_publish_payloads(
    chunk: ChunkDataWindow,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build final_data, track_process diagnostics, and manifest fields."""
    # Frame count is the leading axis of the (T, N, 3) object track tensor.
    frame_count = int(np.asarray(chunk.track_process_data["object_points"]).shape[0])
    track_process = _track_process_payload(chunk.track_process_data)
    final_data = _final_data_payload(
        track_process,
        surface_points=chunk.surface_points,
        interior_points=chunk.interior_points,
    )
    # design_spec_v6_1.md: deformed shape-prior trajectories ride as dedicated
    # final_data per-frame keys; they never widen the object arrays and are
    # not duplicated into the track_process diagnostics.
    for asap_key in ("asap_surface_points", "asap_interior_points"):
        if asap_key in chunk.track_process_data:
            final_data[asap_key] = np.ascontiguousarray(
                np.asarray(chunk.track_process_data[asap_key], dtype=np.float64)
            )
    # Keep the same query schema hash in both payloads so online chunks can be
    # concatenated without changing query semantic identity.
    for key in DATA_PROCESS_QUERY_SCHEMA_KEYS:
        track_process[key] = final_data[key]
    manifest = {
        "frame_count": int(frame_count),
        "chunk_index": None if chunk.chunk_index is None else int(chunk.chunk_index),
        "camera_count": 1,
        "depth_backend": str(chunk.depth_backend),
        "depth_source_internal": str(chunk.depth_source_internal),
        "runtime_contract": DATA_PROCESS_SAM3D_REALTIME_CONTRACT_VERSION,
        "chunk_continuity_contract": "stable_query_schema_hash_and_contiguous_frames",
        "data_process_sam3d_metrics": dict(DATA_PROCESS_SAM3D_METRICS),
        "publish_contract": "online_final_data_chunk",
        "query_schema_version": _scalar_str(final_data["query_schema_version"]),
        "query_schema_hash": _scalar_str(final_data["query_schema_hash"]),
    }
    manifest.update(_quality_manifest_fields(final_data, track_process))
    return final_data, track_process, manifest


__all__ = [
    "DATA_PROCESS_SAM3D_REALTIME_CONTRACT_VERSION",
    "DATA_PROCESS_QUERY_SCHEMA_KEYS",
    "ChunkDataWindow",
    "build_window_publish_payloads",
    "build_query_schema_payload",
]
