"""Build and validate Demo v5.1 realtime final_data chunk payloads.

Demo v5.1 publishes online chunks directly. It no longer materializes each
window as a data_process-style case directory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any, Mapping, Sequence

import numpy as np

DATA_PROCESS_QUERY_SCHEMA_VERSION = "data_process_sam3d_realtime_query_schema_v1"
DATA_PROCESS_SAM3D_REALTIME_CONTRACT_VERSION = (
    "data_process_sam3d_realtime_final_data_v1"
)
TRACK_PROCESS_STATUSES = ("normal", "degraded", "invalid")
# Query schema fields are static over an online run. The hash is intentionally
# part of both final_data and track_process_data so readers can reject mixed
# chunks before comparing large arrays.
DATA_PROCESS_QUERY_SCHEMA_KEYS = (
    "query_schema_version",
    "query_schema_hash",
    "query_ids",
    "query_semantic_labels",
    "object_sample_query_ids",
    "controller_sample_query_ids",
)

DATA_PROCESS_FINAL_DATA_KEYS = (
    "controller_points",
    "controller_final_indices",
    "controller_selected_query_ids",
    "controller_sample_query_ids",
    "object_colors",
    "object_motions_valid",
    "object_points",
    "object_sample_indices",
    "object_selected_query_ids",
    "object_sample_query_ids",
    "object_visibilities",
    "query_schema_version",
    "query_schema_hash",
    "query_ids",
    "query_semantic_labels",
    "surface_points",
    "interior_points",
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
CONTROLLER_TRACK_TIME_KEYS = (
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
    "controller_visibility_policy": "visible_for_whole_chunk_then_motion_consistent",
    "controller_final_count": 30,
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
class DataProcessChunk:
    """In-memory representation of one fixed-size realtime final_data window."""

    track_process_data: Mapping[str, np.ndarray]
    surface_points: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float64)
    )
    interior_points: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float64)
    )
    fps: int = 5
    serial_number: str = "demo-v5-single-camera"
    depth_backend: str = ""
    depth_source_internal: str = ""
    chunk_index: int | None = None
    source_frame_indices: Sequence[int] | None = None


def _array(
    name: str, value: Any, shape_tail: tuple[int, ...] | None = None
) -> np.ndarray:
    arr = np.asarray(value)
    if shape_tail is not None and tuple(arr.shape[-len(shape_tail) :]) != tuple(
        shape_tail
    ):
        raise ValueError(f"{name} must end with shape {shape_tail}, got {arr.shape}")
    return np.ascontiguousarray(arr)


def _ensure_frame_count(chunk: DataProcessChunk) -> int:
    track_points = np.asarray(chunk.track_process_data["object_points"])
    if track_points.ndim != 3 or track_points.shape[-1] != 3:
        raise ValueError("track_process_data['object_points'] must have shape T,N,3")
    frame_count = int(track_points.shape[0])
    if frame_count <= 0:
        raise ValueError("realtime final_data chunk requires at least one frame")
    return int(frame_count)


def _int_vector(value: Any, *, default: np.ndarray | None = None) -> np.ndarray:
    if value is None:
        if default is None:
            return np.empty((0,), dtype=np.int64)
        return np.ascontiguousarray(np.asarray(default, dtype=np.int64).reshape(-1))
    return np.ascontiguousarray(np.asarray(value, dtype=np.int64).reshape(-1))


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
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return str(value.item())
        if value.size == 1:
            return str(value.reshape(-1)[0].item())
    return str(value)


def build_query_schema_payload(
    track_process_data: Mapping[str, Any],
    *,
    object_sample_query_ids: np.ndarray | None = None,
    controller_sample_query_ids: np.ndarray | None = None,
) -> dict[str, Any]:
    """Build query-id metadata shared by final_data and track_process_data."""
    track_process_data = dict(track_process_data)
    object_points = np.asarray(
        track_process_data.get("object_points", np.empty((0, 0, 3)))
    )
    controller_points = np.asarray(
        track_process_data.get("controller_points", np.empty((0, 0, 3)))
    )
    object_count = int(object_points.shape[1]) if object_points.ndim >= 2 else 0
    controller_count = (
        int(controller_points.shape[1]) if controller_points.ndim >= 2 else 0
    )

    # Prepared v5.1 frames usually carry explicit query identity. Synthetic or
    # reconstructed inputs can still derive ids from object/controller query
    # arrays before the canonical payload is written.
    if (
        "query_ids" in track_process_data
        and "query_semantic_labels" in track_process_data
    ):
        query_ids = np.ascontiguousarray(
            np.asarray(track_process_data["query_ids"], dtype=np.int64).reshape(-1)
        )
        query_semantic_labels = np.ascontiguousarray(
            np.asarray(
                track_process_data["query_semantic_labels"], dtype=np.int8
            ).reshape(-1)
        )
        if query_ids.shape != query_semantic_labels.shape:
            raise ValueError(
                "query_ids and query_semantic_labels must have matching shape"
            )
    else:
        object_query_ids = _int_vector(
            track_process_data.get(
                "object_candidate_query_ids",
                track_process_data.get("object_query_indices"),
            ),
            default=np.arange(object_count, dtype=np.int64),
        )
        controller_query_ids = _int_vector(
            track_process_data.get(
                "controller_candidate_query_ids",
                track_process_data.get("controller_query_indices"),
            ),
            default=np.arange(
                object_count, object_count + controller_count, dtype=np.int64
            ),
        )
        query_ids = np.ascontiguousarray(
            np.concatenate([object_query_ids, controller_query_ids]), dtype=np.int64
        )
        query_semantic_labels = np.ascontiguousarray(
            np.concatenate(
                [
                    np.ones((object_query_ids.shape[0],), dtype=np.int8),
                    np.full((controller_query_ids.shape[0],), 2, dtype=np.int8),
                ]
            ),
            dtype=np.int8,
        )
    object_query_ids = _int_vector(
        track_process_data.get(
            "object_candidate_query_ids",
            track_process_data.get("object_query_indices"),
        ),
        default=np.arange(object_count, dtype=np.int64),
    )
    controller_query_ids = _int_vector(
        track_process_data.get(
            "controller_candidate_query_ids",
            track_process_data.get("controller_query_indices"),
        ),
        default=np.arange(
            object_count, object_count + controller_count, dtype=np.int64
        ),
    )

    if object_sample_query_ids is None:
        object_sample_query_ids = _int_vector(
            track_process_data.get(
                "object_track_query_indices",
                track_process_data.get("object_selected_query_ids", object_query_ids),
            )
        )
    if controller_sample_query_ids is None:
        controller_sample_query_ids = _int_vector(
            track_process_data.get(
                "controller_track_query_indices",
                track_process_data.get("controller_selected_query_ids"),
            )
        )
        if controller_sample_query_ids.size == 0:
            fps = _int_vector(
                track_process_data.get("controller_final_indices"),
                default=np.arange(controller_count, dtype=np.int64),
            )
            controller_sample_query_ids = np.full(fps.shape, -1, dtype=np.int64)
            valid = (fps >= 0) & (fps < controller_query_ids.shape[0])
            controller_sample_query_ids[valid] = controller_query_ids[fps[valid]]

    payload: dict[str, Any] = {
        "query_schema_version": DATA_PROCESS_QUERY_SCHEMA_VERSION,
        "query_ids": query_ids,
        "query_semantic_labels": query_semantic_labels,
        "object_sample_query_ids": np.ascontiguousarray(
            np.asarray(object_sample_query_ids, dtype=np.int64).reshape(-1)
        ),
        "controller_sample_query_ids": np.ascontiguousarray(
            np.asarray(controller_sample_query_ids, dtype=np.int64).reshape(-1)
        ),
    }
    payload["query_schema_hash"] = _query_schema_hash(payload)
    return payload


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
        if key not in track_process_data:
            raise ValueError(f"track_process_data missing required key: {key}")
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
        arr = np.asarray(track_process_data[key])
        if key in {"controller_track_mode", "controller_filter_reason"}:
            payload[key] = np.asarray(arr, dtype="<U40")
        elif key == "controller_source_query_ids" or key.endswith("_count"):
            payload[key] = np.ascontiguousarray(arr.astype(np.int64))
        else:
            payload[key] = np.ascontiguousarray(arr.astype(np.float32))
    for key in CONTROLLER_TRACK_STATIC_KEYS:
        if key in track_process_data:
            payload[key] = np.ascontiguousarray(
                np.asarray(track_process_data[key], dtype=np.int64)
            )
    if "track_process_status" in track_process_data:
        payload["track_process_status"] = str(
            np.asarray(track_process_data["track_process_status"]).item()
        )
    selected_controller_count = int(np.asarray(payload["controller_points"]).shape[1])
    controller_mask_len = int(
        np.asarray(payload["controller_mask"]).reshape(-1).shape[0]
    )
    if (
        "controller_mask" in payload
        and "controller_candidate_mask" not in payload
        and controller_mask_len != selected_controller_count
    ):
        payload["controller_candidate_mask"] = np.ascontiguousarray(
            np.asarray(payload["controller_mask"], dtype=bool).reshape(-1)
        )
    if (
        "controller_query_indices" in payload
        and "controller_candidate_query_ids" not in payload
        and controller_mask_len != selected_controller_count
    ):
        payload["controller_candidate_query_ids"] = np.ascontiguousarray(
            np.asarray(payload["controller_query_indices"], dtype=np.int64).reshape(-1)
        )
    _validate_track_shapes(payload)
    return payload


def _sample_object_volume_indices(
    object_points: np.ndarray,
    *,
    surface_points: np.ndarray | None = None,
    interior_points: np.ndarray | None = None,
    volume_sample_size: float = 0.005,
) -> np.ndarray:
    """Choose first-frame object samples on the 5 mm occupancy grid."""
    # Offline parity with data_process_sam3d/data_process_sample.py:L281-L300.
    # That path keeps one first-frame object track per 5 mm voxel, with
    # shape-prior points included in the bounds when available.
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
    """Assemble the realtime final_data.pkl contract."""
    # Offline parity with data_process_sam3d/data_process_sample.py:L250-L352.
    # That path reduces track_process_data into final_data.pkl, carrying object
    # samples, controller handles, and optional surface/interior shape-prior
    # points.
    track_process = dict(track_process)
    object_points = np.asarray(track_process["object_points"])
    # If strict tracking already selected object tracks, preserve that selection.
    # Otherwise sample against the shape prior so final_data remains compact.
    if "object_track_query_indices" in track_process:
        indices = np.arange(object_points.shape[1], dtype=np.int64)
    else:
        indices = _sample_object_volume_indices(
            object_points,
            surface_points=surface_points,
            interior_points=interior_points,
            volume_sample_size=0.005,
        )
    if "object_track_query_indices" in track_process:
        object_sample_indices = indices
    else:
        object_sample_indices = np.asarray(
            track_process.get("object_volume_sample_indices", indices),
            dtype=np.int64,
        ).reshape(-1)
        if object_sample_indices.shape[0] != indices.shape[0]:
            object_sample_indices = indices
    object_query_indices = np.asarray(
        track_process.get(
            "object_query_indices", np.arange(object_points.shape[1], dtype=np.int64)
        ),
        dtype=np.int64,
    ).reshape(-1)
    if "object_track_query_indices" in track_process:
        object_selected_query_ids = np.asarray(
            track_process["object_track_query_indices"], dtype=np.int64
        ).reshape(-1)
    elif object_query_indices.shape[0] == object_points.shape[1]:
        object_selected_query_ids = object_query_indices[indices]
    else:
        object_selected_query_ids = indices

    controller_points = np.asarray(track_process["controller_points"])
    controller_count = int(controller_points.shape[1])
    # Controller points are always the selected control anchors. The *_query_ids
    # arrays record which original query each anchor came from.
    if "controller_track_query_indices" in track_process:
        controller_final_indices = np.arange(controller_count, dtype=np.int64)
        controller_selected_query_ids = np.asarray(
            track_process["controller_track_query_indices"],
            dtype=np.int64,
        ).reshape(-1)
    else:
        controller_final_indices = np.asarray(
            track_process.get(
                "controller_final_indices", np.arange(controller_count, dtype=np.int64)
            ),
            dtype=np.int64,
        ).reshape(-1)
        if controller_final_indices.shape[0] != controller_count:
            controller_final_indices = np.arange(controller_count, dtype=np.int64)
        candidate_count = int(
            max(
                controller_count,
                int(np.max(controller_final_indices)) + 1
                if controller_final_indices.size
                else 0,
            )
        )
        default_controller_query_indices = np.arange(
            object_points.shape[1],
            object_points.shape[1] + candidate_count,
            dtype=np.int64,
        )
        controller_query_indices = np.asarray(
            track_process.get(
                "controller_query_indices", default_controller_query_indices
            ),
            dtype=np.int64,
        ).reshape(-1)
        controller_selected_query_ids = np.full((controller_count,), -1, dtype=np.int64)
        valid = (controller_final_indices >= 0) & (
            controller_final_indices < controller_query_indices.shape[0]
        )
        controller_selected_query_ids[valid] = controller_query_indices[
            controller_final_indices[valid]
        ]
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
    _validate_final_shapes(final)
    return final


def _zero_point_count(points: np.ndarray) -> int:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim < 2 or pts.shape[-1] != 3 or pts.shape[0] == 0:
        return 0
    return int(np.count_nonzero(np.linalg.norm(pts[0], axis=-1) <= 1e-9))


def _quality_manifest_fields(
    final_data: Mapping[str, np.ndarray],
    track_process: Mapping[str, np.ndarray],
) -> dict[str, Any]:
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
    if "controller_track_confidence" in track_process:
        confidence = np.asarray(
            track_process["controller_track_confidence"], dtype=np.float32
        )
        payload["controller_track_mean_confidence"] = (
            float(np.mean(confidence)) if confidence.size else 1.0
        )
        payload["controller_track_low_confidence_ratio"] = (
            float(np.count_nonzero(confidence < 0.25) / confidence.size)
            if confidence.size
            else 0.0
        )
    if "controller_track_mode" in track_process:
        modes = np.asarray(track_process["controller_track_mode"], dtype=str)
        payload["controller_track_direct_frame_count"] = int(
            np.count_nonzero(modes == "direct_valid")
        )
        payload["controller_track_neighbor_recovered_frame_count"] = int(
            np.count_nonzero(np.char.find(modes.astype(str), "bundle_recovered") >= 0)
        )
        payload["controller_track_unrecoverable_frame_count"] = int(
            np.count_nonzero(np.char.find(modes.astype(str), "unrecoverable") >= 0)
        )
    if "track_process_status" in track_process:
        payload["track_process_status"] = str(track_process["track_process_status"])
    return payload


def build_data_process_chunk_payload(
    chunk: DataProcessChunk,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build final_data, track_process diagnostics, and manifest fields."""
    frame_count = _ensure_frame_count(chunk)
    track_process = _track_process_payload(chunk.track_process_data)
    final_data = _final_data_payload(
        track_process,
        surface_points=chunk.surface_points,
        interior_points=chunk.interior_points,
    )
    # Keep the same query schema hash in both payloads so online chunks can be
    # concatenated without changing query semantic identity.
    for key in DATA_PROCESS_QUERY_SCHEMA_KEYS:
        track_process[key] = final_data[key]
    _validate_query_schema_payload(track_process, label="track_process")
    manifest = {
        "frame_count": int(frame_count),
        "chunk_index": None if chunk.chunk_index is None else int(chunk.chunk_index),
        "camera_count": 1,
        "surface_point_count": int(final_data["surface_points"].shape[0]),
        "interior_point_count": int(final_data["interior_points"].shape[0]),
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


def _validate_track_shapes(payload: Mapping[str, np.ndarray]) -> None:
    """Validate the richer track-process payload before publishing."""
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
        raise ValueError(
            "controller_points must have shape T,M,3 matching object frame count"
        )
    controller_mask = np.asarray(payload["controller_mask"])
    if controller_mask.ndim != 1:
        raise ValueError(
            "controller_mask must be a 1D mask over candidate controller points"
        )
    controller_mask_is_candidate_level = (
        controller_mask.shape[0] != controller_points.shape[1]
    )
    controller_candidate_count = int(controller_mask.shape[0])
    controller_count = int(controller_points.shape[1])
    if "controller_final_indices" in payload:
        fps = np.asarray(payload["controller_final_indices"], dtype=np.int64).reshape(
            -1
        )
        if fps.shape[0] != controller_points.shape[1]:
            raise ValueError(
                "controller_final_indices must match selected controller point count"
            )
        if (
            controller_mask_is_candidate_level
            and fps.size
            and np.any((fps >= controller_mask.shape[0]) | (fps < -1))
        ):
            raise ValueError(
                "controller_final_indices must index controller_mask candidates or be -1"
            )
    if "controller_query_indices" in payload:
        query_ids = np.asarray(
            payload["controller_query_indices"], dtype=np.int64
        ).reshape(-1)
        if (
            controller_mask_is_candidate_level
            and query_ids.shape[0] != controller_mask.shape[0]
        ):
            raise ValueError(
                "controller_query_indices must match controller candidate count"
            )
    if "controller_candidate_query_ids" in payload:
        query_ids = np.asarray(
            payload["controller_candidate_query_ids"], dtype=np.int64
        ).reshape(-1)
        if (
            controller_mask_is_candidate_level
            and query_ids.shape[0] != controller_mask.shape[0]
        ):
            raise ValueError(
                "controller_candidate_query_ids must match controller candidate count"
            )
    for key in CONTROLLER_CANDIDATE_TIME_KEYS:
        if key not in payload:
            continue
        arr = np.asarray(payload[key])
        if arr.shape != (frame_count, controller_candidate_count):
            raise ValueError(
                f"{key} must have shape T,N matching controller candidates"
            )
    for key in CONTROLLER_CANDIDATE_POINT_KEYS:
        if key not in payload:
            continue
        arr = np.asarray(payload[key])
        if arr.shape != (frame_count, controller_candidate_count, 3):
            raise ValueError(
                f"{key} must have shape T,N,3 matching controller candidates"
            )
    for key in CONTROLLER_TRACK_TIME_KEYS:
        if key not in payload:
            continue
        arr = np.asarray(payload[key])
        if arr.shape != (frame_count, controller_count):
            raise ValueError(
                f"{key} must have shape T,M matching selected controller tracks"
            )
    for key in CONTROLLER_TRACK_STATIC_KEYS:
        if key not in payload:
            continue
        arr = np.asarray(payload[key])
        if arr.ndim != 2 or arr.shape[0] != controller_count:
            raise ValueError(
                f"{key} must have shape M,K matching selected controller tracks"
            )
    if (
        "track_process_status" in payload
        and str(payload["track_process_status"]) not in TRACK_PROCESS_STATUSES
    ):
        raise ValueError("track_process_status must be normal, degraded, or invalid")
    for key in ("object_volume_sample_indices", "object_track_query_indices"):
        if key not in payload:
            continue
        arr = np.asarray(payload[key]).reshape(-1)
        if arr.shape[0] != object_count:
            raise ValueError(f"{key} must match object point count")


def _validate_query_schema_payload(payload: Mapping[str, Any], *, label: str) -> None:
    """Validate the stable online query identity contract for one payload."""
    for key in DATA_PROCESS_QUERY_SCHEMA_KEYS:
        if key not in payload:
            raise ValueError(f"{label} missing required query schema key: {key}")
    query_ids = np.asarray(payload["query_ids"], dtype=np.int64).reshape(-1)
    query_semantic_labels = np.asarray(
        payload["query_semantic_labels"], dtype=np.int8
    ).reshape(-1)
    if query_ids.shape != query_semantic_labels.shape:
        raise ValueError(
            f"{label} query_ids and query_semantic_labels must have matching shape"
        )
    if not bool(
        np.all(
            np.isin(
                query_semantic_labels,
                np.array([0, 1, 2], dtype=np.int8),
            )
        )
    ):
        raise ValueError(f"{label} query_semantic_labels must contain only 0, 1, or 2")
    if (
        _scalar_str(payload["query_schema_version"])
        != DATA_PROCESS_QUERY_SCHEMA_VERSION
    ):
        raise ValueError(f"{label} unsupported query_schema_version")
    _validate_query_schema_sample_semantics(payload, label=label)
    expected_hash = _query_schema_hash(payload)
    if _scalar_str(payload["query_schema_hash"]) != expected_hash:
        raise ValueError(
            f"{label} query_schema_hash does not match query identity fields"
        )


def _validate_query_schema_sample_semantics(
    payload: Mapping[str, Any], *, label: str
) -> None:
    query_ids = np.asarray(payload["query_ids"], dtype=np.int64).reshape(-1)
    query_semantic_labels = np.asarray(
        payload["query_semantic_labels"], dtype=np.int8
    ).reshape(-1)
    unique_query_ids, counts = np.unique(query_ids, return_counts=True)
    duplicate_ids = unique_query_ids[counts > 1]
    if duplicate_ids.size:
        raise ValueError(
            f"{label} query_ids must be unique; duplicates={duplicate_ids[:5].tolist()}"
        )
    label_by_query_id = {
        int(query_id): int(semantic_label)
        for query_id, semantic_label in zip(
            query_ids.tolist(), query_semantic_labels.tolist()
        )
    }

    def require_sample_semantics(
        key: str, expected_label: int, semantic_name: str
    ) -> None:
        sample_ids = np.asarray(payload[key], dtype=np.int64).reshape(-1)
        if sample_ids.size == 0:
            return
        unique_sample_ids, sample_counts = np.unique(sample_ids, return_counts=True)
        duplicate_sample_ids = unique_sample_ids[sample_counts > 1]
        if duplicate_sample_ids.size:
            raise ValueError(
                f"{label} {key} must be unique; duplicates={duplicate_sample_ids[:5].tolist()}"
            )
        missing = [
            int(value)
            for value in sample_ids.tolist()
            if int(value) not in label_by_query_id
        ]
        if missing:
            raise ValueError(
                f"{label} {key} contains ids not present in query_ids: {missing[:5]}"
            )
        wrong = [
            int(value)
            for value in sample_ids.tolist()
            if label_by_query_id[int(value)] != int(expected_label)
        ]
        if wrong:
            raise ValueError(
                f"{label} {key} must reference {semantic_name} "
                f"semantic queries; wrong_ids={wrong[:5]}"
            )

    require_sample_semantics("object_sample_query_ids", int(1), "object")
    require_sample_semantics("controller_sample_query_ids", int(2), "controller")


def _validate_final_shapes(payload: Mapping[str, np.ndarray]) -> None:
    """Validate final_data.pkl shape, query schema, and finite-point invariants."""
    payload = dict(payload)
    for key in ("surface_points", "interior_points"):
        if key not in payload:
            raise ValueError(f"final_data.pkl missing required key: {key}")
    for key in DATA_PROCESS_FINAL_DATA_KEYS:
        if key not in payload:
            raise ValueError(f"final_data.pkl missing required key: {key}")
    if "controller_mask" in payload:
        raise ValueError(
            "final_data.pkl must not contain candidate-level controller_mask"
        )
    object_points = np.asarray(payload["object_points"], dtype=np.float64)
    if object_points.ndim != 3 or object_points.shape[-1] != 3:
        raise ValueError("object_points must have shape T,N,3")
    frame_count, object_count, _ = object_points.shape
    object_colors = np.asarray(payload["object_colors"], dtype=np.float64)
    if object_colors.shape != object_points.shape:
        raise ValueError("object_colors must match object_points shape")
    for key in ("object_visibilities", "object_motions_valid"):
        arr = np.asarray(payload[key])
        if arr.shape != (frame_count, object_count):
            raise ValueError(f"{key} must have shape T,N matching object_points")
    controller_points = np.asarray(payload["controller_points"], dtype=np.float64)
    if (
        controller_points.ndim != 3
        or controller_points.shape[0] != frame_count
        or controller_points.shape[-1] != 3
    ):
        raise ValueError(
            "controller_points must have shape T,M,3 matching object frame count"
        )
    controller_count = int(controller_points.shape[1])
    for key in (
        "controller_final_indices",
        "controller_selected_query_ids",
        "controller_sample_query_ids",
    ):
        arr = np.asarray(payload[key], dtype=np.int64).reshape(-1)
        if arr.shape[0] != controller_count:
            raise ValueError(f"{key} must match selected controller point count")
    for key in (
        "object_sample_indices",
        "object_selected_query_ids",
        "object_sample_query_ids",
    ):
        arr = np.asarray(payload[key], dtype=np.int64).reshape(-1)
        if arr.shape[0] != object_count:
            raise ValueError(f"{key} must match object point count")
    _validate_query_schema_payload(payload, label="final_data.pkl")
    if not np.isfinite(object_points).all() or not np.isfinite(controller_points).all():
        raise ValueError("object/controller points must be finite")
    if object_points.shape[1] and np.any(
        np.linalg.norm(object_points[0], axis=1) <= 1e-9
    ):
        raise ValueError("first-frame object points contain zero-depth placeholders")
    if controller_points.shape[1] and np.any(
        np.linalg.norm(controller_points[0], axis=1) <= 1e-9
    ):
        raise ValueError(
            "first-frame controller points contain zero-depth placeholders"
        )
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


__all__ = [
    "DATA_PROCESS_FINAL_DATA_KEYS",
    "DATA_PROCESS_SAM3D_REALTIME_CONTRACT_VERSION",
    "DATA_PROCESS_QUERY_SCHEMA_KEYS",
    "DATA_PROCESS_QUERY_SCHEMA_VERSION",
    "DATA_PROCESS_TRACK_PROCESS_KEYS",
    "DATA_PROCESS_SAM3D_METRICS",
    "DataProcessChunk",
    "build_data_process_chunk_payload",
    "build_query_schema_payload",
]
