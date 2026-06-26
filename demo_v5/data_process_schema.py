"""Canonical Demo v5 data_process_sam3d schema names and legacy aliases."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np


LEGACY_TO_CANONICAL_KEYS = {
    "controller_anchor_query_indices": "controller_track_query_indices",
    "controller_anchor_active_query_indices": "controller_track_active_query_indices",
    "controller_anchor_status": "controller_track_status",
    "controller_anchor_bundle_query_ids": "controller_neighbor_query_ids",
    "controller_anchor_source_query_id": "controller_source_query_ids",
    "controller_anchor_observation_mode": "controller_track_mode",
    "controller_anchor_confidence": "controller_track_confidence",
    "controller_anchor_failure_reason": "controller_filter_reason",
    "controller_anchor_bundle_support_count": "controller_neighbor_support_count",
    "controller_anchor_bundle_raw_visible_count": "controller_neighbor_raw_visible_count",
    "controller_anchor_bundle_depth_valid_count": "controller_neighbor_depth_valid_count",
    "controller_anchor_bundle_processed_mask_valid_count": "controller_neighbor_processed_mask_valid_count",
    "controller_anchor_bundle_motion_valid_count": "controller_neighbor_motion_valid_count",
    "controller_anchor_recovery_residual": "controller_neighbor_fit_residual",
    "controller_anchor_mean_confidence": "controller_track_mean_confidence",
    "controller_anchor_low_confidence_ratio": "controller_track_low_confidence_ratio",
    "controller_anchor_mode": "controller_track_selection_mode",
    "controller_anchor_count": "controller_track_count",
    "controller_anchor_direct_count": "controller_track_direct_count",
    "controller_anchor_recovered_count": "controller_track_recovered_count",
    "controller_anchor_direct_frame_count": "controller_track_direct_frame_count",
    "controller_anchor_bundle_recovered_frame_count": "controller_track_neighbor_recovered_frame_count",
    "controller_anchor_unrecoverable_frame_count": "controller_track_unrecoverable_frame_count",
    "controller_anchor_observation_mode_summary": "controller_track_mode_summary",
    "controller_anchor_revived_count": "controller_track_revived_count",
    "controller_anchor_fallback_count": "controller_track_fallback_count",
    "controller_anchor_missing_count": "controller_track_missing_count",
    "object_anchor_mode": "object_track_selection_mode",
    "object_anchor_count": "object_track_count",
    "object_anchor_direct_count": "object_track_direct_count",
    "object_anchor_revived_count": "object_track_revived_count",
    "object_anchor_fallback_count": "object_track_fallback_count",
    "object_anchor_missing_count": "object_track_missing_count",
    "object_anchor_query_indices": "object_track_query_indices",
    "object_anchor_active_query_indices": "object_track_active_query_indices",
    "object_anchor_status": "object_track_status",
    "controller_quality_status": "track_process_status",
    "controller_fps_indices": "controller_final_indices",
    "topology_version": "query_schema_version",
    "topology_hash": "query_schema_hash",
    "futurephystwin_case_root": "data_process_case_root",
    "futurephystwin_base_path": "base_path",
}

CANONICAL_TO_LEGACY_KEYS = {value: key for key, value in LEGACY_TO_CANONICAL_KEYS.items()}

QUERY_SCHEMA_KEYS = (
    "query_schema_version",
    "query_schema_hash",
    "query_ids",
    "query_semantic_labels",
    "object_sample_query_ids",
    "controller_sample_query_ids",
)

LEGACY_QUERY_SCHEMA_KEYS = tuple(CANONICAL_TO_LEGACY_KEYS.get(key, key) for key in QUERY_SCHEMA_KEYS)

TRACK_PROCESS_STATUSES = ("normal", "degraded", "invalid")


def canonical_key(key: str) -> str:
    return LEGACY_TO_CANONICAL_KEYS.get(str(key), str(key))


def legacy_key(key: str) -> str:
    return CANONICAL_TO_LEGACY_KEYS.get(str(key), str(key))


def _compatible_values(left: Any, right: Any) -> bool:
    try:
        left_arr = np.asarray(left)
        right_arr = np.asarray(right)
        if left_arr.shape != right_arr.shape:
            return False
        if left_arr.dtype.kind in {"U", "S", "O"} or right_arr.dtype.kind in {"U", "S", "O"}:
            return bool(np.array_equal(left_arr.astype(str), right_arr.astype(str)))
        return bool(np.array_equal(left_arr, right_arr))
    except Exception:
        return left == right


def normalize_data_process_keys(
    payload: Mapping[str, Any],
    *,
    keep_legacy_aliases: bool = False,
    validate_aliases: bool = True,
) -> dict[str, Any]:
    """Return a copy using canonical data_process_sam3d keys.

    Canonical keys win when both spellings are present. For aliases whose values
    are directly comparable, mismatches are rejected so stale mixed contracts do
    not pass silently.
    """
    normalized: dict[str, Any] = {}
    legacy_aliases: dict[str, Any] = {}
    for key, value in payload.items():
        canonical = canonical_key(str(key))
        if canonical == key:
            normalized[str(key)] = value
        else:
            legacy_aliases[str(key)] = value
            normalized.setdefault(canonical, value)

    if validate_aliases:
        for old_key, old_value in legacy_aliases.items():
            new_key = canonical_key(old_key)
            if new_key not in payload:
                continue
            if new_key in {"query_schema_version", "query_schema_hash"}:
                continue
            if not _compatible_values(payload[new_key], old_value):
                raise ValueError(f"legacy key {old_key!r} conflicts with canonical key {new_key!r}")

    if keep_legacy_aliases:
        normalized.update(legacy_aliases)
    return normalized


def add_legacy_aliases(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy with legacy aliases added for compatibility readers."""
    result = dict(payload)
    for canonical, legacy in CANONICAL_TO_LEGACY_KEYS.items():
        if canonical in result and legacy not in result:
            result[legacy] = result[canonical]
    return result


def status_from_payload(payload: Mapping[str, Any], *, default: str = "normal") -> str:
    normalized = normalize_data_process_keys(payload, validate_aliases=False)
    return str(np.asarray(normalized.get("track_process_status", default)).item())
