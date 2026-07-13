"""Validated timing records for the Demo v6.2 shape-prior critical path."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from demo_v6_2.utils.atomic_io import atomic_json_dump


SHAPE_PRIOR_TIMING_SCHEMA_VERSION = 1
STAGE_PROFILE_STATUS_WAITING = "waiting_for_go"
STAGE_PROFILE_STATUS_COMPLETED = "completed"


def elapsed_ms(start_s: float, end_s: float | None = None) -> float:
    """Return a non-negative ``perf_counter`` duration in milliseconds."""
    end = time.perf_counter() if end_s is None else float(end_s)
    duration_ms = (end - float(start_s)) * 1000.0
    if not math.isfinite(duration_ms) or duration_ms < 0.0:
        raise ValueError(f"invalid timing duration: {duration_ms}")
    return float(duration_ms)


def _validated_duration(value: Any, *, field: str) -> float:
    try:
        duration = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite non-negative number") from exc
    if not math.isfinite(duration) or duration < 0.0:
        raise ValueError(f"{field} must be a finite non-negative number")
    return duration


def _validate_timing_tree(value: Any, *, field: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _validate_timing_tree(item, field=f"{field}.{key}")
        return
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        _validated_duration(value, field=field)
        return
    if field.endswith("_ms"):
        _validated_duration(value, field=field)


def write_stage_profile(
    path: str | Path | None,
    *,
    stage: str,
    status: str,
    execution_mode: str,
    timing_ms: Mapping[str, Any],
    ready_wall_time_s: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Write one subprocess timing snapshot when the parent requested it."""
    if path is None:
        return
    stage_name = str(stage).strip()
    if not stage_name:
        raise ValueError("shape-prior timing stage must be non-empty")
    timing = dict(timing_ms)
    _validate_timing_tree(timing, field=f"{stage_name}.timing_ms")
    payload: dict[str, Any] = {
        "schema_version": SHAPE_PRIOR_TIMING_SCHEMA_VERSION,
        "stage": stage_name,
        "status": str(status),
        "execution_mode": str(execution_mode),
        "worker_pid": int(os.getpid()),
        "snapshot_wall_time_s": float(time.time()),
        "ready_wall_time_s": (
            None if ready_wall_time_s is None else float(ready_wall_time_s)
        ),
        "timing_ms": timing,
    }
    if metadata is not None:
        payload["metadata"] = dict(metadata)
    atomic_json_dump(payload, path)


def load_completed_stage_profile(
    path: str | Path,
    *,
    expected_stage: str,
) -> dict[str, Any]:
    """Load and validate a completed subprocess timing profile."""
    profile_path = Path(path)
    if not profile_path.is_file():
        raise FileNotFoundError(
            f"shape-prior stage {expected_stage!r} did not write {profile_path}"
        )
    try:
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid shape-prior timing JSON: {profile_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(
            f"shape-prior timing profile must be an object: {profile_path}"
        )
    if int(payload.get("schema_version", -1)) != SHAPE_PRIOR_TIMING_SCHEMA_VERSION:
        raise ValueError(f"unsupported shape-prior timing schema: {profile_path}")
    if str(payload.get("stage", "")) != str(expected_stage):
        raise ValueError(
            f"shape-prior timing stage mismatch: expected {expected_stage!r}, "
            f"got {payload.get('stage')!r}"
        )
    if str(payload.get("status", "")) != STAGE_PROFILE_STATUS_COMPLETED:
        raise ValueError(
            f"shape-prior stage {expected_stage!r} timing is not completed"
        )
    timing = payload.get("timing_ms")
    if not isinstance(timing, dict):
        raise ValueError(
            f"shape-prior stage {expected_stage!r} timing_ms must be an object"
        )
    _validate_timing_tree(timing, field=f"{expected_stage}.timing_ms")
    return payload


def _pre_submit_timing(
    frame0: Any,
    *,
    request_start_s: float,
) -> dict[str, Any]:
    """Describe the camera-process critical path before shape-prior submit.

    ``frame0`` is a ``shape_prior_warmup.ShapePriorFrame0Request``; it is
    typed loosely so this timing module needs no warmup import.
    """
    milestones = (
        frame0.warmup_runtime_start_perf_s,
        frame0.frame_receive_perf_s,
        frame0.frame_mask_ready_perf_s,
        frame0.frame_pcd_ready_perf_s,
    )
    if all(value is None for value in milestones):
        return {
            "available": False,
            "reason": "standalone request has no camera warm-up milestones",
        }
    if any(value is None for value in milestones):
        raise ValueError("shape-prior pre-submit milestones must be all present")
    runtime_start_s, receive_s, mask_ready_s, pcd_ready_s = (
        float(value) for value in milestones if value is not None
    )
    ordered = (
        runtime_start_s,
        receive_s,
        mask_ready_s,
        pcd_ready_s,
        float(request_start_s),
    )
    if any(current < previous for previous, current in zip(ordered, ordered[1:])):
        raise ValueError("shape-prior pre-submit milestones are not monotonic")
    return {
        "available": True,
        "runtime_start_to_frame0_receive_ms": elapsed_ms(
            runtime_start_s,
            receive_s,
        ),
        "frame0_receive_to_mask_ready_ms": elapsed_ms(receive_s, mask_ready_s),
        "mask_ready_to_pcd_ready_ms": elapsed_ms(mask_ready_s, pcd_ready_s),
        "pcd_ready_to_shape_prior_submit_ms": elapsed_ms(
            pcd_ready_s,
            request_start_s,
        ),
        "runtime_start_to_shape_prior_submit_ms": elapsed_ms(
            runtime_start_s,
            request_start_s,
        ),
        "frame0_pipeline_timing_ms": dict(frame0.frame0_pipeline_timing_ms),
        "perception_profile": dict(frame0.frame0_perception_profile),
    }


def critical_path_entry(
    *,
    stage: str,
    path_start_s: float,
    stage_start_s: float,
    stage_end_s: float,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one validated sequential critical-path entry."""
    start_offset_ms = elapsed_ms(path_start_s, stage_start_s)
    duration_ms = elapsed_ms(stage_start_s, stage_end_s)
    entry: dict[str, Any] = {
        "stage": str(stage),
        "start_offset_ms": start_offset_ms,
        "end_offset_ms": start_offset_ms + duration_ms,
        "duration_ms": duration_ms,
    }
    if details is not None:
        entry["details"] = dict(details)
    return entry


def build_critical_path_analysis(
    entries: Sequence[Mapping[str, Any]],
    *,
    total_ms: float,
) -> dict[str, Any]:
    """Validate a sequential timeline and rank its optimization targets."""
    total = _validated_duration(total_ms, field="critical_path.total_ms")
    normalized: list[dict[str, Any]] = []
    previous_start = -1.0
    previous_end = 0.0
    for index, raw_entry in enumerate(entries):
        stage = str(raw_entry.get("stage", "")).strip()
        if not stage:
            raise ValueError(f"critical_path[{index}].stage must be non-empty")
        start = _validated_duration(
            raw_entry.get("start_offset_ms"),
            field=f"critical_path[{index}].start_offset_ms",
        )
        duration = _validated_duration(
            raw_entry.get("duration_ms"),
            field=f"critical_path[{index}].duration_ms",
        )
        end = _validated_duration(
            raw_entry.get("end_offset_ms", start + duration),
            field=f"critical_path[{index}].end_offset_ms",
        )
        if start < previous_start or start + 1e-6 < previous_end:
            raise ValueError("shape-prior critical-path stages must be sequential")
        if abs(end - (start + duration)) > 1e-3:
            raise ValueError(
                f"critical_path[{index}] end does not equal start + duration"
            )
        normalized_entry = {
            "stage": stage,
            "start_offset_ms": start,
            "end_offset_ms": end,
            "duration_ms": duration,
        }
        if "details" in raw_entry:
            details = raw_entry["details"]
            if not isinstance(details, Mapping):
                raise ValueError(f"critical_path[{index}].details must be an object")
            normalized_entry["details"] = dict(details)
        normalized.append(normalized_entry)
        previous_start = start
        previous_end = end

    accounted_ms = float(sum(entry["duration_ms"] for entry in normalized))
    if accounted_ms > total + 1e-3:
        raise ValueError("shape-prior accounted time exceeds total wall time")
    ranking = sorted(
        (
            {
                "stage": entry["stage"],
                "duration_ms": entry["duration_ms"],
                "share_percent": (
                    0.0 if total == 0.0 else float(entry["duration_ms"] / total * 100.0)
                ),
            }
            for entry in normalized
        ),
        key=lambda item: item["duration_ms"],
        reverse=True,
    )
    return {
        "schema_version": SHAPE_PRIOR_TIMING_SCHEMA_VERSION,
        "clock": "time.perf_counter wall duration",
        "total_ms": total,
        "accounted_ms": accounted_ms,
        "unattributed_ms": max(0.0, total - accounted_ms),
        "critical_path": normalized,
        "ranking": ranking,
        "bottleneck": None if not ranking else dict(ranking[0]),
    }


__all__ = [
    "SHAPE_PRIOR_TIMING_SCHEMA_VERSION",
    "STAGE_PROFILE_STATUS_COMPLETED",
    "STAGE_PROFILE_STATUS_WAITING",
    "build_critical_path_analysis",
    "critical_path_entry",
    "elapsed_ms",
    "load_completed_stage_profile",
    "write_stage_profile",
]
