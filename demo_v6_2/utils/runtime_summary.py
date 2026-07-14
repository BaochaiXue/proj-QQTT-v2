"""Run-level summary helpers for Demo v6.2 chunk manifests."""

from __future__ import annotations

from typing import Sequence


def runtime_chunk_summary(
    manifests: Sequence[dict[str, object]],
) -> dict[str, object]:
    """Aggregate per-chunk manifests into run-level publish/quality stats."""
    # publish_wall_s values are wall-clock seconds; consecutive differences
    # measure the steady-state publish cadence downstream consumers observed.
    publish_times = [
        float(item["publish_wall_s"])
        for item in manifests
        if item.get("publish_wall_s") is not None
    ]
    intervals = [
        publish_times[idx] - publish_times[idx - 1]
        for idx in range(1, len(publish_times))
    ]
    backlog_values = [
        int(item["backlog_chunks"])
        for item in manifests
        if item.get("backlog_chunks") is not None
    ]
    shape_publish_times = [
        float(item["publish_wall_s"])
        for item in manifests
        if item.get("publish_wall_s") is not None
        and bool(item.get("shape_prior_complete"))
    ]
    # The worst chunk status becomes the run-level status. Statuses missing
    # from the table rank lowest (-1) so a stray label never outranks a real
    # degraded/invalid signal.
    quality_order = {"normal": 0, "degraded": 1, "invalid": 2}
    quality_values = [
        str(item.get("track_process_status", "normal")) for item in manifests
    ]
    track_process_status = "normal"
    if quality_values:
        track_process_status = max(
            quality_values, key=lambda value: quality_order.get(value, -1)
        )
    quality_counts = {
        status: int(sum(1 for value in quality_values if value == status))
        for status in ("normal", "degraded", "invalid")
    }
    invalid_chunks = [
        str(item.get("chunk_name", item.get("chunk_index", "")))
        for item in manifests
        if str(item.get("track_process_status", "normal")) == "invalid"
    ]
    return {
        "first_ready_chunk_wall_s": publish_times[0] if publish_times else None,
        "first_shape_prior_ready_chunk_wall_s": (
            shape_publish_times[0] if shape_publish_times else None
        ),
        "steady_publish_intervals_s": intervals,
        "steady_state_publish_interval_max_s": max(intervals) if intervals else None,
        "max_backlog_chunks": max(backlog_values) if backlog_values else None,
        "track_process_status": track_process_status,
        "track_process_status_counts": quality_counts,
        "track_process_invalid_chunk_count": int(len(invalid_chunks)),
        "track_process_invalid_chunks": invalid_chunks,
        "online_publish_skipped_chunk_count": int(
            sum(
                1
                for item in manifests
                if bool(item.get("online_publish_skipped", False))
            )
        ),
    }
