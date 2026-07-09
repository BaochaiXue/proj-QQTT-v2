"""Warmup-delayed startup row trimming for Demo v6.1 chunk streaming."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from demo_v6_2.chunk_jsonl_tail import _optional_int


# ---------------------------------------------------------------------------
# Warmup-delayed startup rows
# ---------------------------------------------------------------------------
def _row_ready_for_realtime_chunk_start(row: Mapping[str, Any]) -> bool:
    """Return the row ready for realtime chunk start."""
    controller_points = _optional_int(row.get("controller_point_count"))
    if controller_points is not None and int(controller_points) < 30:
        return False
    object_points = _optional_int(row.get("object_point_count"))
    if object_points is not None and int(object_points) <= 0:
        return False
    return True


@dataclass(frozen=True)
class WarmupTrimResult:
    rows: list[dict[str, Any]]
    skipped_count: int


def _trim_warmup_delayed_rows(rows: Sequence[Mapping[str, Any]]) -> WarmupTrimResult:
    """Drop invalid startup rows while preserving warmup frame 0 for chunking.

    Demo v6.1 writes ``input_frames.jsonl`` from camera start, but
    ``frames.jsonl`` is the data_process output stream. With shape-prior warmup,
    the first strict pair can be source frame 0 processed many seconds late; the
    next row then jumps to the realtime source frame after warmup. That delayed
    row is the online frame 0 and must remain in ``chunk_000000``.
    Live RealSense can also emit one strict pair before color-aligned PCD is
    ready; that invalid row has masks but zero controller/object points and must
    not anchor controller FPS.
    """
    trimmed = [dict(row) for row in rows]
    skipped = 0
    while trimmed and not _row_ready_for_realtime_chunk_start(trimmed[0]):
        skipped += 1
        trimmed.pop(0)
    return WarmupTrimResult(rows=trimmed, skipped_count=int(skipped))


@dataclass
class _WarmupStartFilterState:
    pending_rows: list[dict[str, Any]] = field(default_factory=list)
    resolved: bool = False
    skipped_rows: int = 0


def _filter_warmup_start_rows(
    state: _WarmupStartFilterState,
    rows: Sequence[Mapping[str, Any]],
    *,
    capture_finished: bool,
) -> list[dict[str, Any]]:
    """Hold the first live rows until warmup-delayed source rows can be trimmed."""
    if state.resolved:
        return [dict(row) for row in rows]
    state.pending_rows.extend(dict(row) for row in rows)
    trim_result = _trim_warmup_delayed_rows(state.pending_rows)
    if trim_result.skipped_count:
        state.skipped_rows += int(trim_result.skipped_count)
        state.pending_rows = trim_result.rows
    if len(state.pending_rows) < 2 and not bool(capture_finished):
        return []
    state.resolved = True
    output = list(state.pending_rows)
    state.pending_rows = []
    return output
