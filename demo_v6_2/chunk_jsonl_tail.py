"""frames.jsonl tailing helpers for Demo v6.1 chunk streaming.

frames.jsonl is append-only and owned by the camera subprocess. The helpers in
this section either tolerate incomplete rows or normalize source-frame metadata
so chunking stays deterministic during live tailing.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Iterator, Mapping, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# frames.jsonl tailing
# ---------------------------------------------------------------------------
# frames.jsonl is append-only and owned by the camera subprocess. The helpers in
# this section either tolerate incomplete rows or normalize source-frame metadata
# so chunking stays deterministic during live tailing.
def _read_jsonl_from_offset(
    path: Path, offset: int
) -> tuple[list[dict[str, Any]], int]:
    """Read only newly appended complete JSONL rows.

    ``frames.jsonl`` is written by another process. If the writer is in the
    middle of a row, return the rows that were already complete and leave the
    offset at the start of the partial line for the next poll.
    """
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows, int(offset)
    with path.open("r", encoding="utf-8") as handle:
        handle.seek(max(0, int(offset)))
        while True:
            row_offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            stripped = line.strip()
            if stripped:
                try:
                    rows.append(json.loads(stripped))
                except json.JSONDecodeError:
                    return rows, int(row_offset)
        return rows, int(handle.tell())


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Return the iter JSONL."""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _relative_wall_s(origin_s: float) -> float:
    # All *_wall_s manifest timings are seconds since one shared monotonic
    # origin so latencies stay comparable across chunks within a run.
    """Convert an absolute monotonic timestamp into run-relative seconds."""
    return float(time.monotonic() - float(origin_s))


def _complete_chunk_backlog(
    frames_path: Path, *, chunk_size: int, published_chunk_count: int
) -> int:
    """Count fully captured but not-yet-published chunks (manifest telemetry).

    A window is publishable once its borrow row (the next window's first
    row) exists, so the newest full-but-borrowless window does not count as
    backlog while the capture is still running.
    """
    if int(chunk_size) <= 0:
        return 0
    row_count = 0
    if frames_path.is_file():
        with frames_path.open("r", encoding="utf-8") as handle:
            row_count = sum(1 for line in handle if line.strip())
    return max(0, max(0, row_count - 1) // int(chunk_size) - int(published_chunk_count))


def _optional_int(value: Any) -> int | None:
    """Return the optional int."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    """Return the optional float."""
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _rows_source_frame_indices(
    rows: Sequence[Mapping[str, Any]], *, fallback_start: int
) -> list[int]:
    """Per-row source frame index: source_frame_index, else seq, else position."""
    indices: list[int] = []
    for offset, row in enumerate(rows):
        value = _optional_int(row.get("source_frame_index"))
        if value is None:
            value = _optional_int(row.get("seq"))
        indices.append(
            int(value) if value is not None else int(fallback_start) + offset
        )
    return indices


def _rows_source_timestamps(rows: Sequence[Mapping[str, Any]]) -> list[float] | None:
    """Per-row capture timestamps; None unless every row carries a finite one."""
    values: list[float] = []
    for row in rows:
        value = _optional_float(row.get("source_timestamp_s"))
        if value is None:
            return None
        values.append(float(value))
    return values
