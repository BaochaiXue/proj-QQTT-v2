"""Warmup-delayed startup row trimming for Demo v6.2 chunk streaming."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from demo_v6_2.streaming.jsonl_tail import _optional_int


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


class WarmupStartFilter:
    """Hold the first live rows until warmup-delayed source rows can be trimmed."""

    def __init__(self) -> None:
        self._pending_rows: list[dict[str, Any]] = []
        self._resolved = False
        self._skipped_rows = 0

    @property
    def skipped_rows(self) -> int:
        """Startup rows trimmed exactly once before the filter resolved."""
        return int(self._skipped_rows)

    def filter(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        capture_finished: bool,
    ) -> list[dict[str, Any]]:
        """Pass rows through once resolved; buffer and trim the startup rows."""
        if self._resolved:
            return [dict(row) for row in rows]
        self._pending_rows.extend(dict(row) for row in rows)
        self._trim_unready_startup_rows()
        if len(self._pending_rows) < 2 and not bool(capture_finished):
            return []
        self._resolved = True
        output = list(self._pending_rows)
        self._pending_rows = []
        return output

    def _trim_unready_startup_rows(self) -> None:
        """Drop invalid startup rows while preserving warmup frame 0 for chunking.

        Demo v6.2 writes ``input_frames.jsonl`` from camera start, but
        ``frames.jsonl`` is the data_process output stream. With shape-prior
        warmup, the first strict pair can be source frame 0 processed many
        seconds late; the next row then jumps to the realtime source frame
        after warmup. That delayed row is the online frame 0 and must remain
        in ``chunk_000000``.
        Live RealSense can also emit one strict pair before color-aligned PCD
        is ready; that invalid row has masks but zero controller/object points
        and must not anchor controller FPS.
        """
        while self._pending_rows and not _row_ready_for_realtime_chunk_start(
            self._pending_rows[0]
        ):
            self._skipped_rows += 1
            self._pending_rows.pop(0)
