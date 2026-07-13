"""Live pipeline-status event stream shared by every Demo v6.2 process.

This module answers design question 25 — "visualize, in the real demo, what the
pipeline is doing right now (and whether warm-up failed)". The three cooperating
processes (the orchestrator ``main.py``, the camera/tracker/warm-up runtime
``main_data_processing.py``, and the SAM3D shape-prior stages) each append one
JSON line per lifecycle event to ``<base_path>/pipeline_status.jsonl``. The
visualizer tails that file and draws a status band
(``viz_panels.draw_pipeline_status``).

Contract:
- Writing is BEST-EFFORT and never raises: a status write must never break the
  realtime pipeline, change its timing meaningfully, or alter any published
  product. ``pipeline_status.jsonl`` is a brand-new sidecar file, so it is
  invisible to the online chunk / final_data / online_data products and to the
  byte-parity gate.
- Every process opens the file in append mode and writes one whole line per
  event, so concurrent appends from the three processes stay line-atomic.
- Each event carries a wall-clock epoch ``t`` so the renderer can order events
  across processes and show recency; ``stage`` is one of the STAGE_* constants
  below so writers and the renderer agree on a vocabulary.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

STATUS_FILENAME = "pipeline_status.jsonl"

# Canonical stage vocabulary shared by writers and the renderer. ``detail``
# carries the human-readable specifics (e.g. the shape-prior sub-stage name or an
# error message); ``ok=False`` marks a failure the renderer paints red.
STAGE_RUN_START = "run_start"
STAGE_CAPTURE_START = "capture_start"
STAGE_FIRST_FRAME = "first_frame_seeded"
STAGE_SHAPE_PRIOR = "shape_prior"
STAGE_WARMUP_READY = "warmup_ready"
STAGE_CHUNK_COMMITTED = "chunk_committed"
STAGE_DOWNSTREAM_START = "downstream_start"
STAGE_RUN_FINISHED = "run_finished"
STAGE_FATAL = "fatal_error"

# Short human labels the renderer shows for each stage.
STAGE_LABELS = {
    STAGE_RUN_START: "starting",
    STAGE_CAPTURE_START: "capturing",
    STAGE_FIRST_FRAME: "frame-0 seeded",
    STAGE_SHAPE_PRIOR: "shape-prior",
    STAGE_WARMUP_READY: "warm-up ready",
    STAGE_CHUNK_COMMITTED: "streaming chunks",
    STAGE_DOWNSTREAM_START: "downstream started",
    STAGE_RUN_FINISHED: "finished",
    STAGE_FATAL: "FAILED",
}


def status_path(base_path: str | Path) -> Path:
    """Return the fixed ``pipeline_status.jsonl`` path under ``base_path``."""
    return Path(base_path) / STATUS_FILENAME


class PipelineStatusWriter:
    """Best-effort append-only writer; never raises, never blocks the pipeline."""

    def __init__(self, base_path: str | Path | None, source: str) -> None:
        # ``source`` names the emitting process (orchestrator / camera /
        # shape_prior) so the renderer can attribute each event. A ``None``
        # base_path yields a no-op writer (e.g. a camera process launched without
        # a headless capture dir), so call sites never need to guard emits.
        self._path = status_path(base_path) if base_path is not None else None
        self._source = str(source)

    def emit(self, stage: str, detail: str = "", *, ok: bool = True, **fields: Any) -> None:
        """Append one status event. Any failure is swallowed by design."""
        if self._path is None:
            return
        record: dict[str, Any] = {
            "t": time.time(),
            "source": self._source,
            "stage": str(stage),
            "detail": str(detail),
            "ok": bool(ok),
        }
        record.update(fields)
        try:
            line = json.dumps(record, sort_keys=True) + "\n"
            # Append mode (O_APPEND) makes each single-line write atomic across
            # the three cooperating processes.
            with open(self._path, "a", encoding="utf-8") as handle:
                handle.write(line)
        except Exception:
            # Telemetry must never break the realtime pipeline.
            pass


def read_status_events(base_path: str | Path, *, tail: int | None = None) -> list[dict[str, Any]]:
    """Read status events for the renderer, tolerating a torn last line."""
    path = status_path(base_path)
    try:
        raw_lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    if tail is not None:
        raw_lines = raw_lines[-int(tail):]
    events: list[dict[str, Any]] = []
    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            # A live tail can catch a partially-flushed final line; skip it.
            continue
    return events
