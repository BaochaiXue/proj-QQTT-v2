"""Convert Demo v6.1 headless capture rows into realtime chunks.

The camera process appends ``frames.jsonl`` and prepared per-frame NPZ payloads.
This bridge tails that stream, waits for the shape prior only at chunk materialize
time, and publishes online chunk payloads plus the static final_data view.
"""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Callable, Mapping

import numpy as np

from demo_v6_2.chunk_data_output import ChunkDataWriter
from demo_v6_2.online_frame_archive import OnlineFrameArchive
from demo_v6_2 import asap
from demo_v6_2 import phystwin_strict_product as strict
from demo_v6_2 import tracking

from demo_v6_2.chunk_jsonl_tail import (
    _complete_chunk_backlog,
    _iter_jsonl,
    _read_jsonl_from_offset,
    _relative_wall_s,
)
from demo_v6_2.chunk_warmup_trim import (
    _WarmupStartFilterState,
    _filter_warmup_start_rows,
    _trim_warmup_delayed_rows,
)
from demo_v6_2.chunk_capture_meta import (
    _read_json_file_stable,
    _shape_points_for_chunk,
    _shape_points_from_capture,
    _wait_for_asap_case_dir,
)
from demo_v6_2.chunk_window_builder import _prepared_frame_from_row
from demo_v6_2.chunk_materialize import _write_chunk_from_rows


# ---------------------------------------------------------------------------
# Entry points: completed-capture conversion and live tailing
# ---------------------------------------------------------------------------
def write_chunk_data_from_headless_capture(
    capture_dir: str | Path,
    *,
    base_path: str | Path,
    case_prefix: str = "demo_v6_1",
    chunk_frame_count: int = 25,
    fps: int = 5,
    max_chunks: int | None = None,
    surface_points: np.ndarray | None = None,
    interior_points: np.ndarray | None = None,
    on_chunk_written: Callable[[dict[str, Any]], None] | None = None,
    write_online_output: bool = True,
    online_case_name: str | None = None,
    asap_augment: bool = True,
    asap_mesh_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Convert a completed headless capture into online final_data chunks."""
    capture = Path(capture_dir)
    if int(chunk_frame_count) <= 0:
        raise ValueError("chunk_frame_count must be positive")
    metadata = _read_json_file_stable(
        capture / "metadata.json",
        deadline_s=time.monotonic() + 5.0,
        poll_interval_s=0.05,
    )
    shape_surface, shape_interior = _shape_points_from_capture(
        capture,
        metadata,
        surface_points=surface_points,
        interior_points=interior_points,
    )
    serials = metadata.get("serial_numbers") or ["demo-v6-1-single-camera"]
    serial_number = str(serials[0])

    manifests: list[dict[str, Any]] = []
    chunk_size = int(chunk_frame_count)
    chunk_index = 0
    row_buffer: list[dict[str, Any]] = []
    prepared_buffer: list[strict.PreparedPhysTwinFrame] = []
    row_start = 0
    wall_time_origin_s = time.monotonic()
    frames_path = capture / "frames.jsonl"
    trim_result = _trim_warmup_delayed_rows(list(_iter_jsonl(frames_path)))
    rows_to_process = trim_result.rows
    warmup_skipped_rows = trim_result.skipped_count
    online_writer = None
    frame_archive = None
    online_case = str(online_case_name or case_prefix)
    if bool(write_online_output):
        # Only complete windows are published, so the static final_data view is
        # sized to whole chunks (optionally capped by max_chunks).
        full_chunks = len(rows_to_process) // chunk_size
        if max_chunks is not None:
            full_chunks = min(full_chunks, int(max_chunks))
        online_writer = ChunkDataWriter(
            base_path=base_path,
            case_name=online_case,
            chunk_size=chunk_size,
            num_frames_total=full_chunks * chunk_size,
        )
        frame_archive = OnlineFrameArchive(
            base_path=base_path,
            case_name=online_case,
            fps=int(fps),
        )
        # Seed calibrate.pkl/metadata.json (frame_num=0) immediately so
        # downstream consumers launched before the first chunk commit can
        # read the case dir (design_spec_v6_1.md, downstream.mode).
        frame_archive.initialize_case(metadata, serial_number=serial_number)
    # Keep the tracking runtime and query schema alive across chunks. Chunk 0
    # freezes identity (design_spec.md): object columns, controller anchors,
    # and the neighbor table persist for the whole stream, so later windows
    # never reselect object/controller points.
    # Offline parity with data_process_sam3d/data_process_sample.py:L281-L300
    # and data_process_sam3d/data_process_track.py:L338-L356. Offline sampling
    # happens once per case; Demo v6.1 treats the live stream as one case.
    tracking_runtime = tracking.TrackingRuntime()
    # design_spec_v6_1.md: the ASAP augmenter is session-lived like the tracking
    # runtime; it initializes from the first window and fails fast when the
    # aligned shape-prior mesh is missing.
    asap_runtime = (
        asap.AsapRuntime(mesh_path_override=asap_mesh_path)
        if bool(asap_augment)
        else None
    )
    session_query_schema: dict[str, np.ndarray] = {}
    for row_idx, row in enumerate(rows_to_process):
        if max_chunks is not None and len(manifests) >= int(max_chunks):
            break
        row_buffer.append(row)
        prepared_buffer.append(_prepared_frame_from_row(capture, row))
        if len(row_buffer) < chunk_size:
            continue
        window_closed_wall_s = _relative_wall_s(float(wall_time_origin_s))
        # Borrow frame: the next row in the completed capture, when present.
        # The exact-multiple tail window publishes end-of-sequence semantics.
        lookahead_prepared: list[strict.PreparedPhysTwinFrame] = []
        if row_idx + 1 < len(rows_to_process):
            borrow_row = rows_to_process[row_idx + 1]
            lookahead_prepared = [_prepared_frame_from_row(capture, borrow_row)]
        try:
            manifest = _write_chunk_from_rows(
                capture=capture,
                metadata=metadata,
                rows=row_buffer,
                case_prefix=case_prefix,
                chunk_index=chunk_index,
                row_start=row_start,
                row_end=row_idx + 1,
                fps=int(fps),
                serial_number=serial_number,
                surface_points=shape_surface,
                interior_points=shape_interior,
                wall_time_origin_s=wall_time_origin_s,
                window_closed_wall_s=window_closed_wall_s,
                prepared_frames=prepared_buffer,
                backlog_chunks=(
                    lambda path=frames_path,
                    size=chunk_size,
                    published=chunk_index + 1: _complete_chunk_backlog(
                        path,
                        chunk_size=size,
                        published_chunk_count=published,
                    )
                ),
                online_writer=online_writer,
                frame_archive=frame_archive,
                tracking_runtime=tracking_runtime,
                session_query_schema=session_query_schema,
                warmup_skipped_rows=warmup_skipped_rows,
                lookahead_prepared=lookahead_prepared,
                asap_runtime=asap_runtime,
            )
        except Exception:
            # design_spec.md failures (controller selection / recovery) abort
            # the stream; leave a terminal manifest instead of "recording".
            if online_writer is not None:
                online_writer.finish(status="failed")
            raise
        manifests.append(manifest)
        if on_chunk_written is not None:
            on_chunk_written(manifest)
        chunk_index += 1
        row_start = row_idx + 1
        row_buffer = []
        prepared_buffer = []
    if online_writer is not None:
        online_writer.finish()
    return manifests


def _wait_for_metadata(
    capture: Path, *, capture_finished: Callable[[], bool], poll_interval_s: float
) -> Mapping[str, Any]:
    """Wait for for metadata."""
    metadata_path = capture / "metadata.json"
    while True:
        if metadata_path.is_file():
            try:
                return _read_json_file_stable(
                    metadata_path,
                    deadline_s=time.monotonic()
                    + max(0.5, float(poll_interval_s) * 4.0),
                    poll_interval_s=float(poll_interval_s),
                )
            except RuntimeError:
                if capture_finished():
                    raise RuntimeError(
                        f"capture finished before stable metadata appeared: {metadata_path}"
                    )
        elif capture_finished():
            raise RuntimeError(
                f"capture finished before metadata appeared: {metadata_path}"
            )
        time.sleep(max(0.0, float(poll_interval_s)))


def stream_chunk_data_from_headless_capture(
    capture_dir: str | Path,
    *,
    base_path: str | Path,
    case_prefix: str = "demo_v6_1",
    chunk_frame_count: int = 25,
    fps: int = 5,
    max_chunks: int | None = None,
    capture_finished: Callable[[], bool],
    before_poll: Callable[[], None] | None = None,
    poll_interval_s: float = 0.05,
    surface_points: np.ndarray | None = None,
    interior_points: np.ndarray | None = None,
    require_shape_prior: bool = False,
    shape_prior_wait_timeout_s: float = 300.0,
    on_chunk_written: Callable[[dict[str, Any]], None] | None = None,
    write_online_output: bool = True,
    online_case_name: str | None = None,
    asap_augment: bool = True,
    asap_mesh_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Tail a live headless capture and publish each closed window once its
    borrow row (the next window's first row) arrives; at capture end the
    final full window flushes without a borrow row."""
    capture = Path(capture_dir)
    if int(chunk_frame_count) <= 0:
        raise ValueError("chunk_frame_count must be positive")
    metadata = _wait_for_metadata(
        capture,
        capture_finished=capture_finished,
        poll_interval_s=float(poll_interval_s),
    )
    serials = metadata.get("serial_numbers") or ["demo-v6-1-single-camera"]
    serial_number = str(serials[0])
    frames_path = capture / "frames.jsonl"
    manifests: list[dict[str, Any]] = []
    next_row_idx = 0
    frames_offset = 0
    row_start = 0
    row_buffer: list[dict[str, Any]] = []
    prepared_buffer: list[strict.PreparedPhysTwinFrame] = []
    chunk_index = 0
    chunk_size = int(chunk_frame_count)
    wall_time_origin_s = time.monotonic()
    warmup_start_filter = _WarmupStartFilterState()
    online_writer = None
    frame_archive = None
    online_case = str(online_case_name or case_prefix)
    if bool(write_online_output):
        online_writer = ChunkDataWriter(
            base_path=base_path,
            case_name=online_case,
            chunk_size=chunk_size,
            num_frames_total=(
                None if max_chunks is None else int(max_chunks) * int(chunk_size)
            ),
        )
        frame_archive = OnlineFrameArchive(
            base_path=base_path,
            case_name=online_case,
            fps=int(fps),
        )
        # Seed calibrate.pkl/metadata.json (frame_num=0) as soon as capture
        # metadata is known: phystwin_shen downstream consumers start at
        # shape-prior-ready time and read the case dir before the first
        # chunk commits (design_spec_v6_1.md, downstream.mode).
        frame_archive.initialize_case(metadata, serial_number=serial_number)
    # Live streaming uses the same stateful tracking runtime as offline
    # conversion so chunk N+1 continues the identity frozen by chunk 0.
    tracking_runtime = tracking.TrackingRuntime()
    # design_spec_v6_1.md: session-lived ASAP augmenter; initializes from the
    # first materialized window (the aligned mesh exists only after shape-
    # prior warmup) and fails fast when final_mesh.glb is missing.
    asap_runtime = (
        asap.AsapRuntime(mesh_path_override=asap_mesh_path)
        if bool(asap_augment)
        else None
    )
    session_query_schema: dict[str, np.ndarray] = {}
    # A full window is held pending until its borrow row (the next window's
    # first row) arrives, so the published tail row carries a real motion
    # verdict. This trades one output frame of publish latency; steady-state
    # throughput is unchanged. At capture end the pending window flushes
    # without a borrow row (origin end-of-sequence tail semantics) — the
    # final full window is never dropped.
    pending_window: dict[str, Any] | None = None

    def _materialize_pending(
        pending: dict[str, Any],
        lookahead_prepared: list[strict.PreparedPhysTwinFrame],
    ) -> dict[str, Any]:
        latest_metadata, shape_surface, shape_interior = _shape_points_for_chunk(
            capture,
            surface_points=surface_points,
            interior_points=interior_points,
            require_shape_prior=bool(require_shape_prior),
            shape_prior_wait_timeout_s=float(shape_prior_wait_timeout_s),
            capture_finished=capture_finished,
            before_poll=before_poll,
            poll_interval_s=float(poll_interval_s),
        )
        if (
            asap_runtime is not None
            and not asap_runtime.initialized
            and asap_runtime.mesh_path_override is None
        ):
            latest_metadata = _wait_for_asap_case_dir(
                capture,
                latest_metadata,
                shape_prior_wait_timeout_s=float(shape_prior_wait_timeout_s),
                before_poll=before_poll,
                poll_interval_s=float(poll_interval_s),
            )
        try:
            return _write_chunk_from_rows(
                capture=capture,
                metadata=latest_metadata,
                rows=pending["rows"],
                case_prefix=case_prefix,
                chunk_index=int(pending["chunk_index"]),
                row_start=int(pending["row_start"]),
                row_end=int(pending["row_end"]),
                fps=int(fps),
                serial_number=serial_number,
                surface_points=shape_surface,
                interior_points=shape_interior,
                wall_time_origin_s=wall_time_origin_s,
                window_closed_wall_s=float(pending["window_closed_wall_s"]),
                prepared_frames=pending["prepared"],
                backlog_chunks=(
                    lambda path=frames_path,
                    size=chunk_size,
                    published=int(pending["chunk_index"]) + 1: _complete_chunk_backlog(
                        path,
                        chunk_size=size,
                        published_chunk_count=published,
                    )
                ),
                online_writer=online_writer,
                frame_archive=frame_archive,
                tracking_runtime=tracking_runtime,
                session_query_schema=session_query_schema,
                warmup_skipped_rows=warmup_start_filter.skipped_rows,
                lookahead_prepared=lookahead_prepared,
                asap_runtime=asap_runtime,
            )
        except Exception:
            # design_spec.md failures (controller selection / recovery)
            # abort the stream; leave a terminal manifest instead of
            # "recording".
            if online_writer is not None:
                online_writer.finish(status="failed")
            raise

    while True:
        if max_chunks is not None and len(manifests) >= int(max_chunks):
            break
        if before_poll is not None:
            before_poll()
        rows, frames_offset = _read_jsonl_from_offset(frames_path, frames_offset)
        saw_new_rows = bool(rows)
        rows = _filter_warmup_start_rows(
            warmup_start_filter,
            rows,
            capture_finished=bool(capture_finished()),
        )
        for row in rows:
            prepared_row = _prepared_frame_from_row(capture, row)
            if pending_window is not None:
                # This row is the pending window's borrow frame — and also
                # the first row of the next window below.
                manifest = _materialize_pending(pending_window, [prepared_row])
                pending_window = None
                manifests.append(manifest)
                if on_chunk_written is not None:
                    on_chunk_written(manifest)
                chunk_index += 1
                if max_chunks is not None and len(manifests) >= int(max_chunks):
                    break
            # A chunk closes strictly by row count. Shape-prior readiness and
            # final_data materialization are handled at borrow-row arrival so
            # source pacing remains tied to the camera/fake-camera stream.
            row_buffer.append(row)
            prepared_buffer.append(prepared_row)
            next_row_idx += 1
            if len(row_buffer) < chunk_size:
                continue
            pending_window = {
                "rows": row_buffer,
                "prepared": prepared_buffer,
                "chunk_index": chunk_index,
                "row_start": row_start,
                "row_end": next_row_idx,
                "window_closed_wall_s": _relative_wall_s(float(wall_time_origin_s)),
            }
            row_start = next_row_idx
            row_buffer = []
            prepared_buffer = []
        if max_chunks is not None and len(manifests) >= int(max_chunks):
            break
        if capture_finished() and not saw_new_rows:
            if pending_window is not None:
                # Terminal flush: no successor row will arrive.
                manifest = _materialize_pending(pending_window, [])
                pending_window = None
                manifests.append(manifest)
                if on_chunk_written is not None:
                    on_chunk_written(manifest)
                chunk_index += 1
            break
        time.sleep(max(0.0, float(poll_interval_s)))
    if online_writer is not None:
        online_writer.finish()
    return manifests


__all__ = [
    "stream_chunk_data_from_headless_capture",
    "write_chunk_data_from_headless_capture",
]
