"""Chunk materialization and publish for Demo v6.2 chunk streaming.

Pipeline questions Q15-Q21 (see PIPELINE.md): a chunk closes strictly by frame/row
COUNT (``chunk_size`` = ``round(replay_fps * chunk_seconds)``, default 35), its
per-frame arrays are stacked, tracked through the session-lived
``tracking.TrackingRuntime`` (frozen chunk-0 identity + motion-consistency filter +
anchor recovery), optionally ASAP-augmented, then written as
``online_data/chunks/chunk_{id:06d}.pkl`` plus the RGB-D archive. The manifest is
rewritten fsync-ordered on each commit (chunk dumped -> frame fsynced -> manifest
rewritten) so a reader that sees a new manifest is guaranteed durable data.
"""

from __future__ import annotations

from dataclasses import replace as dataclass_replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from demo_v6_2.chunk_data_payload import build_chunk_data_payload
from demo_v6_2.chunk_data_output import ChunkDataWriter
from demo_v6_2.online_frame_archive import (
    OnlineFrameArchive,
    OnlineFrameArchiveError,
)
from demo_v6_2 import asap
from demo_v6_2 import phystwin_strict_product as strict
from demo_v6_2 import tracking
from demo_v6_2.chunk_jsonl_tail import (
    _relative_wall_s,
    _rows_source_frame_indices,
    _rows_source_timestamps,
)
from demo_v6_2.chunk_capture_meta import (
    _controller_track_manifest_fields,
    _object_track_manifest_fields,
)
from demo_v6_2.chunk_window_builder import _chunk_data_window_from_prepared_frames


def _write_chunk_from_rows(
    *,
    capture: Path,
    metadata: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    case_prefix: str,
    chunk_index: int,
    row_start: int,
    row_end: int,
    fps: int,
    serial_number: str,
    surface_points: np.ndarray,
    interior_points: np.ndarray,
    wall_time_origin_s: float,
    window_closed_wall_s: float,
    prepared_frames: Sequence[strict.PreparedPhysTwinFrame],
    backlog_chunks: Callable[[], int] | None = None,
    online_writer: ChunkDataWriter | None = None,
    frame_archive: OnlineFrameArchive | None = None,
    tracking_runtime: tracking.TrackingRuntime | None = None,
    session_query_schema: dict[str, np.ndarray] | None = None,
    warmup_skipped_rows: int = 0,
    lookahead_prepared: Sequence[strict.PreparedPhysTwinFrame] = (),
    asap_runtime: asap.AsapRuntime | None = None,
) -> dict[str, Any]:
    """Materialize one final_data window and optionally commit it online.

    ``lookahead_prepared`` carries the borrow frame, normally the next window's
    first row, so the published tail row gets a real motion verdict. Without it
    at capture end, the tail row publishes origin's end-of-sequence semantics.

    ``asap_runtime`` (design_spec_v6_1.md) fills invalid ``object_points``
    entries in place and publishes deformed shape-prior trajectories as
    dedicated ``asap_surface_points`` / ``asap_interior_points`` keys before
    payload assembly.
    """
    chunk_name = f"{case_prefix}_online_chunk_{chunk_index:04d}"
    source_window_start_s = float(row_start) / float(fps)
    source_window_end_s = float(row_end) / float(fps)
    materialize_start_wall_s = _relative_wall_s(float(wall_time_origin_s))
    published_frames = list(prepared_frames)
    if len(published_frames) != len(rows):
        raise OnlineFrameArchiveError(
            f"chunk {chunk_index}: {len(rows)} capture rows require exactly "
            f"{len(rows)} prepared frames, got {len(published_frames)}"
        )
    prepared_lookahead = list(lookahead_prepared)
    # Prepared frames carry RGB, masks, dense world PCD, full tracks,
    # visibility, and query points synchronized to the same source sequence.
    chunk = _chunk_data_window_from_prepared_frames(
        metadata,
        published_frames,
        surface_points=surface_points,
        interior_points=interior_points,
        fps=int(fps),
        serial_number=serial_number,
        chunk_index=chunk_index,
        tracking_runtime=tracking_runtime,
        session_query_schema=session_query_schema,
        lookahead_frames=prepared_lookahead,
    )
    prepared_count = len(published_frames)
    motion_lookahead_frames = len(prepared_lookahead)
    track_finalize_done_wall_s = max(
        _relative_wall_s(float(wall_time_origin_s)), window_closed_wall_s
    )
    # Track manifest fields describe the REAL tracking output, so compute them
    # before ASAP fills invalid object entries and attaches dedicated
    # shape-prior trajectory keys.
    track_fields = _object_track_manifest_fields(chunk.track_process_data)
    track_fields.update(_controller_track_manifest_fields(chunk.track_process_data))
    asap_summary: dict[str, Any] = {}
    if asap_runtime is not None:
        # design_spec_v6_1.md: the ASAP augmenter runs live at materialization,
        # after the tracking state machine and before payload assembly. It
        # fails fast when final_mesh.glb is unavailable.
        augmented_track_process, asap_summary = asap_runtime.augment_window(
            chunk.track_process_data,
            metadata=metadata,
            surface_points=chunk.surface_points,
            interior_points=chunk.interior_points,
        )
        chunk = dataclass_replace(chunk, track_process_data=augmented_track_process)
    chunk_source_frame_indices = (
        [int(value) for value in chunk.source_frame_indices]
        if chunk.source_frame_indices is not None
        else _rows_source_frame_indices(rows, fallback_start=row_start)
    )
    chunk_source_timestamps_s = _rows_source_timestamps(rows)

    final_data, track_process, manifest = build_chunk_data_payload(chunk)
    manifest.update(
        {
            "chunk_name": chunk_name,
            "online_publish_skipped": False,
        }
    )
    # Provenance plus latency telemetry. backlog_chunks() is sampled here,
    # after track finalization, so it reflects the backlog at publish time.
    manifest.update(
        {
            "source_capture_dir": str(capture),
            "source_row_start": int(row_start),
            "source_row_end": int(row_end),
            "warmup_skipped_rows": int(warmup_skipped_rows),
            "source_window_start_s": source_window_start_s,
            "source_window_end_s": source_window_end_s,
            "chunk_ready_source_seq": int(rows[-1].get("seq", row_end - 1)),
            "chunk_ready_source_frame_index": int(chunk_source_frame_indices[-1]),
            "chunk_ready_source_time_s": (
                None
                if rows[-1].get("source_timestamp_s") is None
                else float(rows[-1]["source_timestamp_s"])
            ),
            "window_closed_wall_s": float(window_closed_wall_s),
            "track_finalize_done_wall_s": float(track_finalize_done_wall_s),
            "materialize_start_wall_s": materialize_start_wall_s,
            "materialize_end_wall_s": track_finalize_done_wall_s,
            "materialize_latency_ms": float(
                (track_finalize_done_wall_s - materialize_start_wall_s) * 1000.0
            ),
            "backlog_chunks": 0 if backlog_chunks is None else int(backlog_chunks()),
            "chunk_materialization_source": "prepared_data_process_frame",
            "prepared_frame_count": int(prepared_count),
            "legacy_reprocess_frame_count": 0,
            "motion_lookahead_frames": int(motion_lookahead_frames),
        }
    )
    if asap_summary:
        manifest.update(asap_summary)
    manifest.update(track_fields)
    timing_floor_s = max(
        float(manifest.get("window_closed_wall_s", 0.0) or 0.0),
        float(manifest.get("track_finalize_done_wall_s", 0.0) or 0.0),
    )
    final_data_written_wall_s = max(
        _relative_wall_s(float(wall_time_origin_s)),
        timing_floor_s,
    )
    manifest["final_data_written_wall_s"] = float(final_data_written_wall_s)
    if online_writer is not None:
        if frame_archive is not None:
            # Raw color/depth land BEFORE the chunk commit so a committed
            # chunk always has its per-frame products; any missing frame
            # aborts the stream through the existing failed-manifest path.
            archive_summary = frame_archive.archive_chunk(
                chunk_id=int(online_writer.latest_committed_chunk + 1),
                metadata=metadata,
                serial_number=serial_number,
                frames=published_frames,
                source_frame_indices=chunk_source_frame_indices,
                source_timestamps_s=chunk_source_timestamps_s,
                online_start_frame=int(online_writer.latest_committed_frame),
            )
            manifest.update(archive_summary)
        online_result = online_writer.commit_chunk_data(
            final_data,
            track_process,
            source_frame_indices=chunk_source_frame_indices,
            source_timestamps_s=chunk_source_timestamps_s,
            status="recording",
        )
        if frame_archive is not None:
            # metadata.json advances only after the chunk commit, so its
            # frame_num never counts frames of an uncommitted chunk.
            frame_archive.publish_metadata()
        manifest.update(
            {
                "online_dir": online_result["online_dir"],
                "online_manifest_path": online_result["online_manifest_path"],
                "online_chunk_path": online_result["online_chunk_path"],
                "online_chunk_id": online_result["online_chunk_id"],
                "online_latest_committed_frame": online_result[
                    "online_latest_committed_frame"
                ],
                "static_data_path": online_result["static_data_path"],
            }
        )
    publish_wall_s = max(
        _relative_wall_s(float(wall_time_origin_s)), final_data_written_wall_s
    )
    manifest["publish_wall_s"] = float(publish_wall_s)
    manifest["publish_latency_ms"] = float(
        (publish_wall_s - window_closed_wall_s) * 1000.0
    )
    manifest["publish_lag_ms"] = float((publish_wall_s - source_window_end_s) * 1000.0)
    return manifest
