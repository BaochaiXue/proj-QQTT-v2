"""Chunk materialization and publish for Demo v6.2 chunk streaming.

Pipeline questions Q16-Q22 (see PIPELINE.md): a chunk closes strictly by frame/row
COUNT (``chunk_size`` = ``round(replay_fps * chunk_seconds)``, default 25), its
per-frame arrays are stacked, tracked through the session-lived
``tracking.TrackingRuntime`` (frozen chunk-0 identity + motion-consistency filter +
anchor recovery), optionally ASAP-augmented, then written as
``online_data/chunks/chunk_{id:06d}.pkl`` plus the RGB-D archive. The manifest is
rewritten fsync-ordered on each commit (chunk dumped -> frame fsynced -> manifest
rewritten) so a reader that sees a new manifest is guaranteed durable data.
"""

from __future__ import annotations

from dataclasses import replace as dataclass_replace
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np

from demo_v6_2.streaming.data_payload import build_window_publish_payloads
from demo_v6_2.streaming.online_frame_archive import OnlineFrameArchiveError
from demo_v6_2 import phystwin_strict_product as strict
from demo_v6_2.streaming.jsonl_tail import (
    _complete_chunk_backlog,
    _relative_wall_s,
    _rows_source_timestamps,
)
from demo_v6_2.streaming.window_builder import _chunk_data_window_from_prepared_frames

if TYPE_CHECKING:
    from demo_v6_2.streaming.session import ChunkStreamSession


# ---------------------------------------------------------------------------
# Track manifest telemetry
# ---------------------------------------------------------------------------
def _controller_track_manifest_fields(
    track_process_data: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the controller track manifest fields."""
    track_process_data = dict(track_process_data)
    query_indices = np.asarray(
        track_process_data["controller_track_query_indices"], dtype=np.int64
    ).reshape(-1)
    active_indices = np.asarray(
        track_process_data.get("controller_track_active_query_indices", query_indices),
        dtype=np.int64,
    ).reshape(-1)
    statuses = np.asarray(
        track_process_data.get("controller_track_status", []), dtype=str
    ).reshape(-1)
    payload = {
        "controller_track_selection_mode": "streaming_stable",
        "controller_track_count": int(len(query_indices)),
        "controller_track_query_indices": [
            int(value) for value in query_indices.tolist()
        ],
        "controller_track_active_query_indices": [
            int(value) for value in active_indices.tolist()
        ],
        "controller_track_direct_count": int(np.count_nonzero(statuses == "direct")),
        "controller_track_proxied_count": int(np.count_nonzero(statuses == "proxied")),
        "controller_track_status": [str(value) for value in statuses.tolist()],
    }
    if "controller_proxied" in track_process_data:
        proxied = np.asarray(track_process_data["controller_proxied"], dtype=bool)
        payload["controller_track_direct_frame_count"] = int(
            proxied.size - np.count_nonzero(proxied)
        )
        payload["controller_track_proxied_frame_count"] = int(
            np.count_nonzero(proxied)
        )
        payload["controller_track_proxied_ratio"] = (
            float(np.count_nonzero(proxied) / proxied.size) if proxied.size else 0.0
        )
    if "track_process_status" in track_process_data:
        payload["track_process_status"] = str(
            np.asarray(track_process_data["track_process_status"]).item()
        )
    return payload


def _object_track_manifest_fields(
    track_process_data: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the object track manifest fields."""
    track_process_data = dict(track_process_data)
    query_indices = np.asarray(
        track_process_data["object_track_query_indices"], dtype=np.int64
    ).reshape(-1)
    active_indices = np.asarray(
        track_process_data.get("object_track_active_query_indices", query_indices),
        dtype=np.int64,
    ).reshape(-1)
    statuses = np.asarray(
        track_process_data.get("object_track_status", []), dtype=str
    ).reshape(-1)
    return {
        "object_track_selection_mode": "streaming_stable",
        "object_track_count": int(len(query_indices)),
        "object_track_query_indices": [int(value) for value in query_indices.tolist()],
        "object_track_active_query_indices": [
            int(value) for value in active_indices.tolist()
        ],
        "object_track_direct_count": int(np.count_nonzero(statuses == "direct")),
        "object_track_revived_count": int(np.count_nonzero(statuses == "revived")),
        "object_track_fallback_count": int(np.count_nonzero(statuses == "fallback")),
        "object_track_missing_count": int(np.count_nonzero(statuses == "missing")),
        "object_track_status_summary": {
            "direct": int(np.count_nonzero(statuses == "direct")),
            "revived": int(np.count_nonzero(statuses == "revived")),
            "fallback": int(np.count_nonzero(statuses == "fallback")),
            "missing": int(np.count_nonzero(statuses == "missing")),
        },
    }


def _materialize_and_commit_window(
    session: ChunkStreamSession,
    pending: Mapping[str, Any],
    borrow_prepared: Sequence[strict.PreparedPhysTwinFrame],
    *,
    metadata: Mapping[str, Any],
    surface_points: np.ndarray,
    interior_points: np.ndarray,
) -> dict[str, Any]:
    """Materialize one pending final_data window and commit it online.

    ``session`` owns every cross-window constant (capture location, cadence,
    the online writer and RGB-D archive, the tracking runtime, the frozen
    query schema); ``pending`` is one closed source window (rows, prepared
    frames, row bounds, close time). ``metadata``/``surface_points``/
    ``interior_points`` are the shape-prior products the caller resolved
    immediately before this call.

    ``borrow_prepared`` carries the borrow frame, normally the next window's
    first row, so the published tail row gets a real motion verdict. Without it
    at capture end, the tail row publishes origin's end-of-sequence semantics.

    ``session.asap_runtime`` (design_spec_v6_1.md) fills invalid
    ``object_points`` entries in place and publishes deformed shape-prior
    trajectories as dedicated ``asap_surface_points`` / ``asap_interior_points``
    keys before payload assembly.
    """
    rows: Sequence[Mapping[str, Any]] = pending["rows"]
    chunk_index = int(pending["chunk_index"])
    row_start = int(pending["row_start"])
    row_end = int(pending["row_end"])
    window_closed_wall_s = float(pending["window_closed_wall_s"])
    chunk_name = f"{session.case_prefix}_online_chunk_{chunk_index:04d}"
    source_window_start_s = float(row_start) / float(session.fps)
    source_window_end_s = float(row_end) / float(session.fps)
    materialize_start_wall_s = _relative_wall_s(session.wall_time_origin_s)
    published_frames = list(pending["prepared"])
    if len(published_frames) != len(rows):
        raise OnlineFrameArchiveError(
            f"chunk {chunk_index}: {len(rows)} capture rows require exactly "
            f"{len(rows)} prepared frames, got {len(published_frames)}"
        )
    borrow_prepared = list(borrow_prepared)
    # Prepared frames carry RGB, masks, dense world PCD, full tracks,
    # visibility, and query points synchronized to the same source sequence.
    chunk = _chunk_data_window_from_prepared_frames(
        session,
        metadata,
        published_frames,
        surface_points=surface_points,
        interior_points=interior_points,
        chunk_index=chunk_index,
        borrow_frames=borrow_prepared,
    )
    prepared_count = len(published_frames)
    motion_lookahead_frames = len(borrow_prepared)
    track_finalize_done_wall_s = max(
        _relative_wall_s(session.wall_time_origin_s), window_closed_wall_s
    )
    # Track manifest fields describe the REAL tracking output, so compute them
    # before ASAP fills invalid object entries and attaches dedicated
    # shape-prior trajectory keys.
    track_fields = _object_track_manifest_fields(chunk.track_process_data)
    track_fields.update(_controller_track_manifest_fields(chunk.track_process_data))
    asap_summary: dict[str, Any] = {}
    if session.asap_runtime is not None:
        # design_spec_v6_1.md: the ASAP augmenter runs live at materialization,
        # after the tracking state machine and before payload assembly. It
        # fails fast when final_mesh.glb is unavailable.
        augmented_track_process, asap_summary = session.asap_runtime.augment_window(
            chunk.track_process_data,
            metadata=metadata,
            surface_points=chunk.surface_points,
            interior_points=chunk.interior_points,
        )
        chunk = dataclass_replace(chunk, track_process_data=augmented_track_process)
    chunk_source_frame_indices = [int(value) for value in chunk.source_frame_indices]
    chunk_source_timestamps_s = _rows_source_timestamps(rows)

    final_data, track_process, manifest = build_window_publish_payloads(
        chunk, volume_sample_size_m=session.volume_sample_size_m
    )
    manifest.update(
        {
            "chunk_name": chunk_name,
            "online_publish_skipped": False,
        }
    )
    # Provenance plus latency telemetry. The chunk backlog is sampled here,
    # after track finalization, so it reflects the backlog at publish time.
    manifest.update(
        {
            "source_capture_dir": str(session.capture),
            "source_row_start": int(row_start),
            "source_row_end": int(row_end),
            "warmup_skipped_rows": int(session.warmup_skipped_rows),
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
            "backlog_chunks": int(
                _complete_chunk_backlog(
                    session.frames_path,
                    chunk_size=session.chunk_size,
                    published_chunk_count=chunk_index + 1,
                )
            ),
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
        _relative_wall_s(session.wall_time_origin_s),
        timing_floor_s,
    )
    manifest["final_data_written_wall_s"] = float(final_data_written_wall_s)
    # Raw color/depth land BEFORE the chunk commit so a committed chunk always
    # has its per-frame products; any missing frame aborts the stream through
    # the session's failed-manifest path.
    archive_summary = session.frame_archive.archive_chunk(
        chunk_id=int(session.online_writer.latest_committed_chunk + 1),
        frames=published_frames,
        source_frame_indices=chunk_source_frame_indices,
        online_start_frame=int(session.online_writer.latest_committed_frame),
    )
    manifest.update(archive_summary)
    online_result = session.online_writer.commit_chunk_data(
        final_data,
        track_process,
        source_frame_indices=chunk_source_frame_indices,
        source_timestamps_s=chunk_source_timestamps_s,
        status="recording",
    )
    # metadata.json advances only after the chunk commit, so its frame_num
    # never counts frames of an uncommitted chunk.
    session.frame_archive.publish_metadata()
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
        _relative_wall_s(session.wall_time_origin_s), final_data_written_wall_s
    )
    manifest["publish_wall_s"] = float(publish_wall_s)
    manifest["publish_latency_ms"] = float(
        (publish_wall_s - window_closed_wall_s) * 1000.0
    )
    manifest["publish_lag_ms"] = float((publish_wall_s - source_window_end_s) * 1000.0)
    return manifest
