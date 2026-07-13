"""Build Demo v6.2 chunk windows from canonical prepared frames."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from demo_v6_2.streaming.chunk_data_payload import ChunkDataWindow
from demo_v6_2.online_frame_archive import OnlineFrameArchiveError
from demo_v6_2 import phystwin_strict_product as strict
from demo_v6_2 import tracking
from demo_v6_2.chunk_capture_meta import (
    _camera_to_world,
    _intrinsics_matrix,
)


# ---------------------------------------------------------------------------
# Chunk materialization and publish
# ---------------------------------------------------------------------------
def _track_input_with_session_query_schema(
    *,
    tracks_yx: np.ndarray,
    visibility: np.ndarray,
    mask_frames: Sequence[Mapping[str, Any]],
    pcd_points: np.ndarray,
    pcd_colors: np.ndarray,
    session_query_schema: dict[str, np.ndarray] | None,
) -> dict[str, np.ndarray]:
    """Build window observations while preserving session-wide query identity.

    The first chunk fixes the query id/semantic-label arrays. Later chunks must
    reuse the same arrays so online output can be concatenated without changing
    object/controller topology.
    """
    # Offline parity with data_process_sam3d/data_process_track.py:L58-L118.
    # That stage labels first-frame tracks by object/controller masks and lifts
    # visible track pixels into world-space PCD samples. Demo v6.1 also pins
    # query ids across realtime chunks so those role labels stay stable online.
    query_ids = None
    query_semantic_labels = None
    if session_query_schema is not None and "query_ids" in session_query_schema:
        query_ids = session_query_schema["query_ids"]
        query_semantic_labels = session_query_schema["query_semantic_labels"]
    track_input = tracking.build_window_observations(
        tracks_yx=tracks_yx,
        visibility=visibility,
        mask_frames=mask_frames,
        pcd_points=pcd_points,
        pcd_colors=pcd_colors,
        query_ids=query_ids,
        query_semantic_labels=query_semantic_labels,
    )
    if session_query_schema is None:
        return track_input
    if "query_ids" not in session_query_schema:
        session_query_schema["query_ids"] = np.ascontiguousarray(
            track_input["query_ids"], dtype=np.int64
        )
        session_query_schema["query_semantic_labels"] = np.ascontiguousarray(
            track_input["query_semantic_labels"],
            dtype=np.int8,
        )
        return track_input
    if not np.array_equal(session_query_schema["query_ids"], track_input["query_ids"]):
        raise ValueError("Demo v6.1 session query_ids changed across chunks")
    if not np.array_equal(
        session_query_schema["query_semantic_labels"],
        track_input["query_semantic_labels"],
    ):
        raise ValueError("Demo v6.1 session query_semantic_labels changed across chunks")
    return track_input


def _prepared_frame_from_row(
    capture_dir: Path,
    row: Mapping[str, Any],
) -> strict.PreparedPhysTwinFrame:
    """Load the required prepared frame referenced by one capture row."""
    path_value = row.get("prepared_phystwin_frame_path")
    if path_value is None:
        raise OnlineFrameArchiveError(
            "headless capture row is missing prepared_phystwin_frame_path: "
            f"seq={row.get('seq')}, "
            f"source_frame_index={row.get('source_frame_index')}"
        )
    path = capture_dir / str(path_value)
    if not path.is_file():
        raise OnlineFrameArchiveError(
            "prepared PhysTwin frame does not exist: "
            f"{path} (seq={row.get('seq')}, "
            f"source_frame_index={row.get('source_frame_index')})"
        )
    return strict.load_prepared_phystwin_frame(path)


def _chunk_data_window_from_prepared_frames(
    metadata: Mapping[str, Any],
    frames: Sequence[strict.PreparedPhysTwinFrame],
    *,
    surface_points: np.ndarray,
    interior_points: np.ndarray,
    fps: int,
    serial_number: str,
    chunk_index: int,
    tracking_runtime: tracking.TrackingRuntime | None = None,
    session_query_schema: dict[str, np.ndarray] | None = None,
    lookahead_frames: Sequence[strict.PreparedPhysTwinFrame] = (),
) -> ChunkDataWindow:
    """Materialize a chunk from prepared per-frame NPZ payloads.

    This is the current v6.1 realtime path. The camera process already wrote
    RGB, masks, dense world PCD, tracks, visibility, and query points for the
    same source frame, so this function only enforces shared queries and runs
    the design_spec.md tracking state machine. ``lookahead_frames`` are borrow
    frames: they extend the motion-consistency domain but are excluded from
    the published window.
    """
    if not frames:
        raise ValueError("prepared data_process chunk requires at least one frame")
    # Prepared frames already carry world-space PCD, but malformed capture
    # metadata remains a hard input error.
    _intrinsics_matrix(metadata)
    _camera_to_world(metadata)
    first_queries = np.asarray(frames[0].query_points_yx, dtype=np.float32).reshape(
        -1, 2
    )

    mask_frames: list[dict[str, np.ndarray]] = []
    tracks: list[np.ndarray] = []
    visibility: list[np.ndarray] = []
    pcd_points: list[np.ndarray] = []
    pcd_colors: list[np.ndarray] = []
    source_frame_indices: list[int] = []

    for frame in list(frames) + list(lookahead_frames):
        queries = np.asarray(frame.query_points_yx, dtype=np.float32).reshape(-1, 2)
        if queries.shape != first_queries.shape or not np.allclose(
            queries, first_queries
        ):
            raise ValueError(
                "prepared data_process frames in one chunk must share query_points_yx"
            )
        mask_frames.append(
            strict.normalize_processed_mask_frame(frame.processed_mask_frame)
        )
        tracks.append(
            np.ascontiguousarray(frame.tracks_yx, dtype=np.float32).reshape(-1, 2)
        )
        visibility.append(
            np.ascontiguousarray(frame.visibility, dtype=bool).reshape(-1)
        )
        pcd_points.append(np.ascontiguousarray(frame.pcd_points, dtype=np.float32))
        pcd_colors.append(np.ascontiguousarray(frame.pcd_colors, dtype=np.uint8))

    for frame in frames:
        source_frame_indices.append(
            int(
                frame.source_frame_index
                if frame.source_frame_index is not None
                else frame.seq
            )
        )

    tracks_yx = np.stack(tracks, axis=0)
    tracker_visibility = np.stack(visibility, axis=0)
    pcd_points_arr = np.stack(pcd_points, axis=0)
    pcd_colors_arr = np.stack(pcd_colors, axis=0)
    # Offline parity with data_process_sam3d/data_process_pcd.py:L84-L149,
    # data_process_sam3d/data_process_mask.py:L42-L152, and
    # data_process_sam3d/data_process_track.py:L37-L135. Prepared frames already
    # contain the PCD and mask products, so this block performs the corresponding
    # track classification/lift step.
    track_input = _track_input_with_session_query_schema(
        tracks_yx=tracks_yx,
        visibility=tracker_visibility,
        mask_frames=mask_frames,
        pcd_points=pcd_points_arr,
        pcd_colors=pcd_colors_arr,
        session_query_schema=session_query_schema,
    )
    # design_spec.md state machine: origin motion consistency, frozen chunk-0
    # identity, per-frame temporary_invalid, and rigid-registration recovery.
    runtime = tracking_runtime if tracking_runtime is not None else tracking.TrackingRuntime()
    track_process = runtime.process_window(
        track_input,
        surface_points=surface_points,
        interior_points=interior_points,
        lookahead_frames=len(lookahead_frames),
    )

    return ChunkDataWindow(
        track_process_data=track_process,
        surface_points=surface_points,
        interior_points=interior_points,
        fps=int(fps),
        serial_number=serial_number,
        depth_backend=str(
            metadata.get("depth_backend") or metadata.get("depth_source", "")
        ),
        depth_source_internal=str(
            metadata.get("depth_source_internal")
            or metadata.get("depth_source")
            or metadata.get("depth_backend", "")
        ),
        chunk_index=int(chunk_index),
        source_frame_indices=source_frame_indices,
    )
