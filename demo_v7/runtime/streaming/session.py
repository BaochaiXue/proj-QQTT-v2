"""Convert Demo v6.2 headless capture rows into realtime chunks.

The camera process appends ``frames.jsonl`` and prepared per-frame NPZ payloads.
This bridge tails that stream as one ``ChunkStreamSession``, waits for the shape
prior only at chunk materialize time, and publishes online chunk payloads plus
the static final_data view.
"""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Callable, Mapping

import numpy as np

from demo_v7.runtime.streaming.data_output import ChunkDataWriter
from demo_v7.runtime.streaming.online_frame_archive import OnlineFrameArchive
from demo_v7.runtime.streaming import asap
from demo_v7.runtime import phystwin_strict_product as strict
from demo_v7.runtime import tracking

from demo_v7.runtime.streaming.jsonl_tail import (
    _read_jsonl_from_offset,
    _relative_wall_s,
)
from demo_v7.runtime.streaming.warmup_trim import WarmupStartFilter
from demo_v7.runtime.shape_prior.case import write_shape_prior_points_npz
from demo_v7.runtime.streaming.capture_meta import (
    _read_json_file_stable,
    _wait_for_asap_case_dir,
    _wait_for_shape_candidates,
)
from demo_v7.runtime.streaming.window_builder import _prepared_frame_from_row
from demo_v7.runtime.streaming.materialize import _materialize_and_commit_window


# ---------------------------------------------------------------------------
# Realtime capture tailing
# ---------------------------------------------------------------------------
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


class ChunkStreamSession:
    """One realtime chunk-streaming session over a live headless capture.

    The session owns every cross-window constant (capture location, cadence,
    the online writer/RGB-D archive, the session-lived tracking and ASAP
    runtimes, the frozen query schema) plus the mutable tail state (row
    buffers, the pending window, committed manifests). Helpers in
    ``materialize`` / ``window_builder`` / ``capture_meta``
    take the session instead of re-plumbing that state through keyword
    parameters.
    """

    def __init__(
        self,
        capture_dir: str | Path,
        *,
        base_path: str | Path,
        # Values with config/default.yaml knobs are required so this session
        # never carries a second copy of a config default.
        case_prefix: str,
        chunk_size: int,
        fps: int,
        max_chunks: int | None = None,
        capture_finished: Callable[[], bool],
        before_poll: Callable[[], None] | None = None,
        poll_interval_s: float,
        surface_points: np.ndarray | None = None,
        interior_points: np.ndarray | None = None,
        require_shape_prior: bool = False,
        shape_prior_wait_timeout_s: float,
        volume_sample_size_m: float,
        points_npz: str | Path | None = None,
        on_chunk_written: Callable[[dict[str, Any]], None] | None = None,
        online_case_name: str | None = None,
        asap_augment: bool = True,
        asap_mesh_path: str | Path | None = None,
    ) -> None:
        self.capture = Path(capture_dir)
        if int(chunk_size) <= 0:
            raise ValueError("chunk_size must be positive")
        self.case_prefix = str(case_prefix)
        self.chunk_size = int(chunk_size)
        self.fps = int(fps)
        self.max_chunks = None if max_chunks is None else int(max_chunks)
        self.capture_finished = capture_finished
        self.before_poll = before_poll
        self.poll_interval_s = float(poll_interval_s)
        self.surface_points = surface_points
        self.interior_points = interior_points
        self.require_shape_prior = bool(require_shape_prior)
        self.shape_prior_wait_timeout_s = float(shape_prior_wait_timeout_s)
        self.volume_sample_size_m = float(volume_sample_size_m)
        # Final structure points are frozen by the chunk-0 unified sampling;
        # points.npz (the downstream launch trigger) is written then.
        self.points_npz = None if points_npz is None else Path(points_npz)
        self.final_surface_points: np.ndarray | None = None
        self.final_interior_points: np.ndarray | None = None
        self.on_chunk_written = on_chunk_written
        metadata = _wait_for_metadata(
            self.capture,
            capture_finished=capture_finished,
            poll_interval_s=self.poll_interval_s,
        )
        self.serial_number = str(metadata.get("serial") or "").strip()
        if not self.serial_number:
            raise RuntimeError(
                "capture metadata is missing the single-camera serial"
            )
        self.frames_path = self.capture / "frames.jsonl"
        self.wall_time_origin_s = time.monotonic()
        self.warmup_start_filter = WarmupStartFilter()
        online_case = str(online_case_name or case_prefix)
        self.online_writer = ChunkDataWriter(
            base_path=base_path,
            case_name=online_case,
            chunk_size=self.chunk_size,
            num_frames_total=(
                None if self.max_chunks is None else self.max_chunks * self.chunk_size
            ),
        )
        self.frame_archive = OnlineFrameArchive(
            base_path=base_path,
            fps=self.fps,
        )
        # Seed calibrate.pkl/metadata.json (frame_num=0) as soon as capture
        # metadata is known: phystwin_shen downstream consumers start at
        # shape-prior-ready time and read the case dir before the first
        # chunk commits (design_spec_v6_1.md, downstream.mode).
        self.frame_archive.initialize_case(metadata)
        # One stateful tracking runtime spans the realtime session so chunk N+1
        # continues the identity frozen by chunk 0.
        self.tracking_runtime = tracking.TrackingRuntime(
            volume_sample_size=self.volume_sample_size_m
        )
        # design_spec_v6_1.md: session-lived ASAP augmenter; initializes from the
        # first materialized window (the aligned mesh exists only after shape-
        # prior warmup) and fails fast when final_mesh.glb is missing.
        self.asap_runtime = (
            asap.AsapRuntime(mesh_path_override=asap_mesh_path)
            if bool(asap_augment)
            else None
        )
        self.session_query_schema: dict[str, np.ndarray] = {}
        # Mutable tail state, advanced only by run().
        self.manifests: list[dict[str, Any]] = []
        self.next_row_idx = 0
        self.frames_offset = 0
        self.row_start = 0
        self.row_buffer: list[dict[str, Any]] = []
        self.prepared_buffer: list[strict.PreparedPhysTwinFrame] = []
        self.chunk_index = 0
        # A full window is held pending until its borrow row (the next window's
        # first row) arrives, so the published tail row carries a real motion
        # verdict. This trades one output frame of publish latency; steady-state
        # throughput is unchanged. At capture end the pending window flushes
        # without a borrow row (origin end-of-sequence tail semantics) — the
        # final full window is never dropped.
        self.pending_window: dict[str, Any] | None = None

    @property
    def warmup_skipped_rows(self) -> int:
        """Startup rows trimmed exactly once before the warmup filter resolved."""
        return int(self.warmup_start_filter.skipped_rows)

    def _chunk_target_reached(self) -> bool:
        return self.max_chunks is not None and len(self.manifests) >= self.max_chunks

    def _materialize_pending(
        self,
        pending: dict[str, Any],
        borrow_prepared: list[strict.PreparedPhysTwinFrame],
    ) -> dict[str, Any]:
        if self.final_surface_points is None:
            # Chunk 0: wait for the RAW candidate pools; the unified sampling
            # inside the identity freeze produces the final structure points.
            latest_metadata, shape_surface, shape_interior = (
                _wait_for_shape_candidates(self)
            )
        else:
            latest_metadata = _wait_for_metadata(
                self.capture,
                capture_finished=self.capture_finished,
                poll_interval_s=self.poll_interval_s,
            )
            shape_surface, shape_interior = None, None
        if (
            self.asap_runtime is not None
            and not self.asap_runtime.initialized
            and self.asap_runtime.mesh_path_override is None
        ):
            latest_metadata = _wait_for_asap_case_dir(self, latest_metadata)
        try:
            result = _materialize_and_commit_window(
                self,
                pending,
                borrow_prepared,
                metadata=latest_metadata,
                surface_points=shape_surface,
                interior_points=shape_interior,
            )
        except Exception:
            # design_spec.md failures (controller selection / recovery)
            # abort the stream; leave a terminal manifest instead of
            # "recording".
            self.online_writer.finish(status="failed")
            raise
        if self.final_surface_points is None:
            # Chunk 0 committed: freeze the unified structure points for the
            # session and publish the downstream launch trigger.
            surface, interior = self.tracking_runtime.frozen_structure_points()
            self.final_surface_points = surface
            self.final_interior_points = interior
            if self.require_shape_prior and self.points_npz is not None:
                write_shape_prior_points_npz(
                    self.points_npz,
                    surface_points=surface,
                    interior_points=interior,
                )
                print(
                    "[chunk-stream] unified structure points frozen "
                    f"(surface={int(surface.shape[0])} "
                    f"interior={int(interior.shape[0])}) -> {self.points_npz}",
                    flush=True,
                )
        return result

    def _commit_pending(
        self, borrow_prepared: list[strict.PreparedPhysTwinFrame]
    ) -> None:
        manifest = self._materialize_pending(self.pending_window, borrow_prepared)
        self.pending_window = None
        self.manifests.append(manifest)
        if self.on_chunk_written is not None:
            self.on_chunk_written(manifest)
        self.chunk_index += 1

    def run(self) -> list[dict[str, Any]]:
        """Tail the live headless capture and publish each closed window once
        its borrow row (the next window's first row) arrives; at capture end
        the final full window flushes without a borrow row."""
        while True:
            if self._chunk_target_reached():
                break
            if self.before_poll is not None:
                self.before_poll()
            rows, self.frames_offset = _read_jsonl_from_offset(
                self.frames_path, self.frames_offset
            )
            saw_new_rows = bool(rows)
            rows = self.warmup_start_filter.filter(
                rows,
                capture_finished=bool(self.capture_finished()),
            )
            for row in rows:
                prepared_row = _prepared_frame_from_row(self.capture, row)
                if self.pending_window is not None:
                    # This row is the pending window's borrow frame — and also
                    # the first row of the next window below.
                    self._commit_pending([prepared_row])
                    if self._chunk_target_reached():
                        break
                # A chunk closes strictly by row count. Shape-prior readiness and
                # final_data materialization are handled at borrow-row arrival so
                # source pacing remains tied to the camera/fake-camera stream.
                self.row_buffer.append(row)
                self.prepared_buffer.append(prepared_row)
                # Real-time RGB-D publication: this frame's color/depth land
                # on disk NOW (frame cadence), not at chunk commit. frame_num
                # in metadata.json still advances only after the owning chunk
                # commits (archive_chunk verifies these streamed files then).
                self.frame_archive.stream_frame(prepared_row)
                self.next_row_idx += 1
                if len(self.row_buffer) < self.chunk_size:
                    continue
                self.pending_window = {
                    "rows": self.row_buffer,
                    "prepared": self.prepared_buffer,
                    "chunk_index": self.chunk_index,
                    "row_start": self.row_start,
                    "row_end": self.next_row_idx,
                    "window_closed_wall_s": _relative_wall_s(self.wall_time_origin_s),
                }
                self.row_start = self.next_row_idx
                self.row_buffer = []
                self.prepared_buffer = []
            if self._chunk_target_reached():
                break
            if self.capture_finished() and not saw_new_rows:
                if self.pending_window is not None:
                    # Terminal flush: no successor row will arrive.
                    self._commit_pending([])
                break
            time.sleep(max(0.0, self.poll_interval_s))
        self._close()
        return self.manifests

    def _close(self) -> None:
        # Streamed frames past the last committed chunk (partial tail window /
        # max_chunks stop) have no owning chunk; drop them so the final tree
        # contains only committed frames and matches frame_num's contract.
        discarded = self.frame_archive.discard_streamed_tail()
        if discarded:
            print(
                f"[demo_v6_1] online_data: discarded {discarded} streamed "
                "frames past the last committed chunk"
            )
        self.online_writer.finish()


__all__ = [
    "ChunkStreamSession",
]
