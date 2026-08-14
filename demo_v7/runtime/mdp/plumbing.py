"""Lossless pipeline plumbing: StageStats, OrderedPacketQueue, SameSeqPairer,
LosslessPipeline, FatalErrorLatch."""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import TYPE_CHECKING, Callable, Generic, TypeVar

from demo_v7.runtime.shape_prior import warmup as shape_prior_warmup
from demo_v7.runtime.mdp.packets import FatalWorkerError, PairedBuildResult
from demo_v7.runtime.pipeline_status import STAGE_FATAL
from demo_v7.runtime.tracking import CONTROLLER_FINAL_COUNT
from demo_v7.runtime.utils.concurrency import packet_seq as _packet_seq

if TYPE_CHECKING:
    from demo_v7.runtime.mdp.packets import (
        FramePacket,
        MaskPacket,
        PcdBuildResult,
        ProcessedFramePacket,
        TrackerMarkerPacket,
    )
    from demo_v7.runtime.pipeline_status import PipelineStatusWriter

class StageStats:
    def __init__(self, window_s: float = 1.0) -> None:
        """Initialize StageStats."""
        self.window_s = float(window_s)
        self._lock = threading.Lock()
        self._times: deque[float] = deque()

    def record(self, now_s: float | None = None) -> None:
        """Record StageStats."""
        now = time.perf_counter() if now_s is None else float(now_s)
        with self._lock:
            self._times.append(now)
            cutoff = now - self.window_s
            while len(self._times) > 1 and self._times[0] < cutoff:
                self._times.popleft()

    @property
    def fps(self) -> float:
        """Return the FPS."""
        with self._lock:
            if len(self._times) < 2:
                return 0.0
            elapsed = self._times[-1] - self._times[0]
            if elapsed <= 0:
                return 0.0
            return float((len(self._times) - 1) / elapsed)


PacketT = TypeVar("PacketT")


class LosslessPipelineError(RuntimeError):
    """Fatal contract violation in the lossless Demo 3.x pipeline."""


class OrderedPacketQueue(Generic[PacketT]):
    """Bounded FIFO packet queue that rejects gaps and silent overwrites."""

    def __init__(self, *, name: str, max_backlog_frames: int) -> None:
        """Initialize OrderedPacketQueue."""
        self.name = str(name)
        self.max_backlog_frames = max(1, int(max_backlog_frames))
        self._condition = threading.Condition()
        self._items: deque[PacketT] = deque()
        self._last_put_seq = -1
        self._last_get_seq = -1
        self._closed = False
        self._max_size_seen = 0
        self._blocked_s = 0.0

    def put(self, packet: PacketT) -> None:
        """Enqueue one packet; raises on seq gaps, overflow, or a closed queue."""
        seq = int(_packet_seq(packet))
        with self._condition:
            if self._closed:
                raise LosslessPipelineError(f"{self.name} queue is closed")
            expected = self._last_put_seq + 1
            if seq != expected:
                raise LosslessPipelineError(
                    f"{self.name} queue expected seq {expected}, got {seq}"
                )
            if len(self._items) >= self.max_backlog_frames:
                raise LosslessPipelineError(
                    "lossless input FPS backlog exceeded "
                    f"stage={self.name} queue_len={len(self._items) + 1} "
                    f"max={self.max_backlog_frames} expected_seq={self._last_get_seq + 1} "
                    f"latest_seq={seq}"
                )
            self._items.append(packet)
            self._last_put_seq = seq
            self._max_size_seen = max(self._max_size_seen, len(self._items))
            self._condition.notify_all()

    def wait_for_capacity(self, *, stop_event: threading.Event, timeout_s: float = 0.05) -> bool:
        """Wait for for capacity."""
        with self._condition:
            waited_from: float | None = None
            while not stop_event.is_set():
                if self._closed:
                    raise LosslessPipelineError(f"{self.name} queue is closed")
                if len(self._items) < self.max_backlog_frames:
                    if waited_from is not None:
                        self._blocked_s += time.perf_counter() - waited_from
                    return True
                if waited_from is None:
                    waited_from = time.perf_counter()
                self._condition.wait(timeout=float(timeout_s))
            if waited_from is not None:
                self._blocked_s += time.perf_counter() - waited_from
            return False

    def put_wait(self, packet: PacketT, *, stop_event: threading.Event, timeout_s: float = 0.05) -> int:
        """Return the put wait."""
        seq = int(_packet_seq(packet))
        with self._condition:
            if self._closed:
                raise LosslessPipelineError(f"{self.name} queue is closed")
            expected = self._last_put_seq + 1
            if seq != expected:
                raise LosslessPipelineError(
                    f"{self.name} queue expected seq {expected}, got {seq}"
                )
            waited_from: float | None = None
            while len(self._items) >= self.max_backlog_frames:
                if stop_event.is_set():
                    return 0
                if self._closed:
                    raise LosslessPipelineError(f"{self.name} queue is closed")
                if waited_from is None:
                    waited_from = time.perf_counter()
                self._condition.wait(timeout=float(timeout_s))
            if waited_from is not None:
                self._blocked_s += time.perf_counter() - waited_from
            self._items.append(packet)
            self._last_put_seq = seq
            self._max_size_seen = max(self._max_size_seen, len(self._items))
            self._condition.notify_all()
            return len(self._items)

    def get(self, *, stop_event: threading.Event, timeout_s: float = 0.05) -> PacketT | None:
        """Return the get."""
        with self._condition:
            while not self._items:
                if self._closed or stop_event.is_set():
                    return None
                self._condition.wait(timeout=float(timeout_s))
            packet = self._items.popleft()
            seq = int(_packet_seq(packet))
            expected = self._last_get_seq + 1
            if seq != expected:
                raise LosslessPipelineError(
                    f"{self.name} queue consumer expected seq {expected}, got {seq}"
                )
            self._last_get_seq = seq
            self._condition.notify_all()
            return packet

    def close(self) -> None:
        """Close OrderedPacketQueue."""
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def telemetry(self) -> dict[str, int | float]:
        """Queue-health snapshot: depth, high-water mark, seqs, blocked time."""
        with self._condition:
            return {
                "len": len(self._items),
                "max_seen": int(self._max_size_seen),
                "put_seq": int(self._last_put_seq),
                "get_seq": int(self._last_get_seq),
                "blocked_s": round(self._blocked_s, 3),
            }

    def reset(self) -> None:
        """Reset OrderedPacketQueue."""
        with self._condition:
            self._items.clear()
            self._last_put_seq = -1
            self._last_get_seq = -1
            self._closed = False
            self._max_size_seen = 0
            self._blocked_s = 0.0
            self._condition.notify_all()


class SameSeqPairer:
    def __init__(self, *, max_backlog_frames: int) -> None:
        """Initialize SameSeqPairer."""
        self.max_backlog_frames = max(1, int(max_backlog_frames))
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._pending_pcd: dict[int, PcdBuildResult] = {}
        self._pending_tracker: dict[int, TrackerMarkerPacket] = {}
        self._expected_seq = 0
        self._pcd_closed = False
        self._tracker_closed = False
        self._max_pending_seen = 0
        self._blocked_s = 0.0

    def reset(self) -> None:
        """Reset SameSeqPairer."""
        with self._condition:
            self._pending_pcd.clear()
            self._pending_tracker.clear()
            self._expected_seq = 0
            self._pcd_closed = False
            self._tracker_closed = False
            self._max_pending_seen = 0
            self._blocked_s = 0.0
            self._condition.notify_all()

    def wait_for_side_capacity(
        self,
        side: str,
        *,
        stop_event: threading.Event,
        timeout_s: float = 0.05,
    ) -> bool:
        """Wait for for side capacity."""
        side_name = str(side)
        if side_name not in {"pcd", "tracker"}:
            raise ValueError("side must be 'pcd' or 'tracker'")
        with self._condition:
            waited_from: float | None = None
            while not stop_event.is_set():
                if side_name == "pcd":
                    if self._pcd_closed:
                        raise LosslessPipelineError("same-seq pairer PCD side is closed")
                    pending = len(self._pending_pcd)
                else:
                    if self._tracker_closed:
                        raise LosslessPipelineError("same-seq pairer tracker side is closed")
                    pending = len(self._pending_tracker)
                if pending < self.max_backlog_frames:
                    if waited_from is not None:
                        self._blocked_s += time.perf_counter() - waited_from
                    return True
                if waited_from is None:
                    waited_from = time.perf_counter()
                self._condition.wait(timeout=float(timeout_s))
            if waited_from is not None:
                self._blocked_s += time.perf_counter() - waited_from
            return False

    def add_pcd_result(self, result: PcdBuildResult) -> list[PairedBuildResult]:
        """Add PCD result."""
        seq = int(result.pcd_packet.seq)
        with self._condition:
            if self._pcd_closed:
                raise LosslessPipelineError("same-seq pairer PCD side is closed")
            if seq < self._expected_seq:
                raise LosslessPipelineError(
                    f"same-seq pairer received stale PCD seq {seq}, expected {self._expected_seq}"
                )
            if seq in self._pending_pcd:
                raise LosslessPipelineError(f"same-seq pairer duplicate PCD seq {seq}")
            self._pending_pcd[seq] = result
            self._max_pending_seen = max(
                self._max_pending_seen, len(self._pending_pcd), len(self._pending_tracker)
            )
            self._check_backlog_locked()
            pairs = self._flush_ready_locked()
            self._condition.notify_all()
            return pairs

    def add_tracker_packet(self, packet: TrackerMarkerPacket) -> list[PairedBuildResult]:
        """Add tracker packet."""
        seq = int(packet.seq)
        with self._condition:
            if self._tracker_closed:
                raise LosslessPipelineError("same-seq pairer tracker side is closed")
            if seq < self._expected_seq:
                raise LosslessPipelineError(
                    f"same-seq pairer received stale tracker seq {seq}, expected {self._expected_seq}"
                )
            if seq in self._pending_tracker:
                raise LosslessPipelineError(f"same-seq pairer duplicate tracker seq {seq}")
            self._pending_tracker[seq] = packet
            self._max_pending_seen = max(
                self._max_pending_seen, len(self._pending_pcd), len(self._pending_tracker)
            )
            self._check_backlog_locked()
            pairs = self._flush_ready_locked()
            self._condition.notify_all()
            return pairs

    def close_pcd(self) -> list[PairedBuildResult]:
        """Close PCD."""
        with self._condition:
            self._pcd_closed = True
            pairs = self._flush_ready_locked()
            self._check_closed_locked()
            self._condition.notify_all()
            return pairs

    def close_tracker(self) -> list[PairedBuildResult]:
        """Close tracker."""
        with self._condition:
            self._tracker_closed = True
            pairs = self._flush_ready_locked()
            self._check_closed_locked()
            self._condition.notify_all()
            return pairs

    @property
    def done(self) -> bool:
        """Return the done."""
        with self._condition:
            return (
                self._pcd_closed
                and self._tracker_closed
                and not self._pending_pcd
                and not self._pending_tracker
            )

    def telemetry(self) -> dict[str, int | float]:
        """Pairer-health snapshot: pending sides, high-water mark, blocked time."""
        with self._condition:
            return {
                "pending_pcd": len(self._pending_pcd),
                "pending_tracker": len(self._pending_tracker),
                "max_pending_seen": int(self._max_pending_seen),
                "expected_seq": int(self._expected_seq),
                "blocked_s": round(self._blocked_s, 3),
            }

    def _flush_ready_locked(self) -> list[PairedBuildResult]:
        """Return the flush ready locked."""
        pairs: list[PairedBuildResult] = []
        while self._expected_seq in self._pending_pcd and self._expected_seq in self._pending_tracker:
            seq = int(self._expected_seq)
            pcd_result = self._pending_pcd.pop(seq)
            tracker_packet = self._pending_tracker.pop(seq)
            pairs.append(PairedBuildResult(seq=seq, pcd_result=pcd_result, tracker_packet=tracker_packet))
            self._expected_seq += 1
        return pairs

    def _check_backlog_locked(self) -> None:
        """Check backlog locked."""
        if len(self._pending_pcd) > self.max_backlog_frames or len(self._pending_tracker) > self.max_backlog_frames:
            raise LosslessPipelineError(
                "lossless input FPS backlog exceeded "
                f"stage=pairer expected_seq={self._expected_seq} "
                f"pending_pcd={len(self._pending_pcd)} pending_tracker={len(self._pending_tracker)} "
                f"max={self.max_backlog_frames}"
            )

    def _check_closed_locked(self) -> None:
        """Check closed locked."""
        if not (self._pcd_closed and self._tracker_closed):
            return
        if self._pending_pcd or self._pending_tracker:
            raise LosslessPipelineError(
                "same-seq pairer closed with unmatched packets "
                f"expected_seq={self._expected_seq} "
                f"pending_pcd={sorted(self._pending_pcd)} "
                f"pending_tracker={sorted(self._pending_tracker)}"
            )


class LosslessPipeline:
    """Strict same-seq lossless pipeline state and its ordering protocol.

    Owns the four gap-free queues (frame -> raw mask -> processed frame ->
    paired output), the same-seq pairer, and the ordered-publish cursor.
    Invariants enforced here:
    - every queue carries contiguous seqs (OrderedPacketQueue rejects gaps);
    - pairer sides mutate only under ``pairer_lock``, and completed pairs are
      enqueued under that same lock so pair order matches pairing order;
    - a pair for seq N is published only after N-1 (stale/skipped seq raises
      LosslessPipelineError);
    - ``first_pair_published`` releases the capture replay clock exactly once.
    """

    def __init__(self, *, max_backlog_frames: int) -> None:
        """Initialize LosslessPipeline."""
        self.max_backlog_frames = max(1, int(max_backlog_frames))
        self.frame_queue: OrderedPacketQueue[FramePacket] = OrderedPacketQueue(
            name="frame", max_backlog_frames=self.max_backlog_frames
        )
        self.mask_queue: OrderedPacketQueue[MaskPacket] = OrderedPacketQueue(
            name="raw-mask", max_backlog_frames=self.max_backlog_frames
        )
        self.processed_frame_queue: OrderedPacketQueue[ProcessedFramePacket] = (
            OrderedPacketQueue(
                name="processed-frame", max_backlog_frames=self.max_backlog_frames
            )
        )
        self.pair_output_queue: OrderedPacketQueue[PairedBuildResult] = (
            OrderedPacketQueue(
                name="pair-output", max_backlog_frames=self.max_backlog_frames
            )
        )
        self.pairer = SameSeqPairer(max_backlog_frames=self.max_backlog_frames)
        self._pairer_lock = threading.Lock()
        self._publish_condition = threading.Condition()
        self._next_publish_seq = 0
        self.processing_done = threading.Event()
        self.first_pair_published = threading.Event()

    def reset(self) -> None:
        """Reset every queue, the pairer, the publish cursor, and the events."""
        self.frame_queue.reset()
        self.mask_queue.reset()
        self.processed_frame_queue.reset()
        self.pair_output_queue.reset()
        self.pairer.reset()
        with self._publish_condition:
            self._next_publish_seq = 0
            self._publish_condition.notify_all()
        self.processing_done.clear()
        self.first_pair_published.clear()

    def close_queues(self) -> None:
        """Close every queue (teardown path)."""
        self.frame_queue.close()
        self.mask_queue.close()
        self.processed_frame_queue.close()
        self.pair_output_queue.close()

    # ---- capture side -----------------------------------------------------
    def submit_frame(self, packet: FramePacket, *, stop_event: threading.Event) -> bool:
        """Enqueue one capture frame; False when stopped while at capacity."""
        return self.frame_queue.put_wait(packet, stop_event=stop_event) > 0

    def finish_capture(self) -> None:
        """Close the frame queue to mark the capture side complete."""
        self.frame_queue.close()

    # ---- segmentation side ------------------------------------------------
    def submit_mask(self, packet: MaskPacket, *, stop_event: threading.Event) -> bool:
        """Enqueue one raw mask packet; False when stopped while at capacity."""
        if not self.mask_queue.wait_for_capacity(stop_event=stop_event):
            return False
        self.mask_queue.put(packet)
        return True

    # ---- pairer sides -----------------------------------------------------
    def submit_pcd_result(
        self, result: PcdBuildResult, *, stop_event: threading.Event
    ) -> bool:
        """Feed the PCD side of the pairer; False when stopped at capacity."""
        if not self.pairer.wait_for_side_capacity("pcd", stop_event=stop_event):
            return False
        with self._pairer_lock:
            self._enqueue_pairs(self.pairer.add_pcd_result(result))
        return True

    def close_pcd_side(self) -> None:
        """Close the PCD side, flushing any pairs it completes."""
        with self._pairer_lock:
            self._enqueue_pairs(self.pairer.close_pcd())
            self.maybe_close_pair_output()

    def submit_tracker_packet(
        self, packet: TrackerMarkerPacket, *, stop_event: threading.Event
    ) -> bool:
        """Feed the tracker side of the pairer; False when stopped at capacity."""
        if not self.pairer.wait_for_side_capacity("tracker", stop_event=stop_event):
            return False
        with self._pairer_lock:
            self._enqueue_pairs(self.pairer.add_tracker_packet(packet))
        return True

    def close_tracker_side(self) -> None:
        """Close the tracker side, flushing any pairs it completes."""
        with self._pairer_lock:
            self._enqueue_pairs(self.pairer.close_tracker())
            self.maybe_close_pair_output()

    def _enqueue_pairs(self, pairs: list[PairedBuildResult]) -> None:
        """Enqueue completed same-seq pairs for ordered publishing."""
        for pair in pairs:
            self.pair_output_queue.put(pair)

    # ---- ordered publish side ----------------------------------------------
    def wait_publish_turn(self, seq: int, *, stop_event: threading.Event) -> bool:
        """Block until ``seq`` is next to publish; False when stopped first."""
        with self._publish_condition:
            while seq != self._next_publish_seq:
                if seq < self._next_publish_seq:
                    raise LosslessPipelineError(
                        f"lossless publish received stale seq {seq}, expected "
                        f"{self._next_publish_seq}"
                    )
                if stop_event.is_set():
                    return False
                self._publish_condition.wait(timeout=0.05)
        return True

    def finish_publish_turn(self, seq: int) -> None:
        """Advance the publish cursor past ``seq``."""
        with self._publish_condition:
            if seq != self._next_publish_seq:
                raise LosslessPipelineError(
                    f"lossless publish expected seq {self._next_publish_seq}, got {seq}"
                )
            self._next_publish_seq += 1
            self._publish_condition.notify_all()

    def telemetry(self) -> dict[str, object]:
        """Per-stage queue-health snapshot for the [queue-telemetry] stream."""
        with self._publish_condition:
            next_publish_seq = int(self._next_publish_seq)
        return {
            "frame": self.frame_queue.telemetry(),
            "mask": self.mask_queue.telemetry(),
            "processed": self.processed_frame_queue.telemetry(),
            "pair_output": self.pair_output_queue.telemetry(),
            "pairer": self.pairer.telemetry(),
            "next_publish_seq": next_publish_seq,
        }

    def maybe_close_pair_output(self) -> None:
        """Close the pair-output queue once the pairer fully drained."""
        if self.pairer.done and not self.processing_done.is_set():
            self.pair_output_queue.close()

    def finish_output(self) -> None:
        """Mark the ordered-publish side complete (ends the run loop)."""
        if not self.processing_done.is_set():
            self.processing_done.set()


class FatalErrorLatch:
    """First-error-wins fatal latch shared by every pipeline worker.

    The first recorded error is printed, surfaced on the live status band,
    and sets ``stop_event``; later errors return the original without side
    effects, so teardown noise never masks the root cause.
    """

    def __init__(
        self, *, status: PipelineStatusWriter, stop_event: threading.Event
    ) -> None:
        """Initialize FatalErrorLatch."""
        self._status = status
        self._stop_event = stop_event
        self._lock = threading.Lock()
        self._fatal: FatalWorkerError | None = None

    def record(self, stage: str, exc: BaseException) -> FatalWorkerError:
        """Record the first fatal worker error and set stop_event."""
        fatal = FatalWorkerError(
            stage=str(stage), exc_type=type(exc).__name__, message=str(exc)
        )
        first = False
        with self._lock:
            if self._fatal is None:
                self._fatal = fatal
                first = True
            else:
                fatal = self._fatal
        if first:
            print(
                f"[FATAL] {fatal.stage} failed: {fatal.exc_type}: {fatal.message}",
                flush=True,
            )
            self._status.emit(
                STAGE_FATAL,
                f"{fatal.stage}: {fatal.message}",
                ok=False,
                exc_type=fatal.exc_type,
            )
            self._stop_event.set()
        return fatal

    def snapshot(self) -> FatalWorkerError | None:
        """Return the first recorded fatal error, if any."""
        with self._lock:
            return self._fatal


def _formal_timeline_rows_gated(
    *, warmup_anchor_written: bool, shape_prior_status: str
) -> bool:
    """design_spec.md warmup/formal timeline split.

    Rows always write until a chunk-ready warmup anchor row has landed (live
    RealSense can emit an invalid strict pair before color-aligned PCD is
    ready; the bridge trims those, so they must not consume the frame-0
    slot). After the anchor, frames processed while the shape prior is still
    computing stay OUT of the formal final_data chunk timeline (they keep
    feeding the trackers and the left preview, which pace by
    input_frames.jsonl). The first frame after the prior is ready becomes
    output frame 1, stitched directly after warmup frame 0 under the
    operator hold-still convention. Terminal states (ready/disabled/failed)
    lift the gate — a failed prior must surface through the chunk bridge's
    shape-prior error path instead of silently stalling the row stream.
    """
    if not warmup_anchor_written:
        return False
    return str(shape_prior_status) in (
        shape_prior_warmup.STATUS_PENDING,
        shape_prior_warmup.STATUS_RUNNING,
    )


class FormalTimelineGate:
    """Warm-up -> formal-timeline transition state for headless products.

    Owns the gate that withholds post-warmup rows (PCD row + tracker sidecar)
    while the shape prior is still computing. Invariants enforced here:
    - the --shape-prior-timeout-ms expiry latches one-way, with one WARN;
    - first-ungated-row metadata is produced exactly once;
    - the chunk-ready warmup anchor is detected exactly once;
    - a run that ends while rows are still gated is a fatal error.
    """

    def __init__(
        self, *, shape_prior_status: Callable[[], str], timeout_ms: int
    ) -> None:
        """Initialize FormalTimelineGate."""
        self._shape_prior_status = shape_prior_status
        self._timeout_ms = int(timeout_ms)
        self.shape_prior_result_written = False
        self.anchor_row_written = False
        self.gated_frame_count = 0
        self._metadata_written = False
        self._started_s: float | None = None
        self._expired = False

    def rows_gated(self) -> bool:
        """True while post-warmup frames must stay out of the chunk timeline.

        The gate carries its own deadline: --shape-prior-timeout-ms bounds how
        long formal rows may be withheld. On expiry rows resume so the chunk
        bridge's shape-prior wait/failure path reports loudly, instead of the
        row stream stalling silently on a hung prior.
        """
        if self._expired:
            return False
        if not _formal_timeline_rows_gated(
            warmup_anchor_written=self.anchor_row_written,
            shape_prior_status=self._shape_prior_status(),
        ):
            return False
        now_s = time.perf_counter()
        if self._started_s is None:
            self._started_s = now_s
        if self._timeout_ms > 0 and (now_s - self._started_s) * 1000.0 >= float(
            self._timeout_ms
        ):
            self._expired = True
            print(
                "[WARN] shape prior still not ready after --shape-prior-timeout-ms="
                f"{self._timeout_ms}; resuming formal chunk rows so the chunk bridge "
                "can surface the shape-prior wait/failure loudly.",
                flush=True,
            )
            return False
        return True

    def note_gated_row(self) -> None:
        """Count one frame withheld from the formal timeline."""
        self.gated_frame_count += 1

    def first_ungated_row_metadata(self, seq: int) -> dict[str, int] | None:
        """Return the gate-summary metadata exactly once, at the first ungated row."""
        if not self.gated_frame_count or self._metadata_written:
            return None
        self._metadata_written = True
        return {
            "formal_timeline_gated_frame_count": int(self.gated_frame_count),
            "formal_timeline_start_seq": int(seq),
        }

    def note_anchor_row(
        self, *, controller_point_count: int, object_point_count: int
    ) -> None:
        """Latch the chunk-ready warmup anchor once a row qualifies."""
        if not self.anchor_row_written:
            self.anchor_row_written = (
                int(controller_point_count) >= CONTROLLER_FINAL_COUNT
                and int(object_point_count) > 0
            )

    def incomplete_run_error(self) -> str | None:
        """Error message when the run ended with rows still gated, else None."""
        if not self.gated_frame_count or self._metadata_written:
            return None
        return (
            "run ended while formal chunk rows were still gated on "
            f"the shape prior ({self.gated_frame_count} frames "
            "withheld); the capture has no formal timeline and cannot be "
            "chunked."
        )


class StageStatsBoard:
    """Per-stage throughput samples and the row-schema stage_fps snapshot."""

    _STAGES = ("capture", "seg", "pcd", "tracker")

    def __init__(self) -> None:
        """Initialize StageStatsBoard."""
        self._stats = {name: StageStats() for name in self._STAGES}

    def record(self, stage: str, now_s: float | None = None) -> None:
        """Record one completed item for ``stage``."""
        self._stats[stage].record(now_s)

    def fps_snapshot(self) -> dict[str, float]:
        """Return the stage_fps mapping serialized into headless rows."""
        return {f"{name}_fps": float(self._stats[name].fps) for name in self._STAGES}


__all__ = [
    "StageStats",
    "StageStatsBoard",
    "LosslessPipelineError",
    "OrderedPacketQueue",
    "SameSeqPairer",
    "LosslessPipeline",
    "FatalErrorLatch",
    "FormalTimelineGate",
]
