"""Lossless pipeline plumbing: StageStats, OrderedPacketQueue, SameSeqPairer."""
from __future__ import annotations

from demo_v6_2.mdp_constants import *  # noqa: F401,F403
from demo_v6_2.mdp_packets import PairedBuildResult

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


@dataclass(frozen=True)
class OrderedQueueStats:
    name: str
    size: int
    max_size: int
    last_put_seq: int
    last_get_seq: int
    closed: bool


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

    def put(self, packet: PacketT) -> int:
        """Return the put."""
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
            return len(self._items)

    def wait_for_capacity(self, *, stop_event: threading.Event, timeout_s: float = 0.05) -> bool:
        """Wait for for capacity."""
        with self._condition:
            while not stop_event.is_set():
                if self._closed:
                    raise LosslessPipelineError(f"{self.name} queue is closed")
                if len(self._items) < self.max_backlog_frames:
                    return True
                self._condition.wait(timeout=float(timeout_s))
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
            while len(self._items) >= self.max_backlog_frames:
                if stop_event.is_set():
                    return 0
                if self._closed:
                    raise LosslessPipelineError(f"{self.name} queue is closed")
                self._condition.wait(timeout=float(timeout_s))
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

    def get_nowait(self) -> PacketT | None:
        """Return the get nowait."""
        with self._condition:
            if not self._items:
                return None
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

    def reset(self) -> None:
        """Reset OrderedPacketQueue."""
        with self._condition:
            self._items.clear()
            self._last_put_seq = -1
            self._last_get_seq = -1
            self._closed = False
            self._max_size_seen = 0
            self._condition.notify_all()

    @property
    def stats(self) -> OrderedQueueStats:
        """Return the stats."""
        with self._condition:
            return OrderedQueueStats(
                name=self.name,
                size=len(self._items),
                max_size=int(self._max_size_seen),
                last_put_seq=int(self._last_put_seq),
                last_get_seq=int(self._last_get_seq),
                closed=bool(self._closed),
            )

    def latest_seq(self) -> int:
        """Return the latest seq."""
        with self._condition:
            return int(self._last_put_seq)

    def pending_count(self) -> int:
        """Return the pending count."""
        with self._condition:
            return len(self._items)

    def is_closed_and_empty(self) -> bool:
        """Return whether closed and empty."""
        with self._condition:
            return bool(self._closed and not self._items)


@dataclass(frozen=True)
class PairerStats:
    expected_seq: int
    pending_pcd: int
    pending_tracker: int
    emitted_seq: int
    pcd_closed: bool
    tracker_closed: bool


class SameSeqPairer:
    def __init__(self, *, max_backlog_frames: int) -> None:
        """Initialize SameSeqPairer."""
        self.max_backlog_frames = max(1, int(max_backlog_frames))
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._pending_pcd: dict[int, PcdBuildResult] = {}
        self._pending_tracker: dict[int, TrackerMarkerPacket] = {}
        self._expected_seq = 0
        self._emitted_seq = -1
        self._pcd_closed = False
        self._tracker_closed = False

    def reset(self) -> None:
        """Reset SameSeqPairer."""
        with self._condition:
            self._pending_pcd.clear()
            self._pending_tracker.clear()
            self._expected_seq = 0
            self._emitted_seq = -1
            self._pcd_closed = False
            self._tracker_closed = False
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
                    return True
                self._condition.wait(timeout=float(timeout_s))
            return False

    def add_pcd_result(self, result: PcdBuildResult) -> list[PairedBuildResult]:
        """Add PCD result."""
        seq = int(result.packet.seq)
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

    @property
    def stats(self) -> PairerStats:
        """Return the stats."""
        with self._condition:
            return PairerStats(
                expected_seq=int(self._expected_seq),
                pending_pcd=len(self._pending_pcd),
                pending_tracker=len(self._pending_tracker),
                emitted_seq=int(self._emitted_seq),
                pcd_closed=bool(self._pcd_closed),
                tracker_closed=bool(self._tracker_closed),
            )

    def _flush_ready_locked(self) -> list[PairedBuildResult]:
        """Return the flush ready locked."""
        pairs: list[PairedBuildResult] = []
        while self._expected_seq in self._pending_pcd and self._expected_seq in self._pending_tracker:
            seq = int(self._expected_seq)
            pcd_result = self._pending_pcd.pop(seq)
            tracker_packet = self._pending_tracker.pop(seq)
            pairs.append(PairedBuildResult(seq=seq, pcd_result=pcd_result, tracker_packet=tracker_packet))
            self._emitted_seq = seq
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


__all__ = [
    "StageStats",
    "PacketT",
    "LosslessPipelineError",
    "OrderedQueueStats",
    "OrderedPacketQueue",
    "PairerStats",
    "SameSeqPairer",
]
