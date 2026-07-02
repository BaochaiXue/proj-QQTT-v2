"""Latest-wins buffers and timing helpers shared by Demo v5.1 realtime threads."""

from __future__ import annotations

import threading
import time
from typing import Generic, TypeVar

T = TypeVar("T")


def elapsed_ms(start_s: float, end_s: float | None = None) -> float:
    return float(((time.perf_counter() if end_s is None else end_s) - start_s) * 1000.0)


def packet_seq(packet: object) -> int:
    try:
        return int(getattr(packet, "seq"))
    except AttributeError as exc:
        raise TypeError("latest-slot packets must expose an integer seq attribute") from exc


class LatestSlot(Generic[T]):
    """Thread-safe single-slot latest-wins buffer."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._packet: T | None = None
        self._last_taken_seq_total = -1
        self._last_taken_seq_window = -1
        self._dropped = 0
        self._dropped_total = 0

    def put(self, packet: T) -> int:
        seq = packet_seq(packet)
        with self._lock:
            if self._packet is not None:
                current_seq = packet_seq(self._packet)
                if current_seq > self._last_taken_seq_total:
                    self._dropped_total += max(1, seq - current_seq)
                if current_seq > self._last_taken_seq_window:
                    self._dropped += max(1, seq - current_seq)
            self._packet = packet
            return self._dropped

    def get_latest_after(self, last_seq: int) -> T | None:
        with self._lock:
            if self._packet is None:
                return None
            seq = packet_seq(self._packet)
            if seq <= last_seq:
                return None
            self._last_taken_seq_total = seq
            self._last_taken_seq_window = seq
            return self._packet

    def latest_seq(self) -> int:
        with self._lock:
            if self._packet is None:
                return -1
            return packet_seq(self._packet)

    def reset_dropped_count(self) -> None:
        with self._lock:
            self._dropped = 0
            if self._packet is not None:
                self._last_taken_seq_window = max(self._last_taken_seq_window, packet_seq(self._packet))

    @property
    def dropped_count(self) -> int:
        with self._lock:
            return self._dropped

    @property
    def total_dropped_count(self) -> int:
        with self._lock:
            return self._dropped_total


class CoalescedPostGate:
    """Allow at most one queued GUI post callback at a time."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending = False

    def try_mark_pending(self) -> bool:
        with self._lock:
            if self._pending:
                return False
            self._pending = True
            return True

    def mark_done(self) -> None:
        with self._lock:
            self._pending = False

    @property
    def pending(self) -> bool:
        with self._lock:
            return self._pending
