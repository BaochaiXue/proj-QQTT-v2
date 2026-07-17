"""Latest-wins buffers and timing helpers shared by Demo v6.2 realtime threads."""

from __future__ import annotations

import threading
import time
from typing import Generic, TypeVar

T = TypeVar("T")


def elapsed_ms(start_s: float, end_s: float | None = None) -> float:
    """Compute elapsed ms."""
    return float(((time.perf_counter() if end_s is None else end_s) - start_s) * 1000.0)


def packet_seq(packet: object) -> int:
    """Return the packet seq."""
    try:
        return int(getattr(packet, "seq"))
    except AttributeError as exc:
        raise TypeError("latest-slot packets must expose an integer seq attribute") from exc


class LatestSlot(Generic[T]):
    """Thread-safe single-slot latest-wins buffer.

    Producers put(); the consumer polls get_latest_after() with the last seq it
    handled. Overwriting a packet the consumer never took counts as a drop.
    """

    def __init__(self) -> None:
        """Initialize LatestSlot."""
        self._lock = threading.Lock()
        self._packet: T | None = None
        self._last_taken_seq_window = -1
        self._dropped = 0

    def put(self, packet: T) -> int:
        """Store packet, count drops if the previous one was never taken; returns window drops."""
        seq = packet_seq(packet)
        with self._lock:
            if self._packet is not None:
                current_seq = packet_seq(self._packet)
                # seq gaps count as multiple drops (producer skipped frames upstream).
                if current_seq > self._last_taken_seq_window:
                    self._dropped += max(1, seq - current_seq)
            self._packet = packet
            return self._dropped

    def get_latest_after(self, last_seq: int) -> T | None:
        """Return the get latest after."""
        with self._lock:
            if self._packet is None:
                return None
            seq = packet_seq(self._packet)
            if seq <= last_seq:
                return None
            self._last_taken_seq_window = seq
            return self._packet

    def latest_seq(self) -> int:
        """Return the latest seq."""
        with self._lock:
            if self._packet is None:
                return -1
            return packet_seq(self._packet)

    @property
    def dropped_count(self) -> int:
        """Return the dropped count."""
        with self._lock:
            return self._dropped
