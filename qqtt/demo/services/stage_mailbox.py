from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Generic, TypeVar


T = TypeVar("T")


STAGE_MAILBOX_POLICY_LATEST_ONLY = "latest-only"
STAGE_MAILBOX_POLICIES = (STAGE_MAILBOX_POLICY_LATEST_ONLY,)


@dataclass(frozen=True)
class StageMailboxSnapshot:
    published: int
    accepted: int
    completed: int
    pending_replaced: int
    active_present: bool
    pending_present: bool


class LatestOnlyStageMailbox(Generic[T]):
    """Latest-wins stage mailbox that never replaces the active bundle."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: T | None = None
        self._pending: T | None = None
        self.published = 0
        self.accepted = 0
        self.completed = 0
        self.pending_replaced = 0

    def publish_latest(self, item: T) -> int:
        with self._lock:
            replaced = int(self._pending is not None)
            self._pending = item
            self.published += 1
            self.pending_replaced += replaced
            return replaced

    def take_next(self) -> T | None:
        with self._lock:
            if self._active is not None or self._pending is None:
                return None
            self._active = self._pending
            self._pending = None
            self.accepted += 1
            return self._active

    def complete_active(self, item: T | None = None) -> T | None:
        with self._lock:
            active = self._active
            if active is None:
                return None
            if item is not None and item is not active:
                return None
            self._active = None
            self.completed += 1
            return active

    def active(self) -> T | None:
        with self._lock:
            return self._active

    def pending(self) -> T | None:
        with self._lock:
            return self._pending

    def snapshot(self) -> dict[str, int | bool]:
        with self._lock:
            snap = StageMailboxSnapshot(
                published=int(self.published),
                accepted=int(self.accepted),
                completed=int(self.completed),
                pending_replaced=int(self.pending_replaced),
                active_present=bool(self._active is not None),
                pending_present=bool(self._pending is not None),
            )
        return {
            "published": snap.published,
            "accepted": snap.accepted,
            "completed": snap.completed,
            "pending_replaced": snap.pending_replaced,
            "active_present": snap.active_present,
            "pending_present": snap.pending_present,
        }


__all__ = [
    "LatestOnlyStageMailbox",
    "STAGE_MAILBOX_POLICIES",
    "STAGE_MAILBOX_POLICY_LATEST_ONLY",
]
