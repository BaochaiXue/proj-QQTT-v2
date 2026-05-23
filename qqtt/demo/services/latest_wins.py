from __future__ import annotations

from dataclasses import dataclass, field
import queue
import threading
from typing import Any, Generic, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class LatestWinsPublishInfo:
    replaced_count: int = 0
    replaced_items: tuple[Any, ...] = field(default_factory=tuple)


class LatestValueSlot(Generic[T]):
    """Thread-safe in-process latest-wins slot."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._item: T | None = None
        self.published = 0
        self.taken = 0
        self.replaced = 0

    def publish_latest(self, item: T) -> int:
        with self._lock:
            replaced = int(self._item is not None)
            self._item = item
            self.published += 1
            self.replaced += replaced
            return replaced

    def take_latest(self) -> T | None:
        with self._lock:
            item = self._item
            self._item = None
            if item is not None:
                self.taken += 1
            return item

    def peek_latest(self) -> T | None:
        with self._lock:
            return self._item

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "published": int(self.published),
                "taken": int(self.taken),
                "replaced": int(self.replaced),
                "pending": int(self._item is not None),
            }


class LatestWinsQueue:
    """CPU latest-wins queue wrapper for non-blocking process IPC."""

    def __init__(self, queue_obj: Any | None = None) -> None:
        self.queue = queue_obj if queue_obj is not None else queue.Queue(maxsize=1)
        self.published = 0
        self.taken = 0
        self.replaced = 0
        self.put_failures = 0

    def publish_latest(self, item: Any) -> int:
        return int(self.publish_latest_with_info(item).replaced_count)

    def publish_latest_with_info(self, item: Any) -> LatestWinsPublishInfo:
        replaced_items = self._drain_items()
        try:
            self.queue.put_nowait(item)
        except queue.Full:
            replaced_items.extend(self._drain_items())
            try:
                self.queue.put_nowait(item)
            except queue.Full:
                self.put_failures += 1
                return LatestWinsPublishInfo(
                    replaced_count=len(replaced_items),
                    replaced_items=tuple(replaced_items),
                )
        self.published += 1
        self.replaced += len(replaced_items)
        return LatestWinsPublishInfo(
            replaced_count=len(replaced_items),
            replaced_items=tuple(replaced_items),
        )

    def take_latest(self) -> Any | None:
        latest = None
        drained = 0
        while True:
            try:
                latest = self.queue.get_nowait()
                drained += 1
            except queue.Empty:
                break
        if drained:
            self.taken += 1
            self.replaced += max(0, drained - 1)
        return latest

    def snapshot(self) -> dict[str, int]:
        return {
            "published": int(self.published),
            "taken": int(self.taken),
            "replaced": int(self.replaced),
            "put_failures": int(self.put_failures),
        }

    def close(self) -> None:
        for method in ("cancel_join_thread", "close"):
            try:
                getattr(self.queue, method)()
            except Exception:
                pass

    def _drain(self) -> int:
        return len(self._drain_items())

    def _drain_items(self) -> list[Any]:
        items: list[Any] = []
        while True:
            try:
                items.append(self.queue.get_nowait())
            except queue.Empty:
                break
        return items


__all__ = [
    "LatestValueSlot",
    "LatestWinsQueue",
    "LatestWinsPublishInfo",
]
