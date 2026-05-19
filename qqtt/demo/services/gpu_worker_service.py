from __future__ import annotations

from dataclasses import dataclass, field
import multiprocessing as mp
import os
import queue
import time
from typing import Any, Callable


@dataclass(frozen=True)
class GpuWorkerConfig:
    name: str
    physical_gpu: str
    start_method: str = "spawn"
    env_extra: dict[str, str] = field(default_factory=dict)
    ready_timeout_s: float = 30.0
    heartbeat_interval_s: float = 1.0


@dataclass(frozen=True)
class GpuWorkerStatus:
    name: str
    pid: int
    alive: bool
    cuda_visible_devices: str
    ready: bool
    model_loaded: bool
    processed_count: int
    dropped_count: int
    last_error: str | None
    last_heartbeat_s: float


class GpuWorkerHandle:
    def __init__(self, config: GpuWorkerConfig, process: Any, status_queue: Any) -> None:
        self.config = config
        self.process = process
        self.status_queue = status_queue
        self.status_events: list[dict[str, Any]] = []

    @property
    def pid(self) -> int | None:
        return getattr(self.process, "pid", None)

    def drain_status_events(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        while True:
            try:
                event = self.status_queue.get_nowait()
            except queue.Empty:
                break
            events.append(dict(event))
        self.status_events.extend(events)
        return events

    def stop(self, timeout_s: float = 3.0) -> None:
        try:
            self.process.join(timeout=timeout_s)
        except Exception:
            pass
        if getattr(self.process, "is_alive", lambda: False)():
            try:
                self.process.terminate()
                self.process.join(timeout=2.0)
            except Exception:
                pass
        for method in ("cancel_join_thread", "close"):
            try:
                getattr(self.status_queue, method)()
            except Exception:
                pass

    def snapshot(self) -> dict[str, Any]:
        self.drain_status_events()
        ready = next((event for event in reversed(self.status_events) if event.get("type") == "ready"), None)
        error = next((event for event in reversed(self.status_events) if event.get("type") == "error"), None)
        return {
            "name": self.config.name,
            "pid": self.pid,
            "alive": bool(getattr(self.process, "is_alive", lambda: False)()),
            "cuda_visible_devices": self.config.physical_gpu,
            "ready": ready,
            "last_error": error,
            "status_events": list(self.status_events),
        }


class GpuWorkerService:
    def start(
        self,
        target: Callable[..., Any],
        args: tuple[Any, ...],
        config: GpuWorkerConfig,
    ) -> GpuWorkerHandle:
        if config.start_method not in {"spawn", "forkserver"}:
            raise ValueError("CUDA GPU workers must use spawn or forkserver")
        context = mp.get_context(config.start_method)
        status_queue = context.Queue(maxsize=32)
        process = context.Process(
            target=_gpu_worker_entrypoint,
            args=(target, args, config, status_queue),
            name=str(config.name),
        )
        process.daemon = True
        process.start()
        return GpuWorkerHandle(config, process, status_queue)


def _gpu_worker_entrypoint(
    target: Callable[..., Any],
    args: tuple[Any, ...],
    config: GpuWorkerConfig,
    status_queue: Any,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.physical_gpu)
    os.environ.update({str(key): str(value) for key, value in config.env_extra.items()})
    status_queue.put(
        {
            "type": "ready",
            "name": config.name,
            "pid": os.getpid(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "ready_perf_s": time.perf_counter(),
        }
    )
    try:
        target(*args)
    except BaseException as exc:
        status_queue.put(
            {
                "type": "error",
                "name": config.name,
                "pid": os.getpid(),
                "error": f"{type(exc).__name__}: {exc}",
                "error_perf_s": time.perf_counter(),
            }
        )
        raise


__all__ = [
    "GpuWorkerConfig",
    "GpuWorkerHandle",
    "GpuWorkerService",
    "GpuWorkerStatus",
]
