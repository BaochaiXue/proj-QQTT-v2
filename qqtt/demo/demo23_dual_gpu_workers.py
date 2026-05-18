from __future__ import annotations

import copy
from dataclasses import dataclass, field
import os
import queue
import time
from typing import Any

from qqtt.demo.three_view_masked_fused_pcd_runtime import (
    CameraMaskPacket,
    CaptureGroup,
    DepthGroup,
    MaskGroup,
    _elapsed_ms,
)


@dataclass(frozen=True)
class WorkerStop:
    reason: str = "stop"


@dataclass(frozen=True)
class WorkerCaptureTask:
    group_id: int
    group: CaptureGroup
    enqueued_perf_s: float = field(default_factory=time.perf_counter)


@dataclass(frozen=True)
class WorkerDepthResult:
    group_id: int
    depth_group: DepthGroup
    worker_profile: dict[str, Any]
    worker_timing: dict[str, Any]


@dataclass(frozen=True)
class WorkerMaskResult:
    group_id: int
    mask_group: MaskGroup
    worker_profile: dict[str, Any]
    worker_timing: dict[str, Any]


@dataclass(frozen=True)
class WorkerErrorResult:
    group_id: int | None
    stage: str
    error: str
    traceback: str


STOP = WorkerStop()


class BoundedLatestTaskQueue:
    """Latest-only bounded queue wrapper for multiprocessing task queues."""

    def __init__(self, task_queue: Any, *, maxsize: int) -> None:
        self.task_queue = task_queue
        self.maxsize = max(1, int(maxsize))
        self.drop_count = 0

    def put_latest(self, task: WorkerCaptureTask) -> int:
        dropped = 0
        while _queue_full(self.task_queue):
            try:
                self.task_queue.get_nowait()
            except queue.Empty:
                break
            dropped += 1
        self.task_queue.put(task)
        self.drop_count += dropped
        return dropped


def _queue_full(task_queue: Any) -> bool:
    try:
        return bool(task_queue.full())
    except Exception:
        return False


def is_stop_task(task: Any) -> bool:
    return isinstance(task, WorkerStop)


def parse_cuda_device_index(device: str | int) -> int:
    if isinstance(device, int):
        if device < 0:
            raise ValueError(f"CUDA device index must be non-negative: {device}")
        return int(device)
    value = str(device).strip().lower()
    if value == "cuda":
        return 0
    if value.startswith("cuda:"):
        index = int(value.split(":", maxsplit=1)[1])
        if index < 0:
            raise ValueError(f"CUDA device index must be non-negative: {device}")
        return index
    raise ValueError(f"Demo 2.3 workers require CUDA devices, got {device!r}")


def _prepare_child_args(args: Any, *, physical_device: str, stage: str) -> Any:
    child_args = copy.deepcopy(args)
    physical_index = parse_cuda_device_index(physical_device)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_index)
    child_args.device = "cuda"
    child_args.gpu_sampling = False
    child_args.parallel_init = False
    if stage == "edgetam":
        child_args.sam31_device = "cuda"
    return child_args


def _set_torch_device() -> None:
    import torch

    torch.cuda.set_device(0)


def _worker_profile(runtime: Any, group_id: int, profile_workers: bool) -> dict[str, Any]:
    if not profile_workers:
        return {}
    return runtime.pop_profile_record(int(group_id))


def run_ffs_worker(args: Any, in_queue: Any, out_queue: Any) -> None:
    import traceback

    group_id: int | None = None
    try:
        child_args = _prepare_child_args(args, physical_device=getattr(args, "ffs_device", "cuda:0"), stage="ffs")
        _set_torch_device()
        from qqtt.demo.demo23_runtime import Demo23WorkerRuntime

        runtime = Demo23WorkerRuntime(child_args)
        runner = runtime._get_or_prepare_ffs_runner()
        aligners = {}
        profile_workers = bool(getattr(args, "dual_gpu_profile_workers", False))
        while True:
            task = in_queue.get()
            if is_stop_task(task):
                break
            if not isinstance(task, WorkerCaptureTask):
                continue
            group_id = int(task.group_id)
            started_s = time.perf_counter()
            depth_group, _h2d = runtime._run_depth_cycle_for_group(
                group=task.group,
                runner=runner,
                aligners=aligners,
            )
            out_queue.put(
                WorkerDepthResult(
                    group_id=group_id,
                    depth_group=depth_group,
                    worker_profile=_worker_profile(runtime, group_id, profile_workers),
                    worker_timing={
                        "stage": "ffs",
                        "device": str(getattr(args, "ffs_device", "cuda:0")),
                        "queued_wait_ms": max(0.0, (started_s - float(task.enqueued_perf_s)) * 1000.0),
                        "worker_period_ms": _elapsed_ms(started_s, time.perf_counter()),
                    },
                )
            )
    except BaseException as exc:
        out_queue.put(
            WorkerErrorResult(
                group_id=group_id,
                stage="ffs",
                error=f"{type(exc).__name__}: {exc}",
                traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            )
        )


def run_edgetam_worker(args: Any, in_queue: Any, out_queue: Any) -> None:
    import traceback

    group_id: int | None = None
    try:
        child_args = _prepare_child_args(args, physical_device=getattr(args, "edgetam_device", "cuda:1"), stage="edgetam")
        _set_torch_device()
        from qqtt.demo.demo23_runtime import Demo23WorkerRuntime

        runtime = Demo23WorkerRuntime(child_args)
        states = runtime._get_or_init_gpu_owner_edgetam_states()
        profile_workers = bool(getattr(args, "dual_gpu_profile_workers", False))
        while True:
            task = in_queue.get()
            if is_stop_task(task):
                break
            if not isinstance(task, WorkerCaptureTask):
                continue
            group_id = int(task.group_id)
            started_s = time.perf_counter()
            mask_packets, edgetam_cycle_ms = runtime._run_gpu_owner_edgetam_cycle(
                states=states,
                group=task.group,
            )
            if mask_packets is None:
                continue
            mask_group = MaskGroup(
                group_id=group_id,
                mask_packets=dict(mask_packets),
                edgetam_stage_wall_ms=float(edgetam_cycle_ms),
                edgetam_stage_sum_model_ms=sum(
                    float(packet.cuda_event_model_ms or packet.model_ms)
                    for packet in mask_packets.values()
                    if isinstance(packet, CameraMaskPacket)
                ),
                edgetam_stage_mode="dual-gpu-batch-vision",
            )
            out_queue.put(
                WorkerMaskResult(
                    group_id=group_id,
                    mask_group=mask_group,
                    worker_profile=_worker_profile(runtime, group_id, profile_workers),
                    worker_timing={
                        "stage": "edgetam",
                        "device": str(getattr(args, "edgetam_device", "cuda:1")),
                        "queued_wait_ms": max(0.0, (started_s - float(task.enqueued_perf_s)) * 1000.0),
                        "worker_period_ms": _elapsed_ms(started_s, time.perf_counter()),
                    },
                )
            )
    except BaseException as exc:
        out_queue.put(
            WorkerErrorResult(
                group_id=group_id,
                stage="edgetam",
                error=f"{type(exc).__name__}: {exc}",
                traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            )
        )
