from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import multiprocessing as mp
import os
import queue
import sys
import time
from typing import Any, Sequence

from qqtt.demo.demo31_dual_gpu_ipc import (
    LatestWinsQueue,
    TrackingInputLitePacket,
    TrackingResultLitePacket,
)
from qqtt.tracking.backends.point_tracker_adapter import (
    LITETRACKER_RUNTIME_PYTORCH,
    TRACKER_BACKEND_COTRACKER3,
    TRACKER_BACKEND_LITETRACKER,
    TRACKER_BACKEND_LOCOTRACK,
    TRACKER_BACKEND_TAPNEXTPP,
    TRACKER_BATCH_QUERY_COUNT_POLICY_FIXED,
    TRACKER_EXECUTION_MODE_AUTO,
    TRACKER_EXECUTION_MODE_BATCH_VIEWS,
    PointTrackerAdapterConfig,
    build_point_tracker_adapter_factory,
    effective_legacy_update_mode,
    normalize_litetracker_runtime,
    normalize_tracker_backend,
    normalize_tracker_batch_query_count_policy,
    normalize_tracker_execution_mode,
    tracker_backend_spec,
)


PROCESS_MODE_SUBPROCESS = "subprocess"
PROCESS_MODE_SPAWN = "spawn"
PROCESS_MODES = (PROCESS_MODE_SUBPROCESS, PROCESS_MODE_SPAWN)
COTRACKER_UPDATE_MODE_AUTO = "auto"
COTRACKER_UPDATE_MODE_BATCH = "batch"


@dataclass(frozen=True)
class CoTrackerProcessConfig:
    camera_ids: tuple[int, ...] = (0, 1, 2)
    cotracker_gpu: str = "1"
    cotracker_backend: str = TRACKER_BACKEND_COTRACKER3
    backend_execution_mode: str = TRACKER_EXECUTION_MODE_BATCH_VIEWS
    query_mode: str = "phystwin_dense"
    query_count_request: str = "auto"
    seed: int = 42
    sampling_device: str = "cuda"
    init_requires_object_and_controller: bool = True
    overlay_max_points_per_camera: int = 0
    overlay_display_scope: str = "controller"
    input_max_age_ms: float = 250.0
    poll_interval_s: float = 0.001
    process_mode: str = PROCESS_MODE_SUBPROCESS
    device: str = "cuda"
    prewarm_backends: bool = True
    update_mode: str = COTRACKER_UPDATE_MODE_BATCH
    trackon2_checkpoint: str | None = None
    trackon2_config: str | None = None
    trackon2_repo_dir: str | None = None
    litetracker_weights: str | None = None
    litetracker_repo_dir: str | None = None
    litetracker_runtime: str = LITETRACKER_RUNTIME_PYTORCH
    litetracker_onnx_dir: str | None = None
    litetracker_export_onnx: bool = False
    litetracker_onnx_opset: int = 17
    litetracker_onnx_optimization_level: int = 5
    locotrack_repo_dir: str | None = None
    locotrack_checkpoint: str | None = None
    locotrack_model_size: str = "small"
    locotrack_window_frames: int = 8
    locotrack_resolution: tuple[int, int] = (256, 256)
    locotrack_query_chunk_size: int = 256
    locotrack_autocast_dtype: str = "bf16"
    tapnet_repo_dir: str | None = None
    tapnextpp_checkpoint: str | None = None
    tapnextpp_image_size: tuple[int, int] = (256, 256)
    tapnextpp_autocast_dtype: str = "fp16"
    tapnextpp_use_certainty: bool = False
    tapnextpp_certainty_radius: int = 8
    tapnextpp_certainty_threshold: float = 0.5
    tapnextpp_compile: bool = False
    tapnextpp_reset_on_reinitialize: bool = True
    tracker_batch_query_count_policy: str = TRACKER_BATCH_QUERY_COUNT_POLICY_FIXED

    @property
    def normalized_tracker_backend(self) -> str:
        return normalize_tracker_backend(self.cotracker_backend)

    @property
    def tracker_family(self) -> str:
        return tracker_backend_spec(self.normalized_tracker_backend).family

    @property
    def tracker_query_dependent_init(self) -> bool:
        return self.normalized_tracker_backend == TRACKER_BACKEND_LITETRACKER

    @property
    def tracker_prewarm_mode(self) -> str:
        if self.normalized_tracker_backend == TRACKER_BACKEND_LITETRACKER:
            return "model_load_only" if bool(self.prewarm_backends) else "lazy_query_init"
        if self.normalized_tracker_backend == TRACKER_BACKEND_LOCOTRACK:
            return "model_load_only" if bool(self.prewarm_backends) else "disabled"
        if self.normalized_tracker_backend == TRACKER_BACKEND_TAPNEXTPP:
            return "model_load_only" if bool(self.prewarm_backends) else "disabled"
        return "backend_model_prewarm" if bool(self.prewarm_backends) else "disabled"

    def to_json_dict(self) -> dict[str, Any]:
        tracker_backend = normalize_tracker_backend(self.cotracker_backend)
        execution_mode = normalize_tracker_execution_mode(self.backend_execution_mode)
        legacy_update = str(self.update_mode).strip().lower().replace("_", "-")
        if execution_mode == TRACKER_EXECUTION_MODE_AUTO and legacy_update in {"batch", "serial"}:
            execution_mode = "batch-views" if legacy_update == "batch" else "serial"
        return {
            "camera_ids": [int(item) for item in self.camera_ids],
            "cotracker_gpu": str(self.cotracker_gpu),
            "cotracker_backend": str(tracker_backend),
            "tracker_backend": str(tracker_backend),
            "backend_execution_mode": str(execution_mode),
            "query_mode": str(self.query_mode),
            "query_count_request": str(self.query_count_request),
            "seed": int(self.seed),
            "sampling_device": str(self.sampling_device),
            "init_requires_object_and_controller": bool(self.init_requires_object_and_controller),
            "overlay_max_points_per_camera": int(self.overlay_max_points_per_camera),
            "overlay_display_scope": str(self.overlay_display_scope),
            "input_max_age_ms": float(self.input_max_age_ms),
            "poll_interval_s": float(self.poll_interval_s),
            "process_mode": str(self.process_mode),
            "device": str(self.device),
            "prewarm_backends": bool(self.prewarm_backends),
            "update_mode": str(effective_legacy_update_mode(execution_mode)),
            "trackon2_checkpoint": self.trackon2_checkpoint,
            "trackon2_config": self.trackon2_config,
            "trackon2_repo_dir": self.trackon2_repo_dir,
            "litetracker_weights": self.litetracker_weights,
            "litetracker_repo_dir": self.litetracker_repo_dir,
            "litetracker_runtime": normalize_litetracker_runtime(self.litetracker_runtime),
            "litetracker_onnx_dir": self.litetracker_onnx_dir,
            "litetracker_export_onnx": bool(self.litetracker_export_onnx),
            "litetracker_onnx_opset": int(self.litetracker_onnx_opset),
            "litetracker_onnx_optimization_level": int(self.litetracker_onnx_optimization_level),
            "locotrack_repo_dir": self.locotrack_repo_dir,
            "locotrack_checkpoint": self.locotrack_checkpoint,
            "locotrack_model_size": str(self.locotrack_model_size),
            "locotrack_window_frames": int(self.locotrack_window_frames),
            "locotrack_resolution": [int(item) for item in self.locotrack_resolution],
            "locotrack_query_chunk_size": int(self.locotrack_query_chunk_size),
            "locotrack_autocast_dtype": str(self.locotrack_autocast_dtype),
            "tapnet_repo_dir": self.tapnet_repo_dir,
            "tapnextpp_checkpoint": self.tapnextpp_checkpoint,
            "tapnextpp_image_size": [int(item) for item in self.tapnextpp_image_size],
            "tapnextpp_autocast_dtype": str(self.tapnextpp_autocast_dtype),
            "tapnextpp_use_certainty": bool(self.tapnextpp_use_certainty),
            "tapnextpp_certainty_radius": int(self.tapnextpp_certainty_radius),
            "tapnextpp_certainty_threshold": float(self.tapnextpp_certainty_threshold),
            "tapnextpp_compile": bool(self.tapnextpp_compile),
            "tapnextpp_reset_on_reinitialize": bool(self.tapnextpp_reset_on_reinitialize),
            "tracker_batch_query_count_policy": normalize_tracker_batch_query_count_policy(
                self.tracker_batch_query_count_policy
            ),
        }

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> "CoTrackerProcessConfig":
        backend = normalize_tracker_backend(payload.get("tracker_backend", payload.get("cotracker_backend", TRACKER_BACKEND_COTRACKER3)))
        execution_mode = normalize_tracker_execution_mode(
            payload.get("backend_execution_mode", payload.get("tracking_backend_execution_mode", payload.get("update_mode", TRACKER_EXECUTION_MODE_BATCH_VIEWS)))
        )
        legacy_update = str(payload.get("update_mode", "")).strip().lower().replace("_", "-")
        if execution_mode == TRACKER_EXECUTION_MODE_AUTO and legacy_update in {"batch", "serial"}:
            execution_mode = "batch-views" if legacy_update == "batch" else "serial"
        return cls(
            camera_ids=tuple(int(item) for item in payload.get("camera_ids", (0, 1, 2))),
            cotracker_gpu=str(payload.get("cotracker_gpu", "1")),
            cotracker_backend=backend,
            backend_execution_mode=execution_mode,
            query_mode=str(payload.get("query_mode", "phystwin_dense")),
            query_count_request=str(payload.get("query_count_request", payload.get("query_count", "auto"))),
            seed=int(payload.get("seed", 42)),
            sampling_device=str(payload.get("sampling_device", "cuda")),
            init_requires_object_and_controller=bool(payload.get("init_requires_object_and_controller", True)),
            overlay_max_points_per_camera=int(payload.get("overlay_max_points_per_camera", 0)),
            overlay_display_scope=str(payload.get("overlay_display_scope", "controller")),
            input_max_age_ms=float(payload.get("input_max_age_ms", 250.0)),
            poll_interval_s=float(payload.get("poll_interval_s", 0.001)),
            process_mode=str(payload.get("process_mode", PROCESS_MODE_SUBPROCESS)),
            device=str(payload.get("device", "cuda")),
            prewarm_backends=bool(payload.get("prewarm_backends", True)),
            update_mode=str(effective_legacy_update_mode(execution_mode)),
            trackon2_checkpoint=payload.get("trackon2_checkpoint"),
            trackon2_config=payload.get("trackon2_config"),
            trackon2_repo_dir=payload.get("trackon2_repo_dir"),
            litetracker_weights=payload.get("litetracker_weights"),
            litetracker_repo_dir=payload.get("litetracker_repo_dir"),
            litetracker_runtime=normalize_litetracker_runtime(
                payload.get("litetracker_runtime", LITETRACKER_RUNTIME_PYTORCH)
            ),
            litetracker_onnx_dir=payload.get("litetracker_onnx_dir"),
            litetracker_export_onnx=bool(payload.get("litetracker_export_onnx", False)),
            litetracker_onnx_opset=int(payload.get("litetracker_onnx_opset", 17)),
            litetracker_onnx_optimization_level=int(payload.get("litetracker_onnx_optimization_level", 5)),
            locotrack_repo_dir=payload.get("locotrack_repo_dir"),
            locotrack_checkpoint=payload.get("locotrack_checkpoint"),
            locotrack_model_size=str(payload.get("locotrack_model_size", "small")),
            locotrack_window_frames=int(payload.get("locotrack_window_frames", 8)),
            locotrack_resolution=tuple(int(item) for item in payload.get("locotrack_resolution", (256, 256))),
            locotrack_query_chunk_size=int(payload.get("locotrack_query_chunk_size", 256)),
            locotrack_autocast_dtype=str(payload.get("locotrack_autocast_dtype", "bf16")),
            tapnet_repo_dir=payload.get("tapnet_repo_dir"),
            tapnextpp_checkpoint=payload.get("tapnextpp_checkpoint"),
            tapnextpp_image_size=tuple(int(item) for item in payload.get("tapnextpp_image_size", (256, 256))),
            tapnextpp_autocast_dtype=str(payload.get("tapnextpp_autocast_dtype", "fp16")),
            tapnextpp_use_certainty=bool(payload.get("tapnextpp_use_certainty", False)),
            tapnextpp_certainty_radius=int(payload.get("tapnextpp_certainty_radius", 8)),
            tapnextpp_certainty_threshold=float(payload.get("tapnextpp_certainty_threshold", 0.5)),
            tapnextpp_compile=bool(payload.get("tapnextpp_compile", False)),
            tapnextpp_reset_on_reinitialize=bool(payload.get("tapnextpp_reset_on_reinitialize", True)),
            tracker_batch_query_count_policy=normalize_tracker_batch_query_count_policy(
                payload.get("tracker_batch_query_count_policy", TRACKER_BATCH_QUERY_COUNT_POLICY_FIXED)
            ),
        )

    @classmethod
    def from_json(cls, raw: str) -> "CoTrackerProcessConfig":
        return cls.from_json_dict(json.loads(raw))

    def to_json(self) -> str:
        return json.dumps(self.to_json_dict(), sort_keys=True)


def build_cotracker_process_env(
    config: CoTrackerProcessConfig,
    *,
    base_env: dict[str, str] | None = None,
) -> dict[str, str]:
    env = dict(os.environ if base_env is None else base_env)
    env["CUDA_VISIBLE_DEVICES"] = str(config.cotracker_gpu)
    env["QQTT_DEMO31_COTRACKER_PROCESS"] = "1"
    env["QQTT_DEMO31_POINT_TRACKER_PROCESS"] = "1"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return env


def configure_cotracker_cuda_environment(config: CoTrackerProcessConfig) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.cotracker_gpu)
    os.environ["QQTT_DEMO31_COTRACKER_PROCESS"] = "1"
    os.environ["QQTT_DEMO31_POINT_TRACKER_PROCESS"] = "1"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def build_cotracker_subprocess_argv(
    config: CoTrackerProcessConfig,
    *,
    python_executable: str | None = None,
) -> list[str]:
    return [
        python_executable or sys.executable,
        "-m",
        "qqtt.demo.demo31_cotracker_process",
        "--config-json",
        config.to_json(),
        "--print-contract",
    ]


class CoTrackerProcessHandle:
    def __init__(
        self,
        *,
        process: Any,
        input_queue: Any,
        output_queue: Any,
        status_queue: Any | None,
        stop_event: Any,
    ) -> None:
        self.process = process
        self.input_endpoint = LatestWinsQueue(input_queue)
        self.output_endpoint = LatestWinsQueue(output_queue)
        self.status_queue = status_queue
        self.stop_event = stop_event
        self.started_s = time.perf_counter()
        self._status_events: list[dict[str, Any]] = []

    @property
    def pid(self) -> int:
        return int(getattr(self.process, "pid", 0) or 0)

    def start(self) -> None:
        if hasattr(self.process, "start"):
            self.process.start()

    def publish_input(self, packet: TrackingInputLitePacket) -> int:
        return self.input_endpoint.publish_latest(packet)

    def get_result(self) -> TrackingResultLitePacket | None:
        return self.output_endpoint.take_latest()

    def stop(self, *, timeout_s: float = 2.0) -> None:
        if hasattr(self.stop_event, "set"):
            self.stop_event.set()
        if hasattr(self.process, "join"):
            self.process.join(float(timeout_s))
        if hasattr(self.process, "is_alive") and self.process.is_alive() and hasattr(self.process, "terminate"):
            self.process.terminate()
            self.process.join(float(timeout_s))

    def drain_status_events(self) -> list[dict[str, Any]]:
        if self.status_queue is None:
            return []
        drained: list[dict[str, Any]] = []
        while True:
            try:
                item = self.status_queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(item, dict):
                receive_perf_s = time.perf_counter()
                event = dict(item)
                event["status_receive_perf_s"] = float(receive_perf_s)
                event["status_receive_after_process_start_s"] = float(
                    max(0.0, receive_perf_s - float(self.started_s))
                )
                ready_perf_s = event.get("ready_perf_s")
                if ready_perf_s is not None:
                    try:
                        ready_perf = float(ready_perf_s)
                    except (TypeError, ValueError):
                        ready_perf = 0.0
                    if ready_perf > 0.0:
                        event["ready_event_after_process_start_s"] = float(
                            max(0.0, ready_perf - float(self.started_s))
                        )
                        event["ready_receive_after_process_start_s"] = float(
                            max(0.0, receive_perf_s - float(self.started_s))
                        )
                        event["ready_queue_lag_ms"] = float(max(0.0, (receive_perf_s - ready_perf) * 1000.0))
                drained.append(event)
        self._status_events.extend(drained)
        return drained

    def snapshot(self) -> dict[str, Any]:
        self.drain_status_events()
        ready_events = [
            event
            for event in self._status_events
            if str(event.get("type")) == "ready"
        ]
        return {
            "pid": self.pid,
            "alive": bool(self.process.is_alive()) if hasattr(self.process, "is_alive") else False,
            "input_endpoint": self.input_endpoint.snapshot(),
            "output_endpoint": self.output_endpoint.snapshot(),
            "status_events": list(self._status_events),
            "ready": ready_events[-1] if ready_events else None,
        }


def start_cotracker_process(
    config: CoTrackerProcessConfig,
    *,
    context_name: str = "spawn",
) -> CoTrackerProcessHandle:
    ctx = mp.get_context(context_name)
    input_queue = ctx.Queue(maxsize=1)
    output_queue = ctx.Queue(maxsize=1)
    status_queue = ctx.Queue(maxsize=16)
    stop_event = ctx.Event()
    process = ctx.Process(
        target=run_cotracker_worker_loop,
        name="demo31-cotracker-gpu",
        args=(config, input_queue, output_queue, stop_event, status_queue),
        daemon=True,
    )
    handle = CoTrackerProcessHandle(
        process=process,
        input_queue=input_queue,
        output_queue=output_queue,
        status_queue=status_queue,
        stop_event=stop_event,
    )
    handle.start()
    return handle


def run_cotracker_worker_loop(
    config: CoTrackerProcessConfig,
    input_queue: Any,
    output_queue: Any,
    stop_event: Any,
    status_queue: Any | None = None,
    *,
    backend_factory: Any | None = None,
) -> dict[str, Any]:
    process_start_s = time.perf_counter()
    configure_cotracker_cuda_environment(config)

    from qqtt.demo.point_tracker_overlay_worker import (  # noqa: PLC0415
        LatestTrackingOverlaySlot,
        PointTrackerOverlayWorker,
    )

    input_endpoint = LatestWinsQueue(input_queue)
    output_endpoint = LatestWinsQueue(output_queue)
    output_slot = LatestTrackingOverlaySlot()
    adapter_factory = backend_factory or build_point_tracker_adapter_factory(
        PointTrackerAdapterConfig(
            backend=normalize_tracker_backend(config.cotracker_backend),
            device=str(config.device),
            trackon2_checkpoint=config.trackon2_checkpoint,
            trackon2_config=config.trackon2_config,
            trackon2_repo_dir=config.trackon2_repo_dir,
            litetracker_weights=config.litetracker_weights,
            litetracker_repo_dir=config.litetracker_repo_dir,
            litetracker_runtime=config.litetracker_runtime,
            litetracker_onnx_dir=config.litetracker_onnx_dir,
            litetracker_export_onnx=config.litetracker_export_onnx,
            litetracker_onnx_opset=config.litetracker_onnx_opset,
            litetracker_onnx_optimization_level=config.litetracker_onnx_optimization_level,
            locotrack_repo_dir=config.locotrack_repo_dir,
            locotrack_checkpoint=config.locotrack_checkpoint,
            locotrack_model_size=config.locotrack_model_size,
            locotrack_window_frames=config.locotrack_window_frames,
            locotrack_resolution=config.locotrack_resolution,
            locotrack_query_chunk_size=config.locotrack_query_chunk_size,
            locotrack_autocast_dtype=config.locotrack_autocast_dtype,
            tapnet_repo_dir=config.tapnet_repo_dir,
            tapnextpp_checkpoint=config.tapnextpp_checkpoint,
            tapnextpp_image_size=config.tapnextpp_image_size,
            tapnextpp_autocast_dtype=config.tapnextpp_autocast_dtype,
            tapnextpp_use_certainty=config.tapnextpp_use_certainty,
            tapnextpp_certainty_radius=config.tapnextpp_certainty_radius,
            tapnextpp_certainty_threshold=config.tapnextpp_certainty_threshold,
            tapnextpp_compile=config.tapnextpp_compile,
            tapnextpp_reset_on_reinitialize=config.tapnextpp_reset_on_reinitialize,
        )
    )
    update_mode = effective_legacy_update_mode(config.backend_execution_mode)
    worker = PointTrackerOverlayWorker(
        camera_ids=tuple(int(item) for item in config.camera_ids),
        backend_factory=adapter_factory,
        output_slot=output_slot,
        query_mode=str(config.query_mode),
        query_count_request=str(config.query_count_request),
        seed=int(config.seed),
        sampling_device=str(config.sampling_device),
        init_requires_object_and_controller=bool(config.init_requires_object_and_controller),
        overlay_max_points_per_camera=int(config.overlay_max_points_per_camera),
        overlay_display_scope=str(config.overlay_display_scope),
        device=str(config.device),
        update_mode=str(update_mode),
        tracker_backend=normalize_tracker_backend(config.cotracker_backend),
        backend_execution_mode=normalize_tracker_execution_mode(config.backend_execution_mode),
        tracker_batch_query_count_policy=normalize_tracker_batch_query_count_policy(
            config.tracker_batch_query_count_policy
        ),
    )
    warmup_profile: dict[str, Any] = {}
    tracker_backend = normalize_tracker_backend(config.cotracker_backend)
    tracker_prewarm_mode = config.tracker_prewarm_mode
    if bool(config.prewarm_backends):
        try:
            warmup_profile = worker.warmup_backends()
        except BaseException as exc:
            if status_queue is not None:
                try:
                    status_queue.put_nowait(
                        {
                            "type": "error",
                            "stage": "tracker_warmup",
                            "legacy_stage": "cotracker_warmup",
                            "error": f"{type(exc).__name__}: {exc}",
                            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                            "tracker_backend": tracker_backend,
                            "tracker_prewarm_mode": tracker_prewarm_mode,
                            "total_init_ms": float((time.perf_counter() - process_start_s) * 1000.0),
                        }
                    )
                except Exception:
                    pass
            raise
    else:
        warmup_profile = {
            "skipped": True,
            "skip_reason": str(tracker_prewarm_mode),
            "tracker_backend": tracker_backend,
            "total_ms": 0.0,
            "per_camera": {},
        }
    if status_queue is not None:
        try:
            status_queue.put_nowait(
                {
                    "type": "ready",
                    "stage": (
                        "litetracker_ready_to_receive_inputs"
                        if tracker_backend == TRACKER_BACKEND_LITETRACKER
                        else "tracker_ready"
                    ),
                    "legacy_stage": "cotracker",
                    "process_kind": "point_tracker_child",
                    "ready_state": "ready_to_receive_inputs",
                    "ready_to_receive_inputs": True,
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "cotracker_backend": tracker_backend,
                    "tracker_backend": tracker_backend,
                    "tracker_family": config.tracker_family,
                    "litetracker_runtime": normalize_litetracker_runtime(config.litetracker_runtime),
                    "litetracker_onnx_dir": config.litetracker_onnx_dir,
                    "litetracker_export_onnx": bool(config.litetracker_export_onnx),
                    "litetracker_onnx_opset": int(config.litetracker_onnx_opset),
                    "litetracker_onnx_optimization_level": int(config.litetracker_onnx_optimization_level),
                    "locotrack_model_size": str(config.locotrack_model_size),
                    "locotrack_window_frames": int(config.locotrack_window_frames),
                    "locotrack_resolution": [int(item) for item in config.locotrack_resolution],
                    "locotrack_query_chunk_size": int(config.locotrack_query_chunk_size),
                    "locotrack_autocast_dtype": str(config.locotrack_autocast_dtype),
                    "locotrack_checkpoint": config.locotrack_checkpoint,
                    "locotrack_repo_dir": config.locotrack_repo_dir,
                    "tapnet_repo_dir": config.tapnet_repo_dir,
                    "tapnextpp_checkpoint": config.tapnextpp_checkpoint,
                    "tapnextpp_image_size": [int(item) for item in config.tapnextpp_image_size],
                    "tapnextpp_autocast_dtype": str(config.tapnextpp_autocast_dtype),
                    "tapnextpp_use_certainty": bool(config.tapnextpp_use_certainty),
                    "tapnextpp_certainty_radius": int(config.tapnextpp_certainty_radius),
                    "tapnextpp_certainty_threshold": float(config.tapnextpp_certainty_threshold),
                    "tapnextpp_compile": bool(config.tapnextpp_compile),
                    "tapnextpp_reset_on_reinitialize": bool(config.tapnextpp_reset_on_reinitialize),
                    "backend_execution_mode": normalize_tracker_execution_mode(config.backend_execution_mode),
                    "prewarm_backends": bool(config.prewarm_backends),
                    "tracker_prewarm_mode": tracker_prewarm_mode,
                    "tracker_query_dependent_init": bool(config.tracker_query_dependent_init),
                    "tracker_query_dependent_init_pending": bool(config.tracker_query_dependent_init),
                    "update_mode": str(update_mode),
                    "warmup_profile": warmup_profile,
                    "tracker_warmup_profile": warmup_profile,
                    "total_init_ms": float((time.perf_counter() - process_start_s) * 1000.0),
                    "ready_perf_s": time.perf_counter(),
                }
            )
        except Exception:
            pass
    processed = 0
    dropped_old = 0
    while not stop_event.is_set():
        packet = input_endpoint.take_latest()
        if packet is None:
            time.sleep(float(config.poll_interval_s))
            continue
        if time.perf_counter() - float(packet.timestamp_s) > float(config.input_max_age_ms) / 1000.0:
            dropped_old += 1
            continue
        overlay = worker.process_group(packet.to_overlay_input_packet())
        processed += 1
        if overlay is not None:
            output_endpoint.publish_latest(TrackingResultLitePacket.from_overlay_packet(overlay))
    return {
        "processed": int(processed),
        "dropped_old": int(dropped_old),
        "worker": worker.snapshot(),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Demo 3.1 CoTracker3 child process helper.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--print-contract", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = CoTrackerProcessConfig.from_json(args.config_json)
    configure_cotracker_cuda_environment(config)
    if args.print_contract:
        print(
            json.dumps(
                {
                    "cotracker_process": True,
                    "point_tracker_process": True,
                    "process_kind": "point_tracker_child",
                    "cotracker_backend": config.cotracker_backend,
                    "tracker_backend": normalize_tracker_backend(config.cotracker_backend),
                    "tracker_family": config.tracker_family,
                    "litetracker_runtime": normalize_litetracker_runtime(config.litetracker_runtime),
                    "litetracker_onnx_dir": config.litetracker_onnx_dir,
                    "litetracker_export_onnx": bool(config.litetracker_export_onnx),
                    "litetracker_onnx_opset": int(config.litetracker_onnx_opset),
                    "litetracker_onnx_optimization_level": int(config.litetracker_onnx_optimization_level),
                    "locotrack_model_size": str(config.locotrack_model_size),
                    "locotrack_window_frames": int(config.locotrack_window_frames),
                    "locotrack_resolution": [int(item) for item in config.locotrack_resolution],
                    "locotrack_query_chunk_size": int(config.locotrack_query_chunk_size),
                    "locotrack_autocast_dtype": str(config.locotrack_autocast_dtype),
                    "locotrack_checkpoint": config.locotrack_checkpoint,
                    "locotrack_repo_dir": config.locotrack_repo_dir,
                    "tapnet_repo_dir": config.tapnet_repo_dir,
                    "tapnextpp_checkpoint": config.tapnextpp_checkpoint,
                    "tapnextpp_image_size": [int(item) for item in config.tapnextpp_image_size],
                    "tapnextpp_autocast_dtype": str(config.tapnextpp_autocast_dtype),
                    "tapnextpp_use_certainty": bool(config.tapnextpp_use_certainty),
                    "tapnextpp_certainty_radius": int(config.tapnextpp_certainty_radius),
                    "tapnextpp_certainty_threshold": float(config.tapnextpp_certainty_threshold),
                    "tapnextpp_compile": bool(config.tapnextpp_compile),
                    "tapnextpp_reset_on_reinitialize": bool(config.tapnextpp_reset_on_reinitialize),
                    "backend_execution_mode": normalize_tracker_execution_mode(config.backend_execution_mode),
                    "cotracker_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "tracking_query_mode": config.query_mode,
                    "tracking_query_count_requested": config.query_count_request,
                    "cotracker_seed": config.seed,
                    "init_requires_object_and_controller": config.init_requires_object_and_controller,
                    "overlay_max_points_per_camera": config.overlay_max_points_per_camera,
                    "overlay_display_scope": config.overlay_display_scope,
                    "prewarm_backends": config.prewarm_backends,
                    "tracker_prewarm_mode": config.tracker_prewarm_mode,
                    "tracker_query_dependent_init": config.tracker_query_dependent_init,
                    "ready_state": "ready_to_receive_inputs",
                    "update_mode": effective_legacy_update_mode(config.backend_execution_mode),
                    "tracker_batch_query_count_policy": normalize_tracker_batch_query_count_policy(
                        config.tracker_batch_query_count_policy
                    ),
                    "cross_gpu_cuda_tensor_transfer": False,
                    "ipc_payload": "cpu_numpy_latest_wins",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    return 0


__all__ = [
    "CoTrackerProcessConfig",
    "CoTrackerProcessHandle",
    "PROCESS_MODE_SPAWN",
    "PROCESS_MODE_SUBPROCESS",
    "PROCESS_MODES",
    "build_cotracker_process_env",
    "build_cotracker_subprocess_argv",
    "configure_cotracker_cuda_environment",
    "main",
    "run_cotracker_worker_loop",
    "start_cotracker_process",
]


if __name__ == "__main__":
    raise SystemExit(main())
