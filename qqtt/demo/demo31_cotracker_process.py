from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import multiprocessing as mp
import os
import sys
import time
from typing import Any, Sequence

from qqtt.demo.demo31_dual_gpu_ipc import (
    LatestWinsQueue,
    TrackingInputLitePacket,
    TrackingResultLitePacket,
)


PROCESS_MODE_SUBPROCESS = "subprocess"
PROCESS_MODE_SPAWN = "spawn"
PROCESS_MODES = (PROCESS_MODE_SUBPROCESS, PROCESS_MODE_SPAWN)


@dataclass(frozen=True)
class CoTrackerProcessConfig:
    camera_ids: tuple[int, ...] = (0, 1, 2)
    cotracker_gpu: str = "1"
    cotracker_backend: str = "cotracker3_online"
    query_mode: str = "phystwin_dense"
    query_count_request: str = "auto"
    seed: int = 42
    sampling_device: str = "cuda"
    init_requires_object_and_controller: bool = True
    overlay_max_points_per_camera: int = 30
    input_max_age_ms: float = 250.0
    poll_interval_s: float = 0.001
    process_mode: str = PROCESS_MODE_SUBPROCESS
    device: str = "cuda"

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "camera_ids": [int(item) for item in self.camera_ids],
            "cotracker_gpu": str(self.cotracker_gpu),
            "cotracker_backend": str(self.cotracker_backend),
            "query_mode": str(self.query_mode),
            "query_count_request": str(self.query_count_request),
            "seed": int(self.seed),
            "sampling_device": str(self.sampling_device),
            "init_requires_object_and_controller": bool(self.init_requires_object_and_controller),
            "overlay_max_points_per_camera": int(self.overlay_max_points_per_camera),
            "input_max_age_ms": float(self.input_max_age_ms),
            "poll_interval_s": float(self.poll_interval_s),
            "process_mode": str(self.process_mode),
            "device": str(self.device),
        }

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> "CoTrackerProcessConfig":
        return cls(
            camera_ids=tuple(int(item) for item in payload.get("camera_ids", (0, 1, 2))),
            cotracker_gpu=str(payload.get("cotracker_gpu", "1")),
            cotracker_backend=str(payload.get("cotracker_backend", "cotracker3_online")),
            query_mode=str(payload.get("query_mode", "phystwin_dense")),
            query_count_request=str(payload.get("query_count_request", payload.get("query_count", "auto"))),
            seed=int(payload.get("seed", 42)),
            sampling_device=str(payload.get("sampling_device", "cuda")),
            init_requires_object_and_controller=bool(payload.get("init_requires_object_and_controller", True)),
            overlay_max_points_per_camera=int(payload.get("overlay_max_points_per_camera", 30)),
            input_max_age_ms=float(payload.get("input_max_age_ms", 250.0)),
            poll_interval_s=float(payload.get("poll_interval_s", 0.001)),
            process_mode=str(payload.get("process_mode", PROCESS_MODE_SUBPROCESS)),
            device=str(payload.get("device", "cuda")),
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
    return env


def configure_cotracker_cuda_environment(config: CoTrackerProcessConfig) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.cotracker_gpu)
    os.environ["QQTT_DEMO31_COTRACKER_PROCESS"] = "1"


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
        stop_event: Any,
    ) -> None:
        self.process = process
        self.input_endpoint = LatestWinsQueue(input_queue)
        self.output_endpoint = LatestWinsQueue(output_queue)
        self.stop_event = stop_event
        self.started_s = time.perf_counter()

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

    def snapshot(self) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "alive": bool(self.process.is_alive()) if hasattr(self.process, "is_alive") else False,
            "input_endpoint": self.input_endpoint.snapshot(),
            "output_endpoint": self.output_endpoint.snapshot(),
        }


def start_cotracker_process(
    config: CoTrackerProcessConfig,
    *,
    context_name: str = "spawn",
) -> CoTrackerProcessHandle:
    ctx = mp.get_context(context_name)
    input_queue = ctx.Queue(maxsize=1)
    output_queue = ctx.Queue(maxsize=1)
    stop_event = ctx.Event()
    process = ctx.Process(
        target=run_cotracker_worker_loop,
        name="demo31-cotracker-gpu",
        args=(config, input_queue, output_queue, stop_event),
        daemon=True,
    )
    handle = CoTrackerProcessHandle(
        process=process,
        input_queue=input_queue,
        output_queue=output_queue,
        stop_event=stop_event,
    )
    handle.start()
    return handle


def run_cotracker_worker_loop(
    config: CoTrackerProcessConfig,
    input_queue: Any,
    output_queue: Any,
    stop_event: Any,
    *,
    backend_factory: Any | None = None,
) -> dict[str, Any]:
    configure_cotracker_cuda_environment(config)

    from qqtt.demo.cotracker3_overlay_worker import (  # noqa: PLC0415
        CoTracker3OverlayWorker,
        LatestTrackingOverlaySlot,
    )

    input_endpoint = LatestWinsQueue(input_queue)
    output_endpoint = LatestWinsQueue(output_queue)
    output_slot = LatestTrackingOverlaySlot()
    worker = CoTracker3OverlayWorker(
        camera_ids=tuple(int(item) for item in config.camera_ids),
        backend_factory=backend_factory,
        output_slot=output_slot,
        query_mode=str(config.query_mode),
        query_count_request=str(config.query_count_request),
        seed=int(config.seed),
        sampling_device=str(config.sampling_device),
        init_requires_object_and_controller=bool(config.init_requires_object_and_controller),
        overlay_max_points_per_camera=int(config.overlay_max_points_per_camera),
        device=str(config.device),
    )
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
                    "cotracker_backend": config.cotracker_backend,
                    "cotracker_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "tracking_query_mode": config.query_mode,
                    "tracking_query_count_requested": config.query_count_request,
                    "cotracker_seed": config.seed,
                    "init_requires_object_and_controller": config.init_requires_object_and_controller,
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
