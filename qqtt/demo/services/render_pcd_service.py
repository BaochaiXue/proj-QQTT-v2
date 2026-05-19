from __future__ import annotations

from dataclasses import dataclass
import os
import time
from typing import Any, Protocol

from qqtt.demo.render_fastpath import (
    LatestOnlyRenderBuffer,
    RenderMicroProfileRecord,
    RenderMicroProfiler,
    summarize_render_records,
)
from qqtt.demo.services.service_types import RenderPacket


class RenderPacketRenderer(Protocol):
    def render_packet(self, packet: RenderPacket) -> dict[str, Any]:
        ...


@dataclass
class RenderServiceConfig:
    window_title: str
    render_mode: str = "pointcloud"
    latest_only: bool = True
    fast_exit_env_var: str = "QQTT_WSLG_OPEN3D_FAST_EXIT"
    max_render_fps: float | None = None
    profile_window_s: float = 5.0


class RenderPcdService:
    """Service facade for latest-wins PCD rendering and render profiles.

    This first extraction owns the buffer/profile contract while the legacy
    Open3D GUI loop is still hosted by the existing runtimes.
    """

    def __init__(self, config: RenderServiceConfig, renderer: RenderPacketRenderer | None = None) -> None:
        self.config = config
        self.renderer = renderer
        self.buffer: LatestOnlyRenderBuffer[RenderPacket] = LatestOnlyRenderBuffer()
        self.profiler = RenderMicroProfiler()
        self.started_s: float | None = None
        self.stopped_s: float | None = None
        self.rendered_count = 0
        self.submit_count = 0
        self.fast_exit_used = False

    def start(self) -> None:
        if self.started_s is None:
            self.started_s = time.perf_counter()

    def submit_latest(self, packet: RenderPacket) -> None:
        self.start()
        self.buffer.publish(packet)
        self.submit_count += 1

    def render_once(self) -> RenderPacket | None:
        packet = self.buffer.take_latest()
        if packet is None:
            return None
        started_s = time.perf_counter()
        record_payload: dict[str, Any] = {}
        if self.renderer is not None:
            record_payload = dict(self.renderer.render_packet(packet))
        elapsed_ms = float((time.perf_counter() - started_s) * 1000.0)
        points_count = sum(int(layer.points_xyz.shape[0]) for layer in (*packet.layers, *packet.overlay_layers))
        colors_count = sum(int(layer.colors_rgb.shape[0]) for layer in (*packet.layers, *packet.overlay_layers))
        record = RenderMicroProfileRecord(
            render_packet_id=int(packet.group_id),
            points_count=int(points_count),
            colors_count=int(colors_count),
            queue_wait_ms=0.0,
            render_total_ms=float(record_payload.pop("render_total_ms", elapsed_ms)),
            extra=record_payload,
        )
        self.profiler.record(record)
        self.rendered_count += 1
        return packet

    def run_until_stopped(self, duration_s: float | None = None) -> None:
        self.start()
        deadline_s = None if duration_s is None else time.perf_counter() + float(duration_s)
        while deadline_s is None or time.perf_counter() < deadline_s:
            if self.render_once() is None:
                time.sleep(0.001)

    def stop(self) -> None:
        self.stopped_s = time.perf_counter()

    def should_fast_exit(self) -> bool:
        return os.environ.get(self.config.fast_exit_env_var) == "1"

    def snapshot(self) -> dict[str, Any]:
        summary = summarize_render_records(self.profiler.records())
        started = self.started_s
        stopped = self.stopped_s or time.perf_counter()
        duration_s = max(0.0, stopped - started) if started is not None else 0.0
        rendered_fps = float(self.rendered_count / duration_s) if duration_s > 0.0 else 0.0
        return {
            "render_window_title": self.config.window_title,
            "render_latest_only": bool(self.config.latest_only),
            "render_fast_exit_used": bool(self.fast_exit_used),
            "rendered_fps": rendered_fps,
            "render_loop_fps": rendered_fps,
            "render_waited_for_object_volume_filter": False,
            "render_buffer": self.buffer.snapshot(),
            "render_micro_profile": summary,
        }


__all__ = [
    "RenderPcdService",
    "RenderPacketRenderer",
    "RenderServiceConfig",
]
