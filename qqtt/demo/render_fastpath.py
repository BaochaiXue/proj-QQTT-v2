from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import json
from pathlib import Path
import threading
import time
from typing import Any, Generic, TypeVar

import numpy as np


RENDER_BACKEND_LEGACY_CURRENT = "legacy-current"
RENDER_BACKEND_LEGACY_INPLACE = "legacy-inplace"
RENDER_BACKEND_TENSOR_O3D_DLPACK = "tensor-o3d-dlpack"
RENDER_BACKENDS = (
    RENDER_BACKEND_LEGACY_CURRENT,
    RENDER_BACKEND_LEGACY_INPLACE,
    RENDER_BACKEND_TENSOR_O3D_DLPACK,
)
DEFAULT_RENDER_BACKEND = RENDER_BACKEND_LEGACY_INPLACE

RENDER_COPY_MODE_SYNC_CPU = "sync-cpu"
RENDER_COPY_MODE_ASYNC_PINNED = "async-pinned"
RENDER_COPY_MODES = (RENDER_COPY_MODE_SYNC_CPU, RENDER_COPY_MODE_ASYNC_PINNED)
DEFAULT_RENDER_COPY_MODE = RENDER_COPY_MODE_SYNC_CPU

RENDER_LAYER_MODE_COMBINED = "combined"
RENDER_LAYER_MODE_SEPARATE = "separate"
RENDER_LAYER_MODES = (RENDER_LAYER_MODE_COMBINED, RENDER_LAYER_MODE_SEPARATE)
DEFAULT_RENDER_LAYER_MODE = RENDER_LAYER_MODE_COMBINED

T = TypeVar("T")


def elapsed_ms(start_s: float, end_s: float | None = None) -> float:
    return float(((time.perf_counter() if end_s is None else end_s) - start_s) * 1000.0)


class LatestOnlyRenderBuffer(Generic[T]):
    """Thread-safe latest-wins render buffer.

    Producers never block on the renderer. If the renderer is behind, the
    pending display packet is replaced and the stale display packet is counted
    as dropped. This does not lower compute/filter PCD quality; it only avoids
    queuing obsolete UI frames.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._packet: T | None = None
        self.published = 0
        self.taken = 0
        self.dropped = 0
        self.backpressure_count = 0

    def publish(self, packet: T) -> None:
        with self._lock:
            if self._packet is not None:
                self.dropped += 1
            self._packet = packet
            self.published += 1

    def take_latest(self) -> T | None:
        with self._lock:
            packet = self._packet
            self._packet = None
            if packet is not None:
                self.taken += 1
            return packet

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "published": int(self.published),
                "taken": int(self.taken),
                "displayed": int(self.taken),
                "dropped": int(self.dropped),
                "pending": int(self._packet is not None),
                "backpressure_count": int(self.backpressure_count),
            }


class CoalescedRenderPostGate:
    """Allow at most one queued GUI render callback at a time."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending = False
        self.posted = 0
        self.coalesced = 0

    def try_mark_pending(self) -> bool:
        with self._lock:
            if self._pending:
                self.coalesced += 1
                return False
            self._pending = True
            self.posted += 1
            return True

    def mark_done(self) -> None:
        with self._lock:
            self._pending = False

    def snapshot(self) -> dict[str, int | bool]:
        with self._lock:
            return {
                "posted": int(self.posted),
                "coalesced": int(self.coalesced),
                "pending": bool(self._pending),
            }


class ColorFloat32InplaceBuffer:
    def __init__(self) -> None:
        self.array: np.ndarray | None = None

    def ensure(self, n_points: int) -> np.ndarray:
        if self.array is None or self.array.shape != (n_points, 3):
            self.array = np.empty((n_points, 3), dtype=np.float32)
        return self.array

    def convert_into(self, colors_rgb_u8: np.ndarray) -> np.ndarray:
        if colors_rgb_u8.ndim != 2 or colors_rgb_u8.shape[1] != 3:
            raise ValueError("colors_rgb_u8 must be an Nx3 array")
        out = self.ensure(int(colors_rgb_u8.shape[0]))
        np.multiply(colors_rgb_u8, np.float32(1.0 / 255.0), out=out, casting="unsafe")
        return out


class PointsFloat32InplaceBuffer:
    def __init__(self) -> None:
        self.array: np.ndarray | None = None

    def ensure(self, n_points: int) -> np.ndarray:
        if self.array is None or self.array.shape != (n_points, 3):
            self.array = np.empty((n_points, 3), dtype=np.float32)
        return self.array

    def copy_into(self, points_xyz_m: np.ndarray) -> np.ndarray:
        if points_xyz_m.ndim != 2 or points_xyz_m.shape[1] != 3:
            raise ValueError("points_xyz_m must be an Nx3 array")
        out = self.ensure(int(points_xyz_m.shape[0]))
        np.copyto(out, points_xyz_m, casting="unsafe")
        return out


class RenderLayerCombiner:
    """Reusable object/controller display combiner.

    The compute/filter output remains split by semantic layer. This helper only
    builds a display packet with the same points and colors so Open3D can update
    one geometry instead of two.
    """

    def __init__(self) -> None:
        self._points: np.ndarray | None = None
        self._colors: np.ndarray | None = None

    def combine(self, layers: Sequence[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, float]:
        started_s = time.perf_counter()
        nonempty = [(points, colors) for points, colors in layers if int(points.shape[0]) > 0]
        if not nonempty:
            return (
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.uint8),
                elapsed_ms(started_s),
            )
        total_points = int(sum(points.shape[0] for points, _colors in nonempty))
        points_out = self._points
        colors_out = self._colors
        if points_out is None or points_out.shape != (total_points, 3):
            points_out = np.empty((total_points, 3), dtype=np.float32)
            self._points = points_out
        if colors_out is None or colors_out.shape != (total_points, 3):
            colors_out = np.empty((total_points, 3), dtype=np.uint8)
            self._colors = colors_out
        offset = 0
        for points, colors in nonempty:
            n_points = int(points.shape[0])
            if colors.shape[0] != n_points:
                raise ValueError("points and colors must have the same length")
            np.copyto(points_out[offset : offset + n_points], points, casting="unsafe")
            np.copyto(colors_out[offset : offset + n_points], colors, casting="unsafe")
            offset += n_points
        return points_out, colors_out, elapsed_ms(started_s)


@dataclass(frozen=True)
class RenderLayerUpdate:
    points_count: int
    colors_count: int
    cpu_format_ms: float
    open3d_points_update_ms: float
    open3d_colors_update_ms: float
    open3d_update_geometry_ms: float
    open3d_add_geometry_ms: float = 0.0
    open3d_remove_geometry_ms: float = 0.0
    geometry_recreated: bool = False
    tensor_rebound: bool = False


class Open3DSceneTensorLayer:
    """Open3D GUI Scene tensor point-cloud layer with a no-recreate fast path."""

    def __init__(
        self,
        *,
        name: str,
        o3d_module: Any,
        o3c_module: Any,
        rendering_module: Any,
        scene: Any,
        material: Any,
        device: Any,
        backend: str = DEFAULT_RENDER_BACKEND,
        min_capacity: int = 0,
    ) -> None:
        if backend not in RENDER_BACKENDS:
            raise ValueError(f"unsupported render backend {backend!r}")
        if int(min_capacity) < 0:
            raise ValueError("min_capacity must be >= 0")
        self.name = str(name)
        self.o3d = o3d_module
        self.o3c = o3c_module
        self.rendering = rendering_module
        self.scene = scene
        self.material = material
        self.backend = backend
        self.effective_backend = backend
        self.pcd = self.o3d.t.geometry.PointCloud(device)
        self.added = False
        self.point_count = 0
        self.capacity = 0
        self.min_capacity = int(min_capacity)
        self._points_buffer = PointsFloat32InplaceBuffer()
        self._colors_buffer = ColorFloat32InplaceBuffer()
        self._refs: dict[str, np.ndarray | None] = {"points": None, "colors": None}

    def update(self, points_xyz_m: np.ndarray, colors_rgb_u8: np.ndarray) -> RenderLayerUpdate:
        format_start_s = time.perf_counter()
        if points_xyz_m.ndim != 2 or points_xyz_m.shape[1] != 3:
            raise ValueError("points_xyz_m must be an Nx3 array")
        if colors_rgb_u8.ndim != 2 or colors_rgb_u8.shape[1] != 3:
            raise ValueError("colors_rgb_u8 must be an Nx3 array")
        n_points = int(points_xyz_m.shape[0])
        if n_points == 0:
            remove_ms = 0.0
            if self.added:
                remove_start_s = time.perf_counter()
                try:
                    self.scene.remove_geometry(self.name)
                except Exception:
                    pass
                remove_ms = elapsed_ms(remove_start_s)
            self.added = False
            self.point_count = 0
            self.capacity = 0
            self._refs["points"] = None
            self._refs["colors"] = None
            return RenderLayerUpdate(
                points_count=0,
                colors_count=int(colors_rgb_u8.shape[0]),
                cpu_format_ms=elapsed_ms(format_start_s),
                open3d_points_update_ms=0.0,
                open3d_colors_update_ms=0.0,
                open3d_update_geometry_ms=0.0,
                open3d_remove_geometry_ms=remove_ms,
                geometry_recreated=remove_ms > 0.0,
            )

        if colors_rgb_u8.shape[0] != n_points:
            raise ValueError("points and colors must have the same length")

        old_capacity = int(self.capacity)
        next_capacity = max(old_capacity, n_points, int(self.min_capacity))
        capacity_changed = next_capacity != old_capacity
        points_update_start_s = time.perf_counter()
        if self.backend == RENDER_BACKEND_LEGACY_INPLACE:
            points = self._points_buffer.ensure(next_capacity)
            np.copyto(points[:n_points], points_xyz_m, casting="unsafe")
            if next_capacity > n_points:
                points[n_points:, 0:2] = np.float32(0.0)
                points[n_points:, 2] = np.float32(-1.0)
            tensor_rebound = capacity_changed or self._refs["points"] is None
            if tensor_rebound:
                self._refs["points"] = points
                self.pcd.point.positions = self.o3c.Tensor.from_numpy(points)
        else:
            points = np.ascontiguousarray(points_xyz_m, dtype=np.float32)
            self._refs["points"] = points
            self.pcd.point.positions = self.o3c.Tensor.from_numpy(points)
            tensor_rebound = True
        points_update_ms = elapsed_ms(points_update_start_s)

        colors_update_start_s = time.perf_counter()
        if self.backend == RENDER_BACKEND_LEGACY_INPLACE:
            colors = self._colors_buffer.ensure(next_capacity)
            np.multiply(colors_rgb_u8, np.float32(1.0 / 255.0), out=colors[:n_points], casting="unsafe")
            if next_capacity > n_points:
                colors[n_points:] = np.float32(0.0)
            colors_rebound = capacity_changed or self._refs["colors"] is None
            if colors_rebound:
                self._refs["colors"] = colors
                self.pcd.point.colors = self.o3c.Tensor.from_numpy(colors)
            tensor_rebound = tensor_rebound or colors_rebound
        else:
            colors = self._colors_buffer.convert_into(colors_rgb_u8).copy()
            self._refs["colors"] = colors
            self.pcd.point.colors = self.o3c.Tensor.from_numpy(colors)
            tensor_rebound = True
        colors_update_ms = elapsed_ms(colors_update_start_s)

        cpu_format_ms = elapsed_ms(format_start_s)
        update_start_s = time.perf_counter()
        add_ms = 0.0
        remove_ms = 0.0
        recreated = False
        if self.backend == RENDER_BACKEND_LEGACY_INPLACE:
            needs_readd = not self.added or capacity_changed
        else:
            point_count_changed = self.added and n_points != self.point_count
            needs_readd = not self.added or point_count_changed
        if needs_readd:
            if self.added:
                remove_start_s = time.perf_counter()
                try:
                    self.scene.remove_geometry(self.name)
                except Exception:
                    pass
                remove_ms = elapsed_ms(remove_start_s)
            add_start_s = time.perf_counter()
            self.scene.add_geometry(self.name, self.pcd, self.material)
            add_ms = elapsed_ms(add_start_s)
            self.added = True
            recreated = True
        else:
            flags = self.rendering.Scene.UPDATE_POINTS_FLAG | self.rendering.Scene.UPDATE_COLORS_FLAG
            try:
                self.scene.scene.update_geometry(self.name, self.pcd, flags)
            except Exception:
                remove_start_s = time.perf_counter()
                try:
                    self.scene.remove_geometry(self.name)
                except Exception:
                    pass
                remove_ms = elapsed_ms(remove_start_s)
                add_start_s = time.perf_counter()
                self.scene.add_geometry(self.name, self.pcd, self.material)
                add_ms = elapsed_ms(add_start_s)
                recreated = True
        update_ms = elapsed_ms(update_start_s)
        self.point_count = n_points
        self.capacity = next_capacity
        return RenderLayerUpdate(
            points_count=n_points,
            colors_count=int(colors_rgb_u8.shape[0]),
            cpu_format_ms=cpu_format_ms,
            open3d_points_update_ms=points_update_ms,
            open3d_colors_update_ms=colors_update_ms,
            open3d_update_geometry_ms=update_ms,
            open3d_add_geometry_ms=add_ms,
            open3d_remove_geometry_ms=remove_ms,
            geometry_recreated=recreated,
            tensor_rebound=tensor_rebound,
        )


@dataclass(frozen=True)
class RenderMicroProfileRecord:
    render_packet_id: int
    points_count: int
    colors_count: int
    queue_wait_ms: float = 0.0
    gpu_to_cpu_copy_ms: float = 0.0
    cpu_format_ms: float = 0.0
    open3d_points_update_ms: float = 0.0
    open3d_colors_update_ms: float = 0.0
    open3d_update_geometry_ms: float = 0.0
    open3d_poll_events_ms: float = 0.0
    open3d_update_renderer_ms: float = 0.0
    render_total_ms: float = 0.0
    backpressure: bool = False
    backend: str = DEFAULT_RENDER_BACKEND
    backend_effective: str = DEFAULT_RENDER_BACKEND
    geometry_recreated: bool = False
    tensor_rebound: bool = False
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "render_packet_id": int(self.render_packet_id),
            "points_count": int(self.points_count),
            "colors_count": int(self.colors_count),
            "queue_wait_ms": float(self.queue_wait_ms),
            "gpu_to_cpu_copy_ms": float(self.gpu_to_cpu_copy_ms),
            "cpu_format_ms": float(self.cpu_format_ms),
            "open3d_points_update_ms": float(self.open3d_points_update_ms),
            "open3d_colors_update_ms": float(self.open3d_colors_update_ms),
            "open3d_update_geometry_ms": float(self.open3d_update_geometry_ms),
            "open3d_poll_events_ms": float(self.open3d_poll_events_ms),
            "open3d_update_renderer_ms": float(self.open3d_update_renderer_ms),
            "render_total_ms": float(self.render_total_ms),
            "backpressure": bool(self.backpressure),
            "backend": self.backend,
            "backend_effective": self.backend_effective,
            "geometry_recreated": bool(self.geometry_recreated),
            "tensor_rebound": bool(self.tensor_rebound),
        }
        payload.update(dict(self.extra))
        return payload


class RenderMicroProfiler:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: list[dict[str, Any]] = []

    def record(self, record: RenderMicroProfileRecord | Mapping[str, Any]) -> None:
        payload = record.to_dict() if isinstance(record, RenderMicroProfileRecord) else dict(record)
        with self._lock:
            self._records.append(payload)

    def records(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(record) for record in self._records]

    def summary(self) -> dict[str, Any]:
        return summarize_render_records(self.records())


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _stats(values: Iterable[float]) -> dict[str, float]:
    clean = [float(value) for value in values]
    if not clean:
        return {"count": 0, "median": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "count": int(len(clean)),
        "median": _percentile(clean, 50.0),
        "p50": _percentile(clean, 50.0),
        "p90": _percentile(clean, 90.0),
        "p95": _percentile(clean, 95.0),
        "max": float(max(clean)),
    }


def summarize_render_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = {
        "render_copy_ms": _stats(float(record.get("gpu_to_cpu_copy_ms", 0.0) or 0.0) for record in records),
        "render_combine_ms": _stats(float(record.get("combine_ms", 0.0) or 0.0) for record in records),
        "render_cpu_format_ms": _stats(float(record.get("cpu_format_ms", 0.0) or 0.0) for record in records),
        "render_open3d_points_update_ms": _stats(float(record.get("open3d_points_update_ms", 0.0) or 0.0) for record in records),
        "render_open3d_colors_update_ms": _stats(float(record.get("open3d_colors_update_ms", 0.0) or 0.0) for record in records),
        "render_open3d_update_ms": _stats(float(record.get("open3d_update_geometry_ms", 0.0) or 0.0) for record in records),
        "render_poll_update_ms": _stats(
            float(record.get("open3d_poll_events_ms", 0.0) or 0.0)
            + float(record.get("open3d_update_renderer_ms", 0.0) or 0.0)
            for record in records
        ),
        "render_total_ms": _stats(float(record.get("render_total_ms", 0.0) or 0.0) for record in records),
        "render_points_count": _stats(float(record.get("points_count", 0.0) or 0.0) for record in records),
    }
    return {
        "render_packets_received": int(len(records)),
        "render_packets_displayed": int(len(records)),
        "render_packets_dropped": 0,
        "render_backpressure_count": int(sum(1 for record in records if bool(record.get("backpressure", False)))),
        "geometry_recreated_count": int(sum(1 for record in records if bool(record.get("geometry_recreated", False)))),
        "tensor_rebound_count": int(sum(1 for record in records if bool(record.get("tensor_rebound", False)))),
        "metrics": metrics,
    }


def torch_to_o3d_tensor_dlpack(tensor: Any) -> Any:
    """Convert a torch tensor to an Open3D tensor through DLPack.

    This helper is intentionally lazy-imported so deterministic tests do not
    require Open3D or torch. The caller must keep the source tensor alive until
    Open3D has consumed the updated geometry.
    """

    import open3d as o3d  # type: ignore
    import torch  # type: ignore

    if not torch.is_tensor(tensor):
        raise TypeError("tensor must be a torch.Tensor")
    return o3d.core.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(tensor.contiguous()))


def write_render_profile_summary(
    *,
    records: Sequence[Mapping[str, Any]],
    output_json: Path,
    output_md: Path | None = None,
    title: str = "Demo 2.2 render backend micro-profile",
) -> dict[str, Any]:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    summary = summarize_render_records(records)
    payload = {"title": title, "summary": summary, "records": [dict(record) for record in records]}
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if output_md is not None:
        metrics = summary.get("metrics", {})
        lines = [
            f"# {title}",
            "",
            f"- packets displayed: `{summary.get('render_packets_displayed', 0)}`",
            f"- backpressure count: `{summary.get('render_backpressure_count', 0)}`",
            f"- geometry recreated count: `{summary.get('geometry_recreated_count', 0)}`",
            "",
            "| Metric | p50 | p90 | p95 | max |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for name in (
            "render_copy_ms",
            "render_combine_ms",
            "render_cpu_format_ms",
            "render_open3d_update_ms",
            "render_poll_update_ms",
            "render_total_ms",
            "render_points_count",
        ):
            stat = metrics.get(name, {})
            lines.append(
                f"| `{name}` | `{stat.get('p50', 0.0):.2f}` | `{stat.get('p90', 0.0):.2f}` | "
                f"`{stat.get('p95', 0.0):.2f}` | `{stat.get('max', 0.0):.2f}` |"
            )
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload
