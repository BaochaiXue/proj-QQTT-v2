"""Open3D GUI helpers: WSLG env setup, module loading, tensor point-cloud layer."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from demo_v5_1.utils.concurrency import elapsed_ms

WSLG_OPEN3D_ENV_UNSET_KEYS = (
    "VK_ICD_FILENAMES",
    "__GLX_VENDOR_LIBRARY_NAME",
    "__EGL_VENDOR_LIBRARY_FILENAMES",
)
WSLG_OPEN3D_ENV_DEFAULTS = {
    "WAYLAND_DISPLAY": "",
    "EGL_PLATFORM": "x11",
    "GALLIUM_DRIVER": "d3d12",
    "MESA_LOADER_DRIVER_OVERRIDE": "d3d12",
    "LIBGL_ALWAYS_SOFTWARE": "0",
    "QQTT_WSLG_OPEN3D_FAST_EXIT": "1",
    "MESA_D3D12_DEFAULT_ADAPTER_NAME": "NVIDIA",
}


def running_under_wsl() -> bool:
    if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"):
        return True
    try:
        return "microsoft" in Path("/proc/version").read_text(encoding="utf-8").lower()
    except OSError:
        return False


def apply_wslg_open3d_env_defaults() -> dict[str, str]:
    if os.environ.get("QQTT_DISABLE_WSLG_OPEN3D_DEFAULTS") == "1":
        return {}
    if not running_under_wsl():
        return {}

    applied: dict[str, str] = {}
    for key in WSLG_OPEN3D_ENV_UNSET_KEYS:
        if key in os.environ:
            os.environ.pop(key, None)
            applied[key] = "<unset>"
    for key, value in WSLG_OPEN3D_ENV_DEFAULTS.items():
        if key == "MESA_D3D12_DEFAULT_ADAPTER_NAME" and key in os.environ:
            continue
        if os.environ.get(key) != value:
            os.environ[key] = value
            applied[key] = value
    return applied


def load_open3d_modules():
    apply_wslg_open3d_env_defaults()
    try:
        import open3d as o3d  # type: ignore
        from open3d.visualization import gui, rendering  # type: ignore
    except ImportError as exc:
        raise RuntimeError("open3d is required to render the realtime point cloud") from exc
    return o3d, gui, rendering


class RenderStats:
    def __init__(self, window_s: float = 1.0) -> None:
        if window_s <= 0:
            raise ValueError("window_s must be positive")
        self.window_s = float(window_s)
        self._samples: deque[tuple[float, float]] = deque()
        self.latest_latency_ms = 0.0

    def record_render(self, *, render_time_s: float, latency_ms: float) -> None:
        self.latest_latency_ms = float(latency_ms)
        self._samples.append((float(render_time_s), float(latency_ms)))
        cutoff = float(render_time_s) - self.window_s
        while len(self._samples) > 1 and self._samples[0][0] < cutoff:
            self._samples.popleft()

    @property
    def render_fps(self) -> float:
        if len(self._samples) < 2:
            return 0.0
        elapsed = self._samples[-1][0] - self._samples[0][0]
        if elapsed <= 0:
            return 0.0
        return float((len(self._samples) - 1) / elapsed)

    @property
    def mean_latency_ms(self) -> float:
        if not self._samples:
            return 0.0
        return float(sum(latency for _, latency in self._samples) / len(self._samples))


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


class _Float32Buffer:
    """Reusable Nx3 float32 buffer for in-place point/color updates."""

    def __init__(self) -> None:
        self.array: np.ndarray | None = None

    def ensure(self, n_points: int) -> np.ndarray:
        if self.array is None or self.array.shape != (n_points, 3):
            self.array = np.empty((n_points, 3), dtype=np.float32)
        return self.array


class Open3DSceneTensorLayer:
    """Open3D GUI Scene tensor point-cloud layer with an in-place no-recreate fast path."""

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
        min_capacity: int = 0,
    ) -> None:
        if int(min_capacity) < 0:
            raise ValueError("min_capacity must be >= 0")
        self.name = str(name)
        self.o3d = o3d_module
        self.o3c = o3c_module
        self.rendering = rendering_module
        self.scene = scene
        self.material = material
        self.pcd = self.o3d.t.geometry.PointCloud(device)
        self.added = False
        self.point_count = 0
        self.capacity = 0
        self.min_capacity = int(min_capacity)
        self._points_buffer = _Float32Buffer()
        self._colors_buffer = _Float32Buffer()
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
                self.scene.remove_geometry(self.name)
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
        points = self._points_buffer.ensure(next_capacity)
        np.copyto(points[:n_points], points_xyz_m, casting="unsafe")
        if next_capacity > n_points:
            points[n_points:, 0:2] = np.float32(0.0)
            points[n_points:, 2] = np.float32(-1.0)
        tensor_rebound = capacity_changed or self._refs["points"] is None
        if tensor_rebound:
            self._refs["points"] = points
            self.pcd.point.positions = self.o3c.Tensor.from_numpy(points)
        points_update_ms = elapsed_ms(points_update_start_s)

        colors_update_start_s = time.perf_counter()
        colors = self._colors_buffer.ensure(next_capacity)
        np.multiply(colors_rgb_u8, np.float32(1.0 / 255.0), out=colors[:n_points], casting="unsafe")
        if next_capacity > n_points:
            colors[n_points:] = np.float32(0.0)
        colors_rebound = capacity_changed or self._refs["colors"] is None
        if colors_rebound:
            self._refs["colors"] = colors
            self.pcd.point.colors = self.o3c.Tensor.from_numpy(colors)
        tensor_rebound = tensor_rebound or colors_rebound
        colors_update_ms = elapsed_ms(colors_update_start_s)

        cpu_format_ms = elapsed_ms(format_start_s)
        update_start_s = time.perf_counter()
        add_ms = 0.0
        remove_ms = 0.0
        recreated = False
        if not self.added or capacity_changed:
            if self.added:
                remove_start_s = time.perf_counter()
                self.scene.remove_geometry(self.name)
                remove_ms = elapsed_ms(remove_start_s)
            add_start_s = time.perf_counter()
            self.scene.add_geometry(self.name, self.pcd, self.material)
            add_ms = elapsed_ms(add_start_s)
            self.added = True
            recreated = True
        else:
            flags = self.rendering.Scene.UPDATE_POINTS_FLAG | self.rendering.Scene.UPDATE_COLORS_FLAG
            self.scene.scene.update_geometry(self.name, self.pcd, flags)
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
