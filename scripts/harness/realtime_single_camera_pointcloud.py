from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, replace
import math
from pathlib import Path
import sys
import threading
import time
from typing import Callable, Generic, TypeVar

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_process.depth_backends import DEFAULT_FFS_REPO, DEFAULT_FFS_TRT_TWO_STAGE_MODEL_DIR

try:
    import cv2  # type: ignore
except ImportError:
    cv2 = None

try:
    from numba import njit  # type: ignore
except ImportError:
    njit = None


SUPPORTED_CAPTURE_FPS = (5, 15, 30, 60)
SUPPORTED_PROFILES = ("848x480", "640x480")
BACKPROJECT_BACKENDS = ("auto", "numpy", "numba")
DEPTH_SOURCES = ("realsense", "ffs")
DEFAULT_PROFILE = "848x480"
DEFAULT_FPS = 60
DEFAULT_CAMERA_MAX_POINTS = 0
DEFAULT_ORBIT_MAX_POINTS = 200000
COORDINATE_FRAME = "camera_color_frame"
GEOMETRY_NAME = "single_d455_live_pointcloud"
DEBUG_LOG_INTERVAL_S = 1.0


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class PipelineTiming:
    wait_ms: float = 0.0
    align_ms: float = 0.0
    frame_copy_ms: float = 0.0
    ffs_ms: float = 0.0
    ffs_align_ms: float = 0.0
    image_mask_ms: float = 0.0
    backproject_ms: float = 0.0
    open3d_convert_ms: float = 0.0
    open3d_update_ms: float = 0.0
    receive_to_render_ms: float = 0.0


@dataclass(frozen=True)
class FramePacket:
    seq: int
    color_bgr: np.ndarray
    intrinsics: CameraIntrinsics
    receive_perf_s: float
    timing: PipelineTiming
    depth_source: str
    depth_u16: np.ndarray | None = None
    depth_scale_m_per_unit: float = 0.0
    ir_left_u8: np.ndarray | None = None
    ir_right_u8: np.ndarray | None = None
    k_ir_left: np.ndarray | None = None
    t_ir_left_to_color: np.ndarray | None = None
    ir_baseline_m: float = 0.0


@dataclass(frozen=True)
class DepthFramePacket:
    seq: int
    color_bgr: np.ndarray
    depth_source: str
    intrinsics: CameraIntrinsics
    receive_perf_s: float
    process_done_perf_s: float
    dropped_capture_frames: int
    dropped_ffs_frames: int
    timing: PipelineTiming
    depth_m: np.ndarray | None = None
    depth_u16: np.ndarray | None = None
    depth_scale_m_per_unit: float = 0.0


@dataclass(frozen=True)
class RawFfsDepthPacket:
    seq: int
    color_bgr: np.ndarray
    depth_left_m: np.ndarray
    intrinsics: CameraIntrinsics
    k_ir_left: np.ndarray
    t_ir_left_to_color: np.ndarray
    k_color: np.ndarray
    receive_perf_s: float
    ffs_done_perf_s: float
    dropped_capture_frames: int
    dropped_ffs_frames: int
    timing: PipelineTiming
    depth_source: str = "ffs"


@dataclass(frozen=True)
class PointCloudPacket:
    seq: int
    points_xyz_m: np.ndarray
    colors_rgb_u8: np.ndarray
    backproject_backend: str
    depth_source: str
    receive_perf_s: float
    process_done_perf_s: float
    dropped_capture_frames: int
    dropped_ffs_frames: int
    timing: PipelineTiming

    @property
    def point_count(self) -> int:
        return int(self.points_xyz_m.shape[0])


@dataclass(frozen=True)
class ImagePacket:
    seq: int
    image_rgb_u8: np.ndarray
    valid_count: int
    depth_source: str
    receive_perf_s: float
    process_done_perf_s: float
    dropped_capture_frames: int
    dropped_ffs_frames: int
    timing: PipelineTiming

    @property
    def point_count(self) -> int:
        return self.valid_count


T = TypeVar("T")


class LatestSlot(Generic[T]):
    """Thread-safe single-slot latest-wins buffer."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._packet: T | None = None
        self._last_taken_seq = -1
        self._dropped = 0

    def put(self, packet: T) -> int:
        seq = _packet_seq(packet)
        with self._lock:
            if self._packet is not None:
                current_seq = _packet_seq(self._packet)
                if current_seq > self._last_taken_seq:
                    self._dropped += max(1, seq - current_seq)
            self._packet = packet
            return self._dropped

    def get_latest_after(self, last_seq: int) -> T | None:
        with self._lock:
            if self._packet is None:
                return None
            seq = _packet_seq(self._packet)
            if seq <= last_seq:
                return None
            self._last_taken_seq = seq
            return self._packet

    @property
    def dropped_count(self) -> int:
        with self._lock:
            return self._dropped


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


def depth_to_render_ms(timing: PipelineTiming) -> float:
    return float(
        timing.align_ms
        + timing.frame_copy_ms
        + timing.ffs_ms
        + timing.ffs_align_ms
        + timing.image_mask_ms
        + timing.backproject_ms
        + timing.open3d_convert_ms
        + timing.open3d_update_ms
    )


def _elapsed_ms(start_s: float, end_s: float) -> float:
    return float((end_s - start_s) * 1000.0)


def _packet_seq(packet: object) -> int:
    try:
        return int(getattr(packet, "seq"))
    except AttributeError as exc:
        raise TypeError("latest-slot packets must expose an integer seq attribute") from exc


class ColorFloat32Buffer:
    """Reusable RGB uint8 -> float32 [0, 1] conversion buffer."""

    def __init__(self) -> None:
        self._arr: np.ndarray | None = None

    def convert(self, colors_rgb_u8: np.ndarray) -> np.ndarray:
        if colors_rgb_u8.ndim != 2 or colors_rgb_u8.shape[1] != 3:
            raise ValueError("colors_rgb_u8 must be an Nx3 array")
        n_points = colors_rgb_u8.shape[0]
        arr = self._arr
        if arr is None or arr.shape != (n_points, 3):
            arr = np.empty((n_points, 3), dtype=np.float32)
            self._arr = arr
        np.multiply(colors_rgb_u8, np.float32(1.0 / 255.0), out=arr, casting="unsafe")
        return arr


def ensure_float32_c_contiguous(points_xyz_m: np.ndarray) -> np.ndarray:
    if points_xyz_m.ndim != 2 or points_xyz_m.shape[1] != 3:
        raise ValueError("points_xyz_m must be an Nx3 array")
    if points_xyz_m.dtype == np.float32 and points_xyz_m.flags["C_CONTIGUOUS"]:
        return points_xyz_m
    return np.ascontiguousarray(points_xyz_m, dtype=np.float32)


def parse_profile(profile: str) -> tuple[int, int]:
    if profile not in SUPPORTED_PROFILES:
        raise ValueError(f"unsupported profile {profile!r}; expected one of {', '.join(SUPPORTED_PROFILES)}")
    width_text, height_text = profile.split("x", 1)
    return int(width_text), int(height_text)


def build_pixel_grid(*, width: int, height: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if stride < 1:
        raise ValueError("stride must be >= 1")
    xs = np.arange(0, width, stride, dtype=np.float32)
    ys = np.arange(0, height, stride, dtype=np.float32)
    return np.meshgrid(xs, ys, indexing="xy")


def build_projection_grid(
    *,
    width: int,
    height: int,
    stride: int,
    intrinsics: CameraIntrinsics,
) -> tuple[np.ndarray, np.ndarray]:
    if intrinsics.fx <= 0 or intrinsics.fy <= 0:
        raise ValueError("intrinsics fx/fy must be positive")
    grid_x, grid_y = build_pixel_grid(width=width, height=height, stride=stride)
    ray_x = (grid_x - np.float32(intrinsics.cx)) / np.float32(intrinsics.fx)
    ray_y = (grid_y - np.float32(intrinsics.cy)) / np.float32(intrinsics.fy)
    return np.ascontiguousarray(ray_x, dtype=np.float32), np.ascontiguousarray(ray_y, dtype=np.float32)


def camera_intrinsics_from_rs(intrinsics: object) -> CameraIntrinsics:
    return CameraIntrinsics(
        fx=float(getattr(intrinsics, "fx")),
        fy=float(getattr(intrinsics, "fy")),
        cx=float(getattr(intrinsics, "ppx")),
        cy=float(getattr(intrinsics, "ppy")),
    )


def camera_intrinsics_to_matrix(intrinsics: CameraIntrinsics) -> np.ndarray:
    return np.array(
        [
            [float(intrinsics.fx), 0.0, float(intrinsics.cx)],
            [0.0, float(intrinsics.fy), float(intrinsics.cy)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def rs_intrinsics_to_matrix(intrinsics: object) -> np.ndarray:
    return camera_intrinsics_to_matrix(camera_intrinsics_from_rs(intrinsics))


def rs_extrinsics_to_matrix(extrinsics: object) -> np.ndarray:
    rotation = list(map(float, getattr(extrinsics, "rotation")))
    translation = list(map(float, getattr(extrinsics, "translation")))
    return np.array(
        [
            [rotation[0], rotation[1], rotation[2], translation[0]],
            [rotation[3], rotation[4], rotation[5], translation[1]],
            [rotation[6], rotation[7], rotation[8], translation[2]],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def rs_translation_norm(extrinsics: object) -> float:
    tx, ty, tz = map(float, getattr(extrinsics, "translation"))
    return float(math.sqrt(tx * tx + ty * ty + tz * tz))


def align_ir_depth_to_color_fast(
    depth_ir_m: np.ndarray,
    K_ir_left: np.ndarray,
    T_ir_left_to_color: np.ndarray,
    K_color: np.ndarray,
    output_shape: tuple[int, int],
    invalid_value: float = 0.0,
) -> np.ndarray:
    depth = np.asarray(depth_ir_m, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError("depth_ir_m must be a 2D array")
    k_ir = np.asarray(K_ir_left, dtype=np.float32).reshape(3, 3)
    k_color = np.asarray(K_color, dtype=np.float32).reshape(3, 3)
    transform = np.asarray(T_ir_left_to_color, dtype=np.float32).reshape(4, 4)
    out_height, out_width = output_shape
    if out_height <= 0 or out_width <= 0:
        raise ValueError("output_shape must be positive")

    valid = np.isfinite(depth) & (depth > 0.0)
    if not np.any(valid):
        return np.full((out_height, out_width), np.float32(invalid_value), dtype=np.float32)

    rows, cols = np.nonzero(valid)
    z = depth[rows, cols].astype(np.float32, copy=False)
    x = (cols.astype(np.float32) - np.float32(k_ir[0, 2])) * z / np.float32(k_ir[0, 0])
    y = (rows.astype(np.float32) - np.float32(k_ir[1, 2])) * z / np.float32(k_ir[1, 1])

    r = transform[:3, :3]
    t = transform[:3, 3]
    x_color = r[0, 0] * x + r[0, 1] * y + r[0, 2] * z + t[0]
    y_color = r[1, 0] * x + r[1, 1] * y + r[1, 2] * z + t[1]
    z_color = r[2, 0] * x + r[2, 1] * y + r[2, 2] * z + t[2]

    front = np.isfinite(z_color) & (z_color > 0.0)
    if not np.any(front):
        return np.full((out_height, out_width), np.float32(invalid_value), dtype=np.float32)

    x_color = x_color[front]
    y_color = y_color[front]
    z_color = z_color[front]
    u = np.rint((x_color / z_color) * np.float32(k_color[0, 0]) + np.float32(k_color[0, 2])).astype(np.int32)
    v = np.rint((y_color / z_color) * np.float32(k_color[1, 1]) + np.float32(k_color[1, 2])).astype(np.int32)

    inside = (u >= 0) & (u < out_width) & (v >= 0) & (v < out_height)
    if not np.any(inside):
        return np.full((out_height, out_width), np.float32(invalid_value), dtype=np.float32)

    flat_idx = (v[inside] * out_width + u[inside]).astype(np.int64, copy=False)
    z_inside = z_color[inside].astype(np.float32, copy=False)
    nearest = np.full(out_height * out_width, np.inf, dtype=np.float32)
    np.minimum.at(nearest, flat_idx, z_inside)
    output = nearest.reshape(out_height, out_width)
    output[~np.isfinite(output)] = np.float32(invalid_value)
    return np.ascontiguousarray(output, dtype=np.float32)


def _depth_bounds_to_u16(
    *,
    depth_scale_m_per_unit: float,
    depth_min_m: float,
    depth_max_m: float,
) -> tuple[int, int]:
    if not math.isfinite(depth_scale_m_per_unit) or depth_scale_m_per_unit <= 0:
        raise ValueError("depth_scale_m_per_unit must be finite and positive")
    if not math.isfinite(depth_min_m) or depth_min_m < 0:
        raise ValueError("depth_min_m must be finite and >= 0")
    if not math.isfinite(depth_max_m):
        raise ValueError("depth_max_m must be finite")
    if depth_max_m > 0 and depth_max_m <= depth_min_m:
        raise ValueError("expected depth_max_m <= 0, or depth_max_m > depth_min_m")

    u16_max = int(np.iinfo(np.uint16).max)
    scale32 = np.float32(depth_scale_m_per_unit)
    if not np.isfinite(scale32) or scale32 <= 0:
        raise ValueError("depth_scale_m_per_unit must be representable as finite positive float32")

    def scaled_m(raw_depth: int) -> float:
        return float(np.float32(raw_depth) * scale32)

    if scaled_m(u16_max) < depth_min_m:
        return u16_max + 1, u16_max
    lower = max(1, min(u16_max, int(math.floor(depth_min_m / float(scale32)))))
    while lower > 1 and scaled_m(lower - 1) >= depth_min_m:
        lower -= 1
    while lower <= u16_max and scaled_m(lower) < depth_min_m:
        lower += 1
    if lower > u16_max:
        return u16_max + 1, u16_max

    upper = u16_max
    if depth_max_m > 0:
        if scaled_m(1) > depth_max_m:
            return lower, 0
        upper = max(0, min(u16_max, int(math.ceil(depth_max_m / float(scale32)))))
        while upper < u16_max and scaled_m(upper + 1) <= depth_max_m:
            upper += 1
        while upper >= 0 and scaled_m(upper) > depth_max_m:
            upper -= 1
    return lower, upper


if njit is not None:

    @njit
    def _backproject_numba_rank_sample(
        depth_u16: np.ndarray,
        color_bgr: np.ndarray,
        ray_x: np.ndarray,
        ray_y: np.ndarray,
        depth_scale: np.float32,
        lower_u16: int,
        upper_u16: int,
        max_points: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        height, width = depth_u16.shape
        n_valid = 0
        for y in range(height):
            for x in range(width):
                depth = int(depth_u16[y, x])
                if lower_u16 <= depth <= upper_u16:
                    n_valid += 1

        if n_valid == 0:
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

        if max_points > 0 and n_valid > max_points:
            n_out = max_points
        else:
            n_out = n_valid

        points = np.empty((n_out, 3), dtype=np.float32)
        colors = np.empty((n_out, 3), dtype=np.uint8)
        valid_rank = 0
        out_idx = 0

        for y in range(height):
            for x in range(width):
                depth = int(depth_u16[y, x])
                if lower_u16 <= depth <= upper_u16:
                    take = False
                    if n_out == n_valid:
                        take = True
                    elif n_out == 1:
                        take = valid_rank == 0
                    else:
                        target_rank = (out_idx * (n_valid - 1)) // (n_out - 1)
                        take = valid_rank == target_rank

                    if take and out_idx < n_out:
                        z = np.float32(depth) * depth_scale
                        points[out_idx, 0] = ray_x[y, x] * z
                        points[out_idx, 1] = ray_y[y, x] * z
                        points[out_idx, 2] = z
                        colors[out_idx, 0] = color_bgr[y, x, 2]
                        colors[out_idx, 1] = color_bgr[y, x, 1]
                        colors[out_idx, 2] = color_bgr[y, x, 0]
                        out_idx += 1

                    valid_rank += 1

        return points, colors

else:
    _backproject_numba_rank_sample = None


def numba_backprojection_available() -> bool:
    return _backproject_numba_rank_sample is not None


def resolve_backproject_backend(
    requested_backend: str,
    *,
    stride: int,
    projection_grid_available: bool,
) -> str:
    if requested_backend not in BACKPROJECT_BACKENDS:
        raise ValueError(f"unsupported backproject backend {requested_backend!r}")
    can_use_numba = numba_backprojection_available() and stride == 1 and projection_grid_available
    if requested_backend == "numpy":
        return "numpy"
    if requested_backend == "numba":
        if not numba_backprojection_available():
            raise ValueError("--backproject-backend numba requires numba to be installed")
        if stride != 1:
            raise ValueError("--backproject-backend numba requires --stride 1")
        if not projection_grid_available:
            raise ValueError("--backproject-backend numba requires a precomputed projection grid")
        return "numba"
    return "numba" if can_use_numba else "numpy"


def warm_up_numba_backprojection() -> None:
    if _backproject_numba_rank_sample is None:
        return
    depth = np.array([[1]], dtype=np.uint16)
    color = np.zeros((1, 1, 3), dtype=np.uint8)
    ray_x = np.zeros((1, 1), dtype=np.float32)
    ray_y = np.zeros((1, 1), dtype=np.float32)
    _backproject_numba_rank_sample(depth, color, ray_x, ray_y, np.float32(0.001), 1, 1, 1)


def build_camera_view_image(
    *,
    color_bgr: np.ndarray,
    depth_u16: np.ndarray,
    depth_scale_m_per_unit: float,
    depth_min_m: float,
    depth_max_m: float,
    splat_px: int = 0,
) -> tuple[np.ndarray, int]:
    if depth_u16.ndim != 2:
        raise ValueError("depth_u16 must be a 2D array")
    if color_bgr.ndim != 3 or color_bgr.shape[2] != 3:
        raise ValueError("color_bgr must be an HxWx3 array")
    if color_bgr.shape[:2] != depth_u16.shape:
        raise ValueError("color and depth shapes must match after depth-to-color alignment")
    if splat_px < 0:
        raise ValueError("splat_px must be >= 0")

    lower_u16, upper_u16 = _depth_bounds_to_u16(
        depth_scale_m_per_unit=depth_scale_m_per_unit,
        depth_min_m=depth_min_m,
        depth_max_m=depth_max_m,
    )
    image_rgb = np.zeros(color_bgr.shape, dtype=np.uint8)
    if lower_u16 > upper_u16:
        return image_rgb, 0

    if splat_px == 0 and cv2 is not None:
        valid_mask_u8 = cv2.inRange(depth_u16, lower_u16, upper_u16)
        valid_count = int(cv2.countNonZero(valid_mask_u8))
        if valid_count == 0:
            return image_rgb, 0
        rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(cv2.bitwise_and(rgb, rgb, mask=valid_mask_u8)), valid_count

    valid = (depth_u16 >= lower_u16) & (depth_u16 <= upper_u16)
    valid_count = int(np.count_nonzero(valid))
    if valid_count == 0:
        return image_rgb, 0

    source_rgb = np.ascontiguousarray(color_bgr[:, :, ::-1])
    if splat_px == 0:
        source_rgb[~valid] = 0
        return source_rgb, valid_count

    _splat_valid_rgb(image_rgb=image_rgb, source_rgb=source_rgb, valid=valid, radius=splat_px)
    return np.ascontiguousarray(image_rgb), valid_count


def _valid_float_depth_mask(*, depth_m: np.ndarray, depth_min_m: float, depth_max_m: float) -> np.ndarray:
    if depth_min_m < 0:
        raise ValueError("depth_min_m must be >= 0")
    if depth_max_m > 0 and depth_max_m <= depth_min_m:
        raise ValueError("expected depth_max_m <= 0, or depth_max_m > depth_min_m")
    valid = np.isfinite(depth_m) & (depth_m > 0.0) & (depth_m >= np.float32(depth_min_m))
    if depth_max_m > 0:
        valid &= depth_m <= np.float32(depth_max_m)
    return valid


def build_camera_view_image_from_depth_m(
    *,
    color_bgr: np.ndarray,
    depth_m: np.ndarray,
    depth_min_m: float,
    depth_max_m: float,
    splat_px: int = 0,
) -> tuple[np.ndarray, int]:
    if depth_m.ndim != 2:
        raise ValueError("depth_m must be a 2D array")
    if color_bgr.ndim != 3 or color_bgr.shape[2] != 3:
        raise ValueError("color_bgr must be an HxWx3 array")
    if color_bgr.shape[:2] != depth_m.shape:
        raise ValueError("color and depth shapes must match after depth-to-color alignment")
    if splat_px < 0:
        raise ValueError("splat_px must be >= 0")

    depth = np.asarray(depth_m, dtype=np.float32)
    valid = _valid_float_depth_mask(depth_m=depth, depth_min_m=depth_min_m, depth_max_m=depth_max_m)
    valid_count = int(np.count_nonzero(valid))
    image_rgb = np.zeros(color_bgr.shape, dtype=np.uint8)
    if valid_count == 0:
        return image_rgb, 0

    source_rgb = np.ascontiguousarray(color_bgr[:, :, ::-1])
    if splat_px == 0:
        source_rgb[~valid] = 0
        return source_rgb, valid_count

    _splat_valid_rgb(image_rgb=image_rgb, source_rgb=source_rgb, valid=valid, radius=splat_px)
    return np.ascontiguousarray(image_rgb), valid_count


def _splat_valid_rgb(*, image_rgb: np.ndarray, source_rgb: np.ndarray, valid: np.ndarray, radius: int) -> None:
    height, width = valid.shape
    for dy in range(-radius, radius + 1):
        src_y0 = max(0, -dy)
        src_y1 = min(height, height - dy)
        dst_y0 = max(0, dy)
        dst_y1 = min(height, height + dy)
        for dx in range(-radius, radius + 1):
            src_x0 = max(0, -dx)
            src_x1 = min(width, width - dx)
            dst_x0 = max(0, dx)
            dst_x1 = min(width, width + dx)
            source_mask = valid[src_y0:src_y1, src_x0:src_x1]
            if not np.any(source_mask):
                continue
            dst_view = image_rgb[dst_y0:dst_y1, dst_x0:dst_x1]
            dst_view[source_mask] = source_rgb[src_y0:src_y1, src_x0:src_x1][source_mask]


def backproject_aligned_rgbd(
    *,
    color_bgr: np.ndarray,
    depth_u16: np.ndarray,
    intrinsics: CameraIntrinsics,
    depth_scale_m_per_unit: float,
    depth_min_m: float,
    depth_max_m: float,
    stride: int,
    max_points: int,
    pixel_grid: tuple[np.ndarray, np.ndarray] | None = None,
    projection_grid: tuple[np.ndarray, np.ndarray] | None = None,
    backproject_backend: str = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    if depth_u16.ndim != 2:
        raise ValueError("depth_u16 must be a 2D array")
    if color_bgr.ndim != 3 or color_bgr.shape[2] != 3:
        raise ValueError("color_bgr must be an HxWx3 array")
    if color_bgr.shape[:2] != depth_u16.shape:
        raise ValueError("color and depth shapes must match after depth-to-color alignment")
    if depth_scale_m_per_unit <= 0:
        raise ValueError("depth_scale_m_per_unit must be positive")
    if depth_min_m < 0:
        raise ValueError("depth_min_m must be >= 0")
    if depth_max_m > 0 and depth_max_m <= depth_min_m:
        raise ValueError("expected depth_max_m <= 0, or depth_max_m > depth_min_m")
    if stride < 1:
        raise ValueError("stride must be >= 1")
    if max_points < 0:
        raise ValueError("max_points must be >= 0")
    effective_backend = resolve_backproject_backend(
        backproject_backend,
        stride=stride,
        projection_grid_available=projection_grid is not None,
    )

    depth_view = depth_u16[::stride, ::stride]
    color_view = color_bgr[::stride, ::stride, :]
    if projection_grid is not None:
        ray_x, ray_y = projection_grid
        if ray_x.shape != depth_view.shape or ray_y.shape != depth_view.shape:
            raise ValueError("projection_grid shape does not match strided depth shape")
    else:
        if pixel_grid is None:
            ray_x, ray_y = build_projection_grid(
                width=depth_u16.shape[1],
                height=depth_u16.shape[0],
                stride=stride,
                intrinsics=intrinsics,
            )
        else:
            grid_x, grid_y = pixel_grid
            if grid_x.shape != depth_view.shape or grid_y.shape != depth_view.shape:
                raise ValueError("pixel_grid shape does not match strided depth shape")
            ray_x = (grid_x - np.float32(intrinsics.cx)) / np.float32(intrinsics.fx)
            ray_y = (grid_y - np.float32(intrinsics.cy)) / np.float32(intrinsics.fy)

    lower_u16, upper_u16 = _depth_bounds_to_u16(
        depth_scale_m_per_unit=depth_scale_m_per_unit,
        depth_min_m=depth_min_m,
        depth_max_m=depth_max_m,
    )
    if lower_u16 > upper_u16:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    if effective_backend == "numba":
        assert _backproject_numba_rank_sample is not None
        return _backproject_numba_rank_sample(
            depth_view,
            color_view,
            ray_x,
            ray_y,
            np.float32(depth_scale_m_per_unit),
            int(lower_u16),
            int(upper_u16),
            int(max_points),
        )

    if stride == 1 and cv2 is not None:
        valid_mask = cv2.inRange(depth_view, lower_u16, upper_u16)
        valid_flat = np.flatnonzero(valid_mask.ravel())
    else:
        valid = (depth_view >= lower_u16) & (depth_view <= upper_u16)
        valid_flat = np.flatnonzero(valid.ravel())

    n_valid = int(valid_flat.size)
    if n_valid == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    if max_points > 0 and n_valid > max_points:
        indices = np.linspace(0, n_valid - 1, max_points, dtype=np.int64)
        valid_flat = valid_flat[indices]

    n_points = int(valid_flat.size)
    depth_flat = depth_view.ravel()
    ray_x_flat = ray_x.ravel()
    ray_y_flat = ray_y.ravel()

    z = depth_flat[valid_flat].astype(np.float32, copy=False) * np.float32(depth_scale_m_per_unit)
    points = np.empty((n_points, 3), dtype=np.float32)
    points[:, 0] = ray_x_flat[valid_flat] * z
    points[:, 1] = ray_y_flat[valid_flat] * z
    points[:, 2] = z

    if stride == 1 and color_bgr.flags["C_CONTIGUOUS"]:
        color_flat = color_bgr.reshape(-1, 3)
    else:
        color_flat = np.ascontiguousarray(color_view).reshape(-1, 3)

    colors_rgb_u8 = np.empty((n_points, 3), dtype=np.uint8)
    colors_rgb_u8[:, 0] = color_flat[valid_flat, 2]
    colors_rgb_u8[:, 1] = color_flat[valid_flat, 1]
    colors_rgb_u8[:, 2] = color_flat[valid_flat, 0]

    return points, colors_rgb_u8


def backproject_aligned_rgbd_depth_m(
    *,
    color_bgr: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: CameraIntrinsics,
    depth_min_m: float,
    depth_max_m: float,
    stride: int,
    max_points: int,
    pixel_grid: tuple[np.ndarray, np.ndarray] | None = None,
    projection_grid: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if depth_m.ndim != 2:
        raise ValueError("depth_m must be a 2D array")
    if color_bgr.ndim != 3 or color_bgr.shape[2] != 3:
        raise ValueError("color_bgr must be an HxWx3 array")
    if color_bgr.shape[:2] != depth_m.shape:
        raise ValueError("color and depth shapes must match after depth-to-color alignment")
    if depth_min_m < 0:
        raise ValueError("depth_min_m must be >= 0")
    if depth_max_m > 0 and depth_max_m <= depth_min_m:
        raise ValueError("expected depth_max_m <= 0, or depth_max_m > depth_min_m")
    if stride < 1:
        raise ValueError("stride must be >= 1")
    if max_points < 0:
        raise ValueError("max_points must be >= 0")

    depth_view = np.asarray(depth_m, dtype=np.float32)[::stride, ::stride]
    color_view = color_bgr[::stride, ::stride, :]
    if projection_grid is not None:
        ray_x, ray_y = projection_grid
        if ray_x.shape != depth_view.shape or ray_y.shape != depth_view.shape:
            raise ValueError("projection_grid shape does not match strided depth shape")
    else:
        if pixel_grid is None:
            ray_x, ray_y = build_projection_grid(
                width=depth_m.shape[1],
                height=depth_m.shape[0],
                stride=stride,
                intrinsics=intrinsics,
            )
        else:
            grid_x, grid_y = pixel_grid
            if grid_x.shape != depth_view.shape or grid_y.shape != depth_view.shape:
                raise ValueError("pixel_grid shape does not match strided depth shape")
            ray_x = (grid_x - np.float32(intrinsics.cx)) / np.float32(intrinsics.fx)
            ray_y = (grid_y - np.float32(intrinsics.cy)) / np.float32(intrinsics.fy)

    valid = _valid_float_depth_mask(depth_m=depth_view, depth_min_m=depth_min_m, depth_max_m=depth_max_m)
    valid_flat = np.flatnonzero(valid.ravel())
    n_valid = int(valid_flat.size)
    if n_valid == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    if max_points > 0 and n_valid > max_points:
        indices = np.linspace(0, n_valid - 1, max_points, dtype=np.int64)
        valid_flat = valid_flat[indices]

    n_points = int(valid_flat.size)
    depth_flat = depth_view.ravel()
    ray_x_flat = ray_x.ravel()
    ray_y_flat = ray_y.ravel()
    z = depth_flat[valid_flat].astype(np.float32, copy=False)
    points = np.empty((n_points, 3), dtype=np.float32)
    points[:, 0] = ray_x_flat[valid_flat] * z
    points[:, 1] = ray_y_flat[valid_flat] * z
    points[:, 2] = z

    if stride == 1 and color_bgr.flags["C_CONTIGUOUS"]:
        color_flat = color_bgr.reshape(-1, 3)
    else:
        color_flat = np.ascontiguousarray(color_view).reshape(-1, 3)

    colors_rgb_u8 = np.empty((n_points, 3), dtype=np.uint8)
    colors_rgb_u8[:, 0] = color_flat[valid_flat, 2]
    colors_rgb_u8[:, 1] = color_flat[valid_flat, 1]
    colors_rgb_u8[:, 2] = color_flat[valid_flat, 0]
    return points, colors_rgb_u8


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Realtime single-D455 camera-frame RGB-D demo. Streams RealSense color+depth or "
            "FFS RGB+IR stereo depth, aligns metric depth to color, and renders either a fast "
            "camera-view valid-depth image or an Open3D point cloud."
        ),
        epilog=(
            f"Point-cloud backend contract: frame={COORDINATE_FRAME}, units=meters, axes=x right/y down/z forward. "
            "The image backend preserves aligned valid depth pixels in camera view. No calibrate.pkl or "
            "multi-camera world transform is used."
        ),
    )
    parser.add_argument("--serial", default=None, help="D400 serial to open. Defaults to the first sorted detected serial.")
    parser.add_argument("--fps", type=int, choices=SUPPORTED_CAPTURE_FPS, default=DEFAULT_FPS, help="Capture frame rate.")
    parser.add_argument(
        "--profile",
        choices=SUPPORTED_PROFILES,
        default=DEFAULT_PROFILE,
        help="Color/depth stream resolution.",
    )
    parser.add_argument(
        "--depth-source",
        choices=DEPTH_SOURCES,
        default="realsense",
        help="Depth source. realsense streams color+depth; ffs streams color+IR-left+IR-right and runs two-stage TensorRT FFS.",
    )
    parser.add_argument(
        "--ffs-repo",
        type=Path,
        default=DEFAULT_FFS_REPO,
        help="Fast-FoundationStereo repo path. Defaults to the repo-root-relative sibling ../Fast-FoundationStereo.",
    )
    parser.add_argument(
        "--ffs-trt-model-dir",
        type=Path,
        default=DEFAULT_FFS_TRT_TWO_STAGE_MODEL_DIR,
        help="Two-stage TensorRT FFS engine directory.",
    )
    parser.add_argument(
        "--ffs-trt-root",
        type=Path,
        default=None,
        help="Optional TensorRT Python package/root override forwarded to the FFS runner.",
    )
    parser.add_argument("--emitter", choices=("auto", "on", "off"), default="auto", help="Depth emitter mode.")
    parser.add_argument("--depth-min-m", type=float, default=0.1, help="Minimum rendered depth in meters.")
    parser.add_argument(
        "--depth-max-m",
        type=float,
        default=0.0,
        help="Maximum rendered depth in meters. Use <=0 to disable far clipping.",
    )
    parser.add_argument("--stride", type=int, default=1, help="Pixel stride for point generation. Default preserves density.")
    parser.add_argument(
        "--max-points",
        type=int,
        default=None,
        help="Maximum rendered points. Omit for view default: camera=0 uncapped, orbit=200000. Set 0 to force uncapped.",
    )
    parser.add_argument("--point-size", type=float, default=2.0, help="Open3D point size.")
    parser.add_argument(
        "--view-mode",
        choices=("camera", "orbit"),
        default="camera",
        help="Open3D view. camera uses D455 color intrinsics; orbit uses a third-person view.",
    )
    parser.add_argument(
        "--render-backend",
        choices=("auto", "image", "pointcloud"),
        default="auto",
        help="Render backend. auto uses image for camera view and pointcloud for orbit view.",
    )
    parser.add_argument(
        "--backproject-backend",
        choices=BACKPROJECT_BACKENDS,
        default="auto",
        help="Point-cloud backprojection backend. auto uses numba when available for the stride-1 projection-grid path.",
    )
    parser.add_argument(
        "--image-splat-px",
        type=int,
        default=0,
        help="Image backend splat radius in pixels. 0 preserves exact valid depth pixels.",
    )
    parser.add_argument(
        "--latency-target-ms",
        type=float,
        default=50.0,
        help="HUD warning threshold only; does not adapt quality.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show profiler timing in the HUD and print per-stage timing to the console once per second.",
    )
    parser.add_argument("--duration-s", type=float, default=0.0, help="Run duration. 0 means until the window closes.")
    return parser


def validate_ffs_paths(*, ffs_repo: Path, model_dir: Path) -> None:
    if not ffs_repo.exists():
        raise ValueError(f"--ffs-repo does not exist: {ffs_repo}")
    if not ffs_repo.is_dir():
        raise ValueError(f"--ffs-repo is not a directory: {ffs_repo}")
    if not model_dir.exists():
        raise ValueError(f"--ffs-trt-model-dir does not exist: {model_dir}")
    if not model_dir.is_dir():
        raise ValueError(f"--ffs-trt-model-dir is not a directory: {model_dir}")
    missing = [
        str(path)
        for path in (
            model_dir / "feature_runner.engine",
            model_dir / "post_runner.engine",
            model_dir / "onnx.yaml",
        )
        if not path.is_file()
    ]
    if missing:
        raise ValueError("missing required two-stage TensorRT FFS artifact files: " + ", ".join(missing))


def validate_args(args: argparse.Namespace) -> None:
    parse_profile(args.profile)
    if args.depth_source not in DEPTH_SOURCES:
        raise ValueError(f"--depth-source must be one of {', '.join(DEPTH_SOURCES)}")
    if args.depth_min_m < 0:
        raise ValueError("--depth-min-m must be >= 0")
    if args.depth_max_m > 0 and args.depth_max_m <= args.depth_min_m:
        raise ValueError("expected --depth-max-m <= 0, or --depth-max-m > --depth-min-m")
    if args.stride < 1:
        raise ValueError("--stride must be >= 1")
    if args.max_points is not None and args.max_points < 0:
        raise ValueError("--max-points must be >= 0")
    if args.point_size <= 0:
        raise ValueError("--point-size must be > 0")
    if args.render_backend == "image" and args.view_mode != "camera":
        raise ValueError("--render-backend image requires --view-mode camera")
    if args.backproject_backend not in BACKPROJECT_BACKENDS:
        raise ValueError(f"--backproject-backend must be one of {', '.join(BACKPROJECT_BACKENDS)}")
    if resolve_render_backend(args) == "pointcloud":
        resolve_backproject_backend(args.backproject_backend, stride=args.stride, projection_grid_available=True)
    if args.image_splat_px < 0:
        raise ValueError("--image-splat-px must be >= 0")
    if args.latency_target_ms <= 0:
        raise ValueError("--latency-target-ms must be > 0")
    if args.duration_s < 0:
        raise ValueError("--duration-s must be >= 0")
    if args.depth_source == "ffs":
        validate_ffs_paths(ffs_repo=Path(args.ffs_repo), model_dir=Path(args.ffs_trt_model_dir))


def resolve_render_backend(args: argparse.Namespace) -> str:
    if args.render_backend != "auto":
        return str(args.render_backend)
    if args.view_mode == "camera":
        return "image"
    return "pointcloud"


def resolve_max_points(args: argparse.Namespace) -> int:
    if args.max_points is not None:
        return int(args.max_points)
    if args.view_mode == "orbit":
        return DEFAULT_ORBIT_MAX_POINTS
    return DEFAULT_CAMERA_MAX_POINTS


def apply_view_defaults(args: argparse.Namespace) -> argparse.Namespace:
    args.max_points = resolve_max_points(args)
    return args


def pointcloud_update_requires_readd(*, geometry_added: bool, current_capacity: int, point_count: int) -> bool:
    return (not geometry_added) or point_count > current_capacity


def _load_realsense_module():
    try:
        import pyrealsense2 as rs  # type: ignore
    except ImportError as exc:
        raise RuntimeError("pyrealsense2 is required to run the realtime D455 demo") from exc
    return rs


def _load_open3d_modules():
    try:
        import open3d as o3d  # type: ignore
        from open3d.visualization import gui, rendering  # type: ignore
    except ImportError as exc:
        raise RuntimeError("open3d is required to render the realtime point cloud") from exc
    return o3d, gui, rendering


def _device_info(device: object, info_key: object) -> str:
    try:
        if hasattr(device, "supports") and device.supports(info_key):
            return str(device.get_info(info_key))
    except Exception:
        return ""
    return ""


def list_d400_serials(rs: object) -> list[str]:
    context = rs.context()
    serials: list[str] = []
    for device in context.query_devices():
        product_line = _device_info(device, rs.camera_info.product_line)
        serial = _device_info(device, rs.camera_info.serial_number)
        if serial and product_line.upper() == "D400":
            serials.append(serial)
    return sorted(serials)


def resolve_serial(rs: object, requested_serial: str | None) -> str:
    serials = list_d400_serials(rs)
    if requested_serial:
        if serials and requested_serial not in serials:
            available = ", ".join(serials)
            raise RuntimeError(f"requested serial {requested_serial!r} is not a detected D400 device; available: {available}")
        return requested_serial
    if not serials:
        raise RuntimeError("no D400 RealSense device detected")
    return serials[0]


def _apply_emitter(profile: object, emitter: str, rs: object) -> None:
    if emitter == "auto":
        return
    try:
        depth_sensor = profile.get_device().first_depth_sensor()
        if depth_sensor.supports(rs.option.emitter_enabled):
            depth_sensor.set_option(rs.option.emitter_enabled, 1.0 if emitter == "on" else 0.0)
    except Exception as exc:
        raise RuntimeError(f"failed to set emitter {emitter!r}: {exc}") from exc


@dataclass(frozen=True)
class RealtimeCameraRuntime:
    pipeline: object
    align: object | None
    serial: str
    intrinsics: CameraIntrinsics
    depth_scale_m_per_unit: float
    k_color: np.ndarray
    k_ir_left: np.ndarray | None = None
    t_ir_left_to_color: np.ndarray | None = None
    ir_baseline_m: float = 0.0


def _start_realsense_pipeline(args: argparse.Namespace) -> RealtimeCameraRuntime:
    rs = _load_realsense_module()
    width, height = parse_profile(args.profile)
    serial = resolve_serial(rs, args.serial)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, args.fps)
    if args.depth_source == "ffs":
        config.enable_stream(rs.stream.infrared, 1, width, height, rs.format.y8, args.fps)
        config.enable_stream(rs.stream.infrared, 2, width, height, rs.format.y8, args.fps)
    else:
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, args.fps)
    profile = pipeline.start(config)
    try:
        _apply_emitter(profile, args.emitter, rs)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = float(depth_sensor.get_depth_scale())
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_stream.get_intrinsics()
        intrinsics = camera_intrinsics_from_rs(intr)
        k_color = rs_intrinsics_to_matrix(intr)
        if args.depth_source == "ffs":
            ir_left_profile = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
            ir_right_profile = profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()
            ir_left_to_right = ir_left_profile.get_extrinsics_to(ir_right_profile)
            ir_left_to_color = ir_left_profile.get_extrinsics_to(color_stream)
            return RealtimeCameraRuntime(
                pipeline=pipeline,
                align=None,
                serial=serial,
                intrinsics=intrinsics,
                depth_scale_m_per_unit=depth_scale,
                k_color=k_color,
                k_ir_left=rs_intrinsics_to_matrix(ir_left_profile.get_intrinsics()),
                t_ir_left_to_color=rs_extrinsics_to_matrix(ir_left_to_color),
                ir_baseline_m=rs_translation_norm(ir_left_to_right),
            )
        align = rs.align(rs.stream.color)
    except Exception:
        pipeline.stop()
        raise
    return RealtimeCameraRuntime(
        pipeline=pipeline,
        align=align,
        serial=serial,
        intrinsics=intrinsics,
        depth_scale_m_per_unit=depth_scale,
        k_color=k_color,
    )


class RealtimeSingleCameraPointCloudDemo:
    def __init__(self, args: argparse.Namespace) -> None:
        if args.max_points is None:
            apply_view_defaults(args)
        self.args = args
        self.width, self.height = parse_profile(args.profile)
        self.pixel_grid = build_pixel_grid(width=self.width, height=self.height, stride=args.stride)
        self.projection_grid: tuple[np.ndarray, np.ndarray] | None = None
        self.render_backend = resolve_render_backend(args)
        self.backproject_backend = "none"
        self.capture_slot: LatestSlot[FramePacket] = LatestSlot()
        self.depth_slot: LatestSlot[DepthFramePacket | RawFfsDepthPacket] = LatestSlot()
        self.render_slot: LatestSlot[PointCloudPacket | ImagePacket] = LatestSlot()
        self.stop_event = threading.Event()
        self.capture_thread: threading.Thread | None = None
        self.depth_thread: threading.Thread | None = None
        self.render_worker_thread: threading.Thread | None = None
        self.pipeline: object | None = None
        self.align: object | None = None
        self.serial = ""
        self.intrinsics = CameraIntrinsics(0.0, 0.0, 0.0, 0.0)
        self.depth_scale_m_per_unit = 0.0
        self.k_color = np.eye(3, dtype=np.float32)
        self.k_ir_left: np.ndarray | None = None
        self.t_ir_left_to_color: np.ndarray | None = None
        self.ir_baseline_m = 0.0
        self.ffs_runner: object | None = None
        self._request_render_update: Callable[[], None] = lambda: None
        self._last_debug_log_s = 0.0

    def run(self) -> int:
        if self.args.depth_source == "ffs":
            self.ffs_runner = self._create_ffs_runner()
        runtime = _start_realsense_pipeline(self.args)
        self.pipeline = runtime.pipeline
        self.align = runtime.align
        self.serial = runtime.serial
        self.intrinsics = runtime.intrinsics
        self.depth_scale_m_per_unit = runtime.depth_scale_m_per_unit
        self.k_color = runtime.k_color
        self.k_ir_left = runtime.k_ir_left
        self.t_ir_left_to_color = runtime.t_ir_left_to_color
        self.ir_baseline_m = runtime.ir_baseline_m
        if self.render_backend == "pointcloud":
            self.projection_grid = build_projection_grid(
                width=self.width,
                height=self.height,
                stride=self.args.stride,
                intrinsics=self.intrinsics,
            )
            self.backproject_backend = resolve_backproject_backend(
                self.args.backproject_backend,
                stride=self.args.stride,
                projection_grid_available=True,
            )
            if self.backproject_backend == "numba":
                warm_up_numba_backprojection()
        elif self.args.depth_source == "realsense" and self.args.backproject_backend == "numba":
            warm_up_numba_backprojection()
        try:
            self._run_open3d_viewer()
        finally:
            self.stop()
        return 0

    def _create_ffs_runner(self) -> object:
        try:
            from data_process.depth_backends import FastFoundationStereoTensorRTRunner

            return FastFoundationStereoTensorRTRunner(
                ffs_repo=Path(self.args.ffs_repo),
                model_dir=Path(self.args.ffs_trt_model_dir),
                trt_root=None if self.args.ffs_trt_root is None else Path(self.args.ffs_trt_root),
            )
        except Exception as exc:
            raise RuntimeError(f"failed to start FFS TensorRT runner: {type(exc).__name__}: {exc}") from exc

    def stop(self) -> None:
        self.stop_event.set()
        pipeline = self.pipeline
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception:
                pass
            self.pipeline = None
        for thread in (self.capture_thread, self.depth_thread, self.render_worker_thread):
            if thread is not None and thread.is_alive():
                thread.join(timeout=2.0)

    def _start_threads(self) -> None:
        self.capture_thread = threading.Thread(target=self._capture_loop, name="single-d455-capture", daemon=True)
        depth_target = self._ffs_depth_loop if self.args.depth_source == "ffs" else self._native_depth_loop
        self.depth_thread = threading.Thread(target=depth_target, name="single-d455-depth", daemon=True)
        self.render_worker_thread = threading.Thread(
            target=self._render_prep_loop,
            name="single-d455-render-prep",
            daemon=True,
        )
        self.capture_thread.start()
        self.depth_thread.start()
        self.render_worker_thread.start()

    def _capture_loop(self) -> None:
        assert self.pipeline is not None
        seq = 0
        while not self.stop_event.is_set():
            try:
                wait_start_s = time.perf_counter()
                frames = self.pipeline.wait_for_frames(1000)
                receive_perf_s = time.perf_counter()
                if self.args.depth_source == "ffs":
                    align_done_s = receive_perf_s
                    color_frame = frames.get_color_frame()
                    ir_left_frame = frames.get_infrared_frame(1)
                    ir_right_frame = frames.get_infrared_frame(2)
                    if not color_frame or not ir_left_frame or not ir_right_frame:
                        continue
                else:
                    assert self.align is not None
                    aligned = self.align.process(frames)
                    align_done_s = time.perf_counter()
                    color_frame = aligned.get_color_frame()
                    depth_frame = aligned.get_depth_frame()
                    if not color_frame or not depth_frame:
                        continue
                copy_start_s = time.perf_counter()
                color_bgr = np.ascontiguousarray(np.asanyarray(color_frame.get_data()).copy())
                if self.args.depth_source == "ffs":
                    ir_left_u8 = np.ascontiguousarray(np.asanyarray(ir_left_frame.get_data()).copy())
                    ir_right_u8 = np.ascontiguousarray(np.asanyarray(ir_right_frame.get_data()).copy())
                    depth_u16 = None
                else:
                    depth_u16 = np.ascontiguousarray(np.asanyarray(depth_frame.get_data()).copy())
                    ir_left_u8 = None
                    ir_right_u8 = None
                copy_done_s = time.perf_counter()
                if color_bgr.shape[:2] != (self.height, self.width):
                    continue
                if self.args.depth_source == "ffs":
                    if ir_left_u8.shape != (self.height, self.width) or ir_right_u8.shape != (self.height, self.width):
                        continue
                elif depth_u16 is None or depth_u16.shape != (self.height, self.width):
                    continue
                seq += 1
                timing = PipelineTiming(
                    wait_ms=_elapsed_ms(wait_start_s, receive_perf_s),
                    align_ms=_elapsed_ms(receive_perf_s, align_done_s),
                    frame_copy_ms=_elapsed_ms(copy_start_s, copy_done_s),
                )
                self.capture_slot.put(
                    FramePacket(
                        seq=seq,
                        color_bgr=color_bgr,
                        intrinsics=self.intrinsics,
                        receive_perf_s=receive_perf_s,
                        timing=timing,
                        depth_source=str(self.args.depth_source),
                        depth_u16=depth_u16,
                        depth_scale_m_per_unit=self.depth_scale_m_per_unit,
                        ir_left_u8=ir_left_u8,
                        ir_right_u8=ir_right_u8,
                        k_ir_left=self.k_ir_left,
                        t_ir_left_to_color=self.t_ir_left_to_color,
                        ir_baseline_m=self.ir_baseline_m,
                    )
                )
            except Exception:
                if not self.stop_event.is_set():
                    self.stop_event.wait(0.01)

    def _native_depth_loop(self) -> None:
        last_frame_seq = -1
        while not self.stop_event.is_set():
            frame = self.capture_slot.get_latest_after(last_frame_seq)
            if frame is None:
                self.stop_event.wait(0.001)
                continue
            last_frame_seq = frame.seq
            if frame.depth_u16 is None:
                continue
            depth_done_s = time.perf_counter()
            self.depth_slot.put(
                DepthFramePacket(
                    seq=frame.seq,
                    color_bgr=frame.color_bgr,
                    depth_source=frame.depth_source,
                    intrinsics=frame.intrinsics,
                    receive_perf_s=frame.receive_perf_s,
                    process_done_perf_s=depth_done_s,
                    dropped_capture_frames=self.capture_slot.dropped_count,
                    dropped_ffs_frames=self.depth_slot.dropped_count,
                    timing=frame.timing,
                    depth_u16=frame.depth_u16,
                    depth_scale_m_per_unit=frame.depth_scale_m_per_unit,
                )
            )

    def _ffs_depth_loop(self) -> None:
        runner = self.ffs_runner
        if runner is None:
            if not self.stop_event.is_set():
                print("[ERROR] FFS depth worker started without a TensorRT runner", flush=True)
                self.stop_event.set()
            return

        last_frame_seq = -1
        while not self.stop_event.is_set():
            frame = self.capture_slot.get_latest_after(last_frame_seq)
            if frame is None:
                self.stop_event.wait(0.001)
                continue
            last_frame_seq = frame.seq
            if (
                frame.ir_left_u8 is None
                or frame.ir_right_u8 is None
                or frame.k_ir_left is None
                or frame.t_ir_left_to_color is None
                or frame.ir_baseline_m <= 0
            ):
                continue
            try:
                ffs_start_s = time.perf_counter()
                output = runner.run_pair(
                    frame.ir_left_u8,
                    frame.ir_right_u8,
                    K_ir_left=frame.k_ir_left,
                    baseline_m=float(frame.ir_baseline_m),
                )
                ffs_done_s = time.perf_counter()
                depth_ir_left_m = np.asarray(output["depth_ir_left_m"], dtype=np.float32)
                k_ir_left_used = np.asarray(output.get("K_ir_left_used", frame.k_ir_left), dtype=np.float32)
            except Exception as exc:
                if not self.stop_event.is_set():
                    print(f"[WARN] FFS depth frame {frame.seq} failed: {type(exc).__name__}: {exc}", flush=True)
                continue

            timing = replace(
                frame.timing,
                ffs_ms=_elapsed_ms(ffs_start_s, ffs_done_s),
            )
            self.depth_slot.put(
                RawFfsDepthPacket(
                    seq=frame.seq,
                    color_bgr=frame.color_bgr,
                    depth_left_m=depth_ir_left_m,
                    intrinsics=frame.intrinsics,
                    k_ir_left=k_ir_left_used,
                    t_ir_left_to_color=frame.t_ir_left_to_color,
                    k_color=self.k_color,
                    receive_perf_s=frame.receive_perf_s,
                    ffs_done_perf_s=ffs_done_s,
                    dropped_capture_frames=self.capture_slot.dropped_count,
                    dropped_ffs_frames=self.depth_slot.dropped_count,
                    depth_source=frame.depth_source,
                    timing=timing,
                )
            )

    def _render_prep_loop(self) -> None:
        last_depth_seq = -1
        while not self.stop_event.is_set():
            frame = self.depth_slot.get_latest_after(last_depth_seq)
            if frame is None:
                self.stop_event.wait(0.001)
                continue
            last_depth_seq = frame.seq
            if self.render_backend == "image":
                self._process_image_frame(frame)
            else:
                self._process_pointcloud_frame(frame)

    def _process_image_frame(self, frame: DepthFramePacket | RawFfsDepthPacket) -> None:
        try:
            if isinstance(frame, RawFfsDepthPacket):
                align_start_s = time.perf_counter()
                depth_color_m = align_ir_depth_to_color_fast(
                    frame.depth_left_m,
                    frame.k_ir_left,
                    frame.t_ir_left_to_color,
                    frame.k_color,
                    output_shape=frame.color_bgr.shape[:2],
                )
                align_done_s = time.perf_counter()
                mask_start_s = time.perf_counter()
                image_rgb, valid_count = build_camera_view_image_from_depth_m(
                    color_bgr=frame.color_bgr,
                    depth_m=depth_color_m,
                    depth_min_m=self.args.depth_min_m,
                    depth_max_m=self.args.depth_max_m,
                    splat_px=self.args.image_splat_px,
                )
                mask_done_s = time.perf_counter()
                timing = replace(
                    frame.timing,
                    ffs_align_ms=_elapsed_ms(align_start_s, align_done_s),
                    image_mask_ms=_elapsed_ms(mask_start_s, mask_done_s),
                )
            elif frame.depth_u16 is not None:
                mask_start_s = time.perf_counter()
                image_rgb, valid_count = build_camera_view_image(
                    color_bgr=frame.color_bgr,
                    depth_u16=frame.depth_u16,
                    depth_scale_m_per_unit=frame.depth_scale_m_per_unit,
                    depth_min_m=self.args.depth_min_m,
                    depth_max_m=self.args.depth_max_m,
                    splat_px=self.args.image_splat_px,
                )
                mask_done_s = time.perf_counter()
                timing = replace(frame.timing, image_mask_ms=_elapsed_ms(mask_start_s, mask_done_s))
            elif frame.depth_m is not None:
                mask_start_s = time.perf_counter()
                image_rgb, valid_count = build_camera_view_image_from_depth_m(
                    color_bgr=frame.color_bgr,
                    depth_m=frame.depth_m,
                    depth_min_m=self.args.depth_min_m,
                    depth_max_m=self.args.depth_max_m,
                    splat_px=self.args.image_splat_px,
                )
                mask_done_s = time.perf_counter()
                timing = replace(frame.timing, image_mask_ms=_elapsed_ms(mask_start_s, mask_done_s))
            else:
                return
        except Exception:
            return
        self.render_slot.put(
            ImagePacket(
                seq=frame.seq,
                image_rgb_u8=image_rgb,
                valid_count=valid_count,
                depth_source=frame.depth_source,
                receive_perf_s=frame.receive_perf_s,
                process_done_perf_s=time.perf_counter(),
                dropped_capture_frames=frame.dropped_capture_frames,
                dropped_ffs_frames=self.depth_slot.dropped_count,
                timing=timing,
            )
        )
        self._request_render_update()

    def _process_pointcloud_frame(self, frame: DepthFramePacket | RawFfsDepthPacket) -> None:
        projection_grid = self.projection_grid
        if projection_grid is None:
            return
        try:
            backproject_backend = self.backproject_backend
            if isinstance(frame, RawFfsDepthPacket):
                align_start_s = time.perf_counter()
                depth_color_m = align_ir_depth_to_color_fast(
                    frame.depth_left_m,
                    frame.k_ir_left,
                    frame.t_ir_left_to_color,
                    frame.k_color,
                    output_shape=frame.color_bgr.shape[:2],
                )
                align_done_s = time.perf_counter()
                backproject_start_s = time.perf_counter()
                points, colors = backproject_aligned_rgbd_depth_m(
                    color_bgr=frame.color_bgr,
                    depth_m=depth_color_m,
                    intrinsics=frame.intrinsics,
                    depth_min_m=self.args.depth_min_m,
                    depth_max_m=self.args.depth_max_m,
                    stride=self.args.stride,
                    max_points=self.args.max_points,
                    projection_grid=projection_grid,
                )
                backproject_done_s = time.perf_counter()
                backproject_backend = "float32-numpy"
                timing = replace(
                    frame.timing,
                    ffs_align_ms=_elapsed_ms(align_start_s, align_done_s),
                    backproject_ms=_elapsed_ms(backproject_start_s, backproject_done_s),
                )
            elif frame.depth_u16 is not None:
                backproject_start_s = time.perf_counter()
                points, colors = backproject_aligned_rgbd(
                    color_bgr=frame.color_bgr,
                    depth_u16=frame.depth_u16,
                    intrinsics=frame.intrinsics,
                    depth_scale_m_per_unit=frame.depth_scale_m_per_unit,
                    depth_min_m=self.args.depth_min_m,
                    depth_max_m=self.args.depth_max_m,
                    stride=self.args.stride,
                    max_points=self.args.max_points,
                    projection_grid=projection_grid,
                    backproject_backend=self.backproject_backend,
                )
                backproject_done_s = time.perf_counter()
                timing = replace(frame.timing, backproject_ms=_elapsed_ms(backproject_start_s, backproject_done_s))
            elif frame.depth_m is not None:
                backproject_start_s = time.perf_counter()
                points, colors = backproject_aligned_rgbd_depth_m(
                    color_bgr=frame.color_bgr,
                    depth_m=frame.depth_m,
                    intrinsics=frame.intrinsics,
                    depth_min_m=self.args.depth_min_m,
                    depth_max_m=self.args.depth_max_m,
                    stride=self.args.stride,
                    max_points=self.args.max_points,
                    projection_grid=projection_grid,
                )
                backproject_done_s = time.perf_counter()
                backproject_backend = "float32-numpy"
                timing = replace(frame.timing, backproject_ms=_elapsed_ms(backproject_start_s, backproject_done_s))
            else:
                return
        except Exception:
            return
        self.render_slot.put(
            PointCloudPacket(
                seq=frame.seq,
                points_xyz_m=points,
                colors_rgb_u8=colors,
                backproject_backend=backproject_backend,
                depth_source=frame.depth_source,
                receive_perf_s=frame.receive_perf_s,
                process_done_perf_s=time.perf_counter(),
                dropped_capture_frames=frame.dropped_capture_frames,
                dropped_ffs_frames=self.depth_slot.dropped_count,
                timing=timing,
            )
        )
        self._request_render_update()

    def _run_open3d_viewer(self) -> None:
        o3d, gui, rendering = _load_open3d_modules()
        o3c = o3d.core
        device = o3c.Device("CPU:0")
        app = gui.Application.instance
        app.initialize()
        window = app.create_window("Realtime Single D455 Point Cloud", 1280, 800)
        scene_widget = None
        image_widget = None
        if self.render_backend == "image":
            blank = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            image_widget = gui.ImageWidget(o3d.geometry.Image(blank))
        else:
            scene_widget = gui.SceneWidget()
            scene_widget.scene = rendering.Open3DScene(window.renderer)
            scene_widget.scene.set_background([0.02, 0.02, 0.02, 1.0])

        hud_label = gui.Label("Waiting for frames...")
        hud_label.text_color = gui.Color(1.0, 1.0, 1.0)
        hud_panel = gui.Vert(0, gui.Margins(8, 8, 8, 8))
        hud_panel.add_child(hud_label)
        if image_widget is not None:
            window.add_child(image_widget)
        elif scene_widget is not None:
            window.add_child(scene_widget)
        window.add_child(hud_panel)

        def on_layout(layout_context: object) -> None:
            rect = window.content_rect
            if image_widget is not None:
                image_widget.frame = rect
            elif scene_widget is not None:
                scene_widget.frame = rect
            em = window.theme.font_size
            preferred = hud_panel.calc_preferred_size(layout_context, gui.Widget.Constraints())
            hud_panel.frame = gui.Rect(
                rect.x + 0.5 * em,
                rect.y + 0.5 * em,
                max(preferred.width, 640),
                max(preferred.height, (12.0 if self.args.debug else 9.0) * em),
            )

        window.set_on_layout(on_layout)
        material = rendering.MaterialRecord()
        material.shader = "defaultUnlit"
        material.point_size = float(self.args.point_size)
        pcd = o3d.t.geometry.PointCloud(device)
        color_float_buffer = ColorFloat32Buffer()
        # Tensor.from_numpy shares CPU memory, so retain the current arrays until the next update.
        tensor_numpy_refs: dict[str, np.ndarray | None] = {"points": None, "colors": None}
        geometry_added = {"value": False}
        geometry_capacity = {"value": 0}
        camera_initialized = {"value": False}
        update_warned = {"value": False}
        render_stats = RenderStats()
        last_render_seq = {"value": -1}
        post_lock = threading.Lock()
        callback_pending = {"value": False}

        def reset_camera() -> None:
            assert scene_widget is not None
            if self.args.view_mode == "camera":
                intrinsic_matrix = np.array(
                    [
                        [self.intrinsics.fx, 0.0, self.intrinsics.cx],
                        [0.0, self.intrinsics.fy, self.intrinsics.cy],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=np.float64,
                )
                extrinsic = np.eye(4, dtype=np.float64)
                bounds = o3d.geometry.AxisAlignedBoundingBox([-10.0, -10.0, 0.01], [10.0, 10.0, 20.0])
                scene_widget.setup_camera(intrinsic_matrix, extrinsic, self.width, self.height, bounds)
                return
            try:
                scene_widget.look_at([0.0, 0.0, 0.8], [0.0, 0.0, -1.0], [0.0, -1.0, 0.0])
            except Exception:
                bounds = o3d.geometry.AxisAlignedBoundingBox([-0.5, -0.35, 0.1], [0.5, 0.35, 1.5])
                scene_widget.setup_camera(60.0, bounds, [0.0, 0.0, 0.0])

        def render_latest() -> None:
            with post_lock:
                callback_pending["value"] = False
            packet = self.render_slot.get_latest_after(last_render_seq["value"])
            if packet is None:
                return
            last_render_seq["value"] = packet.seq
            open3d_convert_ms = 0.0
            open3d_update_ms = 0.0
            if isinstance(packet, ImagePacket):
                assert image_widget is not None
                convert_start_s = time.perf_counter()
                o3d_image = o3d.geometry.Image(packet.image_rgb_u8)
                open3d_convert_ms = _elapsed_ms(convert_start_s, time.perf_counter())
                update_start_s = time.perf_counter()
                image_widget.update_image(o3d_image)
                open3d_update_ms = _elapsed_ms(update_start_s, time.perf_counter())
            elif packet.point_count == 0:
                assert scene_widget is not None
                if geometry_added["value"]:
                    update_start_s = time.perf_counter()
                    try:
                        scene_widget.scene.remove_geometry(GEOMETRY_NAME)
                    except Exception:
                        pass
                    open3d_update_ms = _elapsed_ms(update_start_s, time.perf_counter())
                    geometry_added["value"] = False
                    geometry_capacity["value"] = 0
            else:
                assert scene_widget is not None
                convert_start_s = time.perf_counter()
                points = ensure_float32_c_contiguous(packet.points_xyz_m)
                colors = color_float_buffer.convert(packet.colors_rgb_u8)
                tensor_numpy_refs["points"] = points
                tensor_numpy_refs["colors"] = colors
                pcd.point.positions = o3c.Tensor.from_numpy(points)
                pcd.point.colors = o3c.Tensor.from_numpy(colors)
                open3d_convert_ms = _elapsed_ms(convert_start_s, time.perf_counter())
                update_start_s = time.perf_counter()
                if pointcloud_update_requires_readd(
                    geometry_added=geometry_added["value"],
                    current_capacity=geometry_capacity["value"],
                    point_count=packet.point_count,
                ):
                    if geometry_added["value"]:
                        try:
                            scene_widget.scene.remove_geometry(GEOMETRY_NAME)
                        except Exception:
                            pass
                    scene_widget.scene.add_geometry(GEOMETRY_NAME, pcd, material)
                    geometry_added["value"] = True
                    geometry_capacity["value"] = packet.point_count
                    if not camera_initialized["value"]:
                        reset_camera()
                        camera_initialized["value"] = True
                else:
                    try:
                        flags = rendering.Scene.UPDATE_POINTS_FLAG | rendering.Scene.UPDATE_COLORS_FLAG
                        scene_widget.scene.scene.update_geometry(GEOMETRY_NAME, pcd, flags)
                        geometry_capacity["value"] = max(geometry_capacity["value"], packet.point_count)
                    except Exception as exc:
                        if not update_warned["value"]:
                            print(
                                f"[WARN] update_geometry fallback: {type(exc).__name__}: {exc}",
                                flush=True,
                            )
                            update_warned["value"] = True
                        try:
                            scene_widget.scene.remove_geometry(GEOMETRY_NAME)
                        except Exception:
                            pass
                        scene_widget.scene.add_geometry(GEOMETRY_NAME, pcd, material)
                        geometry_added["value"] = True
                        geometry_capacity["value"] = packet.point_count
                        if not camera_initialized["value"]:
                            reset_camera()
                            camera_initialized["value"] = True
                open3d_update_ms = _elapsed_ms(update_start_s, time.perf_counter())

            render_time_s = time.perf_counter()
            latency_ms = (render_time_s - packet.receive_perf_s) * 1000.0
            timing = replace(
                packet.timing,
                open3d_convert_ms=open3d_convert_ms,
                open3d_update_ms=open3d_update_ms,
                receive_to_render_ms=latency_ms,
            )
            render_stats.record_render(render_time_s=render_time_s, latency_ms=latency_ms)
            hud_label.text = self._format_hud(packet=packet, stats=render_stats, timing=timing)
            self._maybe_log_debug(packet=packet, stats=render_stats, timing=timing, now_s=render_time_s)
            window.post_redraw()

        def request_render_update() -> None:
            with post_lock:
                if callback_pending["value"] or self.stop_event.is_set():
                    return
                callback_pending["value"] = True
            app.post_to_main_thread(window, render_latest)

        def on_close() -> bool:
            self.stop_event.set()
            return True

        self._request_render_update = request_render_update
        window.set_on_close(on_close)
        self._start_threads()

        timer: threading.Timer | None = None
        if self.args.duration_s > 0:
            timer = threading.Timer(
                self.args.duration_s,
                lambda: app.post_to_main_thread(window, lambda: (self.stop_event.set(), window.close())),
            )
            timer.daemon = True
            timer.start()
        try:
            app.run()
        finally:
            if timer is not None:
                timer.cancel()

    def _format_hud(self, *, packet: PointCloudPacket | ImagePacket, stats: RenderStats, timing: PipelineTiming) -> str:
        status = "late" if timing.receive_to_render_ms > self.args.latency_target_ms else "ok"
        max_points = "uncapped" if self.args.max_points == 0 else str(self.args.max_points)
        backproject_backend = getattr(packet, "backproject_backend", self.backproject_backend)
        text = (
            f"render FPS: {stats.render_fps:.1f}\n"
            f"latency: {timing.receive_to_render_ms:.1f} ms ({status}, target {self.args.latency_target_ms:.1f} ms)\n"
            f"mean latency: {stats.mean_latency_ms:.1f} ms\n"
            f"points: {packet.point_count}  max-points: {max_points}\n"
            f"dropped capture frames: {packet.dropped_capture_frames}\n"
            f"dropped depth/render packets: {packet.dropped_ffs_frames}/{self.render_slot.dropped_count}\n"
            f"serial/profile/fps: {self.serial}  {self.args.profile}@{self.args.fps}\n"
            f"depth source: {packet.depth_source}\n"
            f"view/backend/backproject: {self.args.view_mode}/{self.render_backend}/{backproject_backend}\n"
            f"frame: {COORDINATE_FRAME}  meters  x right / y down / z forward"
        )
        if self.args.debug:
            text += (
                "\n"
                f"profiler ms: depth_to_render={depth_to_render_ms(timing):.2f} "
                f"wait={timing.wait_ms:.2f} align={timing.align_ms:.2f} copy={timing.frame_copy_ms:.2f}\n"
                f"ffs={timing.ffs_ms:.2f} ffs_align={timing.ffs_align_ms:.2f} "
                f"mask={timing.image_mask_ms:.2f} backproject={timing.backproject_ms:.2f} "
                f"o3d_convert={timing.open3d_convert_ms:.2f} o3d_update={timing.open3d_update_ms:.2f}"
            )
        return text

    def _maybe_log_debug(
        self,
        *,
        packet: PointCloudPacket | ImagePacket,
        stats: RenderStats,
        timing: PipelineTiming,
        now_s: float,
    ) -> None:
        if not self.args.debug or now_s - self._last_debug_log_s < DEBUG_LOG_INTERVAL_S:
            return
        self._last_debug_log_s = now_s
        print(
            "[pcd-debug] "
            f"seq={packet.seq} "
            f"render_fps={stats.render_fps:.1f} "
            f"receive_to_render_ms={timing.receive_to_render_ms:.2f} "
            f"depth_to_render_ms={depth_to_render_ms(timing):.2f} "
            f"depth_source={packet.depth_source} "
            f"wait_ms={timing.wait_ms:.2f} "
            f"align_ms={timing.align_ms:.2f} "
            f"copy_ms={timing.frame_copy_ms:.2f} "
            f"ffs_ms={timing.ffs_ms:.2f} "
            f"ffs_align_ms={timing.ffs_align_ms:.2f} "
            f"mask_ms={timing.image_mask_ms:.2f} "
            f"backproject_ms={timing.backproject_ms:.2f} "
            f"o3d_convert_ms={timing.open3d_convert_ms:.2f} "
            f"o3d_update_ms={timing.open3d_update_ms:.2f} "
            f"backend={self.render_backend} "
            f"backproject_backend={getattr(packet, 'backproject_backend', self.backproject_backend)} "
            f"points={packet.point_count} "
            f"dropped_capture={packet.dropped_capture_frames} "
            f"dropped_depth={packet.dropped_ffs_frames} "
            f"dropped_render={self.render_slot.dropped_count}",
            flush=True,
        )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_args(args)
        apply_view_defaults(args)
        return RealtimeSingleCameraPointCloudDemo(args).run()
    except (RuntimeError, ValueError) as exc:
        parser.exit(2, f"{parser.prog}: error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
