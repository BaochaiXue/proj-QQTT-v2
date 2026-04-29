from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import threading
import time
from typing import Callable, Generic, TypeVar

import numpy as np


SUPPORTED_CAPTURE_FPS = (5, 15, 30)
SUPPORTED_PROFILES = ("848x480", "640x480")
DEFAULT_PROFILE = "848x480"
DEFAULT_FPS = 30
COORDINATE_FRAME = "camera_color_frame"
GEOMETRY_NAME = "single_d455_live_pointcloud"


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class FramePacket:
    seq: int
    color_bgr: np.ndarray
    depth_u16: np.ndarray
    intrinsics: CameraIntrinsics
    depth_scale_m_per_unit: float
    receive_perf_s: float


@dataclass(frozen=True)
class PointCloudPacket:
    seq: int
    points_xyz_m: np.ndarray
    colors_rgb_u8: np.ndarray
    receive_perf_s: float
    process_done_perf_s: float
    dropped_capture_frames: int

    @property
    def point_count(self) -> int:
        return int(self.points_xyz_m.shape[0])


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


def _packet_seq(packet: object) -> int:
    try:
        return int(getattr(packet, "seq"))
    except AttributeError as exc:
        raise TypeError("latest-slot packets must expose an integer seq attribute") from exc


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

    depth_view = depth_u16[::stride, ::stride]
    color_view = color_bgr[::stride, ::stride, :]
    if pixel_grid is None:
        grid_x, grid_y = build_pixel_grid(width=depth_u16.shape[1], height=depth_u16.shape[0], stride=stride)
    else:
        grid_x, grid_y = pixel_grid
    if grid_x.shape != depth_view.shape or grid_y.shape != depth_view.shape:
        raise ValueError("pixel_grid shape does not match strided depth shape")

    depth_m = depth_view.astype(np.float32, copy=False) * np.float32(depth_scale_m_per_unit)
    valid = (depth_view > 0) & np.isfinite(depth_m) & (depth_m >= depth_min_m)
    if depth_max_m > 0:
        valid &= depth_m <= depth_max_m
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    z = depth_m[valid]
    points = np.empty((z.size, 3), dtype=np.float32)
    points[:, 0] = (grid_x[valid] - np.float32(intrinsics.cx)) * z / np.float32(intrinsics.fx)
    points[:, 1] = (grid_y[valid] - np.float32(intrinsics.cy)) * z / np.float32(intrinsics.fy)
    points[:, 2] = z
    colors_rgb_u8 = color_view[valid][:, [2, 1, 0]]

    if max_points > 0 and points.shape[0] > max_points:
        indices = np.linspace(0, points.shape[0] - 1, max_points, dtype=np.int64)
        points = points[indices]
        colors_rgb_u8 = colors_rgb_u8[indices]

    return np.ascontiguousarray(points), np.ascontiguousarray(colors_rgb_u8)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Realtime single-D455 camera-frame point-cloud demo. Streams color + depth, aligns depth to "
            f"color, backprojects in {COORDINATE_FRAME}, and renders with Open3D."
        ),
        epilog=(
            f"Point-cloud contract: frame={COORDINATE_FRAME}, units=meters, axes=x right/y down/z forward. "
            "No calibrate.pkl or multi-camera world transform is used."
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
    parser.add_argument("--emitter", choices=("auto", "on", "off"), default="auto", help="Depth emitter mode.")
    parser.add_argument("--depth-min-m", type=float, default=0.1, help="Minimum rendered depth in meters.")
    parser.add_argument(
        "--depth-max-m",
        type=float,
        default=0.0,
        help="Maximum rendered depth in meters. Use <=0 to disable far clipping.",
    )
    parser.add_argument("--stride", type=int, default=1, help="Pixel stride for point generation. Default preserves density.")
    parser.add_argument("--max-points", type=int, default=0, help="Maximum rendered points. 0 means uncapped.")
    parser.add_argument("--point-size", type=float, default=2.0, help="Open3D point size.")
    parser.add_argument(
        "--latency-target-ms",
        type=float,
        default=50.0,
        help="HUD warning threshold only; does not adapt quality.",
    )
    parser.add_argument("--duration-s", type=float, default=0.0, help="Run duration. 0 means until the window closes.")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    parse_profile(args.profile)
    if args.depth_min_m < 0:
        raise ValueError("--depth-min-m must be >= 0")
    if args.depth_max_m > 0 and args.depth_max_m <= args.depth_min_m:
        raise ValueError("expected --depth-max-m <= 0, or --depth-max-m > --depth-min-m")
    if args.stride < 1:
        raise ValueError("--stride must be >= 1")
    if args.max_points < 0:
        raise ValueError("--max-points must be >= 0")
    if args.point_size <= 0:
        raise ValueError("--point-size must be > 0")
    if args.latency_target_ms <= 0:
        raise ValueError("--latency-target-ms must be > 0")
    if args.duration_s < 0:
        raise ValueError("--duration-s must be >= 0")


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


def _start_realsense_pipeline(args: argparse.Namespace) -> tuple[object, object, str, CameraIntrinsics, float]:
    rs = _load_realsense_module()
    width, height = parse_profile(args.profile)
    serial = resolve_serial(rs, args.serial)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, args.fps)
    profile = pipeline.start(config)
    try:
        _apply_emitter(profile, args.emitter, rs)
        align = rs.align(rs.stream.color)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = float(depth_sensor.get_depth_scale())
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_stream.get_intrinsics()
        intrinsics = CameraIntrinsics(fx=float(intr.fx), fy=float(intr.fy), cx=float(intr.ppx), cy=float(intr.ppy))
    except Exception:
        pipeline.stop()
        raise
    return pipeline, align, serial, intrinsics, depth_scale


class RealtimeSingleCameraPointCloudDemo:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.width, self.height = parse_profile(args.profile)
        self.pixel_grid = build_pixel_grid(width=self.width, height=self.height, stride=args.stride)
        self.capture_slot: LatestSlot[FramePacket] = LatestSlot()
        self.pointcloud_slot: LatestSlot[PointCloudPacket] = LatestSlot()
        self.stop_event = threading.Event()
        self.capture_thread: threading.Thread | None = None
        self.worker_thread: threading.Thread | None = None
        self.pipeline: object | None = None
        self.align: object | None = None
        self.serial = ""
        self.intrinsics = CameraIntrinsics(0.0, 0.0, 0.0, 0.0)
        self.depth_scale_m_per_unit = 0.0
        self._request_render_update: Callable[[], None] = lambda: None

    def run(self) -> int:
        self.pipeline, self.align, self.serial, self.intrinsics, self.depth_scale_m_per_unit = _start_realsense_pipeline(
            self.args
        )
        try:
            self._run_open3d_viewer()
        finally:
            self.stop()
        return 0

    def stop(self) -> None:
        self.stop_event.set()
        pipeline = self.pipeline
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception:
                pass
            self.pipeline = None
        for thread in (self.capture_thread, self.worker_thread):
            if thread is not None and thread.is_alive():
                thread.join(timeout=2.0)

    def _start_threads(self) -> None:
        self.capture_thread = threading.Thread(target=self._capture_loop, name="single-d455-capture", daemon=True)
        self.worker_thread = threading.Thread(target=self._pointcloud_loop, name="single-d455-pointcloud", daemon=True)
        self.capture_thread.start()
        self.worker_thread.start()

    def _capture_loop(self) -> None:
        assert self.pipeline is not None
        assert self.align is not None
        seq = 0
        while not self.stop_event.is_set():
            try:
                frames = self.pipeline.wait_for_frames(1000)
                receive_perf_s = time.perf_counter()
                aligned = self.align.process(frames)
                color_frame = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue
                color_bgr = np.ascontiguousarray(np.asanyarray(color_frame.get_data()).copy())
                depth_u16 = np.ascontiguousarray(np.asanyarray(depth_frame.get_data()).copy())
                if color_bgr.shape[:2] != (self.height, self.width) or depth_u16.shape != (self.height, self.width):
                    continue
                seq += 1
                self.capture_slot.put(
                    FramePacket(
                        seq=seq,
                        color_bgr=color_bgr,
                        depth_u16=depth_u16,
                        intrinsics=self.intrinsics,
                        depth_scale_m_per_unit=self.depth_scale_m_per_unit,
                        receive_perf_s=receive_perf_s,
                    )
                )
            except Exception:
                if not self.stop_event.is_set():
                    self.stop_event.wait(0.01)

    def _pointcloud_loop(self) -> None:
        last_frame_seq = -1
        while not self.stop_event.is_set():
            frame = self.capture_slot.get_latest_after(last_frame_seq)
            if frame is None:
                self.stop_event.wait(0.001)
                continue
            last_frame_seq = frame.seq
            try:
                points, colors = backproject_aligned_rgbd(
                    color_bgr=frame.color_bgr,
                    depth_u16=frame.depth_u16,
                    intrinsics=frame.intrinsics,
                    depth_scale_m_per_unit=frame.depth_scale_m_per_unit,
                    depth_min_m=self.args.depth_min_m,
                    depth_max_m=self.args.depth_max_m,
                    stride=self.args.stride,
                    max_points=self.args.max_points,
                    pixel_grid=self.pixel_grid,
                )
            except Exception:
                continue
            self.pointcloud_slot.put(
                PointCloudPacket(
                    seq=frame.seq,
                    points_xyz_m=points,
                    colors_rgb_u8=colors,
                    receive_perf_s=frame.receive_perf_s,
                    process_done_perf_s=time.perf_counter(),
                    dropped_capture_frames=self.capture_slot.dropped_count,
                )
            )
            self._request_render_update()

    def _run_open3d_viewer(self) -> None:
        o3d, gui, rendering = _load_open3d_modules()
        app = gui.Application.instance
        app.initialize()
        window = app.create_window("Realtime Single D455 Point Cloud", 1280, 800)
        scene_widget = gui.SceneWidget()
        scene_widget.scene = rendering.Open3DScene(window.renderer)
        scene_widget.scene.set_background([0.02, 0.02, 0.02, 1.0])

        hud_label = gui.Label("Waiting for frames...")
        hud_label.text_color = gui.Color(1.0, 1.0, 1.0)
        hud_panel = gui.Vert(0, gui.Margins(8, 8, 8, 8))
        hud_panel.add_child(hud_label)
        window.add_child(scene_widget)
        window.add_child(hud_panel)

        def on_layout(layout_context: object) -> None:
            rect = window.content_rect
            scene_widget.frame = rect
            em = window.theme.font_size
            preferred = hud_panel.calc_preferred_size(layout_context, gui.Widget.Constraints())
            hud_panel.frame = gui.Rect(rect.x + 0.5 * em, rect.y + 0.5 * em, max(preferred.width, 520), preferred.height)

        window.set_on_layout(on_layout)
        material = rendering.MaterialRecord()
        material.shader = "defaultUnlit"
        material.point_size = float(self.args.point_size)
        pcd = o3d.geometry.PointCloud()
        geometry_added = {"value": False}
        render_stats = RenderStats()
        last_render_seq = {"value": -1}
        post_lock = threading.Lock()
        callback_pending = {"value": False}

        def reset_camera() -> None:
            try:
                scene_widget.look_at([0.0, 0.0, 0.8], [0.0, 0.0, -1.0], [0.0, -1.0, 0.0])
            except Exception:
                bounds = o3d.geometry.AxisAlignedBoundingBox([-0.5, -0.35, 0.1], [0.5, 0.35, 1.5])
                scene_widget.setup_camera(60.0, bounds, [0.0, 0.0, 0.0])

        def render_latest() -> None:
            with post_lock:
                callback_pending["value"] = False
            packet = self.pointcloud_slot.get_latest_after(last_render_seq["value"])
            if packet is None:
                return
            last_render_seq["value"] = packet.seq
            if packet.point_count == 0:
                if geometry_added["value"]:
                    try:
                        scene_widget.scene.remove_geometry(GEOMETRY_NAME)
                    except Exception:
                        pass
                    geometry_added["value"] = False
            else:
                pcd.points = o3d.utility.Vector3dVector(packet.points_xyz_m.astype(np.float64, copy=False))
                colors_float = packet.colors_rgb_u8.astype(np.float64, copy=False) / 255.0
                pcd.colors = o3d.utility.Vector3dVector(colors_float)
                if geometry_added["value"]:
                    try:
                        flags = rendering.Scene.UPDATE_POINTS_FLAG | rendering.Scene.UPDATE_COLORS_FLAG
                        scene_widget.scene.scene.update_geometry(GEOMETRY_NAME, pcd, flags)
                    except Exception:
                        try:
                            scene_widget.scene.remove_geometry(GEOMETRY_NAME)
                        except Exception:
                            pass
                        scene_widget.scene.add_geometry(GEOMETRY_NAME, pcd, material)
                else:
                    scene_widget.scene.add_geometry(GEOMETRY_NAME, pcd, material)
                    geometry_added["value"] = True
                    reset_camera()

            render_time_s = time.perf_counter()
            latency_ms = (render_time_s - packet.receive_perf_s) * 1000.0
            render_stats.record_render(render_time_s=render_time_s, latency_ms=latency_ms)
            hud_label.text = self._format_hud(packet=packet, stats=render_stats, latency_ms=latency_ms)
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

    def _format_hud(self, *, packet: PointCloudPacket, stats: RenderStats, latency_ms: float) -> str:
        status = "late" if latency_ms > self.args.latency_target_ms else "ok"
        max_points = "uncapped" if self.args.max_points == 0 else str(self.args.max_points)
        return (
            f"render FPS: {stats.render_fps:.1f}\n"
            f"latency: {latency_ms:.1f} ms ({status}, target {self.args.latency_target_ms:.1f} ms)\n"
            f"mean latency: {stats.mean_latency_ms:.1f} ms\n"
            f"points: {packet.point_count}  max-points: {max_points}\n"
            f"dropped capture frames: {packet.dropped_capture_frames}\n"
            f"serial/profile/fps: {self.serial}  {self.args.profile}@{self.args.fps}\n"
            f"frame: {COORDINATE_FRAME}  meters  x right / y down / z forward"
        )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_args(args)
        return RealtimeSingleCameraPointCloudDemo(args).run()
    except (RuntimeError, ValueError) as exc:
        parser.exit(2, f"{parser.prog}: error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
