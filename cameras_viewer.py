#!/usr/bin/env python3
"""Simple GUI viewer for the camera-only RealSense workflow.

Shows color (top) + depth colormap (bottom) per camera, tiled in a grid.
Press `q` or `Esc` to exit.
"""

from __future__ import annotations

import argparse
import math
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple


def _is_wsl_environment() -> bool:
    if os.environ.get("WSL_DISTRO_NAME"):
        return True
    try:
        return (
            "microsoft"
            in Path("/proc/sys/kernel/osrelease").read_text(encoding="utf-8").lower()
        )
    except OSError:
        return False


def _configure_qt_platform_default() -> None:
    if os.environ.get("QT_QPA_PLATFORM"):
        return
    if _is_wsl_environment():
        os.environ["QT_QPA_PLATFORM"] = "xcb"


_configure_qt_platform_default()

from data_process.visualization.depth_colormap import (
    DEFAULT_DEPTH_VIS_MAX_M,
    DEFAULT_DEPTH_VIS_MIN_M,
    colorize_depth_units,
)
from qqtt.env.camera.defaults import (
    DEFAULT_FPS,
    DEFAULT_HEIGHT,
    DEFAULT_NUM_CAM,
    DEFAULT_WIDTH,
)

if TYPE_CHECKING:
    import cv2
    import numpy as np
    import pyrealsense2 as rs


def _runtime_imports():
    import cv2
    import numpy as np
    import pyrealsense2 as rs

    return cv2, np, rs


DEFAULT_EXPOSURE_OVERRIDES = {
    "239222303506": 156.0,
    "239222300781": 156.0,
}
MEASURED_FPS_WINDOW_S = 1.0
MIN_MEASURED_FPS_SAMPLES = 2
_SCREEN_SIZE_CACHE: Optional[Tuple[int, int]] = None
CAPTURE_THREAD_TIMEOUT_MS = 100
DEPTH_RENDER_MODE_CHOICES = ("colormap", "fps_placeholder")


def _apply_color_controls(
    *,
    pipeline: Any,
    auto_exposure: bool,
    exposure: float,
    gain: float,
) -> Tuple[float, float, float]:
    """
    Apply color exposure settings and return (auto_exposure, exposure, gain).
    """
    auto_val = float("nan")
    exp_val = float("nan")
    gain_val = float("nan")
    _, _, rs = _runtime_imports()
    try:
        color_sensor = pipeline.get_active_profile().get_device().first_color_sensor()
        if color_sensor.supports(rs.option.enable_auto_exposure):
            color_sensor.set_option(
                rs.option.enable_auto_exposure, 1.0 if auto_exposure else 0.0
            )
            auto_val = color_sensor.get_option(rs.option.enable_auto_exposure)
        if not auto_exposure:
            if color_sensor.supports(rs.option.exposure):
                color_sensor.set_option(rs.option.exposure, float(exposure))
                exp_val = color_sensor.get_option(rs.option.exposure)
            if color_sensor.supports(rs.option.gain):
                color_sensor.set_option(rs.option.gain, float(gain))
                gain_val = color_sensor.get_option(rs.option.gain)
        else:
            if color_sensor.supports(rs.option.exposure):
                exp_val = color_sensor.get_option(rs.option.exposure)
            if color_sensor.supports(rs.option.gain):
                gain_val = color_sensor.get_option(rs.option.gain)
    except Exception:
        pass
    return auto_val, exp_val, gain_val


def _start_pipeline(
    *,
    ctx: Any,
    serial: str,
    profiles: Tuple[Tuple[int, int, int], ...],
) -> Tuple[Any, Tuple[int, int, int]]:
    _, _, rs = _runtime_imports()
    last_err: Optional[BaseException] = None
    for width, height, fps in profiles:
        pipeline = rs.pipeline(ctx)
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        try:
            pipeline.start(config)
            ok = False
            for _ in range(3):
                try:
                    pipeline.wait_for_frames(2000)
                    ok = True
                    break
                except RuntimeError as e:
                    last_err = e
            if not ok:
                raise RuntimeError(
                    f"Frame did not arrive after starting {width}x{height}@{fps}"
                )
            return pipeline, (width, height, fps)
        except Exception as e:
            last_err = e
            try:
                pipeline.stop()
            except Exception:
                pass
            print(
                f"[WARN] Failed to start {width}x{height}@{fps}: "
                f"{type(e).__name__}: {e}",
                flush=True,
            )
            time.sleep(0.2)
    raise RuntimeError(
        f"Failed to start pipeline with candidates={profiles}: {last_err}"
    )


def _build_profiles(
    *,
    req_width: int,
    req_height: int,
    req_fps: int,
    usb_desc: str,
    total_cams: int,
) -> Tuple[Tuple[int, int, int], ...]:
    profiles: List[Tuple[int, int, int]] = []

    if total_cams >= 3:
        # Match calibrate defaults first: requested WH/FPS.
        profiles.extend(
            [
                (req_width, req_height, req_fps),
                (req_width, req_height, 5),
                (req_width, req_height, 15),
                (848, 480, 15),
                (640, 480, 15),
                (640, 480, 5),
                (848, 480, 30),
            ]
        )
    else:
        if usb_desc.startswith("2"):
            profiles.append((req_width, req_height, min(req_fps, 15)))
        else:
            profiles.append((req_width, req_height, req_fps))
        profiles.extend(
            [
                (848, 480, 15),
                (848, 480, 5),
                (640, 480, 30),
                (640, 480, 15),
                (640, 480, 5),
            ]
        )

    seen = set()
    uniq: List[Tuple[int, int, int]] = []
    for p in profiles:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return tuple(uniq)


def _safe_wait_frames(
    pipeline: Any,
    timeout_ms: int = 100,
) -> Optional[Any]:
    try:
        return pipeline.wait_for_frames(timeout_ms)
    except RuntimeError:
        return None


def _make_panel(
    *,
    color: Any,
    depth: Any,
    depth_scale_m_per_unit: float,
    depth_vis_min_m: float,
    depth_vis_max_m: float,
    label_lines: Tuple[str, str],
) -> Any:
    depth_colormap = colorize_depth_units(
        depth,
        depth_scale_m_per_unit=float(depth_scale_m_per_unit),
        depth_min_m=float(depth_vis_min_m),
        depth_max_m=float(depth_vis_max_m),
    )
    return _compose_panel(
        color=color, bottom_image=depth_colormap, label_lines=label_lines
    )


def _compose_panel(
    *,
    color: Any,
    bottom_image: Any,
    label_lines: Tuple[str, str],
) -> Any:
    cv2, np, _ = _runtime_imports()
    color_image = np.asarray(color, dtype=np.uint8)
    lower = np.asarray(bottom_image, dtype=np.uint8)
    if lower.ndim == 2:
        lower = cv2.cvtColor(lower, cv2.COLOR_GRAY2BGR)
    panel = np.vstack([color_image, lower])
    _draw_panel_label(panel, label_lines=label_lines, color_bgr=(255, 255, 255))
    return panel


def _update_recent_frame_times(
    frame_times: deque[float],
    *,
    now_s: float,
    frame_received: bool,
    window_s: float = MEASURED_FPS_WINDOW_S,
) -> None:
    if frame_received:
        frame_times.append(float(now_s))
    cutoff = float(now_s) - float(window_s)
    while frame_times and frame_times[0] < cutoff:
        frame_times.popleft()


def _compute_measured_fps(frame_times: deque[float]) -> float:
    if len(frame_times) < MIN_MEASURED_FPS_SAMPLES:
        return 0.0
    elapsed_s = float(frame_times[-1] - frame_times[0])
    if elapsed_s <= 1e-6:
        return 0.0
    return float((len(frame_times) - 1) / elapsed_s)


def _format_measured_rate_label(
    *,
    fps_value: float,
    sample_count: int,
) -> str:
    if int(sample_count) < MIN_MEASURED_FPS_SAMPLES:
        return "warming"
    return f"{float(fps_value):.1f}"


def _format_panel_label_lines(
    *,
    serial: str,
    usb_desc: str,
    stream_w: int,
    stream_h: int,
    configured_fps: float,
    measured_fps: float,
    measured_sample_count: int,
) -> Tuple[str, str]:
    usb_label = usb_desc or "unknown"
    line1 = (
        f"{serial} usb={usb_label} {stream_w}x{stream_h}@{float(configured_fps):.1f}fps"
    )
    measured_label = _format_measured_rate_label(
        fps_value=measured_fps,
        sample_count=measured_sample_count,
    )
    line2 = f"configured: {float(configured_fps):.1f} | measured: {measured_label}"
    return line1, line2


def _format_depth_debug_lines(
    *,
    measured_fps: float,
    measured_sample_count: int,
) -> Tuple[str, str]:
    return (
        "Depth render disabled",
        f"depth fps: {_format_measured_rate_label(fps_value=measured_fps, sample_count=measured_sample_count)}",
    )


def _draw_panel_label(
    panel: Any,
    *,
    label_lines: Tuple[str, str],
    color_bgr: Tuple[int, int, int],
) -> None:
    cv2, _, _ = _runtime_imports()
    line_height = 26
    top = 8
    bottom = top + line_height * len(label_lines) + 8
    cv2.rectangle(panel, (0, 0), (panel.shape[1] - 1, bottom), (0, 0, 0), -1)
    for idx, line in enumerate(label_lines):
        cv2.putText(
            panel,
            line,
            (8, top + 16 + idx * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color_bgr,
            2,
            cv2.LINE_AA,
        )


def _make_message_bottom(
    *,
    width: int,
    height: int,
    message_lines: Tuple[str, ...],
    color_bgr: Tuple[int, int, int] = (180, 180, 180),
) -> Any:
    cv2, np, _ = _runtime_imports()
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    if not message_lines:
        return canvas
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    line_gap = 18
    line_sizes = [
        cv2.getTextSize(line, font, font_scale, thickness)[0] for line in message_lines
    ]
    total_height = sum(size[1] for size in line_sizes) + line_gap * max(
        0, len(line_sizes) - 1
    )
    y = max(24, (height - total_height) // 2 + line_sizes[0][1])
    for line, size in zip(message_lines, line_sizes):
        x = max(8, (width - size[0]) // 2)
        cv2.putText(
            canvas,
            line,
            (x, y),
            font,
            font_scale,
            color_bgr,
            thickness,
            cv2.LINE_AA,
        )
        y += size[1] + line_gap
    return canvas


def _make_depth_debug_bottom(
    *,
    width: int,
    height: int,
    measured_fps: float,
    measured_sample_count: int,
) -> Any:
    return _make_message_bottom(
        width=width,
        height=height,
        message_lines=_format_depth_debug_lines(
            measured_fps=measured_fps,
            measured_sample_count=measured_sample_count,
        ),
    )


def _fit_to_canvas(
    image: Any,
    target_width: int,
    target_height: int,
    *,
    interpolation: int,
    fill_value: int = 0,
) -> Any:
    cv2, np, _ = _runtime_imports()
    src_height, src_width = image.shape[:2]
    if src_width <= 0 or src_height <= 0:
        raise ValueError(f"Invalid image shape: {image.shape}")

    scale = min(target_width / src_width, target_height / src_height)
    scaled_width = max(1, int(round(src_width * scale)))
    scaled_height = max(1, int(round(src_height * scale)))
    resized = cv2.resize(
        image, (scaled_width, scaled_height), interpolation=interpolation
    )

    if image.ndim == 2:
        canvas = np.full((target_height, target_width), fill_value, dtype=image.dtype)
    else:
        channels = image.shape[2]
        canvas = np.full(
            (target_height, target_width, channels), fill_value, dtype=image.dtype
        )

    y0 = (target_height - scaled_height) // 2
    x0 = (target_width - scaled_width) // 2
    canvas[y0 : y0 + scaled_height, x0 : x0 + scaled_width] = resized
    return canvas


def _query_screen_size() -> tuple[int, int]:
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        width = int(root.winfo_screenwidth())
        height = int(root.winfo_screenheight())
        root.destroy()
        return width, height
    except Exception:
        return 1920, 1080


def _get_screen_size() -> tuple[int, int]:
    global _SCREEN_SIZE_CACHE
    if _SCREEN_SIZE_CACHE is None:
        _SCREEN_SIZE_CACHE = _query_screen_size()
    return _SCREEN_SIZE_CACHE


def _compute_display_target_size(
    *,
    grid_height: int,
    grid_width: int,
    screen_size: Optional[Tuple[int, int]] = None,
) -> Tuple[int, int]:
    if grid_width <= 0 or grid_height <= 0:
        raise ValueError(f"Invalid grid size: {(grid_height, grid_width)}")
    if screen_size is None:
        screen_width, screen_height = _get_screen_size()
    else:
        screen_width, screen_height = screen_size
    screen_width = int(screen_width)
    screen_height = int(screen_height)
    max_width = max(640, screen_width - 120)
    max_height = max(480, screen_height - 180)
    scale = min(max_width / grid_width, max_height / grid_height, 1.0)
    target_width = max(1, int(round(grid_width * scale)))
    target_height = max(1, int(round(grid_height * scale)))
    return target_width, target_height


def _fit_grid_for_display(
    grid: Any,
    *,
    target_size: Optional[Tuple[int, int]] = None,
) -> Any:
    cv2, _, _ = _runtime_imports()
    grid_height, grid_width = grid.shape[:2]
    if target_size is None:
        target_width, target_height = _compute_display_target_size(
            grid_height=grid_height,
            grid_width=grid_width,
        )
    else:
        target_width, target_height = map(int, target_size)
    if target_width == grid_width and target_height == grid_height:
        return grid
    return cv2.resize(grid, (target_width, target_height), interpolation=cv2.INTER_AREA)


def _empty_panel(width: int, height: int, label_lines: Tuple[str, str]) -> np.ndarray:
    cv2, np, _ = _runtime_imports()
    panel = np.zeros((height * 2, width, 3), dtype=np.uint8)
    _draw_panel_label(panel, label_lines=label_lines, color_bgr=(0, 0, 255))
    return panel


def _tile_panels(
    panels: List[Any],
    panel_h: int,
    panel_w: int,
) -> Any:
    _, np, _ = _runtime_imports()
    if not panels:
        return np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    cols = 2
    rows = int(math.ceil(len(panels) / cols))
    grid = np.zeros((rows * panel_h, cols * panel_w, 3), dtype=np.uint8)
    for idx, panel in enumerate(panels):
        r = idx // cols
        c = idx % cols
        y0, y1 = r * panel_h, (r + 1) * panel_h
        x0, x1 = c * panel_w, (c + 1) * panel_w
        grid[y0:y1, x0:x1] = panel
    return grid


def _enumerate_d400_devices(ctx: Any) -> List[Any]:
    _, _, rs = _runtime_imports()
    devices: List[Any] = []
    for dev in ctx.query_devices():
        try:
            name = dev.get_info(rs.camera_info.name).lower()
            product_line = dev.get_info(rs.camera_info.product_line)
        except Exception:
            continue
        if name == "platform camera":
            continue
        if product_line != "D400":
            continue
        devices.append(dev)
    return devices


def _device_serial(dev: Any) -> str:
    _, _, rs = _runtime_imports()
    return str(dev.get_info(rs.camera_info.serial_number))


def _order_devices_by_serial(
    devices: List[Any],
    *,
    serials: Optional[List[str]],
    max_cams: int,
) -> List[Any]:
    by_serial: Dict[str, Any] = {}
    for dev in devices:
        serial = _device_serial(dev)
        if serial in by_serial:
            raise ValueError(f"Duplicate RealSense serial detected: {serial}")
        by_serial[serial] = dev
    if serials:
        missing = [serial for serial in serials if serial not in by_serial]
        if missing:
            raise ValueError(
                f"Requested serials not detected by librealsense: {missing}"
            )
        return [by_serial[serial] for serial in serials]
    return sorted(devices, key=_device_serial)[: int(max_cams)]


def _build_camera_state(
    *,
    serial: str,
    usb_desc: str,
    pipeline: Any,
    align: Any,
    stream_w: int,
    stream_h: int,
    fps_used: int,
) -> dict[str, Any]:
    return {
        "serial": serial,
        "usb": usb_desc,
        "pipeline": pipeline,
        "align": align,
        "fps": float(fps_used),
        "stream_w": int(stream_w),
        "stream_h": int(stream_h),
        "lock": threading.Lock(),
        "capture_thread": None,
        "capture_seq": 0,
        "last_rendered_capture_seq": 0,
        "frame_times": deque(),
        "measured_fps": 0.0,
        "last_color": None,
        "last_depth": None,
        "last_depth_scale_m_per_unit": 0.001,
    }


def _capture_loop(
    cam_state: dict[str, Any],
    *,
    target_width: int,
    target_height: int,
    stop_event: threading.Event,
) -> None:
    cv2, np, _ = _runtime_imports()
    while not stop_event.is_set():
        frames = _safe_wait_frames(
            cam_state["pipeline"],
            timeout_ms=CAPTURE_THREAD_TIMEOUT_MS,
        )
        if not frames:
            continue
        try:
            frames = cam_state["align"].process(frames)
            depth = frames.get_depth_frame()
            color = frames.get_color_frame()
        except RuntimeError:
            continue
        if not depth or not color:
            continue

        color_np = np.asanyarray(color.get_data())
        depth_np = np.asanyarray(depth.get_data())
        try:
            depth_scale_m_per_unit = float(depth.get_units())
        except Exception:
            depth_scale_m_per_unit = 0.001
        if color_np.shape[1] != target_width or color_np.shape[0] != target_height:
            color_np = _fit_to_canvas(
                color_np,
                target_width,
                target_height,
                interpolation=cv2.INTER_LINEAR,
            )
        if depth_np.shape[1] != target_width or depth_np.shape[0] != target_height:
            depth_np = _fit_to_canvas(
                depth_np,
                target_width,
                target_height,
                interpolation=cv2.INTER_NEAREST,
            )

        with cam_state["lock"]:
            cam_state["capture_seq"] = int(cam_state["capture_seq"]) + 1
            cam_state["last_color"] = color_np
            cam_state["last_depth"] = depth_np
            cam_state["last_depth_scale_m_per_unit"] = depth_scale_m_per_unit


def _render_panel(
    cam_state: dict[str, Any],
    *,
    width: int,
    height: int,
    depth_vis_min_m: float,
    depth_vis_max_m: float,
    depth_render_mode: str,
) -> Any:
    _, np, _ = _runtime_imports()
    now_s = time.perf_counter()
    with cam_state["lock"]:
        capture_seq = int(cam_state["capture_seq"])
        frame_received = capture_seq > int(cam_state["last_rendered_capture_seq"])
        if frame_received:
            cam_state["last_rendered_capture_seq"] = capture_seq
        _update_recent_frame_times(
            cam_state["frame_times"],
            now_s=now_s,
            frame_received=frame_received,
        )
        cam_state["measured_fps"] = _compute_measured_fps(cam_state["frame_times"])
        serial = str(cam_state["serial"])
        usb_desc = str(cam_state["usb"])
        stream_w = int(cam_state["stream_w"])
        stream_h = int(cam_state["stream_h"])
        configured_fps = float(cam_state["fps"])
        measured_fps = float(cam_state["measured_fps"])
        measured_sample_count = len(cam_state["frame_times"])
        latest_color = (
            None
            if cam_state["last_color"] is None
            else np.asarray(cam_state["last_color"]).copy()
        )
        latest_depth = (
            None
            if cam_state["last_depth"] is None
            else np.asarray(cam_state["last_depth"]).copy()
        )
        depth_scale_m_per_unit = float(cam_state["last_depth_scale_m_per_unit"])

    label_lines = _format_panel_label_lines(
        serial=serial,
        usb_desc=usb_desc,
        stream_w=stream_w,
        stream_h=stream_h,
        configured_fps=configured_fps,
        measured_fps=measured_fps,
        measured_sample_count=measured_sample_count,
    )
    if latest_color is None or latest_depth is None:
        return _empty_panel(width, height, label_lines)
    if depth_render_mode == "fps_placeholder":
        return _compose_panel(
            color=latest_color,
            bottom_image=_make_depth_debug_bottom(
                width=width,
                height=height,
                measured_fps=measured_fps,
                measured_sample_count=measured_sample_count,
            ),
            label_lines=label_lines,
        )
    return _make_panel(
        color=latest_color,
        depth=latest_depth,
        depth_scale_m_per_unit=depth_scale_m_per_unit,
        depth_vis_min_m=float(depth_vis_min_m),
        depth_vis_max_m=float(depth_vis_max_m),
        label_lines=label_lines,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--max-cams", type=int, default=DEFAULT_NUM_CAM)
    parser.add_argument("--serials", nargs="*", default=None)
    parser.add_argument("--auto-exposure", action="store_true")
    parser.add_argument("--exposure", type=float, default=70.0)
    parser.add_argument("--gain", type=float, default=60.0)
    parser.add_argument(
        "--depth-vis-min-m", type=float, default=DEFAULT_DEPTH_VIS_MIN_M
    )
    parser.add_argument(
        "--depth-vis-max-m", type=float, default=DEFAULT_DEPTH_VIS_MAX_M
    )
    parser.add_argument(
        "--depth-render-mode",
        choices=DEPTH_RENDER_MODE_CHOICES,
        default="colormap",
        help="Bottom-panel depth display mode. Use fps_placeholder to skip depth rendering and show received depth FPS only.",
    )
    args = parser.parse_args()
    cv2, np, rs = _runtime_imports()
    if float(args.depth_vis_max_m) <= float(args.depth_vis_min_m):
        raise ValueError(
            f"--depth-vis-max-m must be greater than --depth-vis-min-m. "
            f"Got {args.depth_vis_min_m=} {args.depth_vis_max_m=}"
        )

    ctx = rs.context()
    devices = _enumerate_d400_devices(ctx)
    if not devices:
        print("No D400 RealSense device detected by librealsense.", flush=True)
        return 2

    devices = _order_devices_by_serial(
        devices,
        serials=args.serials if args.serials else None,
        max_cams=int(args.max_cams),
    )
    serials = [dev.get_info(rs.camera_info.serial_number) for dev in devices]
    print(f"Detected {len(devices)} camera(s): {', '.join(serials)}", flush=True)

    stop_event = threading.Event()
    cams: List[Dict[str, Any]] = []
    for dev in devices:
        serial = dev.get_info(rs.camera_info.serial_number)
        try:
            usb_desc = dev.get_info(rs.camera_info.usb_type_descriptor)
        except Exception:
            usb_desc = ""

        profiles = _build_profiles(
            req_width=args.width,
            req_height=args.height,
            req_fps=args.fps,
            usb_desc=usb_desc,
            total_cams=len(devices),
        )
        try:
            pipeline, (stream_w, stream_h, fps_used) = _start_pipeline(
                ctx=ctx,
                serial=serial,
                profiles=profiles,
            )
        except Exception as e:
            print(
                f"[ERROR] Could not start {serial}: {type(e).__name__}: {e}", flush=True
            )
            continue

        target_exposure = float(DEFAULT_EXPOSURE_OVERRIDES.get(serial, args.exposure))
        ae, exp, g = _apply_color_controls(
            pipeline=pipeline,
            auto_exposure=args.auto_exposure,
            exposure=target_exposure,
            gain=args.gain,
        )
        cam_state = _build_camera_state(
            serial=serial,
            usb_desc=usb_desc,
            pipeline=pipeline,
            align=rs.align(rs.stream.color),
            stream_w=stream_w,
            stream_h=stream_h,
            fps_used=fps_used,
        )
        cam_state["capture_thread"] = threading.Thread(
            target=_capture_loop,
            args=(cam_state,),
            kwargs={
                "target_width": int(args.width),
                "target_height": int(args.height),
                "stop_event": stop_event,
            },
            daemon=True,
            name=f"native-capture-{serial}",
        )
        cam_state["capture_thread"].start()
        cams.append(cam_state)
        print(
            f"Started {serial} usb={usb_desc or 'unknown'} "
            f"at {stream_w}x{stream_h}@{fps_used} "
            f"(AE={ae}, EXP={exp}, GAIN={g}, target_exp={target_exposure})",
            flush=True,
        )

    if not cams:
        print("Failed to start any camera.", flush=True)
        return 3

    print(f"Running with {len(cams)} camera(s). Press q/Esc to quit.", flush=True)

    panel_h = args.height * 2
    panel_w = args.width
    grid_cols = 2
    grid_rows = int(math.ceil(len(cams) / grid_cols))
    display_target_size = _compute_display_target_size(
        grid_height=grid_rows * panel_h,
        grid_width=grid_cols * panel_w,
    )
    window_flags = cv2.WINDOW_NORMAL
    if hasattr(cv2, "WINDOW_KEEPRATIO"):
        window_flags |= cv2.WINDOW_KEEPRATIO
    cv2.namedWindow("RealSense Viewer", window_flags)

    try:
        while True:
            panels: List[np.ndarray] = []
            for cam in cams:
                panels.append(
                    _render_panel(
                        cam,
                        width=int(args.width),
                        height=int(args.height),
                        depth_vis_min_m=float(args.depth_vis_min_m),
                        depth_vis_max_m=float(args.depth_vis_max_m),
                        depth_render_mode=str(args.depth_render_mode),
                    )
                )

            grid = _tile_panels(panels, panel_h, panel_w)
            display_grid = _fit_grid_for_display(grid, target_size=display_target_size)
            cv2.imshow("RealSense Viewer", display_grid)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        stop_event.set()
        for cam in cams:
            try:
                cam["pipeline"].stop()
            except Exception:
                pass
        for cam in cams:
            try:
                cam["capture_thread"].join(timeout=2.0)
            except Exception:
                pass
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
