#!/usr/bin/env python3
"""Live preview for RGB + Fast-FoundationStereo depth.

Shows live RGB (top) and color-aligned FFS depth colormap (bottom) per camera.
Press `q` or `Esc` to exit.
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import queue
import threading
import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

from cameras_viewer import (
    DEFAULT_EXPOSURE_OVERRIDES,
    MIN_MEASURED_FPS_SAMPLES,
    _apply_color_controls,
    _build_profiles,
    _compute_measured_fps,
    _enumerate_d400_devices,
    _fit_grid_for_display,
    _fit_to_canvas,
    _runtime_imports,
    _tile_panels,
    _update_recent_frame_times,
)
from data_process.depth_backends import (
    FastFoundationStereoRunner,
    FastFoundationStereoTensorRTRunner,
    align_depth_to_color,
    load_tensorrt_model_config,
    resolve_tensorrt_image_transform,
)
from data_process.visualization.depth_colormap import (
    DEFAULT_DEPTH_VIS_MAX_M,
    DEFAULT_DEPTH_VIS_MIN_M,
    colorize_depth_meters,
)
from qqtt.env.camera.defaults import DEFAULT_FPS, DEFAULT_HEIGHT, DEFAULT_NUM_CAM, DEFAULT_WIDTH

if TYPE_CHECKING:
    import cv2
    import pyrealsense2 as rs


LATEST_QUEUE_SIZE = 1
CAPTURE_QUEUE_TIMEOUT_MS = 100
RESULT_QUEUE_TIMEOUT_S = 0.1
SHARED_WORKER_IDLE_SLEEP_S = 0.005
DEFAULT_FFS_TRT_MODEL_DIR = Path(__file__).resolve().parent / "data" / "ffs_proof_of_life" / "trt_two_stage_864x480_wsl"


def _intrinsics_to_matrix(intrinsics: Any) -> list[list[float]]:
    return [
        [float(intrinsics.fx), 0.0, float(intrinsics.ppx)],
        [0.0, float(intrinsics.fy), float(intrinsics.ppy)],
        [0.0, 0.0, 1.0],
    ]


def _extrinsics_to_matrix(extrinsics: Any) -> list[list[float]]:
    rotation = list(map(float, extrinsics.rotation))
    translation = list(map(float, extrinsics.translation))
    return [
        [rotation[0], rotation[1], rotation[2], translation[0]],
        [rotation[3], rotation[4], rotation[5], translation[1]],
        [rotation[6], rotation[7], rotation[8], translation[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _translation_norm(extrinsics: Any) -> float:
    tx, ty, tz = map(float, extrinsics.translation)
    return float(math.sqrt(tx * tx + ty * ty + tz * tz))


def _extract_runtime_geometry(pipeline: Any) -> dict[str, Any]:
    _, _, rs = _runtime_imports()
    profile = pipeline.get_active_profile()
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    ir_left_profile = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
    ir_right_profile = profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()
    ir_left_to_color = ir_left_profile.get_extrinsics_to(color_profile)
    ir_left_to_right = ir_left_profile.get_extrinsics_to(ir_right_profile)
    return {
        "K_color": _intrinsics_to_matrix(color_profile.get_intrinsics()),
        "K_ir_left": _intrinsics_to_matrix(ir_left_profile.get_intrinsics()),
        "T_ir_left_to_color": _extrinsics_to_matrix(ir_left_to_color),
        "T_ir_left_to_right": _extrinsics_to_matrix(ir_left_to_right),
        "ir_baseline_m": _translation_norm(ir_left_to_right),
    }


def _start_pipeline_ffs(
    *,
    ctx: Any,
    serial: str,
    profiles: Tuple[Tuple[int, int, int], ...],
) -> Tuple[Any, Tuple[int, int, int], dict[str, Any]]:
    _, _, rs = _runtime_imports()
    last_err: Optional[BaseException] = None
    for width, height, fps in profiles:
        pipeline = rs.pipeline(ctx)
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.infrared, 1, width, height, rs.format.y8, fps)
        config.enable_stream(rs.stream.infrared, 2, width, height, rs.format.y8, fps)
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
            geometry = _extract_runtime_geometry(pipeline)
            return pipeline, (width, height, fps), geometry
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
    raise RuntimeError(f"Failed to start pipeline with candidates={profiles}: {last_err}")


def _rate_label(*, fps_value: float, sample_count: int) -> str:
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
    capture_fps: float,
    capture_sample_count: int,
    ffs_fps: float,
    ffs_sample_count: int,
) -> Tuple[str, str]:
    usb_label = usb_desc or "unknown"
    line1 = f"{serial} usb={usb_label} {stream_w}x{stream_h}@{float(configured_fps):.1f}fps"
    line2 = (
        f"capture: {_rate_label(fps_value=capture_fps, sample_count=capture_sample_count)}"
        f" | ffs: {_rate_label(fps_value=ffs_fps, sample_count=ffs_sample_count)}"
    )
    return line1, line2


def _summarize_runtime_stats(per_camera_stats: List[dict[str, Any]]) -> dict[str, Any]:
    return {
        "camera_count": int(len(per_camera_stats)),
        "aggregate_capture_fps": float(sum(float(item["capture_fps"]) for item in per_camera_stats)),
        "aggregate_ffs_fps": float(sum(float(item["ffs_fps"]) for item in per_camera_stats)),
        "per_camera": list(per_camera_stats),
    }


def _format_runtime_stats_lines(*, elapsed_s: float, runtime_stats: dict[str, Any]) -> Tuple[str, ...]:
    lines: list[str] = [
        (
            f"[stats t={float(elapsed_s):.1f}s cams={int(runtime_stats['camera_count'])}] "
            f"capture_sum={float(runtime_stats['aggregate_capture_fps']):.1f} "
            f"ffs_sum={float(runtime_stats['aggregate_ffs_fps']):.1f}"
        )
    ]
    for item in runtime_stats["per_camera"]:
        lines.append(
            (
                f"[stats cam{int(item['camera_idx'])} {item['serial']}] "
                f"capture={_rate_label(fps_value=float(item['capture_fps']), sample_count=int(item['capture_sample_count']))} "
                f"ffs={_rate_label(fps_value=float(item['ffs_fps']), sample_count=int(item['ffs_sample_count']))} "
                f"infer_ms={float(item['latest_inference_ms']):.1f} "
                f"seq_gap={int(item['seq_gap'])}"
                + (f" error={item['worker_error']}" if item["worker_error"] else "")
            )
        )
    return tuple(lines)


def _collect_runtime_stats(cams: List[dict[str, Any]]) -> dict[str, Any]:
    per_camera_stats: list[dict[str, Any]] = []
    for cam_state in cams:
        with cam_state["lock"]:
            capture_seq = int(cam_state["capture_seq"])
            latest_ffs_capture_seq = int(cam_state["latest_ffs_capture_seq"])
            if latest_ffs_capture_seq >= 0:
                seq_gap = max(0, capture_seq - latest_ffs_capture_seq)
            else:
                seq_gap = max(0, capture_seq)
            per_camera_stats.append(
                {
                    "camera_idx": int(cam_state["camera_idx"]),
                    "serial": str(cam_state["serial"]),
                    "capture_fps": float(cam_state["capture_fps"]),
                    "capture_sample_count": int(len(cam_state["capture_frame_times"])),
                    "ffs_fps": float(cam_state["ffs_fps"]),
                    "ffs_sample_count": int(len(cam_state["ffs_frame_times"])),
                    "latest_inference_ms": float(cam_state["last_inference_s"]) * 1000.0,
                    "seq_gap": int(seq_gap),
                    "worker_error": None if cam_state["worker_error"] is None else str(cam_state["worker_error"]),
                }
            )
    per_camera_stats.sort(key=lambda item: int(item["camera_idx"]))
    return _summarize_runtime_stats(per_camera_stats)


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


def _make_waiting_bottom(
    width: int,
    height: int,
    *,
    message: str,
    color_bgr: Tuple[int, int, int] = (180, 180, 180),
) -> np.ndarray:
    cv2, _, _ = _runtime_imports()
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    text_x = max(8, (width - text_size[0]) // 2)
    text_y = max(24, (height + text_size[1]) // 2)
    cv2.putText(
        canvas,
        message,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color_bgr,
        2,
        cv2.LINE_AA,
    )
    return canvas


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


def _empty_panel(width: int, height: int, label_lines: Tuple[str, str]) -> np.ndarray:
    panel = _compose_panel(
        color=np.zeros((height, width, 3), dtype=np.uint8),
        bottom_image=_make_waiting_bottom(width, height, message="Waiting for color + IR"),
        label_lines=label_lines,
    )
    return panel


def _fit_grid_for_window(grid: Any, *, window_name: str) -> Any:
    del window_name
    # The base viewer's screen-bounded fit path is stable under WSL Qt/OpenCV.
    # Using getWindowImageRect() here can bootstrap the FFS grid into a tiny
    # thumbnail if the initial image rect is reported before the first real frame.
    return _fit_grid_for_display(grid)


def _put_latest(queue_obj: Any, item: Any) -> None:
    try:
        queue_obj.put_nowait(item)
        return
    except queue.Full:
        pass
    try:
        while True:
            queue_obj.get_nowait()
    except queue.Empty:
        pass
    try:
        queue_obj.put_nowait(item)
    except queue.Full:
        pass


def _reproject_ffs_depth_to_color(
    depth_ir_left_m: np.ndarray,
    *,
    K_ir_left: np.ndarray,
    T_ir_left_to_color: np.ndarray,
    K_color: np.ndarray,
    output_shape: tuple[int, int],
) -> np.ndarray:
    return align_depth_to_color(
        np.asarray(depth_ir_left_m, dtype=np.float32),
        np.asarray(K_ir_left, dtype=np.float32),
        np.asarray(T_ir_left_to_color, dtype=np.float32),
        np.asarray(K_color, dtype=np.float32),
        output_shape=output_shape,
        invalid_value=0.0,
    )


def _build_ffs_runner(
    *,
    runner_backend: str,
    ffs_repo: str,
    model_path: str | None,
    ffs_scale: float,
    ffs_valid_iters: int,
    ffs_max_disp: int,
    trt_model_dir: str | None,
    trt_root: str | None,
) -> Any:
    if runner_backend == "pytorch":
        if model_path is None:
            raise ValueError("Missing model_path for PyTorch FFS backend.")
        return FastFoundationStereoRunner(
            ffs_repo=ffs_repo,
            model_path=model_path,
            scale=ffs_scale,
            valid_iters=ffs_valid_iters,
            max_disp=ffs_max_disp,
        )
    if runner_backend == "tensorrt":
        if trt_model_dir is None:
            raise ValueError("Missing trt_model_dir for TensorRT FFS backend.")
        return FastFoundationStereoTensorRTRunner(
            ffs_repo=ffs_repo,
            model_dir=trt_model_dir,
            trt_root=trt_root,
        )
    raise ValueError(f"Unsupported FFS backend: {runner_backend}")


def _drain_shared_worker_next_request(
    *,
    camera_order: List[int],
    request_queues: dict[int, Any],
    closed_camera_indices: set[int],
    start_cursor: int,
) -> tuple[int | None, Any | None, int, set[int]]:
    queue_count = len(camera_order)
    if queue_count <= 0:
        return None, None, 0, set(closed_camera_indices)

    cursor = int(start_cursor) % queue_count
    base_cursor = cursor
    closed = set(int(idx) for idx in closed_camera_indices)
    for offset in range(queue_count):
        pos = (base_cursor + offset) % queue_count
        camera_idx = int(camera_order[pos])
        if camera_idx in closed:
            continue
        try:
            payload = request_queues[camera_idx].get_nowait()
        except queue.Empty:
            continue
        next_cursor = (pos + 1) % queue_count
        if payload is None:
            closed.add(camera_idx)
            cursor = next_cursor
            continue
        return camera_idx, payload, next_cursor, closed
    return None, None, cursor, closed


def _resolve_ffs_worker_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    backend = str(args.ffs_backend)
    if not args.ffs_repo.exists():
        raise FileNotFoundError(f"Missing --ffs_repo: {args.ffs_repo}")

    worker_kwargs: dict[str, Any] = {
        "runner_backend": backend,
        "ffs_repo": str(args.ffs_repo.resolve()),
        "model_path": None,
        "ffs_scale": float(args.ffs_scale),
        "ffs_valid_iters": int(args.ffs_valid_iters),
        "ffs_max_disp": int(args.ffs_max_disp),
        "trt_model_dir": None,
        "trt_root": None,
        "trt_engine_height": None,
        "trt_engine_width": None,
    }
    if backend == "pytorch":
        if args.ffs_model_path is None:
            raise ValueError("--ffs_model_path is required for --ffs_backend pytorch")
        if not args.ffs_model_path.exists():
            raise FileNotFoundError(f"Missing --ffs_model_path: {args.ffs_model_path}")
        worker_kwargs["model_path"] = str(args.ffs_model_path.resolve())
        return worker_kwargs
    if backend != "tensorrt":
        raise ValueError(f"Unsupported --ffs_backend: {backend}")

    if args.ffs_trt_model_dir is None:
        raise ValueError("--ffs_trt_model_dir is required for --ffs_backend tensorrt")
    if not args.ffs_trt_model_dir.exists():
        raise FileNotFoundError(f"Missing --ffs_trt_model_dir: {args.ffs_trt_model_dir}")
    for engine_name in ("feature_runner.engine", "post_runner.engine"):
        engine_path = args.ffs_trt_model_dir / engine_name
        if not engine_path.exists():
            raise FileNotFoundError(f"Missing TensorRT engine: {engine_path}")
    cfg = load_tensorrt_model_config(args.ffs_trt_model_dir)
    worker_kwargs["trt_model_dir"] = str(args.ffs_trt_model_dir.resolve())
    worker_kwargs["trt_engine_height"] = int(cfg["image_size"][0])
    worker_kwargs["trt_engine_width"] = int(cfg["image_size"][1])
    if args.ffs_trt_root is not None:
        if not args.ffs_trt_root.exists():
            raise FileNotFoundError(f"Missing --ffs_trt_root: {args.ffs_trt_root}")
        worker_kwargs["trt_root"] = str(args.ffs_trt_root.resolve())
    return worker_kwargs


def _format_ffs_backend_startup_note(
    *,
    runner_backend: str,
    stream_w: int,
    stream_h: int,
    worker_kwargs: dict[str, Any],
) -> str | None:
    if runner_backend != "tensorrt":
        return None
    engine_h = int(worker_kwargs["trt_engine_height"])
    engine_w = int(worker_kwargs["trt_engine_width"])
    transform = resolve_tensorrt_image_transform(
        input_height=int(stream_h),
        input_width=int(stream_w),
        engine_height=engine_h,
        engine_width=engine_w,
    )
    mode = str(transform["mode"])
    if mode == "match":
        return f"TensorRT engine {engine_w}x{engine_h} matches capture size."
    if mode == "pad":
        return (
            f"TensorRT engine {engine_w}x{engine_h}; "
            f"capture {int(stream_w)}x{int(stream_h)} will be symmetrically padded to "
            f"{engine_w}x{engine_h} before inference."
        )
    return (
        f"TensorRT engine {engine_w}x{engine_h}; "
        f"capture {int(stream_w)}x{int(stream_h)} will be resized before inference."
    )


def _capture_loop(
    cam_state: dict[str, Any],
    *,
    target_width: int,
    target_height: int,
    stop_event: threading.Event,
) -> None:
    cv2, np, _ = _runtime_imports()
    while not stop_event.is_set():
        now_s = time.perf_counter()
        frame_received = False
        try:
            frames = cam_state["pipeline"].wait_for_frames(CAPTURE_QUEUE_TIMEOUT_MS)
            color_frame = frames.get_color_frame() if frames else None
            ir_left_frame = frames.get_infrared_frame(1) if frames else None
            ir_right_frame = frames.get_infrared_frame(2) if frames else None
            if color_frame and ir_left_frame and ir_right_frame:
                color_np = np.asanyarray(color_frame.get_data())
                ir_left_np = np.asanyarray(ir_left_frame.get_data())
                ir_right_np = np.asanyarray(ir_right_frame.get_data())
                if color_np.shape[1] != target_width or color_np.shape[0] != target_height:
                    color_np = _fit_to_canvas(
                        color_np,
                        target_width,
                        target_height,
                        interpolation=cv2.INTER_LINEAR,
                    )
                if ir_left_np.shape[1] != target_width or ir_left_np.shape[0] != target_height:
                    ir_left_np = _fit_to_canvas(
                        ir_left_np,
                        target_width,
                        target_height,
                        interpolation=cv2.INTER_NEAREST,
                    )
                if ir_right_np.shape[1] != target_width or ir_right_np.shape[0] != target_height:
                    ir_right_np = _fit_to_canvas(
                        ir_right_np,
                        target_width,
                        target_height,
                        interpolation=cv2.INTER_NEAREST,
                    )
                with cam_state["lock"]:
                    next_seq = int(cam_state["capture_seq"]) + 1
                    cam_state["capture_seq"] = next_seq
                    cam_state["latest_color"] = color_np
                    cam_state["latest_ir_left"] = ir_left_np
                    cam_state["latest_ir_right"] = ir_right_np
                    _update_recent_frame_times(
                        cam_state["capture_frame_times"],
                        now_s=now_s,
                        frame_received=True,
                    )
                    cam_state["capture_fps"] = _compute_measured_fps(cam_state["capture_frame_times"])
                _put_latest(
                    cam_state["request_queue"],
                    {
                        "capture_seq": next_seq,
                        "ir_left": ir_left_np,
                        "ir_right": ir_right_np,
                    },
                )
                frame_received = True
        except RuntimeError:
            pass

        if not frame_received:
            with cam_state["lock"]:
                _update_recent_frame_times(
                    cam_state["capture_frame_times"],
                    now_s=now_s,
                    frame_received=False,
                )
                cam_state["capture_fps"] = _compute_measured_fps(cam_state["capture_frame_times"])


def _ffs_worker_loop(
    *,
    camera_idx: int,
    serial: str,
    request_queue: Any,
    result_queue: Any,
    runner_backend: str,
    ffs_repo: str,
    model_path: str | None,
    ffs_scale: float,
    ffs_valid_iters: int,
    ffs_max_disp: int,
    trt_model_dir: str | None,
    trt_root: str | None,
    trt_engine_height: int | None,
    trt_engine_width: int | None,
    geometry: dict[str, Any],
    output_shape: tuple[int, int],
) -> None:
    result_frame_times: deque[float] = deque()
    try:
        runner = _build_ffs_runner(
            runner_backend=runner_backend,
            ffs_repo=ffs_repo,
            model_path=model_path,
            ffs_scale=ffs_scale,
            ffs_valid_iters=ffs_valid_iters,
            ffs_max_disp=ffs_max_disp,
            trt_model_dir=trt_model_dir,
            trt_root=trt_root,
        )
    except Exception as exc:
        _put_latest(
            result_queue,
            {
                "camera_idx": int(camera_idx),
                "serial": serial,
                "fatal_error": f"{type(exc).__name__}: {exc}",
            },
        )
        return

    while True:
        try:
            payload = request_queue.get(timeout=RESULT_QUEUE_TIMEOUT_S)
        except queue.Empty:
            continue
        if payload is None:
            return
        try:
            infer_start_s = time.perf_counter()
            run_output = runner.run_pair(
                payload["ir_left"],
                payload["ir_right"],
                K_ir_left=np.asarray(geometry["K_ir_left"], dtype=np.float32),
                baseline_m=float(geometry["ir_baseline_m"]),
            )
            depth_color_m = _reproject_ffs_depth_to_color(
                np.asarray(run_output["depth_ir_left_m"], dtype=np.float32),
                K_ir_left=np.asarray(run_output["K_ir_left_used"], dtype=np.float32),
                T_ir_left_to_color=np.asarray(geometry["T_ir_left_to_color"], dtype=np.float32),
                K_color=np.asarray(geometry["K_color"], dtype=np.float32),
                output_shape=output_shape,
            )
            result_time_s = time.perf_counter()
            _update_recent_frame_times(
                result_frame_times,
                now_s=result_time_s,
                frame_received=True,
            )
            _put_latest(
                result_queue,
                {
                    "camera_idx": int(camera_idx),
                    "serial": serial,
                    "capture_seq": int(payload["capture_seq"]),
                    "depth_color_m": depth_color_m,
                    "worker_ffs_fps": _compute_measured_fps(result_frame_times),
                    "inference_s": float(result_time_s - infer_start_s),
                    "result_time_s": result_time_s,
                },
            )
        except Exception as exc:
            _put_latest(
                result_queue,
                {
                    "camera_idx": int(camera_idx),
                    "serial": serial,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )


def _shared_ffs_worker_loop(
    *,
    request_queues: dict[int, Any],
    result_queues: dict[int, Any],
    camera_serials: dict[int, str],
    runner_backend: str,
    ffs_repo: str,
    model_path: str | None,
    ffs_scale: float,
    ffs_valid_iters: int,
    ffs_max_disp: int,
    trt_model_dir: str | None,
    trt_root: str | None,
    trt_engine_height: int | None,
    trt_engine_width: int | None,
    geometries: dict[int, dict[str, Any]],
    output_shapes: dict[int, tuple[int, int]],
) -> None:
    del trt_engine_height, trt_engine_width
    camera_order = sorted(int(idx) for idx in request_queues.keys())
    result_frame_times: dict[int, deque[float]] = {
        int(camera_idx): deque() for camera_idx in camera_order
    }
    try:
        runner = _build_ffs_runner(
            runner_backend=runner_backend,
            ffs_repo=ffs_repo,
            model_path=model_path,
            ffs_scale=ffs_scale,
            ffs_valid_iters=ffs_valid_iters,
            ffs_max_disp=ffs_max_disp,
            trt_model_dir=trt_model_dir,
            trt_root=trt_root,
        )
    except Exception as exc:
        for camera_idx in camera_order:
            _put_latest(
                result_queues[int(camera_idx)],
                {
                    "camera_idx": int(camera_idx),
                    "serial": str(camera_serials[int(camera_idx)]),
                    "fatal_error": f"{type(exc).__name__}: {exc}",
                },
            )
        return

    cursor = 0
    closed_camera_indices: set[int] = set()
    while True:
        camera_idx, payload, cursor, closed_camera_indices = _drain_shared_worker_next_request(
            camera_order=camera_order,
            request_queues=request_queues,
            closed_camera_indices=closed_camera_indices,
            start_cursor=cursor,
        )
        if payload is None:
            if len(closed_camera_indices) >= len(camera_order):
                return
            time.sleep(SHARED_WORKER_IDLE_SLEEP_S)
            continue

        camera_idx = int(camera_idx)
        geometry = geometries[camera_idx]
        output_shape = output_shapes[camera_idx]
        serial = str(camera_serials[camera_idx])
        try:
            infer_start_s = time.perf_counter()
            run_output = runner.run_pair(
                payload["ir_left"],
                payload["ir_right"],
                K_ir_left=np.asarray(geometry["K_ir_left"], dtype=np.float32),
                baseline_m=float(geometry["ir_baseline_m"]),
            )
            depth_color_m = _reproject_ffs_depth_to_color(
                np.asarray(run_output["depth_ir_left_m"], dtype=np.float32),
                K_ir_left=np.asarray(run_output["K_ir_left_used"], dtype=np.float32),
                T_ir_left_to_color=np.asarray(geometry["T_ir_left_to_color"], dtype=np.float32),
                K_color=np.asarray(geometry["K_color"], dtype=np.float32),
                output_shape=output_shape,
            )
            result_time_s = time.perf_counter()
            _update_recent_frame_times(
                result_frame_times[camera_idx],
                now_s=result_time_s,
                frame_received=True,
            )
            _put_latest(
                result_queues[camera_idx],
                {
                    "camera_idx": camera_idx,
                    "serial": serial,
                    "capture_seq": int(payload["capture_seq"]),
                    "depth_color_m": depth_color_m,
                    "worker_ffs_fps": _compute_measured_fps(result_frame_times[camera_idx]),
                    "inference_s": float(result_time_s - infer_start_s),
                    "result_time_s": result_time_s,
                },
            )
        except Exception as exc:
            _put_latest(
                result_queues[camera_idx],
                {
                    "camera_idx": camera_idx,
                    "serial": serial,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )


def _result_loop(cam_state: dict[str, Any], *, stop_event: threading.Event) -> None:
    while True:
        now_s = time.perf_counter()
        got_result = False
        try:
            payload = cam_state["result_queue"].get(timeout=RESULT_QUEUE_TIMEOUT_S)
            got_result = True
        except queue.Empty:
            payload = None

        with cam_state["lock"]:
            if payload is not None:
                if "fatal_error" in payload:
                    cam_state["worker_error"] = str(payload["fatal_error"])
                elif "error" in payload:
                    cam_state["worker_error"] = str(payload["error"])
                else:
                    cam_state["worker_error"] = None
                    cam_state["latest_ffs_depth_m"] = payload["depth_color_m"]
                    cam_state["latest_ffs_capture_seq"] = int(payload["capture_seq"])
                    cam_state["last_worker_ffs_fps"] = float(payload["worker_ffs_fps"])
                    cam_state["last_inference_s"] = float(payload["inference_s"])
            _update_recent_frame_times(
                cam_state["ffs_frame_times"],
                now_s=now_s,
                frame_received=got_result and payload is not None and "depth_color_m" in payload,
            )
            cam_state["ffs_fps"] = _compute_measured_fps(cam_state["ffs_frame_times"])
        if stop_event.is_set() and not cam_state["worker_process"].is_alive() and not got_result:
            break


def _build_camera_state(
    *,
    camera_idx: int,
    serial: str,
    usb_desc: str,
    pipeline: Any,
    stream_w: int,
    stream_h: int,
    fps_used: int,
    geometry: dict[str, Any],
    request_queue: Any,
    result_queue: Any,
    worker_process: mp.Process,
) -> dict[str, Any]:
    return {
        "camera_idx": int(camera_idx),
        "serial": serial,
        "usb": usb_desc,
        "pipeline": pipeline,
        "stream_w": int(stream_w),
        "stream_h": int(stream_h),
        "fps": float(fps_used),
        "geometry": geometry,
        "request_queue": request_queue,
        "result_queue": result_queue,
        "worker_process": worker_process,
        "lock": threading.Lock(),
        "capture_seq": 0,
        "latest_ffs_capture_seq": -1,
        "latest_color": None,
        "latest_ir_left": None,
        "latest_ir_right": None,
        "latest_ffs_depth_m": None,
        "worker_error": None,
        "last_worker_ffs_fps": 0.0,
        "last_inference_s": 0.0,
        "capture_frame_times": deque(),
        "ffs_frame_times": deque(),
        "capture_fps": 0.0,
        "ffs_fps": 0.0,
    }


def _render_panel(cam_state: dict[str, Any], *, width: int, height: int, depth_vis_min_m: float, depth_vis_max_m: float) -> Any:
    with cam_state["lock"]:
        serial = str(cam_state["serial"])
        usb_desc = str(cam_state["usb"])
        stream_w = int(cam_state["stream_w"])
        stream_h = int(cam_state["stream_h"])
        configured_fps = float(cam_state["fps"])
        capture_fps = float(cam_state["capture_fps"])
        capture_sample_count = len(cam_state["capture_frame_times"])
        ffs_fps = float(cam_state["ffs_fps"])
        ffs_sample_count = len(cam_state["ffs_frame_times"])
        latest_color = None if cam_state["latest_color"] is None else np.asarray(cam_state["latest_color"]).copy()
        latest_ffs_depth_m = None if cam_state["latest_ffs_depth_m"] is None else np.asarray(cam_state["latest_ffs_depth_m"]).copy()
        worker_error = cam_state["worker_error"]

    label_lines = _format_panel_label_lines(
        serial=serial,
        usb_desc=usb_desc,
        stream_w=stream_w,
        stream_h=stream_h,
        configured_fps=configured_fps,
        capture_fps=capture_fps,
        capture_sample_count=capture_sample_count,
        ffs_fps=ffs_fps,
        ffs_sample_count=ffs_sample_count,
    )
    if latest_color is None:
        return _empty_panel(width, height, label_lines)
    render_width = stream_w
    render_height = stream_h
    if latest_ffs_depth_m is None:
        lower = _make_waiting_bottom(
            render_width,
            render_height,
            message="FFS error" if worker_error else "FFS warming...",
            color_bgr=(0, 0, 255) if worker_error else (180, 180, 180),
        )
    else:
        lower = colorize_depth_meters(
            latest_ffs_depth_m,
            depth_min_m=float(depth_vis_min_m),
            depth_max_m=float(depth_vis_max_m),
        )
    panel = _compose_panel(color=latest_color, bottom_image=lower, label_lines=label_lines)
    if panel.shape[0] != height * 2 or panel.shape[1] != width:
        panel = _fit_to_canvas(
            panel,
            width,
            height * 2,
            interpolation=_runtime_imports()[0].INTER_LINEAR,
        )
    return panel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live RGB + Fast-FoundationStereo viewer with color-aligned FFS depth."
    )
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--max-cams", type=int, default=DEFAULT_NUM_CAM)
    parser.add_argument("--auto-exposure", action="store_true")
    parser.add_argument("--exposure", type=float, default=70.0)
    parser.add_argument("--gain", type=float, default=60.0)
    parser.add_argument("--depth-vis-min-m", type=float, default=DEFAULT_DEPTH_VIS_MIN_M)
    parser.add_argument("--depth-vis-max-m", type=float, default=DEFAULT_DEPTH_VIS_MAX_M)
    parser.add_argument("--ffs_backend", choices=("pytorch", "tensorrt"), default="tensorrt")
    parser.add_argument("--ffs_repo", type=Path, required=True)
    parser.add_argument("--ffs_model_path", type=Path, default=None)
    parser.add_argument("--ffs_scale", type=float, default=1.0)
    parser.add_argument("--ffs_valid_iters", type=int, default=8)
    parser.add_argument("--ffs_max_disp", type=int, default=192)
    parser.add_argument("--ffs_worker_mode", choices=("per_camera", "shared"), default="per_camera")
    parser.add_argument("--ffs_trt_model_dir", type=Path, default=DEFAULT_FFS_TRT_MODEL_DIR)
    parser.add_argument("--ffs_trt_root", type=Path, default=None)
    parser.add_argument("--duration-s", type=float, default=0.0)
    parser.add_argument("--stats-log-interval-s", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    mp.freeze_support()
    args = parse_args()
    cv2, np, rs = _runtime_imports()
    if float(args.depth_vis_max_m) <= float(args.depth_vis_min_m):
        raise ValueError(
            f"--depth-vis-max-m must be greater than --depth-vis-min-m. "
            f"Got {args.depth_vis_min_m=} {args.depth_vis_max_m=}"
        )
    worker_runner_kwargs = _resolve_ffs_worker_kwargs(args)

    ctx = rs.context()
    devices = _enumerate_d400_devices(ctx)
    if not devices:
        print("No D400 RealSense device detected by librealsense.", flush=True)
        return 2

    devices = devices[: args.max_cams]
    serials = [dev.get_info(rs.camera_info.serial_number) for dev in devices]
    print(f"Detected {len(devices)} camera(s): {', '.join(serials)}", flush=True)

    mp_ctx = mp.get_context("spawn")
    stop_event = threading.Event()
    cams: List[Dict[str, Any]] = []
    camera_specs: List[Dict[str, Any]] = []
    worker_processes: List[mp.Process] = []

    for camera_idx, dev in enumerate(devices):
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
            pipeline, (stream_w, stream_h, fps_used), geometry = _start_pipeline_ffs(
                ctx=ctx,
                serial=serial,
                profiles=profiles,
            )
        except Exception as exc:
            print(f"[ERROR] Could not start {serial}: {type(exc).__name__}: {exc}", flush=True)
            continue

        target_exposure = float(DEFAULT_EXPOSURE_OVERRIDES.get(serial, args.exposure))
        ae, exp, g = _apply_color_controls(
            pipeline=pipeline,
            auto_exposure=args.auto_exposure,
            exposure=target_exposure,
            gain=args.gain,
        )

        request_queue = mp_ctx.Queue(maxsize=LATEST_QUEUE_SIZE)
        result_queue = mp_ctx.Queue(maxsize=LATEST_QUEUE_SIZE)
        camera_specs.append(
            {
                "camera_idx": int(camera_idx),
                "serial": serial,
                "usb_desc": usb_desc,
                "pipeline": pipeline,
                "stream_w": int(stream_w),
                "stream_h": int(stream_h),
                "fps_used": int(fps_used),
                "geometry": geometry,
                "request_queue": request_queue,
                "result_queue": result_queue,
                "ae": ae,
                "exp": exp,
                "gain": g,
                "target_exposure": target_exposure,
            }
        )

    if not camera_specs:
        print("Failed to start any camera.", flush=True)
        return 3

    worker_mode = str(args.ffs_worker_mode)
    if worker_mode == "per_camera":
        for spec in camera_specs:
            worker_process = mp_ctx.Process(
                target=_ffs_worker_loop,
                kwargs={
                    "camera_idx": int(spec["camera_idx"]),
                    "serial": str(spec["serial"]),
                    "request_queue": spec["request_queue"],
                    "result_queue": spec["result_queue"],
                    **worker_runner_kwargs,
                    "geometry": spec["geometry"],
                    "output_shape": (int(spec["stream_h"]), int(spec["stream_w"])),
                },
                daemon=True,
            )
            worker_process.start()
            spec["worker_process"] = worker_process
            worker_processes.append(worker_process)
    else:
        shared_worker_process = mp_ctx.Process(
            target=_shared_ffs_worker_loop,
            kwargs={
                "request_queues": {
                    int(spec["camera_idx"]): spec["request_queue"] for spec in camera_specs
                },
                "result_queues": {
                    int(spec["camera_idx"]): spec["result_queue"] for spec in camera_specs
                },
                "camera_serials": {
                    int(spec["camera_idx"]): str(spec["serial"]) for spec in camera_specs
                },
                **worker_runner_kwargs,
                "geometries": {
                    int(spec["camera_idx"]): spec["geometry"] for spec in camera_specs
                },
                "output_shapes": {
                    int(spec["camera_idx"]): (int(spec["stream_h"]), int(spec["stream_w"]))
                    for spec in camera_specs
                },
            },
            daemon=True,
        )
        shared_worker_process.start()
        worker_processes.append(shared_worker_process)
        for spec in camera_specs:
            spec["worker_process"] = shared_worker_process

    print(
        f"FFS worker topology: {worker_mode} "
        f"({len(worker_processes)} process(es) for {len(camera_specs)} active camera(s))",
        flush=True,
    )

    for spec in camera_specs:
        cam_state = _build_camera_state(
            camera_idx=int(spec["camera_idx"]),
            serial=str(spec["serial"]),
            usb_desc=str(spec["usb_desc"]),
            pipeline=spec["pipeline"],
            stream_w=int(spec["stream_w"]),
            stream_h=int(spec["stream_h"]),
            fps_used=int(spec["fps_used"]),
            geometry=spec["geometry"],
            request_queue=spec["request_queue"],
            result_queue=spec["result_queue"],
            worker_process=spec["worker_process"],
        )
        cam_state["capture_thread"] = threading.Thread(
            target=_capture_loop,
            args=(cam_state,),
            kwargs={
                "target_width": int(spec["stream_w"]),
                "target_height": int(spec["stream_h"]),
                "stop_event": stop_event,
            },
            daemon=True,
            name=f"capture-{spec['serial']}",
        )
        cam_state["result_thread"] = threading.Thread(
            target=_result_loop,
            args=(cam_state,),
            kwargs={"stop_event": stop_event},
            daemon=True,
            name=f"ffs-result-{spec['serial']}",
        )
        cam_state["capture_thread"].start()
        cam_state["result_thread"].start()
        cams.append(cam_state)

        print(
            f"Started {spec['serial']} usb={spec['usb_desc'] or 'unknown'} "
            f"at {int(spec['stream_w'])}x{int(spec['stream_h'])}@{int(spec['fps_used'])} "
            f"backend={args.ffs_backend} worker_mode={worker_mode} "
            f"(AE={spec['ae']}, EXP={spec['exp']}, GAIN={spec['gain']}, target_exp={float(spec['target_exposure'])})",
            flush=True,
        )
        startup_note = _format_ffs_backend_startup_note(
            runner_backend=str(worker_runner_kwargs["runner_backend"]),
            stream_w=int(spec["stream_w"]),
            stream_h=int(spec["stream_h"]),
            worker_kwargs=worker_runner_kwargs,
        )
        if startup_note:
            print(f"[info {spec['serial']}] {startup_note}", flush=True)

    print(f"Running with {len(cams)} camera(s). Press q/Esc to quit.", flush=True)

    panel_h = args.height * 2
    panel_w = args.width
    window_flags = cv2.WINDOW_NORMAL
    if hasattr(cv2, "WINDOW_KEEPRATIO"):
        window_flags |= cv2.WINDOW_KEEPRATIO
    cv2.namedWindow("RealSense FFS Viewer", window_flags)
    loop_start_s = time.perf_counter()
    next_stats_log_s: Optional[float]
    if float(args.stats_log_interval_s) > 0:
        next_stats_log_s = loop_start_s + float(args.stats_log_interval_s)
    else:
        next_stats_log_s = None

    try:
        while True:
            panels: List[np.ndarray] = []
            for cam_state in cams:
                panels.append(
                    _render_panel(
                        cam_state,
                        width=int(args.width),
                        height=int(args.height),
                        depth_vis_min_m=float(args.depth_vis_min_m),
                        depth_vis_max_m=float(args.depth_vis_max_m),
                    )
                )
            grid = _tile_panels(panels, panel_h, panel_w)
            display_grid = _fit_grid_for_window(grid, window_name="RealSense FFS Viewer")
            cv2.imshow("RealSense FFS Viewer", display_grid)
            key = cv2.waitKey(1) & 0xFF
            now_s = time.perf_counter()
            if next_stats_log_s is not None and now_s >= next_stats_log_s:
                runtime_stats = _collect_runtime_stats(cams)
                for line in _format_runtime_stats_lines(
                    elapsed_s=float(now_s - loop_start_s),
                    runtime_stats=runtime_stats,
                ):
                    print(line, flush=True)
                next_stats_log_s = now_s + float(args.stats_log_interval_s)
            if float(args.duration_s) > 0 and (now_s - loop_start_s) >= float(args.duration_s):
                print(
                    f"Reached --duration-s={float(args.duration_s):.1f}. Stopping viewer.",
                    flush=True,
                )
                break
            if key in (ord("q"), 27):
                break
    finally:
        stop_event.set()
        for cam_state in cams:
            try:
                cam_state["pipeline"].stop()
            except Exception:
                pass
        for cam_state in cams:
            try:
                cam_state["capture_thread"].join(timeout=2.0)
            except Exception:
                pass
        for cam_state in cams:
            _put_latest(cam_state["request_queue"], None)
        for proc in worker_processes:
            proc.join(timeout=5.0)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2.0)
        for cam_state in cams:
            try:
                cam_state["result_thread"].join(timeout=2.0)
            except Exception:
                pass
            try:
                cam_state["request_queue"].close()
                cam_state["result_queue"].close()
            except Exception:
                pass
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
