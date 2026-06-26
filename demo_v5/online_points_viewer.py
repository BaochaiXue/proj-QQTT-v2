#!/usr/bin/env python3
"""Online 2D overlay viewer for Demo v5 object/controller point chunks."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import pickle
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np


DEFAULT_WINDOW_NAME = "Demo v5 online points"
DEFAULT_IMAGE_SIZE = (1280, 720)


@dataclass(frozen=True)
class CameraModel:
    intrinsic: np.ndarray
    camera_to_world: np.ndarray
    image_size: tuple[int, int]
    metadata_fps: float | None


def _require_cv2() -> Any:
    import cv2

    return cv2


def _install_numpy_pickle_aliases() -> None:
    """Allow pickles written by NumPy 2.x to load in older NumPy runtimes."""
    try:
        import numpy.core as numpy_core
    except Exception:
        return
    sys.modules.setdefault("numpy._core", numpy_core)
    for name in ("numeric", "multiarray", "umath", "_multiarray_umath"):
        try:
            module = __import__(f"numpy.core.{name}", fromlist=[name])
        except Exception:
            continue
        sys.modules.setdefault(f"numpy._core.{name}", module)


def load_pickle(path: str | Path) -> Any:
    _install_numpy_pickle_aliases()
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def read_json(path: str | Path) -> dict[str, Any]:
    try:
        text = Path(path).read_text(encoding="utf-8")
        return dict(json.loads(text))
    except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError):
        return {}


def normalize_online_dir(path: str | Path) -> Path:
    value = Path(path).expanduser()
    if value.name == "chunks":
        return value.parent
    return value


def infer_case_dir(online_dir: Path, case_dir: str | Path | None) -> Path:
    if case_dir is not None:
        return Path(case_dir).expanduser()
    if online_dir.parent.name == "online_data":
        return online_dir.parent.parent / "data" / online_dir.name
    return online_dir.parent / "data" / online_dir.name


def _select_camera_array(value: Any, *, cam_idx: int, shape: tuple[int, int]) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape == shape:
        return arr
    if arr.ndim == 3 and arr.shape[1:] == shape and int(arr.shape[0]) > 0:
        idx = min(max(int(cam_idx), 0), int(arr.shape[0]) - 1)
        return np.asarray(arr[idx], dtype=np.float64)
    return None


def _metadata_image_size(metadata: Mapping[str, Any]) -> tuple[int, int]:
    value = metadata.get("WH")
    if value is not None:
        arr = np.asarray(value).reshape(-1)
        if arr.size >= 2:
            width = int(arr[0])
            height = int(arr[1])
            if width > 0 and height > 0:
                return (width, height)
    return DEFAULT_IMAGE_SIZE


def load_camera_model(case_dir: str | Path, *, cam_idx: int) -> CameraModel:
    """Load intrinsics and camera-to-world calibration from the aggregate case."""
    case_path = Path(case_dir)
    metadata = read_json(case_path / "metadata.json")
    image_size = _metadata_image_size(metadata)
    intrinsic = _select_camera_array(metadata.get("intrinsics"), cam_idx=cam_idx, shape=(3, 3))
    if intrinsic is None:
        intrinsic = np.eye(3, dtype=np.float64)
    calibrate_path = case_path / "calibrate.pkl"
    camera_to_world = None
    if calibrate_path.is_file():
        camera_to_world = _select_camera_array(load_pickle(calibrate_path), cam_idx=cam_idx, shape=(4, 4))
    if camera_to_world is None:
        camera_to_world = np.eye(4, dtype=np.float64)
    fps_value = metadata.get("fps")
    try:
        metadata_fps = float(fps_value) if fps_value is not None else None
    except (TypeError, ValueError):
        metadata_fps = None
    if metadata_fps is not None and (not math.isfinite(metadata_fps) or metadata_fps <= 0.0):
        metadata_fps = None
    return CameraModel(
        intrinsic=np.asarray(intrinsic, dtype=np.float64),
        camera_to_world=np.asarray(camera_to_world, dtype=np.float64),
        image_size=image_size,
        metadata_fps=metadata_fps,
    )


def project_world_points_to_pixels(
    points: np.ndarray,
    *,
    intrinsic: np.ndarray,
    camera_to_world: np.ndarray,
    image_size: tuple[int, int],
    visibility: np.ndarray | None = None,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Project world-space Demo v5 points back onto the selected RGB camera."""
    arr = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if arr.size == 0:
        return np.empty((0, 2), dtype=np.int32), np.empty((0,), dtype=np.int64)
    indices = np.arange(arr.shape[0], dtype=np.int64)
    if visibility is not None:
        vis = np.asarray(visibility, dtype=bool).reshape(-1)
        if vis.shape[0] == arr.shape[0]:
            indices = indices[vis]
            arr = arr[vis]
    step = max(int(stride), 1)
    if step > 1:
        indices = indices[::step]
        arr = arr[::step]
    if arr.size == 0:
        return np.empty((0, 2), dtype=np.int32), np.empty((0,), dtype=np.int64)
    world_to_camera = np.linalg.inv(np.asarray(camera_to_world, dtype=np.float64))
    homogeneous = np.concatenate([arr, np.ones((arr.shape[0], 1), dtype=np.float64)], axis=1)
    camera_points = (world_to_camera @ homogeneous.T).T[:, :3]
    finite = np.all(np.isfinite(camera_points), axis=1)
    positive_depth = camera_points[:, 2] > 1e-6
    valid = finite & positive_depth
    if not np.any(valid):
        return np.empty((0, 2), dtype=np.int32), np.empty((0,), dtype=np.int64)
    camera_points = camera_points[valid]
    indices = indices[valid]
    projected = (np.asarray(intrinsic, dtype=np.float64) @ camera_points.T).T
    xy = projected[:, :2] / projected[:, 2:3]
    pixels = np.rint(xy).astype(np.int32)
    width, height = int(image_size[0]), int(image_size[1])
    in_bounds = (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    return pixels[in_bounds], indices[in_bounds]


def _blank_image(image_size: tuple[int, int]) -> np.ndarray:
    width, height = int(image_size[0]), int(image_size[1])
    return np.zeros((height, width, 3), dtype=np.uint8)


def _frame_path_candidates(case_dir: Path, *, cam_idx: int, source_frame: int) -> list[Path]:
    color_dir = case_dir / "color" / str(int(cam_idx))
    return [
        color_dir / f"{int(source_frame)}.png",
        color_dir / f"{int(source_frame):06d}.png",
        color_dir / f"{int(source_frame)}.jpg",
        color_dir / f"{int(source_frame):06d}.jpg",
    ]


def read_background(
    case_dir: Path,
    *,
    cam_idx: int,
    source_frame: int,
    image_size: tuple[int, int],
    use_background: bool,
) -> np.ndarray:
    """Return the source RGB frame when present, otherwise a black canvas."""
    if not use_background:
        return _blank_image(image_size)
    cv2 = _require_cv2()
    for path in _frame_path_candidates(case_dir, cam_idx=cam_idx, source_frame=source_frame):
        if not path.is_file():
            continue
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is not None:
            return image
    return _blank_image(image_size)


def _chunk_frame_count(chunk: Mapping[str, Any]) -> int:
    for key in ("object_points", "controller_points"):
        value = chunk.get(key)
        if value is not None:
            return int(np.asarray(value).shape[0])
    return 0


def _source_frame_for_chunk_frame(chunk: Mapping[str, Any], local_frame: int) -> int:
    source_indices = chunk.get("source_frame_indices")
    if source_indices is not None:
        try:
            return int(source_indices[int(local_frame)])
        except (IndexError, TypeError, ValueError):
            pass
    return int(chunk.get("start_frame", 0)) + int(local_frame)


def parse_bgr_color(text: str) -> tuple[int, int, int]:
    parts = [part.strip() for part in str(text).split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("color must be B,G,R")
    try:
        values = [int(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("color must be B,G,R integers") from exc
    return tuple(max(0, min(255, value)) for value in values)


def _rainbow_colors(point_indices: np.ndarray) -> np.ndarray:
    if point_indices.size == 0:
        return np.empty((0, 3), dtype=np.uint8)
    cv2 = _require_cv2()
    values = np.asarray(point_indices % 256, dtype=np.uint8).reshape(-1, 1)
    return cv2.applyColorMap(values, cv2.COLORMAP_TURBO).reshape(-1, 3)


def object_point_colors(
    chunk: Mapping[str, Any],
    *,
    local_frame: int,
    point_indices: np.ndarray,
    mode: str,
) -> np.ndarray:
    mode_value = str(mode)
    if mode_value == "green":
        return np.tile(np.array([[0, 255, 0]], dtype=np.uint8), (point_indices.shape[0], 1))
    if mode_value == "object-colors":
        colors = chunk.get("object_colors")
        if colors is not None:
            arr = np.asarray(colors)
            if arr.ndim == 3 and local_frame < arr.shape[0] and arr.shape[2] >= 3:
                selected = np.asarray(arr[int(local_frame), point_indices, :3], dtype=np.float64)
                if selected.size:
                    if float(np.nanmax(selected)) <= 1.0:
                        selected = selected * 255.0
                    selected = np.clip(selected, 0.0, 255.0).astype(np.uint8)
                    return selected[:, ::-1]
    return _rainbow_colors(point_indices)


def _draw_points(image: np.ndarray, pixels: np.ndarray, colors: np.ndarray, *, radius: int) -> None:
    if pixels.size == 0:
        return
    cv2 = _require_cv2()
    draw_radius = max(int(radius), 1)
    for (x_value, y_value), color in zip(pixels, colors, strict=False):
        cv2.circle(
            image,
            (int(x_value), int(y_value)),
            draw_radius,
            tuple(int(value) for value in color),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )


def _draw_status(image: np.ndarray, text: str) -> None:
    cv2 = _require_cv2()
    origin = (12, 28)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)


def render_chunk_frame(
    chunk: Mapping[str, Any],
    *,
    local_frame: int,
    case_dir: Path,
    camera: CameraModel,
    cam_idx: int,
    use_background: bool,
    show_invisible_object_points: bool,
    object_stride: int,
    object_radius: int,
    controller_radius: int,
    object_color_mode: str,
    controller_color: tuple[int, int, int],
    fps: float,
) -> np.ndarray:
    """Draw one online chunk frame as colored object/controller pixels."""
    source_frame = _source_frame_for_chunk_frame(chunk, local_frame)
    image = read_background(
        case_dir,
        cam_idx=cam_idx,
        source_frame=source_frame,
        image_size=camera.image_size,
        use_background=use_background,
    )
    image_size = (int(image.shape[1]), int(image.shape[0]))
    object_points = chunk.get("object_points")
    if object_points is not None:
        object_arr = np.asarray(object_points)
        if object_arr.ndim == 3 and int(local_frame) < int(object_arr.shape[0]):
            visibility = None
            if not show_invisible_object_points and chunk.get("object_visibilities") is not None:
                vis_arr = np.asarray(chunk["object_visibilities"])
                if vis_arr.ndim == 2 and int(local_frame) < int(vis_arr.shape[0]):
                    visibility = vis_arr[int(local_frame)]
            object_pixels, object_indices = project_world_points_to_pixels(
                object_arr[int(local_frame)],
                intrinsic=camera.intrinsic,
                camera_to_world=camera.camera_to_world,
                image_size=image_size,
                visibility=visibility,
                stride=object_stride,
            )
            _draw_points(
                image,
                object_pixels,
                object_point_colors(
                    chunk,
                    local_frame=int(local_frame),
                    point_indices=object_indices,
                    mode=object_color_mode,
                ),
                radius=object_radius,
            )
    controller_points = chunk.get("controller_points")
    if controller_points is not None:
        controller_arr = np.asarray(controller_points)
        if controller_arr.ndim == 3 and int(local_frame) < int(controller_arr.shape[0]):
            controller_pixels, controller_indices = project_world_points_to_pixels(
                controller_arr[int(local_frame)],
                intrinsic=camera.intrinsic,
                camera_to_world=camera.camera_to_world,
                image_size=image_size,
                stride=1,
            )
            controller_colors = np.tile(
                np.asarray(controller_color, dtype=np.uint8).reshape(1, 3),
                (controller_indices.shape[0], 1),
            )
            _draw_points(image, controller_pixels, controller_colors, radius=controller_radius)
    frame_count = _chunk_frame_count(chunk)
    chunk_id = int(chunk.get("chunk_id", -1))
    _draw_status(
        image,
        f"chunk {chunk_id:06d}  frame {int(local_frame) + 1}/{frame_count}  source {source_frame}  {fps:g} FPS",
    )
    return image


def _window_is_open(window_name: str) -> bool:
    cv2 = _require_cv2()
    try:
        return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1
    except Exception:
        return True


def _key_requests_quit(key: int) -> bool:
    return key in (27, ord("q"), ord("Q"))


def _wait_with_pause(window_name: str, *, delay_s: float) -> bool:
    cv2 = _require_cv2()
    deadline = time.monotonic() + max(float(delay_s), 0.0)
    paused = False
    while True:
        if not _window_is_open(window_name):
            return False
        wait_s = 0.05 if paused else min(0.05, max(0.0, deadline - time.monotonic()))
        key = cv2.waitKey(max(1, int(wait_s * 1000))) & 0xFF
        if _key_requests_quit(key):
            return False
        if key == ord(" "):
            paused = not paused
            if not paused:
                deadline = time.monotonic()
        if paused:
            continue
        if time.monotonic() >= deadline:
            return True


def play_chunk(
    chunk: Mapping[str, Any],
    *,
    case_dir: Path,
    camera: CameraModel,
    args: argparse.Namespace,
    fps: float,
) -> np.ndarray | None:
    cv2 = _require_cv2()
    period_s = 1.0 / max(float(fps), 1e-6)
    frame_count = _chunk_frame_count(chunk)
    last_image = None
    for local_frame in range(frame_count):
        image = render_chunk_frame(
            chunk,
            local_frame=local_frame,
            case_dir=case_dir,
            camera=camera,
            cam_idx=int(args.cam_idx),
            use_background=not bool(args.no_background),
            show_invisible_object_points=bool(args.show_invisible_object_points),
            object_stride=int(args.object_stride),
            object_radius=int(args.object_radius),
            controller_radius=int(args.controller_radius),
            object_color_mode=str(args.object_color_mode),
            controller_color=args.controller_color,
            fps=fps,
        )
        cv2.imshow(str(args.window_name), image)
        last_image = image
        if not _wait_with_pause(str(args.window_name), delay_s=period_s):
            return None
    return last_image


def wait_for_chunk(
    online_dir: Path,
    *,
    chunk_id: int,
    poll_sec: float,
    window_name: str,
    last_image: np.ndarray | None,
) -> dict[str, Any] | None:
    """Block until the next online chunk appears or the stream finishes."""
    cv2 = _require_cv2()
    chunks_dir = online_dir / "chunks"
    chunk_path = chunks_dir / f"chunk_{int(chunk_id):06d}.pkl"
    while True:
        if chunk_path.is_file():
            return dict(load_pickle(chunk_path))
        manifest = read_json(online_dir / "manifest.json")
        latest = int(manifest.get("latest_committed_chunk", -1))
        status = str(manifest.get("status", "recording"))
        if status == "finished" and latest < int(chunk_id):
            return None
        if last_image is not None:
            waiting = last_image.copy()
            _draw_status(waiting, f"waiting for chunk {int(chunk_id):06d}")
            cv2.imshow(window_name, waiting)
        key = cv2.waitKey(max(1, int(float(poll_sec) * 1000))) & 0xFF
        if _key_requests_quit(key) or not _window_is_open(window_name):
            return None


def resolve_playback_fps(args: argparse.Namespace, camera: CameraModel) -> float:
    fps = None if args.fps is None else float(args.fps)
    if fps is None:
        fps = camera.metadata_fps
    if fps is None:
        fps = 5.0
    if not math.isfinite(float(fps)) or float(fps) <= 0.0:
        raise ValueError("--fps must be positive")
    return float(fps)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Play Demo v5 online object/controller points chunk by chunk.")
    parser.add_argument("--online-dir", type=Path, required=True, help="Path to online_data/<case> or its chunks directory.")
    parser.add_argument("--case-dir", type=Path, default=None, help="Path to data/<case>. Inferred from --online-dir when omitted.")
    parser.add_argument("--cam-idx", type=int, default=0)
    parser.add_argument("--fps", type=float, default=None, help="Playback FPS. Defaults to metadata fps, then 5.")
    parser.add_argument("--poll-sec", type=float, default=0.1)
    parser.add_argument("--start-chunk", type=int, default=0)
    parser.add_argument("--object-stride", type=int, default=1)
    parser.add_argument("--object-radius", type=int, default=4)
    parser.add_argument("--controller-radius", type=int, default=7)
    parser.add_argument("--object-color-mode", choices=("rainbow", "green", "object-colors"), default="rainbow")
    parser.add_argument("--controller-color", type=parse_bgr_color, default=parse_bgr_color("0,0,255"))
    parser.add_argument("--show-invisible-object-points", action="store_true")
    parser.add_argument("--no-background", action="store_true")
    parser.add_argument("--window-name", default=DEFAULT_WINDOW_NAME)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if int(args.cam_idx) < 0:
        raise ValueError("--cam-idx must be non-negative")
    if float(args.poll_sec) <= 0.0:
        raise ValueError("--poll-sec must be positive")
    if int(args.start_chunk) < 0:
        raise ValueError("--start-chunk must be non-negative")
    if int(args.object_stride) <= 0:
        raise ValueError("--object-stride must be positive")
    if int(args.object_radius) <= 0:
        raise ValueError("--object-radius must be positive")
    if int(args.controller_radius) <= 0:
        raise ValueError("--controller-radius must be positive")


def run(args: argparse.Namespace) -> int:
    """Play committed chunks in order, tailing the online directory live."""
    validate_args(args)
    cv2 = _require_cv2()
    online_dir = normalize_online_dir(args.online_dir)
    case_dir = infer_case_dir(online_dir, args.case_dir)
    camera = load_camera_model(case_dir, cam_idx=int(args.cam_idx))
    fps = resolve_playback_fps(args, camera)
    cv2.namedWindow(str(args.window_name), cv2.WINDOW_NORMAL)
    chunk_id = int(args.start_chunk)
    last_image: np.ndarray | None = None
    while True:
        chunk = wait_for_chunk(
            online_dir,
            chunk_id=chunk_id,
            poll_sec=float(args.poll_sec),
            window_name=str(args.window_name),
            last_image=last_image,
        )
        if chunk is None:
            return 0
        last_image = play_chunk(
            chunk,
            case_dir=case_dir,
            camera=camera,
            args=args,
            fps=fps,
        )
        if last_image is None:
            return 0
        chunk_id += 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
