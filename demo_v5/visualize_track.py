#!/usr/bin/env python3
"""Track visualization for Demo v5 object/controller point chunks."""
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


DEFAULT_WINDOW_NAME = "Demo v5 visualize track"
DEFAULT_IMAGE_SIZE = (1280, 720)
DEFAULT_OBJECT_RADIUS = 3
DEFAULT_CONTROLLER_RADIUS = 6
RENDER_MODE_RGB_OVERLAY = "rgb-overlay"
RENDER_MODE_SAM3D_FINAL_DATA = "sam3d-final-data"
RENDER_MODES = (RENDER_MODE_RGB_OVERLAY, RENDER_MODE_SAM3D_FINAL_DATA)


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


def _sam3d_rainbow_colors_bgr(chunk: Mapping[str, Any], point_indices: np.ndarray) -> np.ndarray:
    indices = np.asarray(point_indices, dtype=np.int64).reshape(-1)
    if indices.size == 0:
        return np.empty((0, 3), dtype=np.uint8)
    object_points = chunk.get("object_points")
    if object_points is None:
        normalized = np.zeros((indices.shape[0],), dtype=np.float32)
    else:
        arr = np.asarray(object_points, dtype=np.float64)
        if arr.ndim != 3 or arr.shape[0] == 0 or arr.shape[2] < 2:
            normalized = np.zeros((indices.shape[0],), dtype=np.float32)
        else:
            y_values = np.asarray(arr[0, :, 1], dtype=np.float64).reshape(-1)
            if y_values.size == 0:
                normalized = np.zeros((indices.shape[0],), dtype=np.float32)
            else:
                finite = np.isfinite(y_values)
                if np.any(finite):
                    y_min = float(np.nanmin(y_values[finite]))
                    y_max = float(np.nanmax(y_values[finite]))
                    span = y_max - y_min
                    if math.isfinite(span) and span > 1e-9:
                        selected_y = y_values[np.clip(indices, 0, y_values.shape[0] - 1)]
                        normalized = np.clip((selected_y - y_min) / span, 0.0, 1.0).astype(np.float32)
                    else:
                        normalized = np.zeros((indices.shape[0],), dtype=np.float32)
                else:
                    normalized = np.zeros((indices.shape[0],), dtype=np.float32)
    try:
        import matplotlib.pyplot as plt

        rgb = np.asarray(plt.cm.rainbow(normalized)[:, :3], dtype=np.float32) * 255.0
    except Exception:
        rgb = np.stack(
            [
                255.0 * normalized,
                255.0 * (1.0 - np.abs(normalized - 0.5) * 2.0),
                255.0 * (1.0 - normalized),
            ],
            axis=1,
        )
    return np.ascontiguousarray(np.clip(rgb, 0, 255).astype(np.uint8)[:, ::-1], dtype=np.uint8)


def _sam3d_rainbow_colors_rgb_float(object_points: np.ndarray, point_count: int) -> np.ndarray:
    count = max(0, int(point_count))
    if count == 0:
        return np.empty((0, 3), dtype=np.float64)
    arr = np.asarray(object_points, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[0] == 0 or arr.shape[1] < count or arr.shape[2] < 2:
        normalized = np.zeros((count,), dtype=np.float64)
    else:
        y_values = np.asarray(arr[0, :count, 1], dtype=np.float64)
        finite = np.isfinite(y_values)
        if np.any(finite):
            y_min = float(np.nanmin(y_values[finite]))
            y_max = float(np.nanmax(y_values[finite]))
            span = y_max - y_min
            if math.isfinite(span) and span > 1e-9:
                normalized = np.clip((y_values - y_min) / span, 0.0, 1.0)
            else:
                normalized = np.zeros((count,), dtype=np.float64)
        else:
            normalized = np.zeros((count,), dtype=np.float64)
    try:
        import matplotlib.pyplot as plt

        return np.ascontiguousarray(np.asarray(plt.cm.rainbow(normalized)[:, :3], dtype=np.float64))
    except Exception:
        rgb = np.stack(
            [
                normalized,
                1.0 - np.abs(normalized - 0.5) * 2.0,
                1.0 - normalized,
            ],
            axis=1,
        )
        return np.ascontiguousarray(np.clip(rgb, 0.0, 1.0), dtype=np.float64)


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
    return _sam3d_rainbow_colors_bgr(chunk, point_indices)


def controller_point_colors(
    chunk: Mapping[str, Any],
    *,
    local_frame: int,
    point_indices: np.ndarray,
    fallback_color: tuple[int, int, int],
) -> np.ndarray:
    color = np.asarray(fallback_color, dtype=np.uint8).reshape(1, 3)
    return np.tile(color, (point_indices.shape[0], 1))


def _draw_sam3d_markers(
    image: np.ndarray,
    pixels: np.ndarray,
    colors: np.ndarray,
    *,
    radius: int,
) -> None:
    if pixels.size == 0:
        return
    cv2 = _require_cv2()
    draw_radius = max(int(radius), 1)
    for (x_value, y_value), color in zip(pixels, colors, strict=False):
        center = (int(x_value), int(y_value))
        color_bgr = tuple(int(value) for value in color)
        cv2.circle(
            image,
            center,
            draw_radius,
            color_bgr,
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
            _draw_sam3d_markers(
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
            _draw_sam3d_markers(
                image,
                controller_pixels,
                controller_point_colors(
                    chunk,
                    local_frame=int(local_frame),
                    point_indices=controller_indices,
                    fallback_color=controller_color,
                ),
                radius=controller_radius,
            )
    frame_count = _chunk_frame_count(chunk)
    chunk_id = int(chunk.get("chunk_id", -1))
    _draw_status(
        image,
        f"chunk {chunk_id:06d}  frame {int(local_frame) + 1}/{frame_count}  source {source_frame}  {fps:g} FPS",
    )
    return image


class RgbOverlayRenderer:
    def __init__(self, *, camera: CameraModel, args: argparse.Namespace, fps: float) -> None:
        self._camera = camera
        self._args = args
        self._fps = float(fps)

    def render_frame(self, chunk: Mapping[str, Any], *, local_frame: int, case_dir: Path) -> np.ndarray:
        return render_chunk_frame(
            chunk,
            local_frame=int(local_frame),
            case_dir=case_dir,
            camera=self._camera,
            cam_idx=int(self._args.cam_idx),
            use_background=not bool(self._args.no_background),
            show_invisible_object_points=bool(self._args.show_invisible_object_points),
            object_stride=int(self._args.object_stride),
            object_radius=int(self._args.object_radius),
            controller_radius=int(self._args.controller_radius),
            object_color_mode=str(self._args.object_color_mode),
            controller_color=self._args.controller_color,
            fps=self._fps,
        )

    def close(self) -> None:
        return None


class Sam3DFinalDataRenderer:
    def __init__(self, *, image_size: tuple[int, int], show_invisible_object_points: bool) -> None:
        self._image_size = (int(image_size[0]), int(image_size[1]))
        self._show_invisible_object_points = bool(show_invisible_object_points)
        self._o3d: Any | None = None
        self._vis: Any | None = None
        self._object_pcd: Any | None = None
        self._controller_meshes: list[Any] = []
        self._controller_centers: list[np.ndarray] = []
        self._object_colors: np.ndarray | None = None
        self._object_color_count = -1
        self._initialized = False

    def _require_open3d(self) -> Any:
        if self._o3d is None:
            import open3d as o3d

            self._o3d = o3d
        return self._o3d

    def _ensure_window(self) -> None:
        if self._vis is not None:
            return
        o3d = self._require_open3d()
        self._vis = o3d.visualization.Visualizer()
        width, height = self._image_size
        self._vis.create_window(width=width, height=height, visible=False)
        self._object_pcd = o3d.geometry.PointCloud()

    def _object_visibility(self, chunk: Mapping[str, Any], local_frame: int, point_count: int) -> np.ndarray:
        if self._show_invisible_object_points:
            return np.ones((point_count,), dtype=bool)
        value = chunk.get("object_visibilities")
        if value is None:
            return np.ones((point_count,), dtype=bool)
        arr = np.asarray(value, dtype=bool)
        if arr.ndim == 2 and int(local_frame) < int(arr.shape[0]) and arr.shape[1] == point_count:
            return np.ascontiguousarray(arr[int(local_frame)], dtype=bool)
        return np.ones((point_count,), dtype=bool)

    def _update_object_colors(self, object_points: np.ndarray) -> np.ndarray:
        point_count = int(object_points.shape[1])
        if self._object_colors is None or self._object_color_count != point_count:
            self._object_colors = _sam3d_rainbow_colors_rgb_float(object_points, point_count)
            self._object_color_count = point_count
        return self._object_colors

    def _reset_controller_meshes(self, controller_points: np.ndarray) -> None:
        assert self._vis is not None
        o3d = self._require_open3d()
        for mesh in self._controller_meshes:
            self._vis.remove_geometry(mesh, reset_bounding_box=False)
        self._controller_meshes = []
        self._controller_centers = []
        for origin in np.asarray(controller_points, dtype=np.float64).reshape(-1, 3):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01).translate(origin)
            sphere.paint_uniform_color([1.0, 0.0, 0.0])
            self._controller_meshes.append(sphere)
            self._controller_centers.append(np.asarray(origin, dtype=np.float64))
            self._vis.add_geometry(sphere, reset_bounding_box=False)

    def _set_initial_view(self) -> None:
        assert self._vis is not None
        view_control = self._vis.get_view_control()
        view_control.set_front([1, 0, -2])
        view_control.set_up([0, 0, -1])
        view_control.set_zoom(1)

    def render_frame(self, chunk: Mapping[str, Any], *, local_frame: int, case_dir: Path) -> np.ndarray:
        del case_dir
        self._ensure_window()
        assert self._vis is not None
        assert self._object_pcd is not None
        o3d = self._require_open3d()

        object_arr = np.asarray(chunk.get("object_points"), dtype=np.float64)
        controller_arr = np.asarray(chunk.get("controller_points"), dtype=np.float64)
        if object_arr.ndim != 3 or controller_arr.ndim != 3:
            return _blank_image(self._image_size)
        if int(local_frame) >= int(object_arr.shape[0]) or int(local_frame) >= int(controller_arr.shape[0]):
            return _blank_image(self._image_size)

        object_frame = np.asarray(object_arr[int(local_frame)], dtype=np.float64).reshape(-1, 3)
        object_colors = self._update_object_colors(object_arr)
        visible = self._object_visibility(chunk, int(local_frame), int(object_frame.shape[0]))
        object_valid = visible & np.all(np.isfinite(object_frame), axis=1)
        controller_frame = np.asarray(controller_arr[int(local_frame)], dtype=np.float64).reshape(-1, 3)
        controller_valid = np.all(np.isfinite(controller_frame), axis=1)
        controller_points = controller_frame[controller_valid]

        if not self._initialized:
            self._object_pcd.points = o3d.utility.Vector3dVector(object_frame[object_valid])
            self._object_pcd.colors = o3d.utility.Vector3dVector(object_colors[object_valid])
            self._vis.add_geometry(self._object_pcd)
            self._reset_controller_meshes(controller_points)
            self._set_initial_view()
            self._initialized = True
        else:
            self._object_pcd.points = o3d.utility.Vector3dVector(object_frame[object_valid])
            self._object_pcd.colors = o3d.utility.Vector3dVector(object_colors[object_valid])
            self._vis.update_geometry(self._object_pcd)
            if len(controller_points) != len(self._controller_meshes):
                self._reset_controller_meshes(controller_points)
            for index, sphere in enumerate(self._controller_meshes):
                origin = np.asarray(controller_points[index], dtype=np.float64)
                sphere.translate(origin - self._controller_centers[index])
                self._controller_centers[index] = origin
                self._vis.update_geometry(sphere)

        self._vis.poll_events()
        self._vis.update_renderer()
        frame = np.asarray(self._vis.capture_screen_float_buffer(do_render=True))
        frame = np.clip(frame * 255.0, 0.0, 255.0).astype(np.uint8)
        cv2 = _require_cv2()
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def close(self) -> None:
        if self._vis is not None:
            self._vis.destroy_window()
            self._vis = None


def build_frame_renderer(args: argparse.Namespace, *, camera: CameraModel, fps: float) -> Any:
    render_mode = str(args.render_mode)
    if render_mode == RENDER_MODE_RGB_OVERLAY:
        return RgbOverlayRenderer(camera=camera, args=args, fps=fps)
    if render_mode == RENDER_MODE_SAM3D_FINAL_DATA:
        return Sam3DFinalDataRenderer(
            image_size=camera.image_size,
            show_invisible_object_points=bool(args.show_invisible_object_points),
        )
    raise ValueError(f"unsupported render mode: {render_mode!r}")


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
    renderer: Any,
    args: argparse.Namespace,
    fps: float,
) -> np.ndarray | None:
    cv2 = _require_cv2()
    period_s = 1.0 / max(float(fps), 1e-6)
    frame_count = _chunk_frame_count(chunk)
    last_image = None
    for local_frame in range(frame_count):
        image = renderer.render_frame(
            chunk,
            local_frame=local_frame,
            case_dir=case_dir,
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


def _chunk_sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    try:
        return (int(stem.rsplit("_", 1)[1]), path.name)
    except (IndexError, ValueError):
        return (0, path.name)


def list_available_chunk_paths(online_dir: Path, *, start_chunk: int) -> list[Path]:
    chunks_dir = normalize_online_dir(online_dir) / "chunks"
    paths = sorted(chunks_dir.glob("chunk_*.pkl"), key=_chunk_sort_key)
    start = int(start_chunk)
    return [path for path in paths if _chunk_sort_key(path)[0] >= start]


def render_output_video(args: argparse.Namespace) -> int:
    cv2 = _require_cv2()
    online_dir = normalize_online_dir(args.online_dir)
    case_dir = infer_case_dir(online_dir, args.case_dir)
    camera = load_camera_model(case_dir, cam_idx=int(args.cam_idx))
    fps = resolve_playback_fps(args, camera)
    chunk_paths = list_available_chunk_paths(online_dir, start_chunk=int(args.start_chunk))
    if not chunk_paths:
        raise ValueError(f"no chunk_*.pkl files found under {online_dir / 'chunks'}")
    output_path = Path(args.output_video).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    renderer = build_frame_renderer(args, camera=camera, fps=fps)
    writer = None
    try:
        for chunk_path in chunk_paths:
            chunk = dict(load_pickle(chunk_path))
            frame_count = _chunk_frame_count(chunk)
            for local_frame in range(frame_count):
                image = renderer.render_frame(chunk, local_frame=local_frame, case_dir=case_dir)
                if writer is None:
                    height, width = image.shape[:2]
                    writer = cv2.VideoWriter(
                        str(output_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (int(width), int(height)),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"failed to open VideoWriter for {output_path}")
                writer.write(image)
    finally:
        if writer is not None:
            writer.release()
        renderer.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Play Demo v5 online object/controller points chunk by chunk.")
    parser.add_argument("--online-dir", type=Path, required=True, help="Path to online_data/<case> or its chunks directory.")
    parser.add_argument("--case-dir", type=Path, default=None, help="Path to data/<case>. Inferred from --online-dir when omitted.")
    parser.add_argument("--render-mode", choices=RENDER_MODES, default=RENDER_MODE_RGB_OVERLAY)
    parser.add_argument("--output-video", type=Path, default=None, help="Write existing chunks to MP4 and exit instead of opening a live window.")
    parser.add_argument("--cam-idx", type=int, default=0)
    parser.add_argument("--fps", type=float, default=None, help="Playback FPS. Defaults to metadata fps, then 5.")
    parser.add_argument("--poll-sec", type=float, default=0.1)
    parser.add_argument("--start-chunk", type=int, default=0)
    parser.add_argument("--object-stride", type=int, default=1)
    parser.add_argument("--object-radius", type=int, default=DEFAULT_OBJECT_RADIUS)
    parser.add_argument("--controller-radius", type=int, default=DEFAULT_CONTROLLER_RADIUS)
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
    if args.output_video is not None:
        return render_output_video(args)
    cv2 = _require_cv2()
    online_dir = normalize_online_dir(args.online_dir)
    case_dir = infer_case_dir(online_dir, args.case_dir)
    camera = load_camera_model(case_dir, cam_idx=int(args.cam_idx))
    fps = resolve_playback_fps(args, camera)
    renderer = build_frame_renderer(args, camera=camera, fps=fps)
    cv2.namedWindow(str(args.window_name), cv2.WINDOW_NORMAL)
    chunk_id = int(args.start_chunk)
    last_image: np.ndarray | None = None
    try:
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
                renderer=renderer,
                args=args,
                fps=fps,
            )
            if last_image is None:
                return 0
            chunk_id += 1
    finally:
        renderer.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
