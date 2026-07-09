"""Frame renderers (RGB overlay and Open3D final_data) for the viewer.

Extracted verbatim from ``visualize_track.py`` as part of a behavior-preserving
file split. Depends on ``viz_camera_model``, ``viz_input_timeline``, and
``viz_panels``.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from demo_v6_2.viz_camera_model import CameraModel, _require_cv2
from demo_v6_2.viz_input_timeline import (
    _resolve_capture_dir,
    _resolve_input_rgb_timeline,
    format_input_display_latency,
    load_input_rgb_background_paths,
)
from demo_v6_2.viz_panels import (
    _blank_image,
    _sam3d_rainbow_colors_rgb_float,
    render_chunk_frame,
)


RENDER_MODE_RGB_OVERLAY = "rgb-overlay"
RENDER_MODE_SAM3D_FINAL_DATA = "sam3d-final-data"
RENDER_MODES = (RENDER_MODE_RGB_OVERLAY, RENDER_MODE_SAM3D_FINAL_DATA)


class RgbOverlayRenderer:
    """Render object/controller tracks as overlays on RGB background frames."""

    def __init__(self, *, camera: CameraModel, args: argparse.Namespace, fps: float) -> None:
        """Initialize RgbOverlayRenderer."""
        self._camera = camera
        self._args = args
        self._fps = float(fps)
        self._background_frame_paths = self._load_background_frame_paths(args)

    def _load_background_frame_paths(self, args: argparse.Namespace) -> dict[int, Path]:
        """Load background frame paths."""
        capture_dir = _resolve_capture_dir(args)
        input_timeline = _resolve_input_rgb_timeline(args, capture_dir=capture_dir)
        if capture_dir is None or input_timeline is None:
            return {}
        return load_input_rgb_background_paths(input_timeline, capture_dir=capture_dir)

    def render_frame(self, chunk: Mapping[str, Any], *, local_frame: int, case_dir: Path) -> np.ndarray:
        """Render one chunk frame as an RGB-overlay image."""
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
            background_frame_paths=self._background_frame_paths,
        )

    def close(self) -> None:
        """Release renderer resources."""
        return None


class Sam3DFinalDataRenderer:
    """Render final_data object/controller points through an Open3D visualizer."""

    def __init__(
        self,
        *,
        image_size: tuple[int, int],
        show_invisible_object_points: bool,
        visible: bool = False,
        window_name: str = "final_data output",
        window_position: tuple[int, int] | None = None,
    ) -> None:
        """Initialize Sam3DFinalDataRenderer."""
        self._image_size = (int(image_size[0]), int(image_size[1]))
        self._show_invisible_object_points = bool(show_invisible_object_points)
        self._visible = bool(visible)
        self._window_name = str(window_name)
        self._window_position = window_position
        self._o3d: Any | None = None
        self._vis: Any | None = None
        self._object_pcd: Any | None = None
        self._controller_meshes: list[Any] = []
        self._controller_centers: list[np.ndarray] = []
        self._object_colors: np.ndarray | None = None
        self._object_color_count = -1
        self._initialized = False

    def _require_open3d(self) -> Any:
        """Return validated open3d."""
        if self._o3d is None:
            import open3d as o3d

            self._o3d = o3d
        return self._o3d

    def _ensure_window(self) -> None:
        """Return the ensure window."""
        if self._vis is not None:
            return
        o3d = self._require_open3d()
        self._vis = o3d.visualization.Visualizer()
        width, height = self._image_size
        left = 50
        top = 50
        if self._window_position is not None:
            left, top = (int(self._window_position[0]), int(self._window_position[1]))
        self._vis.create_window(
            window_name=self._window_name,
            width=width,
            height=height,
            left=left,
            top=top,
            visible=self._visible,
        )
        self._object_pcd = o3d.geometry.PointCloud()

    def _object_visibility(self, chunk: Mapping[str, Any], local_frame: int, point_count: int) -> np.ndarray:
        """Return the object visibility."""
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
        """Update object colors."""
        point_count = int(object_points.shape[1])
        if self._object_colors is None or self._object_color_count != point_count:
            self._object_colors = _sam3d_rainbow_colors_rgb_float(object_points, point_count)
            self._object_color_count = point_count
        return self._object_colors

    def _reset_controller_meshes(self, controller_points: np.ndarray) -> None:
        # Controller spheres are cached and translated in place per frame;
        # rebuild them only when the controller point count changes.
        """Reset controller meshes."""
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
        """Set initial view."""
        assert self._vis is not None
        view_control = self._vis.get_view_control()
        view_control.set_front([1, 0, -2])
        view_control.set_up([0, 0, -1])
        view_control.set_zoom(1)

    def poll(self) -> bool:
        """Process Open3D visualizer events and report whether it is alive."""
        self._ensure_window()
        assert self._vis is not None
        alive = self._vis.poll_events()
        self._vis.update_renderer()
        return bool(alive) if alive is not None else True

    def update_frame(self, chunk: Mapping[str, Any], *, local_frame: int, case_dir: Path) -> bool:
        """Update the Open3D scene for one final_data frame."""
        del case_dir
        self._ensure_window()
        assert self._vis is not None
        assert self._object_pcd is not None
        o3d = self._require_open3d()

        object_arr = np.asarray(chunk.get("object_points"), dtype=np.float64)
        controller_arr = np.asarray(chunk.get("controller_points"), dtype=np.float64)
        if object_arr.ndim != 3 or controller_arr.ndim != 3:
            return self.poll()
        if int(local_frame) >= int(object_arr.shape[0]) or int(local_frame) >= int(controller_arr.shape[0]):
            return self.poll()

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

        alive = self._vis.poll_events()
        self._vis.update_renderer()
        return bool(alive) if alive is not None else True

    def render_frame(self, chunk: Mapping[str, Any], *, local_frame: int, case_dir: Path) -> np.ndarray:
        """Render one final_data frame to a BGR image."""
        if not self.update_frame(chunk, local_frame=local_frame, case_dir=case_dir):
            return _blank_image(self._image_size)
        assert self._vis is not None
        frame = np.asarray(self._vis.capture_screen_float_buffer(do_render=True))
        frame = np.clip(frame * 255.0, 0.0, 255.0).astype(np.uint8)
        cv2 = _require_cv2()
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def close(self) -> None:
        """Destroy the Open3D visualizer window when it exists."""
        if self._vis is not None:
            self._vis.destroy_window()
            self._vis = None


class Sam3DGuiFinalDataRenderer:
    """Interactive Open3D GUI renderer with a 2D latency HUD in the 3D window."""

    def __init__(
        self,
        *,
        image_size: tuple[int, int],
        show_invisible_object_points: bool,
        window_name: str = "final_data output",
        window_position: tuple[int, int] | None = None,
        show_latency_overlay: bool = True,
        object_point_size: float = 5.0,
        controller_point_size: float = 18.0,
    ) -> None:
        """Initialize Sam3DGuiFinalDataRenderer."""
        self._image_size = (int(image_size[0]), int(image_size[1]))
        self._show_invisible_object_points = bool(show_invisible_object_points)
        self._window_name = str(window_name)
        self._window_position = window_position
        self._show_latency_overlay = bool(show_latency_overlay)
        self._object_point_size = float(object_point_size)
        self._controller_point_size = float(controller_point_size)
        self._o3d: Any | None = None
        self._gui: Any | None = None
        self._rendering: Any | None = None
        self._window: Any | None = None
        self._scene_widget: Any | None = None
        self._title_label: Any | None = None
        self._latency_label: Any | None = None
        self._object_material: Any | None = None
        self._controller_material: Any | None = None
        self._object_colors: np.ndarray | None = None
        self._object_color_count = -1
        self._camera_initialized = False
        self._closed = False

    def _require_open3d_gui(self) -> tuple[Any, Any, Any]:
        """Return validated open3d gui."""
        if self._o3d is None or self._gui is None or self._rendering is None:
            import open3d as o3d
            from open3d.visualization import gui, rendering

            self._o3d = o3d
            self._gui = gui
            self._rendering = rendering
        return self._o3d, self._gui, self._rendering

    def _ensure_window(self) -> None:
        """Return the ensure window."""
        if self._window is not None:
            return
        _o3d, gui, rendering = self._require_open3d_gui()
        gui.Application.instance.initialize()
        width, height = self._image_size
        left = 50
        top = 50
        if self._window_position is not None:
            left, top = (int(self._window_position[0]), int(self._window_position[1]))
        self._window = gui.Application.instance.create_window(
            self._window_name,
            int(width),
            int(height),
            int(left),
            int(top),
        )
        self._window.set_on_close(self._on_close)
        self._scene_widget = gui.SceneWidget()
        self._scene_widget.scene = rendering.Open3DScene(self._window.renderer)
        self._scene_widget.scene.set_background([0.0, 0.0, 0.0, 1.0])
        self._scene_widget.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)

        self._title_label = gui.Label("final_data output")
        self._title_label.text_color = gui.Color(1.0, 1.0, 1.0)
        self._title_label.background_color = gui.Color(0.0, 0.0, 0.0, 0.45)
        self._latency_label = gui.Label(format_input_display_latency(None))
        self._latency_label.text_color = gui.Color(1.0, 1.0, 1.0)
        self._latency_label.background_color = gui.Color(0.0, 0.0, 0.0, 0.45)
        self._latency_label.visible = self._show_latency_overlay

        self._window.add_child(self._scene_widget)
        self._window.add_child(self._title_label)
        self._window.add_child(self._latency_label)
        self._window.set_on_layout(self._layout)

        self._object_material = rendering.MaterialRecord()
        self._object_material.shader = "defaultUnlit"
        self._object_material.point_size = max(1.0, self._object_point_size)
        self._controller_material = rendering.MaterialRecord()
        self._controller_material.shader = "defaultUnlit"
        self._controller_material.point_size = max(1.0, self._controller_point_size)

    def _layout(self, _layout_context: Any) -> None:
        """Return the layout."""
        if self._window is None or self._scene_widget is None:
            return
        gui = self._gui
        assert gui is not None
        rect = self._window.content_rect
        self._scene_widget.frame = rect
        overlay_width = min(300, max(140, int(rect.width) - 24))
        title_height = 26
        latency_height = 26
        x = int(rect.x + rect.width - overlay_width - 12)
        y = int(rect.y + 10)
        if self._title_label is not None:
            self._title_label.frame = gui.Rect(x, y, overlay_width, title_height)
        if self._latency_label is not None:
            self._latency_label.frame = gui.Rect(x, y + title_height, overlay_width, latency_height)

    def _on_close(self) -> bool:
        """Return the on close."""
        self._closed = True
        return True

    def _object_visibility(self, chunk: Mapping[str, Any], local_frame: int, point_count: int) -> np.ndarray:
        """Return the object visibility."""
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
        """Update object colors."""
        point_count = int(object_points.shape[1])
        if self._object_colors is None or self._object_color_count != point_count:
            self._object_colors = _sam3d_rainbow_colors_rgb_float(object_points, point_count)
            self._object_color_count = point_count
        return self._object_colors

    def _set_latency_label(self, latency_s: float | None) -> None:
        """Set latency label."""
        if self._latency_label is None:
            return
        self._latency_label.text = format_input_display_latency(latency_s)
        self._latency_label.visible = self._show_latency_overlay

    def _remove_geometry_if_present(self, name: str) -> None:
        """Return the remove geometry if present."""
        assert self._scene_widget is not None
        scene = self._scene_widget.scene
        try:
            if scene.has_geometry(name):
                scene.remove_geometry(name)
        except Exception:
            scene.remove_geometry(name)

    def _initialize_camera(self, points: np.ndarray) -> None:
        """Initialize camera."""
        if self._camera_initialized or points.size == 0:
            return
        assert self._scene_widget is not None
        o3d = self._o3d
        assert o3d is not None
        finite_points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        finite_points = finite_points[np.all(np.isfinite(finite_points), axis=1)]
        if finite_points.size == 0:
            return
        bounds = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
            o3d.utility.Vector3dVector(finite_points)
        )
        center = np.asarray(bounds.get_center(), dtype=np.float32)
        extent = float(np.linalg.norm(np.asarray(bounds.get_extent(), dtype=np.float64)))
        if not math.isfinite(extent) or extent <= 1e-6:
            extent = 1.0
        self._scene_widget.setup_camera(60.0, bounds, center)
        eye = center + np.asarray([0.0, -1.2 * extent, 0.8 * extent], dtype=np.float32)
        up = np.asarray([0.0, 0.0, -1.0], dtype=np.float32)
        self._scene_widget.look_at(center, eye, up)
        self._camera_initialized = True

    def poll(self) -> bool:
        """Process Open3D GUI events and report whether the window is open."""
        self._ensure_window()
        if self._closed:
            return False
        assert self._gui is not None
        if self._window is not None:
            self._window.post_redraw()
        alive = self._gui.Application.instance.run_one_tick()
        return bool(alive) and not self._closed

    def update_frame(
        self,
        chunk: Mapping[str, Any],
        *,
        local_frame: int,
        case_dir: Path,
        input_to_display_latency_s: float | None = None,
    ) -> bool:
        """Update the GUI scene and latency HUD for one final_data frame."""
        del case_dir
        self._ensure_window()
        if self._closed:
            return False
        assert self._scene_widget is not None
        assert self._object_material is not None
        assert self._controller_material is not None
        o3d = self._o3d
        assert o3d is not None
        self._set_latency_label(input_to_display_latency_s)

        object_arr = np.asarray(chunk.get("object_points"), dtype=np.float64)
        controller_arr = np.asarray(chunk.get("controller_points"), dtype=np.float64)
        if object_arr.ndim != 3 or controller_arr.ndim != 3:
            return self.poll()
        if int(local_frame) >= int(object_arr.shape[0]) or int(local_frame) >= int(controller_arr.shape[0]):
            return self.poll()

        object_frame = np.asarray(object_arr[int(local_frame)], dtype=np.float64).reshape(-1, 3)
        object_colors = self._update_object_colors(object_arr)
        visible = self._object_visibility(chunk, int(local_frame), int(object_frame.shape[0]))
        object_valid = visible & np.all(np.isfinite(object_frame), axis=1)
        controller_frame = np.asarray(controller_arr[int(local_frame)], dtype=np.float64).reshape(-1, 3)
        controller_valid = np.all(np.isfinite(controller_frame), axis=1)
        controller_points = controller_frame[controller_valid]

        object_pcd = o3d.geometry.PointCloud()
        object_pcd.points = o3d.utility.Vector3dVector(object_frame[object_valid])
        object_pcd.colors = o3d.utility.Vector3dVector(object_colors[object_valid])
        controller_pcd = o3d.geometry.PointCloud()
        controller_pcd.points = o3d.utility.Vector3dVector(controller_points)
        controller_color = np.tile(np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64), (len(controller_points), 1))
        controller_pcd.colors = o3d.utility.Vector3dVector(controller_color)

        self._remove_geometry_if_present("object_points")
        self._remove_geometry_if_present("controller_points")
        self._scene_widget.scene.add_geometry("object_points", object_pcd, self._object_material)
        self._scene_widget.scene.add_geometry("controller_points", controller_pcd, self._controller_material)

        all_points = np.concatenate([object_frame[object_valid], controller_points], axis=0)
        self._initialize_camera(all_points)
        return self.poll()

    def close(self) -> None:
        """Close the Open3D GUI window and mark the renderer closed."""
        if self._window is not None:
            self._window.close()
            self._window = None
        self._closed = True


def build_frame_renderer(args: argparse.Namespace, *, camera: CameraModel, fps: float) -> Any:
    """Create the frame renderer selected by CLI arguments."""
    render_mode = str(args.render_mode)
    if render_mode == RENDER_MODE_RGB_OVERLAY:
        return RgbOverlayRenderer(camera=camera, args=args, fps=fps)
    if render_mode == RENDER_MODE_SAM3D_FINAL_DATA:
        return Sam3DFinalDataRenderer(
            image_size=camera.image_size,
            show_invisible_object_points=bool(args.show_invisible_object_points),
        )
    raise ValueError(f"unsupported render mode: {render_mode!r}")
