"""Interactive Open3D mesh viewer embedded in Qt (drag to orbit).

Open3D's windowed viewer cannot be embedded in a foreign Qt window, so this
widget drives an ``OffscreenRenderer`` instead: every camera change re-renders
the mesh to an image shown in the widget. Interaction model:

- left-drag  = orbit (azimuth/elevation around the mesh bbox center)
- wheel      = zoom (distance)
- double-click = reset view

Renders happen on the GUI thread but are coalesced through a single-shot
timer (~60 Hz cap) and take ~10-20 ms for these single-object meshes, so
dragging stays fluid without touching any pipeline thread. Open3D import and
EGL context creation are lazy (first ``setMeshPath``) and best-effort: any
failure downgrades the widget to a message label — the GUI never dies over a
viewer. No CUDA is involved (EGL/OpenGL only), and everything runs in the
GUI process — the camera service is untouched.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QMouseEvent, QPainter, QPixmap, QWheelEvent
from PySide6.QtWidgets import QSizePolicy, QWidget

_RENDER_W = 960
_RENDER_H = 720
_MIN_ELEVATION_DEG = -89.0
_MAX_ELEVATION_DEG = 89.0
_ZOOM_STEP = 1.12
_ORBIT_DEG_PER_PX = 0.4
# Filament's render_to_image returns LINEAR RGB on this build (verified: the
# raw output is ~6x darker than the albedo texture and does not respond to
# more light); encode to sRGB ourselves via a LUT. Background/point colors
# are therefore specified as the LINEAR values that ENCODE to the intended
# theme sRGB colors.
_IBL_INTENSITY = 90000.0
_SUN_INTENSITY = 200000.0


def _srgb_encode(linear: np.ndarray) -> np.ndarray:
    return np.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * np.power(np.clip(linear, 0.0, 1.0), 1.0 / 2.4) - 0.055,
    )


def _srgb_decode(srgb: np.ndarray) -> np.ndarray:
    return np.where(
        srgb <= 0.04045, srgb / 12.92, np.power((srgb + 0.055) / 1.055, 2.4)
    )


_SRGB_LUT = np.round(_srgb_encode(np.arange(256) / 255.0) * 255.0).astype(np.uint8)
# App theme #202124 expressed in linear so the encoded output matches it.
_BACKGROUND_LINEAR_RGBA = tuple(
    float(v) for v in _srgb_decode(np.asarray([0x20, 0x21, 0x24]) / 255.0)
) + (1.0,)


def _linear_color_of_srgb_u8(rgb: tuple) -> list:
    return [float(v) for v in _srgb_decode(np.asarray(rgb, dtype=np.float64) / 255.0)]


class MeshOrbitView(QWidget):
    """Drag-to-orbit mesh view backed by an Open3D offscreen renderer."""

    def __init__(self, placeholder: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._placeholder = str(placeholder)
        self._error: str | None = None
        self._renderer: Any = None
        self._o3d: Any = None
        self._mesh_path: str | None = None
        self._point_names: list[str] = []
        self._pixmap: QPixmap | None = None
        # Orbit state (spherical around the mesh bbox center).
        self._center = np.zeros(3, dtype=np.float64)
        self._distance = 1.0
        self._home_distance = 1.0
        self._azimuth_deg = 35.0
        self._elevation_deg = 20.0
        self._drag_last: tuple[float, float] | None = None
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(16)
        self._render_timer.timeout.connect(self._render_now)
        self.setMinimumSize(360, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    # ---- public API --------------------------------------------------------

    def setMeshPath(self, path: str) -> None:  # noqa: N802 (Qt style)
        """Load a mesh file (glb/obj/...) and show it; safe to call again."""
        if path == self._mesh_path:
            return
        try:
            self._ensure_renderer()
            model = self._o3d.io.read_triangle_model(str(path))
            # SAM3D glbs bake the look into the albedo texture but ship
            # metallic=1, which swallows the diffuse term under any light —
            # neutralize the PBR params, keep the texture.
            for material in model.materials:
                material.shader = "defaultLit"
                material.base_metallic = 0.0
                material.base_roughness = 0.7
            self._renderer.scene.clear_geometry()
            self._point_names = []
            self._renderer.scene.add_model("mesh", model)
            bounds = self._renderer.scene.bounding_box
            self._center = np.asarray(bounds.get_center(), dtype=np.float64)
            extent = float(np.linalg.norm(np.asarray(bounds.get_extent())))
            self._home_distance = max(extent, 1e-6) * 1.8
            self._distance = self._home_distance
            self._azimuth_deg = 35.0
            self._elevation_deg = 20.0
            self._mesh_path = str(path)
            self._error = None
            self._schedule_render()
        except Exception as exc:
            # Best-effort viewer: never let a render backend kill the GUI.
            self._error = f"mesh 预览不可用: {exc}"
            self._pixmap = None
            self.update()

    def setPointSets(
        self,
        point_sets: dict[str, tuple[np.ndarray, tuple[int, int, int]]],
        *,
        point_size: float = 3.0,
    ) -> None:
        """Show named point clouds ({name: (Nx3 points, sRGB u8 color)}).

        Replaces whatever the widget showed before (mesh or points); use
        ``setSourceVisible`` to toggle one source without reloading. Colors
        are given as display sRGB and converted to the linear values the
        renderer expects (the output pass re-encodes).
        """
        try:
            self._ensure_renderer()
            self._renderer.scene.clear_geometry()
            self._mesh_path = None
            self._point_names = []
            bounds_min: list[np.ndarray] = []
            bounds_max: list[np.ndarray] = []
            for name, (points, srgb_u8) in point_sets.items():
                pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
                if pts.shape[0] == 0:
                    continue
                cloud = self._o3d.geometry.PointCloud()
                cloud.points = self._o3d.utility.Vector3dVector(pts)
                material = self._o3d.visualization.rendering.MaterialRecord()
                material.shader = "defaultUnlit"
                material.base_color = _linear_color_of_srgb_u8(srgb_u8) + [1.0]
                material.point_size = float(point_size)
                self._renderer.scene.add_geometry(str(name), cloud, material)
                self._point_names.append(str(name))
                bounds_min.append(pts.min(axis=0))
                bounds_max.append(pts.max(axis=0))
            if not self._point_names:
                self._error = "补点数据为空"
                self._pixmap = None
                self.update()
                return
            low = np.min(np.stack(bounds_min), axis=0)
            high = np.max(np.stack(bounds_max), axis=0)
            self._center = (low + high) / 2.0
            extent = float(np.linalg.norm(high - low))
            self._home_distance = max(extent, 1e-6) * 1.8
            self._distance = self._home_distance
            self._azimuth_deg = 35.0
            self._elevation_deg = 20.0
            # Non-None sentinel so the interaction handlers engage.
            self._mesh_path = "<points>"
            self._error = None
            self._schedule_render()
        except Exception as exc:
            self._error = f"补点预览不可用: {exc}"
            self._pixmap = None
            self.update()

    def setSourceVisible(self, name: str, visible: bool) -> None:  # noqa: N802
        """Toggle one named point set from ``setPointSets``."""
        if self._renderer is None or name not in self._point_names:
            return
        try:
            self._renderer.scene.show_geometry(str(name), bool(visible))
            self._schedule_render()
        except Exception:
            pass

    def setPlaceholderText(self, text: str) -> None:  # noqa: N802 (Qt style)
        """Replace the idle placeholder line (shown until content loads)."""
        self._placeholder = str(text)
        if self._pixmap is None:
            self.update()

    def clear(self) -> None:
        """Drop the loaded mesh/points and return to the placeholder."""
        self._mesh_path = None
        self._pixmap = None
        self._point_names = []
        if self._renderer is not None:
            try:
                self._renderer.scene.clear_geometry()
            except Exception:
                pass
        self.update()

    # ---- rendering ---------------------------------------------------------

    def _ensure_renderer(self) -> None:
        if self._renderer is not None:
            return
        import open3d as o3d  # noqa: PLC0415 (lazy: EGL context on demand)

        renderer = o3d.visualization.rendering.OffscreenRenderer(
            _RENDER_W, _RENDER_H
        )
        renderer.scene.set_background(list(_BACKGROUND_LINEAR_RGBA))
        renderer.scene.scene.enable_indirect_light(True)
        renderer.scene.scene.set_indirect_light_intensity(_IBL_INTENSITY)
        # The sun acts as a camera headlight: its direction follows the view
        # and is re-set on every render (see _render_now).
        renderer.scene.scene.enable_sun_light(True)
        self._o3d = o3d
        self._renderer = renderer

    def _schedule_render(self) -> None:
        if not self._render_timer.isActive():
            self._render_timer.start()

    def _render_now(self) -> None:
        if self._renderer is None or self._mesh_path is None:
            return
        az = math.radians(self._azimuth_deg)
        el = math.radians(self._elevation_deg)
        eye = self._center + self._distance * np.asarray(
            [
                math.cos(el) * math.cos(az),
                math.cos(el) * math.sin(az),
                math.sin(el),
            ]
        )
        self._renderer.setup_camera(
            60.0, self._center.tolist(), eye.tolist(), [0.0, 0.0, 1.0]
        )
        # Camera headlight: keep the sun on the view direction so the facing
        # side is always lit regardless of orbit angle.
        view_dir = (self._center - eye) / max(
            float(np.linalg.norm(self._center - eye)), 1e-9
        )
        self._renderer.scene.scene.set_sun_light(
            view_dir.tolist(), [1.0, 1.0, 1.0], _SUN_INTENSITY
        )
        image = np.ascontiguousarray(np.asarray(self._renderer.render_to_image()))
        # Linear -> sRGB (see module constants) via LUT; ~1ms at 960x720.
        image = _SRGB_LUT[image]
        height, width = image.shape[:2]
        qimage = QImage(
            image.data, width, height, 3 * width, QImage.Format.Format_RGB888
        ).copy()
        self._pixmap = QPixmap.fromImage(qimage)
        self.update()

    # ---- interaction -------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._mesh_path:
            self._drag_last = (event.position().x(), event.position().y())
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_last is None:
            return
        x, y = event.position().x(), event.position().y()
        last_x, last_y = self._drag_last
        self._drag_last = (x, y)
        self._azimuth_deg = (self._azimuth_deg - (x - last_x) * _ORBIT_DEG_PER_PX) % 360.0
        self._elevation_deg = min(
            _MAX_ELEVATION_DEG,
            max(_MIN_ELEVATION_DEG, self._elevation_deg + (y - last_y) * _ORBIT_DEG_PER_PX),
        )
        self._schedule_render()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._drag_last = None
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    def wheelEvent(self, event: QWheelEvent) -> None:
        if not self._mesh_path:
            return
        if event.angleDelta().y() > 0:
            self._distance /= _ZOOM_STEP
        else:
            self._distance *= _ZOOM_STEP
        self._distance = min(max(self._distance, 1e-4), self._home_distance * 20.0)
        self._schedule_render()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if not self._mesh_path:
            return
        self._distance = self._home_distance
        self._azimuth_deg = 35.0
        self._elevation_deg = 20.0
        self._schedule_render()

    # ---- painting ----------------------------------------------------------

    def paintEvent(self, event: Any) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.palette().window())
        if self._pixmap is not None:
            scaled = self._pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
        else:
            painter.setPen(Qt.GlobalColor.gray)
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                self._error or self._placeholder,
            )
        painter.end()
