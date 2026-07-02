"""Side-by-side RGB / projected-PCD / tracking-overlay panel rendering."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


CAMERA_COLOR_FRAME = "camera_color_frame"
TABLE_WORLD_FRAME_KIND = "table_world_z0"


@dataclass(frozen=True)
class SideBySidePanelHud:
    rgb_seq: int
    paired_seq: int
    input_time_s: float | None
    pipeline_latency_ms: float
    display_latency_ms: float
    startup_hold_s: float
    filter_preset: str
    marker_count: int
    capture_fps: float = 0.0
    seg_fps: float = 0.0
    depth_fps: float = 0.0
    pcd_fps: float = 0.0
    tracker_fps: float = 0.0
    render_fps: float = 0.0
    tracking_background: str = "target-union"
    object_point_count: int = 0
    controller_point_count: int = 0
    query_count: int = 0
    remaining_query_count: int = 0
    remaining_object_query_count: int = 0
    remaining_controller_query_count: int = 0
    remaining_hand_a_query_count: int = 0
    remaining_hand_b_query_count: int = 0
    shape_prior_status: str = "disabled"
    shape_prior_point_count: int = 0

    @property
    def rgb_ahead_frames(self) -> int:
        return compute_rgb_ahead_frames(rgb_seq=self.rgb_seq, paired_seq=self.paired_seq)


@dataclass(frozen=True)
class SideBySidePanelInputs:
    rgb_image_bgr: np.ndarray
    pcd_panel_bgr: np.ndarray
    tracking_panel_bgr: np.ndarray
    hud: SideBySidePanelHud


def compute_rgb_ahead_frames(*, rgb_seq: int, paired_seq: int) -> int:
    return max(0, int(rgb_seq) - int(paired_seq))


def format_side_by_side_fps_line(hud: SideBySidePanelHud) -> str:
    return (
        "FPS cap/seg/depth/pcd/tracker/render: "
        f"{float(hud.capture_fps):.1f}/"
        f"{float(hud.seg_fps):.1f}/"
        f"{float(hud.depth_fps):.1f}/"
        f"{float(hud.pcd_fps):.1f}/"
        f"{float(hud.tracker_fps):.1f}/"
        f"{float(hud.render_fps):.1f}"
    )


def _as_bgr_u8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"panel image must be HxWx3, got {arr.shape}")
    if arr.dtype == np.uint8:
        return np.ascontiguousarray(arr)
    return np.ascontiguousarray(np.clip(arr, 0, 255).astype(np.uint8))


def _resize_to_cell(image: np.ndarray, cell_size: tuple[int, int]) -> np.ndarray:
    width, height = int(cell_size[0]), int(cell_size[1])
    if width <= 0 or height <= 0:
        raise ValueError("cell_size must contain positive width and height")

    image_u8 = _as_bgr_u8(image)
    if image_u8.shape[:2] == (height, width):
        return image_u8.copy()
    return cv2.resize(image_u8, (width, height), interpolation=cv2.INTER_LINEAR)


def _draw_text_lines(image: np.ndarray, lines: list[str], *, origin: tuple[int, int]) -> None:
    if not lines:
        return

    x = max(0, min(int(origin[0]), max(0, image.shape[1] - 1)))
    y = max(0, min(int(origin[1]), max(0, image.shape[0] - 1)))
    scale = 0.38
    thickness = 1
    line_height = 14
    max_text_width = max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0] for line in lines)
    box_width = min(image.shape[1] - x, max(1, max_text_width + 8))
    box_height = min(image.shape[0] - y, max(1, line_height * len(lines) + 5))
    if box_width <= 0 or box_height <= 0:
        return

    overlay = image.copy()
    cv2.rectangle(overlay, (x, y), (x + box_width - 1, y + box_height - 1), (0, 0, 0), thickness=-1)
    cv2.addWeighted(overlay, 0.55, image, 0.45, 0.0, dst=image)

    cursor_y = y + min(11, max(1, box_height - 1))
    for line in lines:
        if cursor_y >= image.shape[0]:
            break
        cv2.putText(
            image,
            line,
            (x + 4, cursor_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        cursor_y += line_height


def _hud_lines(hud: SideBySidePanelHud) -> list[str]:
    input_time = "none" if hud.input_time_s is None else f"{float(hud.input_time_s):.2f}s"
    return [
        f"rgb={int(hud.rgb_seq)} paired={int(hud.paired_seq)} ahead={hud.rgb_ahead_frames}f",
        f"input={input_time} pipe={float(hud.pipeline_latency_ms):.1f}ms disp={float(hud.display_latency_ms):.1f}ms",
        f"hold={float(hud.startup_hold_s):.2f}s filter={hud.filter_preset} markers={int(hud.marker_count)}",
        f"bg={hud.tracking_background} obj={int(hud.object_point_count)} ctrl={int(hud.controller_point_count)}",
        f"shape_prior={hud.shape_prior_status} pts={int(hud.shape_prior_point_count)}",
        format_side_by_side_fps_line(hud),
    ]


def _remaining_query_legend_lines(hud: SideBySidePanelHud) -> list[str]:
    query_count = max(0, int(hud.query_count))
    remaining = max(0, int(hud.remaining_query_count))
    if query_count <= 0 and remaining <= 0:
        return []
    return [
        f"remaining {remaining}/{query_count}",
        f"obj={int(hud.remaining_object_query_count)} ctrl={int(hud.remaining_controller_query_count)}",
        f"hand_a={int(hud.remaining_hand_a_query_count)} hand_b={int(hud.remaining_hand_b_query_count)}",
    ]


def render_side_by_side_panel(
    inputs: SideBySidePanelInputs,
    *,
    cell_size: tuple[int, int] | None = None,
) -> np.ndarray:
    left_source = _as_bgr_u8(inputs.rgb_image_bgr)
    if cell_size is None:
        cell_size = (int(left_source.shape[1]), int(left_source.shape[0]))

    left = _resize_to_cell(left_source, cell_size)
    middle = _resize_to_cell(inputs.pcd_panel_bgr, cell_size)
    right = _resize_to_cell(inputs.tracking_panel_bgr, cell_size)

    panel = np.concatenate([left, middle, right], axis=1)
    _draw_text_lines(panel, _remaining_query_legend_lines(inputs.hud), origin=(2, 2))
    hud_lines = _hud_lines(inputs.hud)
    hud_y = max(0, panel.shape[0] - (14 * len(hud_lines) + 6))
    _draw_text_lines(panel, hud_lines, origin=(2, hud_y))
    return np.ascontiguousarray(panel, dtype=np.uint8)


def _intrinsics_values(intrinsics: Any) -> tuple[float, float, float, float]:
    if isinstance(intrinsics, Mapping):
        return (
            float(intrinsics["fx"]),
            float(intrinsics["fy"]),
            float(intrinsics["cx"]),
            float(intrinsics["cy"]),
        )

    if all(hasattr(intrinsics, name) for name in ("fx", "fy", "cx", "cy")):
        return (
            float(intrinsics.fx),
            float(intrinsics.fy),
            float(intrinsics.cx),
            float(intrinsics.cy),
        )

    matrix = np.asarray(intrinsics)
    if matrix.shape == (3, 3):
        return float(matrix[0, 0]), float(matrix[1, 1]), float(matrix[0, 2]), float(matrix[1, 2])

    raise ValueError("intrinsics must be a mapping, object, or 3x3 matrix with fx/fy/cx/cy")


def _camera_points_for_frame(
    points_xyz: np.ndarray,
    *,
    coordinate_frame: str,
    camera_to_world_c2w: Any | None,
) -> np.ndarray:
    points = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    if coordinate_frame == CAMERA_COLOR_FRAME:
        return points
    if coordinate_frame != TABLE_WORLD_FRAME_KIND:
        raise ValueError(f"unsupported coordinate_frame: {coordinate_frame!r}")
    if camera_to_world_c2w is None:
        raise ValueError("table_world_z0 projection requires camera_to_world_c2w")
    if len(points) == 0:
        return points

    c2w = np.asarray(camera_to_world_c2w, dtype=np.float64).reshape(4, 4)
    w2c = np.linalg.inv(c2w)
    homogeneous = np.concatenate([points.astype(np.float64), np.ones((len(points), 1), dtype=np.float64)], axis=1)
    return np.ascontiguousarray((w2c @ homogeneous.T).T[:, :3], dtype=np.float32)


def _project_points(
    points_xyz: np.ndarray,
    intrinsics: Any,
    *,
    width: int,
    height: int,
    coordinate_frame: str,
    camera_to_world_c2w: Any | None,
) -> tuple[np.ndarray, np.ndarray]:
    points = _camera_points_for_frame(
        points_xyz,
        coordinate_frame=coordinate_frame,
        camera_to_world_c2w=camera_to_world_c2w,
    )
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.int32), np.empty((0,), dtype=bool)

    z = points[:, 2]
    finite_depth = np.isfinite(points).all(axis=1) & (z > np.float32(1e-6))
    pixels = np.zeros((len(points), 2), dtype=np.int32)
    valid = np.zeros((len(points),), dtype=bool)
    if not np.any(finite_depth):
        return pixels, valid

    fx, fy, cx, cy = _intrinsics_values(intrinsics)
    valid_indices = np.flatnonzero(finite_depth)
    valid_points = points[valid_indices]
    valid_z = valid_points[:, 2]
    u = np.rint(valid_points[:, 0] * fx / valid_z + cx).astype(np.int32)
    v = np.rint(valid_points[:, 1] * fy / valid_z + cy).astype(np.int32)
    in_bounds = (u >= 0) & (u < int(width)) & (v >= 0) & (v < int(height))
    pixels[valid_indices, 0] = u
    pixels[valid_indices, 1] = v
    valid[valid_indices] = in_bounds
    return pixels, valid


def _reshape_points(name: str, points_xyz: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xyz, dtype=np.float32)
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} must be an Nx3 array, got {points.shape}")
    return np.ascontiguousarray(points)


def _reshape_rgb(name: str, rgb_u8: np.ndarray) -> np.ndarray:
    colors = np.asarray(rgb_u8, dtype=np.uint8)
    if colors.size == 0:
        return np.empty((0, 3), dtype=np.uint8)
    if colors.ndim != 2 or colors.shape[1] != 3:
        raise ValueError(f"{name} must be an Nx3 array, got {colors.shape}")
    return np.ascontiguousarray(colors)


def _require_same_length(first_name: str, first: np.ndarray, second_name: str, second: np.ndarray) -> None:
    if len(first) != len(second):
        raise ValueError(f"{first_name} length {len(first)} must match {second_name} length {len(second)}")


def _draw_projected_points(
    image_bgr: np.ndarray,
    points_xyz: np.ndarray,
    colors_rgb: np.ndarray,
    intrinsics: Any,
    *,
    points_name: str,
    colors_name: str,
    point_size: int,
    max_points: int,
    coordinate_frame: str,
    camera_to_world_c2w: Any | None,
) -> int:
    points = _reshape_points(points_name, points_xyz)
    colors = _reshape_rgb(colors_name, colors_rgb)
    _require_same_length(points_name, points, colors_name, colors)
    if len(points) == 0:
        return 0

    if int(max_points) > 0 and len(points) > int(max_points):
        indices = np.linspace(0, len(points) - 1, int(max_points), dtype=np.int64)
        points = points[indices]
        colors = colors[indices]

    pixels, valid = _project_points(
        points,
        intrinsics,
        width=image_bgr.shape[1],
        height=image_bgr.shape[0],
        coordinate_frame=coordinate_frame,
        camera_to_world_c2w=camera_to_world_c2w,
    )
    radius = max(1, int(point_size))
    drawn = 0
    for (u, v), ok, rgb in zip(pixels, valid, colors, strict=False):
        if not bool(ok):
            continue
        cv2.circle(image_bgr, (int(u), int(v)), radius, tuple(int(value) for value in rgb[::-1]), thickness=-1)
        drawn += 1
    return drawn


def render_projected_pcd_panel(
    *,
    width: int,
    height: int,
    intrinsics: Any,
    controller_xyz_m: np.ndarray,
    controller_rgb_u8: np.ndarray,
    object_xyz_m: np.ndarray,
    object_rgb_u8: np.ndarray,
    point_size: int,
    max_render_points: int,
    coordinate_frame: str,
    camera_to_world_c2w: Any | None,
    shape_prior_xyz_m: np.ndarray | None = None,
    shape_prior_rgb_u8: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    if int(width) <= 0 or int(height) <= 0:
        raise ValueError("width and height must be positive")

    image = np.zeros((int(height), int(width), 3), dtype=np.uint8)
    controller_count = _draw_projected_points(
        image,
        controller_xyz_m,
        controller_rgb_u8,
        intrinsics,
        points_name="controller_xyz_m",
        colors_name="controller_rgb_u8",
        point_size=point_size,
        max_points=max_render_points,
        coordinate_frame=coordinate_frame,
        camera_to_world_c2w=camera_to_world_c2w,
    )
    object_count = _draw_projected_points(
        image,
        object_xyz_m,
        object_rgb_u8,
        intrinsics,
        points_name="object_xyz_m",
        colors_name="object_rgb_u8",
        point_size=point_size,
        max_points=max_render_points,
        coordinate_frame=coordinate_frame,
        camera_to_world_c2w=camera_to_world_c2w,
    )
    shape_prior_count = _draw_projected_points(
        image,
        np.empty((0, 3), dtype=np.float32) if shape_prior_xyz_m is None else shape_prior_xyz_m,
        np.empty((0, 3), dtype=np.uint8) if shape_prior_rgb_u8 is None else shape_prior_rgb_u8,
        intrinsics,
        points_name="shape_prior_xyz_m",
        colors_name="shape_prior_rgb_u8",
        point_size=point_size,
        max_points=max_render_points,
        coordinate_frame=coordinate_frame,
        camera_to_world_c2w=camera_to_world_c2w,
    )
    return image, {
        "controller_points": controller_count,
        "object_points": object_count,
        "shape_prior_points": shape_prior_count,
    }


def render_tracking_overlay_panel(
    *,
    image_bgr: np.ndarray,
    tracks_yx: np.ndarray,
    visibility: np.ndarray,
    marker_rgb_u8: np.ndarray,
    query_is_object: np.ndarray,
    query_is_controller: np.ndarray,
    query_controller_instance_id: np.ndarray,
    marker_radius: int,
) -> tuple[np.ndarray, dict[str, int]]:
    image = _as_bgr_u8(image_bgr).copy()
    tracks = np.asarray(tracks_yx, dtype=np.float32)
    if tracks.size == 0:
        tracks = np.empty((0, 2), dtype=np.float32)
    if tracks.ndim != 2 or tracks.shape[1] != 2:
        raise ValueError(f"tracks_yx must be an Nx2 array, got {tracks.shape}")
    visible = np.asarray(visibility, dtype=np.float32).reshape(-1) > 0.0
    colors = _reshape_rgb("marker_rgb_u8", marker_rgb_u8)
    is_object = np.asarray(query_is_object, dtype=bool).reshape(-1)
    is_controller = np.asarray(query_is_controller, dtype=bool).reshape(-1)
    controller_instance = np.asarray(query_controller_instance_id, dtype=np.int64).reshape(-1)

    for name, arr in (
        ("visibility", visible),
        ("marker_rgb_u8", colors),
        ("query_is_object", is_object),
        ("query_is_controller", is_controller),
        ("query_controller_instance_id", controller_instance),
    ):
        _require_same_length("tracks_yx", tracks, name, arr)

    counts = {
        "query_points": 0,
        "query_object_points": 0,
        "query_controller_points": 0,
        "query_hand_a_points": 0,
        "query_hand_b_points": 0,
    }
    radius = max(1, int(marker_radius))

    for index in range(len(tracks)):
        if not bool(visible[index]):
            continue
        y, x = float(tracks[index, 0]), float(tracks[index, 1])
        if not np.isfinite([y, x]).all():
            continue
        yy, xx = int(round(y)), int(round(x))
        if yy < 0 or yy >= image.shape[0] or xx < 0 or xx >= image.shape[1]:
            continue

        color_bgr = tuple(int(value) for value in colors[index, ::-1])
        cv2.circle(image, (xx, yy), radius, color_bgr, thickness=-1)
        counts["query_points"] += 1
        if bool(is_controller[index]):
            counts["query_controller_points"] += 1
            if int(controller_instance[index]) == 1:
                counts["query_hand_a_points"] += 1
            elif int(controller_instance[index]) == 2:
                counts["query_hand_b_points"] += 1
        elif bool(is_object[index]):
            counts["query_object_points"] += 1

    return image, counts
