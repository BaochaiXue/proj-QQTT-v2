from __future__ import annotations

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
    tracking_background: str = "target-union"
    object_point_count: int = 0
    controller_point_count: int = 0

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
    hud_y = max(0, panel.shape[0] - 48)
    _draw_text_lines(panel, _hud_lines(inputs.hud), origin=(2, hud_y))
    return np.ascontiguousarray(panel, dtype=np.uint8)


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
    intrinsics: dict[str, Any],
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
    valid = np.isfinite(points).all(axis=1) & (z > np.float32(1e-6))
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    u = np.rint(points[:, 0] * fx / z + cx).astype(np.int32)
    v = np.rint(points[:, 1] * fy / z + cy).astype(np.int32)
    valid &= (u >= 0) & (u < int(width)) & (v >= 0) & (v < int(height))
    return np.stack([u, v], axis=1), valid


def _draw_projected_points(
    image_bgr: np.ndarray,
    points_xyz: np.ndarray,
    colors_rgb: np.ndarray,
    intrinsics: dict[str, Any],
    *,
    point_size: int,
    max_points: int,
    coordinate_frame: str,
    camera_to_world_c2w: Any | None,
) -> int:
    points = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    colors = np.asarray(colors_rgb, dtype=np.uint8).reshape(-1, 3)
    count = min(len(points), len(colors))
    points = points[:count]
    colors = colors[:count]
    if count == 0:
        return 0

    if int(max_points) > 0 and count > int(max_points):
        indices = np.linspace(0, count - 1, int(max_points), dtype=np.int64)
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
    intrinsics: dict[str, Any],
    controller_xyz_m: np.ndarray,
    controller_rgb_u8: np.ndarray,
    object_xyz_m: np.ndarray,
    object_rgb_u8: np.ndarray,
    point_size: int,
    max_render_points: int,
    coordinate_frame: str,
    camera_to_world_c2w: Any | None,
) -> tuple[np.ndarray, dict[str, int]]:
    if int(width) <= 0 or int(height) <= 0:
        raise ValueError("width and height must be positive")

    image = np.zeros((int(height), int(width), 3), dtype=np.uint8)
    controller_count = _draw_projected_points(
        image,
        controller_xyz_m,
        controller_rgb_u8,
        intrinsics,
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
        point_size=point_size,
        max_points=max_render_points,
        coordinate_frame=coordinate_frame,
        camera_to_world_c2w=camera_to_world_c2w,
    )
    return image, {"controller_points": controller_count, "object_points": object_count}


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
    tracks = np.asarray(tracks_yx, dtype=np.float32).reshape(-1, 2)
    visible = np.asarray(visibility, dtype=np.float32).reshape(-1) > 0.0
    colors = np.asarray(marker_rgb_u8, dtype=np.uint8).reshape(-1, 3)
    is_object = np.asarray(query_is_object, dtype=bool).reshape(-1)
    is_controller = np.asarray(query_is_controller, dtype=bool).reshape(-1)
    controller_instance = np.asarray(query_controller_instance_id, dtype=np.int64).reshape(-1)

    count = min(len(tracks), len(visible), len(colors), len(is_object), len(is_controller), len(controller_instance))
    counts = {
        "query_points": 0,
        "query_object_points": 0,
        "query_controller_points": 0,
        "query_hand_a_points": 0,
        "query_hand_b_points": 0,
    }
    radius = max(1, int(marker_radius))

    for index in range(count):
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
