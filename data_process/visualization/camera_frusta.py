from __future__ import annotations

from typing import Any

import numpy as np


CAMERA_COLORS_BGR: tuple[tuple[int, int, int], ...] = (
    (80, 180, 255),
    (120, 220, 120),
    (255, 180, 90),
    (220, 120, 220),
)


def _normalize_vector(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    vec = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-6:
        return np.asarray(fallback, dtype=np.float32)
    return vec / norm


def extract_camera_pose(
    c2w: np.ndarray,
    *,
    camera_idx: int,
    serial: str | None = None,
) -> dict[str, Any]:
    transform = np.asarray(c2w, dtype=np.float32).reshape(4, 4)
    position = transform[:3, 3].astype(np.float32)
    right = _normalize_vector(transform[:3, 0], np.array([1.0, 0.0, 0.0], dtype=np.float32))
    up = _normalize_vector(-transform[:3, 1], np.array([0.0, 0.0, 1.0], dtype=np.float32))
    forward = _normalize_vector(transform[:3, 2], np.array([0.0, 0.0, 1.0], dtype=np.float32))
    short_label = f"Cam{camera_idx}"
    full_label = f"{short_label} | {serial}" if serial else short_label
    return {
        "camera_idx": int(camera_idx),
        "serial": serial,
        "short_label": short_label,
        "label": full_label,
        "position": position,
        "right": right,
        "up": up,
        "forward": forward,
        "color_bgr": CAMERA_COLORS_BGR[int(camera_idx) % len(CAMERA_COLORS_BGR)],
        "transform": transform,
    }


def extract_camera_poses(
    c2w_list: list[np.ndarray],
    *,
    serial_numbers: list[str] | None = None,
    camera_ids: list[int] | None = None,
) -> list[dict[str, Any]]:
    if serial_numbers is None:
        serial_numbers = [None] * len(c2w_list)
    if len(serial_numbers) != len(c2w_list):
        raise ValueError("serial_numbers must match c2w_list length.")
    selected_ids = list(range(len(c2w_list))) if camera_ids is None else [int(idx) for idx in camera_ids]
    poses: list[dict[str, Any]] = []
    for camera_idx in selected_ids:
        if camera_idx < 0 or camera_idx >= len(c2w_list):
            raise ValueError(f"camera_idx out of range: {camera_idx}")
        poses.append(
            extract_camera_pose(
                c2w_list[camera_idx],
                camera_idx=camera_idx,
                serial=serial_numbers[camera_idx],
            )
        )
    return poses


def build_camera_frustum_geometry(
    pose: dict[str, Any],
    *,
    frustum_scale: float,
    aspect_ratio: float = 4.0 / 3.0,
) -> dict[str, Any]:
    origin = np.asarray(pose["position"], dtype=np.float32)
    right = np.asarray(pose["right"], dtype=np.float32)
    up = np.asarray(pose["up"], dtype=np.float32)
    forward = np.asarray(pose["forward"], dtype=np.float32)

    depth = max(1e-3, float(frustum_scale))
    half_h = depth * 0.35
    half_w = half_h * float(aspect_ratio)
    plane_center = origin + forward * depth
    corners = np.stack(
        [
            plane_center - right * half_w + up * half_h,
            plane_center + right * half_w + up * half_h,
            plane_center + right * half_w - up * half_h,
            plane_center - right * half_w - up * half_h,
        ],
        axis=0,
    ).astype(np.float32)
    forward_tip = (origin + forward * depth * 1.25).astype(np.float32)
    label_anchor = (plane_center + up * half_h * 1.25 + right * half_w * 0.15).astype(np.float32)

    segments = [
        (origin, corners[0]),
        (origin, corners[1]),
        (origin, corners[2]),
        (origin, corners[3]),
        (corners[0], corners[1]),
        (corners[1], corners[2]),
        (corners[2], corners[3]),
        (corners[3], corners[0]),
        (origin, forward_tip),
    ]

    return {
        **pose,
        "frustum_scale": depth,
        "frustum_corners": corners,
        "forward_tip": forward_tip,
        "label_anchor": label_anchor,
        "segments": segments,
    }


def collect_camera_geometry_points(camera_geometries: list[dict[str, Any]]) -> np.ndarray:
    points: list[np.ndarray] = []
    for geometry in camera_geometries:
        points.append(np.asarray(geometry["position"], dtype=np.float32))
        points.append(np.asarray(geometry["forward_tip"], dtype=np.float32))
        points.append(np.asarray(geometry["label_anchor"], dtype=np.float32))
        points.append(np.asarray(geometry["frustum_corners"], dtype=np.float32))
    if not points:
        return np.empty((0, 3), dtype=np.float32)
    stacked = [
        item.reshape(-1, 3).astype(np.float32)
        for item in points
    ]
    return np.concatenate(stacked, axis=0)

