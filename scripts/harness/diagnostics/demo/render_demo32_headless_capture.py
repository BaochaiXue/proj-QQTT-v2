#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np

def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "qqtt").is_dir() and (candidate / "scripts").is_dir():
            return candidate
    raise RuntimeError(f"failed to locate repo root from {start}")


ROOT = _find_repo_root(Path(__file__).resolve())
ROOT_STR = str(ROOT)
if ROOT_STR in sys.path:
    sys.path.remove(ROOT_STR)
sys.path.insert(0, ROOT_STR)

from qqtt.demo.query_rainbow import query_rainbow_colors_for_indices
from qqtt.demo.demo32_side_by_side_panel import (
    SideBySidePanelHud,
    SideBySidePanelInputs,
    compute_rgb_ahead_frames,
    render_projected_pcd_panel,
    render_side_by_side_panel,
    render_tracking_overlay_panel,
)


DEMO_VISUAL_MODES = ("pcd", "tracking")
PANEL_MODE_SINGLE = "single"
PANEL_MODE_SIDE_BY_SIDE = "side-by-side"
PANEL_MODES = (PANEL_MODE_SINGLE, PANEL_MODE_SIDE_BY_SIDE)
TABLE_WORLD_FRAME_KIND = "table_world_z0"
CAMERA_COLOR_FRAME = "camera_color_frame"
DEFAULT_TABLE_Z_OVERLAY_THRESHOLDS_M = (0.005, 0.010, 0.020, 0.030)
TABLE_Z_ABOVE_DIRECTION_POSITIVE = "positive"
TABLE_Z_ABOVE_DIRECTION_NEGATIVE = "negative"
TABLE_Z_ABOVE_DIRECTIONS = (
    TABLE_Z_ABOVE_DIRECTION_POSITIVE,
    TABLE_Z_ABOVE_DIRECTION_NEGATIVE,
)
DEFAULT_TABLE_Z_ABOVE_DIRECTION = TABLE_Z_ABOVE_DIRECTION_NEGATIVE
TRACKING_BACKGROUND_MASK_TARGET_UNION = "target-union"
TRACKING_BACKGROUND_MASK_RGB = "rgb"
TRACKING_BACKGROUND_MASK_MODES = (
    TRACKING_BACKGROUND_MASK_TARGET_UNION,
    TRACKING_BACKGROUND_MASK_RGB,
)
TRACKER_MARKER_GATE_PCD_FILTER_RESIDUAL_TABLE_Z = "pcd_filter_residual_table_z"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _table_z_clearance_m(
    points_xyz: np.ndarray,
    *,
    table_z_m: float,
    above_direction: str,
) -> np.ndarray:
    points = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    direction = str(above_direction)
    if direction == TABLE_Z_ABOVE_DIRECTION_POSITIVE:
        return np.ascontiguousarray(points[:, 2] - np.float32(table_z_m), dtype=np.float32)
    if direction == TABLE_Z_ABOVE_DIRECTION_NEGATIVE:
        return np.ascontiguousarray(np.float32(table_z_m) - points[:, 2], dtype=np.float32)
    raise RuntimeError(f"table_z_above_direction must be one of {TABLE_Z_ABOVE_DIRECTIONS}")


def _read_frames(path: Path) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            frames.append(json.loads(line))
    return frames


def _read_input_frames(*, capture_dir: Path, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    timeline = metadata.get("input_rgb_timeline")
    if not timeline:
        return []
    path = _resolve_capture_path(capture_dir, str(timeline))
    if not path.is_file():
        return []
    return _read_frames(path)


def _resolve_capture_path(capture_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else capture_dir / path


def _trajectory_index(capture_dir: Path) -> dict[int, Path]:
    by_seq: dict[int, Path] = {}
    for path in sorted((capture_dir / "query_trajectory").glob("*.npz")):
        try:
            seq = int(path.stem)
        except ValueError:
            continue
        by_seq[seq] = path
    return by_seq


def _trajectory_path_for_frame(
    *,
    capture_dir: Path,
    frame: dict[str, Any],
    trajectory_by_seq: dict[int, Path],
) -> Path | None:
    exact = _resolve_capture_path(capture_dir, str(frame["query_trajectory_path"]))
    if exact.is_file():
        return exact
    seq = int(frame["seq"])
    return trajectory_by_seq.get(seq)


def _read_rgb_frame_bgr(*, capture_dir: Path, frame: dict[str, Any], width: int, height: int) -> np.ndarray:
    if "rgb_path" not in frame:
        raise RuntimeError("tracking render requires rgb_path in frames.jsonl; rerun headless capture")
    rgb_path = _resolve_capture_path(capture_dir, str(frame["rgb_path"]))
    image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read RGB frame: {rgb_path}")
    if image.shape[:2] != (int(height), int(width)):
        image = cv2.resize(image, (int(width), int(height)), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(image, dtype=np.uint8)


def _read_input_rgb_frame_bgr(
    *,
    capture_dir: Path,
    input_frame: dict[str, Any] | None,
    width: int,
    height: int,
) -> np.ndarray | None:
    resolved = _input_rgb_path_for_frame(capture_dir=capture_dir, input_frame=input_frame)
    if resolved is None:
        return None
    rgb_path, _ = resolved
    image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    if image.shape[:2] != (int(height), int(width)):
        image = cv2.resize(image, (int(width), int(height)), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(image, dtype=np.uint8)


def _input_rgb_path_for_frame(
    *,
    capture_dir: Path,
    input_frame: dict[str, Any] | None,
) -> tuple[Path, str] | None:
    if input_frame is None or "seq" not in input_frame:
        return None
    if "input_rgb_path" in input_frame:
        source = str(input_frame["input_rgb_path"])
    elif "rgb_path" in input_frame:
        source = str(input_frame["rgb_path"])
    else:
        source = f"input_rgb/{int(input_frame['seq']):06d}.png"
    return _resolve_capture_path(capture_dir, source), source


def _as_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return result


def _as_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result


def _npz_scalar_int(payload: Any, key: str) -> int | None:
    if key not in payload.files:
        return None
    values = np.asarray(payload[key]).reshape(-1)
    if len(values) == 0:
        return None
    return _as_int_or_none(values[0])


def _npz_scalar_str(payload: Any, key: str) -> str | None:
    if key not in payload.files:
        return None
    values = np.asarray(payload[key]).reshape(-1)
    if len(values) == 0:
        return None
    return str(values[0])


def _marker_residual_audit_counts(
    *,
    frame: dict[str, Any],
    trajectory_path: Path | None,
) -> tuple[int, int] | None:
    checked = _as_int_or_none(frame.get("marker_residual_checked_count"))
    violations = _as_int_or_none(frame.get("marker_residual_violation_count"))
    if checked is not None and violations is not None:
        gate = frame.get("marker_residual_gate")
        if gate is not None:
            if str(gate) != TRACKER_MARKER_GATE_PCD_FILTER_RESIDUAL_TABLE_Z:
                return None
            return max(0, int(checked)), max(0, int(violations))

    if trajectory_path is None or not trajectory_path.is_file():
        return None
    with np.load(trajectory_path, allow_pickle=False) as payload:
        checked = _npz_scalar_int(payload, "marker_residual_checked_count")
        violations = _npz_scalar_int(payload, "marker_residual_violation_count")
        gate = _npz_scalar_str(payload, "marker_residual_gate")
    if checked is None or violations is None:
        return None
    if gate != TRACKER_MARKER_GATE_PCD_FILTER_RESIDUAL_TABLE_Z:
        return None
    return max(0, int(checked)), max(0, int(violations))


def _pair_process_done_time(row: dict[str, Any]) -> float | None:
    value = _as_float_or_none(row.get("pair_process_done_perf_s"))
    if value is not None:
        return value
    value = _as_float_or_none(row.get("pipeline_done_perf_s"))
    if value is not None:
        return value
    return _as_float_or_none(row.get("process_done_perf_s"))


def _latest_input_frame_for_paired_row(
    *,
    input_frames: list[dict[str, Any]],
    paired_row: dict[str, Any],
) -> dict[str, Any] | None:
    if not input_frames:
        return None
    paired_time = _pair_process_done_time(paired_row)
    if paired_time is None:
        paired_time = _as_float_or_none(paired_row.get("receive_perf_s"))
    timed_rows = [
        (index, _as_float_or_none(row.get("receive_perf_s")), row)
        for index, row in enumerate(input_frames)
    ]
    if paired_time is not None:
        eligible = [
            (index, receive_time, row)
            for index, receive_time, row in timed_rows
            if receive_time is not None and receive_time <= paired_time
        ]
        if eligible:
            return max(eligible, key=lambda item: (float(item[1]), int(item[0])))[2]
        paired_seq = int(paired_row.get("seq", 0))
        same_seq_untimed = [
            (index, row)
            for index, receive_time, row in timed_rows
            if receive_time is None and int(row.get("seq", -1)) == paired_seq
        ]
        if same_seq_untimed:
            return max(same_seq_untimed, key=lambda item: int(item[0]))[1]
        return None

    paired_seq = int(paired_row.get("seq", 0))
    sequenced = [(index, int(row["seq"]), row) for index, row in enumerate(input_frames) if "seq" in row]
    eligible_seq = [(index, seq, row) for index, seq, row in sequenced if seq <= paired_seq]
    if eligible_seq:
        return max(eligible_seq, key=lambda item: (int(item[1]), int(item[0])))[2]
    return sequenced[0][2] if sequenced else input_frames[-1]


def _read_target_union_mask(
    *,
    capture_dir: Path,
    frame: dict[str, Any],
    width: int,
    height: int,
) -> np.ndarray:
    if "mask_path" not in frame:
        raise RuntimeError(
            "tracking background target-union requires mask_path in frames.jsonl; "
            "rerun headless capture"
        )
    mask_path = _resolve_capture_path(capture_dir, str(frame["mask_path"]))
    if not mask_path.is_file():
        raise RuntimeError(f"tracking background target-union mask file missing: {mask_path}")
    with np.load(mask_path, allow_pickle=False) as payload:
        missing = [name for name in ("object_mask", "controller_mask") if name not in payload.files]
        if missing:
            raise RuntimeError(
                "tracking background target-union mask payload missing "
                + ", ".join(missing)
                + f": {mask_path}"
            )
        object_mask = np.asarray(payload["object_mask"], dtype=bool)
        controller_mask = np.asarray(payload["controller_mask"], dtype=bool)
    expected_shape = (int(height), int(width))
    if object_mask.shape != expected_shape:
        raise RuntimeError(
            f"object_mask shape {tuple(object_mask.shape)} does not match render shape "
            f"{expected_shape}: {mask_path}"
        )
    if controller_mask.shape != expected_shape:
        raise RuntimeError(
            f"controller_mask shape {tuple(controller_mask.shape)} does not match render shape "
            f"{expected_shape}: {mask_path}"
        )
    return np.ascontiguousarray(np.logical_or(object_mask, controller_mask), dtype=bool)


def _apply_tracking_background_mask(image_bgr: np.ndarray, target_union_mask: np.ndarray) -> int:
    mask = np.asarray(target_union_mask, dtype=bool)
    if mask.ndim != 2:
        raise RuntimeError(f"tracking background mask must be 2D, got shape {tuple(mask.shape)}")
    if image_bgr.shape[:2] != mask.shape:
        raise RuntimeError(
            f"tracking background mask shape {tuple(mask.shape)} does not match image shape "
            f"{tuple(image_bgr.shape[:2])}"
        )
    image_bgr[~mask] = 0
    return int(np.count_nonzero(mask))


def _transform_world_points_to_camera(points_xyz: np.ndarray, camera_to_world_c2w: Any) -> np.ndarray:
    points = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    c2w = np.asarray(camera_to_world_c2w, dtype=np.float32)
    if c2w.shape != (4, 4):
        raise RuntimeError(f"camera_to_world_c2w must be 4x4, got {c2w.shape}")
    if len(points) == 0:
        return np.ascontiguousarray(points, dtype=np.float32)
    w2c = np.linalg.inv(c2w.astype(np.float64)).astype(np.float32)
    homogeneous = np.concatenate([points, np.ones((len(points), 1), dtype=np.float32)], axis=1)
    camera_points = (w2c @ homogeneous.T).T[:, :3]
    return np.ascontiguousarray(camera_points, dtype=np.float32)


def _project_points(
    points_xyz: np.ndarray,
    intrinsics: dict[str, Any],
    *,
    width: int,
    height: int,
    coordinate_frame: str = CAMERA_COLOR_FRAME,
    camera_to_world_c2w: Any | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    if points.size == 0:
        return np.empty((0, 2), dtype=np.int32), np.empty((0,), dtype=bool)
    if str(coordinate_frame) == TABLE_WORLD_FRAME_KIND:
        if camera_to_world_c2w is None:
            raise RuntimeError("table_world_z0 projection requires camera_to_world_c2w in capture metadata")
        points = _transform_world_points_to_camera(points, camera_to_world_c2w)
    z = points[:, 2]
    valid = np.isfinite(points).all(axis=1) & (z > np.float32(1e-6))
    fx = np.float32(intrinsics["fx"])
    fy = np.float32(intrinsics["fy"])
    cx = np.float32(intrinsics["cx"])
    cy = np.float32(intrinsics["cy"])
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
    coordinate_frame: str = CAMERA_COLOR_FRAME,
    camera_to_world_c2w: Any | None = None,
) -> int:
    height, width = image_bgr.shape[:2]
    points = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    colors = np.asarray(colors_rgb, dtype=np.uint8).reshape(-1, 3)
    if len(points) == 0:
        return 0
    if int(max_points) > 0 and len(points) > int(max_points):
        indices = np.linspace(0, len(points) - 1, int(max_points), dtype=np.int64)
        points = points[indices]
        colors = colors[indices]
    order = np.argsort(points[:, 2])[::-1]
    points = points[order]
    colors = colors[order]
    uv, valid = _project_points(
        points,
        intrinsics,
        width=width,
        height=height,
        coordinate_frame=str(coordinate_frame),
        camera_to_world_c2w=camera_to_world_c2w,
    )
    uv = uv[valid]
    colors_bgr = colors[valid][:, ::-1]
    if len(uv) == 0:
        return 0
    radius = max(0, int(point_size) // 2)
    if radius <= 0:
        image_bgr[uv[:, 1], uv[:, 0]] = colors_bgr
    else:
        for dy in range(-radius, radius + 1):
            yy = np.clip(uv[:, 1] + dy, 0, height - 1)
            for dx in range(-radius, radius + 1):
                xx = np.clip(uv[:, 0] + dx, 0, width - 1)
                image_bgr[yy, xx] = colors_bgr
    return int(len(uv))


def _draw_query_points(
    image_bgr: np.ndarray,
    trajectory_path: Path,
    *,
    marker_radius: int,
) -> tuple[int, int, int, int, int]:
    if not trajectory_path.is_file():
        return 0, 0, 0, 0, 0
    payload = np.load(trajectory_path, allow_pickle=False)
    tracks_yx = np.asarray(payload["tracks_yx"], dtype=np.float32).reshape(-1, 2)
    query_indices = np.asarray(payload["query_indices"], dtype=np.int64).reshape(-1)
    if "query_is_object" in payload.files:
        query_is_object = np.asarray(payload["query_is_object"], dtype=bool).reshape(-1)
    else:
        query_is_object = np.ones((len(tracks_yx),), dtype=bool)
    if "query_is_controller" in payload.files:
        query_is_controller = np.asarray(payload["query_is_controller"], dtype=bool).reshape(-1)
    else:
        query_is_controller = np.zeros((len(tracks_yx),), dtype=bool)
    if "marker_rgb_u8" in payload.files:
        marker_rgb_u8 = np.asarray(payload["marker_rgb_u8"], dtype=np.uint8).reshape(-1, 3)
    elif "query_rgb_u8" in payload.files:
        query_rgb_u8 = np.asarray(payload["query_rgb_u8"], dtype=np.uint8).reshape(-1, 3)
        marker_rgb_u8 = np.zeros((len(query_indices), 3), dtype=np.uint8)
        valid_indices = (query_indices >= 0) & (query_indices < len(query_rgb_u8))
        marker_rgb_u8[valid_indices] = query_rgb_u8[query_indices[valid_indices]]
    else:
        query_count = int(payload["query_count"][0]) if "query_count" in payload.files else None
        marker_rgb_u8 = query_rainbow_colors_for_indices(query_indices, query_count=query_count)
    if "visibility" in payload.files:
        visibility = np.asarray(payload["visibility"], dtype=np.float32).reshape(-1)
    else:
        visibility = np.ones((len(tracks_yx),), dtype=np.float32)
    if "query_controller_instance_id" in payload.files:
        query_controller_instance_id = np.asarray(payload["query_controller_instance_id"], dtype=np.int64).reshape(-1)
    else:
        query_controller_instance_id = np.zeros((len(tracks_yx),), dtype=np.int64)
    count = min(
        len(tracks_yx),
        len(query_indices),
        len(query_is_object),
        len(query_is_controller),
        len(marker_rgb_u8),
        len(visibility),
        len(query_controller_instance_id),
    )
    if count == 0:
        return 0, 0, 0, 0, 0
    height, width = image_bgr.shape[:2]
    y = np.rint(tracks_yx[:count, 0]).astype(np.int32)
    x = np.rint(tracks_yx[:count, 1]).astype(np.int32)
    valid = (
        np.isfinite(tracks_yx[:count]).all(axis=1)
        & (visibility[:count] > np.float32(0.5))
        & (x >= 0)
        & (x < int(width))
        & (y >= 0)
        & (y < int(height))
    )
    uv = np.stack([x, y], axis=1)
    visible_uv = uv[valid]
    visible_is_object = query_is_object[:count][valid]
    visible_is_controller = query_is_controller[:count][valid]
    visible_controller_instance_id = query_controller_instance_id[:count][valid]
    visible_colors_bgr = marker_rgb_u8[:count][valid][:, ::-1]
    radius = max(1, int(marker_radius))

    object_count = 0
    controller_count = 0
    object_mask = visible_is_object & ~visible_is_controller
    controller_mask = visible_is_controller
    other_mask = ~(object_mask | controller_mask)
    if np.any(other_mask):
        object_mask = object_mask | other_mask
    object_uv = visible_uv[object_mask]
    object_colors = visible_colors_bgr[object_mask]
    controller_uv = visible_uv[controller_mask]
    controller_colors = visible_colors_bgr[controller_mask]
    for point_uv, color_bgr in zip(object_uv, object_colors):
        cv2.circle(
            image_bgr,
            (int(point_uv[0]), int(point_uv[1])),
            radius,
            tuple(int(value) for value in color_bgr),
            -1,
            cv2.LINE_AA,
        )
        object_count += 1
    for point_uv, color_bgr in zip(controller_uv, controller_colors):
        cv2.circle(
            image_bgr,
            (int(point_uv[0]), int(point_uv[1])),
            radius,
            tuple(int(value) for value in color_bgr),
            -1,
            cv2.LINE_AA,
        )
        controller_count += 1
    hand_a_count = int(np.count_nonzero(visible_controller_instance_id == 1))
    hand_b_count = int(np.count_nonzero(visible_controller_instance_id == 2))
    return int(object_count + controller_count), int(object_count), int(controller_count), hand_a_count, hand_b_count


def _fit_bool_array(values: np.ndarray, length: int, *, fill: bool = False) -> np.ndarray:
    output = np.full((max(0, int(length)),), bool(fill), dtype=bool)
    arr = np.asarray(values, dtype=bool).reshape(-1)
    count = min(len(arr), len(output))
    if count:
        output[:count] = arr[:count]
    return output


def _fit_int_array(values: np.ndarray, length: int, *, fill: int = 0) -> np.ndarray:
    output = np.full((max(0, int(length)),), int(fill), dtype=np.int64)
    arr = np.asarray(values, dtype=np.int64).reshape(-1)
    count = min(len(arr), len(output))
    if count:
        output[:count] = arr[:count]
    return output


def _query_remaining_counts(
    *,
    trajectory_path: Path | None,
    fallback_query_count: int,
    fallback_object_count: int,
    fallback_controller_count: int,
    fallback_hand_a_count: int,
    fallback_hand_b_count: int,
) -> dict[str, int]:
    fallback = {
        "all": max(0, int(fallback_query_count)),
        "query_count": max(0, int(fallback_query_count)),
        "object": max(0, int(fallback_object_count)),
        "controller": max(0, int(fallback_controller_count)),
        "hand_a": max(0, int(fallback_hand_a_count)),
        "hand_b": max(0, int(fallback_hand_b_count)),
    }
    if trajectory_path is None or not trajectory_path.is_file():
        return fallback
    with np.load(trajectory_path, allow_pickle=False) as payload:
        scalar_remaining = _npz_scalar_int(payload, "remaining_query_count")
        if scalar_remaining is not None:
            query_count = _npz_scalar_int(payload, "query_count")
            return {
                "all": max(0, int(scalar_remaining)),
                "query_count": max(0, int(query_count if query_count is not None else scalar_remaining)),
                "object": max(0, int(_npz_scalar_int(payload, "remaining_object_query_count") or 0)),
                "controller": max(0, int(_npz_scalar_int(payload, "remaining_controller_query_count") or 0)),
                "hand_a": max(0, int(_npz_scalar_int(payload, "remaining_hand_a_query_count") or 0)),
                "hand_b": max(0, int(_npz_scalar_int(payload, "remaining_hand_b_query_count") or 0)),
            }
        query_count = _npz_scalar_int(payload, "query_count")
        if query_count is None:
            query_count = int(fallback["all"])
        query_count = max(0, int(query_count))
        if "query_alive_mask" in payload.files:
            alive = _fit_bool_array(payload["query_alive_mask"], query_count)
        else:
            alive = np.ones((query_count,), dtype=bool)
        remaining_all = int(np.count_nonzero(alive))
        if "query_all_controller_instance_id" in payload.files:
            instance_id = _fit_int_array(payload["query_all_controller_instance_id"], query_count)
            hand_a = alive & (instance_id == 1)
            hand_b = alive & (instance_id == 2)
            controller = alive & ((instance_id == 1) | (instance_id == 2))
            if "query_all_target_id" in payload.files:
                target_id = _fit_int_array(payload["query_all_target_id"], query_count)
                obj = alive & (target_id == 2)
                controller = controller | (alive & ((target_id == 1) | (target_id == 3)))
            else:
                obj = np.zeros((query_count,), dtype=bool)
            return {
                "all": remaining_all,
                "query_count": query_count,
                "object": int(np.count_nonzero(obj)),
                "controller": int(np.count_nonzero(controller)),
                "hand_a": int(np.count_nonzero(hand_a)),
                "hand_b": int(np.count_nonzero(hand_b)),
            }
    if query_count > int(fallback["all"]):
        fallback["all"] = query_count
        fallback["query_count"] = query_count
    return fallback


def _read_query_panel_payload(trajectory_path: Path | None) -> dict[str, np.ndarray] | None:
    if trajectory_path is None or not trajectory_path.is_file():
        return None
    with np.load(trajectory_path, allow_pickle=False) as payload:
        tracks_yx = np.asarray(payload["tracks_yx"], dtype=np.float32).reshape(-1, 2)
        query_indices = np.asarray(payload["query_indices"], dtype=np.int64).reshape(-1)
        if "query_is_object" in payload.files:
            query_is_object = np.asarray(payload["query_is_object"], dtype=bool).reshape(-1)
        else:
            query_is_object = np.ones((len(tracks_yx),), dtype=bool)
        if "query_is_controller" in payload.files:
            query_is_controller = np.asarray(payload["query_is_controller"], dtype=bool).reshape(-1)
        else:
            query_is_controller = np.zeros((len(tracks_yx),), dtype=bool)
        if "marker_rgb_u8" in payload.files:
            marker_rgb_u8 = np.asarray(payload["marker_rgb_u8"], dtype=np.uint8).reshape(-1, 3)
        elif "query_rgb_u8" in payload.files:
            query_rgb_u8 = np.asarray(payload["query_rgb_u8"], dtype=np.uint8).reshape(-1, 3)
            marker_rgb_u8 = np.zeros((len(query_indices), 3), dtype=np.uint8)
            valid_indices = (query_indices >= 0) & (query_indices < len(query_rgb_u8))
            marker_rgb_u8[valid_indices] = query_rgb_u8[query_indices[valid_indices]]
        else:
            query_count = int(payload["query_count"][0]) if "query_count" in payload.files else None
            marker_rgb_u8 = query_rainbow_colors_for_indices(query_indices, query_count=query_count)
        if "visibility" in payload.files:
            visibility = np.asarray(payload["visibility"], dtype=np.float32).reshape(-1)
        else:
            visibility = np.ones((len(tracks_yx),), dtype=np.float32)
        if "query_controller_instance_id" in payload.files:
            query_controller_instance_id = np.asarray(
                payload["query_controller_instance_id"],
                dtype=np.int64,
            ).reshape(-1)
        else:
            query_controller_instance_id = np.zeros((len(tracks_yx),), dtype=np.int64)
    count = min(
        len(tracks_yx),
        len(visibility),
        len(marker_rgb_u8),
        len(query_is_object),
        len(query_is_controller),
        len(query_controller_instance_id),
    )
    return {
        "tracks_yx": np.ascontiguousarray(tracks_yx[:count], dtype=np.float32),
        "visibility": np.ascontiguousarray(visibility[:count], dtype=np.float32),
        "marker_rgb_u8": np.ascontiguousarray(marker_rgb_u8[:count], dtype=np.uint8),
        "query_is_object": np.ascontiguousarray(query_is_object[:count], dtype=bool),
        "query_is_controller": np.ascontiguousarray(query_is_controller[:count], dtype=bool),
        "query_controller_instance_id": np.ascontiguousarray(
            query_controller_instance_id[:count],
            dtype=np.int64,
        ),
    }


def _empty_query_panel_payload() -> dict[str, np.ndarray]:
    return {
        "tracks_yx": np.empty((0, 2), dtype=np.float32),
        "visibility": np.empty((0,), dtype=np.float32),
        "marker_rgb_u8": np.empty((0, 3), dtype=np.uint8),
        "query_is_object": np.empty((0,), dtype=bool),
        "query_is_controller": np.empty((0,), dtype=bool),
        "query_controller_instance_id": np.empty((0,), dtype=np.int64),
    }


def _stack_pcd_points(pcd: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    object_xyz = np.asarray(pcd["object_xyz_m"], dtype=np.float32).reshape(-1, 3)
    object_rgb = np.asarray(pcd["object_rgb_u8"], dtype=np.uint8).reshape(-1, 3)
    controller_xyz = np.asarray(pcd["controller_xyz_m"], dtype=np.float32).reshape(-1, 3)
    controller_rgb = np.asarray(pcd["controller_rgb_u8"], dtype=np.uint8).reshape(-1, 3)
    object_labels = np.full((len(object_xyz),), "object", dtype=object)
    controller_labels = np.full((len(controller_xyz),), "controller", dtype=object)
    if len(object_xyz) == 0 and len(controller_xyz) == 0:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.uint8),
            np.empty((0,), dtype=object),
        )
    return (
        np.concatenate([object_xyz, controller_xyz], axis=0),
        np.concatenate([object_rgb, controller_rgb], axis=0),
        np.concatenate([object_labels, controller_labels], axis=0),
    )


def render_table_z_filter_overlay_sweep(
    *,
    capture_dir: Path,
    output_dir: Path,
    fps: float,
    thresholds_m: tuple[float, ...] = DEFAULT_TABLE_Z_OVERLAY_THRESHOLDS_M,
    point_size: int = 2,
    max_render_points: int = 0,
    table_z_above_direction: str | None = None,
) -> dict[str, Any]:
    capture_dir = Path(capture_dir).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = _read_json(capture_dir / "metadata.json")
    frames = _read_frames(capture_dir / "frames.jsonl")
    if not frames:
        raise RuntimeError(f"no saved frames found in {capture_dir / 'frames.jsonl'}")
    width = int(metadata["width"])
    height = int(metadata["height"])
    intrinsics = dict(metadata["intrinsics"])
    pcd_coordinate_frame = str(
        metadata.get("pcd_coordinate_frame")
        or metadata.get("coordinate_frame")
        or CAMERA_COLOR_FRAME
    )
    if pcd_coordinate_frame != TABLE_WORLD_FRAME_KIND:
        raise RuntimeError("table-Z overlay sweep requires table_world_z0 PCD capture")
    camera_to_world_c2w = metadata.get("camera_to_world_c2w")
    if camera_to_world_c2w is None:
        raise RuntimeError("table-Z overlay sweep requires camera_to_world_c2w in capture metadata")
    table_z_m = float(metadata.get("table_z_m", 0.0) or 0.0)
    direction = str(
        table_z_above_direction
        or metadata.get("table_z_above_direction")
        or DEFAULT_TABLE_Z_ABOVE_DIRECTION
    )
    if direction not in TABLE_Z_ABOVE_DIRECTIONS:
        raise RuntimeError(f"table_z_above_direction must be one of {TABLE_Z_ABOVE_DIRECTIONS}")

    threshold_summaries: list[dict[str, Any]] = []
    for threshold_m in tuple(float(value) for value in thresholds_m):
        suffix = f"{threshold_m:.3f}".replace(".", "p")
        output = output_dir / f"table_z_filter_threshold_{suffix}m.mp4"
        writer = cv2.VideoWriter(
            str(output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps),
            (width * 3, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"failed to open video writer: {output}")
        frame_rows: list[dict[str, int]] = []
        try:
            for frame in frames:
                rgb = _read_rgb_frame_bgr(capture_dir=capture_dir, frame=frame, width=width, height=height)
                before = rgb.copy()
                after = rgb.copy()
                removed_panel = rgb.copy()
                pcd_path = _resolve_capture_path(capture_dir, str(frame["pcd_path"]))
                with np.load(pcd_path, allow_pickle=False) as pcd:
                    points, colors, labels = _stack_pcd_points(pcd)
                finite = np.isfinite(points).all(axis=1) if len(points) else np.zeros((0,), dtype=bool)
                clearance = _table_z_clearance_m(
                    points,
                    table_z_m=table_z_m,
                    above_direction=direction,
                )
                removed_mask = finite & (clearance <= np.float32(threshold_m))
                kept_mask = ~removed_mask
                removed_colors = np.tile(np.array([[255, 0, 0]], dtype=np.uint8), (int(np.count_nonzero(removed_mask)), 1))
                _draw_projected_points(
                    before,
                    points,
                    colors,
                    intrinsics,
                    point_size=int(point_size),
                    max_points=int(max_render_points),
                    coordinate_frame=pcd_coordinate_frame,
                    camera_to_world_c2w=camera_to_world_c2w,
                )
                _draw_projected_points(
                    after,
                    points[kept_mask],
                    colors[kept_mask],
                    intrinsics,
                    point_size=int(point_size),
                    max_points=int(max_render_points),
                    coordinate_frame=pcd_coordinate_frame,
                    camera_to_world_c2w=camera_to_world_c2w,
                )
                _draw_projected_points(
                    removed_panel,
                    points[removed_mask],
                    removed_colors,
                    intrinsics,
                    point_size=max(2, int(point_size)),
                    max_points=int(max_render_points),
                    coordinate_frame=pcd_coordinate_frame,
                    camera_to_world_c2w=camera_to_world_c2w,
                )
                writer.write(np.concatenate([before, after, removed_panel], axis=1))
                object_removed = int(np.count_nonzero(removed_mask & (labels == "object")))
                controller_removed = int(np.count_nonzero(removed_mask & (labels == "controller")))
                frame_rows.append(
                    {
                        "seq": int(frame["seq"]),
                        "input_points": int(len(points)),
                        "kept_points": int(np.count_nonzero(kept_mask)),
                        "removed_points": int(np.count_nonzero(removed_mask)),
                        "object_removed_points": object_removed,
                        "controller_removed_points": controller_removed,
                    }
                )
        finally:
            writer.release()
        threshold_summaries.append(
            {
                "threshold_m": float(threshold_m),
                "output": str(output.resolve()),
                "frame_count": int(len(frame_rows)),
                "input_total": int(sum(row["input_points"] for row in frame_rows)),
                "kept_total": int(sum(row["kept_points"] for row in frame_rows)),
                "removed_total": int(sum(row["removed_points"] for row in frame_rows)),
                "frames": frame_rows,
            }
        )

    summary = {
        "capture_dir": str(capture_dir),
        "output_dir": str(output_dir.resolve()),
        "fps": float(fps),
        "frame_count": int(len(frames)),
        "image_size": [int(width), int(height)],
        "pcd_coordinate_frame": pcd_coordinate_frame,
        "table_z_m": table_z_m,
        "table_z_above_direction": direction,
        "thresholds_m": [float(value) for value in thresholds_m],
        "thresholds": threshold_summaries,
        "overlay_columns": ["before", "after", "removed_red"],
    }
    summary_path = output_dir / "table_z_filter_overlay_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def render_capture_to_video(
    *,
    capture_dir: Path,
    output: Path,
    fps: float,
    point_size: int = 2,
    max_render_points: int = 0,
    query_point_radius: int = 3,
    demo_visual_mode: str = "tracking",
    tracking_background_mask: str = TRACKING_BACKGROUND_MASK_TARGET_UNION,
    panel_mode: str = PANEL_MODE_SINGLE,
) -> dict[str, Any]:
    capture_dir = Path(capture_dir).resolve()
    metadata = _read_json(capture_dir / "metadata.json")
    frames = _read_frames(capture_dir / "frames.jsonl")
    if not frames:
        raise RuntimeError(f"no saved frames found in {capture_dir / 'frames.jsonl'}")
    width = int(metadata["width"])
    height = int(metadata["height"])
    intrinsics = dict(metadata["intrinsics"])
    pcd_coordinate_frame = str(
        metadata.get("pcd_coordinate_frame")
        or metadata.get("coordinate_frame")
        or CAMERA_COLOR_FRAME
    )
    camera_to_world_c2w = metadata.get("camera_to_world_c2w")
    if str(demo_visual_mode) not in DEMO_VISUAL_MODES:
        raise ValueError(f"demo_visual_mode must be one of {DEMO_VISUAL_MODES}")
    if str(tracking_background_mask) not in TRACKING_BACKGROUND_MASK_MODES:
        raise ValueError(f"tracking_background_mask must be one of {TRACKING_BACKGROUND_MASK_MODES}")
    if str(panel_mode) not in PANEL_MODES:
        raise ValueError(f"panel_mode must be one of {PANEL_MODES}")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output_width = width * 3 if str(panel_mode) == PANEL_MODE_SIDE_BY_SIDE else width
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (output_width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {output}")
    rendered_counts: list[dict[str, Any]] = []
    trajectory_by_seq = _trajectory_index(capture_dir)
    input_frames = _read_input_frames(capture_dir=capture_dir, metadata=metadata)
    if str(panel_mode) == PANEL_MODE_SIDE_BY_SIDE:
        has_input_timeline = bool(input_frames)
        left_rgb_policy = "latest_input_rgb" if has_input_timeline else "same_seq_fallback"
        sync_policy = (
            "latest_receive_perf_s_lte_pair_process_done_perf_s"
            if has_input_timeline
            else "paired_seq_fallback"
        )
    else:
        left_rgb_policy = "paired_rgb"
        sync_policy = "paired_seq"
    missing_query_frames = 0
    missing_rgb_frames = 0
    tracking_marker_residual_checked_total = 0
    tracking_marker_residual_violation_total = 0
    tracking_marker_residual_violation_frames = 0
    tracking_marker_residual_audit_missing_frames = 0
    try:
        for frame in frames:
            image = np.zeros((height, width, 3), dtype=np.uint8)
            controller_count = object_count = 0
            query_count = query_object_count = query_controller_count = 0
            query_hand_a_count = query_hand_b_count = 0
            remaining_query_counts = {
                "all": 0,
                "query_count": 0,
                "object": 0,
                "controller": 0,
                "hand_a": 0,
                "hand_b": 0,
            }
            tracking_background_mask_pixels = 0
            marker_residual_checked_count = 0
            marker_residual_violation_count = 0
            marker_residual_audit_present = 0
            query_path = None
            if str(panel_mode) == PANEL_MODE_SIDE_BY_SIDE:
                paired_rgb = _read_rgb_frame_bgr(capture_dir=capture_dir, frame=frame, width=width, height=height)
                input_row = _latest_input_frame_for_paired_row(input_frames=input_frames, paired_row=frame)
                resolved_input_rgb_path = _input_rgb_path_for_frame(capture_dir=capture_dir, input_frame=input_row)
                input_rgb_source_path = resolved_input_rgb_path[1] if resolved_input_rgb_path is not None else None
                input_rgb = _read_input_rgb_frame_bgr(
                    capture_dir=capture_dir,
                    input_frame=input_row,
                    width=width,
                    height=height,
                )
                if input_rgb is None:
                    input_rgb = paired_rgb.copy()
                    missing_rgb_frames += 1
                    rgb_seq = int(frame["seq"])
                    input_time_s = _as_float_or_none(frame.get("source_timestamp_s"))
                    if input_time_s is None:
                        input_time_s = _as_float_or_none(frame.get("receive_perf_s"))
                    input_rgb_source_path = str(frame.get("rgb_path") or f"rgb/{int(frame['seq']):06d}.png")
                else:
                    rgb_seq = int(input_row["seq"]) if input_row is not None else int(frame["seq"])
                    input_time_s = None
                    if input_row is not None:
                        input_time_s = _as_float_or_none(input_row.get("source_timestamp_s"))
                        if input_time_s is None:
                            input_time_s = _as_float_or_none(input_row.get("receive_perf_s"))

                pcd_path = _resolve_capture_path(capture_dir, str(frame["pcd_path"]))
                with np.load(pcd_path, allow_pickle=False) as pcd:
                    pcd_panel, pcd_counts = render_projected_pcd_panel(
                        width=width,
                        height=height,
                        intrinsics=intrinsics,
                        controller_xyz_m=pcd["controller_xyz_m"],
                        controller_rgb_u8=pcd["controller_rgb_u8"],
                        object_xyz_m=pcd["object_xyz_m"],
                        object_rgb_u8=pcd["object_rgb_u8"],
                        point_size=int(point_size),
                        max_render_points=int(max_render_points),
                        coordinate_frame=pcd_coordinate_frame,
                        camera_to_world_c2w=camera_to_world_c2w,
                    )
                controller_count = int(pcd_counts["controller_points"])
                object_count = int(pcd_counts["object_points"])

                tracking_image = paired_rgb.copy()
                if str(tracking_background_mask) == TRACKING_BACKGROUND_MASK_TARGET_UNION:
                    target_union_mask = _read_target_union_mask(
                        capture_dir=capture_dir,
                        frame=frame,
                        width=width,
                        height=height,
                    )
                    tracking_background_mask_pixels = _apply_tracking_background_mask(tracking_image, target_union_mask)
                query_path = _trajectory_path_for_frame(
                    capture_dir=capture_dir,
                    frame=frame,
                    trajectory_by_seq=trajectory_by_seq,
                )
                query_payload = _read_query_panel_payload(query_path)
                if query_payload is None:
                    missing_query_frames += 1
                    query_payload = _empty_query_panel_payload()
                tracking_panel, query_counts = render_tracking_overlay_panel(
                    image_bgr=tracking_image,
                    marker_radius=int(query_point_radius),
                    **query_payload,
                )
                query_count = int(query_counts["query_points"])
                query_object_count = int(query_counts["query_object_points"])
                query_controller_count = int(query_counts["query_controller_points"])
                query_hand_a_count = int(query_counts["query_hand_a_points"])
                query_hand_b_count = int(query_counts["query_hand_b_points"])
                remaining_query_counts = _query_remaining_counts(
                    trajectory_path=query_path,
                    fallback_query_count=query_count,
                    fallback_object_count=query_object_count,
                    fallback_controller_count=query_controller_count,
                    fallback_hand_a_count=query_hand_a_count,
                    fallback_hand_b_count=query_hand_b_count,
                )

                receive_time = _as_float_or_none(frame.get("receive_perf_s"))
                pair_process_done_time = _pair_process_done_time(frame)
                if receive_time is not None and pair_process_done_time is not None:
                    display_latency_ms = max(0.0, (pair_process_done_time - receive_time) * 1000.0)
                else:
                    display_latency_ms = _as_float_or_none(frame.get("pipeline_latency_ms")) or 0.0
                pipeline_latency_ms = _as_float_or_none(frame.get("pipeline_latency_ms"))
                if pipeline_latency_ms is None:
                    pipeline_latency_ms = display_latency_ms
                hud = SideBySidePanelHud(
                    rgb_seq=int(rgb_seq),
                    paired_seq=int(frame["seq"]),
                    input_time_s=input_time_s,
                    pipeline_latency_ms=float(pipeline_latency_ms),
                    display_latency_ms=float(display_latency_ms),
                    startup_hold_s=float(metadata.get("startup_hold_s", 0.0) or 0.0),
                    filter_preset=str(frame.get("filter_preset") or metadata.get("pcd_filter_preset") or "unknown"),
                    marker_count=int(frame.get("marker_count", query_count) or 0),
                    tracking_background=str(tracking_background_mask),
                    object_point_count=int(object_count),
                    controller_point_count=int(controller_count),
                    query_count=int(remaining_query_counts["query_count"]),
                    remaining_query_count=int(remaining_query_counts["all"]),
                    remaining_object_query_count=int(remaining_query_counts["object"]),
                    remaining_controller_query_count=int(remaining_query_counts["controller"]),
                    remaining_hand_a_query_count=int(remaining_query_counts["hand_a"]),
                    remaining_hand_b_query_count=int(remaining_query_counts["hand_b"]),
                )
                image = render_side_by_side_panel(
                    SideBySidePanelInputs(
                        rgb_image_bgr=input_rgb,
                        pcd_panel_bgr=pcd_panel,
                        tracking_panel_bgr=tracking_panel,
                        hud=hud,
                    ),
                    cell_size=(width, height),
                )
            elif str(demo_visual_mode) == "tracking":
                image = _read_rgb_frame_bgr(capture_dir=capture_dir, frame=frame, width=width, height=height)
                if str(tracking_background_mask) == TRACKING_BACKGROUND_MASK_TARGET_UNION:
                    target_union_mask = _read_target_union_mask(
                        capture_dir=capture_dir,
                        frame=frame,
                        width=width,
                        height=height,
                    )
                    tracking_background_mask_pixels = _apply_tracking_background_mask(image, target_union_mask)
                query_path = _trajectory_path_for_frame(
                    capture_dir=capture_dir,
                    frame=frame,
                    trajectory_by_seq=trajectory_by_seq,
                )
                if query_path is None or not query_path.is_file():
                    missing_query_frames += 1
                else:
                    (
                        query_count,
                        query_object_count,
                        query_controller_count,
                        query_hand_a_count,
                        query_hand_b_count,
                    ) = _draw_query_points(
                        image,
                        query_path,
                        marker_radius=int(query_point_radius),
                    )
                    remaining_query_counts = _query_remaining_counts(
                        trajectory_path=query_path,
                        fallback_query_count=query_count,
                        fallback_object_count=query_object_count,
                        fallback_controller_count=query_controller_count,
                        fallback_hand_a_count=query_hand_a_count,
                        fallback_hand_b_count=query_hand_b_count,
                    )
            else:
                pcd_path = _resolve_capture_path(capture_dir, str(frame["pcd_path"]))
                pcd = np.load(pcd_path, allow_pickle=False)
                controller_count = _draw_projected_points(
                    image,
                    pcd["controller_xyz_m"],
                    pcd["controller_rgb_u8"],
                    intrinsics,
                    point_size=int(point_size),
                    max_points=int(max_render_points),
                    coordinate_frame=pcd_coordinate_frame,
                    camera_to_world_c2w=camera_to_world_c2w,
                )
                object_count = _draw_projected_points(
                    image,
                    pcd["object_xyz_m"],
                    pcd["object_rgb_u8"],
                    intrinsics,
                    point_size=int(point_size),
                    max_points=int(max_render_points),
                    coordinate_frame=pcd_coordinate_frame,
                    camera_to_world_c2w=camera_to_world_c2w,
                )
            frame_renders_tracking_panel = str(panel_mode) == PANEL_MODE_SIDE_BY_SIDE or str(demo_visual_mode) == "tracking"
            if frame_renders_tracking_panel:
                audit_counts = _marker_residual_audit_counts(frame=frame, trajectory_path=query_path)
                if audit_counts is None:
                    tracking_marker_residual_audit_missing_frames += 1
                else:
                    marker_residual_audit_present = 1
                    marker_residual_checked_count = int(audit_counts[0])
                    marker_residual_violation_count = int(audit_counts[1])
                    tracking_marker_residual_checked_total += marker_residual_checked_count
                    tracking_marker_residual_violation_total += marker_residual_violation_count
                    if marker_residual_violation_count > 0:
                        tracking_marker_residual_violation_frames += 1
                    expected_marker_count = _as_int_or_none(frame.get("marker_count"))
                    if expected_marker_count is None:
                        expected_marker_count = int(query_count)
                    if marker_residual_checked_count < max(0, int(expected_marker_count)):
                        tracking_marker_residual_audit_missing_frames += 1
            writer.write(image)
            rendered_row = {
                "seq": int(frame["seq"]),
                "controller_points": int(controller_count),
                "object_points": int(object_count),
                "query_points": int(query_count),
                "query_object_points": int(query_object_count),
                "query_controller_points": int(query_controller_count),
                "query_hand_a_points": int(query_hand_a_count),
                "query_hand_b_points": int(query_hand_b_count),
                "query_initialized_points": int(remaining_query_counts["query_count"]),
                "remaining_query_points": int(remaining_query_counts["all"]),
                "remaining_query_object_points": int(remaining_query_counts["object"]),
                "remaining_query_controller_points": int(remaining_query_counts["controller"]),
                "remaining_query_hand_a_points": int(remaining_query_counts["hand_a"]),
                "remaining_query_hand_b_points": int(remaining_query_counts["hand_b"]),
                "tracking_background_mask_pixels": int(tracking_background_mask_pixels),
                "query_trajectory_exact": int(query_path is not None and query_path.is_file()),
                "marker_residual_audit_present": int(marker_residual_audit_present),
                "marker_residual_checked_count": int(marker_residual_checked_count),
                "marker_residual_violation_count": int(marker_residual_violation_count),
            }
            if str(panel_mode) == PANEL_MODE_SIDE_BY_SIDE:
                rendered_row.update(
                    {
                        "rgb_seq": int(rgb_seq),
                        "paired_seq": int(frame["seq"]),
                        "input_time_s": input_time_s,
                        "input_rgb_source_path": input_rgb_source_path,
                        "pipeline_latency_ms": float(pipeline_latency_ms),
                        "display_latency_ms": float(display_latency_ms),
                        "rgb_ahead_frames": int(
                            compute_rgb_ahead_frames(rgb_seq=int(rgb_seq), paired_seq=int(frame["seq"]))
                        ),
                    }
                )
            rendered_counts.append(rendered_row)
    finally:
        writer.release()
    renders_tracking_panel = str(panel_mode) == PANEL_MODE_SIDE_BY_SIDE or str(demo_visual_mode) == "tracking"
    tracking_background_mask_source = "none"
    if renders_tracking_panel:
        tracking_background_mask_source = (
            "object_mask|controller_mask"
            if str(tracking_background_mask) == TRACKING_BACKGROUND_MASK_TARGET_UNION
            else "full_rgb"
        )
    tracking_marker_residual_target_met = bool(
        renders_tracking_panel
        and tracking_marker_residual_audit_missing_frames == 0
        and tracking_marker_residual_violation_total == 0
    )
    summary = {
        "capture_dir": str(capture_dir),
        "output": str(output.resolve()),
        "fps": float(fps),
        "frame_count": int(len(frames)),
        "image_size": [int(width), int(height)],
        "saved_pcd_source": metadata.get("saved_pcd_source"),
        "pcd_coordinate_frame": pcd_coordinate_frame,
        "demo_visual_mode": str(demo_visual_mode),
        "panel_mode": str(panel_mode),
        "left_rgb_policy": left_rgb_policy,
        "input_rgb_frame_count": int(len(input_frames)),
        "missing_rgb_frames": int(missing_rgb_frames),
        "sync_policy": sync_policy,
        "tracking_background_mask": str(tracking_background_mask),
        "tracking_background_mask_source": tracking_background_mask_source,
        "tracking_background_mask_pixel_total": int(
            sum(item["tracking_background_mask_pixels"] for item in rendered_counts)
        ),
        "tracking_marker_residual_checked_total": int(tracking_marker_residual_checked_total),
        "tracking_marker_residual_violation_total": int(tracking_marker_residual_violation_total),
        "tracking_marker_residual_violation_frames": int(tracking_marker_residual_violation_frames),
        "tracking_marker_residual_audit_missing_frames": int(tracking_marker_residual_audit_missing_frames),
        "tracking_marker_residual_target_met": bool(tracking_marker_residual_target_met),
        "query_overlay": "phystwin_rgb_current_points_only" if renders_tracking_panel else "none",
        "query_color_mode": "phystwin_rainbow_identity" if renders_tracking_panel else "none",
        "query_match_policy": "exact_same_seq_only",
        "missing_query_frames": int(missing_query_frames),
        "query_count_totals": {
            "hand_a": int(sum(item["query_hand_a_points"] for item in rendered_counts)),
            "hand_b": int(sum(item["query_hand_b_points"] for item in rendered_counts)),
            "controller": int(sum(item["query_controller_points"] for item in rendered_counts)),
            "object": int(sum(item["query_object_points"] for item in rendered_counts)),
            "all": int(sum(item["query_points"] for item in rendered_counts)),
        },
        "remaining_query_count_totals": {
            "hand_a": int(sum(item["remaining_query_hand_a_points"] for item in rendered_counts)),
            "hand_b": int(sum(item["remaining_query_hand_b_points"] for item in rendered_counts)),
            "controller": int(sum(item["remaining_query_controller_points"] for item in rendered_counts)),
            "object": int(sum(item["remaining_query_object_points"] for item in rendered_counts)),
            "all": int(sum(item["remaining_query_points"] for item in rendered_counts)),
        },
        "rendered_counts": rendered_counts,
    }
    summary_path = output.with_suffix(".render_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if str(panel_mode) == PANEL_MODE_SIDE_BY_SIDE:
        panel_summary_path = output.with_suffix(".panel_summary.json")
        panel_summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render Demo 3.2 headless capture artifacts to MP4.")
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--point-size", type=int, default=2)
    parser.add_argument("--max-render-points", type=int, default=0)
    parser.add_argument("--query-point-radius", type=int, default=3)
    parser.add_argument("--demo-visual-mode", choices=DEMO_VISUAL_MODES, default="tracking")
    parser.add_argument("--panel-mode", choices=PANEL_MODES, default=PANEL_MODE_SINGLE)
    parser.add_argument(
        "--table-z-overlay-sweep",
        action="store_true",
        help="Render table-Z before/after/removed RGB overlay sweep instead of the normal demo video.",
    )
    parser.add_argument(
        "--table-z-overlay-output-dir",
        type=Path,
        default=None,
        help="Output directory for --table-z-overlay-sweep. Defaults to <output stem>_table_z_overlay.",
    )
    parser.add_argument(
        "--table-z-threshold-m",
        type=float,
        action="append",
        default=None,
        help="Repeatable table-Z overlay threshold in meters. Defaults to 0.005,0.010,0.020,0.030.",
    )
    parser.add_argument(
        "--table-z-above-direction",
        choices=TABLE_Z_ABOVE_DIRECTIONS,
        default=None,
        help="Override table-world direction away from the tabletop. Defaults to capture metadata, then negative.",
    )
    parser.add_argument(
        "--tracking-background-mask",
        choices=TRACKING_BACKGROUND_MASK_MODES,
        default=TRACKING_BACKGROUND_MASK_TARGET_UNION,
        help="Tracking render RGB background policy: target-union masks table/background, rgb preserves full RGB.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if bool(args.table_z_overlay_sweep):
        output_dir = args.table_z_overlay_output_dir
        if output_dir is None:
            output_dir = args.output.with_suffix("")
            output_dir = output_dir.with_name(f"{output_dir.name}_table_z_overlay")
        summary = render_table_z_filter_overlay_sweep(
            capture_dir=args.capture_dir,
            output_dir=output_dir,
            fps=float(args.fps),
            point_size=int(args.point_size),
            max_render_points=int(args.max_render_points),
            table_z_above_direction=args.table_z_above_direction,
            thresholds_m=tuple(float(value) for value in (args.table_z_threshold_m or DEFAULT_TABLE_Z_OVERLAY_THRESHOLDS_M)),
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    summary = render_capture_to_video(
        capture_dir=args.capture_dir,
        output=args.output,
        fps=float(args.fps),
        point_size=int(args.point_size),
        max_render_points=int(args.max_render_points),
        query_point_radius=int(args.query_point_radius),
        demo_visual_mode=str(args.demo_visual_mode),
        tracking_background_mask=str(args.tracking_background_mask),
        panel_mode=str(args.panel_mode),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
