"""Depth backprojection, world-Z diagnostics, table-Z filter, tracker geometry helpers."""
from __future__ import annotations

from demo_v6_2.mdp_constants import *  # noqa: F401,F403
from demo_v6_2.mdp_packets import MarkerResidualAudit

def _masked_sample_indices(
    *,
    depth_m: np.ndarray,
    mask: np.ndarray,
    depth_min_m: float,
    depth_max_m: float,
    max_points: int,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the masked sample indices."""
    if depth_m.ndim != 2 or mask.ndim != 2:
        raise ValueError("depth_m and mask must be 2D arrays")
    if depth_m.shape != mask.shape:
        raise ValueError("depth and mask shapes must match")
    if max_points < 0:
        raise ValueError("max_points must be >= 0")
    valid = np.isfinite(depth_m) & (depth_m > np.float32(depth_min_m))
    if depth_max_m > 0:
        valid &= depth_m < np.float32(depth_max_m)
    selected = valid & np.asarray(mask, dtype=bool)
    if not np.any(selected):
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    rows, cols = np.nonzero(selected)
    if max_points > 0 and rows.shape[0] > max_points:
        generator = rng if rng is not None else np.random.default_rng()
        indices = generator.choice(rows.shape[0], int(max_points), replace=False)
        rows = rows[indices]
        cols = cols[indices]
    return rows.astype(np.int64, copy=False), cols.astype(np.int64, copy=False)


def erode_binary_mask(mask: np.ndarray, *, erode_pixels: int) -> np.ndarray:
    """Erode a binary mask by the requested pixel radius."""
    pixels = int(erode_pixels)
    if pixels < 0:
        raise ValueError("erode_pixels must be >= 0")
    mask_bool = np.asarray(mask, dtype=bool)
    if pixels == 0 or mask_bool.size == 0 or not np.any(mask_bool):
        return np.ascontiguousarray(mask_bool)

    eroded = mask_bool
    for _ in range(pixels):
        padded = np.pad(eroded, 1, mode="constant", constant_values=False)
        eroded = (
            padded[:-2, :-2]
            & padded[:-2, 1:-1]
            & padded[:-2, 2:]
            & padded[1:-1, :-2]
            & padded[1:-1, 1:-1]
            & padded[1:-1, 2:]
            & padded[2:, :-2]
            & padded[2:, 1:-1]
            & padded[2:, 2:]
        )
        if not np.any(eroded):
            break
    return np.ascontiguousarray(eroded)


def backproject_masked_rgbd(
    *,
    color_bgr: np.ndarray,
    depth_m: np.ndarray,
    mask: np.ndarray,
    ray_x: np.ndarray,
    ray_y: np.ndarray,
    depth_min_m: float,
    depth_max_m: float,
    max_points: int,
    color_mode: str,
    class_rgb: tuple[int, int, int],
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project masked rgbd."""
    if color_bgr.ndim != 3 or color_bgr.shape[2] != 3:
        raise ValueError("color_bgr must be an HxWx3 array")
    if depth_m.shape != color_bgr.shape[:2]:
        raise ValueError("color and depth shapes must match")
    if depth_m.shape != ray_x.shape or depth_m.shape != ray_y.shape:
        raise ValueError("depth and projection grids must have matching shapes")
    if color_mode not in {"rgb", "class"}:
        raise ValueError("color_mode must be 'rgb' or 'class'")

    rows, cols = _masked_sample_indices(
        depth_m=depth_m,
        mask=mask,
        depth_min_m=depth_min_m,
        depth_max_m=depth_max_m,
        max_points=max_points,
        rng=rng,
    )
    if rows.size == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    z = depth_m[rows, cols].astype(np.float32, copy=False)
    x = ray_x[rows, cols].astype(np.float32, copy=False) * z
    y = ray_y[rows, cols].astype(np.float32, copy=False) * z
    points = np.ascontiguousarray(np.stack([x, y, z], axis=1), dtype=np.float32)
    if color_mode == "rgb":
        colors = np.ascontiguousarray(color_bgr[rows, cols, ::-1], dtype=np.uint8)
    else:
        colors = make_solid_colors(points.shape[0], class_rgb)
    return points, colors


def backproject_masked_rgbd_profiled(
    *,
    color_bgr: np.ndarray,
    depth_m: np.ndarray,
    mask: np.ndarray,
    ray_x: np.ndarray,
    ray_y: np.ndarray,
    depth_min_m: float,
    depth_max_m: float,
    max_points: int,
    color_mode: str,
    class_rgb: tuple[int, int, int],
    rng: np.random.Generator | None = None,
    return_yx: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]] | tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Back-project masked rgbd profiled."""
    if color_bgr.ndim != 3 or color_bgr.shape[2] != 3:
        raise ValueError("color_bgr must be an HxWx3 array")
    if depth_m.shape != color_bgr.shape[:2] or depth_m.shape != mask.shape:
        raise ValueError("color, depth, and mask shapes must match")
    if depth_m.shape != ray_x.shape or depth_m.shape != ray_y.shape:
        raise ValueError("depth and projection grids must have matching shapes")
    if max_points < 0:
        raise ValueError("max_points must be >= 0")
    if color_mode not in {"rgb", "class"}:
        raise ValueError("color_mode must be 'rgb' or 'class'")

    timing: dict[str, float] = {}
    started_s = time.perf_counter()
    valid = np.isfinite(depth_m) & (depth_m > np.float32(depth_min_m))
    if depth_max_m > 0:
        valid &= depth_m < np.float32(depth_max_m)
    selected = valid & np.asarray(mask, dtype=bool)
    timing["pcd_mask_intersection_ms"] = _elapsed_ms(started_s, time.perf_counter())

    started_s = time.perf_counter()
    if not np.any(selected):
        timing["pcd_select_ms"] = _elapsed_ms(started_s, time.perf_counter())
        timing["pcd_point_cap_ms"] = 0.0
        timing["pcd_backproject_ms"] = 0.0
        timing["pcd_color_gather_ms"] = 0.0
        timing["pcd_raw_points"] = 0.0
        timing["pcd_cap_points"] = 0.0
        empty_points = np.empty((0, 3), dtype=np.float32)
        empty_colors = np.empty((0, 3), dtype=np.uint8)
        empty_yx = np.empty((0, 2), dtype=np.int64)
        if return_yx:
            return empty_points, empty_colors, empty_yx, timing
        return empty_points, empty_colors, timing
    rows, cols = np.nonzero(selected)
    timing["pcd_raw_points"] = float(rows.shape[0])
    timing["pcd_select_ms"] = _elapsed_ms(started_s, time.perf_counter())

    started_s = time.perf_counter()
    if max_points > 0 and rows.shape[0] > max_points:
        generator = rng if rng is not None else np.random.default_rng()
        indices = generator.choice(rows.shape[0], int(max_points), replace=False)
        rows = rows[indices]
        cols = cols[indices]
    rows = rows.astype(np.int64, copy=False)
    cols = cols.astype(np.int64, copy=False)
    timing["pcd_cap_points"] = float(rows.shape[0])
    timing["pcd_point_cap_ms"] = _elapsed_ms(started_s, time.perf_counter())

    started_s = time.perf_counter()
    z = depth_m[rows, cols].astype(np.float32, copy=False)
    x = ray_x[rows, cols].astype(np.float32, copy=False) * z
    y = ray_y[rows, cols].astype(np.float32, copy=False) * z
    points = np.ascontiguousarray(np.stack([x, y, z], axis=1), dtype=np.float32)
    timing["pcd_backproject_ms"] = _elapsed_ms(started_s, time.perf_counter())

    started_s = time.perf_counter()
    if color_mode == "rgb":
        colors = np.ascontiguousarray(color_bgr[rows, cols, ::-1], dtype=np.uint8)
    else:
        colors = make_solid_colors(points.shape[0], class_rgb)
    timing["pcd_color_gather_ms"] = _elapsed_ms(started_s, time.perf_counter())
    if return_yx:
        yx = np.ascontiguousarray(np.stack([rows, cols], axis=1), dtype=np.int64)
        return points, colors, yx, timing
    return points, colors, timing


def backproject_masked(
    *,
    depth_m: np.ndarray,
    mask: np.ndarray,
    ray_x: np.ndarray,
    ray_y: np.ndarray,
    depth_min_m: float,
    depth_max_m: float,
    max_points: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Back-project masked."""
    if depth_m.shape != ray_x.shape or depth_m.shape != ray_y.shape:
        raise ValueError("depth and projection grids must have matching shapes")
    rows, cols = _masked_sample_indices(
        depth_m=depth_m,
        mask=mask,
        depth_min_m=depth_min_m,
        depth_max_m=depth_max_m,
        max_points=max_points,
        rng=rng,
    )
    if rows.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    z = depth_m[rows, cols].astype(np.float32, copy=False)
    x = ray_x[rows, cols].astype(np.float32, copy=False) * z
    y = ray_y[rows, cols].astype(np.float32, copy=False) * z
    return np.ascontiguousarray(np.stack([x, y, z], axis=1), dtype=np.float32)


def make_solid_colors(point_count: int, rgb: tuple[int, int, int]) -> np.ndarray:
    """Create solid colors."""
    if point_count <= 0:
        return np.empty((0, 3), dtype=np.uint8)
    color = np.asarray(rgb, dtype=np.uint8).reshape(1, 3)
    return np.repeat(color, int(point_count), axis=0)


def _camera_intrinsics_matrix(intrinsics: CameraIntrinsics) -> np.ndarray:
    """Return the camera intrinsics matrix."""
    return np.array(
        [
            [float(intrinsics.fx), 0.0, float(intrinsics.cx)],
            [0.0, float(intrinsics.fy), float(intrinsics.cy)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _transform_points_c2w(points_xyz_m: np.ndarray, c2w: np.ndarray | None) -> np.ndarray:
    """Transform points C2W."""
    points = np.asarray(points_xyz_m, dtype=np.float32).reshape(-1, 3)
    if c2w is None or points.size == 0:
        return np.ascontiguousarray(points, dtype=np.float32)
    matrix = np.asarray(c2w, dtype=np.float32)
    if matrix.shape != (4, 4):
        raise ValueError(f"camera-to-world transform must be 4x4, got {matrix.shape}")
    homogeneous = np.concatenate(
        [points, np.ones((points.shape[0], 1), dtype=np.float32)],
        axis=1,
    )
    world = (matrix @ homogeneous.T).T[:, :3]
    return np.ascontiguousarray(world, dtype=np.float32)


def _z_quantiles(points_xyz_m: np.ndarray) -> dict[str, float | None]:
    """Return the z quantiles."""
    keys = ("min", "p01", "p05", "p10", "p50", "p90", "p95", "p99", "max")
    points = np.asarray(points_xyz_m, dtype=np.float32).reshape(-1, 3)
    z = points[:, 2]
    finite = z[np.isfinite(z)]
    if finite.size == 0:
        # Covers both empty input and all-NaN/inf depth: every quantile is None.
        return {key: None for key in keys}
    quantiles = np.quantile(
        finite.astype(np.float64),
        [0.0, 0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99, 1.0],
    )
    return {key: float(value) for key, value in zip(keys, quantiles)}


def table_z_clearance_m(
    points_xyz_m: np.ndarray,
    *,
    table_z_m: float = TABLE_Z_M,
) -> np.ndarray:
    """Return the table z clearance m."""
    points = np.asarray(points_xyz_m, dtype=np.float32).reshape(-1, 3)
    return np.ascontiguousarray(
        np.float32(table_z_m) - points[:, 2],
        dtype=np.float32,
    )


def _world_z_class_stats(
    points_xyz_m: np.ndarray,
    *,
    table_z_m: float,
    thresholds_m: tuple[float, ...],
) -> dict[str, Any]:
    """Return the world z class stats."""
    points = np.asarray(points_xyz_m, dtype=np.float32).reshape(-1, 3)
    finite = np.isfinite(points).all(axis=1) if len(points) else np.zeros((0,), dtype=bool)
    clearance = table_z_clearance_m(points, table_z_m=table_z_m)
    threshold_rows: list[dict[str, float | int]] = []
    for threshold_m in thresholds_m:
        candidate = finite & (clearance <= np.float32(float(threshold_m)))
        count = int(np.count_nonzero(candidate))
        threshold_rows.append(
            {
                "threshold_m": float(threshold_m),
                "candidate_count": count,
                "candidate_ratio": float(count / max(1, len(points))),
            }
        )
    return {
        "count": int(len(points)),
        "finite_count": int(np.count_nonzero(finite)),
        "z_m": _z_quantiles(points),
        "table_thresholds": threshold_rows,
    }


def build_world_z_diagnostics(
    *,
    object_xyz_m: np.ndarray,
    controller_xyz_m: np.ndarray,
    hand_a_xyz_m: np.ndarray | None = None,
    hand_b_xyz_m: np.ndarray | None = None,
    table_z_m: float = TABLE_Z_M,
    thresholds_m: tuple[float, ...] = DEFAULT_TABLE_Z_DIAGNOSTIC_THRESHOLDS_M,
) -> dict[str, Any]:
    """Build world z diagnostics."""
    thresholds = tuple(float(value) for value in thresholds_m)
    classes: dict[str, Any] = {
        "object": _world_z_class_stats(
            object_xyz_m,
            table_z_m=float(table_z_m),
            thresholds_m=thresholds,
        ),
        "controller": _world_z_class_stats(
            controller_xyz_m,
            table_z_m=float(table_z_m),
            thresholds_m=thresholds,
        ),
    }
    if hand_a_xyz_m is not None:
        classes["hand_a"] = _world_z_class_stats(
            hand_a_xyz_m,
            table_z_m=float(table_z_m),
            thresholds_m=thresholds,
        )
    if hand_b_xyz_m is not None:
        classes["hand_b"] = _world_z_class_stats(
            hand_b_xyz_m,
            table_z_m=float(table_z_m),
            thresholds_m=thresholds,
        )
    return {
        "table_z_m": float(table_z_m),
        "table_z_above_direction": TABLE_Z_ABOVE_DIRECTION,
        "thresholds_m": [float(value) for value in thresholds],
        "classes": classes,
    }


def apply_table_z_filter(
    points_xyz_m: np.ndarray,
    colors_rgb_u8: np.ndarray,
    *,
    enabled: bool,
    threshold_m: float,
    table_z_m: float = TABLE_Z_M,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Apply table z filter."""
    points = np.asarray(points_xyz_m, dtype=np.float32).reshape(-1, 3)
    colors = np.asarray(colors_rgb_u8, dtype=np.uint8).reshape(-1, 3)
    if len(points) != len(colors):
        raise ValueError("points and colors must have the same first dimension")
    if not bool(enabled) or len(points) == 0:
        return np.ascontiguousarray(points, dtype=np.float32), np.ascontiguousarray(colors, dtype=np.uint8), {
            "enabled": bool(enabled),
            "threshold_m": float(threshold_m),
            "table_z_m": float(table_z_m),
            "table_z_above_direction": TABLE_Z_ABOVE_DIRECTION,
            "input_points": int(len(points)),
            "removed_points": 0,
            "output_points": int(len(points)),
            "removed_ratio": 0.0,
        }
    finite = np.isfinite(points).all(axis=1)
    clearance = table_z_clearance_m(points, table_z_m=table_z_m)
    remove = finite & (clearance <= np.float32(float(threshold_m)))
    keep = ~remove
    removed = int(np.count_nonzero(remove))
    return (
        np.ascontiguousarray(points[keep], dtype=np.float32),
        np.ascontiguousarray(colors[keep], dtype=np.uint8),
        {
            "enabled": True,
            "threshold_m": float(threshold_m),
            "table_z_m": float(table_z_m),
            "table_z_above_direction": TABLE_Z_ABOVE_DIRECTION,
            "input_points": int(len(points)),
            "removed_points": removed,
            "output_points": int(np.count_nonzero(keep)),
            "removed_ratio": float(removed / max(1, len(points))),
        },
    )


def apply_table_z_filter_with_yx(
    points_xyz_m: np.ndarray,
    colors_rgb_u8: np.ndarray,
    yx: np.ndarray,
    *,
    enabled: bool,
    threshold_m: float,
    table_z_m: float = TABLE_Z_M,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Apply table z filter with YX."""
    points = np.asarray(points_xyz_m, dtype=np.float32).reshape(-1, 3)
    colors = np.asarray(colors_rgb_u8, dtype=np.uint8).reshape(-1, 3)
    yx_arr = np.asarray(yx, dtype=np.int64).reshape(-1, 2)
    if len(points) != len(colors) or len(points) != len(yx_arr):
        raise ValueError("points, colors, and yx must have the same first dimension")
    if not bool(enabled) or len(points) == 0:
        return (
            np.ascontiguousarray(points, dtype=np.float32),
            np.ascontiguousarray(colors, dtype=np.uint8),
            np.ascontiguousarray(yx_arr, dtype=np.int64),
            {
                "enabled": bool(enabled),
                "threshold_m": float(threshold_m),
                "table_z_m": float(table_z_m),
                "table_z_above_direction": TABLE_Z_ABOVE_DIRECTION,
                "input_points": int(len(points)),
                "removed_points": 0,
                "output_points": int(len(points)),
                "removed_ratio": 0.0,
            },
        )
    finite = np.isfinite(points).all(axis=1)
    clearance = table_z_clearance_m(points, table_z_m=table_z_m)
    remove = finite & (clearance <= np.float32(float(threshold_m)))
    keep = ~remove
    removed = int(np.count_nonzero(remove))
    return (
        np.ascontiguousarray(points[keep], dtype=np.float32),
        np.ascontiguousarray(colors[keep], dtype=np.uint8),
        np.ascontiguousarray(yx_arr[keep], dtype=np.int64),
        {
            "enabled": True,
            "threshold_m": float(threshold_m),
            "table_z_m": float(table_z_m),
            "table_z_above_direction": TABLE_Z_ABOVE_DIRECTION,
            "input_points": int(len(points)),
            "removed_points": removed,
            "output_points": int(np.count_nonzero(keep)),
            "removed_ratio": float(removed / max(1, len(points))),
        },
    )


def _tracker_union_mask(mask_packet: MaskPacket) -> np.ndarray:
    """Return the tracker union mask."""
    controller = np.asarray(mask_packet.controller_mask, dtype=bool)
    obj = np.asarray(mask_packet.object_mask, dtype=bool)
    if controller.shape != obj.shape:
        raise ValueError("controller/object masks must share a shape")
    return np.logical_or(controller, obj)


def _mask_packet_hand_a_mask(mask_packet: MaskPacket) -> np.ndarray:
    """Return the mask packet hand a mask."""
    if mask_packet.hand_a_mask is None:
        return np.asarray(mask_packet.controller_mask, dtype=bool)
    return np.asarray(mask_packet.hand_a_mask, dtype=bool)


def _mask_packet_hand_b_mask(mask_packet: MaskPacket) -> np.ndarray:
    """Return the mask packet hand b mask."""
    if mask_packet.hand_b_mask is None:
        return np.zeros_like(np.asarray(mask_packet.controller_mask, dtype=bool), dtype=bool)
    return np.asarray(mask_packet.hand_b_mask, dtype=bool)


def _classify_query_points_yx(
    query_points_yx: np.ndarray,
    *,
    object_mask: np.ndarray,
    controller_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Classify query points YX."""
    points = np.asarray(query_points_yx, dtype=np.float32).reshape(-1, 2)
    if len(points) == 0:
        empty = np.empty((0,), dtype=bool)
        return empty, empty
    object_bool = np.asarray(object_mask, dtype=bool)
    controller_bool = np.asarray(controller_mask, dtype=bool)
    height, width = object_bool.shape[:2]
    yy = np.clip(np.rint(points[:, 0]).astype(np.int64), 0, height - 1)
    xx = np.clip(np.rint(points[:, 1]).astype(np.int64), 0, width - 1)
    return object_bool[yy, xx].astype(bool), controller_bool[yy, xx].astype(bool)


def _classify_query_targets_yx(
    query_points_yx: np.ndarray,
    *,
    object_mask: np.ndarray,
    hand_a_mask: np.ndarray,
    hand_b_mask: np.ndarray,
    controller_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Classify query targets YX."""
    points = np.asarray(query_points_yx, dtype=np.float32).reshape(-1, 2)
    if len(points) == 0:
        empty_bool = np.empty((0,), dtype=bool)
        empty_int = np.empty((0,), dtype=np.int64)
        return empty_bool, empty_bool, empty_int, empty_int
    object_bool = np.asarray(object_mask, dtype=bool)
    hand_a_bool = np.asarray(hand_a_mask, dtype=bool)
    hand_b_bool = np.asarray(hand_b_mask, dtype=bool)
    controller_bool = np.asarray(controller_mask, dtype=bool)
    height, width = object_bool.shape[:2]
    yy = np.clip(np.rint(points[:, 0]).astype(np.int64), 0, height - 1)
    xx = np.clip(np.rint(points[:, 1]).astype(np.int64), 0, width - 1)
    in_hand_a = hand_a_bool[yy, xx]
    in_hand_b = hand_b_bool[yy, xx] & ~in_hand_a
    in_object = object_bool[yy, xx] & ~(in_hand_a | in_hand_b)
    in_controller = controller_bool[yy, xx] | in_hand_a | in_hand_b
    target_id = np.zeros((len(points),), dtype=np.int64)
    target_id[in_object] = OBJECT_ID
    target_id[in_hand_a] = HAND_A_ID
    target_id[in_hand_b] = HAND_B_ID
    controller_instance_id = np.zeros((len(points),), dtype=np.int64)
    controller_instance_id[in_hand_a] = QUERY_CONTROLLER_INSTANCE_HAND_A
    controller_instance_id[in_hand_b] = QUERY_CONTROLLER_INSTANCE_HAND_B
    return in_object.astype(bool), in_controller.astype(bool), target_id, controller_instance_id


def _mask_from_yx(shape: tuple[int, int], yx: np.ndarray) -> np.ndarray:
    """Return the mask from YX."""
    mask = np.zeros(tuple(shape), dtype=bool)
    coords = np.asarray(yx, dtype=np.int64).reshape(-1, 2)
    if len(coords) == 0:
        return mask
    rows = coords[:, 0]
    cols = coords[:, 1]
    valid = (rows >= 0) & (rows < shape[0]) & (cols >= 0) & (cols < shape[1])
    if np.any(valid):
        mask[rows[valid], cols[valid]] = True
    return np.ascontiguousarray(mask)


def _select_points_by_yx_mask(points_xyz_m: np.ndarray, yx: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Select 3D points whose YX pixel locations fall inside a mask."""
    points = np.asarray(points_xyz_m, dtype=np.float32).reshape(-1, 3)
    coords = np.asarray(yx, dtype=np.int64).reshape(-1, 2)
    if len(points) == 0 or len(coords) == 0:
        return np.empty((0, 3), dtype=np.float32)
    count = min(len(points), len(coords))
    target = np.asarray(mask, dtype=bool)
    rows = coords[:count, 0]
    cols = coords[:count, 1]
    valid = (rows >= 0) & (rows < target.shape[0]) & (cols >= 0) & (cols < target.shape[1])
    keep = np.zeros((count,), dtype=bool)
    if np.any(valid):
        keep[valid] = target[rows[valid], cols[valid]]
    return np.ascontiguousarray(points[:count][keep], dtype=np.float32)


def _tracker_display_visibility(
    visibility: np.ndarray,
    *,
    query_is_object: np.ndarray,
    query_is_controller: np.ndarray,
    display_scope: str,
) -> np.ndarray:
    """Return the tracker display visibility."""
    vis = np.asarray(visibility, dtype=np.float32).reshape(-1)
    scope = str(display_scope)
    if scope == TRACKER_DISPLAY_SCOPE_UNION:
        return vis
    if scope == TRACKER_DISPLAY_SCOPE_OBJECT:
        labels = np.asarray(query_is_object, dtype=bool).reshape(-1)
    else:
        labels = np.asarray(query_is_controller, dtype=bool).reshape(-1)
    if labels.shape[0] != vis.shape[0]:
        fitted = np.zeros_like(vis, dtype=bool)
        fitted[: min(len(labels), len(fitted))] = labels[: min(len(labels), len(fitted))]
        labels = fitted
    return np.where(labels, vis, 0.0).astype(np.float32)


def _tracker_per_target_visibility(
    tracks_yx: np.ndarray,
    visibility: np.ndarray,
    *,
    mask_packet: MaskPacket,
    query_target_id: np.ndarray,
) -> np.ndarray:
    """Return the tracker per target visibility."""
    tracks = np.asarray(tracks_yx, dtype=np.float32).reshape(-1, 2)
    vis = np.asarray(visibility, dtype=np.float32).reshape(-1)
    target_id = np.asarray(query_target_id, dtype=np.int64).reshape(-1)
    count = min(len(tracks), len(vis), len(target_id))
    output = np.zeros((len(vis),), dtype=np.float32)
    if count == 0:
        return output
    object_mask = np.asarray(mask_packet.object_mask, dtype=bool)
    hand_a_mask = _mask_packet_hand_a_mask(mask_packet)
    hand_b_mask = _mask_packet_hand_b_mask(mask_packet)
    height, width = object_mask.shape[:2]
    yy = np.rint(tracks[:count, 0]).astype(np.int64)
    xx = np.rint(tracks[:count, 1]).astype(np.int64)
    finite_tracks = np.isfinite(tracks[:count]).all(axis=1)
    in_bounds = (yy >= 0) & (yy < height) & (xx >= 0) & (xx < width)
    valid = (vis[:count] > 0.0) & finite_tracks & in_bounds
    if not np.any(valid):
        return output
    valid_indices = np.flatnonzero(valid)
    inside_target = np.zeros((count,), dtype=bool)
    valid_targets = target_id[valid_indices]
    hand_a_indices = valid_indices[valid_targets == HAND_A_ID]
    if len(hand_a_indices):
        inside_target[hand_a_indices] = hand_a_mask[yy[hand_a_indices], xx[hand_a_indices]]
    hand_b_indices = valid_indices[valid_targets == HAND_B_ID]
    if len(hand_b_indices):
        inside_target[hand_b_indices] = hand_b_mask[yy[hand_b_indices], xx[hand_b_indices]]
    object_indices = valid_indices[valid_targets == OBJECT_ID]
    if len(object_indices):
        inside_target[object_indices] = object_mask[yy[object_indices], xx[object_indices]]
    output[:count] = np.where(inside_target, vis[:count], 0.0).astype(np.float32)
    return output


def _tracker_lift_valid_mask(
    *,
    tracks_yx: np.ndarray,
    visibility: np.ndarray,
    depth: np.ndarray,
    depth_scale_m_per_unit: float,
    mask: np.ndarray | None,
    depth_min_m: float,
    depth_max_m: float,
) -> np.ndarray:
    """Return the tracker lift valid mask."""
    tracks = np.asarray(tracks_yx, dtype=np.float32).reshape(-1, 2)
    vis = np.asarray(visibility, dtype=np.float32).reshape(-1) > 0.0
    if vis.shape[0] != tracks.shape[0]:
        raise ValueError("visibility length must match tracks_yx")

    depth_arr = np.asarray(depth)
    if np.issubdtype(depth_arr.dtype, np.floating):
        depth_m = depth_arr.astype(np.float32, copy=False)
    else:
        depth_m = depth_arr.astype(np.float32) * np.float32(depth_scale_m_per_unit)
    height, width = depth_m.shape[:2]
    mask_bool = np.ones((height, width), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if mask_bool.shape[:2] != (height, width):
        raise ValueError("tracker lift mask shape must match depth shape")

    yy = np.rint(tracks[:, 0]).astype(np.int64)
    xx = np.rint(tracks[:, 1]).astype(np.int64)
    finite_tracks = np.isfinite(tracks).all(axis=1)
    in_bounds = (yy >= 0) & (yy < height) & (xx >= 0) & (xx < width)
    valid = vis & finite_tracks & in_bounds
    if not np.any(valid):
        return np.zeros((tracks.shape[0],), dtype=bool)

    valid_indices = np.flatnonzero(valid)
    sampled_depth = depth_m[yy[valid_indices], xx[valid_indices]]
    depth_valid = np.isfinite(sampled_depth) & (sampled_depth > 0.0) & (sampled_depth >= np.float32(depth_min_m))
    if np.isfinite(float(depth_max_m)):
        depth_valid &= sampled_depth <= np.float32(depth_max_m)
    inside_mask = mask_bool[yy[valid_indices], xx[valid_indices]]
    valid_out = np.zeros((tracks.shape[0],), dtype=bool)
    valid_out[valid_indices] = depth_valid & inside_mask
    return valid_out


def _query_current_residual_visibility(
    tracks_yx: np.ndarray,
    *,
    query_is_object: np.ndarray,
    query_is_controller: np.ndarray,
    object_residual_mask: np.ndarray,
    controller_residual_mask: np.ndarray,
) -> np.ndarray:
    """Return the query current residual visibility."""
    tracks = np.asarray(tracks_yx, dtype=np.float32).reshape(-1, 2)
    is_object = np.asarray(query_is_object, dtype=bool).reshape(-1)
    is_controller = np.asarray(query_is_controller, dtype=bool).reshape(-1)
    count = min(len(tracks), len(is_object), len(is_controller))
    visible = np.zeros((len(tracks),), dtype=bool)
    if count <= 0:
        return visible

    object_mask = np.asarray(object_residual_mask, dtype=bool)
    controller_mask = np.asarray(controller_residual_mask, dtype=bool)
    if object_mask.shape != controller_mask.shape:
        raise ValueError("object/controller residual masks must share a shape")
    height, width = object_mask.shape[:2]
    yy = np.rint(tracks[:count, 0]).astype(np.int64)
    xx = np.rint(tracks[:count, 1]).astype(np.int64)
    finite = np.isfinite(tracks[:count]).all(axis=1)
    in_bounds = finite & (yy >= 0) & (yy < height) & (xx >= 0) & (xx < width)
    if not np.any(in_bounds):
        return visible

    valid_indices = np.flatnonzero(in_bounds)
    object_indices = valid_indices[is_object[:count][valid_indices]]
    if len(object_indices):
        visible[object_indices] = object_mask[yy[object_indices], xx[object_indices]]
    controller_indices = valid_indices[is_controller[:count][valid_indices]]
    if len(controller_indices):
        visible[controller_indices] |= controller_mask[yy[controller_indices], xx[controller_indices]]
    unlabelled_indices = valid_indices[~(is_object[:count][valid_indices] | is_controller[:count][valid_indices])]
    if len(unlabelled_indices):
        union_mask = np.logical_or(object_mask, controller_mask)
        visible[unlabelled_indices] = union_mask[yy[unlabelled_indices], xx[unlabelled_indices]]
    return visible


def _audit_marker_residual_subset(
    marker_tracks_yx: np.ndarray,
    *,
    object_residual_mask: np.ndarray,
    controller_residual_mask: np.ndarray,
    gate: str = TRACKER_MARKER_GATE_PCD_FILTER_RESIDUAL_TABLE_Z,
) -> MarkerResidualAudit:
    """Audit marker residual subset."""
    tracks = np.asarray(marker_tracks_yx, dtype=np.float32).reshape(-1, 2)
    object_mask = np.asarray(object_residual_mask, dtype=bool)
    controller_mask = np.asarray(controller_residual_mask, dtype=bool)
    if object_mask.shape != controller_mask.shape:
        raise ValueError("object/controller residual masks must share a shape")

    count = int(tracks.shape[0])
    pixels_yx = np.full((count, 2), -1, dtype=np.int64)
    valid = np.zeros((count,), dtype=bool)
    if count <= 0:
        return MarkerResidualAudit(
            pixels_yx=pixels_yx,
            valid=valid,
            violation=np.zeros((0,), dtype=bool),
            checked_count=0,
            violation_count=0,
            gate=str(gate),
        )

    finite = np.isfinite(tracks).all(axis=1)
    if np.any(finite):
        pixels_yx[finite] = np.rint(tracks[finite]).astype(np.int64)

    height, width = object_mask.shape[:2]
    yy = pixels_yx[:, 0]
    xx = pixels_yx[:, 1]
    in_bounds = finite & (yy >= 0) & (yy < int(height)) & (xx >= 0) & (xx < int(width))
    if np.any(in_bounds):
        union_mask = np.logical_or(object_mask, controller_mask)
        valid[in_bounds] = union_mask[yy[in_bounds], xx[in_bounds]]

    violation = ~valid
    return MarkerResidualAudit(
        pixels_yx=np.ascontiguousarray(pixels_yx, dtype=np.int64),
        valid=np.ascontiguousarray(valid, dtype=bool),
        violation=np.ascontiguousarray(violation, dtype=bool),
        checked_count=count,
        violation_count=int(np.count_nonzero(violation)),
        gate=str(gate),
    )


def _select_visible_spread_indices(tracks_yx: np.ndarray, visibility: np.ndarray, *, max_points: int) -> np.ndarray:
    """Select visible spread indices."""
    tracks = np.asarray(tracks_yx, dtype=np.float32).reshape(-1, 2)
    visible = np.flatnonzero(np.asarray(visibility, dtype=np.float32).reshape(-1) > 0.0)
    if len(visible) > 0:
        visible = visible[np.isfinite(tracks[visible]).all(axis=1)]
    # overlay cap is fixed to 0 (draw all visible markers); the former
    # farthest-point subsampling for cap > 0 was unreachable and removed.
    return visible.astype(np.int64)


def _latest_tracker_arrays(result: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return the latest tracker arrays."""
    tracks = np.asarray(result.tracks_yx, dtype=np.float32)
    visibility = np.asarray(result.visibility, dtype=np.float32)
    if tracks.ndim == 4:
        tracks_latest = tracks[0, -1]
        visibility_latest = visibility[0, -1]
    elif tracks.ndim == 3:
        tracks_latest = tracks[-1]
        visibility_latest = visibility[-1]
    elif tracks.ndim == 2:
        tracks_latest = tracks
        visibility_latest = visibility
    else:
        raise ValueError(f"tracker tracks_yx must be 2D, 3D, or 4D; got {tracks.shape}")
    return (
        np.ascontiguousarray(np.asarray(tracks_latest, dtype=np.float32).reshape(-1, 2)),
        np.ascontiguousarray(np.asarray(visibility_latest, dtype=np.float32).reshape(-1)),
    )


__all__ = [
    "_masked_sample_indices",
    "erode_binary_mask",
    "backproject_masked_rgbd",
    "backproject_masked_rgbd_profiled",
    "backproject_masked",
    "make_solid_colors",
    "_camera_intrinsics_matrix",
    "_transform_points_c2w",
    "_z_quantiles",
    "table_z_clearance_m",
    "_world_z_class_stats",
    "build_world_z_diagnostics",
    "apply_table_z_filter",
    "apply_table_z_filter_with_yx",
    "_tracker_union_mask",
    "_mask_packet_hand_a_mask",
    "_mask_packet_hand_b_mask",
    "_classify_query_points_yx",
    "_classify_query_targets_yx",
    "_mask_from_yx",
    "_select_points_by_yx_mask",
    "_tracker_display_visibility",
    "_tracker_per_target_visibility",
    "_tracker_lift_valid_mask",
    "_query_current_residual_visibility",
    "_audit_marker_residual_subset",
    "_select_visible_spread_indices",
    "_latest_tracker_arrays",
]
