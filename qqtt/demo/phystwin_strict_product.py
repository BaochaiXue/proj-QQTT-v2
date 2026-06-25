from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import pickle
from typing import Any, Mapping, Sequence

import numpy as np

from data_process.depth_backends.geometry import transform_points
from qqtt.demo.pcd_postprocess import _detect_radius_outlier_indices
from qqtt.demo.realtime_single_camera_pointcloud import build_projection_grid_from_matrix
from qqtt.tracking.sampling import PHYSTWIN_DENSE_QUERY_POINTS, sample_phystwin_dense


TRACKING_PRODUCT_BACKEND_REALTIME_OVERLAY = "realtime-overlay"
TRACKING_PRODUCT_BACKEND_PHYSTWIN_STRICT = "phystwin-strict-tracking"
TRACKING_PRODUCT_BACKENDS = (
    TRACKING_PRODUCT_BACKEND_REALTIME_OVERLAY,
    TRACKING_PRODUCT_BACKEND_PHYSTWIN_STRICT,
)
DEFAULT_TRACKING_PRODUCT_BACKEND = TRACKING_PRODUCT_BACKEND_REALTIME_OVERLAY

COMPATIBILITY_TARGET_PHYSTWIN = "PhysTwin"
PHYSTWIN_STRICT_EXECUTION_MODE = "workstation_strict"
PHYSTWIN_COMPATIBILITY_PATH_NAME = "cotracker"


@dataclass(frozen=True)
class StrictQuerySample:
    query_txy: np.ndarray
    query_points_yx: np.ndarray
    union_mask: np.ndarray


@dataclass(frozen=True)
class PreparedPhysTwinFrame:
    seq: int
    rgb_frame: np.ndarray
    processed_mask_frame: Mapping[str, np.ndarray]
    pcd_points: np.ndarray
    pcd_colors: np.ndarray
    tracks_yx: np.ndarray
    visibility: np.ndarray
    query_points_yx: np.ndarray
    source_timestamp_s: float | None = None
    source_frame_index: int | None = None
    source_step: int | None = None


def normalize_tracking_product_backend(value: str | None) -> str:
    normalized = str(value or DEFAULT_TRACKING_PRODUCT_BACKEND).strip().lower().replace("_", "-")
    aliases = {
        "overlay": TRACKING_PRODUCT_BACKEND_REALTIME_OVERLAY,
        "realtime": TRACKING_PRODUCT_BACKEND_REALTIME_OVERLAY,
        "tapnextpp-overlay": TRACKING_PRODUCT_BACKEND_REALTIME_OVERLAY,
        "phystwin-strict": TRACKING_PRODUCT_BACKEND_PHYSTWIN_STRICT,
        "phys-twin-strict": TRACKING_PRODUCT_BACKEND_PHYSTWIN_STRICT,
        "phystwin-strict-tracking": TRACKING_PRODUCT_BACKEND_PHYSTWIN_STRICT,
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in TRACKING_PRODUCT_BACKENDS:
        raise ValueError(
            f"unsupported tracking product backend {value!r}; expected one of {TRACKING_PRODUCT_BACKENDS}"
        )
    return normalized


def tracking_product_backend_is_strict(value: str | None) -> bool:
    return normalize_tracking_product_backend(value) == TRACKING_PRODUCT_BACKEND_PHYSTWIN_STRICT


def sample_first_frame_union_queries(
    object_mask: np.ndarray,
    controller_mask: np.ndarray,
    *,
    max_queries: int = PHYSTWIN_DENSE_QUERY_POINTS,
    seed: int = 42,
    camera_idx: int = 0,
) -> StrictQuerySample:
    obj = np.asarray(object_mask, dtype=bool)
    ctrl = np.asarray(controller_mask, dtype=bool)
    if obj.shape != ctrl.shape:
        raise ValueError("object_mask and controller_mask must have the same shape")
    union = np.logical_or(obj, ctrl)
    points_yx = sample_phystwin_dense(union, seed=int(seed), camera_idx=int(camera_idx), torch_device="cpu")
    cap = int(max_queries)
    if cap > 0 and len(points_yx) > cap:
        points_yx = np.ascontiguousarray(points_yx[:cap], dtype=np.float32)
    query_txy = np.zeros((len(points_yx), 3), dtype=np.float32)
    if len(points_yx):
        query_txy[:, 1] = points_yx[:, 1]
        query_txy[:, 2] = points_yx[:, 0]
    return StrictQuerySample(
        query_txy=np.ascontiguousarray(query_txy, dtype=np.float32),
        query_points_yx=np.ascontiguousarray(points_yx, dtype=np.float32),
        union_mask=np.ascontiguousarray(union, dtype=bool),
    )


def _mask_from_frame(frame: Mapping[str, Any], key: str, fallback: np.ndarray | None = None) -> np.ndarray:
    if key in frame and frame[key] is not None:
        return np.asarray(frame[key], dtype=bool)
    if fallback is not None:
        return np.asarray(fallback, dtype=bool)
    raise ValueError(f"missing required mask key {key!r}")


def normalize_processed_mask_frame(frame: Mapping[str, Any]) -> dict[str, np.ndarray]:
    obj = _mask_from_frame(frame, "object")
    if "controller" in frame and frame["controller"] is not None:
        ctrl = np.asarray(frame["controller"], dtype=bool)
    else:
        hand_a = np.asarray(frame.get("hand_a", np.zeros_like(obj, dtype=bool)), dtype=bool)
        hand_b = np.asarray(frame.get("hand_b", np.zeros_like(obj, dtype=bool)), dtype=bool)
        ctrl = np.logical_or(hand_a, hand_b)
    if obj.shape != ctrl.shape:
        raise ValueError("object/controller masks must have the same shape")
    obj = np.asarray(obj, dtype=bool) & ~np.asarray(ctrl, dtype=bool)
    out = {
        "object": np.ascontiguousarray(obj, dtype=bool),
        "controller": np.ascontiguousarray(ctrl, dtype=bool),
    }
    if "hand_a" in frame and frame["hand_a"] is not None:
        out["hand_a"] = np.ascontiguousarray(np.asarray(frame["hand_a"], dtype=bool), dtype=bool)
    if "hand_b" in frame and frame["hand_b"] is not None:
        out["hand_b"] = np.ascontiguousarray(np.asarray(frame["hand_b"], dtype=bool), dtype=bool)
    return out


def write_processed_masks(output_dir: str | Path, frames: Sequence[Mapping[str, Any]]) -> Path:
    root = Path(output_dir)
    mask_dir = root / "mask"
    mask_dir.mkdir(parents=True, exist_ok=True)
    processed = [[normalize_processed_mask_frame(frame)] for frame in frames]
    path = mask_dir / "processed_masks.pkl"
    with path.open("wb") as handle:
        pickle.dump(processed, handle)
    return path


def _intrinsics_to_matrix(intrinsics: Any) -> np.ndarray:
    if isinstance(intrinsics, Mapping):
        fx = float(intrinsics["fx"])
        fy = float(intrinsics["fy"])
        cx = float(intrinsics["cx"])
        cy = float(intrinsics["cy"])
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    if all(hasattr(intrinsics, name) for name in ("fx", "fy", "cx", "cy")):
        return np.array(
            [
                [float(intrinsics.fx), 0.0, float(intrinsics.cx)],
                [0.0, float(intrinsics.fy), 0.0 + float(intrinsics.cy)],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
    return np.asarray(intrinsics, dtype=np.float32).reshape(3, 3)


def dense_world_pcd_grid(
    *,
    depth_m: np.ndarray,
    color_rgb_u8: np.ndarray,
    intrinsics: Any,
    c2w: np.ndarray,
    depth_min_m: float = 0.0,
    depth_max_m: float = float("inf"),
) -> tuple[np.ndarray, np.ndarray]:
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"depth_m must be HxW; got {depth.shape}")
    color = np.asarray(color_rgb_u8, dtype=np.uint8)
    if color.shape[:2] != depth.shape or color.ndim != 3 or color.shape[2] != 3:
        raise ValueError("color_rgb_u8 must have shape HxWx3 matching depth_m")
    height, width = depth.shape
    K = _intrinsics_to_matrix(intrinsics)
    ray_x, ray_y = build_projection_grid_from_matrix(width=width, height=height, K=K)
    finite = np.isfinite(depth)
    valid = finite & (depth > 0.0) & (depth >= np.float32(float(depth_min_m)))
    if np.isfinite(float(depth_max_m)):
        valid &= depth <= np.float32(float(depth_max_m))

    points = np.zeros((height, width, 3), dtype=np.float32)
    if np.any(valid):
        rows, cols = np.nonzero(valid)
        z = depth[rows, cols]
        points_camera = np.stack(
            [
                ray_x[rows, cols].astype(np.float32, copy=False) * z,
                ray_y[rows, cols].astype(np.float32, copy=False) * z,
                z,
            ],
            axis=1,
        ).astype(np.float32)
        points_world = transform_points(points_camera, np.asarray(c2w, dtype=np.float32).reshape(4, 4)).astype(np.float32)
        points[rows, cols] = points_world
    return points[None].astype(np.float32, copy=False), color[None].astype(np.uint8, copy=False)


def apply_depth_validity_to_mask_frame(
    frame: Mapping[str, np.ndarray],
    depth_m: np.ndarray,
) -> dict[str, np.ndarray]:
    depth = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.0)
    normalized = normalize_processed_mask_frame(frame)
    filtered: dict[str, np.ndarray] = {}
    for key, mask in normalized.items():
        arr = np.asarray(mask, dtype=bool)
        if arr.shape != valid.shape:
            raise ValueError(f"mask {key!r} shape {arr.shape} does not match depth shape {valid.shape}")
        filtered[key] = np.ascontiguousarray(arr & valid, dtype=bool)
    return normalize_processed_mask_frame(filtered)


def apply_radius_outlier_to_mask_frame(
    frame: Mapping[str, np.ndarray],
    points_grid: np.ndarray,
    *,
    enabled: bool,
    radius_m: float,
    nb_points: int,
) -> dict[str, np.ndarray]:
    normalized = normalize_processed_mask_frame(frame)
    if not bool(enabled):
        return normalized
    grid = np.asarray(points_grid, dtype=np.float32)
    if grid.ndim == 4:
        grid = grid[0]
    if grid.ndim != 3 or grid.shape[-1] != 3:
        raise ValueError(f"points_grid must have shape H,W,3 or 1,H,W,3; got {grid.shape}")

    filtered = {key: np.asarray(value, dtype=bool).copy() for key, value in normalized.items()}
    for key in ("object", "controller"):
        mask = filtered[key]
        if mask.shape != grid.shape[:2]:
            raise ValueError(f"mask {key!r} shape {mask.shape} does not match points grid {grid.shape[:2]}")
        yy, xx = np.nonzero(mask)
        if len(yy) == 0:
            continue
        class_points = grid[yy, xx]
        finite = np.isfinite(class_points).all(axis=1) & (np.linalg.norm(class_points, axis=1) > 1e-9)
        if not np.all(finite):
            invalid_rows = yy[~finite]
            invalid_cols = xx[~finite]
            filtered[key][invalid_rows, invalid_cols] = False
            yy = yy[finite]
            xx = xx[finite]
            class_points = class_points[finite]
        if len(class_points) == 0:
            continue
        result = _detect_radius_outlier_indices(
            class_points,
            radius_m=float(radius_m),
            nb_points=int(nb_points),
        )
        outlier_indices = np.asarray(result["outlier_indices"], dtype=np.int64)
        if len(outlier_indices):
            filtered[key][yy[outlier_indices], xx[outlier_indices]] = False
    return normalize_processed_mask_frame(filtered)


def prepare_phystwin_frame(
    *,
    seq: int,
    rgb_frame: np.ndarray,
    depth_m: np.ndarray,
    mask_frame: Mapping[str, np.ndarray],
    tracks_yx: np.ndarray,
    visibility: np.ndarray,
    query_points_yx: np.ndarray,
    intrinsics: Any,
    c2w: np.ndarray,
    mask_radius_outlier_filter: bool = True,
    mask_radius_outlier_radius_m: float = 0.01,
    mask_radius_outlier_nb_points: int = 40,
    source_timestamp_s: float | None = None,
    source_frame_index: int | None = None,
    source_step: int | None = None,
) -> PreparedPhysTwinFrame:
    rgb = np.ascontiguousarray(np.asarray(rgb_frame, dtype=np.uint8))
    depth = np.asarray(depth_m, dtype=np.float32)
    points, colors = dense_world_pcd_grid(
        depth_m=depth,
        color_rgb_u8=rgb,
        intrinsics=intrinsics,
        c2w=c2w,
    )
    depth_valid_masks = apply_depth_validity_to_mask_frame(mask_frame, depth)
    processed = apply_radius_outlier_to_mask_frame(
        depth_valid_masks,
        points,
        enabled=bool(mask_radius_outlier_filter),
        radius_m=float(mask_radius_outlier_radius_m),
        nb_points=int(mask_radius_outlier_nb_points),
    )
    tracks = np.ascontiguousarray(np.asarray(tracks_yx, dtype=np.float32).reshape(-1, 2))
    vis = np.ascontiguousarray(np.asarray(visibility, dtype=bool).reshape(-1))
    queries = np.ascontiguousarray(np.asarray(query_points_yx, dtype=np.float32).reshape(-1, 2))
    if tracks.shape[0] != queries.shape[0] or vis.shape[0] != queries.shape[0]:
        raise ValueError(
            "prepared PhysTwin frame requires full tracks/visibility matching query_points_yx; "
            f"tracks={tracks.shape[0]} visibility={vis.shape[0]} queries={queries.shape[0]}"
        )
    return PreparedPhysTwinFrame(
        seq=int(seq),
        rgb_frame=rgb,
        processed_mask_frame=processed,
        pcd_points=np.ascontiguousarray(points, dtype=np.float32),
        pcd_colors=np.ascontiguousarray(colors, dtype=np.uint8),
        tracks_yx=tracks,
        visibility=vis,
        query_points_yx=queries,
        source_timestamp_s=None if source_timestamp_s is None else float(source_timestamp_s),
        source_frame_index=None if source_frame_index is None else int(source_frame_index),
        source_step=None if source_step is None else int(source_step),
    )


def _optional_float_payload(value: float | None) -> np.ndarray:
    return np.asarray([np.nan if value is None else float(value)], dtype=np.float64)


def _optional_int_payload(value: int | None) -> np.ndarray:
    return np.asarray([-1 if value is None else int(value)], dtype=np.int64)


def write_prepared_phystwin_frame(path: str | Path, frame: PreparedPhysTwinFrame) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(f"{output.name}.tmp")
    masks = normalize_processed_mask_frame(frame.processed_mask_frame)
    mask_keys = np.asarray(sorted(masks.keys()))
    payload: dict[str, Any] = {
        "seq": np.asarray([int(frame.seq)], dtype=np.int64),
        "rgb_frame": np.ascontiguousarray(frame.rgb_frame, dtype=np.uint8),
        "pcd_points": np.ascontiguousarray(frame.pcd_points, dtype=np.float32),
        "pcd_colors": np.ascontiguousarray(frame.pcd_colors, dtype=np.uint8),
        "tracks_yx": np.ascontiguousarray(frame.tracks_yx, dtype=np.float32),
        "visibility": np.ascontiguousarray(frame.visibility, dtype=bool),
        "query_points_yx": np.ascontiguousarray(frame.query_points_yx, dtype=np.float32),
        "mask_keys": mask_keys,
        "source_timestamp_s": _optional_float_payload(frame.source_timestamp_s),
        "source_frame_index": _optional_int_payload(frame.source_frame_index),
        "source_step": _optional_int_payload(frame.source_step),
    }
    for key in mask_keys:
        payload[f"mask_{str(key)}"] = np.ascontiguousarray(masks[str(key)], dtype=bool)
    with tmp.open("wb") as handle:
        np.savez(handle, **payload)
    tmp.replace(output)
    return output


def _none_if_negative(value: int) -> int | None:
    return None if int(value) < 0 else int(value)


def load_prepared_phystwin_frame(path: str | Path) -> PreparedPhysTwinFrame:
    payload = np.load(Path(path), allow_pickle=False)
    mask_frame: dict[str, np.ndarray] = {}
    for key in payload["mask_keys"]:
        name = str(key)
        mask_frame[name] = np.ascontiguousarray(np.asarray(payload[f"mask_{name}"], dtype=bool))
    timestamp = float(payload["source_timestamp_s"][0])
    return PreparedPhysTwinFrame(
        seq=int(payload["seq"][0]),
        rgb_frame=np.ascontiguousarray(np.asarray(payload["rgb_frame"], dtype=np.uint8)),
        processed_mask_frame=normalize_processed_mask_frame(mask_frame),
        pcd_points=np.ascontiguousarray(np.asarray(payload["pcd_points"], dtype=np.float32)),
        pcd_colors=np.ascontiguousarray(np.asarray(payload["pcd_colors"], dtype=np.uint8)),
        tracks_yx=np.ascontiguousarray(np.asarray(payload["tracks_yx"], dtype=np.float32).reshape(-1, 2)),
        visibility=np.ascontiguousarray(np.asarray(payload["visibility"], dtype=bool).reshape(-1)),
        query_points_yx=np.ascontiguousarray(np.asarray(payload["query_points_yx"], dtype=np.float32).reshape(-1, 2)),
        source_timestamp_s=None if not np.isfinite(timestamp) else timestamp,
        source_frame_index=_none_if_negative(int(payload["source_frame_index"][0])),
        source_step=_none_if_negative(int(payload["source_step"][0])),
    )


def _round_tracks_to_indices(tracks_yx: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tracks = np.asarray(tracks_yx, dtype=np.float32).reshape(-1, 2)
    yy = np.rint(tracks[:, 0]).astype(np.int64)
    xx = np.rint(tracks[:, 1]).astype(np.int64)
    finite = np.isfinite(tracks).all(axis=1)
    in_bounds = finite & (yy >= 0) & (yy < int(shape[0])) & (xx >= 0) & (xx < int(shape[1]))
    return yy, xx, in_bounds


def build_track_process_input(
    *,
    tracks_yx: np.ndarray,
    visibility: np.ndarray,
    processed_masks: Sequence[Sequence[Mapping[str, Any]]],
    pcd_points: np.ndarray,
    pcd_colors: np.ndarray,
) -> dict[str, np.ndarray]:
    tracks = np.asarray(tracks_yx, dtype=np.float32)
    vis = np.asarray(visibility, dtype=bool)
    points_grid = np.asarray(pcd_points, dtype=np.float32)
    colors_grid = np.asarray(pcd_colors)
    if tracks.ndim != 3 or tracks.shape[-1] != 2:
        raise ValueError("tracks_yx must have shape T,N,2")
    if vis.shape != tracks.shape[:2]:
        raise ValueError("visibility must have shape T,N")
    if points_grid.ndim != 5 or points_grid.shape[1] < 1 or points_grid.shape[-1] != 3:
        raise ValueError("pcd_points must have shape T,C,H,W,3")
    if colors_grid.shape != points_grid.shape:
        raise ValueError("pcd_colors must match pcd_points shape")
    if len(processed_masks) != tracks.shape[0] or points_grid.shape[0] != tracks.shape[0]:
        raise ValueError("processed_masks, pcd_points, and tracks must share T")

    first = normalize_processed_mask_frame(processed_masks[0][0])
    yy0, xx0, in_bounds0 = _round_tracks_to_indices(tracks[0], first["object"].shape)
    object_label = np.zeros((tracks.shape[1],), dtype=bool)
    controller_label = np.zeros((tracks.shape[1],), dtype=bool)
    visible0 = vis[0] & in_bounds0
    if np.any(visible0):
        object_label[visible0] = first["object"][yy0[visible0], xx0[visible0]]
        controller_label[visible0] = first["controller"][yy0[visible0], xx0[visible0]]

    semantic_vis = np.array(vis, dtype=bool, copy=True)
    for frame_idx in range(tracks.shape[0]):
        masks = normalize_processed_mask_frame(processed_masks[frame_idx][0])
        yy, xx, in_bounds = _round_tracks_to_indices(tracks[frame_idx], masks["object"].shape)
        valid = semantic_vis[frame_idx] & in_bounds
        inside = np.zeros((tracks.shape[1],), dtype=bool)
        object_idx = valid & object_label
        controller_idx = valid & controller_label
        if np.any(object_idx):
            inside[object_idx] = masks["object"][yy[object_idx], xx[object_idx]]
        if np.any(controller_idx):
            inside[controller_idx] |= masks["controller"][yy[controller_idx], xx[controller_idx]]
        semantic_vis[frame_idx] &= inside

    track_points = np.zeros((tracks.shape[0], tracks.shape[1], 3), dtype=np.float32)
    track_colors = np.zeros((tracks.shape[0], tracks.shape[1], 3), dtype=np.float32)
    for frame_idx in range(tracks.shape[0]):
        height, width = points_grid.shape[2:4]
        yy, xx, in_bounds = _round_tracks_to_indices(tracks[frame_idx], (height, width))
        valid = semantic_vis[frame_idx] & in_bounds
        if np.any(valid):
            sampled_points = points_grid[frame_idx, 0, yy[valid], xx[valid]]
            finite_depth = np.isfinite(sampled_points).all(axis=1)
            nonzero_depth = np.linalg.norm(sampled_points, axis=1) > 1e-9
            valid_indices = np.flatnonzero(valid)
            invalid_indices = valid_indices[~(finite_depth & nonzero_depth)]
            if len(invalid_indices):
                semantic_vis[frame_idx, invalid_indices] = False
            keep_indices = valid_indices[finite_depth & nonzero_depth]
            if len(keep_indices):
                track_points[frame_idx, keep_indices] = points_grid[frame_idx, 0, yy[keep_indices], xx[keep_indices]]
                track_colors[frame_idx, keep_indices] = colors_grid[frame_idx, 0, yy[keep_indices], xx[keep_indices]].astype(np.float32) / 255.0

    object_indices = np.flatnonzero(object_label)
    controller_indices = np.flatnonzero(controller_label)
    return {
        "query_is_object": object_label,
        "query_is_controller": controller_label,
        "object_query_indices": object_indices.astype(np.int64),
        "controller_query_indices": controller_indices.astype(np.int64),
        "object_points": track_points[:, object_indices, :],
        "object_colors": track_colors[:, object_indices, :],
        "object_visibilities": semantic_vis[:, object_indices],
        "controller_points": track_points[:, controller_indices, :],
        "controller_colors": track_colors[:, controller_indices, :],
        "controller_visibilities": semantic_vis[:, controller_indices],
    }


def _motion_valid_for_class(
    points: np.ndarray,
    visibilities: np.ndarray,
    *,
    neighbor_dist: float,
    min_neighbors: int,
    motion_similarity_m: float,
    once_false_mask: bool,
) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float32)
    vis = np.asarray(visibilities, dtype=bool)
    if pts.ndim != 3 or pts.shape[-1] != 3:
        raise ValueError("points must have shape T,N,3")
    if vis.shape != pts.shape[:2]:
        raise ValueError("visibilities must have shape T,N")
    motions_valid = np.zeros_like(vis, dtype=bool)
    if pts.shape[0] > 1:
        motions_valid[:-1] = vis[:-1] & vis[1:]
    global_mask = np.prod(vis, axis=0).astype(bool) if once_false_mask and vis.size else np.ones((pts.shape[1],), dtype=bool)
    if pts.shape[1] == 0:
        return motions_valid, global_mask
    motions = np.zeros_like(pts, dtype=np.float32)
    motions[:-1] = pts[1:] - pts[:-1]
    from scipy.spatial import cKDTree

    for frame_idx in range(max(0, pts.shape[0] - 1)):
        if once_false_mask:
            motions_valid[frame_idx] &= global_mask
        if not np.any(motions_valid[frame_idx]):
            continue
        tree = cKDTree(pts[frame_idx])
        all_neighbors = tree.query_ball_point(
            pts[frame_idx],
            r=float(neighbor_dist),
            workers=-1,
            return_sorted=False,
        )
        for query_idx in range(pts.shape[1]):
            if once_false_mask and not global_mask[query_idx]:
                motions_valid[frame_idx, query_idx] = False
                continue
            if not motions_valid[frame_idx, query_idx]:
                continue
            neighbors = np.asarray(all_neighbors[query_idx], dtype=np.int64)
            neighbors = neighbors[motions_valid[frame_idx, neighbors]]
            if len(neighbors) < int(min_neighbors):
                motions_valid[frame_idx, query_idx] = False
                if once_false_mask:
                    global_mask[query_idx] = False
                continue
            motion_diff = np.linalg.norm(motions[frame_idx, query_idx] - motions[frame_idx, neighbors], axis=1)
            if int(np.count_nonzero(motion_diff < float(motion_similarity_m))) < 0.5 * float(len(neighbors)):
                motions_valid[frame_idx, query_idx] = False
                if once_false_mask:
                    global_mask[query_idx] = False
        if once_false_mask:
            motions_valid[frame_idx] &= global_mask
    return motions_valid, global_mask.astype(bool, copy=False)


def apply_phystwin_motion_filters(
    track_data: Mapping[str, np.ndarray],
    *,
    neighbor_dist: float = 0.01,
    min_neighbors: int = 5,
    motion_similarity_m: float = 0.005,
) -> dict[str, np.ndarray]:
    result = {key: np.asarray(value).copy() for key, value in track_data.items()}
    object_valid, _object_mask = _motion_valid_for_class(
        result["object_points"],
        result["object_visibilities"],
        neighbor_dist=float(neighbor_dist),
        min_neighbors=int(min_neighbors),
        motion_similarity_m=float(motion_similarity_m),
        once_false_mask=False,
    )
    controller_valid, controller_mask = _motion_valid_for_class(
        result["controller_points"],
        result["controller_visibilities"],
        neighbor_dist=float(neighbor_dist),
        min_neighbors=int(min_neighbors),
        motion_similarity_m=float(motion_similarity_m),
        once_false_mask=True,
    )
    result["object_motions_valid"] = object_valid
    result["controller_motions_valid"] = controller_valid
    result["controller_mask"] = controller_mask
    return result


def _farthest_point_sample_indices(points_xyz: np.ndarray, count: int) -> np.ndarray:
    pts = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    target = int(count)
    if target < 0:
        raise ValueError("count must be >= 0")
    if target == 0:
        return np.empty((0,), dtype=np.int64)
    if len(pts) < target:
        raise RuntimeError(f"controller FPS requires at least {target} points; got {len(pts)}")
    selected = [0]
    min_dist2 = np.sum((pts - pts[0]) ** 2, axis=1)
    for _ in range(1, target):
        idx = int(np.argmax(min_dist2))
        selected.append(idx)
        dist2 = np.sum((pts - pts[idx]) ** 2, axis=1)
        min_dist2 = np.minimum(min_dist2, dist2)
    return np.asarray(selected, dtype=np.int64)


class StreamingControllerAnchorSelector:
    """Keep controller handle order stable across Demo v4 online chunks."""

    def __init__(
        self,
        *,
        count: int = 30,
        revive_knn: int = 4,
    ) -> None:
        if int(count) < 0:
            raise ValueError("count must be >= 0")
        if int(revive_knn) <= 0:
            raise ValueError("revive_knn must be positive")
        self.count = int(count)
        self.revive_knn = int(revive_knn)
        self._initial_query_indices: np.ndarray | None = None
        self._active_query_indices: np.ndarray | None = None
        self._last_points: np.ndarray | None = None

    @property
    def initialized(self) -> bool:
        return self._initial_query_indices is not None

    def select(self, track_data: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        result = {key: np.asarray(value).copy() for key, value in track_data.items()}
        points = np.asarray(result["controller_points"], dtype=np.float32)
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError("controller_points must have shape T,N,3")
        frame_count, candidate_count, _ = points.shape
        query_indices = self._query_indices(result, candidate_count)
        valid_candidates = self._valid_candidate_mask(result, points)

        if not self.initialized:
            selected = self._initial_selection(points, valid_candidates)
            output = np.ascontiguousarray(points[:, selected, :], dtype=np.float32)
            initial_query_indices = np.ascontiguousarray(query_indices[selected], dtype=np.int64)
            self._initial_query_indices = initial_query_indices
            self._active_query_indices = initial_query_indices.copy()
            self._last_points = np.ascontiguousarray(output[-1], dtype=np.float32)
            return self._with_anchor_payload(
                result,
                output,
                selected,
                np.asarray(["direct"] * self.count),
                active_query_indices=initial_query_indices,
            )

        assert self._initial_query_indices is not None
        assert self._active_query_indices is not None
        assert self._last_points is not None

        output = np.zeros((frame_count, self.count, 3), dtype=np.float32)
        selected = np.full((self.count,), -1, dtype=np.int64)
        statuses = np.full((self.count,), "missing", dtype="<U8")
        active_query_indices = self._active_query_indices.copy()
        used_candidates: set[int] = set()
        query_to_candidate = {int(query_id): int(idx) for idx, query_id in enumerate(query_indices.tolist())}

        direct_mask = np.zeros((self.count,), dtype=bool)
        for anchor_idx in range(self.count):
            candidate_idx = self._direct_candidate_for_anchor(
                anchor_idx,
                query_to_candidate=query_to_candidate,
                valid_candidates=valid_candidates,
            )
            if candidate_idx is None or candidate_idx in used_candidates:
                continue
            output[:, anchor_idx, :] = points[:, candidate_idx, :]
            selected[anchor_idx] = int(candidate_idx)
            active_query_indices[anchor_idx] = int(query_indices[candidate_idx])
            statuses[anchor_idx] = "direct"
            used_candidates.add(int(candidate_idx))
            direct_mask[anchor_idx] = True

        for anchor_idx in np.flatnonzero(~direct_mask):
            predicted_first = self._predict_first_frame(anchor_idx, output, direct_mask)
            revived, candidate_idx = self._revive_from_neighbors(
                points,
                valid_candidates,
                used_candidates=used_candidates,
                predicted_first=predicted_first,
            )
            output[:, anchor_idx, :] = revived
            if candidate_idx is None:
                statuses[anchor_idx] = "fallback"
                active_query_indices[anchor_idx] = -1
            else:
                selected[anchor_idx] = int(candidate_idx)
                active_query_indices[anchor_idx] = int(query_indices[candidate_idx])
                statuses[anchor_idx] = "revived"
                used_candidates.add(int(candidate_idx))

        self._active_query_indices = np.ascontiguousarray(active_query_indices, dtype=np.int64)
        self._last_points = np.ascontiguousarray(output[-1], dtype=np.float32)
        return self._with_anchor_payload(
            result,
            output,
            selected,
            statuses,
            active_query_indices=active_query_indices,
        )

    def _query_indices(self, result: Mapping[str, np.ndarray], candidate_count: int) -> np.ndarray:
        value = result.get("controller_query_indices")
        if value is None:
            return np.arange(candidate_count, dtype=np.int64)
        query_indices = np.asarray(value, dtype=np.int64).reshape(-1)
        if query_indices.shape[0] != int(candidate_count):
            raise ValueError(
                "controller_query_indices must match controller candidate count; "
                f"got {query_indices.shape[0]} for {candidate_count}"
            )
        return np.ascontiguousarray(query_indices, dtype=np.int64)

    def _valid_candidate_mask(self, result: Mapping[str, np.ndarray], points: np.ndarray) -> np.ndarray:
        candidate_count = points.shape[1]
        mask = np.asarray(
            result.get("controller_mask", np.ones((candidate_count,), dtype=bool)),
            dtype=bool,
        ).reshape(-1)
        if mask.shape[0] != candidate_count:
            raise ValueError("controller_mask must match controller candidate count")
        finite = np.isfinite(points).all(axis=(0, 2))
        nonzero = np.all(np.linalg.norm(points, axis=2) > 1e-9, axis=0)
        return np.ascontiguousarray(mask & finite & nonzero, dtype=bool)

    def _initial_selection(self, points: np.ndarray, valid_candidates: np.ndarray) -> np.ndarray:
        valid_indices = np.flatnonzero(valid_candidates)
        candidates = points[:, valid_indices, :]
        sample_indices = _farthest_point_sample_indices(candidates[0], self.count)
        return np.ascontiguousarray(valid_indices[sample_indices], dtype=np.int64)

    def _direct_candidate_for_anchor(
        self,
        anchor_idx: int,
        *,
        query_to_candidate: Mapping[int, int],
        valid_candidates: np.ndarray,
    ) -> int | None:
        assert self._initial_query_indices is not None
        assert self._active_query_indices is not None
        for query_id in (
            int(self._initial_query_indices[anchor_idx]),
            int(self._active_query_indices[anchor_idx]),
        ):
            candidate_idx = query_to_candidate.get(query_id)
            if candidate_idx is not None and bool(valid_candidates[candidate_idx]):
                return int(candidate_idx)
        return None

    def _predict_first_frame(
        self,
        anchor_idx: int,
        output: np.ndarray,
        direct_mask: np.ndarray,
    ) -> np.ndarray:
        assert self._last_points is not None
        previous = np.asarray(self._last_points, dtype=np.float32)
        base = previous[int(anchor_idx)]
        direct_indices = np.flatnonzero(direct_mask)
        if len(direct_indices) == 0:
            return np.ascontiguousarray(base, dtype=np.float32)
        direct_previous = previous[direct_indices]
        direct_displacement = output[0, direct_indices, :] - direct_previous
        distances = np.linalg.norm(direct_previous - base[None, :], axis=1)
        order = np.argsort(distances)[: max(1, min(self.revive_knn, len(distances)))]
        weights = 1.0 / np.maximum(distances[order], 1e-6)
        weights = weights / np.sum(weights)
        displacement = np.sum(direct_displacement[order] * weights[:, None], axis=0)
        return np.ascontiguousarray(base + displacement.astype(np.float32), dtype=np.float32)

    def _revive_from_neighbors(
        self,
        points: np.ndarray,
        valid_candidates: np.ndarray,
        *,
        used_candidates: set[int],
        predicted_first: np.ndarray,
    ) -> tuple[np.ndarray, int | None]:
        candidates = np.flatnonzero(valid_candidates)
        if used_candidates:
            candidates = np.asarray(
                [int(idx) for idx in candidates.tolist() if int(idx) not in used_candidates],
                dtype=np.int64,
            )
        if len(candidates) == 0:
            held = np.repeat(np.asarray(predicted_first, dtype=np.float32)[None, :], points.shape[0], axis=0)
            return np.ascontiguousarray(held, dtype=np.float32), None

        candidate_first = points[0, candidates, :]
        distances = np.linalg.norm(candidate_first - np.asarray(predicted_first, dtype=np.float32)[None, :], axis=1)
        order = np.argsort(distances)[: max(1, min(self.revive_knn, len(candidates)))]
        selected = candidates[order]
        weights = 1.0 / np.maximum(distances[order], 1e-6)
        weights = weights / np.sum(weights)
        trajectory = np.sum(points[:, selected, :] * weights[None, :, None], axis=1)
        trajectory = trajectory + (np.asarray(predicted_first, dtype=np.float32) - trajectory[0])[None, :]
        return np.ascontiguousarray(trajectory, dtype=np.float32), int(selected[0])

    def _with_anchor_payload(
        self,
        result: dict[str, np.ndarray],
        output: np.ndarray,
        selected: np.ndarray,
        statuses: np.ndarray,
        *,
        active_query_indices: np.ndarray,
    ) -> dict[str, np.ndarray]:
        assert self._initial_query_indices is not None
        result["controller_points"] = np.ascontiguousarray(output, dtype=np.float32)
        result["controller_fps_indices"] = np.ascontiguousarray(np.asarray(selected, dtype=np.int64))
        result["controller_anchor_query_indices"] = np.ascontiguousarray(self._initial_query_indices, dtype=np.int64)
        result["controller_anchor_active_query_indices"] = np.ascontiguousarray(
            np.asarray(active_query_indices, dtype=np.int64)
        )
        result["controller_anchor_status"] = np.asarray(statuses, dtype="<U8")
        return result


def select_final_controller_points(track_data: Mapping[str, np.ndarray], *, count: int = 30) -> dict[str, np.ndarray]:
    result = {key: np.asarray(value).copy() for key, value in track_data.items()}
    mask = np.asarray(result.get("controller_mask", np.ones((result["controller_points"].shape[1],), dtype=bool)), dtype=bool)
    valid_indices = np.flatnonzero(mask)
    candidates = result["controller_points"][:, valid_indices, :]
    sample_indices = _farthest_point_sample_indices(candidates[0], int(count))
    final_indices = valid_indices[sample_indices]
    result["controller_points"] = np.ascontiguousarray(result["controller_points"][:, final_indices, :], dtype=np.float32)
    result["controller_fps_indices"] = np.asarray(final_indices, dtype=np.int64)
    return result


def sample_object_first_frame_volume(
    track_data: Mapping[str, np.ndarray],
    *,
    volume_sample_size: float = 0.005,
) -> dict[str, np.ndarray]:
    result = {key: np.asarray(value).copy() for key, value in track_data.items()}
    object_points = np.asarray(result["object_points"], dtype=np.float32)
    if object_points.ndim != 3 or object_points.shape[-1] != 3:
        raise ValueError("object_points must have shape T,N,3")
    if object_points.shape[1] == 0:
        return result
    min_bound = np.min(object_points[0], axis=0)
    grid_seen: set[tuple[int, int, int]] = set()
    keep: list[int] = []
    voxel = float(volume_sample_size)
    if voxel <= 0.0:
        raise ValueError("volume_sample_size must be positive")
    for idx, point in enumerate(object_points[0]):
        grid_index = tuple(np.floor((point - min_bound) / np.float32(voxel)).astype(np.int64).tolist())
        if grid_index in grid_seen:
            continue
        grid_seen.add(grid_index)
        keep.append(int(idx))
    indices = np.asarray(keep, dtype=np.int64)
    for key in ("object_points", "object_colors"):
        if key in result:
            result[key] = np.ascontiguousarray(result[key][:, indices, :])
    for key in ("object_visibilities", "object_motions_valid"):
        if key in result:
            result[key] = np.ascontiguousarray(result[key][:, indices])
    result["object_volume_sample_indices"] = indices
    return result


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _load_frame_masks(path: Path) -> dict[str, np.ndarray]:
    payload = np.load(path, allow_pickle=False)
    frame = {
        "object": np.asarray(payload["object_mask"], dtype=bool),
        "controller": np.asarray(payload["controller_mask"], dtype=bool),
    }
    if "hand_a_mask" in payload:
        frame["hand_a"] = np.asarray(payload["hand_a_mask"], dtype=bool)
    if "hand_b_mask" in payload:
        frame["hand_b"] = np.asarray(payload["hand_b_mask"], dtype=bool)
    return frame


def _write_tracking_npz(
    output_dir: Path,
    *,
    tracks_yx: np.ndarray,
    visibility: np.ndarray,
    query_txy: np.ndarray,
) -> None:
    tracking_dir = output_dir / "tracking"
    compat_dir = output_dir / PHYSTWIN_COMPATIBILITY_PATH_NAME
    tracking_dir.mkdir(parents=True, exist_ok=True)
    compat_dir.mkdir(parents=True, exist_ok=True)
    for directory in (tracking_dir, compat_dir):
        np.savez(
            directory / "0.npz",
            tracks=np.ascontiguousarray(tracks_yx, dtype=np.float32),
            visibility=np.ascontiguousarray(visibility, dtype=bool),
            queries_txy=np.ascontiguousarray(query_txy, dtype=np.float32),
        )


def _depth_path_for_row(capture_dir: Path, row: Mapping[str, Any]) -> Path:
    if "depth_color_m_path" in row:
        return capture_dir / str(row["depth_color_m_path"])
    if "ffs_depth_path" in row:
        return capture_dir / str(row["ffs_depth_path"])
    raise KeyError("headless capture row must contain depth_color_m_path or legacy ffs_depth_path")


def _finalize_prepared_only_headless_capture(
    capture: Path,
    out: Path,
    *,
    metadata: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    prepared_paths = [row.get("prepared_phystwin_frame_path") for row in rows]
    missing = [idx for idx, value in enumerate(prepared_paths) if value is None]
    if missing:
        raise KeyError(
            "prepared-only headless capture rows must contain prepared_phystwin_frame_path; "
            f"missing row indices={missing[:5]}"
        )
    first_frame = load_prepared_phystwin_frame(capture / str(prepared_paths[0]))
    manifest = {
        "compatibility_target": COMPATIBILITY_TARGET_PHYSTWIN,
        "tracking_product_backend": TRACKING_PRODUCT_BACKEND_PHYSTWIN_STRICT,
        "tracker_backend": "tapnextpp",
        "mask_backend": "edgetam",
        "depth_backend": str(metadata.get("depth_backend") or metadata.get("depth_source", "")),
        "depth_source_internal": str(
            metadata.get("depth_source_internal")
            or metadata.get("depth_source")
            or metadata.get("depth_backend", "")
        ),
        "execution_mode": PHYSTWIN_STRICT_EXECUTION_MODE,
        "compatibility_path_name": PHYSTWIN_COMPATIBILITY_PATH_NAME,
        "not_actual_cotracker": True,
        "camera_count": 1,
        "frame_count": int(len(rows)),
        "query_count": int(first_frame.query_points_yx.shape[0]),
        "headless_prepared_only": True,
        "chunk_materialization_source": "prepared_phystwin_frame",
        "prepared_frame_count": int(len(prepared_paths)),
        "prepared_frames_dir": "prepared_phystwin",
        "processed_masks_path": None,
        "track_process_data_path": None,
        "final_data_path": None,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _write_pcd_frames(
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    capture_dir: Path,
    metadata: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    pcd_dir = output_dir / "pcd"
    pcd_dir.mkdir(parents=True, exist_ok=True)
    c2w = np.asarray(metadata.get("camera_to_world_c2w") or np.eye(4, dtype=np.float32), dtype=np.float32).reshape(4, 4)
    intrinsics = metadata["intrinsics"]
    all_points: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []
    for idx, row in enumerate(rows):
        depth = np.load(_depth_path_for_row(capture_dir, row))
        rgb = _load_rgb(capture_dir / str(row["rgb_path"]))
        points, colors = dense_world_pcd_grid(
            depth_m=depth,
            color_rgb_u8=rgb,
            intrinsics=intrinsics,
            c2w=c2w,
        )
        np.savez(pcd_dir / f"{idx}.npz", points=points, colors=colors)
        all_points.append(points)
        all_colors.append(colors)
    if not all_points:
        return np.empty((0, 1, 0, 0, 3), dtype=np.float32), np.empty((0, 1, 0, 0, 3), dtype=np.uint8)
    return np.stack(all_points, axis=0), np.stack(all_colors, axis=0)


def _open_video_writer(path: Path, *, size: tuple[int, int], fps: float = 30.0):
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = int(size[0]), int(size[1])
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer for {path}")
    return writer


def _render_tracking_2d_video(
    path: Path,
    *,
    capture_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    tracks_yx: np.ndarray,
    visibility: np.ndarray,
    query_is_object: np.ndarray,
    query_is_controller: np.ndarray,
    size: tuple[int, int] = (848, 480),
) -> None:
    import cv2

    writer = _open_video_writer(path, size=size)
    width, height = int(size[0]), int(size[1])
    is_object = np.asarray(query_is_object, dtype=bool).reshape(-1)
    is_controller = np.asarray(query_is_controller, dtype=bool).reshape(-1)
    for frame_idx, row in enumerate(rows):
        rgb = _load_rgb(capture_dir / str(row["rgb_path"]))
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        src_h, src_w = frame.shape[:2]
        if (src_w, src_h) != (width, height):
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        sx = float(width) / max(1.0, float(src_w))
        sy = float(height) / max(1.0, float(src_h))
        tracks = np.asarray(tracks_yx[frame_idx], dtype=np.float32)
        vis = np.asarray(visibility[frame_idx], dtype=bool)
        finite = np.isfinite(tracks).all(axis=1)
        visible = np.flatnonzero(vis & finite)
        for idx in visible:
            y = int(round(float(tracks[idx, 0]) * sy))
            x = int(round(float(tracks[idx, 1]) * sx))
            if x < 0 or x >= width or y < 0 or y >= height:
                continue
            color = (60, 220, 60) if idx < len(is_object) and is_object[idx] else (40, 80, 255)
            if idx < len(is_controller) and not is_object[idx] and not is_controller[idx]:
                color = (220, 220, 220)
            cv2.circle(frame, (x, y), 2, color, -1, lineType=cv2.LINE_AA)
        cv2.putText(
            frame,
            f"tracking_2d frame={frame_idx} visible={len(visible)}",
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)
    writer.release()


def _world_xy_bounds(*arrays: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    chunks: list[np.ndarray] = []
    for arr in arrays:
        pts = np.asarray(arr, dtype=np.float32).reshape(-1, 3)
        finite = np.isfinite(pts).all(axis=1) & (np.linalg.norm(pts, axis=1) > 0.0)
        if np.any(finite):
            chunks.append(pts[finite, :2])
    if not chunks:
        return np.array([-1.0, -1.0], dtype=np.float32), np.array([1.0, 1.0], dtype=np.float32)
    xy = np.concatenate(chunks, axis=0)
    lo = np.min(xy, axis=0)
    hi = np.max(xy, axis=0)
    span = np.maximum(hi - lo, np.float32(1e-3))
    pad = span * np.float32(0.08)
    return lo - pad, hi + pad


def _draw_world_points(
    frame: np.ndarray,
    points: np.ndarray,
    *,
    bounds: tuple[np.ndarray, np.ndarray],
    color_bgr: tuple[int, int, int],
    radius: int,
) -> int:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    finite = np.isfinite(pts).all(axis=1) & (np.linalg.norm(pts, axis=1) > 0.0)
    pts = pts[finite]
    if len(pts) == 0:
        return 0
    lo, hi = bounds
    width = frame.shape[1]
    height = frame.shape[0]
    span = np.maximum(hi - lo, np.float32(1e-6))
    px = np.clip(((pts[:, 0] - lo[0]) / span[0] * (width - 60) + 30).astype(np.int64), 0, width - 1)
    py = np.clip(((pts[:, 1] - lo[1]) / span[1] * (height - 80) + 50).astype(np.int64), 0, height - 1)
    py = height - 1 - py
    import cv2

    for x, y in zip(px, py):
        cv2.circle(frame, (int(x), int(y)), int(radius), color_bgr, -1, lineType=cv2.LINE_AA)
    return int(len(pts))


def _render_world_track_video(
    path: Path,
    *,
    object_points: np.ndarray,
    object_valid: np.ndarray,
    controller_points: np.ndarray,
    title: str,
    size: tuple[int, int] = (640, 480),
) -> None:
    import cv2

    writer = _open_video_writer(path, size=size)
    frame_count = max(int(np.asarray(object_points).shape[0]), int(np.asarray(controller_points).shape[0]), 1)
    bounds = _world_xy_bounds(object_points, controller_points)
    width, height = int(size[0]), int(size[1])
    for frame_idx in range(frame_count):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        obj = np.asarray(object_points[min(frame_idx, max(0, object_points.shape[0] - 1))], dtype=np.float32).reshape(-1, 3)
        valid = np.asarray(object_valid[min(frame_idx, max(0, object_valid.shape[0] - 1))], dtype=bool).reshape(-1)
        if len(valid) == len(obj):
            obj = obj[valid]
        ctrl = np.asarray(controller_points[min(frame_idx, max(0, controller_points.shape[0] - 1))], dtype=np.float32).reshape(-1, 3)
        obj_count = _draw_world_points(frame, obj, bounds=bounds, color_bgr=(50, 220, 80), radius=2)
        ctrl_count = _draw_world_points(frame, ctrl, bounds=bounds, color_bgr=(40, 40, 255), radius=5)
        cv2.putText(frame, title, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"frame={frame_idx} object={obj_count} controller={ctrl_count}",
            (18, 64),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (210, 230, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)
    writer.release()


def _render_final_pcd_video(
    path: Path,
    *,
    object_points: np.ndarray,
    controller_points: np.ndarray,
    size: tuple[int, int] = (640, 480),
) -> None:
    object_valid = np.ones(np.asarray(object_points).shape[:2], dtype=bool)
    _render_world_track_video(
        path,
        object_points=object_points,
        object_valid=object_valid,
        controller_points=controller_points,
        title="final_pcd 5mm object sample + controller FPS30",
        size=size,
    )


def _render_empty_video(path: Path, *, frame_count: int, label: str, size: tuple[int, int] = (640, 360)) -> None:
    import cv2

    writer = _open_video_writer(path, size=size)
    width, height = int(size[0]), int(size[1])
    count = max(1, int(frame_count))
    for frame_idx in range(count):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(frame, label, (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"frame={frame_idx}", (24, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 220, 255), 2, cv2.LINE_AA)
        writer.write(frame)
    writer.release()


def finalize_headless_capture(
    capture_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    capture = Path(capture_dir)
    out = Path(output_dir) if output_dir is not None else capture / "phystwin_like"
    out.mkdir(parents=True, exist_ok=True)
    metadata_path = capture / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    rows = _read_jsonl(capture / "frames.jsonl")
    if not rows:
        raise RuntimeError(f"no headless frames found in {capture / 'frames.jsonl'}")
    if bool(metadata.get("headless_prepared_only")) or (
        "prepared_phystwin_frame_path" in rows[0] and "mask_path" not in rows[0]
    ):
        return _finalize_prepared_only_headless_capture(capture, out, metadata=metadata, rows=rows)

    mask_frames = [_load_frame_masks(capture / str(row["mask_path"])) for row in rows]
    processed_mask_path = write_processed_masks(out, mask_frames)

    trajectory_payloads = [np.load(capture / str(row["query_trajectory_path"]), allow_pickle=False) for row in rows]
    query_points_yx = np.asarray(trajectory_payloads[0]["query_points_yx"], dtype=np.float32)
    query_txy = np.zeros((len(query_points_yx), 3), dtype=np.float32)
    query_txy[:, 1] = query_points_yx[:, 1]
    query_txy[:, 2] = query_points_yx[:, 0]
    tracks: list[np.ndarray] = []
    visibility: list[np.ndarray] = []
    for payload in trajectory_payloads:
        track_key = "all_tracks_yx" if "all_tracks_yx" in payload.files else "tracks_yx"
        vis_key = "all_tracker_visibility" if "all_tracker_visibility" in payload.files else "visibility"
        current_tracks = np.asarray(payload[track_key], dtype=np.float32).reshape(-1, 2)
        current_vis = np.asarray(payload[vis_key], dtype=bool).reshape(-1)
        if current_tracks.shape[0] != len(query_points_yx):
            raise RuntimeError(
                "strict PhysTwin product requires full per-query tracks; "
                f"got {current_tracks.shape[0]} tracks for {len(query_points_yx)} queries at seq={int(payload['seq'][0])}"
            )
        tracks.append(current_tracks)
        visibility.append(current_vis)
    tracks_yx = np.stack(tracks, axis=0)
    tracker_visibility = np.stack(visibility, axis=0)
    _write_tracking_npz(out, tracks_yx=tracks_yx, visibility=tracker_visibility, query_txy=query_txy)
    pcd_points, pcd_colors = _write_pcd_frames(out, rows, capture_dir=capture, metadata=metadata)

    processed_masks = [[normalize_processed_mask_frame(frame)] for frame in mask_frames]
    track_input = build_track_process_input(
        tracks_yx=tracks_yx,
        visibility=tracker_visibility,
        processed_masks=processed_masks,
        pcd_points=pcd_points,
        pcd_colors=pcd_colors,
    )
    filtered = apply_phystwin_motion_filters(track_input)
    track_process = select_final_controller_points(filtered, count=30)
    track_process_path = out / "track_process_data.pkl"
    with track_process_path.open("wb") as handle:
        pickle.dump(
            {
                "object_points": track_process["object_points"],
                "object_colors": track_process["object_colors"],
                "object_visibilities": track_process["object_visibilities"],
                "object_motions_valid": track_process["object_motions_valid"],
                "controller_points": track_process["controller_points"],
            },
            handle,
        )
    final_data = sample_object_first_frame_volume(track_process, volume_sample_size=0.005)
    final_data_path = out / "final_data.pkl"
    with final_data_path.open("wb") as handle:
        pickle.dump(
            {
                "object_points": final_data["object_points"],
                "object_colors": final_data["object_colors"],
                "object_visibilities": final_data["object_visibilities"],
                "object_motions_valid": final_data["object_motions_valid"],
                "controller_points": final_data["controller_points"],
            },
            handle,
        )

    _render_tracking_2d_video(
        out / "tracking_2d.mp4",
        capture_dir=capture,
        rows=rows,
        tracks_yx=tracks_yx,
        visibility=tracker_visibility,
        query_is_object=track_input["query_is_object"],
        query_is_controller=track_input["query_is_controller"],
    )
    _render_world_track_video(
        out / "track_process_data.mp4",
        object_points=track_process["object_points"],
        object_valid=track_process["object_motions_valid"],
        controller_points=track_process["controller_points"],
        title="track_process_data object motion valid + controller FPS30",
    )
    _render_world_track_video(
        out / "final_data.mp4",
        object_points=final_data["object_points"],
        object_valid=final_data["object_visibilities"],
        controller_points=final_data["controller_points"],
        title="final_data object 5mm sample + controller FPS30",
    )
    if final_data["object_points"].shape[1] or final_data["controller_points"].shape[1]:
        _render_final_pcd_video(
            out / "final_pcd.mp4",
            object_points=final_data["object_points"],
            controller_points=final_data["controller_points"],
        )
    else:
        _render_empty_video(out / "final_pcd.mp4", frame_count=len(rows), label="final_pcd empty")

    manifest = {
        "compatibility_target": COMPATIBILITY_TARGET_PHYSTWIN,
        "tracking_product_backend": TRACKING_PRODUCT_BACKEND_PHYSTWIN_STRICT,
        "tracker_backend": "tapnextpp",
        "mask_backend": "edgetam",
        "depth_backend": str(metadata.get("depth_backend") or metadata.get("depth_source", "")),
        "depth_source_internal": str(
            metadata.get("depth_source_internal")
            or metadata.get("depth_source")
            or metadata.get("depth_backend", "")
        ),
        "execution_mode": PHYSTWIN_STRICT_EXECUTION_MODE,
        "compatibility_path_name": PHYSTWIN_COMPATIBILITY_PATH_NAME,
        "not_actual_cotracker": True,
        "camera_count": 1,
        "frame_count": int(len(rows)),
        "query_count": int(len(query_points_yx)),
        "processed_masks_path": str(processed_mask_path.relative_to(out)),
        "track_process_data_path": str(track_process_path.relative_to(out)),
        "final_data_path": str(final_data_path.relative_to(out)),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


__all__ = [
    "COMPATIBILITY_TARGET_PHYSTWIN",
    "DEFAULT_TRACKING_PRODUCT_BACKEND",
    "PHYSTWIN_STRICT_EXECUTION_MODE",
    "PreparedPhysTwinFrame",
    "TRACKING_PRODUCT_BACKEND_PHYSTWIN_STRICT",
    "TRACKING_PRODUCT_BACKEND_REALTIME_OVERLAY",
    "TRACKING_PRODUCT_BACKENDS",
    "StrictQuerySample",
    "StreamingControllerAnchorSelector",
    "apply_depth_validity_to_mask_frame",
    "apply_phystwin_motion_filters",
    "apply_radius_outlier_to_mask_frame",
    "build_track_process_input",
    "dense_world_pcd_grid",
    "finalize_headless_capture",
    "load_prepared_phystwin_frame",
    "normalize_tracking_product_backend",
    "prepare_phystwin_frame",
    "sample_first_frame_union_queries",
    "sample_object_first_frame_volume",
    "select_final_controller_points",
    "tracking_product_backend_is_strict",
    "write_prepared_phystwin_frame",
    "write_processed_masks",
]
