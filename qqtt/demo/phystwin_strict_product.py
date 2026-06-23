from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import pickle
from typing import Any, Mapping, Sequence

import numpy as np

from data_process.depth_backends.geometry import transform_points
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
            track_points[frame_idx, valid] = points_grid[frame_idx, 0, yy[valid], xx[valid]]
            track_colors[frame_idx, valid] = colors_grid[frame_idx, 0, yy[valid], xx[valid]].astype(np.float32) / 255.0

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
    for frame_idx in range(max(0, pts.shape[0] - 1)):
        current_valid = motions_valid[frame_idx].copy()
        if once_false_mask:
            current_valid &= global_mask
            motions_valid[frame_idx] &= global_mask
        for query_idx in range(pts.shape[1]):
            if once_false_mask and not global_mask[query_idx]:
                motions_valid[frame_idx, query_idx] = False
                continue
            if not motions_valid[frame_idx, query_idx]:
                continue
            distances = np.linalg.norm(pts[frame_idx] - pts[frame_idx, query_idx], axis=1)
            neighbors = np.flatnonzero((distances <= float(neighbor_dist)) & current_valid)
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
    "TRACKING_PRODUCT_BACKEND_PHYSTWIN_STRICT",
    "TRACKING_PRODUCT_BACKEND_REALTIME_OVERLAY",
    "TRACKING_PRODUCT_BACKENDS",
    "StrictQuerySample",
    "apply_phystwin_motion_filters",
    "build_track_process_input",
    "dense_world_pcd_grid",
    "finalize_headless_capture",
    "normalize_tracking_product_backend",
    "sample_first_frame_union_queries",
    "sample_object_first_frame_volume",
    "select_final_controller_points",
    "tracking_product_backend_is_strict",
    "write_processed_masks",
]
