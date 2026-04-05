#!/usr/bin/env python3
"""
Preview fused point cloud from a recorded multi-camera RGB-D sequence.

Supports two visualization backends:
1) open3d (interactive 3D window, if open3d is installed)
2) opencv (top/front/side projection viewer, no open3d required)

Example:
  conda run --no-capture-output -n qqtt-record-min python test/view_fused_pcd.py --case-dir data_collect/20260206_145124
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


def _project_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if (p / ".git").exists():
            return p
    return here.parent


def _resolve_case_dir(case_dir: str) -> Path:
    path = Path(case_dir)
    if path.is_absolute():
        return path
    return (_project_root() / path).resolve()


def _load_metadata(case_dir: Path) -> Dict:
    metadata_path = case_dir / "metadata.json"
    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_c2ws(case_dir: Path) -> np.ndarray:
    cal_path = case_dir / "calibrate.pkl"
    with cal_path.open("rb") as f:
        c2ws = pickle.load(f)
    return np.asarray(c2ws, dtype=np.float64)


def _discover_cam_ids(case_dir: Path) -> List[int]:
    color_dir = case_dir / "color"
    depth_dir = case_dir / "depth"
    if not color_dir.exists() or not depth_dir.exists():
        raise RuntimeError(f"Missing color/ or depth/ folder under: {case_dir}")

    color_ids = {int(p.name) for p in color_dir.iterdir() if p.is_dir() and p.name.isdigit()}
    depth_ids = {int(p.name) for p in depth_dir.iterdir() if p.is_dir() and p.name.isdigit()}
    cam_ids = sorted(color_ids & depth_ids)
    if not cam_ids:
        raise RuntimeError("No valid camera folders found under color/ and depth/.")
    return cam_ids


def _frame_ids_for_cam(case_dir: Path, cam_id: int) -> List[int]:
    color_dir = case_dir / "color" / str(cam_id)
    depth_dir = case_dir / "depth" / str(cam_id)
    color_ids = {int(p.stem) for p in color_dir.glob("*.png")}
    depth_ids = {int(p.stem) for p in depth_dir.glob("*.npy")}
    return sorted(color_ids & depth_ids)


def _common_frame_ids(case_dir: Path, cam_ids: Sequence[int]) -> List[int]:
    frame_sets: List[set] = []
    for cam_id in cam_ids:
        ids = _frame_ids_for_cam(case_dir, cam_id)
        if not ids:
            raise RuntimeError(f"No valid frame files for camera {cam_id}.")
        frame_sets.append(set(ids))
    common = sorted(set.intersection(*frame_sets))
    if not common:
        raise RuntimeError("No common frame index exists across all cameras.")
    return common


def _pixel_grid_cache() -> Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]]:
    return {}


def _depth_to_points_cam(
    depth_m: np.ndarray,
    intrinsic: np.ndarray,
    cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    h, w = depth_m.shape
    key = (h, w)
    if key not in cache:
        xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        cache[key] = (xs, ys)
    xs, ys = cache[key]

    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])

    z = depth_m
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    return np.stack([x, y, z], axis=-1)


def _fill_depth_holes(depth_m: np.ndarray, iterations: int) -> np.ndarray:
    if iterations <= 0:
        return depth_m
    out = depth_m.copy()
    invalid = ~np.isfinite(out)
    invalid |= out <= 0
    if not np.any(invalid):
        return out

    kernel = np.ones((3, 3), dtype=np.uint8)
    for _ in range(iterations):
        if not np.any(invalid):
            break
        dilated = cv2.dilate(out, kernel)
        fill_mask = invalid & (dilated > 0)
        out[fill_mask] = dilated[fill_mask]
        invalid = ~np.isfinite(out)
        invalid |= out <= 0
    return out


def _camera_source_color(cam_id: int) -> np.ndarray:
    # Requested mapping: cam0 red, cam1 blue, cam3 green.
    # For a 3-camera setup this means cam2 is also mapped to green.
    if cam_id == 0:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)  # red
    if cam_id == 1:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)  # blue
    return np.array([0.0, 1.0, 0.0], dtype=np.float32)  # green


def _fuse_one_frame(
    case_dir: Path,
    frame_id: int,
    cam_ids: Sequence[int],
    intrinsics: np.ndarray,
    c2ws: np.ndarray,
    min_depth: float,
    max_depth: float,
    depth_hole_fill: bool,
    hole_fill_iters: int,
    color_by_camera: bool,
    camera_color_mode: str,
    camera_color_alpha: float,
    highlight_cam_id: Optional[int],
    cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:
    points_all: List[np.ndarray] = []
    colors_all: List[np.ndarray] = []

    for cam_id in cam_ids:
        color_path = case_dir / "color" / str(cam_id) / f"{frame_id}.png"
        depth_path = case_dir / "depth" / str(cam_id) / f"{frame_id}.npy"

        color_bgr = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
        if color_bgr is None:
            raise RuntimeError(f"Failed to read color image: {color_path}")
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        depth_raw = np.load(depth_path).astype(np.float32)
        depth_m = depth_raw / 1000.0
        if depth_hole_fill:
            depth_m = _fill_depth_holes(depth_m, iterations=hole_fill_iters)

        points_cam = _depth_to_points_cam(depth_m, intrinsics[cam_id], cache)

        valid = np.isfinite(depth_m)
        valid &= depth_m > min_depth
        valid &= depth_m < max_depth

        points_cam_flat = points_cam.reshape(-1, 3)
        colors_flat = color_rgb.reshape(-1, 3)
        valid_flat = valid.reshape(-1)

        points_cam_valid = points_cam_flat[valid_flat]
        apply_camera_color = color_by_camera and (
            highlight_cam_id is None or cam_id == highlight_cam_id
        )
        if apply_camera_color:
            cam_color = _camera_source_color(cam_id)
            if camera_color_mode == "override":
                colors_valid = np.broadcast_to(
                    cam_color, (points_cam_valid.shape[0], 3)
                ).astype(np.float32, copy=True)
            else:
                # Tint mode: keep original RGB and overlay camera color filter.
                alpha = float(np.clip(camera_color_alpha, 0.0, 1.0))
                base_colors = colors_flat[valid_flat]
                colors_valid = (
                    (1.0 - alpha) * base_colors + alpha * cam_color[None, :]
                ).astype(np.float32, copy=False)
                np.clip(colors_valid, 0.0, 1.0, out=colors_valid)
        else:
            colors_valid = colors_flat[valid_flat]

        ones = np.ones((points_cam_valid.shape[0], 1), dtype=np.float32)
        homo_cam = np.concatenate([points_cam_valid, ones], axis=1)
        points_world = (homo_cam @ c2ws[cam_id].T)[:, :3]

        points_all.append(points_world)
        colors_all.append(colors_valid)

    return np.concatenate(points_all, axis=0), np.concatenate(colors_all, axis=0)


def _voxel_downsample_numpy(
    points: np.ndarray, colors: np.ndarray, voxel_size: float
) -> Tuple[np.ndarray, np.ndarray]:
    if voxel_size <= 0:
        return points, colors
    if points.shape[0] == 0:
        return points, colors
    vox = np.floor(points / voxel_size).astype(np.int64)
    _, unique_idx = np.unique(vox, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)
    return points[unique_idx], colors[unique_idx]


def _flip_updown(points: np.ndarray, enabled: bool) -> np.ndarray:
    if not enabled or points.shape[0] == 0:
        return points
    flipped = points.copy()
    # Mirror across XY plane: keep x/y, invert z.
    flipped[:, 2] = -flipped[:, 2]
    return flipped


def _crop_workspace(
    points: np.ndarray,
    colors: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if points.shape[0] == 0:
        return points, colors
    mask = points[:, 0] >= x_min
    mask &= points[:, 0] <= x_max
    mask &= points[:, 1] >= y_min
    mask &= points[:, 1] <= y_max
    mask &= points[:, 2] >= z_min
    mask &= points[:, 2] <= z_max
    return points[mask], colors[mask]


def _largest_connected_component_by_voxel(
    points: np.ndarray,
    colors: np.ndarray,
    voxel_size: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if points.shape[0] == 0 or voxel_size <= 0:
        return points, colors

    voxels = np.floor(points / voxel_size).astype(np.int32)
    unique_voxels, inverse = np.unique(voxels, axis=0, return_inverse=True)
    if unique_voxels.shape[0] <= 1:
        return points, colors

    # 26-neighborhood keeps diagonally connected surface patches together.
    offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if not (dx == 0 and dy == 0 and dz == 0)
    ]

    key_to_idx = {tuple(v): i for i, v in enumerate(unique_voxels)}
    visited = np.zeros(unique_voxels.shape[0], dtype=bool)
    best_component: List[int] = []

    for start_idx in range(unique_voxels.shape[0]):
        if visited[start_idx]:
            continue

        stack = [start_idx]
        visited[start_idx] = True
        component: List[int] = []

        while stack:
            curr = stack.pop()
            component.append(curr)
            vx, vy, vz = unique_voxels[curr]
            for dx, dy, dz in offsets:
                nei_idx = key_to_idx.get((vx + dx, vy + dy, vz + dz))
                if nei_idx is not None and not visited[nei_idx]:
                    visited[nei_idx] = True
                    stack.append(nei_idx)

        if len(component) > len(best_component):
            best_component = component

    keep_voxel = np.zeros(unique_voxels.shape[0], dtype=bool)
    keep_voxel[np.asarray(best_component, dtype=np.int64)] = True
    keep_point = keep_voxel[inverse]
    return points[keep_point], colors[keep_point]


def _sample_points(
    points: np.ndarray, colors: np.ndarray, max_points: int
) -> Tuple[np.ndarray, np.ndarray]:
    if max_points <= 0 or points.shape[0] <= max_points:
        return points, colors
    idx = np.random.choice(points.shape[0], max_points, replace=False)
    return points[idx], colors[idx]


def _write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    colors_u8 = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors_u8):
            f.write(f"{p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]}\n")


def _try_import_open3d():
    try:
        import open3d as o3d  # type: ignore

        return o3d
    except Exception:
        return None


def _render_projection(points: np.ndarray, colors: np.ndarray, view: str, size: int) -> np.ndarray:
    img = np.zeros((size, size, 3), dtype=np.uint8)
    if points.shape[0] == 0:
        return img

    if view == "top":
        u = points[:, 0]
        v = -points[:, 1]
        d = points[:, 2]
    elif view == "front":
        u = points[:, 0]
        v = -points[:, 2]
        d = points[:, 1]
    elif view == "side":
        u = points[:, 1]
        v = -points[:, 2]
        d = points[:, 0]
    else:
        raise ValueError(f"Unknown view: {view}")

    u_min, u_max = np.percentile(u, 1), np.percentile(u, 99)
    v_min, v_max = np.percentile(v, 1), np.percentile(v, 99)
    if u_max <= u_min:
        u_max = u_min + 1e-6
    if v_max <= v_min:
        v_max = v_min + 1e-6

    px = ((u - u_min) / (u_max - u_min) * (size - 1)).astype(np.int32)
    py = ((v - v_min) / (v_max - v_min) * (size - 1)).astype(np.int32)
    px = np.clip(px, 0, size - 1)
    py = np.clip(py, 0, size - 1)

    order = np.argsort(d)
    bgr = (np.clip(colors, 0.0, 1.0)[:, ::-1] * 255.0).astype(np.uint8)
    img[py[order], px[order]] = bgr[order]
    return img


def _render_opencv_views(points: np.ndarray, colors: np.ndarray, panel: int = 520) -> np.ndarray:
    top = _render_projection(points, colors, "top", panel)
    front = _render_projection(points, colors, "front", panel)
    side = _render_projection(points, colors, "side", panel)
    blank = np.zeros_like(top)

    cv2.putText(top, "Top (X-Y)", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(
        front, "Front (X-Z)", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
    )
    cv2.putText(side, "Side (Y-Z)", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    row1 = np.hstack([top, front])
    row2 = np.hstack([side, blank])
    return np.vstack([row1, row2])


def _prepare_out_path(save_ply: str, frame_id: int, single_frame: bool) -> Optional[Path]:
    if not save_ply:
        return None
    out = Path(save_ply)
    if not out.is_absolute():
        out = (_project_root() / out).resolve()
    if single_frame:
        if out.suffix.lower() != ".ply":
            out = out.with_suffix(".ply")
        return out
    stem = out.stem if out.stem else "fused"
    suffix = out.suffix if out.suffix else ".ply"
    return out.with_name(f"{stem}_{frame_id}{suffix}")


def _run_export_only(
    selected: Sequence[int],
    args,
    case_dir: Path,
    cam_ids: Sequence[int],
    intrinsics: np.ndarray,
    c2ws: np.ndarray,
) -> int:
    if not args.save_ply:
        raise RuntimeError("--export-only requires --save-ply.")

    cache = _pixel_grid_cache()
    for i, frame_id in enumerate(selected, start=1):
        save_points, save_colors, _, _, stats = _process_frame_to_single_object(
            args=args,
            case_dir=case_dir,
            frame_id=frame_id,
            cam_ids=cam_ids,
            intrinsics=intrinsics,
            c2ws=c2ws,
            cache=cache,
        )
        out_path = _prepare_out_path(args.save_ply, frame_id, len(selected) == 1)
        if out_path is None:
            raise RuntimeError("Failed to resolve output PLY path.")
        _write_ply(out_path, save_points, save_colors)
        print(
            f"[{i}/{len(selected)}] frame={frame_id} "
            f"raw={stats['raw']} voxel={stats['voxel']} crop={stats['crop']} "
            f"object={stats['object']} -> {out_path}"
        )
    return 0


def _process_frame_to_single_object(
    args,
    case_dir: Path,
    frame_id: int,
    cam_ids: Sequence[int],
    intrinsics: np.ndarray,
    c2ws: np.ndarray,
    cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]],
    highlight_cam_id: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    effective_highlight_cam_id = highlight_cam_id
    if effective_highlight_cam_id is None and args.highlight_cam_id >= 0:
        effective_highlight_cam_id = args.highlight_cam_id

    points, colors = _fuse_one_frame(
        case_dir=case_dir,
        frame_id=frame_id,
        cam_ids=cam_ids,
        intrinsics=intrinsics,
        c2ws=c2ws,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        depth_hole_fill=args.depth_hole_fill,
        hole_fill_iters=args.hole_fill_iters,
        color_by_camera=args.color_by_camera,
        camera_color_mode=args.camera_color_mode,
        camera_color_alpha=args.camera_color_alpha,
        highlight_cam_id=effective_highlight_cam_id,
        cache=cache,
    )
    points = _flip_updown(points, enabled=args.flip_updown)
    stats: Dict[str, int] = {"raw": int(points.shape[0])}

    points, colors = _voxel_downsample_numpy(points, colors, args.voxel_size)
    stats["voxel"] = int(points.shape[0])

    if args.workspace_crop:
        points, colors = _crop_workspace(
            points=points,
            colors=colors,
            x_min=args.x_min,
            x_max=args.x_max,
            y_min=args.y_min,
            y_max=args.y_max,
            z_min=args.z_min,
            z_max=args.z_max,
        )
    stats["crop"] = int(points.shape[0])

    if args.single_object:
        points, colors = _largest_connected_component_by_voxel(
            points=points,
            colors=colors,
            voxel_size=max(args.object_voxel_size, args.voxel_size, 1e-6),
        )
    stats["object"] = int(points.shape[0])

    vis_points, vis_colors = _sample_points(points, colors, args.max_points)
    stats["vis"] = int(vis_points.shape[0])
    return points, colors, vis_points, vis_colors, stats


def _run_opencv_playback(
    selected: Sequence[int],
    args,
    case_dir: Path,
    cam_ids: Sequence[int],
    intrinsics: np.ndarray,
    c2ws: np.ndarray,
) -> int:
    cache = _pixel_grid_cache()
    win = "Fused PCD Preview (OpenCV) - q/Esc to quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    delay_ms = int(max(1, 1000.0 / max(args.fps, 1e-6)))
    selector_enabled = args.color_by_camera and args.gui_cam_selector and len(cam_ids) > 0

    if selector_enabled:
        cv2.createTrackbar("TintCamIdx", win, 0, len(cam_ids) - 1, lambda _v: None)
        if args.highlight_cam_id in cam_ids:
            init_pos = cam_ids.index(args.highlight_cam_id)
        else:
            init_pos = 0
        cv2.setTrackbarPos("TintCamIdx", win, init_pos)
        print(
            "OpenCV selector enabled: move `TintCamIdx` to choose camera tint target. "
            f"Mapping={cam_ids}"
        )

    try:
        for i, frame_id in enumerate(selected, start=1):
            selected_cam_id: Optional[int] = None
            if args.color_by_camera:
                if selector_enabled:
                    pos = cv2.getTrackbarPos("TintCamIdx", win)
                    pos = int(np.clip(pos, 0, len(cam_ids) - 1))
                    selected_cam_id = cam_ids[pos]
                elif args.highlight_cam_id >= 0:
                    selected_cam_id = args.highlight_cam_id

            save_points, save_colors, vis_points, vis_colors, stats = _process_frame_to_single_object(
                args=args,
                case_dir=case_dir,
                frame_id=frame_id,
                cam_ids=cam_ids,
                intrinsics=intrinsics,
                c2ws=c2ws,
                cache=cache,
                highlight_cam_id=selected_cam_id,
            )

            canvas = _render_opencv_views(vis_points, vis_colors)
            tint_text = "rgb"
            if args.color_by_camera:
                if selected_cam_id is None:
                    tint_text = "tint=all-cams"
                else:
                    tint_text = f"tint=cam{selected_cam_id}"
            cv2.putText(
                canvas,
                (
                    f"frame={frame_id} {tint_text} obj={stats['object']} vis={stats['vis']} "
                    f"[{i}/{len(selected)}]"
                ),
                (10, canvas.shape[0] - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.imshow(win, canvas)
            print(
                f"[{i}/{len(selected)}] frame={frame_id} "
                f"raw={stats['raw']} voxel={stats['voxel']} "
                f"crop={stats['crop']} object={stats['object']}"
            )

            out_path = _prepare_out_path(args.save_ply, frame_id, len(selected) == 1)
            if out_path is not None:
                _write_ply(out_path, save_points, save_colors)
                print(f"Saved: {out_path}")

            key = cv2.waitKey(delay_ms) & 0xFF
            if key in (27, ord("q")):
                break

        print("Playback finished. Press q/Esc in window to exit.")
        while True:
            key = cv2.waitKey(30) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        cv2.destroyAllWindows()

    return 0


def _run_open3d_playback(
    o3d,
    selected: Sequence[int],
    args,
    case_dir: Path,
    cam_ids: Sequence[int],
    intrinsics: np.ndarray,
    c2ws: np.ndarray,
) -> int:
    cache = _pixel_grid_cache()
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Fused PCD Preview (Open3D)", width=1400, height=900)
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    vis.add_geometry(coord)

    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)
    selector_enabled = args.color_by_camera and args.gui_cam_selector and len(cam_ids) > 0
    selector_win = "Tint Selector (Open3D)"
    if selector_enabled:
        cv2.namedWindow(selector_win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(selector_win, 420, 100)
        cv2.createTrackbar("TintCamIdx", selector_win, 0, len(cam_ids) - 1, lambda _v: None)
        if args.highlight_cam_id in cam_ids:
            init_pos = cam_ids.index(args.highlight_cam_id)
        else:
            init_pos = 0
        cv2.setTrackbarPos("TintCamIdx", selector_win, init_pos)
        print(
            "Open3D selector enabled: use `TintCamIdx` window to choose tint camera. "
            f"Mapping={cam_ids}"
        )

    try:
        for i, frame_id in enumerate(selected, start=1):
            selected_cam_id: Optional[int] = None
            if args.color_by_camera:
                if selector_enabled:
                    pos = cv2.getTrackbarPos("TintCamIdx", selector_win)
                    pos = int(np.clip(pos, 0, len(cam_ids) - 1))
                    selected_cam_id = cam_ids[pos]
                elif args.highlight_cam_id >= 0:
                    selected_cam_id = args.highlight_cam_id

            save_points, save_colors, vis_points, vis_colors, stats = _process_frame_to_single_object(
                args=args,
                case_dir=case_dir,
                frame_id=frame_id,
                cam_ids=cam_ids,
                intrinsics=intrinsics,
                c2ws=c2ws,
                cache=cache,
                highlight_cam_id=selected_cam_id,
            )

            pcd.points = o3d.utility.Vector3dVector(vis_points)
            pcd.colors = o3d.utility.Vector3dVector(vis_colors)
            vis.update_geometry(pcd)

            tint_text = "rgb"
            if args.color_by_camera:
                if selected_cam_id is None:
                    tint_text = "tint=all-cams"
                else:
                    tint_text = f"tint=cam{selected_cam_id}"
            print(
                f"[{i}/{len(selected)}] frame={frame_id} "
                f"{tint_text} raw={stats['raw']} voxel={stats['voxel']} "
                f"crop={stats['crop']} object={stats['object']}"
            )

            out_path = _prepare_out_path(args.save_ply, frame_id, len(selected) == 1)
            if out_path is not None:
                _write_ply(out_path, save_points, save_colors)
                print(f"Saved: {out_path}")

            alive = vis.poll_events()
            vis.update_renderer()
            if selector_enabled:
                # Keep the OpenCV selector window responsive.
                cv2.waitKey(1)
            if not alive:
                break
            time.sleep(max(0.0, 1.0 / max(args.fps, 1e-6)))

        print("Playback finished. Close the Open3D window to exit.")
        while vis.poll_events():
            vis.update_renderer()
            if selector_enabled:
                cv2.waitKey(1)
            time.sleep(0.01)
    finally:
        vis.destroy_window()
        if selector_enabled:
            try:
                cv2.destroyWindow(selector_win)
            except Exception:
                pass

    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case-dir",
        type=str,
        default="data_collect/20260206_145124",
        help="Folder containing color/, depth/, metadata.json, calibrate.pkl",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "open3d", "opencv"],
        default="auto",
        help="Visualization backend.",
    )
    parser.add_argument("--frame-step", type=int, default=1, help="Playback step size.")
    parser.add_argument("--fps", type=float, default=8.0, help="Playback FPS.")
    parser.add_argument("--min-depth", type=float, default=0.2, help="Min depth in meters.")
    parser.add_argument("--max-depth", type=float, default=1.5, help="Max depth in meters.")
    parser.add_argument(
        "--color-by-camera",
        dest="color_by_camera",
        action="store_true",
        help="Color points by camera source: cam0=red, cam1=blue, cam2/3=green.",
    )
    parser.add_argument(
        "--no-color-by-camera",
        dest="color_by_camera",
        action="store_false",
        help="Use original RGB colors from each camera image.",
    )
    parser.set_defaults(color_by_camera=False)
    parser.add_argument(
        "--highlight-cam-id",
        type=int,
        default=-1,
        help="When --color-by-camera is on: only tint this camera id, others keep original RGB. Use -1 for all cameras.",
    )
    parser.add_argument(
        "--gui-cam-selector",
        dest="gui_cam_selector",
        action="store_true",
        help="Show a selector (trackbar) to choose one tinted camera viewpoint during GUI playback.",
    )
    parser.add_argument(
        "--no-gui-cam-selector",
        dest="gui_cam_selector",
        action="store_false",
        help="Disable GUI selector and use --highlight-cam-id.",
    )
    parser.set_defaults(gui_cam_selector=True)
    parser.add_argument(
        "--camera-color-mode",
        choices=["tint", "override"],
        default="tint",
        help="When --color-by-camera is on: 'tint' overlays camera color on RGB, 'override' uses pure camera colors.",
    )
    parser.add_argument(
        "--camera-color-alpha",
        type=float,
        default=0.45,
        help="Tint alpha in [0,1] for --camera-color-mode tint.",
    )
    parser.add_argument(
        "--flip-updown",
        action="store_true",
        help="Flip fused point cloud vertically by inverting z axis.",
    )
    parser.add_argument(
        "--no-flip-updown",
        dest="flip_updown",
        action="store_false",
        help="Disable vertical flip.",
    )
    parser.set_defaults(flip_updown=False)
    parser.add_argument(
        "--depth-hole-fill",
        dest="depth_hole_fill",
        action="store_true",
        help="Fill small zero-depth holes before fusion.",
    )
    parser.add_argument(
        "--no-depth-hole-fill",
        dest="depth_hole_fill",
        action="store_false",
        help="Disable depth hole filling.",
    )
    parser.set_defaults(depth_hole_fill=True)
    parser.add_argument(
        "--hole-fill-iters",
        type=int,
        default=2,
        help="Depth hole-fill dilation iterations.",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.004,
        help="Optional voxel downsample size in meters (0 disables).",
    )
    parser.add_argument(
        "--single-object",
        dest="single_object",
        action="store_true",
        help="Keep only the largest connected component after fusion.",
    )
    parser.add_argument(
        "--no-single-object",
        dest="single_object",
        action="store_false",
        help="Disable largest-component filtering.",
    )
    parser.set_defaults(single_object=True)
    parser.add_argument(
        "--object-voxel-size",
        type=float,
        default=0.01,
        help="Voxel size used for connected-component extraction.",
    )
    parser.add_argument(
        "--workspace-crop",
        dest="workspace_crop",
        action="store_true",
        help="Crop fused points to workspace bounds before single-object extraction.",
    )
    parser.add_argument(
        "--no-workspace-crop",
        dest="workspace_crop",
        action="store_false",
        help="Disable workspace crop.",
    )
    parser.set_defaults(workspace_crop=True)
    parser.add_argument("--x-min", type=float, default=-0.05)
    parser.add_argument("--x-max", type=float, default=0.40)
    parser.add_argument("--y-min", type=float, default=-0.20)
    parser.add_argument("--y-max", type=float, default=0.50)
    parser.add_argument("--z-min", type=float, default=-0.15)
    parser.add_argument("--z-max", type=float, default=0.20)
    parser.add_argument(
        "--max-points",
        type=int,
        default=250000,
        help="Randomly keep at most this many points per frame (0 disables).",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=None,
        help="Start from a specific frame id if it exists.",
    )
    parser.add_argument(
        "--single-frame",
        type=int,
        default=None,
        help="Only fuse this frame id.",
    )
    parser.add_argument(
        "--save-ply",
        type=str,
        default="",
        help="Optional output path to save fused PLY (single frame) or stem (sequence).",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Fuse selected frames and save PLY only (no visualization window).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print discovered frames and validate loading first frame.",
    )
    args = parser.parse_args()
    args.camera_color_alpha = float(np.clip(args.camera_color_alpha, 0.0, 1.0))

    case_dir = _resolve_case_dir(args.case_dir)
    if not case_dir.exists():
        raise FileNotFoundError(f"Case directory not found: {case_dir}")

    metadata = _load_metadata(case_dir)
    intrinsics = np.asarray(metadata["intrinsics"], dtype=np.float32)
    c2ws = _load_c2ws(case_dir)
    cam_ids = _discover_cam_ids(case_dir)
    frame_ids = _common_frame_ids(case_dir, cam_ids)

    if intrinsics.shape[0] <= max(cam_ids):
        raise RuntimeError(
            f"Intrinsics count ({intrinsics.shape[0]}) is smaller than max cam id ({max(cam_ids)})."
        )
    if c2ws.shape[0] <= max(cam_ids):
        raise RuntimeError(
            f"Extrinsics count ({c2ws.shape[0]}) is smaller than max cam id ({max(cam_ids)})."
        )

    if args.single_frame is not None:
        if args.single_frame not in frame_ids:
            raise RuntimeError(f"single-frame {args.single_frame} not found in common frame ids.")
        selected = [args.single_frame]
    else:
        selected = frame_ids[:: max(args.frame_step, 1)]
        if args.start_frame is not None:
            selected = [fid for fid in selected if fid >= args.start_frame]
    if not selected:
        raise RuntimeError("No frame selected after applying frame-step/start-frame options.")

    print(f"Case dir: {case_dir}")
    print(f"Cameras: {cam_ids}")
    print(f"Common frames: {len(frame_ids)} ({frame_ids[0]} -> {frame_ids[-1]})")
    print(
        f"Depth filter: [{args.min_depth:.3f}, {args.max_depth:.3f}] m, "
        f"voxel={args.voxel_size:.4f} m, max_points={args.max_points}"
    )
    if args.color_by_camera:
        if args.camera_color_mode == "tint":
            print(
                "Color mode: camera tint overlay "
                f"(cam0=red, cam1=blue, cam2/3=green, alpha={args.camera_color_alpha:.2f})"
            )
        else:
            print("Color mode: camera source override (cam0=red, cam1=blue, cam2/3=green)")
        if args.gui_cam_selector:
            print("Color target: OpenCV GUI selector enabled (one camera tinted at a time).")
        elif args.highlight_cam_id >= 0:
            print(f"Color target: cam{args.highlight_cam_id} only.")
        else:
            print("Color target: all cameras.")
    else:
        print("Color mode: original RGB")
    print(f"Flip up/down (invert z): {args.flip_updown}")
    print(
        f"Depth hole fill: {args.depth_hole_fill} "
        f"(iters={args.hole_fill_iters})"
    )
    print(
        f"Single object: {args.single_object} (component voxel={args.object_voxel_size:.4f} m), "
        f"workspace crop: {args.workspace_crop}"
    )
    if args.workspace_crop:
        print(
            "Workspace bounds: "
            f"x=[{args.x_min:.3f},{args.x_max:.3f}] "
            f"y=[{args.y_min:.3f},{args.y_max:.3f}] "
            f"z=[{args.z_min:.3f},{args.z_max:.3f}]"
        )
    print(f"Selected frames: {len(selected)} (first={selected[0]}, last={selected[-1]})")

    cache = _pixel_grid_cache()
    _, _, _, _, preview_stats = _process_frame_to_single_object(
        args=args,
        case_dir=case_dir,
        frame_id=selected[0],
        cam_ids=cam_ids,
        intrinsics=intrinsics,
        c2ws=c2ws,
        cache=cache,
    )
    print(
        f"Preview frame {selected[0]} points: "
        f"raw={preview_stats['raw']} voxel={preview_stats['voxel']} "
        f"crop={preview_stats['crop']} object={preview_stats['object']}"
    )

    if args.dry_run:
        return 0

    if args.export_only:
        if not args.save_ply:
            args.save_ply = str((case_dir / "fused_object" / "object.ply").resolve())
        return _run_export_only(
            selected=selected,
            args=args,
            case_dir=case_dir,
            cam_ids=cam_ids,
            intrinsics=intrinsics,
            c2ws=c2ws,
        )

    o3d = _try_import_open3d()
    backend = args.backend
    if backend == "auto":
        backend = "open3d" if o3d is not None else "opencv"
    print(f"Using backend: {backend}")

    if backend == "open3d":
        if o3d is None:
            raise RuntimeError(
                "open3d is not installed. Use --backend opencv or install open3d in this env."
            )
        return _run_open3d_playback(
            o3d=o3d,
            selected=selected,
            args=args,
            case_dir=case_dir,
            cam_ids=cam_ids,
            intrinsics=intrinsics,
            c2ws=c2ws,
        )
    return _run_opencv_playback(
        selected=selected,
        args=args,
        case_dir=case_dir,
        cam_ids=cam_ids,
        intrinsics=intrinsics,
        c2ws=c2ws,
    )


if __name__ == "__main__":
    raise SystemExit(main())
