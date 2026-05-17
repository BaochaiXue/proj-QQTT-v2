#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_OUTPUT_ROOT = ROOT / "data" / "experiments" / "demo3_tracking_overlay"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Demo 3 lifted tracking anchors/trails over an aligned fused PCD frame.")
    parser.add_argument("--case-root", type=Path, required=True)
    parser.add_argument("--tracking-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--frame-idx", type=int, default=0)
    parser.add_argument("--trail-len", type=int, default=8)
    parser.add_argument("--max-overlay-points", type=int, default=30)
    parser.add_argument("--cameras", type=str, default="0,1,2")
    parser.add_argument("--viewpoints", type=str, default="cam0,cam1,cam2")
    parser.add_argument("--depth-source", choices=("native", "ffs"), default="native")
    parser.add_argument("--depth-min-m", type=float, default=0.2)
    parser.add_argument("--depth-max-m", type=float, default=1.5)
    parser.add_argument("--mask-dir", type=Path, default=None)
    return parser.parse_args(argv)


def _parse_cameras(spec: str) -> list[int]:
    return [int(item.strip()) for item in str(spec).split(",") if item.strip()]


def _parse_viewpoints(spec: str) -> list[str]:
    return [item.strip() for item in str(spec).split(",") if item.strip()]


def _tracking_npz_path(tracking_root: Path, camera_idx: int) -> Path:
    candidates = (
        tracking_root / "cotracker_like" / f"{camera_idx}.npz",
        tracking_root / f"{camera_idx}.npz",
        tracking_root / f"cam{camera_idx}.npz",
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find tracking NPZ for camera {camera_idx} under {tracking_root}")


def _load_mask(mask_root: Path | None, camera_idx: int, frame_idx: int, shape_hw: tuple[int, int]) -> np.ndarray:
    if mask_root is None:
        return np.ones(shape_hw, dtype=bool)
    candidates = (
        mask_root / str(camera_idx) / f"{frame_idx}.png",
        mask_root / str(camera_idx) / f"{frame_idx}.npy",
        mask_root / f"{camera_idx}_{frame_idx}.png",
        mask_root / f"{camera_idx}_{frame_idx}.npy",
    )
    for path in candidates:
        if not path.exists():
            continue
        if path.suffix == ".npy":
            return np.asarray(np.load(path), dtype=bool)
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Failed to read mask: {path}")
        return image > 0
    return np.ones(shape_hw, dtype=bool)


def _as_uint8_colors(colors: np.ndarray | None, count: int, default: tuple[int, int, int]) -> np.ndarray:
    if colors is None or len(colors) != count:
        return np.tile(np.asarray(default, dtype=np.uint8), (count, 1))
    arr = np.asarray(colors)
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr * (255.0 if arr.max(initial=0.0) <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    return arr.astype(np.uint8)


def _view_axes(viewpoint: str) -> tuple[int, int]:
    name = viewpoint.lower()
    if name in {"cam1", "xz", "side"}:
        return 0, 2
    if name in {"cam2", "yz", "front"}:
        return 1, 2
    return 0, 1


def _project_points(points: np.ndarray, axes: tuple[int, int], size_hw: tuple[int, int]) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.size == 0:
        return np.empty((0, 2), dtype=np.int32)
    height, width = size_hw
    xy = pts[:, list(axes)]
    min_xy = np.nanmin(xy, axis=0)
    max_xy = np.nanmax(xy, axis=0)
    span = np.maximum(max_xy - min_xy, 1e-6)
    norm = (xy - min_xy) / span
    px = np.clip(norm[:, 0] * (width - 24) + 12, 0, width - 1)
    py = np.clip((1.0 - norm[:, 1]) * (height - 24) + 12, 0, height - 1)
    return np.stack([px, py], axis=1).astype(np.int32)


def _draw_points(
    image: np.ndarray,
    points: np.ndarray,
    colors: np.ndarray | None,
    axes: tuple[int, int],
    *,
    radius: int,
    default_color: tuple[int, int, int],
    max_points: int | None = None,
) -> None:
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) == 0:
        return
    draw_colors = _as_uint8_colors(colors, len(pts), default_color)
    if max_points is not None and len(pts) > max_points:
        idx = np.linspace(0, len(pts) - 1, int(max_points), dtype=np.int64)
        pts = pts[idx]
        draw_colors = draw_colors[idx]
    for (x, y), color in zip(_project_points(pts, axes, image.shape[:2]), draw_colors):
        cv2.circle(image, (int(x), int(y)), int(radius), (int(color[2]), int(color[1]), int(color[0])), -1, lineType=cv2.LINE_AA)


def _write_overlay_board(
    output_dir: Path,
    *,
    fused_points: np.ndarray,
    fused_colors: np.ndarray | None,
    anchor_points: np.ndarray,
    anchor_colors: np.ndarray | None,
    trail_points: np.ndarray,
    trail_colors: np.ndarray | None,
    viewpoints: list[str],
) -> tuple[Path, Path]:
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    panel_h, panel_w = 240, 320
    row_titles = ("PCD only", "PCD + anchors", "PCD + anchors + trails")
    board = np.zeros((panel_h * 3, panel_w * len(viewpoints), 3), dtype=np.uint8)
    for col, viewpoint in enumerate(viewpoints):
        axes = _view_axes(viewpoint)
        for row in range(3):
            panel = board[row * panel_h : (row + 1) * panel_h, col * panel_w : (col + 1) * panel_w]
            panel[:] = 8
            _draw_points(panel, fused_points, fused_colors, axes, radius=1, default_color=(180, 180, 170), max_points=12000)
            if row >= 1:
                _draw_points(panel, anchor_points, anchor_colors, axes, radius=3, default_color=(255, 80, 20), max_points=500)
            if row >= 2:
                _draw_points(panel, trail_points, trail_colors, axes, radius=2, default_color=(40, 180, 255), max_points=2000)
            cv2.putText(
                panel,
                f"{viewpoint} | {row_titles[row]}",
                (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )
    frame_path = frames_dir / "frame_000000.png"
    cv2.imwrite(str(frame_path), board)
    video_path = output_dir / "overlay_3view.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 1.0, (board.shape[1], board.shape[0]))
    try:
        writer.write(board)
    finally:
        writer.release()
    return frame_path, video_path


def run_overlay_export(args: argparse.Namespace) -> dict:
    from data_process.visualization.calibration_io import load_calibration_transforms
    from data_process.visualization.io_artifacts import write_ply_ascii
    from data_process.visualization.io_case import (
        get_case_intrinsics,
        get_depth_scale_list,
        load_case_frame_cloud,
        load_case_metadata,
        load_depth_frame,
    )
    from qqtt.tracking.io import load_cotracker_like_npz
    from qqtt.tracking.lift import lift_tracks_to_world

    case_root = Path(args.case_root).resolve()
    tracking_root = Path(args.tracking_root).resolve()
    output_dir = Path(args.output).resolve() if args.output is not None else Path(args.output_root).resolve() / case_root.name / tracking_root.name
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = load_case_metadata(case_root)
    intrinsics = get_case_intrinsics(metadata)
    depth_scales = get_depth_scale_list(metadata, len(metadata["serial_numbers"]))
    c2w_list = load_calibration_transforms(
        case_root / "calibrate.pkl",
        serial_numbers=metadata["serial_numbers"],
        calibration_reference_serials=metadata.get("calibration_reference_serials", metadata["serial_numbers"]),
    )
    depth_source = "realsense" if args.depth_source == "native" else "ffs"
    fused_points, fused_colors, fused_stats = load_case_frame_cloud(
        case_dir=case_root,
        metadata=metadata,
        frame_idx=int(args.frame_idx),
        depth_source=depth_source,
        use_float_ffs_depth_when_available=True,
        voxel_size=None,
        max_points_per_camera=None,
        depth_min_m=float(args.depth_min_m),
        depth_max_m=float(args.depth_max_m),
    )

    all_anchor_points: list[np.ndarray] = []
    all_anchor_colors: list[np.ndarray] = []
    all_trail_points: list[np.ndarray] = []
    all_trail_colors: list[np.ndarray] = []
    per_camera = []
    for camera_idx in _parse_cameras(str(args.cameras)):
        result, tracking_metadata = load_cotracker_like_npz(_tracking_npz_path(tracking_root, camera_idx))
        frame_idx = min(int(args.frame_idx), result.tracks_yx.shape[0] - 1)
        trail_start = max(0, frame_idx - max(1, int(args.trail_len)) + 1)
        lifted = None
        points = np.empty((0, 3), dtype=np.float32)
        for trail_frame_idx in range(trail_start, frame_idx + 1):
            _, depth_m_or_u16, _ = load_depth_frame(
                case_dir=case_root,
                metadata=metadata,
                camera_idx=camera_idx,
                frame_idx=trail_frame_idx,
                depth_source=depth_source,
                use_float_ffs_depth_when_available=True,
            )
            color_path = case_root / "color" / str(camera_idx) / f"{trail_frame_idx}.png"
            color_image = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
            mask = _load_mask(None if args.mask_dir is None else Path(args.mask_dir).resolve(), camera_idx, trail_frame_idx, depth_m_or_u16.shape[:2])
            lifted_this = lift_tracks_to_world(
                tracks_yx_t=result.tracks_yx[trail_frame_idx],
                visibility_t=result.visibility[trail_frame_idx],
                depth_uint16=depth_m_or_u16,
                depth_scale_m_per_unit=float(depth_scales[camera_idx] or 1.0),
                mask=mask,
                K=intrinsics[camera_idx],
                c2w=c2w_list[camera_idx],
                camera_idx=camera_idx,
                colors_rgb=color_image,
                track_ids=result.track_ids,
                depth_min_m=float(args.depth_min_m),
                depth_max_m=float(args.depth_max_m),
                max_tracks=int(args.max_overlay_points),
            )
            trail_points = lifted_this.points_world
            if len(trail_points) > 0:
                all_trail_points.append(trail_points)
                trail_color = np.zeros((len(trail_points), 3), dtype=np.uint8)
                trail_color[:, camera_idx % 3] = 160
                all_trail_colors.append(trail_color)
            if trail_frame_idx == frame_idx:
                lifted = lifted_this
                points = trail_points
                if len(points) > 0:
                    all_anchor_points.append(points)
                    color = np.zeros((len(points), 3), dtype=np.uint8)
                    color[:, camera_idx % 3] = 255
                    all_anchor_colors.append(color)
        if lifted is None:
            raise RuntimeError(f"No lifted frame was evaluated for camera {camera_idx}")
        per_camera.append(
            {
                "camera_idx": camera_idx,
                "tracking_metadata": tracking_metadata,
                "lift_stats": lifted.stats,
                "anchor_count_written": int(len(points)),
            }
        )

    anchor_points = np.concatenate(all_anchor_points, axis=0) if all_anchor_points else np.empty((0, 3), dtype=np.float32)
    anchor_colors = np.concatenate(all_anchor_colors, axis=0) if all_anchor_colors else np.empty((0, 3), dtype=np.uint8)
    trail_points = np.concatenate(all_trail_points, axis=0) if all_trail_points else np.empty((0, 3), dtype=np.float32)
    trail_colors = np.concatenate(all_trail_colors, axis=0) if all_trail_colors else np.empty((0, 3), dtype=np.uint8)
    write_ply_ascii(output_dir / "fused_pcd.ply", fused_points, fused_colors)
    write_ply_ascii(output_dir / "lifted_anchors.ply", anchor_points, anchor_colors)
    write_ply_ascii(output_dir / "lifted_trails.ply", trail_points, trail_colors)
    frame_path, video_path = _write_overlay_board(
        output_dir,
        fused_points=fused_points,
        fused_colors=fused_colors,
        anchor_points=anchor_points,
        anchor_colors=anchor_colors,
        trail_points=trail_points,
        trail_colors=trail_colors,
        viewpoints=_parse_viewpoints(str(args.viewpoints)),
    )
    summary = {
        "case_root": str(case_root),
        "tracking_root": str(tracking_root),
        "frame_idx": int(args.frame_idx),
        "depth_source": str(args.depth_source),
        "viewpoints": _parse_viewpoints(str(args.viewpoints)),
        "fused_point_count": int(len(fused_points)),
        "anchor_point_count": int(len(anchor_points)),
        "trail_point_count": int(len(trail_points)),
        "fused_stats": fused_stats,
        "per_camera": per_camera,
        "artifacts": {
            "fused_pcd": str(output_dir / "fused_pcd.ply"),
            "lifted_anchors": str(output_dir / "lifted_anchors.ply"),
            "lifted_trails": str(output_dir / "lifted_trails.ply"),
            "overlay_stats": str(output_dir / "overlay_stats.json"),
            "overlay_frame": str(frame_path),
            "overlay_video": str(video_path),
        },
    }
    (output_dir / "overlay_stats.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "overlay_summary.md").write_text(
        "\n".join(
            [
                "# Demo 3 Tracking Overlay",
                "",
                f"- frame_idx: {int(args.frame_idx)}",
                f"- fused_point_count: {int(len(fused_points))}",
                f"- anchor_point_count: {int(len(anchor_points))}",
                f"- trail_point_count: {int(len(trail_points))}",
                f"- overlay_video: {video_path}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_overlay_export(args)
    print(f"Demo 3 tracking overlay artifacts written to {Path(summary['artifacts']['overlay_stats']).parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
