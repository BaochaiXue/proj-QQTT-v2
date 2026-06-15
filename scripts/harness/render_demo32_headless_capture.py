#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ROOT_STR = str(ROOT)
if ROOT_STR in sys.path:
    sys.path.remove(ROOT_STR)
sys.path.insert(0, ROOT_STR)

from qqtt.demo.query_rainbow import query_rainbow_colors_for_indices


DEMO_VISUAL_MODES = ("pcd", "tracking")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_frames(path: Path) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            frames.append(json.loads(line))
    return frames


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


def _project_points(points_xyz: np.ndarray, intrinsics: dict[str, Any], *, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    if points.size == 0:
        return np.empty((0, 2), dtype=np.int32), np.empty((0,), dtype=bool)
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
    uv, valid = _project_points(points, intrinsics, width=width, height=height)
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
    intrinsics: dict[str, Any],
    *,
    marker_radius: int,
) -> tuple[int, int, int]:
    if not trajectory_path.is_file():
        return 0, 0, 0
    payload = np.load(trajectory_path, allow_pickle=False)
    marker_xyz = np.asarray(payload["marker_xyz_m"], dtype=np.float32).reshape(-1, 3)
    query_indices = np.asarray(payload["query_indices"], dtype=np.int64).reshape(-1)
    if "query_is_object" in payload.files:
        query_is_object = np.asarray(payload["query_is_object"], dtype=bool).reshape(-1)
    else:
        query_is_object = np.ones((len(marker_xyz),), dtype=bool)
    if "query_is_controller" in payload.files:
        query_is_controller = np.asarray(payload["query_is_controller"], dtype=bool).reshape(-1)
    else:
        query_is_controller = np.zeros((len(marker_xyz),), dtype=bool)
    if "marker_rgb_u8" in payload.files:
        marker_rgb_u8 = np.asarray(payload["marker_rgb_u8"], dtype=np.uint8).reshape(-1, 3)
    else:
        query_count = int(payload["query_count"][0]) if "query_count" in payload.files else None
        marker_rgb_u8 = query_rainbow_colors_for_indices(query_indices, query_count=query_count)
    count = min(len(marker_xyz), len(query_indices), len(query_is_object), len(query_is_controller), len(marker_rgb_u8))
    if count == 0:
        return 0, 0, 0
    height, width = image_bgr.shape[:2]
    uv, valid = _project_points(marker_xyz[:count], intrinsics, width=width, height=height)
    visible_uv = uv[valid]
    visible_is_object = query_is_object[:count][valid]
    visible_is_controller = query_is_controller[:count][valid]
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
    return int(object_count + controller_count), int(object_count), int(controller_count)


def render_capture_to_video(
    *,
    capture_dir: Path,
    output: Path,
    fps: float,
    point_size: int = 2,
    max_render_points: int = 0,
    query_point_radius: int = 3,
    demo_visual_mode: str = "tracking",
) -> dict[str, Any]:
    capture_dir = Path(capture_dir).resolve()
    metadata = _read_json(capture_dir / "metadata.json")
    frames = _read_frames(capture_dir / "frames.jsonl")
    if not frames:
        raise RuntimeError(f"no saved frames found in {capture_dir / 'frames.jsonl'}")
    width = int(metadata["width"])
    height = int(metadata["height"])
    intrinsics = dict(metadata["intrinsics"])
    if str(demo_visual_mode) not in DEMO_VISUAL_MODES:
        raise ValueError(f"demo_visual_mode must be one of {DEMO_VISUAL_MODES}")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {output}")
    rendered_counts: list[dict[str, int]] = []
    trajectory_by_seq = _trajectory_index(capture_dir)
    missing_query_frames = 0
    try:
        for frame in frames:
            image = np.zeros((height, width, 3), dtype=np.uint8)
            pcd_path = _resolve_capture_path(capture_dir, str(frame["pcd_path"]))
            pcd = np.load(pcd_path, allow_pickle=False)
            controller_count = _draw_projected_points(
                image,
                pcd["controller_xyz_m"],
                pcd["controller_rgb_u8"],
                intrinsics,
                point_size=int(point_size),
                max_points=int(max_render_points),
            )
            object_count = _draw_projected_points(
                image,
                pcd["object_xyz_m"],
                pcd["object_rgb_u8"],
                intrinsics,
                point_size=int(point_size),
                max_points=int(max_render_points),
            )
            query_count = query_object_count = query_controller_count = 0
            query_path = None
            if str(demo_visual_mode) == "tracking":
                query_path = _trajectory_path_for_frame(
                    capture_dir=capture_dir,
                    frame=frame,
                    trajectory_by_seq=trajectory_by_seq,
                )
                if query_path is None or not query_path.is_file():
                    missing_query_frames += 1
                else:
                    query_count, query_object_count, query_controller_count = _draw_query_points(
                        image,
                        query_path,
                        intrinsics,
                        marker_radius=int(query_point_radius),
                    )
            writer.write(image)
            rendered_counts.append(
                {
                    "seq": int(frame["seq"]),
                    "controller_points": int(controller_count),
                    "object_points": int(object_count),
                    "query_points": int(query_count),
                    "query_object_points": int(query_object_count),
                    "query_controller_points": int(query_controller_count),
                    "query_trajectory_exact": int(query_path is not None and query_path.is_file()),
                }
            )
    finally:
        writer.release()
    summary = {
        "capture_dir": str(capture_dir),
        "output": str(output.resolve()),
        "fps": float(fps),
        "frame_count": int(len(frames)),
        "image_size": [int(width), int(height)],
        "saved_pcd_source": metadata.get("saved_pcd_source"),
        "demo_visual_mode": str(demo_visual_mode),
        "query_overlay": "current_points_only_rainbow_identity" if str(demo_visual_mode) == "tracking" else "none",
        "query_color_mode": "phystwin_rainbow_identity" if str(demo_visual_mode) == "tracking" else "none",
        "query_match_policy": "exact_same_seq_only",
        "missing_query_frames": int(missing_query_frames),
        "rendered_counts": rendered_counts,
    }
    summary_path = output.with_name("render_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render Demo 3.2 headless enhanced-pt capture artifacts to MP4.")
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--point-size", type=int, default=2)
    parser.add_argument("--max-render-points", type=int, default=0)
    parser.add_argument("--query-point-radius", type=int, default=3)
    parser.add_argument("--demo-visual-mode", choices=DEMO_VISUAL_MODES, default="tracking")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = render_capture_to_video(
        capture_dir=args.capture_dir,
        output=args.output,
        fps=float(args.fps),
        point_size=int(args.point_size),
        max_render_points=int(args.max_render_points),
        query_point_radius=int(args.query_point_radius),
        demo_visual_mode=str(args.demo_visual_mode),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
