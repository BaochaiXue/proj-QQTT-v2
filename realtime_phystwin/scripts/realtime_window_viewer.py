#!/usr/bin/env python3
"""Display online-training windows with prediction history and first-seen state."""

import argparse
from collections import OrderedDict
import json
import os
import pickle
import time
from pathlib import Path

import cv2
import numpy as np


WINDOW_ARRAY_KEYS = (
    "pred_points",
    "gt_object_points",
    "object_colors",
    "object_visibilities",
    "controller_points",
    "frame_indices",
    "online_frame_indices",
)

POINT_KERNELS = {}


def parse_color(value):
    parts = [int(v.strip()) for v in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("color must be B,G,R")
    return tuple(max(0, min(255, v)) for v in parts)


def load_camera(base_path, case_name, cam_idx):
    case_dir = Path(base_path) / case_name
    with open(case_dir / "calibrate.pkl", "rb") as f:
        c2ws = pickle.load(f)
    with open(case_dir / "metadata.json", "r") as f:
        metadata = json.load(f)

    c2w = np.asarray(c2ws[int(cam_idx)], dtype=np.float64)
    w2c = np.linalg.inv(c2w)
    intrinsic = np.asarray(metadata["intrinsics"][int(cam_idx)], dtype=np.float64)
    width, height = metadata["WH"]
    return intrinsic, w2c, int(width), int(height)


def load_npz(path):
    try:
        with np.load(path, allow_pickle=False) as data:
            return {key: data[key].copy() for key in data.files}
    except (OSError, ValueError, EOFError):
        return None


def load_latest_window(path):
    return load_npz(path)


def scalar_int(value):
    return int(np.asarray(value).reshape(-1)[0])


def exported_window_count(export):
    pred_points = export["pred_points"]
    if pred_points.ndim == 3:
        return 1
    if pred_points.ndim != 4:
        raise ValueError(f"Expected pred_points with 3 or 4 dims, got {pred_points.shape}")
    return int(pred_points.shape[0])


def slice_exported_window(export, window_idx):
    pred_points = export["pred_points"]
    if pred_points.ndim == 3:
        selected = dict(export)
        if "window_start" not in selected:
            selected["window_start"] = np.array(
                int(np.asarray(selected["window_starts"]).reshape(-1)[0]),
                dtype=np.int64,
            )
        selected["window_index"] = np.array(0, dtype=np.int64)
        return selected
    if pred_points.ndim != 4:
        raise ValueError(f"Expected pred_points with 3 or 4 dims, got {pred_points.shape}")

    window_starts = export["window_starts"].astype(np.int64)
    selected = {}
    for key, value in export.items():
        if key in WINDOW_ARRAY_KEYS:
            selected[key] = value[window_idx] if value.shape[0] == pred_points.shape[0] else value
        else:
            selected[key] = value
    selected["window_start"] = np.array(int(window_starts[window_idx]), dtype=np.int64)
    selected["window_index"] = np.array(int(window_idx), dtype=np.int64)
    return selected


def exported_windows(export):
    return [
        slice_exported_window(export, window_idx)
        for window_idx in range(exported_window_count(export))
    ]


def first_seen_path(first_seen_dir, window_start):
    return first_seen_dir / f"window_{int(window_start):06d}.npz"


def snapshot_from_window(window):
    return {
        "iteration": scalar_int(window["iteration"]),
        "pred_points": np.asarray(window["pred_points"], dtype=np.float32).copy(),
    }


def load_first_seen_snapshot(first_seen_dir, window_start):
    data = load_npz(first_seen_path(first_seen_dir, window_start))
    if data is None:
        return None
    return {
        "iteration": scalar_int(
            data["first_iteration"]
            if "first_iteration" in data
            else data["iteration"]
        ),
        "pred_points": np.asarray(data["pred_points"], dtype=np.float32).copy(),
        "source": "disk",
    }


def first_seen_window_from_data(data):
    window = dict(data)
    window["iteration"] = np.array(
        scalar_int(
            data["first_iteration"]
            if "first_iteration" in data
            else data["iteration"]
        ),
        dtype=np.int64,
    )
    window["window_start"] = np.array(
        scalar_int(data["window_start"]), dtype=np.int64
    )
    window["window_index"] = np.array(-1, dtype=np.int64)
    return window


def discover_first_seen_windows(window_histories, first_seen_dir):
    if not first_seen_dir.is_dir():
        return []
    added = []
    for path in sorted(first_seen_dir.glob("window_*.npz")):
        data = load_npz(path)
        if data is None or "window_start" not in data:
            continue
        window = first_seen_window_from_data(data)
        start = scalar_int(window["window_start"])
        first = {
            "iteration": scalar_int(window["iteration"]),
            "pred_points": np.asarray(
                window["pred_points"], dtype=np.float32
            ).copy(),
            "source": "disk",
        }
        if start in window_histories:
            state = window_histories[start]
            if state["first"].get("source") != "disk":
                state["first"] = first
            continue
        window_histories[start] = {
            "shared": window,
            "frame_idx": 0,
            "last_seen_iteration": scalar_int(window["iteration"]),
            "first": first,
            "history": {},
            "history_order": [],
        }
        added.append(start)
    return added


def trim_history(state, max_history):
    if int(max_history) <= 0:
        return
    while len(state["history_order"]) > int(max_history):
        remove_iteration = state["history_order"].pop(0)
        state["history"].pop(remove_iteration, None)


def update_window_histories(
    window_histories,
    export,
    first_seen_dir,
    max_history,
    history_every,
):
    iteration = scalar_int(export["iteration"])
    windows = exported_windows(export)
    added = []
    captured = []
    for window in windows:
        start = scalar_int(window["window_start"])
        if start in window_histories:
            state = window_histories[start]
            state["shared"] = window
            state["last_seen_iteration"] = iteration
            state["frame_idx"] %= int(window["pred_points"].shape[0])
        else:
            first = load_first_seen_snapshot(first_seen_dir, start)
            if first is None:
                first = snapshot_from_window(window)
                first["source"] = "viewer-fallback"
            state = {
                "shared": window,
                "frame_idx": 0,
                "last_seen_iteration": iteration,
                "first": first,
                "history": {},
                "history_order": [],
            }
            window_histories[start] = state
            added.append(start)

        if state["first"].get("source") != "disk":
            persisted_first = load_first_seen_snapshot(first_seen_dir, start)
            if persisted_first is not None:
                state["first"] = persisted_first

        should_capture = (
            iteration != int(state["first"]["iteration"])
            and iteration % int(history_every) == 0
            and iteration not in state["history"]
        )
        if should_capture:
            state["history"][iteration] = snapshot_from_window(window)
            state["history_order"].append(iteration)
            state["history_order"].sort()
            trim_history(state, max_history)
            captured.append((start, iteration))

    return added, captured


def history_iterations(window_histories):
    return sorted(
        {
            iteration
            for state in window_histories.values()
            for iteration in state["history_order"]
        },
        reverse=True,
    )


def clamp_scroll(viewport):
    viewport["scroll_x_px"] = max(
        0, min(int(viewport["scroll_x_px"]), int(viewport["max_scroll_x_px"]))
    )
    viewport["scroll_y_px"] = max(
        0, min(int(viewport["scroll_y_px"]), int(viewport["max_scroll_y_px"]))
    )


def scroll_to_latest(viewport):
    viewport["scroll_x_px"] = int(viewport["max_scroll_x_px"])
    viewport["auto_follow_x"] = True


def scroll_to_oldest(viewport):
    viewport["scroll_x_px"] = 0
    viewport["auto_follow_x"] = False


def scroll_to_newest_history(viewport):
    viewport["scroll_y_px"] = 0
    viewport["auto_follow_y"] = True


def scroll_to_oldest_history(viewport):
    viewport["scroll_y_px"] = int(viewport["max_scroll_y_px"])
    viewport["auto_follow_y"] = False


def handle_mouse(event, x, y, _flags, viewport):
    if event == cv2.EVENT_LBUTTONDOWN:
        viewport["dragging"] = True
        viewport["drag_start_x"] = int(x)
        viewport["drag_start_y"] = int(y)
        viewport["drag_start_scroll_x_px"] = int(viewport["scroll_x_px"])
        viewport["drag_start_scroll_y_px"] = int(viewport["scroll_y_px"])
    elif event == cv2.EVENT_MOUSEMOVE and viewport["dragging"]:
        drag_x = int(viewport["drag_start_x"]) - int(x)
        drag_y = int(viewport["drag_start_y"]) - int(y)
        viewport["scroll_x_px"] = (
            int(viewport["drag_start_scroll_x_px"]) + drag_x
        )
        viewport["scroll_y_px"] = (
            int(viewport["drag_start_scroll_y_px"]) + drag_y
        )
        if abs(drag_x) >= 3:
            viewport["auto_follow_x"] = False
        if abs(drag_y) >= 3:
            viewport["auto_follow_y"] = False
        clamp_scroll(viewport)
    elif event == cv2.EVENT_LBUTTONUP:
        viewport["dragging"] = False


def select_points(pred_points, window, frame_idx, mode):
    points = pred_points[frame_idx]
    if mode == "original":
        return points[: scalar_int(window["num_original_points"])]
    if mode == "surface":
        return points[: scalar_int(window["num_surface_points"])]
    return points[: scalar_int(window["num_all_points"])]


def project_points(points, intrinsic, w2c, image_width, image_height, min_depth=1e-6):
    if points.size == 0:
        return np.empty((0, 2), dtype=np.int32)

    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if points.shape[0] == 0:
        return np.empty((0, 2), dtype=np.int32)

    points_h = np.concatenate(
        [points.astype(np.float64), np.ones((points.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    cam = (w2c @ points_h.T).T[:, :3]
    z = cam[:, 2]
    valid = np.isfinite(cam).all(axis=1) & (z > min_depth)
    cam = cam[valid]
    z = z[valid]
    if cam.shape[0] == 0:
        return np.empty((0, 2), dtype=np.int32)

    u = intrinsic[0, 0] * cam[:, 0] / z + intrinsic[0, 2]
    v = intrinsic[1, 1] * cam[:, 1] / z + intrinsic[1, 2]
    uv = np.stack([u, v], axis=1)
    in_frame = (
        np.isfinite(uv).all(axis=1)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < image_width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < image_height)
    )
    return np.rint(uv[in_frame]).astype(np.int32)


def draw_points(image, uv, color, radius):
    if uv.size == 0:
        return

    xy = np.rint(uv).astype(np.int32)
    valid = (
        (xy[:, 0] >= 0)
        & (xy[:, 0] < image.shape[1])
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < image.shape[0])
    )
    xy = xy[valid]
    if xy.size == 0:
        return

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[xy[:, 1], xy[:, 0]] = 255
    radius = max(0, int(radius))
    if radius > 0:
        kernel = POINT_KERNELS.get(radius)
        if kernel is None:
            diameter = radius * 2 + 1
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (diameter, diameter)
            )
            POINT_KERNELS[radius] = kernel
        mask = cv2.dilate(mask, kernel)
    image[mask != 0] = color


class FrameCache:
    def __init__(self, max_items):
        self.max_items = max(1, int(max_items))
        self.images = OrderedDict()

    def load(self, path, tile_width):
        key = (str(path), int(tile_width))
        image = self.images.get(key)
        if image is not None:
            self.images.move_to_end(key)
            return image

        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot read frame image: {path}")
        image = resize_tile(image, tile_width)
        self.images[key] = image
        self.images.move_to_end(key)
        while len(self.images) > self.max_items:
            self.images.popitem(last=False)
        return image


def read_frame_image(args, window, local_idx, frame_cache):
    frame_idx = int(window["frame_indices"][local_idx])
    image_path = (
        Path(args.base_path)
        / args.case_name
        / "color"
        / str(args.cam_idx)
        / f"{frame_idx}.png"
    )
    return frame_cache.load(image_path, args.tile_width)


def draw_label(image, text):
    cv2.rectangle(image, (0, 0), (image.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(
        image,
        text,
        (6, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def resize_tile(image, tile_width):
    tile_width = int(tile_width)
    if tile_width <= 0 or image.shape[1] == tile_width:
        return image
    scale = float(tile_width) / float(image.shape[1])
    tile_height = max(1, int(round(image.shape[0] * scale)))
    return cv2.resize(image, (tile_width, tile_height), interpolation=cv2.INTER_AREA)


def render_raw_frame(args, window, local_idx, frame_cache):
    image = read_frame_image(args, window, local_idx, frame_cache).copy()
    start = scalar_int(window["window_start"])
    frame_idx = int(window["frame_indices"][local_idx])
    draw_label(image, f"raw start={start} frame={frame_idx}")
    return image


def render_overlay_frame(
    args,
    window,
    pred_points,
    local_idx,
    intrinsic,
    w2c,
    label,
    frame_cache,
):
    image = read_frame_image(args, window, local_idx, frame_cache).copy()
    image_height, image_width = image.shape[:2]
    selected_pred_points = select_points(
        pred_points, window, local_idx, args.point_mode
    )
    pred_uv = project_points(
        selected_pred_points, intrinsic, w2c, image_width, image_height
    )
    draw_points(image, pred_uv, args.pred_color, args.radius)

    if "gt_object_points" in window:
        gt_points = window["gt_object_points"][local_idx]
        if "object_visibilities" in window and window["object_visibilities"].size > 0:
            visible = window["object_visibilities"][local_idx].astype(bool)
            gt_points = gt_points[visible]
        gt_uv = project_points(gt_points, intrinsic, w2c, image_width, image_height)
        draw_points(image, gt_uv, args.gt_color, max(1, args.radius - 1))

    if args.draw_controller and "controller_points" in window:
        controller_points = window["controller_points"][local_idx]
        controller_uv = project_points(
            controller_points, intrinsic, w2c, image_width, image_height
        )
        draw_points(image, controller_uv, args.controller_color, args.radius + 1)

    draw_label(image, label)
    return image


def render_missing_tile(args, window_start, label):
    image = np.zeros(
        (int(args.tile_height), int(args.tile_width), 3), dtype=np.uint8
    )
    draw_label(image, f"{label} start={int(window_start)}")
    return image


def render_window_row(
    args,
    visible_items,
    intrinsic,
    w2c,
    row_kind,
    frame_cache,
    iteration=None,
):
    tiles = []
    for start, state in visible_items:
        window = state["shared"]
        local_idx = int(state["frame_idx"]) % int(window["pred_points"].shape[0])
        if row_kind == "raw":
            tiles.append(render_raw_frame(args, window, local_idx, frame_cache))
            continue

        if row_kind == "first":
            snapshot = state["first"]
            label = f"first iter={int(snapshot['iteration'])} start={int(start)}"
        else:
            snapshot = state["history"].get(int(iteration))
            if snapshot is None:
                tiles.append(
                    render_missing_tile(
                        args,
                        start,
                        f"iter={int(iteration)} not sampled",
                    )
                )
                continue
            label = f"iter={int(iteration)} start={int(start)}"

        pred_points = snapshot["pred_points"]
        snapshot_idx = local_idx % int(pred_points.shape[0])
        tiles.append(
            render_overlay_frame(
                args,
                window,
                pred_points,
                snapshot_idx,
                intrinsic,
                w2c,
                label=label,
                frame_cache=frame_cache,
            )
        )
    return np.hstack(tiles)


def render_history_view(
    args,
    window_histories,
    intrinsic,
    w2c,
    viewport,
    frame_cache,
):
    ordered_items = [
        (start, window_histories[start])
        for start in sorted(window_histories)
    ]
    visible_count = min(int(args.max_windows), len(ordered_items))
    viewport_width = visible_count * int(args.tile_width)
    total_width = len(ordered_items) * int(args.tile_width)
    viewport["max_scroll_x_px"] = max(0, total_width - viewport_width)
    if viewport["auto_follow_x"]:
        viewport["scroll_x_px"] = viewport["max_scroll_x_px"]

    iterations = history_iterations(window_histories)
    visible_history_count = min(
        int(args.visible_history_rows), len(iterations)
    )
    history_viewport_height = visible_history_count * int(args.tile_height)
    total_history_height = len(iterations) * int(args.tile_height)
    viewport["max_scroll_y_px"] = max(
        0, total_history_height - history_viewport_height
    )
    if viewport["auto_follow_y"]:
        viewport["scroll_y_px"] = 0
    clamp_scroll(viewport)

    first_col = int(viewport["scroll_x_px"]) // int(args.tile_width)
    crop_left = int(viewport["scroll_x_px"]) % int(args.tile_width)
    render_cols = visible_count + (1 if crop_left > 0 else 0)
    visible_items = ordered_items[first_col : first_col + render_cols]

    raw_row = render_window_row(
        args,
        visible_items,
        intrinsic,
        w2c,
        row_kind="raw",
        frame_cache=frame_cache,
    )
    first_row = render_window_row(
        args,
        visible_items,
        intrinsic,
        w2c,
        row_kind="first",
        frame_cache=frame_cache,
    )
    raw_row = raw_row[:, crop_left : crop_left + viewport_width]
    first_row = first_row[:, crop_left : crop_left + viewport_width]

    rows = [raw_row]
    if visible_history_count > 0:
        first_history_row = (
            int(viewport["scroll_y_px"]) // int(args.tile_height)
        )
        crop_top = int(viewport["scroll_y_px"]) % int(args.tile_height)
        render_rows = visible_history_count + (1 if crop_top > 0 else 0)
        visible_iterations = iterations[
            first_history_row : first_history_row + render_rows
        ]
        history_rows = [
            render_window_row(
                args,
                visible_items,
                intrinsic,
                w2c,
                row_kind="history",
                frame_cache=frame_cache,
                iteration=iteration,
            )[:, crop_left : crop_left + viewport_width]
            for iteration in visible_iterations
        ]
        history_grid = np.vstack(history_rows)
        history_grid = history_grid[
            crop_top : crop_top + history_viewport_height
        ]
        rows.append(history_grid)

    rows.append(first_row)
    return np.vstack(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Display the latest realtime online-training window."
    )
    parser.add_argument("--base_path", default="data/different_types")
    parser.add_argument("--case_name", required=True)
    parser.add_argument("--experiments_dir", default="experiments_online")
    parser.add_argument("--realtime_dir", default=None)
    parser.add_argument("--cam_idx", type=int, default=0)
    parser.add_argument(
        "--point_mode",
        choices=("original", "surface", "all"),
        default="surface",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--pred_color", type=parse_color, default=(0, 255, 0))
    parser.add_argument("--gt_color", type=parse_color, default=(0, 0, 255))
    parser.add_argument("--controller_color", type=parse_color, default=(255, 0, 0))
    parser.add_argument("--draw_gt", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--draw_controller", action="store_true")
    parser.add_argument(
        "--max_windows",
        type=int,
        default=4,
        help="Number of window columns visible at once; all windows remain retained.",
    )
    parser.add_argument(
        "--max_history",
        type=int,
        default=10,
        help="Recent prediction iterations retained per window; 0 keeps all.",
    )
    parser.add_argument(
        "--history_every",
        type=int,
        default=1,
        help="Capture one prediction snapshot every N exported iterations.",
    )
    parser.add_argument(
        "--visible_history_rows",
        type=int,
        default=1,
        help="Prediction-history rows visible between raw and first-seen rows.",
    )
    parser.add_argument("--tile_width", type=int, default=360)
    parser.add_argument(
        "--frame_cache_size",
        type=int,
        default=256,
        help="Maximum number of resized source frames cached in memory.",
    )
    parser.add_argument("--poll_sec", type=float, default=0.1)
    return parser.parse_args()


def main():
    args = parse_args()
    args.max_windows = max(1, int(args.max_windows))
    args.max_history = max(0, int(args.max_history))
    args.history_every = max(1, int(args.history_every))
    args.visible_history_rows = max(1, int(args.visible_history_rows))
    args.tile_width = max(1, int(args.tile_width))
    args.frame_cache_size = max(1, int(args.frame_cache_size))
    realtime_dir = (
        Path(args.realtime_dir)
        if args.realtime_dir is not None
        else Path(args.experiments_dir) / args.case_name / "realtime"
    )
    latest_path = realtime_dir / "latest_window.npz"
    first_seen_dir = realtime_dir / "first_seen"

    intrinsic, w2c, image_width, image_height = load_camera(
        args.base_path, args.case_name, args.cam_idx
    )
    args.tile_height = max(
        1, int(round(image_height * float(args.tile_width) / float(image_width)))
    )
    display_intrinsic = intrinsic.copy()
    display_intrinsic[0, :] *= float(args.tile_width) / float(image_width)
    display_intrinsic[1, :] *= float(args.tile_height) / float(image_height)
    frame_cache = FrameCache(args.frame_cache_size)
    window_histories = {}
    latest_export = None
    last_mtime_ns = None
    last_first_seen_mtime_ns = None
    paused = False
    frame_period_sec = 1.0 / max(args.fps, 1e-6)
    viewport = {
        "scroll_x_px": 0,
        "scroll_y_px": 0,
        "max_scroll_x_px": 0,
        "max_scroll_y_px": 0,
        "dragging": False,
        "drag_start_x": 0,
        "drag_start_y": 0,
        "drag_start_scroll_x_px": 0,
        "drag_start_scroll_y_px": 0,
        "auto_follow_x": True,
        "auto_follow_y": True,
    }

    cv2.namedWindow("online windows", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("online windows", handle_mouse, viewport)
    print(f"[viewer] waiting for {latest_path}")
    print(
        "[viewer] drag with left mouse to scroll windows/history; "
        "keys: q/esc quit, space pause, arrows scroll, "
        "l/o newest/oldest window, u/d newest/oldest history, "
        "+/- visible windows"
    )

    while True:
        if first_seen_dir.is_dir():
            first_seen_mtime_ns = os.stat(first_seen_dir).st_mtime_ns
            if first_seen_mtime_ns != last_first_seen_mtime_ns:
                discovered = discover_first_seen_windows(
                    window_histories, first_seen_dir
                )
                if discovered:
                    print(
                        "[viewer] discovered first-seen windows "
                        f"{sorted(discovered)}"
                    )
                last_first_seen_mtime_ns = first_seen_mtime_ns

        if latest_path.exists():
            mtime_ns = os.stat(latest_path).st_mtime_ns
            if mtime_ns != last_mtime_ns:
                loaded = load_latest_window(latest_path)
                if loaded is not None:
                    latest_export = loaded
                    added, captured = update_window_histories(
                        window_histories,
                        latest_export,
                        first_seen_dir,
                        args.max_history,
                        args.history_every,
                    )
                    print(
                        "[viewer] "
                        f"iter={scalar_int(latest_export['iteration'])}, "
                        f"windows={sorted(window_histories)}, "
                        f"added={added}, captured={captured}"
                    )
                    last_mtime_ns = mtime_ns

        if window_histories:
            render_started = time.perf_counter()
            image = render_history_view(
                args,
                window_histories,
                display_intrinsic,
                w2c,
                viewport,
                frame_cache,
            )
            cv2.imshow("online windows", image)
            render_elapsed = time.perf_counter() - render_started
            remaining_ms = max(
                1, int(round((frame_period_sec - render_elapsed) * 1000.0))
            )
            key = cv2.waitKeyEx(remaining_ms if not paused else 100)
        else:
            key = cv2.waitKeyEx(max(1, int(args.poll_sec * 1000)))

        if key in (27, ord("q")):
            break
        if key == ord(" "):
            paused = not paused
        elif key in (ord("+"), ord("=")):
            args.max_windows += 1
            print(f"[viewer] visible_windows={args.max_windows}")
        elif key in (ord("-"), ord("_")):
            args.max_windows = max(1, args.max_windows - 1)
            print(f"[viewer] visible_windows={args.max_windows}")
        elif key in (ord("l"),):
            scroll_to_latest(viewport)
        elif key in (ord("o"),):
            scroll_to_oldest(viewport)
        elif key in (ord("u"),):
            scroll_to_newest_history(viewport)
        elif key in (ord("d"),):
            scroll_to_oldest_history(viewport)
        elif key in (81, 65361, 2424832, ord("[")):
            viewport["auto_follow_x"] = False
            viewport["scroll_x_px"] -= int(args.tile_width)
            clamp_scroll(viewport)
        elif key in (83, 65363, 2555904, ord("]")):
            viewport["auto_follow_x"] = False
            viewport["scroll_x_px"] += int(args.tile_width)
            clamp_scroll(viewport)
        elif key in (82, 65362, 2490368):
            viewport["auto_follow_y"] = False
            viewport["scroll_y_px"] -= int(args.tile_height)
            clamp_scroll(viewport)
        elif key in (84, 65364, 2621440):
            viewport["auto_follow_y"] = False
            viewport["scroll_y_px"] += int(args.tile_height)
            clamp_scroll(viewport)

        if not paused:
            for state in window_histories.values():
                segment_len = int(state["shared"]["pred_points"].shape[0])
                state["frame_idx"] = (int(state["frame_idx"]) + 1) % segment_len

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
