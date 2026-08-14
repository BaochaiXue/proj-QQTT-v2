"""Rendering helpers: WSLG Open3D env setup and diagnostic video writers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

WSLG_OPEN3D_ENV_UNSET_KEYS = (
    "VK_ICD_FILENAMES",
    "__GLX_VENDOR_LIBRARY_NAME",
    "__EGL_VENDOR_LIBRARY_FILENAMES",
)
WSLG_OPEN3D_ENV_DEFAULTS = {
    "WAYLAND_DISPLAY": "",
    "EGL_PLATFORM": "x11",
    "GALLIUM_DRIVER": "d3d12",
    "MESA_LOADER_DRIVER_OVERRIDE": "d3d12",
    "LIBGL_ALWAYS_SOFTWARE": "0",
    "QQTT_WSLG_OPEN3D_FAST_EXIT": "1",
    "MESA_D3D12_DEFAULT_ADAPTER_NAME": "NVIDIA",
}


def running_under_wsl() -> bool:
    """Return the running under wsl."""
    if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"):
        return True
    try:
        return "microsoft" in Path("/proc/version").read_text(encoding="utf-8").lower()
    except OSError:
        return False


def apply_wslg_open3d_env_defaults() -> dict[str, str]:
    """Apply wslg open3d env defaults."""
    if os.environ.get("QQTT_DISABLE_WSLG_OPEN3D_DEFAULTS") == "1":
        return {}
    if not running_under_wsl():
        return {}

    applied: dict[str, str] = {}
    for key in WSLG_OPEN3D_ENV_UNSET_KEYS:
        if key in os.environ:
            os.environ.pop(key, None)
            applied[key] = "<unset>"
    for key, value in WSLG_OPEN3D_ENV_DEFAULTS.items():
        # Respect an explicit user GPU-adapter choice; force the other defaults.
        if key == "MESA_D3D12_DEFAULT_ADAPTER_NAME" and key in os.environ:
            continue
        if os.environ.get(key) != value:
            os.environ[key] = value
            applied[key] = value
    return applied


# ---------------------------------------------------------------------------
# Diagnostic video rendering (PhysTwin strict-product headless finalize)
# ---------------------------------------------------------------------------


def _load_rgb(path: Path) -> np.ndarray:
    """Load RGB."""
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def open_video_writer(path: Path, *, size: tuple[int, int], fps: float = 30.0):
    """Open video writer."""
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = int(size[0]), int(size[1])
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height)
    )
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
    """Render tracking 2d video."""
    import cv2

    writer = open_video_writer(path, size=size)
    width, height = int(size[0]), int(size[1])
    is_object = np.asarray(query_is_object, dtype=bool).reshape(-1)
    is_controller = np.asarray(query_is_controller, dtype=bool).reshape(-1)
    for frame_idx, row in enumerate(rows):
        rgb = _load_rgb(capture_dir / str(row["rgb_path"]))
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        src_h, src_w = frame.shape[:2]
        if (src_w, src_h) != (width, height):
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        # Track coordinates are in source pixels; scale them into the output size.
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
            # BGR color code: green = object query, red = controller query,
            # light gray = neither semantic class.
            color = (
                (60, 220, 60)
                if idx < len(is_object) and is_object[idx]
                else (40, 80, 255)
            )
            if (
                idx < len(is_controller)
                and not is_object[idx]
                and not is_controller[idx]
            ):
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
    """Padded world-XY bounds over all finite, non-zero points.

    Zero-norm points (the "no depth" sentinel) are excluded so they do not
    drag the view toward the origin; an 8% margin keeps dots off the frame
    edge, and a unit box is the fallback when nothing is valid.
    """
    chunks: list[np.ndarray] = []
    for arr in arrays:
        pts = np.asarray(arr, dtype=np.float32).reshape(-1, 3)
        finite = np.isfinite(pts).all(axis=1) & (np.linalg.norm(pts, axis=1) > 0.0)
        if np.any(finite):
            chunks.append(pts[finite, :2])
    if not chunks:
        return np.array([-1.0, -1.0], dtype=np.float32), np.array(
            [1.0, 1.0], dtype=np.float32
        )
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
    """Scatter world points onto a top-down XY view; returns the drawn count."""
    import cv2

    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    finite = np.isfinite(pts).all(axis=1) & (np.linalg.norm(pts, axis=1) > 0.0)
    pts = pts[finite]
    if len(pts) == 0:
        return 0
    lo, hi = bounds
    height, width = frame.shape[:2]
    span = np.maximum(hi - lo, np.float32(1e-6))
    # Fixed pixel margins leave room for the HUD text; world +Y points up, so
    # flip the row axis after mapping.
    px = np.clip(
        ((pts[:, 0] - lo[0]) / span[0] * (width - 60) + 30).astype(np.int64),
        0,
        width - 1,
    )
    py = np.clip(
        ((pts[:, 1] - lo[1]) / span[1] * (height - 80) + 50).astype(np.int64),
        0,
        height - 1,
    )
    py = height - 1 - py
    for x, y in zip(px, py):
        cv2.circle(
            frame, (int(x), int(y)), int(radius), color_bgr, -1, lineType=cv2.LINE_AA
        )
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
    """Render world track video."""
    import cv2

    writer = open_video_writer(path, size=size)
    frame_count = max(
        int(np.asarray(object_points).shape[0]),
        int(np.asarray(controller_points).shape[0]),
        1,
    )
    # Bounds are computed once over the whole clip so the view does not jitter
    # frame to frame.
    bounds = _world_xy_bounds(object_points, controller_points)
    width, height = int(size[0]), int(size[1])
    for frame_idx in range(frame_count):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Indices clamp to the last frame so a shorter object/controller/valid
        # array simply holds its final state.
        obj = np.asarray(
            object_points[min(frame_idx, max(0, object_points.shape[0] - 1))],
            dtype=np.float32,
        ).reshape(-1, 3)
        valid = np.asarray(
            object_valid[min(frame_idx, max(0, object_valid.shape[0] - 1))], dtype=bool
        ).reshape(-1)
        if len(valid) == len(obj):
            obj = obj[valid]
        ctrl = np.asarray(
            controller_points[min(frame_idx, max(0, controller_points.shape[0] - 1))],
            dtype=np.float32,
        ).reshape(-1, 3)
        obj_count = _draw_world_points(
            frame, obj, bounds=bounds, color_bgr=(50, 220, 80), radius=2
        )
        ctrl_count = _draw_world_points(
            frame, ctrl, bounds=bounds, color_bgr=(40, 40, 255), radius=5
        )
        cv2.putText(
            frame,
            title,
            (18, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
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


def _render_empty_video(
    path: Path, *, frame_count: int, label: str, size: tuple[int, int] = (640, 360)
) -> None:
    """Render empty video."""
    import cv2

    writer = open_video_writer(path, size=size)
    width, height = int(size[0]), int(size[1])
    count = max(1, int(frame_count))
    for frame_idx in range(count):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(
            frame,
            label,
            (24, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"frame={frame_idx}",
            (24, 86),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (180, 220, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)
    writer.release()
