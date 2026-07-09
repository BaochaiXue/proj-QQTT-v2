"""Camera model loading, projection, and shared IO helpers for the viewer.

Extracted verbatim from ``visualize_track.py`` as part of a behavior-preserving
file split. Foundational module: it defines the low-level pickle/JSON IO helpers
and camera calibration parsers shared by the other ``viz_*`` modules.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import pickle
import sys
from typing import Any, Mapping

import numpy as np


DEFAULT_IMAGE_SIZE = (1280, 720)


@dataclass(frozen=True)
class CameraModel:
    """Camera intrinsics, pose, image size, and optional playback FPS."""

    intrinsic: np.ndarray
    camera_to_world: np.ndarray
    image_size: tuple[int, int]
    metadata_fps: float | None


# --- Lazy imports and file loading -------------------------------------------


def _require_cv2() -> Any:
    """Return validated cv2."""
    import cv2

    return cv2


def _install_numpy_pickle_aliases() -> None:
    """Allow pickles written by NumPy 2.x to load in older NumPy runtimes."""
    try:
        import numpy.core as numpy_core
    except Exception:
        return
    sys.modules.setdefault("numpy._core", numpy_core)
    for name in ("numeric", "multiarray", "umath", "_multiarray_umath"):
        try:
            module = __import__(f"numpy.core.{name}", fromlist=[name])
        except Exception:
            continue
        sys.modules.setdefault(f"numpy._core.{name}", module)


def load_pickle(path: str | Path) -> Any:
    """Load a pickle while tolerating NumPy 2.x module aliases."""
    _install_numpy_pickle_aliases()
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def read_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON object, returning an empty dict for missing or invalid input."""
    try:
        text = Path(path).read_text(encoding="utf-8")
        return dict(json.loads(text))
    except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError):
        return {}


def normalize_online_dir(path: str | Path) -> Path:
    """Accept either online_data or online_data/chunks."""
    value = Path(path).expanduser()
    if value.name == "chunks":
        return value.parent
    return value


def infer_case_dir(online_dir: Path, case_dir: str | Path | None) -> Path:
    """Resolve the aggregate data directory for an online stream."""
    if case_dir is not None:
        return Path(case_dir).expanduser()
    if online_dir.name != "online_data":
        raise ValueError(f"online_dir must point to online_data: {online_dir}")
    return online_dir.parent / "data"


# --- Camera model loading and projection -------------------------------------


def _select_camera_array(value: Any, *, cam_idx: int, shape: tuple[int, int]) -> np.ndarray | None:
    # Calibration arrays may be stored per camera (N, *shape) or flat (*shape).
    """Select camera array."""
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape == shape:
        return arr
    if arr.ndim == 3 and arr.shape[1:] == shape and int(arr.shape[0]) > 0:
        idx = min(max(int(cam_idx), 0), int(arr.shape[0]) - 1)
        return np.asarray(arr[idx], dtype=np.float64)
    return None


def _first_case_image_size(case_path: Path, *, cam_idx: int) -> tuple[int, int] | None:
    """Probe case images for the (width, height) of the first readable frame."""
    cv2 = _require_cv2()
    patterns = [
        case_path / "color" / str(int(cam_idx)) / "*.png",
        case_path / "color" / str(int(cam_idx)) / "*.jpg",
        # Fake-live capture dirs store viewer backgrounds under input_rgb,
        # without the aligned-case color/0 sidecar layout.
        case_path / "input_rgb" / "*.png",
        case_path / "input_rgb" / "*.jpg",
    ]
    for pattern in patterns:
        for path in sorted(pattern.parent.glob(pattern.name)):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            height, width = image.shape[:2]
            if int(width) > 0 and int(height) > 0:
                return (int(width), int(height))
    return None


def _intrinsic_matrix_from_metadata(value: Any, *, cam_idx: int) -> np.ndarray | None:
    """Return the intrinsic matrix from metadata."""
    if isinstance(value, Mapping):
        try:
            fx = float(value["fx"])
            fy = float(value["fy"])
            cx = float(value["cx"])
            cy = float(value["cy"])
        except (KeyError, TypeError, ValueError):
            return None
        if not all(math.isfinite(v) for v in (fx, fy, cx, cy)):
            return None
        return np.asarray(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
    matrix = _select_camera_array(value, cam_idx=cam_idx, shape=(3, 3))
    if matrix is not None:
        return np.asarray(matrix, dtype=np.float64)
    return None


def _camera_to_world_from_metadata(
    metadata: Mapping[str, Any],
    *,
    cam_idx: int,
) -> np.ndarray | None:
    # Fake-live captures publish the single-camera pose in metadata instead of
    # writing an aligned-case calibrate.pkl; use it for offline video export.
    """Return the camera to world from metadata."""
    for key in ("camera_to_world_c2w", "camera_to_world"):
        matrix = _select_camera_array(metadata.get(key), cam_idx=cam_idx, shape=(4, 4))
        if matrix is not None:
            return np.asarray(matrix, dtype=np.float64)
    return None


def load_camera_model(case_dir: str | Path, *, cam_idx: int) -> CameraModel:
    """Load intrinsics and camera-to-world calibration from the aggregate case."""
    case_path = Path(case_dir)
    metadata = read_json(case_path / "metadata.json")
    # Image size precedence: metadata "WH" (stored as [width, height]), then
    # the first readable case image, then the 1280x720 default.
    image_size: tuple[int, int] | None = None
    wh_value = metadata.get("WH")
    if wh_value is not None:
        wh = np.asarray(wh_value).reshape(-1)
        if wh.size >= 2:
            width = int(wh[0])
            height = int(wh[1])
            if width > 0 and height > 0:
                image_size = (width, height)
    if image_size is None:
        image_size = _first_case_image_size(case_path, cam_idx=cam_idx)
    if image_size is None:
        image_size = DEFAULT_IMAGE_SIZE
    intrinsic = _intrinsic_matrix_from_metadata(metadata.get("intrinsics"), cam_idx=cam_idx)
    if intrinsic is None:
        intrinsic = np.eye(3, dtype=np.float64)
    calibrate_path = case_path / "calibrate.pkl"
    camera_to_world = None
    if calibrate_path.is_file():
        camera_to_world = _select_camera_array(
            load_pickle(calibrate_path),
            cam_idx=cam_idx,
            shape=(4, 4),
        )
    if camera_to_world is None:
        camera_to_world = _camera_to_world_from_metadata(metadata, cam_idx=cam_idx)
    if camera_to_world is None:
        camera_to_world = np.eye(4, dtype=np.float64)
    fps_value = metadata.get("fps", metadata.get("replay_fps", metadata.get("lossless_input_fps")))
    try:
        metadata_fps = float(fps_value) if fps_value is not None else None
    except (TypeError, ValueError):
        metadata_fps = None
    if metadata_fps is not None and (not math.isfinite(metadata_fps) or metadata_fps <= 0.0):
        metadata_fps = None
    return CameraModel(
        intrinsic=np.asarray(intrinsic, dtype=np.float64),
        camera_to_world=np.asarray(camera_to_world, dtype=np.float64),
        image_size=image_size,
        metadata_fps=metadata_fps,
    )


def project_world_points_to_pixels(
    points: np.ndarray,
    *,
    intrinsic: np.ndarray,
    camera_to_world: np.ndarray,
    image_size: tuple[int, int],
    visibility: np.ndarray | None = None,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Project world-space Demo v6.1 points back onto the selected RGB camera.

    Returns integer pixel coordinates plus the surviving points' original row
    indices, so callers can look up per-point attributes such as colors.
    """
    arr = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if arr.size == 0:
        return np.empty((0, 2), dtype=np.int32), np.empty((0,), dtype=np.int64)
    indices = np.arange(arr.shape[0], dtype=np.int64)
    if visibility is not None:
        vis = np.asarray(visibility, dtype=bool).reshape(-1)
        if vis.shape[0] == arr.shape[0]:
            indices = indices[vis]
            arr = arr[vis]
    step = max(int(stride), 1)
    if step > 1:
        indices = indices[::step]
        arr = arr[::step]
    if arr.size == 0:
        return np.empty((0, 2), dtype=np.int32), np.empty((0,), dtype=np.int64)
    world_to_camera = np.linalg.inv(np.asarray(camera_to_world, dtype=np.float64))
    homogeneous = np.concatenate([arr, np.ones((arr.shape[0], 1), dtype=np.float64)], axis=1)
    camera_points = (world_to_camera @ homogeneous.T).T[:, :3]
    finite = np.all(np.isfinite(camera_points), axis=1)
    positive_depth = camera_points[:, 2] > 1e-6
    valid = finite & positive_depth
    if not np.any(valid):
        return np.empty((0, 2), dtype=np.int32), np.empty((0,), dtype=np.int64)
    camera_points = camera_points[valid]
    indices = indices[valid]
    projected = (np.asarray(intrinsic, dtype=np.float64) @ camera_points.T).T
    xy = projected[:, :2] / projected[:, 2:3]
    pixels = np.rint(xy).astype(np.int32)
    width, height = int(image_size[0]), int(image_size[1])
    in_bounds = (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    return pixels[in_bounds], indices[in_bounds]
