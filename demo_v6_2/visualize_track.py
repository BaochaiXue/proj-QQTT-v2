#!/usr/bin/env python3
"""Track visualization for Demo v6.1 object/controller point chunks.

The viewer can render historical chunk files or follow a live run. In
side-by-side mode the left panel follows camera RGB input while the right panel
chooses the final_data frame whose source timestamp best matches the desired
camera-to-output latency.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
import json
import math
from pathlib import Path
import pickle
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np


DEFAULT_WINDOW_NAME = "Demo v6.1 visualize track"
DEFAULT_IMAGE_SIZE = (1280, 720)
DEFAULT_OBJECT_RADIUS = 3
DEFAULT_CONTROLLER_RADIUS = 6
RENDER_MODE_RGB_OVERLAY = "rgb-overlay"
RENDER_MODE_SAM3D_FINAL_DATA = "sam3d-final-data"
RENDER_MODES = (RENDER_MODE_RGB_OVERLAY, RENDER_MODE_SAM3D_FINAL_DATA)
LAYOUT_OUTPUT_ONLY = "output-only"
LAYOUT_SIDE_BY_SIDE = "side-by-side"
LAYOUTS = (LAYOUT_SIDE_BY_SIDE, LAYOUT_OUTPUT_ONLY)
DEFAULT_RIGHT_BLANK_LABEL = "waiting for first final_data chunk"


# --- Data types and live-playback state --------------------------------------


@dataclass(frozen=True)
class CameraModel:
    """Camera intrinsics, pose, image size, and optional playback FPS."""

    intrinsic: np.ndarray
    camera_to_world: np.ndarray
    image_size: tuple[int, int]
    metadata_fps: float | None


@dataclass(frozen=True)
class InputRgbFrame:
    """One RGB input frame plus source-frame timing metadata."""

    seq: int
    image_bgr: np.ndarray
    path: Path | None
    source_frame_index: int | None
    source_timestamp_s: float | None


class InputReceiveTimeline:
    """Incrementally cache fake-camera receive times keyed by source frame.

    The timeline file is append-only during live capture, so the reader keeps a
    byte offset and only parses newly completed JSONL rows.
    """

    def __init__(self, timeline_path: str | Path | None) -> None:
        """Initialize InputReceiveTimeline."""
        self.timeline_path = None if timeline_path is None else Path(timeline_path).expanduser()
        self.receive_times: dict[int, float] = {}
        self._offset = 0

    def refresh(self) -> None:
        """Read any newly completed receive-timeline rows."""
        path = self.timeline_path
        if path is None:
            return
        try:
            size = path.stat().st_size
        except OSError:
            return
        if int(size) < int(self._offset):
            self._offset = 0
            self.receive_times.clear()
        try:
            with path.open("rb") as handle:
                handle.seek(self._offset)
                for raw_line in handle:
                    self._ingest_line(raw_line)
                self._offset = int(handle.tell())
        except OSError:
            return

    def receive_time(self, source_frame_index: int | None) -> float | None:
        """Return the cached receive time for one source frame."""
        if source_frame_index is None:
            return None
        try:
            key = int(source_frame_index)
        except (TypeError, ValueError):
            return None
        value = self.receive_times.get(key)
        if value is None:
            return None
        receive_s = float(value)
        if not math.isfinite(receive_s):
            return None
        return receive_s

    def _ingest_line(self, raw_line: bytes) -> None:
        """Ingest one JSONL timeline row."""
        text = raw_line.decode("utf-8", errors="replace").strip()
        if not text:
            return
        try:
            row = dict(json.loads(text))
        except (json.JSONDecodeError, TypeError, ValueError):
            return
        try:
            source_frame_index = int(row["source_frame_index"])
            receive_perf_s = float(row["receive_perf_s"])
        except (KeyError, TypeError, ValueError):
            return
        if not math.isfinite(receive_perf_s):
            return
        self.receive_times[source_frame_index] = receive_perf_s


@dataclass
class OutputStreamPlaybackCursor:
    """Small playback state machine for live output-frame progression."""

    fps: float
    output_index: int = 0
    last_step_s: float | None = None

    def advance(self, *, latest: int, now_s: float, paused: bool) -> int:
        """Advance playback by elapsed time while staying within loaded output.

        Advances at most one frame per call, so the caller's poll cadence must
        be at least as fast as the playback FPS to keep up.
        """
        latest_index = max(0, int(latest))
        self.output_index = min(max(int(self.output_index), 0), latest_index)
        now = float(now_s)
        if self.last_step_s is None:
            self.last_step_s = now
            return int(self.output_index)
        if paused or self.output_index >= latest_index:
            self.last_step_s = now
            return int(self.output_index)
        period_s = 1.0 / max(float(self.fps), 1e-6)
        elapsed_s = max(0.0, now - float(self.last_step_s))
        if elapsed_s + 1e-9 < period_s:
            return int(self.output_index)
        self.output_index = min(latest_index, int(self.output_index) + 1)
        self.last_step_s = now
        return int(self.output_index)

    def seek(self, index: int, *, latest: int, now_s: float | None = None) -> int:
        """Move playback to a bounded output-frame index."""
        self.output_index = min(max(int(index), 0), max(0, int(latest)))
        if now_s is not None:
            self.last_step_s = float(now_s)
        return int(self.output_index)


@dataclass
class CameraToFinalDataFpsMeter:
    """Estimate online publish throughput from newly appended output frames."""

    _last_update_s: float | None = None
    _fps: float | None = None

    def seed(self, fps: float | None) -> float | None:
        """Seed the meter with a historical FPS estimate when available."""
        if fps is None:
            return self._fps
        value = float(fps)
        if not math.isfinite(value) or value <= 0.0:
            return self._fps
        self._fps = value
        return self._fps

    def update(self, *, appended_frames: int, now_s: float) -> float | None:
        """Update the FPS estimate from the number of newly appended frames."""
        count = int(appended_frames)
        if count <= 0:
            return self._fps
        now = float(now_s)
        if self._last_update_s is None:
            self._last_update_s = now
            return None
        elapsed = now - float(self._last_update_s)
        self._last_update_s = now
        if elapsed <= 1e-9:
            return self._fps
        self._fps = float(count) / elapsed
        return self._fps


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


def load_fake_input_frame_total(capture_dir: str | Path | None) -> int | None:
    """Return the expected fake-live RGB frame count when metadata provides it."""
    if capture_dir is None:
        return None
    metadata = read_json(Path(capture_dir).expanduser() / "metadata.json")
    if str(metadata.get("input_source") or "") != "fake-live":
        return None
    try:
        total = int(metadata.get("recording_frame_count"))
    except (TypeError, ValueError):
        return None
    return total if total > 0 else None


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


# --- Panel composition and overlay drawing -----------------------------------


def _blank_image(image_size: tuple[int, int]) -> np.ndarray:
    """Return the blank image."""
    width, height = int(image_size[0]), int(image_size[1])
    return np.zeros((height, width, 3), dtype=np.uint8)


def _panel_image(image: np.ndarray | None, *, image_size: tuple[int, int]) -> np.ndarray:
    """Return the panel image."""
    cv2 = _require_cv2()
    width, height = int(image_size[0]), int(image_size[1])
    if image is None:
        return _blank_image((width, height))
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return _blank_image((width, height))
    bgr = np.ascontiguousarray(arr[:, :, :3], dtype=np.uint8)
    if bgr.shape[1] == width and bgr.shape[0] == height:
        return bgr.copy()
    return cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)


def _draw_panel_label(image: np.ndarray, text: str, *, right: bool = False) -> None:
    """Draw panel label."""
    if image.shape[0] < 40 or image.shape[1] < 160:
        return
    cv2 = _require_cv2()
    origin = (12, 28)
    if right:
        (text_width, _text_height), _baseline = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            1,
        )
        origin = (max(12, int(image.shape[1]) - int(text_width) - 12), 28)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_camera_to_final_data_fps_overlay(image: np.ndarray, fps: float | None) -> None:
    """Draw camera to final data FPS overlay."""
    if image.shape[0] < 40 or image.shape[1] < 280:
        return
    cv2 = _require_cv2()
    if fps is None or not math.isfinite(float(fps)):
        text = "camera->final_data -- FPS"
    else:
        text = f"camera->final_data {float(fps):.1f} FPS"
    (text_width, _text_height), _baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        1,
    )
    origin = (max(12, int(image.shape[1]) - int(text_width) - 12), 28)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_fake_rgb_frame_counter_overlay(
    image: np.ndarray,
    input_frame: InputRgbFrame | None,
    *,
    fake_input_frame_total: int | None,
) -> None:
    """Draw fake RGB frame counter overlay."""
    if image.shape[0] < 70 or image.shape[1] < 180:
        return
    # The counter only applies to fake-live replays, where the recording's
    # total frame count is known ahead of time (1-based for display).
    if input_frame is None or input_frame.source_frame_index is None or fake_input_frame_total is None:
        return
    try:
        current = int(input_frame.source_frame_index) + 1
        total = int(fake_input_frame_total)
    except (TypeError, ValueError):
        return
    if current <= 0 or total <= 0:
        return
    text = f"RGB frame {current}/{total}"
    cv2 = _require_cv2()
    origin = (12, 54)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_center_label(image: np.ndarray, text: str) -> None:
    """Draw center label."""
    if image.shape[0] < 60 or image.shape[1] < 160:
        return
    cv2 = _require_cv2()
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
    origin = (
        max(12, (int(image.shape[1]) - int(text_width)) // 2),
        max(32, (int(image.shape[0]) + int(text_height)) // 2),
    )
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1, cv2.LINE_AA)


# --- Input RGB timeline loading ----------------------------------------------


def _input_rgb_path_from_row(row: Mapping[str, Any], *, capture_dir: Path) -> Path | None:
    """Return the input RGB path from row."""
    value = row.get("input_rgb_path")
    if value is not None and str(value).strip():
        path = Path(str(value))
        return path if path.is_absolute() else capture_dir / path
    seq = row.get("seq")
    if seq is not None:
        try:
            seq_int = int(seq)
        except (TypeError, ValueError):
            seq_int = -1
        if seq_int >= 0:
            for directory in ("input_rgb", "rgb"):
                path = capture_dir / directory / f"{seq_int:06d}.png"
                if path.is_file():
                    return path
    return None


def _input_rgb_frame_from_row(row: Mapping[str, Any], *, capture_dir: Path) -> InputRgbFrame | None:
    """Return the input RGB frame from row."""
    cv2 = _require_cv2()
    path = _input_rgb_path_from_row(row, capture_dir=capture_dir)
    if path is None or not path.is_file():
        return None
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    try:
        seq = int(row.get("seq", 0))
    except (TypeError, ValueError):
        seq = 0
    source_frame_index = row.get("source_frame_index")
    try:
        source_frame_index = None if source_frame_index is None else int(source_frame_index)
    except (TypeError, ValueError):
        source_frame_index = None
    source_timestamp_s = row.get("source_timestamp_s")
    try:
        source_timestamp_s = None if source_timestamp_s is None else float(source_timestamp_s)
    except (TypeError, ValueError):
        source_timestamp_s = None
    return InputRgbFrame(
        seq=seq,
        image_bgr=np.ascontiguousarray(image, dtype=np.uint8),
        path=path,
        source_frame_index=source_frame_index,
        source_timestamp_s=source_timestamp_s,
    )


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    """Read JSONL rows."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    rows: list[dict[str, Any]] = []
    for line in lines:
        text = line.strip()
        if not text:
            continue
        try:
            rows.append(dict(json.loads(text)))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    return rows


def load_latest_input_rgb_frame(timeline_path: str | Path, *, capture_dir: str | Path) -> InputRgbFrame | None:
    """Load the newest RGB input frame referenced by an input timeline."""
    capture_path = Path(capture_dir).expanduser()
    for row in reversed(_read_jsonl_rows(Path(timeline_path).expanduser())):
        frame = _input_rgb_frame_from_row(row, capture_dir=capture_path)
        if frame is not None:
            return frame
    return None


def load_input_rgb_frames(timeline_path: str | Path, *, capture_dir: str | Path) -> list[InputRgbFrame]:
    """Load all RGB input frames referenced by an input timeline."""
    capture_path = Path(capture_dir).expanduser()
    frames: list[InputRgbFrame] = []
    for row in _read_jsonl_rows(Path(timeline_path).expanduser()):
        frame = _input_rgb_frame_from_row(row, capture_dir=capture_path)
        if frame is not None:
            frames.append(frame)
    return frames


def load_input_rgb_background_paths(
    timeline_path: str | Path,
    *,
    capture_dir: str | Path,
) -> dict[int, Path]:
    """Map original source-frame ids to fake-live RGB images for output export."""
    capture_path = Path(capture_dir).expanduser()
    paths: dict[int, Path] = {}
    for row in _read_jsonl_rows(Path(timeline_path).expanduser()):
        try:
            source_frame_index = int(row["source_frame_index"])
        except (KeyError, TypeError, ValueError):
            continue
        path = _input_rgb_path_from_row(row, capture_dir=capture_path)
        if path is None or not path.is_file():
            continue
        paths[source_frame_index] = path
    return paths


def render_side_by_side_frame(
    *,
    input_frame: InputRgbFrame | None,
    output_frame: np.ndarray | None,
    image_size: tuple[int, int],
    right_blank_label: str = DEFAULT_RIGHT_BLANK_LABEL,
    camera_to_final_data_fps: float | None = None,
    fake_input_frame_total: int | None = None,
    input_to_display_latency_s: float | None = None,
    show_latency_overlay: bool = True,
) -> np.ndarray:
    """Compose one RGB-input/final_data-output frame for display or video."""
    left = _panel_image(None if input_frame is None else input_frame.image_bgr, image_size=image_size)
    right = _panel_image(output_frame, image_size=image_size)
    if input_frame is None:
        _draw_center_label(left, "waiting for RGB input")
    if output_frame is None:
        _draw_center_label(right, str(right_blank_label))
    _draw_panel_label(left, "RGB input")
    _draw_fake_rgb_frame_counter_overlay(left, input_frame, fake_input_frame_total=fake_input_frame_total)
    _draw_camera_to_final_data_fps_overlay(left, camera_to_final_data_fps)
    _draw_panel_label(right, "final_data output", right=True)
    if show_latency_overlay:
        # Latency HUD: right-aligned, one 26px line below the panel label.
        text = format_input_display_latency(input_to_display_latency_s)
        if right.shape[0] >= 40 and right.shape[1] >= 220:
            cv2 = _require_cv2()
            (text_width, _text_height), _baseline = cv2.getTextSize(
                text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                1,
            )
            origin = (max(12, int(right.shape[1]) - int(text_width) - 12), 54)
            cv2.putText(right, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(right, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    return np.ascontiguousarray(np.concatenate([left, right], axis=1), dtype=np.uint8)


def read_background(
    case_dir: Path,
    *,
    cam_idx: int,
    source_frame: int,
    image_size: tuple[int, int],
    use_background: bool,
    frame_path: Path | None = None,
) -> np.ndarray:
    """Return the source RGB frame when present, otherwise a black canvas."""
    if not use_background:
        return _blank_image(image_size)
    cv2 = _require_cv2()
    if frame_path is not None and frame_path.is_file():
        image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if image is not None:
            return image
    # Probe the aligned-case color/<cam>/ layout before the fake-live
    # input_rgb/ layout, trying plain and zero-padded names, png before jpg.
    color_dir = case_dir / "color" / str(int(cam_idx))
    input_rgb_dir = case_dir / "input_rgb"
    for path in (
        color_dir / f"{int(source_frame)}.png",
        color_dir / f"{int(source_frame):06d}.png",
        color_dir / f"{int(source_frame)}.jpg",
        color_dir / f"{int(source_frame):06d}.jpg",
        input_rgb_dir / f"{int(source_frame)}.png",
        input_rgb_dir / f"{int(source_frame):06d}.png",
        input_rgb_dir / f"{int(source_frame)}.jpg",
        input_rgb_dir / f"{int(source_frame):06d}.jpg",
    ):
        if not path.is_file():
            continue
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is not None:
            return image
    return _blank_image(image_size)


# --- Chunk frame timing and latency ------------------------------------------


def _chunk_frame_count(chunk: Mapping[str, Any]) -> int:
    """Return the chunk frame count."""
    for key in ("object_points", "controller_points"):
        value = chunk.get(key)
        if value is not None:
            return int(np.asarray(value).shape[0])
    return 0


def _source_frame_for_chunk_frame(chunk: Mapping[str, Any], local_frame: int) -> int:
    """Return the source frame for chunk frame."""
    source_indices = chunk.get("source_frame_indices")
    if source_indices is not None:
        try:
            return int(source_indices[int(local_frame)])
        except (IndexError, TypeError, ValueError):
            pass
    return int(chunk.get("start_frame", 0)) + int(local_frame)


def input_display_latency_s(
    chunk: Mapping[str, Any],
    *,
    local_frame: int,
    receive_times: Mapping[int, float],
    now_s: float,
) -> float | None:
    """Measure input receive-to-display latency for one output chunk frame."""
    source_frame = _source_frame_for_chunk_frame(chunk, int(local_frame))
    try:
        receive_perf_s = float(receive_times[int(source_frame)])
    except (KeyError, TypeError, ValueError):
        return None
    now_value = float(now_s)
    latency_s = now_value - receive_perf_s
    if not math.isfinite(latency_s) or latency_s < 0.0:
        return None
    return latency_s


def format_input_display_latency(latency_s: float | None) -> str:
    """Format an input-display latency value for the viewer overlay."""
    if latency_s is None:
        return "input->display --"
    try:
        value = float(latency_s)
    except (TypeError, ValueError):
        return "input->display --"
    if not math.isfinite(value) or value < 0.0:
        return "input->display --"
    rounded = Decimal(str(value + 1e-9)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return f"input->display {rounded:.2f}s"


def _input_display_latency_for_output_index(
    output_frames: Sequence[tuple[Mapping[str, Any], int]],
    *,
    output_index: int,
    receive_times: Mapping[int, float],
    now_s: float,
) -> float | None:
    """Return the input display latency for output index."""
    if not output_frames:
        return None
    idx = min(max(int(output_index), 0), len(output_frames) - 1)
    chunk, local_frame = output_frames[idx]
    return input_display_latency_s(
        chunk,
        local_frame=int(local_frame),
        receive_times=receive_times,
        now_s=float(now_s),
    )


def _source_time_for_chunk_frame(
    chunk: Mapping[str, Any],
    local_frame: int,
    *,
    fps: float,
    allow_frame_index_fallback: bool = False,
) -> float | None:
    """Return the source time for chunk frame."""
    source_timestamps = chunk.get("source_timestamps_s")
    if source_timestamps is not None:
        try:
            value = float(source_timestamps[int(local_frame)])
            if math.isfinite(value):
                return value
        except (IndexError, TypeError, ValueError):
            pass
    source_indices = chunk.get("source_frame_indices")
    if allow_frame_index_fallback and source_indices is not None and math.isfinite(float(fps)) and float(fps) > 0.0:
        try:
            return float(source_indices[int(local_frame)]) / float(fps)
        except (IndexError, TypeError, ValueError):
            pass
    return None


def source_time_input_display_latency_s(
    *,
    input_frame: InputRgbFrame | None,
    output_frames: Sequence[tuple[Mapping[str, Any], int]],
    output_index: int,
    fps: float,
) -> float | None:
    """Estimate latency between current input source time and displayed output."""
    if input_frame is None or input_frame.source_timestamp_s is None or not output_frames:
        return None
    idx = min(max(int(output_index), 0), len(output_frames) - 1)
    chunk, local_frame = output_frames[idx]
    output_source_time = _source_time_for_chunk_frame(chunk, int(local_frame), fps=float(fps))
    if output_source_time is None:
        return None
    origin_chunk, origin_local_frame = output_frames[0]
    origin_source_time = _source_time_for_chunk_frame(origin_chunk, int(origin_local_frame), fps=float(fps))
    if origin_source_time is None:
        return None
    try:
        input_source_time = float(input_frame.source_timestamp_s)
        output_time = float(output_source_time)
        origin_time = float(origin_source_time)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (input_source_time, output_time, origin_time)):
        return None
    input_elapsed_s = input_source_time - origin_time
    output_elapsed_s = output_time - origin_time
    latency_s = input_elapsed_s - output_elapsed_s
    if latency_s < 0.0 or not math.isfinite(latency_s):
        return None
    return float(latency_s)


# --- Marker colors -----------------------------------------------------------


def parse_bgr_color(text: str) -> tuple[int, int, int]:
    """Parse a comma-separated B,G,R color triplet."""
    parts = [part.strip() for part in str(text).split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("color must be B,G,R")
    try:
        values = [int(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("color must be B,G,R integers") from exc
    return tuple(max(0, min(255, value)) for value in values)


def _sam3d_rainbow_colors_bgr(chunk: Mapping[str, Any], point_indices: np.ndarray) -> np.ndarray:
    # Colors are keyed to each point's Y height in the chunk's FIRST frame so
    # a point keeps one stable color for the whole chunk. Falls back to a
    # piecewise-linear rainbow when matplotlib is unavailable.
    """Return the SAM3D rainbow colors BGR."""
    indices = np.asarray(point_indices, dtype=np.int64).reshape(-1)
    if indices.size == 0:
        return np.empty((0, 3), dtype=np.uint8)
    object_points = chunk.get("object_points")
    if object_points is None:
        normalized = np.zeros((indices.shape[0],), dtype=np.float32)
    else:
        arr = np.asarray(object_points, dtype=np.float64)
        if arr.ndim != 3 or arr.shape[0] == 0 or arr.shape[2] < 2:
            normalized = np.zeros((indices.shape[0],), dtype=np.float32)
        else:
            y_values = np.asarray(arr[0, :, 1], dtype=np.float64).reshape(-1)
            if y_values.size == 0:
                normalized = np.zeros((indices.shape[0],), dtype=np.float32)
            else:
                finite = np.isfinite(y_values)
                if np.any(finite):
                    y_min = float(np.nanmin(y_values[finite]))
                    y_max = float(np.nanmax(y_values[finite]))
                    span = y_max - y_min
                    if math.isfinite(span) and span > 1e-9:
                        selected_y = y_values[np.clip(indices, 0, y_values.shape[0] - 1)]
                        normalized = np.clip((selected_y - y_min) / span, 0.0, 1.0).astype(np.float32)
                    else:
                        normalized = np.zeros((indices.shape[0],), dtype=np.float32)
                else:
                    normalized = np.zeros((indices.shape[0],), dtype=np.float32)
    try:
        import matplotlib.pyplot as plt

        rgb = np.asarray(plt.cm.rainbow(normalized)[:, :3], dtype=np.float32) * 255.0
    except Exception:
        rgb = np.stack(
            [
                255.0 * normalized,
                255.0 * (1.0 - np.abs(normalized - 0.5) * 2.0),
                255.0 * (1.0 - normalized),
            ],
            axis=1,
        )
    return np.ascontiguousarray(np.clip(rgb, 0, 255).astype(np.uint8)[:, ::-1], dtype=np.uint8)


def _sam3d_rainbow_colors_rgb_float(object_points: np.ndarray, point_count: int) -> np.ndarray:
    # Open3D variant of _sam3d_rainbow_colors_bgr: same first-frame Y-height
    # keying, but returns float RGB in [0, 1] for point-cloud colors.
    """Return the SAM3D rainbow colors RGB float."""
    count = max(0, int(point_count))
    if count == 0:
        return np.empty((0, 3), dtype=np.float64)
    arr = np.asarray(object_points, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[0] == 0 or arr.shape[1] < count or arr.shape[2] < 2:
        normalized = np.zeros((count,), dtype=np.float64)
    else:
        y_values = np.asarray(arr[0, :count, 1], dtype=np.float64)
        finite = np.isfinite(y_values)
        if np.any(finite):
            y_min = float(np.nanmin(y_values[finite]))
            y_max = float(np.nanmax(y_values[finite]))
            span = y_max - y_min
            if math.isfinite(span) and span > 1e-9:
                normalized = np.clip((y_values - y_min) / span, 0.0, 1.0)
            else:
                normalized = np.zeros((count,), dtype=np.float64)
        else:
            normalized = np.zeros((count,), dtype=np.float64)
    try:
        import matplotlib.pyplot as plt

        return np.ascontiguousarray(np.asarray(plt.cm.rainbow(normalized)[:, :3], dtype=np.float64))
    except Exception:
        rgb = np.stack(
            [
                normalized,
                1.0 - np.abs(normalized - 0.5) * 2.0,
                1.0 - normalized,
            ],
            axis=1,
        )
        return np.ascontiguousarray(np.clip(rgb, 0.0, 1.0), dtype=np.float64)


def object_point_colors(
    chunk: Mapping[str, Any],
    *,
    local_frame: int,
    point_indices: np.ndarray,
    mode: str,
) -> np.ndarray:
    """Resolve BGR colors for projected object point markers."""
    mode_value = str(mode)
    if mode_value == "green":
        return np.tile(np.array([[0, 255, 0]], dtype=np.uint8), (point_indices.shape[0], 1))
    if mode_value == "object-colors":
        colors = chunk.get("object_colors")
        if colors is not None:
            arr = np.asarray(colors)
            if arr.ndim == 3 and local_frame < arr.shape[0] and arr.shape[2] >= 3:
                selected = np.asarray(arr[int(local_frame), point_indices, :3], dtype=np.float64)
                if selected.size:
                    if float(np.nanmax(selected)) <= 1.0:
                        selected = selected * 255.0
                    selected = np.clip(selected, 0.0, 255.0).astype(np.uint8)
                    return selected[:, ::-1]
    return _sam3d_rainbow_colors_bgr(chunk, point_indices)


def controller_point_colors(
    chunk: Mapping[str, Any],
    *,
    local_frame: int,
    point_indices: np.ndarray,
    fallback_color: tuple[int, int, int],
) -> np.ndarray:
    """Resolve BGR colors for projected controller point markers."""
    color = np.asarray(fallback_color, dtype=np.uint8).reshape(1, 3)
    return np.tile(color, (point_indices.shape[0], 1))


# --- Chunk-frame rendering and renderer classes ------------------------------


def _draw_sam3d_markers(
    image: np.ndarray,
    pixels: np.ndarray,
    colors: np.ndarray,
    *,
    radius: int,
) -> None:
    """Draw SAM3D markers."""
    if pixels.size == 0:
        return
    cv2 = _require_cv2()
    draw_radius = max(int(radius), 1)
    for (x_value, y_value), color in zip(pixels, colors, strict=False):
        center = (int(x_value), int(y_value))
        color_bgr = tuple(int(value) for value in color)
        cv2.circle(
            image,
            center,
            draw_radius,
            color_bgr,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )


def _draw_status(image: np.ndarray, text: str) -> None:
    """Draw status text on an output image."""
    cv2 = _require_cv2()
    origin = (12, 28)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)


def render_chunk_frame(
    chunk: Mapping[str, Any],
    *,
    local_frame: int,
    case_dir: Path,
    camera: CameraModel,
    cam_idx: int,
    use_background: bool,
    show_invisible_object_points: bool,
    object_stride: int,
    object_radius: int,
    controller_radius: int,
    object_color_mode: str,
    controller_color: tuple[int, int, int],
    fps: float,
    background_frame_paths: Mapping[int, Path] | None = None,
) -> np.ndarray:
    """Draw one online chunk frame as colored object/controller pixels."""
    source_frame = _source_frame_for_chunk_frame(chunk, local_frame)
    frame_path = None
    if background_frame_paths is not None:
        # Fake-live chunks keep the original recording source ids, while saved
        # RGB files are named by receive sequence; the timeline bridges them.
        frame_path = background_frame_paths.get(int(source_frame))
    image = read_background(
        case_dir,
        cam_idx=cam_idx,
        source_frame=source_frame,
        image_size=camera.image_size,
        use_background=use_background,
        frame_path=frame_path,
    )
    image_size = (int(image.shape[1]), int(image.shape[0]))
    object_points = chunk.get("object_points")
    if object_points is not None:
        object_arr = np.asarray(object_points)
        if object_arr.ndim == 3 and int(local_frame) < int(object_arr.shape[0]):
            visibility = None
            if not show_invisible_object_points and chunk.get("object_visibilities") is not None:
                vis_arr = np.asarray(chunk["object_visibilities"])
                if vis_arr.ndim == 2 and int(local_frame) < int(vis_arr.shape[0]):
                    visibility = vis_arr[int(local_frame)]
            object_pixels, object_indices = project_world_points_to_pixels(
                object_arr[int(local_frame)],
                intrinsic=camera.intrinsic,
                camera_to_world=camera.camera_to_world,
                image_size=image_size,
                visibility=visibility,
                stride=object_stride,
            )
            _draw_sam3d_markers(
                image,
                object_pixels,
                object_point_colors(
                    chunk,
                    local_frame=int(local_frame),
                    point_indices=object_indices,
                    mode=object_color_mode,
                ),
                radius=object_radius,
            )
    controller_points = chunk.get("controller_points")
    if controller_points is not None:
        controller_arr = np.asarray(controller_points)
        if controller_arr.ndim == 3 and int(local_frame) < int(controller_arr.shape[0]):
            controller_pixels, controller_indices = project_world_points_to_pixels(
                controller_arr[int(local_frame)],
                intrinsic=camera.intrinsic,
                camera_to_world=camera.camera_to_world,
                image_size=image_size,
                stride=1,
            )
            _draw_sam3d_markers(
                image,
                controller_pixels,
                controller_point_colors(
                    chunk,
                    local_frame=int(local_frame),
                    point_indices=controller_indices,
                    fallback_color=controller_color,
                ),
                radius=controller_radius,
            )
    frame_count = _chunk_frame_count(chunk)
    chunk_id = int(chunk.get("chunk_id", -1))
    _draw_status(
        image,
        f"chunk {chunk_id:06d}  frame {int(local_frame) + 1}/{frame_count}  source {source_frame}  {fps:g} FPS",
    )
    return image


class RgbOverlayRenderer:
    """Render object/controller tracks as overlays on RGB background frames."""

    def __init__(self, *, camera: CameraModel, args: argparse.Namespace, fps: float) -> None:
        """Initialize RgbOverlayRenderer."""
        self._camera = camera
        self._args = args
        self._fps = float(fps)
        self._background_frame_paths = self._load_background_frame_paths(args)

    def _load_background_frame_paths(self, args: argparse.Namespace) -> dict[int, Path]:
        """Load background frame paths."""
        capture_dir = _resolve_capture_dir(args)
        input_timeline = _resolve_input_rgb_timeline(args, capture_dir=capture_dir)
        if capture_dir is None or input_timeline is None:
            return {}
        return load_input_rgb_background_paths(input_timeline, capture_dir=capture_dir)

    def render_frame(self, chunk: Mapping[str, Any], *, local_frame: int, case_dir: Path) -> np.ndarray:
        """Render one chunk frame as an RGB-overlay image."""
        return render_chunk_frame(
            chunk,
            local_frame=int(local_frame),
            case_dir=case_dir,
            camera=self._camera,
            cam_idx=int(self._args.cam_idx),
            use_background=not bool(self._args.no_background),
            show_invisible_object_points=bool(self._args.show_invisible_object_points),
            object_stride=int(self._args.object_stride),
            object_radius=int(self._args.object_radius),
            controller_radius=int(self._args.controller_radius),
            object_color_mode=str(self._args.object_color_mode),
            controller_color=self._args.controller_color,
            fps=self._fps,
            background_frame_paths=self._background_frame_paths,
        )

    def close(self) -> None:
        """Release renderer resources."""
        return None


class Sam3DFinalDataRenderer:
    """Render final_data object/controller points through an Open3D visualizer."""

    def __init__(
        self,
        *,
        image_size: tuple[int, int],
        show_invisible_object_points: bool,
        visible: bool = False,
        window_name: str = "final_data output",
        window_position: tuple[int, int] | None = None,
    ) -> None:
        """Initialize Sam3DFinalDataRenderer."""
        self._image_size = (int(image_size[0]), int(image_size[1]))
        self._show_invisible_object_points = bool(show_invisible_object_points)
        self._visible = bool(visible)
        self._window_name = str(window_name)
        self._window_position = window_position
        self._o3d: Any | None = None
        self._vis: Any | None = None
        self._object_pcd: Any | None = None
        self._controller_meshes: list[Any] = []
        self._controller_centers: list[np.ndarray] = []
        self._object_colors: np.ndarray | None = None
        self._object_color_count = -1
        self._initialized = False

    def _require_open3d(self) -> Any:
        """Return validated open3d."""
        if self._o3d is None:
            import open3d as o3d

            self._o3d = o3d
        return self._o3d

    def _ensure_window(self) -> None:
        """Return the ensure window."""
        if self._vis is not None:
            return
        o3d = self._require_open3d()
        self._vis = o3d.visualization.Visualizer()
        width, height = self._image_size
        left = 50
        top = 50
        if self._window_position is not None:
            left, top = (int(self._window_position[0]), int(self._window_position[1]))
        self._vis.create_window(
            window_name=self._window_name,
            width=width,
            height=height,
            left=left,
            top=top,
            visible=self._visible,
        )
        self._object_pcd = o3d.geometry.PointCloud()

    def _object_visibility(self, chunk: Mapping[str, Any], local_frame: int, point_count: int) -> np.ndarray:
        """Return the object visibility."""
        if self._show_invisible_object_points:
            return np.ones((point_count,), dtype=bool)
        value = chunk.get("object_visibilities")
        if value is None:
            return np.ones((point_count,), dtype=bool)
        arr = np.asarray(value, dtype=bool)
        if arr.ndim == 2 and int(local_frame) < int(arr.shape[0]) and arr.shape[1] == point_count:
            return np.ascontiguousarray(arr[int(local_frame)], dtype=bool)
        return np.ones((point_count,), dtype=bool)

    def _update_object_colors(self, object_points: np.ndarray) -> np.ndarray:
        """Update object colors."""
        point_count = int(object_points.shape[1])
        if self._object_colors is None or self._object_color_count != point_count:
            self._object_colors = _sam3d_rainbow_colors_rgb_float(object_points, point_count)
            self._object_color_count = point_count
        return self._object_colors

    def _reset_controller_meshes(self, controller_points: np.ndarray) -> None:
        # Controller spheres are cached and translated in place per frame;
        # rebuild them only when the controller point count changes.
        """Reset controller meshes."""
        assert self._vis is not None
        o3d = self._require_open3d()
        for mesh in self._controller_meshes:
            self._vis.remove_geometry(mesh, reset_bounding_box=False)
        self._controller_meshes = []
        self._controller_centers = []
        for origin in np.asarray(controller_points, dtype=np.float64).reshape(-1, 3):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01).translate(origin)
            sphere.paint_uniform_color([1.0, 0.0, 0.0])
            self._controller_meshes.append(sphere)
            self._controller_centers.append(np.asarray(origin, dtype=np.float64))
            self._vis.add_geometry(sphere, reset_bounding_box=False)

    def _set_initial_view(self) -> None:
        """Set initial view."""
        assert self._vis is not None
        view_control = self._vis.get_view_control()
        view_control.set_front([1, 0, -2])
        view_control.set_up([0, 0, -1])
        view_control.set_zoom(1)

    def poll(self) -> bool:
        """Process Open3D visualizer events and report whether it is alive."""
        self._ensure_window()
        assert self._vis is not None
        alive = self._vis.poll_events()
        self._vis.update_renderer()
        return bool(alive) if alive is not None else True

    def update_frame(self, chunk: Mapping[str, Any], *, local_frame: int, case_dir: Path) -> bool:
        """Update the Open3D scene for one final_data frame."""
        del case_dir
        self._ensure_window()
        assert self._vis is not None
        assert self._object_pcd is not None
        o3d = self._require_open3d()

        object_arr = np.asarray(chunk.get("object_points"), dtype=np.float64)
        controller_arr = np.asarray(chunk.get("controller_points"), dtype=np.float64)
        if object_arr.ndim != 3 or controller_arr.ndim != 3:
            return self.poll()
        if int(local_frame) >= int(object_arr.shape[0]) or int(local_frame) >= int(controller_arr.shape[0]):
            return self.poll()

        object_frame = np.asarray(object_arr[int(local_frame)], dtype=np.float64).reshape(-1, 3)
        object_colors = self._update_object_colors(object_arr)
        visible = self._object_visibility(chunk, int(local_frame), int(object_frame.shape[0]))
        object_valid = visible & np.all(np.isfinite(object_frame), axis=1)
        controller_frame = np.asarray(controller_arr[int(local_frame)], dtype=np.float64).reshape(-1, 3)
        controller_valid = np.all(np.isfinite(controller_frame), axis=1)
        controller_points = controller_frame[controller_valid]

        if not self._initialized:
            self._object_pcd.points = o3d.utility.Vector3dVector(object_frame[object_valid])
            self._object_pcd.colors = o3d.utility.Vector3dVector(object_colors[object_valid])
            self._vis.add_geometry(self._object_pcd)
            self._reset_controller_meshes(controller_points)
            self._set_initial_view()
            self._initialized = True
        else:
            self._object_pcd.points = o3d.utility.Vector3dVector(object_frame[object_valid])
            self._object_pcd.colors = o3d.utility.Vector3dVector(object_colors[object_valid])
            self._vis.update_geometry(self._object_pcd)
            if len(controller_points) != len(self._controller_meshes):
                self._reset_controller_meshes(controller_points)
            for index, sphere in enumerate(self._controller_meshes):
                origin = np.asarray(controller_points[index], dtype=np.float64)
                sphere.translate(origin - self._controller_centers[index])
                self._controller_centers[index] = origin
                self._vis.update_geometry(sphere)

        alive = self._vis.poll_events()
        self._vis.update_renderer()
        return bool(alive) if alive is not None else True

    def render_frame(self, chunk: Mapping[str, Any], *, local_frame: int, case_dir: Path) -> np.ndarray:
        """Render one final_data frame to a BGR image."""
        if not self.update_frame(chunk, local_frame=local_frame, case_dir=case_dir):
            return _blank_image(self._image_size)
        assert self._vis is not None
        frame = np.asarray(self._vis.capture_screen_float_buffer(do_render=True))
        frame = np.clip(frame * 255.0, 0.0, 255.0).astype(np.uint8)
        cv2 = _require_cv2()
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def close(self) -> None:
        """Destroy the Open3D visualizer window when it exists."""
        if self._vis is not None:
            self._vis.destroy_window()
            self._vis = None


class Sam3DGuiFinalDataRenderer:
    """Interactive Open3D GUI renderer with a 2D latency HUD in the 3D window."""

    def __init__(
        self,
        *,
        image_size: tuple[int, int],
        show_invisible_object_points: bool,
        window_name: str = "final_data output",
        window_position: tuple[int, int] | None = None,
        show_latency_overlay: bool = True,
        object_point_size: float = 5.0,
        controller_point_size: float = 18.0,
    ) -> None:
        """Initialize Sam3DGuiFinalDataRenderer."""
        self._image_size = (int(image_size[0]), int(image_size[1]))
        self._show_invisible_object_points = bool(show_invisible_object_points)
        self._window_name = str(window_name)
        self._window_position = window_position
        self._show_latency_overlay = bool(show_latency_overlay)
        self._object_point_size = float(object_point_size)
        self._controller_point_size = float(controller_point_size)
        self._o3d: Any | None = None
        self._gui: Any | None = None
        self._rendering: Any | None = None
        self._window: Any | None = None
        self._scene_widget: Any | None = None
        self._title_label: Any | None = None
        self._latency_label: Any | None = None
        self._object_material: Any | None = None
        self._controller_material: Any | None = None
        self._object_colors: np.ndarray | None = None
        self._object_color_count = -1
        self._camera_initialized = False
        self._closed = False

    def _require_open3d_gui(self) -> tuple[Any, Any, Any]:
        """Return validated open3d gui."""
        if self._o3d is None or self._gui is None or self._rendering is None:
            import open3d as o3d
            from open3d.visualization import gui, rendering

            self._o3d = o3d
            self._gui = gui
            self._rendering = rendering
        return self._o3d, self._gui, self._rendering

    def _ensure_window(self) -> None:
        """Return the ensure window."""
        if self._window is not None:
            return
        _o3d, gui, rendering = self._require_open3d_gui()
        gui.Application.instance.initialize()
        width, height = self._image_size
        left = 50
        top = 50
        if self._window_position is not None:
            left, top = (int(self._window_position[0]), int(self._window_position[1]))
        self._window = gui.Application.instance.create_window(
            self._window_name,
            int(width),
            int(height),
            int(left),
            int(top),
        )
        self._window.set_on_close(self._on_close)
        self._scene_widget = gui.SceneWidget()
        self._scene_widget.scene = rendering.Open3DScene(self._window.renderer)
        self._scene_widget.scene.set_background([0.0, 0.0, 0.0, 1.0])
        self._scene_widget.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)

        self._title_label = gui.Label("final_data output")
        self._title_label.text_color = gui.Color(1.0, 1.0, 1.0)
        self._title_label.background_color = gui.Color(0.0, 0.0, 0.0, 0.45)
        self._latency_label = gui.Label(format_input_display_latency(None))
        self._latency_label.text_color = gui.Color(1.0, 1.0, 1.0)
        self._latency_label.background_color = gui.Color(0.0, 0.0, 0.0, 0.45)
        self._latency_label.visible = self._show_latency_overlay

        self._window.add_child(self._scene_widget)
        self._window.add_child(self._title_label)
        self._window.add_child(self._latency_label)
        self._window.set_on_layout(self._layout)

        self._object_material = rendering.MaterialRecord()
        self._object_material.shader = "defaultUnlit"
        self._object_material.point_size = max(1.0, self._object_point_size)
        self._controller_material = rendering.MaterialRecord()
        self._controller_material.shader = "defaultUnlit"
        self._controller_material.point_size = max(1.0, self._controller_point_size)

    def _layout(self, _layout_context: Any) -> None:
        """Return the layout."""
        if self._window is None or self._scene_widget is None:
            return
        gui = self._gui
        assert gui is not None
        rect = self._window.content_rect
        self._scene_widget.frame = rect
        overlay_width = min(300, max(140, int(rect.width) - 24))
        title_height = 26
        latency_height = 26
        x = int(rect.x + rect.width - overlay_width - 12)
        y = int(rect.y + 10)
        if self._title_label is not None:
            self._title_label.frame = gui.Rect(x, y, overlay_width, title_height)
        if self._latency_label is not None:
            self._latency_label.frame = gui.Rect(x, y + title_height, overlay_width, latency_height)

    def _on_close(self) -> bool:
        """Return the on close."""
        self._closed = True
        return True

    def _object_visibility(self, chunk: Mapping[str, Any], local_frame: int, point_count: int) -> np.ndarray:
        """Return the object visibility."""
        if self._show_invisible_object_points:
            return np.ones((point_count,), dtype=bool)
        value = chunk.get("object_visibilities")
        if value is None:
            return np.ones((point_count,), dtype=bool)
        arr = np.asarray(value, dtype=bool)
        if arr.ndim == 2 and int(local_frame) < int(arr.shape[0]) and arr.shape[1] == point_count:
            return np.ascontiguousarray(arr[int(local_frame)], dtype=bool)
        return np.ones((point_count,), dtype=bool)

    def _update_object_colors(self, object_points: np.ndarray) -> np.ndarray:
        """Update object colors."""
        point_count = int(object_points.shape[1])
        if self._object_colors is None or self._object_color_count != point_count:
            self._object_colors = _sam3d_rainbow_colors_rgb_float(object_points, point_count)
            self._object_color_count = point_count
        return self._object_colors

    def _set_latency_label(self, latency_s: float | None) -> None:
        """Set latency label."""
        if self._latency_label is None:
            return
        self._latency_label.text = format_input_display_latency(latency_s)
        self._latency_label.visible = self._show_latency_overlay

    def _remove_geometry_if_present(self, name: str) -> None:
        """Return the remove geometry if present."""
        assert self._scene_widget is not None
        scene = self._scene_widget.scene
        try:
            if scene.has_geometry(name):
                scene.remove_geometry(name)
        except Exception:
            scene.remove_geometry(name)

    def _initialize_camera(self, points: np.ndarray) -> None:
        """Initialize camera."""
        if self._camera_initialized or points.size == 0:
            return
        assert self._scene_widget is not None
        o3d = self._o3d
        assert o3d is not None
        finite_points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        finite_points = finite_points[np.all(np.isfinite(finite_points), axis=1)]
        if finite_points.size == 0:
            return
        bounds = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
            o3d.utility.Vector3dVector(finite_points)
        )
        center = np.asarray(bounds.get_center(), dtype=np.float32)
        extent = float(np.linalg.norm(np.asarray(bounds.get_extent(), dtype=np.float64)))
        if not math.isfinite(extent) or extent <= 1e-6:
            extent = 1.0
        self._scene_widget.setup_camera(60.0, bounds, center)
        eye = center + np.asarray([0.0, -1.2 * extent, 0.8 * extent], dtype=np.float32)
        up = np.asarray([0.0, 0.0, -1.0], dtype=np.float32)
        self._scene_widget.look_at(center, eye, up)
        self._camera_initialized = True

    def poll(self) -> bool:
        """Process Open3D GUI events and report whether the window is open."""
        self._ensure_window()
        if self._closed:
            return False
        assert self._gui is not None
        if self._window is not None:
            self._window.post_redraw()
        alive = self._gui.Application.instance.run_one_tick()
        return bool(alive) and not self._closed

    def update_frame(
        self,
        chunk: Mapping[str, Any],
        *,
        local_frame: int,
        case_dir: Path,
        input_to_display_latency_s: float | None = None,
    ) -> bool:
        """Update the GUI scene and latency HUD for one final_data frame."""
        del case_dir
        self._ensure_window()
        if self._closed:
            return False
        assert self._scene_widget is not None
        assert self._object_material is not None
        assert self._controller_material is not None
        o3d = self._o3d
        assert o3d is not None
        self._set_latency_label(input_to_display_latency_s)

        object_arr = np.asarray(chunk.get("object_points"), dtype=np.float64)
        controller_arr = np.asarray(chunk.get("controller_points"), dtype=np.float64)
        if object_arr.ndim != 3 or controller_arr.ndim != 3:
            return self.poll()
        if int(local_frame) >= int(object_arr.shape[0]) or int(local_frame) >= int(controller_arr.shape[0]):
            return self.poll()

        object_frame = np.asarray(object_arr[int(local_frame)], dtype=np.float64).reshape(-1, 3)
        object_colors = self._update_object_colors(object_arr)
        visible = self._object_visibility(chunk, int(local_frame), int(object_frame.shape[0]))
        object_valid = visible & np.all(np.isfinite(object_frame), axis=1)
        controller_frame = np.asarray(controller_arr[int(local_frame)], dtype=np.float64).reshape(-1, 3)
        controller_valid = np.all(np.isfinite(controller_frame), axis=1)
        controller_points = controller_frame[controller_valid]

        object_pcd = o3d.geometry.PointCloud()
        object_pcd.points = o3d.utility.Vector3dVector(object_frame[object_valid])
        object_pcd.colors = o3d.utility.Vector3dVector(object_colors[object_valid])
        controller_pcd = o3d.geometry.PointCloud()
        controller_pcd.points = o3d.utility.Vector3dVector(controller_points)
        controller_color = np.tile(np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64), (len(controller_points), 1))
        controller_pcd.colors = o3d.utility.Vector3dVector(controller_color)

        self._remove_geometry_if_present("object_points")
        self._remove_geometry_if_present("controller_points")
        self._scene_widget.scene.add_geometry("object_points", object_pcd, self._object_material)
        self._scene_widget.scene.add_geometry("controller_points", controller_pcd, self._controller_material)

        all_points = np.concatenate([object_frame[object_valid], controller_points], axis=0)
        self._initialize_camera(all_points)
        return self.poll()

    def close(self) -> None:
        """Close the Open3D GUI window and mark the renderer closed."""
        if self._window is not None:
            self._window.close()
            self._window = None
        self._closed = True


def build_frame_renderer(args: argparse.Namespace, *, camera: CameraModel, fps: float) -> Any:
    """Create the frame renderer selected by CLI arguments."""
    render_mode = str(args.render_mode)
    if render_mode == RENDER_MODE_RGB_OVERLAY:
        return RgbOverlayRenderer(camera=camera, args=args, fps=fps)
    if render_mode == RENDER_MODE_SAM3D_FINAL_DATA:
        return Sam3DFinalDataRenderer(
            image_size=camera.image_size,
            show_invisible_object_points=bool(args.show_invisible_object_points),
        )
    raise ValueError(f"unsupported render mode: {render_mode!r}")


# --- Interactive playback loops ----------------------------------------------


def _window_is_open(window_name: str) -> bool:
    """Return the window is open."""
    cv2 = _require_cv2()
    try:
        return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1
    except Exception:
        return True


def _key_requests_quit(key: int) -> bool:
    """Return the key requests quit."""
    return key in (27, ord("q"), ord("Q"))


def _wait_with_pause(window_name: str, *, delay_s: float) -> bool:
    """Wait one playback period while servicing space-to-pause and quit keys.

    Returns False when the viewer should stop (quit key or closed window).
    Unpausing resets the deadline so playback resumes immediately.
    """
    cv2 = _require_cv2()
    deadline = time.monotonic() + max(float(delay_s), 0.0)
    paused = False
    while True:
        if not _window_is_open(window_name):
            return False
        wait_s = 0.05 if paused else min(0.05, max(0.0, deadline - time.monotonic()))
        key = cv2.waitKey(max(1, int(wait_s * 1000))) & 0xFF
        if _key_requests_quit(key):
            return False
        if key == ord(" "):
            paused = not paused
            if not paused:
                deadline = time.monotonic()
        if paused:
            continue
        if time.monotonic() >= deadline:
            return True


def play_chunk(
    chunk: Mapping[str, Any],
    *,
    case_dir: Path,
    renderer: Any,
    args: argparse.Namespace,
    fps: float,
) -> np.ndarray | None:
    """Play one chunk frame-by-frame in an OpenCV window."""
    cv2 = _require_cv2()
    period_s = 1.0 / max(float(fps), 1e-6)
    frame_count = _chunk_frame_count(chunk)
    last_image = None
    for local_frame in range(frame_count):
        image = renderer.render_frame(
            chunk,
            local_frame=local_frame,
            case_dir=case_dir,
        )
        cv2.imshow(str(args.window_name), image)
        last_image = image
        if not _wait_with_pause(str(args.window_name), delay_s=period_s):
            return None
    return last_image


def use_interactive_side_by_side(args: argparse.Namespace) -> bool:
    """Return whether side-by-side mode should use the Open3D GUI renderer."""
    return (
        str(getattr(args, "layout", LAYOUT_OUTPUT_ONLY)) == LAYOUT_SIDE_BY_SIDE
        and str(getattr(args, "render_mode", RENDER_MODE_RGB_OVERLAY)) == RENDER_MODE_SAM3D_FINAL_DATA
        and getattr(args, "output_video", None) is None
    )


def run_interactive_side_by_side(args: argparse.Namespace) -> int:
    """Run the Open3D output window next to a live OpenCV RGB input window."""
    cv2 = _require_cv2()
    online_dir = normalize_online_dir(args.online_dir)
    case_dir = infer_case_dir(online_dir, args.case_dir)
    camera = load_camera_model(case_dir, cam_idx=int(args.cam_idx))
    fps = resolve_playback_fps(args, camera)
    capture_dir = _resolve_capture_dir(args)
    input_timeline = _resolve_input_rgb_timeline(args, capture_dir=capture_dir)
    fake_input_frame_total = load_fake_input_frame_total(capture_dir)
    width, height = camera.image_size
    left_window_name = f"{args.window_name} - RGB input"
    right_window_name = f"{args.window_name} - final_data output"
    output_renderer = Sam3DGuiFinalDataRenderer(
        image_size=camera.image_size,
        show_invisible_object_points=bool(args.show_invisible_object_points),
        window_name=right_window_name,
        window_position=(int(width) + 80, 50),
        show_latency_overlay=bool(args.latency_overlay),
    )
    output_frames: list[tuple[dict[str, Any], int]] = []
    loaded_paths: set[Path] = set()
    cursor = OutputStreamPlaybackCursor(fps=fps)
    final_data_fps_meter = CameraToFinalDataFpsMeter()
    paused = False

    cv2.namedWindow(left_window_name, cv2.WINDOW_NORMAL)
    try:
        cv2.resizeWindow(left_window_name, int(width), int(height))
        cv2.moveWindow(left_window_name, 30, 50)
    except Exception:
        pass

    try:
        while True:
            appended = _append_new_output_frames(
                online_dir,
                start_chunk=int(args.start_chunk),
                loaded_paths=loaded_paths,
                output_frames=output_frames,
            )
            latest = max(0, len(output_frames) - 1)
            now_s = time.monotonic()
            camera_to_final_data_fps = final_data_fps_meter.update(
                appended_frames=appended,
                now_s=now_s,
            )
            if camera_to_final_data_fps is None:
                camera_to_final_data_fps = final_data_fps_meter.seed(
                    estimate_historical_camera_to_final_data_fps(
                        online_dir,
                        start_chunk=int(args.start_chunk),
                    )
                )

            input_frame = None
            if capture_dir is not None and input_timeline is not None:
                input_frame = load_latest_input_rgb_frame(input_timeline, capture_dir=capture_dir)
            if output_frames and not paused:
                # The left panel follows the latest camera RGB. The right panel
                # plays only committed chunk frames at the configured 5 FPS.
                cursor.advance(latest=latest, now_s=now_s, paused=False)
            else:
                cursor.advance(latest=latest, now_s=now_s, paused=True)
            # Compose the left panel with the same overlays as the left half
            # of render_side_by_side_frame.
            input_panel = _panel_image(
                None if input_frame is None else input_frame.image_bgr,
                image_size=camera.image_size,
            )
            if input_frame is None:
                _draw_center_label(input_panel, "waiting for RGB input")
            _draw_panel_label(input_panel, "RGB input")
            _draw_fake_rgb_frame_counter_overlay(
                input_panel,
                input_frame,
                fake_input_frame_total=fake_input_frame_total,
            )
            _draw_camera_to_final_data_fps_overlay(input_panel, camera_to_final_data_fps)
            cv2.imshow(left_window_name, input_panel)

            if output_frames:
                chunk, local_frame = output_frames[int(cursor.output_index)]
                latency_s = None
                if bool(args.latency_overlay):
                    latency_s = source_time_input_display_latency_s(
                        input_frame=input_frame,
                        output_frames=output_frames,
                        output_index=int(cursor.output_index),
                        fps=fps,
                    )
                if not output_renderer.update_frame(
                    chunk,
                    local_frame=local_frame,
                    case_dir=case_dir,
                    input_to_display_latency_s=latency_s,
                ):
                    return 0
            else:
                if not output_renderer.poll():
                    return 0

            key = cv2.waitKey(max(1, int(float(args.poll_sec) * 1000))) & 0xFF
            if _key_requests_quit(key) or not _window_is_open(left_window_name):
                return 0
            if key == ord(" "):
                paused = not paused
                cursor.last_step_s = time.monotonic()
            elif key in (ord("f"), ord("F")):
                paused = False
                cursor.seek(latest, latest=latest, now_s=time.monotonic())
    finally:
        output_renderer.close()
        try:
            cv2.destroyWindow(left_window_name)
        except Exception:
            pass


def run_side_by_side(args: argparse.Namespace) -> int:
    """Run the single-window side-by-side viewer/video fallback."""
    cv2 = _require_cv2()
    online_dir = normalize_online_dir(args.online_dir)
    case_dir = infer_case_dir(online_dir, args.case_dir)
    camera = load_camera_model(case_dir, cam_idx=int(args.cam_idx))
    fps = resolve_playback_fps(args, camera)
    renderer = build_frame_renderer(args, camera=camera, fps=fps)
    capture_dir = _resolve_capture_dir(args)
    input_timeline = _resolve_input_rgb_timeline(args, capture_dir=capture_dir)
    fake_input_frame_total = load_fake_input_frame_total(capture_dir)

    window_name = str(args.window_name)
    trackbar_name = "output frame"
    output_frames: list[tuple[dict[str, Any], int]] = []
    loaded_paths: set[Path] = set()
    cursor = OutputStreamPlaybackCursor(fps=fps)
    final_data_fps_meter = CameraToFinalDataFpsMeter()
    follow_latest = bool(args.follow_latest)
    paused = False
    # Guard against feedback: syncing the trackbar position programmatically
    # fires on_trackbar, which must not be treated as a user seek.
    trackbar_guard = {"updating": False}

    def on_trackbar(value: int) -> None:
        """Return the on trackbar."""
        nonlocal follow_latest
        if trackbar_guard["updating"]:
            return
        latest = max(0, len(output_frames) - 1)
        cursor.seek(value, latest=latest, now_s=time.monotonic())
        follow_latest = int(cursor.output_index) >= latest

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.createTrackbar(trackbar_name, window_name, 0, 1, on_trackbar)
    try:
        while True:
            appended = _append_new_output_frames(
                online_dir,
                start_chunk=int(args.start_chunk),
                loaded_paths=loaded_paths,
                output_frames=output_frames,
            )
            latest = max(0, len(output_frames) - 1)
            # OpenCV rejects a trackbar max of 0, and some GUI backends do not
            # implement setTrackbarMax at all.
            try:
                cv2.setTrackbarMax(trackbar_name, window_name, max(1, latest))
            except Exception:
                pass
            now_s = time.monotonic()
            camera_to_final_data_fps = final_data_fps_meter.update(
                appended_frames=appended,
                now_s=now_s,
            )
            if camera_to_final_data_fps is None:
                camera_to_final_data_fps = final_data_fps_meter.seed(
                    estimate_historical_camera_to_final_data_fps(
                        online_dir,
                        start_chunk=int(args.start_chunk),
                    )
                )

            input_frame = None
            if capture_dir is not None and input_timeline is not None:
                input_frame = load_latest_input_rgb_frame(input_timeline, capture_dir=capture_dir)
            if output_frames and follow_latest and not paused:
                # The left panel follows the latest camera RGB. The right panel
                # plays only committed chunk frames at the configured 5 FPS.
                cursor.advance(latest=latest, now_s=now_s, paused=False)
            else:
                cursor.seek(cursor.output_index, latest=latest)
            output_frame = _render_output_timeline_frame(
                output_frames,
                output_index=int(cursor.output_index),
                renderer=renderer,
                case_dir=case_dir,
            )
            input_to_display_latency_s = None
            if bool(args.latency_overlay):
                input_to_display_latency_s = source_time_input_display_latency_s(
                    input_frame=input_frame,
                    output_frames=output_frames,
                    output_index=int(cursor.output_index),
                    fps=fps,
                )
            image = render_side_by_side_frame(
                input_frame=input_frame,
                output_frame=output_frame,
                image_size=camera.image_size,
                right_blank_label=str(args.right_blank_label),
                camera_to_final_data_fps=camera_to_final_data_fps,
                fake_input_frame_total=fake_input_frame_total,
                input_to_display_latency_s=input_to_display_latency_s,
                show_latency_overlay=bool(args.latency_overlay),
            )
            cv2.imshow(window_name, image)
            trackbar_guard["updating"] = True
            try:
                cv2.setTrackbarPos(trackbar_name, window_name, int(cursor.output_index))
            finally:
                trackbar_guard["updating"] = False
            key = cv2.waitKey(max(1, int(float(args.poll_sec) * 1000))) & 0xFF
            if _key_requests_quit(key) or not _window_is_open(window_name):
                return 0
            if key == ord(" "):
                paused = not paused
                cursor.last_step_s = time.monotonic()
            elif key in (ord("f"), ord("F")):
                follow_latest = True
                paused = False
                cursor.seek(latest, latest=latest, now_s=time.monotonic())
    finally:
        renderer.close()


def wait_for_chunk(
    online_dir: Path,
    *,
    chunk_id: int,
    poll_sec: float,
    window_name: str,
    last_image: np.ndarray | None,
) -> dict[str, Any] | None:
    """Block until the next online chunk appears or the stream finishes."""
    cv2 = _require_cv2()
    chunks_dir = online_dir / "chunks"
    chunk_path = chunks_dir / f"chunk_{int(chunk_id):06d}.pkl"
    while True:
        if chunk_path.is_file():
            return dict(load_pickle(chunk_path))
        manifest = read_json(online_dir / "manifest.json")
        latest = int(manifest.get("latest_committed_chunk", -1))
        status = str(manifest.get("status", "recording"))
        if status == "finished" and latest < int(chunk_id):
            return None
        if last_image is not None:
            waiting = last_image.copy()
            _draw_status(waiting, f"waiting for chunk {int(chunk_id):06d}")
            cv2.imshow(window_name, waiting)
        key = cv2.waitKey(max(1, int(float(poll_sec) * 1000))) & 0xFF
        if _key_requests_quit(key) or not _window_is_open(window_name):
            return None


def resolve_playback_fps(args: argparse.Namespace, camera: CameraModel) -> float:
    """Resolve playback FPS from CLI, metadata, or the default fallback."""
    fps = None if args.fps is None else float(args.fps)
    if fps is None:
        fps = camera.metadata_fps
    if fps is None:
        fps = 5.0
    if not math.isfinite(float(fps)) or float(fps) <= 0.0:
        raise ValueError("--fps must be positive")
    return float(fps)


# --- Online chunk discovery and throughput estimation ------------------------


def _chunk_sort_key(path: Path) -> tuple[int, str]:
    # Chunk files are named chunk_<id>.pkl; sort numerically by id.
    """Return the chunk sort key."""
    stem = path.stem
    try:
        return (int(stem.rsplit("_", 1)[1]), path.name)
    except (IndexError, ValueError):
        return (0, path.name)


def list_available_chunk_paths(online_dir: Path, *, start_chunk: int) -> list[Path]:
    """List committed chunk pickle files at or after the requested chunk id."""
    chunks_dir = normalize_online_dir(online_dir) / "chunks"
    paths = sorted(chunks_dir.glob("chunk_*.pkl"), key=_chunk_sort_key)
    start = int(start_chunk)
    return [path for path in paths if _chunk_sort_key(path)[0] >= start]


def _run_root_for_online_dir(online_dir: Path) -> Path | None:
    """Return the run root that owns an online data directory."""
    path = normalize_online_dir(online_dir)
    if path.name != "online_data":
        return None
    return path.parent


def _camera_to_final_data_fps_from_run_manifest(online_dir: Path) -> float | None:
    """Recover publish FPS from the run-level summary when available."""
    run_root = _run_root_for_online_dir(online_dir)
    if run_root is None or not run_root.is_dir():
        return None
    candidates = [run_root / "run_summary.json"]
    for manifest_path in candidates:
        if not manifest_path.is_file():
            continue
        manifest = read_json(manifest_path)
        online_dir_value = str(manifest.get("online_dir", ""))
        if online_dir_value and Path(online_dir_value).name != "online_data":
            continue
        intervals = []
        for value in manifest.get("steady_publish_intervals_s", []) or []:
            try:
                interval = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(interval) and interval > 1e-9:
                intervals.append(interval)
        try:
            chunk_frame_count = int(manifest.get("chunk_frame_count", 0))
        except (TypeError, ValueError):
            chunk_frame_count = 0
        if intervals and chunk_frame_count > 0:
            return float(len(intervals) * chunk_frame_count) / float(sum(intervals))
    return None


def _camera_to_final_data_fps_from_chunk_mtimes(online_dir: Path, *, start_chunk: int) -> float | None:
    """Fallback throughput estimate based on chunk file mtimes.

    The first chunk's frames accumulated before its commit time, so only
    frames committed after the first mtime are divided by the mtime span.
    """
    chunk_infos: list[tuple[float, int]] = []
    for chunk_path in list_available_chunk_paths(online_dir, start_chunk=start_chunk):
        try:
            mtime_s = float(chunk_path.stat().st_mtime)
        except OSError:
            continue
        try:
            chunk = dict(load_pickle(chunk_path))
        except Exception:
            continue
        frame_count = _chunk_frame_count(chunk)
        if frame_count > 0 and math.isfinite(mtime_s):
            chunk_infos.append((mtime_s, int(frame_count)))
    if len(chunk_infos) < 2:
        return None
    chunk_infos.sort(key=lambda item: item[0])
    elapsed_s = float(chunk_infos[-1][0] - chunk_infos[0][0])
    if elapsed_s <= 1e-9:
        return None
    frames_after_first_commit = sum(frame_count for _mtime_s, frame_count in chunk_infos[1:])
    if frames_after_first_commit <= 0:
        return None
    return float(frames_after_first_commit) / elapsed_s


def estimate_historical_camera_to_final_data_fps(online_dir: Path, *, start_chunk: int) -> float | None:
    """Estimate camera->final_data throughput.

    This is used when reopening already committed chunks.
    """
    manifest_fps = _camera_to_final_data_fps_from_run_manifest(online_dir)
    if manifest_fps is not None:
        return manifest_fps
    return _camera_to_final_data_fps_from_chunk_mtimes(online_dir, start_chunk=start_chunk)


def _append_new_output_frames(
    online_dir: Path,
    *,
    start_chunk: int,
    loaded_paths: set[Path],
    output_frames: list[tuple[dict[str, Any], int]],
) -> int:
    """Load each new chunk once and flatten it into frame-level playback rows."""
    appended = 0
    for chunk_path in list_available_chunk_paths(online_dir, start_chunk=start_chunk):
        resolved = chunk_path.resolve()
        if resolved in loaded_paths:
            continue
        try:
            chunk = dict(load_pickle(chunk_path))
        except Exception:
            # Leave the path unmarked so a partially written chunk is retried
            # on the next poll.
            continue
        loaded_paths.add(resolved)
        frame_count = _chunk_frame_count(chunk)
        for local_frame in range(frame_count):
            output_frames.append((chunk, int(local_frame)))
            appended += 1
    return appended


def _render_output_timeline_frame(
    output_frames: Sequence[tuple[dict[str, Any], int]],
    *,
    output_index: int,
    renderer: Any,
    case_dir: Path,
) -> np.ndarray | None:
    """Render output timeline frame."""
    if not output_frames:
        return None
    idx = min(max(int(output_index), 0), len(output_frames) - 1)
    chunk, local_frame = output_frames[idx]
    return renderer.render_frame(chunk, local_frame=local_frame, case_dir=case_dir)


def _resolve_capture_dir(args: argparse.Namespace) -> Path | None:
    """Resolve the capture directory for RGB timeline lookup."""
    value = getattr(args, "capture_dir", None)
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(value).expanduser()


def _resolve_input_rgb_timeline(args: argparse.Namespace, *, capture_dir: Path | None) -> Path | None:
    """Resolve input RGB timeline."""
    value = getattr(args, "input_rgb_timeline", None)
    if value is not None and str(value).strip():
        return Path(value).expanduser()
    if capture_dir is None:
        return None
    return capture_dir / "input_frames.jsonl"


# --- Offline video export ----------------------------------------------------


def render_side_by_side_output_video(args: argparse.Namespace) -> int:
    """Render existing side-by-side frames to an MP4 file."""
    cv2 = _require_cv2()
    online_dir = normalize_online_dir(args.online_dir)
    case_dir = infer_case_dir(online_dir, args.case_dir)
    camera = load_camera_model(case_dir, cam_idx=int(args.cam_idx))
    fps = resolve_playback_fps(args, camera)
    renderer = build_frame_renderer(args, camera=camera, fps=fps)
    capture_dir = _resolve_capture_dir(args)
    input_timeline = _resolve_input_rgb_timeline(args, capture_dir=capture_dir)
    fake_input_frame_total = load_fake_input_frame_total(capture_dir)
    input_frames = (
        []
        if capture_dir is None or input_timeline is None
        else load_input_rgb_frames(input_timeline, capture_dir=capture_dir)
    )
    loaded_paths: set[Path] = set()
    output_frames: list[tuple[dict[str, Any], int]] = []
    _append_new_output_frames(
        online_dir,
        start_chunk=int(args.start_chunk),
        loaded_paths=loaded_paths,
        output_frames=output_frames,
    )
    total_frames = max(len(input_frames), len(output_frames), 1)
    output_path = Path(args.output_video).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    try:
        for index in range(total_frames):
            input_frame = input_frames[min(index, len(input_frames) - 1)] if input_frames else None
            output_frame = None
            if index < len(output_frames):
                output_frame = _render_output_timeline_frame(
                    output_frames,
                    output_index=index,
                    renderer=renderer,
                    case_dir=case_dir,
                )
            image = render_side_by_side_frame(
                input_frame=input_frame,
                output_frame=output_frame,
                image_size=camera.image_size,
                right_blank_label=str(args.right_blank_label),
                fake_input_frame_total=fake_input_frame_total,
                show_latency_overlay=False,
            )
            if writer is None:
                height, width = image.shape[:2]
                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (int(width), int(height)),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"failed to open VideoWriter for {output_path}")
            writer.write(image)
    finally:
        if writer is not None:
            writer.release()
        renderer.close()
    return 0


def render_output_video(args: argparse.Namespace) -> int:
    """Render existing output chunks to an MP4 file."""
    cv2 = _require_cv2()
    if str(getattr(args, "layout", LAYOUT_OUTPUT_ONLY)) == LAYOUT_SIDE_BY_SIDE:
        return render_side_by_side_output_video(args)
    online_dir = normalize_online_dir(args.online_dir)
    case_dir = infer_case_dir(online_dir, args.case_dir)
    camera = load_camera_model(case_dir, cam_idx=int(args.cam_idx))
    fps = resolve_playback_fps(args, camera)
    chunk_paths = list_available_chunk_paths(online_dir, start_chunk=int(args.start_chunk))
    if not chunk_paths:
        raise ValueError(f"no chunk_*.pkl files found under {online_dir / 'chunks'}")
    output_path = Path(args.output_video).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    renderer = build_frame_renderer(args, camera=camera, fps=fps)
    writer = None
    try:
        for chunk_path in chunk_paths:
            chunk = dict(load_pickle(chunk_path))
            frame_count = _chunk_frame_count(chunk)
            for local_frame in range(frame_count):
                image = renderer.render_frame(chunk, local_frame=local_frame, case_dir=case_dir)
                if writer is None:
                    height, width = image.shape[:2]
                    writer = cv2.VideoWriter(
                        str(output_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (int(width), int(height)),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"failed to open VideoWriter for {output_path}")
                writer.write(image)
    finally:
        if writer is not None:
            writer.release()
        renderer.close()
    return 0


# --- CLI entry points --------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the Demo v6.1 chunk viewer CLI parser."""
    parser = argparse.ArgumentParser(
        description="Play Demo v6.1 online object/controller points chunk by chunk."
    )
    parser.add_argument("--layout", choices=LAYOUTS, default=LAYOUT_OUTPUT_ONLY)
    parser.add_argument(
        "--online-dir",
        type=Path,
        required=True,
        help="Path to online_data or its chunks directory.",
    )
    parser.add_argument(
        "--case-dir",
        type=Path,
        default=None,
        help="Path to data. Inferred from --online-dir when omitted.",
    )
    parser.add_argument("--render-mode", choices=RENDER_MODES, default=RENDER_MODE_RGB_OVERLAY)
    parser.add_argument("--output-video", type=Path, default=None, help="Write existing chunks to MP4 and exit instead of opening a live window.")
    parser.add_argument("--capture-dir", type=Path, default=None, help="Headless capture dir containing input_frames.jsonl and input_rgb/*.png.")
    parser.add_argument("--input-rgb-timeline", type=Path, default=None, help="Path to input_frames.jsonl. Defaults to --capture-dir/input_frames.jsonl.")
    parser.add_argument("--right-blank-label", default=DEFAULT_RIGHT_BLANK_LABEL)
    parser.add_argument("--follow-latest", dest="follow_latest", action="store_true", default=True)
    parser.add_argument("--no-follow-latest", dest="follow_latest", action="store_false")
    parser.add_argument("--cam-idx", type=int, default=0)
    parser.add_argument("--fps", type=float, default=None, help="Playback FPS. Defaults to metadata fps, then 5.")
    parser.add_argument("--latency-overlay", dest="latency_overlay", action="store_true", default=True)
    parser.add_argument("--no-latency-overlay", dest="latency_overlay", action="store_false")
    parser.add_argument("--poll-sec", type=float, default=0.1)
    parser.add_argument("--start-chunk", type=int, default=0)
    parser.add_argument("--object-stride", type=int, default=1)
    parser.add_argument("--object-radius", type=int, default=DEFAULT_OBJECT_RADIUS)
    parser.add_argument("--controller-radius", type=int, default=DEFAULT_CONTROLLER_RADIUS)
    parser.add_argument("--object-color-mode", choices=("rainbow", "green", "object-colors"), default="rainbow")
    parser.add_argument("--controller-color", type=parse_bgr_color, default=parse_bgr_color("0,0,255"))
    parser.add_argument("--show-invisible-object-points", action="store_true")
    parser.add_argument("--no-background", action="store_true")
    parser.add_argument("--window-name", default=DEFAULT_WINDOW_NAME)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Validate viewer CLI arguments before opening windows or videos."""
    if str(args.layout) not in LAYOUTS:
        raise ValueError(f"--layout must be one of {', '.join(LAYOUTS)}")
    if int(args.cam_idx) < 0:
        raise ValueError("--cam-idx must be non-negative")
    if float(args.poll_sec) <= 0.0:
        raise ValueError("--poll-sec must be positive")
    if int(args.start_chunk) < 0:
        raise ValueError("--start-chunk must be non-negative")
    if int(args.object_stride) <= 0:
        raise ValueError("--object-stride must be positive")
    if int(args.object_radius) <= 0:
        raise ValueError("--object-radius must be positive")
    if int(args.controller_radius) <= 0:
        raise ValueError("--controller-radius must be positive")


def run(args: argparse.Namespace) -> int:
    """Play committed chunks in order, tailing the online directory live."""
    validate_args(args)
    if args.output_video is not None:
        return render_output_video(args)
    if str(args.layout) == LAYOUT_SIDE_BY_SIDE:
        if use_interactive_side_by_side(args):
            return run_interactive_side_by_side(args)
        return run_side_by_side(args)
    cv2 = _require_cv2()
    online_dir = normalize_online_dir(args.online_dir)
    case_dir = infer_case_dir(online_dir, args.case_dir)
    camera = load_camera_model(case_dir, cam_idx=int(args.cam_idx))
    fps = resolve_playback_fps(args, camera)
    renderer = build_frame_renderer(args, camera=camera, fps=fps)
    cv2.namedWindow(str(args.window_name), cv2.WINDOW_NORMAL)
    chunk_id = int(args.start_chunk)
    last_image: np.ndarray | None = None
    try:
        while True:
            chunk = wait_for_chunk(
                online_dir,
                chunk_id=chunk_id,
                poll_sec=float(args.poll_sec),
                window_name=str(args.window_name),
                last_image=last_image,
            )
            if chunk is None:
                return 0
            last_image = play_chunk(
                chunk,
                case_dir=case_dir,
                renderer=renderer,
                args=args,
                fps=fps,
            )
            if last_image is None:
                return 0
            chunk_id += 1
    finally:
        renderer.close()


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments and run the Demo v6.1 viewer."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
