"""Single-camera shape-prior warmup for Demo v5.1."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import threading
import time
from typing import Any

import numpy as np
from PIL import Image


STATUS_DISABLED = "disabled"
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_READY = "ready"
STATUS_FAILED = "failed"

DEFAULT_SHAPE_PRIOR_TIMEOUT_MS = 180_000
DEFAULT_SHAPE_PRIOR_WARMUP_CUDA_VISIBLE_DEVICES = "1"
CASE_NAME = "shape_prior_frame0"
DEFAULT_SURFACE_POINT_COUNT = 1024
DEFAULT_VOLUME_SAMPLE_SIZE_M = 0.005

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ShapePriorFrame0Request:
    seq: int
    source_timestamp_s: float | None
    input_source: str
    depth_backend: str
    depth_source_internal: str
    rgb_u8: np.ndarray
    object_mask: np.ndarray
    object_observation_mask: np.ndarray | None
    controller_mask: np.ndarray
    depth_color_m: np.ndarray
    k_color: np.ndarray
    camera_to_world_c2w: np.ndarray
    table_z_m: float | None = None
    table_z_above_direction: str | None = None


@dataclass(frozen=True)
class ShapePriorResult:
    seq: int
    source_seq: int | None
    source_timestamp_s: float | None
    status: str
    points_m: np.ndarray
    colors_rgb_u8: np.ndarray
    surface_points_m: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float32)
    )
    interior_points_m: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float32)
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return str(self.status) == STATUS_READY


def default_profile(*, enabled: bool) -> dict[str, Any]:
    status = STATUS_PENDING if enabled else STATUS_DISABLED
    return {
        "shape_prior_enabled": bool(enabled),
        "shape_prior_status": status,
        "shape_prior_source_seq": None,
        "shape_prior_source_time_s": None,
        "shape_prior_submit_ms": 0.0,
        "shape_prior_error": None,
    }


def _as_rgb_u8(value: np.ndarray) -> np.ndarray:
    rgb = np.asarray(value)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("shape prior rgb_u8 must have shape HxWx3")
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.uint8)
    return np.ascontiguousarray(rgb)


def _as_mask(value: np.ndarray, *, shape: tuple[int, int], name: str) -> np.ndarray:
    mask = np.asarray(value, dtype=bool)
    if mask.shape != shape:
        raise ValueError(f"{name} shape {mask.shape} does not match RGB shape {shape}")
    return np.ascontiguousarray(mask)


def _as_depth(value: np.ndarray, *, shape: tuple[int, int]) -> np.ndarray:
    depth = np.asarray(value, dtype=np.float32)
    if depth.shape != shape:
        raise ValueError(f"depth shape {depth.shape} does not match RGB shape {shape}")
    return np.ascontiguousarray(depth)


def _require_controller_name(value: str) -> str:
    name = str(value).strip()
    if not name:
        raise ValueError("shape prior controller_name must be non-empty")
    return name


def _camera_points_world(
    depth_m: np.ndarray,
    k_color: np.ndarray,
    c2w: np.ndarray,
) -> np.ndarray:
    height, width = depth_m.shape
    rows, cols = np.indices((height, width), dtype=np.float32)
    k = np.asarray(k_color, dtype=np.float32).reshape(3, 3)
    z = np.asarray(depth_m, dtype=np.float32)
    x = (cols - np.float32(k[0, 2])) * z / np.float32(k[0, 0])
    y = (rows - np.float32(k[1, 2])) * z / np.float32(k[1, 1])
    points_cam = np.stack([x, y, z], axis=-1)
    points_h = np.concatenate(
        [points_cam, np.ones((height, width, 1), dtype=np.float32)],
        axis=-1,
    )
    points_world = points_h @ np.asarray(c2w, dtype=np.float32).reshape(4, 4).T
    return np.ascontiguousarray(points_world[:, :, :3], dtype=np.float32)


def _write_rgba_input(rgb_u8: np.ndarray, object_mask: np.ndarray, path: Path) -> None:
    alpha = np.where(object_mask, 255, 0).astype(np.uint8)
    rgba = np.dstack([rgb_u8, alpha])
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba).save(path)


def _write_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.where(mask, 255, 0).astype(np.uint8)).save(path)


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _shape_prior_env(cuda_visible_devices: str) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    python_path = [str(REPO_ROOT), str(REPO_ROOT / "demo_v5_1")]
    current = env.get("PYTHONPATH")
    if current:
        python_path.append(current)
    env["PYTHONPATH"] = os.pathsep.join(python_path)
    return env


def _run_stage(command: list[str], *, env: dict[str, str]) -> float:
    start_s = time.perf_counter()
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)
    return (time.perf_counter() - start_s) * 1000.0


def _case_dir(case_root: Path, case_name: str) -> Path:
    return Path(case_root) / str(case_name)


def write_shape_prior_case(
    frame0: ShapePriorFrame0Request,
    *,
    case_root: Path,
    case_name: str,
    controller_name: str,
) -> dict[str, Path]:
    controller_name = _require_controller_name(controller_name)
    rgb = _as_rgb_u8(frame0.rgb_u8)
    image_shape = tuple(rgb.shape[:2])
    object_mask = _as_mask(frame0.object_mask, shape=image_shape, name="object_mask")
    if not np.any(object_mask):
        raise ValueError("shape prior object_mask is empty")
    observation_mask = frame0.object_observation_mask
    if observation_mask is None:
        observation_mask = object_mask
    observation_mask = _as_mask(
        observation_mask,
        shape=image_shape,
        name="object_observation_mask",
    )
    controller_mask = _as_mask(
        frame0.controller_mask,
        shape=image_shape,
        name="controller_mask",
    )
    depth_m = _as_depth(frame0.depth_color_m, shape=image_shape)
    k_color = np.asarray(frame0.k_color, dtype=np.float32).reshape(3, 3)
    c2w = np.asarray(frame0.camera_to_world_c2w, dtype=np.float32).reshape(4, 4)

    case = _case_dir(case_root, case_name)
    color_path = case / "color" / "0" / "0.png"
    rgba_path = case / "shape" / "sam3d_input_rgba.png"
    object_mask_path = case / "mask" / "0" / "0" / "0.png"
    controller_mask_path = case / "mask" / "0" / "1" / "0.png"
    pcd_path = case / "pcd" / "0.npz"

    color_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(color_path)
    _write_rgba_input(rgb, object_mask, rgba_path)
    _write_mask(object_mask, object_mask_path)
    _write_mask(controller_mask, controller_mask_path)
    _write_json(
        {"0": "stuffed animal", "1": str(controller_name)},
        case / "mask" / "mask_info_0.json",
    )
    _write_json(
        {
            "frame_num": 1,
            "intrinsics": [k_color.tolist()],
            "shape_prior_source_seq": int(frame0.seq),
            "shape_prior_source_time_s": frame0.source_timestamp_s,
            "input_source": str(frame0.input_source),
            "depth_backend": str(frame0.depth_backend),
            "depth_source_internal": str(frame0.depth_source_internal),
            "table_z_m": frame0.table_z_m,
            "table_z_above_direction": frame0.table_z_above_direction,
        },
        case / "metadata.json",
    )
    with (case / "calibrate.pkl").open("wb") as handle:
        pickle.dump([c2w], handle, protocol=pickle.HIGHEST_PROTOCOL)

    points_world = _camera_points_world(depth_m, k_color, c2w)
    valid_object = object_mask & observation_mask & np.isfinite(depth_m) & (depth_m > 0)
    object_points = points_world[valid_object]
    object_colors = rgb[valid_object].astype(np.float32) / 255.0
    if object_points.size == 0:
        raise ValueError("shape prior object observation has no valid depth points")

    valid_controller = controller_mask & np.isfinite(depth_m) & (depth_m > 0)
    controller_points = points_world[valid_controller]
    if controller_points.size == 0:
        controller_points = object_points[:1].copy()

    pcd_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        pcd_path,
        points=np.ascontiguousarray(points_world[None], dtype=np.float32),
        colors=np.ascontiguousarray((rgb[None].astype(np.float32) / 255.0)),
    )
    with (case / "mask" / "processed_masks.pkl").open("wb") as handle:
        pickle.dump(
            [[{"object": object_mask, "controller": controller_mask}]],
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    with (case / "track_process_data.pkl").open("wb") as handle:
        pickle.dump(
            {
                "object_points": object_points[None].astype(np.float32),
                "object_colors": object_colors[None].astype(np.float32),
                "object_visibilities": np.ones((1, object_points.shape[0]), dtype=bool),
                "object_motions_valid": np.ones((1, object_points.shape[0]), dtype=bool),
                "controller_points": controller_points[None].astype(np.float32),
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    return {
        "case": case,
        "rgba": rgba_path,
        "pcd": pcd_path,
        "track_process": case / "track_process_data.pkl",
    }


class ShapePriorLocalClient:
    def __init__(
        self,
        *,
        case_root: str | Path,
        controller_name: str,
        cuda_visible_devices: str = DEFAULT_SHAPE_PRIOR_WARMUP_CUDA_VISIBLE_DEVICES,
        case_name: str = CASE_NAME,
        sam3d_root: str | Path | None = None,
        sam3d_config: str | Path | None = None,
    ) -> None:
        self.case_root = Path(case_root)
        self.cuda_visible_devices = str(cuda_visible_devices)
        self.controller_name = _require_controller_name(controller_name)
        self.case_name = str(case_name)
        self.sam3d_root = None if sam3d_root is None else Path(sam3d_root)
        self.sam3d_config = None if sam3d_config is None else Path(sam3d_config)

    def request_shape_prior(self, frame0: ShapePriorFrame0Request) -> ShapePriorResult:
        paths = write_shape_prior_case(
            frame0,
            case_root=self.case_root,
            case_name=self.case_name,
            controller_name=self.controller_name,
        )
        env = _shape_prior_env(self.cuda_visible_devices)
        timings: dict[str, float] = {}

        generate_command = [
            sys.executable,
            "-m",
            "demo_v5_1.shape_prior_generate",
            "--img_path",
            str(paths["rgba"]),
            "--output_dir",
            str(paths["case"] / "shape"),
            "--skip-visualization",
        ]
        if self.sam3d_root is not None:
            generate_command.extend(["--sam3d-root", str(self.sam3d_root)])
        if self.sam3d_config is not None:
            generate_command.extend(["--config", str(self.sam3d_config)])
        timings["shape_prior_generate_ms"] = _run_stage(generate_command, env=env)

        timings["shape_prior_align_ms"] = _run_stage(
            [
                sys.executable,
                "-m",
                "demo_v5_1.shape_prior_align",
                "--base_path",
                str(self.case_root),
                "--case_name",
                self.case_name,
                "--controller_name",
                self.controller_name,
            ],
            env=env,
        )
        timings["shape_prior_sample_ms"] = _run_stage(
            [
                sys.executable,
                "-m",
                "demo_v5_1.shape_prior_sample",
                "--base_path",
                str(self.case_root),
                "--case_name",
                self.case_name,
                "--shape_prior",
                "--num_surface_points",
                str(DEFAULT_SURFACE_POINT_COUNT),
                "--volume_sample_size",
                str(DEFAULT_VOLUME_SAMPLE_SIZE_M),
            ],
            env=env,
        )
        final_data_path = paths["case"] / "final_data.pkl"
        if not final_data_path.is_file():
            raise FileNotFoundError(f"shape-prior sample did not write {final_data_path}")
        with final_data_path.open("rb") as handle:
            final_data = pickle.load(handle)

        surface = np.asarray(final_data["surface_points"], dtype=np.float32).reshape(-1, 3)
        interior = np.asarray(final_data["interior_points"], dtype=np.float32).reshape(-1, 3)
        object_points = np.asarray(final_data["object_points"], dtype=np.float32).reshape(-1, 3)
        points = np.concatenate([surface, interior, object_points], axis=0)
        colors = np.tile(np.array([[86, 180, 233]], dtype=np.uint8), (points.shape[0], 1))
        metadata = {
            **timings,
            "shape_prior_case_dir": str(paths["case"]),
            "shape_prior_warmup_cuda_visible_devices": self.cuda_visible_devices,
            "shape_prior_controller_name": self.controller_name,
            "shape_prior_surface_point_count": int(surface.shape[0]),
            "shape_prior_interior_point_count": int(interior.shape[0]),
            "shape_prior_point_count": int(points.shape[0]),
        }
        return ShapePriorResult(
            seq=int(frame0.seq),
            source_seq=int(frame0.seq),
            source_timestamp_s=frame0.source_timestamp_s,
            status=STATUS_READY,
            points_m=np.ascontiguousarray(points, dtype=np.float32),
            colors_rgb_u8=np.ascontiguousarray(colors, dtype=np.uint8),
            surface_points_m=np.ascontiguousarray(surface, dtype=np.float32),
            interior_points_m=np.ascontiguousarray(interior, dtype=np.float32),
            metadata=metadata,
        )


class ShapePriorWarmupManager:
    def __init__(self, *, enabled: bool, client: ShapePriorLocalClient | None) -> None:
        self.enabled = bool(enabled)
        self.client = client
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._result: ShapePriorResult | None = None
        self._profile = default_profile(enabled=self.enabled)

    def maybe_submit(self, frame0: ShapePriorFrame0Request) -> bool:
        if not self.enabled:
            return False
        if self.client is None:
            raise RuntimeError("shape-prior warmup is enabled without a local client")
        with self._lock:
            if self._thread is not None:
                return False
            self._profile.update(
                {
                    "shape_prior_status": STATUS_RUNNING,
                    "shape_prior_source_seq": int(frame0.seq),
                    "shape_prior_source_time_s": frame0.source_timestamp_s,
                }
            )
            self._thread = threading.Thread(
                target=self._run,
                args=(frame0,),
                name="shape-prior-warmup",
                daemon=True,
            )
            self._thread.start()
            return True

    def _run(self, frame0: ShapePriorFrame0Request) -> None:
        start_s = time.perf_counter()
        try:
            assert self.client is not None
            result = self.client.request_shape_prior(frame0)
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_s) * 1000.0
            with self._lock:
                self._profile.update(
                    {
                        "shape_prior_status": STATUS_FAILED,
                        "shape_prior_submit_ms": elapsed_ms,
                        "shape_prior_error": str(exc),
                    }
                )
            return
        elapsed_ms = (time.perf_counter() - start_s) * 1000.0
        with self._lock:
            self._result = result
            self._profile.update(result.metadata)
            self._profile.update(
                {
                    "shape_prior_status": str(result.status),
                    "shape_prior_submit_ms": elapsed_ms,
                    "shape_prior_error": None,
                    "shape_prior_source_seq": result.source_seq,
                    "shape_prior_source_time_s": result.source_timestamp_s,
                }
            )

    def wait(self, timeout_s: float | None = None) -> ShapePriorResult | None:
        thread = self._thread
        if thread is not None:
            thread.join(timeout=None if timeout_s is None else float(timeout_s))
        return self.ready_result()

    def ready_result(self) -> ShapePriorResult | None:
        with self._lock:
            result = self._result
        if result is not None and result.ready:
            return result
        return None

    def profile(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._profile)


__all__ = [
    "DEFAULT_SHAPE_PRIOR_WARMUP_CUDA_VISIBLE_DEVICES",
    "DEFAULT_SHAPE_PRIOR_TIMEOUT_MS",
    "STATUS_DISABLED",
    "STATUS_FAILED",
    "STATUS_PENDING",
    "STATUS_READY",
    "STATUS_RUNNING",
    "ShapePriorFrame0Request",
    "ShapePriorLocalClient",
    "ShapePriorResult",
    "ShapePriorWarmupManager",
    "default_profile",
    "write_shape_prior_case",
]
