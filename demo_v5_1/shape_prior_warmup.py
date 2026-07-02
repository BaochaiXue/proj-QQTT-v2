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
# Surface/interior sampling counts follow the original PhysTwin offline
# pipeline (data_process_origin/data_process_sample.py defaults).
DEFAULT_SURFACE_POINT_COUNT = 1024
DEFAULT_VOLUME_SAMPLE_SIZE_M = 0.005
# World frame convention: the table plane is z == 0 and points above the
# table have negative z (matches the origin capture convention).
TABLE_Z_ABOVE_DIRECTION = "negative"
POINTS_NPZ = Path("outputs") / "shape_prior" / "points.npz"

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ShapePriorFrame0Request:
    """Frame-0 capture snapshot needed to build an offline-style case dir."""

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
        """Return the ready."""
        return str(self.status) == STATUS_READY


def default_profile(*, enabled: bool) -> dict[str, Any]:
    """Return the default profile."""
    status = STATUS_PENDING if enabled else STATUS_DISABLED
    return {
        "shape_prior_enabled": bool(enabled),
        "shape_prior_status": status,
        "shape_prior_source_seq": None,
        "shape_prior_source_time_s": None,
        "shape_prior_submit_ms": 0.0,
        "shape_prior_error": None,
    }


# ---------------------------------------------------------------------------
# Input validation and small IO helpers
# ---------------------------------------------------------------------------


def _as_mask(value: np.ndarray, *, shape: tuple[int, int], name: str) -> np.ndarray:
    """Return the as mask."""
    mask = np.asarray(value, dtype=bool)
    if mask.shape != shape:
        raise ValueError(f"{name} shape {mask.shape} does not match RGB shape {shape}")
    return np.ascontiguousarray(mask)


def _require_name(value: str, *, field_name: str) -> str:
    """Return validated name."""
    name = str(value).strip()
    if not name:
        raise ValueError(f"shape prior {field_name} must be non-empty")
    return name


def _camera_points_world(
    depth_m: np.ndarray,
    k_color: np.ndarray,
    c2w: np.ndarray,
) -> np.ndarray:
    """Backproject a dense depth map (meters) to an HxWx3 world-point grid.

    Uses the color pinhole intrinsics ``k_color`` (3x3) and the row-major
    camera-to-world extrinsics ``c2w`` (4x4). Pixels keep their grid position
    so downstream masks can index the result directly.
    """
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


def _write_mask(mask: np.ndarray, path: Path) -> None:
    """Write a boolean mask image to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.where(mask, 255, 0).astype(np.uint8)).save(path)


def _write_json(payload: dict[str, Any], path: Path) -> None:
    """Write JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_stage(command: list[str], *, env: dict[str, str]) -> float:
    """Run one pipeline stage as a subprocess and return its wall time in ms.

    Kept as a module-level function: tests patch it to fake the stage chain.
    """
    start_s = time.perf_counter()
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)
    return (time.perf_counter() - start_s) * 1000.0


def _require_stage_file(path: Path, *, stage_name: str) -> None:
    """Return validated stage file."""
    if not path.is_file():
        raise FileNotFoundError(f"{stage_name} did not write {path}")


def _points_array(value: np.ndarray, *, name: str) -> np.ndarray:
    """Return the points array."""
    points = np.asarray(value, dtype=np.float32)
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} must have shape Nx3")
    return np.ascontiguousarray(points, dtype=np.float32)


def write_shape_prior_points_npz(
    path: str | Path,
    *,
    surface_points: np.ndarray,
    interior_points: np.ndarray,
) -> Path:
    """Write shape prior points NPZ."""
    output_path = Path(path)
    surface = _points_array(surface_points, name="surface_points")
    interior = _points_array(interior_points, name="interior_points")
    points = np.concatenate([surface, interior], axis=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        surface_points=surface,
        interior_points=interior,
        points=np.ascontiguousarray(points, dtype=np.float32),
    )
    return output_path


def write_shape_prior_case(
    frame0: ShapePriorFrame0Request,
    *,
    case_root: Path,
    case_name: str,
    object_name: str,
    controller_name: str,
) -> dict[str, Path]:
    """Serialize frame 0 as a one-frame, one-camera offline-style case dir.

    The directory layout (color/, mask/, pcd/, calibrate.pkl, metadata.json,
    processed_masks.pkl, track_process_data.pkl) mirrors what the original
    PhysTwin data_process_origin scripts expect, with camera index 0 and
    frame index 0.
    """
    object_name = _require_name(object_name, field_name="object_name")
    controller_name = _require_name(controller_name, field_name="controller_name")

    rgb = np.asarray(frame0.rgb_u8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("shape prior rgb_u8 must have shape HxWx3")
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.uint8)
    rgb = np.ascontiguousarray(rgb)
    image_shape = tuple(rgb.shape[:2])
    object_mask = _as_mask(frame0.object_mask, shape=image_shape, name="object_mask")
    if not np.any(object_mask):
        raise ValueError("shape prior object_mask is empty")
    # Without an explicit observation mask the segmentation mask doubles as
    # the depth-observation gate.
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
    depth_m = np.asarray(frame0.depth_color_m, dtype=np.float32)
    if depth_m.shape != image_shape:
        raise ValueError(
            f"depth shape {depth_m.shape} does not match RGB shape {image_shape}"
        )
    depth_m = np.ascontiguousarray(depth_m)
    k_color = np.asarray(frame0.k_color, dtype=np.float32).reshape(3, 3)
    c2w = np.asarray(frame0.camera_to_world_c2w, dtype=np.float32).reshape(4, 4)

    case = Path(case_root) / str(case_name)
    color_path = case / "color" / "0" / "0.png"
    shape_dir = case / "shape"
    object_mask_path = case / "mask" / "0" / "0" / "0.png"
    controller_mask_path = case / "mask" / "0" / "1" / "0.png"
    pcd_path = case / "pcd" / "0.npz"

    shape_dir.mkdir(parents=True, exist_ok=True)
    color_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(color_path)
    _write_mask(object_mask, object_mask_path)
    _write_mask(controller_mask, controller_mask_path)
    _write_json(
        {"0": object_name, "1": controller_name},
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
            "table_z_above_direction": TABLE_Z_ABOVE_DIRECTION,
        },
        case / "metadata.json",
    )
    with (case / "calibrate.pkl").open("wb") as handle:
        pickle.dump([c2w], handle, protocol=pickle.HIGHEST_PROTOCOL)

    points_world = _camera_points_world(depth_m, k_color, c2w)
    depth_valid = np.isfinite(depth_m) & (depth_m > 0)
    valid_object = object_mask & observation_mask & depth_valid
    valid_controller = controller_mask & depth_valid
    from demo_v5_1.chunk_data_stream import (  # noqa: PLC0415
        _apply_radius_outlier_to_mask_frame,
    )

    # Intentional parity with data_process_origin/data_process_mask.py: original
    # PhysTwin removes unsupported 3D radius outliers by clearing pixels in
    # processed_masks.pkl, while pcd/0.npz stays as the dense point grid. The
    # align stage then filters with points[processed_mask].
    cleaned_masks = _apply_radius_outlier_to_mask_frame(
        frame={"object": valid_object, "controller": valid_controller},
        points_grid=points_world,
        enabled=True,
        radius_m=0.01,
        nb_points=40,
    )
    valid_object = _as_mask(
        cleaned_masks["object"],
        shape=image_shape,
        name="processed object mask",
    )
    valid_controller = _as_mask(
        cleaned_masks["controller"],
        shape=image_shape,
        name="processed controller mask",
    )
    object_points = points_world[valid_object]
    object_colors = rgb[valid_object].astype(np.float32) / 255.0
    if object_points.size == 0:
        raise ValueError("shape prior object observation has no valid depth points")

    controller_points = points_world[valid_controller]
    if controller_points.size == 0:
        # Downstream stages require at least one controller point; borrow an
        # object point rather than failing the whole warmup.
        controller_points = object_points[:1].copy()

    # pcd/0.npz keeps the dense HxW grid (leading axis = camera index) so the
    # align stage can index it with the processed masks.
    pcd_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        pcd_path,
        points=np.ascontiguousarray(points_world[None], dtype=np.float32),
        colors=np.ascontiguousarray((rgb[None].astype(np.float32) / 255.0)),
        masks=np.ascontiguousarray(depth_valid[None], dtype=bool),
    )
    with (case / "mask" / "processed_masks.pkl").open("wb") as handle:
        pickle.dump(
            [[{"object": valid_object, "controller": valid_controller}]],
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
        "color": color_path,
        "object_mask": object_mask_path,
        "shape": shape_dir,
        "pcd": pcd_path,
        "track_process": case / "track_process_data.pkl",
    }


class ShapePriorLocalClient:
    """Runs the offline shape-prior chain locally on the warmup GPU.

    Stage order matches the original PhysTwin pipeline: image upscale ->
    SAM3.1 image segmentation (in-process) -> SAM3D generate -> align ->
    sample. All subprocess stages inherit CUDA_VISIBLE_DEVICES so they stay
    off the realtime GPU.
    """

    def __init__(
        self,
        *,
        case_root: str | Path,
        object_name: str,
        controller_name: str,
        cuda_visible_devices: str = DEFAULT_SHAPE_PRIOR_WARMUP_CUDA_VISIBLE_DEVICES,
        case_name: str = CASE_NAME,
        points_npz: str | Path = POINTS_NPZ,
        sam3d_root: str | Path | None = None,
        sam3d_config: str | Path | None = None,
        sam31_device: str = "cuda",
        reuse_sam31_model: bool = True,
    ) -> None:
        """Initialize ShapePriorLocalClient."""
        self.case_root = Path(case_root)
        self.cuda_visible_devices = str(cuda_visible_devices)
        self.object_name = _require_name(object_name, field_name="object_name")
        self.controller_name = _require_name(
            controller_name,
            field_name="controller_name",
        )
        self.case_name = str(case_name)
        self.points_npz = Path(points_npz)
        self.sam3d_root = None if sam3d_root is None else Path(sam3d_root)
        self.sam3d_config = None if sam3d_config is None else Path(sam3d_config)
        self.sam31_device = str(sam31_device)
        self.reuse_sam31_model = bool(reuse_sam31_model)

    def request_shape_prior(self, frame0: ShapePriorFrame0Request) -> ShapePriorResult:
        """Request shape prior."""
        paths = write_shape_prior_case(
            frame0,
            case_root=self.case_root,
            case_name=self.case_name,
            object_name=self.object_name,
            controller_name=self.controller_name,
        )
        # Subprocess stages must import demo_v5_1 both as a package (repo
        # root) and as top-level modules (demo_v5_1/), pinned to the warmup
        # GPU via CUDA_VISIBLE_DEVICES.
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.cuda_visible_devices)
        python_path = [str(REPO_ROOT), str(REPO_ROOT / "demo_v5_1")]
        current = env.get("PYTHONPATH")
        if current:
            python_path.append(current)
        env["PYTHONPATH"] = os.pathsep.join(python_path)
        timings: dict[str, float] = {}

        high_resolution_path = paths["shape"] / "high_resolution.png"
        masked_image_path = paths["shape"] / "masked_image.png"

        timings["shape_prior_upscale_ms"] = _run_stage(
            [
                sys.executable,
                "-m",
                "demo_v5_1.utils.image_upscale",
                "--img_path",
                str(paths["color"]),
                "--mask_path",
                str(paths["object_mask"]),
                "--output_path",
                str(high_resolution_path),
                "--category",
                self.object_name,
            ],
            env=env,
        )
        _require_stage_file(high_resolution_path, stage_name="shape-prior upscale")

        from demo_v5_1 import sam31_image_segmentation  # noqa: PLC0415

        segment_start_s = time.perf_counter()
        sam31_image_segmentation.segment_image_to_origin_rgba(
            img_path=high_resolution_path,
            text_prompt=self.object_name,
            output_path=masked_image_path,
            device=self.sam31_device,
            reuse_model=self.reuse_sam31_model,
        )
        timings["shape_prior_segment_image_ms"] = (
            time.perf_counter() - segment_start_s
        ) * 1000.0
        _require_stage_file(masked_image_path, stage_name="shape-prior segment")

        generate_command = [
            sys.executable,
            "-m",
            "demo_v5_1.shape_prior_generate",
            "--img_path",
            str(masked_image_path),
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

        surface = _points_array(final_data["surface_points"], name="surface_points")
        interior = _points_array(final_data["interior_points"], name="interior_points")
        points = np.concatenate([surface, interior], axis=0)
        write_shape_prior_points_npz(
            self.points_npz,
            surface_points=surface,
            interior_points=interior,
        )
        # Uniform display tint for prior points; real per-point colors are
        # not available for sampled surface/interior points.
        colors = np.tile(
            np.array([[86, 180, 233]], dtype=np.uint8),
            (points.shape[0], 1),
        )
        metadata = {
            **timings,
            "shape_prior_case_dir": str(paths["case"]),
            "shape_prior_points_npz": str(self.points_npz),
            "shape_prior_warmup_cuda_visible_devices": self.cuda_visible_devices,
            "shape_prior_object_name": self.object_name,
            "shape_prior_controller_name": self.controller_name,
            "shape_prior_sam31_device": self.sam31_device,
            "shape_prior_sam31_reuse_model": self.reuse_sam31_model,
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
    """One-shot background runner for the shape-prior warmup.

    ``maybe_submit`` accepts only the first frame-0 request; the pipeline runs
    on a daemon thread and its status/timings are mirrored into the profile
    dict under the lock.
    """

    def __init__(self, *, enabled: bool, client: ShapePriorLocalClient | None) -> None:
        """Initialize ShapePriorWarmupManager."""
        self.enabled = bool(enabled)
        self.client = client
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._result: ShapePriorResult | None = None
        self._profile = default_profile(enabled=self.enabled)

    def maybe_submit(self, frame0: ShapePriorFrame0Request) -> bool:
        """Maybe start or update submit."""
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
        """Run ShapePriorWarmupManager."""
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
        """Wait for ShapePriorWarmupManager."""
        thread = self._thread
        if thread is not None:
            thread.join(timeout=None if timeout_s is None else float(timeout_s))
        return self.ready_result()

    def ready_result(self) -> ShapePriorResult | None:
        """Return the ready result."""
        with self._lock:
            result = self._result
        if result is not None and result.ready:
            return result
        return None

    def profile(self) -> dict[str, Any]:
        """Return the profile."""
        with self._lock:
            return dict(self._profile)


__all__ = [
    "DEFAULT_SHAPE_PRIOR_WARMUP_CUDA_VISIBLE_DEVICES",
    "DEFAULT_SHAPE_PRIOR_TIMEOUT_MS",
    "POINTS_NPZ",
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
    "write_shape_prior_points_npz",
]
