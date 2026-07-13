"""Single-camera shape-prior warmup for Demo v6.1."""

from __future__ import annotations

import atexit
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Mapping

import numpy as np
from PIL import Image

from demo_v6_2.shape_prior_timing import (
    _pre_submit_timing,
    build_critical_path_analysis,
    critical_path_entry,
    elapsed_ms,
    load_completed_stage_profile,
)


STATUS_DISABLED = "disabled"
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_READY = "ready"
STATUS_FAILED = "failed"

DEFAULT_SHAPE_PRIOR_TIMEOUT_MS = 180_000
DEFAULT_SHAPE_PRIOR_WARMUP_CUDA_VISIBLE_DEVICES = "0"
CASE_NAME = "shape_prior_frame0"
# Surface/interior sampling counts follow the original PhysTwin offline
# pipeline (data_process_origin/data_process_sample.py defaults).
DEFAULT_SURFACE_POINT_COUNT = 1024
DEFAULT_VOLUME_SAMPLE_SIZE_M = 0.005
# World frame convention: the table plane is z == 0 and points above the
# table have negative z (matches the origin capture convention).
TABLE_Z_ABOVE_DIRECTION = "negative"
POINTS_NPZ = Path("outputs_v6_1") / "shape_prior" / "points.npz"
# Pre-warmed one-shot stage workers: spawned at app boot with --wait-signal so
# model loading happens off the frame-0 critical path; each worker runs its
# stage once on GO and exits, releasing its whole CUDA context.
PREWARM_STAGE_UPSCALE = "upscale"
PREWARM_STAGE_GENERATE = "generate"
PREWARM_STAGE_ALIGN = "align"
PREWARM_STAGES = (
    PREWARM_STAGE_UPSCALE,
    PREWARM_STAGE_GENERATE,
    PREWARM_STAGE_ALIGN,
)
PREWARM_WORKER_EXIT_TIMEOUT_S = 10.0

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
    controller_mask: np.ndarray
    depth_color_m: np.ndarray
    depth_valid_mask: np.ndarray
    points_world_m: np.ndarray
    k_color: np.ndarray
    camera_to_world_c2w: np.ndarray
    table_z_m: float | None = None
    warmup_runtime_start_perf_s: float | None = None
    frame_receive_perf_s: float | None = None
    frame_mask_ready_perf_s: float | None = None
    frame_pcd_ready_perf_s: float | None = None
    frame0_pipeline_timing_ms: dict[str, float] = field(default_factory=dict)
    frame0_perception_profile: dict[str, Any] = field(default_factory=dict)


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
        "shape_prior_request_total_ms": 0.0,
        "warmup_runtime_start_to_shape_prior_ready_ms": None,
        "warmup_shape_prior_ready_to_gate_open_ms": None,
        "warmup_total_ms": None,
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


def _write_mask(mask: np.ndarray, path: Path) -> None:
    """Write a boolean mask image to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.where(mask, 255, 0).astype(np.uint8)).save(path)


def _write_json(payload: dict[str, Any], path: Path) -> None:
    """Write JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


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
    depth_valid = _as_mask(
        frame0.depth_valid_mask,
        shape=image_shape,
        name="depth_valid_mask",
    )
    points_world = np.asarray(frame0.points_world_m, dtype=np.float32)
    if points_world.shape != (*image_shape, 3):
        raise ValueError(
            "shape prior points_world_m must have shape "
            f"{(*image_shape, 3)}; got {points_world.shape}"
        )
    points_world = np.ascontiguousarray(points_world)
    # Masks are guaranteed depth-valid subsets upstream by
    # ProcessedFramePacket.__post_init__ (mdp_packets), so no re-check here.
    k_color = np.asarray(frame0.k_color, dtype=np.float32).reshape(3, 3)
    c2w = np.asarray(frame0.camera_to_world_c2w, dtype=np.float32).reshape(4, 4)
    if not np.isfinite(k_color).all():
        raise ValueError("shape prior color intrinsics must be finite")
    if not np.isfinite(c2w).all():
        raise ValueError("shape prior camera-to-world transform must be finite")
    if not np.isfinite(points_world[object_mask | controller_mask]).all():
        raise ValueError("shape prior processed masks contain non-finite 3D points")

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

    object_points = points_world[object_mask]
    object_colors = rgb[object_mask].astype(np.float32) / 255.0
    if object_points.size == 0:
        raise ValueError("shape prior object observation has no valid depth points")

    controller_points = points_world[controller_mask]
    if controller_points.size == 0:
        raise ValueError("shape prior controller observation has no valid points")

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
                "object_motions_valid": np.ones(
                    (1, object_points.shape[0]), dtype=bool
                ),
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
        self._prewarm_workers: dict[str, subprocess.Popen[str]] = {}
        self._prewarm_lock = threading.Lock()
        self._atexit_registered = False

    def _stage_profile_path(self, stage: str) -> Path:
        """Return the fixed detailed timing path for one subprocess stage."""
        return self.case_root / self.case_name / "shape" / "timing" / f"{stage}.json"

    def _stage_env(self) -> dict[str, str]:
        """Environment shared by all stage subprocesses (cold or pre-warmed)."""
        # Subprocess stages must import demo_v6_1 both as a package (repo
        # root) and as top-level modules (demo_v6_1/), pinned to the warmup
        # GPU via CUDA_VISIBLE_DEVICES.
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.cuda_visible_devices)
        python_path = [str(REPO_ROOT), str(REPO_ROOT / "demo_v6_2")]
        current = env.get("PYTHONPATH")
        if current:
            python_path.append(current)
        env["PYTHONPATH"] = os.pathsep.join(python_path)
        return env

    def _stage_commands(self) -> dict[str, list[str]]:
        """Stage commands; paths mirror the write_shape_prior_case layout."""
        case = Path(self.case_root) / self.case_name
        shape_dir = case / "shape"
        upscale = [
            sys.executable,
            "-m",
            "demo_v6_2.utils.image_upscale",
            "--img_path",
            str(case / "color" / "0" / "0.png"),
            "--mask_path",
            str(case / "mask" / "0" / "0" / "0.png"),
            "--output_path",
            str(shape_dir / "high_resolution.png"),
            "--category",
            self.object_name,
            "--profile-json",
            str(self._stage_profile_path(PREWARM_STAGE_UPSCALE)),
        ]
        generate = [
            sys.executable,
            "-m",
            "demo_v6_2.shape_prior_generate",
            "--img_path",
            str(shape_dir / "masked_image.png"),
            "--output_dir",
            str(shape_dir),
            "--skip-visualization",
            "--profile-json",
            str(self._stage_profile_path(PREWARM_STAGE_GENERATE)),
        ]
        if self.sam3d_root is not None:
            generate.extend(["--sam3d-root", str(self.sam3d_root)])
        if self.sam3d_config is not None:
            generate.extend(["--config", str(self.sam3d_config)])
        align = [
            sys.executable,
            "-m",
            "demo_v6_2.shape_prior_align",
            "--base_path",
            str(self.case_root),
            "--case_name",
            self.case_name,
            "--controller_name",
            self.controller_name,
            "--profile-json",
            str(self._stage_profile_path(PREWARM_STAGE_ALIGN)),
        ]
        return {
            PREWARM_STAGE_UPSCALE: upscale,
            PREWARM_STAGE_GENERATE: generate,
            PREWARM_STAGE_ALIGN: align,
        }

    def prewarm(self) -> None:
        """Spawn pre-warmed one-shot workers for the heavy subprocess stages.

        Each worker front-loads what it safely can (stage inputs need not
        exist yet), then blocks on stdin until ``request_shape_prior`` writes
        GO after the frame-0 case dir is on disk. Workers exit after one
        request, so no VRAM stays allocated once the warmup finishes.

        VRAM budget on the single warmup GPU: upscale (SD-x4 weights) and
        align (SuperGlue) preload fully, but the generate worker only runs
        its CPU-side import tree before GO -- SAM3D weights would otherwise
        be resident during the upscale stage's inference peak, which does not
        fit on a 24GB card. Weights move to the GPU after GO, exactly like
        the cold path's serial ordering.
        """
        with self._prewarm_lock:
            env = self._stage_env()
            commands = self._stage_commands()
            for stage in PREWARM_STAGES:
                if stage in self._prewarm_workers:
                    continue
                self._prewarm_workers[stage] = subprocess.Popen(
                    [*commands[stage], "--wait-signal"],
                    cwd=REPO_ROOT,
                    env=env,
                    stdin=subprocess.PIPE,
                    text=True,
                )
            if not self._atexit_registered:
                atexit.register(self.close)
                self._atexit_registered = True

    def close(self) -> None:
        """Ask any unused pre-warmed workers to exit and reap them."""
        with self._prewarm_lock:
            workers = dict(self._prewarm_workers)
            self._prewarm_workers.clear()
        for worker in workers.values():
            if worker.poll() is not None:
                continue
            try:
                if worker.stdin is not None:
                    worker.stdin.write("EXIT\n")
                    worker.stdin.flush()
                    worker.stdin.close()
            except (BrokenPipeError, OSError, ValueError):
                pass
            try:
                worker.wait(timeout=PREWARM_WORKER_EXIT_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                worker.terminate()
                try:
                    worker.wait(timeout=PREWARM_WORKER_EXIT_TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    worker.wait()

    def _run_prewarmed_stage(
        self, worker: subprocess.Popen[str], *, stage_name: str
    ) -> tuple[float, float]:
        """Signal GO and return its critical-path time and wall timestamp."""
        if worker.poll() is not None:
            raise RuntimeError(
                f"pre-warmed {stage_name} worker exited before GO "
                f"with code {worker.returncode}"
            )
        start_s = time.perf_counter()
        go_wall_time_s = time.time()
        assert worker.stdin is not None
        worker.stdin.write("GO\n")
        worker.stdin.flush()
        worker.stdin.close()
        returncode = worker.wait()
        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, worker.args)
        return elapsed_ms(start_s), float(go_wall_time_s)

    def _run_stage_maybe_prewarmed(
        self,
        stage: str,
        command: list[str],
        *,
        env: dict[str, str],
        prewarmed_stages: list[str],
    ) -> tuple[float, dict[str, Any]]:
        """Run a stage via its pre-warmed worker when present, else cold."""
        with self._prewarm_lock:
            worker = self._prewarm_workers.pop(stage, None)
        if worker is None:
            stage_ms = _run_stage(command, env=env)
            return stage_ms, {
                "execution_mode": "cold",
                "critical_path_ms": float(stage_ms),
                "go_wall_time_s": None,
            }
        stage_ms, go_wall_time_s = self._run_prewarmed_stage(
            worker,
            stage_name=stage,
        )
        prewarmed_stages.append(stage)
        return stage_ms, {
            "execution_mode": "prewarmed",
            "critical_path_ms": float(stage_ms),
            "go_wall_time_s": float(go_wall_time_s),
        }

    def _completed_stage_details(
        self,
        stage: str,
        *,
        orchestration: dict[str, Any],
    ) -> dict[str, Any]:
        """Load child timing and attribute any unfinished prewarm at GO."""
        profile = load_completed_stage_profile(
            self._stage_profile_path(stage),
            expected_stage=stage,
        )
        expected_mode = str(orchestration["execution_mode"])
        if str(profile.get("execution_mode")) != expected_mode:
            raise ValueError(
                f"shape-prior {stage} execution mode mismatch: "
                f"parent={expected_mode!r} child={profile.get('execution_mode')!r}"
            )
        go_wall_time_s = orchestration.get("go_wall_time_s")
        ready_wall_time_s = profile.get("ready_wall_time_s")
        readiness: dict[str, Any] = {
            "ready_before_go": None,
            "ready_lead_ms": None,
            "startup_tail_on_critical_path_ms": None,
        }
        if expected_mode == "prewarmed":
            if go_wall_time_s is None or ready_wall_time_s is None:
                raise ValueError(
                    f"prewarmed shape-prior stage {stage!r} lacks READY timing"
                )
            delta_ms = (float(go_wall_time_s) - float(ready_wall_time_s)) * 1000.0
            readiness = {
                "ready_before_go": bool(delta_ms >= 0.0),
                "ready_lead_ms": max(0.0, float(delta_ms)),
                "startup_tail_on_critical_path_ms": max(
                    0.0,
                    float(-delta_ms),
                ),
            }
        critical_path_ms = float(orchestration["critical_path_ms"])
        snapshot_wall_time_s = float(profile["snapshot_wall_time_s"])
        if go_wall_time_s is not None:
            profile_snapshot_from_parent_start_ms = max(
                0.0,
                (snapshot_wall_time_s - float(go_wall_time_s)) * 1000.0,
            )
        else:
            process_lifetime_ms = profile["timing_ms"].get("process_lifetime_ms")
            profile_snapshot_from_parent_start_ms = (
                None if process_lifetime_ms is None else float(process_lifetime_ms)
            )
        exit_after_profile_ms = (
            None
            if profile_snapshot_from_parent_start_ms is None
            else max(
                0.0,
                critical_path_ms - profile_snapshot_from_parent_start_ms,
            )
        )
        details = dict(profile)
        details["orchestration"] = {
            **orchestration,
            **readiness,
            "profile_snapshot_from_parent_start_ms": (
                profile_snapshot_from_parent_start_ms
            ),
            "profile_snapshot_to_parent_return_ms": exit_after_profile_ms,
        }
        return details

    def request_shape_prior(self, frame0: ShapePriorFrame0Request) -> ShapePriorResult:
        """Request shape prior."""
        request_start_s = time.perf_counter()
        critical_path: list[dict[str, Any]] = []

        case_start_s = time.perf_counter()
        paths = write_shape_prior_case(
            frame0,
            case_root=self.case_root,
            case_name=self.case_name,
            object_name=self.object_name,
            controller_name=self.controller_name,
        )
        case_end_s = time.perf_counter()
        critical_path.append(
            critical_path_entry(
                stage="case_write",
                path_start_s=request_start_s,
                stage_start_s=case_start_s,
                stage_end_s=case_end_s,
            )
        )
        env = self._stage_env()
        commands = self._stage_commands()
        prewarmed_stages: list[str] = []

        high_resolution_path = paths["shape"] / "high_resolution.png"
        masked_image_path = paths["shape"] / "masked_image.png"

        def record_stage(
            stage: str,
            run: Callable[[], Any],
            build_details: Callable[[Any], Mapping[str, Any]],
            *,
            require_file: tuple[Path, str] | None = None,
        ) -> None:
            """Run one stage, then append its validated critical-path entry."""
            stage_start_s = time.perf_counter()
            outcome = run()
            stage_end_s = time.perf_counter()
            if require_file is not None:
                _require_stage_file(require_file[0], stage_name=require_file[1])
            critical_path.append(
                critical_path_entry(
                    stage=stage,
                    path_start_s=request_start_s,
                    stage_start_s=stage_start_s,
                    stage_end_s=stage_end_s,
                    details=build_details(outcome),
                )
            )

        def record_subprocess_stage(
            stage: str,
            *,
            require_file: tuple[Path, str] | None = None,
        ) -> None:
            """Record one cold-or-prewarmed subprocess stage run."""
            record_stage(
                stage,
                lambda: self._run_stage_maybe_prewarmed(
                    stage, commands[stage], env=env, prewarmed_stages=prewarmed_stages
                ),
                lambda outcome: self._completed_stage_details(
                    stage, orchestration=outcome[1]
                ),
                require_file=require_file,
            )

        record_subprocess_stage(
            PREWARM_STAGE_UPSCALE,
            require_file=(high_resolution_path, "shape-prior upscale"),
        )

        from demo_v6_2.perception import (  # noqa: PLC0415
            sam31_image_segmentation,
        )

        record_stage(
            "segment_image",
            lambda: sam31_image_segmentation.segment_image_to_origin_rgba(
                img_path=high_resolution_path,
                text_prompt=self.object_name,
                output_path=masked_image_path,
                device=self.sam31_device,
                reuse_model=self.reuse_sam31_model,
            ),
            lambda outcome: outcome[1],
            require_file=(masked_image_path, "shape-prior segment"),
        )
        record_subprocess_stage(PREWARM_STAGE_GENERATE)
        record_subprocess_stage(PREWARM_STAGE_ALIGN)

        sample_command = [
            sys.executable,
            "-m",
            "demo_v6_2.shape_prior_sample",
            "--base_path",
            str(self.case_root),
            "--case_name",
            self.case_name,
            "--shape_prior",
            "--num_surface_points",
            str(DEFAULT_SURFACE_POINT_COUNT),
            "--volume_sample_size",
            str(DEFAULT_VOLUME_SAMPLE_SIZE_M),
            "--profile-json",
            str(self._stage_profile_path("sample")),
        ]
        record_stage(
            "sample",
            lambda: _run_stage(sample_command, env=env),
            lambda sample_ms: self._completed_stage_details(
                "sample",
                orchestration={
                    "execution_mode": "cold",
                    "critical_path_ms": float(sample_ms),
                    "go_wall_time_s": None,
                },
            ),
        )

        finalize_start_s = time.perf_counter()
        final_data_path = paths["case"] / "final_data.pkl"
        _require_stage_file(final_data_path, stage_name="shape-prior sample")
        load_start_s = time.perf_counter()
        with final_data_path.open("rb") as handle:
            final_data = pickle.load(handle)
        load_end_s = time.perf_counter()

        surface = _points_array(final_data["surface_points"], name="surface_points")
        interior = _points_array(final_data["interior_points"], name="interior_points")
        points = np.concatenate([surface, interior], axis=0)
        points_write_start_s = time.perf_counter()
        write_shape_prior_points_npz(
            self.points_npz,
            surface_points=surface,
            interior_points=interior,
        )
        points_write_end_s = time.perf_counter()
        # Uniform display tint for prior points; real per-point colors are
        # not available for sampled surface/interior points.
        colors = np.tile(
            np.array([[86, 180, 233]], dtype=np.uint8),
            (points.shape[0], 1),
        )
        finalize_end_s = time.perf_counter()
        critical_path.append(
            critical_path_entry(
                stage="result_finalize",
                path_start_s=request_start_s,
                stage_start_s=finalize_start_s,
                stage_end_s=finalize_end_s,
                details={
                    "final_data_load_ms": elapsed_ms(load_start_s, load_end_s),
                    "points_npz_write_ms": elapsed_ms(
                        points_write_start_s,
                        points_write_end_s,
                    ),
                },
            )
        )
        request_total_ms = elapsed_ms(request_start_s, finalize_end_s)
        timing_analysis = build_critical_path_analysis(
            critical_path,
            total_ms=request_total_ms,
        )
        timing_analysis["pre_submit"] = _pre_submit_timing(
            frame0,
            request_start_s=request_start_s,
        )
        stage_ms = {
            entry["stage"]: float(entry["duration_ms"])
            for entry in timing_analysis["critical_path"]
        }
        metadata = {
            "shape_prior_case_write_ms": stage_ms["case_write"],
            "shape_prior_upscale_ms": stage_ms["upscale"],
            "shape_prior_segment_image_ms": stage_ms["segment_image"],
            "shape_prior_generate_ms": stage_ms["generate"],
            "shape_prior_align_ms": stage_ms["align"],
            "shape_prior_sample_ms": stage_ms["sample"],
            "shape_prior_result_finalize_ms": stage_ms["result_finalize"],
            "shape_prior_request_total_ms": request_total_ms,
            "shape_prior_timing": timing_analysis,
            "shape_prior_case_dir": str(paths["case"]),
            "shape_prior_points_npz": str(self.points_npz),
            "shape_prior_warmup_cuda_visible_devices": self.cuda_visible_devices,
            "shape_prior_object_name": self.object_name,
            "shape_prior_controller_name": self.controller_name,
            "shape_prior_sam31_device": self.sam31_device,
            "shape_prior_sam31_reuse_model": self.reuse_sam31_model,
            "shape_prior_prewarmed_stages": list(prewarmed_stages),
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
        self._warmup_runtime_start_perf_s: float | None = None
        self._ready_perf_s: float | None = None
        self._gate_open_perf_s: float | None = None

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
            self._warmup_runtime_start_perf_s = frame0.warmup_runtime_start_perf_s
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
            failed_s = time.perf_counter()
            request_total_ms = elapsed_ms(start_s, failed_s)
            with self._lock:
                self._profile.update(
                    {
                        "shape_prior_status": STATUS_FAILED,
                        "shape_prior_request_total_ms": request_total_ms,
                        "shape_prior_error": str(exc),
                    }
                )
                if self._warmup_runtime_start_perf_s is not None:
                    self._profile["warmup_runtime_start_to_shape_prior_failure_ms"] = (
                        elapsed_ms(
                            self._warmup_runtime_start_perf_s,
                            failed_s,
                        )
                    )
            return
        ready_perf_s = time.perf_counter()
        request_total_ms = elapsed_ms(start_s, ready_perf_s)
        with self._lock:
            self._ready_perf_s = ready_perf_s
            self._result = result
            self._profile.update(result.metadata)
            self._profile.update(
                {
                    "shape_prior_status": str(result.status),
                    "shape_prior_request_total_ms": request_total_ms,
                    "shape_prior_ready_wall_time_s": float(time.time()),
                    "shape_prior_error": None,
                    "shape_prior_source_seq": result.source_seq,
                    "shape_prior_source_time_s": result.source_timestamp_s,
                }
            )
            if self._warmup_runtime_start_perf_s is not None:
                self._profile["warmup_runtime_start_to_shape_prior_ready_ms"] = (
                    elapsed_ms(
                        self._warmup_runtime_start_perf_s,
                        ready_perf_s,
                    )
                )

    def mark_gate_open(self) -> None:
        """Record the first moment the formal timeline observes READY."""
        gate_open_perf_s = time.perf_counter()
        gate_open_wall_time_s = time.time()
        with self._lock:
            if self._gate_open_perf_s is not None:
                return
            if self._ready_perf_s is None:
                raise RuntimeError("shape-prior gate opened before READY")
            self._gate_open_perf_s = gate_open_perf_s
            ready_to_gate_ms = elapsed_ms(self._ready_perf_s, gate_open_perf_s)
            self._profile.update(
                {
                    "shape_prior_gate_open_wall_time_s": float(gate_open_wall_time_s),
                    "warmup_shape_prior_ready_to_gate_open_ms": ready_to_gate_ms,
                }
            )
            if self._warmup_runtime_start_perf_s is not None:
                self._profile["warmup_total_ms"] = elapsed_ms(
                    self._warmup_runtime_start_perf_s,
                    gate_open_perf_s,
                )

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
