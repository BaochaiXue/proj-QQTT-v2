"""Single-camera shape-prior warmup for Demo v6.2."""

from __future__ import annotations

import atexit
from dataclasses import dataclass, field
import os
from pathlib import Path
import pickle
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Mapping

import numpy as np

from demo_v6_2.shape_prior.case import (
    ShapePriorFrame0Request,
    points_array,
    require_name,
    write_shape_prior_case,
    write_shape_prior_points_npz,
)
from demo_v6_2.shape_prior import mesh_cache
from demo_v6_2.shape_prior.mesh_cache import (
    ShapePriorMeshCache,
    normalize_object_id,
)
from demo_v6_2.tracking import DEFAULT_VOLUME_SAMPLE_SIZE_M
from demo_v6_2.shape_prior.timing import (
    build_critical_path_analysis,
    critical_path_entry,
    elapsed_ms,
    load_completed_stage_profile,
    pre_submit_timing,
)
from demo_v6_2.utils.atomic_io import atomic_json_dump


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
# Mirrors demo_v6_2/shape_prior/generate.py DEFAULT_SEED; passed explicitly so
# the mesh-cache manifest records the exact generation seed.
DEFAULT_GENERATE_SEED = 42
POINTS_NPZ = Path("outputs") / "shape_prior" / "points.npz"
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

REPO_ROOT = Path(__file__).resolve().parents[2]


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


def _run_stage(command: list[str], *, env: dict[str, str]) -> float:
    """Run one pipeline stage as a subprocess and return its wall time in ms."""
    start_s = time.perf_counter()
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)
    return (time.perf_counter() - start_s) * 1000.0


def _require_stage_file(path: Path, *, stage_name: str) -> None:
    """Return validated stage file."""
    if not path.is_file():
        raise FileNotFoundError(f"{stage_name} did not write {path}")


class PrewarmWorkerPool:
    """Pre-warmed one-shot stage workers: spawn once, single GO, guaranteed reap."""

    def __init__(self) -> None:
        """Initialize PrewarmWorkerPool."""
        self._workers: dict[str, subprocess.Popen[str]] = {}
        self._lock = threading.Lock()
        self._atexit_registered = False

    def spawn(
        self,
        commands: Mapping[str, list[str]],
        env: dict[str, str],
        *,
        active_stages: tuple[str, ...],
    ) -> None:
        """Spawn one worker for every explicitly selected prewarm stage."""
        with self._lock:
            for stage in active_stages:
                if stage not in PREWARM_STAGES:
                    raise ValueError(f"unsupported shape-prior prewarm stage: {stage}")
                if stage in self._workers:
                    continue
                self._workers[stage] = subprocess.Popen(
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
        with self._lock:
            workers = dict(self._workers)
            self._workers.clear()
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

    def pop_and_go(self, stage: str) -> tuple[float, float] | None:
        """Pop the stage's worker (single-use) and signal GO, else None.

        Returns the worker's critical-path time in ms and the GO wall
        timestamp.
        """
        with self._lock:
            worker = self._workers.pop(stage, None)
        if worker is None:
            return None
        if worker.poll() is not None:
            raise RuntimeError(
                f"pre-warmed {stage} worker exited before GO "
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
        object_prompt: str,
        controller_name: str,
        cache_root: str | Path,
        object_id: str | None = None,
        cuda_visible_devices: str = DEFAULT_SHAPE_PRIOR_WARMUP_CUDA_VISIBLE_DEVICES,
        case_name: str = CASE_NAME,
        points_npz: str | Path = POINTS_NPZ,
        sam3d_root: str | Path | None = None,
        sam3d_config: str | Path | None = None,
        sam31_device: str = "cuda",
        volume_sample_size_m: float = DEFAULT_VOLUME_SAMPLE_SIZE_M,
    ) -> None:
        """Initialize ShapePriorLocalClient and resolve the mesh cache."""
        if float(volume_sample_size_m) <= 0.0:
            raise ValueError("volume_sample_size_m must be positive")
        self.volume_sample_size_m = float(volume_sample_size_m)
        self.case_root = Path(case_root)
        self.cuda_visible_devices = str(cuda_visible_devices)
        # object_prompt is the SAM3.1 semantic label; object_id is the cache
        # identity (a specific instance + asset version). They are distinct: the
        # prompt is never a cache key.
        self.object_prompt = require_name(object_prompt, field_name="object_prompt")
        self.controller_name = require_name(
            controller_name,
            field_name="controller_name",
        )
        self.case_name = str(case_name)
        self.points_npz = Path(points_npz)
        self.sam3d_root = None if sam3d_root is None else Path(sam3d_root)
        self.sam3d_config = None if sam3d_config is None else Path(sam3d_config)
        self.sam31_device = str(sam31_device)
        self._prewarm_pool = PrewarmWorkerPool()
        # Resolve the cache before any worker pre-warms (a corrupt entry raises
        # here, before prewarm, so the run fails at startup instead of silently
        # regenerating). cache_root is only touched when a cache is enabled.
        self.object_id = normalize_object_id(object_id)
        self._cache = ShapePriorMeshCache(
            object_id=self.object_id,
            cache_root=cache_root,
        )
        self._cache_resolution = self._cache.resolve()
        self.reuse_sam31_model = not self._cache_resolution.hit

    @property
    def cache_resolution(self) -> mesh_cache.CacheResolution:
        """Return the startup-resolved cache decision for this run."""
        return self._cache_resolution

    @property
    def cache_root(self) -> Path:
        """Return the resolved persistent cache root from configuration."""
        return self._cache.cache_root

    @property
    def requires_generation(self) -> bool:
        """Return whether this run executes upscale, segment, and generate."""
        return not self._cache_resolution.hit

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
            "demo_v6_2.shape_prior.upscale",
            "--img_path",
            str(case / "color" / "0" / "0.png"),
            "--mask_path",
            str(case / "mask" / "0" / "0" / "0.png"),
            "--output_path",
            str(shape_dir / "high_resolution.png"),
            "--category",
            self.object_prompt,
            "--profile-json",
            str(self._stage_profile_path(PREWARM_STAGE_UPSCALE)),
        ]
        generate = [
            sys.executable,
            "-m",
            "demo_v6_2.shape_prior.generate",
            "--img_path",
            str(shape_dir / "masked_image.png"),
            "--output_dir",
            str(shape_dir),
            "--seed",
            str(DEFAULT_GENERATE_SEED),
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
            "demo_v6_2.shape_prior.align",
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

    def _prewarm_stages(self) -> tuple[str, ...]:
        """Return the explicit prewarm-stage set for this run's cache state."""
        if self._cache_resolution.hit:
            return (PREWARM_STAGE_ALIGN,)
        return PREWARM_STAGES

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

        On a cache hit only the align worker is pre-warmed (see
        ``_prewarm_stages``); upscale and generate are skipped entirely.
        """
        self._prewarm_pool.spawn(
            self._stage_commands(),
            self._stage_env(),
            active_stages=self._prewarm_stages(),
        )

    def close(self) -> None:
        """Ask any unused pre-warmed workers to exit and reap them."""
        self._prewarm_pool.close()

    def _run_stage_maybe_prewarmed(
        self,
        stage: str,
        command: list[str],
        *,
        env: dict[str, str],
        prewarmed_stages: list[str],
    ) -> tuple[float, dict[str, Any]]:
        """Run a stage via its pre-warmed worker when present, else cold."""
        prewarmed = self._prewarm_pool.pop_and_go(stage)
        if prewarmed is None:
            stage_ms = _run_stage(command, env=env)
            return stage_ms, {
                "execution_mode": "cold",
                "critical_path_ms": float(stage_ms),
                "go_wall_time_s": None,
            }
        stage_ms, go_wall_time_s = prewarmed
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
            object_name=self.object_prompt,
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
        ) -> Any:
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
            return outcome

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

        def record_skipped_stage(stage: str) -> None:
            """Emit a zero-duration critical-path entry for a cache-hit skip."""
            now_s = time.perf_counter()
            critical_path.append(
                critical_path_entry(
                    stage=stage,
                    path_start_s=request_start_s,
                    stage_start_s=now_s,
                    stage_end_s=now_s,
                    details={"execution_mode": "skipped_cache_hit"},
                )
            )

        resolution = self._cache_resolution
        object_glb_path = paths["shape"] / mesh_cache.MESH_FILENAME
        canonical_mesh_source = "cache" if resolution.hit else "generated"
        canonical_mesh_sha256: str | None = None
        cache_publish_ms = 0.0

        if resolution.hit:
            # Cache hit: skip upscale + second SAM3.1 segment + SAM3D generate,
            # and materialize the canonical mesh from disk into this run's case.
            # The "generate" stage slot records the materialization instead.
            record_skipped_stage(PREWARM_STAGE_UPSCALE)
            record_skipped_stage("segment_image")
            canonical_mesh_sha256 = str(
                record_stage(
                    PREWARM_STAGE_GENERATE,
                    lambda: self._cache.materialize(
                        resolution=resolution, dest_glb=object_glb_path
                    ),
                    lambda sha: {
                        "execution_mode": "cache_hit_materialize",
                        "canonical_mesh_source": "cache",
                        "canonical_mesh_sha256": str(sha),
                        "cache_entry_dir": str(resolution.entry_dir),
                    },
                    require_file=(object_glb_path, "cache materialize"),
                )
            )
        else:
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
                    text_prompt=self.object_prompt,
                    output_path=masked_image_path,
                    device=self.sam31_device,
                    reuse_model=self.reuse_sam31_model,
                ),
                lambda outcome: outcome[1],
                require_file=(masked_image_path, "shape-prior segment"),
            )

            def generate_canonical_mesh() -> dict[str, Any]:
                orchestration = self._run_stage_maybe_prewarmed(
                    PREWARM_STAGE_GENERATE,
                    commands[PREWARM_STAGE_GENERATE],
                    env=env,
                    prewarmed_stages=prewarmed_stages,
                )
                publish_ms = 0.0
                if resolution.enabled:
                    # Publish before align so a later alignment failure does not
                    # force the same canonical mesh to be regenerated.
                    publish_start_s = time.perf_counter()
                    published = self._cache.publish(
                        source_glb=object_glb_path,
                        object_prompt_at_generation=self.object_prompt,
                        generator_seed=DEFAULT_GENERATE_SEED,
                    )
                    publish_ms = elapsed_ms(publish_start_s)
                    mesh_sha256 = str(published["mesh_sha256"])
                else:
                    mesh_cache.validate_mesh_glb(object_glb_path)
                    mesh_sha256 = mesh_cache.sha256_file(object_glb_path)
                return {
                    "orchestration": orchestration[1],
                    "mesh_sha256": mesh_sha256,
                    "cache_publish_ms": publish_ms,
                }

            def generated_mesh_details(outcome: Mapping[str, Any]) -> dict[str, Any]:
                details = self._completed_stage_details(
                    PREWARM_STAGE_GENERATE,
                    orchestration=dict(outcome["orchestration"]),
                )
                details["canonical_mesh"] = {
                    "source": "generated",
                    "mesh_sha256": str(outcome["mesh_sha256"]),
                    "cache_status": str(resolution.status),
                    "cache_publish_ms": float(outcome["cache_publish_ms"]),
                }
                return details

            generated = record_stage(
                PREWARM_STAGE_GENERATE,
                generate_canonical_mesh,
                generated_mesh_details,
                require_file=(object_glb_path, "shape-prior generate"),
            )
            canonical_mesh_sha256 = str(generated["mesh_sha256"])
            cache_publish_ms = float(generated["cache_publish_ms"])

        record_subprocess_stage(PREWARM_STAGE_ALIGN)

        sample_command = [
            sys.executable,
            "-m",
            "demo_v6_2.shape_prior.sample",
            "--base_path",
            str(self.case_root),
            "--case_name",
            self.case_name,
            "--shape_prior",
            "--num_surface_points",
            str(DEFAULT_SURFACE_POINT_COUNT),
            "--volume_sample_size",
            str(self.volume_sample_size_m),
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

        surface = points_array(final_data["surface_points"], name="surface_points")
        interior = points_array(final_data["interior_points"], name="interior_points")
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
        timing_analysis["pre_submit"] = pre_submit_timing(
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
            "shape_prior_object_name": self.object_prompt,
            "shape_prior_object_prompt": self.object_prompt,
            "shape_prior_controller_name": self.controller_name,
            "shape_prior_sam31_device": self.sam31_device,
            "shape_prior_sam31_reuse_model": self.reuse_sam31_model,
            "shape_prior_prewarmed_stages": list(prewarmed_stages),
            "shape_prior_surface_point_count": int(surface.shape[0]),
            "shape_prior_interior_point_count": int(interior.shape[0]),
            "shape_prior_point_count": int(points.shape[0]),
            "shape_prior_cache_enabled": bool(resolution.enabled),
            "shape_prior_cache_hit": (
                None if not resolution.enabled else bool(resolution.hit)
            ),
            "shape_prior_cache_status": str(resolution.status),
            "shape_prior_cache_object_id": resolution.object_id,
            "shape_prior_cache_root": str(self.cache_root),
            "shape_prior_object_prompt_at_generation": (
                str(resolution.manifest["object_prompt_at_generation"])
                if resolution.hit
                else self.object_prompt
            ),
            "shape_prior_cache_entry_dir": (
                str(resolution.entry_dir) if resolution.enabled else None
            ),
            "shape_prior_cache_mesh_sha256": (
                canonical_mesh_sha256 if resolution.enabled else None
            ),
            "shape_prior_canonical_mesh_source": canonical_mesh_source,
            "shape_prior_canonical_mesh_sha256": canonical_mesh_sha256,
            "shape_prior_cache_materialize_ms": (
                stage_ms["generate"] if resolution.hit else 0.0
            ),
            "shape_prior_cache_publish_ms": cache_publish_ms,
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
    dict under the lock. The runtime identity fields enrich the profile
    payload; ``profile_json`` is the optional on-disk mirror path.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        client: ShapePriorLocalClient | None,
        input_source: str,
        depth_backend_label: str,
        depth_source: str,
        profile_json: Path | None,
    ) -> None:
        """Initialize ShapePriorWarmupManager."""
        self.enabled = bool(enabled)
        self.client = client
        self.input_source = str(input_source)
        self.depth_backend_label = str(depth_backend_label)
        self.depth_source = str(depth_source)
        self.profile_json = None if profile_json is None else Path(profile_json)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._result: ShapePriorResult | None = None
        self._profile = default_profile(enabled=self.enabled)
        if client is not None:
            # Cache identity/status is resolved at client construction, so it is
            # known even before the request runs -- surface it up front so a
            # failure profile still carries the cache status and object id.
            resolution = client.cache_resolution
            self._profile.update(
                {
                    "shape_prior_cache_enabled": bool(resolution.enabled),
                    "shape_prior_cache_hit": (
                        None if not resolution.enabled else bool(resolution.hit)
                    ),
                    "shape_prior_cache_status": str(resolution.status),
                    "shape_prior_cache_object_id": resolution.object_id,
                    "shape_prior_cache_root": str(client.cache_root),
                }
            )
        self._warmup_runtime_start_perf_s: float | None = None
        self._ready_perf_s: float | None = None
        self._gate_open_perf_s: float | None = None

    @property
    def requires_sam31_reuse(self) -> bool:
        """Return whether shape-prior generation needs the initial SAM3.1 model."""
        return bool(
            self.enabled and self.client is not None and self.client.requires_generation
        )

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

    def profile_payload(self) -> dict[str, Any]:
        """Return the profile enriched with the runtime identity fields."""
        payload = self.profile()
        if payload.get("input_source") is None:
            payload["input_source"] = self.input_source
        if payload.get("depth_backend") is None:
            payload["depth_backend"] = self.depth_backend_label
        if payload.get("depth_source_internal") is None:
            payload["depth_source_internal"] = self.depth_source
        return payload

    def write_profile_json(self, payload: dict[str, Any] | None = None) -> None:
        """Write the profile payload to ``profile_json`` when configured."""
        if self.profile_json is None:
            return
        data = self.profile_payload() if payload is None else dict(payload)
        atomic_json_dump(data, self.profile_json)


__all__ = [
    "DEFAULT_SHAPE_PRIOR_WARMUP_CUDA_VISIBLE_DEVICES",
    "DEFAULT_SHAPE_PRIOR_TIMEOUT_MS",
    "POINTS_NPZ",
    "STATUS_DISABLED",
    "STATUS_PENDING",
    "STATUS_READY",
    "STATUS_RUNNING",
    "ShapePriorLocalClient",
    "ShapePriorResult",
    "ShapePriorWarmupManager",
    "default_profile",
]
