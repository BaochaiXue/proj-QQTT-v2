"""Gaussian-splats feature manager: worker lifecycle + generate/align flow.

Owned by the staged runtime. Lifecycle:

- ``start()`` after the shape-prior chain is READY (the align chain's
  best_match.pkl must exist for the world transform): spawns the persistent
  ``triposplat_worker`` subprocess on the shape-prior GPU and queues the
  first generation.
- ``regenerate(seed)`` from CMD_REGEN_GAUSSIAN (REVIEW 拣选/换seed): one
  in-flight generation at a time; a re-roll replaces the artifacts.
- ``shutdown()`` before FORMAL launches: the worker exits and frees the
  shape-prior GPU for the PhysTwin children; the aligned world ply stays on
  disk for the live renderer.

After each worker "done": the canonical->world alignment runs in the
manager thread (CPU: registration + mesh2world replay + ARAP residual),
the aligned overlay still is rendered (service GPU, milliseconds), and the
GUI gets EVT_ARTIFACTS kind=gaussian. Every failure is display-only: the
feature disables itself loudly (EVT_ERROR) and never touches the pipeline.
"""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

from demo_v7.service import gaussian_options


class GaussianManager:
    """One run's gaussian generation + alignment orchestration."""

    def __init__(
        self,
        *,
        case_dir: Path,
        out_dir: Path,
        controller_name: str,
        cuda_visible_devices: str,
        emit_progress: Callable[..., None],
        emit_artifacts: Callable[[str, dict[str, str]], None],
        emit_error: Callable[[str, str], None],
        seed: int = gaussian_options.DEFAULT_GAUSSIAN_SEED,
        num_gaussians: int = gaussian_options.DEFAULT_NUM_GAUSSIANS,
        steps: int = gaussian_options.DEFAULT_GAUSSIAN_STEPS,
    ) -> None:
        self.case_dir = Path(case_dir)
        self.out_dir = Path(out_dir)
        self.controller_name = str(controller_name)
        self.cuda_visible_devices = str(cuda_visible_devices)
        self._emit_progress = emit_progress
        self._emit_artifacts = emit_artifacts
        self._emit_error = emit_error
        self.seed = int(seed)
        self.num_gaussians = int(num_gaussians)
        self.steps = int(steps)
        self._proc: subprocess.Popen[str] | None = None
        self._reader: threading.Thread | None = None
        self._lock = threading.Lock()
        self._busy = False
        self._closed = False
        self.world_ply_path = self.out_dir / "gaussian_world.ply"

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        """Spawn the worker and queue the first generation (non-blocking)."""
        try:
            gaussian_options.ensure_triposplat_available()
            self.out_dir.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = self.cuda_visible_devices
            worker = Path(__file__).resolve().parent / "triposplat_worker.py"
            self._proc = subprocess.Popen(
                [
                    sys.executable,
                    str(worker),
                    "--triposplat-repo",
                    str(gaussian_options.TRIPOSPLAT_REPO),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=None,  # worker logs ride the service's stderr
                text=True,
                env=env,
            )
        except Exception as exc:
            self._emit_error("gaussian", f"worker spawn failed: {exc}")
            return
        self._reader = threading.Thread(
            target=self._reader_loop, name="gaussian-worker-reader", daemon=True
        )
        self._reader.start()
        self.regenerate(self.seed)

    def shutdown(self) -> None:
        """Ask the worker to exit (idempotent; bounded wait then kill)."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            proc = self._proc
        if proc is None:
            return
        try:
            if proc.stdin is not None:
                proc.stdin.write(json.dumps({"cmd": "exit"}) + "\n")
                proc.stdin.flush()
                proc.stdin.close()
        except Exception:
            pass
        try:
            proc.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            proc.kill()

    # -- commands ------------------------------------------------------------

    def regenerate(self, seed: int | None) -> bool:
        """Queue one generation; False when busy/closed (caller acks that)."""
        with self._lock:
            if self._closed or self._proc is None or self._proc.stdin is None:
                return False
            if self._busy:
                return False
            self._busy = True
            if seed is None:
                # A fresh die roll that never repeats the current seed.
                seed = (self.seed + int(time.time())) % 1_000_000 or 1
            self.seed = int(seed)
            request = {
                "cmd": "generate",
                "image": str(self.case_dir / "shape" / "masked_image.png"),
                "out_dir": str(self.out_dir),
                "seed": self.seed,
                "num_gaussians": self.num_gaussians,
                "steps": self.steps,
            }
            try:
                self._proc.stdin.write(json.dumps(request) + "\n")
                self._proc.stdin.flush()
            except Exception as exc:
                self._busy = False
                self._emit_error("gaussian", f"worker request failed: {exc}")
                return False
        self._emit_progress(
            "gaussian", f"生成中(seed={self.seed}, {self.num_gaussians} splats)"
        )
        return True

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._busy

    def has_world_ply(self) -> bool:
        return self.world_ply_path.is_file()

    # -- worker events -------------------------------------------------------

    def _reader_loop(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            try:
                self._on_worker_event(event)
            except Exception as exc:
                with self._lock:
                    self._busy = False
                self._emit_error("gaussian", f"{type(exc).__name__}: {exc}")
        # EOF: worker exited. Normal after shutdown(); an error mid-run
        # already produced an error event above.

    def _on_worker_event(self, event: dict[str, Any]) -> None:
        kind = event.get("event")
        if kind == "ready":
            self._emit_progress(
                "gaussian", f"TripoSplat 就绪({event.get('load_s', '?')}s)"
            )
        elif kind == "progress":
            step, total = int(event.get("step", 0)), int(event.get("total", 1))
            if step in (1, total // 2, total):
                self._emit_progress("gaussian", f"采样 {step}/{total}")
        elif kind == "error":
            with self._lock:
                self._busy = False
            self._emit_error("gaussian", str(event.get("message")))
            # Mirror into the Gaussian tab's status line (ok=False).
            self._emit_progress("gaussian", str(event.get("message")), ok=False)
        elif kind == "done":
            try:
                artifacts = self._align_and_collect(event)
                self._emit_artifacts("gaussian", artifacts)
                self._emit_progress(
                    "gaussian",
                    f"gaussian 就绪(seed={event.get('seed')}, "
                    f"{event.get('num_splats')} splats, "
                    f"{event.get('generation_s')}s)",
                )
            finally:
                with self._lock:
                    self._busy = False

    # -- alignment -----------------------------------------------------------

    def _align_and_collect(self, done_event: dict[str, Any]) -> dict[str, str]:
        """World-align the fresh canonical ply + render the overlay still."""
        import cv2
        import numpy as np

        from demo_v7.service.gaussian_align import align_gaussian_to_world
        from demo_v7.service.gaussian_utils import (
            load_gaussian_ply,
            render_gaussians,
            save_gaussian_ply,
        )

        ply_path = Path(done_event["ply"])
        splats = load_gaussian_ply(ply_path)
        world, alignment = align_gaussian_to_world(
            splats, self.case_dir, self.controller_name
        )
        save_gaussian_ply(self.world_ply_path, world)

        with open(self.case_dir / "calibrate.pkl", "rb") as handle:
            c2w = np.asarray(pickle.load(handle)[0], dtype=np.float64)
        intrinsics = np.asarray(
            json.loads((self.case_dir / "metadata.json").read_text())["intrinsics"]
        )[0]
        frame_bgr = cv2.imread(str(self.case_dir / "color" / "0" / "0.png"))
        height, width = frame_bgr.shape[:2]
        rgb, alpha = render_gaussians(
            world,
            viewmat=np.linalg.inv(c2w),
            intrinsics=intrinsics,
            width=width,
            height=height,
            background=(0.0, 0.0, 0.0),
        )
        blend = alpha[..., None] * 0.65
        overlay = (
            frame_bgr.astype(np.float32) * (1 - blend)
            + rgb[..., ::-1].astype(np.float32) * blend
        ).astype(np.uint8)
        overlay_path = self.out_dir / "gaussian_world_overlay.png"
        cv2.imwrite(str(overlay_path), overlay)

        provenance_path = Path(done_event["provenance"])
        provenance = json.loads(provenance_path.read_text())
        provenance["alignment"] = alignment.provenance()
        provenance_path.write_text(json.dumps(provenance, indent=1))

        return {
            "turntable": str(done_event["contact_sheet"]),
            "prepared": str(done_event["prepared"]),
            "world_overlay": str(overlay_path),
            "ply": str(ply_path),
            "world_ply": str(self.world_ply_path),
            "provenance": str(provenance_path),
        }
