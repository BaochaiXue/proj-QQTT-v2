"""mesh_surface gaussian manager: derive splats from the aligned world mesh.

Drop-in lifecycle twin of ``GaussianManager`` (same public surface, selected
by the staged runtime on ``gaussian_backend == mesh_surface``), but with no
worker subprocess and no alignment chain at all: the TRELLIS.2 mesh is the
single geometry truth, so once the shape-prior chain is READY the aligned
``shape/matching/final_mesh.glb`` (world frame, ARAP-refined, textured) is
deterministically gaussianized — every splat hard-bound to a triangle via
face_id + barycentric anchors. Registration, ICP, ARAP residual transfer,
floater pruning and the self-align upgrade are all structurally unnecessary
here: there is no second geometry to reconcile.

Artifacts (same names the GUI's Gaussian tab consumes):
- ``gaussian_world.ply``      world-frame splats (the live renderer input)
- ``gaussian_anchors.npz``    face_id + barycentric + rest topology + hash
- ``gaussian_world_overlay.png`` frame-0 camera overlay still
- ``gaussian_provenance.json``  backend/seed/splat count + the measured
  center-to-mesh replay error (numerically ~0 by construction; recorded so
  the hard-binding claim is auditable per run)

Every failure is display-only (EVT_ERROR + feature self-disable), matching
the TripoSplat manager's fail-soft contract.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Callable

from demo_v7.service import gaussian_options


class MeshSurfaceGaussianManager:
    """One run's mesh-surface gaussian derivation (no subprocess)."""

    def __init__(
        self,
        *,
        case_dir: Path,
        out_dir: Path,
        emit_progress: Callable[..., None],
        emit_artifacts: Callable[[str, dict[str, str]], None],
        emit_error: Callable[[str, str], None],
        seed: int = gaussian_options.DEFAULT_GAUSSIAN_SEED,
        target_splats: int | None = None,
    ) -> None:
        self.case_dir = Path(case_dir)
        self.out_dir = Path(out_dir)
        self._emit_progress = emit_progress
        self._emit_artifacts = emit_artifacts
        self._emit_error = emit_error
        self.seed = int(seed)
        if target_splats is None:
            from demo_v7.service.mesh_surface_gaussian import (
                DEFAULT_TARGET_SPLATS,
            )

            target_splats = DEFAULT_TARGET_SPLATS
        self.target_splats = int(target_splats)
        self._lock = threading.Lock()
        self._busy = False
        self._closed = False
        self._case_ready = threading.Event()
        self._first_gen: threading.Thread | None = None
        self.world_ply_path = self.out_dir / "gaussian_world.ply"
        self.anchors_path = self.out_dir / "gaussian_anchors.npz"
        self.mesh_path = self.case_dir / "shape" / "matching" / "final_mesh.glb"

    # -- lifecycle (GaussianManager-compatible surface) ----------------------

    def start(self) -> None:
        """No subprocess to spawn; just claim the output dir."""
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            self._emit_error("gaussian", f"mesh_surface out_dir: {exc}")
            return
        self._emit_progress(
            "gaussian", "mesh_surface 后端:等待 shape prior 对齐 mesh…"
        )

    def notify_submitted(self) -> None:
        """No masked-image dependency — generation waits on chain READY."""

    def notify_case_ready(self) -> None:
        """Chain READY: final_mesh.glb is settled — derive the splats."""
        self._case_ready.set()
        with self._lock:
            if self._closed or self._first_gen is not None:
                return
            self._first_gen = threading.Thread(
                target=lambda: self._generate(self.seed),
                name="gaussian-mesh-surface",
                daemon=True,
            )
        self._first_gen.start()

    def shutdown(self) -> None:
        with self._lock:
            self._closed = True

    # -- commands ------------------------------------------------------------

    def regenerate(self, seed: int | None) -> bool:
        """Re-derive with a new sampling seed (REVIEW 拣选 re-roll)."""
        with self._lock:
            if self._closed or self._busy or not self._case_ready.is_set():
                return False
        if seed is None:
            seed = (self.seed + int(time.time())) % 1_000_000 or 1
        threading.Thread(
            target=lambda: self._generate(int(seed)),
            name="gaussian-mesh-surface-regen",
            daemon=True,
        ).start()
        return True

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._busy

    def has_world_ply(self) -> bool:
        return self.world_ply_path.is_file() and self.anchors_path.is_file()

    # -- derivation ----------------------------------------------------------

    def _generate(self, seed: int) -> None:
        with self._lock:
            if self._closed or self._busy:
                return
            self._busy = True
        try:
            self.seed = int(seed)
            started_s = time.perf_counter()
            if not self.mesh_path.is_file():
                raise FileNotFoundError(
                    f"aligned mesh missing: {self.mesh_path}"
                )
            self._emit_progress(
                "gaussian",
                f"从对齐 mesh 派生表面高斯(seed={self.seed}, "
                f"目标 {self.target_splats} splats)…",
            )
            artifacts, num_splats = self._derive_and_collect()
            self._emit_artifacts("gaussian", artifacts)
            self._emit_progress(
                "gaussian",
                f"gaussian 就绪(mesh_surface, seed={self.seed}, "
                f"{num_splats} splats, "
                f"{time.perf_counter() - started_s:.1f}s)",
            )
        except Exception as exc:
            self._emit_error("gaussian", f"{type(exc).__name__}: {exc}")
            self._emit_progress(
                "gaussian", f"mesh_surface 派生失败: {exc}", ok=False
            )
        finally:
            with self._lock:
                self._busy = False

    def _derive_and_collect(self) -> tuple[dict[str, str], int]:
        import numpy as np

        from demo_v7.service.gaussian_manager import render_world_overlay
        from demo_v7.service.gaussian_utils import save_gaussian_ply
        from demo_v7.service.mesh_surface_gaussian import (
            gaussianize_mesh,
            replay_splat_means,
            save_anchors,
        )

        splats, anchors = gaussianize_mesh(
            self.mesh_path, target_splats=self.target_splats, seed=self.seed
        )
        # Hard-binding audit: replaying the anchors must reproduce the splat
        # centers (float32-exactly). Recorded in provenance every run.
        replayed = replay_splat_means(
            anchors.rest_vertices.astype(np.float64),
            anchors.faces.astype(np.int64),
            anchors.face_index,
            anchors.barycentric.astype(np.float64),
        )
        center_err_m = float(
            np.linalg.norm(replayed - splats.means, axis=1).max()
        )
        if center_err_m > 1e-4:
            raise ValueError(
                f"anchor replay error {center_err_m:.6f} m — binding broken"
            )
        save_anchors(self.anchors_path, anchors)
        save_gaussian_ply(self.world_ply_path, splats)

        self._emit_progress("gaussian", "渲染世界系叠加图…")
        overlay_path = render_world_overlay(self.case_dir, self.out_dir, splats)

        provenance = {
            "backend": gaussian_options.GAUSSIAN_MESH_SURFACE,
            "mesh": str(self.mesh_path),
            "seed": self.seed,
            "target_splats": self.target_splats,
            "num_splats": len(splats),
            "topology_sha256": anchors.topology_sha256,
            "center_to_mesh_replay_max_m": center_err_m,
            "alignment": {"method": "mesh_surface"},
        }
        provenance_path = self.out_dir / "gaussian_provenance.json"
        provenance_path.write_text(json.dumps(provenance, indent=1))
        artifacts = {
            "world_overlay": str(overlay_path),
            "world_ply": str(self.world_ply_path),
            "anchors": str(self.anchors_path),
            "provenance": str(provenance_path),
        }
        return artifacts, len(splats)
