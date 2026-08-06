"""demo_v7 shape-prior generate-backend clients (v6.2 stays untouched).

``ShapePriorLocalClient._stage_commands()`` is the single source of every
stage subprocess argv (both the prewarm pool spawn and the cold fallback), so
swapping ONLY the generate entry re-backends the chain while upscale /
segment / align / sample — and every contract around them — run the exact
v6.2 code.

Generate-stage contract the replacement honors (see warmup.py):

- reads ``<case>/shape/masked_image.png`` (RGBA, alpha = object mask);
- writes ``<case>/shape/object.glb`` (trimesh-loadable; align re-solves
  pose+scale so TRELLIS.2's [-0.5,0.5]^3 box is fine);
- writes the schema-v1 COMPLETED profile at ``shape/timing/generate.json``
  with the execution mode the parent orchestrated (cold vs prewarmed);
- when prewarmed (the pool appends ``--wait-signal``) it implements the
  stdin WAITING/GO handshake — the runner reuses the v6.2 ``StageProfileRun``
  for exact lifecycle fidelity.
"""

from __future__ import annotations

from pathlib import Path

from demo_v6_2.shape_prior import warmup as shape_prior_warmup

from demo_v7.service.backend_options import (
    BACKEND_SAM3D,
    BACKEND_TRELLIS2,
    TRELLIS2_PYTHON,
    TRELLIS2_REPO,
    ensure_trellis2_available,
    normalize_backend,
)

TRELLIS2_RUNNER = Path(__file__).resolve().parent / "trellis2_generate.py"


class Trellis2ShapePriorClient(shape_prior_warmup.ShapePriorLocalClient):
    """v6.2 chain with the generate stage re-pointed at TRELLIS.2.

    The runner executes under the trellis2 conda env's absolute python;
    the shared ``_stage_env()`` stays untouched (its CUDA_VISIBLE_DEVICES
    pins the warmup GPU for TRELLIS.2 too, and its repo-root PYTHONPATH is
    what lets the runner import the v6.2 timing contract) — the runner
    itself front-loads the TRELLIS.2 checkout onto ``sys.path`` so those
    repo entries never shadow it.
    """

    def _stage_commands(self) -> dict[str, list[str]]:
        commands = dict(super()._stage_commands())
        case = Path(self.case_root) / self.case_name
        shape_dir = case / "shape"
        commands[shape_prior_warmup.PREWARM_STAGE_GENERATE] = [
            str(TRELLIS2_PYTHON),
            str(TRELLIS2_RUNNER),
            "--img_path",
            str(shape_dir / "masked_image.png"),
            "--output_dir",
            str(shape_dir),
            "--trellis2-repo",
            str(TRELLIS2_REPO),
            "--seed",
            str(shape_prior_warmup.DEFAULT_GENERATE_SEED),
            "--skip-visualization",
            "--profile-json",
            str(
                self._stage_profile_path(shape_prior_warmup.PREWARM_STAGE_GENERATE)
            ),
        ]
        return commands


def create_shape_prior_client(
    backend: str | None, **kwargs
) -> shape_prior_warmup.ShapePriorLocalClient:
    """Construct the client for ``backend`` (sam3d/trellis2; none has none).

    ``kwargs`` are the unchanged ``ShapePriorLocalClient`` constructor
    arguments (the staged runtime passes the same set for every backend).
    """
    resolved = normalize_backend(backend)
    if resolved == BACKEND_SAM3D:
        return shape_prior_warmup.ShapePriorLocalClient(**kwargs)
    if resolved == BACKEND_TRELLIS2:
        ensure_trellis2_available()
        return Trellis2ShapePriorClient(**kwargs)
    raise ValueError(f"backend {resolved!r} does not use a shape-prior client")
