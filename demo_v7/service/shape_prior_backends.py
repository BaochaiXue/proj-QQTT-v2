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
SAMPLE_ASAP_SAFE_RUNNER = Path(__file__).resolve().parent / "sample_asap_safe.py"
UPSCALE_PASSTHROUGH_RUNNER = (
    Path(__file__).resolve().parent / "upscale_passthrough.py"
)


class _NoUpscaleStageMixin:
    """Swap the upscale stage for the crop-only passthrough runner.

    Same interpreter, same CLI flags, same WAITING/GO lifecycle and profile
    schema — ``high_resolution.png`` just holds the original-resolution
    mask-bbox crop, so the untouched SAM3.1 segment + generate + align +
    sample chain runs unchanged on it. Composes with any client class whose
    ``_stage_commands`` keeps the v6.2 upscale argv shape.
    """

    def _stage_commands(self) -> dict[str, list[str]]:
        commands = dict(super()._stage_commands())  # type: ignore[misc]
        upscale = list(commands[shape_prior_warmup.PREWARM_STAGE_UPSCALE])
        assert upscale[1:3] == ["-m", "demo_v6_2.shape_prior.upscale"], upscale
        commands[shape_prior_warmup.PREWARM_STAGE_UPSCALE] = [
            upscale[0],
            str(UPSCALE_PASSTHROUGH_RUNNER),
            *upscale[3:],
        ]
        return commands


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
        # Sample runs via the zero-extent cleanup wrapper (same CLI, same
        # interpreter, same GO protocol): align's final ARAP leaves a few
        # hundred exactly-zero-extent collapsed faces in final_mesh.glb, and
        # on the TRELLIS.2 topology those make the downstream ASAP
        # deformation's solver fail to factorize (see sample_asap_safe.py).
        sample = list(commands[shape_prior_warmup.PREWARM_STAGE_SAMPLE])
        assert sample[1:3] == ["-m", "demo_v6_2.shape_prior.sample"], sample
        commands[shape_prior_warmup.PREWARM_STAGE_SAMPLE] = [
            sample[0],
            str(SAMPLE_ASAP_SAFE_RUNNER),
            *sample[3:],
        ]
        return commands


class NoUpscaleShapePriorClient(
    _NoUpscaleStageMixin, shape_prior_warmup.ShapePriorLocalClient
):
    """sam3d chain with the upscale stage swapped for the crop passthrough."""


class NoUpscaleTrellis2ShapePriorClient(
    _NoUpscaleStageMixin, Trellis2ShapePriorClient
):
    """trellis2 chain with the upscale stage swapped for the passthrough."""


def create_shape_prior_client(
    backend: str | None, *, use_upscale: bool = True, **kwargs
) -> shape_prior_warmup.ShapePriorLocalClient:
    """Construct the client for ``backend`` (sam3d/trellis2; none has none).

    ``use_upscale`` False swaps ONLY the upscale stage for the crop-only
    passthrough (GUI 上采样 toggle); ``kwargs`` are the unchanged
    ``ShapePriorLocalClient`` constructor arguments (the staged runtime
    passes the same set for every backend).
    """
    resolved = normalize_backend(backend)
    if resolved == BACKEND_SAM3D:
        if use_upscale:
            return shape_prior_warmup.ShapePriorLocalClient(**kwargs)
        return NoUpscaleShapePriorClient(**kwargs)
    if resolved == BACKEND_TRELLIS2:
        ensure_trellis2_available()
        if use_upscale:
            return Trellis2ShapePriorClient(**kwargs)
        return NoUpscaleTrellis2ShapePriorClient(**kwargs)
    raise ValueError(f"backend {resolved!r} does not use a shape-prior client")
