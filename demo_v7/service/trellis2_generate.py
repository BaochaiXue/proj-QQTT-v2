"""TRELLIS.2 generate-stage CLI: shape/masked_image.png -> shape/object.glb.

Drop-in replacement for ``demo_v6_2.shape_prior.generate`` (same CLI surface,
same StageProfileRun WAITING/GO/COMPLETED lifecycle, same output contract) so
the untouched v6.2 align/sample stages run unchanged afterwards. It is meant
to run under the ``trellis2`` conda env python — the parent-side argv is built
by ``demo_v7.service.shape_prior_backends.Trellis2ShapePriorClient``.

Environment landmines handled here (memory: trellis2-integration):

- the ``trellis2`` package is repo-local, never pip-installed -> the checkout
  passed via ``--trellis2-repo`` is prepended to ``sys.path`` before any
  heavy import;
- ``briaai/RMBG-2.0`` (rembg) is a gated HF repo -> stubbed out before
  ``from_pretrained``; our input is always an RGBA masked image, for which
  ``preprocess_image`` never calls rembg;
- CUDA_HOME must point at the env's own cuda-toolkit 12.4 (system has only
  12.8/13.x) for any first-use nvdiffrast JIT build during texture baking;
- align renders the glb through pytorch3d's experimental GLB loader, which
  cannot decode EXT_texture_webp -> PNG textures (``extension_webp=False``).
"""

import time


_MODULE_IMPORT_START_S = time.perf_counter()


import os  # noqa: E402
import sys  # noqa: E402
from argparse import ArgumentParser  # noqa: E402
from pathlib import Path  # noqa: E402


# align's candidate rendering + ARAP are tuned for SAM3D-scale meshes (the
# sloth reference: 5.6k verts / 8.6k faces); TRELLIS.2's native ~490k faces
# would be ~57x that, so decimate to the same order of magnitude.
DECIMATION_TARGET_FACES = 16000
TEXTURE_SIZE = 2048
DEFAULT_SEED = 42
_ACTIVE_TIMING_FIELDS = (
    "module_import_ms",
    "pre_go_prepare_ms",
    "model_load_ms",
    "bulk_cuda_ms",
    "input_decode_ms",
    "pipeline_run_ms",
    "mesh_export_ms",
    # SAM3D-only fields kept at 0.0 so generate.json stays field-uniform
    # across backends for timeline consumers.
    "gaussian_export_ms",
    "visualization_export_ms",
)


def build_parser():
    """Build the command-line argument parser (generate.py CLI mirror)."""
    parser = ArgumentParser(
        description="Generate shape prior via TRELLIS.2 (demo_v7 backend)."
    )
    parser.add_argument("--img_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--trellis2-repo",
        type=str,
        required=True,
        help="TRELLIS.2 checkout (repo-local `trellis2` package).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="microsoft/TRELLIS.2-4B",
        help="HF pipeline id (must already be in the local HF cache).",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--profile-json", type=Path, default=None)
    parser.add_argument(
        "--skip-visualization",
        action="store_true",
        help="Accepted for generate.py argv parity; TRELLIS.2 emits no video.",
    )
    parser.add_argument(
        "--wait-signal",
        dest="wait_signal",
        action="store_true",
        help="Load the pipeline to CPU RAM, then block on stdin for GO.",
    )
    return parser


def _bootstrap_paths(trellis2_repo):
    """Front-load sys.path + CUDA env before any torch/trellis2 import.

    Must win over the parent's _stage_env PYTHONPATH (repo_root +
    repo_root/demo_v6_2 come first there and would otherwise shadow module
    names inside the TRELLIS.2 checkout).
    """
    repo = Path(trellis2_repo).expanduser().resolve()
    if not (repo / "trellis2" / "__init__.py").is_file():
        raise FileNotFoundError(f"not a TRELLIS.2 checkout: {repo}")
    repo_str = str(repo)
    if repo_str in sys.path:
        sys.path.remove(repo_str)
    sys.path.insert(0, repo_str)
    # The env's own cuda-toolkit (12.4) for potential nvdiffrast JIT builds.
    # FORCED, not setdefault: the parent forwards the ambient shell env, and
    # this machine's ambient CUDA_HOME points at the system toolkit (13.x) —
    # a JIT build against it would mismatch torch cu124.
    env_prefix = Path(sys.executable).resolve().parents[1]
    if (env_prefix / "bin" / "nvcc").is_file():
        os.environ["CUDA_HOME"] = str(env_prefix)
        os.environ["PATH"] = os.pathsep.join(
            [str(env_prefix / "bin"), os.environ.get("PATH", "")]
        )
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # The checkpoint contract is cache-only (backend_options fail-fasts on a
    # missing snapshot); offline mode also kills per-boot etag revalidation
    # round-trips. setdefault keeps an operator override authoritative.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")


def _stub_gated_rembg():
    """Disable the gated briaai/RMBG-2.0 background-removal model.

    ``preprocess_image`` uses the input's own alpha channel whenever it is
    RGBA with a real mask (our case, always), so the stub is never invoked;
    it exists purely so ``from_pretrained`` does not 403 on the gated repo.
    """
    from trellis2.pipelines import rembg as rembg_mod

    class _NoRembg:
        def __init__(self, *args, **kwargs):
            pass

        def to(self, *args, **kwargs):
            return self

        def cpu(self):
            return self

        def __call__(self, image):
            raise RuntimeError(
                "rembg is disabled (gated HF repo); the input must be an "
                "RGBA masked image"
            )

    rembg_mod.BiRefNet = _NoRembg


from contextlib import contextmanager


@contextmanager
def _init_skipped():
    """No-op torch's RANDOM weight initializers for the enclosed scope.

    Measured (2026-08-06): 57.3s of the from_pretrained wall time is the CPU
    random init of five 1.3B DiTs — values that load_state_dict immediately
    overwrites. Only the random fillers are patched; deterministic ones
    (zeros_/ones_/constant_) stay live for buffers. Bit-safety is enforced
    separately by the checkpoint-coverage assertion in
    ``_patch_component_loader`` — a parameter the checkpoint does not cover
    would keep its (now skipped) init, so that case raises instead.
    """
    import torch

    names = (
        "uniform_",
        "normal_",
        "trunc_normal_",
        "kaiming_uniform_",
        "kaiming_normal_",
        "xavier_uniform_",
        "xavier_normal_",
        "orthogonal_",
    )
    saved = {name: getattr(torch.nn.init, name) for name in names}

    def _noop(tensor, *args, **kwargs):
        return tensor

    try:
        for name in names:
            setattr(torch.nn.init, name, _noop)
        yield
    finally:
        for name, fn in saved.items():
            setattr(torch.nn.init, name, fn)


def _patch_component_loader():
    """Wrap trellis2.models.from_pretrained: skip init + assert coverage.

    Scope: the pipeline base class loads ONLY the trellis2 component models
    (DiTs/VAEs) through this function; DINOv3 and rembg are constructed
    afterwards by the image-to-3d subclass, so transformers' own missing-key
    init paths are never patched. Coverage assertion: every state_dict entry
    of the constructed model must exist in the loaded checkpoint (captured
    via safetensors load_file), except ``rope_phases`` — a buffer computed
    in __init__ (sparse_structure_flow.py) untouched by nn.init. Anything
    else missing means the skipped init would actually matter: raise loudly
    rather than silently degrade.
    """
    if os.environ.get("DEMO_V7_T2_FAST", "1") == "0":
        return
    import trellis2.models as t2_models

    original = t2_models.from_pretrained
    if getattr(original, "_v7_skip_init", False):
        return

    def _from_pretrained_skip_init(path, **kwargs):
        import safetensors.torch as st

        captured = {}
        real_load_file = st.load_file

        def _capturing_load_file(file, *args, **load_kwargs):
            state = real_load_file(file, *args, **load_kwargs)
            captured["keys"] = set(state.keys())
            return state

        st.load_file = _capturing_load_file
        try:
            with _init_skipped():
                model = original(path, **kwargs)
        finally:
            st.load_file = real_load_file
        checkpoint_keys = captured.get("keys", set())
        missing = [
            key
            for key in model.state_dict()
            if key not in checkpoint_keys and "rope_phases" not in key
        ]
        if missing:
            raise RuntimeError(
                f"skip-init loaded {path} but the checkpoint does not cover "
                f"{len(missing)} tensors (e.g. {missing[:5]}); their random "
                "init was skipped — refusing to run with undefined weights"
            )
        return model

    _from_pretrained_skip_init._v7_skip_init = True
    t2_models.from_pretrained = _from_pretrained_skip_init


def _load_pipeline(model_id):
    """from_pretrained to CPU RAM only (VRAM stays untouched until run)."""
    _stub_gated_rembg()
    _patch_component_loader()
    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(model_id)
    # low_vram mode (checkpoint default) stages models on/off the GPU per
    # pipeline step; .cuda() only records the target device.
    pipeline.cuda()
    return pipeline


# Bulk residency needs the full weight set (~9G fp16) plus the run's
# activation peak with headroom for cublasLt workspaces; below this free-VRAM
# line the staged low_vram path is the safe choice (measured: a crowded GPU
# fails inside cublasLt heuristics, not with a clean OOM).
_BULK_CUDA_MIN_FREE_BYTES = 14 * 1024**3


def _bulk_cuda(pipeline, timing_ms):
    """Move the whole pipeline to the GPU once (drop low_vram staging).

    low_vram=True (checkpoint default) round-trips each model over PCIe per
    pipeline step, serialized with compute. One bulk transfer after GO is
    strictly the same math on the same device — only the staging disappears,
    so the output is unchanged either way. Adaptive: on a crowded GPU
    (foreign processes; the shape-prior GPU is contested on this box) the
    staged path is kept — robustness over the ~5s win. Must not run pre-GO:
    the WAITING contract keeps VRAM free for the upscale stage's peak.
    """
    if os.environ.get("DEMO_V7_T2_FAST", "1") == "0":
        return
    import torch

    bulk_start_s = time.perf_counter()
    free_bytes, _total = torch.cuda.mem_get_info()
    if free_bytes < _BULK_CUDA_MIN_FREE_BYTES:
        print(
            f"[trellis2-generate] {free_bytes / 1024**3:.1f}G free VRAM < "
            f"{_BULK_CUDA_MIN_FREE_BYTES / 1024**3:.0f}G: keeping low_vram "
            "staging (same output, slower)",
            flush=True,
        )
        return
    pipeline.low_vram = False
    pipeline.cuda()
    timing_ms["bulk_cuda_ms"] = _elapsed_ms(bulk_start_s)


def _arap_safe_face_mask(vertices, faces):
    """Faces that survive align's exact-weld + ARAP (True = keep).

    align.py builds its ARAP mesh via o3d ``remove_duplicated_vertices()``
    (exact position weld). A face whose corners collapse to duplicate
    indices under that weld — or whose area is ~0 (collinear corners) —
    yields nan/inf cotangent weights and the ARAP solver fails to factorize
    ("Failed to build solver"). o_voxel's atlas export produces a handful of
    such slivers along UV seams; SAM3D's exporter never does, which is why
    the unchanged align stage only breaks on this backend.
    """
    import numpy as np

    verts = np.asarray(vertices, dtype=np.float64)
    tris = np.asarray(faces, dtype=np.int64)
    _, weld = np.unique(verts, axis=0, return_inverse=True)
    welded = weld[tris]
    distinct = (
        (welded[:, 0] != welded[:, 1])
        & (welded[:, 1] != welded[:, 2])
        & (welded[:, 0] != welded[:, 2])
    )
    corners = verts[tris]
    areas = 0.5 * np.linalg.norm(
        np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]),
        axis=1,
    )
    return distinct & (areas > 1e-12)


def _validated_rgba(img_path):
    """Open the masked image, enforcing the RGBA-with-mask input contract."""
    import numpy as np
    from PIL import Image

    image = Image.open(img_path).convert("RGBA")
    alpha = np.asarray(image)[:, :, 3]
    if bool(np.all(alpha == 255)):
        # Same guard as generate.py's rgba_to_sam3d_inputs: without a real
        # alpha mask the (stubbed) rembg path would be taken.
        raise ValueError("Image must contain an alpha foreground mask.")
    return image


def _pin_flex_gemm_autotune():
    """Pin the texture-bake grid_sample kernel to the known 4090-best config.

    flex_gemm autotunes with the exact masked-texel count in the key, which
    is unique per object — the persistent cache never hits and EVERY run
    re-benchmarks 12 tile configs (~1.55s inside mesh_export). The kernel's
    output is tile-size independent (verified bit-identical across configs);
    with a single config Triton skips benchmarking entirely. Fail-soft: if
    flex_gemm internals moved, keep stock autotune.
    """
    if os.environ.get("DEMO_V7_T2_FAST", "1") == "0":
        return
    try:
        import triton
        from flex_gemm.kernels.triton.grid_sample import (
            indice_weighed_sum_fwd as fwd_mod,
        )

        fwd_mod.indice_weighed_sum_fwd_kernel.configs = [
            triton.Config({"BM": 16, "BK": 8}, num_warps=2)
        ]
    except Exception as exc:
        print(
            f"[trellis2-generate] flex_gemm autotune pin skipped: {exc}",
            flush=True,
        )


@contextmanager
def _fast_png_encode():
    """PNG compress_level 1 for the glb export (trimesh hardcodes level 6).

    zlib level only trades bytes for time on a LOSSLESS stream — decoded
    pixels are bit-identical; the glb grows ~1MB and the export saves ~1.8s.
    Scoped: PIL's default is restored right after the export.
    """
    if os.environ.get("DEMO_V7_T2_FAST", "1") == "0":
        yield
        return
    from PIL import Image

    real_save = Image.Image.save

    def _save_fast_png(self, fp, format=None, **params):
        if str(format or "").lower() == "png" and "compress_level" not in params:
            params["compress_level"] = 1
        return real_save(self, fp, format=format, **params)

    Image.Image.save = _save_fast_png
    try:
        yield
    finally:
        Image.Image.save = real_save


def run_trellis2_shape_prior(args, pipeline=None, *, timing_ms=None):
    """Generate + bake + export object.glb; returns the active timings."""
    timings = {} if timing_ms is None else timing_ms
    if pipeline is None:
        prepare_start_s = time.perf_counter()
        _bootstrap_paths(args.trellis2_repo)
        timings["pre_go_prepare_ms"] = _elapsed_ms(prepare_start_s)

        model_load_start_s = time.perf_counter()
        pipeline = _load_pipeline(args.model)
        timings["model_load_ms"] = _elapsed_ms(model_load_start_s)
        _bulk_cuda(pipeline, timings)

    input_decode_start_s = time.perf_counter()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image = _validated_rgba(args.img_path)
    timings["input_decode_ms"] = _elapsed_ms(input_decode_start_s)

    pipeline_run_start_s = time.perf_counter()
    mesh = pipeline.run(image, seed=int(args.seed))[0]
    timings["pipeline_run_ms"] = _elapsed_ms(pipeline_run_start_s)

    mesh_export_start_s = time.perf_counter()
    import o_voxel

    _pin_flex_gemm_autotune()
    mesh.simplify(16777216)
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=DECIMATION_TARGET_FACES,
        texture_size=TEXTURE_SIZE,
        remesh=True,
        verbose=False,
    )
    keep = _arap_safe_face_mask(glb.vertices, glb.faces)
    if not bool(keep.all()):
        dropped = int((~keep).sum())
        # Quality guarantee (owner rule 2026-08-06): the filter may only ever
        # remove zero-extent junk — faces with no renderable area, hence no
        # effect on SuperGlue matching, PnP, scale, or sampling (verified
        # pixel-identical across 16 candidate-render viewpoints). A drop
        # fraction beyond this bound cannot be that; fail the stage loudly
        # instead of silently shipping a degraded mesh.
        if dropped > max(2, len(keep) // 1000):
            raise ValueError(
                f"ARAP-safety filter wants to drop {dropped}/{len(keep)} "
                "faces — far beyond zero-extent junk; refusing to degrade "
                "the mesh (inspect the o_voxel export)"
            )
        print(
            f"[trellis2-generate] dropping {dropped} zero-extent ARAP-unsafe "
            f"sliver face(s) of {len(keep)} (render-invisible; "
            "alignment inputs unchanged)",
            flush=True,
        )
        glb.update_faces(keep)
        glb.remove_unreferenced_vertices()
    # PNG textures: pytorch3d's experimental GLB reader (align) and Open3D
    # (the GUI mesh view) cannot decode EXT_texture_webp.
    with _fast_png_encode():
        glb.export(str(output_dir / "object.glb"), extension_webp=False)
    timings["mesh_export_ms"] = _elapsed_ms(mesh_export_start_s)
    return timings


def _elapsed_ms(start_s):
    """Local perf-counter delta in ms (timing.elapsed_ms mirror, pre-import)."""
    duration_ms = (time.perf_counter() - float(start_s)) * 1000.0
    if duration_ms < 0.0:
        raise ValueError(f"invalid timing duration: {duration_ms}")
    return float(duration_ms)


def _import_stage_profile_run():
    """Import the v6.2 timing contract (stdlib-only import chain).

    The parent's _stage_env already puts the repo root on PYTHONPATH; the
    fallback covers direct CLI invocations of this script.
    """
    try:
        from demo_v6_2.shape_prior.timing import StageProfileRun
    except ModuleNotFoundError:
        repo_root = str(Path(__file__).resolve().parents[2])
        if repo_root not in sys.path:
            sys.path.append(repo_root)
        from demo_v6_2.shape_prior.timing import StageProfileRun
    return StageProfileRun


def main(argv=None):
    """Run the command-line entry point (generate.py lifecycle mirror)."""
    module_import_ms = _elapsed_ms(_MODULE_IMPORT_START_S)
    args = build_parser().parse_args(argv)
    StageProfileRun = _import_stage_profile_run()
    run = StageProfileRun(
        stage="generate",
        profile_json=args.profile_json,
        wait_signal=args.wait_signal,
        timing_ms=dict.fromkeys(
            (*_ACTIVE_TIMING_FIELDS, "go_wait_ms", "total_ms", "process_lifetime_ms"),
            0.0,
        ),
        active_fields=_ACTIVE_TIMING_FIELDS,
        process_started_s=_MODULE_IMPORT_START_S,
    )
    timing_ms = run.timing_ms
    timing_ms["module_import_ms"] = float(module_import_ms)

    if args.wait_signal:
        # Pre-GO work is CPU-only: sys.path/CUDA bootstrap plus
        # from_pretrained into CPU RAM (~17G; the box has 251G). low_vram
        # staging keeps VRAM free until the post-GO run, so the upscale
        # stage's inference peak never overlaps model weights — the same
        # sequencing contract as the SAM3D worker, at a much lower VRAM
        # footprint (TRELLIS.2 peaks ~2.7G allocated).
        prepare_start_s = time.perf_counter()
        _bootstrap_paths(args.trellis2_repo)
        timing_ms["pre_go_prepare_ms"] = _elapsed_ms(prepare_start_s)

        model_load_start_s = time.perf_counter()
        pipeline = _load_pipeline(args.model)
        timing_ms["model_load_ms"] = _elapsed_ms(model_load_start_s)

        run.write_waiting()
        if not run.wait_for_go():
            return

        _bulk_cuda(pipeline, timing_ms)
        run_trellis2_shape_prior(args, pipeline=pipeline, timing_ms=timing_ms)
    else:
        run_trellis2_shape_prior(args, timing_ms=timing_ms)

    run.write_completed()


if __name__ == "__main__":
    main()
