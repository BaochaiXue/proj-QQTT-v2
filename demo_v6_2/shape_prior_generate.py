import time


_MODULE_IMPORT_START_S = time.perf_counter()


import os  # noqa: E402
from argparse import ArgumentParser  # noqa: E402
from pathlib import Path  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from demo_v6_2.shape_prior_timing import (  # noqa: E402
    STAGE_PROFILE_STATUS_COMPLETED,
    STAGE_PROFILE_STATUS_WAITING,
    elapsed_ms,
    write_stage_profile,
)


# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or
# 'xformers', default is 'flash-attn'
os.environ["SPCONV_ALGO"] = "native"  # Can be 'native' or 'auto'.


DEFAULT_SAM3D_ROOT_CANDIDATES = [
    Path(os.environ[key]).expanduser()
    for key in ("SAM3D_ROOT", "MVSAM3D_ROOT")
    if os.environ.get(key)
]
DEFAULT_SAM3D_ROOT_CANDIDATES += [
    Path("sam-3d-objects"),
    Path("vendor/demo_runtime/sam-3d-objects"),
    Path("/home/xinjie/external/sam-3d-objects"),
    Path("/home/xinjie/external/MV-SAM3D"),
]
DEFAULT_SEED = 42
_ACTIVE_TIMING_FIELDS = (
    "module_import_ms",
    "pre_go_prepare_ms",
    "model_load_ms",
    "input_decode_ms",
    "pipeline_run_ms",
    "mesh_export_ms",
    "gaussian_export_ms",
    "visualization_export_ms",
)


def build_parser():
    """Build the command-line argument parser."""
    parser = ArgumentParser(
        description="Generate shape prior via SAM3D with the original API."
    )
    parser.add_argument("--img_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--sam3d-root",
        type=str,
        default=None,
        help="SAM3D checkout containing notebook/inference.py.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="SAM3D pipeline config. Defaults under --sam3d-root.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--profile-json", type=Path, default=None)
    parser.add_argument(
        "--skip-visualization",
        action="store_true",
        help="Skip optional visualization.mp4 generation.",
    )
    parser.add_argument(
        "--wait-signal",
        dest="wait_signal",
        action="store_true",
        help="Load the SAM3D pipeline, then block on stdin for GO before running.",
    )
    return parser


def resolve_sam3d_root(value=None):
    """Resolve SAM3D root."""
    candidates = [Path(value).expanduser()] if value else []
    candidates.extend(DEFAULT_SAM3D_ROOT_CANDIDATES)
    for candidate in candidates:
        root = candidate.resolve()
        if (root / "notebook" / "inference.py").is_file() and (
            root / "sam3d_objects"
        ).is_dir():
            return root
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "SAM3D root not found. Set SAM3D_ROOT or pass --sam3d-root. "
        f"Searched: {searched}"
    )


def default_config_for_root(root):
    """Return the default config for root."""
    for candidate in (
        root / "checkpoints" / "hf" / "pipeline.yaml",
        root / "checkpoints" / "hf" / "checkpoints" / "pipeline.yaml",
    ):
        if candidate.exists():
            return candidate
    return root / "checkpoints" / "hf" / "pipeline.yaml"


def load_inference_class(root):
    """Load inference class."""
    for path in (root, root / "notebook"):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    from inference import Inference  # type: ignore

    return Inference


def rgba_to_sam3d_inputs(image):
    """Return the RGBA to SAM3D inputs."""
    final_im = image.convert("RGBA")
    rgba = np.asarray(final_im, dtype=np.uint8)
    alpha = rgba[:, :, 3]
    if np.all(alpha == 255):
        raise ValueError("Image must contain an alpha foreground mask.")
    image_rgb = np.ascontiguousarray(rgba[:, :, :3], dtype=np.uint8)
    mask = np.ascontiguousarray((alpha > 0).astype(np.uint8) * 255)
    return image_rgb, mask


def _first(value):
    """Return the first."""
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def export_mesh(mesh_obj, path):
    """Return the export mesh."""
    if mesh_obj is None:
        raise ValueError("SAM3D output did not include a mesh/glb object.")
    if hasattr(mesh_obj, "success") and getattr(mesh_obj, "success") is False:
        raise ValueError("SAM3D mesh extraction failed.")
    if hasattr(mesh_obj, "export"):
        mesh_obj.export(path)
        return

    vertices = getattr(mesh_obj, "vertices", None)
    faces = getattr(mesh_obj, "faces", None)
    if vertices is None or faces is None:
        raise AttributeError(f"Mesh object cannot be exported to {path}.")
    if hasattr(vertices, "detach"):
        vertices = vertices.detach().cpu().numpy()
    if hasattr(faces, "detach"):
        faces = faces.detach().cpu().numpy()

    import trimesh

    trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(path)


def resolve_inference_inputs(args):
    """Resolve the SAM3D checkout and import its Inference class (CPU-only)."""
    sam3d_root = resolve_sam3d_root(args.sam3d_root)
    config = Path(args.config).expanduser() if args.config else None
    config = config or default_config_for_root(sam3d_root)
    if not config.exists():
        raise FileNotFoundError(f"SAM3D config not found: {config}")
    Inference = load_inference_class(sam3d_root)
    return Inference, config


def load_sam3d_inference(args):
    """Resolve the SAM3D checkout and load its inference pipeline (the slow part)."""
    Inference, config = resolve_inference_inputs(args)
    return Inference(str(config), compile=False)


def run_sam3d_shape_prior(args, infer=None, *, timing_ms=None):
    """Run the unchanged SAM3D algorithm and return its active timings."""
    timings = {} if timing_ms is None else timing_ms
    if infer is None:
        prepare_start_s = time.perf_counter()
        Inference, config = resolve_inference_inputs(args)
        timings["pre_go_prepare_ms"] = elapsed_ms(prepare_start_s)

        model_load_start_s = time.perf_counter()
        infer = Inference(str(config), compile=False)
        timings["model_load_ms"] = elapsed_ms(model_load_start_s)

    input_decode_start_s = time.perf_counter()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_rgb, mask = rgba_to_sam3d_inputs(Image.open(args.img_path))
    timings["input_decode_ms"] = elapsed_ms(input_decode_start_s)

    pipeline_run_start_s = time.perf_counter()
    pipeline = getattr(infer, "_pipeline", None)
    if pipeline is not None and hasattr(pipeline, "run"):
        if hasattr(pipeline, "rendering_engine"):
            pipeline.rendering_engine = "nvdiffrast"
        outputs = pipeline.run(
            image_rgb,
            mask,
            seed=int(args.seed),
            with_mesh_postprocess=True,
            with_texture_baking=True,
            with_layout_postprocess=True,
            use_vertex_color=False,
        )
    else:
        outputs = infer(image_rgb, mask, seed=int(args.seed))
    timings["pipeline_run_ms"] = elapsed_ms(pipeline_run_start_s)

    mesh_export_start_s = time.perf_counter()
    mesh_obj = _first(outputs.get("glb"))
    if mesh_obj is None:
        mesh_obj = _first(outputs.get("mesh"))
    mesh_path = output_dir / "object.glb"
    export_mesh(mesh_obj, mesh_path)
    timings["mesh_export_ms"] = elapsed_ms(mesh_export_start_s)

    gaussian_export_start_s = time.perf_counter()
    gaussian = _first(outputs.get("gaussian"))
    if gaussian is None:
        gaussian = _first(outputs.get("gs"))
    if gaussian is not None and hasattr(gaussian, "save_ply"):
        gaussian.save_ply(output_dir / "object.ply")
    timings["gaussian_export_ms"] = elapsed_ms(gaussian_export_start_s)

    visualization_start_s = time.perf_counter()
    if not args.skip_visualization:
        frames = outputs.get("video") or outputs.get("visualization")
        if frames is not None:
            import imageio

            imageio.mimsave(output_dir / "visualization.mp4", frames, fps=30)
    timings["visualization_export_ms"] = elapsed_ms(visualization_start_s)
    return timings


def _new_timing_record(*, module_import_ms: float) -> dict[str, float]:
    """Return all generate timing fields with explicit zero values."""
    timing_ms = {field: 0.0 for field in _ACTIVE_TIMING_FIELDS}
    timing_ms.update(
        {
            "go_wait_ms": 0.0,
            "total_ms": 0.0,
            "process_lifetime_ms": 0.0,
        }
    )
    timing_ms["module_import_ms"] = float(module_import_ms)
    return timing_ms


def _active_total_ms(timing_ms: dict[str, float]) -> float:
    """Return active stage time, intentionally excluding the GO idle wait."""
    return float(sum(timing_ms[field] for field in _ACTIVE_TIMING_FIELDS))


def main(argv=None):
    """Run the command-line entry point."""
    module_import_ms = elapsed_ms(_MODULE_IMPORT_START_S)
    args = build_parser().parse_args(argv)
    execution_mode = "prewarmed" if args.wait_signal else "cold"
    timing_ms = _new_timing_record(module_import_ms=module_import_ms)
    ready_wall_time_s = None

    if args.wait_signal:
        from demo_v6_2.utils import stage_prewarm

        # Pre-GO work is CPU-only (checkout resolution + the sam3d_objects
        # import tree). Weights go to the GPU only after GO: the upscale
        # stage's inference peak plus resident SAM3D weights do not fit on
        # one 24GB warmup GPU, so they must never overlap.
        prepare_start_s = time.perf_counter()
        Inference, config = resolve_inference_inputs(args)
        timing_ms["pre_go_prepare_ms"] = elapsed_ms(prepare_start_s)

        ready_wall_time_s = time.time()
        timing_ms["total_ms"] = _active_total_ms(timing_ms)
        timing_ms["process_lifetime_ms"] = elapsed_ms(_MODULE_IMPORT_START_S)
        write_stage_profile(
            args.profile_json,
            stage="generate",
            status=STAGE_PROFILE_STATUS_WAITING,
            execution_mode=execution_mode,
            timing_ms=timing_ms,
            ready_wall_time_s=ready_wall_time_s,
        )

        go_wait_start_s = time.perf_counter()
        should_run = stage_prewarm.wait_for_go("generate")
        timing_ms["go_wait_ms"] = elapsed_ms(go_wait_start_s)
        if not should_run:
            return

        model_load_start_s = time.perf_counter()
        infer = Inference(str(config), compile=False)
        timing_ms["model_load_ms"] = elapsed_ms(model_load_start_s)
        run_sam3d_shape_prior(args, infer=infer, timing_ms=timing_ms)
    else:
        run_sam3d_shape_prior(args, timing_ms=timing_ms)

    timing_ms["total_ms"] = _active_total_ms(timing_ms)
    timing_ms["process_lifetime_ms"] = elapsed_ms(_MODULE_IMPORT_START_S)
    write_stage_profile(
        args.profile_json,
        stage="generate",
        status=STAGE_PROFILE_STATUS_COMPLETED,
        execution_mode=execution_mode,
        timing_ms=timing_ms,
        ready_wall_time_s=ready_wall_time_s,
    )


if __name__ == "__main__":
    main()
