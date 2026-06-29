#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from PIL import Image


def _resolve_repo_root() -> Path:
    """Resolve the repo root for the Demo v5.1-local worker."""
    current_path = Path(__file__).resolve()
    return current_path.parent.parent


REPO_ROOT = _resolve_repo_root()
REPO_ROOT_STR = str(REPO_ROOT)
if REPO_ROOT_STR in sys.path:
    sys.path.remove(REPO_ROOT_STR)
sys.path.insert(0, REPO_ROOT_STR)

from demo_v5_1.shape_prior import (  # noqa: E402
    DEFAULT_SHAPE_PRIOR_RENDER_RGB,
    SHAPE_PRIOR_STATUS_FAILED,
    SHAPE_PRIOR_STATUS_READY,
    ShapePriorFrame0Request,
    ShapePriorResult,
    align_mesh_to_observation,
    observation_points_world,
    pack_shape_prior_result,
    sample_shape_prior_points,
    unpack_shape_prior_request,
)
from demo_v5_1.shape_prior_warmup import (  # noqa: E402
    prepare_shape_prior_worker_startup,
)


DEFAULT_RUNTIME_ASSET_ROOT = Path("vendor") / "demo_runtime"
DEFAULT_SAM3D_ROOT = DEFAULT_RUNTIME_ASSET_ROOT / "sam-3d-objects"
DEFAULT_UPSCALER_ROOT = DEFAULT_RUNTIME_ASSET_ROOT / "stable-diffusion-x4-upscaler"
DEFAULT_UPSCALE_CATEGORY = "stuffed animal"


def _elapsed_ms(start_s: float, end_s: float | None = None) -> float:
    return (
        (time.perf_counter() if end_s is None else float(end_s)) - float(start_s)
    ) * 1000.0


def _default_config_for_root(root: Path) -> Path:
    for candidate in (
        root / "checkpoints" / "hf" / "pipeline.yaml",
        root / "checkpoints" / "hf" / "checkpoints" / "pipeline.yaml",
    ):
        if candidate.exists():
            return candidate
    return root / "checkpoints" / "hf" / "pipeline.yaml"


def _sample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    finite = np.isfinite(arr).all(axis=1)
    arr = arr[finite]
    if int(max_points) > 0 and len(arr) > int(max_points):
        indices = np.linspace(0, len(arr) - 1, int(max_points), dtype=np.int64)
        arr = arr[indices]
    return np.ascontiguousarray(arr, dtype=np.float32)


def _sam3d_output_mesh(outputs: dict[str, Any]) -> Any:
    import trimesh

    for key in ("glb", "mesh"):
        mesh_like = outputs.get(key)
        if isinstance(mesh_like, trimesh.Trimesh):
            return mesh_like.copy()
        if isinstance(mesh_like, trimesh.Scene):
            mesh = mesh_like.dump(concatenate=True)
            if isinstance(mesh, trimesh.Trimesh):
                return mesh.copy()
    raise RuntimeError("SAM3D output did not include a trimesh mesh/glb object")


def _object_crop_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(np.asarray(mask, dtype=bool))
    if coords.size == 0:
        raise ValueError("shape-prior object mask is empty")
    y0 = int(np.min(coords[:, 0]))
    y1 = int(np.max(coords[:, 0]))
    x0 = int(np.min(coords[:, 1]))
    x1 = int(np.max(coords[:, 1]))
    center_x = (x0 + x1) / 2.0
    center_y = (y0 + y1) / 2.0
    size = max(x1 - x0 + 1, y1 - y0 + 1)
    size = max(2, int(math.ceil(size * 1.2)))
    half = size / 2.0
    return (
        int(math.floor(center_x - half)),
        int(math.floor(center_y - half)),
        int(math.ceil(center_x + half)),
        int(math.ceil(center_y + half)),
    )


class ShapePriorSam3DWorker:
    def __init__(
        self,
        *,
        sam3d_root: Path,
        config: Path | None,
        device: str,
        seed: int,
        max_points: int,
        upscale_category: str = DEFAULT_UPSCALE_CATEGORY,
    ) -> None:
        self.sam3d_root = Path(sam3d_root).expanduser()
        self.config = Path(config).expanduser() if config is not None else None
        self.device = str(device)
        self.seed = int(seed)
        self.max_points = int(max_points)
        self.upscale_category = str(upscale_category)
        self._inference: Any | None = None
        self._upscaler: Any | None = None
        self._last_sam3d_model_load_ms = 0.0
        self._last_upscaler_model_load_ms = 0.0
        self._startup_metadata: dict[str, Any] = {}

    def startup_metadata(self) -> dict[str, Any]:
        return dict(self._startup_metadata)

    def _load_upscaler(self) -> Any:
        if self._upscaler is not None:
            self._last_upscaler_model_load_ms = 0.0
            return self._upscaler
        start_s = time.perf_counter()
        from diffusers import StableDiffusionUpscalePipeline
        import torch

        pipeline = StableDiffusionUpscalePipeline.from_pretrained(
            str(DEFAULT_UPSCALER_ROOT),
            torch_dtype=torch.float16,
        )
        self._upscaler = pipeline.to(self.device)
        self._last_upscaler_model_load_ms = _elapsed_ms(start_s)
        return self._upscaler

    def _release_upscaler(self) -> None:
        if self._upscaler is None:
            return
        self._upscaler = None
        try:
            import gc
            import torch

            gc.collect()
            if str(self.device).startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _load_inference(self) -> Any:
        if self._inference is not None:
            self._last_sam3d_model_load_ms = 0.0
            return self._inference
        start_s = time.perf_counter()
        root = self.sam3d_root.resolve()
        if (
            not (root / "notebook" / "inference.py").exists()
            or not (root / "sam3d_objects").exists()
        ):
            raise FileNotFoundError(
                f"SAM3D root must contain notebook/inference.py and sam3d_objects: {root}"
            )
        for path in (root, root / "notebook"):
            path_s = str(path)
            if path_s not in sys.path:
                sys.path.insert(0, path_s)
        from inference import Inference  # type: ignore

        config = self.config or _default_config_for_root(root)
        if not config.exists():
            raise FileNotFoundError(f"SAM3D config not found: {config}")
        self._inference = Inference(str(config), compile=False)
        self._last_sam3d_model_load_ms = _elapsed_ms(start_s)
        return self._inference

    def preload_models(self) -> dict[str, Any]:
        start_s = time.perf_counter()
        self._load_upscaler()
        self._startup_metadata["worker_preload_upscaler_ms"] = _elapsed_ms(start_s)
        start_s = time.perf_counter()
        self._load_inference()
        self._startup_metadata["worker_preload_sam3d_ms"] = _elapsed_ms(start_s)
        self._startup_metadata["worker_preloaded_models"] = True
        return self.startup_metadata()

    def _upscaled_sam3d_input(
        self,
        frame0: ShapePriorFrame0Request,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        upscaler = self._load_upscaler()
        upscaler_model_load_ms = float(
            getattr(self, "_last_upscaler_model_load_ms", 0.0)
        )
        self._last_upscaler_model_load_ms = 0.0
        upscale_start_s = time.perf_counter()
        rgb = np.ascontiguousarray(frame0.rgb_u8, dtype=np.uint8)
        object_mask = np.asarray(frame0.object_mask, dtype=bool)
        crop_box = _object_crop_box(object_mask)
        crop_rgb = Image.fromarray(rgb).crop(crop_box)
        crop_mask_u8 = object_mask.astype(np.uint8) * 255
        crop_mask = Image.fromarray(crop_mask_u8).crop(crop_box)
        prompt = f"Hand manipulates a {self.upscale_category}."
        upscaled = upscaler(prompt=prompt, image=crop_rgb).images[0]
        upscaled_rgb = np.ascontiguousarray(
            np.asarray(upscaled.convert("RGB"), dtype=np.uint8)
        )
        image_upscale_ms = _elapsed_ms(upscale_start_s)

        mask_start_s = time.perf_counter()
        resized_mask = crop_mask.resize(upscaled.size, Image.Resampling.NEAREST)
        mask_u8 = (np.asarray(resized_mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
        mask_refinement_ms = _elapsed_ms(mask_start_s)
        self._release_upscaler()
        if int(np.count_nonzero(mask_u8)) <= 0:
            raise RuntimeError("upscaled shape-prior mask is empty")
        return (
            upscaled_rgb,
            np.ascontiguousarray(mask_u8, dtype=np.uint8),
            {
                "upscaler_model_load_ms": upscaler_model_load_ms,
                "image_upscale_ms": image_upscale_ms,
                "mask_refinement_ms": mask_refinement_ms,
                "sam3d_input_mask_pixels": int(np.count_nonzero(mask_u8)),
            },
        )

    def _canonical_points_from_sam3d(
        self, frame0: ShapePriorFrame0Request
    ) -> tuple[Any, np.ndarray, dict[str, Any]]:
        image_rgb, mask_u8, prep_stats = self._upscaled_sam3d_input(frame0)
        infer = self._load_inference()
        sam3d_model_load_ms = float(getattr(self, "_last_sam3d_model_load_ms", 0.0))
        self._last_sam3d_model_load_ms = 0.0
        start_s = time.perf_counter()
        outputs = infer._pipeline.run(  # type: ignore[attr-defined]
            image_rgb,
            mask_u8,
            seed=self.seed,
            with_mesh_postprocess=True,
            with_texture_baking=False,
            with_layout_postprocess=True,
            use_vertex_color=False,
        )
        canonical_mesh = _sam3d_output_mesh(outputs)
        mesh_vertices = np.asarray(canonical_mesh.vertices, dtype=np.float32).reshape(
            -1, 3
        )
        canonical = _sample_points(mesh_vertices, self.max_points)
        if len(canonical) < 3:
            raise RuntimeError("SAM3D canonical mesh has fewer than 3 finite vertices")
        stats = {
            "sam3d_model_load_ms": sam3d_model_load_ms,
            "sam3d_inference_ms": _elapsed_ms(start_s),
            "sam3d_vertex_count": int(len(mesh_vertices)),
        }
        stats.update(prep_stats)
        stats.update(self.startup_metadata())
        return canonical_mesh, canonical, stats

    def handle(self, frame0: ShapePriorFrame0Request) -> list[bytes]:
        total_start_s = time.perf_counter()
        try:
            observation_start_s = time.perf_counter()
            observation = observation_points_world(frame0, max_points=self.max_points)
            observation_ms = _elapsed_ms(observation_start_s)
            if len(observation) < 3:
                raise RuntimeError(
                    "shape prior requires at least 3 valid object observation depth points"
                )
            canonical_mesh, _canonical, sam3d_stats = self._canonical_points_from_sam3d(
                frame0
            )
            align_start_s = time.perf_counter()
            aligned_mesh, aligned_points, align_stats = align_mesh_to_observation(
                canonical_mesh,
                observation,
            )
            align_ms = _elapsed_ms(align_start_s)
            sampling_start_s = time.perf_counter()
            samples = sample_shape_prior_points(
                aligned_mesh,
                observation,
            )
            surface_points = samples.surface_points_m
            interior_points = samples.interior_points_m
            structure = np.concatenate([surface_points, interior_points], axis=0)
            points = _sample_points(
                structure if len(structure) else aligned_points,
                self.max_points,
            )
            metadata = dict(sam3d_stats)
            metadata.update(align_stats)
            metadata.update(samples.metadata)
            metadata.update(
                {
                    "single_view_alignment_ms": align_ms,
                    "sampling_ms": observation_ms + _elapsed_ms(sampling_start_s),
                    "shape_prior_sampling_ms": _elapsed_ms(sampling_start_s),
                    "shape_prior_total_ms": _elapsed_ms(total_start_s),
                    "shape_prior_point_count": int(len(points)),
                }
            )
            metadata.update(self.startup_metadata())
            colors = np.full(
                (len(points), 3), DEFAULT_SHAPE_PRIOR_RENDER_RGB, dtype=np.uint8
            )
            metadata.update(
                {
                    "shape_backend": "sam3d-objects",
                    "shape_prior_source_seq": int(frame0.seq),
                    "shape_prior_source_time_s": frame0.source_timestamp_s,
                    "input_source": str(frame0.input_source),
                    "depth_backend": str(frame0.depth_backend),
                    "depth_source_internal": str(frame0.depth_source_internal),
                }
            )
            return pack_shape_prior_result(
                ShapePriorResult(
                    seq=int(frame0.seq),
                    source_seq=int(frame0.seq),
                    source_timestamp_s=frame0.source_timestamp_s,
                    status=SHAPE_PRIOR_STATUS_READY,
                    points_m=points,
                    colors_rgb_u8=colors,
                    surface_points_m=surface_points,
                    interior_points_m=interior_points,
                    metadata=metadata,
                )
            )
        except Exception as exc:
            return pack_shape_prior_result(
                ShapePriorResult(
                    seq=int(frame0.seq),
                    source_seq=int(frame0.seq),
                    source_timestamp_s=frame0.source_timestamp_s,
                    status=SHAPE_PRIOR_STATUS_FAILED,
                    metadata=self.startup_metadata(),
                    error=str(exc),
                )
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Long-lived remote SAM3D shape-prior worker for single-camera demos."
    )
    parser.add_argument(
        "--bind", default="tcp://0.0.0.0:7100", help="ZeroMQ REP bind endpoint."
    )
    parser.add_argument("--sam3d-root", type=Path, default=DEFAULT_SAM3D_ROOT)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="SAM3D pipeline YAML. Defaults under --sam3d-root.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-points", type=int, default=60000)
    parser.add_argument(
        "--upscale-category",
        default=DEFAULT_UPSCALE_CATEGORY,
        help="Category text used in the data_process_sam3d x4 upscaler prompt.",
    )
    parser.add_argument(
        "--preload-models",
        action="store_true",
        help="Load the x4 upscaler and SAM3D inference model before binding the worker endpoint.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    worker = ShapePriorSam3DWorker(
        sam3d_root=Path(args.sam3d_root),
        config=args.config,
        device=str(args.device),
        seed=int(args.seed),
        max_points=int(args.max_points),
        upscale_category=str(args.upscale_category),
    )
    try:
        startup_metadata = prepare_shape_prior_worker_startup(
            worker,
            preload_models=bool(args.preload_models),
        )
    except Exception as exc:
        print(
            f"[shape-prior-worker] startup failed: {exc}", file=sys.stderr, flush=True
        )
        return 1
    import zmq

    context = zmq.Context.instance()
    socket = context.socket(zmq.REP)
    socket.bind(str(args.bind))
    print(
        "[shape-prior-worker] "
        f"ready bind={args.bind} sam3d_root={args.sam3d_root} "
        f"preload={startup_metadata.get('worker_preloaded_models', False)} "
        f"worker_ready_ms={float(startup_metadata.get('worker_ready_ms', 0.0)):.1f}",
        flush=True,
    )
    while True:
        parts = socket.recv_multipart()
        try:
            frame0 = unpack_shape_prior_request(parts)
            reply = worker.handle(frame0)
        except Exception as exc:
            reply = pack_shape_prior_result(
                ShapePriorResult(
                    seq=-1,
                    status=SHAPE_PRIOR_STATUS_FAILED,
                    metadata=worker.startup_metadata(),
                    error=str(exc),
                )
            )
        socket.send_multipart(reply)


if __name__ == "__main__":
    raise SystemExit(main())
