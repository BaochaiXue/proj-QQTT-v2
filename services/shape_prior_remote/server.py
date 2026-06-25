#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from PIL import Image


def _resolve_repo_root() -> Path:
    candidates = [Path(__file__).resolve().parents[2]]
    env_root = os.environ.get("QQTT_REPO_ROOT")
    if env_root:
        candidates.append(Path(env_root))
    candidates.append(Path.cwd())
    for candidate in candidates:
        root = candidate.expanduser().resolve()
        if (root / "qqtt").is_dir() and (root / "services").is_dir():
            return root
    return Path(__file__).resolve().parents[2]


REPO_ROOT = _resolve_repo_root()
REPO_ROOT_STR = str(REPO_ROOT)
if REPO_ROOT_STR in sys.path:
    sys.path.remove(REPO_ROOT_STR)
sys.path.insert(0, REPO_ROOT_STR)

from qqtt.demo.shape_prior_warmup import DEFAULT_SHAPE_PRIOR_RENDER_RGB  # noqa: E402
from qqtt.demo.single_view_shape_align import (  # noqa: E402
    ShapeAlignmentConfig,
    align_canonical_shape_to_observation,
)
from qqtt.demo.single_view_shape_prior_sampling import (  # noqa: E402
    SimpleShapeMesh,
    sample_data_process_sam3d_single_view_shape_prior_points,
)
from services.shape_prior_remote.protocol import (  # noqa: E402
    ShapePriorRequest,
    build_error_response_parts,
    build_shape_prior_response_parts,
    parse_shape_prior_request_parts,
)


DEFAULT_RUNTIME_ASSET_ROOT = Path("vendor") / "demo_runtime"
DEFAULT_SAM3D_ROOT = DEFAULT_RUNTIME_ASSET_ROOT / "sam-3d-objects"
DEFAULT_FUTUREPHYSTWIN_ROOT = DEFAULT_RUNTIME_ASSET_ROOT / "FuturePhysTwin"
DEFAULT_UPSCALE_CATEGORY = "stuffed animal"
_WARMUP_IMAGE_SIZE = 64


def _elapsed_ms(start_s: float, end_s: float | None = None) -> float:
    return ((time.perf_counter() if end_s is None else float(end_s)) - float(start_s)) * 1000.0


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


def _default_startup_metadata() -> dict[str, Any]:
    return {
        "worker_preload_upscaler_ms": 0.0,
        "worker_preload_sam3d_ms": 0.0,
        "worker_dummy_warmup_ms": 0.0,
        "worker_ready_ms": 0.0,
        "worker_preloaded_models": False,
        "worker_warmed_models": False,
    }


def _mesh_vertices(mesh_like: Any) -> np.ndarray | None:
    if mesh_like is None:
        return None
    vertices = getattr(mesh_like, "vertices", None)
    if vertices is not None:
        if hasattr(vertices, "detach"):
            vertices = vertices.detach().cpu().numpy()
        return np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    if isinstance(mesh_like, dict):
        candidates = mesh_like.values()
    elif isinstance(mesh_like, (list, tuple)):
        candidates = mesh_like
    else:
        geometry = getattr(mesh_like, "geometry", None)
        if isinstance(geometry, dict):
            candidates = geometry.values()
        elif geometry is not None and geometry is not mesh_like:
            candidates = (geometry,)
        else:
            dump = getattr(mesh_like, "dump", None)
            if callable(dump):
                try:
                    dumped = dump(concatenate=True)
                except TypeError:
                    dumped = dump()
                return _mesh_vertices(dumped)
            return None
    parts = []
    for candidate in candidates:
        candidate_vertices = _mesh_vertices(candidate)
        if candidate_vertices is not None and len(candidate_vertices) > 0:
            parts.append(candidate_vertices)
    if not parts:
        return None
    return np.concatenate(parts, axis=0).astype(np.float32, copy=False)


def _mesh_to_trimesh(mesh_like: Any) -> Any | None:
    if mesh_like is None:
        return None
    try:
        import trimesh
    except Exception:
        trimesh = None

    if trimesh is not None and isinstance(mesh_like, trimesh.Trimesh):
        return mesh_like.copy()
    if trimesh is not None and isinstance(mesh_like, trimesh.Scene):
        dumped = mesh_like.dump(concatenate=True)
        return dumped.copy() if isinstance(dumped, trimesh.Trimesh) else None
    vertices = getattr(mesh_like, "vertices", None)
    faces = getattr(mesh_like, "faces", None)
    if vertices is not None and faces is not None:
        if hasattr(vertices, "detach"):
            vertices = vertices.detach().cpu().numpy()
        if hasattr(faces, "detach"):
            faces = faces.detach().cpu().numpy()
        vertices_arr = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
        faces_arr = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
        if len(vertices_arr) and len(faces_arr):
            if trimesh is None:
                return SimpleShapeMesh(vertices=vertices_arr, faces=faces_arr)
            return trimesh.Trimesh(vertices=vertices_arr, faces=faces_arr, process=False)
    if isinstance(mesh_like, dict):
        candidates = mesh_like.values()
    elif isinstance(mesh_like, (list, tuple)):
        candidates = mesh_like
    else:
        geometry = getattr(mesh_like, "geometry", None)
        if isinstance(geometry, dict):
            candidates = geometry.values()
        elif geometry is not None and geometry is not mesh_like:
            candidates = (geometry,)
        else:
            dump = getattr(mesh_like, "dump", None)
            if callable(dump):
                try:
                    return _mesh_to_trimesh(dump(concatenate=True))
                except TypeError:
                    return _mesh_to_trimesh(dump())
                except Exception:
                    return None
            return None
    parts = []
    for candidate in candidates:
        mesh = _mesh_to_trimesh(candidate)
        if mesh is not None and len(mesh.vertices) and len(mesh.faces):
            parts.append(mesh)
    if not parts:
        return None
    if trimesh is not None:
        return trimesh.util.concatenate(parts)
    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    offset = 0
    for mesh in parts:
        verts = np.asarray(mesh.vertices, dtype=np.float32).reshape(-1, 3)
        tri = np.asarray(mesh.faces, dtype=np.int64).reshape(-1, 3)
        vertices.append(verts)
        faces.append(tri + offset)
        offset += len(verts)
    return SimpleShapeMesh(vertices=np.concatenate(vertices, axis=0), faces=np.concatenate(faces, axis=0))


def _transform_trimesh(mesh: Any, *, scale: float, rotation: np.ndarray, translation: np.ndarray) -> Any:
    transformed = mesh.copy()
    vertices = np.asarray(transformed.vertices, dtype=np.float32).reshape(-1, 3)
    aligned = np.float32(scale) * (vertices @ np.asarray(rotation, dtype=np.float32).reshape(3, 3).T)
    aligned = aligned + np.asarray(translation, dtype=np.float32).reshape(1, 3)
    transformed.vertices = aligned
    return transformed


def _object_observation_points_world(request: ShapePriorRequest, *, max_points: int) -> np.ndarray:
    observation_mask = getattr(request, "object_observation_mask", None)
    mask = np.asarray(request.object_mask if observation_mask is None else observation_mask, dtype=bool)
    depth = np.asarray(request.depth_color_m, dtype=np.float32)
    valid = mask & np.isfinite(depth) & (depth > np.float32(0.0))
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32)
    rows, cols = np.nonzero(valid)
    z = depth[rows, cols].astype(np.float32, copy=False)
    k = np.asarray(request.k_color, dtype=np.float32).reshape(3, 3)
    fx = float(k[0, 0])
    fy = float(k[1, 1])
    cx = float(k[0, 2])
    cy = float(k[1, 2])
    if fx == 0.0 or fy == 0.0:
        raise ValueError("invalid k_color focal length")
    x = (cols.astype(np.float32) - np.float32(cx)) * z / np.float32(fx)
    y = (rows.astype(np.float32) - np.float32(cy)) * z / np.float32(fy)
    camera_points = np.stack([x, y, z], axis=1).astype(np.float32, copy=False)
    c2w = np.asarray(request.camera_to_world_c2w, dtype=np.float32).reshape(4, 4)
    homogeneous = np.concatenate([camera_points, np.ones((len(camera_points), 1), dtype=np.float32)], axis=1)
    world = (c2w @ homogeneous.T).T[:, :3]
    return _sample_points(world, max_points)


def _request_mask_metadata(request: ShapePriorRequest) -> dict[str, int]:
    observation_mask = getattr(request, "object_observation_mask", None)
    if observation_mask is None:
        observation_mask = request.object_mask
    return {
        "object_mask_pixels": int(np.count_nonzero(np.asarray(request.object_mask, dtype=bool))),
        "object_observation_mask_pixels": int(np.count_nonzero(np.asarray(observation_mask, dtype=bool))),
    }


def _alignment_config_from_request(request: ShapePriorRequest) -> ShapeAlignmentConfig:
    metadata = dict(request.metadata)
    above_direction = str(metadata.get("table_z_above_direction", "negative"))
    if above_direction not in {"positive", "negative"}:
        raise ValueError("table_z_above_direction must be positive or negative")
    return ShapeAlignmentConfig(
        table_z_m=float(metadata.get("table_z_m", 0.0)),
        above_direction=above_direction,
    )


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
        echo_observation: bool = False,
    ) -> None:
        self.sam3d_root = Path(sam3d_root).expanduser()
        self.config = Path(config).expanduser() if config is not None else None
        self.device = str(device)
        self.seed = int(seed)
        self.max_points = int(max_points)
        self.upscale_category = str(upscale_category)
        self.echo_observation = bool(echo_observation)
        self._inference: Any | None = None
        self._upscaler: Any | None = None
        self._model_load_ms = 0.0
        self._last_sam3d_model_load_ms = 0.0
        self._last_upscaler_model_load_ms = 0.0
        self._startup_metadata: dict[str, Any] = _default_startup_metadata()
        self._last_canonical_mesh: Any | None = None

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
            "stabilityai/stable-diffusion-x4-upscaler",
            torch_dtype=torch.float16,
        )
        self._upscaler = pipeline.to(self.device)
        self._last_upscaler_model_load_ms = _elapsed_ms(start_s)
        return self._upscaler

    def _load_inference(self) -> Any:
        if self._inference is not None:
            self._last_sam3d_model_load_ms = 0.0
            return self._inference
        start_s = time.perf_counter()
        root = self.sam3d_root.resolve()
        if not (root / "notebook" / "inference.py").exists() or not (root / "sam3d_objects").exists():
            raise FileNotFoundError(f"SAM3D root must contain notebook/inference.py and sam3d_objects: {root}")
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
        self._model_load_ms = self._last_sam3d_model_load_ms
        return self._inference

    def preload_models(self) -> dict[str, Any]:
        upscaler_start_s = time.perf_counter()
        self._load_upscaler()
        upscaler_elapsed_ms = _elapsed_ms(upscaler_start_s)
        upscaler_load_ms = float(getattr(self, "_last_upscaler_model_load_ms", 0.0))
        self._last_upscaler_model_load_ms = 0.0

        sam3d_start_s = time.perf_counter()
        self._load_inference()
        sam3d_elapsed_ms = _elapsed_ms(sam3d_start_s)
        sam3d_load_ms = float(getattr(self, "_last_sam3d_model_load_ms", 0.0))
        self._last_sam3d_model_load_ms = 0.0
        self._model_load_ms = 0.0

        self._startup_metadata.update(
            {
                "worker_preload_upscaler_ms": upscaler_load_ms if upscaler_load_ms > 0.0 else upscaler_elapsed_ms,
                "worker_preload_sam3d_ms": sam3d_load_ms if sam3d_load_ms > 0.0 else sam3d_elapsed_ms,
                "worker_preloaded_models": True,
            }
        )
        return self.startup_metadata()

    def run_dummy_warmup(self) -> dict[str, Any]:
        warmup_start_s = time.perf_counter()
        rgb = np.full((_WARMUP_IMAGE_SIZE, _WARMUP_IMAGE_SIZE, 3), 127, dtype=np.uint8)
        gradient = np.linspace(80, 180, _WARMUP_IMAGE_SIZE, dtype=np.uint8)
        rgb[:, :, 0] = gradient[None, :]
        rgb[:, :, 1] = gradient[:, None]
        mask = np.zeros((_WARMUP_IMAGE_SIZE, _WARMUP_IMAGE_SIZE), dtype=np.uint8)
        margin = _WARMUP_IMAGE_SIZE // 4
        mask[margin:-margin, margin:-margin] = 255

        prompt = f"Hand manipulates a {self.upscale_category}."
        upscaler = self._load_upscaler()
        upscaled = upscaler(prompt=prompt, image=Image.fromarray(rgb)).images[0]
        upscaled_rgb = np.ascontiguousarray(np.asarray(upscaled.convert("RGB"), dtype=np.uint8))
        mask_u8 = np.asarray(
            Image.fromarray(mask).resize(upscaled.size, Image.Resampling.NEAREST),
            dtype=np.uint8,
        )
        if int(np.count_nonzero(mask_u8)) <= 0:
            raise RuntimeError("dummy shape-prior warmup mask is empty")

        infer = self._load_inference()
        outputs = infer._pipeline.run(  # type: ignore[attr-defined]
            upscaled_rgb,
            np.ascontiguousarray(mask_u8, dtype=np.uint8),
            seed=self.seed,
            with_mesh_postprocess=True,
            with_texture_baking=False,
            with_layout_postprocess=True,
            use_vertex_color=False,
        )
        mesh_like = outputs.get("glb", None)
        mesh = _mesh_to_trimesh(mesh_like)
        if mesh is None:
            mesh_like = outputs.get("mesh", None)
            mesh = _mesh_to_trimesh(mesh_like)
        vertices = np.asarray(mesh.vertices, dtype=np.float32).reshape(-1, 3) if mesh is not None else _mesh_vertices(mesh_like)
        if vertices is None or len(vertices) <= 0:
            raise RuntimeError("dummy SAM3D warmup did not produce convertible mesh geometry")
        try:
            import torch

            if str(self.device).startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize(self.device)
        except Exception:
            pass
        self._last_canonical_mesh = None
        self._last_upscaler_model_load_ms = 0.0
        self._last_sam3d_model_load_ms = 0.0
        self._model_load_ms = 0.0
        self._startup_metadata.update(
            {
                "worker_dummy_warmup_ms": _elapsed_ms(warmup_start_s),
                "worker_warmed_models": True,
            }
        )
        return self.startup_metadata()

    def _upscaled_sam3d_input(
        self,
        request: ShapePriorRequest,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        upscaler = self._load_upscaler()
        upscaler_model_load_ms = float(getattr(self, "_last_upscaler_model_load_ms", 0.0))
        self._last_upscaler_model_load_ms = 0.0
        upscale_start_s = time.perf_counter()
        rgb = np.ascontiguousarray(request.rgb_u8, dtype=np.uint8)
        object_mask = np.asarray(request.object_mask, dtype=bool)
        crop_box = _object_crop_box(object_mask)
        crop_rgb = Image.fromarray(rgb).crop(crop_box)
        crop_mask_u8 = (object_mask.astype(np.uint8) * 255)
        crop_mask = Image.fromarray(crop_mask_u8).crop(crop_box)
        prompt = f"Hand manipulates a {self.upscale_category}."
        upscaled = upscaler(prompt=prompt, image=crop_rgb).images[0]
        upscaled_rgb = np.ascontiguousarray(np.asarray(upscaled.convert("RGB"), dtype=np.uint8))
        image_upscale_ms = _elapsed_ms(upscale_start_s)

        mask_start_s = time.perf_counter()
        resized_mask = crop_mask.resize(upscaled.size, Image.Resampling.NEAREST)
        mask_u8 = (np.asarray(resized_mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
        mask_refinement_ms = _elapsed_ms(mask_start_s)
        if int(np.count_nonzero(mask_u8)) <= 0:
            raise RuntimeError("upscaled shape-prior mask is empty")
        return upscaled_rgb, np.ascontiguousarray(mask_u8, dtype=np.uint8), {
            "upscaler_model_load_ms": upscaler_model_load_ms,
            "image_upscale_ms": image_upscale_ms,
            "mask_refinement_ms": mask_refinement_ms,
            "shape_prior_upscale_prompt": prompt,
            "sam3d_original_rgb_shape": [int(v) for v in rgb.shape],
            "sam3d_crop_box_xyxy": [int(v) for v in crop_box],
            "sam3d_crop_shape": [int(crop_rgb.height), int(crop_rgb.width), 3],
            "sam3d_input_shape": [int(v) for v in upscaled_rgb.shape],
            "sam3d_input_mask_pixels": int(np.count_nonzero(mask_u8)),
        }

    def _canonical_points_from_sam3d(self, request: ShapePriorRequest) -> tuple[np.ndarray, dict[str, Any]]:
        image_rgb, mask_u8, prep_stats = self._upscaled_sam3d_input(request)
        infer = self._load_inference()
        sam3d_model_load_ms = float(getattr(self, "_last_sam3d_model_load_ms", 0.0))
        self._last_sam3d_model_load_ms = 0.0
        self._model_load_ms = 0.0
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
        mesh_like = outputs.get("glb", None)
        canonical_mesh = _mesh_to_trimesh(mesh_like)
        mesh_vertices = np.asarray(canonical_mesh.vertices, dtype=np.float32).reshape(-1, 3) if canonical_mesh is not None else _mesh_vertices(mesh_like)
        mesh_source = "glb"
        if mesh_vertices is None or len(mesh_vertices) <= 0:
            mesh_like = outputs.get("mesh", None)
            canonical_mesh = _mesh_to_trimesh(mesh_like)
            mesh_vertices = np.asarray(canonical_mesh.vertices, dtype=np.float32).reshape(-1, 3) if canonical_mesh is not None else _mesh_vertices(mesh_like)
            mesh_source = "mesh"
        if mesh_vertices is None or len(mesh_vertices) <= 0:
            raise RuntimeError("SAM3D output did not include a mesh/glb object")
        if hasattr(mesh_vertices, "detach"):
            mesh_vertices = mesh_vertices.detach().cpu().numpy()
        self._last_canonical_mesh = canonical_mesh
        canonical = _sample_points(np.asarray(mesh_vertices, dtype=np.float32).reshape(-1, 3), self.max_points)
        if len(canonical) < 3:
            raise RuntimeError("SAM3D canonical mesh has fewer than 3 finite vertices")
        stats = {
            "sam3d_model_load_ms": sam3d_model_load_ms,
            "sam3d_inference_ms": _elapsed_ms(start_s),
            "geometry_export_ms": 0.0,
            "sam3d_mesh_source": mesh_source,
        }
        stats.update(prep_stats)
        stats.update(self.startup_metadata())
        return canonical, stats

    def handle(self, request: ShapePriorRequest) -> list[bytes]:
        total_start_s = time.perf_counter()
        request_id = str(request.metadata.get("request_id", ""))
        seq = int(request.metadata.get("seq", -1))
        try:
            observation_start_s = time.perf_counter()
            observation = _object_observation_points_world(request, max_points=self.max_points)
            observation_ms = _elapsed_ms(observation_start_s)
            if len(observation) < 3:
                raise RuntimeError("shape prior requires at least 3 valid object observation depth points")
            if self.echo_observation:
                points = observation
                surface_points = np.empty((0, 3), dtype=np.float32)
                interior_points = np.empty((0, 3), dtype=np.float32)
                metadata = {
                    "sam3d_model_load_ms": 0.0,
                    "upscaler_model_load_ms": 0.0,
                    "sam3d_inference_ms": 0.0,
                    "geometry_export_ms": 0.0,
                    "single_view_alignment_ms": 0.0,
                    "sampling_ms": observation_ms,
                    "shape_prior_total_ms": _elapsed_ms(total_start_s),
                    "alignment_valid": True,
                    "echo_observation": True,
                }
            else:
                self._last_canonical_mesh = None
                canonical, sam3d_stats = self._canonical_points_from_sam3d(request)
                align_start_s = time.perf_counter()
                aligned = align_canonical_shape_to_observation(
                    canonical,
                    observation,
                    config=_alignment_config_from_request(request),
                )
                align_ms = _elapsed_ms(align_start_s)
                if not aligned.valid:
                    raise RuntimeError(f"shape-prior single-view alignment invalid: {aligned.validation}")
                surface_points = np.empty((0, 3), dtype=np.float32)
                interior_points = np.empty((0, 3), dtype=np.float32)
                canonical_mesh = self._last_canonical_mesh
                sampling_start_s = time.perf_counter()
                if canonical_mesh is not None:
                    aligned_mesh = _transform_trimesh(
                        canonical_mesh,
                        scale=float(aligned.scale),
                        rotation=aligned.rotation,
                        translation=aligned.translation,
                    )
                    samples = sample_data_process_sam3d_single_view_shape_prior_points(
                        aligned_mesh,
                        observation,
                    )
                    surface_points = samples.surface_points_m
                    interior_points = samples.interior_points_m
                    sam3d_stats.update(samples.metadata)
                structure = np.concatenate([surface_points, interior_points], axis=0)
                points = _sample_points(structure if len(structure) else aligned.aligned_points_m, self.max_points)
                metadata = dict(sam3d_stats)
                metadata.update(
                    {
                        "single_view_alignment_ms": align_ms,
                        "sampling_ms": observation_ms + _elapsed_ms(sampling_start_s),
                        "single_view_shape_prior_sampling_ms": _elapsed_ms(sampling_start_s),
                        "shape_prior_total_ms": _elapsed_ms(total_start_s),
                        "alignment_valid": True,
                        "alignment": aligned.validation,
                    }
                )
            metadata.update(self.startup_metadata())
            metadata.update(_request_mask_metadata(request))
            colors = np.full((len(points), 3), DEFAULT_SHAPE_PRIOR_RENDER_RGB, dtype=np.uint8)
            metadata.update(
                {
                    "shape_backend": "sam3d-objects",
                    "shape_prior_source_seq": seq,
                    "shape_prior_source_time_s": request.metadata.get("source_timestamp_s"),
                    "input_source": request.metadata.get("input_source"),
                    "depth_backend": request.metadata.get("depth_backend"),
                    "depth_source_internal": request.metadata.get("depth_source_internal"),
                }
            )
            return build_shape_prior_response_parts(
                request_id=request_id,
                seq=seq,
                status="ready",
                points_m=points,
                colors_rgb_u8=colors,
                surface_points_m=surface_points,
                interior_points_m=interior_points,
                metadata=metadata,
            )
        except Exception as exc:
            return build_error_response_parts(
                request_id=request_id,
                seq=seq,
                error=str(exc),
                metadata=self.startup_metadata(),
            )


def _prepare_worker_startup(
    worker: ShapePriorSam3DWorker,
    *,
    preload_models: bool,
    warmup_models: bool,
) -> dict[str, Any]:
    startup_start_s = time.perf_counter()
    should_preload = bool(preload_models) or bool(warmup_models)
    if should_preload:
        worker.preload_models()
        worker._startup_metadata["worker_preloaded_models"] = True
    if bool(warmup_models):
        worker.run_dummy_warmup()
        worker._startup_metadata["worker_warmed_models"] = True
    worker._startup_metadata["worker_ready_ms"] = _elapsed_ms(startup_start_s)
    return worker.startup_metadata()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Long-lived remote SAM3D shape-prior worker for Demo 3.2.")
    parser.add_argument("--bind", default="tcp://0.0.0.0:7100", help="ZeroMQ REP bind endpoint.")
    parser.add_argument("--sam3d-root", type=Path, default=DEFAULT_SAM3D_ROOT)
    parser.add_argument("--futurephystwin-root", type=Path, default=DEFAULT_FUTUREPHYSTWIN_ROOT)
    parser.add_argument("--config", type=Path, default=None, help="SAM3D pipeline YAML. Defaults under --sam3d-root.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-points", type=int, default=60000)
    parser.add_argument(
        "--upscale-category",
        default=DEFAULT_UPSCALE_CATEGORY,
        help="Category text used in the data_process_sam3d x4 upscaler prompt.",
    )
    parser.add_argument(
        "--echo-observation",
        action="store_true",
        help="Protocol/debug mode: return first-frame object observation PCD without loading SAM3D.",
    )
    parser.add_argument(
        "--preload-models",
        action="store_true",
        help="Load the x4 upscaler and SAM3D inference model before binding the worker endpoint.",
    )
    parser.add_argument(
        "--warmup-models",
        action="store_true",
        help="Run a strict deterministic dummy upscaler + SAM3D + mesh-conversion warmup before ready.",
    )
    parser.add_argument("--debug", action="store_true")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if bool(args.warmup_models):
        args.preload_models = True
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    worker = ShapePriorSam3DWorker(
        sam3d_root=Path(args.sam3d_root),
        config=args.config,
        device=str(args.device),
        seed=int(args.seed),
        max_points=int(args.max_points),
        upscale_category=str(args.upscale_category),
        echo_observation=bool(args.echo_observation),
    )
    try:
        startup_metadata = _prepare_worker_startup(
            worker,
            preload_models=bool(args.preload_models),
            warmup_models=bool(args.warmup_models),
        )
    except Exception as exc:
        print(f"[shape-prior-worker] startup failed: {exc}", file=sys.stderr, flush=True)
        return 1
    import zmq

    context = zmq.Context.instance()
    socket = context.socket(zmq.REP)
    socket.bind(str(args.bind))
    print(
        "[shape-prior-worker] "
        f"ready bind={args.bind} sam3d_root={args.sam3d_root} "
        f"echo={args.echo_observation} preload={startup_metadata.get('worker_preloaded_models')} "
        f"warmup={startup_metadata.get('worker_warmed_models')} "
        f"worker_ready_ms={float(startup_metadata.get('worker_ready_ms', 0.0)):.1f}",
        flush=True,
    )
    while True:
        parts = socket.recv_multipart()
        recv_s = time.perf_counter()
        try:
            request = parse_shape_prior_request_parts(parts)
            reply = worker.handle(request)
        except Exception as exc:
            reply = build_error_response_parts(request_id="", seq=-1, error=str(exc))
        if bool(args.debug):
            metadata = {}
            try:
                import json

                metadata = json.loads(reply[0].decode("utf-8"))
            except Exception:
                pass
            print(
                "[shape-prior-worker] "
                f"seq={metadata.get('seq')} status={metadata.get('status')} "
                f"points={metadata.get('point_count')} total_ms={_elapsed_ms(recv_s):.1f} "
                f"error={metadata.get('error')}",
                flush=True,
            )
        socket.send_multipart(reply)


if __name__ == "__main__":
    raise SystemExit(main())
