#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


def _resolve_repo_root() -> Path:
    candidates = [Path(__file__).resolve().parents[2], Path.cwd()]
    env_root = os.environ.get("QQTT_REPO_ROOT")
    if env_root:
        candidates.insert(0, Path(env_root))
    for candidate in candidates:
        root = candidate.expanduser().resolve()
        if (root / "qqtt").is_dir() and (root / "services").is_dir():
            return root
    return Path(__file__).resolve().parents[2]


REPO_ROOT = _resolve_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qqtt.demo.shape_prior_warmup import DEFAULT_SHAPE_PRIOR_RENDER_RGB  # noqa: E402
from qqtt.demo.single_view_shape_align import (  # noqa: E402
    ShapeAlignmentConfig,
    align_canonical_shape_to_observation,
)
from services.shape_prior_remote.protocol import (  # noqa: E402
    ShapePriorRequest,
    build_error_response_parts,
    build_shape_prior_response_parts,
    parse_shape_prior_request_parts,
)


DEFAULT_SAM3D_ROOT = Path("/home/xinjie/external/sam-3d-objects")
DEFAULT_FUTUREPHYSTWIN_ROOT = Path("/home/xinjie/FuturePhysTwin")


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


def _object_observation_points_world(request: ShapePriorRequest, *, max_points: int) -> np.ndarray:
    mask = np.asarray(request.object_mask, dtype=bool)
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


class ShapePriorSam3DWorker:
    def __init__(
        self,
        *,
        sam3d_root: Path,
        config: Path | None,
        device: str,
        seed: int,
        max_points: int,
        echo_observation: bool = False,
    ) -> None:
        self.sam3d_root = Path(sam3d_root).expanduser()
        self.config = Path(config).expanduser() if config is not None else None
        self.device = str(device)
        self.seed = int(seed)
        self.max_points = int(max_points)
        self.echo_observation = bool(echo_observation)
        self._inference: Any | None = None
        self._model_load_ms = 0.0

    def _load_inference(self) -> Any:
        if self._inference is not None:
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
        self._model_load_ms = _elapsed_ms(start_s)
        return self._inference

    def _canonical_points_from_sam3d(self, request: ShapePriorRequest) -> tuple[np.ndarray, dict[str, Any]]:
        infer = self._load_inference()
        start_s = time.perf_counter()
        mask_u8 = (np.asarray(request.object_mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
        outputs = infer._pipeline.run(  # type: ignore[attr-defined]
            np.ascontiguousarray(request.rgb_u8, dtype=np.uint8),
            mask_u8,
            seed=self.seed,
            with_mesh_postprocess=True,
            with_texture_baking=False,
            with_layout_postprocess=True,
            use_vertex_color=False,
        )
        mesh_obj = outputs.get("glb", None)
        if mesh_obj is None:
            mesh_list = outputs.get("mesh", [])
            mesh_obj = mesh_list[0] if mesh_list else None
        if mesh_obj is None:
            raise RuntimeError("SAM3D output did not include a mesh/glb object")
        vertices = getattr(mesh_obj, "vertices", None)
        if vertices is None and hasattr(mesh_obj, "geometry"):
            vertices = getattr(mesh_obj.geometry, "vertices", None)
        if vertices is None:
            raise RuntimeError("SAM3D mesh output has no vertices")
        if hasattr(vertices, "detach"):
            vertices = vertices.detach().cpu().numpy()
        canonical = _sample_points(np.asarray(vertices, dtype=np.float32).reshape(-1, 3), self.max_points)
        if len(canonical) < 3:
            raise RuntimeError("SAM3D canonical mesh has fewer than 3 finite vertices")
        return canonical, {
            "sam3d_model_load_ms": float(self._model_load_ms),
            "sam3d_inference_ms": _elapsed_ms(start_s),
            "geometry_export_ms": 0.0,
        }

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
                metadata = {
                    "sam3d_model_load_ms": 0.0,
                    "sam3d_inference_ms": 0.0,
                    "geometry_export_ms": 0.0,
                    "single_view_alignment_ms": 0.0,
                    "sampling_ms": observation_ms,
                    "shape_prior_total_ms": _elapsed_ms(total_start_s),
                    "alignment_valid": True,
                    "echo_observation": True,
                }
            else:
                canonical, sam3d_stats = self._canonical_points_from_sam3d(request)
                align_start_s = time.perf_counter()
                aligned = align_canonical_shape_to_observation(
                    canonical,
                    observation,
                    config=ShapeAlignmentConfig(),
                )
                align_ms = _elapsed_ms(align_start_s)
                if not aligned.valid:
                    raise RuntimeError(f"shape-prior single-view alignment invalid: {aligned.validation}")
                points = _sample_points(aligned.aligned_points_m, self.max_points)
                metadata = dict(sam3d_stats)
                metadata.update(
                    {
                        "single_view_alignment_ms": align_ms,
                        "sampling_ms": observation_ms,
                        "shape_prior_total_ms": _elapsed_ms(total_start_s),
                        "alignment_valid": True,
                        "alignment": aligned.validation,
                    }
                )
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
                metadata=metadata,
            )
        except Exception as exc:
            return build_error_response_parts(request_id=request_id, seq=seq, error=str(exc))


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
        "--echo-observation",
        action="store_true",
        help="Protocol/debug mode: return first-frame object observation PCD without loading SAM3D.",
    )
    parser.add_argument("--debug", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    worker = ShapePriorSam3DWorker(
        sam3d_root=Path(args.sam3d_root),
        config=args.config,
        device=str(args.device),
        seed=int(args.seed),
        max_points=int(args.max_points),
        echo_observation=bool(args.echo_observation),
    )
    import zmq

    context = zmq.Context.instance()
    socket = context.socket(zmq.REP)
    socket.bind(str(args.bind))
    print(f"[shape-prior-worker] bind={args.bind} sam3d_root={args.sam3d_root} echo={args.echo_observation}", flush=True)
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
                f"points={metadata.get('point_count')} total_ms={_elapsed_ms(recv_s):.1f}",
                flush=True,
            )
        socket.send_multipart(reply)


if __name__ == "__main__":
    raise SystemExit(main())
