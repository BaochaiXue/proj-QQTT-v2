#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_OUTPUT_DIR = ROOT / "docs/generated/demo31_tapnextpp_onnx_trt_feasibility"
DEFAULT_ARTIFACT_DIR = ROOT / "data/cache/demo31_tapnextpp_onnx_trt_feasibility"
DEFAULT_TAPNET_REPO = ROOT / "external/tapnet"
DEFAULT_TAPNEXTPP_CHECKPOINT = ROOT / "checkpoints/tapnextpp/tapnextpp_ckpt.pt"

TAPNEXT_WIDTH = 768
TAPNEXT_DEPTH = 12
TAPNEXT_PATCH_SIZE = 8
TAPNEXT_CONV1D_WIDTH = 3


def _csv_ints(value: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        return tuple(int(part.strip()) for part in value.split(",") if part.strip())
    return tuple(int(item) for item in value)


def _parse_image_size(value: str | Sequence[int]) -> tuple[int, int]:
    if isinstance(value, str):
        raw = value.strip().lower().replace("x", ",")
        parts = [int(part.strip()) for part in raw.split(",") if part.strip()]
    else:
        parts = [int(item) for item in value]
    if len(parts) == 1:
        return (parts[0], parts[0])
    if len(parts) == 2:
        return (parts[0], parts[1])
    raise argparse.ArgumentTypeError("--image-size must be H,W, HxW, or a square size.")


def _format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{float(num_bytes):.2f} B"


def _module_status(name: str) -> dict[str, Any]:
    spec = importlib.util.find_spec(name)
    status: dict[str, Any] = {"module": name, "available": spec is not None, "version": ""}
    if spec is None:
        return status
    try:
        module = __import__(name)
        status["version"] = str(getattr(module, "__version__", ""))
    except Exception as exc:
        status["available"] = False
        status["import_error"] = f"{type(exc).__name__}: {exc}"
    return status


def _glob_env_libs() -> dict[str, list[str]]:
    prefixes = [Path(sys.prefix)]
    patterns = {
        "cudnn": [
            "lib/libcudnn.so*",
            "lib/python*/site-packages/nvidia/cudnn/lib/libcudnn.so*",
        ],
        "tensorrt": [
            "lib/libnvinfer.so*",
            "lib/python*/site-packages/tensorrt_libs/libnvinfer.so*",
        ],
        "cuda_runtime": [
            "lib/libcudart.so*",
            "lib/python*/site-packages/nvidia/cuda_runtime/lib/libcudart.so*",
        ],
    }
    found: dict[str, list[str]] = {}
    for key, pats in patterns.items():
        values: list[str] = []
        for prefix in prefixes:
            for pattern in pats:
                values.extend(str(path) for path in prefix.glob(pattern))
        found[key] = sorted(set(values))
    return found


def recommended_gpu_library_dirs() -> list[str]:
    libs = _glob_env_libs()
    dirs: list[str] = []
    for key in ("cudnn", "tensorrt", "cuda_runtime"):
        for path in libs.get(key, []):
            dirs.append(str(Path(path).parent))
    dirs.append(str(Path(sys.prefix) / "lib"))
    return sorted(dict.fromkeys(dirs))


def apply_recommended_gpu_library_path() -> str:
    current = [part for part in os.environ.get("LD_LIBRARY_PATH", "").split(":") if part]
    merged = recommended_gpu_library_dirs() + current
    value = ":".join(dict.fromkeys(merged))
    os.environ["LD_LIBRARY_PATH"] = value
    return value


def detect_runtime_stack() -> dict[str, Any]:
    stack: dict[str, Any] = {
        "python": sys.version.split()[0],
        "prefix": sys.prefix,
        "modules": {name: _module_status(name) for name in ("torch", "onnx", "onnxruntime", "tensorrt", "torch_tensorrt")},
        "libraries": _glob_env_libs(),
        "recommended_ld_library_path": ":".join(recommended_gpu_library_dirs()),
        "current_ld_library_path": os.environ.get("LD_LIBRARY_PATH", ""),
    }
    try:
        import torch

        stack["torch_cuda"] = {
            "torch_version": str(torch.__version__),
            "cuda_version": str(torch.version.cuda),
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
            "devices": [torch.cuda.get_device_name(idx) for idx in range(torch.cuda.device_count())],
        }
    except Exception as exc:
        stack["torch_cuda"] = {"error": f"{type(exc).__name__}: {exc}"}
    try:
        import onnxruntime as ort

        providers = list(ort.get_available_providers())
        stack["onnxruntime_providers"] = providers
        stack["onnxruntime_cuda_listed"] = "CUDAExecutionProvider" in providers
        stack["onnxruntime_tensorrt_listed"] = "TensorrtExecutionProvider" in providers
    except Exception as exc:
        stack["onnxruntime_providers"] = []
        stack["onnxruntime_error"] = f"{type(exc).__name__}: {exc}"
    return stack


def estimate_tapnext_state_bytes(*, batch_size: int, query_count: int, image_size: tuple[int, int]) -> dict[str, Any]:
    patch_tokens = int(image_size[0] // TAPNEXT_PATCH_SIZE) * int(image_size[1] // TAPNEXT_PATCH_SIZE)
    tokens_per_batch = int(patch_tokens) + int(query_count)
    total_tokens = int(batch_size) * int(tokens_per_batch)
    rg_lru_state_bytes_per_layer = total_tokens * TAPNEXT_WIDTH * 4
    conv1d_state_bytes_per_layer = total_tokens * TAPNEXT_CONV1D_WIDTH * TAPNEXT_WIDTH * 2
    hidden_state_bytes_per_layer = rg_lru_state_bytes_per_layer + conv1d_state_bytes_per_layer
    hidden_state_bytes = hidden_state_bytes_per_layer * TAPNEXT_DEPTH
    query_points_bytes = int(batch_size) * int(query_count) * 3 * 4
    tracks_bytes = int(batch_size) * int(query_count) * 2 * 4
    visible_bytes = int(batch_size) * int(query_count) * 2
    return {
        "batch_size": int(batch_size),
        "query_count_per_view": int(query_count),
        "total_query_count": int(batch_size) * int(query_count),
        "image_size": [int(image_size[0]), int(image_size[1])],
        "patch_tokens": int(patch_tokens),
        "tokens_per_batch": int(tokens_per_batch),
        "total_tokens": int(total_tokens),
        "state_tensor_count": int(TAPNEXT_DEPTH * 2),
        "rg_lru_state_bytes_per_layer": int(rg_lru_state_bytes_per_layer),
        "conv1d_state_bytes_per_layer": int(conv1d_state_bytes_per_layer),
        "hidden_state_bytes": int(hidden_state_bytes),
        "hidden_state_mib": float(hidden_state_bytes / (1024.0 * 1024.0)),
        "state_input_output_bytes_min": int(hidden_state_bytes * 2),
        "state_input_output_mib_min": float((hidden_state_bytes * 2) / (1024.0 * 1024.0)),
        "query_points_bytes": int(query_points_bytes),
        "tracks_output_bytes": int(tracks_bytes),
        "visible_output_bytes": int(visible_bytes),
        "hidden_state_human": _format_bytes(hidden_state_bytes),
        "state_input_output_human_min": _format_bytes(hidden_state_bytes * 2),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe Demo 3.1 TAPNext++ ONNX/TensorRT feasibility without touching the live backend.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--tapnet-repo-dir", type=Path, default=DEFAULT_TAPNET_REPO)
    parser.add_argument("--tapnextpp-checkpoint", type=Path, default=DEFAULT_TAPNEXTPP_CHECKPOINT)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-sizes", type=_csv_ints, default=(1, 3), help="Small actual probe batch sizes.")
    parser.add_argument("--query-counts", type=_csv_ints, default=(8,), help="Small actual probe query counts.")
    parser.add_argument("--target-query-counts", type=_csv_ints, default=(1365, 4096), help="Target/stress byte estimates.")
    parser.add_argument("--target-batch-size", type=int, default=3)
    parser.add_argument("--image-size", type=_parse_image_size, default=(256, 256))
    parser.add_argument("--autocast-dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--skip-model-load", action="store_true")
    parser.add_argument("--attempt-torch-export", action="store_true")
    parser.add_argument("--onnx-export-mode", choices=("none", "const-state", "flat-state", "both"), default="none")
    parser.add_argument("--attempt-ort-session", action="store_true")
    parser.add_argument("--attempt-trt-session", action="store_true")
    parser.add_argument("--ort-session-timeout-s", type=float, default=45.0)
    parser.add_argument("--use-recommended-gpu-lib-paths", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    return parser


def _prepend_tapnet_repo(repo_dir: Path) -> None:
    repo = Path(repo_dir).expanduser().resolve()
    path = str(repo)
    if path not in sys.path:
        sys.path.insert(0, path)


def _points(query_count: int, image_size: tuple[int, int]) -> np.ndarray:
    height, width = int(image_size[0]), int(image_size[1])
    cols = int(np.ceil(np.sqrt(float(query_count) * float(width) / float(max(height, 1)))))
    rows = int(np.ceil(float(query_count) / float(max(cols, 1))))
    ys = np.linspace(4, max(height - 5, 4), max(rows, 1), dtype=np.float32)
    xs = np.linspace(4, max(width - 5, 4), max(cols, 1), dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    return np.ascontiguousarray(np.stack([yy.reshape(-1), xx.reshape(-1)], axis=1)[:query_count], dtype=np.float32)


def _frame(rng: np.random.Generator, image_size: tuple[int, int]) -> np.ndarray:
    return np.ascontiguousarray(
        rng.integers(0, 255, size=(int(image_size[0]), int(image_size[1]), 3), dtype=np.uint8),
        dtype=np.uint8,
    )


def _tensor_bytes(tensor: Any) -> int:
    try:
        return int(tensor.numel()) * int(tensor.element_size())
    except Exception:
        return 0


def flatten_tapnext_state(state: Any) -> list[Any]:
    flat: list[Any] = []
    for cache in getattr(state, "hidden_state", []) or []:
        flat.append(cache.rg_lru_state)
        flat.append(cache.conv1d_state)
    return flat


def summarize_state(state: Any) -> dict[str, Any]:
    flat = flatten_tapnext_state(state)
    rg_shapes: list[list[int]] = []
    conv_shapes: list[list[int]] = []
    rg_bytes = 0
    conv_bytes = 0
    for idx in range(0, len(flat), 2):
        rg = flat[idx]
        conv = flat[idx + 1]
        rg_shapes.append([int(item) for item in tuple(rg.shape)])
        conv_shapes.append([int(item) for item in tuple(conv.shape)])
        rg_bytes += _tensor_bytes(rg)
        conv_bytes += _tensor_bytes(conv)
    total_bytes = rg_bytes + conv_bytes
    return {
        "state_type": type(state).__name__,
        "step": int(getattr(state, "step", 0)),
        "query_points_shape": [int(item) for item in tuple(getattr(state, "query_points").shape)],
        "query_points_dtype": str(getattr(state, "query_points").dtype),
        "query_points_device": str(getattr(state, "query_points").device),
        "hidden_layer_count": int(len(getattr(state, "hidden_state", []) or [])),
        "flat_state_tensor_count": int(len(flat)),
        "rg_lru_state_shapes_first_last": [rg_shapes[0], rg_shapes[-1]] if rg_shapes else [],
        "conv1d_state_shapes_first_last": [conv_shapes[0], conv_shapes[-1]] if conv_shapes else [],
        "rg_lru_state_dtype": str(flat[0].dtype) if flat else "",
        "conv1d_state_dtype": str(flat[1].dtype) if len(flat) > 1 else "",
        "rg_lru_state_bytes": int(rg_bytes),
        "conv1d_state_bytes": int(conv_bytes),
        "hidden_state_bytes": int(total_bytes),
        "hidden_state_human": _format_bytes(total_bytes),
        "state_input_output_bytes_min": int(total_bytes * 2),
        "state_input_output_human_min": _format_bytes(total_bytes * 2),
    }


def _load_adapter(args: argparse.Namespace):
    from qqtt.tracking.backends.tapnextpp_adapter import TAPNextPPAdapter

    return TAPNextPPAdapter(
        device=str(args.device),
        repo_dir=str(args.tapnet_repo_dir),
        checkpoint=str(args.tapnextpp_checkpoint),
        image_size=tuple(int(item) for item in args.image_size),
        autocast_dtype=str(args.autocast_dtype),
        fast_postprocess=True,
    )


def _build_case(adapter: Any, *, batch_size: int, query_count: int, image_size: tuple[int, int], seed: int) -> tuple[Any, Any]:
    rng = np.random.default_rng(int(seed))
    frames = [_frame(rng, image_size) for _ in range(int(batch_size))]
    video, source_shape = adapter._frames_to_video_tensor(frames, camera_ids=tuple(range(int(batch_size))))
    points = _points(int(query_count), image_size)
    query_one = adapter._queries_yx_to_tyx_tensor(points, source_shape_hw=source_shape)
    query = query_one[None].repeat(int(batch_size), 1, 1).contiguous()
    return video, query


def _run_initial_and_recurrent(adapter: Any, *, video: Any, query: Any) -> tuple[Any, Any, Any]:
    import torch

    model = adapter._load_model()
    with torch.no_grad(), adapter._autocast_context():
        first = model(video=video, query_points=query)
        recurrent = model(video=video, state=first[3])
    adapter._sync_cuda_if_needed()
    return model, first, recurrent


def _max_abs(value: Any) -> float:
    try:
        return float(value.detach().abs().max().cpu().item())
    except Exception:
        return 0.0


def _step_invariance_probe(adapter: Any, *, model: Any, video: Any, state: Any) -> dict[str, Any]:
    import torch
    from tapnet.tapnext.tapnext_torch import TAPNextTrackingState

    if int(getattr(state, "step", 0)) < 1:
        return {"attempted": False, "reason": "state.step must be >= 1"}
    state_one = TAPNextTrackingState(step=1, query_points=state.query_points, hidden_state=state.hidden_state)
    state_five = TAPNextTrackingState(step=5, query_points=state.query_points, hidden_state=state.hidden_state)
    with torch.no_grad(), adapter._autocast_context():
        out_one = model(video=video, state=state_one)
        out_five = model(video=video, state=state_five)
    adapter._sync_cuda_if_needed()
    track_diff = _max_abs(out_one[0] - out_five[0])
    visible_diff = _max_abs(out_one[2].float() - out_five[2].float())
    return {
        "attempted": True,
        "step_one": 1,
        "step_five": 5,
        "max_track_abs_diff": float(track_diff),
        "max_visible_logit_abs_diff": float(visible_diff),
        "same_outputs": bool(track_diff == 0.0 and visible_diff == 0.0),
        "note": "For recurrent steps after initialization, step only keeps query t negative; this smoke checks fixed-step wrappers.",
    }


class _ConstStateWrapper:
    pass


def _make_const_state_wrapper(model: Any, state: Any):
    import torch

    class ConstStateWrapper(torch.nn.Module):
        def __init__(self, wrapped: Any, frozen_state: Any) -> None:
            super().__init__()
            self.wrapped = wrapped
            self.frozen_state = frozen_state

        def forward(self, video: Any) -> tuple[Any, Any]:
            tracks, _track_logits, visible_logits, _state = self.wrapped(video=video, state=self.frozen_state)
            return tracks, visible_logits

    return ConstStateWrapper(model, state).eval()


def _make_flat_state_wrapper(model: Any):
    import torch
    from tapnet.tapnext.tapnext_lru_modules import RecurrentBlockCache
    from tapnet.tapnext.tapnext_torch import TAPNextTrackingState

    class FlatStateWrapper(torch.nn.Module):
        def __init__(self, wrapped: Any) -> None:
            super().__init__()
            self.wrapped = wrapped

        def forward(self, video: Any, query_points: Any, *flat_state: Any) -> tuple[Any, ...]:
            hidden = []
            for idx in range(0, len(flat_state), 2):
                hidden.append(
                    RecurrentBlockCache(
                        rg_lru_state=flat_state[idx],
                        conv1d_state=flat_state[idx + 1],
                    )
                )
            state = TAPNextTrackingState(step=1, query_points=query_points, hidden_state=hidden)
            tracks, _track_logits, visible_logits, next_state = self.wrapped(video=video, state=state)
            next_flat: list[Any] = []
            for cache in next_state.hidden_state:
                next_flat.append(cache.rg_lru_state)
                next_flat.append(cache.conv1d_state)
            return (tracks, visible_logits, *next_flat)

    return FlatStateWrapper(model).eval()


def _onnx_file_sizes(path: Path) -> dict[str, Any]:
    data_path = Path(str(path) + ".data")
    total = int(path.stat().st_size) if path.exists() else 0
    data = int(data_path.stat().st_size) if data_path.exists() else 0
    return {
        "onnx_path": str(path),
        "onnx_bytes": total,
        "external_data_path": str(data_path) if data_path.exists() else "",
        "external_data_bytes": data,
        "total_artifact_bytes": total + data,
        "total_artifact_human": _format_bytes(total + data),
    }


def _inspect_onnx(path: Path) -> dict[str, Any]:
    try:
        import onnx

        model = onnx.load(str(path), load_external_data=False)
        counts = Counter(node.op_type for node in model.graph.node)
        equations: list[str] = []
        for node in model.graph.node:
            if node.op_type == "Einsum":
                for attr in node.attribute:
                    if attr.name == "equation":
                        equations.append(attr.s.decode("utf-8", errors="replace"))
        return {
            "ok": True,
            "node_count": int(len(model.graph.node)),
            "initializer_count": int(len(model.graph.initializer)),
            "input_count": int(len(model.graph.input)),
            "output_count": int(len(model.graph.output)),
            "op_counts_top": dict(counts.most_common(20)),
            "einsum_equations": sorted(set(equations)),
            "contains_uppercase_einsum": any(any(ch.isupper() for ch in eq) for eq in equations),
        }
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _export_onnx(
    *,
    mode: str,
    model: Any,
    state: Any,
    video: Any,
    artifact_dir: Path,
    batch_size: int,
    query_count: int,
    opset: int,
) -> dict[str, Any]:
    import torch

    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / f"tapnextpp_{mode}_b{int(batch_size)}_q{int(query_count)}.onnx"
    started = time.perf_counter()
    try:
        if mode == "const-state":
            wrapper = _make_const_state_wrapper(model, state)
            torch.onnx.export(
                wrapper,
                (video,),
                str(path),
                opset_version=int(opset),
                input_names=["video"],
                output_names=["tracks", "visible_logits"],
                dynamo=True,
            )
        elif mode == "flat-state":
            wrapper = _make_flat_state_wrapper(model)
            flat = flatten_tapnext_state(state)
            torch.onnx.export(
                wrapper,
                (video, state.query_points, *flat),
                str(path),
                opset_version=int(opset),
                input_names=["video", "query_points"] + [f"state_{idx}" for idx in range(len(flat))],
                dynamo=True,
            )
        else:
            raise ValueError(f"unsupported ONNX export mode: {mode}")
        elapsed_ms = float((time.perf_counter() - started) * 1000.0)
        return {
            "mode": mode,
            "status": "ok",
            "export_ms": elapsed_ms,
            **_onnx_file_sizes(path),
            "onnx_inspection": _inspect_onnx(path),
        }
    except Exception as exc:
        return {
            "mode": mode,
            "status": "fail",
            "export_ms": float((time.perf_counter() - started) * 1000.0),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback_tail": traceback.format_exc(limit=4).splitlines()[-12:],
        }


def _torch_export_probe(model: Any, *, video: Any, state: Any) -> dict[str, Any]:
    import torch

    started = time.perf_counter()
    try:
        exported = torch.export.export(model, (), kwargs={"video": video, "state": state}, strict=False)
        return {
            "status": "ok",
            "export_ms": float((time.perf_counter() - started) * 1000.0),
            "node_count": int(len(list(exported.graph_module.graph.nodes))),
        }
    except Exception as exc:
        return {
            "status": "fail",
            "export_ms": float((time.perf_counter() - started) * 1000.0),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback_tail": traceback.format_exc(limit=4).splitlines()[-12:],
        }


def _ort_session_worker(
    onnx_path: str,
    provider_kind: str,
    engine_cache_path: str,
    trt_fp16: bool,
    use_recommended_paths: bool,
    queue: Any,
) -> None:
    if use_recommended_paths:
        apply_recommended_gpu_library_path()
    try:
        import onnxruntime as ort

        if provider_kind == "trt":
            providers: list[Any] = [
                (
                    "TensorrtExecutionProvider",
                    {
                        "trt_fp16_enable": bool(trt_fp16),
                        "trt_engine_cache_enable": True,
                        "trt_engine_cache_path": str(engine_cache_path),
                    },
                ),
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]
        elif provider_kind == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        started = time.perf_counter()
        session = ort.InferenceSession(str(onnx_path), providers=providers)
        queue.put(
            {
                "provider_kind": provider_kind,
                "status": "ok",
                "session_create_ms": float((time.perf_counter() - started) * 1000.0),
                "available_providers": list(ort.get_available_providers()),
                "actual_providers": list(session.get_providers()),
                "input_count": int(len(session.get_inputs())),
                "output_count": int(len(session.get_outputs())),
            }
        )
    except Exception as exc:
        queue.put(
            {
                "provider_kind": provider_kind,
                "status": "fail",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback_tail": traceback.format_exc(limit=4).splitlines()[-12:],
            }
        )


def _probe_ort_session(
    *,
    onnx_path: Path,
    provider_kind: str,
    artifact_dir: Path,
    timeout_s: float,
    use_recommended_paths: bool,
) -> dict[str, Any]:
    context = mp.get_context("spawn")
    queue: Any = context.Queue()
    process = context.Process(
        target=_ort_session_worker,
        args=(str(onnx_path), provider_kind, str(artifact_dir / "trt_engine_cache"), True, bool(use_recommended_paths), queue),
    )
    started = time.perf_counter()
    process.start()
    process.join(float(timeout_s))
    elapsed_ms = float((time.perf_counter() - started) * 1000.0)
    if process.is_alive():
        process.terminate()
        process.join(5)
        return {
            "provider_kind": provider_kind,
            "status": "timeout",
            "timeout_s": float(timeout_s),
            "elapsed_ms": elapsed_ms,
            "note": "Session construction did not complete inside the timeout.",
        }
    if not queue.empty():
        result = queue.get()
        result["elapsed_ms"] = elapsed_ms
        return result
    return {
        "provider_kind": provider_kind,
        "status": "fail",
        "elapsed_ms": elapsed_ms,
        "error": f"worker exited with code {process.exitcode} without a result",
    }


def _case_payload(
    *,
    args: argparse.Namespace,
    adapter: Any,
    batch_size: int,
    query_count: int,
) -> dict[str, Any]:
    image_size = tuple(int(item) for item in args.image_size)
    video, query = _build_case(adapter, batch_size=int(batch_size), query_count=int(query_count), image_size=image_size, seed=int(args.seed))
    model, first, recurrent = _run_initial_and_recurrent(adapter, video=video, query=query)
    state = first[3]
    recurrent_state = recurrent[3]
    case: dict[str, Any] = {
        "batch_size": int(batch_size),
        "query_count_per_view": int(query_count),
        "total_query_count": int(batch_size) * int(query_count),
        "image_size": [int(image_size[0]), int(image_size[1])],
        "video_shape": [int(item) for item in tuple(video.shape)],
        "query_shape": [int(item) for item in tuple(query.shape)],
        "first_tracks_shape": [int(item) for item in tuple(first[0].shape)],
        "recurrent_tracks_shape": [int(item) for item in tuple(recurrent[0].shape)],
        "visible_logits_shape": [int(item) for item in tuple(recurrent[2].shape)],
        "state_after_first": summarize_state(state),
        "state_after_recurrent": summarize_state(recurrent_state),
        "step_invariance": _step_invariance_probe(adapter, model=model, video=video, state=recurrent_state),
    }
    if bool(args.attempt_torch_export):
        case["torch_export_recurrent_state"] = _torch_export_probe(model, video=video, state=state)
    export_modes: list[str] = []
    if args.onnx_export_mode == "both":
        export_modes = ["const-state", "flat-state"]
    elif args.onnx_export_mode != "none":
        export_modes = [str(args.onnx_export_mode)]
    onnx_exports = []
    for mode in export_modes:
        export = _export_onnx(
            mode=mode,
            model=model,
            state=state,
            video=video,
            artifact_dir=Path(args.artifact_dir),
            batch_size=int(batch_size),
            query_count=int(query_count),
            opset=int(args.opset),
        )
        if export.get("status") == "ok" and (bool(args.attempt_ort_session) or bool(args.attempt_trt_session)):
            path = Path(str(export.get("onnx_path", "")))
            sessions = []
            if bool(args.attempt_ort_session):
                sessions.append(
                    _probe_ort_session(
                        onnx_path=path,
                        provider_kind="cuda",
                        artifact_dir=Path(args.artifact_dir),
                        timeout_s=float(args.ort_session_timeout_s),
                        use_recommended_paths=bool(args.use_recommended_gpu_lib_paths),
                    )
                )
            if bool(args.attempt_trt_session):
                sessions.append(
                    _probe_ort_session(
                        onnx_path=path,
                        provider_kind="trt",
                        artifact_dir=Path(args.artifact_dir),
                        timeout_s=float(args.ort_session_timeout_s),
                        use_recommended_paths=bool(args.use_recommended_gpu_lib_paths),
                    )
                )
            export["ort_sessions"] = sessions
        onnx_exports.append(export)
    if onnx_exports:
        case["onnx_exports"] = onnx_exports
    return case


def _derive_conclusion(payload: dict[str, Any]) -> dict[str, Any]:
    cases = payload.get("actual_probe_cases", [])
    flat_exports = [
        export
        for case in cases
        for export in case.get("onnx_exports", [])
        if export.get("mode") == "flat-state"
    ]
    flat_ok = any(export.get("status") == "ok" for export in flat_exports)
    trt_sessions = [
        session
        for export in flat_exports
        for session in export.get("ort_sessions", [])
        if session.get("provider_kind") == "trt"
    ]
    trt_ready = any("TensorrtExecutionProvider" in session.get("actual_providers", []) for session in trt_sessions if session.get("status") == "ok")
    trt_timeout = any(session.get("status") == "timeout" for session in trt_sessions)
    stack = payload.get("runtime_stack", {})
    torch_trt_available = bool(stack.get("modules", {}).get("torch_tensorrt", {}).get("available"))
    if trt_ready:
        status = "trt_session_ready_for_tiny_probe"
    elif flat_ok and (trt_timeout or trt_sessions):
        status = "exportable_but_trt_session_not_ready"
    elif flat_ok:
        status = "flat_state_onnx_exportable_but_runtime_unproven"
    elif cases:
        status = "state_probe_ok_but_export_unproven"
    else:
        status = "stack_only"
    blockers: list[str] = []
    if not torch_trt_available:
        blockers.append("torch_tensorrt is not installed in demo_3_1_max.")
    providers = stack.get("onnxruntime_providers", [])
    if "TensorrtExecutionProvider" not in providers:
        blockers.append("ONNX Runtime TensorRT EP is not listed.")
    if flat_exports:
        for export in flat_exports:
            inspection = export.get("onnx_inspection", {})
            if inspection.get("contains_uppercase_einsum"):
                blockers.append("Exported ONNX contains uppercase Einsum equations such as ...td,cdD->c...tD; TensorRT importer rejects these.")
                break
    return {
        "status": status,
        "safe_for_live_runtime": False,
        "do_not_integrate_by_default": True,
        "blockers": sorted(set(blockers)),
        "recommendation": (
            "Keep Demo 3.1 on the existing PyTorch TAPNext++ backend. "
            "The plausible next step is a fixed-shape recurrent-cell lowering that rewrites TAPNext++ MLP Einsum into TensorRT-friendly matmul/linear ops, "
            "then rebuilds a flat-state ONNX/TRT probe for B=3,q1365 before any runtime integration."
        ),
    }


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    conclusion = payload.get("conclusion", {})
    stack = payload.get("runtime_stack", {})
    lines = [
        "# Demo 3.1 TAPNext++ ONNX/TRT Feasibility",
        "",
        "This is an isolated model-only probe. It does not modify or enable ONNX/TRT in the live Demo 3.1 TAPNext++ backend.",
        "",
        f"- Status: `{conclusion.get('status', 'unknown')}`",
        f"- Live runtime changed: `False`",
        f"- Recommendation: {conclusion.get('recommendation', '')}",
        "",
        "## Runtime Stack",
        "",
        f"- Torch: `{stack.get('modules', {}).get('torch', {}).get('version', '')}`",
        f"- ONNX: `{stack.get('modules', {}).get('onnx', {}).get('version', '')}`",
        f"- ONNX Runtime: `{stack.get('modules', {}).get('onnxruntime', {}).get('version', '')}`",
        f"- TensorRT Python: `{stack.get('modules', {}).get('tensorrt', {}).get('version', '')}`",
        f"- Torch-TensorRT: `{'available' if stack.get('modules', {}).get('torch_tensorrt', {}).get('available') else 'missing'}`",
        f"- ORT providers listed: `{', '.join(stack.get('onnxruntime_providers', []))}`",
        "",
        "## State Size Estimates",
        "",
        "| Case | Total Points | Hidden State | Min State I/O Per Step |",
        "| --- | ---: | ---: | ---: |",
    ]
    for estimate in payload.get("target_state_estimates", []):
        label = "q1365/view target" if int(estimate.get("query_count_per_view", 0)) == 1365 else "q4096/view stress"
        lines.append(
            f"| {label} | {int(estimate.get('total_query_count', 0))} | {estimate.get('hidden_state_human', '')} | {estimate.get('state_input_output_human_min', '')} |"
        )
    lines.extend(["", "## Actual Small Probes", ""])
    for case in payload.get("actual_probe_cases", []):
        lines.append(
            f"- B={case.get('batch_size')} q={case.get('query_count_per_view')}: "
            f"state `{case.get('state_after_first', {}).get('hidden_state_human', '')}`, "
            f"flat tensors `{case.get('state_after_first', {}).get('flat_state_tensor_count', 0)}`, "
            f"step-invariance `{case.get('step_invariance', {}).get('same_outputs', 'n/a')}`."
        )
        if "torch_export_recurrent_state" in case:
            export = case["torch_export_recurrent_state"]
            lines.append(f"  Torch export recurrent state: `{export.get('status')}` nodes `{export.get('node_count', '')}`.")
        for export in case.get("onnx_exports", []):
            inspection = export.get("onnx_inspection", {})
            lines.append(
                f"  ONNX `{export.get('mode')}`: `{export.get('status')}`, "
                f"artifact `{export.get('total_artifact_human', '')}`, "
                f"nodes `{inspection.get('node_count', '')}`, "
                f"Einsum `{', '.join(inspection.get('einsum_equations', [])[:3])}`."
            )
            for session in export.get("ort_sessions", []):
                lines.append(
                    f"  ORT `{session.get('provider_kind')}` session: `{session.get('status')}`, "
                    f"actual providers `{', '.join(session.get('actual_providers', []))}`, "
                    f"elapsed `{session.get('elapsed_ms', 0.0):.1f}ms`."
                )
    if conclusion.get("blockers"):
        lines.extend(["", "## Blockers", ""])
        for blocker in conclusion.get("blockers", []):
            lines.append(f"- {blocker}")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- q1365/view is the 4095-total-point target. q4096/view is a 12288-total-point stress case.",
            "- A deployable engine must keep TAPNext++ quality by carrying the recurrent state as inputs and outputs.",
            "- Constant-state ONNX exports are useful only as an operator translation smoke test; they are not a live tracker.",
            "- The existing PyTorch Demo 3.1 path should stay the default until a flat-state engine is both correct and faster.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    if bool(args.use_recommended_gpu_lib_paths):
        apply_recommended_gpu_library_path()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_size = tuple(int(item) for item in args.image_size)
    payload: dict[str, Any] = {
        "probe": "demo31_tapnextpp_onnx_trt_feasibility",
        "live_runtime_changed": False,
        "tapnet_repo_dir": str(args.tapnet_repo_dir),
        "tapnextpp_checkpoint": str(args.tapnextpp_checkpoint),
        "device": str(args.device),
        "image_size": [int(image_size[0]), int(image_size[1])],
        "autocast_dtype": str(args.autocast_dtype),
        "runtime_stack": detect_runtime_stack(),
        "target_state_estimates": [
            estimate_tapnext_state_bytes(
                batch_size=int(args.target_batch_size),
                query_count=int(query_count),
                image_size=image_size,
            )
            for query_count in _csv_ints(args.target_query_counts)
        ],
        "actual_probe_cases": [],
    }
    if not bool(args.skip_model_load):
        _prepend_tapnet_repo(Path(args.tapnet_repo_dir))
        adapter = _load_adapter(args)
        for batch_size in _csv_ints(args.batch_sizes):
            for query_count in _csv_ints(args.query_counts):
                payload["actual_probe_cases"].append(
                    _case_payload(
                        args=args,
                        adapter=adapter,
                        batch_size=int(batch_size),
                        query_count=int(query_count),
                    )
                )
    payload["conclusion"] = _derive_conclusion(payload)
    output_json = output_dir / "summary.json"
    output_md = output_dir / "summary.md"
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    _write_markdown(output_md, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    payload = run_probe(args)
    print(json.dumps(payload["conclusion"], indent=2, sort_keys=True, default=str))
    print(Path(args.output_dir) / "summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
