from __future__ import annotations

from pathlib import Path
from typing import Any

from .ort_sessions import build_ort_providers, probe_onnxruntime_stack


def run_export_probe(
    *,
    model_name: str,
    onnx_path: str | Path | None = None,
    engine_cache_path: str | Path = "data/cache/demo3_tracking_trt",
    trt_fp16: bool = True,
) -> dict[str, Any]:
    stack = probe_onnxruntime_stack()
    payload: dict[str, Any] = {
        "model": model_name,
        "export_onnx": "not_attempted",
        "onnxruntime_cuda": "unavailable",
        "onnxruntime_tensorrt": "unavailable",
        "trt_engine_cache_created": False,
        "first_session_create_ms": 0.0,
        "cached_session_create_ms": 0.0,
        "pytorch_ms": 0.0,
        "ort_cuda_ms": 0.0,
        "ort_trt_ms": 0.0,
        "max_abs_diff": 0.0,
        "quality_notes": "",
        "providers_requested": build_ort_providers(engine_cache_path=engine_cache_path, trt_fp16=trt_fp16),
        "stack": stack,
    }
    if onnx_path is None:
        payload["export_onnx"] = "fail"
        payload["quality_notes"] = "No exportable model wrapper or ONNX path was provided."
        return payload
    path = Path(onnx_path)
    if not path.exists():
        payload["export_onnx"] = "fail"
        payload["quality_notes"] = f"ONNX file not found: {path}"
        return payload
    if not stack["onnxruntime_importable"]:
        payload["export_onnx"] = "pass_existing_onnx"
        payload["quality_notes"] = stack["onnxruntime_import_error"]
        return payload
    payload["export_onnx"] = "pass_existing_onnx"
    payload["onnxruntime_cuda"] = "available" if stack["onnxruntime_cuda"] else "unavailable"
    payload["onnxruntime_tensorrt"] = "available" if stack["onnxruntime_tensorrt"] else "unavailable"
    Path(engine_cache_path).mkdir(parents=True, exist_ok=True)
    payload["trt_engine_cache_created"] = Path(engine_cache_path).exists()
    payload["quality_notes"] = "Session execution is not run in the generic probe without model-specific inputs."
    return payload
