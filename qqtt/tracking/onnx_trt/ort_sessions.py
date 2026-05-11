from __future__ import annotations

from pathlib import Path
from typing import Any


def tensorrt_provider_options(
    *,
    engine_cache_path: str | Path = "data/cache/demo3_tracking_trt",
    fp16: bool = True,
    timing_cache: bool = True,
) -> dict[str, Any]:
    return {
        "trt_fp16_enable": bool(fp16),
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": str(engine_cache_path),
        "trt_timing_cache_enable": bool(timing_cache),
    }


def build_ort_providers(
    *,
    engine_cache_path: str | Path = "data/cache/demo3_tracking_trt",
    trt_fp16: bool = True,
    include_tensorrt: bool = True,
    include_cuda: bool = True,
    include_cpu: bool = True,
) -> list[Any]:
    providers: list[Any] = []
    if include_tensorrt:
        providers.append(("TensorrtExecutionProvider", tensorrt_provider_options(engine_cache_path=engine_cache_path, fp16=trt_fp16)))
    if include_cuda:
        providers.append("CUDAExecutionProvider")
    if include_cpu:
        providers.append("CPUExecutionProvider")
    return providers


def probe_onnxruntime_stack() -> dict[str, Any]:
    try:
        import onnxruntime as ort
    except Exception as exc:
        return {
            "onnxruntime_importable": False,
            "onnxruntime_import_error": str(exc),
            "onnxruntime_providers": [],
            "onnxruntime_cuda": False,
            "onnxruntime_tensorrt": False,
        }
    providers = list(ort.get_available_providers())
    return {
        "onnxruntime_importable": True,
        "onnxruntime_import_error": "",
        "onnxruntime_providers": providers,
        "onnxruntime_cuda": "CUDAExecutionProvider" in providers,
        "onnxruntime_tensorrt": "TensorrtExecutionProvider" in providers,
    }
