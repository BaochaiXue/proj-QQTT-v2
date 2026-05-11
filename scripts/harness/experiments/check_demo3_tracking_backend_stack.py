#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_JSON = ROOT / "docs" / "generated" / "demo3_tracking_backend_stack.json"
DEFAULT_MD = ROOT / "docs" / "generated" / "demo3_tracking_backend_stack.md"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe Demo 3 tracking backend system stack and optional dependencies.")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--external-root", type=Path, default=Path("/home/zhangxinjie/external_tracking_backends"))
    return parser.parse_args(argv)


def _run_text(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, check=False, text=True, capture_output=True, timeout=5).stdout.strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def _gpu_name_from_nvidia_smi() -> str:
    if shutil.which("nvidia-smi") is None:
        return ""
    text = _run_text(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    return text.splitlines()[0].strip() if text else ""


def probe_system() -> dict[str, Any]:
    system: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "nvidia_smi_available": shutil.which("nvidia-smi") is not None,
        "gpu_name": _gpu_name_from_nvidia_smi(),
        "cuda_available": False,
        "torch_cuda": "",
        "torch_version": "",
        "tensorrt_importable": False,
        "tensorrt_import_error": "",
        "onnxruntime_importable": False,
        "onnxruntime_providers": [],
        "cv2_cuda_device_count": 0,
    }
    try:
        import torch

        system["torch_version"] = str(torch.__version__)
        system["torch_cuda"] = str(torch.version.cuda)
        system["cuda_available"] = bool(torch.cuda.is_available())
        if not system["gpu_name"] and torch.cuda.is_available():
            system["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        system["torch_import_error"] = str(exc)
    try:
        import tensorrt as trt  # type: ignore

        system["tensorrt_importable"] = True
        system["tensorrt_version"] = str(getattr(trt, "__version__", "unknown"))
    except Exception as exc:
        system["tensorrt_import_error"] = str(exc)
    try:
        import onnxruntime as ort

        system["onnxruntime_importable"] = True
        system["onnxruntime_version"] = str(getattr(ort, "__version__", "unknown"))
        system["onnxruntime_providers"] = list(ort.get_available_providers())
    except Exception as exc:
        system["onnxruntime_import_error"] = str(exc)
    try:
        import cv2

        system["cv2_version"] = str(cv2.__version__)
        system["cv2_cuda_device_count"] = int(cv2.cuda.getCudaEnabledDeviceCount()) if hasattr(cv2, "cuda") else 0
    except Exception as exc:
        system["cv2_import_error"] = str(exc)
    return system


def _install_hint(name: str) -> str:
    hints = {
        "nvofa": "Clone NVIDIA/NVIDIAOpticalFlowSDK and build a flow helper binary.",
        "vpi_lk": "Install NVIDIA VPI Python bindings in the selected conda environment.",
        "tapnext": "Clone google-deepmind/tapnet and configure TAPNext/TAPNext++ checkpoints.",
        "locotrack": "Clone cvlab-kaist/locotrack and configure weights.",
        "tapir": "Clone google-deepmind/tapnet and configure TAPIR/BootsTAPIR checkpoints.",
        "onnxruntime_cuda": "Install onnxruntime-gpu with CUDAExecutionProvider support.",
        "onnxruntime_tensorrt": "Install onnxruntime-gpu build with TensorrtExecutionProvider and TensorRT Python libs.",
    }
    return hints.get(name, "")


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    from qqtt.tracking.registry import check_backend_availability

    system = probe_system()
    backend_names = ["cotracker3_online", "nvofa", "vpi_lk", "tapnext", "locotrack", "tapir"]
    backends = {name: item.to_dict() for name, item in check_backend_availability(backend_names).items()}
    for name, item in backends.items():
        item["install_hint"] = _install_hint(name)
    providers = set(system.get("onnxruntime_providers", []))
    backends["onnxruntime_cuda"] = {
        "backend": "onnxruntime_cuda",
        "available": "CUDAExecutionProvider" in providers,
        "reason": "CUDAExecutionProvider found" if "CUDAExecutionProvider" in providers else "CUDAExecutionProvider not found",
        "install_hint": _install_hint("onnxruntime_cuda"),
    }
    backends["onnxruntime_tensorrt"] = {
        "backend": "onnxruntime_tensorrt",
        "available": "TensorrtExecutionProvider" in providers,
        "reason": "TensorrtExecutionProvider found" if "TensorrtExecutionProvider" in providers else "TensorrtExecutionProvider not found",
        "install_hint": _install_hint("onnxruntime_tensorrt"),
    }
    return {"system": system, "backends": backends, "external_root": str(Path(args.external_root))}


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = ["# Demo 3 Tracking Backend Stack", ""]
    system = report["system"]
    lines.extend(
        [
            f"- python: {system.get('python', '')}",
            f"- gpu_name: {system.get('gpu_name', '')}",
            f"- cuda_available: {system.get('cuda_available', False)}",
            f"- torch_cuda: {system.get('torch_cuda', '')}",
            f"- tensorrt_importable: {system.get('tensorrt_importable', False)}",
            f"- onnxruntime_importable: {system.get('onnxruntime_importable', False)}",
            f"- onnxruntime_providers: {', '.join(system.get('onnxruntime_providers', []))}",
            "",
            "## Backends",
            "",
        ]
    )
    for name, item in sorted(report["backends"].items()):
        state = "available" if item.get("available") else "unavailable"
        lines.append(f"- {name}: {state} - {item.get('reason', '')}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(args.output_md, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
