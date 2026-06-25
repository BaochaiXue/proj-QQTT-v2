#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]


MAIN_MODULES = (
    "numpy",
    "PIL",
    "cv2",
    "open3d",
    "zmq",
    "scipy",
    "matplotlib",
    "yaml",
    "pyrealsense2",
    "trimesh",
    "torch",
    "torchvision",
    "warp",
    "cma",
    "wandb",
)

SHAPE_PRIOR_MODULES = (
    "numpy",
    "PIL",
    "cv2",
    "zmq",
    "scipy",
    "trimesh",
    "torch",
    "torchvision",
    "diffusers",
    "transformers",
    "accelerate",
    "kaolin",
    "pytorch3d",
    "moge",
)

COMMON_ASSET_PATHS = (
    "vendor/demo_runtime/EdgeTAM-hf",
    "vendor/demo_runtime/tapnet",
    "vendor/demo_runtime/checkpoints/tapnextpp/tapnextpp_ckpt.pt",
    "realtime_phystwin/train_online_zero_then_first.py",
    "table_calibrate.pkl",
)

SHAPE_PRIOR_ASSET_PATHS = (
    "vendor/demo_runtime/sam-3d-objects/notebook/inference.py",
    "vendor/demo_runtime/sam-3d-objects/sam3d_objects",
    "vendor/demo_runtime/stable-diffusion-x4-upscaler",
    "vendor/demo_runtime/FuturePhysTwin",
    "vendor/demo_runtime/dinov2",
    "vendor/demo_runtime/checkpoints/dinov2/dinov2_vitl14_reg4_pretrain.pth",
    "vendor/demo_runtime/checkpoints/MoGe-vitl/model.pt",
)


def _version(module: object) -> str:
    return str(getattr(module, "__version__", "import-ok"))


def _check_imports(modules: Iterable[str]) -> list[str]:
    errors: list[str] = []
    for name in modules:
        try:
            module = importlib.import_module(name)
        except Exception as exc:
            errors.append(f"import {name}: {type(exc).__name__}: {exc}")
            continue
        print(f"[ok] import {name}: {_version(module)}")
    return errors


def _check_paths(paths: Iterable[str]) -> list[str]:
    errors: list[str] = []
    for relative in paths:
        path = REPO_ROOT / relative
        if path.exists():
            print(f"[ok] path {relative}")
        else:
            errors.append(f"missing path: {relative}")
    return errors


def _check_shape_prior_source_import() -> list[str]:
    root = REPO_ROOT / "vendor" / "demo_runtime" / "sam-3d-objects"
    for path in (root, root / "notebook"):
        path_s = str(path)
        if path_s not in sys.path:
            sys.path.insert(0, path_s)
    try:
        module = importlib.import_module("sam3d_objects")
    except Exception as exc:
        return [f"import sam3d_objects from vendor source: {type(exc).__name__}: {exc}"]
    print(f"[ok] import sam3d_objects from vendor source: {_version(module)}")
    return []


def _check_cuda(required: bool) -> list[str]:
    try:
        import torch
    except Exception as exc:
        return [f"import torch for CUDA check: {type(exc).__name__}: {exc}"] if required else []
    available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if available else 0
    print(f"[info] torch cuda available={available} device_count={device_count}")
    if required and not available:
        return ["torch.cuda.is_available() is false"]
    return []


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check Demo v5 runtime environment and repo-local assets.")
    parser.add_argument("--role", choices=("main", "shape-prior", "all"), default="all")
    parser.add_argument("--require-cuda", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    errors: list[str] = []
    role = str(args.role)
    if role in {"main", "all"}:
        print("[check] Demo v5 main environment")
        errors.extend(_check_imports(MAIN_MODULES))
        errors.extend(_check_paths(COMMON_ASSET_PATHS))
    if role in {"shape-prior", "all"}:
        print("[check] Demo v5 shape-prior worker environment")
        errors.extend(_check_imports(SHAPE_PRIOR_MODULES))
        errors.extend(_check_paths(SHAPE_PRIOR_ASSET_PATHS))
        errors.extend(_check_shape_prior_source_import())
    errors.extend(_check_cuda(bool(args.require_cuda)))
    if errors:
        print("[fail] Demo v5 environment check failed:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("[ok] Demo v5 environment check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
