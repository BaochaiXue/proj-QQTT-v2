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
    "termcolor",
    "pynput",
    "pyrender",
    "rtree",
    "pyglet",
    "atomics",
    "stannum",
    "taichi",
    "kornia",
    "plyfile",
    "gsplat",
    "simple_knn._C",
    "diff_gaussian_rasterization",
    "pytorch3d",
    "transformers",
    "einops",
    "timm",
    "ftfy",
    "iopath",
    "pycocotools",
    "sam3",
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
    "omegaconf",
    "hydra",
    "kaolin",
    "pytorch3d",
    "moge",
    "loguru",
    "astor",
    "easydict",
    "optree",
    "lightning",
    "imageio",
    "fvcore",
    "timm",
    "einops",
    "einops_exts",
    "seaborn",
    "gradio",
    "xatlas",
    "pymeshfix",
    "igraph",
    "pyvista",
    "spconv",
    "nvdiffrast",
    "xformers",
    "gsplat",
)

COMMON_ASSET_PATHS = (
    "vendor/demo_runtime/EdgeTAM-hf",
    "vendor/demo_runtime/tapnet",
    "vendor/demo_runtime/checkpoints/tapnextpp/tapnextpp_ckpt.pt",
    "vendor/demo_runtime/checkpoints/sam31/sam3.1_multiplex.pt",
    "vendor/demo_runtime/checkpoints/sam31/bpe_simple_vocab_16e6.txt.gz",
    "realtime_phystwin/train_online_zero_then_first.py",
    "table_calibrate.pkl",
)

SHAPE_PRIOR_ASSET_PATHS = (
    "vendor/demo_runtime/sam-3d-objects/notebook/inference.py",
    "vendor/demo_runtime/sam-3d-objects/sam3d_objects",
    "vendor/demo_runtime/stable-diffusion-x4-upscaler",
    "vendor/demo_runtime/stable-diffusion-x4-upscaler/text_encoder/model.safetensors",
    "vendor/demo_runtime/stable-diffusion-x4-upscaler/unet/diffusion_pytorch_model.safetensors",
    "vendor/demo_runtime/stable-diffusion-x4-upscaler/vae/diffusion_pytorch_model.safetensors",
    "vendor/demo_runtime/FuturePhysTwin",
    "vendor/demo_runtime/dinov2",
    "vendor/demo_runtime/checkpoints/dinov2/dinov2_vitl14_reg4_pretrain.pth",
    "vendor/demo_runtime/checkpoints/MoGe-vitl/model.pt",
    "vendor/demo_runtime/checkpoints/sam3d/hf/slat_decoder_gs.ckpt",
    "vendor/demo_runtime/checkpoints/sam3d/hf/slat_decoder_gs_4.ckpt",
    "vendor/demo_runtime/checkpoints/sam3d/hf/slat_decoder_mesh.ckpt",
    "vendor/demo_runtime/checkpoints/sam3d/hf/slat_generator.ckpt",
    "vendor/demo_runtime/checkpoints/sam3d/hf/ss_decoder.ckpt",
    "vendor/demo_runtime/checkpoints/sam3d/hf/ss_generator.ckpt",
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


def _check_main_source_imports() -> list[str]:
    root = REPO_ROOT / "vendor" / "demo_runtime" / "tapnet"
    root_s = str(root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)
    errors: list[str] = []
    for name in (
        "tapnet.tapnext.tapnext_torch",
        "tapnet.tapnext.tapnext_torch_utils",
        "sam3.model_builder",
    ):
        try:
            module = importlib.import_module(name)
        except Exception as exc:
            errors.append(f"import {name} from vendor source: {type(exc).__name__}: {exc}")
            continue
        print(f"[ok] import {name} from vendor source: {_version(module)}")
    try:
        from transformers import AutoConfig, EdgeTamVideoModel, Sam2VideoProcessor  # noqa: F401
    except Exception as exc:
        errors.append(f"import EdgeTAM transformers classes: {type(exc).__name__}: {exc}")
    else:
        print("[ok] import EdgeTAM transformers classes: import-ok")
        model_dir = REPO_ROOT / "vendor" / "demo_runtime" / "EdgeTAM-hf"
        try:
            AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True)
        except Exception as exc:
            errors.append(f"load EdgeTAM config: {type(exc).__name__}: {exc}")
        else:
            print("[ok] load EdgeTAM config: import-ok")
    phystwin_root = REPO_ROOT / "realtime_phystwin"
    phystwin_root_s = str(phystwin_root)
    if phystwin_root_s not in sys.path:
        sys.path.insert(0, phystwin_root_s)
    for name in (
        "optimize_online_cma",
        "train_online_warp",
    ):
        try:
            module = importlib.import_module(name)
        except Exception as exc:
            errors.append(f"import {name} from realtime_phystwin: {type(exc).__name__}: {exc}")
            continue
        print(f"[ok] import {name} from realtime_phystwin: {_version(module)}")
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
        errors.extend(_check_main_source_imports())
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
