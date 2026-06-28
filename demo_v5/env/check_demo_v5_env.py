#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
import shutil
import subprocess
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
    "yaml",
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


def _is_executable(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def _check_nvcc_version(path: Path) -> list[str]:
    try:
        result = subprocess.run(
            [str(path), "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except Exception as exc:
        return [f"run nvcc --version at {path}: {type(exc).__name__}: {exc}"]
    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        detail = stderr or stdout or f"exit code {result.returncode}"
        return [f"run nvcc --version at {path}: {detail}"]
    version_line = next(
        (line.strip() for line in result.stdout.splitlines() if "release" in line),
        result.stdout.splitlines()[-1].strip() if result.stdout.splitlines() else "version-ok",
    )
    print(f"[ok] nvcc {path}: {version_line}")
    return []


def _check_nvcc_toolchain() -> list[str]:
    errors: list[str] = []
    candidates: list[Path] = []

    cudacxx = os.environ.get("CUDACXX")
    if cudacxx:
        cudacxx_path = Path(cudacxx).expanduser()
        if _is_executable(cudacxx_path):
            candidates.append(cudacxx_path)
        else:
            errors.append(f"CUDACXX points to missing or non-executable nvcc: {cudacxx_path}")

    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home:
        cuda_home_nvcc = Path(cuda_home).expanduser() / "bin" / "nvcc"
        if _is_executable(cuda_home_nvcc):
            candidates.append(cuda_home_nvcc)
        else:
            errors.append(f"CUDA_HOME/bin/nvcc is missing or non-executable: {cuda_home_nvcc}")

    path_nvcc = shutil.which("nvcc")
    if path_nvcc:
        candidates.append(Path(path_nvcc))
    else:
        errors.append("nvcc is not on PATH")

    if not candidates:
        return errors

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        errors.extend(_check_nvcc_version(candidate))
    return errors


def _check_gsplat_runtime_smoke() -> list[str]:
    try:
        import torch
        from gsplat import rasterization
    except Exception as exc:
        return [f"import gsplat rasterization for CUDA smoke: {type(exc).__name__}: {exc}"]

    if not torch.cuda.is_available():
        return ["torch.cuda.is_available() is false for gsplat CUDA smoke"]

    try:
        device = torch.device("cuda:0")
        with torch.no_grad():
            means = torch.tensor([[0.0, 0.0, 2.0]], device=device)
            quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
            scales = torch.tensor([[0.1, 0.1, 0.1]], device=device)
            opacities = torch.tensor([0.9], device=device)
            colors = torch.tensor([[1.0, 0.0, 0.0]], device=device)
            viewmats = torch.eye(4, device=device)[None]
            intrinsics = torch.tensor(
                [[[32.0, 0.0, 16.0], [0.0, 32.0, 16.0], [0.0, 0.0, 1.0]]],
                device=device,
            )
            render_colors, render_alphas, _ = rasterization(
                means,
                quats,
                scales,
                opacities,
                colors,
                viewmats,
                intrinsics,
                width=32,
                height=32,
            )
            torch.cuda.synchronize(device)
    except Exception as exc:
        return [
            "gsplat CUDA rasterization smoke failed: "
            f"{type(exc).__name__}: {exc}. Ensure nvcc is available through "
            "CUDACXX, CUDA_HOME/bin/nvcc, or PATH before running SAM3D."
        ]

    print(
        "[ok] gsplat CUDA rasterization smoke: "
        f"colors={tuple(render_colors.shape)} alphas={tuple(render_alphas.shape)}"
    )
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
    if bool(args.require_cuda) and role in {"shape-prior", "all"}:
        errors.extend(_check_nvcc_toolchain())
        errors.extend(_check_gsplat_runtime_smoke())
    if errors:
        print("[fail] Demo v5 environment check failed:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("[ok] Demo v5 environment check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
