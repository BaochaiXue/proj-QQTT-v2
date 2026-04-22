from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.harness.verify_ffs_tensorrt_wsl import build_engine_from_onnx

DEFAULT_FFS_REPO = Path("/home/zhangxinjie/Fast-FoundationStereo")
DEFAULT_MODEL_PATH = DEFAULT_FFS_REPO / "weights" / "23-36-37" / "model_best_bp2_serialize.pth"
DEFAULT_OUT_DIR = ROOT / "data" / "ffs_proof_of_life" / "trt_single_engine_batch3_864x480_wsl_fp32"


def _disable_torch_compile(torch_module) -> None:
    def identity_compile(fn=None, *args, **kwargs):
        if fn is None:
            def decorator(inner):
                return inner

            return decorator
        return fn

    torch_module.compile = identity_compile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Fast-FoundationStereo single-engine TensorRT on WSL/Linux.")
    parser.add_argument("--ffs_repo", default=str(DEFAULT_FFS_REPO))
    parser.add_argument("--model_path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=864)
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--valid_iters", type=int, default=4)
    parser.add_argument("--max_disp", type=int, default=192)
    parser.add_argument("--workspace_gib", type=int, default=8)
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def ensure_required_paths(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required paths: {missing}")


def prepare_python_runtime(ffs_repo: Path):
    import torch

    _disable_torch_compile(torch)
    if str(ffs_repo) not in sys.path:
        sys.path.insert(0, str(ffs_repo))
    return torch


def export_single_engine_onnx(
    *,
    torch_module,
    ffs_repo: Path,
    model_path: Path,
    out_dir: Path,
    height: int,
    width: int,
    batch_size: int,
    valid_iters: int,
    max_disp: int,
) -> None:
    import yaml
    from omegaconf import OmegaConf

    _disable_torch_compile(torch_module)
    if str(ffs_repo) not in sys.path:
        sys.path.insert(0, str(ffs_repo))

    import core.foundation_stereo as foundation_stereo_module
    import core.foundation_stereo as _fs_module
    import torch.nn as nn
    import torch.nn.functional as F

    def _build_gwc_volume_onnx(refimg_fea, targetimg_fea, maxdisp, num_groups, normalize=True):
        dtype = refimg_fea.dtype
        bsz, channels, height_04, width_04 = refimg_fea.shape
        channels_per_group = channels // num_groups
        ref_volume = refimg_fea.unsqueeze(2).expand(bsz, channels, maxdisp, height_04, width_04)
        shifted = [
            F.pad(targetimg_fea, (disp, 0, 0, 0), "constant", 0.0)[:, :, :, :width_04]
            for disp in range(maxdisp)
        ]
        target_volume = torch_module.stack(shifted, dim=2)
        ref_volume = ref_volume.view(bsz, num_groups, channels_per_group, maxdisp, height_04, width_04)
        target_volume = target_volume.view(bsz, num_groups, channels_per_group, maxdisp, height_04, width_04)
        if normalize:
            ref_volume = F.normalize(ref_volume.float(), dim=2).to(dtype)
            target_volume = F.normalize(target_volume.float(), dim=2).to(dtype)
        return (ref_volume * target_volume).sum(dim=2).contiguous()

    def _build_concat_volume_onnx(refimg_fea, targetimg_fea, maxdisp):
        bsz, channels, height_04, width_04 = refimg_fea.shape
        ref_volume = refimg_fea.unsqueeze(2).expand(bsz, channels, maxdisp, height_04, width_04)
        shifted = [
            F.pad(targetimg_fea, (disp, 0, 0, 0), "constant", 0.0)[:, :, :, :width_04]
            for disp in range(maxdisp)
        ]
        target_volume = torch_module.stack(shifted, dim=2)
        return torch_module.cat((ref_volume, target_volume), dim=1).contiguous()

    class FastFoundationStereoSingleOnnx(nn.Module):
        def __init__(self, model) -> None:
            super().__init__()
            self.model = model

        @torch_module.no_grad()
        def forward(self, left_image, right_image):
            return self.model.forward(
                left_image,
                right_image,
                iters=self.model.args.valid_iters,
                test_mode=True,
                optimize_build_volume="pytorch1",
            )

    model = torch_module.load(str(model_path), map_location="cpu", weights_only=False)
    model.args.max_disp = max_disp
    model.args.valid_iters = valid_iters
    model.args.mixed_precision = False
    model.cuda().eval()
    wrapper = FastFoundationStereoSingleOnnx(model).cuda().eval()

    left_img = torch_module.randn(batch_size, 3, height, width, device="cuda")
    right_img = torch_module.randn(batch_size, 3, height, width, device="cuda")

    _fs_module.normalize_image = lambda img: img
    _fs_module.build_gwc_volume_optimized_pytorch1 = _build_gwc_volume_onnx
    _fs_module.build_concat_volume_optimized_pytorch1 = _build_concat_volume_onnx

    onnx_path = out_dir / "fast_foundationstereo.onnx"
    torch_module.onnx.export(
        wrapper,
        (left_img, right_img),
        str(onnx_path),
        opset_version=17,
        input_names=["left_image", "right_image"],
        output_names=["disparity"],
        do_constant_folding=True,
    )

    cfg = OmegaConf.to_container(model.args)
    cfg["image_size"] = [height, width]
    with open(out_dir / "fast_foundationstereo.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle)


def resolve_demo_paths(ffs_repo: Path) -> tuple[Path, Path, Path]:
    left_file = ffs_repo / "demo_data" / "left.png"
    right_file = ffs_repo / "demo_data" / "right.png"
    intrinsic_file = ffs_repo / "demo_data" / "K.txt"
    return left_file, right_file, intrinsic_file


def run_demo_smoke(*, ffs_repo: Path, model_dir: Path, batch_size: int) -> dict[str, float]:
    if str(ffs_repo) not in sys.path:
        sys.path.insert(0, str(ffs_repo))

    from data_process.depth_backends.fast_foundation_stereo import FastFoundationStereoSingleEngineTensorRTRunner
    from Utils import depth2xyzmap, o3d, toOpen3dCloud, vis_disparity

    left_file, right_file, intrinsic_file = resolve_demo_paths(ffs_repo)
    out_dir = model_dir / "demo_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    left = imageio.imread(left_file)[..., :3]
    right = imageio.imread(right_file)[..., :3]
    with open(intrinsic_file, "r", encoding="utf-8") as handle:
        lines = handle.readlines()
    K = np.array(list(map(float, lines[0].rstrip().split())), dtype=np.float32).reshape(3, 3)
    baseline = float(lines[1])

    runner = FastFoundationStereoSingleEngineTensorRTRunner(
        ffs_repo=ffs_repo,
        model_dir=model_dir,
    )
    batch_outputs = runner.run_batch(
        [
            {
                "left_image": left,
                "right_image": right,
                "K_ir_left": K,
                "baseline_m": baseline,
                "audit_mode": True,
            }
            for _ in range(batch_size)
        ]
    )
    first_output = batch_outputs[0]
    disparity = np.asarray(first_output["disparity"], dtype=np.float32)
    depth = np.asarray(first_output["depth_ir_left_m"], dtype=np.float32)
    K_used = np.asarray(first_output["K_ir_left_used"], dtype=np.float32)
    vis = vis_disparity(disparity, min_val=None, max_val=None, cmap=None, color_map=cv2.COLORMAP_TURBO)
    prep_left = cv2.resize(left, (disparity.shape[1], disparity.shape[0]), interpolation=cv2.INTER_LINEAR)
    prep_right = cv2.resize(right, (disparity.shape[1], disparity.shape[0]), interpolation=cv2.INTER_LINEAR)
    imageio.imwrite(out_dir / "left.png", prep_left)
    imageio.imwrite(out_dir / "right.png", prep_right)
    imageio.imwrite(out_dir / "disp_vis.png", np.concatenate([prep_left, prep_right, vis], axis=1))
    np.save(out_dir / "depth_meter.npy", depth)
    xyz_map = depth2xyzmap(depth, K_used)
    pcd = toOpen3dCloud(xyz_map.reshape(-1, 3), prep_left.reshape(-1, 3))
    pts = np.asarray(pcd.points)
    keep = (pts[:, 2] > 0) & np.isfinite(pts[:, 2]) & (pts[:, 2] <= 100.0)
    pcd = pcd.select_by_index(np.where(keep)[0])
    o3d.io.write_point_cloud(str(out_dir / "cloud.ply"), pcd)
    return dict(first_output["audit_stats"])


def main() -> int:
    args = parse_args()
    ffs_repo = Path(args.ffs_repo).resolve()
    model_path = Path(args.model_path).resolve()
    out_dir = Path(args.out_dir).resolve()
    ensure_required_paths([ffs_repo, model_path, *resolve_demo_paths(ffs_repo)])
    if int(args.height) % 32 != 0 or int(args.width) % 32 != 0:
        raise ValueError(f"Expected engine size divisible by 32, got {args.width}x{args.height}")
    if int(args.batch_size) <= 0:
        raise ValueError(f"Expected positive --batch_size, got {args.batch_size}")

    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch_module = prepare_python_runtime(ffs_repo)

    export_single_engine_onnx(
        torch_module=torch_module,
        ffs_repo=ffs_repo,
        model_path=model_path,
        out_dir=out_dir,
        height=int(args.height),
        width=int(args.width),
        batch_size=int(args.batch_size),
        valid_iters=int(args.valid_iters),
        max_disp=int(args.max_disp),
    )
    build_engine_from_onnx(
        onnx_path=out_dir / "fast_foundationstereo.onnx",
        engine_path=out_dir / "fast_foundationstereo.engine",
        log_path=out_dir / "single_engine_build.log",
        workspace_gib=int(args.workspace_gib),
        fp16=bool(args.fp16),
    )
    audit_stats = run_demo_smoke(
        ffs_repo=ffs_repo,
        model_dir=out_dir,
        batch_size=int(args.batch_size),
    )
    print(f"single-engine batch TensorRT artifacts written to {out_dir}")
    print(f"audit_stats={audit_stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
