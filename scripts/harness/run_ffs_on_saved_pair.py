from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Fast-FoundationStereo on a saved D455 IR pair.")
    parser.add_argument("--sample_dir", required=True)
    parser.add_argument("--ffs_repo", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--valid_iters", type=int, default=8)
    parser.add_argument("--max_disp", type=int, default=192)
    parser.add_argument("--get_pc", type=int, choices=(0, 1), default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    import cv2
    import imageio.v2 as imageio
    import numpy as np
    import torch
    import yaml
    from omegaconf import OmegaConf

    from scripts.harness.ffs_geometry import disparity_to_metric_depth, write_ffs_intrinsic_file

    sample_dir = Path(args.sample_dir).resolve()
    ffs_repo = Path(args.ffs_repo).resolve()
    model_path = Path(args.model_path).resolve()
    out_dir = Path(args.out_dir).resolve()

    def identity_compile(fn=None, *compile_args, **compile_kwargs):
        if fn is None:
            def decorator(inner):
                return inner
            return decorator
        return fn

    torch.compile = identity_compile

    metadata_path = sample_dir / "metadata.json"
    left_path = sample_dir / "ir_left.png"
    right_path = sample_dir / "ir_right.png"
    for path in (metadata_path, left_path, right_path, model_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing required path: {path}")

    sys.path.insert(0, str(ffs_repo))
    from core.utils.utils import InputPadder
    from Utils import AMP_DTYPE, depth2xyzmap, o3d, set_logging_format, set_seed, toOpen3dCloud, vis_disparity

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    K_ir_left = np.asarray(metadata["K_ir_left"], dtype=np.float32)
    baseline_m = float(metadata["ir_baseline_m"])

    with open(model_path.parent / "cfg.yaml", "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg.update(
        {
            "model_dir": str(model_path),
            "left_file": str(left_path),
            "right_file": str(right_path),
            "out_dir": str(out_dir),
            "scale": float(args.scale),
            "valid_iters": int(args.valid_iters),
            "max_disp": int(args.max_disp),
            "get_pc": int(args.get_pc),
        }
    )
    cfg = OmegaConf.create(cfg)

    set_logging_format()
    set_seed(0)
    torch.autograd.set_grad_enabled(False)

    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = torch.load(str(model_path), map_location="cpu", weights_only=False)
    model.args.valid_iters = cfg.valid_iters
    model.args.max_disp = cfg.max_disp
    model.cuda().eval()

    img0 = imageio.imread(left_path)
    img1 = imageio.imread(right_path)
    if img0.ndim == 2:
        img0 = np.tile(img0[..., None], (1, 1, 3))
        img1 = np.tile(img1[..., None], (1, 1, 3))
    img0 = img0[..., :3]
    img1 = img1[..., :3]

    if cfg.scale != 1.0:
        img0 = cv2.resize(img0, dsize=None, fx=cfg.scale, fy=cfg.scale, interpolation=cv2.INTER_LINEAR)
        img1 = cv2.resize(img1, dsize=(img0.shape[1], img0.shape[0]), interpolation=cv2.INTER_LINEAR)

    height, width = img0.shape[:2]
    img0_ori = img0.copy()
    img1_ori = img1.copy()
    imageio.imwrite(out_dir / "left.png", img0)
    imageio.imwrite(out_dir / "right.png", img1)

    img0_tensor = torch.as_tensor(img0).cuda().float()[None].permute(0, 3, 1, 2)
    img1_tensor = torch.as_tensor(img1).cuda().float()[None].permute(0, 3, 1, 2)
    padder = InputPadder(img0_tensor.shape, divis_by=32, force_square=False)
    img0_tensor, img1_tensor = padder.pad(img0_tensor, img1_tensor)

    logging.info("Starting Fast-FoundationStereo forward pass")
    with torch.amp.autocast("cuda", enabled=True, dtype=AMP_DTYPE):
        disparity = model.forward(
            img0_tensor,
            img1_tensor,
            iters=cfg.valid_iters,
            test_mode=True,
            optimize_build_volume="pytorch1",
        )
    disparity = padder.unpad(disparity.float())
    disparity = disparity.data.cpu().numpy().reshape(height, width).clip(0, None).astype(np.float32)
    np.save(out_dir / "disparity_raw.npy", disparity)

    vis = vis_disparity(disparity, min_val=None, max_val=None, cmap=None, color_map=cv2.COLORMAP_TURBO)
    imageio.imwrite(out_dir / "disp_vis.png", np.concatenate([img0_ori, img1_ori, vis], axis=1))

    K_scaled = K_ir_left.copy()
    K_scaled[:2] *= float(cfg.scale)
    write_ffs_intrinsic_file(out_dir / "intrinsics_ffs.txt", K_scaled, baseline_m)

    depth_ir = disparity_to_metric_depth(disparity, fx_ir=float(K_scaled[0, 0]), baseline_m=baseline_m)
    np.save(out_dir / "depth_ir_left_float_m.npy", depth_ir)

    run_metadata = {
        "sample_dir": str(sample_dir),
        "ffs_repo": str(ffs_repo),
        "model_path": str(model_path),
        "scale": float(cfg.scale),
        "valid_iters": int(cfg.valid_iters),
        "max_disp": int(cfg.max_disp),
        "get_pc": int(cfg.get_pc),
        "K_ir_left_used": K_scaled.tolist(),
        "baseline_m": baseline_m,
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")
    (out_dir / "command_log.txt").write_text(
        " ".join(
            [
                str(Path(__file__).resolve()),
                "--sample_dir",
                str(sample_dir),
                "--ffs_repo",
                str(ffs_repo),
                "--model_path",
                str(model_path),
                "--out_dir",
                str(out_dir),
                "--scale",
                str(cfg.scale),
                "--valid_iters",
                str(cfg.valid_iters),
                "--max_disp",
                str(cfg.max_disp),
                "--get_pc",
                str(cfg.get_pc),
            ]
        ),
        encoding="utf-8",
    )
    shutil.copy2(metadata_path, out_dir / "metadata.json")

    if cfg.get_pc:
        xyz_map = depth2xyzmap(depth_ir, K_scaled)
        pcd = toOpen3dCloud(xyz_map.reshape(-1, 3), img0_ori.reshape(-1, 3))
        keep_mask = np.isfinite(np.asarray(pcd.points)[:, 2]) & (np.asarray(pcd.points)[:, 2] > 0)
        keep_ids = np.arange(len(np.asarray(pcd.points)))[keep_mask]
        pcd = pcd.select_by_index(keep_ids)
        o3d.io.write_point_cloud(str(out_dir / "cloud.ply"), pcd)

    print(f"Saved Fast-FoundationStereo outputs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
