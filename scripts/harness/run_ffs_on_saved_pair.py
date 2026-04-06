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
    import numpy as np
    from data_process.depth_backends import FastFoundationStereoRunner, write_ffs_intrinsic_file

    sample_dir = Path(args.sample_dir).resolve()
    ffs_repo = Path(args.ffs_repo).resolve()
    model_path = Path(args.model_path).resolve()
    out_dir = Path(args.out_dir).resolve()

    metadata_path = sample_dir / "metadata.json"
    left_path = sample_dir / "ir_left.png"
    right_path = sample_dir / "ir_right.png"
    for path in (metadata_path, left_path, right_path, model_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing required path: {path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    K_ir_left = np.asarray(metadata["K_ir_left"], dtype=np.float32)
    baseline_m = float(metadata["ir_baseline_m"])

    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    left_image = cv2.imread(str(left_path), cv2.IMREAD_UNCHANGED)
    right_image = cv2.imread(str(right_path), cv2.IMREAD_UNCHANGED)
    if left_image is None or right_image is None:
        raise RuntimeError("Failed to load saved IR stereo pair")

    runner = FastFoundationStereoRunner(
        ffs_repo=ffs_repo,
        model_path=model_path,
        scale=args.scale,
        valid_iters=args.valid_iters,
        max_disp=args.max_disp,
    )
    logging.info("Starting Fast-FoundationStereo forward pass")
    run_output = runner.run_pair(
        left_image,
        right_image,
        K_ir_left=K_ir_left,
        baseline_m=baseline_m,
    )
    disparity = np.asarray(run_output["disparity"], dtype=np.float32)
    depth_ir = np.asarray(run_output["depth_ir_left_m"], dtype=np.float32)
    K_scaled = np.asarray(run_output["K_ir_left_used"], dtype=np.float32)
    np.save(out_dir / "disparity_raw.npy", disparity)
    np.save(out_dir / "depth_ir_left_float_m.npy", depth_ir)

    disp_vis = cv2.applyColorMap(cv2.convertScaleAbs(disparity, alpha=255.0 / max(1.0, float(disparity.max()))), cv2.COLORMAP_TURBO)
    left_preview = left_image if left_image.ndim == 3 else cv2.cvtColor(left_image, cv2.COLOR_GRAY2BGR)
    right_preview = right_image if right_image.ndim == 3 else cv2.cvtColor(right_image, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(out_dir / "disp_vis.png"), np.concatenate([left_preview, right_preview, disp_vis], axis=1))

    run_metadata = {
        "sample_dir": str(sample_dir),
        "ffs_repo": str(ffs_repo),
        "model_path": str(model_path),
        "scale": float(args.scale),
        "valid_iters": int(args.valid_iters),
        "max_disp": int(args.max_disp),
        "get_pc": int(args.get_pc),
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
                str(args.scale),
                "--valid_iters",
                str(args.valid_iters),
                "--max_disp",
                str(args.max_disp),
                "--get_pc",
                str(args.get_pc),
            ]
        ),
        encoding="utf-8",
    )
    shutil.copy2(metadata_path, out_dir / "metadata.json")

    if args.get_pc:
        sys.path.insert(0, str(ffs_repo))
        from Utils import depth2xyzmap, o3d, toOpen3dCloud

        left_preview = left_image if left_image.ndim == 3 else cv2.cvtColor(left_image, cv2.COLOR_GRAY2BGR)
        xyz_map = depth2xyzmap(depth_ir, K_scaled)
        pcd = toOpen3dCloud(xyz_map.reshape(-1, 3), left_preview.reshape(-1, 3))
        valid = np.isfinite(np.asarray(pcd.points)[:, 2]) & (np.asarray(pcd.points)[:, 2] > 0)
        pcd = pcd.select_by_index(np.arange(len(np.asarray(pcd.points)))[valid])
        o3d.io.write_point_cloud(str(out_dir / "cloud.ply"), pcd)

    print(f"Saved Fast-FoundationStereo outputs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
