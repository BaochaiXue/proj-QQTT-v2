from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path

import cv2
import imageio
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FFS_REPO = Path("/home/zhangxinjie/Fast-FoundationStereo")
DEFAULT_MODEL_PATH = DEFAULT_FFS_REPO / "weights" / "23-36-37" / "model_best_bp2_serialize.pth"
DEFAULT_OUT_DIR = ROOT / "data" / "ffs_proof_of_life" / "trt_two_stage_864x480_wsl"


def _disable_torch_compile(torch_module) -> None:
    def identity_compile(fn=None, *args, **kwargs):
        if fn is None:
            def decorator(inner):
                return inner

            return decorator
        return fn

    torch_module.compile = identity_compile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Fast-FoundationStereo TensorRT on WSL/Linux.")
    parser.add_argument("--ffs_repo", default=str(DEFAULT_FFS_REPO))
    parser.add_argument("--model_path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=864)
    parser.add_argument("--valid_iters", type=int, default=4)
    parser.add_argument("--max_disp", type=int, default=192)
    parser.add_argument("--workspace_gib", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--total", type=int, default=20)
    parser.add_argument("--zfar", type=float, default=100.0)
    parser.add_argument("--skip_profiles", action="store_true")
    return parser.parse_args()


def resolve_demo_paths(ffs_repo: Path) -> tuple[Path, Path, Path]:
    left_file = ffs_repo / "demo_data" / "left.png"
    right_file = ffs_repo / "demo_data" / "right.png"
    intrinsic_file = ffs_repo / "demo_data" / "K.txt"
    return left_file, right_file, intrinsic_file


def ensure_required_paths(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required paths: {missing}")


def prepare_python_runtime(ffs_repo: Path):
    import torch

    _disable_torch_compile(torch)
    if str(ffs_repo) not in sys.path:
        sys.path.insert(0, str(ffs_repo))

    import core.foundation_stereo as foundation_stereo
    import core.submodule as submodule

    # Keep the host-side GWC path consistent with the QQTT TensorRT runner.
    replacement = submodule.build_gwc_volume_optimized_pytorch1
    submodule.build_gwc_volume_triton = replacement
    foundation_stereo.build_gwc_volume_triton = replacement
    return torch, foundation_stereo


def verify_python_deps() -> str:
    import tensorrt as trt

    assert trt.Builder(trt.Logger())
    assert hasattr(trt, "OnnxParser")
    return trt.__version__


def export_onnx(
    *,
    torch_module,
    foundation_stereo,
    model_path: Path,
    out_dir: Path,
    height: int,
    width: int,
    valid_iters: int,
    max_disp: int,
) -> None:
    import yaml
    from omegaconf import OmegaConf

    model = torch_module.load(str(model_path), map_location="cpu", weights_only=False)
    model.args.max_disp = max_disp
    model.args.valid_iters = valid_iters
    model.cuda().eval()

    feature_runner = foundation_stereo.TrtFeatureRunner(model).cuda().eval()
    post_runner = foundation_stereo.TrtPostRunner(model).cuda().eval()
    left_img = torch_module.randn(1, 3, height, width, device="cuda", dtype=torch_module.float32) * 255
    right_img = torch_module.randn(1, 3, height, width, device="cuda", dtype=torch_module.float32) * 255

    torch_module.onnx.export(
        feature_runner,
        (left_img, right_img),
        str(out_dir / "feature_runner.onnx"),
        opset_version=17,
        input_names=["left", "right"],
        output_names=[
            "features_left_04",
            "features_left_08",
            "features_left_16",
            "features_left_32",
            "features_right_04",
            "stem_2x",
        ],
        do_constant_folding=True,
    )

    features_left_04, features_left_08, features_left_16, features_left_32, features_right_04, stem_2x = feature_runner(
        left_img,
        right_img,
    )
    gwc_volume = foundation_stereo.build_gwc_volume_triton(
        features_left_04.half(),
        features_right_04.half(),
        max_disp // 4,
        model.cv_group,
        normalize=model.args.normalize,
    )
    torch_module.onnx.export(
        post_runner,
        (
            features_left_04,
            features_left_08,
            features_left_16,
            features_left_32,
            features_right_04,
            stem_2x,
            gwc_volume,
        ),
        str(out_dir / "post_runner.onnx"),
        opset_version=17,
        input_names=[
            "features_left_04",
            "features_left_08",
            "features_left_16",
            "features_left_32",
            "features_right_04",
            "stem_2x",
            "gwc_volume",
        ],
        output_names=["disp"],
        do_constant_folding=True,
    )

    with open(out_dir / "onnx.yaml", "w", encoding="utf-8") as handle:
        cfg = OmegaConf.to_container(model.args)
        cfg["image_size"] = [height, width]
        yaml.safe_dump(cfg, handle)


def build_engine_from_onnx(
    *,
    onnx_path: Path,
    engine_path: Path,
    log_path: Path,
    workspace_gib: int,
    fp16: bool,
) -> None:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errors = [str(parser.get_error(idx)) for idx in range(parser.num_errors)]
        log_path.write_text("\n".join(errors), encoding="utf-8")
        raise RuntimeError(f"Failed to parse ONNX file {onnx_path}; see {log_path}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gib) << 30)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        log_path.write_text("builder.build_serialized_network returned None\n", encoding="utf-8")
        raise RuntimeError(f"Failed to build TensorRT engine for {onnx_path}; see {log_path}")
    engine_path.write_bytes(bytes(serialized_engine))
    log_path.write_text(
        "\n".join(
            [
                f"onnx={onnx_path}",
                f"engine={engine_path}",
                f"workspace_gib={int(workspace_gib)}",
                f"fp16={bool(fp16)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def load_cfg(onnx_dir: Path) -> dict:
    import yaml

    with open(onnx_dir / "onnx.yaml", "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def run_demo(
    *,
    torch_module,
    foundation_stereo,
    onnx_dir: Path,
    left_file: Path,
    right_file: Path,
    intrinsic_file: Path,
    out_dir: Path,
    zfar: float,
) -> None:
    from omegaconf import OmegaConf
    from Utils import depth2xyzmap, o3d, set_logging_format, set_seed, toOpen3dCloud, vis_disparity

    set_logging_format()
    set_seed(0)
    cfg = load_cfg(onnx_dir)
    cfg.update(
        {
            "onnx_dir": str(onnx_dir),
            "left_file": str(left_file),
            "right_file": str(right_file),
            "intrinsic_file": str(intrinsic_file),
            "out_dir": str(out_dir),
            "remove_invisible": 0,
            "denoise_cloud": 0,
            "get_pc": 1,
            "zfar": zfar,
        }
    )
    args = OmegaConf.create(cfg)
    model = foundation_stereo.TrtRunner(
        args,
        str(onnx_dir / "feature_runner.engine"),
        str(onnx_dir / "post_runner.engine"),
    )

    img0 = imageio.imread(left_file)
    img1 = imageio.imread(right_file)
    if len(img0.shape) == 2:
        img0 = np.tile(img0[..., None], (1, 1, 3))
        img1 = np.tile(img1[..., None], (1, 1, 3))
    img0 = img0[..., :3]
    img1 = img1[..., :3]

    fx = args.image_size[1] / img0.shape[1]
    fy = args.image_size[0] / img0.shape[0]
    img0 = cv2.resize(img0, fx=fx, fy=fy, dsize=None)
    img1 = cv2.resize(img1, fx=fx, fy=fy, dsize=None)
    out_dir.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(out_dir / "left.png", img0)
    imageio.imwrite(out_dir / "right.png", img1)

    img0_tensor = torch_module.as_tensor(img0).cuda().float()[None].permute(0, 3, 1, 2)
    img1_tensor = torch_module.as_tensor(img1).cuda().float()[None].permute(0, 3, 1, 2)
    disp = model.forward(img0_tensor, img1_tensor)
    height, width = img0.shape[:2]
    disp = disp.data.cpu().numpy().reshape(height, width).clip(0, None) * 1 / fx

    vis = vis_disparity(disp, min_val=None, max_val=None, cmap=None, color_map=cv2.COLORMAP_TURBO)
    vis = np.concatenate([img0, img1, vis], axis=1)
    imageio.imwrite(out_dir / "disp_vis.png", vis)

    with open(intrinsic_file, "r", encoding="utf-8") as handle:
        lines = handle.readlines()
    K = np.array(list(map(float, lines[0].rstrip().split()))).astype(np.float32).reshape(3, 3)
    baseline = float(lines[1])
    K[:2] *= np.array([fx, fy], dtype=np.float32)[:, np.newaxis]
    depth = K[0, 0] * baseline / disp
    np.save(out_dir / "depth_meter.npy", depth)
    xyz_map = depth2xyzmap(depth, K)
    pcd = toOpen3dCloud(xyz_map.reshape(-1, 3), img0.reshape(-1, 3))
    keep_mask = (np.asarray(pcd.points)[:, 2] > 0) & (np.asarray(pcd.points)[:, 2] <= zfar)
    keep_ids = np.arange(len(np.asarray(pcd.points)))[keep_mask]
    pcd = pcd.select_by_index(keep_ids)
    o3d.io.write_point_cloud(str(out_dir / "cloud.ply"), pcd)


def profile_tensorrt(
    *,
    torch_module,
    foundation_stereo,
    onnx_dir: Path,
    warmup: int,
    total: int,
) -> float:
    from omegaconf import OmegaConf

    cfg = load_cfg(onnx_dir)
    cfg["onnx_dir"] = str(onnx_dir)
    args = OmegaConf.create(cfg)
    model = foundation_stereo.TrtRunner(
        args,
        str(onnx_dir / "feature_runner.engine"),
        str(onnx_dir / "post_runner.engine"),
    )
    height, width = int(args.image_size[0]), int(args.image_size[1])
    img0 = torch_module.randint(0, 256, (1, 3, height, width), dtype=torch_module.float32).cuda()
    img1 = torch_module.randint(0, 256, (1, 3, height, width), dtype=torch_module.float32).cuda()
    times = []
    for idx in range(total):
        torch_module.cuda.synchronize()
        start = time.perf_counter()
        model.forward(img0, img1)
        torch_module.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        logging.info("TensorRT iter %02d: %.1f ms%s", idx, elapsed * 1000.0, " (warmup)" if idx < warmup else "")
    measure_times = times[warmup:]
    return float(np.mean(measure_times) * 1000.0)


def profile_pytorch(
    *,
    torch_module,
    model_path: Path,
    height: int,
    width: int,
    valid_iters: int,
    max_disp: int,
    warmup: int,
    total: int,
) -> float:
    import yaml
    from core.utils.utils import InputPadder
    from Utils import AMP_DTYPE, set_logging_format, set_seed
    from omegaconf import OmegaConf

    set_logging_format()
    set_seed(0)
    with open(model_path.parent / "cfg.yaml", "r", encoding="utf-8") as handle:
        cfg: dict = yaml.safe_load(handle)
    cfg.update(
        {
            "model_dir": str(model_path),
            "valid_iters": valid_iters,
            "max_disp": max_disp,
        }
    )
    args = OmegaConf.create(cfg)
    model = torch_module.load(str(model_path), map_location="cpu", weights_only=False)
    model.args.valid_iters = valid_iters
    model.args.max_disp = max_disp
    model.cuda().eval()
    img0 = torch_module.randint(0, 256, (1, 3, height, width), dtype=torch_module.float32).cuda()
    img1 = torch_module.randint(0, 256, (1, 3, height, width), dtype=torch_module.float32).cuda()
    padder = InputPadder(img0.shape, divis_by=32, force_square=False)
    img0, img1 = padder.pad(img0, img1)

    times = []
    with torch_module.amp.autocast("cuda", enabled=True, dtype=AMP_DTYPE):
        for idx in range(total):
            torch_module.cuda.synchronize()
            start = time.perf_counter()
            model.forward(img0, img1, iters=args.valid_iters, test_mode=True, optimize_build_volume="pytorch1")
            torch_module.cuda.synchronize()
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            logging.info("PyTorch iter %02d: %.1f ms%s", idx, elapsed * 1000.0, " (warmup)" if idx < warmup else "")
    measure_times = times[warmup:]
    return float(np.mean(measure_times) * 1000.0)


def main() -> int:
    args = parse_args()
    ffs_repo = Path(args.ffs_repo).resolve()
    model_path = Path(args.model_path).resolve()
    out_dir = Path(args.out_dir).resolve()
    left_file, right_file, intrinsic_file = resolve_demo_paths(ffs_repo)
    ensure_required_paths([ffs_repo, model_path, left_file, right_file, intrinsic_file])
    if int(args.height) % 32 != 0 or int(args.width) % 32 != 0:
        raise ValueError(f"Expected engine size divisible by 32, got {args.width}x{args.height}")

    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch_module, foundation_stereo = prepare_python_runtime(ffs_repo)
    trt_version = verify_python_deps()
    logging.info("TensorRT Python runtime: %s", trt_version)

    export_onnx(
        torch_module=torch_module,
        foundation_stereo=foundation_stereo,
        model_path=model_path,
        out_dir=out_dir,
        height=args.height,
        width=args.width,
        valid_iters=args.valid_iters,
        max_disp=args.max_disp,
    )
    build_engine_from_onnx(
        onnx_path=out_dir / "feature_runner.onnx",
        engine_path=out_dir / "feature_runner.engine",
        log_path=out_dir / "feature_engine_build.log",
        workspace_gib=args.workspace_gib,
        fp16=True,
    )
    build_engine_from_onnx(
        onnx_path=out_dir / "post_runner.onnx",
        engine_path=out_dir / "post_runner.engine",
        log_path=out_dir / "post_engine_build.log",
        workspace_gib=args.workspace_gib,
        fp16=True,
    )
    run_demo(
        torch_module=torch_module,
        foundation_stereo=foundation_stereo,
        onnx_dir=out_dir,
        left_file=left_file,
        right_file=right_file,
        intrinsic_file=intrinsic_file,
        out_dir=out_dir / "demo_out",
        zfar=args.zfar,
    )

    if args.skip_profiles:
        print(f"Verified TensorRT ONNX/engine/demo outputs in {out_dir}")
        return 0

    trt_ms = profile_tensorrt(
        torch_module=torch_module,
        foundation_stereo=foundation_stereo,
        onnx_dir=out_dir,
        warmup=args.warmup,
        total=args.total,
    )
    pytorch_ms = profile_pytorch(
        torch_module=torch_module,
        model_path=model_path,
        height=args.height,
        width=args.width,
        valid_iters=args.valid_iters,
        max_disp=args.max_disp,
        warmup=args.warmup,
        total=args.total,
    )
    print(f"TensorRT average after warmup: {trt_ms:.1f} ms")
    print(f"PyTorch average after warmup: {pytorch_ms:.1f} ms")
    print(f"Verified TensorRT ONNX/engine/demo/profile outputs in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
