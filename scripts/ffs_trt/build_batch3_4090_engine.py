from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_FFS_REPO = Path("/home/xinjie/Fast-FoundationStereo")
DEFAULT_WEIGHT = DEFAULT_FFS_REPO / "weights" / "20-30-48" / "model_best_bp2_serialize.pth"
DEFAULT_OUT_DIR = (
    ROOT
    / "data"
    / "experiments"
    / "ffs_trt_4090_848x480_pad864_builderopt5_batch3"
    / "engines"
    / "model_20-30-48_iters_4_res_480x864_batch3"
)
DEFAULT_TIMING_CACHE = (
    ROOT
    / "data"
    / "experiments"
    / "ffs_trt_4090_848x480_pad864_builderopt5_batch3"
    / "timing_cache.bin"
)
BATCH1_PATH_TOKEN = "ffs_trt_4090_848x480_pad864_builderopt5/engines/model_20-30-48_iters_4_res_480x864"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an isolated RTX 4090 Fast-FoundationStereo TensorRT batch=3 engine.")
    parser.add_argument("--ffs-repo", type=Path, default=DEFAULT_FFS_REPO)
    parser.add_argument("--weight", type=Path, default=DEFAULT_WEIGHT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--model", default="20-30-48")
    parser.add_argument("--valid-iters", type=int, default=4)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=864)
    parser.add_argument("--capture-height", type=int, default=480)
    parser.add_argument("--capture-width", type=int, default=848)
    parser.add_argument("--builder-optimization-level", type=int, default=5)
    parser.add_argument("--max-disp", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--workspace-gib", type=int, default=8)
    parser.add_argument("--timing-cache", type=Path, default=DEFAULT_TIMING_CACHE)
    parser.add_argument("--debug", action="store_true")
    return parser


def _require_paths(*paths: Path) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required path(s): " + ", ".join(missing))


def _guard_output_dir(out_dir: Path) -> None:
    resolved = str(out_dir.resolve())
    if BATCH1_PATH_TOKEN in resolved or not out_dir.name.endswith("_batch3"):
        raise ValueError(
            "Refusing to write batch=3 artifacts outside the dedicated batch3 directory. "
            f"Got --out-dir={out_dir}"
        )


def _make_timing_cache(config: Any, timing_cache: Path | None) -> None:
    if timing_cache is None:
        return
    payload = timing_cache.read_bytes() if timing_cache.exists() else b""
    cache = config.create_timing_cache(payload)
    config.set_timing_cache(cache, ignore_mismatch=False)


def _save_timing_cache(config: Any, timing_cache: Path | None) -> None:
    if timing_cache is None:
        return
    cache = config.get_timing_cache()
    if cache is None:
        return
    timing_cache.parent.mkdir(parents=True, exist_ok=True)
    timing_cache.write_bytes(bytes(cache.serialize()))


def build_engine_from_onnx(
    *,
    onnx_path: Path,
    engine_path: Path,
    log_path: Path,
    workspace_gib: int,
    fp16: bool,
    builder_optimization_level: int,
    timing_cache: Path | None,
) -> dict[str, Any]:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    parse_start_s = time.perf_counter()
    if not parser.parse(onnx_path.read_bytes()):
        errors = [str(parser.get_error(idx)) for idx in range(parser.num_errors)]
        log_path.write_text("\n".join(errors), encoding="utf-8")
        raise RuntimeError(f"Failed to parse ONNX file {onnx_path}; see {log_path}")
    parse_ms = (time.perf_counter() - parse_start_s) * 1000.0

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gib) << 30)
    config.builder_optimization_level = int(builder_optimization_level)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    _make_timing_cache(config, timing_cache)

    build_start_s = time.perf_counter()
    serialized_engine = builder.build_serialized_network(network, config)
    build_ms = (time.perf_counter() - build_start_s) * 1000.0
    if serialized_engine is None:
        log_path.write_text("builder.build_serialized_network returned None\n", encoding="utf-8")
        raise RuntimeError(f"Failed to build TensorRT engine for {onnx_path}; see {log_path}")
    engine_path.write_bytes(bytes(serialized_engine))
    _save_timing_cache(config, timing_cache)

    input_shapes: dict[str, list[int]] = {}
    output_shapes: dict[str, list[int]] = {}
    for idx in range(network.num_inputs):
        tensor = network.get_input(idx)
        input_shapes[tensor.name] = [int(dim) for dim in tensor.shape]
    for idx in range(network.num_outputs):
        tensor = network.get_output(idx)
        output_shapes[tensor.name] = [int(dim) for dim in tensor.shape]

    lines = [
        f"onnx={onnx_path}",
        f"engine={engine_path}",
        f"workspace_gib={int(workspace_gib)}",
        f"fp16={bool(fp16)}",
        f"builder_optimization_level={int(builder_optimization_level)}",
        f"timing_cache={timing_cache}",
        f"parse_ms={parse_ms:.2f}",
        f"build_ms={build_ms:.2f}",
        f"input_shapes={json.dumps(input_shapes, sort_keys=True)}",
        f"output_shapes={json.dumps(output_shapes, sort_keys=True)}",
    ]
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "onnx": str(onnx_path),
        "engine": str(engine_path),
        "workspace_gib": int(workspace_gib),
        "fp16": bool(fp16),
        "builder_optimization_level": int(builder_optimization_level),
        "timing_cache": "" if timing_cache is None else str(timing_cache),
        "parse_ms": float(parse_ms),
        "build_ms": float(build_ms),
        "input_shapes": input_shapes,
        "output_shapes": output_shapes,
    }


def _environment_metadata(*, trt_version: str) -> dict[str, Any]:
    import torch

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    capability = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else ()
    return {
        "gpu": gpu_name,
        "gpu_capability": list(capability),
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
        "tensorrt": str(trt_version),
        "cuda_home": os.environ.get("CUDA_HOME", ""),
        "torch_cuda_arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST", ""),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }


def write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        import yaml
    except Exception:
        return
    path.with_suffix(".yaml").write_text(yaml.safe_dump(metadata, sort_keys=True), encoding="utf-8")


def rewrite_pairwise_average_pool_to_reduce_mean(onnx_path: Path) -> list[str]:
    """Rewrite static 1x2 AveragePool nodes that TensorRT fails to tactic-select at batch=3.

    The FFS post runner trace contains two pairwise horizontal averages after reshaping
    stereo tensors into a very large effective N dimension. TensorRT 10.16 can build the
    batch=1 graph, but the same AveragePool nodes with batch=3 may fail all backend
    strategies. Reshape + ReduceMean is semantically equivalent for these static shapes
    and avoids that pooling tactic path.
    """
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    model = onnx.load(str(onnx_path))
    inferred = onnx.shape_inference.infer_shapes(model)
    shapes: dict[str, list[int]] = {}
    for value_info in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output):
        tensor_type = value_info.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        dims: list[int] = []
        for dim in tensor_type.shape.dim:
            if not dim.dim_value:
                dims = []
                break
            dims.append(int(dim.dim_value))
        if dims:
            shapes[value_info.name] = dims

    rewritten: list[str] = []
    new_nodes = []
    for node in model.graph.node:
        if node.op_type != "AveragePool":
            new_nodes.append(node)
            continue
        attrs = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
        if (
            list(attrs.get("kernel_shape", [])) != [1, 2]
            or list(attrs.get("strides", [])) != [1, 2]
            or list(attrs.get("pads", [])) != [0, 0, 0, 0]
            or int(attrs.get("ceil_mode", 0)) != 0
        ):
            new_nodes.append(node)
            continue
        input_name = node.input[0]
        output_name = node.output[0]
        input_shape = shapes.get(input_name)
        if input_shape is None or len(input_shape) != 4 or input_shape[-1] % 2 != 0:
            new_nodes.append(node)
            continue

        reshape_shape = [input_shape[0], input_shape[1], input_shape[2], input_shape[3] // 2, 2]
        shape_name = f"{output_name}_pairwise_reduce_shape"
        reshaped_name = f"{output_name}_pairwise_reduce_reshape"
        model.graph.initializer.extend(
            [
                numpy_helper.from_array(np.asarray(reshape_shape, dtype=np.int64), name=shape_name),
            ]
        )
        new_nodes.append(
            helper.make_node(
                "Reshape",
                [input_name, shape_name],
                [reshaped_name],
                name=f"{node.name}_PairwiseReshape",
            )
        )
        new_nodes.append(
            helper.make_node(
                "ReduceMean",
                [reshaped_name],
                [output_name],
                name=f"{node.name}_PairwiseReduceMean",
                axes=[4],
                keepdims=0,
            )
        )
        rewritten.append(node.name or output_name)

    if not rewritten:
        return []
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    onnx.checker.check_model(model)
    onnx.save(model, str(onnx_path))
    return rewritten


def patch_batch_safe_gwc_volume_triton(foundation_stereo: Any) -> None:
    """Patch only this batch3 build process for a batch>1 stride issue in upstream FFS."""
    import torch
    import core.submodule as submodule

    triton = getattr(submodule, "triton", None)
    if triton is None:
        raise RuntimeError("Triton is required for the two-stage FFS TensorRT export path.")
    kernel = getattr(submodule, "_gwc_triton_kernel")

    @torch.no_grad()
    def build_gwc_volume_triton_batch_safe(
        refimg_fea: Any,
        targetimg_fea: Any,
        maxdisp: int,
        num_groups: int,
        normalize: bool = True,
    ) -> Any:
        if triton is None:
            raise RuntimeError("Triton is not available. Please install triton to use build_gwc_volume_triton.")
        batch, channels, height, width = refimg_fea.shape
        assert maxdisp > 0 and channels % num_groups == 0
        group_channels = channels // num_groups
        in_dtype = refimg_fea.dtype if refimg_fea.dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.float32

        if normalize:
            ref_norm = refimg_fea.float().reshape(batch, num_groups, group_channels, height, width).norm(dim=2)
            tar_norm = targetimg_fea.float().reshape(batch, num_groups, group_channels, height, width).norm(dim=2)
            ref_norm = ref_norm.permute(0, 2, 1, 3).reshape(batch * height, num_groups, width).to(in_dtype).contiguous()
            tar_norm = tar_norm.permute(0, 2, 1, 3).reshape(batch * height, num_groups, width).to(in_dtype).contiguous()
        else:
            ref_norm = refimg_fea.new_empty((1, 1, 1), dtype=in_dtype)
            tar_norm = refimg_fea.new_empty((1, 1, 1), dtype=in_dtype)

        ref = refimg_fea.to(in_dtype)
        tar = targetimg_fea.to(in_dtype)
        ref_bhwc = ref.permute(0, 2, 3, 1).reshape(batch * height, width, channels).contiguous()
        tar_bhwc = tar.permute(0, 2, 3, 1).reshape(batch * height, width, channels).contiguous()
        out_bhw = torch.empty((batch * height, num_groups, maxdisp, width), device=ref.device, dtype=in_dtype)
        batch_height = batch * height
        d_eff = min(maxdisp, width)
        grid = lambda meta: (
            batch_height * num_groups,
            triton.cdiv(d_eff, meta["BLOCK_D"]),
            triton.cdiv(width, meta["BLOCK_W"]),
        )
        kernel[grid](
            ref_bhwc,
            tar_bhwc,
            ref_norm,
            tar_norm,
            out_bhw,
            batch_height,
            channels,
            width,
            d_eff,
            num_groups,
            group_channels,
            ref_bhwc.stride(0),
            ref_bhwc.stride(1),
            ref_bhwc.stride(2),
            tar_bhwc.stride(0),
            tar_bhwc.stride(1),
            tar_bhwc.stride(2),
            ref_norm.stride(0),
            ref_norm.stride(1),
            ref_norm.stride(2),
            out_bhw.stride(0),
            out_bhw.stride(1),
            out_bhw.stride(2),
            out_bhw.stride(3),
            NORMALIZE=normalize,
        )
        if d_eff < maxdisp:
            out_bhw[:, :, d_eff:, :] = 0
        return out_bhw.reshape(batch, height, num_groups, maxdisp, width).permute(0, 2, 3, 1, 4).contiguous()

    submodule.build_gwc_volume_triton = build_gwc_volume_triton_batch_safe
    foundation_stereo.build_gwc_volume_triton = build_gwc_volume_triton_batch_safe


def export_onnx_batch3(
    *,
    torch_module: Any,
    foundation_stereo: Any,
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

    model = torch_module.load(str(model_path), map_location="cpu", weights_only=False)
    model.args.max_disp = int(max_disp)
    model.args.valid_iters = int(valid_iters)
    model.cuda().eval()

    feature_runner = foundation_stereo.TrtFeatureRunner(model).cuda().eval()
    post_runner = foundation_stereo.TrtPostRunner(model).cuda().eval()
    left_img = torch_module.randn(batch_size, 3, height, width, device="cuda", dtype=torch_module.float32) * 255
    right_img = torch_module.randn(batch_size, 3, height, width, device="cuda", dtype=torch_module.float32) * 255

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
        dynamo=False,
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
        dynamo=False,
    )
    rewritten = rewrite_pairwise_average_pool_to_reduce_mean(out_dir / "post_runner.onnx")
    if rewritten:
        (out_dir / "post_runner_batch3_onnx_rewrites.json").write_text(
            json.dumps({"rewritten_average_pool_nodes": rewritten}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    with open(out_dir / "onnx.yaml", "w", encoding="utf-8") as handle:
        cfg = OmegaConf.to_container(model.args)
        cfg["image_size"] = [int(height), int(width)]
        yaml.safe_dump(cfg, handle)


def main(argv: list[str] | None = None) -> int:
    from data_process.depth_backends.fast_foundation_stereo import resolve_tensorrt_engine_static_batch_size
    from scripts.harness.verify_ffs_tensorrt_wsl import prepare_python_runtime, verify_python_deps

    args = build_parser().parse_args(argv)
    if int(args.batch_size) != 3:
        raise ValueError("This experiment script is batch=3-specific; expected --batch-size 3.")
    if int(args.height) % 32 != 0 or int(args.width) % 32 != 0:
        raise ValueError(f"TensorRT engine shape must be divisible by 32, got {args.height}x{args.width}.")
    if int(args.builder_optimization_level) < 0:
        raise ValueError("--builder-optimization-level must be non-negative.")
    out_dir = Path(args.out_dir).resolve()
    _guard_output_dir(out_dir)
    _require_paths(Path(args.ffs_repo), Path(args.weight))
    out_dir.mkdir(parents=True, exist_ok=True)

    torch_module, foundation_stereo = prepare_python_runtime(Path(args.ffs_repo).resolve())
    patch_batch_safe_gwc_volume_triton(foundation_stereo)
    trt_version = verify_python_deps()
    if bool(args.debug):
        print(f"[batch3-build] TensorRT={trt_version} out_dir={out_dir}", flush=True)

    export_onnx_batch3(
        torch_module=torch_module,
        foundation_stereo=foundation_stereo,
        model_path=Path(args.weight).resolve(),
        out_dir=out_dir,
        height=int(args.height),
        width=int(args.width),
        batch_size=int(args.batch_size),
        valid_iters=int(args.valid_iters),
        max_disp=int(args.max_disp),
    )
    feature_build = build_engine_from_onnx(
        onnx_path=out_dir / "feature_runner.onnx",
        engine_path=out_dir / "feature_runner.engine",
        log_path=out_dir / "feature_engine_build.log",
        workspace_gib=int(args.workspace_gib),
        fp16=True,
        builder_optimization_level=int(args.builder_optimization_level),
        timing_cache=Path(args.timing_cache).resolve(),
    )
    post_build = build_engine_from_onnx(
        onnx_path=out_dir / "post_runner.onnx",
        engine_path=out_dir / "post_runner.engine",
        log_path=out_dir / "post_engine_build.log",
        workspace_gib=int(args.workspace_gib),
        fp16=True,
        builder_optimization_level=int(args.builder_optimization_level),
        timing_cache=Path(args.timing_cache).resolve(),
    )

    static_batch_size = resolve_tensorrt_engine_static_batch_size(trt_mode="two_stage", model_dir=out_dir)
    if int(static_batch_size) != int(args.batch_size):
        raise RuntimeError(f"Built engine static batch={static_batch_size}, expected {args.batch_size}.")

    metadata = {
        "batch_size": int(args.batch_size),
        "model": str(args.model),
        "valid_iters": int(args.valid_iters),
        "height": int(args.height),
        "width": int(args.width),
        "capture_height": int(args.capture_height),
        "capture_width": int(args.capture_width),
        "builder_optimization_level": int(args.builder_optimization_level),
        "max_disp": int(args.max_disp),
        "ffs_repo": str(Path(args.ffs_repo).resolve()),
        "weight": str(Path(args.weight).resolve()),
        "out_dir": str(out_dir),
        "timing_cache": str(Path(args.timing_cache).resolve()),
        "static_batch_size": int(static_batch_size),
        "feature_build": feature_build,
        "post_build": post_build,
        "environment": _environment_metadata(trt_version=trt_version),
    }
    write_metadata(out_dir / "batch3_metadata.json", metadata)
    print(json.dumps({"status": "pass", "out_dir": str(out_dir), "static_batch_size": int(static_batch_size)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
