from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
MEAN_RGB = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD_RGB = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _trt_to_torch_dtype(dtype: Any) -> Any:
    import tensorrt as trt
    import torch

    if dtype == trt.float32:
        return torch.float32
    if dtype == trt.float16:
        return torch.float16
    if dtype == trt.int32:
        return torch.int32
    if dtype == trt.int64:
        return torch.int64
    if dtype == trt.bool:
        return torch.bool
    raise TypeError(f"Unsupported TensorRT dtype: {dtype}")


def _load_engine(path: Path) -> tuple[Any, Any]:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(path.read_bytes())
    if engine is None:
        raise RuntimeError(f"Failed to deserialize TensorRT engine: {path}")
    context = engine.create_execution_context()
    return engine, context


def _io_names(engine: Any) -> tuple[list[str], list[str]]:
    import tensorrt as trt

    inputs: list[str] = []
    outputs: list[str] = []
    for idx in range(engine.num_io_tensors):
        name = engine.get_tensor_name(idx)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            inputs.append(name)
        elif mode == trt.TensorIOMode.OUTPUT:
            outputs.append(name)
    return inputs, outputs


def _prepare_context(context: Any, inputs: dict[str, Any]) -> None:
    for name, tensor in inputs.items():
        context.set_input_shape(name, tuple(int(dim) for dim in tensor.shape))
    missing = context.infer_shapes()
    if missing:
        raise RuntimeError(f"TensorRT shape inference missing tensors: {missing}")


def _allocate_outputs(engine: Any, context: Any, output_names: list[str]) -> dict[str, Any]:
    import torch

    outputs: dict[str, Any] = {}
    for name in output_names:
        shape = tuple(int(dim) for dim in context.get_tensor_shape(name))
        dtype = _trt_to_torch_dtype(engine.get_tensor_dtype(name))
        outputs[name] = torch.empty(shape, device="cuda", dtype=dtype)
    return outputs


def _bind_tensors(context: Any, tensors: dict[str, Any]) -> None:
    for name, tensor in tensors.items():
        context.set_tensor_address(name, int(tensor.data_ptr()))


def _execute(context: Any, stream: Any) -> None:
    ok = context.execute_async_v3(stream_handle=int(stream.cuda_stream))
    if not ok:
        raise RuntimeError("TensorRT execute_async_v3 returned false")


def _timing_stats(values_ms: list[float]) -> dict[str, float]:
    arr = np.asarray(values_ms, dtype=np.float64)
    return {
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "p90_ms": float(np.percentile(arr, 90)),
        "p95_ms": float(np.percentile(arr, 95)),
        "fps_from_mean": float(1000.0 / arr.mean()),
    }


def _numeric_frame_paths(case_dir: Path, camera_idx: int) -> list[Path]:
    frame_dir = case_dir / "color" / str(camera_idx)
    paths = [path for path in frame_dir.glob("*.png") if path.stem.isdigit()]
    return sorted(paths, key=lambda path: int(path.stem))


def _preprocess_image(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    arr = rgb.astype(np.float32) / 255.0
    arr = (arr - MEAN_RGB) / STD_RGB
    return np.ascontiguousarray(arr.transpose(2, 0, 1)[None, ...])


def _bbox_from_mask(path: Path, *, default_width: int, default_height: int) -> tuple[float, float, float, float]:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None or not np.any(mask > 0):
        return (0.0, 0.0, float(default_width - 1), float(default_height - 1))
    ys, xs = np.nonzero(mask > 0)
    return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))


def _prompt_tensors(
    *,
    mask_path: Path,
    image_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    height, width = image.shape[:2]
    x0, y0, x1, y1 = _bbox_from_mask(mask_path, default_width=width, default_height=height)
    sx = 1024.0 / float(width)
    sy = 1024.0 / float(height)
    box = np.array([x0 * sx, y0 * sy, x1 * sx, y1 * sy], dtype=np.float32).reshape(1, 1, 4)
    point = np.array([[(box[0, 0, 0] + box[0, 0, 2]) * 0.5, (box[0, 0, 1] + box[0, 0, 3]) * 0.5]], dtype=np.float32)
    point = point.reshape(1, 1, 1, 2)
    labels = np.ones((1, 1, 1), dtype=np.int64)
    return point, labels, box


def _load_baseline_ms(path: Path) -> float | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in ("ms_per_frame", "mean_ms", "avg_ms"):
        if key in payload:
            return float(payload[key])
    if "fps" in payload:
        fps = float(payload["fps"])
        return 1000.0 / fps if fps > 0 else None
    if "timing" in payload and isinstance(payload["timing"], dict):
        timing = payload["timing"]
        for key in ("ms_per_frame", "mean_ms", "avg_ms"):
            if key in timing:
                return float(timing[key])
    return None


def benchmark_camera(
    *,
    camera_idx: int,
    case_dir: Path,
    mask_root: Path,
    encoder_engine_path: Path,
    decoder_engine_path: Path,
    warmups: int,
    timed_iters: int,
) -> dict[str, Any]:
    import torch

    encoder_engine, encoder_context = _load_engine(encoder_engine_path)
    decoder_engine, decoder_context = _load_engine(decoder_engine_path)
    encoder_inputs, encoder_outputs_names = _io_names(encoder_engine)
    decoder_inputs, decoder_outputs_names = _io_names(decoder_engine)
    if encoder_inputs != ["pixel_values"]:
        raise RuntimeError(f"Unexpected encoder inputs: {encoder_inputs}")

    frame_paths = _numeric_frame_paths(case_dir, camera_idx)
    if not frame_paths:
        raise FileNotFoundError(f"No PNG frames found for camera {camera_idx} under {case_dir}")

    frames_cuda: list[Any] = []
    prompt_cuda: list[dict[str, Any]] = []
    for frame_path in frame_paths:
        frame_idx = int(frame_path.stem)
        pixel_values = torch.from_numpy(_preprocess_image(frame_path)).to(device="cuda")
        mask_path = mask_root / str(camera_idx) / "0" / f"{frame_idx}.png"
        point, labels, box = _prompt_tensors(mask_path=mask_path, image_path=frame_path)
        frames_cuda.append(pixel_values)
        prompt_cuda.append(
            {
                "input_points": torch.from_numpy(point).to(device="cuda"),
                "input_labels": torch.from_numpy(labels).to(device="cuda"),
                "input_boxes": torch.from_numpy(box).to(device="cuda"),
            }
        )

    encoder_first_inputs = {"pixel_values": frames_cuda[0]}
    _prepare_context(encoder_context, encoder_first_inputs)
    encoder_outputs = _allocate_outputs(encoder_engine, encoder_context, encoder_outputs_names)
    _bind_tensors(encoder_context, {**encoder_first_inputs, **encoder_outputs})

    decoder_first_inputs = {**prompt_cuda[0], **encoder_outputs}
    _prepare_context(decoder_context, decoder_first_inputs)
    decoder_outputs = _allocate_outputs(decoder_engine, decoder_context, decoder_outputs_names)
    _bind_tensors(decoder_context, {**decoder_first_inputs, **decoder_outputs})

    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    def run_one(index: int, *, timed: bool) -> tuple[float, float, float] | None:
        frame = frames_cuda[index % len(frames_cuda)]
        prompts = prompt_cuda[index % len(prompt_cuda)]
        _bind_tensors(encoder_context, {"pixel_values": frame, **encoder_outputs})
        _bind_tensors(decoder_context, {**prompts, **encoder_outputs, **decoder_outputs})
        if not timed:
            _execute(encoder_context, stream)
            _execute(decoder_context, stream)
            return None

        start = torch.cuda.Event(enable_timing=True)
        after_encoder = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        _execute(encoder_context, stream)
        after_encoder.record(stream)
        _execute(decoder_context, stream)
        end.record(stream)
        end.synchronize()
        encoder_ms = float(start.elapsed_time(after_encoder))
        decoder_ms = float(after_encoder.elapsed_time(end))
        total_ms = float(start.elapsed_time(end))
        return encoder_ms, decoder_ms, total_ms

    for idx in range(int(warmups)):
        run_one(idx, timed=False)
    torch.cuda.synchronize()

    encoder_ms_values: list[float] = []
    decoder_ms_values: list[float] = []
    total_ms_values: list[float] = []
    for idx in range(int(timed_iters)):
        values = run_one(idx, timed=True)
        assert values is not None
        encoder_ms, decoder_ms, total_ms = values
        encoder_ms_values.append(encoder_ms)
        decoder_ms_values.append(decoder_ms)
        total_ms_values.append(total_ms)

    pred_masks = decoder_outputs.get("pred_masks")
    mask_stats = None
    if pred_masks is not None:
        mask_cpu = pred_masks.detach().float().cpu().numpy()
        mask_stats = {
            "shape": list(mask_cpu.shape),
            "mean": float(mask_cpu.mean()),
            "min": float(mask_cpu.min()),
            "max": float(mask_cpu.max()),
            "positive_ratio": float((mask_cpu > 0.0).mean()),
        }

    baseline_path = ROOT / "result/sloth_base_motion_ffs_mask_overlay_3x3/timings" / f"sloth_base_motion_ffs_cam{camera_idx}_edgetam.json"
    baseline_ms = _load_baseline_ms(baseline_path)
    total_mean = mean(total_ms_values)
    return {
        "camera_idx": int(camera_idx),
        "frames_available": len(frame_paths),
        "warmups": int(warmups),
        "timed_iters": int(timed_iters),
        "encoder": _timing_stats(encoder_ms_values),
        "decoder": _timing_stats(decoder_ms_values),
        "encoder_plus_decoder": _timing_stats(total_ms_values),
        "mask_output_stats": mask_stats,
        "baseline_edgetam_compiled_ms": baseline_ms,
        "speedup_vs_existing_edgetam_compiled": (baseline_ms / total_mean if baseline_ms else None),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark EdgeTAM TensorRT component engines on a real QQTT case.")
    parser.add_argument("--case-dir", type=Path, default=ROOT / "data/different_types/sloth_base_motion_ffs")
    parser.add_argument(
        "--mask-root",
        type=Path,
        default=ROOT / "result/sloth_base_motion_ffs_mask_overlay_3x3/sam31_masks/mask",
    )
    parser.add_argument("--encoder-engine", type=Path, required=True)
    parser.add_argument("--decoder-engine", type=Path, required=True)
    parser.add_argument("--camera-ids", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--timed-iters", type=int, default=100)
    parser.add_argument("--json-output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = [
        benchmark_camera(
            camera_idx=int(camera_idx),
            case_dir=args.case_dir.resolve(),
            mask_root=args.mask_root.resolve(),
            encoder_engine_path=args.encoder_engine.resolve(),
            decoder_engine_path=args.decoder_engine.resolve(),
            warmups=int(args.warmups),
            timed_iters=int(args.timed_iters),
        )
        for camera_idx in args.camera_ids
    ]

    total_means = [record["encoder_plus_decoder"]["mean_ms"] for record in records]
    payload = {
        "case_dir": str(args.case_dir.resolve()),
        "mask_root": str(args.mask_root.resolve()),
        "encoder_engine": str(args.encoder_engine.resolve()),
        "decoder_engine": str(args.decoder_engine.resolve()),
        "records": records,
        "aggregate": {
            "mean_encoder_plus_decoder_ms": float(mean(total_means)),
            "median_camera_encoder_plus_decoder_ms": float(median(total_means)),
            "fps_from_mean_encoder_plus_decoder": float(1000.0 / mean(total_means)),
        },
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    for record in records:
        total = record["encoder_plus_decoder"]
        print(
            f"cam{record['camera_idx']}: "
            f"encoder={record['encoder']['mean_ms']:.3f} ms, "
            f"decoder={record['decoder']['mean_ms']:.3f} ms, "
            f"total={total['mean_ms']:.3f} ms ({total['fps_from_mean']:.2f} FPS)"
        )
    print(
        "aggregate: "
        f"{payload['aggregate']['mean_encoder_plus_decoder_ms']:.3f} ms, "
        f"{payload['aggregate']['fps_from_mean_encoder_plus_decoder']:.2f} FPS"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
