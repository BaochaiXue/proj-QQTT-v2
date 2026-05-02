from __future__ import annotations

import argparse
import json
import statistics
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_MODEL_ID = "yonigozlan/EdgeTAM-hf"
np: Any = None
Image: Any = None
torch: Any = None
transformers: Any = None
EdgeTamVideoInferenceSession: Any = None
EdgeTamVideoModel: Any = None
Sam2VideoProcessor: Any = None


def _load_runtime_dependencies() -> None:
    global EdgeTamVideoInferenceSession
    global EdgeTamVideoModel
    global Image
    global Sam2VideoProcessor
    global np
    global torch
    global transformers

    import numpy as runtime_np
    from PIL import Image as runtime_image
    import torch as runtime_torch
    import transformers as runtime_transformers
    from transformers import (
        EdgeTamVideoInferenceSession as runtime_edge_session,
        EdgeTamVideoModel as runtime_model,
        Sam2VideoProcessor as runtime_processor,
    )

    np = runtime_np
    Image = runtime_image
    torch = runtime_torch
    transformers = runtime_transformers
    EdgeTamVideoInferenceSession = runtime_edge_session
    EdgeTamVideoModel = runtime_model
    Sam2VideoProcessor = runtime_processor


def _dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(f"unsupported dtype: {name}") from exc


def _make_frames(width: int, height: int, count: int) -> list[Image.Image]:
    frames: list[Image.Image] = []
    square_w = max(48, width // 5)
    square_h = max(48, height // 4)
    y0 = max(0, height // 2 - square_h // 2)
    x_start = max(0, width // 4)
    max_step = max(1, (width - x_start - square_w - 1) // max(1, count - 1))

    for idx in range(count):
        image = np.zeros((height, width, 3), dtype=np.uint8)
        x0 = min(width - square_w, x_start + idx * max_step)
        image[y0 : y0 + square_h, x0 : x0 + square_w] = 255
        frames.append(Image.fromarray(image))
    return frames


def _sync_if_needed(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _tensor_shape(value: Any) -> list[int]:
    return [int(dim) for dim in tuple(value.shape)]


def _env_report(device: str) -> dict[str, Any]:
    report: dict[str, Any] = {
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": device,
        "classes": {
            "EdgeTamVideoModel": EdgeTamVideoModel.__module__,
            "Sam2VideoProcessor": Sam2VideoProcessor.__module__,
            "EdgeTamVideoInferenceSession": EdgeTamVideoInferenceSession.__module__,
        },
    }
    if torch.cuda.is_available():
        report["gpu"] = torch.cuda.get_device_name(0)
    return report


def _init_streaming_session(
    processor: Sam2VideoProcessor,
    device: str,
    dtype: torch.dtype,
    mode: str,
) -> Any:
    if mode == "edgetam":
        return EdgeTamVideoInferenceSession(
            video=None,
            inference_device=device,
            inference_state_device=device,
            video_storage_device=device,
            dtype=dtype,
        )
    if mode == "processor":
        return processor.init_video_session(
            inference_device=device,
            dtype=dtype,
        )
    raise ValueError(f"unsupported session init mode: {mode}")


def _write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# HF EdgeTAM Streaming Validation",
        "",
        f"- Timestamp UTC: `{result['timestamp_utc']}`",
        f"- Status: `{result['status']}`",
        f"- Model: `{result['model_id']}`",
        f"- Environment: `{result['environment']}`",
        f"- Device: `{result['env'].get('device')}`",
        f"- GPU: `{result['env'].get('gpu', 'n/a')}`",
        f"- Torch: `{result['env'].get('torch')}`",
        f"- Torch CUDA: `{result['env'].get('torch_cuda')}`",
        f"- Transformers: `{result['env'].get('transformers')}`",
        "",
        "## API",
        "",
        "- `EdgeTamVideoModel`: import OK",
        "- `Sam2VideoProcessor`: import OK",
        "- `EdgeTamVideoInferenceSession`: import OK",
        f"- Session init mode: `{result['session']['init_mode']}`",
        f"- Session class: `{result['session']['class']}`",
        "- Streaming session initialized without a complete video.",
        "- Frame 0 point prompt was added with `original_size`.",
        "",
        "## Synthetic Streaming Smoke",
        "",
        f"- Frames: `{result['frames']}`",
        f"- Frame size: `{result['height']}x{result['width']}`",
        f"- Dtype: `{result['dtype']}`",
        f"- Mean latency ms: `{result['latency_ms']['mean']:.3f}`",
        f"- Median latency ms: `{result['latency_ms']['median']:.3f}`",
        f"- Min latency ms: `{result['latency_ms']['min']:.3f}`",
        f"- Max latency ms: `{result['latency_ms']['max']:.3f}`",
        "",
        "## Per Frame",
        "",
        "| frame | latency_ms | mask_shape | mask_mean |",
        "| ---: | ---: | --- | ---: |",
    ]
    for item in result["per_frame"]:
        lines.append(
            f"| {item['frame_idx']} | {item['latency_ms']:.3f} | "
            f"`{item['mask_shape']}` | {item['mask_mean']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            "HF EdgeTAMVideo streaming proof-of-life passed on synthetic frames. "
            "This confirms the session-style protocol works separately from the patched official EdgeTAM backend.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = args.device
    dtype = _dtype_from_name(args.dtype)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested, but torch.cuda.is_available() is false")

    print(f"loading model: {args.model_id}")
    model = EdgeTamVideoModel.from_pretrained(args.model_id).to(device, dtype=dtype).eval()
    processor = Sam2VideoProcessor.from_pretrained(args.model_id)
    frames = _make_frames(args.width, args.height, args.frames)

    session = _init_streaming_session(processor, device=device, dtype=dtype, mode=args.session_init)
    print(f"session: {type(session).__module__}.{type(session).__name__}")

    per_frame: list[dict[str, Any]] = []
    autocast_ctx = torch.autocast("cuda", dtype=dtype) if device.startswith("cuda") else nullcontext()

    with torch.inference_mode(), autocast_ctx:
        for frame_idx, frame in enumerate(frames):
            inputs = processor(images=frame, device=device, return_tensors="pt")
            pixel_values = inputs.pixel_values[0].to(device=device, dtype=dtype)

            if frame_idx == 0:
                x = args.width // 4 + max(48, args.width // 5) // 2
                y = args.height // 2
                processor.add_inputs_to_inference_session(
                    inference_session=session,
                    frame_idx=0,
                    obj_ids=1,
                    input_points=[[[[float(x), float(y)]]]],
                    input_labels=[[[1]]],
                    original_size=inputs.original_sizes[0],
                )

            _sync_if_needed(device)
            start = time.perf_counter()
            output = model(
                inference_session=session,
                frame=pixel_values,
            )
            _sync_if_needed(device)
            latency_ms = (time.perf_counter() - start) * 1000.0

            masks = processor.post_process_masks(
                [output.pred_masks],
                original_sizes=inputs.original_sizes,
                binarize=False,
            )[0]
            mask_mean = float(masks.float().mean().item())
            item = {
                "frame_idx": frame_idx,
                "latency_ms": latency_ms,
                "mask_shape": _tensor_shape(masks),
                "mask_mean": mask_mean,
            }
            per_frame.append(item)
            print(
                f"frame={frame_idx}, latency_ms={latency_ms:.2f}, "
                f"mask_shape={tuple(item['mask_shape'])}, mask_mean={mask_mean:.6f}"
            )

    latencies = [item["latency_ms"] for item in per_frame]
    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "environment": "edgetam-hf-stream",
        "model_id": args.model_id,
        "frames": args.frames,
        "width": args.width,
        "height": args.height,
        "dtype": args.dtype,
        "env": _env_report(device),
        "session": {
            "init_mode": args.session_init,
            "class": f"{type(session).__module__}.{type(session).__name__}",
        },
        "latency_ms": {
            "mean": statistics.fmean(latencies),
            "median": statistics.median(latencies),
            "min": min(latencies),
            "max": max(latencies),
        },
        "per_frame": per_frame,
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify Hugging Face EdgeTAMVideo frame-by-frame streaming inference."
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument(
        "--session-init",
        choices=("edgetam", "processor"),
        default="edgetam",
        help="Use EdgeTamVideoInferenceSession directly or Sam2VideoProcessor.init_video_session().",
    )
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _load_runtime_dependencies()
    result = run(args)

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"wrote JSON: {args.json_output}")
    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        _write_markdown(args.markdown_output, result)
        print(f"wrote markdown: {args.markdown_output}")

    print("HF EdgeTAM streaming smoke PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
