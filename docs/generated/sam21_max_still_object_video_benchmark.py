#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor


DEFAULT_CASE_DIR = Path(
    "/home/zhangxinjie/proj-QQTT-v2/data_collect/"
    "both_30_still_object_round1_20260428"
)
DEFAULT_CKPT = Path(
    "/home/zhangxinjie/.cache/huggingface/sam2.1/sam2.1_hiera_large.pt"
)
DEFAULT_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEFAULT_WORK_DIR = Path("/tmp/qqtt_sam21_max_video_benchmark")

CAMERA_BOXES = {
    "0": [285.0, 95.0, 610.0, 479.0],
    "1": [600.0, 135.0, 820.0, 360.0],
    "2": [90.0, 65.0, 340.0, 330.0],
}


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def mark_compile_step() -> None:
    marker = getattr(torch.compiler, "cudagraph_mark_step_begin", None)
    if marker is not None:
        marker()


def timer_ms(fn):
    cuda_sync()
    start = time.perf_counter()
    value = fn()
    cuda_sync()
    return value, (time.perf_counter() - start) * 1000.0


def sorted_frames(frame_dir: Path) -> list[Path]:
    frames = sorted(frame_dir.glob("*.png"), key=lambda p: int(p.stem))
    if not frames:
        frames = sorted(frame_dir.glob("*.jpg"), key=lambda p: int(p.stem))
    return frames


def prepare_jpeg_frames(case_dir: Path, camera: str, frame_count: int, work_dir: Path) -> Path:
    source_dir = case_dir / "color" / camera
    frames = sorted_frames(source_dir)
    if len(frames) < frame_count:
        raise RuntimeError(f"{source_dir} has {len(frames)} frames, need {frame_count}")

    out_dir = work_dir / f"camera_{camera}_jpg"
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, src in enumerate(frames[:frame_count]):
        dst = out_dir / f"{idx:05d}.jpg"
        if dst.exists():
            continue
        Image.open(src).convert("RGB").save(dst, quality=95)
    return out_dir


def autocast_context(use_bfloat16: bool):
    if not use_bfloat16:
        return nullcontext()
    return torch.autocast("cuda", dtype=torch.bfloat16)


def run_image_smoke(args, use_bfloat16: bool):
    frame = sorted_frames(args.case_dir / "color" / args.cameras[0])[0]
    box = np.array(CAMERA_BOXES[args.cameras[0]], dtype=np.float32)

    def work():
        model = build_sam2(args.config, str(args.checkpoint), device="cuda")
        predictor = SAM2ImagePredictor(model)
        image = np.array(Image.open(frame).convert("RGB"))
        predictor.set_image(image)
        masks, scores, _ = predictor.predict(box=box, multimask_output=False)
        return {
            "frame": str(frame),
            "mask_shape": list(masks.shape),
            "score": float(scores[0]),
            "mask_pixels": int(masks[0].sum()),
        }

    with torch.inference_mode(), autocast_context(use_bfloat16):
        result, elapsed_ms = timer_ms(work)
    result["elapsed_ms"] = elapsed_ms
    torch.cuda.empty_cache()
    return result


def run_video(args, use_bfloat16: bool):
    predictor = build_sam2_video_predictor(
        args.config,
        str(args.checkpoint),
        device="cuda",
        vos_optimized=True,
    )
    meta = {
        "predictor_class": type(predictor).__name__,
        "compile_image_encoder_requested": True,
        "compiled_vos_components_requested": [
            "image_encoder",
            "memory_encoder",
            "memory_attention",
            "sam_prompt_encoder",
            "sam_mask_decoder",
        ],
        "vos_optimized_requested": True,
    }

    def run_camera(camera: str):
        video_dir = prepare_jpeg_frames(args.case_dir, camera, args.frames, args.work_dir)
        box = np.array(CAMERA_BOXES[camera], dtype=np.float32)

        def init_state():
            return predictor.init_state(
                video_path=str(video_dir),
                offload_video_to_cpu=False,
                offload_state_to_cpu=False,
                async_loading_frames=False,
            )

        with torch.inference_mode(), autocast_context(use_bfloat16):
            mark_compile_step()
            state, init_ms = timer_ms(init_state)

            def prompt():
                mark_compile_step()
                return predictor.add_new_points_or_box(
                    state,
                    frame_idx=0,
                    obj_id=1,
                    box=box,
                )

            (_, obj_ids, prompt_masks), prompt_ms = timer_ms(prompt)
            prompt_mask_pixels = int((prompt_masks[0] > 0).sum().item())

            def propagate():
                frame_outputs = 0
                last_mask_logits = None
                iterator = predictor.propagate_in_video(
                    state,
                    start_frame_idx=0,
                    max_frame_num_to_track=args.frames,
                    reverse=False,
                )
                while True:
                    mark_compile_step()
                    try:
                        _, _, out_mask_logits = next(iterator)
                    except StopIteration:
                        break
                    frame_outputs += 1
                    last_mask_logits = out_mask_logits
                return frame_outputs, last_mask_logits

            (frame_outputs, last_mask_logits), propagate_ms = timer_ms(propagate)
            last_mask_pixels = (
                int((last_mask_logits[0] > 0).sum().item())
                if last_mask_logits is not None
                else 0
            )

        total_ms = init_ms + prompt_ms + propagate_ms
        result = {
            "camera": camera,
            "video_dir": str(video_dir),
            "box_xyxy": CAMERA_BOXES[camera],
            "obj_ids": [int(x) for x in obj_ids],
            "frames_requested": args.frames,
            "frames_output": frame_outputs,
            "init_ms": init_ms,
            "prompt_ms": prompt_ms,
            "propagate_ms": propagate_ms,
            "propagate_ms_per_output_frame": propagate_ms / frame_outputs,
            "pipeline_ms_per_requested_frame": total_ms / args.frames,
            "prompt_mask_pixels": prompt_mask_pixels,
            "last_mask_pixels": last_mask_pixels,
        }

        del state
        torch.cuda.empty_cache()
        return result

    if args.warmup_camera is not None:
        meta["warmup"] = run_camera(args.warmup_camera)

    results = []
    for camera in args.cameras:
        results.append(run_camera(camera))

    meta["cameras"] = results
    meta["mean_propagate_ms_per_output_frame"] = float(
        np.mean([item["propagate_ms_per_output_frame"] for item in results])
    )
    meta["mean_pipeline_ms_per_requested_frame"] = float(
        np.mean([item["pipeline_ms_per_requested_frame"] for item in results])
    )
    return meta


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", type=Path, default=DEFAULT_CASE_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--config", default=DEFAULT_CFG)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--cameras", nargs="+", default=["0", "1", "2"])
    parser.add_argument("--warmup-camera")
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--no-image-smoke", action="store_true")
    parser.add_argument("--no-bfloat16", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SAM21-max verification")
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    cameras_to_check = list(args.cameras)
    if args.warmup_camera is not None:
        cameras_to_check.append(args.warmup_camera)
    for camera in cameras_to_check:
        if camera not in CAMERA_BOXES:
            raise KeyError(f"No default box for camera {camera}")

    use_bfloat16 = not args.no_bfloat16
    result = {
        "case_dir": str(args.case_dir),
        "checkpoint": str(args.checkpoint),
        "config": args.config,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "frames_per_camera": args.frames,
        "bfloat16_autocast": use_bfloat16,
    }
    if not args.no_image_smoke:
        result["image_smoke"] = run_image_smoke(args, use_bfloat16)
    result["video"] = run_video(args, use_bfloat16)

    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
