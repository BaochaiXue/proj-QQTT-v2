#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch

SAM_OBJECT_ID = 1
OUTPUT_OBJECT_ID = 0
EDGETAM_COMPILE_EAGER = "eager"
EDGETAM_COMPILE_IMAGE_ENCODER = "compile_image_encoder"
EDGETAM_COMPILE_NO_POS_CACHE = "compile_image_encoder_no_pos_cache_patch"


def _parse_text_prompts(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _normalize_label(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def _sorted_frame_tokens(case_dir: Path, *, camera_idx: int, frames: int | None) -> list[str]:
    paths = sorted((case_dir / "color" / str(int(camera_idx))).glob("*.png"), key=lambda path: int(path.stem))
    if frames is not None:
        paths = paths[: int(frames)]
    if not paths:
        raise FileNotFoundError(f"No RGB frames found for camera {camera_idx}: {case_dir}")
    return [path.stem for path in paths]


def _load_mask_info(mask_root: Path, *, camera_idx: int) -> dict[int, str]:
    info_path = mask_root / "mask" / f"mask_info_{int(camera_idx)}.json"
    if not info_path.is_file():
        return {}
    payload = json.loads(info_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Mask info must be a dict: {info_path}")
    return {int(key): str(value) for key, value in payload.items()}


def _mask_shape_from_color(case_dir: Path, *, camera_idx: int, frame_token: str) -> tuple[int, int]:
    image = cv2.imread(str(case_dir / "color" / str(int(camera_idx)) / f"{frame_token}.png"), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Missing RGB frame for mask shape: {case_dir} cam{camera_idx} frame {frame_token}")
    return int(image.shape[0]), int(image.shape[1])


def _load_union_mask(
    *,
    case_dir: Path,
    mask_root: Path | None = None,
    camera_idx: int,
    frame_token: str,
    text_prompt: str,
) -> np.ndarray:
    mask_root = Path(mask_root).resolve() if mask_root is not None else case_dir / "sam31_masks"
    prompts = {_normalize_label(item) for item in _parse_text_prompts(text_prompt)}
    mask_info = _load_mask_info(mask_root, camera_idx=int(camera_idx))
    matched_ids = [obj_id for obj_id, label in mask_info.items() if _normalize_label(label) in prompts]
    if not matched_ids:
        raise RuntimeError(f"No SAM3.1 masks match prompt {text_prompt!r} for cam{camera_idx}")
    height, width = _mask_shape_from_color(case_dir, camera_idx=int(camera_idx), frame_token=str(frame_token))
    union = np.zeros((height, width), dtype=bool)
    for obj_id in matched_ids:
        path = mask_root / "mask" / str(int(camera_idx)) / str(int(obj_id)) / f"{frame_token}.png"
        if not path.is_file():
            continue
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"Failed to read SAM3.1 mask: {path}")
        if image.shape[:2] != union.shape:
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)
        union |= image > 0
    if not np.any(union):
        raise RuntimeError(f"SAM3.1 union mask is empty for cam{camera_idx} frame {frame_token}")
    return union


def _prepare_jpeg_video_dir(
    *,
    case_dir: Path,
    camera_idx: int,
    frame_tokens: Sequence[str],
    work_dir: Path,
) -> Path:
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    for session_idx, frame_token in enumerate(frame_tokens):
        src = case_dir / "color" / str(int(camera_idx)) / f"{frame_token}.png"
        image = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Missing RGB frame: {src}")
        cv2.imwrite(str(work_dir / f"{session_idx:05d}.jpg"), image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return work_dir


def _cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time_ms(fn: Any) -> tuple[Any, float]:
    _cuda_sync()
    start = time.perf_counter()
    value = fn()
    _cuda_sync()
    return value, float((time.perf_counter() - start) * 1000.0)


def _write_single_object_masks(
    *,
    mask_root: Path,
    camera_idx: int,
    object_label: str,
    masks_by_frame_token: dict[str, np.ndarray],
    overwrite: bool,
) -> dict[str, Any]:
    mask_dir = mask_root / "mask"
    camera_dir = mask_dir / str(int(camera_idx))
    info_path = mask_dir / f"mask_info_{int(camera_idx)}.json"
    if (camera_dir.exists() or info_path.exists()) and overwrite:
        shutil.rmtree(camera_dir, ignore_errors=True)
        if info_path.exists():
            info_path.unlink()
    if camera_dir.exists() or info_path.exists():
        raise FileExistsError(f"EdgeTAM mask output already exists for camera {camera_idx}: {mask_root}")
    object_dir = camera_dir / str(OUTPUT_OBJECT_ID)
    object_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    info_path.write_text(json.dumps({str(OUTPUT_OBJECT_ID): str(object_label)}, indent=2), encoding="utf-8")
    saved = 0
    for frame_token, mask in sorted(masks_by_frame_token.items(), key=lambda item: int(item[0])):
        cv2.imwrite(str(object_dir / f"{frame_token}.png"), np.asarray(mask, dtype=np.uint8) * 255)
        saved += 1
    return {
        "mask_root": str(mask_root),
        "camera_idx": int(camera_idx),
        "object_id": int(OUTPUT_OBJECT_ID),
        "object_label": str(object_label),
        "saved_frame_count": int(saved),
    }


def _configure_torch() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("EdgeTAM worker requires CUDA.")
    props = torch.cuda.get_device_properties(0)
    if int(props.major) >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def _position_encoding_forward_no_cache():
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_embed = (
            torch.arange(1, x.shape[-2] + 1, dtype=torch.float32, device=x.device)
            .view(1, -1, 1)
            .repeat(x.shape[0], 1, x.shape[-1])
        )
        x_embed = (
            torch.arange(1, x.shape[-1] + 1, dtype=torch.float32, device=x.device)
            .view(1, 1, -1)
            .repeat(x.shape[0], x.shape[-2], 1)
        )

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

    return forward


def _patch_position_encoding_no_cache() -> bool:
    from sam2.modeling.position_encoding import PositionEmbeddingSine

    if getattr(PositionEmbeddingSine.forward, "_qqtt_no_cache_patch", False):
        return False
    forward = _position_encoding_forward_no_cache()
    forward._qqtt_no_cache_patch = True  # type: ignore[attr-defined]
    PositionEmbeddingSine.forward = forward
    return True


def _build_predictor(*, model_cfg: str, checkpoint: str, compile_mode: str):
    from sam2.build_sam import build_sam2_video_predictor

    compile_mode = str(compile_mode)
    overrides: list[str] = []
    metadata: dict[str, Any] = {
        "compile_mode": compile_mode,
        "hydra_overrides_extra": overrides,
        "position_encoding_no_cache_patch": False,
    }
    if compile_mode == EDGETAM_COMPILE_NO_POS_CACHE:
        metadata["position_encoding_no_cache_patch"] = True
        metadata["position_encoding_no_cache_patch_applied"] = _patch_position_encoding_no_cache()
        overrides.append("model.compile_image_encoder=true")
    elif compile_mode == EDGETAM_COMPILE_IMAGE_ENCODER:
        overrides.append("model.compile_image_encoder=true")
    elif compile_mode != EDGETAM_COMPILE_EAGER:
        raise ValueError(f"Unsupported EdgeTAM compile mode: {compile_mode}")

    predictor = build_sam2_video_predictor(
        str(model_cfg),
        str(checkpoint),
        device="cuda",
        hydra_overrides_extra=overrides,
    )
    metadata["hydra_overrides_extra"] = list(overrides)
    return predictor, metadata


def run_worker(args: argparse.Namespace) -> dict[str, Any]:
    _configure_torch()
    case_dir = Path(args.case_dir).resolve()
    output_mask_root = Path(args.output_mask_root).resolve()
    result_json = Path(args.result_json).resolve()
    result_json.parent.mkdir(parents=True, exist_ok=True)

    frame_tokens = _sorted_frame_tokens(case_dir, camera_idx=int(args.camera_idx), frames=args.frames)
    sam31_mask_root = Path(args.sam31_mask_root).resolve() if args.sam31_mask_root is not None else case_dir / "sam31_masks"
    init_mask = _load_union_mask(
        case_dir=case_dir,
        mask_root=sam31_mask_root,
        camera_idx=int(args.camera_idx),
        frame_token=frame_tokens[0],
        text_prompt=str(args.text_prompt),
    )
    object_label = (_parse_text_prompts(str(args.text_prompt)) or [str(args.text_prompt)])[0]

    temp_root = Path(tempfile.mkdtemp(prefix="edgetam_worker_", dir=str(result_json.parent)))
    video_dir = temp_root / "video_jpg"
    _prepare_jpeg_video_dir(
        case_dir=case_dir,
        camera_idx=int(args.camera_idx),
        frame_tokens=frame_tokens,
        work_dir=video_dir,
    )

    predictor = None
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            predictor, compile_metadata = _build_predictor(
                model_cfg=str(args.model_cfg),
                checkpoint=str(args.checkpoint),
                compile_mode=str(args.compile_mode),
            )

            def init_state() -> Any:
                return predictor.init_state(str(video_dir))

            def prompt_state(state: Any) -> Any:
                return predictor.add_new_mask(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=int(SAM_OBJECT_ID),
                    mask=np.asarray(init_mask, dtype=bool),
                )

            def propagate_state(state: Any, *, collect: bool) -> tuple[int, dict[str, np.ndarray]]:
                outputs: dict[str, np.ndarray] = {}
                count = 0
                for out_frame_idx, _out_obj_ids, out_mask_logits in predictor.propagate_in_video(state):
                    count += 1
                    if collect:
                        token = frame_tokens[int(out_frame_idx)]
                        outputs[str(token)] = (out_mask_logits[0, 0].detach().cpu().numpy() > 0)
                _cuda_sync()
                return count, outputs

            warmup_records: list[dict[str, Any]] = []
            warm_state, warm_init_ms = _time_ms(init_state)
            _warm_prompt, warm_prompt_ms = _time_ms(lambda: prompt_state(warm_state))
            for warmup_idx in range(max(0, int(args.warmup_runs))):
                (warm_count, _warm_outputs), warm_ms = _time_ms(lambda: propagate_state(warm_state, collect=False))
                warmup_records.append(
                    {
                        "warmup_idx": int(warmup_idx),
                        "frame_count": int(warm_count),
                        "propagate_ms": float(warm_ms),
                    }
                )
            del warm_state
            torch.cuda.empty_cache()

            timed_state, init_state_ms = _time_ms(init_state)
            prompt_response, prompt_ms = _time_ms(lambda: prompt_state(timed_state))
            (timed_frame_count, _timed_outputs), timed_propagate_ms = _time_ms(
                lambda: propagate_state(timed_state, collect=False)
            )
            del timed_state
            torch.cuda.empty_cache()

            collect_state, _collect_init_ms = _time_ms(init_state)
            _collect_prompt, _collect_prompt_ms = _time_ms(lambda: prompt_state(collect_state))
            (collect_frame_count, masks_by_frame), collect_ms = _time_ms(
                lambda: propagate_state(collect_state, collect=True)
            )
            del collect_state
            torch.cuda.empty_cache()

        if len(masks_by_frame) != len(frame_tokens):
            raise RuntimeError(f"EdgeTAM saved {len(masks_by_frame)} masks, expected {len(frame_tokens)}")
        write_summary = _write_single_object_masks(
            mask_root=output_mask_root,
            camera_idx=int(args.camera_idx),
            object_label=str(object_label),
            masks_by_frame_token=masks_by_frame,
            overwrite=bool(args.overwrite),
        )
        try:
            prompt_obj_ids = [int(item) for item in prompt_response[1]]
        except Exception:
            prompt_obj_ids = []

        result = {
            "case_key": str(args.case_key),
            "case_dir": str(case_dir),
            "text_prompt": str(args.text_prompt),
            "camera_idx": int(args.camera_idx),
            "checkpoint_key": "edgetam",
            "checkpoint_label": "EdgeTAM",
            "checkpoint_path": str(Path(args.checkpoint).resolve()),
            "config": str(args.model_cfg),
            "compile_mode": str(args.compile_mode),
            "compile_metadata": compile_metadata,
            "init_mode": "mask",
            "sam31_mask_root": str(sam31_mask_root),
            "prompt_source": "sam31_frame0_union_mask",
            "frames_requested": int(len(frame_tokens)),
            "frame_tokens": [str(item) for item in frame_tokens],
            "warmup_runs": int(args.warmup_runs),
            "warmup_init_state_ms": float(warm_init_ms),
            "warmup_prompt_ms": float(warm_prompt_ms),
            "warmup_records": warmup_records,
            "timed_frame_count": int(timed_frame_count),
            "init_state_ms": float(init_state_ms),
            "prompt_ms": float(prompt_ms),
            "timed_propagate_ms": float(timed_propagate_ms),
            "inference_ms_per_frame": float(timed_propagate_ms / max(1, timed_frame_count)),
            "fps": float(1000.0 * timed_frame_count / max(1e-9, timed_propagate_ms)),
            "timing_contract": (
                "edgetam_no_output_propagate_after_"
                f"{int(args.warmup_runs)}_warmup_propagations; "
                "model load, jpeg prep, init_state, prompt, warmup, and mask collection excluded"
            ),
            "prompt_obj_ids": prompt_obj_ids,
            "mask_output": write_summary,
            "mask_collection_frame_count": int(collect_frame_count),
            "mask_collection_ms": float(collect_ms),
            "process_guard": {
                "mode": "single_case_single_view_edgetam_worker",
                "pid": int(os.getpid()),
                "python": sys.executable,
                "cuda_device_name": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "bfloat16_autocast": True,
                "speed_pass_collects_masks": False,
                "compile_mode": str(args.compile_mode),
            },
        }
        result_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EdgeTAM video tracking masks for one aligned case/camera.")
    parser.add_argument("--case-key", required=True)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--text-prompt", required=True)
    parser.add_argument("--camera-idx", type=int, required=True)
    parser.add_argument("--output-mask-root", type=Path, required=True)
    parser.add_argument("--sam31-mask-root", type=Path)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--all-frames", action="store_true")
    parser.add_argument("--checkpoint", default="checkpoints/edgetam.pt")
    parser.add_argument("--model-cfg", default="configs/edgetam.yaml")
    parser.add_argument(
        "--compile-mode",
        choices=(EDGETAM_COMPILE_EAGER, EDGETAM_COMPILE_IMAGE_ENCODER, EDGETAM_COMPILE_NO_POS_CACHE),
        default=EDGETAM_COMPILE_EAGER,
    )
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.all_frames:
        args.frames = None
    return args


def main(argv: list[str] | None = None) -> int:
    result = run_worker(parse_args(argv))
    print(
        f"[edgetam-worker] {result['case_key']} cam{result['camera_idx']}: "
        f"{result['inference_ms_per_frame']:.2f} ms/frame {result['fps']:.2f} FPS",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
