#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
import traceback
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_ID = "yonigozlan/EdgeTAM-hf"
DEFAULT_OUTPUT_DIR = ROOT / "result/hf_edgetam_streaming_realcase"
DEFAULT_DOC_MD = ROOT / "docs/generated/hf_edgetam_streaming_realcase_benchmark.md"
DEFAULT_DOC_JSON = ROOT / "docs/generated/hf_edgetam_streaming_realcase_results.json"
DEFAULT_QUALITY_JSON = ROOT / "docs/generated/hf_edgetam_streaming_quality.json"
DEFAULT_STILL_SAM21_MASK_ROOT = (
    ROOT / "result/sam21_checkpoint_ladder_3x5_time_gifs_ffs203048_iter4_trt_level5_stable_throughput/masks"
)
DEFAULT_DYNAMICS_SAM21_MASK_ROOT = (
    ROOT
    / "result/sam21_dynamics_checkpoint_ladder_3x5_time_gifs_ffs203048_iter4_trt_level5_maskinit_stable_throughput/masks"
)
DEFAULT_PROMPT_MODES = ("point", "box", "mask")
ALL_PROMPT_MODES = ("point", "box", "mask", "previous_mask")
DEFAULT_CAMERA_IDS = (0, 1, 2)
OBJECT_ID = 1
COMPILE_MODE_NONE = "none"
COMPILE_MODE_MODEL_DEFAULT = "model-default"
COMPILE_MODE_MODEL_REDUCE_OVERHEAD = "model-reduce-overhead"
COMPILE_MODE_VISION_REDUCE_OVERHEAD = "vision-reduce-overhead"
COMPILE_MODE_COMPONENTS_REDUCE_OVERHEAD = "components-reduce-overhead"
ALL_COMPILE_MODES = (
    COMPILE_MODE_NONE,
    COMPILE_MODE_MODEL_DEFAULT,
    COMPILE_MODE_MODEL_REDUCE_OVERHEAD,
    COMPILE_MODE_VISION_REDUCE_OVERHEAD,
    COMPILE_MODE_COMPONENTS_REDUCE_OVERHEAD,
)
COMPONENT_COMPILE_TARGETS = ("vision_encoder", "memory_attention", "memory_encoder", "mask_decoder")

cv2: Any = None
np: Any = None
Image: Any = None
torch: Any = None
transformers: Any = None
EdgeTamVideoInferenceSession: Any = None
EdgeTamVideoModel: Any = None
Sam2VideoProcessor: Any = None


@dataclass(frozen=True)
class CaseSpec:
    key: str
    label: str
    case_dir: Path
    text_prompt: str
    sam21_mask_root: Path
    sam31_mask_root: Path


def _default_case_specs() -> list[CaseSpec]:
    still_round1 = ROOT / "data/still_object/ffs203048_iter4_trt_level5/both_30_still_object_round1_20260428"
    still_round2 = ROOT / "data/still_object/ffs203048_iter4_trt_level5/both_30_still_object_round2_20260428"
    still_round3 = ROOT / "data/still_object/ffs203048_iter4_trt_level5/both_30_still_object_round3_20260428"
    still_round4 = ROOT / "data/still_object/ffs203048_iter4_trt_level5/both_30_still_object_round4_20260428"
    rope_round1 = ROOT / "data/still_rope/ffs203048_iter4_trt_level5/both_30_still_rope_round1_20260428"
    rope_round2 = ROOT / "data/still_rope/ffs203048_iter4_trt_level5/both_30_still_rope_round2_20260428"
    dynamics_round1 = ROOT / "data/dynamics/ffs_dynamics_round1_20260414"
    return [
        CaseSpec(
            key="still_object_round1",
            label="Still Object R1",
            case_dir=still_round1,
            text_prompt="stuffed animal",
            sam21_mask_root=DEFAULT_STILL_SAM21_MASK_ROOT,
            sam31_mask_root=still_round1 / "sam31_masks",
        ),
        CaseSpec(
            key="still_object_round2",
            label="Still Object R2",
            case_dir=still_round2,
            text_prompt="stuffed animal",
            sam21_mask_root=DEFAULT_STILL_SAM21_MASK_ROOT,
            sam31_mask_root=still_round2 / "sam31_masks",
        ),
        CaseSpec(
            key="still_object_round3",
            label="Still Object R3",
            case_dir=still_round3,
            text_prompt="stuffed animal",
            sam21_mask_root=DEFAULT_STILL_SAM21_MASK_ROOT,
            sam31_mask_root=still_round3 / "sam31_masks",
        ),
        CaseSpec(
            key="still_object_round4",
            label="Still Object R4",
            case_dir=still_round4,
            text_prompt="stuffed animal",
            sam21_mask_root=DEFAULT_STILL_SAM21_MASK_ROOT,
            sam31_mask_root=still_round4 / "sam31_masks",
        ),
        CaseSpec(
            key="still_rope_round1",
            label="Still Rope R1",
            case_dir=rope_round1,
            text_prompt="white twisted rope on the blue box, white thick twisted rope on top of the blue box",
            sam21_mask_root=DEFAULT_STILL_SAM21_MASK_ROOT,
            sam31_mask_root=rope_round1 / "sam31_masks",
        ),
        CaseSpec(
            key="still_rope_round2",
            label="Still Rope R2",
            case_dir=rope_round2,
            text_prompt="white twisted rope lying on the wooden table",
            sam21_mask_root=DEFAULT_STILL_SAM21_MASK_ROOT,
            sam31_mask_root=rope_round2 / "sam31_masks",
        ),
        CaseSpec(
            key="ffs_dynamics_round1",
            label="FFS Dynamics R1",
            case_dir=dynamics_round1,
            text_prompt="sloth",
            sam21_mask_root=DEFAULT_DYNAMICS_SAM21_MASK_ROOT,
            sam31_mask_root=dynamics_round1 / "sam31_masks",
        ),
    ]


def _load_runtime_dependencies() -> None:
    global EdgeTamVideoInferenceSession
    global EdgeTamVideoModel
    global Image
    global Sam2VideoProcessor
    global cv2
    global np
    global torch
    global transformers

    import cv2 as runtime_cv2
    import numpy as runtime_np
    from PIL import Image as runtime_image
    import torch as runtime_torch
    import transformers as runtime_transformers
    from transformers import (
        EdgeTamVideoInferenceSession as runtime_edge_session,
        EdgeTamVideoModel as runtime_model,
        Sam2VideoProcessor as runtime_processor,
    )

    cv2 = runtime_cv2
    np = runtime_np
    Image = runtime_image
    torch = runtime_torch
    transformers = runtime_transformers
    EdgeTamVideoInferenceSession = runtime_edge_session
    EdgeTamVideoModel = runtime_model
    Sam2VideoProcessor = runtime_processor


def _dtype_from_name(name: str) -> Any:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[str(name)]


def _sync_if_needed(device: str) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def _time_ms(device: str, fn: Any) -> tuple[Any, float]:
    _sync_if_needed(device)
    start = time.perf_counter()
    value = fn()
    _sync_if_needed(device)
    return value, float((time.perf_counter() - start) * 1000.0)


def _comma_list(value: str | None) -> list[str] | None:
    if value is None or not str(value).strip():
        return None
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _parse_ints(value: str | None, *, default: Sequence[int]) -> tuple[int, ...]:
    items = _comma_list(value)
    if items is None:
        return tuple(int(item) for item in default)
    return tuple(int(item) for item in items)


def _parse_modes(value: str | None) -> tuple[str, ...]:
    items = _comma_list(value)
    if items is None:
        return DEFAULT_PROMPT_MODES
    modes = tuple(str(item) for item in items)
    unsupported = [item for item in modes if item not in ALL_PROMPT_MODES]
    if unsupported:
        raise ValueError(f"Unsupported prompt modes: {unsupported}")
    return modes


def _resolve_cli_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _custom_case_from_args(args: argparse.Namespace) -> CaseSpec | None:
    if args.case_dir is None:
        return None
    if args.case_keys:
        raise ValueError("--case-keys cannot be combined with --case-dir")
    missing = [
        name
        for name, value in (
            ("--text-prompt", args.text_prompt),
            ("--sam31-mask-root", args.sam31_mask_root),
        )
        if value is None or not str(value).strip()
    ]
    if missing:
        raise ValueError(f"Custom case requires {', '.join(missing)}")
    case_dir = _resolve_cli_path(args.case_dir)
    case_key = str(args.case_key or case_dir.name)
    return CaseSpec(
        key=case_key,
        label=str(args.case_label or case_key),
        case_dir=case_dir,
        text_prompt=str(args.text_prompt),
        sam21_mask_root=_resolve_cli_path(args.sam31_mask_root),
        sam31_mask_root=_resolve_cli_path(args.sam31_mask_root),
    )


def _select_cases(args: argparse.Namespace) -> list[CaseSpec]:
    custom_case = _custom_case_from_args(args)
    if custom_case is not None:
        return [custom_case]
    cases = _default_case_specs()
    selected = _comma_list(args.case_keys)
    if selected is None:
        return cases
    by_key = {case.key: case for case in cases}
    missing = [key for key in selected if key not in by_key]
    if missing:
        raise ValueError(f"Unknown case keys: {missing}")
    return [by_key[key] for key in selected]


def _sorted_frame_tokens(case_dir: Path, *, camera_idx: int, frames: int | None) -> list[str]:
    color_dir = Path(case_dir) / "color" / str(int(camera_idx))
    paths = sorted(color_dir.glob("*.png"), key=lambda path: int(path.stem))
    if frames is not None:
        paths = paths[: int(frames)]
    if not paths:
        raise FileNotFoundError(f"No color PNG frames found: {color_dir}")
    return [path.stem for path in paths]


def _normalize_label(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def _text_prompt_labels(text_prompt: str) -> set[str]:
    return {_normalize_label(item) for item in str(text_prompt).split(",") if item.strip()}


def _primary_text_prompt_label(text_prompt: str) -> str:
    for item in str(text_prompt).split(","):
        label = " ".join(item.strip().split())
        if label:
            return label
    return str(text_prompt)


def _read_mask_info(mask_root: Path, *, camera_idx: int) -> dict[int, str]:
    root = Path(mask_root)
    if root.name == "mask":
        root = root.parent
    info_path = root / "mask" / f"mask_info_{int(camera_idx)}.json"
    if not info_path.is_file():
        return {}
    payload = json.loads(info_path.read_text(encoding="utf-8"))
    return {int(key): str(value) for key, value in dict(payload).items()}


def _image_shape(case_dir: Path, *, camera_idx: int, frame_token: str) -> tuple[int, int]:
    path = Path(case_dir) / "color" / str(int(camera_idx)) / f"{frame_token}.png"
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Missing color frame: {path}")
    return int(image.shape[0]), int(image.shape[1])


def _load_union_mask(
    *,
    mask_root: Path,
    case_dir: Path,
    camera_idx: int,
    frame_token: str,
    text_prompt: str,
    fallback_all_objects: bool = False,
) -> Any:
    root = Path(mask_root)
    if root.name == "mask":
        root = root.parent
    info = _read_mask_info(root, camera_idx=int(camera_idx))
    prompts = _text_prompt_labels(text_prompt)
    matched_ids = [
        int(obj_id)
        for obj_id, label in sorted(info.items())
        if _normalize_label(label) in prompts
    ]
    if not matched_ids and fallback_all_objects:
        matched_ids = [int(obj_id) for obj_id in sorted(info)]
    if not matched_ids:
        raise RuntimeError(f"No masks match {text_prompt!r} in {root} cam{camera_idx}")

    height, width = _image_shape(case_dir, camera_idx=int(camera_idx), frame_token=str(frame_token))
    union = np.zeros((height, width), dtype=bool)
    for obj_id in matched_ids:
        path = root / "mask" / str(int(camera_idx)) / str(int(obj_id)) / f"{frame_token}.png"
        if not path.is_file():
            continue
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"Failed to read mask: {path}")
        if image.shape[:2] != union.shape:
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)
        union |= image > 0
    return union


def _mask_centroid(mask: Any) -> tuple[float | None, float | None]:
    ys, xs = np.where(np.asarray(mask, dtype=bool))
    if len(xs) == 0:
        return None, None
    return float(np.mean(xs)), float(np.mean(ys))


def _mask_bbox(mask: Any) -> list[float | None]:
    ys, xs = np.where(np.asarray(mask, dtype=bool))
    if len(xs) == 0:
        return [None, None, None, None]
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def _bbox_xyxy_from_mask(mask: Any, *, padding_px: int) -> list[float]:
    bbox = _mask_bbox(mask)
    if any(item is None for item in bbox):
        raise ValueError("Cannot create bbox from empty mask")
    height, width = np.asarray(mask).shape[:2]
    x0, y0, x1, y1 = [float(item) for item in bbox]
    return [
        float(max(0, x0 - int(padding_px))),
        float(max(0, y0 - int(padding_px))),
        float(min(width - 1, x1 + int(padding_px))),
        float(min(height - 1, y1 + int(padding_px))),
    ]


def _mask_iou(a: Any, b: Any) -> float:
    ma = np.asarray(a, dtype=bool)
    mb = np.asarray(b, dtype=bool)
    union = int(np.count_nonzero(ma | mb))
    if union == 0:
        return 1.0
    return float(np.count_nonzero(ma & mb) / union)


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), float(pct)))


def _latency_summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "p90": _percentile(values, 90),
        "p95": _percentile(values, 95),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _compile_mode_to_torch_mode(compile_mode: str) -> str | None:
    if compile_mode in {COMPILE_MODE_MODEL_DEFAULT}:
        return "default"
    if compile_mode in {
        COMPILE_MODE_MODEL_REDUCE_OVERHEAD,
        COMPILE_MODE_VISION_REDUCE_OVERHEAD,
        COMPILE_MODE_COMPONENTS_REDUCE_OVERHEAD,
    }:
        return "reduce-overhead"
    if compile_mode == COMPILE_MODE_NONE:
        return None
    raise ValueError(f"Unsupported compile mode: {compile_mode}")


def _compile_targets_for_mode(model: Any, compile_mode: str) -> tuple[str, ...]:
    if compile_mode == COMPILE_MODE_NONE:
        return ()
    if compile_mode in {COMPILE_MODE_MODEL_DEFAULT, COMPILE_MODE_MODEL_REDUCE_OVERHEAD}:
        return ("<model>",)
    if compile_mode == COMPILE_MODE_VISION_REDUCE_OVERHEAD:
        return tuple(name for name in ("vision_encoder",) if hasattr(model, name))
    if compile_mode == COMPILE_MODE_COMPONENTS_REDUCE_OVERHEAD:
        return tuple(name for name in COMPONENT_COMPILE_TARGETS if hasattr(model, name))
    raise ValueError(f"Unsupported compile mode: {compile_mode}")


def _apply_compile_mode(model: Any, compile_mode: str) -> tuple[Any, dict[str, Any]]:
    compile_mode = str(compile_mode)
    requested_targets = (
        ()
        if compile_mode == COMPILE_MODE_NONE
        else ("<model>",)
        if compile_mode in {COMPILE_MODE_MODEL_DEFAULT, COMPILE_MODE_MODEL_REDUCE_OVERHEAD}
        else ("vision_encoder",)
        if compile_mode == COMPILE_MODE_VISION_REDUCE_OVERHEAD
        else COMPONENT_COMPILE_TARGETS
    )
    targets = _compile_targets_for_mode(model, compile_mode)
    missing_targets = [name for name in requested_targets if name not in targets]
    torch_mode = _compile_mode_to_torch_mode(compile_mode)
    metadata: dict[str, Any] = {
        "compile_mode": compile_mode,
        "enabled": compile_mode != COMPILE_MODE_NONE,
        "torch_compile_available": bool(hasattr(torch, "compile")) if torch is not None else False,
        "torch_compile_mode": torch_mode,
        "fullgraph": False,
        "dynamic": False,
        "requested_targets": list(requested_targets),
        "applied_targets": list(targets),
        "missing_targets": missing_targets,
        "whole_model_compiled": False,
        "wrap_ms": 0.0,
    }
    if compile_mode == COMPILE_MODE_NONE:
        return model, metadata
    if torch is None or not hasattr(torch, "compile"):
        raise RuntimeError("Requested --compile-mode but torch.compile is not available.")
    if not targets:
        raise RuntimeError(f"Requested --compile-mode {compile_mode!r}, but no compile targets were found.")

    started = time.perf_counter()
    if targets == ("<model>",):
        model = torch.compile(
            model,
            mode=torch_mode,
            fullgraph=False,
            dynamic=False,
        )
        metadata["whole_model_compiled"] = True
        metadata["wrap_ms"] = float((time.perf_counter() - started) * 1000.0)
        return model, metadata

    for target in targets:
        module = getattr(model, target)
        setattr(
            model,
            target,
            torch.compile(
                module,
                mode=torch_mode,
                fullgraph=False,
                dynamic=False,
            ),
        )
    metadata["wrap_ms"] = float((time.perf_counter() - started) * 1000.0)
    return model, metadata


def _depth_path(case_dir: Path, *, camera_idx: int, frame_token: str) -> Path | None:
    candidates = [
        case_dir / "depth_ffs_float_m" / str(int(camera_idx)) / f"{frame_token}.npy",
        case_dir / "depth_ffs_native_like_postprocess_float_m" / str(int(camera_idx)) / f"{frame_token}.npy",
        case_dir / "depth_ffs" / str(int(camera_idx)) / f"{frame_token}.npy",
        case_dir / "depth" / str(int(camera_idx)) / f"{frame_token}.npy",
        case_dir / "depth" / str(int(camera_idx)) / f"{frame_token}.png",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def _load_depth_valid_mask(case_dir: Path, *, camera_idx: int, frame_token: str) -> Any | None:
    path = _depth_path(case_dir, camera_idx=int(camera_idx), frame_token=str(frame_token))
    if path is None:
        return None
    if path.suffix == ".npy":
        depth = np.load(str(path))
    else:
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        return None
    return np.isfinite(depth) & (depth > 0)


def _extract_mask(mask_tensor: Any) -> Any:
    value = mask_tensor
    while hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
        break
    array = np.asarray(value)
    array = np.squeeze(array)
    if array.ndim != 2:
        raise RuntimeError(f"Expected 2-D mask after squeeze, got {array.shape}")
    return array > 0


def _write_single_object_masks(
    *,
    mask_root: Path,
    case_key: str,
    prompt_mode: str,
    object_label: str,
    camera_idx: int,
    masks_by_frame: dict[str, Any],
    overwrite: bool,
) -> dict[str, Any]:
    root = Path(mask_root) / str(case_key) / str(prompt_mode)
    mask_dir = root / "mask"
    camera_dir = mask_dir / str(int(camera_idx))
    info_path = mask_dir / f"mask_info_{int(camera_idx)}.json"
    if overwrite:
        if camera_dir.exists():
            import shutil

            shutil.rmtree(camera_dir)
        if info_path.exists():
            info_path.unlink()
    if camera_dir.exists() or info_path.exists():
        raise FileExistsError(f"Mask output already exists: {camera_dir}")
    object_dir = camera_dir / str(OBJECT_ID)
    object_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    info_path.write_text(json.dumps({str(OBJECT_ID): str(object_label)}, indent=2), encoding="utf-8")
    for frame_token, mask in sorted(masks_by_frame.items(), key=lambda item: int(item[0])):
        cv2.imwrite(str(object_dir / f"{frame_token}.png"), np.asarray(mask, dtype=np.uint8) * 255)
    return {
        "mask_root": str(root),
        "camera_idx": int(camera_idx),
        "object_id": int(OBJECT_ID),
        "object_label": str(object_label),
        "saved_frame_count": int(len(masks_by_frame)),
    }


def _init_session(*, device: str, dtype: Any, height: int, width: int) -> Any:
    return EdgeTamVideoInferenceSession(
        video=None,
        video_height=int(height),
        video_width=int(width),
        inference_device=device,
        inference_state_device=device,
        video_storage_device=device,
        dtype=dtype,
    )


def _add_prompt(
    *,
    processor: Any,
    session: Any,
    prompt_mode: str,
    frame_idx: int,
    init_mask: Any,
    original_size: Any,
    bbox_padding_px: int,
) -> None:
    if prompt_mode == "point":
        cx, cy = _mask_centroid(init_mask)
        if cx is None or cy is None:
            raise ValueError("Cannot create point prompt from empty mask")
        processor.add_inputs_to_inference_session(
            inference_session=session,
            frame_idx=int(frame_idx),
            obj_ids=OBJECT_ID,
            input_points=[[[[float(cx), float(cy)]]]],
            input_labels=[[[1]]],
            original_size=original_size,
        )
        return
    if prompt_mode == "box":
        box = _bbox_xyxy_from_mask(init_mask, padding_px=int(bbox_padding_px))
        processor.add_inputs_to_inference_session(
            inference_session=session,
            frame_idx=int(frame_idx),
            obj_ids=OBJECT_ID,
            input_boxes=[[[float(item) for item in box]]],
            original_size=original_size,
        )
        return
    if prompt_mode in {"mask", "previous_mask"}:
        processor.add_inputs_to_inference_session(
            inference_session=session,
            frame_idx=int(frame_idx),
            obj_ids=OBJECT_ID,
            input_masks=np.asarray(init_mask, dtype=bool),
        )
        return
    raise ValueError(f"Unsupported prompt mode: {prompt_mode}")


def _compare_against_variant(
    *,
    masks_by_frame: dict[str, Any],
    case: CaseSpec,
    camera_idx: int,
    variant_key: str,
) -> dict[str, Any] | None:
    variant_root = Path(case.sam21_mask_root) / case.key / str(variant_key)
    if not (variant_root / "mask").is_dir():
        return None
    ious: list[float] = []
    missing: list[str] = []
    for frame_token, mask in masks_by_frame.items():
        try:
            ref = _load_union_mask(
                mask_root=variant_root,
                case_dir=case.case_dir,
                camera_idx=int(camera_idx),
                frame_token=str(frame_token),
                text_prompt=case.text_prompt,
                fallback_all_objects=True,
            )
            ious.append(_mask_iou(mask, ref))
        except Exception:
            missing.append(str(frame_token))
    return {
        "variant": str(variant_key),
        "available": bool(ious),
        "frame_count": int(len(ious)),
        "missing_frames": missing,
        "iou_mean": float(statistics.fmean(ious)) if ious else None,
        "iou_min": float(min(ious)) if ious else None,
        "iou_median": float(statistics.median(ious)) if ious else None,
    }


def _quality_summary(
    *,
    masks_by_frame: dict[str, Any],
    case: CaseSpec,
    camera_idx: int,
) -> dict[str, Any]:
    frame_tokens = list(masks_by_frame)
    masks = [masks_by_frame[token] for token in frame_tokens]
    areas = [int(np.count_nonzero(mask)) for mask in masks]
    centroids = [_mask_centroid(mask) for mask in masks]
    bboxes = [_mask_bbox(mask) for mask in masks]
    frame0 = masks[0]
    iou_to_frame0 = [_mask_iou(mask, frame0) for mask in masks]

    c0 = centroids[0]
    b0 = bboxes[0]
    per_frame: list[dict[str, Any]] = []
    ffs_output_counts: list[int] = []
    ffs_ref_counts: list[int] = []
    ffs_delta_counts: list[int] = []
    failure_frames: list[str] = []

    for token, mask, area, centroid, bbox, iou0 in zip(frame_tokens, masks, areas, centroids, bboxes, iou_to_frame0, strict=True):
        if area <= 0:
            failure_frames.append(str(token))
        centroid_drift = None
        if None not in centroid and None not in c0:
            centroid_drift = float(math.hypot(float(centroid[0]) - float(c0[0]), float(centroid[1]) - float(c0[1])))
        bbox_drift = None
        if all(item is not None for item in bbox) and all(item is not None for item in b0):
            bbox_drift = float(max(abs(float(a) - float(b)) for a, b in zip(bbox, b0, strict=True)))

        depth_valid = _load_depth_valid_mask(case.case_dir, camera_idx=int(camera_idx), frame_token=str(token))
        output_count = None
        ref_count = None
        delta_count = None
        if depth_valid is not None:
            output_count = int(np.count_nonzero(np.asarray(mask, dtype=bool) & depth_valid))
            try:
                ref_mask = _load_union_mask(
                    mask_root=case.sam31_mask_root,
                    case_dir=case.case_dir,
                    camera_idx=int(camera_idx),
                    frame_token=str(token),
                    text_prompt=case.text_prompt,
                    fallback_all_objects=False,
                )
                ref_count = int(np.count_nonzero(np.asarray(ref_mask, dtype=bool) & depth_valid))
                delta_count = int(output_count - ref_count)
                ffs_ref_counts.append(ref_count)
                ffs_delta_counts.append(delta_count)
            except Exception:
                pass
            ffs_output_counts.append(output_count)

        per_frame.append(
            {
                "frame_token": str(token),
                "area": int(area),
                "centroid": [centroid[0], centroid[1]],
                "bbox_xyxy": bbox,
                "centroid_drift_px": centroid_drift,
                "bbox_linf_drift_px": bbox_drift,
                "iou_to_frame0": float(iou0),
                "ffs_output_point_count": output_count,
                "ffs_sam31_point_count": ref_count,
                "ffs_point_count_delta": delta_count,
            }
        )

    areas_np = np.asarray(areas, dtype=np.float64)
    centroid_drifts = [float(item["centroid_drift_px"]) for item in per_frame if item["centroid_drift_px"] is not None]
    bbox_drifts = [float(item["bbox_linf_drift_px"]) for item in per_frame if item["bbox_linf_drift_px"] is not None]
    comparisons = {
        key: value
        for key, value in (
            ("small", _compare_against_variant(masks_by_frame=masks_by_frame, case=case, camera_idx=camera_idx, variant_key="small")),
            ("tiny", _compare_against_variant(masks_by_frame=masks_by_frame, case=case, camera_idx=camera_idx, variant_key="tiny")),
        )
        if value is not None
    }
    return {
        "frame_count": int(len(frame_tokens)),
        "area_mean": float(np.mean(areas_np)) if len(areas_np) else 0.0,
        "area_std": float(np.std(areas_np)) if len(areas_np) else 0.0,
        "area_std_over_mean": float(np.std(areas_np) / max(1.0, float(np.mean(areas_np)))) if len(areas_np) else 0.0,
        "area_min": int(np.min(areas_np)) if len(areas_np) else 0,
        "area_max": int(np.max(areas_np)) if len(areas_np) else 0,
        "centroid_drift_px_max": float(max(centroid_drifts)) if centroid_drifts else None,
        "centroid_drift_px_mean": float(statistics.fmean(centroid_drifts)) if centroid_drifts else None,
        "bbox_linf_drift_px_max": float(max(bbox_drifts)) if bbox_drifts else None,
        "bbox_linf_drift_px_mean": float(statistics.fmean(bbox_drifts)) if bbox_drifts else None,
        "iou_to_frame0_mean": float(statistics.fmean(iou_to_frame0)) if iou_to_frame0 else 1.0,
        "iou_to_frame0_min": float(min(iou_to_frame0)) if iou_to_frame0 else 1.0,
        "sam21_iou": comparisons,
        "ffs_point_counts": {
            "output_mean": float(statistics.fmean(ffs_output_counts)) if ffs_output_counts else None,
            "sam31_mean": float(statistics.fmean(ffs_ref_counts)) if ffs_ref_counts else None,
            "delta_mean": float(statistics.fmean(ffs_delta_counts)) if ffs_delta_counts else None,
            "delta_min": int(min(ffs_delta_counts)) if ffs_delta_counts else None,
            "delta_max": int(max(ffs_delta_counts)) if ffs_delta_counts else None,
        },
        "failure_frames": failure_frames,
        "per_frame": per_frame,
    }


def _run_one_stream(
    *,
    model: Any,
    processor: Any,
    case: CaseSpec,
    camera_idx: int,
    prompt_mode: str,
    frame_tokens: Sequence[str],
    device: str,
    dtype: Any,
    bbox_padding_px: int,
    output_mask_root: Path,
    write_masks: bool,
    overwrite: bool,
) -> dict[str, Any]:
    height, width = _image_shape(case.case_dir, camera_idx=int(camera_idx), frame_token=str(frame_tokens[0]))
    session = _init_session(device=device, dtype=dtype, height=height, width=width)
    init_mask = _load_union_mask(
        mask_root=case.sam31_mask_root,
        case_dir=case.case_dir,
        camera_idx=int(camera_idx),
        frame_token=str(frame_tokens[0]),
        text_prompt=case.text_prompt,
    )

    frame_records: list[dict[str, Any]] = []
    masks_by_frame: dict[str, Any] = {}
    previous_mask = None
    started = time.perf_counter()

    for frame_idx, frame_token in enumerate(frame_tokens):
        color_path = case.case_dir / "color" / str(int(camera_idx)) / f"{frame_token}.png"
        image = Image.open(color_path).convert("RGB")

        inputs, preprocess_ms = _time_ms(
            device,
            lambda image=image: processor(images=image, device=device, return_tensors="pt"),
        )
        pixel_values = inputs.pixel_values[0].to(device=device, dtype=dtype)

        prompt_ms = 0.0
        if frame_idx == 0:
            _unused, prompt_ms = _time_ms(
                device,
                lambda: _add_prompt(
                    processor=processor,
                    session=session,
                    prompt_mode=prompt_mode,
                    frame_idx=0,
                    init_mask=init_mask,
                    original_size=inputs.original_sizes[0],
                    bbox_padding_px=int(bbox_padding_px),
                ),
            )
        elif prompt_mode == "previous_mask" and previous_mask is not None:
            _unused, prompt_ms = _time_ms(
                device,
                lambda previous_mask=previous_mask: processor.add_inputs_to_inference_session(
                    inference_session=session,
                    frame_idx=int(frame_idx),
                    obj_ids=OBJECT_ID,
                    input_masks=np.asarray(previous_mask, dtype=bool),
                ),
            )

        output, model_ms = _time_ms(
            device,
            lambda pixel_values=pixel_values: model(
                inference_session=session,
                frame=pixel_values,
            ),
        )
        post_masks, postprocess_ms = _time_ms(
            device,
            lambda output=output, original_sizes=inputs.original_sizes: processor.post_process_masks(
                [output.pred_masks],
                original_sizes=original_sizes,
                binarize=False,
            )[0],
        )
        mask = _extract_mask(post_masks)
        previous_mask = mask
        masks_by_frame[str(frame_token)] = mask
        frame_total_ms = float(preprocess_ms + prompt_ms + model_ms + postprocess_ms)
        frame_records.append(
            {
                "frame_idx": int(frame_idx),
                "frame_token": str(frame_token),
                "preprocess_ms": float(preprocess_ms),
                "prompt_ms": float(prompt_ms),
                "model_ms": float(model_ms),
                "postprocess_ms": float(postprocess_ms),
                "frame_total_ms": frame_total_ms,
                "mask_area": int(np.count_nonzero(mask)),
            }
        )

    wall_ms = float((time.perf_counter() - started) * 1000.0)
    quality = _quality_summary(masks_by_frame=masks_by_frame, case=case, camera_idx=int(camera_idx))
    mask_output = None
    if write_masks:
        mask_output = _write_single_object_masks(
            mask_root=output_mask_root,
            case_key=case.key,
            prompt_mode=prompt_mode,
            object_label=_primary_text_prompt_label(case.text_prompt),
            camera_idx=int(camera_idx),
            masks_by_frame=masks_by_frame,
            overwrite=bool(overwrite),
        )

    frame_total = [float(item["frame_total_ms"]) for item in frame_records]
    model_values = [float(item["model_ms"]) for item in frame_records]
    subsequent_total = frame_total[1:]
    subsequent_model = model_values[1:]
    return {
        "status": "pass",
        "case_key": case.key,
        "case_label": case.label,
        "case_dir": str(case.case_dir),
        "camera_idx": int(camera_idx),
        "prompt_mode": str(prompt_mode),
        "text_prompt": case.text_prompt,
        "frame_count": int(len(frame_records)),
        "frame_tokens": [str(item) for item in frame_tokens],
        "first_frame_latency_ms": float(frame_total[0]) if frame_total else 0.0,
        "first_frame_model_ms": float(model_values[0]) if model_values else 0.0,
        "subsequent_frame_latency_ms": _latency_summary(subsequent_total),
        "subsequent_model_ms": _latency_summary(subsequent_model),
        "preprocess_ms": _latency_summary([float(item["preprocess_ms"]) for item in frame_records]),
        "model_ms": _latency_summary(model_values),
        "postprocess_ms": _latency_summary([float(item["postprocess_ms"]) for item in frame_records]),
        "prompt_ms": _latency_summary([float(item["prompt_ms"]) for item in frame_records if float(item["prompt_ms"]) > 0]),
        "streaming_total_ms": float(sum(frame_total)),
        "wall_ms": wall_ms,
        "end_to_end_streaming_fps": float(1000.0 * len(frame_records) / max(1e-9, sum(frame_total))),
        "model_only_streaming_fps": float(1000.0 * len(frame_records) / max(1e-9, sum(model_values))),
        "mask_output": mask_output,
        "quality": quality,
        "frames": frame_records,
    }


def _aggregate_jobs(jobs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    passed = [job for job in jobs if job.get("status") == "pass"]
    failed = [job for job in jobs if job.get("status") != "pass"]
    by_mode: dict[str, list[dict[str, Any]]] = {}
    for job in passed:
        by_mode.setdefault(str(job["prompt_mode"]), []).append(job)
    mode_summary = {}
    for mode, items in sorted(by_mode.items()):
        mode_summary[mode] = {
            "job_count": int(len(items)),
            "first_frame_latency_ms": _latency_summary([float(item["first_frame_latency_ms"]) for item in items]),
            "subsequent_frame_latency_median_ms": _latency_summary(
                [float(item["subsequent_frame_latency_ms"]["median"]) for item in items]
            ),
            "end_to_end_streaming_fps": _latency_summary([float(item["end_to_end_streaming_fps"]) for item in items]),
            "model_only_streaming_fps": _latency_summary([float(item["model_only_streaming_fps"]) for item in items]),
            "area_std_over_mean": _latency_summary(
                [float(item["quality"]["area_std_over_mean"]) for item in items]
            ),
            "iou_to_frame0_min": _latency_summary(
                [float(item["quality"]["iou_to_frame0_min"]) for item in items]
            ),
        }
    return {
        "job_count": int(len(jobs)),
        "passed": int(len(passed)),
        "failed": int(len(failed)),
        "mode_summary": mode_summary,
    }


def _env_report(device: str) -> dict[str, Any]:
    report = {
        "python": sys.executable,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": str(device),
    }
    if torch.cuda.is_available():
        report["gpu"] = torch.cuda.get_device_name(0)
    return report


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    aggregate = payload["aggregate"]
    lines = [
        "# HF EdgeTAM Streaming Realcase Benchmark",
        "",
        f"- Timestamp UTC: `{payload['timestamp_utc']}`",
        f"- Status: `{'pass' if aggregate['failed'] == 0 else 'partial'}`",
        f"- Model: `{payload['model_id']}`",
        f"- Environment: `edgetam-hf-stream`",
        f"- Compile mode: `{payload.get('compile_mode', COMPILE_MODE_NONE)}`",
        f"- Device: `{payload['env'].get('device')}`",
        f"- GPU: `{payload['env'].get('gpu', 'n/a')}`",
        f"- Torch: `{payload['env'].get('torch')}`",
        f"- Transformers: `{payload['env'].get('transformers')}`",
        f"- Jobs: `{aggregate['passed']}/{aggregate['job_count']}` passed",
        "",
        "## Contract",
        "",
        "- Uses real aligned QQTT color frames, not synthetic frames.",
        "- Reads one PNG frame at a time and calls the HF streaming model with a persistent inference session.",
        "- Does not pass a full video path, MP4, or offline video-folder input to EdgeTAM.",
        "- Does not modify `edgetam-max`; intended environment is `edgetam-hf-stream`.",
        "- Measures frame-by-frame streaming with persistent inference sessions.",
        "- This remains an experimental benchmark, not a production backend.",
        "",
        "## Compile",
        "",
        f"- Mode: `{payload.get('compile_mode', COMPILE_MODE_NONE)}`",
        f"- Enabled: `{payload.get('compile_metadata', {}).get('enabled', False)}`",
        f"- Applied targets: `{', '.join(payload.get('compile_metadata', {}).get('applied_targets', [])) or 'none'}`",
        f"- `torch.compile` mode: `{payload.get('compile_metadata', {}).get('torch_compile_mode')}`",
        f"- `fullgraph`: `{payload.get('compile_metadata', {}).get('fullgraph')}`",
        f"- `dynamic`: `{payload.get('compile_metadata', {}).get('dynamic')}`",
        "",
        "## Mode Summary",
        "",
        "| mode | jobs | first median ms | subsequent median ms | e2e FPS median | model FPS median | IoU-to-frame0 min median |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode, item in sorted(aggregate["mode_summary"].items()):
        lines.append(
            f"| {mode} | {item['job_count']} | "
            f"{item['first_frame_latency_ms']['median']:.2f} | "
            f"{item['subsequent_frame_latency_median_ms']['median']:.2f} | "
            f"{item['end_to_end_streaming_fps']['median']:.2f} | "
            f"{item['model_only_streaming_fps']['median']:.2f} | "
            f"{item['iou_to_frame0_min']['median']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Jobs",
            "",
            "| case | cam | mode | frames | first ms | subsequent median ms | p95 ms | e2e FPS | small IoU mean | tiny IoU mean | failures |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for job in payload["jobs"]:
        if job.get("status") != "pass":
            lines.append(
                f"| {job.get('case_key')} | {job.get('camera_idx')} | {job.get('prompt_mode')} | 0 | "
                f"n/a | n/a | n/a | n/a | n/a | n/a | 1 |"
            )
            continue
        small = job["quality"].get("sam21_iou", {}).get("small", {})
        tiny = job["quality"].get("sam21_iou", {}).get("tiny", {})
        lines.append(
            f"| {job['case_key']} | {job['camera_idx']} | {job['prompt_mode']} | {job['frame_count']} | "
            f"{job['first_frame_latency_ms']:.2f} | "
            f"{job['subsequent_frame_latency_ms']['median']:.2f} | "
            f"{job['subsequent_frame_latency_ms']['p95']:.2f} | "
            f"{job['end_to_end_streaming_fps']:.2f} | "
            f"{(small.get('iou_mean') if small.get('iou_mean') is not None else float('nan')):.4f} | "
            f"{(tiny.get('iou_mean') if tiny.get('iou_mean') is not None else float('nan')):.4f} | "
            f"{len(job['quality'].get('failure_frames', []))} |"
        )
    if aggregate["failed"]:
        lines.extend(["", "## Failures", ""])
        for job in payload["jobs"]:
            if job.get("status") != "pass":
                lines.append(f"- `{job.get('case_key')}` cam`{job.get('camera_idx')}` `{job.get('prompt_mode')}`: `{job.get('error')}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    dtype = _dtype_from_name(args.dtype)
    cases = _select_cases(args)
    camera_ids = _parse_ints(args.camera_ids, default=DEFAULT_CAMERA_IDS)
    prompt_modes = _parse_modes(args.prompt_modes)
    if bool(args.include_previous_mask) and "previous_mask" not in prompt_modes:
        prompt_modes = (*prompt_modes, "previous_mask")

    print(f"[hf-edgetam-realcase] loading model: {args.model_id}", flush=True)
    model = EdgeTamVideoModel.from_pretrained(str(args.model_id)).to(str(args.device), dtype=dtype).eval()
    model, compile_metadata = _apply_compile_mode(model, str(args.compile_mode))
    if compile_metadata["enabled"]:
        print(
            "[hf-edgetam-realcase] compile mode="
            f"{compile_metadata['compile_mode']} targets={compile_metadata['applied_targets']}",
            flush=True,
        )
    processor = Sam2VideoProcessor.from_pretrained(str(args.model_id))
    autocast_ctx = torch.autocast("cuda", dtype=dtype) if str(args.device).startswith("cuda") else nullcontext()

    jobs: list[dict[str, Any]] = []
    warmup_record: dict[str, Any] | None = None
    warmup_failed = False
    with torch.inference_mode(), autocast_ctx:
        if not bool(args.no_warmup):
            warm_case = cases[0]
            warm_camera = int(camera_ids[0])
            warm_frames = _sorted_frame_tokens(
                warm_case.case_dir,
                camera_idx=warm_camera,
                frames=max(1, int(args.warmup_frames)),
            )
            print(
                f"[hf-edgetam-realcase] warmup case={warm_case.key} cam={warm_camera} "
                f"mode={prompt_modes[0]} frames={len(warm_frames)}",
                flush=True,
            )
            try:
                warmup_record = _run_one_stream(
                    model=model,
                    processor=processor,
                    case=warm_case,
                    camera_idx=warm_camera,
                    prompt_mode=str(prompt_modes[0]),
                    frame_tokens=warm_frames,
                    device=str(args.device),
                    dtype=dtype,
                    bbox_padding_px=int(args.bbox_padding_px),
                    output_mask_root=Path(args.output_dir) / "_warmup_masks",
                    write_masks=False,
                    overwrite=True,
                )
                warmup_record["compile_mode"] = str(args.compile_mode)
            except Exception as exc:
                warmup_failed = True
                warmup_record = {
                    "status": "failed",
                    "phase": "warmup",
                    "case_key": warm_case.key,
                    "case_label": warm_case.label,
                    "case_dir": str(warm_case.case_dir),
                    "camera_idx": int(warm_camera),
                    "prompt_mode": str(prompt_modes[0]),
                    "compile_mode": str(args.compile_mode),
                    "text_prompt": warm_case.text_prompt,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                jobs.append(warmup_record)
                print(
                    f"[hf-edgetam-realcase] WARMUP FAILED case={warm_case.key} cam={warm_camera} "
                    f"mode={prompt_modes[0]} compile={args.compile_mode}: {type(exc).__name__}: {exc}",
                    flush=True,
                )
        if not warmup_failed:
            for case in cases:
                for camera_idx in camera_ids:
                    frame_tokens = _sorted_frame_tokens(case.case_dir, camera_idx=int(camera_idx), frames=args.frames)
                    for prompt_mode in prompt_modes:
                        print(
                            f"[hf-edgetam-realcase] case={case.key} cam={camera_idx} "
                            f"mode={prompt_mode} frames={len(frame_tokens)}",
                            flush=True,
                        )
                        try:
                            job = _run_one_stream(
                                model=model,
                                processor=processor,
                                case=case,
                                camera_idx=int(camera_idx),
                                prompt_mode=str(prompt_mode),
                                frame_tokens=frame_tokens,
                                device=str(args.device),
                                dtype=dtype,
                                bbox_padding_px=int(args.bbox_padding_px),
                                output_mask_root=Path(args.output_dir) / "masks",
                                write_masks=not bool(args.no_write_masks),
                                overwrite=bool(args.overwrite),
                            )
                            job["compile_mode"] = str(args.compile_mode)
                        except Exception as exc:
                            job = {
                                "status": "failed",
                                "case_key": case.key,
                                "case_label": case.label,
                                "case_dir": str(case.case_dir),
                                "camera_idx": int(camera_idx),
                                "prompt_mode": str(prompt_mode),
                                "compile_mode": str(args.compile_mode),
                                "text_prompt": case.text_prompt,
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                                "traceback": traceback.format_exc(),
                            }
                            print(
                                f"[hf-edgetam-realcase] FAILED case={case.key} cam={camera_idx} "
                                f"mode={prompt_mode}: {type(exc).__name__}: {exc}",
                                flush=True,
                            )
                        jobs.append(job)

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": str(args.model_id),
        "environment": "edgetam-hf-stream",
        "compile_mode": str(args.compile_mode),
        "compile_metadata": compile_metadata,
        "prompt_modes": list(prompt_modes),
        "camera_ids": [int(item) for item in camera_ids],
        "frames": None if args.frames is None else int(args.frames),
        "streaming_contract": {
            "frame_by_frame_streaming": True,
            "offline_video_input_used": False,
            "frame_source": "png_loop",
            "video_path_argument_used": False,
        },
        "warmup": {
            "enabled": not bool(args.no_warmup),
            "frames": int(args.warmup_frames),
            "compile_warmup_frames": int(args.warmup_frames) if compile_metadata["enabled"] and not bool(args.no_warmup) else 0,
            "record": warmup_record,
        },
        "env": _env_report(str(args.device)),
        "cases": [
            {
                "key": case.key,
                "label": case.label,
                "case_dir": str(case.case_dir),
                "text_prompt": case.text_prompt,
                "sam21_mask_root": str(case.sam21_mask_root),
                "sam31_mask_root": str(case.sam31_mask_root),
            }
            for case in cases
        ],
        "jobs": jobs,
        "aggregate": _aggregate_jobs(jobs),
    }
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark HF EdgeTAMVideo streaming on real aligned QQTT cases."
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_DOC_JSON)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_DOC_MD)
    parser.add_argument("--quality-output", type=Path, default=DEFAULT_QUALITY_JSON)
    parser.add_argument("--case-keys", help="Comma-separated subset of default case keys.")
    parser.add_argument("--case-dir", type=Path, help="Custom QQTT case directory with color/{cam}/{frame}.png.")
    parser.add_argument("--case-key", help="Custom case key. Defaults to --case-dir name.")
    parser.add_argument("--case-label", help="Custom case label. Defaults to --case-key.")
    parser.add_argument("--text-prompt", help="Custom case text prompt used to load the SAM3.1 prompt mask.")
    parser.add_argument("--sam31-mask-root", type=Path, help="Custom SAM3.1 mask root with mask/ sidecars.")
    parser.add_argument("--camera-ids", help="Comma-separated camera ids. Defaults to 0,1,2.")
    parser.add_argument("--prompt-modes", help="Comma-separated modes: point,box,mask,previous_mask.")
    parser.add_argument("--include-previous-mask", action="store_true")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--all-frames", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument(
        "--compile-mode",
        choices=ALL_COMPILE_MODES,
        default=COMPILE_MODE_VISION_REDUCE_OVERHEAD,
        help=(
            "Optional torch.compile ablation mode. Default uses the validated "
            "vision encoder reduce-overhead compile path; pass `none` for the "
            "eager streaming control."
        ),
    )
    parser.add_argument("--bbox-padding-px", type=int, default=0)
    parser.add_argument("--warmup-frames", type=int, default=3)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-write-masks", action="store_true")
    args = parser.parse_args(argv)
    if args.all_frames:
        args.frames = None
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _load_runtime_dependencies()
    payload = run(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[hf-edgetam-realcase] wrote JSON: {args.json_output}", flush=True)

    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    _write_markdown(args.markdown_output, payload)
    print(f"[hf-edgetam-realcase] wrote markdown: {args.markdown_output}", flush=True)

    quality_payload = {
        "timestamp_utc": payload["timestamp_utc"],
        "model_id": payload["model_id"],
        "environment": payload["environment"],
        "compile_mode": payload["compile_mode"],
        "compile_metadata": payload["compile_metadata"],
        "aggregate": payload["aggregate"],
        "quality": [
            {
                "case_key": job.get("case_key"),
                "camera_idx": job.get("camera_idx"),
                "prompt_mode": job.get("prompt_mode"),
                "status": job.get("status"),
                "quality": job.get("quality"),
            }
            for job in payload["jobs"]
        ],
    }
    args.quality_output.parent.mkdir(parents=True, exist_ok=True)
    args.quality_output.write_text(json.dumps(quality_payload, indent=2), encoding="utf-8")
    print(f"[hf-edgetam-realcase] wrote quality JSON: {args.quality_output}", flush=True)

    if payload["aggregate"]["failed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
