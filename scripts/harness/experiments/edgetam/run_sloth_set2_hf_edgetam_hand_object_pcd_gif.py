#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import traceback
import shutil
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_process.visualization.io_artifacts import write_image, write_json, write_ply_ascii
from data_process.visualization.io_case import load_case_frame_camera_clouds, load_case_metadata
from scripts.harness.experiments import run_hf_edgetam_streaming_realcase as hf_stream


CASE_KEY = "sloth_set_2_motion_ffs"
CASE_LABEL = "Sloth Set 2 Motion FFS"
DEFAULT_MODEL_ID = hf_stream.DEFAULT_MODEL_ID
DEFAULT_CASE_DIR = ROOT / "data/different_types/sloth_set_2_motion_ffs"
DEFAULT_OUTPUT_ROOT = ROOT / "result/sloth_set_2_motion_ffs_hf_edgetam_hand_object_pcd"
DEFAULT_SAM31_MASK_ROOT = DEFAULT_OUTPUT_ROOT / "sam31_masks"
DEFAULT_MASK_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT / "hf_edgetam_streaming_multi_object/masks" / CASE_KEY
DEFAULT_PCD_OUTPUT_DIR = DEFAULT_OUTPUT_ROOT / "pcd_gif"
DEFAULT_OUTPUT_NAME = "sloth_set_2_motion_ffs_hf_edgetam_hand_object_pcd"
DEFAULT_ENHANCED_OUTPUT_NAME = "sloth_set_2_motion_ffs_hf_edgetam_hand_object_pcd_enhanced_pt"
DEFAULT_TWO_HAND_OUTPUT_NAME = "sloth_set_2_motion_ffs_hf_edgetam_object_two_hands_pcd"
DEFAULT_TWO_HAND_ENHANCED_OUTPUT_NAME = "sloth_set_2_motion_ffs_hf_edgetam_object_two_hands_pcd_enhanced_pt"
DEFAULT_STREAMING_JSON = (
    ROOT / "docs/generated/sloth_set_2_motion_ffs_hf_edgetam_hand_object_streaming_results.json"
)
DEFAULT_DOC_JSON = ROOT / "docs/generated/sloth_set_2_motion_ffs_hf_edgetam_hand_object_pcd_results.json"
DEFAULT_DOC_MD = ROOT / "docs/generated/sloth_set_2_motion_ffs_hf_edgetam_hand_object_pcd_benchmark.md"
DEFAULT_OBJECT_INIT_MASK_ROOT = ROOT / "result/sloth_set_2_motion_ffs_hf_edgetam_streaming_pcd_xor/sam31_masks"
DEFAULT_HAND_INSTANCE_MASK_ROOT = DEFAULT_OUTPUT_ROOT / "sam31_raw_multi_prompt_masks"
DEFAULT_CAMERA_IDS = (0, 1, 2)
OBJECT_ID = 1
HAND_ID = 2
LEFT_HAND_ID = 2
RIGHT_HAND_ID = 3
PCD_POSTPROCESS_NONE = "none"
PCD_POSTPROCESS_PT_FILTER = "pt-filter"
PCD_POSTPROCESS_ENHANCED_PT = "enhanced-pt"
PCD_POSTPROCESS_MODES = (PCD_POSTPROCESS_NONE, PCD_POSTPROCESS_PT_FILTER, PCD_POSTPROCESS_ENHANCED_PT)
MANIPULATOR_POSTPROCESS_LABELS = frozenset(("controller", "hand", "hands", "left hand", "right hand"))
DEFAULT_PHYSTWIN_RADIUS_M = 0.01
DEFAULT_PHYSTWIN_NB_POINTS = 40
DEFAULT_ENHANCED_COMPONENT_VOXEL_SIZE_M = 0.01
DEFAULT_ENHANCED_KEEP_NEAR_MAIN_GAP_M = 0.0


@dataclass(frozen=True)
class TrackedObject:
    obj_id: int
    label: str
    row_label: str


def tracked_objects(object_label: str, hand_label: str) -> tuple[TrackedObject, TrackedObject]:
    return (
        TrackedObject(obj_id=OBJECT_ID, label=str(object_label), row_label=str(object_label)),
        TrackedObject(obj_id=HAND_ID, label=str(hand_label), row_label=str(hand_label)),
    )


def tracked_two_hand_objects(
    object_label: str,
    left_hand_label: str,
    right_hand_label: str,
) -> tuple[TrackedObject, TrackedObject, TrackedObject]:
    return (
        TrackedObject(obj_id=OBJECT_ID, label=str(object_label), row_label=str(object_label)),
        TrackedObject(obj_id=LEFT_HAND_ID, label=str(left_hand_label), row_label=str(left_hand_label)),
        TrackedObject(obj_id=RIGHT_HAND_ID, label=str(right_hand_label), row_label=str(right_hand_label)),
    )


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _parse_camera_ids(value: str | None) -> tuple[int, ...]:
    if value is None or not str(value).strip():
        return DEFAULT_CAMERA_IDS
    return tuple(int(item.strip()) for item in str(value).split(",") if item.strip())


def _sorted_frame_tokens(case_dir: Path, *, camera_idx: int, frames: int | None) -> list[str]:
    color_dir = Path(case_dir) / "color" / str(int(camera_idx))
    paths = sorted(color_dir.glob("*.png"), key=lambda path: int(path.stem))
    if frames is not None:
        paths = paths[: int(frames)]
    if not paths:
        raise FileNotFoundError(f"No PNG frames found: {color_dir}")
    return [path.stem for path in paths]


def _normalize_label(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def _is_manipulator_label(value: str) -> bool:
    normalized = _normalize_label(value)
    return normalized in MANIPULATOR_POSTPROCESS_LABELS or normalized.endswith(" hand")


def _root_with_mask_dir(mask_root: Path) -> Path:
    root = Path(mask_root)
    return root.parent if root.name == "mask" else root


def _read_mask_info(mask_root: Path, *, camera_idx: int) -> dict[int, str]:
    root = _root_with_mask_dir(mask_root)
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


def _load_label_mask(
    *,
    mask_root: Path,
    case_dir: Path,
    camera_idx: int,
    frame_token: str,
    label: str,
) -> np.ndarray:
    root = _root_with_mask_dir(mask_root)
    info = _read_mask_info(root, camera_idx=int(camera_idx))
    wanted = _normalize_label(label)
    matched_ids = [int(obj_id) for obj_id, item_label in sorted(info.items()) if _normalize_label(item_label) == wanted]
    if not matched_ids:
        raise RuntimeError(f"No masks match label {label!r} in {root} cam{camera_idx}")

    height, width = _image_shape(case_dir, camera_idx=int(camera_idx), frame_token=str(frame_token))
    union = np.zeros((height, width), dtype=bool)
    loaded_any = False
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
        loaded_any = True
    if not loaded_any:
        raise FileNotFoundError(f"No mask PNGs for label {label!r} frame {frame_token} in {root} cam{camera_idx}")
    return union


def _write_mask_png(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.asarray(mask, dtype=np.uint8) * 255)


def _mask_centroid_x(mask: np.ndarray) -> float:
    _ys, xs = np.nonzero(np.asarray(mask, dtype=bool))
    if len(xs) == 0:
        return float("inf")
    return float(np.mean(xs))


def _load_object_id_mask(
    *,
    mask_root: Path,
    case_dir: Path,
    camera_idx: int,
    frame_token: str,
    obj_id: int,
) -> np.ndarray:
    root = _root_with_mask_dir(mask_root)
    height, width = _image_shape(case_dir, camera_idx=int(camera_idx), frame_token=str(frame_token))
    path = root / "mask" / str(int(camera_idx)) / str(int(obj_id)) / f"{frame_token}.png"
    if not path.is_file():
        raise FileNotFoundError(f"Missing object mask: {path}")
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Failed to read mask: {path}")
    if image.shape[:2] != (height, width):
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)
    return image > 0


def build_object_two_hands_init_mask_root(
    *,
    case_dir: Path,
    output_root: Path,
    object_mask_root: Path,
    hand_instance_mask_root: Path,
    camera_ids: Sequence[int],
    frame_token: str,
    object_label: str,
    left_hand_label: str,
    right_hand_label: str,
    source_hand_label: str = "hand",
    overwrite: bool,
) -> dict[str, Any]:
    output_path = Path(output_root)
    mask_dir = output_path / "mask"
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"Init mask root already exists: {output_path}")
        shutil.rmtree(output_path)
    mask_dir.mkdir(parents=True, exist_ok=True)

    objects = tracked_two_hand_objects(object_label, left_hand_label, right_hand_label)
    assignment_by_camera: dict[str, Any] = {}
    for camera_idx in [int(item) for item in camera_ids]:
        object_mask = _load_label_mask(
            mask_root=object_mask_root,
            case_dir=case_dir,
            camera_idx=camera_idx,
            frame_token=str(frame_token),
            label=object_label,
        )

        hand_info = _read_mask_info(hand_instance_mask_root, camera_idx=camera_idx)
        hand_candidates: list[dict[str, Any]] = []
        for hand_obj_id, label in sorted(hand_info.items()):
            if _normalize_label(label) != _normalize_label(source_hand_label):
                continue
            try:
                mask = _load_object_id_mask(
                    mask_root=hand_instance_mask_root,
                    case_dir=case_dir,
                    camera_idx=camera_idx,
                    frame_token=str(frame_token),
                    obj_id=int(hand_obj_id),
                )
            except FileNotFoundError:
                continue
            area = int(np.count_nonzero(mask))
            if area <= 0:
                continue
            hand_candidates.append(
                {
                    "source_obj_id": int(hand_obj_id),
                    "centroid_x": _mask_centroid_x(mask),
                    "area": area,
                    "mask": mask,
                }
            )
        if len(hand_candidates) < 2:
            raise RuntimeError(
                f"Need at least two hand instances in {hand_instance_mask_root} cam{camera_idx} "
                f"frame {frame_token}, found {len(hand_candidates)}"
            )

        hand_candidates = sorted(hand_candidates, key=lambda item: (float(item["centroid_x"]), -int(item["area"])))
        left = hand_candidates[0]
        right = hand_candidates[-1]

        info_path = mask_dir / f"mask_info_{camera_idx}.json"
        info_path.write_text(
            json.dumps({str(obj.obj_id): obj.label for obj in objects}, indent=2),
            encoding="utf-8",
        )
        _write_mask_png(mask_dir / str(camera_idx) / str(OBJECT_ID) / f"{frame_token}.png", object_mask)
        _write_mask_png(mask_dir / str(camera_idx) / str(LEFT_HAND_ID) / f"{frame_token}.png", left["mask"])
        _write_mask_png(mask_dir / str(camera_idx) / str(RIGHT_HAND_ID) / f"{frame_token}.png", right["mask"])
        assignment_by_camera[str(camera_idx)] = {
            "frame_token": str(frame_token),
            "object_source_root": str(object_mask_root),
            "hand_source_root": str(hand_instance_mask_root),
            "source_hand_label": str(source_hand_label),
            "object_id_map": {str(obj.obj_id): obj.label for obj in objects},
            "left_hand": {
                "source_obj_id": int(left["source_obj_id"]),
                "centroid_x": float(left["centroid_x"]),
                "area": int(left["area"]),
            },
            "right_hand": {
                "source_obj_id": int(right["source_obj_id"]),
                "centroid_x": float(right["centroid_x"]),
                "area": int(right["area"]),
            },
            "all_hand_candidates": [
                {
                    "source_obj_id": int(item["source_obj_id"]),
                    "centroid_x": float(item["centroid_x"]),
                    "area": int(item["area"]),
                }
                for item in hand_candidates
            ],
        }

    summary = {
        "case_root": str(case_dir),
        "output_dir": str(output_path),
        "source_mode": "existing_frame0_masks",
        "camera_ids": [int(item) for item in camera_ids],
        "frame_token": str(frame_token),
        "frame0_only": True,
        "text_prompt": f"{object_label},{left_hand_label},{right_hand_label}",
        "parsed_prompts": [str(object_label), str(left_hand_label), str(right_hand_label)],
        "object_id_map": {str(obj.obj_id): obj.label for obj in objects},
        "assignment_rule": "image_x_centroid_leftmost_rightmost",
        "merge_note": (
            "Canonical three-object init root built from the existing stuffed animal masks "
            "and the two raw SAM3.1 hand instances; hand instances are split by frame-0 image x-centroid."
        ),
        "assignment_by_camera": assignment_by_camera,
    }
    (output_path / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _extract_binary_mask(mask_tensor: Any) -> np.ndarray:
    value = mask_tensor
    while hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
        break
    array = np.asarray(value)
    array = np.squeeze(array)
    if array.ndim != 2:
        raise RuntimeError(f"Expected 2-D mask after squeeze, got {array.shape}")
    return array > 0


def _coerce_object_ids(value: Any) -> list[int]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        value = value.tolist()
    return [int(item) for item in list(value)]


def _extract_object_masks_from_hf_output(output: Any, post_masks: Any) -> dict[int, np.ndarray]:
    object_ids = _coerce_object_ids(getattr(output, "object_ids"))
    if len(object_ids) != len(post_masks):
        raise RuntimeError(f"HF output object_ids length {len(object_ids)} != mask length {len(post_masks)}")
    masks: dict[int, np.ndarray] = {}
    for idx, obj_id in enumerate(object_ids):
        masks[int(obj_id)] = _extract_binary_mask(post_masks[idx])
    return masks


def _write_multi_object_masks(
    *,
    mask_root: Path,
    camera_idx: int,
    objects: Sequence[TrackedObject],
    masks_by_object_frame: Mapping[int, Mapping[str, np.ndarray]],
    overwrite: bool,
) -> dict[str, Any]:
    root = Path(mask_root)
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

    mask_dir.mkdir(parents=True, exist_ok=True)
    camera_dir.mkdir(parents=True, exist_ok=True)
    label_by_id = {str(int(obj.obj_id)): str(obj.label) for obj in objects}
    info_path.write_text(json.dumps(label_by_id, indent=2), encoding="utf-8")

    saved_counts: dict[str, int] = {}
    for obj in objects:
        frame_masks = dict(masks_by_object_frame.get(int(obj.obj_id), {}))
        object_dir = camera_dir / str(int(obj.obj_id))
        object_dir.mkdir(parents=True, exist_ok=True)
        for frame_token, mask in sorted(frame_masks.items(), key=lambda item: int(item[0])):
            cv2.imwrite(str(object_dir / f"{frame_token}.png"), np.asarray(mask, dtype=np.uint8) * 255)
        saved_counts[str(int(obj.obj_id))] = int(len(frame_masks))

    return {
        "mask_root": str(root),
        "camera_idx": int(camera_idx),
        "object_labels": label_by_id,
        "saved_frame_counts": saved_counts,
    }


def _latency_summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(statistics.median([float(item) for item in values])),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _mask_quality_summary(masks_by_frame: Mapping[str, np.ndarray]) -> dict[str, Any]:
    areas = [int(np.count_nonzero(mask)) for _token, mask in sorted(masks_by_frame.items(), key=lambda item: int(item[0]))]
    return {
        "frame_count": int(len(areas)),
        "area_mean": float(np.mean(areas)) if areas else 0.0,
        "area_std": float(np.std(areas)) if areas else 0.0,
        "area_min": int(np.min(areas)) if areas else 0,
        "area_max": int(np.max(areas)) if areas else 0,
        "failure_frames": [
            str(token)
            for token, mask in sorted(masks_by_frame.items(), key=lambda item: int(item[0]))
            if int(np.count_nonzero(mask)) <= 0
        ],
    }


def _init_session(*, device: str, dtype: Any, height: int, width: int) -> Any:
    return hf_stream.EdgeTamVideoInferenceSession(
        video=None,
        video_height=int(height),
        video_width=int(width),
        inference_device=device,
        inference_state_device=device,
        video_storage_device=device,
        dtype=dtype,
    )


def _add_multi_object_mask_prompt(
    *,
    processor: Any,
    session: Any,
    objects: Sequence[TrackedObject],
    init_masks: Mapping[int, np.ndarray],
) -> None:
    processor.add_inputs_to_inference_session(
        inference_session=session,
        frame_idx=0,
        obj_ids=[int(obj.obj_id) for obj in objects],
        input_masks=[np.asarray(init_masks[int(obj.obj_id)], dtype=bool) for obj in objects],
    )


def _run_one_multi_object_camera(
    *,
    model: Any,
    processor: Any,
    case_dir: Path,
    case_key: str,
    camera_idx: int,
    frame_tokens: Sequence[str],
    objects: Sequence[TrackedObject],
    sam31_mask_root: Path,
    output_mask_root: Path,
    device: str,
    dtype: Any,
    write_masks: bool,
    overwrite: bool,
) -> dict[str, Any]:
    height, width = _image_shape(case_dir, camera_idx=int(camera_idx), frame_token=str(frame_tokens[0]))
    session = _init_session(device=device, dtype=dtype, height=height, width=width)
    init_masks = {
        int(obj.obj_id): _load_label_mask(
            mask_root=sam31_mask_root,
            case_dir=case_dir,
            camera_idx=int(camera_idx),
            frame_token=str(frame_tokens[0]),
            label=obj.label,
        )
        for obj in objects
    }

    frame_records: list[dict[str, Any]] = []
    masks_by_object_frame: dict[int, dict[str, np.ndarray]] = {int(obj.obj_id): {} for obj in objects}
    started = time.perf_counter()

    for frame_idx, frame_token in enumerate(frame_tokens):
        color_path = case_dir / "color" / str(int(camera_idx)) / f"{frame_token}.png"
        image = Image.open(color_path).convert("RGB")
        inputs, preprocess_ms = hf_stream._time_ms(
            device,
            lambda image=image: processor(images=image, device=device, return_tensors="pt"),
        )
        pixel_values = inputs.pixel_values[0].to(device=device, dtype=dtype)

        prompt_ms = 0.0
        if frame_idx == 0:
            _unused, prompt_ms = hf_stream._time_ms(
                device,
                lambda: _add_multi_object_mask_prompt(
                    processor=processor,
                    session=session,
                    objects=objects,
                    init_masks=init_masks,
                ),
            )

        output, model_ms = hf_stream._time_ms(
            device,
            lambda pixel_values=pixel_values: model(inference_session=session, frame=pixel_values),
        )
        post_masks, postprocess_ms = hf_stream._time_ms(
            device,
            lambda output=output, original_sizes=inputs.original_sizes: processor.post_process_masks(
                [output.pred_masks],
                original_sizes=original_sizes,
                binarize=False,
            )[0],
        )
        object_masks = _extract_object_masks_from_hf_output(output, post_masks)
        object_areas: dict[str, int] = {}
        for obj in objects:
            obj_id = int(obj.obj_id)
            if obj_id not in object_masks:
                raise RuntimeError(f"HF output missing tracked object id {obj_id}")
            mask = object_masks[obj_id]
            masks_by_object_frame[obj_id][str(frame_token)] = mask
            object_areas[str(obj_id)] = int(np.count_nonzero(mask))

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
                "object_areas": object_areas,
            }
        )
        image.close()

    mask_output = None
    if write_masks:
        mask_output = _write_multi_object_masks(
            mask_root=output_mask_root,
            camera_idx=int(camera_idx),
            objects=objects,
            masks_by_object_frame=masks_by_object_frame,
            overwrite=bool(overwrite),
        )

    frame_total = [float(item["frame_total_ms"]) for item in frame_records]
    model_values = [float(item["model_ms"]) for item in frame_records]
    object_quality = {
        str(obj.obj_id): {
            "label": obj.label,
            **_mask_quality_summary(masks_by_object_frame[int(obj.obj_id)]),
        }
        for obj in objects
    }
    return {
        "status": "pass",
        "case_key": str(case_key),
        "case_label": CASE_LABEL,
        "case_dir": str(case_dir),
        "camera_idx": int(camera_idx),
        "frame_count": int(len(frame_records)),
        "frame_tokens": [str(item) for item in frame_tokens],
        "object_ids": [int(obj.obj_id) for obj in objects],
        "object_labels": {str(int(obj.obj_id)): str(obj.label) for obj in objects},
        "first_frame_latency_ms": float(frame_total[0]) if frame_total else 0.0,
        "first_frame_model_ms": float(model_values[0]) if model_values else 0.0,
        "subsequent_frame_latency_ms": _latency_summary(frame_total[1:]),
        "subsequent_model_ms": _latency_summary(model_values[1:]),
        "preprocess_ms": _latency_summary([float(item["preprocess_ms"]) for item in frame_records]),
        "model_ms": _latency_summary(model_values),
        "postprocess_ms": _latency_summary([float(item["postprocess_ms"]) for item in frame_records]),
        "prompt_ms": _latency_summary([float(item["prompt_ms"]) for item in frame_records if float(item["prompt_ms"]) > 0]),
        "streaming_total_ms": float(sum(frame_total)),
        "wall_ms": float((time.perf_counter() - started) * 1000.0),
        "end_to_end_streaming_fps": float(1000.0 * len(frame_records) / max(1e-9, sum(frame_total))),
        "model_only_streaming_fps": float(1000.0 * len(frame_records) / max(1e-9, sum(model_values))),
        "mask_output": mask_output,
        "quality_by_object": object_quality,
        "frames": frame_records,
    }


def _aggregate_stream_jobs(jobs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    passed = [job for job in jobs if job.get("status") == "pass"]
    failed = [job for job in jobs if job.get("status") != "pass"]
    return {
        "job_count": int(len(jobs)),
        "passed": int(len(passed)),
        "failed": int(len(failed)),
        "first_frame_latency_ms": _latency_summary([float(job["first_frame_latency_ms"]) for job in passed]),
        "subsequent_frame_latency_median_ms": _latency_summary(
            [float(job["subsequent_frame_latency_ms"]["median"]) for job in passed]
        ),
        "end_to_end_streaming_fps": _latency_summary([float(job["end_to_end_streaming_fps"]) for job in passed]),
        "model_only_streaming_fps": _latency_summary([float(job["model_only_streaming_fps"]) for job in passed]),
    }


def run_multi_object_streaming(args: argparse.Namespace, objects: Sequence[TrackedObject]) -> dict[str, Any]:
    case_dir = _resolve_path(args.case_dir)
    sam31_mask_root = _resolve_path(args.sam31_mask_root)
    mask_output_root = _resolve_path(args.output_root) / "hf_edgetam_streaming_multi_object/masks" / CASE_KEY
    camera_ids = _parse_camera_ids(args.camera_ids)
    dtype = hf_stream._dtype_from_name(args.dtype)
    if str(args.device).startswith("cuda") and not hf_stream.torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    print(f"[hf-edgetam-hand-object] loading model: {args.model_id}", flush=True)
    model = hf_stream.EdgeTamVideoModel.from_pretrained(str(args.model_id)).to(str(args.device), dtype=dtype).eval()
    model, compile_metadata = hf_stream._apply_compile_mode(model, str(args.compile_mode))
    if compile_metadata["enabled"]:
        print(
            "[hf-edgetam-hand-object] compile mode="
            f"{compile_metadata['compile_mode']} targets={compile_metadata['applied_targets']}",
            flush=True,
        )
    processor = hf_stream.Sam2VideoProcessor.from_pretrained(str(args.model_id))
    autocast_ctx = (
        hf_stream.torch.autocast("cuda", dtype=dtype)
        if str(args.device).startswith("cuda")
        else nullcontext()
    )

    jobs: list[dict[str, Any]] = []
    warmup_record: dict[str, Any] | None = None
    with hf_stream.torch.inference_mode(), autocast_ctx:
        if not bool(args.no_warmup):
            warm_camera = int(camera_ids[0])
            warm_frames = _sorted_frame_tokens(
                case_dir,
                camera_idx=warm_camera,
                frames=max(1, int(args.warmup_frames)),
            )
            print(
                f"[hf-edgetam-hand-object] warmup cam={warm_camera} frames={len(warm_frames)}",
                flush=True,
            )
            warmup_record = _run_one_multi_object_camera(
                model=model,
                processor=processor,
                case_dir=case_dir,
                case_key=CASE_KEY,
                camera_idx=warm_camera,
                frame_tokens=warm_frames,
                objects=objects,
                sam31_mask_root=sam31_mask_root,
                output_mask_root=mask_output_root / "_warmup_masks",
                device=str(args.device),
                dtype=dtype,
                write_masks=False,
                overwrite=True,
            )
            warmup_record["compile_mode"] = str(args.compile_mode)

        for camera_idx in camera_ids:
            frame_tokens = _sorted_frame_tokens(case_dir, camera_idx=int(camera_idx), frames=args.frames)
            print(
                f"[hf-edgetam-hand-object] case={CASE_KEY} cam={camera_idx} "
                f"frames={len(frame_tokens)} objects={[obj.label for obj in objects]}",
                flush=True,
            )
            try:
                job = _run_one_multi_object_camera(
                    model=model,
                    processor=processor,
                    case_dir=case_dir,
                    case_key=CASE_KEY,
                    camera_idx=int(camera_idx),
                    frame_tokens=frame_tokens,
                    objects=objects,
                    sam31_mask_root=sam31_mask_root,
                    output_mask_root=mask_output_root,
                    device=str(args.device),
                    dtype=dtype,
                    write_masks=True,
                    overwrite=bool(args.overwrite),
                )
                job["compile_mode"] = str(args.compile_mode)
            except Exception as exc:
                job = {
                    "status": "failed",
                    "case_key": CASE_KEY,
                    "case_label": CASE_LABEL,
                    "case_dir": str(case_dir),
                    "camera_idx": int(camera_idx),
                    "compile_mode": str(args.compile_mode),
                    "object_labels": {str(int(obj.obj_id)): str(obj.label) for obj in objects},
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                print(
                    f"[hf-edgetam-hand-object] FAILED cam={camera_idx}: {type(exc).__name__}: {exc}",
                    flush=True,
                )
            jobs.append(job)

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": str(args.model_id),
        "environment": "edgetam-hf-stream",
        "compile_mode": str(args.compile_mode),
        "compile_metadata": compile_metadata,
        "case": {
            "key": CASE_KEY,
            "label": CASE_LABEL,
            "case_dir": str(case_dir),
            "sam31_mask_root": str(sam31_mask_root),
            "mask_output_root": str(mask_output_root),
        },
        "objects": [
            {"object_id": int(obj.obj_id), "label": str(obj.label), "row_label": str(obj.row_label)}
            for obj in objects
        ],
        "camera_ids": [int(item) for item in camera_ids],
        "frames": None if args.frames is None else int(args.frames),
        "streaming_contract": {
            "frame_by_frame_streaming": True,
            "offline_video_input_used": False,
            "frame_source": "png_loop",
            "video_path_argument_used": False,
            "frame0_prompt": "sam31_mask_multi_object",
        },
        "warmup": {
            "enabled": not bool(args.no_warmup),
            "frames": int(args.warmup_frames),
            "compile_warmup_frames": int(args.warmup_frames) if compile_metadata["enabled"] and not bool(args.no_warmup) else 0,
            "record": warmup_record,
        },
        "env": hf_stream._env_report(str(args.device)),
        "jobs": jobs,
        "aggregate": _aggregate_stream_jobs(jobs),
    }


def metadata_with_depth_scale_override(
    metadata: Mapping[str, Any],
    *,
    depth_scale_override_m_per_unit: float,
) -> tuple[dict[str, Any], bool]:
    payload = dict(metadata)
    camera_count = len(payload.get("serial_numbers", []))
    if camera_count <= 0:
        raise ValueError("metadata must contain at least one serial number.")
    existing = payload.get("depth_scale_m_per_unit")
    if existing is None:
        payload["depth_scale_m_per_unit"] = [float(depth_scale_override_m_per_unit) for _ in range(camera_count)]
        return payload, True
    if not isinstance(existing, list):
        existing = [existing for _ in range(camera_count)]
    if len(existing) != camera_count or any(item is None for item in existing):
        payload["depth_scale_m_per_unit"] = [float(depth_scale_override_m_per_unit) for _ in range(camera_count)]
        return payload, True
    payload["depth_scale_m_per_unit"] = [float(item) for item in existing]
    return payload, False


def _format_point_count(point_count: int) -> str:
    count = int(point_count)
    if count >= 1_000_000:
        return f"{count / 1_000_000.0:.1f}M"
    if count >= 1000:
        return f"{count / 1000.0:.1f}k"
    return str(count)


def _point_mask_values(mask: np.ndarray, source_pixel_uv: np.ndarray) -> np.ndarray:
    pixel_uv = np.asarray(source_pixel_uv, dtype=np.int32).reshape(-1, 2)
    if len(pixel_uv) == 0:
        return np.zeros((0,), dtype=bool)
    height, width = np.asarray(mask).shape[:2]
    x = np.clip(pixel_uv[:, 0], 0, width - 1)
    y = np.clip(pixel_uv[:, 1], 0, height - 1)
    return np.asarray(mask, dtype=bool)[y, x]


def _fuse_masked_cloud(
    *,
    camera_clouds: Sequence[dict[str, Any]],
    masks_by_camera: Mapping[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    point_sets: list[np.ndarray] = []
    color_sets: list[np.ndarray] = []
    per_camera: list[dict[str, Any]] = []
    for camera_cloud in camera_clouds:
        camera_idx = int(camera_cloud["camera_idx"])
        points = np.asarray(camera_cloud["points"], dtype=np.float32).reshape(-1, 3)
        colors = np.asarray(camera_cloud["colors"], dtype=np.uint8).reshape(-1, 3)
        selected = _point_mask_values(masks_by_camera[camera_idx], np.asarray(camera_cloud["source_pixel_uv"]))
        point_sets.append(points[selected])
        color_sets.append(colors[selected])
        per_camera.append(
            {
                "camera_idx": camera_idx,
                "input_point_count": int(len(points)),
                "masked_point_count": int(np.count_nonzero(selected)),
            }
        )
    if point_sets:
        return np.concatenate(point_sets, axis=0), np.concatenate(color_sets, axis=0), per_camera
    return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8), per_camera


def _apply_pcd_postprocess(
    *,
    points: np.ndarray,
    colors: np.ndarray,
    mode: str,
    phystwin_radius_m: float,
    phystwin_nb_points: int,
    enhanced_component_voxel_size_m: float,
    enhanced_keep_near_main_gap_m: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    point_array = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    color_array = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    if str(mode) == PCD_POSTPROCESS_NONE:
        return point_array, color_array, {
            "enabled": False,
            "mode": PCD_POSTPROCESS_NONE,
            "input_point_count": int(len(point_array)),
            "output_point_count": int(len(point_array)),
        }
    if str(mode) == PCD_POSTPROCESS_PT_FILTER:
        from data_process.visualization.experiments.ffs_confidence_filter_pcd_compare import (
            _apply_phystwin_like_radius_postprocess,
        )

        filtered_points, filtered_colors, stats = _apply_phystwin_like_radius_postprocess(
            points=point_array,
            colors=color_array,
            enabled=True,
            radius_m=float(phystwin_radius_m),
            nb_points=int(phystwin_nb_points),
        )
        stats["output_point_count"] = int(len(filtered_points))
        return filtered_points, filtered_colors, stats
    if str(mode) != PCD_POSTPROCESS_ENHANCED_PT:
        raise ValueError(f"Unsupported PCD postprocess mode: {mode}")

    # Lazy import keeps HF streaming usable in edgetam-hf-stream, where Open3D is
    # not installed but enhanced render-only runs are executed in SAM21-max.
    from data_process.visualization.experiments.ffs_confidence_filter_pcd_compare import (
        _apply_enhanced_phystwin_like_postprocess,
    )

    filtered_points, filtered_colors, stats = _apply_enhanced_phystwin_like_postprocess(
        points=point_array,
        colors=color_array,
        enabled=True,
        radius_m=float(phystwin_radius_m),
        nb_points=int(phystwin_nb_points),
        component_voxel_size_m=float(enhanced_component_voxel_size_m),
        keep_near_main_gap_m=float(enhanced_keep_near_main_gap_m),
    )
    return filtered_points, filtered_colors, stats


def _postprocess_mode_for_object(
    obj: TrackedObject,
    *,
    default_mode: str,
    controller_mode: str | None,
) -> str:
    if controller_mode is not None and _is_manipulator_label(obj.label):
        return str(controller_mode)
    normalized_default = str(default_mode)
    if normalized_default == PCD_POSTPROCESS_ENHANCED_PT and _is_manipulator_label(obj.label):
        return PCD_POSTPROCESS_PT_FILTER
    return normalized_default


def _postprocess_label_suffix(mode: str) -> str:
    if str(mode) == PCD_POSTPROCESS_ENHANCED_PT:
        return " | ePT"
    if str(mode) == PCD_POSTPROCESS_PT_FILTER:
        return " | PT"
    return ""


def _deterministic_point_cap(points: np.ndarray, colors: np.ndarray, *, max_points: int | None) -> tuple[np.ndarray, np.ndarray]:
    point_array = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    color_array = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    if max_points is None or len(point_array) <= int(max_points):
        return point_array, color_array
    indices = np.linspace(0, len(point_array) - 1, num=int(max_points), dtype=np.int64)
    return point_array[indices], color_array[indices]


def _scale_intrinsic_matrix(
    intrinsic_matrix: np.ndarray,
    *,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> np.ndarray:
    source_w, source_h = source_size
    target_w, target_h = target_size
    scaled = np.asarray(intrinsic_matrix, dtype=np.float32).copy()
    scaled[0, 0] *= float(target_w) / float(source_w)
    scaled[0, 2] *= float(target_w) / float(source_w)
    scaled[1, 1] *= float(target_h) / float(source_h)
    scaled[1, 2] *= float(target_h) / float(source_h)
    return scaled


def _build_view_specs(
    camera_clouds: Sequence[dict[str, Any]],
    *,
    tile_width: int,
    tile_height: int,
) -> dict[int, dict[str, Any]]:
    specs: dict[int, dict[str, Any]] = {}
    for camera_cloud in camera_clouds:
        camera_idx = int(camera_cloud["camera_idx"])
        image = cv2.imread(str(camera_cloud["color_path"]), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Missing color frame: {camera_cloud['color_path']}")
        source_h, source_w = image.shape[:2]
        c2w = np.asarray(camera_cloud["c2w"], dtype=np.float32).reshape(4, 4)
        specs[camera_idx] = {
            "intrinsic_matrix": _scale_intrinsic_matrix(
                np.asarray(camera_cloud["K_color"], dtype=np.float32),
                source_size=(int(source_w), int(source_h)),
                target_size=(int(tile_width), int(tile_height)),
            ),
            "extrinsic_matrix": np.linalg.inv(c2w).astype(np.float32),
            "image_size": [int(tile_width), int(tile_height)],
            "source_image_size": [int(source_w), int(source_h)],
        }
    return specs


def render_pinhole_point_cloud(
    points: np.ndarray,
    colors: np.ndarray,
    *,
    intrinsic_matrix: np.ndarray,
    extrinsic_matrix: np.ndarray,
    width: int,
    height: int,
    point_radius_px: int = 1,
    background_bgr: tuple[int, int, int] = (28, 28, 30),
) -> np.ndarray:
    point_array = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    color_array = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    canvas = np.full((int(height), int(width), 3), background_bgr, dtype=np.uint8)
    if len(point_array) == 0:
        return canvas

    homogeneous = np.concatenate([point_array, np.ones((len(point_array), 1), dtype=np.float32)], axis=1)
    camera_points = homogeneous @ np.asarray(extrinsic_matrix, dtype=np.float32).reshape(4, 4).T
    xyz = camera_points[:, :3]
    z = xyz[:, 2]
    valid = z > 1e-6
    if not np.any(valid):
        return canvas
    xyz = xyz[valid]
    z = z[valid]
    color_values = color_array[valid]
    K = np.asarray(intrinsic_matrix, dtype=np.float32).reshape(3, 3)
    u = K[0, 0] * (xyz[:, 0] / z) + K[0, 2]
    v = K[1, 1] * (xyz[:, 1] / z) + K[1, 2]
    inside = (u >= 0) & (u < int(width)) & (v >= 0) & (v < int(height))
    if not np.any(inside):
        return canvas
    u_i = np.rint(u[inside]).astype(np.int32)
    v_i = np.rint(v[inside]).astype(np.int32)
    z_i = z[inside]
    color_i = color_values[inside]

    order = np.argsort(z_i)[::-1]
    u_i = u_i[order]
    v_i = v_i[order]
    color_i = color_i[order]
    radius = max(0, int(point_radius_px))
    offsets = [(0, 0)]
    if radius > 0:
        offsets = [
            (dx, dy)
            for dy in range(-radius, radius + 1)
            for dx in range(-radius, radius + 1)
            if dx * dx + dy * dy <= radius * radius
        ]
    for dx, dy in offsets:
        uu = u_i + int(dx)
        vv = v_i + int(dy)
        ok = (uu >= 0) & (uu < int(width)) & (vv >= 0) & (vv < int(height))
        if np.any(ok):
            canvas[vv[ok], uu[ok]] = color_i[ok]
    return canvas


def _draw_text_fit(
    image: np.ndarray,
    *,
    text: str,
    origin: tuple[int, int],
    max_width: int,
    font_scale: float,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    scale = float(font_scale)
    while scale > 0.30:
        width = cv2.getTextSize(str(text), cv2.FONT_HERSHEY_SIMPLEX, scale, int(thickness))[0][0]
        if width <= int(max_width):
            break
        scale -= 0.035
    cv2.putText(
        image,
        str(text),
        (int(origin[0]), int(origin[1])),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        int(thickness),
        cv2.LINE_AA,
    )


def label_tile(image: np.ndarray, *, label: str, tile_width: int, tile_height: int) -> np.ndarray:
    tile = cv2.resize(np.asarray(image, dtype=np.uint8), (int(tile_width), int(tile_height)), interpolation=cv2.INTER_AREA)
    cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, 24), (0, 0, 0), -1)
    _draw_text_fit(
        tile,
        text=str(label),
        origin=(8, 17),
        max_width=max(16, int(tile_width) - 16),
        font_scale=0.48,
        color=(255, 255, 255),
        thickness=1,
    )
    return tile


def compose_panel(
    *,
    title_lines: Sequence[str],
    row_headers: Sequence[str],
    column_headers: Sequence[str],
    image_rows: Sequence[Sequence[np.ndarray]],
    row_label_width: int,
) -> np.ndarray:
    expected_rows = len(row_headers)
    expected_cols = len(column_headers)
    if len(image_rows) != expected_rows or any(len(row) != expected_cols for row in image_rows):
        raise ValueError(f"Panel requires exactly {expected_rows} rows of {expected_cols} images.")
    tile_h, tile_w = image_rows[0][0].shape[:2]
    if any(tile.shape[:2] != (tile_h, tile_w) for row in image_rows for tile in row):
        raise ValueError("All panel tiles must have the same shape.")

    title_h = 84
    header_h = 38
    body_h = tile_h * expected_rows
    body_w = int(row_label_width) + tile_w * expected_cols
    title = np.full((title_h, body_w, 3), (10, 10, 10), dtype=np.uint8)
    for line_idx, line in enumerate(list(title_lines)[:2]):
        _draw_text_fit(
            title,
            text=str(line),
            origin=(16, 30 + line_idx * 28),
            max_width=body_w - 32,
            font_scale=0.82 if line_idx == 0 else 0.56,
            color=(255, 255, 255) if line_idx == 0 else (220, 220, 220),
            thickness=2 if line_idx == 0 else 1,
        )

    header = np.full((header_h, body_w, 3), (18, 18, 18), dtype=np.uint8)
    header[:, : int(row_label_width)] = (14, 14, 14)
    for col_idx, column_header in enumerate(column_headers):
        x0 = int(row_label_width) + col_idx * tile_w
        _draw_text_fit(
            header,
            text=str(column_header),
            origin=(x0 + 8, 26),
            max_width=tile_w - 16,
            font_scale=0.58,
            color=(255, 255, 255),
            thickness=2,
        )

    body = np.full((body_h, body_w, 3), (24, 24, 24), dtype=np.uint8)
    for row_idx, (row_header, row_tiles) in enumerate(zip(row_headers, image_rows, strict=True)):
        y0 = row_idx * tile_h
        body[y0 : y0 + tile_h, : int(row_label_width)] = (12, 12, 12)
        _draw_text_fit(
            body,
            text=str(row_header),
            origin=(12, y0 + max(26, tile_h // 2)),
            max_width=int(row_label_width) - 20,
            font_scale=0.62,
            color=(255, 255, 255),
            thickness=2,
        )
        for col_idx, tile in enumerate(row_tiles):
            x0 = int(row_label_width) + col_idx * tile_w
            body[y0 : y0 + tile_h, x0 : x0 + tile_w] = tile
    return np.vstack([title, header, body])


def _write_gif(path: Path, frames_bgr: Sequence[np.ndarray], *, fps: int) -> None:
    if not frames_bgr:
        raise ValueError("No frames to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = [Image.fromarray(cv2.cvtColor(np.asarray(frame, dtype=np.uint8), cv2.COLOR_BGR2RGB)) for frame in frames_bgr]
    duration_ms = max(20, int(round(1000.0 / max(1, int(fps)))))
    frames[0].save(
        str(path),
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def render_hand_object_pcd_panel(
    *,
    case_dir: Path,
    mask_root: Path,
    mask_source_label: str = "HF EdgeTAM",
    output_dir: Path,
    output_name: str,
    objects: Sequence[TrackedObject],
    camera_ids: Sequence[int],
    frames: int | None,
    tile_width: int,
    tile_height: int,
    row_label_width: int,
    gif_fps: int,
    max_points_per_camera: int | None,
    max_points_per_render: int | None,
    depth_min_m: float,
    depth_max_m: float,
    depth_scale_override_m_per_unit: float,
    point_radius_px: int,
    pcd_postprocess_mode: str = PCD_POSTPROCESS_NONE,
    controller_pcd_postprocess_mode: str | None = None,
    phystwin_radius_m: float = DEFAULT_PHYSTWIN_RADIUS_M,
    phystwin_nb_points: int = DEFAULT_PHYSTWIN_NB_POINTS,
    enhanced_component_voxel_size_m: float = DEFAULT_ENHANCED_COMPONENT_VOXEL_SIZE_M,
    enhanced_keep_near_main_gap_m: float = DEFAULT_ENHANCED_KEEP_NEAR_MAIN_GAP_M,
) -> dict[str, Any]:
    case_path = Path(case_dir)
    output_path = Path(output_dir)
    gif_dir = output_path / "gifs"
    first_dir = output_path / "first_frames"
    ply_dir = output_path / "first_frame_ply"
    gif_dir.mkdir(parents=True, exist_ok=True)
    first_dir.mkdir(parents=True, exist_ok=True)
    ply_dir.mkdir(parents=True, exist_ok=True)
    gif_path = gif_dir / f"{output_name}.gif"
    first_frame_path = first_dir / f"{output_name}_first.png"

    metadata_raw = load_case_metadata(case_path)
    metadata, depth_scale_override_used = metadata_with_depth_scale_override(
        metadata_raw,
        depth_scale_override_m_per_unit=float(depth_scale_override_m_per_unit),
    )
    camera_ids = tuple(int(item) for item in camera_ids)
    frame_tokens = _sorted_frame_tokens(case_path, camera_idx=int(camera_ids[0]), frames=frames)
    view_specs: dict[int, dict[str, Any]] | None = None
    frames_bgr: list[np.ndarray] = []
    per_frame: list[dict[str, Any]] = []
    aggregate_points: dict[str, list[int]] = {str(obj.obj_id): [] for obj in objects}
    aggregate_raw_points: dict[str, list[int]] = {str(obj.obj_id): [] for obj in objects}
    object_row_text = ", ".join(str(obj.row_label) for obj in objects)
    postprocess_mode_by_object = {
        str(int(obj.obj_id)): _postprocess_mode_for_object(
            obj,
            default_mode=str(pcd_postprocess_mode),
            controller_mode=controller_pcd_postprocess_mode,
        )
        for obj in objects
    }

    for frame_idx, frame_token in enumerate(frame_tokens):
        camera_clouds, camera_stats = load_case_frame_camera_clouds(
            case_dir=case_path,
            metadata=metadata,
            frame_idx=int(frame_token),
            depth_source="ffs",
            use_float_ffs_depth_when_available=True,
            max_points_per_camera=max_points_per_camera,
            depth_min_m=float(depth_min_m),
            depth_max_m=float(depth_max_m),
        )
        _unused_camera_stats = camera_stats
        camera_clouds = [cloud for cloud in camera_clouds if int(cloud["camera_idx"]) in set(camera_ids)]
        if view_specs is None:
            view_specs = _build_view_specs(camera_clouds, tile_width=int(tile_width), tile_height=int(tile_height))

        image_rows: list[list[np.ndarray]] = []
        frame_objects: list[dict[str, Any]] = []
        for obj in objects:
            masks_by_camera = {
                int(camera_idx): _load_label_mask(
                    mask_root=mask_root,
                    case_dir=case_path,
                    camera_idx=int(camera_idx),
                    frame_token=str(frame_token),
                    label=obj.label,
                )
                for camera_idx in camera_ids
            }
            points, colors, per_camera = _fuse_masked_cloud(camera_clouds=camera_clouds, masks_by_camera=masks_by_camera)
            raw_point_count = int(len(points))
            object_postprocess_mode = postprocess_mode_by_object[str(int(obj.obj_id))]
            points, colors, postprocess_stats = _apply_pcd_postprocess(
                points=points,
                colors=colors,
                mode=object_postprocess_mode,
                phystwin_radius_m=float(phystwin_radius_m),
                phystwin_nb_points=int(phystwin_nb_points),
                enhanced_component_voxel_size_m=float(enhanced_component_voxel_size_m),
                enhanced_keep_near_main_gap_m=float(enhanced_keep_near_main_gap_m),
            )
            aggregate_raw_points[str(obj.obj_id)].append(raw_point_count)
            aggregate_points[str(obj.obj_id)].append(int(len(points)))
            if frame_idx == 0:
                write_ply_ascii(ply_dir / f"{output_name}_{obj.obj_id}_{obj.label.replace(' ', '_')}_frame0000.ply", points, colors)

            render_points, render_colors = _deterministic_point_cap(
                points,
                colors,
                max_points=max_points_per_render,
            )
            label_suffix = _postprocess_label_suffix(object_postprocess_mode)
            label = f"{obj.label} | {_format_point_count(len(points))} pts{label_suffix}"
            row_tiles = []
            for camera_idx in camera_ids:
                spec = view_specs[int(camera_idx)]
                tile = render_pinhole_point_cloud(
                    render_points,
                    render_colors,
                    intrinsic_matrix=np.asarray(spec["intrinsic_matrix"], dtype=np.float32),
                    extrinsic_matrix=np.asarray(spec["extrinsic_matrix"], dtype=np.float32),
                    width=int(tile_width),
                    height=int(tile_height),
                    point_radius_px=int(point_radius_px),
                )
                row_tiles.append(label_tile(tile, label=label, tile_width=int(tile_width), tile_height=int(tile_height)))
            image_rows.append(row_tiles)
            frame_objects.append(
                {
                    "object_id": int(obj.obj_id),
                    "label": str(obj.label),
                    "raw_point_count": raw_point_count,
                    "point_count": int(len(points)),
                    "per_camera": per_camera,
                    "postprocess_mode": object_postprocess_mode,
                    "postprocess_stats": postprocess_stats,
                }
            )

        board = compose_panel(
            title_lines=[
                f"{CASE_LABEL} | {mask_source_label} fused PCD | frame {frame_idx + 1}/{len(frame_tokens)}",
                f"rows=masked objects | columns=original camera pinhole views | FFS depth masked by {mask_source_label}",
            ],
            row_headers=[obj.row_label for obj in objects],
            column_headers=[f"cam{camera_idx}" for camera_idx in camera_ids],
            image_rows=image_rows,
            row_label_width=int(row_label_width),
        )
        if frame_idx == 0:
            write_image(first_frame_path, board)
        frames_bgr.append(board)
        per_frame.append({"frame_idx": int(frame_idx), "frame_token": str(frame_token), "objects": frame_objects})
        if frame_idx == 0 or frame_idx + 1 == len(frame_tokens) or (frame_idx + 1) % 10 == 0:
            print(f"[hf-edgetam-hand-object] rendered PCD frame {frame_idx + 1}/{len(frame_tokens)}", flush=True)

    _write_gif(gif_path, frames_bgr, fps=int(gif_fps))
    aggregate = {}
    for obj in objects:
        values = aggregate_points[str(obj.obj_id)]
        raw_values = aggregate_raw_points[str(obj.obj_id)]
        aggregate[str(obj.obj_id)] = {
            "label": obj.label,
            "raw_point_count_mean": float(np.mean(raw_values)) if raw_values else 0.0,
            "raw_point_count_min": int(np.min(raw_values)) if raw_values else 0,
            "raw_point_count_max": int(np.max(raw_values)) if raw_values else 0,
            "point_count_mean": float(np.mean(values)) if values else 0.0,
            "point_count_min": int(np.min(values)) if values else 0,
            "point_count_max": int(np.max(values)) if values else 0,
            "sample_count": int(len(values)),
        }

    return {
        "case_key": CASE_KEY,
        "case_label": CASE_LABEL,
        "case_dir": str(case_path),
        "frames": int(len(frame_tokens)),
        "camera_ids": [int(item) for item in camera_ids],
        "objects": [
            {"object_id": int(obj.obj_id), "label": str(obj.label), "row_label": str(obj.row_label)}
            for obj in objects
        ],
        "gif_path": str(gif_path),
        "first_frame_path": str(first_frame_path),
        "first_frame_ply_dir": str(ply_dir),
        "mask_root": str(mask_root),
        "mask_source_label": str(mask_source_label),
        "view_mode": "fused_pcd_original_camera_pinhole",
        "render_contract": {
            "rows": object_row_text,
            "columns": "selected original camera pinhole views",
            "point_source": f"FFS depth filtered by {mask_source_label} masks",
            "point_colors": "original RGB camera colors",
            "qualitative_only": True,
            "postprocess_mode": str(pcd_postprocess_mode),
            "per_object_postprocess_modes": postprocess_mode_by_object,
        },
        "depth": {
            "source": "depth",
            "depth_source_argument": "ffs",
            "use_float_ffs_depth_when_available": True,
            "depth_scale_override_m_per_unit": float(depth_scale_override_m_per_unit),
            "depth_scale_override_used": bool(depth_scale_override_used),
            "depth_min_m": float(depth_min_m),
            "depth_max_m": float(depth_max_m),
        },
        "postprocess": {
            "mode": str(pcd_postprocess_mode),
            "enabled": str(pcd_postprocess_mode) != PCD_POSTPROCESS_NONE,
            "controller_mode": None if controller_pcd_postprocess_mode is None else str(controller_pcd_postprocess_mode),
            "manipulator_default_mode": (
                PCD_POSTPROCESS_PT_FILTER
                if str(pcd_postprocess_mode) == PCD_POSTPROCESS_ENHANCED_PT
                else None
            ),
            "per_object_modes": postprocess_mode_by_object,
            "phystwin_radius_m": float(phystwin_radius_m),
            "phystwin_nb_points": int(phystwin_nb_points),
            "enhanced_component_voxel_size_m": float(enhanced_component_voxel_size_m),
            "enhanced_keep_near_main_gap_m": float(enhanced_keep_near_main_gap_m),
        },
        "aggregate": aggregate,
        "frames_detail": per_frame,
    }


def write_report(markdown_path: Path, *, pcd_summary: Mapping[str, Any], streaming_payload: Mapping[str, Any]) -> None:
    aggregate = streaming_payload.get("aggregate", {})
    compile_metadata = streaming_payload.get("compile_metadata", {})
    sam31_summary = pcd_summary.get("sam31_mask_summary", {})
    postprocess_summary = pcd_summary.get("postprocess", {})
    per_object_modes = dict(postprocess_summary.get("per_object_modes", {}))
    sam31_root = (
        sam31_summary.get("output_dir")
        or streaming_payload.get("case", {}).get("sam31_mask_root")
        or ""
    )
    object_specs = list(pcd_summary.get("objects", []))
    object_id_text = ", ".join(
        f"{item.get('object_id')}={item.get('label')}" for item in object_specs
    )
    mask_source_label = str(pcd_summary.get("mask_source_label") or "HF EdgeTAM")
    is_hf_edgetam_source = _normalize_label(mask_source_label) == "hf edgetam"
    manipulator_object_ids = [
        str(item.get("object_id"))
        for item in object_specs
        if _is_manipulator_label(str(item.get("label", "")))
    ]
    manipulator_effective_modes = sorted(
        {
            str(per_object_modes.get(obj_id, postprocess_summary.get("mode", PCD_POSTPROCESS_NONE)))
            for obj_id in manipulator_object_ids
        }
    )
    manipulator_effective_text = ", ".join(manipulator_effective_modes) if manipulator_effective_modes else "none"
    lines = [
        f"# Sloth Set 2 {mask_source_label} Object/Hands PCD Panel",
        "",
        "## Output",
        "",
        f"- GIF: `{pcd_summary.get('gif_path')}`",
        f"- First frame: `{pcd_summary.get('first_frame_path')}`",
        f"- First-frame PLY dir: `{pcd_summary.get('first_frame_ply_dir')}`",
        (
            f"- Streaming JSON: `{streaming_payload.get('streaming_results_path', '')}`"
            if is_hf_edgetam_source
            else "- Streaming JSON: `not written; render-only mask panel`"
        ),
        f"- Frames: `{pcd_summary.get('frames')}`",
        f"- Cameras: `{pcd_summary.get('camera_ids')}`",
        "",
        "## Streaming Contract" if is_hf_edgetam_source else "## Mask Source Contract",
        "",
        (
            "- `frame_by_frame_streaming=true`"
            if is_hf_edgetam_source
            else "- `frame_by_frame_streaming=false` for this render-only SAM3.1 mask panel."
        ),
        (
            "- `offline_video_input_used=false`"
            if is_hf_edgetam_source
            else "- No HF EdgeTAM tracking or propagation was run for this panel."
        ),
        "- `frame_source=png_loop`",
        (
            f"- Frame 0 prompt is SAM3.1 mask prompt for `{len(object_specs)}` objects."
            if is_hf_edgetam_source
            else f"- All frames are masked directly by `{mask_source_label}` masks."
        ),
        "- This is a qualitative PCD panel, not an XOR quality benchmark.",
        f"- PCD postprocess mode: `{pcd_summary.get('postprocess', {}).get('mode', PCD_POSTPROCESS_NONE)}`",
        "",
        "## SAM3.1 Init Root",
        "",
        f"- Root: `{sam31_root}`",
        f"- Render mask root: `{pcd_summary.get('mask_root')}`",
        f"- Object IDs: `{object_id_text}`",
        f"- Note: {sam31_summary.get('merge_note', 'standard SAM3.1 mask root')}",
        *(
            [
                "",
                "## Compile",
                "",
                f"- Mode: `{streaming_payload.get('compile_mode')}`",
                f"- Enabled: `{compile_metadata.get('enabled')}`",
                f"- Applied targets: `{', '.join(compile_metadata.get('applied_targets', [])) or 'none'}`",
                f"- Torch compile mode: `{compile_metadata.get('torch_compile_mode')}`",
                "",
                "## Streaming Summary",
                "",
                f"- Jobs passed: `{aggregate.get('passed')}/{aggregate.get('job_count')}`",
                f"- First-frame median: `{aggregate.get('first_frame_latency_ms', {}).get('median', 0.0):.2f} ms`",
                f"- Subsequent median: `{aggregate.get('subsequent_frame_latency_median_ms', {}).get('median', 0.0):.2f} ms`",
                f"- Median E2E FPS: `{aggregate.get('end_to_end_streaming_fps', {}).get('median', 0.0):.2f}`",
                f"- Median model-only FPS: `{aggregate.get('model_only_streaming_fps', {}).get('median', 0.0):.2f}`",
            ]
            if is_hf_edgetam_source
            else []
        ),
        "",
        "## PCD Postprocess",
        "",
        f"- Default mode: `{postprocess_summary.get('mode')}`",
        f"- Controller/hand override: `{postprocess_summary.get('controller_mode') or 'semantic default'}`",
        f"- Controller/hand effective mode: `{manipulator_effective_text}`",
        f"- Radius: `{postprocess_summary.get('phystwin_radius_m')}`",
        f"- Neighbors: `{postprocess_summary.get('phystwin_nb_points')}`",
        f"- Component voxel size: `{postprocess_summary.get('enhanced_component_voxel_size_m')}`",
        f"- Keep-near-main gap: `{postprocess_summary.get('enhanced_keep_near_main_gap_m')}`",
        "",
        "## Objects",
        "",
        "| object id | label | postprocess | mean raw pts | mean output pts | min | max |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for obj_id, item in sorted(pcd_summary.get("aggregate", {}).items(), key=lambda pair: int(pair[0])):
        lines.append(
            f"| {obj_id} | {item.get('label')} | "
            f"{per_object_modes.get(str(obj_id), postprocess_summary.get('mode'))} | "
            f"{item.get('raw_point_count_mean', 0.0):.1f} | "
            f"{item.get('point_count_mean', 0.0):.1f} | "
            f"{item.get('point_count_min', 0)} | "
            f"{item.get('point_count_max', 0)} |"
        )
    labels = {
        _normalize_label(str(item.get("label", "")))
        for item in pcd_summary.get("objects", [])
    }
    if any(_is_manipulator_label(label) for label in labels):
        lines.extend(
            [
                "",
                "## Controller/Hand Warning",
                "",
                "- `controller` follows the PhysTwin-style convention: all hand instances are merged into one controller mask/PCD.",
                "- Object rows use `enhanced-pt` for cleaner presentation when that global mode is selected; controller/hand rows use the simpler `pt-filter` by default.",
                "- If `enhanced-pt` is enabled on controller/hand rows with an explicit override, it can remove sparse fingertips, contact patches, or partial hand points that may matter for manipulation.",
                "- Do not interpret controller output as per-hand identity. Per-hand workflows need an explicit 3D cross-view identity mapping.",
            ]
        )
    if is_hf_edgetam_source:
        lines.extend(
            [
                "",
                "## Jobs",
                "",
                "| cam | frames | first ms | subsequent median ms | p95 ms | e2e FPS | model FPS | failures |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for job in streaming_payload.get("jobs", []):
            if job.get("status") != "pass":
                lines.append(f"| {job.get('camera_idx')} | 0 | n/a | n/a | n/a | n/a | n/a | 1 |")
                continue
            failures = sum(
                len(item.get("failure_frames", []))
                for item in dict(job.get("quality_by_object", {})).values()
            )
            lines.append(
                f"| {job.get('camera_idx')} | {job.get('frame_count')} | "
                f"{float(job.get('first_frame_latency_ms', 0.0)):.2f} | "
                f"{float(job.get('subsequent_frame_latency_ms', {}).get('median', 0.0)):.2f} | "
                f"{float(job.get('subsequent_frame_latency_ms', {}).get('p95', 0.0)):.2f} | "
                f"{float(job.get('end_to_end_streaming_fps', 0.0)):.2f} | "
                f"{float(job.get('model_only_streaming_fps', 0.0)):.2f} | "
                f"{failures} |"
            )
    if aggregate.get("failed"):
        lines.extend(["", "## Failures", ""])
        for job in streaming_payload.get("jobs", []):
            if job.get("status") != "pass":
                lines.append(f"- cam`{job.get('camera_idx')}`: `{job.get('error_type')}: {job.get('error')}`")
    lines.append("")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track Sloth Set 2 hand/object with HF EdgeTAM streaming and render a fused PCD GIF."
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--case-dir", type=Path, default=DEFAULT_CASE_DIR)
    parser.add_argument("--sam31-mask-root", type=Path, default=DEFAULT_SAM31_MASK_ROOT)
    parser.add_argument(
        "--render-mask-root",
        type=Path,
        default=None,
        help=(
            "Optional mask root for render-only panels. When omitted, renders "
            "from the HF EdgeTAM multi-object mask root under --output-root."
        ),
    )
    parser.add_argument("--mask-source-label", default="HF EdgeTAM")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--object-label", default="stuffed animal")
    parser.add_argument("--hand-label", default="hand")
    parser.add_argument(
        "--track-two-hands",
        action="store_true",
        help="Track three objects: object, left hand, and right hand.",
    )
    parser.add_argument("--left-hand-label", default="left hand")
    parser.add_argument("--right-hand-label", default="right hand")
    parser.add_argument(
        "--build-two-hand-init-root",
        action="store_true",
        help=(
            "Build --sam31-mask-root as a canonical frame-0 init root from an object mask root "
            "and a raw two-hand SAM3.1 instance mask root before running EdgeTAM."
        ),
    )
    parser.add_argument("--object-init-mask-root", type=Path, default=DEFAULT_OBJECT_INIT_MASK_ROOT)
    parser.add_argument("--hand-instance-mask-root", type=Path, default=DEFAULT_HAND_INSTANCE_MASK_ROOT)
    parser.add_argument("--source-hand-label", default="hand")
    parser.add_argument("--init-frame-token", default="0")
    parser.add_argument("--camera-ids", default="0,1,2")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--all-frames", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument(
        "--compile-mode",
        choices=hf_stream.ALL_COMPILE_MODES,
        default=hf_stream.COMPILE_MODE_VISION_REDUCE_OVERHEAD,
    )
    parser.add_argument("--warmup-frames", type=int, default=3)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Skip HF EdgeTAM tracking and render from existing masks under --output-root.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--pcd-output-dir", type=Path, default=None)
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--pcd-postprocess-mode", choices=PCD_POSTPROCESS_MODES, default=PCD_POSTPROCESS_NONE)
    parser.add_argument(
        "--controller-pcd-postprocess-mode",
        choices=PCD_POSTPROCESS_MODES,
        default=None,
        help="Optional per-row override for rows labeled controller/hand. By default, enhanced object runs keep these rows on pt-filter.",
    )
    parser.add_argument("--phystwin-radius-m", type=float, default=DEFAULT_PHYSTWIN_RADIUS_M)
    parser.add_argument("--phystwin-nb-points", type=int, default=DEFAULT_PHYSTWIN_NB_POINTS)
    parser.add_argument("--enhanced-component-voxel-size-m", type=float, default=DEFAULT_ENHANCED_COMPONENT_VOXEL_SIZE_M)
    parser.add_argument("--enhanced-keep-near-main-gap-m", type=float, default=DEFAULT_ENHANCED_KEEP_NEAR_MAIN_GAP_M)
    parser.add_argument("--tile-width", type=int, default=320)
    parser.add_argument("--tile-height", type=int, default=240)
    parser.add_argument("--row-label-width", type=int, default=112)
    parser.add_argument("--gif-fps", type=int, default=12)
    parser.add_argument("--max-points-per-camera", type=int, default=None)
    parser.add_argument("--max-points-per-render", type=int, default=180000)
    parser.add_argument("--depth-min-m", type=float, default=0.20)
    parser.add_argument("--depth-max-m", type=float, default=1.50)
    parser.add_argument("--depth-scale-override-m-per-unit", type=float, default=0.001)
    parser.add_argument("--point-radius-px", type=int, default=1)
    parser.add_argument("--streaming-json", type=Path, default=DEFAULT_STREAMING_JSON)
    parser.add_argument("--doc-json", type=Path, default=DEFAULT_DOC_JSON)
    parser.add_argument("--doc-md", type=Path, default=DEFAULT_DOC_MD)
    args = parser.parse_args(argv)
    if args.all_frames:
        args.frames = None
    return args


def run(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    objects = (
        tracked_two_hand_objects(args.object_label, args.left_hand_label, args.right_hand_label)
        if bool(args.track_two_hands)
        else tracked_objects(args.object_label, args.hand_label)
    )
    output_root = _resolve_path(args.output_root)
    case_dir = _resolve_path(args.case_dir)
    mask_root = (
        _resolve_path(args.render_mask_root)
        if args.render_mask_root is not None
        else output_root / "hf_edgetam_streaming_multi_object/masks" / CASE_KEY
    )
    camera_ids = _parse_camera_ids(args.camera_ids)
    built_init_summary = None

    if bool(args.build_two_hand_init_root):
        if not bool(args.track_two_hands):
            raise ValueError("--build-two-hand-init-root requires --track-two-hands")
        built_init_summary = build_object_two_hands_init_mask_root(
            case_dir=case_dir,
            output_root=_resolve_path(args.sam31_mask_root),
            object_mask_root=_resolve_path(args.object_init_mask_root),
            hand_instance_mask_root=_resolve_path(args.hand_instance_mask_root),
            camera_ids=camera_ids,
            frame_token=str(args.init_frame_token),
            object_label=str(args.object_label),
            left_hand_label=str(args.left_hand_label),
            right_hand_label=str(args.right_hand_label),
            source_hand_label=str(args.source_hand_label),
            overwrite=bool(args.overwrite),
        )

    if bool(args.render_only):
        if args.streaming_json.is_file():
            streaming_payload = json.loads(args.streaming_json.read_text(encoding="utf-8"))
            streaming_payload["streaming_results_path"] = str(args.streaming_json)
        else:
            streaming_payload = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "model_id": str(args.model_id),
                "environment": "render-only",
                "compile_mode": str(args.compile_mode),
                "compile_metadata": {"enabled": False, "applied_targets": [], "torch_compile_mode": None},
                "streaming_results_path": str(args.streaming_json),
                "aggregate": {"job_count": 0, "passed": 0, "failed": 0},
                "jobs": [],
            }
    else:
        streaming_payload = run_multi_object_streaming(args, objects)
    if streaming_payload["aggregate"]["failed"]:
        return streaming_payload, {}

    if args.output_name:
        output_name = str(args.output_name)
    elif bool(args.track_two_hands):
        output_name = (
            DEFAULT_TWO_HAND_ENHANCED_OUTPUT_NAME
            if str(args.pcd_postprocess_mode) == PCD_POSTPROCESS_ENHANCED_PT
            else DEFAULT_TWO_HAND_OUTPUT_NAME
        )
    else:
        output_name = (
            DEFAULT_ENHANCED_OUTPUT_NAME
            if str(args.pcd_postprocess_mode) == PCD_POSTPROCESS_ENHANCED_PT
            else DEFAULT_OUTPUT_NAME
        )
    pcd_output_dir = (
        _resolve_path(args.pcd_output_dir)
        if args.pcd_output_dir is not None
        else output_root / (
            "pcd_gif_enhanced_pt"
            if str(args.pcd_postprocess_mode) == PCD_POSTPROCESS_ENHANCED_PT
            else "pcd_gif"
        )
    )
    pcd_summary = render_hand_object_pcd_panel(
        case_dir=case_dir,
        mask_root=mask_root,
        mask_source_label=str(args.mask_source_label),
        output_dir=pcd_output_dir,
        output_name=output_name,
        objects=objects,
        camera_ids=camera_ids,
        frames=args.frames,
        tile_width=int(args.tile_width),
        tile_height=int(args.tile_height),
        row_label_width=int(args.row_label_width),
        gif_fps=int(args.gif_fps),
        max_points_per_camera=args.max_points_per_camera,
        max_points_per_render=args.max_points_per_render,
        depth_min_m=float(args.depth_min_m),
        depth_max_m=float(args.depth_max_m),
        depth_scale_override_m_per_unit=float(args.depth_scale_override_m_per_unit),
        point_radius_px=int(args.point_radius_px),
        pcd_postprocess_mode=str(args.pcd_postprocess_mode),
        controller_pcd_postprocess_mode=args.controller_pcd_postprocess_mode,
        phystwin_radius_m=float(args.phystwin_radius_m),
        phystwin_nb_points=int(args.phystwin_nb_points),
        enhanced_component_voxel_size_m=float(args.enhanced_component_voxel_size_m),
        enhanced_keep_near_main_gap_m=float(args.enhanced_keep_near_main_gap_m),
    )
    sam31_summary_path = _resolve_path(args.sam31_mask_root) / "summary.json"
    if built_init_summary is not None:
        pcd_summary["sam31_mask_summary"] = built_init_summary
    elif sam31_summary_path.is_file():
        pcd_summary["sam31_mask_summary"] = json.loads(sam31_summary_path.read_text(encoding="utf-8"))
    return streaming_payload, pcd_summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not bool(args.render_only):
        hf_stream._load_runtime_dependencies()
    streaming_payload, pcd_summary = run(args)

    if not bool(args.render_only):
        args.streaming_json.parent.mkdir(parents=True, exist_ok=True)
        streaming_payload["streaming_results_path"] = str(args.streaming_json)
        args.streaming_json.write_text(json.dumps(streaming_payload, indent=2), encoding="utf-8")
        print(f"[hf-edgetam-hand-object] wrote streaming JSON: {args.streaming_json}", flush=True)

    if streaming_payload["aggregate"]["failed"]:
        return 1

    combined_payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "streaming": streaming_payload,
        "pcd_panel": pcd_summary,
    }
    write_json(args.doc_json, combined_payload)
    print(f"[hf-edgetam-hand-object] wrote PCD JSON: {args.doc_json}", flush=True)

    write_report(args.doc_md, pcd_summary=pcd_summary, streaming_payload=streaming_payload)
    print(f"[hf-edgetam-hand-object] wrote markdown: {args.doc_md}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
