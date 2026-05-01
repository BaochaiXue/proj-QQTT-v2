from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import imageio.v2 as imageio
import numpy as np

from ..io_artifacts import write_image, write_json, write_ply_ascii
from ..io_case import get_frame_count, load_case_frame_camera_clouds, load_case_metadata
from ..workflows.masked_camera_view_compare import _image_size_from_color_path, _scale_intrinsic_matrix
from ..workflows.masked_pointcloud_compare import (
    filter_camera_clouds_with_pixel_masks,
    parse_text_prompts,
)
from .enhanced_phystwin_postprocess_pcd_compare import (
    DEFAULT_ENHANCED_COMPONENT_VOXEL_SIZE_M,
    DEFAULT_ENHANCED_KEEP_NEAR_MAIN_GAP_M,
)
from .ffs_confidence_filter_pcd_compare import _apply_enhanced_phystwin_like_postprocess
from .native_ffs_fused_pcd_compare import DEFAULT_PHYSTWIN_NB_POINTS, DEFAULT_PHYSTWIN_RADIUS_M


DEFAULT_OUTPUT_DIR = (
    "data/experiments/"
    "sam21_checkpoint_ladder_3x5_time_gifs_ffs203048_iter4_trt_level5"
)
DEFAULT_SAM2_CHECKPOINT_CACHE = Path.home() / ".cache" / "huggingface" / "sam2.1"
DEFAULT_SAM2_BASE_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824"
DEFAULT_DOC_BENCHMARK_MD = Path("docs/generated/sam21_max_round2_benchmark.md")
DEFAULT_DOC_BENCHMARK_JSON = Path("docs/generated/sam21_max_round2_benchmark_results.json")
DEFAULT_DOC_QUALITY_JSON = Path("docs/generated/sam21_max_round2_mask_quality.json")
DEFAULT_CAMERA_IDS: tuple[int, ...] = (0, 1, 2)
DEFAULT_VARIANT_ORDER: tuple[str, ...] = ("sam31", "large", "base_plus", "small", "tiny")
SAM21_OBJECT_ID = 0
DEFAULT_STABLE_WARMUP_RUNS = 5


@dataclass(frozen=True)
class LadderCaseSpec:
    key: str
    label: str
    output_name: str
    case_dir: Path
    text_prompt: str


@dataclass(frozen=True)
class LadderCheckpointSpec:
    key: str
    label: str
    filename: str
    config: str
    url: str

    def checkpoint_path(self, cache_dir: Path) -> Path:
        return Path(cache_dir).expanduser().resolve() / self.filename


def default_ladder_case_specs(*, root: Path) -> list[LadderCaseSpec]:
    repo_root = Path(root).resolve()
    return [
        LadderCaseSpec(
            key="still_object_round1",
            label="Still Object R1",
            output_name="still_object_round1_3x5_time",
            case_dir=repo_root / "data/still_object/ffs203048_iter4_trt_level5/both_30_still_object_round1_20260428",
            text_prompt="stuffed animal",
        ),
        LadderCaseSpec(
            key="still_object_round2",
            label="Still Object R2",
            output_name="still_object_round2_3x5_time",
            case_dir=repo_root / "data/still_object/ffs203048_iter4_trt_level5/both_30_still_object_round2_20260428",
            text_prompt="stuffed animal",
        ),
        LadderCaseSpec(
            key="still_object_round3",
            label="Still Object R3",
            output_name="still_object_round3_3x5_time",
            case_dir=repo_root / "data/still_object/ffs203048_iter4_trt_level5/both_30_still_object_round3_20260428",
            text_prompt="stuffed animal",
        ),
        LadderCaseSpec(
            key="still_object_round4",
            label="Still Object R4",
            output_name="still_object_round4_3x5_time",
            case_dir=repo_root / "data/still_object/ffs203048_iter4_trt_level5/both_30_still_object_round4_20260428",
            text_prompt="stuffed animal",
        ),
        LadderCaseSpec(
            key="still_rope_round1",
            label="Still Rope R1",
            output_name="still_rope_round1_3x5_time",
            case_dir=repo_root / "data/still_rope/ffs203048_iter4_trt_level5/both_30_still_rope_round1_20260428",
            text_prompt="white twisted rope on the blue box, white thick twisted rope on top of the blue box",
        ),
        LadderCaseSpec(
            key="still_rope_round2",
            label="Still Rope R2",
            output_name="still_rope_round2_3x5_time",
            case_dir=repo_root / "data/still_rope/ffs203048_iter4_trt_level5/both_30_still_rope_round2_20260428",
            text_prompt="white twisted rope lying on the wooden table",
        ),
    ]


def default_ladder_checkpoint_specs() -> list[LadderCheckpointSpec]:
    base = DEFAULT_SAM2_BASE_URL
    return [
        LadderCheckpointSpec(
            key="large",
            label="SAM2.1 Large",
            filename="sam2.1_hiera_large.pt",
            config="configs/sam2.1/sam2.1_hiera_l.yaml",
            url=f"{base}/sam2.1_hiera_large.pt",
        ),
        LadderCheckpointSpec(
            key="base_plus",
            label="SAM2.1 Base+",
            filename="sam2.1_hiera_base_plus.pt",
            config="configs/sam2.1/sam2.1_hiera_b+.yaml",
            url=f"{base}/sam2.1_hiera_base_plus.pt",
        ),
        LadderCheckpointSpec(
            key="small",
            label="SAM2.1 Small",
            filename="sam2.1_hiera_small.pt",
            config="configs/sam2.1/sam2.1_hiera_s.yaml",
            url=f"{base}/sam2.1_hiera_small.pt",
        ),
        LadderCheckpointSpec(
            key="tiny",
            label="SAM2.1 Tiny",
            filename="sam2.1_hiera_tiny.pt",
            config="configs/sam2.1/sam2.1_hiera_t.yaml",
            url=f"{base}/sam2.1_hiera_tiny.pt",
        ),
    ]


def case_to_json(spec: LadderCaseSpec) -> dict[str, Any]:
    return {
        "key": spec.key,
        "label": spec.label,
        "output_name": spec.output_name,
        "case_dir": str(spec.case_dir),
        "text_prompt": spec.text_prompt,
    }


def checkpoint_to_json(spec: LadderCheckpointSpec, *, checkpoint_cache: Path) -> dict[str, Any]:
    return {
        "key": spec.key,
        "label": spec.label,
        "filename": spec.filename,
        "checkpoint": str(spec.checkpoint_path(checkpoint_cache)),
        "config": spec.config,
        "url": spec.url,
    }


def ensure_sam21_checkpoints(
    specs: Sequence[LadderCheckpointSpec],
    *,
    checkpoint_cache: Path,
    download_missing: bool,
) -> list[dict[str, Any]]:
    checkpoint_cache = Path(checkpoint_cache).expanduser().resolve()
    checkpoint_cache.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for spec in specs:
        path = spec.checkpoint_path(checkpoint_cache)
        downloaded = False
        if not path.is_file():
            if not download_missing:
                raise FileNotFoundError(f"Missing SAM2.1 checkpoint: {path}")
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            if tmp_path.exists():
                tmp_path.unlink()
            print(f"[sam21] downloading {spec.filename}", flush=True)
            urllib.request.urlretrieve(spec.url, tmp_path)
            tmp_path.replace(path)
            downloaded = True
        records.append(
            {
                **checkpoint_to_json(spec, checkpoint_cache=checkpoint_cache),
                "exists": path.is_file(),
                "size_bytes": int(path.stat().st_size) if path.is_file() else 0,
                "downloaded": downloaded,
            }
        )
    return records


def _resolve_mask_root(mask_root: str | Path) -> Path:
    root = Path(mask_root).resolve()
    if root.name == "mask":
        root = root.parent
    if not (root / "mask").is_dir():
        raise FileNotFoundError(f"Mask root does not contain mask/: {root}")
    return root


def load_mask_info(mask_root: str | Path, *, camera_idx: int) -> dict[int, str]:
    root = _resolve_mask_root(mask_root)
    info_path = root / "mask" / f"mask_info_{int(camera_idx)}.json"
    if not info_path.is_file():
        return {}
    payload = json.loads(info_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Mask info must be a dict: {info_path}")
    return {int(key): str(value) for key, value in payload.items()}


def _mask_shape_from_color(case_dir: Path, *, camera_idx: int, frame_token: str) -> tuple[int, int]:
    image_path = Path(case_dir) / "color" / str(int(camera_idx)) / f"{frame_token}.png"
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Missing RGB frame for mask shape: {image_path}")
    return int(image.shape[0]), int(image.shape[1])


def load_union_mask(
    *,
    mask_root: str | Path,
    case_dir: str | Path,
    camera_idx: int,
    frame_token: str,
    text_prompt: str,
) -> np.ndarray:
    root = _resolve_mask_root(mask_root)
    prompts = {" ".join(item.strip().lower().split()) for item in parse_text_prompts(text_prompt)}
    mask_info = load_mask_info(root, camera_idx=int(camera_idx))
    height, width = _mask_shape_from_color(Path(case_dir), camera_idx=int(camera_idx), frame_token=str(frame_token))
    union_mask = np.zeros((height, width), dtype=bool)
    matched_object_ids = [
        int(obj_id)
        for obj_id, label in mask_info.items()
        if " ".join(str(label).strip().lower().split()) in prompts
    ]
    if not matched_object_ids:
        raise RuntimeError(
            f"No mask objects match prompt {text_prompt!r} for camera {camera_idx} in {root}"
        )
    for obj_id in matched_object_ids:
        path = root / "mask" / str(int(camera_idx)) / str(int(obj_id)) / f"{frame_token}.png"
        if not path.is_file():
            continue
        mask_image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask_image is None:
            raise RuntimeError(f"Failed to read mask image: {path}")
        if mask_image.shape[:2] != union_mask.shape:
            mask_image = cv2.resize(mask_image, (width, height), interpolation=cv2.INTER_NEAREST)
        union_mask |= mask_image > 0
    return union_mask


def matched_mask_labels(
    *,
    mask_root: str | Path,
    camera_idx: int,
    text_prompt: str,
) -> list[str]:
    prompts = {" ".join(item.strip().lower().split()) for item in parse_text_prompts(text_prompt)}
    labels = []
    for _obj_id, label in sorted(load_mask_info(mask_root, camera_idx=int(camera_idx)).items()):
        normalized = " ".join(str(label).strip().lower().split())
        if normalized in prompts:
            labels.append(str(label))
    return labels


def bbox_xyxy_from_mask(mask: np.ndarray, *, padding_px: int = 0, min_size_px: int = 2) -> list[float]:
    mask_array = np.asarray(mask, dtype=bool)
    if mask_array.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {mask_array.shape}")
    ys, xs = np.where(mask_array)
    if len(xs) == 0:
        raise ValueError("Cannot derive a bbox from an empty mask.")
    height, width = mask_array.shape
    x0 = int(max(0, xs.min() - int(padding_px)))
    y0 = int(max(0, ys.min() - int(padding_px)))
    x1 = int(min(width - 1, xs.max() + int(padding_px)))
    y1 = int(min(height - 1, ys.max() + int(padding_px)))
    if x1 - x0 + 1 < int(min_size_px):
        grow = int(math.ceil((int(min_size_px) - (x1 - x0 + 1)) / 2.0))
        x0 = max(0, x0 - grow)
        x1 = min(width - 1, x1 + grow)
    if y1 - y0 + 1 < int(min_size_px):
        grow = int(math.ceil((int(min_size_px) - (y1 - y0 + 1)) / 2.0))
        y0 = max(0, y0 - grow)
        y1 = min(height - 1, y1 + grow)
    return [float(x0), float(y0), float(x1), float(y1)]


def write_single_object_masks(
    *,
    mask_root: str | Path,
    camera_idx: int,
    object_label: str,
    masks_by_frame_token: dict[str, np.ndarray],
    overwrite: bool,
) -> dict[str, Any]:
    root = Path(mask_root).resolve()
    mask_dir = root / "mask"
    camera_dir = mask_dir / str(int(camera_idx))
    info_path = mask_dir / f"mask_info_{int(camera_idx)}.json"
    if (camera_dir.exists() or info_path.exists()) and bool(overwrite):
        shutil.rmtree(camera_dir, ignore_errors=True)
        if info_path.exists():
            info_path.unlink()
    if camera_dir.exists() or info_path.exists():
        raise FileExistsError(f"SAM2 mask output already exists for camera {camera_idx}: {root}")
    object_dir = camera_dir / str(SAM21_OBJECT_ID)
    object_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    info_path.write_text(json.dumps({str(SAM21_OBJECT_ID): str(object_label)}, indent=2), encoding="utf-8")
    saved = 0
    for frame_token, mask in sorted(masks_by_frame_token.items(), key=lambda item: int(item[0])):
        mask_image = np.asarray(mask, dtype=np.uint8) * 255
        cv2.imwrite(str(object_dir / f"{frame_token}.png"), mask_image)
        saved += 1
    return {
        "mask_root": str(root),
        "camera_idx": int(camera_idx),
        "object_id": int(SAM21_OBJECT_ID),
        "object_label": str(object_label),
        "saved_frame_count": int(saved),
    }


def sorted_case_frame_tokens(case_dir: str | Path, *, camera_idx: int = 0, frames: int | None = None) -> list[str]:
    color_dir = Path(case_dir) / "color" / str(int(camera_idx))
    paths = sorted(color_dir.glob("*.png"), key=lambda path: int(path.stem))
    if frames is not None:
        paths = paths[: int(frames)]
    if not paths:
        raise FileNotFoundError(f"No PNG frames found under {color_dir}")
    return [path.stem for path in paths]


def prepare_jpeg_video_dir(
    *,
    case_dir: str | Path,
    camera_idx: int,
    frame_tokens: Sequence[str],
    work_dir: str | Path,
) -> Path:
    out_dir = Path(work_dir).resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for session_idx, frame_token in enumerate(frame_tokens):
        src = Path(case_dir) / "color" / str(int(camera_idx)) / f"{frame_token}.png"
        image = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Missing RGB frame: {src}")
        cv2.imwrite(str(out_dir / f"{session_idx:05d}.jpg"), image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return out_dir


def _cuda_sync(torch_module: Any) -> None:
    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()


def _mark_compile_step(torch_module: Any) -> None:
    compiler = getattr(torch_module, "compiler", None)
    marker = None if compiler is None else getattr(compiler, "cudagraph_mark_step_begin", None)
    if marker is not None:
        marker()


def _configure_torch_for_sam21(torch_module: Any) -> None:
    if not torch_module.cuda.is_available():
        raise RuntimeError("SAM2.1 video predictor requires CUDA for this benchmark.")
    props = torch_module.cuda.get_device_properties(0)
    if int(props.major) >= 8:
        torch_module.backends.cuda.matmul.allow_tf32 = True
        torch_module.backends.cudnn.allow_tf32 = True


def _time_ms(torch_module: Any, fn: Any) -> tuple[Any, float]:
    _cuda_sync(torch_module)
    start = time.perf_counter()
    value = fn()
    _cuda_sync(torch_module)
    return value, float((time.perf_counter() - start) * 1000.0)


def run_sam21_worker(
    *,
    case_key: str,
    case_dir: str | Path,
    text_prompt: str,
    camera_idx: int,
    checkpoint_key: str,
    checkpoint_label: str,
    checkpoint_path: str | Path,
    config: str,
    output_mask_root: str | Path,
    result_json: str | Path,
    frames: int,
    bbox_padding_px: int = 0,
    overwrite: bool = False,
) -> dict[str, Any]:
    import torch
    from sam2.build_sam import build_sam2_video_predictor

    _configure_torch_for_sam21(torch)
    case_dir = Path(case_dir).resolve()
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing SAM2.1 checkpoint: {checkpoint_path}")

    frame_tokens = sorted_case_frame_tokens(case_dir, camera_idx=int(camera_idx), frames=int(frames))
    sam31_mask = load_union_mask(
        mask_root=case_dir / "sam31_masks",
        case_dir=case_dir,
        camera_idx=int(camera_idx),
        frame_token=frame_tokens[0],
        text_prompt=text_prompt,
    )
    sam21_object_label = (
        matched_mask_labels(
            mask_root=case_dir / "sam31_masks",
            camera_idx=int(camera_idx),
            text_prompt=text_prompt,
        )
        or parse_text_prompts(text_prompt)
        or [str(text_prompt)]
    )[0]
    box_xyxy = bbox_xyxy_from_mask(sam31_mask, padding_px=int(bbox_padding_px))

    result_path = Path(result_json).resolve()
    result_path.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix="sam21_worker_", dir=str(result_path.parent)))
    video_dir = temp_root / "video_jpg"
    prepare_jpeg_video_dir(
        case_dir=case_dir,
        camera_idx=int(camera_idx),
        frame_tokens=frame_tokens,
        work_dir=video_dir,
    )

    predictor = None
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            predictor = build_sam2_video_predictor(
                str(config),
                str(checkpoint_path),
                device="cuda",
                vos_optimized=True,
            )

            def init_state() -> Any:
                _mark_compile_step(torch)
                return predictor.init_state(
                    video_path=str(video_dir),
                    offload_video_to_cpu=False,
                    offload_state_to_cpu=False,
                    async_loading_frames=False,
                )

            def prompt_state(state: Any) -> Any:
                _mark_compile_step(torch)
                return predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=int(SAM21_OBJECT_ID),
                    box=np.asarray(box_xyxy, dtype=np.float32),
                )

            def propagate_state(state: Any, *, collect: bool) -> tuple[int, dict[str, np.ndarray]]:
                outputs: dict[str, np.ndarray] = {}
                iterator = predictor.propagate_in_video(
                    state,
                    start_frame_idx=0,
                    max_frame_num_to_track=len(frame_tokens),
                    reverse=False,
                )
                output_count = 0
                while True:
                    _mark_compile_step(torch)
                    try:
                        out_frame_idx, _out_obj_ids, out_mask_logits = next(iterator)
                    except StopIteration:
                        break
                    output_count += 1
                    if collect:
                        token = frame_tokens[int(out_frame_idx)]
                        mask_tensor = (out_mask_logits[0] > 0).detach()
                        outputs[str(token)] = mask_tensor.squeeze().cpu().numpy().astype(bool)
                _cuda_sync(torch)
                return output_count, outputs

            warm_state, warm_init_ms = _time_ms(torch, init_state)
            _prompt_response, warm_prompt_ms = _time_ms(torch, lambda: prompt_state(warm_state))
            (warmup_frame_count, _warm_masks), warmup_propagate_ms = _time_ms(
                torch,
                lambda: propagate_state(warm_state, collect=False),
            )
            del warm_state
            torch.cuda.empty_cache()

            timed_state, init_state_ms = _time_ms(torch, init_state)
            prompt_response, prompt_ms = _time_ms(torch, lambda: prompt_state(timed_state))
            (timed_frame_count, masks_by_frame), timed_propagate_ms = _time_ms(
                torch,
                lambda: propagate_state(timed_state, collect=True),
            )
            del timed_state
            torch.cuda.empty_cache()

        if len(masks_by_frame) != len(frame_tokens):
            raise RuntimeError(
                f"SAM2.1 worker saved {len(masks_by_frame)} masks, expected {len(frame_tokens)}"
            )
        write_summary = write_single_object_masks(
            mask_root=output_mask_root,
            camera_idx=int(camera_idx),
            object_label=str(sam21_object_label),
            masks_by_frame_token=masks_by_frame,
            overwrite=bool(overwrite),
        )
        try:
            prompt_obj_ids = [int(item) for item in prompt_response[1]]
        except Exception:
            prompt_obj_ids = []
        result = {
            "case_key": str(case_key),
            "case_dir": str(case_dir),
            "text_prompt": str(text_prompt),
            "sam21_object_label": str(sam21_object_label),
            "camera_idx": int(camera_idx),
            "checkpoint_key": str(checkpoint_key),
            "checkpoint_label": str(checkpoint_label),
            "checkpoint_path": str(checkpoint_path),
            "config": str(config),
            "bbox_source": "sam31_frame0_union_mask",
            "bbox_xyxy": [float(item) for item in box_xyxy],
            "frames_requested": int(len(frame_tokens)),
            "frame_tokens": [str(item) for item in frame_tokens],
            "warmup_frame_count": int(warmup_frame_count),
            "timed_frame_count": int(timed_frame_count),
            "warmup_init_state_ms": float(warm_init_ms),
            "warmup_prompt_ms": float(warm_prompt_ms),
            "warmup_propagate_ms": float(warmup_propagate_ms),
            "init_state_ms": float(init_state_ms),
            "prompt_ms": float(prompt_ms),
            "timed_propagate_ms": float(timed_propagate_ms),
            "inference_ms_per_frame": float(timed_propagate_ms / max(1, timed_frame_count)),
            "fps": float(1000.0 * timed_frame_count / max(1e-9, timed_propagate_ms)),
            "prompt_obj_ids": prompt_obj_ids,
            "mask_output": write_summary,
            "process_guard": {
                "mode": "single_case_single_view_checkpoint_worker",
                "pid": int(os.getpid()),
                "python": sys.executable,
                "cuda_device_name": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "vos_optimized_requested": True,
                "bfloat16_autocast": True,
            },
        }
        write_json(result_path, result)
        return result
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
        if predictor is not None and hasattr(predictor, "reset_state"):
            pass


def build_stable_job_manifest(
    *,
    case_specs: Sequence[LadderCaseSpec],
    checkpoint_spec: LadderCheckpointSpec,
    checkpoint_cache: str | Path,
    output_dir: str | Path,
    frames: int,
    camera_ids: Sequence[int] = DEFAULT_CAMERA_IDS,
    bbox_padding_px: int = 0,
) -> dict[str, Any]:
    return {
        "checkpoint_key": checkpoint_spec.key,
        "checkpoint_label": checkpoint_spec.label,
        "checkpoint_path": str(checkpoint_spec.checkpoint_path(Path(checkpoint_cache))),
        "config": checkpoint_spec.config,
        "frames": int(frames),
        "bbox_padding_px": int(bbox_padding_px),
        "camera_ids": [int(item) for item in camera_ids],
        "jobs": [
            {
                "case_key": case_spec.key,
                "case_label": case_spec.label,
                "case_dir": str(case_spec.case_dir),
                "text_prompt": case_spec.text_prompt,
                "camera_idx": int(camera_idx),
                "output_mask_root": str(Path(output_dir).resolve() / "masks" / case_spec.key / checkpoint_spec.key),
            }
            for case_spec in case_specs
            for camera_idx in [int(item) for item in camera_ids]
        ],
    }


def _prepare_stable_job(
    *,
    job: dict[str, Any],
    frames: int,
    bbox_padding_px: int,
    temp_root: Path,
) -> dict[str, Any]:
    case_dir = Path(job["case_dir"]).resolve()
    camera_idx = int(job["camera_idx"])
    text_prompt = str(job["text_prompt"])
    frame_tokens = sorted_case_frame_tokens(case_dir, camera_idx=camera_idx, frames=int(frames))
    sam31_mask = load_union_mask(
        mask_root=case_dir / "sam31_masks",
        case_dir=case_dir,
        camera_idx=camera_idx,
        frame_token=frame_tokens[0],
        text_prompt=text_prompt,
    )
    object_label = (
        matched_mask_labels(
            mask_root=case_dir / "sam31_masks",
            camera_idx=camera_idx,
            text_prompt=text_prompt,
        )
        or parse_text_prompts(text_prompt)
        or [text_prompt]
    )[0]
    box_xyxy = bbox_xyxy_from_mask(sam31_mask, padding_px=int(bbox_padding_px))
    video_dir = temp_root / "video_jpg" / str(job["case_key"]) / f"cam{camera_idx}"
    prepare_jpeg_video_dir(
        case_dir=case_dir,
        camera_idx=camera_idx,
        frame_tokens=frame_tokens,
        work_dir=video_dir,
    )
    return {
        **job,
        "case_dir": str(case_dir),
        "camera_idx": camera_idx,
        "frame_tokens": [str(item) for item in frame_tokens],
        "sam21_object_label": str(object_label),
        "bbox_source": "sam31_frame0_union_mask",
        "bbox_xyxy": [float(item) for item in box_xyxy],
        "video_dir": str(video_dir),
    }


def run_sam21_stable_worker(
    *,
    checkpoint_key: str,
    checkpoint_label: str,
    checkpoint_path: str | Path,
    config: str,
    job_manifest: str | Path,
    result_json: str | Path,
    warmup_runs: int = DEFAULT_STABLE_WARMUP_RUNS,
    speed_use_step_marker: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    import torch
    from sam2.build_sam import build_sam2_video_predictor

    _configure_torch_for_sam21(torch)
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing SAM2.1 checkpoint: {checkpoint_path}")
    manifest = json.loads(Path(job_manifest).read_text(encoding="utf-8"))
    frames = int(manifest.get("frames", 30))
    bbox_padding_px = int(manifest.get("bbox_padding_px", 0))
    raw_jobs = list(manifest.get("jobs", []))
    if not raw_jobs:
        raise ValueError(f"Stable SAM2.1 worker manifest has no jobs: {job_manifest}")

    result_path = Path(result_json).resolve()
    result_path.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix=f"sam21_stable_{checkpoint_key}_", dir=str(result_path.parent)))
    predictor = None
    try:
        prepared_jobs = [
            _prepare_stable_job(
                job=dict(job),
                frames=frames,
                bbox_padding_px=bbox_padding_px,
                temp_root=temp_root,
            )
            for job in raw_jobs
        ]
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            predictor = build_sam2_video_predictor(
                str(config),
                str(checkpoint_path),
                device="cuda",
                vos_optimized=True,
            )

            def init_state(job: dict[str, Any]) -> Any:
                return predictor.init_state(
                    video_path=str(job["video_dir"]),
                    offload_video_to_cpu=False,
                    offload_state_to_cpu=False,
                    async_loading_frames=False,
                )

            def prompt_state(state: Any, job: dict[str, Any]) -> Any:
                return predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=int(SAM21_OBJECT_ID),
                    box=np.asarray(job["bbox_xyxy"], dtype=np.float32),
                )

            def propagate_state(
                state: Any,
                job: dict[str, Any],
                *,
                collect: bool,
                use_step_marker: bool,
            ) -> tuple[int, dict[str, np.ndarray]]:
                frame_tokens = [str(item) for item in job["frame_tokens"]]
                outputs: dict[str, np.ndarray] = {}
                iterator = predictor.propagate_in_video(
                    state,
                    start_frame_idx=0,
                    max_frame_num_to_track=len(frame_tokens),
                    reverse=False,
                )
                output_count = 0
                while True:
                    if use_step_marker:
                        _mark_compile_step(torch)
                    try:
                        out_frame_idx, _out_obj_ids, out_mask_logits = next(iterator)
                    except StopIteration:
                        break
                    output_count += 1
                    if collect:
                        token = frame_tokens[int(out_frame_idx)]
                        mask_tensor = (out_mask_logits[0] > 0).detach()
                        outputs[str(token)] = mask_tensor.squeeze().cpu().numpy().astype(bool)
                _cuda_sync(torch)
                return output_count, outputs

            warmup_records: list[dict[str, Any]] = []
            for job in prepared_jobs:
                state, warm_init_ms = _time_ms(torch, lambda job=job: init_state(job))
                _prompt_response, warm_prompt_ms = _time_ms(
                    torch,
                    lambda state=state, job=job: prompt_state(state, job),
                )
                warmup_ms: list[float] = []
                warmup_frame_counts: list[int] = []
                for _warmup_idx in range(max(0, int(warmup_runs))):
                    (frame_count, _outputs), elapsed_ms = _time_ms(
                        torch,
                        lambda state=state, job=job: propagate_state(
                            state,
                            job,
                            collect=False,
                            use_step_marker=bool(speed_use_step_marker),
                        ),
                    )
                    warmup_ms.append(float(elapsed_ms))
                    warmup_frame_counts.append(int(frame_count))
                warmup_records.append(
                    {
                        "case_key": str(job["case_key"]),
                        "camera_idx": int(job["camera_idx"]),
                        "init_state_ms": float(warm_init_ms),
                        "prompt_ms": float(warm_prompt_ms),
                        "warmup_runs": int(warmup_runs),
                        "warmup_propagate_ms": warmup_ms,
                        "warmup_frame_counts": warmup_frame_counts,
                    }
                )
                del state

            timing_records: list[dict[str, Any]] = []
            speed_phase_started = time.perf_counter()
            for job in prepared_jobs:
                state, init_state_ms = _time_ms(torch, lambda job=job: init_state(job))
                prompt_response, prompt_ms = _time_ms(
                    torch,
                    lambda state=state, job=job: prompt_state(state, job),
                )
                (timed_frame_count, _speed_outputs), timed_propagate_ms = _time_ms(
                    torch,
                    lambda state=state, job=job: propagate_state(
                        state,
                        job,
                        collect=False,
                        use_step_marker=bool(speed_use_step_marker),
                    ),
                )
                try:
                    prompt_obj_ids = [int(item) for item in prompt_response[1]]
                except Exception:
                    prompt_obj_ids = []
                record = {
                    "case_key": str(job["case_key"]),
                    "case_dir": str(job["case_dir"]),
                    "text_prompt": str(job["text_prompt"]),
                    "sam21_object_label": str(job["sam21_object_label"]),
                    "camera_idx": int(job["camera_idx"]),
                    "checkpoint_key": str(checkpoint_key),
                    "checkpoint_label": str(checkpoint_label),
                    "checkpoint_path": str(checkpoint_path),
                    "config": str(config),
                    "bbox_source": str(job["bbox_source"]),
                    "bbox_xyxy": [float(item) for item in job["bbox_xyxy"]],
                    "frames_requested": int(len(job["frame_tokens"])),
                    "frame_tokens": [str(item) for item in job["frame_tokens"]],
                    "warmup_runs": int(warmup_runs),
                    "timed_frame_count": int(timed_frame_count),
                    "init_state_ms": float(init_state_ms),
                    "prompt_ms": float(prompt_ms),
                    "timed_propagate_ms": float(timed_propagate_ms),
                    "inference_ms_per_frame": float(timed_propagate_ms / max(1, timed_frame_count)),
                    "fps": float(1000.0 * timed_frame_count / max(1e-9, timed_propagate_ms)),
                    "timing_contract": (
                        "stable_throughput_no_output_"
                        f"{'marker' if speed_use_step_marker else 'no_marker'}_after_"
                        f"{int(warmup_runs)}_warmup_propagations_per_job"
                    ),
                    "prompt_obj_ids": prompt_obj_ids,
                    "process_guard": {
                        "mode": "single_checkpoint_stable_throughput_worker",
                        "pid": int(os.getpid()),
                        "python": sys.executable,
                        "cuda_device_name": torch.cuda.get_device_name(0),
                        "torch": torch.__version__,
                        "torch_cuda": torch.version.cuda,
                        "vos_optimized_requested": True,
                        "bfloat16_autocast": True,
                        "speed_pass_collects_masks": False,
                        "speed_pass_uses_cudagraph_step_marker": bool(speed_use_step_marker),
                    },
                }
                timing_records.append(record)
                del state
            _cuda_sync(torch)
            speed_phase_wall_ms = float((time.perf_counter() - speed_phase_started) * 1000.0)

            for record, job in zip(timing_records, prepared_jobs, strict=True):
                state, _collect_init_ms = _time_ms(torch, lambda job=job: init_state(job))
                _collect_prompt_response, _collect_prompt_ms = _time_ms(
                    torch,
                    lambda state=state, job=job: prompt_state(state, job),
                )
                (collect_frame_count, masks_by_frame), collect_ms = _time_ms(
                    torch,
                    lambda state=state, job=job: propagate_state(
                        state,
                        job,
                        collect=True,
                        use_step_marker=True,
                    ),
                )
                if len(masks_by_frame) != len(job["frame_tokens"]):
                    raise RuntimeError(
                        f"SAM2.1 stable worker saved {len(masks_by_frame)} masks, "
                        f"expected {len(job['frame_tokens'])}"
                    )
                write_summary = write_single_object_masks(
                    mask_root=job["output_mask_root"],
                    camera_idx=int(job["camera_idx"]),
                    object_label=str(job["sam21_object_label"]),
                    masks_by_frame_token=masks_by_frame,
                    overwrite=bool(overwrite),
                )
                record["mask_output"] = write_summary
                record["mask_collection_frame_count"] = int(collect_frame_count)
                record["mask_collection_ms"] = float(collect_ms)
                record["mask_collection_uses_cudagraph_step_marker"] = True
                del state
            torch.cuda.empty_cache()

        total_timed_frames = int(sum(int(record["timed_frame_count"]) for record in timing_records))
        total_timed_ms = float(sum(float(record["timed_propagate_ms"]) for record in timing_records))
        summary = {
            "checkpoint_key": str(checkpoint_key),
            "checkpoint_label": str(checkpoint_label),
            "checkpoint_path": str(checkpoint_path),
            "config": str(config),
            "job_count": int(len(timing_records)),
            "warmup_runs_per_job": int(warmup_runs),
            "total_timed_frames": total_timed_frames,
            "total_timed_propagate_ms": total_timed_ms,
            "aggregate_ms_per_frame": float(total_timed_ms / max(1, total_timed_frames)),
            "aggregate_fps": float(1000.0 * total_timed_frames / max(1e-9, total_timed_ms)),
            "speed_phase_wall_ms": speed_phase_wall_ms,
            "speed_phase_wall_fps_including_state_setup": float(
                1000.0 * total_timed_frames / max(1e-9, speed_phase_wall_ms)
            ),
            "timing_contract": (
                "one checkpoint worker, all selected jobs, no-output warmup propagations "
                "per job, then no-output speed propagation; "
                "mask collection is separate and excluded"
            ),
            "speed_pass_uses_cudagraph_step_marker": bool(speed_use_step_marker),
            "warmup_records": warmup_records,
            "timing_records": timing_records,
        }
        write_json(result_path, summary)
        return summary
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
        if predictor is not None and hasattr(predictor, "reset_state"):
            pass


def run_sam21_stable_workers_by_checkpoint(
    *,
    script_path: str | Path,
    case_specs: Sequence[LadderCaseSpec],
    checkpoint_specs: Sequence[LadderCheckpointSpec],
    checkpoint_cache: str | Path,
    output_dir: str | Path,
    frames: int,
    camera_ids: Sequence[int] = DEFAULT_CAMERA_IDS,
    bbox_padding_px: int = 0,
    warmup_runs: int = DEFAULT_STABLE_WARMUP_RUNS,
    speed_use_step_marker: bool = True,
    overwrite: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    output_dir = Path(output_dir).resolve()
    checkpoint_cache = Path(checkpoint_cache).expanduser().resolve()
    timing_dir = output_dir / "timings"
    manifest_dir = output_dir / "stable_manifests"
    summary_dir = output_dir / "stable_checkpoint_summaries"
    worker_log_dir = output_dir / "logs" / "sam21_stable_workers"
    for path in (timing_dir, manifest_dir, summary_dir, worker_log_dir):
        path.mkdir(parents=True, exist_ok=True)
    all_records: list[dict[str, Any]] = []
    checkpoint_summaries: list[dict[str, Any]] = []
    for checkpoint_spec in checkpoint_specs:
        manifest = build_stable_job_manifest(
            case_specs=case_specs,
            checkpoint_spec=checkpoint_spec,
            checkpoint_cache=checkpoint_cache,
            output_dir=output_dir,
            frames=int(frames),
            camera_ids=camera_ids,
            bbox_padding_px=int(bbox_padding_px),
        )
        manifest_path = manifest_dir / f"{checkpoint_spec.key}.json"
        summary_path = summary_dir / f"{checkpoint_spec.key}.json"
        write_json(manifest_path, manifest)
        command = [
            sys.executable,
            str(Path(script_path).resolve()),
            "--stable-worker",
            "--checkpoint-key",
            checkpoint_spec.key,
            "--checkpoint-label",
            checkpoint_spec.label,
            "--checkpoint",
            str(checkpoint_spec.checkpoint_path(checkpoint_cache)),
            "--config",
            checkpoint_spec.config,
            "--job-manifest",
            str(manifest_path),
            "--result-json",
            str(summary_path),
            "--stable-warmup-runs",
            str(int(warmup_runs)),
        ]
        if not speed_use_step_marker:
            command.append("--stable-no-speed-step-marker")
        if overwrite:
            command.append("--overwrite")
        print(
            f"[sam21] stable worker checkpoint={checkpoint_spec.key} jobs={len(manifest['jobs'])}",
            flush=True,
        )
        log_path = worker_log_dir / f"{checkpoint_spec.key}.log"
        env = os.environ.copy()
        env.setdefault("TQDM_DISABLE", "1")
        with log_path.open("w", encoding="utf-8") as log_handle:
            try:
                subprocess.run(
                    command,
                    check=True,
                    cwd=Path(__file__).resolve().parents[3],
                    env=env,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                )
            except subprocess.CalledProcessError:
                log_handle.flush()
                tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:])
                print(f"[sam21] stable worker failed; log tail from {log_path}:\n{tail}", flush=True)
                raise
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["worker_log_path"] = str(log_path)
        write_json(summary_path, summary)
        checkpoint_summaries.append(summary)
        for record in summary.get("timing_records", []):
            record["worker_log_path"] = str(log_path)
            result_json = timing_dir / f"{record['case_key']}_cam{int(record['camera_idx'])}_{checkpoint_spec.key}.json"
            write_json(result_json, record)
            all_records.append(record)
    return all_records, checkpoint_summaries


def run_sam21_workers_sequentially(
    *,
    script_path: str | Path,
    case_specs: Sequence[LadderCaseSpec],
    checkpoint_specs: Sequence[LadderCheckpointSpec],
    checkpoint_cache: str | Path,
    output_dir: str | Path,
    frames: int,
    camera_ids: Sequence[int] = DEFAULT_CAMERA_IDS,
    bbox_padding_px: int = 0,
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    output_dir = Path(output_dir).resolve()
    checkpoint_cache = Path(checkpoint_cache).expanduser().resolve()
    timing_dir = output_dir / "timings"
    worker_log_dir = output_dir / "logs" / "sam21_workers"
    worker_log_dir.mkdir(parents=True, exist_ok=True)
    result_records: list[dict[str, Any]] = []
    for case_spec in case_specs:
        for camera_idx in [int(item) for item in camera_ids]:
            for checkpoint_spec in checkpoint_specs:
                mask_root = output_dir / "masks" / case_spec.key / checkpoint_spec.key
                result_json = timing_dir / f"{case_spec.key}_cam{camera_idx}_{checkpoint_spec.key}.json"
                command = [
                    sys.executable,
                    str(Path(script_path).resolve()),
                    "--worker",
                    "--case-key",
                    case_spec.key,
                    "--case-dir",
                    str(case_spec.case_dir),
                    "--text-prompt",
                    case_spec.text_prompt,
                    "--camera-idx",
                    str(camera_idx),
                    "--checkpoint-key",
                    checkpoint_spec.key,
                    "--checkpoint-label",
                    checkpoint_spec.label,
                    "--checkpoint",
                    str(checkpoint_spec.checkpoint_path(checkpoint_cache)),
                    "--config",
                    checkpoint_spec.config,
                    "--output-mask-root",
                    str(mask_root),
                    "--result-json",
                    str(result_json),
                    "--frames",
                    str(int(frames)),
                    "--bbox-padding-px",
                    str(int(bbox_padding_px)),
                ]
                if overwrite:
                    command.append("--overwrite")
                print(
                    f"[sam21] worker case={case_spec.key} cam={camera_idx} checkpoint={checkpoint_spec.key}",
                    flush=True,
                )
                log_path = worker_log_dir / f"{case_spec.key}_cam{camera_idx}_{checkpoint_spec.key}.log"
                env = os.environ.copy()
                env.setdefault("TQDM_DISABLE", "1")
                with log_path.open("w", encoding="utf-8") as log_handle:
                    try:
                        subprocess.run(
                            command,
                            check=True,
                            cwd=Path(__file__).resolve().parents[3],
                            env=env,
                            stdout=log_handle,
                            stderr=subprocess.STDOUT,
                        )
                    except subprocess.CalledProcessError:
                        log_handle.flush()
                        tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:])
                        print(f"[sam21] worker failed; log tail from {log_path}:\n{tail}", flush=True)
                        raise
                record = json.loads(result_json.read_text(encoding="utf-8"))
                record["worker_log_path"] = str(log_path)
                write_json(result_json, record)
                result_records.append(record)
    return result_records


def timing_lookup(timing_records: Sequence[dict[str, Any]]) -> dict[tuple[str, int, str], dict[str, Any]]:
    lookup: dict[tuple[str, int, str], dict[str, Any]] = {}
    for record in timing_records:
        lookup[
            (
                str(record["case_key"]),
                int(record["camera_idx"]),
                str(record["checkpoint_key"]),
            )
        ] = dict(record)
    return lookup


def aggregate_timing_records(timing_records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_checkpoint: dict[str, list[float]] = {}
    by_case_checkpoint: dict[str, dict[str, list[float]]] = {}
    for record in timing_records:
        checkpoint_key = str(record["checkpoint_key"])
        case_key = str(record["case_key"])
        value = float(record["inference_ms_per_frame"])
        by_checkpoint.setdefault(checkpoint_key, []).append(value)
        by_case_checkpoint.setdefault(case_key, {}).setdefault(checkpoint_key, []).append(value)
    checkpoint_summary = {
        key: {
            "mean_inference_ms_per_frame": float(np.mean(values)),
            "mean_fps": float(1000.0 / max(1e-9, float(np.mean(values)))),
            "sample_count": int(len(values)),
        }
        for key, values in sorted(by_checkpoint.items())
    }
    case_summary = {
        case_key: {
            checkpoint_key: {
                "mean_inference_ms_per_frame": float(np.mean(values)),
                "mean_fps": float(1000.0 / max(1e-9, float(np.mean(values)))),
                "sample_count": int(len(values)),
            }
            for checkpoint_key, values in sorted(checkpoint_values.items())
        }
        for case_key, checkpoint_values in sorted(by_case_checkpoint.items())
    }
    return {
        "by_checkpoint": checkpoint_summary,
        "by_case_checkpoint": case_summary,
        "record_count": int(len(timing_records)),
    }


def _format_point_count(point_count: int) -> str:
    count = int(point_count)
    if count >= 1_000_000:
        return f"{count / 1_000_000.0:.1f}M"
    if count >= 1000:
        return f"{count / 1000.0:.1f}k"
    return str(count)


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


def compose_3x5_panel(
    *,
    title_lines: Sequence[str],
    row_headers: Sequence[str],
    column_headers: Sequence[str],
    image_rows: Sequence[Sequence[np.ndarray]],
    row_label_width: int = 92,
) -> np.ndarray:
    if len(row_headers) != 3:
        raise ValueError(f"3x5 panel requires 3 row headers, got {len(row_headers)}")
    if len(column_headers) != 5:
        raise ValueError(f"3x5 panel requires 5 column headers, got {len(column_headers)}")
    if len(image_rows) != 3 or any(len(row) != 5 for row in image_rows):
        raise ValueError("3x5 panel requires exactly 3 rows of 5 images.")
    tile_h, tile_w = image_rows[0][0].shape[:2]
    if any(tile.shape[:2] != (tile_h, tile_w) for row in image_rows for tile in row):
        raise ValueError("All 3x5 panel tiles must have the same shape.")

    title_h = 84
    header_h = 38
    body_h = tile_h * 3
    body_w = int(row_label_width) + tile_w * 5
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


def _deterministic_point_cap(points: np.ndarray, colors: np.ndarray, *, max_points: int | None) -> tuple[np.ndarray, np.ndarray]:
    point_array = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    color_array = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    if max_points is None or len(point_array) <= int(max_points):
        return point_array, color_array
    indices = np.linspace(0, len(point_array) - 1, num=int(max_points), dtype=np.int64)
    return point_array[indices], color_array[indices]


def _build_variant_roots(
    *,
    case_spec: LadderCaseSpec,
    output_dir: Path,
    checkpoint_specs: Sequence[LadderCheckpointSpec],
) -> dict[str, Path]:
    roots = {"sam31": case_spec.case_dir / "sam31_masks"}
    for checkpoint_spec in checkpoint_specs:
        roots[checkpoint_spec.key] = output_dir / "masks" / case_spec.key / checkpoint_spec.key
    return roots


def _variant_label(variant_key: str, checkpoint_specs: Sequence[LadderCheckpointSpec]) -> str:
    if variant_key == "sam31":
        return "SAM3.1"
    for spec in checkpoint_specs:
        if spec.key == variant_key:
            return spec.label.replace("SAM2.1 ", "")
    return variant_key


def _cell_label(
    *,
    case_key: str,
    camera_idx: int,
    variant_key: str,
    point_count: int,
    timing_by_cell: dict[tuple[str, int, str], dict[str, Any]],
) -> str:
    if variant_key == "sam31":
        return f"SAM3.1 | existing mask | {_format_point_count(point_count)} pts"
    timing = timing_by_cell.get((str(case_key), int(camera_idx), str(variant_key)), {})
    ms = float(timing.get("inference_ms_per_frame", 0.0))
    fps = float(timing.get("fps", 0.0))
    return f"{variant_key} | {ms:.1f} ms/f | {fps:.1f} FPS | {_format_point_count(point_count)} pts"


def _fuse_cloud_arrays(camera_clouds: Sequence[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    point_sets = [
        np.asarray(camera_cloud["points"], dtype=np.float32).reshape(-1, 3)
        for camera_cloud in camera_clouds
        if len(camera_cloud["points"]) > 0
    ]
    color_sets = [
        np.asarray(camera_cloud["colors"], dtype=np.uint8).reshape(-1, 3)
        for camera_cloud in camera_clouds
        if len(camera_cloud["points"]) > 0
    ]
    if not point_sets:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
    if len(point_sets) == 1:
        return point_sets[0], color_sets[0]
    return np.concatenate(point_sets, axis=0), np.concatenate(color_sets, axis=0)


def build_frame_cells(
    *,
    case_spec: LadderCaseSpec,
    metadata: dict[str, Any],
    output_dir: Path,
    checkpoint_specs: Sequence[LadderCheckpointSpec],
    frame_idx: int,
    depth_min_m: float,
    depth_max_m: float,
    max_points_per_camera: int | None,
    phystwin_radius_m: float,
    phystwin_nb_points: int,
    enhanced_component_voxel_size_m: float,
    enhanced_keep_near_main_gap_m: float,
) -> tuple[dict[int, dict[str, dict[str, Any]]], dict[str, Any]]:
    variant_roots = _build_variant_roots(case_spec=case_spec, output_dir=output_dir, checkpoint_specs=checkpoint_specs)
    camera_clouds, camera_stats = load_case_frame_camera_clouds(
        case_dir=case_spec.case_dir,
        metadata=metadata,
        frame_idx=int(frame_idx),
        depth_source="ffs",
        use_float_ffs_depth_when_available=True,
        max_points_per_camera=max_points_per_camera,
        depth_min_m=float(depth_min_m),
        depth_max_m=float(depth_max_m),
    )
    cells_by_variant: dict[str, dict[str, Any]] = {}
    for variant_key in DEFAULT_VARIANT_ORDER:
        pixel_mask_by_camera: dict[int, np.ndarray] = {}
        for camera_cloud in camera_clouds:
            camera_idx = int(camera_cloud["camera_idx"])
            mask = load_union_mask(
                mask_root=variant_roots[variant_key],
                case_dir=case_spec.case_dir,
                camera_idx=camera_idx,
                frame_token=str(frame_idx),
                text_prompt=case_spec.text_prompt,
            )
            pixel_mask_by_camera[camera_idx] = mask
        filtered_clouds, mask_metrics = filter_camera_clouds_with_pixel_masks(
            camera_clouds,
            pixel_mask_by_camera=pixel_mask_by_camera,
        )
        raw_points, raw_colors = _fuse_cloud_arrays(filtered_clouds)
        points, colors, postprocess_stats = _apply_enhanced_phystwin_like_postprocess(
            points=raw_points,
            colors=raw_colors,
            enabled=True,
            radius_m=float(phystwin_radius_m),
            nb_points=int(phystwin_nb_points),
            component_voxel_size_m=float(enhanced_component_voxel_size_m),
            keep_near_main_gap_m=float(enhanced_keep_near_main_gap_m),
        )
        cells_by_variant[variant_key] = {
            "points": points,
            "colors": colors,
            "mask_pixel_count": int(sum(np.count_nonzero(mask) for mask in pixel_mask_by_camera.values())),
            "raw_point_count": int(len(raw_points)),
            "point_count": int(len(points)),
            "mask_metrics": mask_metrics,
            "postprocess_stats": postprocess_stats,
        }
    cells = {
        int(camera_cloud["camera_idx"]): {variant_key: dict(cell) for variant_key, cell in cells_by_variant.items()}
        for camera_cloud in camera_clouds
    }
    camera_stats["original_view_source_cameras"] = [
        {
            "camera_idx": int(camera_cloud["camera_idx"]),
            "serial": str(camera_cloud["serial"]),
            "K_color": np.asarray(camera_cloud["K_color"], dtype=np.float32).tolist(),
            "c2w": np.asarray(camera_cloud["c2w"], dtype=np.float32).tolist(),
            "color_path": str(camera_cloud["color_path"]),
        }
        for camera_cloud in camera_clouds
    ]
    return cells, camera_stats


def _build_original_camera_view_specs(
    camera_stats: dict[str, Any],
    *,
    tile_width: int,
    tile_height: int,
) -> dict[int, dict[str, Any]]:
    specs: dict[int, dict[str, Any]] = {}
    for camera in camera_stats.get("original_view_source_cameras", camera_stats.get("per_camera", [])):
        camera_idx = int(camera["camera_idx"])
        source_size = _image_size_from_color_path(camera["color_path"])
        target_size = (int(tile_width), int(tile_height))
        c2w = np.asarray(camera["c2w"], dtype=np.float32).reshape(4, 4)
        specs[camera_idx] = {
            "camera_idx": camera_idx,
            "serial": str(camera.get("serial", camera_idx)),
            "intrinsic_matrix": _scale_intrinsic_matrix(
                np.asarray(camera["K_color"], dtype=np.float32),
                source_size=source_size,
                target_size=target_size,
            ),
            "extrinsic_matrix": np.linalg.inv(c2w).astype(np.float32),
            "image_size": [int(tile_width), int(tile_height)],
            "source_image_size": [int(source_size[0]), int(source_size[1])],
        }
    missing = [camera_idx for camera_idx in DEFAULT_CAMERA_IDS if camera_idx not in specs]
    if missing:
        raise ValueError(f"Missing original camera view specs for camera ids: {missing}")
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


def _render_cell_tile(
    *,
    cell: dict[str, Any],
    label: str,
    view_spec: dict[str, Any],
    tile_width: int,
    tile_height: int,
    max_points_per_render: int | None,
) -> np.ndarray:
    points, colors = _deterministic_point_cap(
        np.asarray(cell["points"], dtype=np.float32),
        np.asarray(cell["colors"], dtype=np.uint8),
        max_points=max_points_per_render,
    )
    rendered = render_pinhole_point_cloud(
        points,
        colors,
        intrinsic_matrix=np.asarray(view_spec["intrinsic_matrix"], dtype=np.float32),
        extrinsic_matrix=np.asarray(view_spec["extrinsic_matrix"], dtype=np.float32),
        width=int(tile_width),
        height=int(tile_height),
        point_radius_px=1,
    )
    return label_tile(rendered, label=label, tile_width=int(tile_width), tile_height=int(tile_height))


def render_case_ladder_gif(
    *,
    case_spec: LadderCaseSpec,
    checkpoint_specs: Sequence[LadderCheckpointSpec],
    output_dir: str | Path,
    timing_records: Sequence[dict[str, Any]],
    frames: int = 30,
    gif_fps: int = 6,
    tile_width: int = 260,
    tile_height: int = 180,
    row_label_width: int = 92,
    depth_min_m: float = 0.2,
    depth_max_m: float = 1.5,
    max_points_per_camera: int | None = None,
    max_points_per_render: int | None = 80_000,
    save_first_frame_ply: bool = True,
    phystwin_radius_m: float = DEFAULT_PHYSTWIN_RADIUS_M,
    phystwin_nb_points: int = DEFAULT_PHYSTWIN_NB_POINTS,
    enhanced_component_voxel_size_m: float = DEFAULT_ENHANCED_COMPONENT_VOXEL_SIZE_M,
    enhanced_keep_near_main_gap_m: float = DEFAULT_ENHANCED_KEEP_NEAR_MAIN_GAP_M,
) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    gif_dir = output_dir / "gifs"
    first_dir = output_dir / "first_frames"
    ply_dir = output_dir / "first_frame_ply" / case_spec.key
    gif_path = gif_dir / f"{case_spec.output_name}.gif"
    first_frame_path = first_dir / f"{case_spec.output_name}_first.png"
    metadata = load_case_metadata(case_spec.case_dir)
    available_frames = min(int(frames), get_frame_count(metadata))
    if available_frames <= 0:
        raise RuntimeError(f"No frames available for {case_spec.case_dir}")
    timing_by_cell = timing_lookup(timing_records)

    first_cells, first_camera_stats = build_frame_cells(
        case_spec=case_spec,
        metadata=metadata,
        output_dir=output_dir,
        checkpoint_specs=checkpoint_specs,
        frame_idx=0,
        depth_min_m=float(depth_min_m),
        depth_max_m=float(depth_max_m),
        max_points_per_camera=max_points_per_camera,
        phystwin_radius_m=float(phystwin_radius_m),
        phystwin_nb_points=int(phystwin_nb_points),
        enhanced_component_voxel_size_m=float(enhanced_component_voxel_size_m),
        enhanced_keep_near_main_gap_m=float(enhanced_keep_near_main_gap_m),
    )
    view_specs = _build_original_camera_view_specs(
        first_camera_stats,
        tile_width=int(tile_width),
        tile_height=int(tile_height),
    )

    if save_first_frame_ply:
        for camera_idx in DEFAULT_CAMERA_IDS:
            for variant_key in DEFAULT_VARIANT_ORDER:
                cell = first_cells[int(camera_idx)][variant_key]
                write_ply_ascii(
                    ply_dir / f"{case_spec.key}_cam{int(camera_idx)}_{variant_key}_frame0000.ply",
                    np.asarray(cell["points"], dtype=np.float32),
                    np.asarray(cell["colors"], dtype=np.uint8),
                )

    column_headers = ["SAM3.1 existing", *[_variant_label(item.key, checkpoint_specs) for item in checkpoint_specs]]
    row_headers = [f"cam{camera_idx}" for camera_idx in DEFAULT_CAMERA_IDS]
    title_lines = [
        f"{case_spec.label} | SAM3.1 vs SAM2.1 checkpoint ladder | FFS RGB PCD after mask",
        (
            f"enhanced PT-like postprocess r={float(phystwin_radius_m):.3f}m/"
            f"{int(phystwin_nb_points)}nn comp={float(enhanced_component_voxel_size_m):.3f}m | "
            f"original camera-view pinhole render | time frames={available_frames}"
        ),
    ]

    frame_summaries: list[dict[str, Any]] = []
    gif_dir.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(gif_path), mode="I", fps=max(1, int(gif_fps)), loop=0) as writer:
        for frame_idx in range(available_frames):
            if frame_idx == 0:
                frame_cells = first_cells
                camera_stats = first_camera_stats
            else:
                frame_cells, camera_stats = build_frame_cells(
                    case_spec=case_spec,
                    metadata=metadata,
                    output_dir=output_dir,
                    checkpoint_specs=checkpoint_specs,
                    frame_idx=frame_idx,
                    depth_min_m=float(depth_min_m),
                    depth_max_m=float(depth_max_m),
                    max_points_per_camera=max_points_per_camera,
                    phystwin_radius_m=float(phystwin_radius_m),
                    phystwin_nb_points=int(phystwin_nb_points),
                    enhanced_component_voxel_size_m=float(enhanced_component_voxel_size_m),
                    enhanced_keep_near_main_gap_m=float(enhanced_keep_near_main_gap_m),
                )
            image_rows: list[list[np.ndarray]] = []
            frame_cell_stats: list[dict[str, Any]] = []
            for camera_idx in DEFAULT_CAMERA_IDS:
                row_tiles: list[np.ndarray] = []
                for variant_key in DEFAULT_VARIANT_ORDER:
                    cell = frame_cells[int(camera_idx)][variant_key]
                    frame_cell_stats.append(
                        {
                            "frame_idx": int(frame_idx),
                            "camera_idx": int(camera_idx),
                            "variant_key": str(variant_key),
                            "mask_pixel_count": int(cell["mask_pixel_count"]),
                            "raw_point_count": int(cell["raw_point_count"]),
                            "point_count": int(cell["point_count"]),
                            "postprocess_stats": cell["postprocess_stats"],
                        }
                    )
                    label = _cell_label(
                        case_key=case_spec.key,
                        camera_idx=int(camera_idx),
                        variant_key=variant_key,
                        point_count=int(cell["point_count"]),
                        timing_by_cell=timing_by_cell,
                    )
                    row_tiles.append(
                        _render_cell_tile(
                            cell=cell,
                            label=label,
                            view_spec=view_specs[int(camera_idx)],
                            tile_width=int(tile_width),
                            tile_height=int(tile_height),
                            max_points_per_render=max_points_per_render,
                        )
                    )
                image_rows.append(row_tiles)
            board = compose_3x5_panel(
                title_lines=title_lines,
                row_headers=row_headers,
                column_headers=column_headers,
                image_rows=image_rows,
                row_label_width=int(row_label_width),
            )
            if frame_idx == 0:
                write_image(first_frame_path, board)
            writer.append_data(cv2.cvtColor(board, cv2.COLOR_BGR2RGB))
            frame_summaries.append(
                {
                    "frame_idx": int(frame_idx),
                    "camera_stats": camera_stats,
                    "cells": frame_cell_stats,
                }
            )
            if frame_idx == 0 or frame_idx + 1 == available_frames or (frame_idx + 1) % 10 == 0:
                print(f"[sam21] rendered {case_spec.key} frame {frame_idx + 1}/{available_frames}", flush=True)

    return {
        "case_key": case_spec.key,
        "case_label": case_spec.label,
        "case_dir": str(case_spec.case_dir),
        "gif_path": str(gif_path),
        "first_frame_path": str(first_frame_path),
        "first_frame_ply_dir": str(ply_dir) if save_first_frame_ply else None,
        "frames": int(available_frames),
        "gif_fps": int(gif_fps),
        "tile_width": int(tile_width),
        "tile_height": int(tile_height),
        "row_label_width": int(row_label_width),
        "view_mode": "original_camera_pinhole",
        "view_specs": {
            str(camera_idx): _serialize_view_config(view_spec)
            for camera_idx, view_spec in sorted(view_specs.items())
        },
        "frame_summaries": frame_summaries,
    }


def _serialize_view_config(view_config: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in view_config.items():
        if isinstance(value, np.ndarray):
            result[key] = value.astype(float).tolist()
        elif isinstance(value, np.generic):
            result[key] = value.item()
        else:
            result[key] = value
    return result


def _mask_centroid(mask: np.ndarray) -> list[float | None]:
    ys, xs = np.where(np.asarray(mask, dtype=bool))
    if len(xs) == 0:
        return [None, None]
    return [float(np.mean(xs)), float(np.mean(ys))]


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    ma = np.asarray(a, dtype=bool)
    mb = np.asarray(b, dtype=bool)
    union = int(np.count_nonzero(ma | mb))
    if union == 0:
        return 1.0
    return float(np.count_nonzero(ma & mb) / union)


def compute_mask_quality_for_root(
    *,
    mask_root: str | Path,
    case_dir: str | Path,
    camera_idx: int,
    text_prompt: str,
    frame_tokens: Sequence[str],
) -> dict[str, Any]:
    masks = [
        load_union_mask(
            mask_root=mask_root,
            case_dir=case_dir,
            camera_idx=int(camera_idx),
            frame_token=str(frame_token),
            text_prompt=text_prompt,
        )
        for frame_token in frame_tokens
    ]
    area_curve = [int(np.count_nonzero(mask)) for mask in masks]
    areas = np.asarray(area_curve, dtype=np.float64)
    centroids = [_mask_centroid(mask) for mask in masks]
    frame0 = masks[0]
    iou_curve = [_mask_iou(mask, frame0) for mask in masks]
    ious = np.asarray(iou_curve, dtype=np.float64)
    valid_centroids = np.asarray(
        [[np.nan if value is None else float(value) for value in centroid] for centroid in centroids],
        dtype=np.float64,
    )
    per_frame = []
    for frame_token, area, centroid, iou in zip(frame_tokens, area_curve, centroids, iou_curve, strict=True):
        per_frame.append(
            {
                "frame_token": str(frame_token),
                "area": int(area),
                "centroid_x": centroid[0],
                "centroid_y": centroid[1],
                "iou_to_frame0": float(iou),
            }
        )
    return {
        "frame_count": int(len(masks)),
        "area_curve": area_curve,
        "centroid_curve": centroids,
        "iou_to_frame0_curve": [float(item) for item in iou_curve],
        "per_frame": per_frame,
        "area_mean": float(np.mean(areas)) if len(areas) else 0.0,
        "area_std": float(np.std(areas)) if len(areas) else 0.0,
        "area_std_over_mean": float(np.std(areas) / max(1.0, float(np.mean(areas)))) if len(areas) else 0.0,
        "area_min": int(np.min(areas)) if len(areas) else 0,
        "area_max": int(np.max(areas)) if len(areas) else 0,
        "centroid_std_x": float(np.nanstd(valid_centroids[:, 0])) if len(valid_centroids) else 0.0,
        "centroid_std_y": float(np.nanstd(valid_centroids[:, 1])) if len(valid_centroids) else 0.0,
        "iou_to_frame0_mean": float(np.mean(ious)) if len(ious) else 1.0,
        "iou_to_frame0_min": float(np.min(ious)) if len(ious) else 1.0,
    }


def compute_ladder_mask_quality(
    *,
    case_specs: Sequence[LadderCaseSpec],
    checkpoint_specs: Sequence[LadderCheckpointSpec],
    output_dir: str | Path,
    frames: int,
    camera_ids: Sequence[int] = DEFAULT_CAMERA_IDS,
) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    cases: dict[str, Any] = {}
    for case_spec in case_specs:
        frame_tokens = sorted_case_frame_tokens(case_spec.case_dir, camera_idx=0, frames=int(frames))
        variant_roots = _build_variant_roots(case_spec=case_spec, output_dir=output_dir, checkpoint_specs=checkpoint_specs)
        case_payload: dict[str, Any] = {
            "label": case_spec.label,
            "case_dir": str(case_spec.case_dir),
            "variants": {},
        }
        for variant_key, mask_root in variant_roots.items():
            variant_payload: dict[str, Any] = {}
            for camera_idx in [int(item) for item in camera_ids]:
                variant_payload[str(camera_idx)] = compute_mask_quality_for_root(
                    mask_root=mask_root,
                    case_dir=case_spec.case_dir,
                    camera_idx=camera_idx,
                    text_prompt=case_spec.text_prompt,
                    frame_tokens=frame_tokens,
                )
            case_payload["variants"][variant_key] = variant_payload
        cases[case_spec.key] = case_payload
    return {
        "frames": int(frames),
        "camera_ids": [int(item) for item in camera_ids],
        "cases": cases,
    }


def write_benchmark_report(
    *,
    markdown_path: str | Path,
    benchmark_payload: dict[str, Any],
    quality_payload: dict[str, Any],
) -> None:
    aggregate = benchmark_payload.get("timing_aggregate", {}).get("by_checkpoint", {})
    lines = [
        "# SAM21-max Round2 Benchmark",
        "",
        "## Environment",
        "",
    ]
    env = benchmark_payload.get("environment", {})
    for key in ("python", "torch", "torch_cuda", "torchvision", "sam2_file", "cuda_available", "gpu"):
        if key in env:
            lines.append(f"- {key}: `{env[key]}`")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
        ]
    )
    for item in benchmark_payload.get("gif_summaries", []):
        lines.append(f"- {item['case_label']}: `{item['gif_path']}`")
    lines.extend(
        [
            "",
            "## SAM2.1 Speed Ladder",
            "",
            "| checkpoint | mean ms/frame | mean FPS | samples |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for checkpoint_key in ("large", "base_plus", "small", "tiny"):
        item = aggregate.get(checkpoint_key, {})
        if not item:
            continue
        lines.append(
            f"| {checkpoint_key} | {float(item['mean_inference_ms_per_frame']):.2f} | "
            f"{float(item['mean_fps']):.2f} | {int(item['sample_count'])} |"
        )
    stable_summaries = benchmark_payload.get("stable_checkpoint_summaries", [])
    if stable_summaries:
        lines.extend(
            [
                "",
                "## Stable Worker Aggregate",
                "",
                "| checkpoint | prop-only ms/frame | prop-only FPS | sweep wall FPS incl. setup | timed frames |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for item in stable_summaries:
            lines.append(
                f"| {item['checkpoint_key']} | {float(item['aggregate_ms_per_frame']):.2f} | "
                f"{float(item['aggregate_fps']):.2f} | "
                f"{float(item['speed_phase_wall_fps_including_state_setup']):.2f} | "
                f"{int(item['total_timed_frames'])} |"
            )
    lines.extend(
        [
            "",
            "## Mask Stability",
            "",
            "| case | variant | cam | area std/mean | min IoU(frame,0) |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for case_key, case_payload in quality_payload.get("cases", {}).items():
        for variant_key, by_camera in case_payload.get("variants", {}).items():
            for camera_key, stats in by_camera.items():
                lines.append(
                    f"| {case_key} | {variant_key} | {camera_key} | "
                    f"{float(stats['area_std_over_mean']):.4f} | {float(stats['iou_to_frame0_min']):.4f} |"
                )
    lines.extend(
        [
            "",
            "## Timing Contract",
            "",
        ]
    )
    if benchmark_payload.get("sam21_timing_protocol") == "stable_throughput":
        marker_label = (
            "with per-step cudagraph markers"
            if benchmark_payload.get("stable_speed_uses_cudagraph_step_marker", True)
            else "without per-step cudagraph markers"
        )
        lines.append(
            "The reported SAM2.1 FPS is no-output propagation timing "
            f"{marker_label} after "
            f"{int(benchmark_payload.get('stable_warmup_runs', DEFAULT_STABLE_WARMUP_RUNS))} "
            "warmup propagations per case/camera job in one long-lived checkpoint worker. "
            "Model load, JPEG preparation, init_state, prompt, warmup propagation, and separate mask collection are excluded."
        )
    else:
        lines.append(
            "The reported SAM2.1 FPS is propagate timing from the timed pass in the diagnostic worker. "
            "Model load, JPEG preparation, init_state, prompt, and full warmup propagation are recorded but excluded; "
            "the timed pass materializes masks for panel output."
        )
    lines.extend(
        [
            "",
        ]
    )
    Path(markdown_path).parent.mkdir(parents=True, exist_ok=True)
    Path(markdown_path).write_text("\n".join(lines), encoding="utf-8")


def collect_environment_report() -> dict[str, Any]:
    report: dict[str, Any] = {"python": sys.version.split()[0]}
    try:
        import torch

        report["torch"] = torch.__version__
        report["torch_cuda"] = torch.version.cuda
        report["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            report["gpu"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        report["torch_error"] = str(exc)
    try:
        import torchvision

        report["torchvision"] = torchvision.__version__
    except Exception as exc:
        report["torchvision_error"] = str(exc)
    try:
        import sam2

        report["sam2_file"] = str(Path(sam2.__file__).resolve())
    except Exception as exc:
        report["sam2_error"] = str(exc)
    return report


def run_ladder_workflow(
    *,
    script_path: str | Path,
    root: str | Path,
    output_dir: str | Path,
    checkpoint_cache: str | Path = DEFAULT_SAM2_CHECKPOINT_CACHE,
    case_keys: Sequence[str] | None = None,
    checkpoint_keys: Sequence[str] | None = None,
    frames: int = 30,
    gif_fps: int = 6,
    tile_width: int = 260,
    tile_height: int = 180,
    row_label_width: int = 92,
    depth_min_m: float = 0.2,
    depth_max_m: float = 1.5,
    max_points_per_camera: int | None = None,
    max_points_per_render: int | None = 80_000,
    bbox_padding_px: int = 0,
    download_missing_checkpoints: bool = True,
    run_sam2: bool = True,
    render_gifs: bool = True,
    overwrite: bool = False,
    stable_throughput: bool = False,
    stable_warmup_runs: int = DEFAULT_STABLE_WARMUP_RUNS,
    stable_speed_use_step_marker: bool = True,
    save_first_frame_ply: bool = True,
    phystwin_radius_m: float = DEFAULT_PHYSTWIN_RADIUS_M,
    phystwin_nb_points: int = DEFAULT_PHYSTWIN_NB_POINTS,
    enhanced_component_voxel_size_m: float = DEFAULT_ENHANCED_COMPONENT_VOXEL_SIZE_M,
    enhanced_keep_near_main_gap_m: float = DEFAULT_ENHANCED_KEEP_NEAR_MAIN_GAP_M,
    docs_generated_root: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(root).resolve()
    output_dir = Path(output_dir).resolve()
    if overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_summary_path = output_dir / "summary.json"
    existing_summary: dict[str, Any] = {}
    if existing_summary_path.exists():
        existing_summary = json.loads(existing_summary_path.read_text(encoding="utf-8"))

    cases = default_ladder_case_specs(root=root)
    if case_keys is not None:
        allowed = {str(item) for item in case_keys}
        cases = [item for item in cases if item.key in allowed]
    checkpoints = default_ladder_checkpoint_specs()
    if checkpoint_keys is not None:
        allowed_checkpoints = {str(item) for item in checkpoint_keys}
        checkpoints = [item for item in checkpoints if item.key in allowed_checkpoints]
    if not cases:
        raise ValueError("No cases selected.")
    if not checkpoints:
        raise ValueError("No checkpoints selected.")

    checkpoint_records = ensure_sam21_checkpoints(
        checkpoints,
        checkpoint_cache=Path(checkpoint_cache),
        download_missing=bool(download_missing_checkpoints),
    )

    timing_records: list[dict[str, Any]]
    stable_checkpoint_summaries: list[dict[str, Any]] = []
    timing_dir = output_dir / "timings"
    if run_sam2:
        if stable_throughput:
            timing_records, stable_checkpoint_summaries = run_sam21_stable_workers_by_checkpoint(
                script_path=script_path,
                case_specs=cases,
                checkpoint_specs=checkpoints,
                checkpoint_cache=Path(checkpoint_cache),
                output_dir=output_dir,
                frames=int(frames),
                camera_ids=DEFAULT_CAMERA_IDS,
                bbox_padding_px=int(bbox_padding_px),
                warmup_runs=int(stable_warmup_runs),
                speed_use_step_marker=bool(stable_speed_use_step_marker),
                overwrite=bool(overwrite),
            )
        else:
            timing_records = run_sam21_workers_sequentially(
                script_path=script_path,
                case_specs=cases,
                checkpoint_specs=checkpoints,
                checkpoint_cache=Path(checkpoint_cache),
                output_dir=output_dir,
                frames=int(frames),
                camera_ids=DEFAULT_CAMERA_IDS,
                bbox_padding_px=int(bbox_padding_px),
                overwrite=bool(overwrite),
            )
    else:
        timing_records = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(timing_dir.glob("*.json"))
        ]
        stable_checkpoint_summaries = list(existing_summary.get("stable_checkpoint_summaries", []))

    gif_summaries: list[dict[str, Any]] = []
    if render_gifs:
        for case_spec in cases:
            gif_summaries.append(
                render_case_ladder_gif(
                    case_spec=case_spec,
                    checkpoint_specs=checkpoints,
                    output_dir=output_dir,
                    timing_records=timing_records,
                    frames=int(frames),
                    gif_fps=int(gif_fps),
                    tile_width=int(tile_width),
                    tile_height=int(tile_height),
                    row_label_width=int(row_label_width),
                    depth_min_m=float(depth_min_m),
                    depth_max_m=float(depth_max_m),
                    max_points_per_camera=max_points_per_camera,
                    max_points_per_render=max_points_per_render,
                    save_first_frame_ply=bool(save_first_frame_ply),
                    phystwin_radius_m=float(phystwin_radius_m),
                    phystwin_nb_points=int(phystwin_nb_points),
                    enhanced_component_voxel_size_m=float(enhanced_component_voxel_size_m),
                    enhanced_keep_near_main_gap_m=float(enhanced_keep_near_main_gap_m),
                )
            )
    else:
        gif_summaries = list(existing_summary.get("gif_summaries", []))

    quality_payload = compute_ladder_mask_quality(
        case_specs=cases,
        checkpoint_specs=checkpoints,
        output_dir=output_dir,
        frames=int(frames),
        camera_ids=DEFAULT_CAMERA_IDS,
    )
    benchmark_payload = {
        "output_dir": str(output_dir),
        "cases": [case_to_json(item) for item in cases],
        "checkpoints": checkpoint_records,
        "frames": int(frames),
        "gif_fps": int(gif_fps),
        "tile_width": int(tile_width),
        "tile_height": int(tile_height),
        "row_label_width": int(row_label_width),
        "sam21_timing_protocol": "stable_throughput" if stable_throughput else "diagnostic_mask_output",
        "stable_warmup_runs": int(stable_warmup_runs) if stable_throughput else None,
        "stable_speed_uses_cudagraph_step_marker": bool(stable_speed_use_step_marker) if stable_throughput else None,
        "depth_source": "ffs",
        "use_float_ffs_depth_when_available": True,
        "enhanced_phystwin_like_postprocess": {
            "enabled": True,
            "radius_m": float(phystwin_radius_m),
            "nb_points": int(phystwin_nb_points),
            "component_voxel_size_m": float(enhanced_component_voxel_size_m),
            "keep_near_main_gap_m": float(enhanced_keep_near_main_gap_m),
        },
        "timing_records": timing_records,
        "timing_aggregate": aggregate_timing_records(timing_records),
        "stable_checkpoint_summaries": stable_checkpoint_summaries,
        "gif_summaries": gif_summaries,
        "environment": collect_environment_report(),
    }
    write_json(output_dir / "summary.json", benchmark_payload)
    write_json(output_dir / "mask_quality.json", quality_payload)

    docs_root = root if docs_generated_root is None else Path(docs_generated_root).resolve()
    benchmark_json_path = docs_root / DEFAULT_DOC_BENCHMARK_JSON
    quality_json_path = docs_root / DEFAULT_DOC_QUALITY_JSON
    benchmark_md_path = docs_root / DEFAULT_DOC_BENCHMARK_MD
    write_json(benchmark_json_path, benchmark_payload)
    write_json(quality_json_path, quality_payload)
    write_benchmark_report(
        markdown_path=benchmark_md_path,
        benchmark_payload=benchmark_payload,
        quality_payload=quality_payload,
    )
    benchmark_payload["docs"] = {
        "benchmark_md": str(benchmark_md_path),
        "benchmark_json": str(benchmark_json_path),
        "quality_json": str(quality_json_path),
    }
    write_json(output_dir / "summary.json", benchmark_payload)
    return benchmark_payload
