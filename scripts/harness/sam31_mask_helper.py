from __future__ import annotations

import inspect
import json
import os
import platform
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


QQTT_SAM31_CHECKPOINT_ENV = "QQTT_SAM31_CHECKPOINT"
QQTT_SAM31_BPE_PATH_ENV = "QQTT_SAM31_BPE_PATH"
DEFAULT_OUTPUT_DIR_NAME = "sam31_masks"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")
BPE_VOCAB_NAME = "bpe_simple_vocab_16e6.txt.gz"
PROMPT_SPLIT_PATTERN = re.compile(r"[,\n;]+|(?<!\d)\.(?!\d)")
_CUDA_AUTOCAST_CONTEXT = None


@dataclass(slots=True)
class ColorSource:
    camera_idx: int
    mode: str
    path: Path
    frame_paths: list[Path] | None = None


def parse_text_prompts(text_prompt: str) -> list[str]:
    prompts: list[str] = []
    for chunk in PROMPT_SPLIT_PATTERN.split(text_prompt):
        normalized = " ".join(chunk.strip().lower().split())
        if normalized and normalized not in prompts:
            prompts.append(normalized)
    return prompts


def _frame_sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    return (0, f"{int(stem):09d}") if stem.isdigit() else (1, stem.lower())


def _list_frame_images(frame_dir: Path) -> list[Path]:
    frames = [path for path in frame_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(frames, key=_frame_sort_key)


def discover_color_sources(
    case_root: str | Path,
    *,
    camera_ids: Sequence[int],
    source_mode: str = "auto",
) -> dict[int, ColorSource]:
    root = Path(case_root).resolve()
    color_root = root / "color"
    if not color_root.is_dir():
        raise FileNotFoundError(f"Missing color directory: {color_root}")
    if source_mode not in {"auto", "mp4", "frames"}:
        raise ValueError(f"Unsupported source_mode: {source_mode}")

    sources: dict[int, ColorSource] = {}
    for camera_idx in [int(item) for item in camera_ids]:
        mp4_path = color_root / f"{camera_idx}.mp4"
        frame_dir = color_root / str(camera_idx)
        frame_paths = _list_frame_images(frame_dir) if frame_dir.is_dir() else []

        if source_mode in {"auto", "mp4"} and mp4_path.is_file():
            sources[camera_idx] = ColorSource(camera_idx=camera_idx, mode="mp4", path=mp4_path)
            continue
        if source_mode in {"auto", "frames"} and frame_paths:
            sources[camera_idx] = ColorSource(
                camera_idx=camera_idx,
                mode="frames",
                path=frame_dir,
                frame_paths=frame_paths,
            )
            continue

        raise FileNotFoundError(
            f"Unable to resolve RGB source for camera {camera_idx} under `{color_root}` "
            f"with source_mode={source_mode!r}. Expected `{mp4_path.name}` or `{frame_dir}`."
        )
    return sources


def default_output_dir(case_root: str | Path) -> Path:
    return Path(case_root).resolve() / DEFAULT_OUTPUT_DIR_NAME


def build_mask_output_path(output_dir: str | Path, *, camera_idx: int, obj_id: int, frame_token: str) -> Path:
    return Path(output_dir).resolve() / "mask" / str(int(camera_idx)) / str(int(obj_id)) / f"{frame_token}.png"


def _resolve_sam3_video_predictor_builder(model_builder_module: Any) -> tuple[Any, str]:
    if hasattr(model_builder_module, "build_sam3_predictor"):
        return model_builder_module.build_sam3_predictor, "build_sam3_predictor"
    if hasattr(model_builder_module, "build_sam3_video_predictor"):
        return model_builder_module.build_sam3_video_predictor, "build_sam3_video_predictor"
    raise ImportError(
        "sam3.model_builder does not expose `build_sam3_predictor` or `build_sam3_video_predictor`."
    )


def _build_sam31_builder_kwargs(
    builder_name: str,
    *,
    checkpoint_path: str,
    bpe_path: str | None,
    async_loading_frames: bool,
    compile_model: bool,
    max_num_objects: int,
) -> dict[str, Any]:
    if builder_name == "build_sam3_predictor":
        return {
            "checkpoint_path": checkpoint_path,
            "version": "sam3.1",
            "compile": compile_model,
            "warm_up": False,
            "max_num_objects": max_num_objects,
            "use_fa3": False,
            "async_loading_frames": async_loading_frames,
        }

    if builder_name == "build_sam3_video_predictor":
        builder_kwargs = {
            "checkpoint_path": checkpoint_path,
            "async_loading_frames": async_loading_frames,
        }
        if bpe_path is not None:
            builder_kwargs["bpe_path"] = bpe_path
        return builder_kwargs

    raise ValueError(f"Unsupported SAM 3 builder: {builder_name}")


def _call_download_ckpt_from_hf(download_ckpt_from_hf: Any) -> str:
    signature = inspect.signature(download_ckpt_from_hf)
    if "version" in signature.parameters:
        return str(download_ckpt_from_hf(version="sam3.1"))
    return str(download_ckpt_from_hf())


def resolve_sam31_bpe_path(checkpoint_path: str | Path | None = None) -> str | None:
    candidates: list[Path] = []

    bpe_override = os.getenv(QQTT_SAM31_BPE_PATH_ENV)
    if bpe_override:
        candidates.append(Path(bpe_override).expanduser())

    if checkpoint_path is not None:
        checkpoint_dir = Path(checkpoint_path).expanduser().resolve().parent
        candidates.append(checkpoint_dir / BPE_VOCAB_NAME)

    try:
        import sam3  # noqa: PLC0415

        sam3_root = Path(sam3.__file__).resolve().parent
        candidates.append(sam3_root / "assets" / BPE_VOCAB_NAME)
        candidates.append(sam3_root.parent / "assets" / BPE_VOCAB_NAME)
    except Exception:
        pass

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return str(resolved)
    return None


def _load_runtime_deps():
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    import sam3.model_builder as sam3_model_builder  # noqa: PLC0415

    build_video_predictor, builder_name = _resolve_sam3_video_predictor_builder(sam3_model_builder)
    download_ckpt_from_hf = sam3_model_builder.download_ckpt_from_hf

    return cv2, np, torch, build_video_predictor, builder_name, download_ckpt_from_hf


def _configure_torch_inference(torch_module) -> None:
    global _CUDA_AUTOCAST_CONTEXT

    if not torch_module.cuda.is_available():
        return

    # On this Windows + RTX 5090 path, SAM 3.1 can trip over CUDA SDPA kernel
    # selection when global bfloat16 autocast is enabled. Force the safer math
    # SDP path instead of relying on unavailable flash / mem-efficient kernels.
    if platform.system() == "Windows":
        if hasattr(torch_module.backends.cuda, "enable_flash_sdp"):
            torch_module.backends.cuda.enable_flash_sdp(False)
        if hasattr(torch_module.backends.cuda, "enable_mem_efficient_sdp"):
            torch_module.backends.cuda.enable_mem_efficient_sdp(False)
        if hasattr(torch_module.backends.cuda, "enable_cudnn_sdp"):
            torch_module.backends.cuda.enable_cudnn_sdp(False)
        if hasattr(torch_module.backends.cuda, "enable_math_sdp"):
            torch_module.backends.cuda.enable_math_sdp(True)
        return

    if _CUDA_AUTOCAST_CONTEXT is None:
        _CUDA_AUTOCAST_CONTEXT = torch_module.autocast(device_type="cuda", dtype=torch_module.bfloat16)
        _CUDA_AUTOCAST_CONTEXT.__enter__()

    device_properties = torch_module.cuda.get_device_properties(torch_module.cuda.current_device())
    if device_properties.major >= 8:
        torch_module.backends.cuda.matmul.allow_tf32 = True
        torch_module.backends.cudnn.allow_tf32 = True


def resolve_sam31_checkpoint_path(checkpoint_path: str | Path | None = None) -> str:
    if checkpoint_path is not None:
        resolved = Path(checkpoint_path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"--checkpoint points to a missing file: {resolved}")
        return str(resolved)

    checkpoint_override = os.getenv(QQTT_SAM31_CHECKPOINT_ENV)
    if checkpoint_override:
        resolved = Path(checkpoint_override).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(
                f"{QQTT_SAM31_CHECKPOINT_ENV} points to a missing file: {resolved}"
            )
        return str(resolved)

    _, _, _, _, _, download_ckpt_from_hf = _load_runtime_deps()
    try:
        return str(Path(_call_download_ckpt_from_hf(download_ckpt_from_hf)).resolve())
    except Exception as exc:
        raise RuntimeError(
            "Unable to resolve the SAM 3.1 checkpoint. Run `hf auth login`, accept "
            "https://huggingface.co/facebook/sam3.1, or pass --checkpoint / set "
            f"{QQTT_SAM31_CHECKPOINT_ENV}."
        ) from exc


def build_sam31_video_predictor(
    *,
    checkpoint_path: str | Path | None = None,
    async_loading_frames: bool = False,
    compile_model: bool = False,
    max_num_objects: int = 16,
):
    _, _, torch_module, build_video_predictor, builder_name, _ = _load_runtime_deps()
    if not torch_module.cuda.is_available():
        raise RuntimeError("The upstream SAM 3.1 video predictor currently requires CUDA.")

    _configure_torch_inference(torch_module)
    resolved_checkpoint = resolve_sam31_checkpoint_path(checkpoint_path)
    resolved_bpe_path = resolve_sam31_bpe_path(resolved_checkpoint)
    predictor = build_video_predictor(
        **_build_sam31_builder_kwargs(
            builder_name,
            checkpoint_path=resolved_checkpoint,
            bpe_path=resolved_bpe_path,
            async_loading_frames=async_loading_frames,
            compile_model=compile_model,
            max_num_objects=max_num_objects,
        )
    )
    return predictor, resolved_checkpoint


def _prepare_session_frames(source: ColorSource, *, session_dir: Path) -> dict[int, str]:
    cv2, _, _, _, _ = _load_runtime_deps()
    if session_dir.exists():
        shutil.rmtree(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)

    frame_token_by_index: dict[int, str] = {}

    if source.mode == "mp4":
        capture = cv2.VideoCapture(str(source.path))
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open video for frame extraction: {source.path}")
        frame_idx = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                target_path = session_dir / f"{frame_idx:05d}.jpg"
                cv2.imwrite(str(target_path), frame)
                frame_token_by_index[frame_idx] = str(frame_idx)
                frame_idx += 1
        finally:
            capture.release()
        if frame_idx == 0:
            raise RuntimeError(f"No frames were extracted from `{source.path}`.")
        return frame_token_by_index

    if source.mode != "frames" or not source.frame_paths:
        raise ValueError(f"Unsupported color source for frame preparation: {source}")

    for session_idx, frame_path in enumerate(source.frame_paths):
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Unable to read RGB frame: {frame_path}")
        target_path = session_dir / f"{session_idx:05d}.jpg"
        cv2.imwrite(str(target_path), frame)
        frame_token_by_index[session_idx] = frame_path.stem

    if not frame_token_by_index:
        raise RuntimeError(f"No image frames were found under `{source.path}`.")
    return frame_token_by_index


def _select_output_indices(outputs: dict[str, Any], *, keep_all_instances: bool) -> list[int]:
    _, np, _, _, _ = _load_runtime_deps()
    out_obj_ids = np.asarray(outputs.get("out_obj_ids", []), dtype=np.int64).reshape(-1)
    if out_obj_ids.size == 0:
        return []

    if keep_all_instances or out_obj_ids.size == 1:
        return list(range(out_obj_ids.size))

    out_probs = np.asarray(outputs.get("out_probs", []), dtype=np.float32).reshape(-1)
    if out_probs.size == out_obj_ids.size:
        return [int(np.argmax(out_probs))]

    out_masks = np.asarray(outputs.get("out_binary_masks", []))
    if out_masks.ndim == 2:
        out_masks = out_masks[None, ...]
    if out_masks.ndim >= 3 and out_masks.shape[0] == out_obj_ids.size:
        areas = out_masks.reshape(out_masks.shape[0], -1).sum(axis=1)
        return [int(np.argmax(areas))]

    return [0]


def _collect_frame_segments(outputs: dict[str, Any], *, allowed_obj_ids: set[int] | None = None) -> dict[int, np.ndarray]:
    _, np, _, _, _ = _load_runtime_deps()
    out_obj_ids = np.asarray(outputs.get("out_obj_ids", []), dtype=np.int64).reshape(-1)
    if out_obj_ids.size == 0:
        return {}

    out_masks = np.asarray(outputs.get("out_binary_masks", []))
    if out_masks.ndim == 2:
        out_masks = out_masks[None, ...]

    frame_segments: dict[int, np.ndarray] = {}
    for idx, obj_id in enumerate(out_obj_ids.tolist()):
        obj_id = int(obj_id)
        if allowed_obj_ids is not None and obj_id not in allowed_obj_ids:
            continue
        frame_segments[obj_id] = np.asarray(out_masks[idx]).astype(bool)
    return frame_segments


def _prepare_camera_output(output_dir: Path, *, camera_idx: int, overwrite: bool) -> None:
    mask_root = output_dir / "mask"
    camera_mask_dir = mask_root / str(int(camera_idx))
    camera_info_path = mask_root / f"mask_info_{int(camera_idx)}.json"
    if camera_mask_dir.exists() or camera_info_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Mask output already exists for camera {camera_idx} under `{mask_root}`. "
                "Pass --overwrite to replace it."
            )
        shutil.rmtree(camera_mask_dir, ignore_errors=True)
        if camera_info_path.exists():
            camera_info_path.unlink()


def run_camera_segmentation(
    *,
    source: ColorSource,
    output_dir: str | Path,
    text_prompt: str,
    checkpoint_path: str | Path | None = None,
    ann_frame_index: int = 0,
    keep_session_frames: bool = False,
    session_root: str | Path | None = None,
    overwrite: bool = False,
    async_loading_frames: bool = False,
    compile_model: bool = False,
    max_num_objects: int = 16,
) -> dict[str, Any]:
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    _prepare_camera_output(output_root, camera_idx=source.camera_idx, overwrite=overwrite)

    prompts = parse_text_prompts(text_prompt)
    if not prompts:
        raise ValueError("text_prompt must contain at least one non-empty prompt.")

    session_base_dir = (
        Path(session_root).resolve()
        if session_root is not None
        else output_root / "_sam31_session_frames"
    )
    session_base_dir.mkdir(parents=True, exist_ok=True)
    session_dir = Path(
        tempfile.mkdtemp(prefix=f"cam{int(source.camera_idx)}_", dir=str(session_base_dir))
    )
    frame_token_by_index = _prepare_session_frames(source, session_dir=session_dir)

    predictor, resolved_checkpoint = build_sam31_video_predictor(
        checkpoint_path=checkpoint_path,
        async_loading_frames=async_loading_frames,
        compile_model=compile_model,
        max_num_objects=max_num_objects,
    )
    session_id = None
    selected_obj_to_label: dict[int, str] = {}
    initial_frame_segments: dict[int, np.ndarray] = {}

    try:
        session_id = predictor.handle_request(
            {"type": "start_session", "resource_path": str(session_dir)}
        )["session_id"]

        for prompt_idx, prompt in enumerate(prompts):
            response = predictor.handle_request(
                {
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": int(ann_frame_index),
                    "text": prompt,
                }
            )
            _, np, _, _, _ = _load_runtime_deps()
            outputs = response["outputs"]
            candidate_ids = np.asarray(outputs.get("out_obj_ids", []), dtype=np.int64).reshape(-1)

            if candidate_ids.size == 0:
                continue

            keep_all_instances = len(prompts) == 1 or prompt_idx > 0
            selected_indices = _select_output_indices(outputs, keep_all_instances=keep_all_instances)
            selected_ids = {int(candidate_ids[idx]) for idx in selected_indices}

            for candidate_id in candidate_ids.tolist():
                candidate_id = int(candidate_id)
                if candidate_id in selected_ids:
                    selected_obj_to_label[candidate_id] = prompt
                    continue
                predictor.handle_request(
                    {
                        "type": "remove_object",
                        "session_id": session_id,
                        "frame_index": int(ann_frame_index),
                        "obj_id": candidate_id,
                    }
                )

            initial_frame_segments.update(
                _collect_frame_segments(outputs, allowed_obj_ids=selected_ids)
            )

        if not selected_obj_to_label:
            raise RuntimeError(
                f"SAM 3.1 did not register any object for prompt `{text_prompt}` "
                f"for camera {source.camera_idx}."
            )

        tracked_obj_ids = set(selected_obj_to_label)
        video_segments: dict[int, dict[int, np.ndarray]] = {}
        for response in predictor.handle_stream_request(
            {
                "type": "propagate_in_video",
                "session_id": session_id,
                "start_frame_index": int(ann_frame_index),
                "propagation_direction": "forward",
            }
        ):
            frame_idx = int(response["frame_index"])
            video_segments[frame_idx] = _collect_frame_segments(
                response["outputs"],
                allowed_obj_ids=tracked_obj_ids,
            )

        if int(ann_frame_index) not in video_segments and initial_frame_segments:
            video_segments[int(ann_frame_index)] = initial_frame_segments
    finally:
        if session_id is not None:
            predictor.handle_request(
                {
                    "type": "close_session",
                    "session_id": session_id,
                    "run_gc_collect": True,
                }
            )
        if hasattr(predictor, "shutdown"):
            predictor.shutdown()

    cv2, np, _, _, _ = _load_runtime_deps()
    mask_root = output_root / "mask"
    mask_root.mkdir(parents=True, exist_ok=True)
    with (mask_root / f"mask_info_{int(source.camera_idx)}.json").open("w", encoding="utf-8") as handle:
        json.dump({str(key): value for key, value in sorted(selected_obj_to_label.items())}, handle, indent=2)

    per_object_frame_counts: dict[str, int] = {str(int(obj_id)): 0 for obj_id in sorted(selected_obj_to_label)}
    saved_frame_count = 0
    for frame_idx, masks in sorted(video_segments.items()):
        frame_token = frame_token_by_index.get(int(frame_idx), str(int(frame_idx)))
        if masks:
            saved_frame_count += 1
        for obj_id, mask in masks.items():
            output_path = build_mask_output_path(
                output_root,
                camera_idx=source.camera_idx,
                obj_id=obj_id,
                frame_token=frame_token,
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), np.asarray(mask, dtype=np.uint8) * 255)
            per_object_frame_counts[str(int(obj_id))] = per_object_frame_counts.get(str(int(obj_id)), 0) + 1

    if not keep_session_frames:
        shutil.rmtree(session_dir, ignore_errors=True)
        if session_root is None and session_base_dir.exists() and not any(session_base_dir.iterdir()):
            session_base_dir.rmdir()

    return {
        "camera_idx": int(source.camera_idx),
        "source_mode": source.mode,
        "source_path": str(source.path.resolve()),
        "frame_count": int(len(frame_token_by_index)),
        "saved_frame_count": int(saved_frame_count),
        "tracked_object_count": int(len(selected_obj_to_label)),
        "tracked_object_labels": {str(key): value for key, value in sorted(selected_obj_to_label.items())},
        "per_object_frame_counts": per_object_frame_counts,
        "checkpoint_path": resolved_checkpoint,
        "session_frame_dir": str(session_dir) if keep_session_frames else None,
        "output_mask_root": str((output_root / "mask" / str(int(source.camera_idx))).resolve()),
    }


def run_case_segmentation(
    *,
    case_root: str | Path,
    text_prompt: str,
    camera_ids: Sequence[int],
    output_dir: str | Path | None = None,
    source_mode: str = "auto",
    checkpoint_path: str | Path | None = None,
    ann_frame_index: int = 0,
    keep_session_frames: bool = False,
    session_root: str | Path | None = None,
    overwrite: bool = False,
    async_loading_frames: bool = False,
    compile_model: bool = False,
    max_num_objects: int = 16,
) -> dict[str, Any]:
    case_root = Path(case_root).resolve()
    output_root = default_output_dir(case_root) if output_dir is None else Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    sources = discover_color_sources(case_root, camera_ids=camera_ids, source_mode=source_mode)
    camera_summaries = []
    resolved_checkpoint = None
    for camera_idx in [int(item) for item in camera_ids]:
        summary = run_camera_segmentation(
            source=sources[camera_idx],
            output_dir=output_root,
            text_prompt=text_prompt,
            checkpoint_path=checkpoint_path,
            ann_frame_index=ann_frame_index,
            keep_session_frames=keep_session_frames,
            session_root=session_root,
            overwrite=overwrite,
            async_loading_frames=async_loading_frames,
            compile_model=compile_model,
            max_num_objects=max_num_objects,
        )
        resolved_checkpoint = summary["checkpoint_path"]
        camera_summaries.append(summary)

    result = {
        "case_root": str(case_root),
        "output_dir": str(output_root),
        "source_mode": source_mode,
        "camera_ids": [int(item) for item in camera_ids],
        "text_prompt": text_prompt,
        "parsed_prompts": parse_text_prompts(text_prompt),
        "ann_frame_index": int(ann_frame_index),
        "checkpoint_path": resolved_checkpoint,
        "camera_summaries": camera_summaries,
    }
    with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result
