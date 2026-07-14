"""Demo v6.1-owned SAM3.1 single-image text segmentation."""

from __future__ import annotations

import os
import platform
import re
import time
from argparse import ArgumentParser
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from demo_v6_2.shape_prior.timing import elapsed_ms


QQTT_SAM31_CHECKPOINT_ENV = "QQTT_SAM31_CHECKPOINT"
QQTT_SAM31_BPE_PATH_ENV = "QQTT_SAM31_BPE_PATH"
BPE_VOCAB_NAME = "bpe_simple_vocab_16e6.txt.gz"
# Prompts split on commas/newlines/semicolons, plus periods that are not part
# of a decimal number (so labels like "sam 3.1" stay intact).
PROMPT_SPLIT_PATTERN = re.compile(r"[,\n;]+|(?<!\d)\.(?!\d)")

# Keyed by (checkpoint, bpe path, compile flag, confidence threshold, device);
# lets reuse_model callers skip the multi-second model load on repeat requests.
_IMAGE_PROCESSOR_CACHE: dict[
    tuple[str, str | None, bool, float, str],
    tuple[Any, Any],
] = {}


def parse_text_prompts(text_prompt: str) -> list[str]:
    # Lowercase, collapse whitespace, and de-duplicate while keeping order.
    """Parse text prompts."""
    prompts: list[str] = []
    for chunk in PROMPT_SPLIT_PATTERN.split(text_prompt):
        normalized = " ".join(chunk.strip().lower().split())
        if normalized and normalized not in prompts:
            prompts.append(normalized)
    return prompts


def resolve_sam31_bpe_path(checkpoint_path: str | Path | None = None) -> str | None:
    # Search order: env override, next to the checkpoint, then the sam3 package
    # assets. Returning None lets the model builder use its bundled default.
    """Resolve the SAM 3.1 BPE vocabulary path."""
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


def resolve_sam31_checkpoint_path(checkpoint_path: str | Path | None = None) -> str:
    """Resolve the SAM 3.1 checkpoint path."""
    if checkpoint_path is not None:
        resolved = Path(checkpoint_path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(
                f"--checkpoint points to a missing file: {resolved}"
            )
        return str(resolved)

    checkpoint_override = os.getenv(QQTT_SAM31_CHECKPOINT_ENV)
    if checkpoint_override:
        resolved = Path(checkpoint_override).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(
                f"{QQTT_SAM31_CHECKPOINT_ENV} points to a missing file: {resolved}"
            )
        return str(resolved)

    from sam3.model_builder import download_ckpt_from_hf  # noqa: PLC0415

    try:
        return str(Path(download_ckpt_from_hf(version="sam3.1")).resolve())
    except Exception as exc:
        raise RuntimeError(
            "Unable to resolve the SAM 3.1 checkpoint. Run `hf auth login`, "
            "accept https://huggingface.co/facebook/sam3.1, or set "
            f"{QQTT_SAM31_CHECKPOINT_ENV}."
        ) from exc


def _configure_torch_inference(torch_module: Any) -> None:
    """Apply idempotent process-wide inference settings (SDP backends, TF32).

    Autocast is intentionally NOT entered here: torch autocast state is
    thread-local, so a long-lived context entered on one thread can never be
    exited safely from another. Inference wraps itself in a scoped autocast
    via _inference_autocast instead.
    """
    if not torch_module.cuda.is_available():
        return

    # Fused SDP kernels are unreliable on Windows CUDA builds: force the math
    # backend there and skip the autocast/TF32 setup entirely.
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

    # TF32 matmul is only worthwhile (and supported) on Ampere+ (SM 8.x).
    device_properties = torch_module.cuda.get_device_properties(
        torch_module.cuda.current_device()
    )
    if device_properties.major >= 8:
        torch_module.backends.cuda.matmul.allow_tf32 = True
        torch_module.backends.cudnn.allow_tf32 = True


def _inference_autocast(torch_module: Any) -> Any:
    """Scoped bfloat16 autocast for one inference call on the calling thread.

    Entered and exited on the same thread within run_image_segmentation, so
    no autocast state ever leaks across calls or threads. Windows keeps the
    math-SDP fp32 path (see _configure_torch_inference).
    """
    if torch_module.cuda.is_available() and platform.system() != "Windows":
        return torch_module.autocast(device_type="cuda", dtype=torch_module.bfloat16)
    return nullcontext()


def clear_sam31_image_processor_cache() -> None:
    """Return the clear sam31 image processor cache."""
    _IMAGE_PROCESSOR_CACHE.clear()


def release_sam31_image_segmentation_runtime() -> None:
    """Drop the cached SAM3.1 model/processor so their GPU memory can be freed.

    There is no autocast state to unwind: inference uses a per-call scoped
    autocast (_inference_autocast).
    """
    clear_sam31_image_processor_cache()


def _build_sam31_image_processor(
    *,
    checkpoint_path: str | Path | None,
    compile_model: bool,
    confidence_threshold: float,
    device: str,
    reuse_model: bool,
) -> tuple[Any, Any, str, str | None, dict[str, float | bool]]:
    # Each stage is timed separately; the dict lands in the "timing_ms" payload
    # of run_image_segmentation so callers can attribute warmup cost.
    """Build sam31 image processor."""
    total_start_s = time.perf_counter()

    import_start_s = time.perf_counter()
    import torch  # noqa: PLC0415

    from sam3.model.sam3_image_processor import Sam3Processor  # noqa: PLC0415
    from sam3.model_builder import build_sam3_image_model  # noqa: PLC0415

    import_ms = float((time.perf_counter() - import_start_s) * 1000.0)
    if not torch.cuda.is_available():
        raise RuntimeError("The upstream SAM 3.1 image model currently requires CUDA.")

    configure_start_s = time.perf_counter()
    _configure_torch_inference(torch)
    configure_ms = float((time.perf_counter() - configure_start_s) * 1000.0)

    resolve_start_s = time.perf_counter()
    resolved_checkpoint = resolve_sam31_checkpoint_path(checkpoint_path)
    resolved_bpe_path = resolve_sam31_bpe_path(resolved_checkpoint)
    resolve_ms = float((time.perf_counter() - resolve_start_s) * 1000.0)

    cache_key = (
        resolved_checkpoint,
        resolved_bpe_path,
        bool(compile_model),
        float(confidence_threshold),
        str(device),
    )
    if reuse_model and cache_key in _IMAGE_PROCESSOR_CACHE:
        model, processor = _IMAGE_PROCESSOR_CACHE[cache_key]
        return (
            model,
            processor,
            resolved_checkpoint,
            resolved_bpe_path,
            {
                "cache_hit": True,
                "import_ms": import_ms,
                "configure_ms": configure_ms,
                "resolve_paths_ms": resolve_ms,
                "model_load_ms": 0.0,
                "processor_init_ms": 0.0,
                "total_ms": float((time.perf_counter() - total_start_s) * 1000.0),
            },
        )

    model_start_s = time.perf_counter()
    model = build_sam3_image_model(
        bpe_path=resolved_bpe_path,
        checkpoint_path=resolved_checkpoint,
        load_from_HF=False,
        device=device,
        eval_mode=True,
        compile=compile_model,
    )
    model_load_ms = float((time.perf_counter() - model_start_s) * 1000.0)

    processor_start_s = time.perf_counter()
    processor = Sam3Processor(
        model,
        device=device,
        confidence_threshold=float(confidence_threshold),
    )
    processor_init_ms = float((time.perf_counter() - processor_start_s) * 1000.0)
    if reuse_model:
        _IMAGE_PROCESSOR_CACHE[cache_key] = (model, processor)

    return (
        model,
        processor,
        resolved_checkpoint,
        resolved_bpe_path,
        {
            "cache_hit": False,
            "import_ms": import_ms,
            "configure_ms": configure_ms,
            "resolve_paths_ms": resolve_ms,
            "model_load_ms": model_load_ms,
            "processor_init_ms": processor_init_ms,
            "total_ms": float((time.perf_counter() - total_start_s) * 1000.0),
        },
    )


def _as_numpy_array(value: Any) -> np.ndarray:
    # Duck-typed tensor-to-numpy conversion; works whether the processor hands
    # back torch tensors or plain arrays. bfloat16 has no numpy equivalent, so
    # it is upcast to float32 first.
    """Coerce the input into numpy array."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if str(getattr(value, "dtype", "")) == "torch.bfloat16" and hasattr(
        value,
        "float",
    ):
        value = value.float()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _normalize_mask_stack(masks: np.ndarray) -> np.ndarray:
    """Normalize (H, W) / (N, 1, H, W) mask layouts to (N, H, W)."""
    if masks.ndim == 2:
        masks = masks[None, ...]
    if masks.ndim >= 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    return masks


def _select_image_output_indices(
    state: dict[str, Any],
    *,
    keep_all_instances: bool,
) -> list[int]:
    """Select image output indices."""
    masks = _as_numpy_array(state.get("masks", []))
    if masks.size == 0:
        return []
    masks = _normalize_mask_stack(masks)

    object_count = int(masks.shape[0])
    if object_count == 0:
        return []
    if keep_all_instances or object_count == 1:
        return list(range(object_count))

    # Single-instance selection: prefer the highest-score detection; fall back
    # to the largest mask when scores are missing or malformed.
    scores = _as_numpy_array(state.get("scores", []))
    if scores.size == object_count:
        return [int(scores.reshape(-1).argmax())]

    areas = masks.reshape(object_count, -1).sum(axis=1)
    return [int(areas.argmax())]


def _collect_image_prompt_masks(
    state: dict[str, Any],
    *,
    selected_indices: set[int],
) -> list[np.ndarray]:
    """Return the collect image prompt masks."""
    masks = _as_numpy_array(state.get("masks", []))
    if masks.size == 0:
        return []
    masks = _normalize_mask_stack(masks)

    output: list[np.ndarray] = []
    for idx in sorted(int(item) for item in selected_indices):
        if idx < 0 or idx >= int(masks.shape[0]):
            continue
        output.append(np.ascontiguousarray(masks[idx].astype(bool)))
    return output


def run_image_segmentation(
    *,
    image: Any,
    text_prompt: str,
    checkpoint_path: str | Path | None = None,
    compile_model: bool = False,
    max_num_objects: int = 16,
    confidence_threshold: float = 0.5,
    device: str = "cuda",
    reuse_model: bool = False,
) -> dict[str, Any]:
    """Run SAM3.1 text segmentation on one image."""

    prompts = parse_text_prompts(text_prompt)
    if not prompts:
        raise ValueError("text_prompt must contain at least one non-empty prompt")

    total_start_s = time.perf_counter()
    (
        model,
        processor,
        resolved_checkpoint,
        resolved_bpe_path,
        build_timing,
    ) = _build_sam31_image_processor(
        checkpoint_path=checkpoint_path,
        compile_model=compile_model,
        confidence_threshold=float(confidence_threshold),
        device=device,
        reuse_model=bool(reuse_model),
    )
    # The processor holds the model; drop the redundant local reference.
    del model

    import torch  # noqa: PLC0415

    masks_by_label: dict[str, list[np.ndarray]] = {prompt: [] for prompt in prompts}
    per_prompt_counts: dict[str, int] = {}
    prompt_timing_ms: dict[str, float] = {}
    # The autocast scope covers the image embedding and every prompt so the
    # whole call runs under bfloat16, then fully unwinds on this same thread.
    with _inference_autocast(torch):
        set_image_start_s = time.perf_counter()
        state = processor.set_image(image)
        set_image_ms = float((time.perf_counter() - set_image_start_s) * 1000.0)

        with torch.inference_mode():
            for prompt_idx, prompt in enumerate(prompts):
                prompt_start_s = time.perf_counter()
                # Prompts share one image embedding; only the text state is
                # reset between prompts.
                if prompt_idx > 0:
                    processor.reset_all_prompts(state)
                state = processor.set_text_prompt(prompt, state)
                # Instance policy: single-prompt requests and every prompt
                # after the first keep all instances; only the first prompt of
                # a multi-prompt request is reduced to its best instance. Demo
                # callers rely on this by ordering prompts object-first,
                # controller-second (one object mask, both hand instances
                # kept).
                keep_all_instances = len(prompts) == 1 or prompt_idx > 0
                selected_indices = set(
                    _select_image_output_indices(
                        state,
                        keep_all_instances=keep_all_instances,
                    )
                )
                prompt_masks = _collect_image_prompt_masks(
                    state,
                    selected_indices=selected_indices,
                )
                masks_by_label[prompt].extend(prompt_masks)
                per_prompt_counts[prompt] = int(len(prompt_masks))
                prompt_timing_ms[prompt] = float(
                    (time.perf_counter() - prompt_start_s) * 1000.0
                )

    return {
        "checkpoint_path": resolved_checkpoint,
        "bpe_path": resolved_bpe_path,
        "text_prompt": text_prompt,
        "parsed_prompts": prompts,
        "masks_by_label": masks_by_label,
        "per_prompt_counts": per_prompt_counts,
        "inference_mode": "sam31-image-one-frame",
        "max_num_objects": int(max_num_objects),
        "timing_ms": {
            **build_timing,
            "set_image_ms": set_image_ms,
            "prompt_total_ms": float(sum(prompt_timing_ms.values())),
            "prompt_ms_by_label": prompt_timing_ms,
            "total_ms": float((time.perf_counter() - total_start_s) * 1000.0),
        },
    }


def segment_image_to_origin_rgba(
    *,
    img_path: str | Path,
    text_prompt: str,
    output_path: str | Path,
    device: str = "cuda",
    reuse_model: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Segment one image and return its RGBA export timing breakdown."""
    total_start_s = time.perf_counter()
    prompt_labels = parse_text_prompts(text_prompt)
    if len(prompt_labels) != 1:
        raise ValueError("--TEXT_PROMPT must resolve to exactly one prompt")

    # The image is read twice on purpose: cv2 supplies the raw pixels for the
    # RGBA export while PIL feeds the model in the format the processor expects.
    input_read_start_s = time.perf_counter()
    image_path = Path(img_path)
    raw_img = cv2.imread(str(image_path))
    if raw_img is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")

    with Image.open(image_path) as image:
        model_image = image.convert("RGB")
    input_read_ms = elapsed_ms(input_read_start_s)

    result = run_image_segmentation(
        image=model_image,
        text_prompt=text_prompt,
        checkpoint_path=None,
        compile_model=False,
        max_num_objects=16,
        confidence_threshold=0.5,
        device=device,
        reuse_model=bool(reuse_model),
    )
    inference_timing = result.get("timing_ms")
    if not isinstance(inference_timing, dict):
        raise RuntimeError("SAM3.1 result timing_ms must be an object")

    # Union every instance mask returned for the label; the RGBA export wants
    # one foreground mask, not per-instance masks.
    mask_union_start_s = time.perf_counter()
    label = prompt_labels[0]
    label_masks = list(result["masks_by_label"].get(label, []))
    if not label_masks:
        raise RuntimeError(f"SAM3.1 did not produce a mask for label {label!r}")
    mask = np.zeros_like(label_masks[0], dtype=bool)
    for label_mask in label_masks:
        label_mask_bool = np.asarray(label_mask, dtype=bool)
        if label_mask_bool.shape != mask.shape:
            raise RuntimeError("SAM3.1 masks for one label have inconsistent shapes")
        mask |= label_mask_bool
    mask = np.ascontiguousarray(mask)
    if mask.shape != tuple(raw_img.shape[:2]):
        raise RuntimeError(
            f"SAM3.1 mask shape {mask.shape} does not match image shape "
            f"{tuple(raw_img.shape[:2])}"
        )
    mask_union_ms = elapsed_ms(mask_union_start_s)

    # Origin-parity export: foreground keeps the original pixel values, the
    # alpha channel is 255 inside the mask and 0 (with black RGB) outside.
    output_write_start_s = time.perf_counter()
    h, w = mask.shape
    ref_img = np.zeros((h, w, 4), dtype=np.uint8)
    mask_bool = mask > 0
    ref_img[mask_bool, :3] = raw_img[mask_bool]
    ref_img[:, :, 3] = mask_bool.astype(np.uint8) * 255

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), ref_img):
        raise RuntimeError(f"Unable to write masked image: {output}")
    output_write_ms = elapsed_ms(output_write_start_s)
    timing = {
        "input_read_ms": input_read_ms,
        "inference": dict(inference_timing),
        "mask_union_ms": mask_union_ms,
        "output_write_ms": output_write_ms,
        "total_ms": elapsed_ms(total_start_s),
    }
    return output, timing


def build_parser() -> ArgumentParser:
    """Build the command-line argument parser."""
    parser = ArgumentParser()
    parser.add_argument("--img_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--TEXT_PROMPT", type=str, required=True)
    return parser


def main() -> None:
    """Run the command-line entry point."""
    args = build_parser().parse_args()
    segment_image_to_origin_rgba(
        img_path=args.img_path,
        text_prompt=args.TEXT_PROMPT,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
