"""Segmentation (EdgeTAM) helpers & model timing."""
from __future__ import annotations

from demo_v6_2.mdp_constants import *  # noqa: F401,F403

def extract_object_masks_from_hf_output(
    output: Any,
    post_masks: Any,
    *,
    mask_logit_threshold: float = DEFAULT_EDGETAM_MASK_LOGIT_THRESHOLD,
) -> dict[int, np.ndarray]:
    # HF EdgeTAM may hand back object ids as a torch tensor, ndarray, scalar, or list.
    """Extract object masks from HF output."""
    ids_value = getattr(output, "object_ids")
    if hasattr(ids_value, "detach"):
        ids_value = ids_value.detach().cpu().tolist()
    if isinstance(ids_value, np.ndarray):
        ids_value = ids_value.tolist()
    if isinstance(ids_value, (int, np.integer)):
        object_ids = [int(ids_value)]
    else:
        object_ids = [int(item) for item in list(ids_value)]
    if len(object_ids) != len(post_masks):
        raise RuntimeError(f"HF output object_ids length {len(object_ids)} != mask length {len(post_masks)}")
    masks: dict[int, np.ndarray] = {}
    for idx, obj_id in enumerate(object_ids):
        # Masks may be GPU tensors with singleton dims; normalize each to a contiguous HxW bool array.
        value = post_masks[idx]
        if hasattr(value, "detach"):
            value = value.detach().float().cpu().numpy()
        array = np.squeeze(np.asarray(value))
        if array.ndim != 2:
            raise RuntimeError(f"expected 2D mask after squeeze, got {array.shape}")
        masks[int(obj_id)] = np.ascontiguousarray(
            array > float(mask_logit_threshold)
        )
    return masks


def _load_hf_streaming_runtime() -> Any:
    """Load HF streaming runtime."""
    from scripts.harness.experiments.edgetam import run_hf_edgetam_streaming_realcase as hf_stream

    hf_stream._load_runtime_dependencies()
    return hf_stream


def _time_runtime_ms(
    fn: Callable[[], Any],
) -> tuple[Any, float, float, float]:
    """Measure the wall time of ``fn`` in milliseconds.

    Returns ``(value, elapsed_ms, pre_sync_ms, post_sync_ms)``; the sync fields
    stay 0.0 (the optional CUDA-sync profiling was CLI-gated and unreachable).
    """
    started = time.perf_counter()
    value = fn()
    elapsed_ms = _elapsed_ms(started, time.perf_counter())
    return value, elapsed_ms, 0.0, 0.0


def _time_model_forward(fn: Callable[[], Any]) -> tuple[Any, float, float, float, float]:
    """Measure the wall time of a model forward in milliseconds.

    Returns ``(value, wall_ms, cuda_event_ms, pre_sync_ms, post_sync_ms)``; the
    cuda-event and sync fields stay 0.0 (their profiling was CLI-gated and
    unreachable).
    """
    started_s = time.perf_counter()
    value = fn()
    wall_ms = _elapsed_ms(started_s, time.perf_counter())
    return value, wall_ms, 0.0, 0.0, 0.0


__all__ = [
    "extract_object_masks_from_hf_output",
    "_load_hf_streaming_runtime",
    "_time_runtime_ms",
    "_time_model_forward",
]
