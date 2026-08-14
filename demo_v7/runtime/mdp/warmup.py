"""Frame-0 SAM3.1 seed for the Demo v6.2 camera subprocess: mask bundling,
controller hand splitting, and CUDA cleanup."""

from __future__ import annotations

import argparse
import gc
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from demo_v7.runtime.utils.camera import bgr_to_pil_rgb

if TYPE_CHECKING:
    from demo_v7.runtime.mdp.cli import RunMode

DEFAULT_SAM31_DEVICE = "cuda"


@dataclass(frozen=True)
class InitialMaskBundle:
    controller_mask: np.ndarray
    object_mask: np.ndarray
    hand_a_mask: np.ndarray | None = None
    hand_b_mask: np.ndarray | None = None


@dataclass(frozen=True)
class Sam31FrameTiming:
    """Frame-0 SAM3.1 timings, threaded into the warm-up perception profile."""

    timing_ms: dict[str, Any] = field(default_factory=dict)
    trim_cleanup_ms: float = 0.0
    release_cleanup_ms: float = 0.0


# ---------------------------------------------------------------------------
# Mask post-processing (label union, controller hand splitting)
# ---------------------------------------------------------------------------


def _union_masks(masks: list[np.ndarray], *, label: str) -> np.ndarray:
    # SAM3.1 returns one mask per detected instance; downstream tracking wants a
    # single foreground mask per label, so OR all instances together.
    """Return the union masks."""
    if not masks:
        raise RuntimeError(f"SAM3.1 did not produce a mask for label {label!r}")
    output = np.zeros_like(masks[0], dtype=bool)
    for mask in masks:
        if mask.shape != output.shape:
            raise RuntimeError("SAM3.1 masks for one label have inconsistent shapes")
        output |= mask.astype(bool)
    return np.ascontiguousarray(output)


def _mask_area(mask: np.ndarray) -> int:
    """Return the mask area."""
    return int(np.count_nonzero(np.asarray(mask, dtype=bool)))


def _mask_centroid_x(mask: np.ndarray) -> float:
    # Sort key for left-to-right hand ordering; empty masks sort last (+inf).
    """Return the mask centroid x."""
    coords = np.argwhere(np.asarray(mask, dtype=bool))
    if coords.size == 0:
        return float("inf")
    return float(coords[:, 1].mean())


def _connected_components_by_area(mask: np.ndarray) -> list[np.ndarray]:
    """Return 8-connected components sorted largest-first."""
    mask_bool = np.asarray(mask, dtype=bool)
    if not np.any(mask_bool):
        return []
    import cv2  # noqa: PLC0415

    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask_bool.astype(np.uint8),
        8,
    )
    components: list[tuple[int, np.ndarray]] = []
    for label_idx in range(1, int(count)):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area > 0:
            components.append((area, labels == label_idx))
    components.sort(key=lambda item: item[0], reverse=True)
    return [
        np.ascontiguousarray(component, dtype=bool) for _area, component in components
    ]


def split_controller_hand_instances(
    controller_masks: list[np.ndarray],
    *,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the split controller hand instances."""
    masks = [
        np.ascontiguousarray(mask, dtype=bool)
        for mask in controller_masks
        if _mask_area(mask) > 0
    ]
    # Two+ instance masks: keep the two largest. A single (merged) mask: split
    # it into connected components and keep the two largest.
    if len(masks) >= 2:
        candidates = sorted(masks, key=_mask_area, reverse=True)[:2]
    elif len(masks) == 1:
        candidates = _connected_components_by_area(masks[0])[:2]
    else:
        candidates = []
    if len(candidates) < 2:
        raise RuntimeError(
            f"SAM3.1 did not produce two separable controller masks for {label!r}; "
            "three-identity demo requires two visible hands in frame 0"
        )
    # Stable identity assignment: hand A is the leftmost hand in the image.
    candidates = sorted(candidates, key=_mask_centroid_x)
    return (
        np.ascontiguousarray(candidates[0], dtype=bool),
        np.ascontiguousarray(candidates[1], dtype=bool),
    )


# ---------------------------------------------------------------------------
# SAM3.1 runtime cleanup
# ---------------------------------------------------------------------------
# Two levels of cleanup: release_* drops the cached model runtime and
# empties the CUDA allocator; a plain allocator trim (_reclaim_cuda_memory)
# keeps the cached model alive (for the next camera or shape-prior warmup) and
# only returns freed blocks to CUDA.


def _reclaim_cuda_memory(
    device: str,
    *,
    warn_context: str,
    ipc_collect: bool = False,
) -> float:
    """Run gc.collect + CUDA synchronize/empty_cache, warning on failure.

    ``warn_context`` names the operation in the ``[WARN] SAM3.1 <ctx> failed``
    message; ``ipc_collect`` additionally returns inter-process CUDA blocks.
    Returns elapsed milliseconds for this reclaim body.
    """
    started_s = time.perf_counter()
    gc.collect()
    try:
        import torch  # noqa: PLC0415

        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if ipc_collect and hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception as exc:
        print(
            f"[WARN] SAM3.1 {warn_context} failed: {type(exc).__name__}: {exc}",
            flush=True,
        )
    return (time.perf_counter() - started_s) * 1000.0


def release_sam31_runtime_resources(device: str = DEFAULT_SAM31_DEVICE) -> float:
    """Return the release sam31 runtime resources."""
    from demo_v7.runtime.perception import sam31_image_segmentation

    started_s = time.perf_counter()
    try:
        sam31_image_segmentation.release_sam31_image_segmentation_runtime()
    except Exception as exc:
        print(
            f"[WARN] SAM3.1 runtime cleanup failed: {type(exc).__name__}: {exc}",
            flush=True,
        )

    _reclaim_cuda_memory(device, warn_context="CUDA cleanup", ipc_collect=True)
    return (time.perf_counter() - started_s) * 1000.0


# ---------------------------------------------------------------------------
# SAM3.1 model preload (overlaps EdgeTAM load + frame-0 wait)
# ---------------------------------------------------------------------------


class Sam31PreloadThread:
    """Load the SAM3.1 model on a daemon thread while frame 0 is still pending.

    Today's frame-0 sequence pays the SAM3.1 checkpoint load only after frame 0
    arrives; this thread moves that load to seg-worker start so it overlaps the
    EdgeTAM load and the camera's first frame. ``wait_for_model`` re-raises any
    load failure at the exact point the lazy in-line build would have raised.
    """

    def __init__(self, *, device: str) -> None:
        """Initialize Sam31PreloadThread."""
        self._device = str(device)
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._preload_ms: float = 0.0

    def start(self) -> None:
        """Start the background load; safe to call at most once."""
        if self._thread is not None:
            raise RuntimeError("SAM3.1 preload already started")
        self._thread = threading.Thread(
            target=self._run,
            name="sam31-preload",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        from demo_v7.runtime.perception.sam31_image_segmentation import (  # noqa: PLC0415
            preload_sam31_image_runtime,
        )

        started_s = time.perf_counter()
        try:
            preload_sam31_image_runtime(device=self._device)
        except BaseException as exc:  # re-raised on the seg worker in wait
            self._error = exc
        finally:
            self._preload_ms = (time.perf_counter() - started_s) * 1000.0

    def wait_done(self, timeout: float) -> bool:
        """Wait for the load to finish without re-raising; True when done.

        Readiness-barrier probe only — failures still re-raise at
        ``wait_for_model`` on the seg worker.
        """
        if self._thread is None:
            return True
        self._thread.join(max(0.0, float(timeout)))
        return not self._thread.is_alive()

    def wait_for_model(self) -> dict[str, float]:
        """Block until the load finishes; re-raise its failure; return timings."""
        if self._thread is None:
            return {"preload_ms": 0.0, "join_wait_ms": 0.0}
        join_started_s = time.perf_counter()
        self._thread.join()
        join_wait_ms = (time.perf_counter() - join_started_s) * 1000.0
        if self._error is not None:
            raise self._error
        return {
            "preload_ms": float(self._preload_ms),
            "join_wait_ms": float(join_wait_ms),
        }


# ---------------------------------------------------------------------------
# SAM3.1 frame-0 segmentation and initial-mask resolution
# ---------------------------------------------------------------------------


def run_sam31_first_frame_mask_bundle(
    color_bgr: np.ndarray,
    args: argparse.Namespace,
    mode: RunMode,
    *,
    reuse_sam31_runtime: bool,
    defer_release: bool = False,
) -> tuple[InitialMaskBundle, Sam31FrameTiming]:
    """Run SAM3.1 on frame 0 and return the mask bundle plus its timings.

    ``defer_release=True`` (cache-hit path only) skips the ~0.3s release
    cleanup here so the caller can run it off the frame-0 critical path,
    strictly after the frame-0 EdgeTAM forward; exception paths still release
    inline so a failed warm-up never leaks the SAM3.1 runtime.
    """
    from demo_v7.runtime.perception.sam31_image_segmentation import (
        parse_text_prompts,
        run_image_segmentation,
    )

    # Prompt order matters: run_image_segmentation keeps only the best instance
    # for the first prompt of a multi-prompt request and all instances for the
    # later ones, so the object prompt must come before the controller prompt
    # (one object mask, both hand instances preserved).
    object_prompt = str(args.shape_prior_object_prompt)
    prompt_labels = []
    if mode.object_tracking_enabled:
        prompt_labels.append(object_prompt)
    if mode.controller_tracking_enabled:
        prompt_labels.append(str("hand"))
    if not prompt_labels:
        empty = np.zeros(tuple(color_bgr.shape[:2]), dtype=bool)
        return (
            InitialMaskBundle(controller_mask=empty, object_mask=empty),
            Sam31FrameTiming(),
        )
    text_prompt = ",".join(prompt_labels)
    # reuse_model=True consumes the Sam31PreloadThread cache entry (or caches
    # the lazily built model). Whether the model SURVIVES this call is decided
    # by reuse_sam31_runtime below: a mesh-cache miss keeps it for the second
    # segmentation (trim only); a mesh-cache hit releases it entirely.
    reuse_sam31_runtime = bool(reuse_sam31_runtime)
    trim_cleanup_ms = 0.0
    release_cleanup_ms = 0.0
    try:
        result = run_image_segmentation(
            image=bgr_to_pil_rgb(color_bgr),
            text_prompt=text_prompt,
            checkpoint_path=None,
            compile_model=False,
            max_num_objects=16,
            device=str(args.device),
            reuse_model=True,
        )
    except BaseException:
        if reuse_sam31_runtime:
            _reclaim_cuda_memory(str(args.device), warn_context="CUDA trim")
        else:
            release_sam31_runtime_resources(str(args.device))
        raise
    if reuse_sam31_runtime:
        trim_cleanup_ms = _reclaim_cuda_memory(
            str(args.device), warn_context="CUDA trim"
        )
    elif not defer_release:
        release_cleanup_ms = release_sam31_runtime_resources(str(args.device))
    timing = Sam31FrameTiming(
        timing_ms=dict(result.get("timing_ms", {}) or {}),
        trim_cleanup_ms=float(trim_cleanup_ms),
        release_cleanup_ms=float(release_cleanup_ms),
    )

    masks_by_label = result["masks_by_label"]
    object_mask: np.ndarray | None = None
    controller_mask: np.ndarray | None = None
    controller_masks: list[np.ndarray] = []
    if mode.object_tracking_enabled:
        object_label = parse_text_prompts(object_prompt)[0]
        object_mask = _union_masks(
            list(masks_by_label.get(object_label, [])),
            label=object_prompt,
        )
    if mode.controller_tracking_enabled:
        controller_label = parse_text_prompts(str("hand"))[0]
        controller_masks = list(masks_by_label.get(controller_label, []))
        controller_mask = _union_masks(
            controller_masks,
            label="hand",
        )
    # Disabled identities get all-false masks of the enabled identity's shape;
    # object-only mode returns early because there are no hands to split.
    if object_mask is None and controller_mask is None:
        empty = np.zeros(tuple(color_bgr.shape[:2]), dtype=bool)
        return InitialMaskBundle(controller_mask=empty, object_mask=empty), timing
    if object_mask is None:
        object_mask = np.zeros_like(controller_mask, dtype=bool)
    if controller_mask is None:
        empty_controller = np.zeros_like(object_mask, dtype=bool)
        return (
            InitialMaskBundle(
                controller_mask=empty_controller,
                object_mask=object_mask,
            ),
            timing,
        )
    # The published controller mask is rebuilt from the two hand instances so
    # it stays consistent with hand_a/hand_b even if SAM3.1 returned extras.
    hand_a_mask, hand_b_mask = split_controller_hand_instances(
        controller_masks,
        label=str("hand"),
    )
    controller_mask = np.logical_or(hand_a_mask, hand_b_mask)
    return (
        InitialMaskBundle(
            controller_mask=np.ascontiguousarray(controller_mask, dtype=bool),
            object_mask=np.ascontiguousarray(object_mask, dtype=bool),
            hand_a_mask=np.ascontiguousarray(hand_a_mask, dtype=bool),
            hand_b_mask=np.ascontiguousarray(hand_b_mask, dtype=bool),
        ),
        timing,
    )
