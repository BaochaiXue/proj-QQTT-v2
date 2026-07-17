"""Segmentation stage: EdgeTAM streaming over captured frames.

Pipeline questions Q8-Q15 (see PIPELINE.md): warm-up anchors on the FIRST single
frame (``_wait_for_first_frame`` reads it with a sentinel seq of -1 and seeds
EdgeTAM once); later frames are consumed by the tracker but not chunked until the
formal gate lifts. The dominant warm-up cost is the SAM3D shape-prior chain
(submitted here, generated in ``shape_prior_warmup``); warm-up produces the frozen
frame-0 masks + seeded EdgeTAM session + shape-prior artifacts. The formal timeline
starts at output frame 1, one frame after the warm-up frame-0 anchor, once
shape-prior is READY. Warm-up failures route to the shared ``fatal`` latch and
tear the process down (surfaced on the live status band, Q23).
"""

from __future__ import annotations

import argparse
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from demo_v6_2.shape_prior import warmup as shape_prior_warmup
from demo_v6_2.mdp import warmup
from demo_v6_2.mdp.cli import active_object_ids
from demo_v6_2.mdp.constants import (
    DEFAULT_EDGETAM_COMPILE_MODE,
    DEFAULT_EDGETAM_LIVE_SESSION_KEEP_FRAMES,
    DEFAULT_EDGETAM_MASK_LOGIT_THRESHOLD,
    DEFAULT_EDGETAM_MODEL_ID,
    HAND_A_ID,
    HAND_B_ID,
    OBJECT_ID,
)
from demo_v6_2.mdp.packets import FramePacket, MaskPacket
from demo_v6_2.utils.camera import bgr_to_pil_rgb
from demo_v6_2.utils.concurrency import elapsed_ms as _elapsed_ms

from demo_v6_2.mdp.cli import RunMode
from demo_v6_2.mdp.plumbing import (
    FatalErrorLatch,
    LosslessPipeline,
    StageStatsBoard,
)
from demo_v6_2.utils.concurrency import LatestSlot

if TYPE_CHECKING:
    from demo_v6_2.mdp.session import CameraSession
    from demo_v6_2.mdp.warmup_preview import WarmupRgbPreview


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


def _time_runtime_ms(fn: Callable[[], Any]) -> tuple[Any, float]:
    """Run ``fn`` and return ``(value, elapsed_ms)``."""
    started = time.perf_counter()
    value = fn()
    return value, _elapsed_ms(started, time.perf_counter())


def _time_model_forward(fn: Callable[[], Any]) -> tuple[Any, float]:
    """Run a model forward and return ``(value, wall_ms)``."""
    started_s = time.perf_counter()
    value = fn()
    return value, _elapsed_ms(started_s, time.perf_counter())


@dataclass(frozen=True)
class SegmentationWarmupState:
    hf_stream: Any
    torch_module: Any
    dtype: Any
    model: Any
    processor: Any
    first_frame: Any | None
    initial_masks: warmup.InitialMaskBundle | None


class SegmentationStage:
    """EdgeTAM segmentation over captured frames, seeded by frame-0 SAM3.1.

    Owns the warm-up perception profile and the frame-0 seeding sequence
    (EdgeTAM load -> frame-0 wait -> SAM3.1 initial masks).
    """

    def __init__(
        self,
        *,
        args: argparse.Namespace,
        mode: RunMode,
        session: CameraSession,
        lossless: LosslessPipeline,
        capture_slot: LatestSlot[FramePacket],
        mask_slot: LatestSlot[MaskPacket],
        stage_stats: StageStatsBoard,
        shape_prior_manager: shape_prior_warmup.ShapePriorWarmupManager,
        warmup_rgb_preview: WarmupRgbPreview,
        first_frame_segmented: threading.Event,
        stop_event: threading.Event,
        fatal: FatalErrorLatch,
    ) -> None:
        """Initialize SegmentationStage."""
        self.args = args
        self.mode = mode
        self.session = session
        self.lossless = lossless
        self.capture_slot = capture_slot
        self.mask_slot = mask_slot
        self.stage_stats = stage_stats
        self.shape_prior_manager = shape_prior_manager
        self.warmup_rgb_preview = warmup_rgb_preview
        self._first_frame_segmented = first_frame_segmented
        self.stop_event = stop_event
        self.fatal = fatal
        self.warmup_perception_profile: dict[str, Any] = {}
        # Stamped by the composition root when run() begins.
        self.warmup_runtime_start_perf_s: float | None = None

    def _prepare_warmup(self) -> SegmentationWarmupState:
        """Load EdgeTAM, wait for frame 0, and seed SAM3.1 initial masks."""
        prepare_start_s = time.perf_counter()
        hf_stream, torch_module, dtype, model, processor = self._init_hf_model()
        model_ready_s = time.perf_counter()
        frame_wait_start_s = time.perf_counter()
        first_frame = self._wait_for_first_frame()
        frame_wait_end_s = time.perf_counter()
        # first_frame is None when capture shut down before frame 0; the seg
        # worker treats that state as a clean early exit rather than an error.
        if first_frame is None:
            self.warmup_perception_profile["segmentation_warmup"] = {
                "edgetam_init_ms": (model_ready_s - prepare_start_s) * 1000.0,
                "frame_wait_ms": (frame_wait_end_s - frame_wait_start_s) * 1000.0,
                "total_ms": (frame_wait_end_s - prepare_start_s) * 1000.0,
                "frame0_available": False,
            }
            return SegmentationWarmupState(
                hf_stream=hf_stream,
                torch_module=torch_module,
                dtype=dtype,
                model=model,
                processor=processor,
                first_frame=None,
                initial_masks=None,
            )
        initial_masks_start_s = time.perf_counter()
        # Resolve the initial mask bundle by running SAM3.1 on the live first
        # frame.
        expected_shape = tuple(first_frame.color_bgr.shape[:2])
        initial_masks, sam31_timing = warmup.run_sam31_first_frame_mask_bundle(
            first_frame.color_bgr, self.args, self.mode
        )
        if (
            initial_masks.controller_mask.shape != expected_shape
            or initial_masks.object_mask.shape != expected_shape
        ):
            raise RuntimeError(
                "SAM3.1 frame-0 masks do not match captured frame shape"
            )
        initial_masks_end_s = time.perf_counter()
        self.warmup_perception_profile["segmentation_warmup"] = {
            "edgetam_init_ms": (model_ready_s - prepare_start_s) * 1000.0,
            "frame_wait_ms": (frame_wait_end_s - frame_wait_start_s) * 1000.0,
            "initial_mask_bundle_ms": (initial_masks_end_s - initial_masks_start_s)
            * 1000.0,
            "initial_sam31": dict(sam31_timing.timing_ms),
            "sam31_trim_cleanup_ms": float(sam31_timing.trim_cleanup_ms),
            "sam31_release_cleanup_ms": float(sam31_timing.release_cleanup_ms),
            "total_ms": (initial_masks_end_s - prepare_start_s) * 1000.0,
            "frame0_available": True,
        }
        return SegmentationWarmupState(
            hf_stream=hf_stream,
            torch_module=torch_module,
            dtype=dtype,
            model=model,
            processor=processor,
            first_frame=first_frame,
            initial_masks=initial_masks,
        )

    def _init_hf_model(self) -> tuple[Any, Any, Any, Any, Any]:
        """Return the init HF model."""
        init_start_s = time.perf_counter()
        runtime_load_start_s = time.perf_counter()
        hf_stream = _load_hf_streaming_runtime()
        torch_module = hf_stream.torch
        if (
            str(self.args.device).startswith("cuda")
            and not torch_module.cuda.is_available()
        ):
            raise RuntimeError(
                "CUDA device requested but torch.cuda.is_available() is false"
            )
        dtype = hf_stream._dtype_from_name(self.args.dtype)
        runtime_load_end_s = time.perf_counter()
        model_load_start_s = time.perf_counter()
        model = hf_stream.EdgeTamVideoModel.from_pretrained(DEFAULT_EDGETAM_MODEL_ID).to(
            self.args.device,
            dtype=dtype,
        )
        model.eval()
        model_load_end_s = time.perf_counter()
        compile_start_s = time.perf_counter()
        model, compile_metadata = hf_stream._apply_compile_mode(
            model, DEFAULT_EDGETAM_COMPILE_MODE
        )
        compile_end_s = time.perf_counter()
        processor_load_start_s = time.perf_counter()
        processor = hf_stream.Sam2VideoProcessor.from_pretrained(DEFAULT_EDGETAM_MODEL_ID)
        processor_load_end_s = time.perf_counter()
        self.warmup_perception_profile["edgetam_runtime_init"] = {
            "runtime_import_ms": _elapsed_ms(
                runtime_load_start_s,
                runtime_load_end_s,
            ),
            "model_load_ms": _elapsed_ms(
                model_load_start_s,
                model_load_end_s,
            ),
            "compile_ms": _elapsed_ms(compile_start_s, compile_end_s),
            "processor_load_ms": _elapsed_ms(
                processor_load_start_s,
                processor_load_end_s,
            ),
            "total_ms": _elapsed_ms(init_start_s, processor_load_end_s),
        }
        print(
            "[edgetam] "
            f"model={DEFAULT_EDGETAM_MODEL_ID} device={self.args.device} dtype={self.args.dtype} "
            f"track_mode={self.args.track_mode} compile_mode={DEFAULT_EDGETAM_COMPILE_MODE} "
            f"applied={compile_metadata.get('applied_targets', [])}",
            flush=True,
        )
        return hf_stream, torch_module, dtype, model, processor

    def _publish_mask_packet(self, packet: MaskPacket) -> None:
        """Publish raw masks to diagnostics and the canonical formal stage."""
        self.mask_slot.put(packet)
        if self.mode.lossless_enabled:
            self.lossless.submit_mask(packet, stop_event=self.stop_event)

    def run(self) -> None:
        """EdgeTAM segmentation worker loop."""
        try:
            warmup_state = self._prepare_warmup()
            first_frame = warmup_state.first_frame
            if first_frame is None:
                return
            initial_masks = warmup_state.initial_masks
            if initial_masks is None:
                raise RuntimeError("segmentation warmup did not produce frame-0 masks")
            session_start_s = time.perf_counter()
            session = warmup_state.hf_stream.EdgeTamVideoInferenceSession(
                video=None,
                video_height=int(first_frame.color_bgr.shape[0]),
                video_width=int(first_frame.color_bgr.shape[1]),
                inference_device=self.args.device,
                inference_state_device=self.args.device,
                video_storage_device=self.args.device,
                dtype=warmup_state.dtype,
            )
            session_end_s = time.perf_counter()
            self.warmup_perception_profile["edgetam_session_init_ms"] = _elapsed_ms(
                session_start_s, session_end_s
            )
            with warmup_state.torch_module.inference_mode():
                first_packet = self._run_segmentation_frame(
                    hf_stream=warmup_state.hf_stream,
                    torch_module=warmup_state.torch_module,
                    dtype=warmup_state.dtype,
                    model=warmup_state.model,
                    processor=warmup_state.processor,
                    session=session,
                    frame=first_frame,
                    initial_masks=initial_masks,
                    add_prompt=True,
                )
                self._publish_mask_packet(first_packet)
                self.stage_stats.record("seg", first_packet.process_done_perf_s)
                if self.mode.lossless_enabled or self.mode.fake_live_input:
                    self._first_frame_segmented.set()
                if not self.shape_prior_manager.enabled:
                    # Without shape-prior warm-up the frame-0 seed IS the whole
                    # warm-up, so the live RGB preview closes here; with it the
                    # preview closes at the WARMUP_FINISHED banner instead.
                    self.warmup_rgb_preview.close()
                last_seq = first_frame.seq
                while not self.stop_event.is_set():
                    if self.mode.lossless_enabled:
                        frame = self.lossless.frame_queue.get(
                            stop_event=self.stop_event
                        )
                        if frame is None:
                            break
                    else:
                        frame = self.capture_slot.get_latest_after(last_seq)
                        if frame is None:
                            time.sleep(0.001)
                            continue
                    last_seq = frame.seq
                    try:
                        packet = self._run_segmentation_frame(
                            hf_stream=warmup_state.hf_stream,
                            torch_module=warmup_state.torch_module,
                            dtype=warmup_state.dtype,
                            model=warmup_state.model,
                            processor=warmup_state.processor,
                            session=session,
                            frame=frame,
                            initial_masks=initial_masks,
                            add_prompt=False,
                        )
                    except Exception as exc:
                        self.fatal.record("EdgeTAM segmentation", exc)
                        break
                    self._publish_mask_packet(packet)
                    self.stage_stats.record("seg", packet.process_done_perf_s)
                if self.mode.lossless_enabled:
                    self.lossless.mask_queue.close()
        except Exception as exc:
            if not self.stop_event.is_set():
                self.fatal.record("segmentation worker", exc)
            if self.mode.lossless_enabled:
                self.lossless.mask_queue.close()

    def _wait_for_first_frame(self) -> FramePacket | None:
        """Wait for for first frame."""
        if self.mode.lossless_enabled:
            return self.lossless.frame_queue.get(stop_event=self.stop_event)
        while not self.stop_event.is_set():
            frame = self.capture_slot.get_latest_after(-1)
            if frame is not None:
                return frame
            time.sleep(0.005)
        return None

    def _autocast_context(self, torch_module: Any) -> Any:
        """Return the autocast context."""
        if not str(self.args.device).startswith("cuda") or self.args.dtype == "float32":
            return nullcontext()
        dtype = (
            torch_module.bfloat16
            if self.args.dtype == "bfloat16"
            else torch_module.float16
        )
        return torch_module.autocast("cuda", dtype=dtype)

    def _prune_edgetam_live_session(
        self, session: Any, *, current_frame_idx: int
    ) -> None:
        """Prune edgetam live session."""
        keep_frames = int(DEFAULT_EDGETAM_LIVE_SESSION_KEEP_FRAMES)
        min_frame_idx = int(current_frame_idx) - keep_frames + 1

        processed_frames = getattr(session, "processed_frames", None)
        if isinstance(processed_frames, dict):
            for frame_idx in list(processed_frames):
                if int(frame_idx) < min_frame_idx:
                    processed_frames.pop(frame_idx, None)

        output_dict_per_obj = getattr(session, "output_dict_per_obj", None)
        if isinstance(output_dict_per_obj, dict):
            for output_dict in output_dict_per_obj.values():
                if not isinstance(output_dict, dict):
                    continue
                non_cond_outputs = output_dict.get("non_cond_frame_outputs")
                if isinstance(non_cond_outputs, dict):
                    for frame_idx in list(non_cond_outputs):
                        if int(frame_idx) < min_frame_idx:
                            non_cond_outputs.pop(frame_idx, None)

        frames_tracked_per_obj = getattr(session, "frames_tracked_per_obj", None)
        if isinstance(frames_tracked_per_obj, dict):
            for tracked_frames in frames_tracked_per_obj.values():
                if not isinstance(tracked_frames, dict):
                    continue
                for frame_idx in list(tracked_frames):
                    if int(frame_idx) < min_frame_idx:
                        tracked_frames.pop(frame_idx, None)

    def _run_segmentation_frame(
        self,
        *,
        hf_stream: Any,
        torch_module: Any,
        dtype: Any,
        model: Any,
        processor: Any,
        session: Any,
        frame: FramePacket,
        initial_masks: warmup.InitialMaskBundle,
        add_prompt: bool,
    ) -> MaskPacket:
        """Run segmentation frame."""
        image = bgr_to_pil_rgb(frame.color_bgr)
        inputs, preprocess_ms = _time_runtime_ms(
            lambda: processor(
                images=image, device=self.args.device, return_tensors="pt"
            ),
        )
        pixel_values = inputs.pixel_values[0].to(device=self.args.device, dtype=dtype)
        prompt_ms = 0.0
        with self._autocast_context(torch_module):
            if add_prompt:
                prompt_obj_ids: list[int] = []
                prompt_masks: list[np.ndarray] = []
                if self.mode.controller_tracking_enabled:
                    prompt_obj_ids.append(HAND_A_ID)
                    prompt_masks.append(
                        np.asarray(initial_masks.hand_a_mask, dtype=bool)
                    )
                if self.mode.object_tracking_enabled:
                    prompt_obj_ids.append(OBJECT_ID)
                    prompt_masks.append(
                        np.asarray(initial_masks.object_mask, dtype=bool)
                    )
                if self.mode.controller_tracking_enabled:
                    prompt_obj_ids.append(HAND_B_ID)
                    prompt_masks.append(
                        np.asarray(initial_masks.hand_b_mask, dtype=bool)
                    )
                _unused, prompt_ms = _time_runtime_ms(
                    lambda: processor.add_inputs_to_inference_session(
                        inference_session=session,
                        frame_idx=int(frame.seq),
                        obj_ids=prompt_obj_ids,
                        input_masks=prompt_masks,
                    ),
                )
            output, wall_model_ms = _time_model_forward(
                lambda: model(
                    inference_session=session,
                    frame=pixel_values,
                    frame_idx=int(frame.seq),
                ),
            )
            post_masks, postprocess_ms = _time_runtime_ms(
                lambda: processor.post_process_masks(
                    [output.pred_masks],
                    original_sizes=inputs.original_sizes,
                    binarize=False,
                )[0],
            )
        masks_by_id = extract_object_masks_from_hf_output(
            output,
            post_masks,
            mask_logit_threshold=float(self.args.edgetam_mask_logit_threshold),
        )
        missing = [
            obj_id
            for obj_id in active_object_ids(self.args)
            if obj_id not in masks_by_id
        ]
        if missing:
            raise RuntimeError(f"HF output missing tracked object ids: {missing}")
        reference_mask = next(iter(masks_by_id.values()))
        object_mask = masks_by_id.get(OBJECT_ID)
        if object_mask is None:
            object_mask = np.zeros_like(reference_mask, dtype=bool)
        hand_a_mask = masks_by_id.get(HAND_A_ID)
        if hand_a_mask is None:
            hand_a_mask = np.zeros_like(reference_mask, dtype=bool)
        hand_b_mask = masks_by_id.get(HAND_B_ID)
        if hand_b_mask is None:
            hand_b_mask = np.zeros_like(reference_mask, dtype=bool)
        controller_mask = np.logical_or(hand_a_mask, hand_b_mask)
        self._prune_edgetam_live_session(
            session, current_frame_idx=int(output.frame_idx)
        )
        process_done_s = time.perf_counter()
        timing = replace(
            frame.timing,
            preprocess_ms=preprocess_ms,
            prompt_ms=prompt_ms,
            model_ms=wall_model_ms,
            wall_model_ms=wall_model_ms,
            postprocess_ms=postprocess_ms,
            mask_ms=float(preprocess_ms + prompt_ms + wall_model_ms + postprocess_ms),
        )
        return MaskPacket(
            seq=frame.seq,
            color_bgr=frame.color_bgr,
            depth_source=frame.depth_source,
            intrinsics=frame.intrinsics,
            depth_scale_m_per_unit=frame.depth_scale_m_per_unit,
            receive_perf_s=frame.receive_perf_s,
            process_done_perf_s=process_done_s,
            dropped_capture_frames=self.capture_slot.dropped_count,
            timing=timing,
            controller_mask=np.ascontiguousarray(controller_mask, dtype=bool),
            object_mask=np.ascontiguousarray(object_mask, dtype=bool),
            hand_a_mask=np.ascontiguousarray(hand_a_mask, dtype=bool),
            hand_b_mask=np.ascontiguousarray(hand_b_mask, dtype=bool),
            depth_u16=frame.depth_u16,
            ir_left_u8=frame.ir_left_u8,
            ir_right_u8=frame.ir_right_u8,
            k_ir_left=frame.k_ir_left,
            t_ir_left_to_color=frame.t_ir_left_to_color,
            k_color=frame.k_color,
            ir_baseline_m=frame.ir_baseline_m,
            source_timestamp_s=frame.source_timestamp_s,
            source_frame_index=frame.source_frame_index,
            source_step=frame.source_step,
        )


__all__ = ["SegmentationStage", "SegmentationWarmupState"]
