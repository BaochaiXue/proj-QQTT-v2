"""Main warmup helpers for Demo v5.1 realtime capture and perception."""

from __future__ import annotations

import argparse
import gc
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from qqtt.demo.pcd_filter_fast import AsyncPcdFilterWorker
from qqtt.demo.realtime_single_camera_pointcloud import (
    apply_wslg_open3d_env_defaults,
    build_projection_grid,
    warm_up_numba_ffs_align,
)
from services.ffs_remote import FfsRemoteDepthClient

TRACK_MODE_CONTROLLER_OBJECT = "controller-object"
TRACK_MODE_OBJECT_ONLY = "object-only"
TRACK_MODE_CONTROLLER_ONLY = "controller-only"
CONTROLLER_INSTANCE_MODE_SINGLE = "single"
CONTROLLER_INSTANCE_MODE_TWO_HANDS = "two-hands"
INIT_MODE_SAM31_FIRST_FRAME = "sam31-first-frame"
INIT_MODE_SAVED_MASKS = "saved-masks"
DEFAULT_SAM31_DEVICE = "cuda"


@dataclass(frozen=True)
class InitialMaskBundle:
    controller_mask: np.ndarray
    object_mask: np.ndarray
    hand_a_mask: np.ndarray | None = None
    hand_b_mask: np.ndarray | None = None


@dataclass(frozen=True)
class SegmentationWarmupState:
    hf_stream: Any
    torch_module: Any
    dtype: Any
    model: Any
    processor: Any
    first_frame: Any | None
    initial_masks: InitialMaskBundle | None


def _resolve_path(value: str | Path, *, repo_root: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path(repo_root) / path).resolve()


def _load_gray_image(path: Path) -> np.ndarray:
    try:
        from PIL import Image

        return np.asarray(Image.open(path).convert("L"))
    except Exception as exc:
        raise ValueError(f"failed to load mask image {path}: {exc}") from exc


def load_binary_mask(
    path: str | Path,
    *,
    expected_shape: tuple[int, int],
    repo_root: Path,
) -> np.ndarray:
    mask_path = _resolve_path(path, repo_root=repo_root)
    image = _load_gray_image(mask_path)
    if image.ndim != 2:
        raise ValueError(f"mask must be a 2D image: {mask_path}")
    if tuple(image.shape) != tuple(expected_shape):
        raise ValueError(
            f"mask shape {tuple(image.shape)} does not match frame shape "
            f"{tuple(expected_shape)}: {mask_path}"
        )
    return np.ascontiguousarray(image > 0)


def object_tracking_enabled(args_or_track_mode: argparse.Namespace | str) -> bool:
    track_mode = (
        args_or_track_mode
        if isinstance(args_or_track_mode, str)
        else args_or_track_mode.track_mode
    )
    return str(track_mode) in {TRACK_MODE_CONTROLLER_OBJECT, TRACK_MODE_OBJECT_ONLY}


def controller_tracking_enabled(args_or_track_mode: argparse.Namespace | str) -> bool:
    track_mode = (
        args_or_track_mode
        if isinstance(args_or_track_mode, str)
        else args_or_track_mode.track_mode
    )
    return str(track_mode) in {
        TRACK_MODE_CONTROLLER_OBJECT,
        TRACK_MODE_CONTROLLER_ONLY,
    }


def three_identity_controller_enabled(args: argparse.Namespace) -> bool:
    return bool(
        controller_tracking_enabled(args)
        and str(
            getattr(args, "controller_instance_mode", CONTROLLER_INSTANCE_MODE_SINGLE)
        )
        == CONTROLLER_INSTANCE_MODE_TWO_HANDS
    )


def bgr_to_pil_rgb(color_bgr: np.ndarray) -> Any:
    from PIL import Image

    return Image.fromarray(np.ascontiguousarray(color_bgr[:, :, ::-1]))


def _union_masks(masks: list[np.ndarray], *, label: str) -> np.ndarray:
    if not masks:
        raise RuntimeError(f"SAM3.1 did not produce a mask for label {label!r}")
    output = np.zeros_like(masks[0], dtype=bool)
    for mask in masks:
        if mask.shape != output.shape:
            raise RuntimeError("SAM3.1 masks for one label have inconsistent shapes")
        output |= mask.astype(bool)
    return np.ascontiguousarray(output)


def _mask_area(mask: np.ndarray) -> int:
    return int(np.count_nonzero(np.asarray(mask, dtype=bool)))


def _mask_centroid_x(mask: np.ndarray) -> float:
    coords = np.argwhere(np.asarray(mask, dtype=bool))
    if coords.size == 0:
        return float("inf")
    return float(coords[:, 1].mean())


def _connected_components_by_area(mask: np.ndarray) -> list[np.ndarray]:
    mask_bool = np.asarray(mask, dtype=bool)
    if not np.any(mask_bool):
        return []
    try:
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
            np.ascontiguousarray(component, dtype=bool)
            for _area, component in components
        ]
    except Exception:
        height, width = mask_bool.shape[:2]
        seen = np.zeros_like(mask_bool, dtype=bool)
        components = []
        for start_y, start_x in np.argwhere(mask_bool):
            if seen[start_y, start_x]:
                continue
            stack = [(int(start_y), int(start_x))]
            seen[start_y, start_x] = True
            coords: list[tuple[int, int]] = []
            while stack:
                y, x = stack.pop()
                coords.append((y, x))
                for ny in (y - 1, y, y + 1):
                    for nx in (x - 1, x, x + 1):
                        if ny == y and nx == x:
                            continue
                        if (
                            0 <= ny < height
                            and 0 <= nx < width
                            and mask_bool[ny, nx]
                            and not seen[ny, nx]
                        ):
                            seen[ny, nx] = True
                            stack.append((ny, nx))
            component = np.zeros_like(mask_bool, dtype=bool)
            yy, xx = np.asarray(coords, dtype=np.int64).T
            component[yy, xx] = True
            components.append(component)
        components.sort(key=_mask_area, reverse=True)
        return [np.ascontiguousarray(component, dtype=bool) for component in components]


def split_controller_hand_instances(
    controller_masks: list[np.ndarray],
    *,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    masks = [
        np.ascontiguousarray(mask, dtype=bool)
        for mask in controller_masks
        if _mask_area(mask) > 0
    ]
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
    candidates = sorted(candidates, key=_mask_centroid_x)
    return (
        np.ascontiguousarray(candidates[0], dtype=bool),
        np.ascontiguousarray(candidates[1], dtype=bool),
    )


def release_sam31_runtime_resources(device: str = DEFAULT_SAM31_DEVICE) -> float:
    from demo_v5_1 import sam31_image_segmentation

    started_s = time.perf_counter()
    try:
        sam31_image_segmentation.release_sam31_image_segmentation_runtime()
    except Exception as exc:
        print(
            f"[WARN] SAM3.1 runtime cleanup failed: {type(exc).__name__}: {exc}",
            flush=True,
        )

    gc.collect()
    try:
        import torch  # noqa: PLC0415

        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception as exc:
        print(
            f"[WARN] SAM3.1 CUDA cleanup failed: {type(exc).__name__}: {exc}",
            flush=True,
        )
    return (time.perf_counter() - started_s) * 1000.0


def trim_sam31_cuda_allocator(device: str = DEFAULT_SAM31_DEVICE) -> float:
    started_s = time.perf_counter()
    gc.collect()
    try:
        import torch  # noqa: PLC0415

        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception as exc:
        print(
            f"[WARN] SAM3.1 CUDA trim failed: {type(exc).__name__}: {exc}",
            flush=True,
        )
    return (time.perf_counter() - started_s) * 1000.0


def run_sam31_first_frame_mask_bundle(
    color_bgr: np.ndarray,
    args: argparse.Namespace,
) -> InitialMaskBundle:
    from demo_v5_1.sam31_image_segmentation import (
        parse_text_prompts,
        run_image_segmentation,
    )

    prompt_labels = []
    if object_tracking_enabled(args):
        prompt_labels.append(str(args.object_prompt))
    if controller_tracking_enabled(args):
        prompt_labels.append(str(args.controller_prompt))
    if not prompt_labels:
        empty = np.zeros(tuple(color_bgr.shape[:2]), dtype=bool)
        return InitialMaskBundle(controller_mask=empty, object_mask=empty)
    text_prompt = ",".join(prompt_labels)
    reuse_sam31_runtime = bool(
        getattr(args, "sam31_cache_init_model", False)
        or getattr(args, "shape_prior_warmup", False)
    )
    keep_runtime_until_all_cameras_init = bool(
        getattr(args, "sam31_keep_runtime_until_all_cameras_init", False)
        or getattr(args, "shape_prior_warmup", False)
    )
    try:
        result = run_image_segmentation(
            image=bgr_to_pil_rgb(color_bgr),
            text_prompt=text_prompt,
            checkpoint_path=None,
            compile_model=False,
            max_num_objects=16,
            device=str(args.device),
            reuse_model=reuse_sam31_runtime,
        )
        setattr(args, "_sam31_last_timing_ms", result.get("timing_ms", {}))
    finally:
        if keep_runtime_until_all_cameras_init:
            trim_ms = trim_sam31_cuda_allocator(str(args.device))
            setattr(args, "_sam31_last_trim_cleanup_ms", float(trim_ms))
        else:
            release_ms = release_sam31_runtime_resources(str(args.device))
            setattr(args, "_sam31_last_release_cleanup_ms", float(release_ms))

    masks_by_label = result["masks_by_label"]
    object_mask: np.ndarray | None = None
    controller_mask: np.ndarray | None = None
    controller_masks: list[np.ndarray] = []
    if object_tracking_enabled(args):
        object_label = parse_text_prompts(str(args.object_prompt))[0]
        object_mask = _union_masks(
            list(masks_by_label.get(object_label, [])),
            label=args.object_prompt,
        )
    if controller_tracking_enabled(args):
        controller_label = parse_text_prompts(str(args.controller_prompt))[0]
        controller_masks = list(masks_by_label.get(controller_label, []))
        controller_mask = _union_masks(
            controller_masks,
            label=args.controller_prompt,
        )
    if object_mask is None and controller_mask is None:
        empty = np.zeros(tuple(color_bgr.shape[:2]), dtype=bool)
        return InitialMaskBundle(controller_mask=empty, object_mask=empty)
    if object_mask is None:
        object_mask = np.zeros_like(controller_mask, dtype=bool)
    if controller_mask is None:
        empty_controller = np.zeros_like(object_mask, dtype=bool)
        return InitialMaskBundle(
            controller_mask=empty_controller,
            object_mask=object_mask,
        )
    if three_identity_controller_enabled(args):
        hand_a_mask, hand_b_mask = split_controller_hand_instances(
            controller_masks,
            label=str(args.controller_prompt),
        )
        controller_mask = np.logical_or(hand_a_mask, hand_b_mask)
    else:
        hand_a_mask = np.ascontiguousarray(controller_mask, dtype=bool)
        hand_b_mask = np.zeros_like(hand_a_mask, dtype=bool)
    return InitialMaskBundle(
        controller_mask=np.ascontiguousarray(controller_mask, dtype=bool),
        object_mask=np.ascontiguousarray(object_mask, dtype=bool),
        hand_a_mask=np.ascontiguousarray(hand_a_mask, dtype=bool),
        hand_b_mask=np.ascontiguousarray(hand_b_mask, dtype=bool),
    )


def run_sam31_first_frame_masks(
    color_bgr: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    bundle = run_sam31_first_frame_mask_bundle(color_bgr, args)
    return bundle.controller_mask, bundle.object_mask


def resolve_initial_mask_bundle(
    frame: Any,
    args: argparse.Namespace,
    *,
    repo_root: Path,
) -> InitialMaskBundle:
    expected_shape = tuple(frame.color_bgr.shape[:2])
    if args.init_mode == INIT_MODE_SAVED_MASKS:
        object_mask = (
            load_binary_mask(
                args.object_init_mask,
                expected_shape=expected_shape,
                repo_root=repo_root,
            )
            if object_tracking_enabled(args)
            else None
        )
        controller_mask = (
            load_binary_mask(
                args.controller_init_mask,
                expected_shape=expected_shape,
                repo_root=repo_root,
            )
            if controller_tracking_enabled(args)
            else None
        )
        if object_mask is None and controller_mask is None:
            empty = np.zeros(expected_shape, dtype=bool)
            return InitialMaskBundle(controller_mask=empty, object_mask=empty)
        if object_mask is None:
            object_mask = np.zeros_like(controller_mask, dtype=bool)
        if controller_mask is None:
            controller_mask = np.zeros_like(object_mask, dtype=bool)
        if three_identity_controller_enabled(args):
            hand_a_mask, hand_b_mask = split_controller_hand_instances(
                [controller_mask],
                label=str(args.controller_prompt),
            )
            controller_mask = np.logical_or(hand_a_mask, hand_b_mask)
        else:
            hand_a_mask = np.ascontiguousarray(controller_mask, dtype=bool)
            hand_b_mask = np.zeros_like(hand_a_mask, dtype=bool)
        return InitialMaskBundle(
            controller_mask=np.ascontiguousarray(controller_mask, dtype=bool),
            object_mask=np.ascontiguousarray(object_mask, dtype=bool),
            hand_a_mask=np.ascontiguousarray(hand_a_mask, dtype=bool),
            hand_b_mask=np.ascontiguousarray(hand_b_mask, dtype=bool),
        )
    if args.init_mode == INIT_MODE_SAM31_FIRST_FRAME:
        bundle = run_sam31_first_frame_mask_bundle(frame.color_bgr, args)
        if (
            bundle.controller_mask.shape != expected_shape
            or bundle.object_mask.shape != expected_shape
        ):
            raise RuntimeError("SAM3.1 frame-0 masks do not match captured frame shape")
        return bundle
    raise ValueError(f"unsupported init mode: {args.init_mode}")


def resolve_initial_masks(
    frame: Any,
    args: argparse.Namespace,
    *,
    repo_root: Path,
) -> tuple[np.ndarray, np.ndarray]:
    bundle = resolve_initial_mask_bundle(frame, args, repo_root=repo_root)
    return bundle.controller_mask, bundle.object_mask


def prepare_runtime_services_and_source(
    demo: Any,
    *,
    pcd_filter_enabled: Callable[[argparse.Namespace], bool],
    is_replay_input_source: Callable[[str], bool],
    recording_source_cls: type,
    start_realsense_pipeline: Callable[[argparse.Namespace], Any],
    fake_live_input_source: str,
    fake_live_frame_selection_policy: str,
) -> None:
    args = demo.args
    apply_wslg_open3d_env_defaults()
    if args.depth_source == "ffs":
        demo.ffs_runner = demo._create_ffs_runner()
        warm_up_numba_ffs_align()
    elif args.depth_source == "ffs_remote":
        demo.ffs_remote_client = FfsRemoteDepthClient(
            endpoint=str(args.ffs_remote_endpoint),
            timeout_ms=int(args.ffs_remote_timeout_ms),
            return_type=str(args.ffs_remote_return),
            compression=str(args.ffs_remote_compress),
            max_inflight=int(args.ffs_remote_max_inflight),
        )
    if args.enable_remote_ffs_quality:
        endpoint = str(args.remote_ffs_quality_endpoint or args.ffs_remote_endpoint)
        demo.remote_quality_client = FfsRemoteDepthClient(
            endpoint=endpoint,
            timeout_ms=int(args.remote_ffs_quality_timeout_ms),
            return_type=str(args.remote_ffs_quality_return),
            compression=str(args.remote_ffs_quality_compress),
            max_inflight=1,
        )
    if pcd_filter_enabled(args) and str(args.pcd_filter_mode) == "async":
        demo.filter_worker = AsyncPcdFilterWorker(demo._filter_pcd_input)
        demo.filter_worker.start()
    if is_replay_input_source(str(args.input_source)):
        demo.recording_source = recording_source_cls(
            args.recording_case,
            replay_fps=float(args.replay_fps),
            depth_source=str(args.depth_source),
        )
        demo.width = demo.recording_source.width
        demo.height = demo.recording_source.height
        demo.runtime = demo.recording_source.make_runtime()
        replay_label = (
            "fake-live"
            if args.input_source == fake_live_input_source
            else "recording-replay"
        )
        frame_selection = (
            fake_live_frame_selection_policy
            if args.input_source == fake_live_input_source
            else "sequential"
        )
        print(
            f"[{replay_label}] "
            f"case={demo.recording_source.case_path} "
            f"frames={demo.recording_source.frame_count} "
            f"replay_fps={demo.recording_source.effective_fps:g} "
            f"recording_fps={demo.recording_source.recording_fps:g} "
            f"first_step={demo.recording_source.steps[0]} "
            f"serial={demo.recording_source.serial} "
            f"depth_source={demo.recording_source.depth_source} "
            f"ir_stereo={str(demo.recording_source.has_ir_stereo).lower()} "
            f"frame_selection={frame_selection}",
            flush=True,
        )
    else:
        demo.runtime = start_realsense_pipeline(args)


def prepare_runtime_projection_and_capture(
    demo: Any,
    *,
    headless_capture_enabled: Callable[[argparse.Namespace], bool],
    headless_capture_writer_cls: type,
) -> None:
    demo._initialize_table_calibration()
    demo.ray_x, demo.ray_y = build_projection_grid(
        width=demo.width,
        height=demo.height,
        stride=1,
        intrinsics=demo.runtime.intrinsics,
    )
    if headless_capture_enabled(demo.args):
        demo.headless_capture_writer = headless_capture_writer_cls(
            demo.args.headless_capture_dir,
            metadata=demo._build_headless_capture_metadata(),
        )
        print(
            f"[headless-capture] dir={demo.headless_capture_writer.output_dir}",
            flush=True,
        )


def prepare_segmentation_warmup(
    demo: Any, *, repo_root: Path
) -> SegmentationWarmupState:
    hf_stream, torch_module, dtype, model, processor = demo._init_hf_model()
    first_frame = demo._wait_for_first_frame()
    if first_frame is None:
        return SegmentationWarmupState(
            hf_stream=hf_stream,
            torch_module=torch_module,
            dtype=dtype,
            model=model,
            processor=processor,
            first_frame=None,
            initial_masks=None,
        )
    initial_masks = resolve_initial_mask_bundle(
        first_frame,
        demo.args,
        repo_root=repo_root,
    )
    return SegmentationWarmupState(
        hf_stream=hf_stream,
        torch_module=torch_module,
        dtype=dtype,
        model=model,
        processor=processor,
        first_frame=first_frame,
        initial_masks=initial_masks,
    )
