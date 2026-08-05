"""Frame-0 derived computation for the demo_v7 WARMUP state (no tracking).

Pure orchestration of existing demo_v6_2 functions over ONE captured frame:
SAM3.1 initial-mask bundle (the exact v6.2 frame-0 seeding source) ->
canonical processed frame / class PCDs (the exact FormalProductStage math)
-> shape-prior frame-0 request (the exact ShapePriorPublisher construction),
plus the on-disk review artifacts the GUI shows. No numeric step is
re-implemented here; every one is imported from demo_v6_2.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from qqtt.env.camera.table_calibration import TABLE_WORLD_FRAME_KIND

from demo_v6_2.mdp import warmup as v62_warmup
from demo_v6_2.mdp.cli import RunMode, depth_backend_label
from demo_v6_2.mdp.packets import (
    MaskedPcdPacket,
    MaskPacket,
    PcdBuildResult,
    PipelineTiming,
    ProcessedFramePacket,
)
from demo_v6_2.phystwin_strict_product import (
    PHYSTWIN_DEPTH_MAX_M,
    PHYSTWIN_DEPTH_MIN_M,
    apply_depth_validity_to_mask_frame,
    apply_radius_outlier_to_mask_frame,
    dense_world_pcd_grid,
)
from demo_v6_2.shape_prior import case as shape_prior_case
from demo_v6_2.utils.concurrency import elapsed_ms as _elapsed_ms

from demo_v7.ipc.protocol import ARTIFACT_KIND_FRAME0, ARTIFACT_KIND_MASKS

if TYPE_CHECKING:
    from demo_v6_2.mdp.session import CameraSession
    from demo_v6_2.shape_prior.warmup import ShapePriorWarmupManager

# The v7 frame-0 pipeline processes exactly one frozen frame; its packets all
# carry the sequence number the formal run will later restart from.
FRAME0_SEQ = 0

# Review overlay: 50% mask tint over the captured RGB (README 摆位/结果屏).
OVERLAY_ALPHA = 0.5
# BGR tints for the overlay png (object green, hand A red, hand B blue).
OVERLAY_TINTS_BGR: dict[str, tuple[int, int, int]] = {
    "object": (0, 200, 0),
    "hand_a": (0, 0, 230),
    "hand_b": (230, 0, 0),
}


@dataclass(frozen=True)
class Frame0Bundle:
    """One frozen frame plus its SAM3.1 masks — input to all derived steps.

    ``depth_u16`` is the native color-aligned RealSense depth; the v7 frame-0
    pipeline supports only the RGB-D depth path (FFS depth would need the
    session's depth engine and is not part of the v7 flow). The ``source_*``
    fields carry the candidate FramePacket's recording provenance (stamped by
    the fake-live replayer; None on a live camera) so the shape-prior request
    and downstream packets keep the v6.2 provenance chain.
    """

    color_bgr: np.ndarray
    depth_u16: np.ndarray
    intrinsics: Any
    depth_scale_m_per_unit: float
    object_mask: np.ndarray
    hand_a_mask: np.ndarray
    hand_b_mask: np.ndarray
    source_timestamp_s: float | None = None
    source_frame_index: int | None = None
    source_step: int | None = None


# ---------------------------------------------------------------------------
# SAM3.1 frame-0 masks (v6.2 seeding source, single frame)
# ---------------------------------------------------------------------------


def compute_sam31_masks(
    color_bgr: np.ndarray,
    *,
    device: str,
    args: argparse.Namespace,
    mode: RunMode | None = None,
    reuse_sam31_runtime: bool = True,
    profile_out: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the v6.2 frame-0 SAM3.1 mask bundle and return (object, hand_a, hand_b).

    Exactly the seeding source SegmentationStage._prepare_warmup uses:
    ``warmup.run_sam31_first_frame_mask_bundle`` — including the "two
    separable controller masks" gate, which raises the same RuntimeError.
    ``defer_release`` is always False here: v7's warmup runs no EdgeTAM
    forward to hide the release behind, and inline release is always safe.
    ``reuse_sam31_runtime`` must be ``manager.requires_sam31_reuse`` when a
    shape-prior generation may still need the model (v6.2 semantics); the
    default True only trims the allocator and never drops the cached model.
    ``profile_out``, when given, receives the segmentation_warmup-style
    timing entries for the frame0_perception_profile.
    """
    if mode is None:
        mode = RunMode.from_args(args)
    if str(device) != str(getattr(args, "device", device)):
        # run_sam31_first_frame_mask_bundle reads args.device for both the
        # segmentation and its CUDA cleanup; honor the explicit device
        # parameter without mutating the caller's namespace.
        args = argparse.Namespace(**vars(args))
        args.device = str(device)
    started_s = time.perf_counter()
    initial_masks, sam31_timing = v62_warmup.run_sam31_first_frame_mask_bundle(
        color_bgr,
        args,
        mode,
        reuse_sam31_runtime=bool(reuse_sam31_runtime),
        defer_release=False,
    )
    expected_shape = tuple(color_bgr.shape[:2])
    if (
        initial_masks.controller_mask.shape != expected_shape
        or initial_masks.object_mask.shape != expected_shape
    ):
        raise RuntimeError("SAM3.1 frame-0 masks do not match captured frame shape")
    if profile_out is not None:
        profile_out["segmentation_warmup"] = {
            "initial_mask_bundle_ms": _elapsed_ms(started_s, time.perf_counter()),
            "initial_sam31": dict(sam31_timing.timing_ms),
            "sam31_trim_cleanup_ms": float(sam31_timing.trim_cleanup_ms),
            "sam31_release_cleanup_ms": float(sam31_timing.release_cleanup_ms),
            "sam31_release_deferred": False,
            "frame0_available": True,
        }
    # Disabled-controller modes yield None hand masks in v6.2; the v7 bundle
    # carries concrete arrays, so map None to all-false of the object shape.
    hand_a = initial_masks.hand_a_mask
    hand_b = initial_masks.hand_b_mask
    if hand_a is None:
        hand_a = np.zeros_like(initial_masks.object_mask, dtype=bool)
    if hand_b is None:
        hand_b = np.zeros_like(initial_masks.object_mask, dtype=bool)
    return (
        np.ascontiguousarray(initial_masks.object_mask, dtype=bool),
        np.ascontiguousarray(hand_a, dtype=bool),
        np.ascontiguousarray(hand_b, dtype=bool),
    )


# ---------------------------------------------------------------------------
# Canonical processed frame + class PCDs (FormalProductStage math, one frame)
# ---------------------------------------------------------------------------


def _frame0_mask_packet(
    bundle: Frame0Bundle,
    *,
    args: argparse.Namespace,
    session: CameraSession,
    receive_perf_s: float,
) -> MaskPacket:
    """Wrap the bundle as the seq-0 MaskPacket the v6.2 PCD math consumes."""
    controller_mask = np.ascontiguousarray(
        np.asarray(bundle.hand_a_mask, dtype=bool)
        | np.asarray(bundle.hand_b_mask, dtype=bool),
        dtype=bool,
    )
    k_color: np.ndarray | None = None
    if session.camera_runtime is not None:
        k_color = np.asarray(session.camera_runtime.k_color, dtype=np.float32)
    return MaskPacket(
        seq=FRAME0_SEQ,
        color_bgr=np.ascontiguousarray(bundle.color_bgr, dtype=np.uint8),
        depth_source=str(getattr(args, "depth_source", "realsense")),
        intrinsics=bundle.intrinsics,
        depth_scale_m_per_unit=float(bundle.depth_scale_m_per_unit),
        receive_perf_s=float(receive_perf_s),
        process_done_perf_s=time.perf_counter(),
        dropped_capture_frames=0,
        timing=PipelineTiming(),
        controller_mask=controller_mask,
        object_mask=np.ascontiguousarray(bundle.object_mask, dtype=bool),
        hand_a_mask=np.ascontiguousarray(bundle.hand_a_mask, dtype=bool),
        hand_b_mask=np.ascontiguousarray(bundle.hand_b_mask, dtype=bool),
        depth_u16=np.ascontiguousarray(bundle.depth_u16),
        k_color=k_color,
        source_timestamp_s=bundle.source_timestamp_s,
        source_frame_index=bundle.source_frame_index,
        source_step=bundle.source_step,
    )


def build_frame0_processed(
    bundle: Frame0Bundle,
    *,
    args: argparse.Namespace,
    session: CameraSession,
) -> PcdBuildResult:
    """Build the origin-style processed frame + class PCDs for frame 0.

    Same call pattern as FormalProductStage._build_processed_frame_result
    (demo_v6_2/mdp/formal_products.py) restricted to the RGB-D depth branch:
    dense_world_pcd_grid -> apply_depth_validity_to_mask_frame ->
    apply_radius_outlier_to_mask_frame, with the identical empty-mask gates.
    """
    started_s = time.perf_counter()
    if session.table_c2w is None:
        raise RuntimeError("formal processed frames require camera-to-world calibration")
    c2w = np.asarray(session.table_c2w, dtype=np.float32)
    if c2w.shape != (4, 4) or not np.isfinite(c2w).all():
        raise RuntimeError("camera-to-world calibration must be a finite 4x4")
    if bundle.depth_u16 is None:
        raise RuntimeError("frame-0 processed frame requires RGB-D depth")

    mask_packet = _frame0_mask_packet(
        bundle, args=args, session=session, receive_perf_s=started_s
    )
    depth_convert_start_s = time.perf_counter()
    depth_m = np.ascontiguousarray(
        np.asarray(mask_packet.depth_u16).astype(np.float32)
        * np.float32(mask_packet.depth_scale_m_per_unit)
    )
    depth_timing = {
        "ffs_ms": 0.0,
        "ffs_align_ms": 0.0,
        "depth_convert_ms": _elapsed_ms(depth_convert_start_s, time.perf_counter()),
    }
    rgb_u8 = np.ascontiguousarray(mask_packet.color_bgr[:, :, ::-1], dtype=np.uint8)
    pcd_points, pcd_colors = dense_world_pcd_grid(
        depth_m=depth_m,
        color_rgb_u8=rgb_u8,
        intrinsics=mask_packet.intrinsics,
        c2w=c2w,
    )
    raw_masks = {
        "object": mask_packet.object_mask,
        "controller": mask_packet.controller_mask,
        "hand_a": mask_packet.hand_a_mask,
        "hand_b": mask_packet.hand_b_mask,
    }
    depth_valid_masks = apply_depth_validity_to_mask_frame(raw_masks, depth_m)
    processed_masks = apply_radius_outlier_to_mask_frame(depth_valid_masks, pcd_points)
    object_mask = np.ascontiguousarray(processed_masks["object"], dtype=bool)
    controller_mask = np.ascontiguousarray(processed_masks["controller"], dtype=bool)
    if not np.any(object_mask):
        raise RuntimeError(f"processed object mask is empty at seq {mask_packet.seq}")
    if not np.any(controller_mask):
        raise RuntimeError(
            f"processed controller mask is empty at seq {mask_packet.seq}"
        )

    processed_mask_packet = replace(
        mask_packet,
        controller_mask=controller_mask,
        object_mask=object_mask,
        hand_a_mask=np.ascontiguousarray(
            processed_masks.get("hand_a", np.zeros_like(controller_mask)),
            dtype=bool,
        ),
        hand_b_mask=np.ascontiguousarray(
            processed_masks.get("hand_b", np.zeros_like(controller_mask)),
            dtype=bool,
        ),
    )
    depth_valid_mask = np.ascontiguousarray(
        np.isfinite(depth_m)
        & (depth_m > np.float32(PHYSTWIN_DEPTH_MIN_M))
        & (depth_m < np.float32(PHYSTWIN_DEPTH_MAX_M)),
        dtype=bool,
    )
    processed_frame = ProcessedFramePacket(
        seq=int(mask_packet.seq),
        mask_packet=processed_mask_packet,
        depth_m=np.ascontiguousarray(depth_m, dtype=np.float32),
        depth_valid_mask=depth_valid_mask,
        pcd_points=np.ascontiguousarray(pcd_points, dtype=np.float32),
        pcd_colors=np.ascontiguousarray(pcd_colors, dtype=np.uint8),
    )

    points_grid = processed_frame.pcd_points[0]
    colors_grid = processed_frame.pcd_colors[0]
    controller_xyz = np.ascontiguousarray(
        points_grid[controller_mask], dtype=np.float32
    ).reshape(-1, 3)
    object_xyz = np.ascontiguousarray(
        points_grid[object_mask], dtype=np.float32
    ).reshape(-1, 3)
    if str(args.pcd_color_mode) == "class":
        controller_colors = np.tile(
            np.asarray(args.controller_color, dtype=np.uint8),
            (len(controller_xyz), 1),
        )
        object_colors = np.tile(
            np.asarray(args.object_color, dtype=np.uint8),
            (len(object_xyz), 1),
        )
    else:
        controller_colors = colors_grid[controller_mask]
        object_colors = colors_grid[object_mask]

    done_s = time.perf_counter()
    timing = replace(
        mask_packet.timing,
        **depth_timing,
        pcd_ms=_elapsed_ms(started_s, done_s),
    )
    packet = MaskedPcdPacket(
        seq=mask_packet.seq,
        controller_xyz_m=controller_xyz,
        controller_colors_rgb_u8=np.ascontiguousarray(
            controller_colors, dtype=np.uint8
        ).reshape(-1, 3),
        object_xyz_m=object_xyz,
        object_colors_rgb_u8=np.ascontiguousarray(
            object_colors, dtype=np.uint8
        ).reshape(-1, 3),
        intrinsics=mask_packet.intrinsics,
        receive_perf_s=mask_packet.receive_perf_s,
        process_done_perf_s=done_s,
        dropped_capture_frames=mask_packet.dropped_capture_frames,
        timing=timing,
        coordinate_frame=TABLE_WORLD_FRAME_KIND,
        source_timestamp_s=mask_packet.source_timestamp_s,
        source_frame_index=mask_packet.source_frame_index,
        source_step=mask_packet.source_step,
    )
    return PcdBuildResult(pcd_packet=packet, processed_frame=processed_frame)


# ---------------------------------------------------------------------------
# Shape-prior frame-0 submission (ShapePriorPublisher construction, one shot)
# ---------------------------------------------------------------------------


def submit_shape_prior(
    manager: ShapePriorWarmupManager,
    processed: PcdBuildResult,
    *,
    args: argparse.Namespace,
    session: CameraSession,
    warmup_start_perf_s: float | None = None,
    perception_profile: dict[str, Any] | None = None,
) -> bool:
    """Build the v6.2 ShapePriorFrame0Request and submit it; True when accepted.

    Field-for-field the request ShapePriorPublisher._frame0_request_from_pcd_result
    builds (demo_v6_2/mdp/shape_prior_flow.py), fed into
    ``ShapePriorWarmupManager.maybe_submit``; a successful submit mirrors the
    publisher's ``write_profile_json()``. ``session`` supplies table_c2w and
    the k_color fallback the publisher reads from its CameraSession.
    """
    if not bool(args.shape_prior_warmup):
        return False
    if session.table_c2w is None:
        raise RuntimeError("shape-prior frame 0 requires camera-to-world calibration")
    processed_frame = processed.processed_frame
    mask_packet = processed_frame.mask_packet
    if int(processed.pcd_packet.seq) != int(mask_packet.seq):
        raise RuntimeError("shape-prior PCD/mask sequence mismatch")
    k_color = mask_packet.k_color
    if k_color is None and session.camera_runtime is not None:
        k_color = np.asarray(session.camera_runtime.k_color, dtype=np.float32)
    if k_color is None:
        raise RuntimeError("shape-prior frame 0 requires color intrinsics")
    request = shape_prior_case.ShapePriorFrame0Request(
        seq=int(mask_packet.seq),
        source_timestamp_s=mask_packet.source_timestamp_s,
        input_source=str(args.input_source),
        depth_backend=depth_backend_label(args),
        depth_source_internal=str(args.depth_source),
        rgb_u8=mask_packet.color_bgr[:, :, ::-1],
        object_mask=mask_packet.object_mask,
        controller_mask=mask_packet.controller_mask,
        depth_color_m=processed_frame.depth_m,
        depth_valid_mask=processed_frame.depth_valid_mask,
        points_world_m=processed_frame.pcd_points[0],
        k_color=k_color,
        camera_to_world_c2w=session.table_c2w,
        warmup_runtime_start_perf_s=warmup_start_perf_s,
        frame_receive_perf_s=float(mask_packet.receive_perf_s),
        frame_mask_ready_perf_s=float(mask_packet.process_done_perf_s),
        frame_pcd_ready_perf_s=float(processed.pcd_packet.process_done_perf_s),
        frame0_pipeline_timing_ms={
            key: float(value)
            for key, value in asdict(processed.pcd_packet.timing).items()
        },
        frame0_perception_profile=dict(perception_profile or {}),
    )
    submitted = bool(manager.maybe_submit(request))
    if submitted:
        manager.write_profile_json()
    return submitted


# ---------------------------------------------------------------------------
# Review artifacts (frame-0 stills + mask pngs + 50%-alpha overlay)
# ---------------------------------------------------------------------------


def _depth_preview_bgr(depth_u16: np.ndarray) -> np.ndarray:
    """Render a JET-colormapped depth preview; invalid (zero) pixels stay black."""
    import cv2  # noqa: PLC0415

    depth = np.asarray(depth_u16)
    valid = depth > 0
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    if np.any(valid):
        values = depth[valid].astype(np.float32)
        lo = float(values.min())
        hi = float(values.max())
        span = (hi - lo) if hi > lo else 1.0
        # 1..255 keeps valid pixels distinguishable from the invalid black 0.
        normalized[valid] = (
            1.0 + (values - lo) * (254.0 / span)
        ).astype(np.uint8)
    preview = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    preview[~valid] = 0
    return preview


def _mask_png_u8(mask: np.ndarray) -> np.ndarray:
    """Return the 0/255 uint8 image for a boolean mask."""
    return np.where(np.asarray(mask, dtype=bool), 255, 0).astype(np.uint8)


def _overlay_bgr(bundle: Frame0Bundle, *, alpha: float) -> np.ndarray:
    """Blend the class tints over the captured RGB at the given alpha."""
    overlay = np.asarray(bundle.color_bgr, dtype=np.float32).copy()
    masks = {
        "object": bundle.object_mask,
        "hand_a": bundle.hand_a_mask,
        "hand_b": bundle.hand_b_mask,
    }
    for name, mask in masks.items():
        selected = np.asarray(mask, dtype=bool)
        if not np.any(selected):
            continue
        tint = np.asarray(OVERLAY_TINTS_BGR[name], dtype=np.float32)
        overlay[selected] = overlay[selected] * (1.0 - alpha) + tint * alpha
    return np.clip(overlay, 0.0, 255.0).astype(np.uint8)


def save_review_artifacts(
    run_dir: Path,
    bundle: Frame0Bundle,
    *,
    alpha: float = OVERLAY_ALPHA,
) -> dict[str, dict[str, str]]:
    """Write frame-0 review artifacts and return {ARTIFACT_KIND: {name: path}}.

    Layout under ``run_dir / "frame0"``: the captured RGB png, the raw
    uint16 depth npy plus its colormap preview png, one png per mask
    (object/hand_a/hand_b), and the 50%-alpha class overlay png. Paths are
    absolute strings ready for the EVT_ARTIFACTS event payload.
    """
    import cv2  # noqa: PLC0415

    out_dir = (Path(run_dir) / "frame0").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    def _write_png(name: str, image_bgr_or_gray: np.ndarray) -> str:
        path = out_dir / name
        if not cv2.imwrite(str(path), image_bgr_or_gray):
            raise RuntimeError(f"failed to write review artifact {path}")
        return str(path)

    frame0_paths: dict[str, str] = {
        "rgb": _write_png(
            "frame0_rgb.png", np.ascontiguousarray(bundle.color_bgr, dtype=np.uint8)
        ),
    }
    depth_npy_path = out_dir / "frame0_depth.npy"
    np.save(depth_npy_path, np.ascontiguousarray(bundle.depth_u16))
    frame0_paths["depth_npy"] = str(depth_npy_path)
    frame0_paths["depth_preview"] = _write_png(
        "frame0_depth_preview.png", _depth_preview_bgr(bundle.depth_u16)
    )

    mask_paths: dict[str, str] = {
        "object": _write_png("mask_object.png", _mask_png_u8(bundle.object_mask)),
        "hand_a": _write_png("mask_hand_a.png", _mask_png_u8(bundle.hand_a_mask)),
        "hand_b": _write_png("mask_hand_b.png", _mask_png_u8(bundle.hand_b_mask)),
        "overlay": _write_png(
            "mask_overlay.png", _overlay_bgr(bundle, alpha=float(alpha))
        ),
    }
    return {
        ARTIFACT_KIND_FRAME0: frame0_paths,
        ARTIFACT_KIND_MASKS: mask_paths,
    }


__all__ = [
    "FRAME0_SEQ",
    "Frame0Bundle",
    "build_frame0_processed",
    "compute_sam31_masks",
    "save_review_artifacts",
    "submit_shape_prior",
]
