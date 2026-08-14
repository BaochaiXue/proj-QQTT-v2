"""Synthetic-array tests for demo_v7/service/frame0_pipeline.py.

No GPU, no SAM3.1 checkpoint: the sam31 call boundary
(``demo_v7.runtime.perception.sam31_image_segmentation.run_image_segmentation``,
lazily imported inside the v6.2 bundle function) is monkeypatched, so the
real v6.2 union/split/gate code still runs on synthetic masks. Geometry
tests use a dense-enough synthetic frame that the real radius-outlier
filter keeps mask interiors.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from qqtt.env.camera.table_calibration import TABLE_WORLD_FRAME_KIND

from demo_v7.runtime.mdp.cli import RunMode
from demo_v7.runtime.mdp.packets import PcdBuildResult
from demo_v7.runtime.perception import sam31_image_segmentation
from demo_v7.runtime.shape_prior.case import ShapePriorFrame0Request
from demo_v7.runtime.utils.camera import CameraIntrinsics

from demo_v7.ipc.protocol import ARTIFACT_KIND_FRAME0, ARTIFACT_KIND_MASKS
from demo_v7.service import frame0_pipeline
from demo_v7.service.frame0_pipeline import Frame0Bundle

H = W = 64
OBJECT_PROMPT = "sloth"


def _mode() -> RunMode:
    return RunMode(
        tracker_enabled=True,
        lossless_enabled=True,
        lossless_input_fps=30.0,
        controller_tracking_enabled=True,
        object_tracking_enabled=True,
        fake_live_input=True,
        depth_backend_label="realsense",
    )


def _args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "device": "cpu",
        "shape_prior_object_prompt": OBJECT_PROMPT,
        "depth_source": "realsense",
        "input_source": "fake_live",
        "pcd_color_mode": "rgb",
        "controller_color": (255, 0, 0),
        "object_color": (0, 255, 0),
        "shape_prior_warmup": True,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _rect_mask(r0: int, r1: int, c0: int, c1: int) -> np.ndarray:
    mask = np.zeros((H, W), dtype=bool)
    mask[r0:r1, c0:c1] = True
    return mask


def _object_mask() -> np.ndarray:
    return _rect_mask(8, 30, 20, 44)


def _hand_a_mask() -> np.ndarray:
    return _rect_mask(38, 60, 4, 20)


def _hand_b_mask() -> np.ndarray:
    return _rect_mask(38, 60, 44, 60)


def _color_bgr() -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)


def _bundle(
    *,
    depth_u16: np.ndarray | None = None,
    object_mask: np.ndarray | None = None,
) -> Frame0Bundle:
    if depth_u16 is None:
        depth_u16 = np.full((H, W), 1000, dtype=np.uint16)  # 1.0 m at 1mm units
    return Frame0Bundle(
        color_bgr=_color_bgr(),
        depth_u16=depth_u16,
        # fx=600 at 1 m depth -> ~1.7mm pixel pitch, dense enough that the
        # real 0.01m/40-neighbor radius filter keeps mask interiors.
        intrinsics=CameraIntrinsics(fx=600.0, fy=600.0, cx=32.0, cy=32.0),
        depth_scale_m_per_unit=0.001,
        object_mask=_object_mask() if object_mask is None else object_mask,
        hand_a_mask=_hand_a_mask(),
        hand_b_mask=_hand_b_mask(),
    )


def _session(c2w: np.ndarray | None = None) -> SimpleNamespace:
    k_color = np.array(
        [[600.0, 0.0, 32.0], [0.0, 600.0, 32.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return SimpleNamespace(
        table_c2w=np.eye(4, dtype=np.float32) if c2w is None else c2w,
        camera_runtime=SimpleNamespace(k_color=k_color),
    )


def _patch_sam31(
    monkeypatch: pytest.MonkeyPatch,
    *,
    object_masks: list[np.ndarray],
    hand_masks: list[np.ndarray],
) -> list[dict[str, object]]:
    """Patch the lazily imported sam31 call; return the recorded call kwargs."""
    calls: list[dict[str, object]] = []

    def fake_run_image_segmentation(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {
            "masks_by_label": {OBJECT_PROMPT: object_masks, "hand": hand_masks},
            "timing_ms": {"total_ms": 1.0},
        }

    monkeypatch.setattr(
        sam31_image_segmentation,
        "run_image_segmentation",
        fake_run_image_segmentation,
    )
    return calls


# ---------------------------------------------------------------------------
# compute_sam31_masks
# ---------------------------------------------------------------------------


def test_compute_sam31_masks_two_hand_instances(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left, right = _hand_a_mask(), _hand_b_mask()
    calls = _patch_sam31(
        monkeypatch,
        object_masks=[_object_mask()],
        hand_masks=[right, left],  # order scrambled: centroid must fix identity
    )
    profile: dict[str, object] = {}
    object_mask, hand_a, hand_b = frame0_pipeline.compute_sam31_masks(
        _color_bgr(),
        device="cpu",
        args=_args(),
        mode=_mode(),
        profile_out=profile,
    )
    assert np.array_equal(object_mask, _object_mask())
    assert np.array_equal(hand_a, left)  # hand A is the leftmost hand
    assert np.array_equal(hand_b, right)
    assert len(calls) == 1
    # Object prompt must precede the controller prompt (v6.2 best-instance rule).
    assert calls[0]["text_prompt"] == f"{OBJECT_PROMPT},hand"
    assert calls[0]["reuse_model"] is True
    warmup_profile = profile["segmentation_warmup"]
    assert warmup_profile["frame0_available"] is True
    assert warmup_profile["initial_sam31"] == {"total_ms": 1.0}


def test_compute_sam31_masks_splits_merged_controller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    merged = np.logical_or(_hand_a_mask(), _hand_b_mask())
    _patch_sam31(monkeypatch, object_masks=[_object_mask()], hand_masks=[merged])
    _object, hand_a, hand_b = frame0_pipeline.compute_sam31_masks(
        _color_bgr(), device="cpu", args=_args(), mode=_mode()
    )
    assert np.array_equal(hand_a, _hand_a_mask())
    assert np.array_equal(hand_b, _hand_b_mask())


def test_compute_sam31_masks_gate_raises_on_single_hand(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_sam31(
        monkeypatch, object_masks=[_object_mask()], hand_masks=[_hand_a_mask()]
    )
    with pytest.raises(RuntimeError, match="two separable controller masks"):
        frame0_pipeline.compute_sam31_masks(
            _color_bgr(), device="cpu", args=_args(), mode=_mode()
        )


def test_compute_sam31_masks_device_override_leaves_args_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_sam31(
        monkeypatch,
        object_masks=[_object_mask()],
        hand_masks=[_hand_a_mask(), _hand_b_mask()],
    )
    args = _args(device="cuda")
    frame0_pipeline.compute_sam31_masks(
        _color_bgr(), device="cpu", args=args, mode=_mode()
    )
    assert calls[0]["device"] == "cpu"
    assert args.device == "cuda"


# ---------------------------------------------------------------------------
# build_frame0_processed
# ---------------------------------------------------------------------------


def test_build_frame0_processed_geometry() -> None:
    bundle = _bundle()
    result = frame0_pipeline.build_frame0_processed(
        bundle, args=_args(), session=_session()
    )
    assert isinstance(result, PcdBuildResult)
    packet = result.pcd_packet
    frame = result.processed_frame

    assert packet.seq == frame0_pipeline.FRAME0_SEQ
    assert packet.coordinate_frame == TABLE_WORLD_FRAME_KIND
    assert frame.pcd_points.shape == (1, H, W, 3)
    assert frame.pcd_colors.shape == (1, H, W, 3)
    assert frame.depth_m.shape == (H, W)
    assert np.all(frame.depth_valid_mask)  # every pixel is at a valid 1.0 m

    mask_packet = frame.mask_packet
    # Processed masks are subsets of the raw masks (depth gate + radius
    # outlier only remove pixels) and stay non-empty.
    assert np.any(mask_packet.object_mask)
    assert not np.any(mask_packet.object_mask & ~bundle.object_mask)
    raw_controller = bundle.hand_a_mask | bundle.hand_b_mask
    assert np.any(mask_packet.controller_mask)
    assert not np.any(mask_packet.controller_mask & ~raw_controller)
    # Hand identities remain controller-mask subsets (v6.2 invariant).
    assert not np.any(mask_packet.hand_a_mask & ~mask_packet.controller_mask)
    assert not np.any(mask_packet.hand_b_mask & ~mask_packet.controller_mask)

    # Identity c2w + flat 1.0m depth: every masked world point sits at z=1.
    assert packet.object_xyz_m.shape[1] == 3
    assert packet.object_xyz_m.shape[0] == int(np.count_nonzero(mask_packet.object_mask))
    assert np.allclose(packet.object_xyz_m[:, 2], 1.0, atol=1e-5)
    assert np.allclose(packet.controller_xyz_m[:, 2], 1.0, atol=1e-5)

    # rgb color mode: packet colors are the frame's RGB at the mask pixels.
    rgb = bundle.color_bgr[:, :, ::-1]
    assert np.array_equal(packet.object_colors_rgb_u8, rgb[mask_packet.object_mask])
    assert packet.timing.pcd_ms > 0.0


def test_build_frame0_processed_class_color_mode() -> None:
    result = frame0_pipeline.build_frame0_processed(
        _bundle(), args=_args(pcd_color_mode="class"), session=_session()
    )
    packet = result.pcd_packet
    assert np.all(packet.object_colors_rgb_u8 == np.array([0, 255, 0], dtype=np.uint8))
    assert np.all(
        packet.controller_colors_rgb_u8 == np.array([255, 0, 0], dtype=np.uint8)
    )


def test_build_frame0_processed_depth_gate_removes_invalid_pixels() -> None:
    depth = np.full((H, W), 1000, dtype=np.uint16)
    hole = _rect_mask(12, 18, 24, 40)  # inside the object mask
    depth[hole] = 0
    result = frame0_pipeline.build_frame0_processed(
        _bundle(depth_u16=depth), args=_args(), session=_session()
    )
    mask_packet = result.processed_frame.mask_packet
    assert not np.any(mask_packet.object_mask & hole)
    assert np.any(mask_packet.object_mask)


def test_build_frame0_processed_empty_object_raises() -> None:
    depth = np.full((H, W), 1000, dtype=np.uint16)
    depth[_object_mask()] = 0  # object entirely at invalid depth
    with pytest.raises(RuntimeError, match="processed object mask is empty"):
        frame0_pipeline.build_frame0_processed(
            _bundle(depth_u16=depth), args=_args(), session=_session()
        )


def test_build_frame0_processed_requires_calibration() -> None:
    session = _session()
    session.table_c2w = None
    with pytest.raises(RuntimeError, match="camera-to-world calibration"):
        frame0_pipeline.build_frame0_processed(
            _bundle(), args=_args(), session=session
        )


# ---------------------------------------------------------------------------
# submit_shape_prior
# ---------------------------------------------------------------------------


class _RecordingManager:
    """Stands in for ShapePriorWarmupManager at the maybe_submit boundary."""

    def __init__(self, *, accept: bool = True) -> None:
        self.accept = bool(accept)
        self.requests: list[ShapePriorFrame0Request] = []
        self.profile_writes = 0

    def maybe_submit(self, frame0: ShapePriorFrame0Request) -> bool:
        self.requests.append(frame0)
        return self.accept

    def write_profile_json(self) -> None:
        self.profile_writes += 1


def test_submit_shape_prior_builds_v62_request() -> None:
    bundle = _bundle()
    session = _session()
    args = _args()
    processed = frame0_pipeline.build_frame0_processed(
        bundle, args=args, session=session
    )
    manager = _RecordingManager()
    submitted = frame0_pipeline.submit_shape_prior(
        manager,
        processed,
        args=args,
        session=session,
        warmup_start_perf_s=123.0,
        perception_profile={"segmentation_warmup": {"frame0_available": True}},
    )
    assert submitted is True
    assert manager.profile_writes == 1
    request = manager.requests[0]
    assert isinstance(request, ShapePriorFrame0Request)
    assert request.seq == frame0_pipeline.FRAME0_SEQ
    assert request.input_source == "fake_live"
    assert request.depth_backend == "realsense"
    assert request.depth_source_internal == "realsense"
    assert np.array_equal(request.rgb_u8, bundle.color_bgr[:, :, ::-1])
    mask_packet = processed.processed_frame.mask_packet
    assert np.array_equal(request.object_mask, mask_packet.object_mask)
    assert np.array_equal(request.controller_mask, mask_packet.controller_mask)
    assert request.points_world_m.shape == (H, W, 3)
    assert np.array_equal(request.k_color, mask_packet.k_color)
    assert request.camera_to_world_c2w is session.table_c2w
    assert request.warmup_runtime_start_perf_s == 123.0
    assert request.frame0_pipeline_timing_ms["pcd_ms"] > 0.0
    assert request.frame0_perception_profile == {
        "segmentation_warmup": {"frame0_available": True}
    }


def test_submit_shape_prior_disabled_returns_false() -> None:
    session = _session()
    args = _args(shape_prior_warmup=False)
    processed = frame0_pipeline.build_frame0_processed(
        _bundle(), args=args, session=session
    )
    manager = _RecordingManager()
    assert (
        frame0_pipeline.submit_shape_prior(
            manager, processed, args=args, session=session
        )
        is False
    )
    assert manager.requests == []
    assert manager.profile_writes == 0


def test_submit_shape_prior_rejected_submit_skips_profile_write() -> None:
    session = _session()
    args = _args()
    processed = frame0_pipeline.build_frame0_processed(
        _bundle(), args=args, session=session
    )
    manager = _RecordingManager(accept=False)
    assert (
        frame0_pipeline.submit_shape_prior(
            manager, processed, args=args, session=session
        )
        is False
    )
    assert manager.profile_writes == 0


# ---------------------------------------------------------------------------
# save_review_artifacts
# ---------------------------------------------------------------------------


def test_save_review_artifacts_layout_and_content(tmp_path: Path) -> None:
    import cv2

    bundle = _bundle()
    artifacts = frame0_pipeline.save_review_artifacts(tmp_path, bundle)
    assert set(artifacts) == {ARTIFACT_KIND_FRAME0, ARTIFACT_KIND_MASKS}
    frame0_paths = artifacts[ARTIFACT_KIND_FRAME0]
    mask_paths = artifacts[ARTIFACT_KIND_MASKS]
    assert set(frame0_paths) == {"rgb", "depth_npy", "depth_preview"}
    assert set(mask_paths) == {"object", "hand_a", "hand_b", "overlay"}
    for path_str in [*frame0_paths.values(), *mask_paths.values()]:
        path = Path(path_str)
        assert path.is_absolute()
        assert path.exists()

    rgb_png = cv2.imread(frame0_paths["rgb"], cv2.IMREAD_COLOR)
    assert np.array_equal(rgb_png, bundle.color_bgr)
    assert np.array_equal(np.load(frame0_paths["depth_npy"]), bundle.depth_u16)

    object_png = cv2.imread(mask_paths["object"], cv2.IMREAD_GRAYSCALE)
    assert np.array_equal(object_png > 0, bundle.object_mask)

    overlay = cv2.imread(mask_paths["overlay"], cv2.IMREAD_COLOR)
    tinted = bundle.object_mask | bundle.hand_a_mask | bundle.hand_b_mask
    assert np.array_equal(overlay[~tinted], bundle.color_bgr[~tinted])
    assert np.any(overlay[tinted] != bundle.color_bgr[tinted])

    preview = cv2.imread(frame0_paths["depth_preview"], cv2.IMREAD_COLOR)
    assert preview.shape == (H, W, 3)


def test_save_review_artifacts_depth_preview_invalid_pixels_black(
    tmp_path: Path,
) -> None:
    import cv2

    depth = np.full((H, W), 1000, dtype=np.uint16)
    depth[:8] = 0
    artifacts = frame0_pipeline.save_review_artifacts(
        tmp_path, _bundle(depth_u16=depth)
    )
    preview = cv2.imread(
        artifacts[ARTIFACT_KIND_FRAME0]["depth_preview"], cv2.IMREAD_COLOR
    )
    assert np.all(preview[:8] == 0)
    assert np.any(preview[8:] != 0)
