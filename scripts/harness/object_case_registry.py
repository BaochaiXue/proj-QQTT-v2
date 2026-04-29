from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RawObjectCaptureSpec:
    object_set: str
    round_id: str
    raw_case_name: str
    capture_mode: str
    streams: tuple[str, ...]
    width: int
    height: int
    fps: int
    notes: str = ""


STATIC_OBJECT_RAW_CASES: tuple[RawObjectCaptureSpec, ...] = (
    RawObjectCaptureSpec(
        object_set="static_object",
        round_id="round4",
        raw_case_name="both_30_static_round4_20260427",
        capture_mode="both_eval",
        streams=("color", "depth", "ir_left", "ir_right"),
        width=848,
        height=480,
        fps=30,
    ),
    RawObjectCaptureSpec(
        object_set="static_object",
        round_id="round5",
        raw_case_name="both_30_static_round5_20260427",
        capture_mode="both_eval",
        streams=("color", "depth", "ir_left", "ir_right"),
        width=848,
        height=480,
        fps=30,
    ),
    RawObjectCaptureSpec(
        object_set="static_object",
        round_id="round6",
        raw_case_name="both_30_static_round6_20260427",
        capture_mode="both_eval",
        streams=("color", "depth", "ir_left", "ir_right"),
        width=848,
        height=480,
        fps=30,
    ),
    RawObjectCaptureSpec(
        object_set="static_object",
        round_id="round7",
        raw_case_name="both_30_static_round7_20260428",
        capture_mode="both_eval",
        streams=("color", "depth", "ir_left", "ir_right"),
        width=848,
        height=480,
        fps=30,
        notes="Emitter-on static object capture used by current FFS/SAM timing checks.",
    ),
)

STILL_OBJECT_RAW_CASES: tuple[RawObjectCaptureSpec, ...] = (
    RawObjectCaptureSpec(
        object_set="still_object",
        round_id="round1",
        raw_case_name="both_30_still_object_round1_20260428",
        capture_mode="both_eval",
        streams=("color", "depth", "ir_left", "ir_right"),
        width=848,
        height=480,
        fps=30,
        notes="Still-object round1 capture; keep separate from previous static-object rounds.",
    ),
    RawObjectCaptureSpec(
        object_set="still_object",
        round_id="round2",
        raw_case_name="both_30_still_object_round2_20260428",
        capture_mode="both_eval",
        streams=("color", "depth", "ir_left", "ir_right"),
        width=848,
        height=480,
        fps=30,
        notes="Still-object round2 capture; keep separate from previous static-object rounds.",
    ),
    RawObjectCaptureSpec(
        object_set="still_object",
        round_id="round3",
        raw_case_name="both_30_still_object_round3_20260428",
        capture_mode="both_eval",
        streams=("color", "depth", "ir_left", "ir_right"),
        width=848,
        height=480,
        fps=30,
        notes="Still-object round3 capture; trimmed to 30 frames per camera after a one-frame recorder overrun.",
    ),
    RawObjectCaptureSpec(
        object_set="still_object",
        round_id="round4",
        raw_case_name="both_30_still_object_round4_20260428",
        capture_mode="both_eval",
        streams=("color", "depth", "ir_left", "ir_right"),
        width=848,
        height=480,
        fps=30,
        notes="Still-object round4 capture; first start hit a RealSense power-state error and the successful retry wrote 30 frames per camera.",
    ),
    RawObjectCaptureSpec(
        object_set="still_object",
        round_id="round7",
        raw_case_name="both_30_still_object_round7_20260428",
        capture_mode="both_eval",
        streams=("color", "depth", "ir_left", "ir_right"),
        width=848,
        height=480,
        fps=30,
        notes="Still-object round7 capture; round number intentionally follows the user's non-contiguous sequence.",
    ),
    RawObjectCaptureSpec(
        object_set="still_object",
        round_id="round8",
        raw_case_name="both_30_still_object_round8_20260428",
        capture_mode="both_eval",
        streams=("color", "depth", "ir_left", "ir_right"),
        width=848,
        height=480,
        fps=30,
        notes="Still-object round8 capture; round number intentionally follows the user's non-contiguous sequence.",
    ),
)

OBJECT_RAW_CASES: tuple[RawObjectCaptureSpec, ...] = STATIC_OBJECT_RAW_CASES + STILL_OBJECT_RAW_CASES


def get_raw_object_capture_spec(*, object_set: str, round_id: str) -> RawObjectCaptureSpec:
    normalized_object_set = str(object_set).strip().lower()
    normalized_round_id = str(round_id).strip().lower()
    for spec in OBJECT_RAW_CASES:
        if spec.object_set == normalized_object_set and spec.round_id == normalized_round_id:
            return spec
    raise KeyError(f"Unknown raw object capture: object_set={object_set!r}, round_id={round_id!r}")
