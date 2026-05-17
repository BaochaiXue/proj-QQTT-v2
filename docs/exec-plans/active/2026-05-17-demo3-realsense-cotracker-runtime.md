# Demo 3 RealSense CoTracker Runtime

## Goal

Restructure Demo 3 around a realtime visualization contract: exactly three
RealSense RGB-D cameras, HF EdgeTAM semantic masks, RealSense-depth fused PCD,
and async CoTracker3 online tracking overlay. Demo 3 is not a FuturePhysTwin
offline data-processing pipeline.

## Scope

- Add a thin `demo_v3` public entrypoint.
- Add a `qqtt.demo.demo3_runtime` facade with dry-run contract validation.
- Enforce exactly three cameras and RealSense-only depth in Demo 3.
- Keep FFS, FFS TensorRT, FFS remote, and FFS IR alignment out of the Demo 3
  live contract.
- Add a CoTracker3 overlay worker contract with latest-wins async output and
  frame-by-frame online update semantics.
- Add overlay lift helpers for `y,x` tracks using RealSense depth, intrinsics,
  and `c2w`.
- Add tests for contract validation, CoTracker worker timing/latest-wins
  behavior, and 2D-to-3D overlay lift.
- Document that FuturePhysTwin `track_process_data.pkl`, inverse physics,
  final controller selection, and full PhysTwin post-processing are out of
  scope for Demo 3.

## Non-Goals

- Do not modify Demo 2.2 behavior.
- Do not implement FuturePhysTwin inverse physics or `track_process_data.pkl`.
- Do not add FFS fallback or FFS CLI knobs to Demo 3.
- Do not make CoTracker block the main fused-PCD render path.
- Do not default to 5000 visualized tracks in the realtime overlay.

## Validation

- Dry-run CLI contract tests.
- CoTracker overlay worker unit tests with fake backends.
- Overlay lift unit tests.
- `python scripts/harness/check_all.py` in `demo_2_max` after implementation.
