# Demo 3 Tracking Backend Overlay

## Goal

Implement the first repo-local slice of Demo 3 as a three-camera tracking backend benchmark plus PhysTwin-style 3D temporal anchor overlay. CoTracker3 online is the first real backend, while NVOFA, TAPNext, LocoTrack, TAPIR, and VPI LK are represented through availability checks until their external runtimes are installed.

## Scope

- Add a Demo 3 contract doc describing the benchmark and overlay semantics.
- Add `qqtt/tracking/` with backend interface, registry, deterministic query sampling, CoTracker-like NPZ IO, metrics, and 2D track to world-space anchor lifting.
- Add experiment harness CLIs for backend availability and offline benchmark output.
- Add Demo 2.2 optional tracking overlay flags and profile contract fields, disabled by default.
- Add deterministic smoke tests for coordinate convention, sampling, IO, lift filtering, availability, and harness output.

## Non-Goals For This Slice

- Do not run tracking in the Demo 2.2 render hot path by default.
- Do not require external tracking repos or model weights in CI.
- Do not add tracking dependencies to the camera-only environment installer.
- Do not claim NVOFA/TAPNext/LocoTrack support until availability probes pass on the operator machine.

## Validation

- Targeted unittest modules for `qqtt.tracking`.
- Harness help/availability commands.
- `python scripts/harness/check_all.py`.

## Outcome

- Added the Demo 3 tracking contract, `qqtt.tracking` package, CoTracker3 baseline wrapper, fake CI backend, deterministic sampling, PhysTwin-compatible NPZ IO, 3D lift, metrics, benchmark harness, and offline overlay harness.
- Added optional Demo 2.2 tracking overlay flags, disabled by default.
- Added deterministic smoke tests for y,x convention, sampling, IO, lift filtering, registry behavior, fake benchmark output, and overlay export.
- `python scripts/harness/check_all.py` passes in `demo_2_max`.
