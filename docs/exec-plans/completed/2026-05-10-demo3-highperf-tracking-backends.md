# Demo 3 High-Performance Tracking Backend Probes

## Goal

Extend Demo 3 beyond the CoTracker3 baseline with dependency-gated probes for high-performance tracking candidates: NVOFA, VPI LK, TAPNext/TAPNext++, LocoTrack, TAPIR, and ONNX Runtime CUDA/TensorRT execution-provider probes.

## Scope

- Add a fast stack availability report under `docs/generated/`.
- Add optional install helper scripts that target an isolated conda environment based on the Demo 2 stack.
- Add backend stubs/probes for NVOFA, VPI LK, TAPNext, LocoTrack, and TAPIR.
- Add ONNX Runtime TensorRT provider configuration and a non-fatal export/session probe CLI.
- Extend the benchmark harness with `auto_highperf` availability selection.
- Keep missing optional dependencies non-fatal for `check_all.py`.

## Non-Goals

- Do not force-install external dependencies during deterministic checks.
- Do not mutate `demo_2_max` unless the operator explicitly asks the optional installer to do so.
- Do not claim NVOFA is long-term TAP; it is frame-to-frame optical-flow propagation.
- Do not put dense neural tracking into the live Demo 2.2 hot path by default.

## Validation

- Stack probe writes JSON/MD.
- Optional dependency smoke tests pass without NVOFA/VPI/TAPNext/LocoTrack installed.
- `python scripts/harness/check_all.py`.

## Outcome

- Added dependency-gated backend probes, optional installer, ONNX/TensorRT provider probe, and high-performance backend docs.
- Created `demo3_trackers` by cloning `demo_2_max`; tapnet is importable there, LocoTrack and NVOFA repos are cloned, and VPI remains unavailable.
- Confirmed ONNX Runtime CUDA and TensorRT execution providers are available.
- Kept NVOFA, VPI LK, TAPNext, LocoTrack, and TAPIR optional and non-fatal until their helper/runtime wrappers and checkpoints are configured.
- `python scripts/harness/check_all.py` passes in `demo_2_max`.
