# Demo v4 Single-GPU Native RealSense Realtime Optimization Acceptance

## Goal

Make the native RealSense backend run through the full fake-realtime
camera-to-final-data pipeline on one GPU at 5 FPS without lowering mask,
tracking, PCD, or shape-prior data quality, then prove the penultimate and
fifth-from-last generated chunks are accepted by FuturePhysTwin 0-order and
1-order optimization.

## Requirements

- Use full fake realtime camera input, not offline shortcuts or prebuilt chunk
  surgery.
- Keep `depth_backend=native-realsense` and Demo v4/Demo 3.2 quality defaults:
  EdgeTAM masks, TAPNext++ tracking, same-query identity, table-world PCD
  filtering, data-process-compatible SAM3D single-view shape prior sampling.
- Route all Demo-side GPU work for native RealSense through one visible CUDA
  device/logical `cuda:0`; do not use a second GPU to meet realtime.
- Preserve 5 FPS source cadence and verify steady-state chunk output around
  25-frame, 5-second chunks after startup.
- Record per-chunk wall-clock materialization/publish timing in every chunk
  manifest so realtime cadence claims are based on publish intervals and
  backlog, not only nominal source windows.
- Select the penultimate and fifth-from-last chunks from the realtime run and
  run FuturePhysTwin 0-order and 1-order optimization on both.
- If optimization rejects the current single-view final_data format, add
  compatibility support without degrading the produced data.

## Steps

1. Inspect current Demo v4 CLI/device routing, generated chunk manifests, and
   FuturePhysTwin optimization commands.
2. Add failing tests for any missing single-GPU routing or single-view optimizer
   compatibility behavior.
3. Implement the minimal code/docs needed so native RealSense Demo v4 can run on
   one GPU and expose deterministic validation commands.
4. Run full fake realtime native RealSense Demo v4 with shape-prior worker
   preload/warmup and collect chunks.
5. Audit cadence/FPS, per-chunk materialization latency, publish lag, backlog,
   GPU routing metadata, chunk count, masks/tracks/PCD/shape fields, and select
   chunk `N-1` plus chunk `N-4`.
6. Run FuturePhysTwin 0-order and 1-order optimization on both selected chunks,
   record final losses/outputs, and save a generated validation report.
7. Run focused tests and smoke validation; commit and push to `origin/single-camera`.
