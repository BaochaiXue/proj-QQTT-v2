# Demo v4 Single-GPU Native RealSense Realtime Optimization Acceptance

## Goal

Make the native RealSense backend run through the full fake-realtime
camera-to-final-data pipeline on one GPU at 5 FPS without lowering mask,
tracking, PCD, or shape-prior data quality. Keep single-GPU as the default
product path, retain an explicit dual-GPU isolation mode, then prove the
penultimate and fifth-from-last single-GPU chunks are accepted by
FuturePhysTwin 0-order and 1-order optimization.

## Requirements

- Use full fake realtime camera input, not offline shortcuts or prebuilt chunk
  surgery.
- Keep `depth_backend=native-realsense` and Demo v4/Demo 3.2 quality defaults:
  EdgeTAM masks, TAPNext++ tracking, same-query identity, table-world PCD
  filtering, data-process-compatible SAM3D single-view shape prior sampling.
- Route all Demo-side GPU work for native RealSense through one visible CUDA
  device/logical `cuda:0`; do not use a second GPU to meet realtime.
- Expose `--gpu-mode {single,dual}` with `single` as the default. `single`
  resolves Demo 3.2 to `CUDA_VISIBLE_DEVICES=0`; `dual` resolves Demo 3.2 to
  `CUDA_VISIBLE_DEVICES=1` so a local SAM3D worker can stay on physical GPU0.
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
7. Run a dual-GPU fake-realtime route probe that launches Demo 3.2 with
   `--gpu-mode dual` and writes at least one complete chunk.
8. Run focused tests and smoke validation; commit and push to `origin/single-camera`.

## Current Validation Results

- Default dry-run resolves `gpu_mode=single` and
  `demo32_cuda_visible_devices=0`.
- Dual dry-run resolves `gpu_mode=dual` and `demo32_cuda_visible_devices=1`.
- Single-GPU full fake-realtime run:
  `result/demo_v4/full_fake_realtime_native_single_gpu_fast_20260624/cases`
  produced seven 25-frame chunks with steady publish intervals
  `[4.722, 4.969, 4.904, 5.135, 4.874, 5.049]` seconds and zero backlog.
- Dual-GPU fake-realtime route probe:
  `result/demo_v4/dual_gpu_route_probe_20260624/cases` produced one complete
  25-frame chunk with `gpu_mode=dual`, `demo32_cuda_visible_devices=1`, and
  finite object/controller/surface/interior points.
- FuturePhysTwin accepted single-GPU chunks `0006` and `0003` through CMA and
  `train_warp.py`; exact commands and losses are recorded in
  `docs/generated/demo_v4_futurephystwin_validation_20260624.md`.
