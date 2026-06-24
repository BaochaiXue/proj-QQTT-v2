# Demo v4 Unlimited Full Fake-Realtime Run

## Goal

Remove the Demo v4 default seven-chunk cap, rerun the full three-minute
native-RealSense fake-live stream, and validate the full-run second-last and
fifth-last chunks with FuturePhysTwin 0-order CMA plus 1-order training.

## Requirements

- `--max-chunks` remains available as an explicit debug/short-run limiter.
- The default is unlimited chunk streaming.
- In fake-live mode, unlimited means run until the recording source finishes;
  do not synthesize offline chunks or skip the realtime headless capture path.
- Keep `gpu_mode=single`, `depth_backend=native-realsense`, 5 FPS replay, 25
  frames per 5-second chunk, TAPNext++, EdgeTAM, strict tracking, PCD/filter
  defaults, and data-process-compatible shape-prior point fields.
- Use the full run's `validation_chunk_cases`, which select chunk `N-1` and
  chunk `N-4`, for FuturePhysTwin optimization.

## Steps

1. Change Demo v4 default `max_chunks` to unlimited and update docs/tests.
2. Run focused tests for Demo v4 CLI and chunk writer behavior.
3. Run the full fake-live native RealSense single-GPU experiment with no
   `--max-chunks` argument.
4. Audit chunk count, publish intervals, backlog, selected validation chunks,
   and final-data geometry.
5. Run FuturePhysTwin `optimize_cma.py` and `train_warp.py` on the selected
   chunks.
6. Run smoke validation, commit, push, and report exact results.

## Results

- Default `max_chunks` is now unlimited; `--max-chunks` remains an explicit
  debug limiter.
- Full fake-realtime camera run:
  `result/demo_v4/full_fake_realtime_native_single_gpu_unlimited_20260624_130952`
- The run used `gpu_mode=single`, `depth_backend=native-realsense`, 5 FPS
  output chunking, 25 frames per chunk, and external SAM3D-generated
  surface/interior points from the prior bootstrap artifact.
- It produced 32 complete chunks. Each chunk has 25 frames, `backlog_chunks=0`,
  `depth_backend=native-realsense`, `depth_source_internal=realsense`, and
  `ready_marker_atomic_rename` publish contract.
- Publish cadence after startup:
  p50 `4.995s`, p95 `5.415s`, max `5.576s`.
- First chunk was source window `0.0-5.0s` and published at wall `27.669s`;
  last chunk was source window `155.0-160.0s` and published at wall `182.641s`.
- Validation chunks selected from the full run:
  `demo_v4_native_single_gpu_unlimited_chunk_0031` and
  `demo_v4_native_single_gpu_unlimited_chunk_0028`.

## FuturePhysTwin Optimization Results

- `demo_v4_native_single_gpu_unlimited_chunk_0031`
  - 0-order CMA optimal error: `0.00012115112303945352`
  - 1-order train final iteration 199 loss: `0.00010096399182657478`
  - Checkpoints: `optimal_params.pkl`, `best_180.pth`, `iter_199.pth`
- `demo_v4_native_single_gpu_unlimited_chunk_0028`
  - 0-order CMA optimal error: `0.000206732975129853`
  - 1-order train final iteration 199 loss: `0.0001026425043164636`
  - Checkpoints: `optimal_params.pkl`, `best_199.pth`, `iter_199.pth`

## Verification

- Focused unit tests:
  `conda run -n demo_2_max --no-capture-output python -m unittest tests.test_demo_v4_futurephystwin_chunks tests.test_phystwin_strict_product tests.test_realtime_masked_edgetam_pcd_filter tests.test_demo32_shape_prior_warmup tests.test_single_demo_v3_runtime tests.test_single_demo_tapnextpp_overlay`
  passed with 227 tests.
- Smoke validation:
  `conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile smoke`
  passed with 301 tests and `smoke checks passed`.
