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
