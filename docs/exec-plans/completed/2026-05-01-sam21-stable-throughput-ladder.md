# 2026-05-01 SAM2.1 Stable Throughput Ladder

## Goal

Rerun the SAM2.1 checkpoint ladder with a stable-throughput timing contract that keeps each checkpoint worker hot across all six aligned cases and three cameras, instead of timing one short case/camera/checkpoint worker at a time.

## Plan

1. Add a stable-throughput worker mode for SAM2.1 ladder timing.
2. Build each SAM2.1 checkpoint once per worker with `vos_optimized=True`, `bfloat16` autocast, and TF32 enabled when available.
3. Prepare all selected case/camera 30-frame JPEG video directories before timed inference.
4. Warm up each case/camera job for five full `propagate_in_video` passes without collecting masks.
5. Run the selected case/camera jobs continuously in one timed checkpoint worker, collecting masks for the panel artifacts and recording per-job plus aggregate stable-throughput timing.
6. Keep the existing diagnostic worker path available for comparison, and update report text so the timing contract is explicit.
7. Validate with focused unit tests and run the full six-case artifact generation in `SAM21-max`.

## Validation

- `conda run --no-capture-output -n SAM21-max python -m unittest -v tests.test_sam21_checkpoint_ladder_panel_smoke`
- `conda run --no-capture-output -n SAM21-max python scripts/harness/experiments/run_sam21_checkpoint_ladder_3x5_gifs.py --overwrite --stable-throughput`

## Outcome

- Added `--stable-throughput` to the SAM2.1 checkpoint ladder CLI.
- Ran one long-lived worker per checkpoint over six cases and three cameras.
- Each case/camera job uses five warmup propagations before the no-output speed pass.
- A no-marker run failed on the local Torch 2.11 / CUDA 13 CUDA Graph path with the known overwritten CUDAGraph tensor error, so the completed run records per-step cudagraph markers in the speed pass.
- Mask collection is separate from the speed pass and excluded from the reported FPS.
- Wrote artifacts under `data/experiments/sam21_checkpoint_ladder_3x5_time_gifs_ffs203048_iter4_trt_level5_stable_throughput/`.
- Updated `docs/generated/sam21_max_round2_benchmark.md`, `docs/generated/sam21_max_round2_benchmark_results.json`, and `docs/generated/sam21_max_round2_mask_quality.json`.
- Validation passed:
  - `conda run --no-capture-output -n SAM21-max python -m unittest -v tests.test_sam21_checkpoint_ladder_panel_smoke`
  - `conda run --no-capture-output -n SAM21-max python scripts/harness/check_all.py`

## Completion

Move this plan to `docs/exec-plans/completed/` after the stable-throughput artifacts and generated reports are written.
