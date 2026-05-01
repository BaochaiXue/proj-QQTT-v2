# 2026-05-01 SAM2.1 Checkpoint Ladder 3x5 Panel

## Goal

Generate six 3x5 time GIF benchmark panels for still-object rounds 1-4 and still-rope rounds 1-2. Rows are the three camera views, columns are existing SAM3.1 masks plus SAM2.1 large/base_plus/small/tiny video tracking masks.

## Plan

1. Add an experiment-only SAM2.1 checkpoint ladder workflow under `data_process/visualization/experiments/`.
2. Add an operator CLI under `scripts/harness/experiments/` and register it in the harness catalog.
3. Use existing `case_dir/sam31_masks` as the SAM3.1 baseline and as frame-0 bbox prompt source for SAM2.1.
4. Run every `(case, camera, checkpoint)` SAM2.1 benchmark in a separate sequential worker process with `vos_optimized=True`; exclude model load, frame preparation, init, prompt, and warmup from `inference_ms_per_frame`.
5. Render masked FFS RGB point clouds with enhanced PhysTwin-like postprocessing and write GIF, PNG, PLY, timing JSON, mask-quality JSON, and generated markdown report artifacts.
6. Add deterministic tests for bbox derivation, mask output schema, timing aggregation, and 3x5 panel shape.

## Validation

- Run focused unit tests for the new workflow.
- Run `python scripts/harness/check_all.py` in `SAM21-max`.
- Run the full six-case artifact generation after missing SAM2.1 checkpoints are available.

## Completion

Move this plan to `docs/exec-plans/completed/` after artifacts and reports are generated.
