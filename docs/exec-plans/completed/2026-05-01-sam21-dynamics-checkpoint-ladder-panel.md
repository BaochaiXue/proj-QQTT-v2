# 2026-05-01 SAM2.1 Dynamics Checkpoint Ladder Panel

## Goal

Generate two dynamics 3x5 time GIF panels comparing existing/generated SAM3.1 masks with SAM2.1 large/base_plus/small/tiny video tracking masks.

## Plan

- Add a dynamics case set for `ffs_dynamics_round1_20260414` and `ffs_dynamics_round2_20260415`.
- Generate missing `sloth` SAM3.1 sidecar masks before SAM2.1 runs.
- Generate an experiment-local FFS depth cache using the repo default `20-30-48 / valid_iters=4 / 848x480->864x480 / builderOpt5` TensorRT path, without overwriting the source aligned cases.
- Add SAM2.1 `mask` init mode using `predictor.add_new_mask(...)`; keep `box` init as the existing default for previous workflows.
- Keep SAM2.1 stable throughput timing as no-output propagate-only timing after five warmup propagations per case/view job.
- Render fused masked FFS RGB PCD through original camera pinhole views with enhanced PhysTwin-like postprocessing.

## Validation

- Add deterministic smoke coverage for dynamics case specs, mask init records, timing aggregation, depth override loading, and 3x5 composer shape.
- Run the focused unittest and `scripts/harness/check_all.py`.
- Run the full dynamics artifact command in `SAM21-max`, with SAM3.1 and FFS side jobs dispatched through `FFS-SAM-RS`.
