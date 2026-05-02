# 2026-05-01 EdgeTAM Compiled Dynamics Round1 3x6 Panel

## Goal

Regenerate the `ffs_dynamics_round1` 3x6 GIF panel with EdgeTAM as the sixth
column, using the compiled EdgeTAM image-encoder path that avoids the local
position-encoding CUDA Graph cache crash. Reuse the existing SAM3.1, SAM2.1,
and FFS depth-cache results for the other five columns.

## Plan

- Keep all changes experiment-only under `scripts/harness/experiments/` and
  `data_process/visualization/experiments/`.
- Extend the EdgeTAM video-mask worker with explicit compile modes:
  - `eager`
  - `compile_image_encoder`
  - `compile_image_encoder_no_pos_cache_patch`
- Use `compile_image_encoder_no_pos_cache_patch` for the round1 3x6 workflow.
- Keep the output mask schema compatible with the existing panel renderer.
- Label the panel/report as `EdgeTAM compiled`.
- Reuse existing SAM2.1 timing/masks and the experiment-local FFS depth cache.

## Validation

- Add deterministic coverage for the EdgeTAM compile mode parser default.
- Run the focused unittest for the ladder panel.
- Run the full round1 3x6 artifact command with `--skip-sam31-preflight` and
  existing FFS/SAM2.1 caches.
- Run `scripts/harness/check_all.py`.

## Outcome

- Added EdgeTAM worker compile modes, including the process-local
  no-position-cache compiled mode.
- Regenerated `ffs_dynamics_round1_3x6_time_edgetam.gif` with the compiled
  EdgeTAM column.
- Reused the existing SAM3.1/SAM2.1/FFS caches for the other columns.
- Verified the generated GIF is 71 frames at 1652x662.
- Recorded EdgeTAM compiled propagation-only timing:
  - cam0: 14.67 ms/frame, 68.19 FPS
  - cam1: 16.21 ms/frame, 61.67 FPS
  - cam2: 15.74 ms/frame, 63.53 FPS
  - mean: 15.54 ms/frame, 64.35 FPS
- `conda run --no-capture-output -n SAM21-max python -m unittest tests.test_sam21_checkpoint_ladder_panel_smoke` passed.
- `conda run --no-capture-output -n SAM21-max python scripts/harness/check_all.py` passed.
