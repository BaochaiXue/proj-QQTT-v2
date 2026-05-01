# 2026-05-01 SAM2.1 Ladder Original Camera View Rerender

## Goal

Overwrite the existing SAM2.1 checkpoint ladder 3x5 time GIFs so each PCD cell is rendered from the original calibrated camera viewpoint instead of the previous shared oblique orthographic view.

## Plan

1. Reuse the existing SAM2.1 masks, timing JSON, and benchmark reports; do not rerun SAM2.1 inference.
2. Change the experiment-only ladder panel renderer to build per-camera original-view pinhole camera parameters from aligned `K_color` and `c2w`, matching the `visual_compare_masked_camera_views.py` rendering contract.
3. Use a fast deterministic pinhole z-buffer rasterizer for the 30-frame GIF workload while preserving RGB coloring, per-cell FPS labels, FFS float depth, and enhanced PhysTwin-like postprocess.
4. Add or update deterministic smoke coverage so the 3x5 composer/render path validates original camera view metadata.
5. Rerender the six GIFs in `data/experiments/sam21_checkpoint_ladder_3x5_time_gifs_ffs203048_iter4_trt_level5/` with `--skip-sam2`.
6. Run focused unit tests and `scripts/harness/check_all.py`.

## Validation

- `conda run --no-capture-output -n SAM21-max python -m unittest -v tests.test_sam21_checkpoint_ladder_panel_smoke`
- `conda run --no-capture-output -n SAM21-max python scripts/harness/experiments/run_sam21_checkpoint_ladder_3x5_gifs.py --skip-sam2`
- `conda run --no-capture-output -n SAM21-max python scripts/harness/check_all.py`

## Completion

Move this plan to `docs/exec-plans/completed/` after the six GIFs are overwritten and validation passes.
