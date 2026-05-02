# 2026-05-01 sloth_base_motion_ffs Fused PCD Overlay 2x3

## Goal

Render a 2x3 time GIF for `data/different_types/sloth_base_motion_ffs`
that compares fused masked PCD output from `SAM2.1 Small` and compiled
`EdgeTAM` against the existing `SAM3.1` masks.

## Plan

1. Add an experiment-only fused PCD overlay workflow under
   `data_process/visualization/experiments/`.
2. Add a harness CLI under `scripts/harness/experiments/` that reuses the
   existing sloth mask and timing artifacts without rerunning segmentation.
3. Render each cell as fused RGB PCD from all three cameras, projected through
   the requested original camera pinhole view:
   - overlap keeps RGB color
   - SAM3.1-only points are red
   - candidate-only points are cyan
4. Use explicit experiment-local `depth_scale_override_m_per_unit=0.001`
   because the cleaned `data/different_types` metadata does not contain
   `depth_scale_m_per_unit`.
5. Save GIF, first-frame PNG, first-frame representative PLYs, markdown report,
   JSON report, and update generated-doc / harness indexes.
6. Add deterministic smoke coverage for panel shape, category coloring,
   point-weighted metrics, and depth-scale override behavior.

## Validation

- `conda run --no-capture-output -n SAM21-max python -m unittest tests.test_sam21_checkpoint_ladder_panel_smoke`
- `conda run --no-capture-output -n SAM21-max python scripts/harness/check_all.py`
- Full render command for all 86 frames.

## Outcome

- Rendered
  `result/sloth_base_motion_ffs_fused_pcd_overlay_2x3/gifs/sloth_base_motion_ffs_fused_pcd_overlay_2x3_small_edgetam_compiled.gif`.
- Wrote report files:
  - `docs/generated/sloth_base_motion_ffs_fused_pcd_overlay_2x3_benchmark.md`
  - `docs/generated/sloth_base_motion_ffs_fused_pcd_overlay_2x3_results.json`
