# 2026-05-01 SAM2.1/EdgeTAM Mask Overlay 3x3 GIF

## Goal

Generate a 3x3 time GIF for `ffs_dynamics_round1` that compares only the masks
from SAM2.1 Small, SAM2.1 Tiny, and compiled EdgeTAM against the existing
SAM3.1 masks.

## Plan

- Keep the work experiment-only.
- Reuse existing masks; do not rerun SAM2.1 or EdgeTAM.
- Rows are `cam0/cam1/cam2`.
- Columns are `SAM2.1 Small`, `SAM2.1 Tiny`, and `EdgeTAM compiled`.
- Render RGB background with mask-difference overlay:
  - green: candidate and SAM3.1 overlap
  - red: SAM3.1 only
  - cyan: candidate only
- Label each cell with IoU and candidate/SAM3.1 area ratio.
- Generate GIF, first-frame PNG, JSON summary, and Markdown report.

## Validation

- Add deterministic unit coverage for mask comparison stats and overlay shape.
- Run focused unittest and `scripts/harness/check_all.py`.
- Render the full 71-frame GIF from existing round1 masks.

## Outcome

- Added `data_process/visualization/experiments/sam21_mask_overlay_panel.py`.
- Added `scripts/harness/experiments/visualize_sam21_edgetam_mask_overlay_3x3_gif.py`.
- Generated:
  - `result/sam21_dynamics_checkpoint_ladder_3x5_time_gifs_ffs203048_iter4_trt_level5_maskinit_stable_throughput/gifs/ffs_dynamics_round1_mask_overlay_3x3_small_tiny_edgetam_compiled.gif`
  - `result/sam21_dynamics_checkpoint_ladder_3x5_time_gifs_ffs203048_iter4_trt_level5_maskinit_stable_throughput/first_frames/ffs_dynamics_round1_mask_overlay_3x3_small_tiny_edgetam_compiled_first.png`
  - `docs/generated/sam21_edgetam_mask_overlay_3x3_benchmark.md`
  - `docs/generated/sam21_edgetam_mask_overlay_3x3_results.json`
- Verified the GIF is 71 frames at 1052x662.
- Focused unittest passed.
- `conda run --no-capture-output -n SAM21-max python scripts/harness/check_all.py` passed.
