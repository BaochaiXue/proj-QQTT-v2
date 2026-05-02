# 2026-05-01 sloth_base_motion_ffs Mask Overlay 3x3

## Goal

Regenerate SAM3.1, SAM2.1 Small/Tiny, and compiled EdgeTAM masks for
`data/different_types/sloth_base_motion_ffs`, then render a 3x3 time GIF that
shows mask XOR differences against SAM3.1.

## Plan

- Keep generated masks and outputs under
  `result/sloth_base_motion_ffs_mask_overlay_3x3/`.
- Do not write `sam31_masks` into the source case directory.
- Extend SAM2.1 and EdgeTAM experiment workers to accept an external
  `sam31_mask_root`.
- Add a dedicated experiment CLI that sequentially runs:
  - SAM3.1 mask generation in `FFS-SAM-RS`
  - SAM2.1 Small/Tiny mask tracking in `SAM21-max`
  - compiled EdgeTAM mask tracking in `edgetam-max`
  - 3x3 black-background XOR overlay GIF rendering
- Add deterministic tests for the new external-mask-root and black-background
  overlay behavior.

## Validation

- Run focused unittest for the SAM2.1/overlay smoke tests.
- Run `scripts/harness/check_all.py`.
- Run the full artifact workflow for all 86 frames.

## Outcome

- Added `scripts/harness/experiments/run_sloth_base_motion_mask_overlay_3x3_gif.py`.
- Extended SAM2.1 and EdgeTAM experiment workers to consume an external
  experiment-local SAM3.1 mask root.
- Generated SAM3.1 masks under
  `result/sloth_base_motion_ffs_mask_overlay_3x3/sam31_masks/`; the source case
  was not modified.
- Used `sloth,stuffed animal` as the SAM3.1 mask-generation and mask-union
  prompt alias because the single `sloth` prompt did not register an object on
  cam0.
- Generated:
  - `result/sloth_base_motion_ffs_mask_overlay_3x3/gifs/sloth_base_motion_ffs_mask_overlay_3x3_small_tiny_edgetam_compiled.gif`
  - `result/sloth_base_motion_ffs_mask_overlay_3x3/first_frames/sloth_base_motion_ffs_mask_overlay_3x3_small_tiny_edgetam_compiled_first.png`
  - `docs/generated/sloth_base_motion_ffs_mask_overlay_3x3_benchmark.md`
  - `docs/generated/sloth_base_motion_ffs_mask_overlay_3x3_results.json`
- Verified the GIF is 86 frames at 1052x662.
- Focused unittest passed.
- `conda run --no-capture-output -n SAM21-max python scripts/harness/check_all.py` passed.
