# Sloth Set 2 HF EdgeTAM Hand/Object PCD Panel

## Goal

Build a dedicated experiment that tracks `stuffed animal` and `hand` with HF EdgeTAM streaming on `data/different_types/sloth_set_2_motion_ffs`, then renders a qualitative 2x3 fused PCD GIF panel.

## Scope

- Use SAM3.1 frame-0 masks as initialization masks for two object IDs.
- Keep EdgeTAM inference strictly frame-by-frame from PNG frames.
- Save multi-object masks in the existing `mask/mask_info_{cam}.json` plus `mask/{cam}/{obj_id}/{frame}.png` schema.
- Render rows as object labels and columns as original camera pinhole views.
- Keep the new workflow in `scripts/harness/experiments/`; do not change formal recording or alignment code.

## Implementation

- Added `scripts/harness/experiments/run_sloth_set2_hf_edgetam_hand_object_pcd_gif.py`.
- Registered the experiment in `scripts/harness/_catalog.py`.
- Added generated-doc references for the hand/object reports.
- Added focused tests for stable object ID mapping, multi-object mask writer schema, HF output `object_ids` extraction, and synthetic 2x3 PCD render.

## Experiment Outcome

- Generated SAM3.1 masks under `result/sloth_set_2_motion_ffs_hf_edgetam_hand_object_pcd/sam31_masks`.
- The direct multi-prompt SAM3.1 run returned both frame-0 objects with label `hand`; the final init root therefore merges the existing single-prompt `stuffed animal` masks with the unioned `hand` masks while leaving the generator unchanged.
- Ran HF EdgeTAM frame-by-frame streaming on all 93 frames and `cam0/cam1/cam2`.
- Default compile mode was `vision-reduce-overhead`.

## Outputs

- `result/sloth_set_2_motion_ffs_hf_edgetam_hand_object_pcd/pcd_gif/gifs/sloth_set_2_motion_ffs_hf_edgetam_hand_object_pcd.gif`
- `result/sloth_set_2_motion_ffs_hf_edgetam_hand_object_pcd/pcd_gif/first_frames/sloth_set_2_motion_ffs_hf_edgetam_hand_object_pcd_first.png`
- `result/sloth_set_2_motion_ffs_hf_edgetam_hand_object_pcd/pcd_gif/first_frame_ply/`
- `docs/generated/sloth_set_2_motion_ffs_hf_edgetam_hand_object_pcd_benchmark.md`
- `docs/generated/sloth_set_2_motion_ffs_hf_edgetam_hand_object_pcd_results.json`
- `docs/generated/sloth_set_2_motion_ffs_hf_edgetam_hand_object_streaming_results.json`

## Validation

- `conda run --no-capture-output -n SAM21-max python scripts/harness/check_harness_catalog.py`
- `conda run --no-capture-output -n SAM21-max python -m unittest -v tests.test_sam21_checkpoint_ladder_panel_smoke`
- `conda run --no-capture-output -n SAM21-max python scripts/harness/check_all.py`

All validation passed.
