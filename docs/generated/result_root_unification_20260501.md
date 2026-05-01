# Result Root Unification

- Date: `2026-05-01`
- Canonical local result root: `result/`
- Formal aligned data root remains: `data/`
- Raw recording root remains: `data_collect/`

## Moved

- `data/experiments/*` -> `result/*`
- `data/experiments/_archived_obsolete/*` -> `result/_archived_obsolete/*`
- `data/experiments/_archived_invalid_for_qqtt/*` -> `result/_archived_invalid_for_qqtt/*`
- Legacy top-level `result/*` outputs from the earlier visualization era ->
  `result/_archived_obsolete/legacy_result_root_20260410_20260415/`

`data/experiments/` was removed after it became empty.

## Deleted

Clearly disposable local result outputs:

- `result/grouped_resolution_triplet_smoke`
- `result/tmp_visualize`
- `result/logs/sam31_timing_round7_cam0_human_hand_object_20260428_165848.log`

## Renamed / Archived

Moved obsolete but useful roots out of top-level `result/`:

- `result/sam21_checkpoint_ladder_3x5_time_gifs_ffs203048_iter4_trt_level5` ->
  `result/_archived_obsolete/sam21_checkpoint_ladder_3x5_time_gifs_ffs203048_iter4_trt_level5_nonstable_obsolete_20260429`
- `result/ffs_trt_two_stage_weight_speed_scale1_iter4_20260428` ->
  `result/_archived_obsolete/ffs_trt_two_stage_weight_speed_scale1_iter4_pre_builderopt5_obsolete_20260428`

## Current Top-Level Results To Prefer

- `result/ffs_trt_static_rounds_848x480_pad864_builderopt5_rtx5090_laptop_20260428`
- `result/ffs_static_replay_matrix_concurrent3view_20260422_fullrun`
- `result/sam21_checkpoint_ladder_3x5_time_gifs_ffs203048_iter4_trt_level5_stable_throughput`
- `result/sam21_dynamics_checkpoint_ladder_3x5_time_gifs_ffs203048_iter4_trt_level5_maskinit_stable_throughput`
- `result/edgetam_still_object_round1_cam0_validation_20260501`

Reporting rule: do not cite anything under `result/_archived_obsolete/` or
`result/_archived_invalid_for_qqtt/` as a current result.
