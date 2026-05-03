# Sloth Set 2 HF EdgeTAM Hand A/B 3x3 PCD GIF

## Goal

Relabel the previous `left hand` / `right hand` three-object output as conservative `hand A` / `hand B` and regenerate the 3x3 enhanced-PT PCD GIF.

## Scope

- Do not rerun HF EdgeTAM tracking.
- Reuse the existing stable `obj_id=2` and `obj_id=3` masks from the three-object run.
- Create a relabeled result root and generated reports that avoid physical left/right claims.
- Keep the previous left/right artifacts for audit history.

## Validation

- Verify the relabeled mask schema is `1=stuffed animal`, `2=hand A`, `3=hand B`.
- Render the enhanced-PT 3x3 GIF from existing masks.
- Run deterministic harness checks.

## Outcome

- Created relabeled result root `result/sloth_set_2_motion_ffs_hf_edgetam_object_hand_ab_pcd/`.
- Reused the existing three-object HF EdgeTAM masks without rerunning tracking.
- Relabeled mask schema to:
  - `1=stuffed animal`
  - `2=hand A`
  - `3=hand B`
- Generated enhanced-PT 3x3 PCD GIF:
  - `result/sloth_set_2_motion_ffs_hf_edgetam_object_hand_ab_pcd/pcd_gif_enhanced_pt/gifs/sloth_set_2_motion_ffs_hf_edgetam_object_hand_ab_pcd_enhanced_pt.gif`
- Wrote generated reports:
  - `docs/generated/sloth_set_2_motion_ffs_hf_edgetam_object_hand_ab_streaming_results.json`
  - `docs/generated/sloth_set_2_motion_ffs_hf_edgetam_object_hand_ab_pcd_enhanced_pt_benchmark.md`
  - `docs/generated/sloth_set_2_motion_ffs_hf_edgetam_object_hand_ab_pcd_enhanced_pt_results.json`
- Validation passed:
  - `python scripts/harness/check_harness_catalog.py`
  - focused 3x3 panel smoke
  - `python scripts/harness/check_all.py`
