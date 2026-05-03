# Sloth Set 2 HF EdgeTAM Object + Controller PCD GIF

## Goal

Follow the PhysTwin-style controller convention for the Sloth Set 2 hand/object visualization: keep the object separate, but merge both hands into a single `controller` row instead of maintaining cross-view hand instance IDs.

## Scope

- Do not rerun HF EdgeTAM tracking.
- Reuse the existing union-hand two-object masks from the earlier `stuffed animal` + `hand` run.
- Relabel `hand` to `controller` in a new result root.
- Render an enhanced-PT 2x3 GIF for `stuffed animal` and `controller`.
- Document that enhanced-PT cleanup on hands/controller is display-only and can remove meaningful contact/finger points.

## Validation

- Verify relabeled mask schema is `1=stuffed animal`, `2=controller`.
- Render the enhanced-PT GIF from existing masks.
- Run catalog/focused checks and `scripts/harness/check_all.py`.

## Outcome

- Created PhysTwin-style controller result root:
  - `result/sloth_set_2_motion_ffs_hf_edgetam_object_controller_pcd/`
- Reused existing union-hand HF EdgeTAM masks without rerunning tracking.
- Relabeled mask schema to:
  - `1=stuffed animal`
  - `2=controller`
- Generated enhanced-PT controller GIF:
  - `result/sloth_set_2_motion_ffs_hf_edgetam_object_controller_pcd/pcd_gif_enhanced_pt/gifs/sloth_set_2_motion_ffs_hf_edgetam_object_controller_pcd_enhanced_pt.gif`
- Wrote generated reports:
  - `docs/generated/sloth_set_2_motion_ffs_hf_edgetam_object_controller_streaming_results.json`
  - `docs/generated/sloth_set_2_motion_ffs_hf_edgetam_object_controller_pcd_enhanced_pt_benchmark.md`
  - `docs/generated/sloth_set_2_motion_ffs_hf_edgetam_object_controller_pcd_enhanced_pt_results.json`
- Added a `Hand / Controller PCD Warning` section to `scripts/harness/README.md`.
- Validation passed:
  - `python -m py_compile scripts/harness/experiments/run_sloth_set2_hf_edgetam_hand_object_pcd_gif.py`
  - `python scripts/harness/check_harness_catalog.py`
  - focused 2x3 hand/object PCD panel tests
  - `python scripts/harness/check_all.py`
