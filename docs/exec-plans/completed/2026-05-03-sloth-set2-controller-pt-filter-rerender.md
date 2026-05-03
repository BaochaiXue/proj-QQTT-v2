# Sloth Set 2 Controller PT-Filter Rerender

## Goal

Correct the PhysTwin-style object/controller PCD GIF so the object row uses enhanced-PT cleanup while the controller row uses the simpler PhysTwin-like radius filter.

## Scope

- Do not rerun HF EdgeTAM tracking.
- Add per-row PCD postprocess support to the Sloth Set 2 hand/object renderer.
- Rerender the object/controller GIF from existing masks:
  - `stuffed animal`: `enhanced-pt`
  - `controller`: `pt-filter`
- Keep the earlier all-enhanced controller artifact for audit history.

## Validation

- Verify the generated report records per-object postprocess modes.
- Run focused tests and deterministic harness checks.

## Outcome

- Added `pt-filter` as a renderer postprocess mode for the simple PhysTwin-like radius-neighbor cleanup.
- Added per-row controller override support:
  - default/object rows can use `enhanced-pt`
  - rows labeled `controller` can use `pt-filter`
- Rerendered the Sloth Set 2 object/controller panel without rerunning tracking:
  - object row: `enhanced-pt`
  - controller row: `pt-filter`
- Generated outputs:
  - `result/sloth_set_2_motion_ffs_hf_edgetam_object_controller_pcd/pcd_gif_object_enhanced_controller_pt_filter/gifs/sloth_set_2_motion_ffs_hf_edgetam_object_enhanced_controller_pt_filter_pcd.gif`
  - `docs/generated/sloth_set_2_motion_ffs_hf_edgetam_object_enhanced_controller_pt_filter_pcd_benchmark.md`
  - `docs/generated/sloth_set_2_motion_ffs_hf_edgetam_object_enhanced_controller_pt_filter_pcd_results.json`
- Updated harness documentation to prefer controller `pt-filter` and warn about controller `enhanced-pt`.
- Validation passed:
  - `python -m py_compile scripts/harness/experiments/run_sloth_set2_hf_edgetam_hand_object_pcd_gif.py`
  - `python scripts/harness/check_harness_catalog.py`
  - focused controller override tests
  - `python scripts/harness/check_all.py`
