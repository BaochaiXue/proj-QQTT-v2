# Sloth Set 2 HF EdgeTAM Hand/Object Enhanced PT GIF

## Goal

Generate a second hand/object fused PCD GIF that applies enhanced PhysTwin-like point-cloud postprocessing after EdgeTAM mask filtering.

## Scope

- Reuse existing HF EdgeTAM multi-object mask outputs; do not rerun tracking unless explicitly requested.
- Keep the previous raw masked PCD GIF unchanged.
- Add a renderer option for `none` vs `enhanced-pt`.
- Write enhanced outputs under a separate result subdirectory and generated report.

## Validation

- Add or update focused smoke coverage for the enhanced postprocess mode.
- Run catalog and focused tests.
- Run `scripts/harness/check_all.py`.

## Outcome

- Added a render-only `enhanced-pt` PCD postprocess mode to the Sloth Set 2 HF EdgeTAM hand/object GIF experiment.
- Reused existing frame-by-frame HF EdgeTAM masks; tracking was not rerun for this enhanced GIF.
- Generated enhanced outputs under `result/sloth_set_2_motion_ffs_hf_edgetam_hand_object_pcd/pcd_gif_enhanced_pt/`.
- Wrote generated reports:
  - `docs/generated/sloth_set_2_motion_ffs_hf_edgetam_hand_object_pcd_enhanced_pt_benchmark.md`
  - `docs/generated/sloth_set_2_motion_ffs_hf_edgetam_hand_object_pcd_enhanced_pt_results.json`
- Validation passed:
  - `python -m py_compile scripts/harness/experiments/run_sloth_set2_hf_edgetam_hand_object_pcd_gif.py`
  - focused hand/object raw and enhanced panel smoke tests
  - `python scripts/harness/check_harness_catalog.py`
  - `python scripts/harness/check_all.py`
