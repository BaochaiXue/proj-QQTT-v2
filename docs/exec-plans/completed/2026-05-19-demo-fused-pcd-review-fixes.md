# Demo Fused PCD Review Fixes

## Goal

Fix the fused point-cloud demo issues found during review: Demo 3.1 tracking
overlay/depth alignment, semantic fusion layer keying, stale fused-PCD smoke
test import, and live-runtime dependence on experiment-only postprocess helpers.

## Scope

- Keep changes inside sanctioned demo/tracking diagnostic code and focused tests.
- Preserve the existing user-facing object/controller prompt behavior.
- Do not change formal recording/alignment outputs.
- Do not touch generated calibration artifacts already present in the worktree.

## Plan

1. Add a stable demo PCD postprocess helper module and point live fusion to it.
2. Make semantic fusion use stable semantic roles internally instead of prompt
   label strings.
3. Cache Demo 3.1 lift inputs by tracking group and lift overlay results with
   the matching group instead of whichever depth happens to be latest.
4. Fix the stale fused-PCD smoke import and add focused regression coverage.
5. Run targeted unit tests plus the deterministic harness check if practical.

## Validation Targets

- `conda run -n demo_2_max --no-capture-output python -m unittest -v tests.test_demo_v2_1_three_view_fused_pcd_smoke tests.test_demo31_dual_gpu_contract tests.test_demo31_ipc_latest_wins`
- `conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py`

## Results

- Added `qqtt/demo/pcd_postprocess.py` as the stable demo-owned source for
  PhysTwin-like radius and enhanced component PCD filters.
- Updated single-camera and three-view demo PCD filter paths to import the
  stable demo helper instead of experiment-only visualization code.
- Updated semantic fused layer grouping to use `OBJECT_ID` / `CONTROLLER_ID`
  internally, so identical prompt labels cannot merge object and controller
  clouds before filtering.
- Added Demo 3.1 group-aligned lift-input caching. CoTracker overlay results
  are now lifted with the depth/intrinsics/c2w/mask snapshot from the result's
  source group, not the latest render depth.
- Fixed the stale fused-PCD smoke test import and added regression coverage for
  prompt-label collisions and Demo 3.1 group-aligned overlay lifting.
- PASS:
  `/home/xinjie/miniforge3/bin/conda run -n demo_2_max --no-capture-output python -m unittest -v tests.test_demo_v2_1_three_view_fused_pcd_smoke tests.test_demo_v2_2_async_filtered_fused_pcd_smoke tests.test_demo31_dual_gpu_contract tests.test_demo31_ipc_latest_wins`
- PASS:
  `/home/xinjie/miniforge3/bin/conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py`
