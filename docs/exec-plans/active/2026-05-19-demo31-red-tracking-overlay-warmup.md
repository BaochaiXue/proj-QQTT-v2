# Demo 3.1 Red Tracking Overlay Warmup

## Goal

Make Demo 3.1 rendered output start with visible tracking points instead of
showing semantic PCD first and waiting for CoTracker later.

## Scope

- Render tracking overlay points in high-contrast red.
- Keep Demo 3.1 rendered frames blocked during warmup until a non-empty
  CoTracker overlay can be lifted into world coordinates.
- Preserve nonblocking behavior after the first overlay is rendered by reusing
  the latest non-stale CoTracker result.
- Add contract/profile fields and tests that make the warmup gate explicit.

## Non-Goals

- Do not change raw CoTracker query sampling, mask semantics, SAM3.1, EdgeTAM,
  RealSense capture, or Open3D renderer internals.
- Do not make the shared Demo 2.x tracking overlay path the owner for Demo 3.1.

## Validation

Run:

```bash
conda run -n demo_2_max --no-capture-output python -m py_compile \
  qqtt/demo/demo3_runtime.py \
  qqtt/demo/demo31_runtime.py \
  qqtt/demo/demo31_profile.py

conda run -n demo_2_max --no-capture-output python -m unittest -v \
  tests.test_demo3_contract \
  tests.test_demo31_dual_gpu_contract
```

## Progress

- Added a red shared tracking overlay color constant for Demo 3 / Demo 3.1.
- Added Demo 3.1 `--wait-for-tracking-overlay` / `--no-wait-for-tracking-overlay`.
- Changed Demo 3.1's overlay cap default to `0`, meaning render all visible
  controller-labeled CoTracker tracks instead of only 30 per camera.
- Made the default rendered Demo 3.1 path skip startup render packets until a
  non-empty CoTracker overlay can be lifted into world coordinates.
- Kept post-warmup rendering nonblocking by reusing the latest non-stale
  tracking result.
- Updated Demo 3.1 contract/profile fields and deterministic tests.
