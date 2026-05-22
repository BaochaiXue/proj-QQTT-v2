# Demo 3.1/3.2 Demo-Mode Controller Erode Default

## Goal

Make Demo 3.1 and Demo 3.2 use `controller_mask_erode_px=0` by default in
`demo` mode, matching the current experiment-mode default.

## Scope

- Changed the mode-specific demo default from `1` to `0`.
- Kept explicit `--controller-mask-erode-px` overrides working.
- Updated focused contract tests so both Demo 3.1 and Demo 3.2 report zero
  erode in `demo` mode unless explicitly overridden.

## Outcome

`--mode demo` now keeps the controller mask un-eroded by default in both live
demo entrypoints. Explicit overrides still win.

## Validation

- `conda run -n demo_2_max --no-capture-output python -m unittest -v tests.test_demo31_dual_gpu_contract`
- `conda run -n demo_2_max --no-capture-output python demo_v3_1/realtime_three_view_cotracker3_realsense_overlay_dual4090.py --dry-run --mode demo --camera-ids 0,1,2 --mask-gpu 0 --cotracker-gpu 1 --require-two-cuda`
- `conda run -n demo_2_max --no-capture-output python demo_v3_2/realtime_three_view_litetracker_ffs_dual4090.py --dry-run --mode demo --camera-ids 0,1,2 --mask-gpu 0 --cotracker-gpu 1 --require-two-cuda`
- `conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py`
- `git diff --check`
