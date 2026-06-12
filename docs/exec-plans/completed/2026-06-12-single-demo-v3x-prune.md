# Single Demo V3.x Prune

## Objective

Make the `single_demo_v3*` entrypoints read as true single-camera demos rather
than copied three-camera Demo 3.x surfaces with disabled multi-camera fields.

## Scope

- Remove single-demo contract fields and CLI flags that only exist to negate
  three-camera sync, world fusion, batch-3 FFS, dual-GPU tracking, shape-prior
  warmup, or mandatory calibration.
- Keep the actual single-camera live delegate path for RealSense depth and FFS
  depth.
- Update deterministic tests and operator docs to assert the reduced
  single-camera contract.

## Non-Goals

- Rewrite the underlying EdgeTAM or point-cloud render implementations.
- Change the main branch three-camera baseline.
- Automate hardware validation.

## Validation

- `conda run -n demo_2_max --no-capture-output python -m unittest -v tests.test_single_demo_v3_runtime tests.test_check_all_smoke`
- `conda run -n demo_2_max --no-capture-output python single_demo_v3/realtime_single_camera_realsense_masked_pcd.py --dry-run`
- `conda run -n demo_2_max --no-capture-output python single_demo_v3_2/realtime_single_camera_ffs_masked_pcd.py --dry-run`
- `conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py`

## Status

- 2026-06-12: Completed on branch `single-camera` after `git pull --ff-only
  origin main` reported up to date.
