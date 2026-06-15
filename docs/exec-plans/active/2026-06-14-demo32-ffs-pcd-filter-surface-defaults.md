# Demo 3.2 FFS PCD Filter Surface Defaults

## Problem

Demo 3.2 can show scattered FFS point-cloud points even with
`--enable-pcd-filter`. The wrapper only forwards the on/off switch and a few
component controls, while the delegate keeps the generic realtime filter
defaults. Those defaults use a 40-neighbor radius filter after a 20k voxel cap,
which can be too strict for capped single-camera FFS surfaces and can trigger
raw/capped fallback.

## Plan

- Keep the existing EdgeTAM, FFS, TAPNext++, and Open3D render paths unchanged.
- Expose and forward the existing PCD filter controls from the Demo 3.x wrapper:
  filter scheduling mode, object/controller filter types and caps, radius,
  neighbor count, component voxel size, filter cadence, and budget controls.
- When FFS Demo 3.2/3.3 runs with `--enable-pcd-filter`, use lighter
  enhanced-pt defaults suitable for capped realtime surfaces unless explicitly
  overridden.
- Keep async filtering as the default to avoid freezing the demo, but make the
  filtered output fresher by default for FFS demos.
- Update runtime contract/tests so dry-run reveals the active filter behavior.

## Validation

- `conda run -n demo_2_max --no-capture-output python -m unittest -v tests.test_single_demo_v3_runtime tests.test_realtime_masked_edgetam_pcd_filter`
- `conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py`
- Manual Demo 3.2 fake-live smoke with `--enable-pcd-filter --debug`.
