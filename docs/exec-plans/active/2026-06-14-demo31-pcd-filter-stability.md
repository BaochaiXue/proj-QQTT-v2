# Demo 3.1 PCD Filter Stability

## Problem

Demo 3.1 fake-live showed valid object/controller masks while the rendered
point-cloud layers disappeared or looked stale when `--enable-pcd-filter` was
enabled. Debug telemetry showed nonzero raw semantic points but zero filtered
object points, plus async filtered outputs older than the current frame.

## Plan

- Keep the existing mask and tracking path unchanged.
- Prevent filters from erasing nonempty semantic layers: if a filter produces
  zero points from nonzero capped input, render capped input for that layer.
- Add an async filter age guard so stale filtered outputs do not replace the
  current frame's raw PCD.
- Make the controller filter default enhanced-pt so the default
  controller keep-components value actually preserves two disconnected hand
  components.
- Fall back to capped current-frame controller PCD when controller filtering
  retains less than half of its capped points.
- Start Demo 3.x Open3D windows from a third-person orbit view by default,
  with the original camera view available via `--view-mode camera`.
- Surface the age guard through Demo 3.x launcher contract/argv.
- Add focused unit tests for empty-filter fallback and stale-filter rejection.

## Validation

- `conda run -n demo_2_max --no-capture-output python -m unittest -v tests.test_realtime_masked_edgetam_pcd_filter tests.test_single_demo_v3_runtime tests.test_check_all_smoke`
- `conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py`
- `conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py --full`
