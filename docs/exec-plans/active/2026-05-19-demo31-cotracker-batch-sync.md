# Demo 3.1 CoTracker Batch And Sync Hardening

## Goal

Fix Demo 3.1 CoTracker throughput and sync observability issues:

- Avoid serial per-camera CoTracker updates when a backend can process the
  three camera streams as a batch.
- Keep a safe serial fallback for injected/fake or non-batch backends.
- Make latest-reuse mask/render/overlay freshness visible in profile and
  contract fields so rendered FPS is not confused with synchronized tracking.

## Plan

- Inspect the current CoTracker backend contract and Demo 3.1 process path.
- Add a batch-capable backend path for `CoTracker3OverlayWorker` with B=3
  update when supported.
- Preserve per-camera backend initialization and serial update as fallback.
- Add profile/snapshot fields that distinguish strict same-group data from
  latest-reuse mask and latest overlay lifting.
- Add tests for batch dispatch, serial fallback, and freshness fields.
- Make batch-view execution the default for Demo 3.0 / Demo 3.1 so three
  RealSense views are sent through one backend batch call instead of silently
  publishing serial CoTracker results.

## Validation

- Focused unittest coverage for CoTracker worker and Demo 3.1 profile/contract.
- `python -m py_compile` for touched modules.
- `conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py`.

## Results

- Added batch-first CoTracker dispatch with serial fallback and profile fields.
- Switched the public default from auto/serial fallback to explicit
  `batch` / `batch-views`; serial and auto remain available as explicit
  debugging overrides.
- Added latest-reuse mask/source group freshness reporting for Demo 3.1.
- Passed touched-module `py_compile`.
- Passed focused conda unittest modules for Demo 3 / Demo 3.1 tracking paths.
- Passed quick `scripts/harness/check_all.py`.
- Full `scripts/harness/check_all.py --full` still fails on the pre-existing
  `scripts/harness/realtime_single_camera_pointcloud.py` wrapper importing a
  missing `demo_v2` package; that failure is outside this CoTracker change.
