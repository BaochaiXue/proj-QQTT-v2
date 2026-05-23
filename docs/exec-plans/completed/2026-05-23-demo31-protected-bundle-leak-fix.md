# Demo 3.1 / 3.2 Protected Bundle Leak And Metrics Fix

## Goal

Fix protected bundle leaks in Demo 3.1 / 3.2 tracker-result-driven rendering and make same-bundle profile metrics distinguish rendered-only success from tracker-result-level miss rate.

## Scope

- Return replaced latest-wins tracker inputs to the runtime and unprotect their frame bundles.
- Add protected-bundle TTL/window pruning as a safety net for stale or dropped tracker inputs.
- Make `BundleStore.take_for_tracker_result()` consume matched bundles.
- Unprotect both tracker result group and matched render/bundle group on nearest-debug paths.
- Fix empty `BundleProvenance` same-bundle proof and add clearer summary ratios.

## Implementation

- Added `LatestWinsQueue.publish_latest_with_info()` while preserving the old `publish_latest()` integer API.
- Added `TrackingInputPublishResult` so the Demo 3.1 process client reports replaced tracker-input group IDs.
- Demo runtime now unprotects replaced tracker-input bundles immediately and also prunes protected bundles by TTL/window.
- Tracker-driven render paths now unprotect overlay and matched bundle groups across success, blocked, missing, and nearest-debug cases.
- `BundleStore.take_for_tracker_result()` now consumes exact/nearest matched bundles to prevent duplicate tracker results from rendering the same bundle twice.
- `BundleProvenance.asdict()` no longer treats `bundle_group_id` alone as same-bundle proof.
- Added profile fields for replaced input groups, protected-bundle pruning, oldest protected age, and tracker-result-denominator same-bundle ratios.
- Documented that `display_last_complete_while_waiting=true` means the Open3D viewer keeps the last complete frame visible, not that runtime republishes stale frames at target FPS.

## Validation

- `conda run -n demo_2_max --no-capture-output python -m pytest tests/test_demo_services_latest_wins.py tests/test_frame_bundle_service.py tests/test_stage_mailbox.py tests/test_demo31_dual_gpu_contract.py tests/test_demo31_ipc_latest_wins.py -q`
  - Passed: `86 passed, 6 subtests passed`
- `conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py`
  - Passed quick deterministic harness: `365 tests`, plus scope/catalog/boundary checks.
