# Demo 3.1 / Demo 3.2 same-bundle latest-wins pipeline

## Goal

Ensure rendered tracking PCD results come from one exact three-camera frame bundle.
Async stages may drop stale bundles before they start processing, but a stage that
accepts a bundle must only append results to that bundle's `group_id`.

## Decisions

- Add shared `stage_mailbox` and `frame_bundle_service` foundations.
- Keep existing Demo 3.1 caches as a compatibility bridge while routing exact
  tracker-result render matching through the shared bundle store.
- Default policy is `same-bundle-latest-wins`.
- Nearest/latest fallback is only available under explicit debug policy.
- Tracker child IPC remains CPU RGB/masks/tracks only; depth, intrinsics, and c2w
  stay in the main process.

## Validation

- Focused service tests for mailbox and bundle store.
- Demo 3.1 contract tests for default policy and exact bundle matching.
- Demo 3.1 / Demo 3.2 dry-run contract checks.
- `conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py`.
