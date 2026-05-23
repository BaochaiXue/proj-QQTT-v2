# Demo 3.1 / 3.2 Full Same-Bundle Mailbox Completion

## Goal

Complete the same-bundle latest-wins implementation for Demo 3.1 and Demo 3.2 by wiring the shared stage mailbox into the live runtime and making rendered-profile summaries explicitly report display/new-complete-bundle FPS plus per-stage pending drops.

## Scope

- Keep the current `BundleStore` exact-group tracker-result-driven render path.
- Keep legacy pending render/fusion caches as compatibility bridges.
- Wire `LatestOnlyStageMailbox` into the capture/depth/mask join stage so accepted bundles are protected from pending replacement until the runtime completes processing that joined bundle.
- Expose mailbox counters and per-stage pending/drop fields in runtime snapshots and profile summaries.
- Add focused deterministic tests for mailbox runtime use and the new summary fields.

## Result

Implemented. Demo 3.1 / 3.2 now use the shared latest-only stage mailbox in the capture/depth/mask join buffer. The active joined bundle is not replaced until the runtime completes that exact `group_id`; pending joined bundles remain latest-wins. Profile summaries now include `display_loop_fps`, `new_complete_bundle_fps`, `stage_mailbox_pending_drop_count`, and per-stage pending/drop counters.

## Validation

- `conda run -n demo_2_max --no-capture-output python -m pytest tests/test_frame_bundle_service.py tests/test_stage_mailbox.py tests/test_demo31_dual_gpu_contract.py -q`
  - Passed: `70 passed, 6 subtests passed`
- `conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py`
  - Passed: `361 tests`

## Notes

Hardware rendered profiling is still manual. This change makes the next 60s Demo 3.1/3.2 rendered profiles able to prove same-bundle FPS/drop behavior without relying on mixed latest caches.
