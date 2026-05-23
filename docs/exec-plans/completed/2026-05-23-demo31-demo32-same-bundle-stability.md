# Demo 3.1 / 3.2 Same-Bundle Stability Guards

## Goal

Stabilize the default same-bundle runtime before long live profiling. Keep the existing Demo 3.1 / 3.2 architecture and defaults, but remove the stage-mailbox active-completion footgun and add profile/test guardrails around bundle consumption and Demo 3.2 all-tracks lifting.

## Scope

- Fix `LatestOnlyStageMailbox.complete_active()` so completion is not identity-only.
- Add tests for replaced immutable bundle completion and pending handoff.
- Add runtime/profile counters for bundle-taken render/lift failure paths.
- Add tests that tracker input publication happens after bundle attach/protection.
- Add Demo 3.2 all-tracks lift cap with deterministic selection and profile fields.
- Update Demo 3.2 docs and profile schema as needed.

## Implementation

- `LatestOnlyStageMailbox.complete_active()` now supports no-arg completion, `group_id` completion, and same-`group_id` replacement completion without relying only on Python object identity.
- Runtime summaries now expose bundle-consumption guard counters:
  - `bundle_taken_then_render_failed_count`
  - `bundle_consumed_without_render_count`
  - `bundle_taken_render_success_count`
  - `bundle_taken_surface_anchor_missing_count`
  - `bundle_taken_lift_input_missing_count`
- Demo 3.2 all-tracks lift now defaults to a render-only cap of `512` points per camera with `visible-spread` selection. Passing `0` keeps all valid lifted tracks for explicit debug/quality runs.
- All-tracks lift profile entries now include candidate, selected, rendered, cap-applied, timing, and exact-depth-group fields.
- Demo 3.2 README and Demo 3.1 runtime contract docs now describe the all-tracks cap and new profile counters.

## Validation

- `conda run -n demo_2_max --no-capture-output python -m pytest tests/test_stage_mailbox.py tests/test_frame_bundle_service.py tests/test_demo31_dual_gpu_contract.py -q`
  - Passed: `81 passed, 6 subtests passed`
- `conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py`
  - Passed quick deterministic harness: `368 tests`, plus scope/catalog/boundary checks.
