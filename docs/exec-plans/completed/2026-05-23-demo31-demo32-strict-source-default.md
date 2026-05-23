# Demo 3.1 / 3.2 Strict Same-Source Default

## Goal

Make strict same-source same-bundle rendering the default behavior for Demo 3.1 and Demo 3.2. A newly rendered result must come from one three-camera batch group: RGB/depth/mask/query/tracker result/PCD/lift/render all share the same `group_id`.

## Scope

- Change Demo 3.1 / 3.2 defaults from latest-reuse/exact-target to strict-source same-bundle.
- Keep latest-reuse behavior available only as explicit debug/diagnostic mode.
- Gate tracker input publication when strict-source inputs are incomplete or mask source is reused.
- Treat tracker result group `N` without same-source bundle `N` as an invariant violation by default.
- Add profile fields and tests for invariant violations and incomplete bundle drops.

## Outcome

- Demo 3.1 and Demo 3.2 dry-run contracts now default to:
  - `fusion_mask_policy = strict`
  - `batch_bundle_policy = strict-source`
  - `frame_bundle_policy = strict-source`
  - `same_bundle_invariant_fail_fast = true`
  - `render_waited_for_mask = true`
- Explicit `--fusion-mask-policy latest-reuse` remains available as a diagnostic path and maps to latest-reuse debug bundle behavior when callers do not set a frame/batch policy explicitly.
- Strict tracker input publication now requires same-group RGB, depth, mask, intrinsics, c2w, object/controller masks, and a renderable BundleStore entry before a tracker input is published.
- Strict tracker-result handling now treats missing exact same-source bundles as invariant violations and increments explicit profile counters.
- Runtime and profile schema now expose strict invariant counters and incomplete-bundle drops.
- Demo 3.1 / 3.2 docs now describe rendered FPS as strict same-source track-ready FPS.

## Validation

- Passed:
  `conda run -n demo_2_max --no-capture-output python -m pytest tests/test_demo_services_latest_wins.py tests/test_frame_bundle_service.py tests/test_stage_mailbox.py tests/test_demo31_dual_gpu_contract.py tests/test_demo31_ipc_latest_wins.py -q`
- Passed:
  `conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py`
- Dry-run spot checks confirmed Demo 3.1 and Demo 3.2 print strict-source defaults.
