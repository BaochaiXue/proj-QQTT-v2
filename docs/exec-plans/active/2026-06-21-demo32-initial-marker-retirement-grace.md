# Demo 3.2 Initial Marker Retirement Grace

## Goal

Prevent the first tracker marker packet after TAPNext++ query initialization from permanently retiring queries, while keeping current-frame residual/table-Z display gating active.

## Scope

- Runtime change in `qqtt/demo/realtime_masked_edgetam_pcd.py`.
- Focused retirement tests in `tests/test_single_demo_tapnextpp_overlay.py`.
- Small documentation note for Demo 3.2 tracking marker semantics.
- No CLI or metadata field renames.

## Steps

1. Update retirement tests so the initialization marker packet can hide invalid markers without reducing `query_alive_mask`.
2. Add a test proving a query hidden on the grace frame can reappear on the next frame if it passes residual/table-Z.
3. Record the tracker query initialization `seq` in runtime state.
4. Skip the permanent `alive &= residual_visibility` mutation only when building a marker packet for that initialization `seq`.
5. Keep display gating unchanged on the grace frame by applying residual/table-Z visibility before alive masking.
6. Update docs to say permanent retirement starts after the initialization marker frame.
7. Run focused unit tests, the requested runtime test set, smoke validation, and a Demo 3.2 fake-live panel smoke if the environment can open the GUI.

## Validation Commands

```bash
conda run -n demo_2_max --no-capture-output python -m unittest tests.test_single_demo_tapnextpp_overlay
conda run -n demo_2_max --no-capture-output python -m unittest tests.test_single_demo_v3_runtime tests.test_single_demo_tapnextpp_overlay tests.test_demo32_headless_render_helper tests.test_demo32_side_by_side_panel
conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile smoke
```
