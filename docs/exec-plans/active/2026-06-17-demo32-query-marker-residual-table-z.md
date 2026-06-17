# Demo 3.2 Query Marker Residual + Table-Z Gate

## Goal

Make Demo 3.2 TAPNext++ query markers obey the same filtered residual and
table-Z constraints as the rendered PCD. Initial query selection and per-frame
displayed markers should both come from the active filter preset residual after
table-Z filtering.

## Implementation Steps

1. Add focused tests for filtered residual masks after table-Z and strict marker
   display gating.
2. Refactor residual-mask construction so filtered survivor `yx` is preserved
   through table-Z filtering alongside `xyz` and colors.
3. Use the post-filter, post-table-Z residual masks for initial TAPNext++ query
   selection.
4. Gate per-frame marker display by current residual masks before lift; hidden
   markers stay tracked internally but are absent from rendered/saved marker
   arrays and counts.
5. Add metadata/docs note that tracker markers use the
   `pcd_filter_residual_table_z` display gate.
6. Run targeted unit tests, a short fake-live headless tracking capture/render,
   and the smoke harness.

## Validation

- `conda run -n demo_2_max --no-capture-output python -m unittest tests.test_single_demo_v3_runtime tests.test_single_demo_tapnextpp_overlay tests.test_demo32_headless_render_helper tests.test_realtime_masked_edgetam_pcd_filter`
- Short Demo 3.2 fake-live headless tracking capture and render; verify query
  totals are nonzero and missing query frames are zero.
- `conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile smoke`
