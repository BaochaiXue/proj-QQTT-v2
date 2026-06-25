# Demo v4 Final Data Controller Mask Contract

## Goal

Remove the orphaned candidate-level `controller_mask` from Demo v4
`final_data.pkl` outputs after controller FPS has reduced
`controller_points` to the selected controller columns.

## Root Cause

`controller_mask` is produced before FPS over the full controller candidate set,
but `controller_points` in `final_data.pkl` is already reduced to the final
selected controller points. Without candidate point arrays or mapping fields in
the final artifact, the mask no longer indexes anything in that file.

## Design

- Treat `final_data.pkl` as the consumer-facing product.
- Do not write `controller_mask` into `final_data.pkl`.
- Preserve interpretable static mapping fields:
  - `controller_fps_indices`
  - `controller_selected_query_ids`
  - `object_sample_indices`
  - `object_selected_query_ids`
- Keep candidate-level `controller_mask` only in `track_process_data.pkl`, where
  trace fields can explain it.
- Extend the writer to preserve optional candidate/query trace fields in
  `track_process_data.pkl`.

## Validation

- Add tests where the controller candidate mask has a larger length than the
  final 30 controller points.
- Verify `final_data.pkl` has no `controller_mask`.
- Verify selected controller/object query ids and sample indices are present.
- Run Demo v4 focused tests.

## Status

- Incorporated into the Demo v4 stable topology payload contract work.
- Focused Demo v4/strict product/realtime online topology tests pass with this
  contract:
  `70 passed`.
- Smoke validation passes:
  `302 tests OK`, `smoke checks passed`.
