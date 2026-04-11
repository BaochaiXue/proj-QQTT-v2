## Goal

Re-run a longer-duration `both_eval` stability test for the current 3-camera D455 setup, reconcile the result with the short 30-frame burst experiment, and refresh the repo's current probe/preflight source of truth accordingly.

## Non-Goals

- no new capture mode
- no relaxation of the default `both_eval` gate unless long-duration evidence supports it
- no retention of temporary experimental raw data

## Files To Touch

- `docs/generated/d455_stream_probe_results.json`
- `docs/generated/d455_stream_probe_results.md`
- `docs/generated/both_eval_30_frame_validation.md`
- `docs/HARDWARE_VALIDATION.md`

## Validation Plan

1. run a targeted long-duration `rgbd_ir_pair` probe for the current 3-camera serial set
2. merge the refreshed result into the current generated probe report
3. clean temporary probe artifacts
4. run deterministic checks
