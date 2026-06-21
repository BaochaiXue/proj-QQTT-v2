# Demo 3.2 Default Disable Marker Retirement

## Goal

Make permanent filtered tracking-marker retirement opt-in instead of default-on, because the current live effect is too aggressive for demo review.

## Scope

- Change Demo 3.x launcher and masked PCD runtime parser defaults to `tracker_retire_filtered_markers=False`.
- Keep `--tracker-retire-filtered-markers` as the explicit opt-in flag.
- Keep `--no-tracker-retire-filtered-markers` accepted for compatibility and explicit debugging.
- Update tests and docs so default tracking markers use per-frame residual/table-Z display gating without permanent removal.

## Validation

```bash
conda run -n demo_2_max --no-capture-output python -m unittest tests.test_single_demo_v3_runtime tests.test_single_demo_tapnextpp_overlay
conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile smoke
```
