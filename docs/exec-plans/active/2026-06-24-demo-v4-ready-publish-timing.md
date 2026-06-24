# Demo v4 READY Publish Timing

## Goal

Measure Demo v4 chunk publish latency from the source window closing until the
FuturePhysTwin case is fully validated, marked READY, atomically renamed, and
visible to consumers.

## Steps

- [x] Add tests for the new timing fields and true READY-visible publish
      latency.
- [x] Move final write, validation, READY, and atomic rename timing into the
      FuturePhysTwin chunk writer.
- [x] Update the headless bridge so window close and strict track finalize
      timing are recorded before writer publication.
- [x] Update Demo v4 docs to make `publish_latency_ms` the consumer-visible
      speed metric.
- [x] Run focused Demo v4 tests.
- [x] Run the smoke validation profile without unrelated dirty-worktree
      failures.

## Validation

- `conda run -n demo_2_max --no-capture-output python -m unittest tests.test_demo_v4_futurephystwin_chunks`
- `conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile smoke`

## Current Validation Notes

- Focused READY publish timing/atomic tests pass.
- Full Demo v4 unittest passes.
- Smoke validation passes.
