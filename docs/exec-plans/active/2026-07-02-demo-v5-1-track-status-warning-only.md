# Demo v5.1 Track Status Warning-Only Semantics

## Requirement

Problem:
`track_process_status` values currently affect Demo v5.1 runtime behavior.
`degraded` chunks can be skipped from `online_data`, and `invalid` chunks can
stop chunk materialization or make the runner return a failure code. That
creates hidden time gaps and turns a quality label into control flow.

Required final behavior:
`normal`, `degraded`, and `invalid` are warning/diagnostic metadata only. They
must remain visible in chunk payloads, static aggregate data, manifests, and run
summaries, but they must not affect chunk publication, callbacks, stream
continuation, stop reason, or process exit code.

Inputs:
Demo v5.1 `track_process_status` emitted by
`demo_v5_1/phystwin_strict_product.py` and propagated through chunk payloads.

Outputs:
Continuous online chunks and static aggregate data for every materialized
window, regardless of `track_process_status`, plus warning metadata for
downstream consumers.

State changes:
Remove status-based publish skipping and invalid-stop behavior. Remove the
obsolete degraded-publish CLI switch.

Invalid cases:
Topology/schema mismatches and missing required data still fail at their
existing validation boundaries. Only the quality status label is warning-only.

Constraints:
Keep the change scoped to Demo v5.1 chunk publication and runner semantics.
Do not alter how tracking computes quality labels.

Unknowns:
No correctness-blocking unknowns.

## Plan

- [x] Inspect current status-driven publishing, stream, and runner behavior.
- [x] Spawn read-only agents to cross-check behavior paths and test coverage.
- [x] Remove status-based skip/stop/exit behavior.
- [x] Remove obsolete degraded-publish plumbing.
- [x] Update docs to describe warning-only semantics.
- [x] Run focused unit tests.
- [x] Run smoke validation.

## Validation

- `PYTHONDONTWRITEBYTECODE=1 conda run -n demo_2_max --no-capture-output python -m unittest tests.test_demo_v5_1_chunk_data tests.test_demo_v5_1_default_config`
  passed.
- `PYTHONDONTWRITEBYTECODE=1 conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile smoke`
  passed.
