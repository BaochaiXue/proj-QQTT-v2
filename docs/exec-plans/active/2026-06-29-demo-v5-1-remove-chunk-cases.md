# Demo v5.1 Flatten Output Paths

## Requirement

Problem:
Demo v5.1 outputs are nested under run-specific capture directories and
case-prefix directories. That makes the active shape-prior, online chunks, and
aggregate final data hard to inspect.

Required final behavior:
Demo v5.1 writes the user-visible products to fixed paths under `outputs/`:

- `outputs/shape_prior_case/shape_prior_frame0/final_data.pkl`
- `outputs/shape_prior/points.npz`
- `outputs/data/final_data.pkl`
- `outputs/online_data/chunks/chunk_000000.pkl`

The capture producer may use `outputs/capture/` for `frames.jsonl`,
`prepared_phystwin/`, and metadata. That directory is internal scratch, not a
final result directory.

Inputs:
Existing live/fake-live capture frames, source-headless captures, shape-prior
surface/interior arrays, chunk settings, and visualizer settings.

Outputs:
Fixed `outputs/` paths above plus `outputs/online_data/manifest.json` and
`outputs/run_summary.json`.

State changes:
Each new live/fake-live run clears stale `capture`, `shape_prior_case`,
`shape_prior`, `data`, and `online_data` outputs before writing. Source-headless
conversion clears generated outputs but does not delete the source capture.

Invalid cases:
Keep existing fail-fast validation for malformed chunk tensors, changed query
schema, invalid/degraded track status, and missing required shape prior.

Constraints:
Do not add compatibility switches. Do not preserve case-prefix output
directories. Do not change shape-prior geometry behavior in this path-only
change.

Unknowns:
None affecting correctness.

## Plan

- [x] Flatten online chunk and aggregate final-data writer paths.
- [x] Fix orchestrator capture, cleanup, and run-summary paths.
- [x] Route shape-prior case and points export to fixed `outputs/` paths.
- [x] Update visualizer path inference for flat `online_data`.
- [x] Update focused tests for fixed output paths and pure shape-prior
      `points.npz`.
- [x] Run focused compile/unit validation.
