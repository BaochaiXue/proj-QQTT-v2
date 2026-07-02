# Demo v5.1 Cross-Chunk Tracking Design Spec

**Goal:** Add a detailed Chinese design spec for Demo v5.1 cross-chunk
tracking and its difference from the original data process paths.

## Requirement

Problem:
`demo_v5_1/design_spec.md` does not exist. Existing `pipeline.md` explains the
warmup path, but it does not explain how online chunks preserve object and
controller identity across chunk boundaries.

Required final behavior:
Create `demo_v5_1/design_spec.md` with a careful explanation of:

- How fixed-size online chunks are closed and published.
- How tracker query identity, object columns, and controller handles remain
  stable across chunks.
- How shape-prior points gate materialization without defining tracking
  identity.
- How Demo v5.1 differs from `data_process/record_data_align.py`.
- How Demo v5.1 reuses but changes the original `data_process_origin` /
  `data_process_sam3d` tracking and final-data semantics.

Inputs:
Current `demo_v5_1` implementation, `demo_v5_1/phystwin_strict_product.py`,
`data_process/record_data_align.py`, `data_process_origin/`, and
`realtime_phystwin` online reader expectations.

Outputs:
`demo_v5_1/design_spec.md`.

State changes:
Documentation only.

Invalid cases:
The spec must not imply that per-chunk tracking can reselect topology or that
aligned-case generation carries object/controller query state.

Constraints:
Keep runtime code unchanged. Keep the document scoped to Demo v5.1 and current
single-camera online tracking behavior.

Unknowns:
No correctness-blocking unknowns.

## Plan

- [x] Inspect current Demo v5.1 pipeline, chunk writer, streaming selectors, and
      original data process paths.
- [x] Use parallel explorer agents to cross-check the design and risks.
- [x] Add this execution plan.
- [x] Add `demo_v5_1/design_spec.md`.
- [x] Run focused documentation validation.

## Validation

- `git diff --check -- demo_v5_1/design_spec.md docs/exec-plans/active/2026-07-01-demo-v5-1-cross-chunk-tracking-design.md` passed.
- `conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile smoke` passed.
