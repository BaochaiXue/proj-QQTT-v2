# Demo v7 fake-live execution and bug fixes

## Requirement

- Problem: exercise the current Demo v7 fake-live camera and fix reproducible
  failures found in the actual workflow.
- Required final behavior: preview, frame-0 capture/confirmation, warmup,
  review/reposition, formal tracking/chunks, and shutdown complete with valid
  artifacts and explicit errors on failed processing.
- Inputs: existing recorded case and current Demo v7 configuration/models.
- Outputs: isolated run artifacts, reproducible regression tests, and an exact
  command/outcome report under `docs/generated/`.
- State changes: local run directories and narrowly scoped fixes in Demo v7.
- Invalid cases: broken inputs, failed stages, invalid transitions, and missing
  products must not be reported as successful runs.
- Constraints: main branch; preserve fake camera continuous-stream semantics;
  keep model weights external; do not change formal recording/alignment.
- Unknowns: actual runtime failures and GUI coverage, to resolve by execution
  and inspection before choosing fixes.

## Minimal design

- First run the existing headless driver against real local GPU dependencies
  using a fresh output directory, alongside deterministic Demo v7 tests.
- Inspect runtime logs and products, then exercise the GUI/lifecycle boundaries.
- Change only modules implicated by reproduced bugs, documenting each root
  cause and the direct correction here before implementation.
- Validate each correction with a regression test and repeat the affected
  integration path; run the repository-required smoke validation.
- Record evidence, commit validated changes, and push `origin main`.

## Progress

- Confirmed clean `main`; `git pull --ff-only origin main` reports up to date.
- Read the readable-code skill, Demo v7 README/configuration, driver, and prior
  implementation record. Available environment: `demo_2_max`, two RTX 4090 GPUs.
- The AGENTS-referenced `docs/SCOPE.md` and `docs/WORKFLOWS.md` are absent in
  this checkout; use current Demo v7 contracts and existing scope guards.
