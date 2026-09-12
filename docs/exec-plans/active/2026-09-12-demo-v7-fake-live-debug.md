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
- Baseline: 150 Demo v7 tests and repository smoke profile pass. The actual
  default TRELLIS.2/upscale/TripoSplat warmup and Gaussian re-roll complete.
- Reproduced a chunk-failure reporting bug with three failing regressions:
  the parent never sends its chunk error to the GUI or stops production;
  `wait_for_state(FINISHED)` returns despite an already-latched chunk failure;
  the GUI displays a normal finish after chunk processing has failed.
- Direct correction: forward the existing error event with `where=chunk_stream`,
  stop formal capture and downstream processing, make waiters inspect failures
  before successful states, and keep the GUI's fatal screen terminal until a
  new session. No protocol messages or fallback behavior are added.
- Another local session committed mesh-surface changes as `51fbb32` during
  this run, also picking up this plan and the three new (then failing) tests.
  Keep its implementation intact and commit this task's fixes separately.
- First GPU drive passed: 11 chunks, 56 processed Gaussian frames (55 sent),
  valid points.npz, successful self-alignment, and clean shutdown.
- GUI drive exercised capture/retake; its temporary assertion used the wrong
  Gaussian artifact filename and was corrected before a fresh run. Stopping
  that first service also reproduced terminal link errors only appearing in
  the status bar. Treat these existing error sources as terminal in the GUI,
  latch the failure for waiters/status, and cover late FINISHED/HELLO events.

## Validation and outcome

- Default GPU drive: PASS, 11 chunks, 57 contiguous formal frames, Gaussian
  generation/re-roll/live rendering and fresh shape-prior products verified.
- Actual Qt GUI with default PhysTwin downstream: PASS, capture/retake/confirm,
  review/reposition, formal/stop/finish; 26 chunks, 131 contiguous formal frames.
  PhysTwin ran two CMA iterations and loaded 130 chunk frames into Train-Online;
  normal GUI teardown then stopped its process group.
- Short recorded-case EOF + shape-prior none: PASS, actual replay-ended dialog,
  6 chunks, 32 contiguous frames, and no shape-prior points.npz.
- Seven regression cases cover chunk error delivery, stopping producers,
  failure-first waiters, sticky GUI failures for all three terminal sources,
  persisted link failures, and expected disconnects during requested shutdown.
- Full Demo v7 suite: 159 passed. Repository smoke: passed (66 unittest cases,
  configured CLI checks and all guards). `git diff --check` passes.
- Exact commands, interrupted diagnostic explanation, results, and artifact
  paths: `docs/generated/2026-09-12-demo-v7-fake-live-debug-proof.md`.
- Final process inspection found no remaining camera service or downstream
  supervisor from these runs. No real-camera hardware validation was claimed.

Implementation and validation complete; commit and push the scoped changes to
`origin main` according to the repository workflow.
