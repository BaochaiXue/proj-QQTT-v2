# Demo v6.2 -> Phystwin_shen Full-Pipeline Integration

## Goal

Replace Demo v6.2's direct `train_online_warp.py` plus single-viewer launch with
one `scripts/run_online_full_pipeline.py` supervisor. Demo v6.2 remains the
online-data producer, the only maintained runtime-parameter source, and the
owner of downstream process lifetime.

## Fixed decisions

- Demo's published timeline and PhysTwin physics both run at 5 FPS.
- `/home/xinjie/Phystwin_shen/configs/real.yaml` is globally changed to
  `FPS: 5`, `dt: 5e-5`, and `num_substeps: 4000`.
- All PhysTwin children run with the `demo_2_max` interpreter.
- `demo_v6_2/config/default.yaml` is the maintained source for common, CMA,
  train, and both viewer parameters. The Demo launcher passes those values as
  explicit full-pipeline CLI overrides.
- Demo starts the supervisor only after shape-prior `points.npz` exists, which
  also means the shape-prior GPU work has completed.
- Demo waits for the supervisor after publishing `manifest.status=finished`.
  `train.stop_when_finished` only controls the trainer's own stop condition.
- On any Demo failure or operator interruption, Demo terminates the complete
  PhysTwin process group (wrapper, active CMA/train child, and both viewers),
  escalating SIGTERM to SIGKILL.
- Before launch, Demo kills processes blocking both configured viewer ports.
- The PhysTwin online reader is not changed; Demo owns failure cleanup.
- The existing online chunk, ASAP, calibration, metadata, and RGB-D schemas do
  not change.

## Implementation

1. Add explicit, typed full-pipeline CLI overrides to
   `Phystwin_shen/scripts/run_online_full_pipeline.py` and apply them to the
   loaded config before command construction.
2. Synchronize Demo's `phystwin_shen` YAML section with the current external
   full-pipeline defaults, keeping `stop_when_finished: true` as requested.
3. Replace the two-child launcher in `demo_v6_2/phystwin_shen_launch.py` with
   one supervisor command, one log, and one process handle.
4. Update Demo validation, dry-run contracts, summaries, and lifecycle logic.
5. Keep port takeover local to Demo and extend it to CMA and train viewer ports.
6. Update downstream tests, pipeline documentation, and design documentation.

## Invalid states

- Missing wrapper/config/interpreter fails before camera launch.
- Disabled ASAP fails when the full pipeline is selected because the PhysTwin
  reader requires per-frame ASAP surface and interior trajectories.
- Invalid pipeline numbers or ports fail in Demo argument validation.
- A supervisor non-zero exit fails the Demo immediately while capture is live
  or after capture completes.
- Failure cleanup never reports success after killing the downstream group.

## Validation

- Focused Demo v6.2 downstream tests.
- External wrapper CLI/config unit tests or deterministic dry-run assertions.
- Base-environment dry run proving every external child uses `demo_2_max`.
- Current online-data fixture contract load.
- Repository smoke validation profile.
- Manual GPU/hardware execution remains separate and must not be faked.

## Progress

- [x] Requirements and ownership decisions recorded.
- [x] External wrapper overrides implemented.
- [x] Demo launcher and lifecycle implemented.
- [x] Tests and docs updated.
- [x] Deterministic validation passed.
- [x] Single-camera change committed and pushed.
- [x] Phystwin_shen online-branch change committed and pushed (`74dde0c`).

## Validation results

- Focused Demo v6.2 suite: `65 passed, 6 subtests passed`.
- Downstream lifecycle suite: `33 tests`, including the base-environment
  prefix assertion, real dual-port takeover, leader-exited PGID cleanup, and
  camera stop-race probes.
- External full-pipeline config tests: `3 passed`.
- Cross-repository dry run: wrapper, both viewers, Stage 1, and train all use
  `/home/xinjie/miniforge3/envs/demo_2_max/bin/python`; local values produce
  `segment_len=30`, ports `8765/8766`, `configs/real.yaml`, and
  `--stop_when_finished`.
- Physics invariant: `5e-5 * 4000 = 0.2 s = 1 / 5 FPS`.
- Repository smoke profile: `221 tests`, all CLI/guard checks passed.
- Scoped Ruff checks and `git diff --check` passed. The external
  `trainer_warp.py` retains unrelated pre-existing lint findings outside the
  touched online-loop block.
- Real camera/GPU execution was not run; hardware validation remains manual.
