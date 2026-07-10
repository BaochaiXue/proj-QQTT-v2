# Demo v6.2 stage-specific Phystwin window configuration

## Requirement

Problem:
Demo v6.2's local Phystwin configuration places `batch_size`, `segment_len`,
and `segment_stride` under `stage1` and `train`, matching Phystwin's native
common-then-stage merge behavior. The Demo-to-Phystwin CLI bridge still
requires those keys under `common` and exposes no stage-specific override
flags, so validation fails before the camera starts.

Required final behavior:

- `common` may provide shared defaults for the three window parameters.
- `stage1`, `stage2`, and `train` may override each parameter independently.
- Every enabled stage must resolve all three parameters from its own section
  or from `common`; missing or non-positive values fail before camera startup.
- Demo passes every configured value explicitly to the external full-pipeline
  wrapper, whose argparse contract accepts and applies the stage overrides.
- Existing direct Phystwin YAML behavior and unrelated Demo products remain
  unchanged.

Inputs:

- `demo_v6_2/config/default.yaml::phystwin_shen`.
- `/home/xinjie/Phystwin_shen/configs/online_full_pipeline.yaml`.
- Explicit Demo-generated `--common_*`, `--stage1_*`, `--stage2_*`, and
  `--train_*` wrapper arguments.

Outputs:

- One validated external supervisor command with the intended effective
  window parameters for each enabled stage.

Invalid cases:

- An enabled stage lacks any effective window parameter.
- A configured window parameter is not a positive integer.
- Either repository exposes a YAML runtime leaf without a corresponding CLI
  override.

Constraints:

- Preserve all unrelated uncommitted changes in the Demo repository.
- Keep external code and configuration in the external checkout.
- Do not add compatibility parsing or silently fall back after validation.

Unknowns:

- None. Disabled Stage 2 may omit stage-specific values; if enabled later, it
  must define them or inherit explicit values from `common`.

## Plan

- [x] Extend the external wrapper's supported override schema and argparse
  flags for all three computational stages.
- [x] Test CLI application and common-then-stage precedence in the external
  wrapper.
- [x] Split required and optional overrides in Demo v6.2, forwarding configured
  common/stage window values without requiring duplicates.
- [x] Validate effective values for each enabled stage before launch.
- [x] Update focused Demo tests and operator-facing configuration docs.
- [x] Run external tests/parser dry-run, Demo focused tests, and repository
  smoke validation.

## Validation

- `conda run -n demo_2_max --no-capture-output python \
  tests/test_online_full_pipeline_config.py` in `/home/xinjie/Phystwin_shen`.
- External wrapper dry-run with explicit stage-specific window overrides.
- `conda run -n demo_2_max --no-capture-output python -m pytest \
  tests/test_demo_v6_2_downstream.py -q`.
- `conda run -n demo_2_max --no-capture-output python \
  scripts/harness/validation/run.py --profile smoke`.

Results:

- External wrapper config suite: 4 tests passed.
- External parser/child-command dry-run: Stage 1 `2/10/10`, Stage 2
  `3/20/20`, and Train `5/30/30` reached the correct child commands.
- Demo dry-run: exit 0 with Stage 1 and Train stage-specific flags and no
  absent common/stage2 window flags.
- Demo downstream suite: 48 tests and 17 subtests passed.
- Repository smoke: 243 tests passed; all guards and help probes passed.
- Exact proof commands and outcomes are retained in
  `docs/generated/2026-07-10-phystwin-stage-window-cli-proof.md`.
