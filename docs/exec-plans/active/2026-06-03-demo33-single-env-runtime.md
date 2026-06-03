# Demo 3.3 Single-Environment Runtime

## Goal

Run Demo 3.3 live runtime and its FuturePhysTwin/SAM3D shape-prior route from
one conda environment, `demo_3_3_max`, instead of defaulting shape-prior stages
to a separate `phystwin-max` launcher.

## Scope

- Change the Demo 3.3 shape-prior default Python launcher to the current Python
  interpreter.
- Keep explicit `--futurephystwin-python` override support for environment A/B
  tests.
- Update Demo 3.3 docs and contract tests.

## Non-Goals

- Do not reduce SAM3D, align, sample, tracker, or render quality settings.
- Do not merge all shape-prior stages into one Python process in this change.
  This removes the conda environment switch, but subprocess stage startup/model
  import costs remain.

## Validation

- Dry-run Demo 3.3 under `demo_3_3_max`.
- Focused Demo 3.3 contract tests.
- QQTT quick harness if the unified environment has the expected test stack.

## Result

- Changed Demo 3.3 shape-prior default launcher to `sys.executable`, so a
  live process started with `conda run -n demo_3_3_max python ...` launches the
  FuturePhysTwin route with
  `/home/xinjie/miniforge3/envs/demo_3_3_max/bin/python`.
- Preserved the explicit `--futurephystwin-python` override.
- Updated Demo 3.3 docs and the dry-run contract expectation.

## Verification Evidence

- Dry-run under `demo_3_3_max` reported:
  - `futurephystwin_python =
    /home/xinjie/miniforge3/envs/demo_3_3_max/bin/python`
  - `shape_prior_start_policy = after-teardown`
  - `shape_prior_skip_route_visualizations = true`
  - `overlay_display_scope = union`
- Focused unit tests passed in `demo_3_3_max` via `unittest`: 9 tests OK.
- Focused pytest checks passed in `demo_2_max`: 9 passed.
- `conda run -n demo_2_max --no-capture-output python
  scripts/harness/check_all.py` passed: 389 tests OK.
- Fresh live Demo 3.3 run under `demo_3_3_max` completed and launched a
  detached shape-prior completion worker:
  `docs/generated/demo33_single_env_fresh_demo_3_3_max_20260603_45s_profile_shared_runtime.json`.
- The detached completion worker reached `shape_prior_status = ready` with all
  five FuturePhysTwin route stages using
  `/home/xinjie/miniforge3/envs/demo_3_3_max/bin/python`:
  `image_upscale`, `segment_util_image`, `shape_prior_sam3d`, `align`, and
  `data_process_sample`.
