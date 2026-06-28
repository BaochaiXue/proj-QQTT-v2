# Demo v5.1 Default Config YAML Plan

Goal: move the Demo v5.1 orchestration defaults from `realtime_data_process_sam3d.py` into `demo_v5_1/config/default.yaml` while preserving the current CLI defaults and runtime behavior.

## Scope

- Add `demo_v5_1/config/default.yaml` with the default values currently defined in `demo_v5_1/realtime_data_process_sam3d.py` lines 41-106.
- Load that YAML at module import and initialize the existing `DEFAULT_*`, layout, GPU-map, and checkpoint constants from it.
- Keep existing CLI names, resolver helpers, and command-building behavior unchanged.
- Require PyYAML in the Demo v5/v5.1 main and shape-prior environments, and
  load the default config directly with `yaml.safe_load`.
- Remove the standard-library fallback parser now that PyYAML is an explicit
  environment dependency.

## Checklist

- [x] Add a failing test that proves the default constants are backed by `default.yaml`.
- [x] Add `demo_v5_1/config/default.yaml`.
- [x] Update `demo_v5_1/realtime_data_process_sam3d.py` to load defaults from YAML.
- [x] Run focused tests and compile checks.
- [x] Add a failing test that rejects fallback default-config parser helpers.
- [x] Add PyYAML to every Demo v5/v5.1 pip requirements file that lacked it.
- [x] Remove the fallback parser and deleted helper module.
- [x] Install/verify PyYAML in the local conda environments.
- [x] Run focused config tests and smoke validation.

## Validation Results

- Red check: `conda run -n demo_2_max --no-capture-output python -m unittest tests.test_demo_v5_1_default_config` failed before implementation because `DEFAULT_CONFIG_PATH` did not exist.
- Green check: `conda run -n demo_2_max --no-capture-output python -m unittest tests.test_demo_v5_1_default_config` passed.
- `conda run -n demo_2_max --no-capture-output python demo_v5_1/realtime_data_process_sam3d.py --dry-run` passed and printed the default contract from the YAML-backed constants.
- `conda run -n demo_2_max --no-capture-output python -m py_compile demo_v5_1/realtime_data_process_sam3d.py tests/test_demo_v5_1_default_config.py` passed.
- Red check for fallback removal: `conda run -n demo_2_max --no-capture-output python -m unittest tests.test_demo_v5_1_default_config` failed before implementation because `_parse_default_config_scalar` still existed.
- Green check after fallback removal: `conda run -n demo_2_max --no-capture-output python -m unittest tests.test_demo_v5_1_default_config` passed.
- PyYAML install/verify: `base`, `demo3-max`, `demo_2_max`, `demo_3_1_max`, `demo_3_3_max`, and `phystwin-max` all import `yaml` at version `6.0.3`.
- `conda run -n demo_2_max --no-capture-output python demo_v5_1/realtime_data_process_sam3d.py --dry-run` passed after making PyYAML mandatory.
- Smoke validation: `conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile smoke` still fails at the unittest batch because the current workspace lacks many expected `tests.test_*` modules, such as `tests.test_record_preflight_policy_smoke` and `tests.test_record_data_align_smoke`.
