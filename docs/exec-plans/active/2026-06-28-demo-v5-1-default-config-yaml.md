# Demo v5.1 Default Config YAML Plan

Goal: move the Demo v5.1 orchestration defaults from `realtime_data_process_sam3d.py` into `demo_v5_1/config/default.yaml` while preserving the current CLI defaults and runtime behavior.

## Scope

- Add `demo_v5_1/config/default.yaml` with the default values currently defined in `demo_v5_1/realtime_data_process_sam3d.py` lines 41-106.
- Load that YAML at module import and initialize the existing `DEFAULT_*`, layout, GPU-map, and checkpoint constants from it.
- Keep resolver helpers and command-building behavior unchanged.
- Later user direction renamed the camera-runtime model routing keys and CLI
  options to `perception_device`, `tracker_device`, and `inference_dtype`,
  without retaining old aliases.
- Require PyYAML in the Demo v5/v5.1 main and shape-prior environments, and
  load the default config directly with `yaml.safe_load`.
- Remove the standard-library fallback parser now that PyYAML is an explicit
  environment dependency.
- Keep the loaded YAML as the current grouped structure instead of flattening
  it into backward-compatible root-level keys.
- Read runtime constants through a single grouped `_cfg(section, key)` helper.
- Remove the SAM 3.1 checkpoint env helper that silently probes for the
  vendored file; use an existing environment variable or the YAML path only.

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
- [x] Add a failing test that rejects grouped YAML flattening.
- [x] Remove grouped-to-flat compatibility from the default config loader.
- [x] Re-run focused grouped-config tests, compile checks, and dry-run.
- [x] Add a failing test for YAML-provided SAM 3.1 checkpoint env propagation.
- [x] Remove the SAM 3.1 checkpoint file-probe env helper.
- [x] Add a failing test that rejects the old camera-runtime model routing
      names and accepts `--perception-device`, `--tracker-device`, and
      `--inference-dtype`.
- [x] Rename the Demo v5.1 YAML keys, constants, CLI options, dry-run manifest
      fields, and subprocess argument wiring.
- [x] Update Demo v5.1 README camera-runtime option docs.

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
- Red check for grouped-only config: `conda run -n demo_2_max --no-capture-output python -m unittest tests.test_demo_v5_1_default_config` failed before implementation because `_flatten_default_config` still existed and `load_default_config()` returned flat keys.
- Green check after removing flattening: `conda run -n demo_2_max --no-capture-output python -m unittest tests.test_demo_v5_1_default_config` passed.
- Compile check after removing flattening: `conda run -n demo_2_max --no-capture-output python -m py_compile demo_v5_1/realtime_data_process_sam3d.py tests/test_demo_v5_1_default_config.py` passed.
- Dry-run after removing flattening: `conda run -n demo_2_max --no-capture-output python demo_v5_1/realtime_data_process_sam3d.py --dry-run` passed.
- Smoke validation after removing flattening still fails in the unittest batch
  because the current workspace lacks expected modules including
  `tests.test_record_preflight_policy_smoke`,
  `tests.test_record_data_realtime_align_smoke`, and
  `tests.test_record_data_align_smoke`.
- Red check for checkpoint env cleanup:
  `conda run -n demo_2_max --no-capture-output python -m unittest tests.test_demo_v5_1_default_config.DemoV51DefaultConfigTest.test_camera_env_uses_yaml_sam31_checkpoint_without_file_probe tests.test_demo_v5_1_default_config.DemoV51DefaultConfigTest.test_default_config_access_uses_single_cfg_helper`
  failed before implementation because `_apply_default_sam31_checkpoint_env`
  still existed and skipped missing YAML paths after `is_file()`.
- Green check after checkpoint env cleanup:
  `conda run -n demo_2_max --no-capture-output python -m unittest tests.test_demo_v5_1_default_config`
  passed.
- Compile check after checkpoint env cleanup:
  `conda run -n demo_2_max --no-capture-output python -m py_compile demo_v5_1/realtime_data_process_sam3d.py tests/test_demo_v5_1_default_config.py`
  passed.
- Dry-run after checkpoint env cleanup:
  `conda run -n demo_2_max --no-capture-output python demo_v5_1/realtime_data_process_sam3d.py --dry-run`
  passed.
- Smoke validation after checkpoint env cleanup still fails in the unittest
  batch for the same missing-module reason, including
  `tests.test_record_preflight_policy_smoke`,
  `tests.test_record_data_realtime_align_smoke`, and
  `tests.test_record_data_align_smoke`.
- Red check for camera-runtime model routing rename:
  `conda run -n demo_2_max --no-capture-output python -m unittest tests.test_demo_v5_1_default_config`
  failed before implementation because the old YAML keys, constants, and CLI
  options still existed while the new names were absent.
- Green check after the rename:
  `conda run -n demo_2_max --no-capture-output python -m unittest tests.test_demo_v5_1_default_config`
  passed.
- Related regression:
  `conda run -n demo_2_max --no-capture-output python -m unittest tests.test_demo_v5_legacy_key_cleanup tests.test_demo_v5_1_default_config tests.test_demo_v5_1_aggregate_invariants tests.test_demo_v5_1_split_payload tests.test_demo_v5_1_tools_io tests.test_validation_smoke_manifest`
  passed.
- Compile check:
  `conda run -n demo_2_max --no-capture-output python -m py_compile demo_v5_1/main.py tests/test_demo_v5_1_default_config.py`
  passed.
- Dry-run check:
  `conda run -n demo_2_max --no-capture-output python demo_v5_1/main.py --dry-run --no-shape-prior-warmup --point-viewer-mode disabled --perception-device cuda:0 --tracker-device cuda:0 --inference-dtype float32`
  passed and emitted `perception_device`, `tracker_device`, and
  `inference_dtype`.
- Fresh smoke validation:
  `conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile smoke`
  passed.
