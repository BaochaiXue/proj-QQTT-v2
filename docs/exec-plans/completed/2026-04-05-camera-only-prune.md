# 2026-04-05 Camera-Only Prune

## Problem Statement

`proj-QQTT-v2` 目前仍然包含大量下游数字孪生、逆物理、Gaussian Splatting、评估和交互代码。新的仓库目标已经收缩为相机采集与对齐，因此需要对仓库做破坏性收缩，删除所有越界内容，并建立一个轻量 harness-engineering 工作流来约束未来修改。

## New Repo Charter

> This repository handles 3-camera RealSense preview, calibration, synchronized RGB-D recording, and aligned case generation up to `data_process/record_data_align.py`.

## Scope Table

| Area | Status | Notes |
| --- | --- | --- |
| RealSense camera preview/debug | In scope | `cameras_viewer.py` |
| Multi-camera calibration | In scope | `cameras_calibrate.py`, `qqtt/env/camera/**` |
| Multi-camera RGB-D recording | In scope | `record_data.py`, `qqtt/env/camera/**` |
| Raw recording alignment/trimming | In scope | `data_process/record_data_align.py` |
| Segmentation / dense tracking / shape prior | Out of scope | Remove |
| Point-cloud / final_data packaging beyond alignment | Out of scope | Remove |
| Inverse physics / Warp / training / inference | Out of scope | Remove |
| Gaussian Splatting / rendering | Out of scope | Remove |
| Downstream evaluation / visualization / teleop | Out of scope | Remove |

## Files And Folders To Keep

- `qqtt/env/camera/**`
- `qqtt/env/__init__.py`
- simplified `qqtt/__init__.py`
- `cameras_viewer.py`
- `cameras_calibrate.py`
- `record_data.py`
- `data_process/record_data_align.py`
- `env_install/` with camera-only installer
- `README.md`
- harness docs under `docs/`
- scope guard under `scripts/harness/`
- smoke test and fixture under `tests/`

## Files And Folders To Delete

Directories targeted for deletion:

- `gaussian_splatting/`
- `configs/`
- `taichi_simulator_test/`
- `qqtt/data/`
- `qqtt/engine/`
- `qqtt/model/`
- `qqtt/utils/` if unused after import audit
- `data_process/models/`
- `data_process/utils/`
- `test/`

Within `data_process/`, delete everything except `record_data_align.py` unless import audit proves otherwise.

Top-level files targeted for deletion:

- `process_data.py`
- `optimize_cma.py`
- `train_warp.py`
- `inference_warp.py`
- `inference_optimization_warp.py`
- `final_pipeline.sh`
- all `evaluate_*`
- all `export_*`
- all `gs_*`
- all `visualize_*`
- `interactive_*`
- `physics_dynamics_module.py`
- `prepare_results.py`
- `outdomain_exp.py`
- `script*.py`
- `combine_video.py`
- `optical_frames.py`
- `data_config.csv`
- `env_install/download_pretrained_models.sh`

## Code Changes Required

1. Simplify `qqtt/__init__.py` to export only `CameraSystem`.
2. Add coherent argparse CLIs to `cameras_calibrate.py` and `record_data.py`.
3. Refactor `data_process/record_data_align.py`:
   - default `--base_path=./data_collect`
   - default `--output_path=./data`
   - infer camera count from metadata
   - default fps from metadata unless overridden
   - validate inputs clearly
   - make mp4 generation optional
   - preserve `calibrate.pkl` and `metadata.json`
4. Slim `env_install/env_install.sh` to camera-only dependencies.
5. Rewrite README and add harness docs.
6. Add deterministic scope guard and smoke test.

## Evaluator Plan

Deterministic checks:

- `python cameras_viewer.py --help`
- `python cameras_calibrate.py --help`
- `python record_data.py --help`
- `python data_process/record_data_align.py --help`
- `python -m scripts.harness.check_scope`
- smoke test for `record_data_align.py`

CI should run only deterministic checks. Hardware is manual validation.

## Manual Hardware Validation Plan

1. Connect 3 RealSense cameras.
2. Verify viewer launches and shows all cameras.
3. Run calibration and confirm `calibrate.pkl` is written.
4. Run recording and confirm raw case folder contains color/depth/metadata.
5. Run alignment and confirm output case under `data/`.

## Progress Log

- 2026-04-05: started repo audit and import audit for kept camera/alignment chain.
- 2026-04-05: confirmed kept import chain only depends on `qqtt/env/camera/**`, stdlib, and camera runtime dependencies.
- 2026-04-05: deleted downstream physics, rendering, evaluation, teleop, and tracking code.
- 2026-04-05: simplified `qqtt` exports and refactored camera entrypoints into coherent CLIs.
- 2026-04-05: rewrote `record_data_align.py` as a camera-only alignment tool with new defaults and optional mp4 generation.
- 2026-04-05: rewrote README and added harness docs, scope guard, smoke test, and lightweight CI.
- 2026-04-05: deterministic checks passed:
  - `python cameras_viewer.py --help`
  - `python cameras_calibrate.py --help`
  - `python record_data.py --help`
  - `python data_process/record_data_align.py --help`
  - `python -m scripts.harness.check_scope`
  - `python -m unittest -v tests.test_record_data_align_smoke`

## Decisions / Tradeoffs

- Git history is the backup; no archival folder will be created.
- Hardware validation will remain manual; CI will only assert deterministic behavior.
- Local output folders like `data_collect/` and `data/` remain workflow outputs, but not versioned assets.
- CLI help was made import-light so deterministic checks can run without installing RealSense runtime packages in CI.
- `record_data_align.py` now defaults to skipping mp4 generation so CI and smoke tests do not depend on `ffmpeg`.

## Completion Summary

Completed on 2026-04-05.

The repo now ends at camera preview, calibration, recording, and aligned-case generation. Downstream digital-twin, physics, Gaussian Splatting, and evaluation code has been removed. Deterministic guardrails and smoke tests are in place to keep future changes inside the new camera-only charter.
