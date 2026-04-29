# 2026-04-28 FFS-SAM-RS Runtime Defaults

## Goal

Make the current harness-engineering default for visualization and realtime FFS work explicit:

- environment: `FFS-SAM-RS`
- checkpoint: `20-30-48`
- valid iterations: `4`
- primary runtime: two-stage ONNX/TensorRT
- TensorRT builder optimization level: `5`
- real QQTT input shape: `848x480`, padded to `864x480`

## Scope

- Update live viewer defaults to point at the validated `20-30-48 / iter4 / builderOpt5` two-stage TRT artifact.
- Update harness and workflow docs so new realtime/visualization work starts from `FFS-SAM-RS` and the level-5 TRT artifact.
- Update default PyTorch checkpoint/iteration values in experiment-only visualization CLIs where they still rerun PyTorch FFS.
- Do not change formal depth output contracts.
- Do not force confidence-logit experiments onto TRT until a TRT confidence/logits export exists.

## Validation

- Inspect the selected artifact metadata.
- Run deterministic quick checks after edits.

## Result

- Added shared FFS defaults for `FFS-SAM-RS`, checkpoint `20-30-48`, `valid_iters=4`, `max_disp=192`, and the level-5 two-stage TensorRT artifact.
- Updated realtime/viewer and visualization harness defaults and docs to use the shared policy.
- Kept PyTorch as the explicit path for confidence-logit experiments because the current TensorRT artifact does not export confidence logits.
- Installed `atomics==1.0.3` into `FFS-SAM-RS` so the QQTT RealSense runtime imports cleanly.
- Validation passed with `/home/zhangxinjie/miniconda3/envs/FFS-SAM-RS/bin/python scripts/harness/check_all.py`.
