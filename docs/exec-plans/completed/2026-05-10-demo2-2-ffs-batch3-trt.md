# 2026-05-10 Demo 2.2 FFS Batch3 TensorRT

## Goal
Add an isolated Fast-FoundationStereo TensorRT batch=3 path for Demo 2.2 on
RTX 5090 Laptop without touching the existing batch=1 TensorRT artifact or
default runtime behavior.

## Plan
- Add a 5090-specific batch=3 build wrapper for `20-30-48`, `valid_iters=4`,
  `480x864`, `builderOptimizationLevel=5`.
- Store batch=3 engines under a dedicated result path ending in `_batch3`.
- Keep Demo 2.2 default on batch=1 unless `--ffs-trt-batch-size 3` is passed.
- When batch=3 is selected, default `--ffs-trt-model-dir` to the isolated
  batch=3 engine path unless the operator explicitly overrides it.
- In Demo 2.2 FFS cycles, call `runner.run_batch()` once for cam0/cam1/cam2
  instead of three sequential `run_pair()` calls.
- Record batch size and batch timing in contracts/profiles.

## Validation
- Deterministic contract and batch-dispatch tests.
- `scripts/harness/check_all.py`.
- Build the isolated batch=3 engine if the local TensorRT/FFS environment is
  available.
- Run a Demo 2.2 hardware profile using `--ffs-trt-batch-size 3`.

## Results
- Built isolated batch=3 engines under
  `result/ffs_trt_static_rounds_848x480_pad864_builderopt5_rtx5090_laptop_batch3/`.
- Static TensorRT batch size verified as `3`.
- Added Demo 2.2 `--ffs-trt-batch-size 3` option and isolated default model
  directory selection.
- Batch=3 hardware profile:
  `docs/generated/demo2_2_async_filter_batch3_ffs_20s_warmup_20s_formal_profile.md`.
- Build/profile report:
  `docs/generated/demo2_2_ffs_batch3_trt_build_and_profile.md`.
