# 2026-05-10 Demo 2.2 Default FFS Batch3

## Goal

Make the Demo 2.2 default wrapper use the isolated Fast-FoundationStereo
TensorRT batch=3 engine path.

## Scope

- Keep the global Demo 2.1 parser default at batch=1.
- Keep explicit `--ffs-trt-batch-size 1` as the operator rollback path.
- For the Demo 2.2 async-filter preset, default `--ffs-trt-batch-size` to `3`
  and default `--ffs-trt-model-dir` to the isolated batch=3 engine path.
- Update deterministic tests and generated docs.

## Validation

- PASS: Demo 2.2 dry-run contract reports `trt_batch_size=3` and the isolated
  batch=3 model path.
- PASS: Demo 2.2 explicit `--ffs-trt-batch-size 1` remains batch=1.
- PASS: targeted Demo 2.2 / Demo 2.1 smoke tests.
- PASS: `scripts/harness/check_all.py`.
