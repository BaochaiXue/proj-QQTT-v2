# 2026-05-19 Demo 3.2 Same-GPU FFS EdgeTAM

## Goal

Change Demo 3.2 GPU ownership so FFS TensorRT depth and SAM3.1/HF EdgeTAM masks run on the same physical GPU, while LiteTracker remains isolated on the tracker GPU.

## Non-Goals

- no change to Demo 3.1 RealSense-depth default split
- no change to LiteTracker lazy query-init semantics
- no change to surface-snapped red control markers
- no change to FFS engine artifact or model parameters

## Files To Touch

- `qqtt/demo/demo31_runtime.py`
- `demo_v3_2/README.md`
- focused Demo 3.2 contract tests
- this exec plan

## Implementation Plan

1. Keep Demo 3.2 FFS on physical GPU0.
2. Move Demo 3.2 SAM3.1 / HF EdgeTAM device fields to physical GPU0 as well.
3. Keep LiteTracker child process on physical GPU1.
4. Update contract/dry-run wording so the GPU ownership is explicit.
5. Update tests to lock the new placement.

## Validation Plan

- `conda run -n demo_2_max --no-capture-output python -m unittest -v tests.test_demo31_dual_gpu_contract`
- `conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py`

## Validation Result

- PASS: `python -m py_compile qqtt/demo/demo31_runtime.py qqtt/demo/demo32_runtime.py qqtt/demo/services/profile_schema.py`
- PASS: focused Demo 3.2 placement/profile tests
- PASS: Demo 3.2 dry-run reports `ffs_gpu_physical=0`, `edgetam_gpu_physical=0`, `sam31_gpu_physical=0`, `litetracker_gpu_physical=1`
- PASS: `python scripts/harness/check_all.py`
