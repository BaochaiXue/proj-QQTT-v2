# Demo 3.2 FFS LiteTracker

## Goal

Create Demo 3.2 as a copied/organized Demo 3.1 lineage that uses FFS TensorRT
builderOptimizationLevel=5 batch=3 depth asynchronously before EdgeTAM and a
LiteTracker serial backend for tracking.

## Requirements

- Add a Demo 3.2 entrypoint and docs rather than mutating Demo 3.1 semantics.
- Reuse Demo 3.1 dual-GPU point-tracker process/lift/render behavior.
- Reuse the Demo 2.3 FFS batch=3 opt=5 depth contract.
- Default tracker backend to `litetracker` and serial execution.
- Expose a dry-run contract that makes the pipeline order explicit:
  capture -> FFS -> EdgeTAM -> tracker -> render/diagnostics.

## Plan

1. Inspect Demo 3.1 bridge defaults and Demo 2.3 FFS batch=3 preset wiring.
2. Add Demo 3.2 mode support without breaking Demo 3.1 defaults.
3. Add a `demo_v3_2/` entrypoint and README.
4. Add contract tests for FFS batch=3 opt=5 and LiteTracker serial defaults.
5. Run targeted deterministic tests.

## Results

- Added `demo_v3_2/` as the Demo 3.2 entrypoint/docs layer copied from the
  Demo 3.1 dual-4090 overlay lineage.
- Added the `demo3.2-ffs-litetracker` preset to the shared Demo 3.1 runtime
  bridge while preserving Demo 3.1 RealSense-depth defaults.
- Demo 3.2 dry-run now declares FFS TensorRT
  `builderOptimizationLevel=5`, `trt_batch_size=3`, the Demo 2.3
  `dual-gpu-split` shared runtime path, and LiteTracker serial tracking.
- Runtime commands and profile contract use `demo_3_1_max`; this environment
  owns the current point-tracker/CoTracker/LiteTracker dependency stack.
- Updated profile/schema docs and harness check lists so Demo 3.2 appears in
  deterministic validation.
- Validation passed:
  - `/home/xinjie/miniforge3/bin/conda run -n demo_3_1_max --no-capture-output python -m py_compile qqtt/demo/demo31_runtime.py qqtt/demo/services/profile_schema.py demo_v3_2/realtime_three_view_litetracker_ffs_dual4090.py tests/test_demo31_dual_gpu_contract.py tests/test_check_all_smoke.py`
  - `/home/xinjie/miniforge3/bin/conda run -n demo_3_1_max --no-capture-output python -m unittest -v tests.test_demo31_dual_gpu_contract tests.test_check_all_smoke`
  - `/home/xinjie/miniforge3/bin/conda run -n demo_3_1_max --no-capture-output python demo_v3_2/realtime_three_view_litetracker_ffs_dual4090.py --dry-run --camera-ids 0,1,2 --mask-gpu 0 --cotracker-gpu 1 --require-two-cuda --calibrate-path calibrate.pkl`
  - `/home/xinjie/miniforge3/bin/conda run -n demo_3_1_max --no-capture-output python scripts/harness/check_all.py`
