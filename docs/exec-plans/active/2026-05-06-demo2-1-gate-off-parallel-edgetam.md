# Demo 2.1 Gate-Off Parallel EdgeTAM Benchmark

## Goal

Make Demo 2.1 default to `gpu_gate=off` and rerun the three-camera
controller-object EdgeTAM benchmark with true parallel workers.

## Scope

- Keep the Demo 2.1 quality contract unchanged:
  - live SAM3.1 first-frame init
  - HF EdgeTAMVideo streaming
  - FFS-derived depth for full quality runs
  - timestamp-nearest temporal grouping
  - object enhanced-PT and controller pt-filter
- Change GPU gate defaults to off for Demo 2.1 presets unless explicitly
  overridden on the CLI.
- Add a PyTorch CUDAGraph step boundary before compiled EdgeTAM model calls so
  gate-off parallel workers can be tested instead of failing on overwritten
  graph outputs.
- Update smoke tests and generated notes for the new default.

## Validation

- `python -m py_compile demo_v2_1/realtime_three_view_masked_fused_pcd.py`
- `conda run --no-capture-output -n demo_2_max python -m unittest -v tests.test_demo_v2_1_three_view_fused_pcd_smoke`
- `conda run --no-capture-output -n demo_2_max python scripts/harness/check_all.py`
- Live benchmark:
  - three cameras
  - controller-object
  - controller prompt `towel`
  - object prompt `stuffed animal`
  - `depth-source none`
  - `gpu_gate=off`
  - `fusion-target-fps=15`
