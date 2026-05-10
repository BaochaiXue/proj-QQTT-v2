# 2026-05-10 Demo 2.1.5 Parallel EdgeTAM Eager Probe

## Goal

Test whether the Demo 2.1.5 parallel EdgeTAM worker path can run without
`torch.compile`.

## Plan

1. Add an explicit `compile-mode=none` eager path for EdgeTAM.
2. Add a public Demo 2.1.5 `--no-compile-edgetam` switch that maps to
   `--compile-mode none`.
3. Keep the compiled path unchanged.
4. Run focused deterministic tests and a hardware profile with
   `--parallel-edgetam --no-compile-edgetam`.

## Validation

- `conda run --no-capture-output -n demo_2_max python -m unittest tests.test_demo_v2_1_5_realsense_depth_smoke tests.test_demo_v2_1_three_view_fused_pcd_smoke`
- Demo 2.1.5 eager parallel dry-run.
- Demo 2.1.5 eager parallel hardware profile.

## Result

- Focused tests: PASS.
- Dry-run: PASS; `compile_mode=none`, `gpu_pipeline=separate-workers`,
  `edgetam_model_topology=replicated`, `gpu_gate=off`.
- Hardware profile: completed without CUDAGraph overwrite or fatal error.
- Profile: `docs/generated/demo2_1_5_parallel_edgetam_eager_probe_profile.json`.
- Warmup-excluded EdgeTAM model median:
  - cam0 `184.41 ms`
  - cam1 `188.11 ms`
  - cam2 `189.92 ms`
- Warmup-excluded GPU util median/p95/max: `24.00 / 28.15 / 31.00 %`.
- Warmup-excluded raw fusion/filter/fusion FPS: about `0.65 FPS` because the
  separate depth/mask workers produced many missing-mask fusion timeouts.
