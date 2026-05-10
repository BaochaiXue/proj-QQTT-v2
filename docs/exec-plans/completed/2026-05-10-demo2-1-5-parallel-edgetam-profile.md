# 2026-05-10 Demo 2.1.5 Parallel EdgeTAM Profile

## Goal

Add a clear Demo 2.1.5 public option for running cam0/cam1/cam2 EdgeTAM tracking in parallel, then profile GPU utilization with NVML.

## Plan

1. Add `--parallel-edgetam` to `demo_v2_1_5/realtime_three_view_async_filtered_fused_pcd.py`.
2. Map it to the existing `demo2.1.5-staged-parallel-5fps` runtime preset, which uses native RealSense depth plus parallel per-camera EdgeTAM.
3. Keep `--experimental-staged-parallel` as a backwards-compatible alias.
4. Add deterministic tests for the new public option and its conflict behavior.
5. Run focused tests and `scripts/harness/check_all.py`.
6. Run a hardware NVML profile and write generated report artifacts.

## Validation

Completed.

- Focused unit tests passed:
  - `conda run --no-capture-output -n demo_2_max python -m unittest tests.test_demo_v2_1_5_realsense_depth_smoke tests.test_demo_v2_2_async_filtered_fused_pcd_smoke tests.test_demo_v2_1_three_view_fused_pcd_smoke`
- Full quick deterministic harness passed:
  - `conda run --no-capture-output -n demo_2_max python scripts/harness/check_all.py`
- Hardware NVML profile command:
  - `conda run --no-capture-output -n demo_2_max python demo_v2_1_5/realtime_three_view_async_filtered_fused_pcd.py --parallel-edgetam --duration-s 120 --warmup-s 70 --gpu-sampling --gpu-sampling-interval-s 0.1 --profile-json-output docs/generated/demo2_1_5_parallel_edgetam_gpu_nvml_profile.json`

## Result

- `--parallel-edgetam` now maps Demo 2.1.5 to `demo2.1.5-staged-parallel-5fps`.
- The old `--experimental-staged-parallel` option remains available.
- The staged parallel path uses EdgeTAM `vision-default` compile mode. The more aggressive `vision-reduce-overhead` CUDA graph path crashed in the thread-pool parallel run with a PyTorch Inductor CUDA graph TLS assertion.
- The hardware profile completed successfully and wrote:
  - `docs/generated/demo2_1_5_parallel_edgetam_gpu_nvml_profile.md`
  - `docs/generated/demo2_1_5_parallel_edgetam_gpu_nvml_profile.json`
- Parallel EdgeTAM is currently slower than the previous Demo 2.1.5 single-owner profile:
  - parallel EdgeTAM render FPS: `4.72`
  - previous single-owner render FPS: `7.29`
  - parallel EdgeTAM GPU util median/p95/max: `21 / 23 / 27 %`
  - previous single-owner GPU util median/p95/max: `30 / 37 / 39 %`

## Conclusion

The new option is useful as a reproducible profiling path, but it should not replace the default Demo 2.1.5 path. On this RTX 5090 Laptop setup, thread-level three-camera EdgeTAM parallelism loses the faster reduce-overhead compile mode and increases per-camera model latency.
