# 2026-05-10 Demo 2 GPU Utilization Sampling

## Goal

Add an optional GPU utilization sampler that can be enabled for both Demo 2.2 and Demo 2.1.5 profiles. Keep it off by default and record samples/summaries in the existing profile JSON and Markdown reports.

## Plan

1. Add shared runtime CLI flags for GPU sampling interval and backend.
2. Implement a lightweight background sampler using NVML when available, with `nvidia-smi` fallback.
3. Start the sampler with the demo runtime and stop it before profile writing.
4. Include raw samples plus full-run and after-warmup summaries in profile JSON.
5. Add Markdown summary rows for GPU utilization, memory, power, and clocks.
6. Expose simple `--gpu-sampling` / `--gpu-sampling-interval-s` flags in the Demo 2.2 and Demo 2.1.5 wrappers.
7. Add deterministic tests for argument translation, sampler summarization, and profile payload inclusion.

## Validation

- PASS: `conda run --no-capture-output -n demo_2_max python -m unittest tests.test_demo_v2_2_async_filtered_fused_pcd_smoke tests.test_demo_v2_1_5_realsense_depth_smoke`
- PASS: `conda run --no-capture-output -n demo_2_max python scripts/harness/check_all.py`

## Results

- Added optional GPU sampling flags to Demo 2.2 and Demo 2.1.5 wrappers.
- Added shared `GpuUtilizationSampler` with NVML-first and `nvidia-smi` fallback behavior.
- Profile JSON now records diagnostics, raw samples, full-run summary, and after-warmup summary under `gpu_sampling`.
- Profile Markdown now includes a warmup-excluded `GPU Sampling` section.
- Local probe confirmed `nvidia-smi` fallback works in WSL; `pynvml` is not installed in `demo_2_max`.
