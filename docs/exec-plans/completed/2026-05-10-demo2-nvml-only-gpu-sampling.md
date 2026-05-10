# 2026-05-10 Demo 2 NVML-Only GPU Sampling

## Goal

Make Demo 2 GPU sampling NVML-only. Do not use `nvidia-smi` subprocess fallback for Demo 2.2 or Demo 2.1.5 profiles.

## Plan

1. Restrict GPU sampling backend choices to `nvml` and default to `nvml`.
2. Remove the `nvidia-smi` sampler implementation and fallback path.
3. Update Demo 2.2 / Demo 2.1.5 smoke tests so `nvidia-smi` is rejected and NVML is the explicit default.
4. Update generated documentation to state that NVML Python bindings are required for GPU sampling.
5. Verify focused tests and `scripts/harness/check_all.py`.

## Validation

- PASS: `conda run --no-capture-output -n demo_2_max python -m unittest tests.test_demo_v2_2_async_filtered_fused_pcd_smoke tests.test_demo_v2_1_5_realsense_depth_smoke`
- PASS: Demo 2.2 dry-run with `--gpu-sampling` records `gpu_sampling.backend=nvml`.
- PASS: Demo 2.1.5 dry-run with `--gpu-sampling` records `gpu_sampling.backend=nvml`.
- PASS: direct `GpuUtilizationSampler` NVML smoke in `demo_2_max` collected samples from `NVIDIA GeForce RTX 5090 Laptop GPU`.
- PASS: `conda run --no-capture-output -n demo_2_max python scripts/harness/check_all.py`

## Results

- Restricted GPU sampling backend choices to `("nvml",)`.
- Removed the `nvidia-smi` subprocess sampler implementation and fallback path.
- Installed `nvidia-ml-py==13.595.45` into `demo_2_max` and documented it in `docs/envs.md`.
- Updated tests so `--gpu-sampling-backend nvidia-smi` is rejected by both Demo 2.2 and Demo 2.1.5 wrappers.
