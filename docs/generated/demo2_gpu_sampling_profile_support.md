# Demo 2 GPU Sampling Profile Support

Date: 2026-05-10

## Scope

Demo 2.2 and Demo 2.1.5 expose optional GPU utilization sampling for live profiles. The sampler is off by default and does not change the runtime pipeline unless explicitly enabled.

As of 2026-05-10, Demo 2 GPU sampling is NVML-only. The old `nvidia-smi` subprocess fallback is intentionally removed because subprocess sampling adds avoidable overhead and jitter.

## Commands

Demo 2.2:

```bash
conda run --no-capture-output -n demo_2_max \
  python demo_v2_2/realtime_three_view_async_filtered_fused_pcd.py \
  --duration-s 60 \
  --warmup-s 20 \
  --gpu-sampling \
  --gpu-sampling-interval-s 0.5 \
  --profile-json-output docs/generated/demo2_2_gpu_sampled_profile.json
```

Demo 2.1.5:

```bash
conda run --no-capture-output -n demo_2_max \
  python demo_v2_1_5/realtime_three_view_async_filtered_fused_pcd.py \
  --duration-s 60 \
  --warmup-s 20 \
  --gpu-sampling \
  --gpu-sampling-interval-s 0.5 \
  --profile-json-output docs/generated/demo2_1_5_gpu_sampled_profile.json
```

## Recorded Fields

Profile JSON now includes `gpu_sampling` with:

- sampler diagnostics: enabled, requested backend, backend used, interval, device index, errors
- raw samples with relative `sample_s`
- full-run summary
- after-warmup summary

Each sample can include:

- `gpu_util_pct`
- `memory_util_pct`
- `memory_used_mb`
- `memory_total_mb`
- `power_w`
- `power_limit_w`
- `sm_clock_mhz`
- `mem_clock_mhz`
- `temperature_c`

The Markdown profile report adds a `GPU Sampling` section with warmup-excluded median / p90 / p95 / max values.

## NVML Requirement

The only accepted backend is:

```text
--gpu-sampling-backend nvml
```

The `demo_2_max` environment must provide the NVML Python binding (`pynvml`, distributed by `nvidia-ml-py`). If NVML cannot initialize, the profile records a GPU sampling error and no fallback sampler is started.

Local validation on 2026-05-10:

```text
conda env: demo_2_max
installed package: nvidia-ml-py==13.595.45
NVML probe: PASS
GPU: NVIDIA GeForce RTX 5090 Laptop GPU
sampler smoke: PASS, backend_used=nvml, sample_count=3 over a 1.2s probe
```
