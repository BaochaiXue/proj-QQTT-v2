# Demo 2 GPU Sampling Profile Support

Date: 2026-05-10

## Scope

Demo 2.2 and Demo 2.1.5 now expose optional GPU utilization sampling for live profiles. The sampler is off by default and does not change the runtime pipeline unless explicitly enabled.

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

## Local Probe

In the current `demo_2_max` environment, `pynvml` is not installed, so `auto` falls back to `nvidia-smi`. The WSL `nvidia-smi` path works:

```text
/usr/lib/wsl/lib/nvidia-smi
```

The fallback sampled GPU utilization, memory, power, SM clock, memory clock, and temperature successfully in a short standalone probe. Because `nvidia-smi` starts a subprocess for each sample, use `--gpu-sampling-interval-s 0.5` or slower unless NVML is installed.
