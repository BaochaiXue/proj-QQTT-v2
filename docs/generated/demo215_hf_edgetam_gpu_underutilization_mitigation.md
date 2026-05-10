# Demo 2.1.5 HF EdgeTAM GPU Underutilization Mitigation

- target machine: `WSL Ubuntu RTX 5090 Laptop`
- defaults changed: `False`
- principle: Optimize p50/p90 latency and end-to-end p90; GPU utilization is diagnostic, not the primary KPI.

## Implemented Flags

- `profile_edgetam_stages`: `True`
- `profile_nsys_markers`: `True`
- `profile_sync`: `True`
- `mask_postprocess_cuda_inline`: `True`
- `dtype_float32_ablation`: `True`
- `compile_submodule_modes`: `True`
- `live_fast_native_preset`: `True`
- `live_quality_ffs_preset`: `True`
- `mask_only_debug_preset`: `True`

## Benchmark Matrix

| name | depth | compile | dtype | command |
| --- | --- | --- | --- | --- |
| `mask_only_none_bfloat16` | `none` | `none` | `bfloat16` | `conda run --no-capture-output -n demo_2_max python demo_v2_1_5/realtime_three_view_async_filtered_fused_pcd.py --mask-only-debug --compile-mode none --dtype bfloat16 --mask-postprocess cuda-inline --duration-s 90 --warmup-s 45 --render-mode none --gpu-sampling --gpu-sampling-interval-s 0.2 --profile-cuda-events --profile-edgetam-stages` |
| `mask_only_none_float16` | `none` | `none` | `float16` | `conda run --no-capture-output -n demo_2_max python demo_v2_1_5/realtime_three_view_async_filtered_fused_pcd.py --mask-only-debug --compile-mode none --dtype float16 --mask-postprocess cuda-inline --duration-s 90 --warmup-s 45 --render-mode none --gpu-sampling --gpu-sampling-interval-s 0.2 --profile-cuda-events --profile-edgetam-stages` |
| `mask_only_none_float32` | `none` | `none` | `float32` | `conda run --no-capture-output -n demo_2_max python demo_v2_1_5/realtime_three_view_async_filtered_fused_pcd.py --mask-only-debug --compile-mode none --dtype float32 --mask-postprocess cuda-inline --duration-s 90 --warmup-s 45 --render-mode none --gpu-sampling --gpu-sampling-interval-s 0.2 --profile-cuda-events --profile-edgetam-stages` |
| `mask_only_vision-reduce-overhead_bfloat16` | `none` | `vision-reduce-overhead` | `bfloat16` | `conda run --no-capture-output -n demo_2_max python demo_v2_1_5/realtime_three_view_async_filtered_fused_pcd.py --mask-only-debug --compile-mode vision-reduce-overhead --dtype bfloat16 --mask-postprocess cuda-inline --duration-s 90 --warmup-s 45 --render-mode none --gpu-sampling --gpu-sampling-interval-s 0.2 --profile-cuda-events --profile-edgetam-stages` |
| `mask_only_vision-reduce-overhead_float16` | `none` | `vision-reduce-overhead` | `float16` | `conda run --no-capture-output -n demo_2_max python demo_v2_1_5/realtime_three_view_async_filtered_fused_pcd.py --mask-only-debug --compile-mode vision-reduce-overhead --dtype float16 --mask-postprocess cuda-inline --duration-s 90 --warmup-s 45 --render-mode none --gpu-sampling --gpu-sampling-interval-s 0.2 --profile-cuda-events --profile-edgetam-stages` |
| `mask_only_vision-reduce-overhead_float32` | `none` | `vision-reduce-overhead` | `float32` | `conda run --no-capture-output -n demo_2_max python demo_v2_1_5/realtime_three_view_async_filtered_fused_pcd.py --mask-only-debug --compile-mode vision-reduce-overhead --dtype float32 --mask-postprocess cuda-inline --duration-s 90 --warmup-s 45 --render-mode none --gpu-sampling --gpu-sampling-interval-s 0.2 --profile-cuda-events --profile-edgetam-stages` |
| `mask_only_components-reduce-overhead_bfloat16` | `none` | `components-reduce-overhead` | `bfloat16` | `conda run --no-capture-output -n demo_2_max python demo_v2_1_5/realtime_three_view_async_filtered_fused_pcd.py --mask-only-debug --compile-mode components-reduce-overhead --dtype bfloat16 --mask-postprocess cuda-inline --duration-s 90 --warmup-s 45 --render-mode none --gpu-sampling --gpu-sampling-interval-s 0.2 --profile-cuda-events --profile-edgetam-stages` |
| `mask_only_components-reduce-overhead_float16` | `none` | `components-reduce-overhead` | `float16` | `conda run --no-capture-output -n demo_2_max python demo_v2_1_5/realtime_three_view_async_filtered_fused_pcd.py --mask-only-debug --compile-mode components-reduce-overhead --dtype float16 --mask-postprocess cuda-inline --duration-s 90 --warmup-s 45 --render-mode none --gpu-sampling --gpu-sampling-interval-s 0.2 --profile-cuda-events --profile-edgetam-stages` |
| `mask_only_components-reduce-overhead_float32` | `none` | `components-reduce-overhead` | `float32` | `conda run --no-capture-output -n demo_2_max python demo_v2_1_5/realtime_three_view_async_filtered_fused_pcd.py --mask-only-debug --compile-mode components-reduce-overhead --dtype float32 --mask-postprocess cuda-inline --duration-s 90 --warmup-s 45 --render-mode none --gpu-sampling --gpu-sampling-interval-s 0.2 --profile-cuda-events --profile-edgetam-stages` |
| `live_fast_native` | `realsense` | `vision-reduce-overhead` | `bfloat16` | `conda run --no-capture-output -n demo_2_max python demo_v2_1_5/realtime_three_view_async_filtered_fused_pcd.py --live-fast-native --compile-mode vision-reduce-overhead --dtype bfloat16 --mask-postprocess cuda-inline --duration-s 90 --warmup-s 45 --render-mode none --gpu-sampling --gpu-sampling-interval-s 0.2 --profile-cuda-events --profile-edgetam-stages` |
| `live_quality_ffs` | `ffs` | `vision-reduce-overhead` | `bfloat16` | `conda run --no-capture-output -n demo_2_max python demo_v2_1_5/realtime_three_view_async_filtered_fused_pcd.py --live-quality-ffs --compile-mode vision-reduce-overhead --dtype bfloat16 --mask-postprocess cuda-inline --duration-s 90 --warmup-s 45 --render-mode none --gpu-sampling --gpu-sampling-interval-s 0.2 --profile-cuda-events --profile-edgetam-stages` |

## Existing Profile Summaries

| profile | depth | compile | dtype | EdgeTAM p50 | EdgeTAM p90 | fusion FPS | GPU p50/p95 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| `docs/generated/demo2_1_5_parallel_edgetam_eager_probe_profile.json` | `realsense` | `None` | `None` | `187.48` | `209.24` | `0.65` | `24.0/28.1` |

## Decision Rules

- `hf_keep_default`: Keep HF default unless another backend improves model_ms p90 and e2e p90 with mask parity.
- `cuda_inline`: Promote only if mask parity passes and p90 improves.
- `component_compile`: Treat as experimental until warmup and graph-break behavior are stable.
- `ffs_quality`: Keep FFS quality mode separate from native fast mode.
