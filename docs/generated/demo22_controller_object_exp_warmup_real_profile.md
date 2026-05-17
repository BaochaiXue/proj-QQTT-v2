# Demo 2.2 controller-object-exp warmup and real profile

## Run context

```text
experiment_mode: controller-object-exp
object: stuffed animal
controller: towel
camera_count: 3 RealSense D455 visible in WSL
serials: 239222300412, 239222303506, 239222300781
mask_postprocess: cuda-inline
dtype: bfloat16
compile_mode: vision-reduce-overhead
```

The first warmup attempt hit a SAM3.1 / torchvision parallel-import race, so the successful runs below used `--no-parallel-init`.

## Warmup no-render

```text
profile: docs/generated/demo22_controller_object_exp_warmup_no_render_profile.json
warmup_exclude_s: 20
render_mode: none
capture_group_fps: 13.401
raw_fusion_fps: 5.156
filter_fps: 5.158
fusion_fps: 5.158
render_fps: 0.000
complete_fusion_groups: 206
```

Post-warmup stage timings:

```text
edgetam_total_ms p50/p90/p95: 38.578 / 44.599 / 47.112
ffs_cycle_ms p50/p90/p95: 70.728 / 75.880 / 78.329
gpu_owner_total_ms p50/p90/p95: 186.152 / 205.746 / 220.459
raw_fusion_total_ms p50/p90/p95: 11.113 / 12.654 / 13.721
filter_total_ms p50/p90/p95: 38.437 / 43.053 / 44.491
```

GPU sampling after warmup:

```text
gpu_util_pct p50/p90/p95/max: 46.0 / 50.9 / 52.0 / 84.0
memory_used_mb p50/p90/p95/max: 5151 / 7041 / 7314 / 7315
power_w p50/p90/p95/max: 119.3 / 151.2 / 167.5 / 236.4
```

## Real pointcloud experiment

```text
profile: docs/generated/demo22_controller_object_exp_real_pointcloud_profile.json
log: docs/generated/demo22_controller_object_exp_real_pointcloud_20260517_173838.log
warmup_exclude_s: 30
render_mode: pointcloud
renderer: legacy-inplace, sync-cpu copy mode
render_micro_profile: enabled
capture_group_fps: 14.278
raw_fusion_fps: 4.710
filter_fps: 4.710
fusion_fps: 4.710
render_fps: 4.710
complete_fusion_groups: 464
rendered_groups: 464
render_backpressure_count: 0
```

Post-warmup stage timings:

```text
edgetam_total_ms p50/p90/p95: 42.895 / 50.683 / 54.081
ffs_cycle_ms p50/p90/p95: 75.811 / 79.566 / 81.423
gpu_owner_total_ms p50/p90/p95: 205.408 / 228.765 / 237.719
raw_fusion_total_ms p50/p90/p95: 11.590 / 13.388 / 13.789
filter_total_ms p50/p90/p95: 40.351 / 43.770 / 45.415
fusion_total_ms p50/p90/p95: 52.009 / 56.324 / 58.648
render_total_ms p50/p90/p95: 2.239 / 2.885 / 3.371
```

Render micro-profile:

```text
render_points_count p50/p90/p95: 31112 / 31206 / 31228
render_cpu_format_ms p50/p90/p95: 0.301 / 0.460 / 0.549
render_open3d_update_ms p50/p90/p95: 1.591 / 2.211 / 2.644
render_total_ms p50/p90/p95/max: 2.239 / 2.885 / 3.371 / 7.604
```

GPU sampling after warmup:

```text
gpu_util_pct p50/p90/p95/max: 44.0 / 48.0 / 49.2 / 53.0
memory_used_mb p50/p90/p95/max: 8594 / 12939 / 13461 / 13736
power_w p50/p90/p95/max: 123.8 / 141.1 / 155.1 / 212.2
```

## Decision

```text
real_render_fps_valid: yes
final_fps_source: real pointcloud profile render_fps/filter_fps
final_fps: 4.710
15_fps_target_pass: no
```

This is a real render-path profile, not a no-render or mask-only number. The current bottleneck is upstream supply, dominated by the serialized GPU owner path: FFS cycle plus three-camera EdgeTAM and CPU-side fusion/filter work. The Open3D pointcloud update itself is not the main limiter in this run.
