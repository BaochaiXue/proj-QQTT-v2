# Demo 2.2 controller-object-exp batch-vision profile summary

Date: 2026-05-17

Mode:

```text
object: stuffed animal
controller: towel
experiment_mode: controller-object-exp
EdgeTAM path: shared model + batch vision encoder + per-camera session state
FFS: local TensorRT batch=3
renderer: pointcloud, render_every_n=1
```

## Files

```text
warmup/no-render:
  docs/generated/demo22_controller_object_exp_batchvision_warmup_no_render_profile.json
real/pointcloud:
  docs/generated/demo22_controller_object_exp_batchvision_real_pointcloud_profile.json
serial baselines:
  docs/generated/demo22_controller_object_exp_warmup_no_render_profile.json
  docs/generated/demo22_controller_object_exp_real_pointcloud_profile.json
```

## Warmup no-render

| Metric | Serial | Batch vision |
| --- | ---: | ---: |
| raw_fusion_fps | 5.156 | 6.105 |
| filter_fps | 5.158 | 6.106 |
| gpu_owner p50 / p90 / p95 ms | 186.152 / 205.746 / 220.459 | 157.995 / 168.623 / 172.154 |
| ffs p50 / p90 / p95 ms | 70.728 / 75.880 / 78.329 | 67.849 / 71.781 / 74.670 |
| batch vision model p50 / p90 / p95 ms | 0.000 / 0.000 / 0.000 | 11.424 / 14.024 / 15.044 |
| batch vision total p50 / p90 / p95 ms | 0.000 / 0.000 / 0.000 | 18.946 / 22.360 / 23.007 |
| cam0 model p50 / p90 / p95 ms | 35.349 / 41.273 / 43.749 | 23.193 / 26.677 / 28.177 |
| cam1 model p50 / p90 / p95 ms | 35.183 / 41.175 / 43.891 | 22.610 / 26.303 / 27.100 |
| cam2 model p50 / p90 / p95 ms | 34.986 / 40.752 / 42.584 | 22.475 / 25.199 / 26.743 |

## Real pointcloud

| Metric | Serial | Batch vision |
| --- | ---: | ---: |
| final_fps / filter_fps | 4.710 | 5.689 |
| render_fps | 4.710 | 5.689 |
| gpu_owner p50 / p90 / p95 ms | 205.408 / 228.765 / 237.719 | 168.104 / 184.845 / 193.019 |
| ffs p50 / p90 / p95 ms | 75.811 / 79.566 / 81.423 | 72.725 / 77.918 / 79.794 |
| batch vision model p50 / p90 / p95 ms | 0.000 / 0.000 / 0.000 | 12.533 / 15.631 / 16.883 |
| batch vision total p50 / p90 / p95 ms | 0.000 / 0.000 / 0.000 | 20.380 / 24.265 / 25.806 |
| cam0 model p50 / p90 / p95 ms | 39.968 / 47.706 / 50.651 | 24.741 / 28.714 / 30.026 |
| cam1 model p50 / p90 / p95 ms | 37.773 / 46.027 / 48.852 | 24.178 / 29.067 / 31.057 |
| cam2 model p50 / p90 / p95 ms | 40.195 / 47.702 / 50.591 | 23.747 / 28.319 / 29.858 |
| filter p50 / p90 / p95 ms | 40.351 / 43.770 / 45.415 | 40.078 / 45.068 / 47.321 |
| render p50 / p90 / p95 ms | 2.239 / 2.885 / 3.371 | 2.036 / 2.668 / 2.890 |
| display packet period p50 / p90 / p95 ms | n/a | 168.320 / 184.885 / 196.616 |

## Decision

```text
final_fps_source: Demo 2.2 real pointcloud profile filter_fps
final_fps: 5.689
15_fps_target_pass: false
batch_vision_speedup_vs_serial_pointcloud: 1.208x
```

Batch vision is a real improvement, but it does not change the architecture
enough to reach 15 FPS. The current remaining bottleneck is still the
single-owner GPU period: local FFS plus batch vision plus three per-camera
state/model updates. The next code change should be a staged pipeline scheduler,
not renderer work.
