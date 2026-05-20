# Demo 3.1 performance profile

- preset: `demo2.1.5-live-fast-native`
- canonical preset: `demo2.1.5-live-fast-native`
- target FPS: `30.00`
- capture group target FPS: `30.00`
- compile mode: `vision-reduce-overhead`
- dtype: `bfloat16`
- EdgeTAM input path: `pil`
- mask postprocess: `hf`
- EdgeTAM live session keep frames: `64`
- render backend: `legacy-inplace`
- render latest-only: `True`
- render copy mode: `sync-cpu`
- render FPS after warmup: `3.45`
- raw fusion FPS after warmup: `0.00`
- filter output FPS after warmup: `0.00`
- fusion FPS after warmup: `15.31`
- stage period p50 after warmup: `64.84 ms`
- display packet period p50 after warmup: `152.19 ms`
- groups after warmup: `1593`
- complete fused groups after warmup: `812`
- rendered groups after warmup: `176`
- complete group ratio after warmup: `0.510`
- stage drop count after warmup: `0`
- raw fused pending replacements total: `0`
- render buffer dropped total: `0`
- target deficit: `26.55`
- bottleneck class: `upstream_supply`
- GPU pipeline: `single-owner`
- single-owner order: `ffs-then-edgetam`
- filter scheduler: `none`
- render filtered only: `False`
- pin memory mode: `off`
- FFS input staging: `pinned`
- H2D stream mode: `default`

## Init Profile

| Stage | value |
| --- | ---: |
| parallel init max wait ms | `0.01` |
| camera startup ms | `10961.41` |
| EdgeTAM model load ms | `530.79` |
| EdgeTAM compile wrap ms | `636.88` |
| EdgeTAM compile prewarm ms | `n/a` |
| EdgeTAM warmup/first forward ms | `129.91` |
| SAM3.1 model load ms | `7739.40` |
| SAM3.1 cam0 segment ms | `604.36` |
| SAM3.1 cam1 segment ms | `122.15` |
| SAM3.1 cam2 segment ms | `119.73` |
| FFS runner init ms | `n/a` |
| FFS first run ms | `n/a` |
| session init + prompt add ms | `2.79` |
| SAM3.1 release cleanup ms | `319.86` |
| time to first complete group s | `20.44` |
| time to first rendered group s | `20.92` |

## GPU Sampling

- backend requested: `nvml`
- backend used: `unavailable`
- device index: `0`
- interval s: `0.500`
- samples after warmup: `226`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `gpu_util_pct` | `46.00` | `96.00` | `96.00` | `97.00` |
| `memory_util_pct` | `12.00` | `82.00` | `83.00` | `86.00` |
| `memory_used_mb` | `7538.69` | `8333.56` | `8367.66` | `8437.88` |
| `power_w` | `157.42` | `313.13` | `315.64` | `319.28` |
| `sm_clock_mhz` | `2670.00` | `2670.00` | `2670.00` | `2685.00` |
| `mem_clock_mhz` | `10251.00` | `10251.00` | `10251.00` | `10251.00` |
| `temperature_c` | `51.00` | `70.00` | `71.00` | `73.00` |

## Throughput periods

| Event | median ms | p90 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| `capture_group_period_ms` | `33.49` | `36.89` | `38.51` | `44.03` |
| `display_packet_publish_period_ms` | `152.19` | `755.15` | `1243.72` | `1802.64` |
| `edgetam_stage_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_stage_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `filter_output_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `fusion_publish_period_ms` | `65.03` | `71.72` | `73.72` | `109.40` |
| `gpu_owner_publish_period_ms` | `64.84` | `69.99` | `72.18` | `113.72` |
| `raw_fusion_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_period_ms` | `152.73` | `757.75` | `1245.43` | `1804.10` |
| `stage_join_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `11.57` | `25.88` | `29.63` | `42.95` |
| `edgetam_model_ms` | `16.30` | `25.40` | `26.91` | `58.06` |
| `edgetam_preprocess_ms` | `0.56` | `0.78` | `0.93` | `2.46` |
| `edgetam_prompt_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_postprocess_ms` | `0.02` | `0.04` | `0.04` | `0.77` |
| `edgetam_mask_resize_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_mask_threshold_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_mask_to_cpu_ms` | `0.41` | `0.52` | `0.61` | `7.50` |
| `edgetam_total_ms` | `16.91` | `26.07` | `27.58` | `58.68` |
| `ffs_cycle_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_batch_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_model_ms` | `1.24` | `1.57` | `1.74` | `39.50` |
| `edgetam_batch_vision_total_ms` | `7.25` | `9.96` | `10.97` | `48.26` |
| `edgetam_batch_vision_preprocess_ms` | `1.68` | `2.33` | `2.78` | `7.37` |
| `edgetam_cam0_model_ms` | `23.96` | `27.63` | `28.95` | `58.06` |
| `edgetam_cam1_model_ms` | `15.18` | `19.13` | `20.70` | `28.91` |
| `edgetam_cam2_model_ms` | `14.63` | `17.57` | `18.99` | `35.03` |
| `edgetam_cam0_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_cam1_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_cam2_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edge_pin_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edge_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edge_cam0_pin_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edge_cam1_pin_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edge_cam2_pin_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edge_cam0_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edge_cam1_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edge_cam2_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_stage_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam0_stage_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam1_stage_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam2_stage_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam0_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam1_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam2_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `gpu_owner_total_ms` | `64.80` | `69.96` | `72.18` | `114.61` |
| `gpu_owner_ffs_cycle_ms` | `0.28` | `0.54` | `0.79` | `2.60` |
| `gpu_owner_edgetam_cycle_ms` | `64.41` | `69.65` | `71.75` | `113.39` |
| `raw_fusion_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `fusion_total_ms` | `18.78` | `21.38` | `22.47` | `30.14` |
| `filter_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `filter_input_age_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `object_enhanced_pt_ms` | `3.76` | `4.43` | `4.67` | `7.02` |
| `controller_pt_filter_ms` | `5.59` | `6.70` | `7.13` | `12.08` |
| `render_total_ms` | `1.54` | `1.97` | `2.25` | `7.77` |
| `render_queue_wait_ms` | `210.37` | `263.98` | `272.24` | `298.16` |
| `render_gpu_to_cpu_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_combine_ms` | `0.08` | `0.16` | `0.24` | `5.34` |
| `render_cpu_format_ms` | `0.25` | `0.38` | `0.44` | `5.47` |
| `render_open3d_points_update_ms` | `0.08` | `0.11` | `0.14` | `0.24` |
| `render_open3d_colors_update_ms` | `0.07` | `0.13` | `0.16` | `4.81` |
| `render_open3d_update_geometry_ms` | `1.20` | `1.54` | `1.75` | `7.53` |
| `render_poll_events_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_update_renderer_ms` | `0.02` | `0.02` | `0.02` | `0.10` |
| `open3d_object_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_controller_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `216` | `7.02` | `24000` | `5719` |
| `328` | `6.65` | `24000` | `5768` |
| `630` | `6.47` | `24000` | `5776` |
| `347` | `6.18` | `24000` | `5768` |
| `1628` | `6.08` | `24000` | `5808` |
| `1454` | `5.96` | `24000` | `5776` |
| `1215` | `5.85` | `24000` | `5774` |
| `451` | `5.83` | `24000` | `5735` |
| `1064` | `5.79` | `24000` | `5725` |
| `585` | `5.76` | `24000` | `5773` |
