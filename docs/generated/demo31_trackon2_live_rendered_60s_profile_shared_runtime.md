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
- render FPS after warmup: `1.00`
- raw fusion FPS after warmup: `0.00`
- filter output FPS after warmup: `0.00`
- fusion FPS after warmup: `15.38`
- stage period p50 after warmup: `64.55 ms`
- display packet period p50 after warmup: `938.10 ms`
- groups after warmup: `1585`
- complete fused groups after warmup: `818`
- rendered groups after warmup: `52`
- complete group ratio after warmup: `0.516`
- stage drop count after warmup: `0`
- raw fused pending replacements total: `0`
- render buffer dropped total: `0`
- target deficit: `29.00`
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
| camera startup ms | `11022.21` |
| EdgeTAM model load ms | `564.35` |
| EdgeTAM compile wrap ms | `732.06` |
| EdgeTAM compile prewarm ms | `n/a` |
| EdgeTAM warmup/first forward ms | `128.63` |
| SAM3.1 model load ms | `8106.41` |
| SAM3.1 cam0 segment ms | `571.10` |
| SAM3.1 cam1 segment ms | `123.52` |
| SAM3.1 cam2 segment ms | `125.05` |
| FFS runner init ms | `n/a` |
| FFS first run ms | `n/a` |
| session init + prompt add ms | `2.87` |
| SAM3.1 release cleanup ms | `313.85` |
| time to first complete group s | `19.93` |
| time to first rendered group s | `20.84` |

## GPU Sampling

- backend requested: `nvml`
- backend used: `unavailable`
- device index: `0`
- interval s: `0.500`
- samples after warmup: `226`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `gpu_util_pct` | `46.00` | `100.00` | `100.00` | `100.00` |
| `memory_util_pct` | `12.00` | `74.00` | `76.00` | `78.00` |
| `memory_used_mb` | `3595.56` | `8562.66` | `8598.08` | `8629.06` |
| `power_w` | `157.17` | `404.45` | `405.97` | `407.60` |
| `sm_clock_mhz` | `2655.00` | `2670.00` | `2685.00` | `2685.00` |
| `mem_clock_mhz` | `10251.00` | `10251.00` | `10251.00` | `10501.00` |
| `temperature_c` | `53.00` | `84.00` | `84.00` | `87.00` |

## Throughput periods

| Event | median ms | p90 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| `capture_group_period_ms` | `33.51` | `36.94` | `38.39` | `43.12` |
| `display_packet_publish_period_ms` | `938.10` | `1947.54` | `2394.01` | `2877.28` |
| `edgetam_stage_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_stage_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `filter_output_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `fusion_publish_period_ms` | `64.08` | `73.75` | `76.86` | `86.82` |
| `gpu_owner_publish_period_ms` | `64.55` | `71.57` | `73.85` | `83.96` |
| `raw_fusion_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_period_ms` | `935.95` | `1945.70` | `2397.78` | `2877.90` |
| `stage_join_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `13.43` | `23.81` | `27.77` | `41.14` |
| `edgetam_model_ms` | `15.55` | `27.05` | `29.67` | `43.26` |
| `edgetam_preprocess_ms` | `0.56` | `0.73` | `0.80` | `2.26` |
| `edgetam_prompt_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_postprocess_ms` | `0.02` | `0.04` | `0.04` | `1.76` |
| `edgetam_mask_resize_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_mask_threshold_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_mask_to_cpu_ms` | `0.41` | `0.51` | `0.59` | `7.41` |
| `edgetam_total_ms` | `16.12` | `27.90` | `30.35` | `43.80` |
| `ffs_cycle_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_batch_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_model_ms` | `1.24` | `1.49` | `1.63` | `3.36` |
| `edgetam_batch_vision_total_ms` | `7.11` | `9.23` | `10.14` | `16.84` |
| `edgetam_batch_vision_preprocess_ms` | `1.69` | `2.19` | `2.41` | `6.78` |
| `edgetam_cam0_model_ms` | `24.21` | `30.70` | `32.76` | `43.26` |
| `edgetam_cam1_model_ms` | `14.78` | `17.98` | `20.41` | `27.30` |
| `edgetam_cam2_model_ms` | `14.00` | `16.38` | `17.34` | `24.60` |
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
| `gpu_owner_total_ms` | `64.51` | `71.52` | `73.81` | `83.92` |
| `gpu_owner_ffs_cycle_ms` | `0.26` | `0.55` | `0.82` | `1.44` |
| `gpu_owner_edgetam_cycle_ms` | `64.16` | `71.17` | `73.34` | `83.54` |
| `raw_fusion_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `fusion_total_ms` | `18.85` | `21.25` | `22.09` | `25.89` |
| `filter_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `filter_input_age_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `object_enhanced_pt_ms` | `3.72` | `4.29` | `4.65` | `5.94` |
| `controller_pt_filter_ms` | `6.12` | `6.81` | `7.17` | `11.60` |
| `render_total_ms` | `1.58` | `1.97` | `2.04` | `2.49` |
| `render_queue_wait_ms` | `523.08` | `568.05` | `589.72` | `598.17` |
| `render_gpu_to_cpu_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_combine_ms` | `0.09` | `0.13` | `0.17` | `0.31` |
| `render_cpu_format_ms` | `0.25` | `0.38` | `0.41` | `1.09` |
| `render_open3d_points_update_ms` | `0.08` | `0.12` | `0.13` | `0.21` |
| `render_open3d_colors_update_ms` | `0.06` | `0.11` | `0.16` | `0.93` |
| `render_open3d_update_geometry_ms` | `1.26` | `1.52` | `1.59` | `1.70` |
| `render_poll_events_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_update_renderer_ms` | `0.02` | `0.02` | `0.03` | `0.04` |
| `open3d_object_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_controller_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `1320` | `5.94` | `24000` | `5738` |
| `250` | `5.87` | `24000` | `5687` |
| `320` | `5.54` | `24000` | `5768` |
| `1547` | `5.25` | `24000` | `5789` |
| `877` | `5.15` | `24000` | `5660` |
| `556` | `5.15` | `24000` | `5690` |
| `1248` | `5.14` | `24000` | `5763` |
| `919` | `5.14` | `24000` | `5698` |
| `684` | `5.12` | `24000` | `5688` |
| `596` | `5.10` | `24000` | `5690` |
