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
- render FPS after warmup: `0.00`
- raw fusion FPS after warmup: `0.00`
- filter output FPS after warmup: `0.00`
- fusion FPS after warmup: `15.19`
- stage period p50 after warmup: `65.33 ms`
- display packet period p50 after warmup: `67.08 ms`
- groups after warmup: `1413`
- complete fused groups after warmup: `719`
- rendered groups after warmup: `0`
- complete group ratio after warmup: `0.509`
- stage drop count after warmup: `0`
- raw fused pending replacements total: `0`
- render buffer dropped total: `410`
- target deficit: `30.00`
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
| camera startup ms | `11014.93` |
| EdgeTAM model load ms | `532.46` |
| EdgeTAM compile wrap ms | `603.54` |
| EdgeTAM compile prewarm ms | `n/a` |
| EdgeTAM warmup/first forward ms | `124.91` |
| SAM3.1 model load ms | `7462.63` |
| SAM3.1 cam0 segment ms | `589.66` |
| SAM3.1 cam1 segment ms | `121.60` |
| SAM3.1 cam2 segment ms | `120.70` |
| FFS runner init ms | `n/a` |
| FFS first run ms | `n/a` |
| session init + prompt add ms | `2.60` |
| SAM3.1 release cleanup ms | `277.18` |
| time to first complete group s | `19.18` |
| time to first rendered group s | `n/a` |

## GPU Sampling

- backend requested: `nvml`
- backend used: `unavailable`
- device index: `0`
- interval s: `0.500`
- samples after warmup: `202`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `gpu_util_pct` | `44.00` | `100.00` | `100.00` | `100.00` |
| `memory_util_pct` | `11.00` | `69.00` | `76.00` | `79.00` |
| `memory_used_mb` | `3539.38` | `23904.07` | `24045.92` | `24419.00` |
| `power_w` | `154.60` | `366.61` | `373.09` | `389.04` |
| `sm_clock_mhz` | `2670.00` | `2670.00` | `2670.00` | `2685.00` |
| `mem_clock_mhz` | `10251.00` | `10251.00` | `10251.00` | `10501.00` |
| `temperature_c` | `50.50` | `74.00` | `76.00` | `79.00` |

## Throughput periods

| Event | median ms | p90 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| `capture_group_period_ms` | `33.49` | `37.41` | `38.59` | `42.83` |
| `display_packet_publish_period_ms` | `67.08` | `76.02` | `137.95` | `2997.83` |
| `edgetam_stage_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_stage_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `filter_output_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `fusion_publish_period_ms` | `66.07` | `73.48` | `75.03` | `83.36` |
| `gpu_owner_publish_period_ms` | `65.33` | `71.27` | `73.32` | `81.53` |
| `raw_fusion_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `stage_join_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `11.63` | `26.05` | `29.76` | `42.92` |
| `edgetam_model_ms` | `16.22` | `26.77` | `28.56` | `35.00` |
| `edgetam_preprocess_ms` | `0.56` | `0.71` | `0.80` | `5.34` |
| `edgetam_prompt_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_postprocess_ms` | `0.03` | `0.04` | `0.04` | `4.87` |
| `edgetam_mask_resize_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_mask_threshold_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_mask_to_cpu_ms` | `0.42` | `0.52` | `0.60` | `6.81` |
| `edgetam_total_ms` | `16.82` | `27.38` | `29.30` | `35.64` |
| `ffs_cycle_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_batch_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_model_ms` | `1.22` | `1.48` | `1.61` | `2.56` |
| `edgetam_batch_vision_total_ms` | `7.02` | `8.95` | `9.97` | `21.60` |
| `edgetam_batch_vision_preprocess_ms` | `1.67` | `2.13` | `2.41` | `16.03` |
| `edgetam_cam0_model_ms` | `24.04` | `29.34` | `30.88` | `35.00` |
| `edgetam_cam1_model_ms` | `15.65` | `18.60` | `20.90` | `25.16` |
| `edgetam_cam2_model_ms` | `14.68` | `16.42` | `16.88` | `19.03` |
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
| `gpu_owner_total_ms` | `65.28` | `71.23` | `73.28` | `81.50` |
| `gpu_owner_ffs_cycle_ms` | `0.31` | `0.57` | `0.85` | `1.70` |
| `gpu_owner_edgetam_cycle_ms` | `64.90` | `70.79` | `72.91` | `81.05` |
| `raw_fusion_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `fusion_total_ms` | `17.96` | `20.26` | `21.17` | `24.13` |
| `filter_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `filter_input_age_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `object_enhanced_pt_ms` | `3.66` | `4.23` | `4.40` | `5.73` |
| `controller_pt_filter_ms` | `5.61` | `6.24` | `6.52` | `8.55` |
| `render_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_queue_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_gpu_to_cpu_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_combine_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_cpu_format_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_open3d_points_update_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_open3d_colors_update_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_open3d_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_poll_events_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_update_renderer_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_object_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_controller_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `1126` | `5.73` | `24000` | `5788` |
| `1130` | `5.34` | `24000` | `5804` |
| `1124` | `5.34` | `24000` | `5785` |
| `601` | `5.24` | `24000` | `5754` |
| `401` | `5.21` | `24000` | `5770` |
| `425` | `5.16` | `24000` | `5703` |
| `453` | `5.10` | `24000` | `5749` |
| `1040` | `5.10` | `24000` | `5711` |
| `143` | `4.95` | `24000` | `5794` |
| `1424` | `4.94` | `24000` | `5806` |
