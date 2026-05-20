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
- render FPS after warmup: `0.14`
- raw fusion FPS after warmup: `0.00`
- filter output FPS after warmup: `0.00`
- fusion FPS after warmup: `15.56`
- stage period p50 after warmup: `64.02 ms`
- display packet period p50 after warmup: `4895.79 ms`
- groups after warmup: `3387`
- complete fused groups after warmup: `1765`
- rendered groups after warmup: `16`
- complete group ratio after warmup: `0.521`
- stage drop count after warmup: `0`
- raw fused pending replacements total: `0`
- render buffer dropped total: `0`
- target deficit: `29.86`
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
| camera startup ms | `11014.31` |
| EdgeTAM model load ms | `743.25` |
| EdgeTAM compile wrap ms | `449.90` |
| EdgeTAM compile prewarm ms | `n/a` |
| EdgeTAM warmup/first forward ms | `125.23` |
| SAM3.1 model load ms | `7552.74` |
| SAM3.1 cam0 segment ms | `571.77` |
| SAM3.1 cam1 segment ms | `125.14` |
| SAM3.1 cam2 segment ms | `122.35` |
| FFS runner init ms | `n/a` |
| FFS first run ms | `n/a` |
| session init + prompt add ms | `2.84` |
| SAM3.1 release cleanup ms | `308.51` |
| time to first complete group s | `20.29` |
| time to first rendered group s | `26.06` |

## GPU Sampling

- backend requested: `nvml`
- backend used: `unavailable`
- device index: `0`
- interval s: `0.500`
- samples after warmup: `474`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `gpu_util_pct` | `45.00` | `100.00` | `100.00` | `100.00` |
| `memory_util_pct` | `11.00` | `69.70` | `76.35` | `80.00` |
| `memory_used_mb` | `3569.53` | `24079.68` | `24189.63` | `24467.44` |
| `power_w` | `156.93` | `365.16` | `372.68` | `403.71` |
| `sm_clock_mhz` | `2670.00` | `2670.00` | `2670.00` | `2685.00` |
| `mem_clock_mhz` | `10251.00` | `10251.00` | `10251.00` | `10501.00` |
| `temperature_c` | `51.00` | `73.00` | `76.00` | `82.00` |

## Throughput periods

| Event | median ms | p90 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| `capture_group_period_ms` | `33.49` | `36.97` | `38.45` | `42.40` |
| `display_packet_publish_period_ms` | `4895.79` | `14453.18` | `14553.10` | `14626.13` |
| `edgetam_stage_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_stage_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `filter_output_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `fusion_publish_period_ms` | `64.40` | `71.88` | `73.97` | `83.19` |
| `gpu_owner_publish_period_ms` | `64.02` | `69.41` | `71.32` | `94.00` |
| `raw_fusion_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_period_ms` | `4889.99` | `14450.60` | `14553.52` | `14626.04` |
| `stage_join_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `13.72` | `23.88` | `25.08` | `42.77` |
| `edgetam_model_ms` | `15.59` | `26.14` | `27.80` | `39.48` |
| `edgetam_preprocess_ms` | `0.56` | `0.74` | `0.84` | `1.85` |
| `edgetam_prompt_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_postprocess_ms` | `0.03` | `0.04` | `0.04` | `5.95` |
| `edgetam_mask_resize_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_mask_threshold_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_mask_to_cpu_ms` | `0.41` | `0.52` | `0.60` | `6.61` |
| `edgetam_total_ms` | `16.19` | `26.87` | `28.56` | `40.01` |
| `ffs_cycle_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_batch_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_model_ms` | `1.22` | `1.47` | `1.59` | `33.78` |
| `edgetam_batch_vision_total_ms` | `7.13` | `9.39` | `10.39` | `42.79` |
| `edgetam_batch_vision_preprocess_ms` | `1.67` | `2.21` | `2.51` | `5.54` |
| `edgetam_cam0_model_ms` | `23.84` | `28.53` | `29.84` | `36.65` |
| `edgetam_cam1_model_ms` | `14.92` | `17.87` | `19.83` | `39.48` |
| `edgetam_cam2_model_ms` | `14.17` | `16.13` | `16.77` | `20.81` |
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
| `gpu_owner_total_ms` | `63.98` | `69.36` | `71.31` | `100.19` |
| `gpu_owner_ffs_cycle_ms` | `0.32` | `0.61` | `0.89` | `1.79` |
| `gpu_owner_edgetam_cycle_ms` | `63.56` | `68.91` | `70.88` | `98.80` |
| `raw_fusion_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `fusion_total_ms` | `18.18` | `20.19` | `20.99` | `27.19` |
| `filter_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `filter_input_age_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `object_enhanced_pt_ms` | `3.68` | `4.18` | `4.44` | `6.21` |
| `controller_pt_filter_ms` | `5.69` | `6.34` | `6.60` | `9.30` |
| `render_total_ms` | `1.75` | `2.38` | `2.76` | `2.98` |
| `render_queue_wait_ms` | `3972.16` | `4041.07` | `4084.04` | `4187.46` |
| `render_gpu_to_cpu_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_combine_ms` | `0.07` | `0.13` | `0.21` | `0.40` |
| `render_cpu_format_ms` | `0.23` | `0.34` | `0.42` | `0.58` |
| `render_open3d_points_update_ms` | `0.08` | `0.10` | `0.10` | `0.11` |
| `render_open3d_colors_update_ms` | `0.07` | `0.08` | `0.10` | `0.13` |
| `render_open3d_update_geometry_ms` | `1.29` | `1.52` | `1.60` | `1.81` |
| `render_poll_events_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_update_renderer_ms` | `0.02` | `0.02` | `0.02` | `0.03` |
| `open3d_object_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_controller_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `1820` | `6.21` | `24000` | `5750` |
| `1099` | `5.81` | `24000` | `5782` |
| `1552` | `5.46` | `24000` | `5706` |
| `3322` | `5.45` | `24000` | `5781` |
| `2043` | `5.44` | `24000` | `5761` |
| `575` | `5.37` | `24000` | `5734` |
| `1042` | `5.30` | `24000` | `5768` |
| `2455` | `5.29` | `24000` | `5779` |
| `2216` | `5.25` | `24000` | `5856` |
| `1337` | `5.22` | `24000` | `5736` |
