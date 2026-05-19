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
- render FPS after warmup: `9.28`
- raw fusion FPS after warmup: `0.00`
- filter output FPS after warmup: `0.00`
- fusion FPS after warmup: `15.53`
- stage period p50 after warmup: `64.10 ms`
- display packet period p50 after warmup: `66.91 ms`
- groups after warmup: `1032`
- complete fused groups after warmup: `533`
- rendered groups after warmup: `234`
- complete group ratio after warmup: `0.516`
- stage drop count after warmup: `0`
- raw fused pending replacements total: `0`
- render buffer dropped total: `1`
- target deficit: `20.72`
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
| camera startup ms | `11012.28` |
| EdgeTAM model load ms | `548.29` |
| EdgeTAM compile wrap ms | `611.39` |
| EdgeTAM compile prewarm ms | `n/a` |
| EdgeTAM warmup/first forward ms | `135.87` |
| SAM3.1 model load ms | `7264.81` |
| SAM3.1 cam0 segment ms | `554.57` |
| SAM3.1 cam1 segment ms | `131.99` |
| SAM3.1 cam2 segment ms | `119.04` |
| FFS runner init ms | `n/a` |
| FFS first run ms | `n/a` |
| session init + prompt add ms | `2.58` |
| SAM3.1 release cleanup ms | `300.52` |
| time to first complete group s | `20.28` |
| time to first rendered group s | `29.11` |

## GPU Sampling

- backend requested: `nvml`
- backend used: `unavailable`
- device index: `0`
- interval s: `0.500`
- samples after warmup: `158`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `gpu_util_pct` | `45.00` | `100.00` | `100.00` | `100.00` |
| `memory_util_pct` | `11.00` | `70.30` | `73.00` | `74.00` |
| `memory_used_mb` | `3537.88` | `15921.75` | `15922.91` | `15941.12` |
| `power_w` | `156.52` | `413.14` | `417.89` | `422.64` |
| `sm_clock_mhz` | `2670.00` | `2670.00` | `2685.00` | `2685.00` |
| `mem_clock_mhz` | `10251.00` | `10251.00` | `10251.00` | `10501.00` |
| `temperature_c` | `48.00` | `72.00` | `73.15` | `78.00` |

## Throughput periods

| Event | median ms | p90 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| `capture_group_period_ms` | `33.46` | `36.82` | `38.84` | `42.83` |
| `display_packet_publish_period_ms` | `66.91` | `77.62` | `421.54` | `1567.88` |
| `edgetam_stage_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_stage_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `filter_output_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `fusion_publish_period_ms` | `64.93` | `71.71` | `73.86` | `81.36` |
| `gpu_owner_publish_period_ms` | `64.10` | `69.57` | `71.69` | `81.24` |
| `raw_fusion_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_period_ms` | `66.66` | `83.64` | `416.07` | `1568.46` |
| `stage_join_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `5.68` | `25.70` | `26.84` | `34.87` |
| `edgetam_model_ms` | `16.01` | `24.57` | `26.08` | `34.25` |
| `edgetam_preprocess_ms` | `0.66` | `1.10` | `1.30` | `4.67` |
| `edgetam_prompt_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_postprocess_ms` | `0.02` | `0.04` | `0.04` | `1.57` |
| `edgetam_mask_resize_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_mask_threshold_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_mask_to_cpu_ms` | `0.41` | `0.54` | `0.69` | `7.60` |
| `edgetam_total_ms` | `16.64` | `25.38` | `26.76` | `35.21` |
| `ffs_cycle_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_batch_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_model_ms` | `1.22` | `1.49` | `1.63` | `53.24` |
| `edgetam_batch_vision_total_ms` | `7.86` | `9.95` | `10.63` | `73.47` |
| `edgetam_batch_vision_preprocess_ms` | `1.99` | `3.29` | `3.88` | `14.00` |
| `edgetam_cam0_model_ms` | `22.82` | `26.68` | `27.94` | `34.25` |
| `edgetam_cam1_model_ms` | `15.56` | `21.79` | `22.94` | `27.14` |
| `edgetam_cam2_model_ms` | `13.75` | `16.17` | `16.86` | `19.60` |
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
| `gpu_owner_total_ms` | `64.08` | `69.62` | `71.79` | `131.61` |
| `gpu_owner_ffs_cycle_ms` | `0.27` | `0.50` | `0.83` | `1.62` |
| `gpu_owner_edgetam_cycle_ms` | `63.74` | `69.15` | `71.36` | `130.19` |
| `raw_fusion_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `fusion_total_ms` | `23.76` | `26.66` | `27.69` | `32.19` |
| `filter_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `filter_input_age_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `object_enhanced_pt_ms` | `3.67` | `4.24` | `4.56` | `6.53` |
| `controller_pt_filter_ms` | `9.51` | `10.55` | `10.94` | `12.14` |
| `render_total_ms` | `1.66` | `2.03` | `2.21` | `4.92` |
| `render_queue_wait_ms` | `9.17` | `9.53` | `9.66` | `97.79` |
| `render_gpu_to_cpu_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_combine_ms` | `0.08` | `0.13` | `0.16` | `1.02` |
| `render_cpu_format_ms` | `0.26` | `0.38` | `0.43` | `1.14` |
| `render_open3d_points_update_ms` | `0.08` | `0.11` | `0.13` | `0.24` |
| `render_open3d_colors_update_ms` | `0.08` | `0.16` | `0.19` | `0.27` |
| `render_open3d_update_geometry_ms` | `1.33` | `1.64` | `1.74` | `4.47` |
| `render_poll_events_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_update_renderer_ms` | `0.02` | `0.02` | `0.03` | `0.06` |
| `open3d_object_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_controller_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `133` | `6.53` | `24000` | `5619` |
| `855` | `6.00` | `24000` | `5699` |
| `1057` | `5.31` | `24000` | `5695` |
| `294` | `5.21` | `24000` | `5665` |
| `307` | `5.03` | `24000` | `5651` |
| `1080` | `5.03` | `24000` | `5651` |
| `354` | `4.99` | `24000` | `5592` |
| `587` | `4.98` | `24000` | `5694` |
| `258` | `4.93` | `24000` | `5655` |
| `723` | `4.92` | `24000` | `5644` |
