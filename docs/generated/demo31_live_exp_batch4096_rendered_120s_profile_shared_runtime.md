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
- render FPS after warmup: `0.15`
- raw fusion FPS after warmup: `0.00`
- filter output FPS after warmup: `0.00`
- fusion FPS after warmup: `15.48`
- stage period p50 after warmup: `64.41 ms`
- display packet period p50 after warmup: `4896.83 ms`
- groups after warmup: `3389`
- complete fused groups after warmup: `1756`
- rendered groups after warmup: `12`
- complete group ratio after warmup: `0.518`
- stage drop count after warmup: `0`
- raw fused pending replacements total: `0`
- render buffer dropped total: `0`
- target deficit: `29.85`
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
| camera startup ms | `11024.62` |
| EdgeTAM model load ms | `555.30` |
| EdgeTAM compile wrap ms | `595.06` |
| EdgeTAM compile prewarm ms | `n/a` |
| EdgeTAM warmup/first forward ms | `129.48` |
| SAM3.1 model load ms | `7564.60` |
| SAM3.1 cam0 segment ms | `594.53` |
| SAM3.1 cam1 segment ms | `121.55` |
| SAM3.1 cam2 segment ms | `124.43` |
| FFS runner init ms | `n/a` |
| FFS first run ms | `n/a` |
| session init + prompt add ms | `2.37` |
| SAM3.1 release cleanup ms | `298.06` |
| time to first complete group s | `20.39` |
| time to first rendered group s | `40.79` |

## GPU Sampling

- backend requested: `nvml`
- backend used: `unavailable`
- device index: `0`
- interval s: `0.500`
- samples after warmup: `474`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `gpu_util_pct` | `45.00` | `100.00` | `100.00` | `100.00` |
| `memory_util_pct` | `11.00` | `74.00` | `77.00` | `84.00` |
| `memory_used_mb` | `3591.50` | `24124.81` | `24214.29` | `24518.81` |
| `power_w` | `157.43` | `368.22` | `375.25` | `396.74` |
| `sm_clock_mhz` | `2670.00` | `2670.00` | `2670.00` | `2685.00` |
| `mem_clock_mhz` | `10251.00` | `10251.00` | `10251.00` | `10501.00` |
| `temperature_c` | `52.00` | `78.00` | `79.00` | `83.00` |

## Throughput periods

| Event | median ms | p90 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| `capture_group_period_ms` | `33.48` | `36.94` | `38.67` | `42.68` |
| `display_packet_publish_period_ms` | `4896.83` | `9413.81` | `11882.69` | `14351.57` |
| `edgetam_stage_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_stage_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `filter_output_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `fusion_publish_period_ms` | `64.66` | `71.90` | `74.08` | `82.94` |
| `gpu_owner_publish_period_ms` | `64.41` | `69.83` | `71.76` | `80.37` |
| `raw_fusion_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_period_ms` | `4897.61` | `9414.76` | `11880.76` | `14346.77` |
| `stage_join_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `12.15` | `23.97` | `28.29` | `43.67` |
| `edgetam_model_ms` | `15.46` | `26.55` | `28.40` | `37.68` |
| `edgetam_preprocess_ms` | `0.56` | `0.74` | `0.85` | `2.74` |
| `edgetam_prompt_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_postprocess_ms` | `0.03` | `0.04` | `0.04` | `0.59` |
| `edgetam_mask_resize_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_mask_threshold_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_mask_to_cpu_ms` | `0.41` | `0.53` | `0.61` | `8.37` |
| `edgetam_total_ms` | `16.06` | `27.28` | `29.07` | `38.20` |
| `ffs_cycle_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_batch_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_model_ms` | `1.23` | `1.47` | `1.56` | `32.92` |
| `edgetam_batch_vision_total_ms` | `7.16` | `9.78` | `10.61` | `41.92` |
| `edgetam_batch_vision_preprocess_ms` | `1.68` | `2.22` | `2.53` | `8.23` |
| `edgetam_cam0_model_ms` | `24.11` | `29.17` | `30.34` | `37.68` |
| `edgetam_cam1_model_ms` | `14.81` | `17.51` | `20.23` | `31.51` |
| `edgetam_cam2_model_ms` | `14.09` | `15.96` | `16.61` | `20.02` |
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
| `gpu_owner_total_ms` | `64.38` | `69.79` | `71.77` | `104.40` |
| `gpu_owner_ffs_cycle_ms` | `0.33` | `0.56` | `0.85` | `1.88` |
| `gpu_owner_edgetam_cycle_ms` | `63.98` | `69.41` | `71.35` | `102.81` |
| `raw_fusion_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `fusion_total_ms` | `18.60` | `20.81` | `21.72` | `25.53` |
| `filter_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `filter_input_age_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `object_enhanced_pt_ms` | `3.68` | `4.20` | `4.44` | `5.77` |
| `controller_pt_filter_ms` | `6.15` | `6.92` | `7.23` | `10.38` |
| `render_total_ms` | `1.95` | `2.47` | `2.82` | `3.21` |
| `render_queue_wait_ms` | `3947.46` | `4217.28` | `4261.10` | `4285.47` |
| `render_gpu_to_cpu_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_combine_ms` | `0.08` | `0.11` | `0.15` | `0.19` |
| `render_cpu_format_ms` | `0.26` | `0.33` | `0.33` | `0.34` |
| `render_open3d_points_update_ms` | `0.09` | `0.10` | `0.11` | `0.11` |
| `render_open3d_colors_update_ms` | `0.08` | `0.10` | `0.11` | `0.13` |
| `render_open3d_update_geometry_ms` | `1.47` | `1.68` | `1.70` | `1.72` |
| `render_poll_events_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_update_renderer_ms` | `0.02` | `0.02` | `0.02` | `0.03` |
| `open3d_object_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_controller_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `188` | `5.77` | `24000` | `5707` |
| `243` | `5.53` | `24000` | `5716` |
| `2756` | `5.51` | `24000` | `5792` |
| `465` | `5.28` | `24000` | `5715` |
| `1495` | `5.27` | `24000` | `5750` |
| `467` | `5.26` | `24000` | `5725` |
| `2090` | `5.25` | `24000` | `5783` |
| `138` | `5.24` | `24000` | `5746` |
| `1721` | `5.20` | `24000` | `5730` |
| `1097` | `5.19` | `24000` | `5783` |
