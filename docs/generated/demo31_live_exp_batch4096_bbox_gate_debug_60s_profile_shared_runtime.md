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
- fusion FPS after warmup: `15.43`
- stage period p50 after warmup: `64.64 ms`
- display packet period p50 after warmup: `7211.21 ms`
- groups after warmup: `1303`
- complete fused groups after warmup: `674`
- rendered groups after warmup: `5`
- complete group ratio after warmup: `0.517`
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
| camera startup ms | `11038.80` |
| EdgeTAM model load ms | `550.73` |
| EdgeTAM compile wrap ms | `581.52` |
| EdgeTAM compile prewarm ms | `n/a` |
| EdgeTAM warmup/first forward ms | `127.40` |
| SAM3.1 model load ms | `7422.57` |
| SAM3.1 cam0 segment ms | `596.64` |
| SAM3.1 cam1 segment ms | `123.91` |
| SAM3.1 cam2 segment ms | `122.58` |
| FFS runner init ms | `n/a` |
| FFS first run ms | `n/a` |
| session init + prompt add ms | `3.00` |
| SAM3.1 release cleanup ms | `301.26` |
| time to first complete group s | `19.95` |
| time to first rendered group s | `30.50` |

## GPU Sampling

- backend requested: `nvml`
- backend used: `unavailable`
- device index: `0`
- interval s: `0.500`
- samples after warmup: `190`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `gpu_util_pct` | `44.00` | `100.00` | `100.00` | `100.00` |
| `memory_util_pct` | `11.00` | `72.00` | `74.55` | `79.00` |
| `memory_used_mb` | `3567.81` | `23943.16` | `24128.39` | `24283.44` |
| `power_w` | `155.13` | `361.03` | `370.21` | `389.50` |
| `sm_clock_mhz` | `2670.00` | `2670.00` | `2670.00` | `2685.00` |
| `mem_clock_mhz` | `10251.00` | `10251.00` | `10251.00` | `10501.00` |
| `temperature_c` | `50.00` | `74.00` | `75.00` | `78.00` |

## Throughput periods

| Event | median ms | p90 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| `capture_group_period_ms` | `33.47` | `37.30` | `38.68` | `42.32` |
| `display_packet_publish_period_ms` | `7211.21` | `9521.35` | `9524.11` | `9526.88` |
| `edgetam_stage_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_stage_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `filter_output_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `fusion_publish_period_ms` | `64.72` | `73.42` | `75.22` | `83.39` |
| `gpu_owner_publish_period_ms` | `64.64` | `70.37` | `72.34` | `81.17` |
| `raw_fusion_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_period_ms` | `7215.44` | `9519.14` | `9520.72` | `9522.30` |
| `stage_join_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `12.00` | `25.15` | `26.23` | `41.36` |
| `edgetam_model_ms` | `15.79` | `26.19` | `28.68` | `34.50` |
| `edgetam_preprocess_ms` | `0.56` | `0.73` | `0.79` | `1.29` |
| `edgetam_prompt_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_postprocess_ms` | `0.02` | `0.04` | `0.04` | `4.84` |
| `edgetam_mask_resize_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_mask_threshold_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_mask_to_cpu_ms` | `0.41` | `0.52` | `0.62` | `8.61` |
| `edgetam_total_ms` | `16.36` | `27.14` | `29.43` | `35.41` |
| `ffs_cycle_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_batch_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_model_ms` | `1.23` | `1.44` | `1.56` | `2.60` |
| `edgetam_batch_vision_total_ms` | `7.14` | `9.00` | `10.00` | `13.44` |
| `edgetam_batch_vision_preprocess_ms` | `1.68` | `2.18` | `2.37` | `3.88` |
| `edgetam_cam0_model_ms` | `23.67` | `29.91` | `31.02` | `34.50` |
| `edgetam_cam1_model_ms` | `15.05` | `19.50` | `21.25` | `24.52` |
| `edgetam_cam2_model_ms` | `14.29` | `16.18` | `16.91` | `21.39` |
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
| `gpu_owner_total_ms` | `64.54` | `70.33` | `72.29` | `81.14` |
| `gpu_owner_ffs_cycle_ms` | `0.28` | `0.45` | `0.66` | `1.78` |
| `gpu_owner_edgetam_cycle_ms` | `64.25` | `70.01` | `71.97` | `80.85` |
| `raw_fusion_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `fusion_total_ms` | `18.34` | `20.62` | `21.22` | `24.34` |
| `filter_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `filter_input_age_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `object_enhanced_pt_ms` | `3.70` | `4.28` | `4.53` | `5.40` |
| `controller_pt_filter_ms` | `5.77` | `6.41` | `6.64` | `8.76` |
| `render_total_ms` | `1.58` | `2.35` | `2.57` | `2.79` |
| `render_queue_wait_ms` | `3945.26` | `4082.76` | `4105.84` | `4128.91` |
| `render_gpu_to_cpu_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_combine_ms` | `0.07` | `0.10` | `0.10` | `0.10` |
| `render_cpu_format_ms` | `0.22` | `0.27` | `0.27` | `0.28` |
| `render_open3d_points_update_ms` | `0.08` | `0.09` | `0.10` | `0.10` |
| `render_open3d_colors_update_ms` | `0.06` | `0.08` | `0.08` | `0.08` |
| `render_open3d_update_geometry_ms` | `1.13` | `1.38` | `1.43` | `1.48` |
| `render_poll_events_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_update_renderer_ms` | `0.02` | `0.02` | `0.02` | `0.02` |
| `open3d_object_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_controller_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `1245` | `5.40` | `24000` | `5730` |
| `287` | `5.20` | `24000` | `5746` |
| `1004` | `5.20` | `24000` | `5772` |
| `508` | `5.06` | `24000` | `5782` |
| `1403` | `5.06` | `24000` | `5782` |
| `979` | `5.03` | `24000` | `5750` |
| `1251` | `4.95` | `24000` | `5699` |
| `1278` | `4.94` | `24000` | `5752` |
| `408` | `4.93` | `24000` | `5768` |
| `167` | `4.92` | `24000` | `5761` |
