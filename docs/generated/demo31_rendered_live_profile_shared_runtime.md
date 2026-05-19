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
- render FPS after warmup: `8.24`
- raw fusion FPS after warmup: `0.00`
- filter output FPS after warmup: `0.00`
- fusion FPS after warmup: `15.37`
- stage period p50 after warmup: `64.93 ms`
- display packet period p50 after warmup: `66.64 ms`
- groups after warmup: `3393`
- complete fused groups after warmup: `1748`
- rendered groups after warmup: `916`
- complete group ratio after warmup: `0.515`
- stage drop count after warmup: `0`
- raw fused pending replacements total: `0`
- render buffer dropped total: `148`
- target deficit: `21.76`
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
| camera startup ms | `11110.65` |
| EdgeTAM model load ms | `760.92` |
| EdgeTAM compile wrap ms | `428.38` |
| EdgeTAM compile prewarm ms | `n/a` |
| EdgeTAM warmup/first forward ms | `129.44` |
| SAM3.1 model load ms | `7258.79` |
| SAM3.1 cam0 segment ms | `563.71` |
| SAM3.1 cam1 segment ms | `125.11` |
| SAM3.1 cam2 segment ms | `121.78` |
| FFS runner init ms | `n/a` |
| FFS first run ms | `n/a` |
| session init + prompt add ms | `2.78` |
| SAM3.1 release cleanup ms | `302.42` |
| time to first complete group s | `20.35` |
| time to first rendered group s | `20.36` |

## GPU Sampling

- backend requested: `nvml`
- backend used: `unavailable`
- device index: `0`
- interval s: `0.500`
- samples after warmup: `474`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `gpu_util_pct` | `43.00` | `100.00` | `100.00` | `100.00` |
| `memory_util_pct` | `11.00` | `70.70` | `72.00` | `74.00` |
| `memory_used_mb` | `3648.38` | `16251.88` | `16293.73` | `16373.06` |
| `power_w` | `161.77` | `421.45` | `424.25` | `437.74` |
| `sm_clock_mhz` | `2655.00` | `2670.00` | `2670.00` | `2670.00` |
| `mem_clock_mhz` | `10251.00` | `10251.00` | `10251.00` | `10501.00` |
| `temperature_c` | `59.00` | `84.00` | `86.00` | `90.00` |

## Throughput periods

| Event | median ms | p90 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| `capture_group_period_ms` | `33.51` | `36.81` | `38.74` | `45.26` |
| `display_packet_publish_period_ms` | `66.64` | `78.46` | `358.54` | `2035.14` |
| `edgetam_stage_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_stage_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `filter_output_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `fusion_publish_period_ms` | `65.26` | `73.01` | `75.39` | `83.19` |
| `gpu_owner_publish_period_ms` | `64.93` | `70.69` | `72.40` | `81.26` |
| `raw_fusion_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_period_ms` | `66.91` | `243.55` | `446.36` | `1843.04` |
| `stage_join_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `6.92` | `25.56` | `28.66` | `52.17` |
| `edgetam_model_ms` | `15.82` | `25.34` | `27.16` | `38.34` |
| `edgetam_preprocess_ms` | `0.65` | `1.11` | `1.29` | `3.90` |
| `edgetam_prompt_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_postprocess_ms` | `0.02` | `0.04` | `0.04` | `6.21` |
| `edgetam_mask_resize_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_mask_threshold_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_mask_to_cpu_ms` | `0.40` | `0.52` | `0.60` | `7.68` |
| `edgetam_total_ms` | `16.39` | `26.22` | `27.94` | `39.04` |
| `ffs_cycle_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_batch_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_model_ms` | `1.27` | `1.62` | `1.92` | `43.27` |
| `edgetam_batch_vision_total_ms` | `8.02` | `10.64` | `11.52` | `61.13` |
| `edgetam_batch_vision_preprocess_ms` | `1.96` | `3.34` | `3.86` | `11.70` |
| `edgetam_cam0_model_ms` | `23.76` | `28.04` | `29.50` | `38.34` |
| `edgetam_cam1_model_ms` | `15.30` | `18.99` | `20.91` | `26.27` |
| `edgetam_cam2_model_ms` | `13.80` | `15.80` | `16.40` | `21.75` |
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
| `gpu_owner_total_ms` | `64.90` | `70.66` | `72.37` | `119.17` |
| `gpu_owner_ffs_cycle_ms` | `0.27` | `0.49` | `0.78` | `1.86` |
| `gpu_owner_edgetam_cycle_ms` | `64.54` | `70.30` | `72.08` | `117.85` |
| `raw_fusion_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `fusion_total_ms` | `22.69` | `25.44` | `26.47` | `31.52` |
| `filter_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `filter_input_age_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `object_enhanced_pt_ms` | `3.64` | `4.30` | `4.56` | `6.26` |
| `controller_pt_filter_ms` | `9.16` | `10.18` | `10.55` | `13.63` |
| `render_total_ms` | `1.66` | `1.96` | `2.08` | `6.44` |
| `render_queue_wait_ms` | `9.17` | `9.68` | `11.63` | `691.76` |
| `render_gpu_to_cpu_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_combine_ms` | `0.08` | `0.14` | `0.18` | `1.20` |
| `render_cpu_format_ms` | `0.27` | `0.39` | `0.45` | `1.47` |
| `render_open3d_points_update_ms` | `0.08` | `0.11` | `0.13` | `1.19` |
| `render_open3d_colors_update_ms` | `0.09` | `0.17` | `0.18` | `0.57` |
| `render_open3d_update_geometry_ms` | `1.32` | `1.55` | `1.62` | `6.11` |
| `render_poll_events_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_update_renderer_ms` | `0.02` | `0.02` | `0.03` | `0.05` |
| `open3d_object_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_controller_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `2293` | `6.26` | `24000` | `5643` |
| `1393` | `5.90` | `24000` | `5693` |
| `3328` | `5.79` | `24000` | `5659` |
| `1820` | `5.72` | `24000` | `5648` |
| `0` | `5.65` | `24000` | `5464` |
| `372` | `5.58` | `24000` | `5637` |
| `235` | `5.31` | `24000` | `5657` |
| `2544` | `5.25` | `24000` | `5654` |
| `512` | `5.21` | `24000` | `5597` |
| `791` | `5.19` | `24000` | `5667` |
