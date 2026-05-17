# Demo 2.2 performance profile

- preset: `demo2.2-async-filter-5fps`
- canonical preset: `demo2.2-async-filter-5fps`
- target FPS: `15.00`
- capture group target FPS: `15.00`
- compile mode: `vision-reduce-overhead`
- dtype: `bfloat16`
- EdgeTAM input path: `pil`
- mask postprocess: `hf`
- render backend: `legacy-inplace`
- render latest-only: `True`
- render copy mode: `sync-cpu`
- render FPS after warmup: `5.21`
- Demo 2.2 PASS threshold: `14.40 FPS`
- Demo 2.2 result: `FAIL`
- raw fusion FPS after warmup: `5.21`
- filter output FPS after warmup: `5.21`
- fusion FPS after warmup: `5.21`
- stage period p50 after warmup: `184.81 ms`
- display packet period p50 after warmup: `185.64 ms`
- groups after warmup: `1245`
- complete fused groups after warmup: `467`
- rendered groups after warmup: `467`
- complete group ratio after warmup: `0.375`
- stage drop count after warmup: `22`
- raw fused pending replacements total: `0`
- render buffer dropped total: `0`
- target deficit: `9.79`
- bottleneck class: `upstream_supply`
- GPU pipeline: `overlapped-stages`
- single-owner order: `cross_group_overlap`
- filter scheduler: `async`
- render filtered only: `True`
- pin memory mode: `off`
- FFS input staging: `pinned`
- H2D stream mode: `default`

## Init Profile

| Stage | value |
| --- | ---: |
| parallel init max wait ms | `4419.37` |
| camera startup ms | `7690.23` |
| EdgeTAM model load ms | `1697.20` |
| EdgeTAM compile wrap ms | `0.01` |
| EdgeTAM compile prewarm ms | `1523.26` |
| EdgeTAM warmup/first forward ms | `77.78` |
| SAM3.1 model load ms | `10687.67` |
| SAM3.1 cam0 segment ms | `427.49` |
| SAM3.1 cam1 segment ms | `188.35` |
| SAM3.1 cam2 segment ms | `190.44` |
| FFS runner init ms | `6980.31` |
| FFS first run ms | `1155.15` |
| session init + prompt add ms | `5.21` |
| SAM3.1 release cleanup ms | `298.16` |
| time to first complete group s | `18.66` |
| time to first rendered group s | `18.66` |

## GPU Sampling

- backend requested: `nvml`
- backend used: `nvml`
- device index: `0`
- interval s: `0.500`
- samples after warmup: `177`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `gpu_util_pct` | `48.00` | `56.00` | `59.00` | `64.00` |
| `memory_util_pct` | `15.00` | `21.00` | `23.00` | `28.00` |
| `memory_used_mb` | `11273.48` | `15201.08` | `15620.28` | `15833.48` |
| `power_w` | `135.08` | `159.34` | `188.03` | `246.07` |
| `sm_clock_mhz` | `180.00` | `180.00` | `180.00` | `180.00` |
| `mem_clock_mhz` | `14001.00` | `14001.00` | `14001.00` | `14001.00` |
| `temperature_c` | `70.00` | `74.00` | `74.00` | `75.00` |

## Throughput periods

| Event | median ms | p90 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| `capture_group_period_ms` | `66.78` | `86.43` | `96.30` | `389.60` |
| `display_packet_publish_period_ms` | `185.64` | `209.75` | `222.01` | `423.02` |
| `edgetam_stage_publish_period_ms` | `184.67` | `211.76` | `220.39` | `421.53` |
| `ffs_stage_publish_period_ms` | `185.11` | `211.42` | `225.60` | `425.01` |
| `filter_output_publish_period_ms` | `185.64` | `209.77` | `222.01` | `422.95` |
| `fusion_publish_period_ms` | `185.64` | `209.76` | `222.01` | `422.95` |
| `gpu_owner_publish_period_ms` | `184.81` | `210.69` | `221.73` | `421.68` |
| `raw_fusion_publish_period_ms` | `184.71` | `210.67` | `221.73` | `421.68` |
| `render_period_ms` | `185.01` | `211.08` | `222.44` | `422.89` |
| `stage_join_publish_period_ms` | `184.81` | `210.69` | `221.72` | `421.68` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `17.21` | `47.10` | `55.93` | `65.63` |
| `edgetam_model_ms` | `35.34` | `46.68` | `49.68` | `65.59` |
| `edgetam_preprocess_ms` | `1.61` | `2.19` | `2.37` | `3.08` |
| `edgetam_prompt_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_postprocess_ms` | `0.05` | `0.09` | `0.12` | `0.51` |
| `edgetam_mask_resize_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_mask_threshold_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_mask_to_cpu_ms` | `0.90` | `1.14` | `1.28` | `2.93` |
| `edgetam_total_ms` | `36.53` | `47.83` | `51.07` | `66.93` |
| `ffs_cycle_ms` | `90.32` | `100.18` | `105.40` | `311.18` |
| `ffs_batch_ms` | `61.00` | `66.63` | `70.94` | `275.34` |
| `ffs_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_model_ms` | `62.79` | `71.57` | `74.24` | `276.59` |
| `edgetam_batch_vision_total_ms` | `74.08` | `84.08` | `87.02` | `288.58` |
| `edgetam_batch_vision_preprocess_ms` | `4.82` | `6.56` | `7.11` | `9.24` |
| `edgetam_cam0_model_ms` | `40.79` | `49.42` | `52.64` | `65.59` |
| `edgetam_cam1_model_ms` | `31.23` | `43.57` | `46.10` | `61.39` |
| `edgetam_cam2_model_ms` | `32.30` | `43.62` | `47.23` | `57.50` |
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
| `ffs_stage_ms` | `2.62` | `3.90` | `4.19` | `8.36` |
| `ffs_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam0_stage_ms` | `2.62` | `3.88` | `4.18` | `8.36` |
| `ffs_cam1_stage_ms` | `2.62` | `3.88` | `4.18` | `8.36` |
| `ffs_cam2_stage_ms` | `2.62` | `3.88` | `4.18` | `8.36` |
| `ffs_cam0_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam1_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam2_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `gpu_owner_total_ms` | `184.62` | `211.38` | `219.95` | `421.51` |
| `gpu_owner_ffs_cycle_ms` | `90.30` | `100.19` | `105.43` | `311.18` |
| `gpu_owner_edgetam_cycle_ms` | `184.62` | `211.38` | `219.95` | `421.51` |
| `raw_fusion_total_ms` | `15.44` | `18.19` | `19.19` | `22.74` |
| `fusion_total_ms` | `62.57` | `70.04` | `74.12` | `281.06` |
| `filter_total_ms` | `47.00` | `53.36` | `57.01` | `264.03` |
| `filter_input_age_ms` | `47.63` | `53.69` | `57.48` | `264.80` |
| `object_enhanced_pt_ms` | `38.79` | `44.64` | `47.53` | `256.25` |
| `controller_pt_filter_ms` | `7.88` | `9.23` | `9.76` | `12.79` |
| `render_total_ms` | `2.06` | `2.58` | `2.80` | `4.28` |
| `render_queue_wait_ms` | `9.50` | `10.00` | `10.11` | `10.91` |
| `render_gpu_to_cpu_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_combine_ms` | `0.19` | `0.40` | `0.48` | `1.18` |
| `render_cpu_format_ms` | `0.46` | `0.76` | `0.88` | `1.86` |
| `render_open3d_points_update_ms` | `0.12` | `0.23` | `0.29` | `1.42` |
| `render_open3d_colors_update_ms` | `0.11` | `0.21` | `0.25` | `0.70` |
| `render_open3d_update_geometry_ms` | `1.48` | `1.90` | `2.11` | `3.19` |
| `render_poll_events_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_update_renderer_ms` | `0.03` | `0.03` | `0.04` | `0.17` |
| `open3d_object_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_controller_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `812` | `256.25` | `50587` | `14282` |
| `1452` | `246.99` | `50630` | `14210` |
| `892` | `246.57` | `50636` | `14178` |
| `1217` | `246.02` | `50605` | `14174` |
| `1136` | `244.86` | `50560` | `14339` |
| `974` | `239.93` | `50621` | `14227` |
| `1371` | `239.35` | `50608` | `14296` |
| `1291` | `236.95` | `50639` | `14363` |
| `733` | `236.58` | `50579` | `14246` |
| `451` | `235.94` | `50565` | `14152` |
