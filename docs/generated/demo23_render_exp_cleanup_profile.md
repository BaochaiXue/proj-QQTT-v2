# Demo 2.3 performance profile

- preset: `demo2.3-dual4090-maxfps`
- canonical preset: `demo2.3-dual4090-maxfps`
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
- render FPS after warmup: `12.19`
- raw fusion FPS after warmup: `12.20`
- filter output FPS after warmup: `12.20`
- fusion FPS after warmup: `12.20`
- stage period p50 after warmup: `80.60 ms`
- display packet period p50 after warmup: `80.22 ms`
- groups after warmup: `1248`
- complete fused groups after warmup: `403`
- rendered groups after warmup: `401`
- complete group ratio after warmup: `0.323`
- stage drop count after warmup: `0`
- raw fused pending replacements total: `0`
- render buffer dropped total: `1`
- target deficit: `17.81`
- bottleneck class: `upstream_supply`
- GPU pipeline: `dual-gpu-split`
- single-owner order: `dual_gpu_process_split`
- filter scheduler: `async`
- render filtered only: `True`
- pin memory mode: `off`
- FFS input staging: `pinned`
- H2D stream mode: `default`

## Init Profile

| Stage | value |
| --- | ---: |
| parallel init max wait ms | `n/a` |
| camera startup ms | `11082.10` |
| EdgeTAM model load ms | `n/a` |
| EdgeTAM compile wrap ms | `n/a` |
| EdgeTAM compile prewarm ms | `n/a` |
| EdgeTAM warmup/first forward ms | `n/a` |
| SAM3.1 model load ms | `n/a` |
| SAM3.1 cam0 segment ms | `n/a` |
| SAM3.1 cam1 segment ms | `n/a` |
| SAM3.1 cam2 segment ms | `n/a` |
| FFS runner init ms | `n/a` |
| FFS first run ms | `n/a` |
| session init + prompt add ms | `n/a` |
| SAM3.1 release cleanup ms | `n/a` |
| time to first complete group s | `26.19` |
| time to first rendered group s | `26.20` |

## GPU Sampling

- backend requested: `nvml`
- backend used: `unavailable`
- device index: `0`
- interval s: `0.500`
- samples after warmup: `263`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `gpu_util_pct` | `53.00` | `63.00` | `66.00` | `100.00` |
| `memory_util_pct` | `12.00` | `51.00` | `53.90` | `99.00` |
| `memory_used_mb` | `3671.88` | `6137.75` | `6138.00` | `8501.75` |
| `power_w` | `168.10` | `294.04` | `297.08` | `308.53` |
| `sm_clock_mhz` | `2655.00` | `2670.00` | `2670.00` | `2670.00` |
| `mem_clock_mhz` | `10251.00` | `10251.00` | `10251.00` | `10501.00` |
| `temperature_c` | `57.00` | `63.00` | `63.00` | `65.00` |

## Throughput periods

| Event | median ms | p90 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| `capture_group_period_ms` | `33.38` | `67.22` | `72.01` | `78.22` |
| `display_packet_publish_period_ms` | `80.22` | `93.46` | `100.80` | `136.39` |
| `edgetam_stage_publish_period_ms` | `69.76` | `87.39` | `110.53` | `160.05` |
| `ffs_stage_publish_period_ms` | `44.98` | `75.28` | `78.92` | `126.26` |
| `filter_output_publish_period_ms` | `80.22` | `93.45` | `100.79` | `136.39` |
| `fusion_publish_period_ms` | `80.22` | `93.45` | `100.79` | `136.39` |
| `gpu_owner_publish_period_ms` | `80.60` | `100.07` | `108.19` | `139.10` |
| `raw_fusion_publish_period_ms` | `80.59` | `100.06` | `108.18` | `139.08` |
| `render_period_ms` | `80.64` | `93.98` | `101.55` | `135.45` |
| `stage_join_publish_period_ms` | `80.60` | `100.06` | `108.19` | `139.10` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `4.72` | `13.63` | `26.21` | `54.92` |
| `edgetam_model_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_preprocess_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_prompt_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_postprocess_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_mask_resize_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_mask_threshold_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_mask_to_cpu_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cycle_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_batch_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_model_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_preprocess_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_cam0_model_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_cam1_model_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_cam2_model_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
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
| `gpu_owner_total_ms` | `54.77` | `58.97` | `59.79` | `85.83` |
| `gpu_owner_ffs_cycle_ms` | `39.69` | `42.66` | `43.32` | `47.44` |
| `gpu_owner_edgetam_cycle_ms` | `54.77` | `58.97` | `59.79` | `85.83` |
| `raw_fusion_total_ms` | `51.68` | `66.86` | `68.31` | `72.85` |
| `fusion_total_ms` | `122.20` | `141.00` | `143.28` | `152.44` |
| `filter_total_ms` | `73.01` | `77.79` | `79.16` | `86.30` |
| `filter_input_age_ms` | `74.03` | `85.65` | `90.12` | `133.83` |
| `object_enhanced_pt_ms` | `55.98` | `60.30` | `61.63` | `69.66` |
| `controller_pt_filter_ms` | `16.83` | `20.60` | `21.65` | `27.04` |
| `render_total_ms` | `1.84` | `4.29` | `6.25` | `16.96` |
| `render_queue_wait_ms` | `8.50` | `9.72` | `10.06` | `18.90` |
| `render_gpu_to_cpu_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_combine_ms` | `0.07` | `0.19` | `0.28` | `5.18` |
| `render_cpu_format_ms` | `0.24` | `0.49` | `0.74` | `10.46` |
| `render_open3d_points_update_ms` | `0.07` | `0.11` | `0.15` | `0.86` |
| `render_open3d_colors_update_ms` | `0.07` | `0.16` | `0.22` | `5.18` |
| `render_open3d_update_geometry_ms` | `1.28` | `2.37` | `4.31` | `10.06` |
| `render_poll_events_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_update_renderer_ms` | `0.02` | `0.03` | `0.03` | `0.07` |
| `open3d_object_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_controller_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `1107` | `69.66` | `50773` | `1613` |
| `632` | `68.58` | `49007` | `1326` |
| `1121` | `67.00` | `50718` | `1595` |
| `846` | `65.82` | `48971` | `1347` |
| `1213` | `64.43` | `50738` | `1584` |
| `504` | `63.63` | `48988` | `1329` |
| `640` | `63.54` | `49030` | `1305` |
| `1211` | `63.45` | `50770` | `1589` |
| `1188` | `63.36` | `50737` | `1604` |
| `722` | `63.20` | `48983` | `1323` |
