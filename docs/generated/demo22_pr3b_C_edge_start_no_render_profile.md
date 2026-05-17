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
- render FPS after warmup: `0.00`
- Demo 2.2 PASS threshold: `14.40 FPS`
- Demo 2.2 result: `FAIL`
- raw fusion FPS after warmup: `5.61`
- filter output FPS after warmup: `5.61`
- fusion FPS after warmup: `5.61`
- stage period p50 after warmup: `166.91 ms`
- display packet period p50 after warmup: `167.32 ms`
- groups after warmup: `1213`
- complete fused groups after warmup: `489`
- rendered groups after warmup: `0`
- complete group ratio after warmup: `0.403`
- stage drop count after warmup: `21`
- raw fused pending replacements total: `0`
- render buffer dropped total: `628`
- target deficit: `15.00`
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
| parallel init max wait ms | `8118.53` |
| camera startup ms | `6031.44` |
| EdgeTAM model load ms | `2906.28` |
| EdgeTAM compile wrap ms | `0.01` |
| EdgeTAM compile prewarm ms | `1189.32` |
| EdgeTAM warmup/first forward ms | `86.48` |
| SAM3.1 model load ms | `9426.27` |
| SAM3.1 cam0 segment ms | `1163.08` |
| SAM3.1 cam1 segment ms | `227.74` |
| SAM3.1 cam2 segment ms | `214.99` |
| FFS runner init ms | `5034.68` |
| FFS first run ms | `1126.25` |
| session init + prompt add ms | `5.30` |
| SAM3.1 release cleanup ms | `257.22` |
| time to first complete group s | `17.70` |
| time to first rendered group s | `n/a` |

## GPU Sampling

- backend requested: `nvml`
- backend used: `nvml`
- device index: `0`
- interval s: `0.500`
- samples after warmup: `176`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `gpu_util_pct` | `51.00` | `61.00` | `64.00` | `69.00` |
| `memory_util_pct` | `15.50` | `22.00` | `23.25` | `29.00` |
| `memory_used_mb` | `11662.10` | `15802.10` | `16324.10` | `16535.10` |
| `power_w` | `135.28` | `167.59` | `189.39` | `245.64` |
| `sm_clock_mhz` | `180.00` | `180.00` | `180.00` | `180.00` |
| `mem_clock_mhz` | `14001.00` | `14001.00` | `14001.00` | `14001.00` |
| `temperature_c` | `69.00` | `71.50` | `72.25` | `74.00` |

## Throughput periods

| Event | median ms | p90 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| `capture_group_period_ms` | `66.74` | `84.24` | `92.22` | `478.88` |
| `display_packet_publish_period_ms` | `167.32` | `219.67` | `243.23` | `448.31` |
| `edgetam_stage_publish_period_ms` | `166.61` | `222.36` | `244.02` | `397.60` |
| `ffs_stage_publish_period_ms` | `167.44` | `221.24` | `242.75` | `458.28` |
| `filter_output_publish_period_ms` | `167.32` | `219.67` | `243.22` | `448.26` |
| `fusion_publish_period_ms` | `167.32` | `219.67` | `243.22` | `448.26` |
| `gpu_owner_publish_period_ms` | `166.91` | `221.55` | `243.60` | `402.01` |
| `raw_fusion_publish_period_ms` | `166.92` | `221.54` | `243.60` | `401.96` |
| `render_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `stage_join_publish_period_ms` | `166.91` | `221.54` | `243.61` | `402.01` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `37.38` | `50.36` | `51.75` | `65.97` |
| `edgetam_model_ms` | `31.50` | `46.15` | `51.87` | `72.12` |
| `edgetam_preprocess_ms` | `1.58` | `2.00` | `2.11` | `2.98` |
| `edgetam_prompt_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_postprocess_ms` | `0.05` | `0.08` | `0.10` | `0.31` |
| `edgetam_mask_resize_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_mask_threshold_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_mask_to_cpu_ms` | `0.86` | `1.06` | `1.14` | `9.58` |
| `edgetam_total_ms` | `32.54` | `47.19` | `53.33` | `73.26` |
| `ffs_cycle_ms` | `82.28` | `94.29` | `100.60` | `296.12` |
| `ffs_batch_ms` | `59.05` | `63.05` | `65.45` | `257.79` |
| `ffs_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_model_ms` | `56.37` | `66.96` | `71.21` | `278.66` |
| `edgetam_batch_vision_total_ms` | `67.20` | `77.86` | `82.21` | `289.09` |
| `edgetam_batch_vision_preprocess_ms` | `4.73` | `5.99` | `6.34` | `8.94` |
| `edgetam_cam0_model_ms` | `34.62` | `48.12` | `53.60` | `72.12` |
| `edgetam_cam1_model_ms` | `29.33` | `44.09` | `49.55` | `66.78` |
| `edgetam_cam2_model_ms` | `28.82` | `43.36` | `51.46` | `67.30` |
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
| `ffs_stage_ms` | `2.54` | `3.53` | `3.85` | `9.45` |
| `ffs_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam0_stage_ms` | `2.54` | `3.53` | `3.85` | `9.45` |
| `ffs_cam1_stage_ms` | `2.54` | `3.53` | `3.85` | `9.45` |
| `ffs_cam2_stage_ms` | `2.54` | `3.53` | `3.85` | `9.45` |
| `ffs_cam0_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam1_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam2_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `gpu_owner_total_ms` | `166.59` | `221.46` | `243.07` | `393.91` |
| `gpu_owner_ffs_cycle_ms` | `82.26` | `94.29` | `100.65` | `296.12` |
| `gpu_owner_edgetam_cycle_ms` | `166.59` | `221.46` | `243.07` | `393.91` |
| `raw_fusion_total_ms` | `14.16` | `16.28` | `17.21` | `22.26` |
| `fusion_total_ms` | `55.81` | `63.98` | `67.90` | `262.29` |
| `filter_total_ms` | `41.48` | `48.90` | `51.80` | `248.86` |
| `filter_input_age_ms` | `42.07` | `49.38` | `52.33` | `249.21` |
| `object_enhanced_pt_ms` | `33.71` | `40.38` | `43.54` | `241.31` |
| `controller_pt_filter_ms` | `7.57` | `9.12` | `9.69` | `11.59` |
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
| `207` | `249.91` | `50373` | `14170` |
| `268` | `242.82` | `50329` | `14083` |
| `1340` | `241.31` | `50394` | `14135` |
| `1425` | `240.50` | `50434` | `14096` |
| `464` | `238.20` | `50380` | `14133` |
| `396` | `237.34` | `50379` | `13999` |
| `538` | `237.16` | `50434` | `14092` |
| `1189` | `237.08` | `50343` | `14055` |
| `1496` | `236.78` | `50355` | `14102` |
| `968` | `234.84` | `50424` | `14039` |
