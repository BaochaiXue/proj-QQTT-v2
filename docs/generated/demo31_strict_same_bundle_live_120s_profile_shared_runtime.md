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
- render FPS after warmup: `6.00`
- raw fusion FPS after warmup: `0.00`
- filter output FPS after warmup: `0.00`
- fusion FPS after warmup: `6.00`
- stage period p50 after warmup: `112.13 ms`
- display packet period p50 after warmup: `154.69 ms`
- groups after warmup: `3000`
- complete fused groups after warmup: `675`
- rendered groups after warmup: `675`
- complete group ratio after warmup: `0.225`
- stage drop count after warmup: `0`
- raw fused pending replacements total: `0`
- render buffer dropped total: `0`
- target deficit: `24.00`
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
| parallel init max wait ms | `0.00` |
| camera startup ms | `10970.66` |
| EdgeTAM model load ms | `517.64` |
| EdgeTAM compile wrap ms | `589.67` |
| EdgeTAM compile prewarm ms | `n/a` |
| EdgeTAM warmup/first forward ms | `113.27` |
| SAM3.1 model load ms | `6779.26` |
| SAM3.1 cam0 segment ms | `525.62` |
| SAM3.1 cam1 segment ms | `121.65` |
| SAM3.1 cam2 segment ms | `117.44` |
| FFS runner init ms | `n/a` |
| FFS first run ms | `n/a` |
| session init + prompt add ms | `2.21` |
| SAM3.1 release cleanup ms | `306.96` |
| time to first complete group s | `19.88` |
| time to first rendered group s | `19.90` |

## GPU Sampling

- backend requested: `nvml`
- backend used: `unavailable`
- device index: `0`
- interval s: `0.500`
- samples after warmup: `451`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `gpu_util_pct` | `40.00` | `100.00` | `100.00` | `100.00` |
| `memory_util_pct` | `11.00` | `77.00` | `78.00` | `79.00` |
| `memory_used_mb` | `2976.12` | `8088.12` | `8109.12` | `8185.94` |
| `power_w` | `134.17` | `379.29` | `383.17` | `388.85` |
| `sm_clock_mhz` | `2655.00` | `2670.00` | `2670.00` | `2685.00` |
| `mem_clock_mhz` | `10251.00` | `10251.00` | `10251.00` | `10501.00` |
| `temperature_c` | `52.00` | `82.00` | `83.00` | `85.00` |

## Throughput periods

| Event | median ms | p90 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| `capture_group_period_ms` | `33.43` | `50.59` | `55.53` | `353.63` |
| `display_packet_publish_period_ms` | `154.69` | `166.53` | `267.77` | `464.96` |
| `edgetam_stage_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_stage_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `filter_output_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `fusion_publish_period_ms` | `154.59` | `166.33` | `267.30` | `465.17` |
| `gpu_owner_publish_period_ms` | `112.13` | `123.58` | `126.78` | `487.09` |
| `raw_fusion_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_period_ms` | `155.31` | `168.95` | `267.31` | `480.27` |
| `stage_join_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `12.00` | `24.55` | `26.76` | `42.90` |
| `edgetam_model_ms` | `16.93` | `72.50` | `75.88` | `377.32` |
| `edgetam_preprocess_ms` | `0.58` | `0.92` | `2.35` | `107.02` |
| `edgetam_prompt_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_postprocess_ms` | `0.03` | `0.04` | `0.05` | `14.78` |
| `edgetam_mask_resize_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_mask_threshold_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_mask_to_cpu_ms` | `0.41` | `0.56` | `0.65` | `23.17` |
| `edgetam_total_ms` | `17.54` | `73.27` | `76.60` | `377.94` |
| `ffs_cycle_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_batch_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_model_ms` | `1.24` | `1.52` | `1.70` | `22.80` |
| `edgetam_batch_vision_total_ms` | `7.29` | `23.91` | `38.70` | `333.45` |
| `edgetam_batch_vision_preprocess_ms` | `1.74` | `2.76` | `7.03` | `321.05` |
| `edgetam_cam0_model_ms` | `19.08` | `73.39` | `76.78` | `369.50` |
| `edgetam_cam1_model_ms` | `15.57` | `71.83` | `74.95` | `372.49` |
| `edgetam_cam2_model_ms` | `14.97` | `71.63` | `75.53` | `377.32` |
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
| `gpu_owner_total_ms` | `112.07` | `123.54` | `126.75` | `487.05` |
| `gpu_owner_ffs_cycle_ms` | `0.31` | `0.86` | `1.64` | `22.38` |
| `gpu_owner_edgetam_cycle_ms` | `111.62` | `123.09` | `126.17` | `486.68` |
| `raw_fusion_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `fusion_total_ms` | `76.95` | `83.64` | `86.60` | `380.92` |
| `filter_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `filter_input_age_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `object_enhanced_pt_ms` | `39.79` | `43.67` | `45.66` | `341.20` |
| `controller_pt_filter_ms` | `27.22` | `30.31` | `31.73` | `332.69` |
| `render_total_ms` | `1.71` | `1.99` | `2.15` | `3.83` |
| `render_queue_wait_ms` | `14.97` | `16.44` | `18.08` | `30.92` |
| `render_gpu_to_cpu_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_combine_ms` | `0.09` | `0.15` | `0.20` | `1.85` |
| `render_cpu_format_ms` | `0.27` | `0.38` | `0.48` | `2.07` |
| `render_open3d_points_update_ms` | `0.08` | `0.11` | `0.12` | `0.43` |
| `render_open3d_colors_update_ms` | `0.08` | `0.16` | `0.18` | `1.76` |
| `render_open3d_update_geometry_ms` | `1.36` | `1.60` | `1.67` | `3.51` |
| `render_poll_events_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_update_renderer_ms` | `0.02` | `0.02` | `0.03` | `0.06` |
| `open3d_object_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_controller_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `2508` | `341.20` | `24000` | `8481` |
| `2388` | `333.91` | `24000` | `8451` |
| `2623` | `333.88` | `24000` | `8424` |
| `2740` | `333.12` | `24000` | `8465` |
| `1457` | `331.83` | `24000` | `8466` |
| `1572` | `330.65` | `24000` | `8438` |
| `1802` | `329.07` | `24000` | `8474` |
| `2157` | `327.39` | `24000` | `8449` |
| `1342` | `325.30` | `24000` | `8509` |
| `1920` | `323.03` | `24000` | `8413` |
