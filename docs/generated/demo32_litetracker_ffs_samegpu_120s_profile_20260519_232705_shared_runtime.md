# Demo 3.2 performance profile

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
- render FPS after warmup: `6.85`
- raw fusion FPS after warmup: `15.56`
- filter output FPS after warmup: `15.56`
- fusion FPS after warmup: `15.56`
- stage period p50 after warmup: `62.81 ms`
- display packet period p50 after warmup: `145.02 ms`
- groups after warmup: `3372`
- complete fused groups after warmup: `1664`
- rendered groups after warmup: `718`
- complete group ratio after warmup: `0.493`
- stage drop count after warmup: `1`
- raw fused pending replacements total: `0`
- render buffer dropped total: `0`
- target deficit: `23.15`
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
| camera startup ms | `10828.76` |
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
| time to first complete group s | `26.96` |
| time to first rendered group s | `29.07` |

## GPU Sampling

- backend requested: `nvml`
- backend used: `unavailable`
- device index: `0`
- interval s: `0.500`
- samples after warmup: `492`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `gpu_util_pct` | `88.00` | `94.00` | `96.00` | `100.00` |
| `memory_util_pct` | `48.00` | `82.00` | `84.00` | `87.00` |
| `memory_used_mb` | `6666.69` | `7739.91` | `7762.15` | `8926.69` |
| `power_w` | `307.31` | `328.62` | `331.67` | `337.90` |
| `sm_clock_mhz` | `2655.00` | `2655.00` | `2670.00` | `2685.00` |
| `mem_clock_mhz` | `10251.00` | `10251.00` | `10251.00` | `10501.00` |
| `temperature_c` | `67.00` | `77.00` | `79.00` | `81.00` |

## Throughput periods

| Event | median ms | p90 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| `capture_group_period_ms` | `33.47` | `36.54` | `38.02` | `196.04` |
| `display_packet_publish_period_ms` | `145.02` | `155.06` | `158.14` | `479.19` |
| `edgetam_stage_publish_period_ms` | `69.46` | `90.52` | `102.40` | `284.75` |
| `ffs_stage_publish_period_ms` | `64.30` | `71.40` | `72.89` | `217.09` |
| `filter_output_publish_period_ms` | `62.26` | `85.27` | `88.59` | `250.47` |
| `fusion_publish_period_ms` | `62.26` | `85.28` | `88.59` | `250.48` |
| `gpu_owner_publish_period_ms` | `62.81` | `84.43` | `87.21` | `245.58` |
| `raw_fusion_publish_period_ms` | `62.81` | `84.50` | `87.16` | `244.07` |
| `render_period_ms` | `145.18` | `157.55` | `161.54` | `483.38` |
| `stage_join_publish_period_ms` | `62.81` | `84.42` | `87.21` | `245.58` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `9.14` | `23.65` | `36.11` | `47.49` |
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
| `gpu_owner_total_ms` | `65.19` | `70.55` | `72.31` | `12986.78` |
| `gpu_owner_ffs_cycle_ms` | `61.03` | `64.68` | `65.54` | `85.88` |
| `gpu_owner_edgetam_cycle_ms` | `64.80` | `70.53` | `72.30` | `12986.78` |
| `raw_fusion_total_ms` | `8.41` | `11.21` | `12.36` | `168.64` |
| `fusion_total_ms` | `18.88` | `23.36` | `25.02` | `186.33` |
| `filter_total_ms` | `10.15` | `13.19` | `14.67` | `24.97` |
| `filter_input_age_ms` | `10.78` | `13.86` | `15.30` | `25.57` |
| `object_enhanced_pt_ms` | `4.20` | `5.50` | `6.14` | `10.66` |
| `controller_pt_filter_ms` | `5.68` | `8.47` | `9.89` | `20.89` |
| `render_total_ms` | `1.51` | `4.76` | `5.99` | `17.21` |
| `render_queue_wait_ms` | `207.32` | `256.96` | `267.53` | `2102.58` |
| `render_gpu_to_cpu_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_combine_ms` | `0.07` | `0.18` | `0.30` | `4.78` |
| `render_cpu_format_ms` | `0.21` | `0.42` | `0.58` | `5.19` |
| `render_open3d_points_update_ms` | `0.07` | `0.10` | `0.12` | `4.77` |
| `render_open3d_colors_update_ms` | `0.05` | `0.11` | `0.15` | `2.13` |
| `render_open3d_update_geometry_ms` | `1.20` | `4.35` | `5.47` | `16.60` |
| `render_poll_events_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_update_renderer_ms` | `0.02` | `0.02` | `0.03` | `0.62` |
| `open3d_object_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_controller_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `1252` | `10.66` | `24000` | `6267` |
| `684` | `9.10` | `24000` | `6280` |
| `960` | `8.91` | `24000` | `6221` |
| `2371` | `8.57` | `24000` | `6248` |
| `401` | `8.55` | `24000` | `6200` |
| `3144` | `8.34` | `24000` | `6243` |
| `1559` | `8.20` | `24000` | `6212` |
| `2883` | `7.73` | `24000` | `6275` |
| `1300` | `7.64` | `24000` | `6234` |
| `2261` | `7.62` | `24000` | `6289` |
