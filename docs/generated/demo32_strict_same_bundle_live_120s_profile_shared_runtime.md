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
- render FPS after warmup: `5.93`
- raw fusion FPS after warmup: `5.93`
- filter output FPS after warmup: `5.93`
- fusion FPS after warmup: `5.93`
- stage period p50 after warmup: `106.57 ms`
- display packet period p50 after warmup: `139.83 ms`
- groups after warmup: `3071`
- complete fused groups after warmup: `625`
- rendered groups after warmup: `624`
- complete group ratio after warmup: `0.204`
- stage drop count after warmup: `3`
- raw fused pending replacements total: `0`
- render buffer dropped total: `0`
- target deficit: `24.07`
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
| camera startup ms | `10870.93` |
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
| time to first complete group s | `29.00` |
| time to first rendered group s | `29.05` |

## GPU Sampling

- backend requested: `nvml`
- backend used: `unavailable`
- device index: `0`
- interval s: `0.500`
- samples after warmup: `485`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `gpu_util_pct` | `86.00` | `99.00` | `100.00` | `100.00` |
| `memory_util_pct` | `46.00` | `95.60` | `99.80` | `100.00` |
| `memory_used_mb` | `6609.00` | `9131.89` | `9165.77` | `9192.06` |
| `power_w` | `296.62` | `325.42` | `346.39` | `363.19` |
| `sm_clock_mhz` | `2655.00` | `2655.00` | `2655.00` | `2670.00` |
| `mem_clock_mhz` | `10251.00` | `10251.00` | `10251.00` | `10501.00` |
| `temperature_c` | `67.00` | `78.00` | `80.00` | `83.00` |

## Throughput periods

| Event | median ms | p90 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| `capture_group_period_ms` | `33.39` | `54.56` | `60.32` | `240.95` |
| `display_packet_publish_period_ms` | `139.83` | `286.74` | `313.19` | `574.56` |
| `edgetam_stage_publish_period_ms` | `68.99` | `95.57` | `104.11` | `345.25` |
| `ffs_stage_publish_period_ms` | `59.67` | `97.29` | `104.43` | `350.47` |
| `filter_output_publish_period_ms` | `139.10` | `283.77` | `310.86` | `565.03` |
| `fusion_publish_period_ms` | `139.10` | `283.76` | `310.86` | `565.03` |
| `gpu_owner_publish_period_ms` | `106.57` | `220.76` | `287.45` | `627.41` |
| `raw_fusion_publish_period_ms` | `138.65` | `279.46` | `296.84` | `565.55` |
| `render_period_ms` | `140.38` | `282.86` | `315.57` | `584.43` |
| `stage_join_publish_period_ms` | `106.57` | `220.76` | `287.44` | `627.41` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `19.00` | `25.36` | `27.13` | `65.67` |
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
| `gpu_owner_total_ms` | `64.41` | `69.87` | `71.49` | `89.92` |
| `gpu_owner_ffs_cycle_ms` | `61.56` | `64.43` | `65.30` | `89.92` |
| `gpu_owner_edgetam_cycle_ms` | `64.03` | `69.81` | `71.31` | `78.43` |
| `raw_fusion_total_ms` | `9.09` | `13.74` | `15.36` | `24.82` |
| `fusion_total_ms` | `84.93` | `94.96` | `242.93` | `278.76` |
| `filter_total_ms` | `74.98` | `84.20` | `230.85` | `266.96` |
| `filter_input_age_ms` | `75.00` | `84.22` | `230.88` | `266.99` |
| `object_enhanced_pt_ms` | `45.10` | `49.99` | `53.16` | `226.55` |
| `controller_pt_filter_ms` | `29.86` | `35.55` | `38.22` | `218.87` |
| `render_total_ms` | `3.89` | `5.17` | `5.72` | `10.48` |
| `render_queue_wait_ms` | `40.25` | `50.03` | `53.32` | `67.80` |
| `render_gpu_to_cpu_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_combine_ms` | `0.13` | `0.33` | `0.48` | `4.75` |
| `render_cpu_format_ms` | `0.39` | `0.75` | `1.06` | `5.08` |
| `render_open3d_points_update_ms` | `0.10` | `0.17` | `0.26` | `1.61` |
| `render_open3d_colors_update_ms` | `0.12` | `0.30` | `0.36` | `1.95` |
| `render_open3d_update_geometry_ms` | `3.34` | `4.41` | `4.87` | `9.72` |
| `render_poll_events_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_update_renderer_ms` | `0.02` | `0.03` | `0.03` | `0.05` |
| `open3d_object_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_controller_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `3167` | `226.55` | `24000` | `8760` |
| `2469` | `224.27` | `24000` | `8875` |
| `2631` | `223.58` | `24000` | `8763` |
| `2382` | `220.29` | `24000` | `8832` |
| `2306` | `218.77` | `24000` | `8796` |
| `1438` | `216.38` | `24000` | `8836` |
| `565` | `211.90` | `24000` | `8849` |
| `1374` | `211.78` | `24000` | `8838` |
| `1503` | `211.58` | `24000` | `8771` |
| `1857` | `211.48` | `24000` | `8787` |
