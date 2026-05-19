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
- render FPS after warmup: `11.42`
- raw fusion FPS after warmup: `11.42`
- filter output FPS after warmup: `11.43`
- fusion FPS after warmup: `11.43`
- stage period p50 after warmup: `85.26 ms`
- display packet period p50 after warmup: `84.37 ms`
- groups after warmup: `2035`
- complete fused groups after warmup: `719`
- rendered groups after warmup: `718`
- complete group ratio after warmup: `0.353`
- stage drop count after warmup: `0`
- raw fused pending replacements total: `0`
- render buffer dropped total: `0`
- target deficit: `18.58`
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
| camera startup ms | `11369.79` |
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
| time to first complete group s | `26.51` |
| time to first rendered group s | `26.52` |

## GPU Sampling

- backend requested: `nvml`
- backend used: `unavailable`
- device index: `0`
- interval s: `0.500`
- samples after warmup: `380`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `gpu_util_pct` | `55.00` | `65.00` | `66.00` | `98.00` |
| `memory_util_pct` | `14.00` | `52.00` | `53.00` | `69.00` |
| `memory_used_mb` | `3671.88` | `5961.56` | `5963.38` | `8336.88` |
| `power_w` | `171.63` | `291.60` | `294.64` | `300.58` |
| `sm_clock_mhz` | `2655.00` | `2670.00` | `2670.00` | `2670.00` |
| `mem_clock_mhz` | `10251.00` | `10251.00` | `10251.00` | `10501.00` |
| `temperature_c` | `60.00` | `63.00` | `64.00` | `66.00` |

## Throughput periods

| Event | median ms | p90 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| `capture_group_period_ms` | `33.34` | `71.85` | `75.52` | `113.81` |
| `display_packet_publish_period_ms` | `84.37` | `101.53` | `111.73` | `194.17` |
| `edgetam_stage_publish_period_ms` | `69.79` | `91.41` | `113.71` | `188.90` |
| `ffs_stage_publish_period_ms` | `45.19` | `80.00` | `84.23` | `156.27` |
| `filter_output_publish_period_ms` | `84.37` | `101.54` | `111.72` | `194.18` |
| `fusion_publish_period_ms` | `84.37` | `101.54` | `111.73` | `194.18` |
| `gpu_owner_publish_period_ms` | `85.26` | `108.89` | `118.91` | `245.32` |
| `raw_fusion_publish_period_ms` | `85.26` | `108.89` | `118.90` | `245.34` |
| `render_period_ms` | `84.34` | `102.15` | `112.32` | `209.82` |
| `stage_join_publish_period_ms` | `85.26` | `108.89` | `118.91` | `245.33` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `13.74` | `27.54` | `29.31` | `49.48` |
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
| `gpu_owner_total_ms` | `54.66` | `58.88` | `60.15` | `101.61` |
| `gpu_owner_ffs_cycle_ms` | `39.67` | `42.64` | `44.07` | `101.61` |
| `gpu_owner_edgetam_cycle_ms` | `54.66` | `58.79` | `60.02` | `97.70` |
| `raw_fusion_total_ms` | `52.77` | `70.66` | `72.94` | `138.37` |
| `fusion_total_ms` | `128.29` | `148.97` | `152.02` | `227.90` |
| `filter_total_ms` | `76.54` | `81.35` | `82.98` | `152.66` |
| `filter_input_age_ms` | `77.92` | `90.91` | `95.78` | `156.43` |
| `object_enhanced_pt_ms` | `58.59` | `62.25` | `64.15` | `133.73` |
| `controller_pt_filter_ms` | `17.74` | `21.14` | `21.99` | `40.05` |
| `render_total_ms` | `1.83` | `3.93` | `5.45` | `65.43` |
| `render_queue_wait_ms` | `8.48` | `9.52` | `9.91` | `17.09` |
| `render_gpu_to_cpu_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_combine_ms` | `0.08` | `0.16` | `0.22` | `3.14` |
| `render_cpu_format_ms` | `0.24` | `0.43` | `0.71` | `3.35` |
| `render_open3d_points_update_ms` | `0.07` | `0.11` | `0.13` | `3.16` |
| `render_open3d_colors_update_ms` | `0.07` | `0.15` | `0.20` | `2.67` |
| `render_open3d_update_geometry_ms` | `1.27` | `2.74` | `3.90` | `64.62` |
| `render_poll_events_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_update_renderer_ms` | `0.02` | `0.03` | `0.03` | `0.21` |
| `open3d_object_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_controller_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `1702` | `133.73` | `48986` | `1325` |
| `1699` | `124.69` | `48982` | `1328` |
| `1781` | `104.66` | `48965` | `1335` |
| `1694` | `93.02` | `48961` | `1303` |
| `1731` | `90.20` | `48900` | `1321` |
| `1754` | `89.48` | `48947` | `1326` |
| `1800` | `81.27` | `48924` | `1328` |
| `1704` | `78.56` | `48969` | `1321` |
| `900` | `77.66` | `48923` | `1341` |
| `1697` | `76.02` | `48946` | `1313` |
