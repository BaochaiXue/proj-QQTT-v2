# Demo 2.1 performance profile

- preset: `demo2.3-dual4090-maxfps`
- canonical preset: `demo2.3-dual4090-maxfps`
- target FPS: `15.00`
- capture group target FPS: `15.00`
- compile mode: `vision-reduce-overhead`
- dtype: `bfloat16`
- EdgeTAM input path: `pil`
- mask postprocess: `hf`
- render backend: `legacy-inplace`
- render latest-only: `True`
- render copy mode: `sync-cpu`
- render FPS after warmup: `14.08`
- raw fusion FPS after warmup: `14.08`
- filter output FPS after warmup: `14.08`
- fusion FPS after warmup: `14.08`
- stage period p50 after warmup: `66.47 ms`
- display packet period p50 after warmup: `66.09 ms`
- groups after warmup: `889`
- complete fused groups after warmup: `888`
- rendered groups after warmup: `887`
- complete group ratio after warmup: `0.999`
- stage drop count after warmup: `0`
- raw fused pending replacements total: `0`
- render buffer dropped total: `0`
- target deficit: `0.92`
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
| camera startup ms | `11541.50` |
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
| time to first complete group s | `31.10` |
| time to first rendered group s | `31.11` |

## GPU Sampling

- backend requested: `nvml`
- backend used: `unavailable`
- device index: `0`
- interval s: `0.500`
- samples after warmup: `388`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `gpu_util_pct` | `36.00` | `55.00` | `58.00` | `68.00` |
| `memory_util_pct` | `14.00` | `36.00` | `41.00` | `50.00` |
| `memory_used_mb` | `3671.88` | `19667.90` | `22325.65` | `24512.06` |
| `power_w` | `156.48` | `207.60` | `216.06` | `224.19` |
| `sm_clock_mhz` | `2655.00` | `2670.00` | `2670.00` | `2685.00` |
| `mem_clock_mhz` | `10251.00` | `10251.00` | `10251.00` | `10501.00` |
| `temperature_c` | `55.00` | `59.00` | `59.00` | `60.00` |

## Throughput periods

| Event | median ms | p90 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| `capture_group_period_ms` | `66.86` | `69.32` | `74.19` | `196.39` |
| `display_packet_publish_period_ms` | `66.09` | `73.68` | `76.87` | `224.35` |
| `edgetam_stage_publish_period_ms` | `66.72` | `76.31` | `79.21` | `207.82` |
| `ffs_stage_publish_period_ms` | `66.57` | `75.30` | `77.80` | `218.93` |
| `filter_output_publish_period_ms` | `66.09` | `73.68` | `76.88` | `224.35` |
| `fusion_publish_period_ms` | `66.09` | `73.68` | `76.88` | `224.35` |
| `gpu_owner_publish_period_ms` | `66.47` | `72.01` | `75.05` | `209.10` |
| `raw_fusion_publish_period_ms` | `66.46` | `72.03` | `75.04` | `209.11` |
| `render_period_ms` | `66.52` | `76.12` | `81.18` | `223.47` |
| `stage_join_publish_period_ms` | `66.47` | `72.01` | `75.05` | `209.10` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `10.12` | `24.52` | `26.92` | `35.88` |
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
| `gpu_owner_total_ms` | `53.64` | `57.24` | `58.22` | `89.52` |
| `gpu_owner_ffs_cycle_ms` | `39.27` | `41.91` | `43.42` | `58.36` |
| `gpu_owner_edgetam_cycle_ms` | `53.64` | `57.24` | `58.22` | `89.52` |
| `raw_fusion_total_ms` | `12.44` | `16.84` | `18.63` | `29.39` |
| `fusion_total_ms` | `54.77` | `62.37` | `70.74` | `214.86` |
| `filter_total_ms` | `41.98` | `46.99` | `51.86` | `200.89` |
| `filter_input_age_ms` | `42.59` | `47.63` | `52.26` | `201.09` |
| `object_enhanced_pt_ms` | `28.62` | `31.97` | `33.51` | `182.27` |
| `controller_pt_filter_ms` | `13.11` | `16.32` | `18.45` | `22.55` |
| `render_total_ms` | `2.18` | `3.37` | `3.99` | `10.49` |
| `render_queue_wait_ms` | `8.81` | `9.89` | `10.64` | `12.62` |
| `render_gpu_to_cpu_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_combine_ms` | `0.09` | `0.17` | `0.25` | `3.15` |
| `render_cpu_format_ms` | `0.28` | `0.48` | `0.72` | `3.28` |
| `render_open3d_points_update_ms` | `0.08` | `0.11` | `0.14` | `2.87` |
| `render_open3d_colors_update_ms` | `0.09` | `0.20` | `0.23` | `3.01` |
| `render_open3d_update_geometry_ms` | `1.81` | `2.77` | `3.35` | `10.23` |
| `render_poll_events_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_update_renderer_ms` | `0.02` | `0.02` | `0.03` | `0.08` |
| `open3d_object_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_controller_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `931` | `182.27` | `48040` | `11670` |
| `300` | `177.68` | `48092` | `11706` |
| `959` | `176.81` | `48103` | `11591` |
| `821` | `173.74` | `48191` | `11723` |
| `409` | `173.60` | `48132` | `11577` |
| `463` | `173.54` | `48161` | `11624` |
| `355` | `172.98` | `48094` | `11658` |
| `1153` | `172.64` | `48097` | `11735` |
| `987` | `172.12` | `48142` | `11589` |
| `849` | `171.82` | `48119` | `11650` |
