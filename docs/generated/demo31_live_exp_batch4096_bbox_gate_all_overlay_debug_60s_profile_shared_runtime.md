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
- render FPS after warmup: `0.00`
- raw fusion FPS after warmup: `0.00`
- filter output FPS after warmup: `0.00`
- fusion FPS after warmup: `15.12`
- stage period p50 after warmup: `65.89 ms`
- display packet period p50 after warmup: `0.00 ms`
- groups after warmup: `293`
- complete fused groups after warmup: `148`
- rendered groups after warmup: `1`
- complete group ratio after warmup: `0.505`
- stage drop count after warmup: `0`
- raw fused pending replacements total: `0`
- render buffer dropped total: `0`
- target deficit: `30.00`
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
| camera startup ms | `11034.93` |
| EdgeTAM model load ms | `569.05` |
| EdgeTAM compile wrap ms | `599.97` |
| EdgeTAM compile prewarm ms | `n/a` |
| EdgeTAM warmup/first forward ms | `128.39` |
| SAM3.1 model load ms | `7993.86` |
| SAM3.1 cam0 segment ms | `587.10` |
| SAM3.1 cam1 segment ms | `122.62` |
| SAM3.1 cam2 segment ms | `123.39` |
| FFS runner init ms | `n/a` |
| FFS first run ms | `n/a` |
| session init + prompt add ms | `2.97` |
| SAM3.1 release cleanup ms | `308.86` |
| time to first complete group s | `20.01` |
| time to first rendered group s | `25.76` |

## GPU Sampling

- backend requested: `nvml`
- backend used: `unavailable`
- device index: `0`
- interval s: `0.500`
- samples after warmup: `54`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `gpu_util_pct` | `42.00` | `100.00` | `100.00` | `100.00` |
| `memory_util_pct` | `9.00` | `71.10` | `76.00` | `77.00` |
| `memory_used_mb` | `3548.69` | `23991.50` | `24094.29` | `24267.62` |
| `power_w` | `151.09` | `355.83` | `367.79` | `370.93` |
| `sm_clock_mhz` | `2670.00` | `2685.00` | `2685.00` | `2685.00` |
| `mem_clock_mhz` | `10251.00` | `10251.00` | `10501.00` | `10501.00` |
| `temperature_c` | `45.50` | `63.80` | `68.00` | `69.00` |

## Throughput periods

| Event | median ms | p90 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| `capture_group_period_ms` | `33.41` | `36.78` | `38.46` | `42.42` |
| `display_packet_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_stage_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_stage_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `filter_output_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `fusion_publish_period_ms` | `66.32` | `76.10` | `78.44` | `86.47` |
| `gpu_owner_publish_period_ms` | `65.89` | `71.93` | `74.92` | `92.25` |
| `raw_fusion_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `stage_join_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `11.34` | `25.10` | `29.33` | `38.37` |
| `edgetam_model_ms` | `16.09` | `26.15` | `29.35` | `35.65` |
| `edgetam_preprocess_ms` | `0.58` | `0.77` | `1.05` | `9.16` |
| `edgetam_prompt_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_postprocess_ms` | `0.03` | `0.04` | `0.05` | `5.69` |
| `edgetam_mask_resize_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_mask_threshold_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_mask_to_cpu_ms` | `0.42` | `0.55` | `0.65` | `7.33` |
| `edgetam_total_ms` | `16.68` | `27.34` | `30.33` | `36.89` |
| `ffs_cycle_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_batch_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_model_ms` | `1.21` | `1.48` | `1.59` | `2.10` |
| `edgetam_batch_vision_total_ms` | `7.06` | `9.12` | `10.34` | `33.35` |
| `edgetam_batch_vision_preprocess_ms` | `1.73` | `2.31` | `3.03` | `27.49` |
| `edgetam_cam0_model_ms` | `24.19` | `30.37` | `31.79` | `35.65` |
| `edgetam_cam1_model_ms` | `15.50` | `20.42` | `22.21` | `27.13` |
| `edgetam_cam2_model_ms` | `14.57` | `16.36` | `16.92` | `18.70` |
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
| `gpu_owner_total_ms` | `65.86` | `71.88` | `74.85` | `92.22` |
| `gpu_owner_ffs_cycle_ms` | `0.26` | `0.45` | `0.69` | `1.37` |
| `gpu_owner_edgetam_cycle_ms` | `65.46` | `71.63` | `74.54` | `91.91` |
| `raw_fusion_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `fusion_total_ms` | `18.48` | `20.94` | `21.93` | `25.51` |
| `filter_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `filter_input_age_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `object_enhanced_pt_ms` | `3.69` | `4.21` | `4.45` | `5.32` |
| `controller_pt_filter_ms` | `6.05` | `6.78` | `6.99` | `8.15` |
| `render_total_ms` | `3.19` | `3.19` | `3.19` | `3.19` |
| `render_queue_wait_ms` | `3834.73` | `3834.73` | `3834.73` | `3834.73` |
| `render_gpu_to_cpu_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_combine_ms` | `0.09` | `0.09` | `0.09` | `0.09` |
| `render_cpu_format_ms` | `0.37` | `0.37` | `0.37` | `0.37` |
| `render_open3d_points_update_ms` | `0.08` | `0.08` | `0.08` | `0.08` |
| `render_open3d_colors_update_ms` | `0.20` | `0.20` | `0.20` | `0.20` |
| `render_open3d_update_geometry_ms` | `1.37` | `1.37` | `1.37` | `1.37` |
| `render_poll_events_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_update_renderer_ms` | `0.02` | `0.02` | `0.02` | `0.02` |
| `open3d_object_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_controller_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `312` | `5.32` | `24000` | `5692` |
| `219` | `4.93` | `24000` | `5729` |
| `131` | `4.93` | `24000` | `5690` |
| `324` | `4.74` | `24000` | `5655` |
| `361` | `4.73` | `24000` | `5701` |
| `322` | `4.68` | `24000` | `5670` |
| `316` | `4.64` | `24000` | `5731` |
| `205` | `4.46` | `24000` | `5709` |
| `234` | `4.43` | `24000` | `5722` |
| `141` | `4.39` | `24000` | `5729` |
