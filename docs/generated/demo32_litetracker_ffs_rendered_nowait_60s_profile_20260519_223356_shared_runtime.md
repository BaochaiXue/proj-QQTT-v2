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
- render FPS after warmup: `0.00`
- raw fusion FPS after warmup: `22.55`
- filter output FPS after warmup: `22.55`
- fusion FPS after warmup: `22.55`
- stage period p50 after warmup: `44.19 ms`
- display packet period p50 after warmup: `0.00 ms`
- groups after warmup: `1602`
- complete fused groups after warmup: `1075`
- rendered groups after warmup: `0`
- complete group ratio after warmup: `0.671`
- stage drop count after warmup: `0`
- raw fused pending replacements total: `0`
- render buffer dropped total: `0`
- target deficit: `30.00`
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
| camera startup ms | `10969.18` |
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
| time to first complete group s | `26.74` |
| time to first rendered group s | `n/a` |

## GPU Sampling

- backend requested: `nvml`
- backend used: `unavailable`
- device index: `0`
- interval s: `0.500`
- samples after warmup: `254`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `gpu_util_pct` | `54.00` | `64.00` | `65.00` | `67.00` |
| `memory_util_pct` | `12.00` | `51.00` | `52.00` | `53.00` |
| `memory_used_mb` | `4175.38` | `6797.50` | `6839.14` | `9031.38` |
| `power_w` | `165.81` | `302.67` | `304.78` | `311.37` |
| `sm_clock_mhz` | `2655.00` | `2685.00` | `2685.00` | `2685.00` |
| `mem_clock_mhz` | `10251.00` | `10251.00` | `10251.00` | `10501.00` |
| `temperature_c` | `58.00` | `65.00` | `65.00` | `67.00` |

## Throughput periods

| Event | median ms | p90 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| `capture_group_period_ms` | `33.54` | `38.10` | `39.02` | `192.76` |
| `display_packet_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_stage_publish_period_ms` | `60.23` | `71.85` | `74.05` | `219.83` |
| `ffs_stage_publish_period_ms` | `44.12` | `48.20` | `49.78` | `209.66` |
| `filter_output_publish_period_ms` | `44.17` | `48.82` | `49.96` | `204.41` |
| `fusion_publish_period_ms` | `44.17` | `48.82` | `49.95` | `204.42` |
| `gpu_owner_publish_period_ms` | `44.19` | `48.13` | `49.32` | `203.71` |
| `raw_fusion_publish_period_ms` | `44.19` | `48.20` | `49.40` | `203.80` |
| `render_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `stage_join_publish_period_ms` | `44.19` | `48.13` | `49.32` | `203.71` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `12.79` | `27.06` | `28.94` | `43.88` |
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
| `gpu_owner_total_ms` | `56.04` | `59.20` | `60.46` | `12277.80` |
| `gpu_owner_ffs_cycle_ms` | `39.06` | `40.39` | `40.91` | `43.36` |
| `gpu_owner_edgetam_cycle_ms` | `56.04` | `59.20` | `60.46` | `12277.80` |
| `raw_fusion_total_ms` | `10.21` | `13.00` | `13.83` | `19.69` |
| `fusion_total_ms` | `23.84` | `27.18` | `28.19` | `34.51` |
| `filter_total_ms` | `13.14` | `15.40` | `16.23` | `20.72` |
| `filter_input_age_ms` | `13.77` | `16.10` | `16.81` | `21.19` |
| `object_enhanced_pt_ms` | `4.12` | `5.25` | `5.85` | `8.52` |
| `controller_pt_filter_ms` | `8.79` | `10.84` | `11.50` | `15.73` |
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
| `1523` | `8.52` | `24000` | `6233` |
| `1096` | `7.56` | `24000` | `6201` |
| `1468` | `7.40` | `24000` | `6262` |
| `381` | `7.22` | `24000` | `6169` |
| `1377` | `7.16` | `24000` | `6214` |
| `1680` | `7.14` | `24000` | `6241` |
| `396` | `7.02` | `24000` | `6243` |
| `365` | `6.96` | `24000` | `6140` |
| `1312` | `6.89` | `24000` | `6220` |
| `1729` | `6.79` | `24000` | `6231` |
