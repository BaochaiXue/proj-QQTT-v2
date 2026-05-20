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
- render FPS after warmup: `6.63`
- raw fusion FPS after warmup: `0.00`
- filter output FPS after warmup: `0.00`
- fusion FPS after warmup: `14.63`
- stage period p50 after warmup: `66.56 ms`
- display packet period p50 after warmup: `149.11 ms`
- groups after warmup: `1583`
- complete fused groups after warmup: `778`
- rendered groups after warmup: `350`
- complete group ratio after warmup: `0.491`
- stage drop count after warmup: `0`
- raw fused pending replacements total: `0`
- render buffer dropped total: `0`
- target deficit: `23.37`
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
| camera startup ms | `10996.32` |
| EdgeTAM model load ms | `585.51` |
| EdgeTAM compile wrap ms | `656.96` |
| EdgeTAM compile prewarm ms | `n/a` |
| EdgeTAM warmup/first forward ms | `118.61` |
| SAM3.1 model load ms | `7664.40` |
| SAM3.1 cam0 segment ms | `584.99` |
| SAM3.1 cam1 segment ms | `121.69` |
| SAM3.1 cam2 segment ms | `125.36` |
| FFS runner init ms | `n/a` |
| FFS first run ms | `n/a` |
| session init + prompt add ms | `2.58` |
| SAM3.1 release cleanup ms | `319.76` |
| time to first complete group s | `19.97` |
| time to first rendered group s | `20.45` |

## GPU Sampling

- backend requested: `nvml`
- backend used: `unavailable`
- device index: `0`
- interval s: `0.500`
- samples after warmup: `222`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `gpu_util_pct` | `47.00` | `95.00` | `96.00` | `97.00` |
| `memory_util_pct` | `12.00` | `82.00` | `83.00` | `86.00` |
| `memory_used_mb` | `3682.75` | `8006.33` | `8047.76` | `8085.38` |
| `power_w` | `154.40` | `320.13` | `322.34` | `327.96` |
| `sm_clock_mhz` | `2670.00` | `2670.00` | `2670.00` | `2670.00` |
| `mem_clock_mhz` | `10251.00` | `10251.00` | `10251.00` | `10501.00` |
| `temperature_c` | `53.00` | `73.00` | `74.00` | `75.00` |

## Throughput periods

| Event | median ms | p90 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| `capture_group_period_ms` | `33.53` | `37.36` | `38.72` | `63.56` |
| `display_packet_publish_period_ms` | `149.11` | `155.98` | `161.03` | `330.31` |
| `edgetam_stage_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_stage_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `filter_output_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `fusion_publish_period_ms` | `71.09` | `85.79` | `89.36` | `210.73` |
| `gpu_owner_publish_period_ms` | `66.56` | `74.30` | `77.67` | `225.49` |
| `raw_fusion_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_period_ms` | `149.03` | `160.29` | `167.07` | `341.11` |
| `stage_join_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `13.49` | `24.70` | `27.14` | `59.80` |
| `edgetam_model_ms` | `18.48` | `25.41` | `27.31` | `103.41` |
| `edgetam_preprocess_ms` | `0.56` | `0.75` | `0.86` | `2.96` |
| `edgetam_prompt_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_postprocess_ms` | `0.03` | `0.04` | `0.05` | `3.03` |
| `edgetam_mask_resize_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_mask_threshold_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_mask_to_cpu_ms` | `0.42` | `0.54` | `0.64` | `14.29` |
| `edgetam_total_ms` | `19.21` | `26.04` | `27.88` | `104.01` |
| `ffs_cycle_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_batch_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_model_ms` | `1.22` | `1.52` | `1.73` | `8.00` |
| `edgetam_batch_vision_total_ms` | `7.17` | `9.60` | `10.50` | `50.53` |
| `edgetam_batch_vision_preprocess_ms` | `1.67` | `2.25` | `2.59` | `8.89` |
| `edgetam_cam0_model_ms` | `21.95` | `26.26` | `27.54` | `70.28` |
| `edgetam_cam1_model_ms` | `20.58` | `26.67` | `28.62` | `103.41` |
| `edgetam_cam2_model_ms` | `14.68` | `18.67` | `20.18` | `68.33` |
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
| `gpu_owner_total_ms` | `66.53` | `74.27` | `77.61` | `225.46` |
| `gpu_owner_ffs_cycle_ms` | `0.27` | `0.57` | `0.83` | `2.50` |
| `gpu_owner_edgetam_cycle_ms` | `66.14` | `73.97` | `77.27` | `225.05` |
| `raw_fusion_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `fusion_total_ms` | `18.89` | `22.07` | `23.74` | `89.14` |
| `filter_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `filter_input_age_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `object_enhanced_pt_ms` | `3.75` | `4.47` | `5.00` | `16.22` |
| `controller_pt_filter_ms` | `6.16` | `7.41` | `8.07` | `28.84` |
| `render_total_ms` | `1.59` | `1.89` | `2.09` | `41.58` |
| `render_queue_wait_ms` | `208.74` | `267.60` | `277.52` | `381.75` |
| `render_gpu_to_cpu_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_combine_ms` | `0.09` | `0.17` | `0.24` | `7.80` |
| `render_cpu_format_ms` | `0.24` | `0.39` | `0.47` | `8.06` |
| `render_open3d_points_update_ms` | `0.08` | `0.11` | `0.12` | `1.12` |
| `render_open3d_colors_update_ms` | `0.07` | `0.12` | `0.15` | `1.38` |
| `render_open3d_update_geometry_ms` | `1.26` | `1.49` | `1.58` | `33.42` |
| `render_poll_events_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_update_renderer_ms` | `0.02` | `0.02` | `0.03` | `0.04` |
| `open3d_object_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_controller_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `180` | `16.22` | `24000` | `3265` |
| `176` | `15.95` | `24000` | `4867` |
| `1439` | `12.19` | `24000` | `5789` |
| `1069` | `9.25` | `24000` | `5833` |
| `174` | `8.86` | `24000` | `5833` |
| `1444` | `8.80` | `24000` | `4607` |
| `1529` | `8.38` | `24000` | `5766` |
| `1072` | `8.16` | `24000` | `4906` |
| `1472` | `7.86` | `24000` | `5806` |
| `1431` | `7.42` | `24000` | `5759` |
