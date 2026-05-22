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
- render FPS after warmup: `6.38`
- raw fusion FPS after warmup: `0.00`
- filter output FPS after warmup: `0.00`
- fusion FPS after warmup: `15.02`
- stage period p50 after warmup: `65.71 ms`
- display packet period p50 after warmup: `156.32 ms`
- groups after warmup: `3259`
- complete fused groups after warmup: `1546`
- rendered groups after warmup: `653`
- complete group ratio after warmup: `0.474`
- stage drop count after warmup: `0`
- raw fused pending replacements total: `0`
- render buffer dropped total: `0`
- target deficit: `23.62`
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
| camera startup ms | `10966.36` |
| EdgeTAM model load ms | `533.66` |
| EdgeTAM compile wrap ms | `607.48` |
| EdgeTAM compile prewarm ms | `n/a` |
| EdgeTAM warmup/first forward ms | `131.66` |
| SAM3.1 model load ms | `7446.83` |
| SAM3.1 cam0 segment ms | `535.30` |
| SAM3.1 cam1 segment ms | `121.72` |
| SAM3.1 cam2 segment ms | `119.90` |
| FFS runner init ms | `n/a` |
| FFS first run ms | `n/a` |
| session init + prompt add ms | `2.65` |
| SAM3.1 release cleanup ms | `305.67` |
| time to first complete group s | `30.43` |
| time to first rendered group s | `30.90` |

## GPU Sampling

- backend requested: `nvml`
- backend used: `unavailable`
- device index: `0`
- interval s: `0.500`
- samples after warmup: `463`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `gpu_util_pct` | `46.00` | `100.00` | `100.00` | `100.00` |
| `memory_util_pct` | `12.00` | `77.00` | `78.00` | `100.00` |
| `memory_used_mb` | `3442.94` | `8276.36` | `8297.18` | `8453.12` |
| `power_w` | `157.18` | `379.47` | `380.68` | `386.95` |
| `sm_clock_mhz` | `2655.00` | `2670.00` | `2670.00` | `2670.00` |
| `mem_clock_mhz` | `10251.00` | `10251.00` | `10251.00` | `10501.00` |
| `temperature_c` | `53.00` | `84.00` | `85.00` | `87.00` |

## Throughput periods

| Event | median ms | p90 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| `capture_group_period_ms` | `33.54` | `37.27` | `39.31` | `503.13` |
| `display_packet_publish_period_ms` | `156.32` | `162.40` | `164.06` | `468.07` |
| `edgetam_stage_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_stage_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `filter_output_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `fusion_publish_period_ms` | `67.68` | `84.05` | `87.09` | `520.19` |
| `gpu_owner_publish_period_ms` | `65.71` | `73.28` | `75.42` | `520.34` |
| `raw_fusion_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_period_ms` | `156.50` | `165.40` | `167.28` | `467.71` |
| `stage_join_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `11.80` | `25.32` | `28.05` | `44.63` |
| `edgetam_model_ms` | `18.02` | `24.65` | `26.74` | `476.99` |
| `edgetam_preprocess_ms` | `0.54` | `0.72` | `0.82` | `1.60` |
| `edgetam_prompt_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_postprocess_ms` | `0.02` | `0.04` | `0.04` | `5.46` |
| `edgetam_mask_resize_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_mask_threshold_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_mask_to_cpu_ms` | `0.42` | `0.55` | `0.66` | `8.33` |
| `edgetam_total_ms` | `18.67` | `25.24` | `27.36` | `477.54` |
| `ffs_cycle_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_batch_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_model_ms` | `1.21` | `1.46` | `1.59` | `70.50` |
| `edgetam_batch_vision_total_ms` | `7.38` | `10.04` | `10.74` | `81.92` |
| `edgetam_batch_vision_preprocess_ms` | `1.63` | `2.17` | `2.47` | `4.80` |
| `edgetam_cam0_model_ms` | `21.85` | `26.30` | `28.16` | `476.99` |
| `edgetam_cam1_model_ms` | `20.14` | `25.39` | `27.25` | `44.01` |
| `edgetam_cam2_model_ms` | `14.07` | `17.61` | `19.34` | `30.86` |
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
| `gpu_owner_total_ms` | `65.67` | `73.24` | `75.41` | `520.29` |
| `gpu_owner_ffs_cycle_ms` | `0.33` | `0.86` | `1.26` | `2.96` |
| `gpu_owner_edgetam_cycle_ms` | `65.21` | `72.77` | `74.83` | `519.96` |
| `raw_fusion_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `fusion_total_ms` | `21.97` | `25.54` | `26.51` | `485.37` |
| `filter_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `filter_input_age_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `object_enhanced_pt_ms` | `3.70` | `4.41` | `4.79` | `7.67` |
| `controller_pt_filter_ms` | `9.38` | `11.16` | `11.73` | `15.41` |
| `render_total_ms` | `1.59` | `1.88` | `2.03` | `7.16` |
| `render_queue_wait_ms` | `211.59` | `267.47` | `273.32` | `505.43` |
| `render_gpu_to_cpu_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_combine_ms` | `0.09` | `0.18` | `0.28` | `5.70` |
| `render_cpu_format_ms` | `0.27` | `0.42` | `0.59` | `5.83` |
| `render_open3d_points_update_ms` | `0.08` | `0.11` | `0.13` | `5.17` |
| `render_open3d_colors_update_ms` | `0.08` | `0.15` | `0.17` | `5.55` |
| `render_open3d_update_geometry_ms` | `1.23` | `1.43` | `1.47` | `4.20` |
| `render_poll_events_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_update_renderer_ms` | `0.01` | `0.02` | `0.02` | `0.03` |
| `open3d_object_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_controller_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `2942` | `7.67` | `24000` | `6259` |
| `1589` | `7.13` | `24000` | `6273` |
| `1920` | `7.12` | `24000` | `6355` |
| `438` | `6.95` | `24000` | `6260` |
| `3061` | `6.79` | `24000` | `6312` |
| `1754` | `6.12` | `24000` | `6302` |
| `1601` | `5.80` | `24000` | `6294` |
| `715` | `5.64` | `24000` | `6252` |
| `2609` | `5.62` | `24000` | `6336` |
| `3377` | `5.61` | `24000` | `6293` |
