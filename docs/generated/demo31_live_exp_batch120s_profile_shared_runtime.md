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
- fusion FPS after warmup: `14.56`
- stage period p50 after warmup: `67.30 ms`
- display packet period p50 after warmup: `0.00 ms`
- groups after warmup: `840`
- complete fused groups after warmup: `407`
- rendered groups after warmup: `0`
- complete group ratio after warmup: `0.485`
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
| camera startup ms | `11035.32` |
| EdgeTAM model load ms | `720.66` |
| EdgeTAM compile wrap ms | `447.28` |
| EdgeTAM compile prewarm ms | `n/a` |
| EdgeTAM warmup/first forward ms | `131.70` |
| SAM3.1 model load ms | `7528.79` |
| SAM3.1 cam0 segment ms | `593.80` |
| SAM3.1 cam1 segment ms | `124.93` |
| SAM3.1 cam2 segment ms | `118.81` |
| FFS runner init ms | `n/a` |
| FFS first run ms | `n/a` |
| session init + prompt add ms | `2.44` |
| SAM3.1 release cleanup ms | `302.07` |
| time to first complete group s | `20.32` |
| time to first rendered group s | `n/a` |

## GPU Sampling

- backend requested: `nvml`
- backend used: `unavailable`
- device index: `0`
- interval s: `0.500`
- samples after warmup: `123`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `gpu_util_pct` | `41.00` | `45.00` | `47.00` | `100.00` |
| `memory_util_pct` | `9.00` | `23.00` | `25.00` | `32.00` |
| `memory_used_mb` | `3593.44` | `3702.35` | `4146.31` | `21737.38` |
| `power_w` | `81.78` | `153.00` | `153.36` | `155.62` |
| `sm_clock_mhz` | `2670.00` | `2670.00` | `2685.00` | `2685.00` |
| `mem_clock_mhz` | `10251.00` | `10501.00` | `10501.00` | `10501.00` |
| `temperature_c` | `49.00` | `55.80` | `57.00` | `59.00` |

## Throughput periods

| Event | median ms | p90 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| `capture_group_period_ms` | `33.52` | `37.34` | `38.76` | `50.54` |
| `display_packet_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_stage_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_stage_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `filter_output_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `fusion_publish_period_ms` | `67.82` | `77.59` | `80.80` | `125.21` |
| `gpu_owner_publish_period_ms` | `67.30` | `74.93` | `77.37` | `133.60` |
| `raw_fusion_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `stage_join_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `11.12` | `25.67` | `28.58` | `40.71` |
| `edgetam_model_ms` | `16.26` | `29.05` | `31.80` | `64.92` |
| `edgetam_preprocess_ms` | `0.60` | `0.82` | `0.90` | `3.05` |
| `edgetam_prompt_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_postprocess_ms` | `0.03` | `0.04` | `0.05` | `0.46` |
| `edgetam_mask_resize_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_mask_threshold_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_mask_to_cpu_ms` | `0.43` | `0.58` | `0.66` | `7.02` |
| `edgetam_total_ms` | `16.90` | `29.81` | `32.57` | `65.57` |
| `ffs_cycle_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_batch_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_model_ms` | `1.27` | `1.65` | `1.82` | `33.24` |
| `edgetam_batch_vision_total_ms` | `7.36` | `9.36` | `10.42` | `41.98` |
| `edgetam_batch_vision_preprocess_ms` | `1.79` | `2.46` | `2.68` | `9.16` |
| `edgetam_cam0_model_ms` | `25.41` | `32.53` | `34.21` | `59.60` |
| `edgetam_cam1_model_ms` | `15.68` | `18.85` | `21.90` | `27.61` |
| `edgetam_cam2_model_ms` | `14.79` | `16.64` | `17.49` | `64.92` |
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
| `gpu_owner_total_ms` | `67.28` | `74.93` | `77.82` | `133.56` |
| `gpu_owner_ffs_cycle_ms` | `0.34` | `0.60` | `0.91` | `4.93` |
| `gpu_owner_edgetam_cycle_ms` | `66.69` | `74.49` | `77.42` | `133.22` |
| `raw_fusion_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `fusion_total_ms` | `19.09` | `21.44` | `22.41` | `45.49` |
| `filter_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `filter_input_age_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `object_enhanced_pt_ms` | `3.73` | `4.47` | `4.81` | `19.56` |
| `controller_pt_filter_ms` | `6.26` | `7.21` | `7.62` | `18.44` |
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
| `331` | `19.56` | `24000` | `5403` |
| `323` | `17.22` | `24000` | `5740` |
| `398` | `11.10` | `24000` | `5712` |
| `374` | `10.36` | `24000` | `5686` |
| `342` | `8.91` | `24000` | `5083` |
| `445` | `7.36` | `24000` | `5726` |
| `344` | `7.04` | `24000` | `4295` |
| `791` | `5.66` | `24000` | `5718` |
| `254` | `5.50` | `24000` | `5705` |
| `570` | `5.44` | `24000` | `5732` |
