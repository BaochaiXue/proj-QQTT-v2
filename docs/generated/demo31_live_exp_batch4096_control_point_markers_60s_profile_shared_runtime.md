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
- render FPS after warmup: `0.14`
- raw fusion FPS after warmup: `0.00`
- filter output FPS after warmup: `0.00`
- fusion FPS after warmup: `15.22`
- stage period p50 after warmup: `65.34 ms`
- display packet period p50 after warmup: `4435.70 ms`
- groups after warmup: `1594`
- complete fused groups after warmup: `813`
- rendered groups after warmup: `4`
- complete group ratio after warmup: `0.510`
- stage drop count after warmup: `0`
- raw fused pending replacements total: `0`
- render buffer dropped total: `0`
- target deficit: `29.86`
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
| camera startup ms | `11014.18` |
| EdgeTAM model load ms | `528.36` |
| EdgeTAM compile wrap ms | `599.08` |
| EdgeTAM compile prewarm ms | `n/a` |
| EdgeTAM warmup/first forward ms | `132.23` |
| SAM3.1 model load ms | `7500.23` |
| SAM3.1 cam0 segment ms | `596.21` |
| SAM3.1 cam1 segment ms | `125.81` |
| SAM3.1 cam2 segment ms | `126.16` |
| FFS runner init ms | `n/a` |
| FFS first run ms | `n/a` |
| session init + prompt add ms | `3.11` |
| SAM3.1 release cleanup ms | `299.14` |
| time to first complete group s | `20.01` |
| time to first rendered group s | `38.74` |

## GPU Sampling

- backend requested: `nvml`
- backend used: `unavailable`
- device index: `0`
- interval s: `0.500`
- samples after warmup: `226`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `gpu_util_pct` | `44.00` | `100.00` | `100.00` | `100.00` |
| `memory_util_pct` | `11.00` | `71.00` | `76.00` | `78.00` |
| `memory_used_mb` | `3537.38` | `24155.50` | `24331.48` | `24493.62` |
| `power_w` | `154.92` | `365.20` | `384.11` | `415.76` |
| `sm_clock_mhz` | `2670.00` | `2670.00` | `2670.00` | `2685.00` |
| `mem_clock_mhz` | `10251.00` | `10251.00` | `10251.00` | `10501.00` |
| `temperature_c` | `51.00` | `74.50` | `76.00` | `78.00` |

## Throughput periods

| Event | median ms | p90 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| `capture_group_period_ms` | `33.50` | `36.93` | `38.75` | `46.40` |
| `display_packet_publish_period_ms` | `4435.70` | `11661.12` | `12564.29` | `13467.47` |
| `edgetam_stage_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_stage_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `filter_output_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `fusion_publish_period_ms` | `65.52` | `73.99` | `75.89` | `117.26` |
| `gpu_owner_publish_period_ms` | `65.34` | `71.21` | `73.36` | `145.85` |
| `raw_fusion_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_period_ms` | `4439.75` | `11668.45` | `12572.04` | `13475.63` |
| `stage_join_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `10.56` | `26.27` | `28.99` | `42.50` |
| `edgetam_model_ms` | `15.81` | `27.07` | `29.16` | `61.08` |
| `edgetam_preprocess_ms` | `0.57` | `0.76` | `0.86` | `1.99` |
| `edgetam_prompt_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_postprocess_ms` | `0.03` | `0.04` | `0.04` | `4.99` |
| `edgetam_mask_resize_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_mask_threshold_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_mask_to_cpu_ms` | `0.42` | `0.54` | `0.63` | `7.46` |
| `edgetam_total_ms` | `16.41` | `27.92` | `29.93` | `62.72` |
| `ffs_cycle_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_batch_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_model_ms` | `1.23` | `1.51` | `1.63` | `4.58` |
| `edgetam_batch_vision_total_ms` | `7.11` | `9.07` | `10.16` | `21.14` |
| `edgetam_batch_vision_preprocess_ms` | `1.72` | `2.27` | `2.57` | `5.96` |
| `edgetam_cam0_model_ms` | `24.58` | `30.17` | `31.44` | `61.08` |
| `edgetam_cam1_model_ms` | `15.20` | `18.32` | `20.75` | `36.31` |
| `edgetam_cam2_model_ms` | `14.30` | `16.33` | `16.96` | `22.35` |
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
| `gpu_owner_total_ms` | `65.30` | `71.29` | `73.36` | `145.82` |
| `gpu_owner_ffs_cycle_ms` | `0.33` | `0.61` | `0.89` | `1.68` |
| `gpu_owner_edgetam_cycle_ms` | `64.92` | `70.80` | `73.00` | `145.46` |
| `raw_fusion_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `fusion_total_ms` | `18.68` | `21.01` | `21.98` | `40.44` |
| `filter_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `filter_input_age_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `object_enhanced_pt_ms` | `3.71` | `4.26` | `4.45` | `5.38` |
| `controller_pt_filter_ms` | `6.19` | `6.91` | `7.25` | `9.72` |
| `render_total_ms` | `2.31` | `2.78` | `2.86` | `2.93` |
| `render_queue_wait_ms` | `3346.07` | `3518.55` | `3549.95` | `3581.35` |
| `render_gpu_to_cpu_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_combine_ms` | `0.10` | `0.21` | `0.23` | `0.25` |
| `render_cpu_format_ms` | `0.25` | `0.44` | `0.47` | `0.51` |
| `render_open3d_points_update_ms` | `0.08` | `0.13` | `0.14` | `0.15` |
| `render_open3d_colors_update_ms` | `0.08` | `0.11` | `0.11` | `0.12` |
| `render_open3d_update_geometry_ms` | `1.39` | `1.50` | `1.52` | `1.54` |
| `render_poll_events_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_update_renderer_ms` | `0.01` | `0.03` | `0.03` | `0.04` |
| `open3d_object_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_controller_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `293` | `5.38` | `24000` | `5814` |
| `881` | `5.37` | `24000` | `5779` |
| `266` | `5.27` | `24000` | `5760` |
| `137` | `5.21` | `24000` | `5790` |
| `1517` | `5.19` | `24000` | `5736` |
| `481` | `5.14` | `24000` | `5796` |
| `313` | `5.14` | `24000` | `5796` |
| `423` | `5.08` | `24000` | `5798` |
| `1099` | `5.05` | `24000` | `5758` |
| `352` | `5.02` | `24000` | `5810` |
