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
- render FPS after warmup: `8.69`
- raw fusion FPS after warmup: `0.00`
- filter output FPS after warmup: `0.00`
- fusion FPS after warmup: `15.36`
- stage period p50 after warmup: `65.07 ms`
- display packet period p50 after warmup: `67.06 ms`
- groups after warmup: `8840`
- complete fused groups after warmup: `4561`
- rendered groups after warmup: `2581`
- complete group ratio after warmup: `0.516`
- stage drop count after warmup: `0`
- raw fused pending replacements total: `0`
- render buffer dropped total: `225`
- target deficit: `21.31`
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
| camera startup ms | `10971.88` |
| EdgeTAM model load ms | `522.00` |
| EdgeTAM compile wrap ms | `603.59` |
| EdgeTAM compile prewarm ms | `n/a` |
| EdgeTAM warmup/first forward ms | `124.88` |
| SAM3.1 model load ms | `7281.54` |
| SAM3.1 cam0 segment ms | `572.70` |
| SAM3.1 cam1 segment ms | `122.98` |
| SAM3.1 cam2 segment ms | `122.22` |
| FFS runner init ms | `n/a` |
| FFS first run ms | `n/a` |
| session init + prompt add ms | `2.76` |
| SAM3.1 release cleanup ms | `309.10` |
| time to first complete group s | `20.06` |
| time to first rendered group s | `20.07` |

## GPU Sampling

- backend requested: `nvml`
- backend used: `unavailable`
- device index: `0`
- interval s: `0.500`
- samples after warmup: `1194`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `gpu_util_pct` | `44.50` | `100.00` | `100.00` | `100.00` |
| `memory_util_pct` | `11.00` | `69.00` | `71.00` | `75.00` |
| `memory_used_mb` | `3991.97` | `16631.50` | `16677.08` | `16950.56` |
| `power_w` | `159.14` | `419.62` | `423.42` | `436.65` |
| `sm_clock_mhz` | `2670.00` | `2670.00` | `2670.00` | `2685.00` |
| `mem_clock_mhz` | `10251.00` | `10251.00` | `10251.00` | `10501.00` |
| `temperature_c` | `53.00` | `89.00` | `90.00` | `95.00` |

## Throughput periods

| Event | median ms | p90 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| `capture_group_period_ms` | `33.50` | `36.74` | `38.60` | `53.95` |
| `display_packet_publish_period_ms` | `67.06` | `77.32` | `425.89` | `2175.28` |
| `edgetam_stage_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_stage_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `filter_output_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `fusion_publish_period_ms` | `65.46` | `72.64` | `74.78` | `134.14` |
| `gpu_owner_publish_period_ms` | `65.07` | `70.28` | `71.89` | `111.72` |
| `raw_fusion_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_period_ms` | `66.73` | `191.60` | `439.29` | `2177.31` |
| `stage_join_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `6.67` | `20.30` | `25.36` | `51.42` |
| `edgetam_model_ms` | `15.93` | `25.39` | `26.78` | `58.70` |
| `edgetam_preprocess_ms` | `0.65` | `1.06` | `1.22` | `2.55` |
| `edgetam_prompt_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_postprocess_ms` | `0.02` | `0.03` | `0.04` | `6.41` |
| `edgetam_mask_resize_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_mask_threshold_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_mask_to_cpu_ms` | `0.41` | `0.59` | `1.07` | `9.04` |
| `edgetam_total_ms` | `16.55` | `26.22` | `27.68` | `59.27` |
| `ffs_cycle_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_batch_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_model_ms` | `1.26` | `1.57` | `1.75` | `7.33` |
| `edgetam_batch_vision_total_ms` | `7.86` | `10.07` | `10.85` | `30.07` |
| `edgetam_batch_vision_preprocess_ms` | `1.95` | `3.19` | `3.66` | `7.64` |
| `edgetam_cam0_model_ms` | `23.75` | `27.57` | `28.84` | `57.47` |
| `edgetam_cam1_model_ms` | `15.49` | `19.40` | `21.50` | `58.70` |
| `edgetam_cam2_model_ms` | `13.86` | `15.83` | `16.60` | `34.86` |
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
| `gpu_owner_total_ms` | `65.03` | `70.24` | `71.85` | `111.69` |
| `gpu_owner_ffs_cycle_ms` | `0.29` | `0.56` | `0.80` | `3.13` |
| `gpu_owner_edgetam_cycle_ms` | `64.66` | `69.89` | `71.46` | `111.41` |
| `raw_fusion_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `fusion_total_ms` | `22.48` | `25.15` | `26.08` | `64.56` |
| `filter_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `filter_input_age_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `object_enhanced_pt_ms` | `3.64` | `4.18` | `4.42` | `13.91` |
| `controller_pt_filter_ms` | `9.07` | `9.98` | `10.36` | `25.05` |
| `render_total_ms` | `1.64` | `1.93` | `2.01` | `18.38` |
| `render_queue_wait_ms` | `9.16` | `9.67` | `10.14` | `356.38` |
| `render_gpu_to_cpu_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_combine_ms` | `0.08` | `0.13` | `0.16` | `1.73` |
| `render_cpu_format_ms` | `0.25` | `0.37` | `0.42` | `1.97` |
| `render_open3d_points_update_ms` | `0.08` | `0.11` | `0.13` | `0.98` |
| `render_open3d_colors_update_ms` | `0.07` | `0.15` | `0.17` | `0.97` |
| `render_open3d_update_geometry_ms` | `1.32` | `1.56` | `1.63` | `18.09` |
| `render_poll_events_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_update_renderer_ms` | `0.02` | `0.02` | `0.03` | `0.06` |
| `open3d_object_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_controller_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `5313` | `13.91` | `24000` | `5657` |
| `5225` | `8.76` | `24000` | `5656` |
| `5230` | `8.15` | `24000` | `5681` |
| `5325` | `7.26` | `24000` | `5665` |
| `5327` | `7.14` | `24000` | `5714` |
| `4174` | `6.02` | `24000` | `5699` |
| `5983` | `5.90` | `24000` | `5708` |
| `7238` | `5.86` | `24000` | `5682` |
| `3393` | `5.78` | `24000` | `5644` |
| `2608` | `5.62` | `24000` | `5633` |
