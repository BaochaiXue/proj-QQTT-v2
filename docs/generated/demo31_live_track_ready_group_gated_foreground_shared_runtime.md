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
- render FPS after warmup: `0.10`
- raw fusion FPS after warmup: `0.00`
- filter output FPS after warmup: `0.00`
- fusion FPS after warmup: `15.29`
- stage period p50 after warmup: `65.23 ms`
- display packet period p50 after warmup: `8991.75 ms`
- groups after warmup: `2540`
- complete fused groups after warmup: `1300`
- rendered groups after warmup: `8`
- complete group ratio after warmup: `0.512`
- stage drop count after warmup: `0`
- raw fused pending replacements total: `0`
- render buffer dropped total: `0`
- target deficit: `29.90`
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
| camera startup ms | `11055.00` |
| EdgeTAM model load ms | `894.65` |
| EdgeTAM compile wrap ms | `629.95` |
| EdgeTAM compile prewarm ms | `n/a` |
| EdgeTAM warmup/first forward ms | `127.58` |
| SAM3.1 model load ms | `7299.16` |
| SAM3.1 cam0 segment ms | `555.82` |
| SAM3.1 cam1 segment ms | `131.75` |
| SAM3.1 cam2 segment ms | `118.82` |
| FFS runner init ms | `n/a` |
| FFS first run ms | `n/a` |
| session init + prompt add ms | `2.73` |
| SAM3.1 release cleanup ms | `303.58` |
| time to first complete group s | `20.36` |
| time to first rendered group s | `28.85` |

## GPU Sampling

- backend requested: `nvml`
- backend used: `unavailable`
- device index: `0`
- interval s: `0.500`
- samples after warmup: `350`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `gpu_util_pct` | `44.00` | `100.00` | `100.00` | `100.00` |
| `memory_util_pct` | `11.00` | `70.00` | `71.00` | `75.00` |
| `memory_used_mb` | `3558.66` | `15934.07` | `15961.73` | `16016.12` |
| `power_w` | `156.23` | `423.31` | `425.36` | `432.71` |
| `sm_clock_mhz` | `2670.00` | `2670.00` | `2670.00` | `2685.00` |
| `mem_clock_mhz` | `10251.00` | `10251.00` | `10251.00` | `10501.00` |
| `temperature_c` | `53.00` | `82.00` | `83.00` | `86.00` |

## Throughput periods

| Event | median ms | p90 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| `capture_group_period_ms` | `33.50` | `36.75` | `38.90` | `43.75` |
| `display_packet_publish_period_ms` | `8991.75` | `14867.43` | `16407.34` | `17947.25` |
| `edgetam_stage_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_stage_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `filter_output_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `fusion_publish_period_ms` | `65.59` | `72.63` | `74.74` | `94.16` |
| `gpu_owner_publish_period_ms` | `65.23` | `70.62` | `72.07` | `85.43` |
| `raw_fusion_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_period_ms` | `9058.28` | `15307.08` | `16627.19` | `17947.30` |
| `stage_join_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `12.56` | `25.74` | `36.22` | `44.80` |
| `edgetam_model_ms` | `15.91` | `25.07` | `26.80` | `36.91` |
| `edgetam_preprocess_ms` | `0.67` | `1.08` | `1.26` | `3.46` |
| `edgetam_prompt_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_postprocess_ms` | `0.02` | `0.04` | `0.04` | `6.57` |
| `edgetam_mask_resize_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_mask_threshold_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_mask_to_cpu_ms` | `0.41` | `0.53` | `0.62` | `9.92` |
| `edgetam_total_ms` | `16.50` | `26.03` | `27.67` | `37.54` |
| `ffs_cycle_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_batch_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_model_ms` | `1.27` | `1.59` | `1.75` | `42.01` |
| `edgetam_batch_vision_total_ms` | `7.94` | `10.56` | `11.48` | `58.61` |
| `edgetam_batch_vision_preprocess_ms` | `2.01` | `3.24` | `3.78` | `10.38` |
| `edgetam_cam0_model_ms` | `23.54` | `27.69` | `29.46` | `36.91` |
| `edgetam_cam1_model_ms` | `15.47` | `21.37` | `22.69` | `27.00` |
| `edgetam_cam2_model_ms` | `14.07` | `16.01` | `16.69` | `20.53` |
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
| `gpu_owner_total_ms` | `65.20` | `70.58` | `72.04` | `114.29` |
| `gpu_owner_ffs_cycle_ms` | `0.29` | `0.51` | `0.79` | `1.67` |
| `gpu_owner_edgetam_cycle_ms` | `64.85` | `70.19` | `71.73` | `112.62` |
| `raw_fusion_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `fusion_total_ms` | `24.47` | `27.14` | `28.08` | `37.26` |
| `filter_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `filter_input_age_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `object_enhanced_pt_ms` | `3.65` | `4.25` | `4.49` | `6.06` |
| `controller_pt_filter_ms` | `9.66` | `10.52` | `10.89` | `15.57` |
| `render_total_ms` | `1.86` | `2.77` | `3.25` | `3.72` |
| `render_queue_wait_ms` | `3634.75` | `3664.12` | `3667.46` | `3670.81` |
| `render_gpu_to_cpu_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_combine_ms` | `0.08` | `0.12` | `0.12` | `0.12` |
| `render_cpu_format_ms` | `0.24` | `0.39` | `0.42` | `0.45` |
| `render_open3d_points_update_ms` | `0.10` | `0.14` | `0.16` | `0.18` |
| `render_open3d_colors_update_ms` | `0.06` | `0.14` | `0.18` | `0.22` |
| `render_open3d_update_geometry_ms` | `1.55` | `1.88` | `1.95` | `2.03` |
| `render_poll_events_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_update_renderer_ms` | `0.02` | `0.02` | `0.02` | `0.02` |
| `open3d_object_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_controller_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `335` | `6.06` | `24000` | `5576` |
| `2103` | `6.00` | `24000` | `5585` |
| `1717` | `5.93` | `24000` | `5626` |
| `2625` | `5.83` | `24000` | `5621` |
| `534` | `5.77` | `24000` | `5621` |
| `2042` | `5.38` | `24000` | `5619` |
| `1847` | `5.37` | `24000` | `5613` |
| `2492` | `5.31` | `24000` | `5689` |
| `1113` | `5.06` | `24000` | `5628` |
| `1571` | `5.05` | `24000` | `5669` |
