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
- render FPS after warmup: `0.31`
- raw fusion FPS after warmup: `0.00`
- filter output FPS after warmup: `0.00`
- fusion FPS after warmup: `15.34`
- stage period p50 after warmup: `64.77 ms`
- display packet period p50 after warmup: `3698.35 ms`
- groups after warmup: `435`
- complete fused groups after warmup: `223`
- rendered groups after warmup: `4`
- complete group ratio after warmup: `0.513`
- stage drop count after warmup: `0`
- raw fused pending replacements total: `0`
- render buffer dropped total: `0`
- target deficit: `29.69`
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
| camera startup ms | `11023.86` |
| EdgeTAM model load ms | `771.21` |
| EdgeTAM compile wrap ms | `424.06` |
| EdgeTAM compile prewarm ms | `n/a` |
| EdgeTAM warmup/first forward ms | `134.62` |
| SAM3.1 model load ms | `7494.97` |
| SAM3.1 cam0 segment ms | `601.93` |
| SAM3.1 cam1 segment ms | `121.58` |
| SAM3.1 cam2 segment ms | `124.50` |
| FFS runner init ms | `n/a` |
| FFS first run ms | `n/a` |
| session init + prompt add ms | `3.28` |
| SAM3.1 release cleanup ms | `296.06` |
| time to first complete group s | `19.90` |
| time to first rendered group s | `22.42` |

## GPU Sampling

- backend requested: `nvml`
- backend used: `unavailable`
- device index: `0`
- interval s: `0.500`
- samples after warmup: `70`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `gpu_util_pct` | `39.00` | `87.00` | `91.75` | `98.00` |
| `memory_util_pct` | `7.00` | `30.00` | `51.75` | `57.00` |
| `memory_used_mb` | `3584.03` | `9076.88` | `9078.02` | `9084.69` |
| `power_w` | `134.02` | `152.71` | `153.43` | `154.22` |
| `sm_clock_mhz` | `2670.00` | `2670.00` | `2685.00` | `2685.00` |
| `mem_clock_mhz` | `10251.00` | `10251.00` | `10251.00` | `10501.00` |
| `temperature_c` | `48.50` | `55.10` | `58.55` | `69.00` |

## Throughput periods

| Event | median ms | p90 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| `capture_group_period_ms` | `33.49` | `36.84` | `38.52` | `44.51` |
| `display_packet_publish_period_ms` | `3698.35` | `4464.60` | `4560.38` | `4656.16` |
| `edgetam_stage_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_stage_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `filter_output_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `fusion_publish_period_ms` | `64.95` | `72.30` | `73.92` | `80.98` |
| `gpu_owner_publish_period_ms` | `64.77` | `69.97` | `71.87` | `76.37` |
| `raw_fusion_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_period_ms` | `3698.90` | `4458.63` | `4553.60` | `4648.57` |
| `stage_join_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `10.63` | `24.04` | `27.17` | `40.63` |
| `edgetam_model_ms` | `15.97` | `25.42` | `26.77` | `32.54` |
| `edgetam_preprocess_ms` | `0.59` | `0.79` | `0.90` | `1.30` |
| `edgetam_prompt_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_postprocess_ms` | `0.03` | `0.04` | `0.04` | `5.39` |
| `edgetam_mask_resize_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_mask_threshold_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_mask_to_cpu_ms` | `0.42` | `0.56` | `0.64` | `8.10` |
| `edgetam_total_ms` | `16.56` | `26.12` | `27.34` | `33.05` |
| `ffs_cycle_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_batch_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_model_ms` | `1.24` | `1.48` | `1.68` | `3.87` |
| `edgetam_batch_vision_total_ms` | `7.18` | `9.71` | `10.55` | `12.41` |
| `edgetam_batch_vision_preprocess_ms` | `1.76` | `2.36` | `2.68` | `3.89` |
| `edgetam_cam0_model_ms` | `24.03` | `27.45` | `28.27` | `32.54` |
| `edgetam_cam1_model_ms` | `15.57` | `19.23` | `22.07` | `27.06` |
| `edgetam_cam2_model_ms` | `14.58` | `16.10` | `16.55` | `19.63` |
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
| `gpu_owner_total_ms` | `64.75` | `69.92` | `71.81` | `76.34` |
| `gpu_owner_ffs_cycle_ms` | `0.32` | `0.57` | `0.90` | `2.35` |
| `gpu_owner_edgetam_cycle_ms` | `64.27` | `69.56` | `71.12` | `75.79` |
| `raw_fusion_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `fusion_total_ms` | `18.96` | `21.42` | `21.87` | `25.55` |
| `filter_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `filter_input_age_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `object_enhanced_pt_ms` | `3.81` | `4.26` | `4.57` | `8.35` |
| `controller_pt_filter_ms` | `5.87` | `6.45` | `7.14` | `9.57` |
| `render_total_ms` | `2.15` | `2.63` | `2.71` | `2.79` |
| `render_queue_wait_ms` | `399.49` | `556.54` | `590.18` | `623.82` |
| `render_gpu_to_cpu_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_combine_ms` | `0.10` | `0.12` | `0.13` | `0.13` |
| `render_cpu_format_ms` | `0.26` | `0.32` | `0.33` | `0.33` |
| `render_open3d_points_update_ms` | `0.08` | `0.10` | `0.10` | `0.10` |
| `render_open3d_colors_update_ms` | `0.07` | `0.12` | `0.13` | `0.14` |
| `render_open3d_update_geometry_ms` | `1.41` | `1.51` | `1.52` | `1.53` |
| `render_poll_events_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_update_renderer_ms` | `0.02` | `0.02` | `0.02` | `0.02` |
| `open3d_object_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_controller_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `399` | `8.35` | `24000` | `5668` |
| `279` | `5.55` | `24000` | `5693` |
| `486` | `5.55` | `24000` | `5724` |
| `518` | `5.24` | `24000` | `5701` |
| `243` | `5.14` | `24000` | `5700` |
| `351` | `5.04` | `24000` | `5715` |
| `537` | `4.92` | `24000` | `5715` |
| `381` | `4.85` | `24000` | `5656` |
| `314` | `4.84` | `24000` | `5699` |
| `553` | `4.73` | `24000` | `5686` |
