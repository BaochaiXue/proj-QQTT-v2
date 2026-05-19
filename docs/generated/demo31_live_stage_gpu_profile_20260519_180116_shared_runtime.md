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
- render FPS after warmup: `0.16`
- raw fusion FPS after warmup: `0.00`
- filter output FPS after warmup: `0.00`
- fusion FPS after warmup: `15.43`
- stage period p50 after warmup: `64.41 ms`
- display packet period p50 after warmup: `4517.19 ms`
- groups after warmup: `3375`
- complete fused groups after warmup: `1749`
- rendered groups after warmup: `17`
- complete group ratio after warmup: `0.518`
- stage drop count after warmup: `0`
- raw fused pending replacements total: `0`
- render buffer dropped total: `0`
- target deficit: `29.84`
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
| camera startup ms | `11085.10` |
| EdgeTAM model load ms | `734.29` |
| EdgeTAM compile wrap ms | `405.15` |
| EdgeTAM compile prewarm ms | `n/a` |
| EdgeTAM warmup/first forward ms | `133.61` |
| SAM3.1 model load ms | `7075.87` |
| SAM3.1 cam0 segment ms | `552.88` |
| SAM3.1 cam1 segment ms | `121.75` |
| SAM3.1 cam2 segment ms | `124.26` |
| FFS runner init ms | `n/a` |
| FFS first run ms | `n/a` |
| session init + prompt add ms | `3.21` |
| SAM3.1 release cleanup ms | `303.48` |
| time to first complete group s | `20.14` |
| time to first rendered group s | `32.99` |

## GPU Sampling

- backend requested: `nvml`
- backend used: `unavailable`
- device index: `0`
- interval s: `0.200`
- samples after warmup: `1174`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `gpu_util_pct` | `44.00` | `100.00` | `100.00` | `100.00` |
| `memory_util_pct` | `11.00` | `70.00` | `72.00` | `76.00` |
| `memory_used_mb` | `3606.56` | `16083.19` | `16136.46` | `16269.75` |
| `power_w` | `157.41` | `421.92` | `425.43` | `430.83` |
| `sm_clock_mhz` | `2670.00` | `2670.00` | `2670.00` | `2685.00` |
| `mem_clock_mhz` | `10251.00` | `10251.00` | `10251.00` | `10501.00` |
| `temperature_c` | `51.00` | `82.00` | `84.00` | `88.00` |

## Throughput periods

| Event | median ms | p90 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| `capture_group_period_ms` | `33.49` | `37.25` | `39.40` | `46.09` |
| `display_packet_publish_period_ms` | `4517.19` | `8974.58` | `10102.62` | `13436.90` |
| `edgetam_stage_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_stage_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `filter_output_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `fusion_publish_period_ms` | `64.85` | `71.79` | `73.61` | `152.42` |
| `gpu_owner_publish_period_ms` | `64.41` | `69.51` | `71.01` | `135.20` |
| `raw_fusion_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_period_ms` | `4516.77` | `8970.71` | `10101.56` | `13436.35` |
| `stage_join_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `5.75` | `19.57` | `26.85` | `39.23` |
| `edgetam_model_ms` | `15.68` | `24.62` | `26.27` | `51.04` |
| `edgetam_preprocess_ms` | `0.65` | `1.07` | `1.21` | `3.90` |
| `edgetam_prompt_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_postprocess_ms` | `0.03` | `0.04` | `0.04` | `7.44` |
| `edgetam_mask_resize_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_mask_threshold_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_mask_to_cpu_ms` | `0.41` | `0.53` | `0.64` | `9.58` |
| `edgetam_total_ms` | `16.31` | `25.56` | `27.19` | `54.84` |
| `ffs_cycle_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_batch_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_model_ms` | `1.25` | `1.54` | `1.72` | `39.11` |
| `edgetam_batch_vision_total_ms` | `7.82` | `10.08` | `10.86` | `60.28` |
| `edgetam_batch_vision_preprocess_ms` | `1.96` | `3.20` | `3.62` | `11.71` |
| `edgetam_cam0_model_ms` | `23.17` | `26.95` | `28.45` | `51.04` |
| `edgetam_cam1_model_ms` | `15.07` | `21.48` | `22.67` | `45.50` |
| `edgetam_cam2_model_ms` | `14.02` | `16.03` | `16.55` | `29.73` |
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
| `gpu_owner_total_ms` | `64.38` | `69.47` | `70.98` | `135.16` |
| `gpu_owner_ffs_cycle_ms` | `0.32` | `0.54` | `0.81` | `3.92` |
| `gpu_owner_edgetam_cycle_ms` | `63.97` | `69.08` | `70.61` | `134.81` |
| `raw_fusion_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `fusion_total_ms` | `24.08` | `26.90` | `27.69` | `64.47` |
| `filter_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `filter_input_age_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `object_enhanced_pt_ms` | `3.63` | `4.22` | `4.45` | `12.81` |
| `controller_pt_filter_ms` | `9.62` | `10.64` | `11.06` | `24.60` |
| `render_total_ms` | `1.94` | `2.32` | `2.53` | `3.31` |
| `render_queue_wait_ms` | `3617.92` | `3644.08` | `3647.96` | `3655.87` |
| `render_gpu_to_cpu_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_combine_ms` | `0.08` | `0.12` | `0.17` | `0.33` |
| `render_cpu_format_ms` | `0.29` | `0.33` | `0.38` | `0.58` |
| `render_open3d_points_update_ms` | `0.09` | `0.11` | `0.12` | `0.13` |
| `render_open3d_colors_update_ms` | `0.08` | `0.11` | `0.12` | `0.17` |
| `render_open3d_update_geometry_ms` | `1.56` | `1.80` | `1.96` | `1.97` |
| `render_poll_events_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_update_renderer_ms` | `0.02` | `0.02` | `0.02` | `0.03` |
| `open3d_object_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_controller_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `1506` | `12.81` | `24000` | `5629` |
| `1511` | `10.68` | `24000` | `4517` |
| `1543` | `10.35` | `24000` | `5643` |
| `1601` | `8.64` | `24000` | `5680` |
| `1495` | `8.53` | `24000` | `5653` |
| `1615` | `8.47` | `24000` | `5629` |
| `2496` | `5.64` | `24000` | `5647` |
| `158` | `5.53` | `24000` | `5683` |
| `2329` | `5.53` | `24000` | `5639` |
| `2488` | `5.50` | `24000` | `5655` |
