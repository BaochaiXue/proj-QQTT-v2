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
- render FPS after warmup: `9.50`
- raw fusion FPS after warmup: `0.00`
- filter output FPS after warmup: `0.00`
- fusion FPS after warmup: `15.33`
- stage period p50 after warmup: `65.14 ms`
- display packet period p50 after warmup: `67.86 ms`
- groups after warmup: `1177`
- complete fused groups after warmup: `603`
- rendered groups after warmup: `294`
- complete group ratio after warmup: `0.512`
- stage drop count after warmup: `0`
- raw fused pending replacements total: `0`
- render buffer dropped total: `15`
- target deficit: `20.50`
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
| camera startup ms | `11006.14` |
| EdgeTAM model load ms | `732.97` |
| EdgeTAM compile wrap ms | `425.79` |
| EdgeTAM compile prewarm ms | `n/a` |
| EdgeTAM warmup/first forward ms | `124.68` |
| SAM3.1 model load ms | `7262.92` |
| SAM3.1 cam0 segment ms | `583.28` |
| SAM3.1 cam1 segment ms | `123.51` |
| SAM3.1 cam2 segment ms | `121.35` |
| FFS runner init ms | `n/a` |
| FFS first run ms | `n/a` |
| session init + prompt add ms | `2.86` |
| SAM3.1 release cleanup ms | `316.57` |
| time to first complete group s | `20.25` |
| time to first rendered group s | `28.71` |

## GPU Sampling

- backend requested: `nvml`
- backend used: `unavailable`
- device index: `0`
- interval s: `0.500`
- samples after warmup: `174`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `gpu_util_pct` | `44.00` | `100.00` | `100.00` | `100.00` |
| `memory_util_pct` | `11.00` | `69.70` | `73.00` | `75.00` |
| `memory_used_mb` | `3811.81` | `16446.31` | `16556.26` | `16662.31` |
| `power_w` | `155.15` | `415.70` | `417.54` | `423.90` |
| `sm_clock_mhz` | `2670.00` | `2670.00` | `2670.00` | `2685.00` |
| `mem_clock_mhz` | `10251.00` | `10251.00` | `10251.00` | `10501.00` |
| `temperature_c` | `49.00` | `77.00` | `77.00` | `82.00` |

## Throughput periods

| Event | median ms | p90 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| `capture_group_period_ms` | `33.53` | `36.74` | `38.56` | `45.94` |
| `display_packet_publish_period_ms` | `67.86` | `76.52` | `426.64` | `1090.48` |
| `edgetam_stage_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_stage_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `filter_output_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `fusion_publish_period_ms` | `66.01` | `72.81` | `74.01` | `83.11` |
| `gpu_owner_publish_period_ms` | `65.14` | `70.57` | `71.99` | `79.35` |
| `raw_fusion_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_period_ms` | `67.14` | `173.43` | `431.21` | `1090.61` |
| `stage_join_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `18.78` | `22.48` | `23.78` | `38.04` |
| `edgetam_model_ms` | `16.20` | `24.99` | `26.54` | `35.24` |
| `edgetam_preprocess_ms` | `0.67` | `1.09` | `1.30` | `3.24` |
| `edgetam_prompt_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_postprocess_ms` | `0.02` | `0.04` | `0.04` | `2.63` |
| `edgetam_mask_resize_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_mask_threshold_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_mask_to_cpu_ms` | `0.41` | `0.55` | `0.68` | `9.01` |
| `edgetam_total_ms` | `16.81` | `25.90` | `27.38` | `35.94` |
| `ffs_cycle_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_batch_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_model_ms` | `1.26` | `1.52` | `1.71` | `49.85` |
| `edgetam_batch_vision_total_ms` | `7.97` | `10.12` | `10.94` | `66.33` |
| `edgetam_batch_vision_preprocess_ms` | `2.01` | `3.27` | `3.91` | `9.72` |
| `edgetam_cam0_model_ms` | `23.06` | `27.24` | `28.18` | `35.24` |
| `edgetam_cam1_model_ms` | `15.99` | `21.62` | `23.24` | `29.10` |
| `edgetam_cam2_model_ms` | `13.96` | `15.88` | `16.43` | `18.90` |
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
| `gpu_owner_total_ms` | `65.10` | `70.56` | `72.08` | `123.11` |
| `gpu_owner_ffs_cycle_ms` | `0.32` | `0.51` | `0.82` | `1.49` |
| `gpu_owner_edgetam_cycle_ms` | `64.68` | `70.16` | `71.65` | `122.22` |
| `raw_fusion_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `fusion_total_ms` | `24.06` | `27.26` | `27.94` | `31.07` |
| `filter_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `filter_input_age_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `object_enhanced_pt_ms` | `3.63` | `4.24` | `4.54` | `6.62` |
| `controller_pt_filter_ms` | `9.48` | `10.41` | `10.88` | `14.20` |
| `render_total_ms` | `1.70` | `2.11` | `2.34` | `3.70` |
| `render_queue_wait_ms` | `9.12` | `9.94` | `11.61` | `92.89` |
| `render_gpu_to_cpu_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_combine_ms` | `0.09` | `0.14` | `0.17` | `0.48` |
| `render_cpu_format_ms` | `0.27` | `0.40` | `0.46` | `0.74` |
| `render_open3d_points_update_ms` | `0.08` | `0.11` | `0.12` | `0.24` |
| `render_open3d_colors_update_ms` | `0.08` | `0.17` | `0.20` | `0.32` |
| `render_open3d_update_geometry_ms` | `1.37` | `1.70` | `1.92` | `3.40` |
| `render_poll_events_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_update_renderer_ms` | `0.02` | `0.02` | `0.03` | `0.05` |
| `open3d_object_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_controller_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `241` | `6.62` | `24000` | `5663` |
| `569` | `5.65` | `24000` | `5649` |
| `956` | `5.40` | `24000` | `5676` |
| `214` | `5.36` | `24000` | `5644` |
| `198` | `5.28` | `24000` | `5614` |
| `950` | `5.27` | `24000` | `5666` |
| `191` | `5.25` | `24000` | `5687` |
| `534` | `5.09` | `24000` | `5592` |
| `975` | `5.05` | `24000` | `5614` |
| `237` | `5.02` | `24000` | `5653` |
