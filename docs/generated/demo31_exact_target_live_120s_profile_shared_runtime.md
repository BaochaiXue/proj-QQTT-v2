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
- fusion FPS after warmup: `7.69`
- stage period p50 after warmup: `119.17 ms`
- display packet period p50 after warmup: `0.00 ms`
- groups after warmup: `2711`
- complete fused groups after warmup: `817`
- rendered groups after warmup: `0`
- complete group ratio after warmup: `0.301`
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
| parallel init max wait ms | `4268.70` |
| camera startup ms | `4271.90` |
| EdgeTAM model load ms | `551.58` |
| EdgeTAM compile wrap ms | `598.21` |
| EdgeTAM compile prewarm ms | `n/a` |
| EdgeTAM warmup/first forward ms | `113.20` |
| SAM3.1 model load ms | `7832.71` |
| SAM3.1 cam0 segment ms | `545.14` |
| SAM3.1 cam1 segment ms | `119.72` |
| SAM3.1 cam2 segment ms | `120.16` |
| FFS runner init ms | `n/a` |
| FFS first run ms | `n/a` |
| session init + prompt add ms | `2.26` |
| SAM3.1 release cleanup ms | `308.08` |
| time to first complete group s | `17.04` |
| time to first rendered group s | `n/a` |

## GPU Sampling

- backend requested: `nvml`
- backend used: `unavailable`
- device index: `0`
- interval s: `0.500`
- samples after warmup: `420`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `gpu_util_pct` | `29.50` | `100.00` | `100.00` | `100.00` |
| `memory_util_pct` | `7.00` | `78.00` | `78.00` | `79.00` |
| `memory_used_mb` | `2976.12` | `8244.19` | `8294.69` | `8358.31` |
| `power_w` | `106.00` | `377.28` | `381.12` | `386.00` |
| `sm_clock_mhz` | `2572.50` | `2655.00` | `2655.00` | `2670.00` |
| `mem_clock_mhz` | `10251.00` | `10251.00` | `10251.00` | `10501.00` |
| `temperature_c` | `57.00` | `81.00` | `82.00` | `84.00` |

## Throughput periods

| Event | median ms | p90 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| `capture_group_period_ms` | `32.95` | `56.37` | `61.90` | `358.22` |
| `display_packet_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_stage_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_stage_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `filter_output_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `fusion_publish_period_ms` | `119.25` | `128.56` | `133.81` | `418.98` |
| `gpu_owner_publish_period_ms` | `119.17` | `125.97` | `130.13` | `419.72` |
| `raw_fusion_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `stage_join_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `11.24` | `24.83` | `27.34` | `56.90` |
| `edgetam_model_ms` | `19.07` | `77.25` | `79.77` | `377.01` |
| `edgetam_preprocess_ms` | `0.55` | `0.69` | `0.77` | `1.87` |
| `edgetam_prompt_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_postprocess_ms` | `0.03` | `0.04` | `0.05` | `5.12` |
| `edgetam_mask_resize_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_mask_threshold_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_mask_to_cpu_ms` | `0.42` | `0.51` | `0.63` | `20.03` |
| `edgetam_total_ms` | `19.68` | `77.83` | `80.43` | `377.57` |
| `ffs_cycle_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_batch_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_model_ms` | `1.18` | `1.39` | `1.53` | `2.43` |
| `edgetam_batch_vision_total_ms` | `6.95` | `9.20` | `10.23` | `14.69` |
| `edgetam_batch_vision_preprocess_ms` | `1.64` | `2.06` | `2.30` | `5.61` |
| `edgetam_cam0_model_ms` | `18.95` | `21.42` | `22.65` | `58.08` |
| `edgetam_cam1_model_ms` | `75.45` | `80.87` | `83.80` | `377.01` |
| `edgetam_cam2_model_ms` | `14.41` | `16.50` | `17.62` | `308.72` |
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
| `gpu_owner_total_ms` | `119.13` | `125.97` | `130.10` | `419.68` |
| `gpu_owner_ffs_cycle_ms` | `0.28` | `0.58` | `0.81` | `1.67` |
| `gpu_owner_edgetam_cycle_ms` | `118.76` | `125.58` | `129.75` | `419.40` |
| `raw_fusion_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `fusion_total_ms` | `77.69` | `83.39` | `86.72` | `378.97` |
| `filter_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `filter_input_age_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `object_enhanced_pt_ms` | `40.00` | `43.46` | `44.95` | `331.40` |
| `controller_pt_filter_ms` | `27.76` | `30.72` | `31.99` | `323.20` |
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
| `2103` | `331.40` | `24000` | `8518` |
| `2893` | `331.27` | `24000` | `8479` |
| `2454` | `327.06` | `24000` | `8486` |
| `2716` | `324.97` | `24000` | `8525` |
| `2368` | `324.36` | `24000` | `8512` |
| `2188` | `323.95` | `24000` | `8491` |
| `1674` | `318.75` | `24000` | `8454` |
| `1587` | `315.92` | `24000` | `8532` |
| `1418` | `313.90` | `24000` | `8538` |
| `661` | `312.15` | `24000` | `8468` |
