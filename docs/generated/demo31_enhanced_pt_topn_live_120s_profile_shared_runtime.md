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
- fusion FPS after warmup: `7.60`
- stage period p50 after warmup: `120.32 ms`
- display packet period p50 after warmup: `0.00 ms`
- groups after warmup: `2820`
- complete fused groups after warmup: `776`
- rendered groups after warmup: `0`
- complete group ratio after warmup: `0.275`
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
| camera startup ms | `11037.98` |
| EdgeTAM model load ms | `611.73` |
| EdgeTAM compile wrap ms | `684.99` |
| EdgeTAM compile prewarm ms | `n/a` |
| EdgeTAM warmup/first forward ms | `128.40` |
| SAM3.1 model load ms | `7078.78` |
| SAM3.1 cam0 segment ms | `651.03` |
| SAM3.1 cam1 segment ms | `120.07` |
| SAM3.1 cam2 segment ms | `119.64` |
| FFS runner init ms | `n/a` |
| FFS first run ms | `n/a` |
| session init + prompt add ms | `2.57` |
| SAM3.1 release cleanup ms | `296.34` |
| time to first complete group s | `30.48` |
| time to first rendered group s | `n/a` |

## GPU Sampling

- backend requested: `nvml`
- backend used: `unavailable`
- device index: `0`
- interval s: `0.500`
- samples after warmup: `448`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `gpu_util_pct` | `28.00` | `100.00` | `100.00` | `100.00` |
| `memory_util_pct` | `6.00` | `76.00` | `78.00` | `100.00` |
| `memory_used_mb` | `2994.12` | `8160.88` | `8193.36` | `8216.25` |
| `power_w` | `103.50` | `374.58` | `378.75` | `387.78` |
| `sm_clock_mhz` | `2565.00` | `2655.00` | `2670.00` | `2670.00` |
| `mem_clock_mhz` | `10251.00` | `10251.00` | `10251.00` | `10501.00` |
| `temperature_c` | `52.00` | `80.00` | `81.00` | `84.00` |

## Throughput periods

| Event | median ms | p90 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| `capture_group_period_ms` | `33.23` | `57.73` | `63.71` | `480.61` |
| `display_packet_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_stage_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_stage_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `filter_output_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `fusion_publish_period_ms` | `120.25` | `128.98` | `133.31` | `537.26` |
| `gpu_owner_publish_period_ms` | `120.32` | `127.41` | `130.26` | `563.07` |
| `raw_fusion_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `stage_join_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `10.34` | `26.62` | `28.21` | `56.77` |
| `edgetam_model_ms` | `18.89` | `78.19` | `80.54` | `516.74` |
| `edgetam_preprocess_ms` | `0.54` | `0.71` | `0.78` | `1.13` |
| `edgetam_prompt_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_postprocess_ms` | `0.02` | `0.04` | `0.05` | `13.14` |
| `edgetam_mask_resize_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_mask_threshold_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_mask_to_cpu_ms` | `0.42` | `0.55` | `0.66` | `418.69` |
| `edgetam_total_ms` | `19.54` | `78.85` | `81.24` | `517.37` |
| `ffs_cycle_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_batch_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_model_ms` | `1.20` | `1.41` | `1.55` | `89.75` |
| `edgetam_batch_vision_total_ms` | `7.00` | `9.51` | `10.23` | `98.08` |
| `edgetam_batch_vision_preprocess_ms` | `1.63` | `2.14` | `2.34` | `3.38` |
| `edgetam_cam0_model_ms` | `18.67` | `21.54` | `23.78` | `65.09` |
| `edgetam_cam1_model_ms` | `76.00` | `81.56` | `84.32` | `516.74` |
| `edgetam_cam2_model_ms` | `14.42` | `16.62` | `30.65` | `80.72` |
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
| `gpu_owner_total_ms` | `120.30` | `127.44` | `130.50` | `563.03` |
| `gpu_owner_ffs_cycle_ms` | `0.28` | `0.66` | `0.94` | `2.29` |
| `gpu_owner_edgetam_cycle_ms` | `119.88` | `126.91` | `129.84` | `562.66` |
| `raw_fusion_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `fusion_total_ms` | `78.31` | `84.39` | `86.65` | `528.96` |
| `filter_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `filter_input_age_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `object_enhanced_pt_ms` | `40.39` | `43.97` | `45.17` | `445.92` |
| `controller_pt_filter_ms` | `27.80` | `31.35` | `32.89` | `469.60` |
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
| `2939` | `445.92` | `24000` | `8499` |
| `2580` | `444.65` | `24000` | `8458` |
| `929` | `437.92` | `24000` | `8428` |
| `1047` | `436.41` | `24000` | `8469` |
| `2462` | `434.20` | `24000` | `8500` |
| `584` | `432.00` | `24000` | `8451` |
| `813` | `415.84` | `24000` | `8423` |
| `0` | `53.06` | `24000` | `8273` |
| `1212` | `50.52` | `24000` | `8410` |
| `2782` | `49.30` | `24000` | `8509` |
