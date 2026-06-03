# Demo 3.3 performance profile

- preset: `demo2.3-dual4090-maxfps`
- canonical preset: `demo2.3-dual4090-maxfps`
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
- render FPS after warmup: `4.07`
- raw fusion FPS after warmup: `4.07`
- filter output FPS after warmup: `4.06`
- fusion FPS after warmup: `4.06`
- stage period p50 after warmup: `91.52 ms`
- display packet period p50 after warmup: `232.81 ms`
- groups after warmup: `1093`
- complete fused groups after warmup: `97`
- rendered groups after warmup: `96`
- complete group ratio after warmup: `0.089`
- stage drop count after warmup: `0`
- raw fused pending replacements total: `0`
- render buffer dropped total: `0`
- target deficit: `25.93`
- bottleneck class: `upstream_supply`
- GPU pipeline: `dual-gpu-split`
- single-owner order: `dual_gpu_process_split`
- filter scheduler: `async`
- render filtered only: `True`
- pin memory mode: `off`
- FFS input staging: `pinned`
- H2D stream mode: `default`

## Shape Prior Warmup

- enabled: `True`
- status: `case_ready`
- case dir: `/home/xinjie/proj-QQTT-v2/result/demo32_ffs_tapnextpp/demo33_shape_prior_warmup/20260603-172344/case`
- object points0: `76637`
- surface points: `0`
- interior points: `0`
- structure points: `0`
- affects tracker input: `False`
- affects live observation PCD: `False`

## Init Profile

| Stage | value |
| --- | ---: |
| parallel init max wait ms | `n/a` |
| camera startup ms | `10741.53` |
| EdgeTAM model load ms | `n/a` |
| EdgeTAM compile wrap ms | `n/a` |
| EdgeTAM compile prewarm ms | `n/a` |
| EdgeTAM warmup/first forward ms | `n/a` |
| SAM3.1 model load ms | `n/a` |
| SAM3.1 cam0 segment ms | `n/a` |
| SAM3.1 cam1 segment ms | `n/a` |
| SAM3.1 cam2 segment ms | `n/a` |
| FFS runner init ms | `n/a` |
| FFS first run ms | `n/a` |
| session init + prompt add ms | `n/a` |
| SAM3.1 release cleanup ms | `n/a` |
| time to first complete group s | `34.80` |
| time to first rendered group s | `34.96` |

## GPU Sampling

- backend requested: `nvml`
- backend used: `unavailable`
- device index: `0`
- interval s: `0.500`
- samples after warmup: `196`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `gpu_util_pct` | `70.00` | `90.00` | `93.25` | `97.00` |
| `memory_util_pct` | `39.00` | `55.00` | `59.00` | `62.00` |
| `memory_used_mb` | `6105.44` | `7760.25` | `7765.31` | `8353.44` |
| `power_w` | `274.85` | `299.26` | `321.96` | `352.41` |
| `sm_clock_mhz` | `2655.00` | `2670.00` | `2670.00` | `2685.00` |
| `mem_clock_mhz` | `10251.00` | `10251.00` | `10251.00` | `10501.00` |
| `temperature_c` | `60.50` | `66.00` | `67.25` | `71.00` |

## Throughput periods

| Event | median ms | p90 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| `capture_group_period_ms` | `33.40` | `39.92` | `53.52` | `230.42` |
| `display_packet_publish_period_ms` | `232.81` | `270.19` | `385.12` | `434.21` |
| `edgetam_stage_publish_period_ms` | `68.52` | `99.51` | `106.50` | `1571.04` |
| `ffs_stage_publish_period_ms` | `58.69` | `95.76` | `106.80` | `1574.14` |
| `filter_output_publish_period_ms` | `233.79` | `263.45` | `379.38` | `434.89` |
| `fusion_publish_period_ms` | `233.79` | `263.45` | `379.37` | `434.89` |
| `gpu_owner_publish_period_ms` | `91.52` | `232.02` | `280.46` | `471.60` |
| `raw_fusion_publish_period_ms` | `233.11` | `266.99` | `383.86` | `423.34` |
| `render_period_ms` | `232.42` | `270.67` | `384.97` | `478.81` |
| `stage_join_publish_period_ms` | `91.52` | `232.02` | `280.46` | `471.60` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `12.83` | `16.49` | `21.84` | `42.28` |
| `edgetam_model_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_preprocess_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_prompt_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_postprocess_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_mask_resize_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_mask_threshold_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_mask_to_cpu_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cycle_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_batch_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_model_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_preprocess_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_cam0_model_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_cam1_model_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_cam2_model_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
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
| `gpu_owner_total_ms` | `64.62` | `70.77` | `72.76` | `101.34` |
| `gpu_owner_ffs_cycle_ms` | `62.23` | `65.04` | `65.62` | `81.76` |
| `gpu_owner_edgetam_cycle_ms` | `63.65` | `70.58` | `72.53` | `101.34` |
| `raw_fusion_total_ms` | `12.42` | `17.37` | `19.47` | `21.20` |
| `fusion_total_ms` | `89.49` | `101.61` | `237.85` | `255.58` |
| `filter_total_ms` | `76.58` | `87.42` | `224.87` | `244.13` |
| `filter_input_age_ms` | `76.61` | `87.44` | `224.89` | `244.15` |
| `object_enhanced_pt_ms` | `47.75` | `54.52` | `84.80` | `209.13` |
| `controller_pt_filter_ms` | `29.92` | `34.61` | `35.93` | `196.35` |
| `render_total_ms` | `4.08` | `5.62` | `7.35` | `17.69` |
| `render_queue_wait_ms` | `147.15` | `159.20` | `166.56` | `238.19` |
| `render_gpu_to_cpu_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_combine_ms` | `0.13` | `0.70` | `1.63` | `11.40` |
| `render_cpu_format_ms` | `0.36` | `1.05` | `2.03` | `13.95` |
| `render_open3d_points_update_ms` | `0.10` | `0.16` | `0.22` | `2.30` |
| `render_open3d_colors_update_ms` | `0.11` | `0.30` | `0.34` | `2.23` |
| `render_open3d_update_geometry_ms` | `3.50` | `4.41` | `4.71` | `10.50` |
| `render_poll_events_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_update_renderer_ms` | `0.02` | `0.03` | `0.03` | `0.04` |
| `open3d_object_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_controller_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `963` | `209.13` | `24000` | `9569` |
| `863` | `206.34` | `24000` | `9459` |
| `1227` | `204.61` | `24000` | `9625` |
| `1043` | `199.96` | `24000` | `9569` |
| `777` | `197.66` | `24000` | `9413` |
| `781` | `56.59` | `24000` | `9577` |
| `789` | `56.58` | `24000` | `9456` |
| `1126` | `56.40` | `24000` | `9568` |
| `976` | `55.70` | `24000` | `9626` |
| `1163` | `54.53` | `24000` | `9546` |
