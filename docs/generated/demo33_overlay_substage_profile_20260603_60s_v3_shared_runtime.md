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
- render FPS after warmup: `4.03`
- raw fusion FPS after warmup: `4.02`
- filter output FPS after warmup: `4.03`
- fusion FPS after warmup: `4.03`
- stage period p50 after warmup: `94.43 ms`
- display packet period p50 after warmup: `234.66 ms`
- groups after warmup: `1516`
- complete fused groups after warmup: `157`
- rendered groups after warmup: `156`
- complete group ratio after warmup: `0.104`
- stage drop count after warmup: `0`
- raw fused pending replacements total: `0`
- render buffer dropped total: `0`
- target deficit: `25.97`
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
- case dir: `/home/xinjie/proj-QQTT-v2/result/demo32_ffs_tapnextpp/demo33_shape_prior_warmup/20260603-185853/case`
- object points0: `77000`
- surface points: `0`
- interior points: `0`
- structure points: `0`
- affects tracker input: `False`
- affects live observation PCD: `False`

## Init Profile

| Stage | value |
| --- | ---: |
| parallel init max wait ms | `n/a` |
| camera startup ms | `10711.49` |
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
| time to first complete group s | `35.02` |
| time to first rendered group s | `35.20` |

## GPU Sampling

- backend requested: `nvml`
- backend used: `unavailable`
- device index: `0`
- interval s: `0.500`
- samples after warmup: `254`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `gpu_util_pct` | `75.00` | `94.00` | `97.00` | `99.00` |
| `memory_util_pct` | `40.00` | `58.00` | `59.00` | `61.00` |
| `memory_used_mb` | `6105.44` | `8078.44` | `8091.79` | `8171.69` |
| `power_w` | `281.65` | `303.29` | `316.48` | `342.03` |
| `sm_clock_mhz` | `2655.00` | `2670.00` | `2670.00` | `2670.00` |
| `mem_clock_mhz` | `10251.00` | `10251.00` | `10251.00` | `10501.00` |
| `temperature_c` | `63.00` | `71.00` | `72.00` | `74.00` |

## Throughput periods

| Event | median ms | p90 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| `capture_group_period_ms` | `33.40` | `43.99` | `57.74` | `218.14` |
| `display_packet_publish_period_ms` | `234.66` | `278.14` | `374.65` | `413.67` |
| `edgetam_stage_publish_period_ms` | `69.14` | `101.01` | `108.97` | `1140.78` |
| `ffs_stage_publish_period_ms` | `58.80` | `99.46` | `108.52` | `1176.73` |
| `filter_output_publish_period_ms` | `234.42` | `275.82` | `379.92` | `411.91` |
| `fusion_publish_period_ms` | `234.42` | `275.82` | `379.93` | `411.91` |
| `gpu_owner_publish_period_ms` | `94.43` | `228.32` | `278.47` | `620.43` |
| `raw_fusion_publish_period_ms` | `235.45` | `276.22` | `375.61` | `434.72` |
| `render_period_ms` | `235.09` | `277.65` | `372.21` | `416.07` |
| `stage_join_publish_period_ms` | `94.43` | `228.32` | `278.47` | `620.43` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `11.51` | `19.30` | `22.00` | `42.18` |
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
| `gpu_owner_total_ms` | `64.82` | `70.81` | `73.02` | `96.19` |
| `gpu_owner_ffs_cycle_ms` | `61.93` | `65.02` | `66.01` | `73.03` |
| `gpu_owner_edgetam_cycle_ms` | `64.08` | `70.64` | `72.94` | `96.19` |
| `raw_fusion_total_ms` | `12.34` | `17.77` | `18.95` | `21.92` |
| `fusion_total_ms` | `90.40` | `104.77` | `235.97` | `265.01` |
| `filter_total_ms` | `77.56` | `89.45` | `223.38` | `248.98` |
| `filter_input_age_ms` | `77.59` | `89.48` | `223.40` | `249.02` |
| `object_enhanced_pt_ms` | `47.10` | `54.33` | `57.20` | `216.55` |
| `controller_pt_filter_ms` | `30.32` | `36.26` | `44.20` | `202.85` |
| `render_total_ms` | `4.19` | `6.04` | `7.32` | `20.12` |
| `render_queue_wait_ms` | `147.98` | `162.23` | `164.76` | `177.71` |
| `render_gpu_to_cpu_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_combine_ms` | `0.13` | `0.48` | `1.32` | `4.01` |
| `render_cpu_format_ms` | `0.41` | `1.29` | `2.41` | `13.78` |
| `render_open3d_points_update_ms` | `0.11` | `0.19` | `0.23` | `0.86` |
| `render_open3d_colors_update_ms` | `0.12` | `0.31` | `0.50` | `13.55` |
| `render_open3d_update_geometry_ms` | `3.46` | `4.55` | `5.15` | `9.94` |
| `render_poll_events_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_update_renderer_ms` | `0.02` | `0.03` | `0.03` | `0.04` |
| `open3d_object_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_controller_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `1678` | `216.55` | `24000` | `9513` |
| `950` | `207.36` | `24000` | `9583` |
| `779` | `203.17` | `24000` | `9499` |
| `864` | `197.16` | `24000` | `9582` |
| `1132` | `192.69` | `24000` | `9625` |
| `823` | `58.74` | `24000` | `9575` |
| `1387` | `57.99` | `24000` | `9508` |
| `795` | `57.80` | `24000` | `9557` |
| `1318` | `57.05` | `24000` | `9447` |
| `1354` | `56.74` | `24000` | `9581` |
