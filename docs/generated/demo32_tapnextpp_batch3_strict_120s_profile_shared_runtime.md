# Demo 3.2 performance profile

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
- render FPS after warmup: `5.52`
- raw fusion FPS after warmup: `5.54`
- filter output FPS after warmup: `5.53`
- fusion FPS after warmup: `5.53`
- stage period p50 after warmup: `113.52 ms`
- display packet period p50 after warmup: `157.17 ms`
- groups after warmup: `3086`
- complete fused groups after warmup: `571`
- rendered groups after warmup: `568`
- complete group ratio after warmup: `0.185`
- stage drop count after warmup: `2`
- raw fused pending replacements total: `0`
- render buffer dropped total: `2`
- target deficit: `24.48`
- bottleneck class: `upstream_supply`
- GPU pipeline: `dual-gpu-split`
- single-owner order: `dual_gpu_process_split`
- filter scheduler: `async`
- render filtered only: `True`
- pin memory mode: `off`
- FFS input staging: `pinned`
- H2D stream mode: `default`

## Init Profile

| Stage | value |
| --- | ---: |
| parallel init max wait ms | `n/a` |
| camera startup ms | `10700.87` |
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
| time to first complete group s | `30.84` |
| time to first rendered group s | `30.88` |

## GPU Sampling

- backend requested: `nvml`
- backend used: `unavailable`
- device index: `0`
- interval s: `0.500`
- samples after warmup: `484`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `gpu_util_pct` | `86.00` | `100.00` | `100.00` | `100.00` |
| `memory_util_pct` | `46.00` | `77.00` | `78.00` | `79.00` |
| `memory_used_mb` | `6609.00` | `8152.69` | `8171.27` | `8869.00` |
| `power_w` | `304.52` | `365.75` | `377.00` | `387.34` |
| `sm_clock_mhz` | `2655.00` | `2655.00` | `2655.00` | `2670.00` |
| `mem_clock_mhz` | `10251.00` | `10251.00` | `10251.00` | `10501.00` |
| `temperature_c` | `69.00` | `85.00` | `87.00` | `90.00` |

## Throughput periods

| Event | median ms | p90 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| `capture_group_period_ms` | `33.43` | `52.99` | `58.69` | `247.55` |
| `display_packet_publish_period_ms` | `157.17` | `284.89` | `330.36` | `689.27` |
| `edgetam_stage_publish_period_ms` | `68.47` | `96.81` | `103.18` | `366.76` |
| `ffs_stage_publish_period_ms` | `58.65` | `97.82` | `105.74` | `378.35` |
| `filter_output_publish_period_ms` | `158.08` | `288.34` | `322.63` | `684.46` |
| `fusion_publish_period_ms` | `158.08` | `288.35` | `322.63` | `684.45` |
| `gpu_owner_publish_period_ms` | `113.52` | `219.80` | `290.97` | `726.31` |
| `raw_fusion_publish_period_ms` | `157.53` | `277.27` | `305.81` | `694.39` |
| `render_period_ms` | `157.84` | `283.15` | `330.16` | `686.30` |
| `stage_join_publish_period_ms` | `113.52` | `219.80` | `290.97` | `726.31` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `6.05` | `24.31` | `26.35` | `64.57` |
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
| `gpu_owner_total_ms` | `64.33` | `68.63` | `70.23` | `102.70` |
| `gpu_owner_ffs_cycle_ms` | `61.80` | `64.71` | `65.86` | `85.01` |
| `gpu_owner_edgetam_cycle_ms` | `63.72` | `68.39` | `69.91` | `102.70` |
| `raw_fusion_total_ms` | `9.85` | `15.27` | `17.04` | `23.07` |
| `fusion_total_ms` | `85.83` | `96.89` | `246.41` | `277.94` |
| `filter_total_ms` | `75.54` | `83.98` | `237.51` | `267.35` |
| `filter_input_age_ms` | `75.57` | `84.01` | `237.53` | `267.38` |
| `object_enhanced_pt_ms` | `44.69` | `49.94` | `53.32` | `225.08` |
| `controller_pt_filter_ms` | `31.22` | `36.42` | `38.39` | `220.24` |
| `render_total_ms` | `4.02` | `5.33` | `6.14` | `16.10` |
| `render_queue_wait_ms` | `40.63` | `49.23` | `52.63` | `141.12` |
| `render_gpu_to_cpu_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_combine_ms` | `0.13` | `0.36` | `0.52` | `3.75` |
| `render_cpu_format_ms` | `0.41` | `0.89` | `1.69` | `11.96` |
| `render_open3d_points_update_ms` | `0.11` | `0.23` | `0.33` | `11.63` |
| `render_open3d_colors_update_ms` | `0.12` | `0.33` | `0.39` | `2.43` |
| `render_open3d_update_geometry_ms` | `3.45` | `4.39` | `4.78` | `10.20` |
| `render_poll_events_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_update_renderer_ms` | `0.02` | `0.03` | `0.03` | `0.05` |
| `open3d_object_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_controller_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `2349` | `225.08` | `24000` | `8802` |
| `2770` | `223.98` | `24000` | `8826` |
| `1423` | `217.97` | `24000` | `8776` |
| `2048` | `215.81` | `24000` | `8768` |
| `1104` | `214.13` | `24000` | `8758` |
| `1590` | `212.12` | `24000` | `8785` |
| `715` | `211.84` | `24000` | `8719` |
| `947` | `209.66` | `24000` | `8800` |
| `1804` | `208.18` | `24000` | `8777` |
| `1344` | `207.79` | `24000` | `8797` |
