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
- render FPS after warmup: `3.62`
- raw fusion FPS after warmup: `3.63`
- filter output FPS after warmup: `3.63`
- fusion FPS after warmup: `3.63`
- stage period p50 after warmup: `92.53 ms`
- display packet period p50 after warmup: `256.98 ms`
- groups after warmup: `4808`
- complete fused groups after warmup: `579`
- rendered groups after warmup: `578`
- complete group ratio after warmup: `0.120`
- stage drop count after warmup: `0`
- raw fused pending replacements total: `0`
- render buffer dropped total: `0`
- target deficit: `26.38`
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
- status: `error`
- case dir: `/home/xinjie/proj-QQTT-v2/result/demo32_ffs_tapnextpp/demo33_shape_prior_warmup/20260603-170126/case`
- object points0: `0`
- surface points: `0`
- interior points: `0`
- structure points: `0`
- affects tracker input: `False`
- affects live observation PCD: `False`

## Init Profile

| Stage | value |
| --- | ---: |
| parallel init max wait ms | `n/a` |
| camera startup ms | `10716.06` |
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
| time to first complete group s | `33.88` |
| time to first rendered group s | `34.04` |

## GPU Sampling

- backend requested: `nvml`
- backend used: `unavailable`
- device index: `0`
- interval s: `0.500`
- samples after warmup: `724`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `gpu_util_pct` | `84.00` | `99.00` | `100.00` | `100.00` |
| `memory_util_pct` | `45.00` | `59.00` | `60.00` | `62.00` |
| `memory_used_mb` | `6105.50` | `12588.76` | `21369.62` | `23620.56` |
| `power_w` | `293.63` | `365.77` | `372.66` | `396.29` |
| `sm_clock_mhz` | `2655.00` | `2655.00` | `2655.00` | `2670.00` |
| `mem_clock_mhz` | `10251.00` | `10251.00` | `10251.00` | `10501.00` |
| `temperature_c` | `65.00` | `82.00` | `82.00` | `84.00` |

## Throughput periods

| Event | median ms | p90 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| `capture_group_period_ms` | `33.43` | `47.16` | `59.56` | `287.40` |
| `display_packet_publish_period_ms` | `256.98` | `324.56` | `435.12` | `515.32` |
| `edgetam_stage_publish_period_ms` | `69.47` | `104.04` | `112.34` | `711.31` |
| `ffs_stage_publish_period_ms` | `61.82` | `105.40` | `116.29` | `723.00` |
| `filter_output_publish_period_ms` | `256.75` | `318.16` | `426.38` | `505.89` |
| `fusion_publish_period_ms` | `256.76` | `318.16` | `426.38` | `505.90` |
| `gpu_owner_publish_period_ms` | `92.53` | `250.90` | `293.14` | `674.54` |
| `raw_fusion_publish_period_ms` | `258.88` | `314.74` | `425.21` | `516.03` |
| `render_period_ms` | `258.83` | `324.17` | `435.57` | `513.40` |
| `stage_join_publish_period_ms` | `92.53` | `250.90` | `293.14` | `674.54` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `15.89` | `26.50` | `32.85` | `49.99` |
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
| `gpu_owner_total_ms` | `64.92` | `70.97` | `73.02` | `115.35` |
| `gpu_owner_ffs_cycle_ms` | `61.79` | `65.17` | `66.28` | `115.35` |
| `gpu_owner_edgetam_cycle_ms` | `64.12` | `70.71` | `72.30` | `99.54` |
| `raw_fusion_total_ms` | `12.20` | `19.41` | `21.54` | `32.96` |
| `fusion_total_ms` | `94.88` | `116.52` | `266.19` | `357.73` |
| `filter_total_ms` | `81.50` | `101.04` | `253.67` | `350.28` |
| `filter_input_age_ms` | `81.53` | `101.07` | `253.70` | `350.31` |
| `object_enhanced_pt_ms` | `48.96` | `59.10` | `73.26` | `256.38` |
| `controller_pt_filter_ms` | `32.46` | `39.24` | `41.82` | `290.80` |
| `render_total_ms` | `4.11` | `6.16` | `7.70` | `24.11` |
| `render_queue_wait_ms` | `159.26` | `176.51` | `183.73` | `231.78` |
| `render_gpu_to_cpu_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_combine_ms` | `0.14` | `0.53` | `1.53` | `19.37` |
| `render_cpu_format_ms` | `0.39` | `1.02` | `2.47` | `19.96` |
| `render_open3d_points_update_ms` | `0.11` | `0.19` | `0.24` | `13.80` |
| `render_open3d_colors_update_ms` | `0.11` | `0.23` | `0.33` | `12.80` |
| `render_open3d_update_geometry_ms` | `3.49` | `4.58` | `5.76` | `20.91` |
| `render_poll_events_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_update_renderer_ms` | `0.02` | `0.03` | `0.03` | `0.08` |
| `open3d_object_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_controller_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `4807` | `256.38` | `24000` | `9520` |
| `4588` | `255.17` | `24000` | `9530` |
| `4485` | `254.79` | `24000` | `9556` |
| `4919` | `251.31` | `24000` | `9634` |
| `4131` | `242.33` | `24000` | `9462` |
| `3414` | `241.94` | `24000` | `9720` |
| `3006` | `234.13` | `24000` | `9544` |
| `3315` | `233.77` | `24000` | `9501` |
| `3506` | `233.76` | `24000` | `9611` |
| `3710` | `233.36` | `24000` | `9508` |
