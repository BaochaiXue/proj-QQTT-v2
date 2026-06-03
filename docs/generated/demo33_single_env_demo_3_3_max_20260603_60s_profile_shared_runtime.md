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
- render FPS after warmup: `4.04`
- raw fusion FPS after warmup: `4.04`
- filter output FPS after warmup: `4.04`
- fusion FPS after warmup: `4.04`
- stage period p50 after warmup: `89.23 ms`
- display packet period p50 after warmup: `235.71 ms`
- groups after warmup: `1511`
- complete fused groups after warmup: `155`
- rendered groups after warmup: `154`
- complete group ratio after warmup: `0.103`
- stage drop count after warmup: `1`
- raw fused pending replacements total: `0`
- render buffer dropped total: `0`
- target deficit: `25.96`
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
- case dir: `/home/xinjie/proj-QQTT-v2/result/demo32_ffs_tapnextpp/demo33_shape_prior_warmup/20260603-181603/case`
- object points0: `76945`
- surface points: `0`
- interior points: `0`
- structure points: `0`
- affects tracker input: `False`
- affects live observation PCD: `False`

## Init Profile

| Stage | value |
| --- | ---: |
| parallel init max wait ms | `n/a` |
| camera startup ms | `10698.36` |
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
| time to first complete group s | `35.48` |
| time to first rendered group s | `35.64` |

## GPU Sampling

- backend requested: `nvml`
- backend used: `unavailable`
- device index: `0`
- interval s: `0.500`
- samples after warmup: `256`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `gpu_util_pct` | `76.00` | `91.00` | `95.00` | `99.00` |
| `memory_util_pct` | `41.00` | `55.00` | `57.00` | `61.00` |
| `memory_used_mb` | `6105.44` | `7999.62` | `8043.67` | `8365.44` |
| `power_w` | `282.32` | `305.52` | `326.62` | `349.97` |
| `sm_clock_mhz` | `2655.00` | `2670.00` | `2670.00` | `2670.00` |
| `mem_clock_mhz` | `10251.00` | `10251.00` | `10251.00` | `10501.00` |
| `temperature_c` | `63.00` | `72.00` | `73.00` | `75.00` |

## Throughput periods

| Event | median ms | p90 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| `capture_group_period_ms` | `33.38` | `42.53` | `57.22` | `223.77` |
| `display_packet_publish_period_ms` | `235.71` | `258.23` | `386.64` | `415.93` |
| `edgetam_stage_publish_period_ms` | `68.65` | `98.58` | `106.59` | `1589.88` |
| `ffs_stage_publish_period_ms` | `59.48` | `99.72` | `106.21` | `1592.50` |
| `filter_output_publish_period_ms` | `236.19` | `258.89` | `389.81` | `417.88` |
| `fusion_publish_period_ms` | `236.19` | `258.88` | `389.81` | `417.88` |
| `gpu_owner_publish_period_ms` | `89.23` | `218.99` | `264.43` | `502.18` |
| `raw_fusion_publish_period_ms` | `237.06` | `262.29` | `389.01` | `410.14` |
| `render_period_ms` | `235.59` | `263.51` | `384.38` | `432.33` |
| `stage_join_publish_period_ms` | `89.23` | `218.99` | `264.43` | `502.18` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `12.17` | `23.89` | `24.54` | `42.23` |
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
| `gpu_owner_total_ms` | `64.44` | `70.45` | `72.33` | `87.12` |
| `gpu_owner_ffs_cycle_ms` | `62.62` | `65.04` | `66.01` | `73.22` |
| `gpu_owner_edgetam_cycle_ms` | `63.95` | `70.15` | `71.88` | `87.12` |
| `raw_fusion_total_ms` | `12.99` | `17.94` | `19.55` | `27.20` |
| `fusion_total_ms` | `90.60` | `101.61` | `242.41` | `262.54` |
| `filter_total_ms` | `77.59` | `87.79` | `231.97` | `248.85` |
| `filter_input_age_ms` | `77.61` | `87.82` | `232.00` | `248.88` |
| `object_enhanced_pt_ms` | `47.70` | `53.74` | `99.59` | `210.76` |
| `controller_pt_filter_ms` | `29.90` | `35.56` | `37.47` | `196.94` |
| `render_total_ms` | `4.05` | `5.43` | `7.02` | `19.90` |
| `render_queue_wait_ms` | `149.43` | `166.51` | `168.64` | `177.63` |
| `render_gpu_to_cpu_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_combine_ms` | `0.14` | `0.75` | `2.14` | `16.33` |
| `render_cpu_format_ms` | `0.43` | `1.44` | `2.56` | `16.69` |
| `render_open3d_points_update_ms` | `0.10` | `0.21` | `0.25` | `1.69` |
| `render_open3d_colors_update_ms` | `0.12` | `0.25` | `0.34` | `13.56` |
| `render_open3d_update_geometry_ms` | `3.39` | `4.32` | `4.47` | `6.74` |
| `render_poll_events_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_update_renderer_ms` | `0.02` | `0.03` | `0.03` | `0.04` |
| `open3d_object_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_controller_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `963` | `210.76` | `24000` | `9480` |
| `876` | `207.18` | `24000` | `9471` |
| `1600` | `207.17` | `24000` | `9596` |
| `703` | `206.48` | `24000` | `9564` |
| `1229` | `206.45` | `24000` | `9631` |
| `1139` | `204.07` | `24000` | `9617` |
| `1051` | `198.17` | `24000` | `9577` |
| `790` | `196.65` | `24000` | `9604` |
| `921` | `58.00` | `24000` | `9552` |
| `1505` | `57.31` | `24000` | `9545` |
