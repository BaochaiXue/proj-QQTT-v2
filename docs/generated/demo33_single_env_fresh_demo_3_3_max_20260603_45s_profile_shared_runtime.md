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
- stage period p50 after warmup: `87.69 ms`
- display packet period p50 after warmup: `231.78 ms`
- groups after warmup: `1097`
- complete fused groups after warmup: `96`
- rendered groups after warmup: `95`
- complete group ratio after warmup: `0.088`
- stage drop count after warmup: `0`
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
- case dir: `/home/xinjie/proj-QQTT-v2/result/demo32_ffs_tapnextpp/demo33_shape_prior_warmup/20260603-182928/case`
- object points0: `77128`
- surface points: `0`
- interior points: `0`
- structure points: `0`
- affects tracker input: `False`
- affects live observation PCD: `False`

## Init Profile

| Stage | value |
| --- | ---: |
| parallel init max wait ms | `n/a` |
| camera startup ms | `10717.62` |
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
| time to first complete group s | `35.09` |
| time to first rendered group s | `35.27` |

## GPU Sampling

- backend requested: `nvml`
- backend used: `unavailable`
- device index: `0`
- interval s: `0.500`
- samples after warmup: `196`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `gpu_util_pct` | `63.00` | `90.00` | `95.25` | `98.00` |
| `memory_util_pct` | `39.00` | `53.00` | `57.50` | `61.00` |
| `memory_used_mb` | `6105.44` | `7921.25` | `7960.19` | `8353.44` |
| `power_w` | `269.93` | `300.39` | `303.66` | `325.01` |
| `sm_clock_mhz` | `2655.00` | `2670.00` | `2670.00` | `2670.00` |
| `mem_clock_mhz` | `10251.00` | `10251.00` | `10251.00` | `10501.00` |
| `temperature_c` | `61.00` | `70.00` | `71.00` | `73.00` |

## Throughput periods

| Event | median ms | p90 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| `capture_group_period_ms` | `33.39` | `38.95` | `55.06` | `236.01` |
| `display_packet_publish_period_ms` | `231.78` | `278.64` | `379.59` | `455.86` |
| `edgetam_stage_publish_period_ms` | `68.51` | `96.97` | `106.14` | `1464.18` |
| `ffs_stage_publish_period_ms` | `58.42` | `93.01` | `104.38` | `1515.73` |
| `filter_output_publish_period_ms` | `232.45` | `276.04` | `380.93` | `444.44` |
| `fusion_publish_period_ms` | `232.44` | `276.04` | `380.93` | `444.44` |
| `gpu_owner_publish_period_ms` | `87.69` | `250.63` | `294.54` | `434.25` |
| `raw_fusion_publish_period_ms` | `232.19` | `288.69` | `381.09` | `422.53` |
| `render_period_ms` | `232.87` | `289.64` | `376.14` | `464.26` |
| `stage_join_publish_period_ms` | `87.69` | `250.63` | `294.54` | `434.26` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `10.66` | `26.07` | `27.01` | `36.76` |
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
| `gpu_owner_total_ms` | `64.52` | `69.52` | `71.99` | `106.16` |
| `gpu_owner_ffs_cycle_ms` | `62.19` | `64.95` | `65.93` | `102.42` |
| `gpu_owner_edgetam_cycle_ms` | `63.81` | `69.31` | `71.56` | `106.16` |
| `raw_fusion_total_ms` | `12.89` | `16.98` | `19.04` | `20.84` |
| `fusion_total_ms` | `89.42` | `99.27` | `239.80` | `256.31` |
| `filter_total_ms` | `76.01` | `87.66` | `229.88` | `242.90` |
| `filter_input_age_ms` | `76.03` | `87.69` | `229.91` | `242.93` |
| `object_enhanced_pt_ms` | `46.81` | `53.13` | `91.45` | `210.26` |
| `controller_pt_filter_ms` | `29.38` | `34.65` | `37.15` | `189.08` |
| `render_total_ms` | `4.01` | `5.50` | `5.85` | `23.67` |
| `render_queue_wait_ms` | `147.45` | `158.01` | `160.98` | `176.68` |
| `render_gpu_to_cpu_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_combine_ms` | `0.13` | `0.44` | `1.45` | `11.14` |
| `render_cpu_format_ms` | `0.37` | `0.85` | `1.81` | `11.38` |
| `render_open3d_points_update_ms` | `0.11` | `0.15` | `0.20` | `1.64` |
| `render_open3d_colors_update_ms` | `0.10` | `0.17` | `0.23` | `0.45` |
| `render_open3d_update_geometry_ms` | `3.39` | `4.42` | `4.74` | `5.34` |
| `render_poll_events_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_update_renderer_ms` | `0.02` | `0.03` | `0.03` | `0.05` |
| `open3d_object_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_controller_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `1045` | `210.26` | `24000` | `9490` |
| `948` | `208.85` | `24000` | `9445` |
| `1221` | `205.57` | `24000` | `9527` |
| `1137` | `205.45` | `24000` | `9559` |
| `690` | `192.92` | `24000` | `9609` |
| `409` | `57.63` | `24000` | `9564` |
| `627` | `55.68` | `24000` | `9631` |
| `697` | `55.17` | `24000` | `9536` |
| `725` | `53.42` | `24000` | `9628` |
| `1212` | `53.29` | `24000` | `9666` |
