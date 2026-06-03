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
- render FPS after warmup: `3.59`
- raw fusion FPS after warmup: `3.60`
- filter output FPS after warmup: `3.60`
- fusion FPS after warmup: `3.60`
- stage period p50 after warmup: `106.08 ms`
- display packet period p50 after warmup: `241.88 ms`
- groups after warmup: `4839`
- complete fused groups after warmup: `575`
- rendered groups after warmup: `573`
- complete group ratio after warmup: `0.119`
- stage drop count after warmup: `7`
- raw fused pending replacements total: `0`
- render buffer dropped total: `1`
- target deficit: `26.41`
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
- case dir: `/home/xinjie/proj-QQTT-v2/result/demo32_ffs_tapnextpp/demo33_shape_prior_warmup/20260603-170703/case`
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
| camera startup ms | `10956.34` |
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
| time to first complete group s | `34.10` |
| time to first rendered group s | `34.26` |

## GPU Sampling

- backend requested: `nvml`
- backend used: `unavailable`
- device index: `0`
- interval s: `0.500`
- samples after warmup: `728`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `gpu_util_pct` | `83.00` | `100.00` | `100.00` | `100.00` |
| `memory_util_pct` | `44.00` | `57.00` | `60.00` | `62.00` |
| `memory_used_mb` | `7783.53` | `15582.81` | `20564.81` | `24150.81` |
| `power_w` | `287.87` | `346.75` | `350.00` | `364.52` |
| `sm_clock_mhz` | `2655.00` | `2670.00` | `2670.00` | `2670.00` |
| `mem_clock_mhz` | `10251.00` | `10251.00` | `10251.00` | `10501.00` |
| `temperature_c` | `65.00` | `75.30` | `78.00` | `81.00` |

## Throughput periods

| Event | median ms | p90 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| `capture_group_period_ms` | `33.38` | `47.09` | `58.49` | `269.45` |
| `display_packet_publish_period_ms` | `241.88` | `397.11` | `439.85` | `1249.92` |
| `edgetam_stage_publish_period_ms` | `73.19` | `109.93` | `123.22` | `695.47` |
| `ffs_stage_publish_period_ms` | `66.55` | `109.14` | `120.89` | `694.19` |
| `filter_output_publish_period_ms` | `243.40` | `398.09` | `431.31` | `1246.81` |
| `fusion_publish_period_ms` | `243.40` | `398.09` | `431.31` | `1246.81` |
| `gpu_owner_publish_period_ms` | `106.08` | `274.84` | `388.25` | `1292.38` |
| `raw_fusion_publish_period_ms` | `242.98` | `400.68` | `439.40` | `1240.41` |
| `render_period_ms` | `243.41` | `399.36` | `445.00` | `1249.56` |
| `stage_join_publish_period_ms` | `106.08` | `274.84` | `388.25` | `1292.38` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `18.89` | `24.62` | `27.63` | `50.64` |
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
| `gpu_owner_total_ms` | `66.05` | `98.33` | `99.97` | `126.86` |
| `gpu_owner_ffs_cycle_ms` | `62.81` | `90.81` | `93.19` | `126.86` |
| `gpu_owner_edgetam_cycle_ms` | `65.67` | `97.81` | `99.79` | `110.07` |
| `raw_fusion_total_ms` | `11.71` | `18.19` | `20.43` | `29.52` |
| `fusion_total_ms` | `90.85` | `106.62` | `256.66` | `322.79` |
| `filter_total_ms` | `78.21` | `92.76` | `244.22` | `302.67` |
| `filter_input_age_ms` | `78.24` | `92.79` | `244.26` | `302.71` |
| `object_enhanced_pt_ms` | `47.13` | `55.20` | `71.44` | `272.74` |
| `controller_pt_filter_ms` | `30.99` | `36.21` | `39.52` | `253.19` |
| `render_total_ms` | `3.98` | `5.74` | `7.00` | `18.55` |
| `render_queue_wait_ms` | `151.36` | `169.32` | `177.69` | `435.68` |
| `render_gpu_to_cpu_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_combine_ms` | `0.13` | `0.56` | `1.03` | `13.91` |
| `render_cpu_format_ms` | `0.39` | `1.10` | `2.09` | `15.44` |
| `render_open3d_points_update_ms` | `0.11` | `0.19` | `0.25` | `1.87` |
| `render_open3d_colors_update_ms` | `0.11` | `0.24` | `0.37` | `11.29` |
| `render_open3d_update_geometry_ms` | `3.39` | `4.32` | `4.93` | `12.92` |
| `render_poll_events_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_update_renderer_ms` | `0.02` | `0.03` | `0.03` | `0.05` |
| `open3d_object_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_controller_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `2965` | `272.74` | `24000` | `9546` |
| `4961` | `241.74` | `24000` | `9478` |
| `3834` | `238.92` | `24000` | `9516` |
| `3519` | `237.45` | `24000` | `9581` |
| `3622` | `236.72` | `24000` | `9545` |
| `3718` | `232.58` | `24000` | `9436` |
| `1966` | `229.83` | `24000` | `9482` |
| `4045` | `227.87` | `24000` | `9521` |
| `2454` | `226.73` | `24000` | `9504` |
| `4355` | `226.44` | `24000` | `9493` |
