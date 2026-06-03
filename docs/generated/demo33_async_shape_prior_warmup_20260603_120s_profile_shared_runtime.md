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
- render FPS after warmup: `3.45`
- raw fusion FPS after warmup: `3.46`
- filter output FPS after warmup: `3.46`
- fusion FPS after warmup: `3.46`
- stage period p50 after warmup: `104.30 ms`
- display packet period p50 after warmup: `241.81 ms`
- groups after warmup: `3183`
- complete fused groups after warmup: `344`
- rendered groups after warmup: `343`
- complete group ratio after warmup: `0.108`
- stage drop count after warmup: `1`
- raw fused pending replacements total: `0`
- render buffer dropped total: `0`
- target deficit: `26.55`
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
- case dir: `/home/xinjie/proj-QQTT-v2/result/demo32_ffs_tapnextpp/demo33_shape_prior_warmup/20260603-164950/case`
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
| camera startup ms | `10735.09` |
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
| time to first complete group s | `34.22` |
| time to first rendered group s | `34.37` |

## GPU Sampling

- backend requested: `nvml`
- backend used: `unavailable`
- device index: `0`
- interval s: `0.500`
- samples after warmup: `492`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `gpu_util_pct` | `84.50` | `100.00` | `100.00` | `100.00` |
| `memory_util_pct` | `44.00` | `58.00` | `60.00` | `61.00` |
| `memory_used_mb` | `7868.25` | `17254.75` | `21436.75` | `24050.75` |
| `power_w` | `284.32` | `345.46` | `350.10` | `359.06` |
| `sm_clock_mhz` | `2655.00` | `2670.00` | `2670.00` | `2670.00` |
| `mem_clock_mhz` | `10251.00` | `10251.00` | `10251.00` | `10501.00` |
| `temperature_c` | `65.00` | `73.00` | `74.00` | `77.00` |

## Throughput periods

| Event | median ms | p90 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| `capture_group_period_ms` | `33.41` | `44.67` | `55.79` | `258.98` |
| `display_packet_publish_period_ms` | `241.81` | `414.08` | `570.69` | `1201.42` |
| `edgetam_stage_publish_period_ms` | `78.37` | `111.85` | `127.82` | `584.51` |
| `ffs_stage_publish_period_ms` | `68.61` | `109.74` | `123.60` | `609.36` |
| `filter_output_publish_period_ms` | `241.40` | `414.37` | `560.22` | `1192.02` |
| `fusion_publish_period_ms` | `241.41` | `414.38` | `560.23` | `1192.03` |
| `gpu_owner_publish_period_ms` | `104.30` | `299.26` | `417.42` | `1336.99` |
| `raw_fusion_publish_period_ms` | `241.30` | `421.74` | `558.82` | `1179.16` |
| `render_period_ms` | `241.72` | `417.58` | `572.13` | `1194.91` |
| `stage_join_publish_period_ms` | `104.30` | `299.26` | `417.42` | `1336.99` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `14.64` | `25.28` | `30.38` | `43.74` |
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
| `gpu_owner_total_ms` | `68.13` | `99.70` | `100.45` | `105.65` |
| `gpu_owner_ffs_cycle_ms` | `63.54` | `92.46` | `93.78` | `105.65` |
| `gpu_owner_edgetam_cycle_ms` | `67.73` | `99.52` | `100.27` | `104.66` |
| `raw_fusion_total_ms` | `11.94` | `17.28` | `19.21` | `23.05` |
| `fusion_total_ms` | `89.85` | `104.22` | `250.01` | `297.74` |
| `filter_total_ms` | `76.75` | `93.51` | `236.00` | `276.42` |
| `filter_input_age_ms` | `76.78` | `93.53` | `236.03` | `276.45` |
| `object_enhanced_pt_ms` | `46.36` | `55.11` | `60.17` | `247.35` |
| `controller_pt_filter_ms` | `30.37` | `36.36` | `40.31` | `226.01` |
| `render_total_ms` | `3.94` | `5.08` | `5.65` | `18.22` |
| `render_queue_wait_ms` | `148.78` | `164.81` | `169.47` | `229.92` |
| `render_gpu_to_cpu_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_combine_ms` | `0.13` | `0.50` | `0.98` | `14.37` |
| `render_cpu_format_ms` | `0.39` | `0.97` | `1.78` | `14.71` |
| `render_open3d_points_update_ms` | `0.11` | `0.19` | `0.25` | `1.60` |
| `render_open3d_colors_update_ms` | `0.11` | `0.23` | `0.34` | `11.22` |
| `render_open3d_update_geometry_ms` | `3.33` | `4.31` | `4.46` | `12.03` |
| `render_poll_events_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_update_renderer_ms` | `0.02` | `0.03` | `0.03` | `0.06` |
| `open3d_object_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_controller_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `2366` | `247.35` | `24000` | `9543` |
| `2568` | `230.50` | `24000` | `9487` |
| `2189` | `227.99` | `24000` | `9555` |
| `2959` | `225.63` | `24000` | `9562` |
| `2276` | `225.61` | `24000` | `9485` |
| `1905` | `224.30` | `24000` | `9519` |
| `3081` | `217.01` | `24000` | `9586` |
| `1443` | `213.00` | `24000` | `9541` |
| `1619` | `211.05` | `24000` | `9527` |
| `958` | `207.01` | `24000` | `9547` |
