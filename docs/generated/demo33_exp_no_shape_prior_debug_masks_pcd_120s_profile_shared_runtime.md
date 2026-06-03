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
- render FPS after warmup: `5.48`
- raw fusion FPS after warmup: `5.47`
- filter output FPS after warmup: `5.48`
- fusion FPS after warmup: `5.48`
- stage period p50 after warmup: `113.58 ms`
- display packet period p50 after warmup: `157.91 ms`
- groups after warmup: `3068`
- complete fused groups after warmup: `549`
- rendered groups after warmup: `548`
- complete group ratio after warmup: `0.179`
- stage drop count after warmup: `4`
- raw fused pending replacements total: `0`
- render buffer dropped total: `0`
- target deficit: `24.52`
- bottleneck class: `upstream_supply`
- GPU pipeline: `dual-gpu-split`
- single-owner order: `dual_gpu_process_split`
- filter scheduler: `async`
- render filtered only: `True`
- pin memory mode: `off`
- FFS input staging: `pinned`
- H2D stream mode: `default`

## Shape Prior Warmup

- enabled: `False`
- status: `disabled`
- case dir: `result/demo32_ffs_tapnextpp/demo33_shape_prior_warmup/<run_id>/case`
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
| camera startup ms | `10765.96` |
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
| time to first complete group s | `33.69` |
| time to first rendered group s | `33.72` |

## GPU Sampling

- backend requested: `nvml`
- backend used: `unavailable`
- device index: `0`
- interval s: `0.500`
- samples after warmup: `483`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `gpu_util_pct` | `85.00` | `95.00` | `97.00` | `98.00` |
| `memory_util_pct` | `46.00` | `58.00` | `60.00` | `62.00` |
| `memory_used_mb` | `6105.44` | `7921.75` | `7954.07` | `8365.44` |
| `power_w` | `292.03` | `333.70` | `347.49` | `362.09` |
| `sm_clock_mhz` | `2655.00` | `2655.00` | `2670.00` | `2685.00` |
| `mem_clock_mhz` | `10251.00` | `10251.00` | `10251.00` | `10501.00` |
| `temperature_c` | `65.00` | `78.00` | `79.90` | `82.00` |

## Throughput periods

| Event | median ms | p90 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| `capture_group_period_ms` | `33.48` | `53.58` | `59.72` | `243.47` |
| `display_packet_publish_period_ms` | `157.91` | `291.78` | `321.68` | `545.90` |
| `edgetam_stage_publish_period_ms` | `69.36` | `98.05` | `105.74` | `277.17` |
| `ffs_stage_publish_period_ms` | `61.65` | `98.10` | `108.57` | `292.64` |
| `filter_output_publish_period_ms` | `157.84` | `294.37` | `324.52` | `549.22` |
| `fusion_publish_period_ms` | `157.84` | `294.37` | `324.51` | `549.21` |
| `gpu_owner_publish_period_ms` | `113.58` | `260.04` | `295.17` | `857.81` |
| `raw_fusion_publish_period_ms` | `157.54` | `278.66` | `291.05` | `557.82` |
| `render_period_ms` | `158.36` | `292.41` | `322.97` | `543.63` |
| `stage_join_publish_period_ms` | `113.58` | `260.04` | `295.17` | `857.81` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `11.15` | `27.16` | `27.54` | `63.26` |
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
| `gpu_owner_total_ms` | `64.81` | `70.31` | `71.99` | `101.26` |
| `gpu_owner_ffs_cycle_ms` | `61.95` | `65.07` | `65.96` | `101.26` |
| `gpu_owner_edgetam_cycle_ms` | `64.19` | `70.00` | `71.68` | `86.49` |
| `raw_fusion_total_ms` | `14.15` | `20.68` | `23.31` | `126.17` |
| `fusion_total_ms` | `93.48` | `110.64` | `253.36` | `357.63` |
| `filter_total_ms` | `78.69` | `87.35` | `239.36` | `265.11` |
| `filter_input_age_ms` | `78.72` | `87.38` | `239.39` | `265.14` |
| `object_enhanced_pt_ms` | `45.38` | `50.79` | `53.69` | `227.31` |
| `controller_pt_filter_ms` | `33.59` | `38.79` | `41.75` | `220.94` |
| `render_total_ms` | `3.68` | `4.72` | `5.85` | `16.19` |
| `render_queue_wait_ms` | `24.47` | `33.98` | `36.13` | `41.07` |
| `render_gpu_to_cpu_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_combine_ms` | `0.12` | `0.27` | `0.44` | `3.77` |
| `render_cpu_format_ms` | `0.38` | `0.69` | `0.89` | `3.97` |
| `render_open3d_points_update_ms` | `0.10` | `0.16` | `0.20` | `1.59` |
| `render_open3d_colors_update_ms` | `0.12` | `0.29` | `0.35` | `2.55` |
| `render_open3d_update_geometry_ms` | `3.15` | `4.09` | `4.69` | `12.08` |
| `render_poll_events_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_update_renderer_ms` | `0.02` | `0.03` | `0.03` | `0.05` |
| `open3d_object_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_controller_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `2546` | `227.31` | `24000` | `9133` |
| `2855` | `225.29` | `24000` | `9040` |
| `3125` | `225.18` | `24000` | `9078` |
| `2047` | `219.10` | `24000` | `9082` |
| `1642` | `217.24` | `24000` | `9019` |
| `1571` | `216.58` | `24000` | `9024` |
| `1507` | `214.63` | `24000` | `8997` |
| `1305` | `214.22` | `24000` | `9112` |
| `2620` | `213.50` | `24000` | `9079` |
| `365` | `211.34` | `24000` | `9084` |
