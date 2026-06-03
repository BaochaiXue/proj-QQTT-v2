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
- render FPS after warmup: `5.66`
- raw fusion FPS after warmup: `5.65`
- filter output FPS after warmup: `5.66`
- fusion FPS after warmup: `5.66`
- stage period p50 after warmup: `105.24 ms`
- display packet period p50 after warmup: `155.80 ms`
- groups after warmup: `2261`
- complete fused groups after warmup: `397`
- rendered groups after warmup: `396`
- complete group ratio after warmup: `0.176`
- stage drop count after warmup: `2`
- raw fused pending replacements total: `0`
- render buffer dropped total: `0`
- target deficit: `24.34`
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
| camera startup ms | `10727.95` |
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
| time to first complete group s | `33.75` |
| time to first rendered group s | `33.79` |

## GPU Sampling

- backend requested: `nvml`
- backend used: `unavailable`
- device index: `0`
- interval s: `0.500`
- samples after warmup: `366`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `gpu_util_pct` | `85.50` | `93.00` | `97.00` | `98.00` |
| `memory_util_pct` | `45.00` | `57.00` | `59.00` | `61.00` |
| `memory_used_mb` | `6105.44` | `7963.38` | `7976.50` | `8365.44` |
| `power_w` | `294.76` | `331.39` | `343.95` | `358.15` |
| `sm_clock_mhz` | `2655.00` | `2655.00` | `2666.25` | `2670.00` |
| `mem_clock_mhz` | `10251.00` | `10251.00` | `10251.00` | `10501.00` |
| `temperature_c` | `64.00` | `77.00` | `79.00` | `80.00` |

## Throughput periods

| Event | median ms | p90 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| `capture_group_period_ms` | `33.50` | `53.06` | `58.98` | `225.24` |
| `display_packet_publish_period_ms` | `155.80` | `288.16` | `310.77` | `432.42` |
| `edgetam_stage_publish_period_ms` | `69.39` | `98.73` | `106.58` | `272.68` |
| `ffs_stage_publish_period_ms` | `59.95` | `99.02` | `110.79` | `288.46` |
| `filter_output_publish_period_ms` | `156.17` | `288.37` | `315.85` | `444.95` |
| `fusion_publish_period_ms` | `156.16` | `288.38` | `315.86` | `444.95` |
| `gpu_owner_publish_period_ms` | `105.24` | `226.34` | `290.37` | `503.46` |
| `raw_fusion_publish_period_ms` | `156.28` | `272.53` | `286.58` | `444.52` |
| `render_period_ms` | `156.64` | `287.81` | `315.48` | `449.28` |
| `stage_join_publish_period_ms` | `105.24` | `226.33` | `290.37` | `503.45` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `16.77` | `24.36` | `24.54` | `46.69` |
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
| `gpu_owner_total_ms` | `64.40` | `69.89` | `72.52` | `101.82` |
| `gpu_owner_ffs_cycle_ms` | `61.82` | `65.15` | `66.06` | `97.63` |
| `gpu_owner_edgetam_cycle_ms` | `63.90` | `69.58` | `72.45` | `101.82` |
| `raw_fusion_total_ms` | `9.74` | `14.76` | `16.39` | `22.31` |
| `fusion_total_ms` | `89.30` | `100.51` | `244.77` | `266.20` |
| `filter_total_ms` | `78.70` | `88.64` | `234.91` | `256.53` |
| `filter_input_age_ms` | `78.72` | `88.67` | `234.94` | `256.56` |
| `object_enhanced_pt_ms` | `44.38` | `50.64` | `51.78` | `224.29` |
| `controller_pt_filter_ms` | `34.34` | `39.53` | `47.12` | `210.97` |
| `render_total_ms` | `3.36` | `4.14` | `4.83` | `9.11` |
| `render_queue_wait_ms` | `36.19` | `46.13` | `48.03` | `57.88` |
| `render_gpu_to_cpu_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_combine_ms` | `0.11` | `0.31` | `0.46` | `3.74` |
| `render_cpu_format_ms` | `0.35` | `0.67` | `0.98` | `4.38` |
| `render_open3d_points_update_ms` | `0.10` | `0.15` | `0.19` | `1.76` |
| `render_open3d_colors_update_ms` | `0.11` | `0.27` | `0.31` | `1.68` |
| `render_open3d_update_geometry_ms` | `2.86` | `3.59` | `3.74` | `7.33` |
| `render_poll_events_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_update_renderer_ms` | `0.02` | `0.03` | `0.03` | `0.04` |
| `open3d_object_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_controller_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `1456` | `224.29` | `24000` | `8935` |
| `894` | `214.03` | `24000` | `8812` |
| `1887` | `213.66` | `24000` | `8862` |
| `365` | `213.01` | `24000` | `8863` |
| `1828` | `209.16` | `24000` | `8789` |
| `1020` | `209.06` | `24000` | `8896` |
| `2012` | `208.04` | `24000` | `8784` |
| `1154` | `204.58` | `24000` | `8795` |
| `1581` | `199.52` | `24000` | `8926` |
| `1701` | `198.08` | `24000` | `8775` |
