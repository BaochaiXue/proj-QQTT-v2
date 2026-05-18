# Demo 2.1 performance profile

- preset: `demo2.3-dual4090-maxfps`
- canonical preset: `demo2.3-dual4090-maxfps`
- target FPS: `30.00`
- capture group target FPS: `30.00`
- compile mode: `vision-reduce-overhead`
- dtype: `bfloat16`
- EdgeTAM input path: `pil`
- mask postprocess: `hf`
- render backend: `legacy-inplace`
- render latest-only: `True`
- render copy mode: `sync-cpu`
- render FPS after warmup: `12.77`
- raw fusion FPS after warmup: `12.74`
- filter output FPS after warmup: `12.77`
- fusion FPS after warmup: `12.77`
- stage period p50 after warmup: `67.17 ms`
- display packet period p50 after warmup: `66.02 ms`
- groups after warmup: `1453`
- complete fused groups after warmup: `702`
- rendered groups after warmup: `702`
- complete group ratio after warmup: `0.483`
- stage drop count after warmup: `0`
- raw fused pending replacements total: `0`
- render buffer dropped total: `0`
- target deficit: `17.23`
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
| camera startup ms | `11522.94` |
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
| time to first complete group s | `31.36` |
| time to first rendered group s | `31.37` |

## GPU Sampling

- backend requested: `nvml`
- backend used: `unavailable`
- device index: `0`
- interval s: `0.500`
- samples after warmup: `396`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `gpu_util_pct` | `50.00` | `61.00` | `64.00` | `86.00` |
| `memory_util_pct` | `12.00` | `48.00` | `50.25` | `54.00` |
| `memory_used_mb` | `3671.88` | `18777.16` | `21805.77` | `24451.88` |
| `power_w` | `158.60` | `285.98` | `292.25` | `302.33` |
| `sm_clock_mhz` | `2655.00` | `2685.00` | `2685.00` | `2685.00` |
| `mem_clock_mhz` | `10251.00` | `10251.00` | `10251.00` | `10501.00` |
| `temperature_c` | `54.00` | `64.00` | `65.00` | `67.00` |

## Throughput periods

| Event | median ms | p90 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| `capture_group_period_ms` | `34.08` | `53.84` | `57.12` | `208.93` |
| `display_packet_publish_period_ms` | `66.02` | `111.58` | `121.64` | `326.34` |
| `edgetam_stage_publish_period_ms` | `60.86` | `74.18` | `88.41` | `217.81` |
| `ffs_stage_publish_period_ms` | `44.70` | `64.21` | `68.78` | `217.45` |
| `filter_output_publish_period_ms` | `66.02` | `111.61` | `121.65` | `326.34` |
| `fusion_publish_period_ms` | `66.02` | `111.61` | `121.65` | `326.35` |
| `gpu_owner_publish_period_ms` | `67.17` | `112.89` | `124.77` | `222.86` |
| `raw_fusion_publish_period_ms` | `67.18` | `112.89` | `124.77` | `222.86` |
| `render_period_ms` | `67.02` | `110.14` | `123.11` | `326.51` |
| `stage_join_publish_period_ms` | `67.17` | `112.89` | `124.77` | `222.86` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `5.27` | `18.77` | `28.83` | `62.76` |
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
| `gpu_owner_total_ms` | `54.63` | `58.56` | `59.63` | `102.98` |
| `gpu_owner_ffs_cycle_ms` | `39.84` | `43.20` | `44.54` | `56.41` |
| `gpu_owner_edgetam_cycle_ms` | `54.62` | `58.56` | `59.63` | `102.98` |
| `raw_fusion_total_ms` | `16.33` | `27.01` | `39.29` | `66.28` |
| `fusion_total_ms` | `69.18` | `91.48` | `104.93` | `226.14` |
| `filter_total_ms` | `51.36` | `58.56` | `64.07` | `204.71` |
| `filter_input_age_ms` | `51.98` | `59.01` | `65.74` | `205.66` |
| `object_enhanced_pt_ms` | `36.06` | `42.52` | `46.80` | `189.00` |
| `controller_pt_filter_ms` | `15.37` | `19.02` | `20.29` | `24.23` |
| `render_total_ms` | `2.22` | `4.51` | `5.81` | `15.54` |
| `render_queue_wait_ms` | `8.62` | `9.82` | `10.78` | `18.89` |
| `render_gpu_to_cpu_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_combine_ms` | `0.08` | `0.20` | `0.35` | `5.20` |
| `render_cpu_format_ms` | `0.27` | `0.59` | `1.63` | `11.15` |
| `render_open3d_points_update_ms` | `0.08` | `0.11` | `0.15` | `5.80` |
| `render_open3d_colors_update_ms` | `0.08` | `0.21` | `0.30` | `4.67` |
| `render_open3d_update_geometry_ms` | `1.83` | `3.94` | `4.93` | `14.83` |
| `render_poll_events_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_update_renderer_ms` | `0.02` | `0.03` | `0.03` | `3.01` |
| `open3d_object_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_controller_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `1551` | `189.00` | `49203` | `12300` |
| `1200` | `188.96` | `49420` | `12400` |
| `890` | `187.82` | `49470` | `12391` |
| `1416` | `187.55` | `49510` | `12304` |
| `1314` | `187.36` | `49468` | `12451` |
| `1507` | `186.95` | `49144` | `12201` |
| `1788` | `186.30` | `46558` | `11385` |
| `1463` | `186.05` | `49125` | `12229` |
| `1374` | `185.93` | `49502` | `12273` |
| `1909` | `185.67` | `46510` | `11246` |
