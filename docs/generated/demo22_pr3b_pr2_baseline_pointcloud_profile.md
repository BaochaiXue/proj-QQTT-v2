# Demo 2.2 performance profile

- preset: `demo2.2-async-filter-5fps`
- canonical preset: `demo2.2-async-filter-5fps`
- target FPS: `15.00`
- capture group target FPS: `15.00`
- compile mode: `vision-reduce-overhead`
- dtype: `bfloat16`
- EdgeTAM input path: `pil`
- mask postprocess: `hf`
- render backend: `legacy-inplace`
- render latest-only: `True`
- render copy mode: `sync-cpu`
- render FPS after warmup: `5.28`
- Demo 2.2 PASS threshold: `14.40 FPS`
- Demo 2.2 result: `FAIL`
- raw fusion FPS after warmup: `5.28`
- filter output FPS after warmup: `5.28`
- fusion FPS after warmup: `5.28`
- stage period p50 after warmup: `182.47 ms`
- display packet period p50 after warmup: `183.06 ms`
- groups after warmup: `1267`
- complete fused groups after warmup: `475`
- rendered groups after warmup: `475`
- complete group ratio after warmup: `0.375`
- stage drop count after warmup: `9`
- raw fused pending replacements total: `0`
- render buffer dropped total: `0`
- target deficit: `9.72`
- bottleneck class: `upstream_supply`
- GPU pipeline: `single-owner`
- single-owner order: `ffs-then-edgetam`
- filter scheduler: `async`
- render filtered only: `True`
- pin memory mode: `off`
- FFS input staging: `pinned`
- H2D stream mode: `default`

## Init Profile

| Stage | value |
| --- | ---: |
| parallel init max wait ms | `3376.42` |
| camera startup ms | `8304.29` |
| EdgeTAM model load ms | `2511.69` |
| EdgeTAM compile wrap ms | `0.01` |
| EdgeTAM compile prewarm ms | `1364.98` |
| EdgeTAM warmup/first forward ms | `84.36` |
| SAM3.1 model load ms | `10655.31` |
| SAM3.1 cam0 segment ms | `390.87` |
| SAM3.1 cam1 segment ms | `199.35` |
| SAM3.1 cam2 segment ms | `200.55` |
| FFS runner init ms | `7184.64` |
| FFS first run ms | `1101.72` |
| session init + prompt add ms | `5.29` |
| SAM3.1 release cleanup ms | `257.82` |
| time to first complete group s | `18.50` |
| time to first rendered group s | `18.52` |

## GPU Sampling

- backend requested: `nvml`
- backend used: `nvml`
- device index: `0`
- interval s: `0.500`
- samples after warmup: `176`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `gpu_util_pct` | `48.50` | `53.00` | `54.25` | `59.00` |
| `memory_util_pct` | `13.00` | `16.00` | `17.00` | `21.00` |
| `memory_used_mb` | `11268.56` | `15205.56` | `15623.56` | `16032.56` |
| `power_w` | `120.70` | `154.42` | `173.38` | `243.13` |
| `sm_clock_mhz` | `180.00` | `180.00` | `180.00` | `180.00` |
| `mem_clock_mhz` | `14001.00` | `14001.00` | `14001.00` | `14001.00` |
| `temperature_c` | `68.00` | `71.00` | `72.00` | `73.00` |

## Throughput periods

| Event | median ms | p90 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| `capture_group_period_ms` | `66.81` | `84.63` | `92.56` | `330.27` |
| `display_packet_publish_period_ms` | `183.06` | `198.66` | `207.01` | `425.34` |
| `edgetam_stage_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_stage_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `filter_output_publish_period_ms` | `183.07` | `198.67` | `207.03` | `425.31` |
| `fusion_publish_period_ms` | `183.07` | `198.67` | `207.03` | `425.32` |
| `gpu_owner_publish_period_ms` | `182.47` | `198.60` | `207.32` | `420.34` |
| `raw_fusion_publish_period_ms` | `182.61` | `199.00` | `207.17` | `416.49` |
| `render_period_ms` | `182.76` | `200.14` | `207.64` | `434.56` |
| `stage_join_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `37.72` | `47.70` | `51.54` | `65.82` |
| `edgetam_model_ms` | `26.27` | `30.45` | `32.10` | `38.42` |
| `edgetam_preprocess_ms` | `1.10` | `1.37` | `1.47` | `2.07` |
| `edgetam_prompt_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_postprocess_ms` | `0.04` | `0.07` | `0.08` | `0.30` |
| `edgetam_mask_resize_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_mask_threshold_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_mask_to_cpu_ms` | `0.85` | `1.01` | `1.10` | `9.61` |
| `edgetam_total_ms` | `27.35` | `31.58` | `33.18` | `39.65` |
| `ffs_cycle_ms` | `77.06` | `83.69` | `86.56` | `295.68` |
| `ffs_batch_ms` | `51.22` | `57.29` | `61.23` | `267.18` |
| `ffs_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_model_ms` | `13.95` | `17.76` | `19.11` | `34.89` |
| `edgetam_batch_vision_total_ms` | `22.10` | `27.42` | `28.91` | `43.66` |
| `edgetam_batch_vision_preprocess_ms` | `3.29` | `4.11` | `4.41` | `6.20` |
| `edgetam_cam0_model_ms` | `26.97` | `31.21` | `32.53` | `37.37` |
| `edgetam_cam1_model_ms` | `25.94` | `29.95` | `31.35` | `34.32` |
| `edgetam_cam2_model_ms` | `25.89` | `30.07` | `32.04` | `38.42` |
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
| `ffs_stage_ms` | `1.77` | `2.52` | `2.68` | `3.74` |
| `ffs_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam0_stage_ms` | `1.77` | `2.52` | `2.68` | `3.74` |
| `ffs_cam1_stage_ms` | `1.77` | `2.52` | `2.68` | `3.74` |
| `ffs_cam2_stage_ms` | `1.77` | `2.52` | `2.68` | `3.74` |
| `ffs_cam0_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam1_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam2_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `gpu_owner_total_ms` | `182.40` | `198.51` | `207.22` | `420.29` |
| `gpu_owner_ffs_cycle_ms` | `77.06` | `83.69` | `86.56` | `295.68` |
| `gpu_owner_edgetam_cycle_ms` | `104.77` | `116.12` | `120.89` | `139.57` |
| `raw_fusion_total_ms` | `10.99` | `12.83` | `13.25` | `15.80` |
| `fusion_total_ms` | `55.08` | `62.40` | `66.04` | `275.38` |
| `filter_total_ms` | `43.99` | `50.54` | `53.54` | `263.63` |
| `filter_input_age_ms` | `44.55` | `51.08` | `54.03` | `263.92` |
| `object_enhanced_pt_ms` | `36.45` | `42.65` | `46.00` | `255.85` |
| `controller_pt_filter_ms` | `7.30` | `8.53` | `8.94` | `10.82` |
| `render_total_ms` | `1.81` | `2.25` | `2.45` | `4.64` |
| `render_queue_wait_ms` | `5.10` | `10.02` | `10.15` | `10.65` |
| `render_gpu_to_cpu_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_combine_ms` | `0.11` | `0.19` | `0.26` | `0.71` |
| `render_cpu_format_ms` | `0.29` | `0.48` | `0.69` | `0.99` |
| `render_open3d_points_update_ms` | `0.09` | `0.14` | `0.16` | `0.64` |
| `render_open3d_colors_update_ms` | `0.07` | `0.15` | `0.21` | `0.63` |
| `render_open3d_update_geometry_ms` | `1.41` | `1.80` | `1.91` | `3.67` |
| `render_poll_events_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_update_renderer_ms` | `0.03` | `0.03` | `0.04` | `0.13` |
| `open3d_object_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_controller_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `932` | `255.85` | `50729` | `14288` |
| `1098` | `250.55` | `50741` | `14292` |
| `699` | `249.46` | `50688` | `14214` |
| `1342` | `246.54` | `50535` | `14233` |
| `1426` | `239.75` | `50783` | `14320` |
| `1509` | `239.41` | `50718` | `14290` |
| `1589` | `237.05` | `50720` | `14357` |
| `623` | `234.82` | `50495` | `14315` |
| `1182` | `234.01` | `50516` | `14234` |
| `472` | `233.48` | `50525` | `14272` |
