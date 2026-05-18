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
- render FPS after warmup: `12.48`
- raw fusion FPS after warmup: `12.47`
- filter output FPS after warmup: `12.48`
- fusion FPS after warmup: `12.48`
- stage period p50 after warmup: `67.95 ms`
- display packet period p50 after warmup: `67.62 ms`
- groups after warmup: `1479`
- complete fused groups after warmup: `536`
- rendered groups after warmup: `536`
- complete group ratio after warmup: `0.362`
- stage drop count after warmup: `1`
- raw fused pending replacements total: `0`
- render buffer dropped total: `0`
- target deficit: `17.52`
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
| camera startup ms | `11513.95` |
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
| time to first complete group s | `31.48` |
| time to first rendered group s | `31.49` |

## GPU Sampling

- backend requested: `nvml`
- backend used: `unavailable`
- device index: `0`
- interval s: `0.500`
- samples after warmup: `248`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `gpu_util_pct` | `53.00` | `60.30` | `63.65` | `100.00` |
| `memory_util_pct` | `14.00` | `47.30` | `51.00` | `98.00` |
| `memory_used_mb` | `3671.88` | `15919.71` | `17809.77` | `17869.12` |
| `power_w` | `166.18` | `292.24` | `294.77` | `301.22` |
| `sm_clock_mhz` | `2655.00` | `2685.00` | `2685.00` | `2685.00` |
| `mem_clock_mhz` | `10251.00` | `10251.00` | `10251.00` | `10501.00` |
| `temperature_c` | `58.00` | `63.00` | `64.00` | `66.00` |

## Throughput periods

| Event | median ms | p90 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| `capture_group_period_ms` | `33.51` | `50.61` | `55.13` | `205.68` |
| `display_packet_publish_period_ms` | `67.62` | `111.65` | `126.43` | `287.77` |
| `edgetam_stage_publish_period_ms` | `61.28` | `73.41` | `86.07` | `222.47` |
| `ffs_stage_publish_period_ms` | `44.86` | `61.40` | `66.23` | `222.12` |
| `filter_output_publish_period_ms` | `67.62` | `111.66` | `126.44` | `287.77` |
| `fusion_publish_period_ms` | `67.62` | `111.66` | `126.44` | `287.77` |
| `gpu_owner_publish_period_ms` | `67.95` | `114.57` | `127.04` | `254.13` |
| `raw_fusion_publish_period_ms` | `67.94` | `114.58` | `127.04` | `254.12` |
| `render_period_ms` | `68.84` | `113.50` | `126.31` | `282.37` |
| `stage_join_publish_period_ms` | `67.95` | `114.57` | `127.04` | `254.13` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `11.13` | `27.04` | `27.56` | `51.62` |
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
| `gpu_owner_total_ms` | `54.19` | `57.96` | `58.92` | `98.69` |
| `gpu_owner_ffs_cycle_ms` | `39.63` | `42.93` | `44.12` | `59.06` |
| `gpu_owner_edgetam_cycle_ms` | `54.19` | `57.96` | `58.88` | `98.69` |
| `raw_fusion_total_ms` | `15.06` | `26.76` | `38.36` | `50.56` |
| `fusion_total_ms` | `66.91` | `90.12` | `104.38` | `222.50` |
| `filter_total_ms` | `50.83` | `59.01` | `63.16` | `202.22` |
| `filter_input_age_ms` | `51.54` | `59.86` | `64.60` | `203.24` |
| `object_enhanced_pt_ms` | `35.15` | `42.64` | `47.50` | `188.93` |
| `controller_pt_filter_ms` | `15.60` | `19.17` | `19.89` | `22.67` |
| `render_total_ms` | `2.15` | `4.50` | `5.71` | `21.32` |
| `render_queue_wait_ms` | `8.54` | `9.75` | `10.64` | `16.97` |
| `render_gpu_to_cpu_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_combine_ms` | `0.08` | `0.23` | `0.58` | `7.46` |
| `render_cpu_format_ms` | `0.29` | `0.65` | `1.40` | `8.22` |
| `render_open3d_points_update_ms` | `0.08` | `0.12` | `0.15` | `1.66` |
| `render_open3d_colors_update_ms` | `0.09` | `0.21` | `0.30` | `4.54` |
| `render_open3d_update_geometry_ms` | `1.76` | `3.59` | `4.96` | `16.58` |
| `render_poll_events_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_update_renderer_ms` | `0.02` | `0.03` | `0.03` | `0.07` |
| `open3d_object_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_controller_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `877` | `188.93` | `50013` | `12415` |
| `567` | `183.40` | `50085` | `12502` |
| `1378` | `179.59` | `46474` | `11446` |
| `507` | `179.30` | `49768` | `12393` |
| `982` | `179.09` | `50144` | `12622` |
| `1083` | `178.06` | `50064` | `12550` |
| `1495` | `177.59` | `46293` | `11315` |
| `619` | `177.46` | `50096` | `12618` |
| `1192` | `177.19` | `46384` | `11368` |
| `1028` | `176.47` | `49971` | `12496` |
