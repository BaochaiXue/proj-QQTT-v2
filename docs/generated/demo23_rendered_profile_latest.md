# Demo 2.3 performance profile

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
- render FPS after warmup: `12.44`
- raw fusion FPS after warmup: `12.44`
- filter output FPS after warmup: `12.44`
- fusion FPS after warmup: `12.44`
- stage period p50 after warmup: `66.86 ms`
- display packet period p50 after warmup: `67.96 ms`
- groups after warmup: `2563`
- complete fused groups after warmup: `1169`
- rendered groups after warmup: `1169`
- complete group ratio after warmup: `0.456`
- stage drop count after warmup: `0`
- raw fused pending replacements total: `0`
- render buffer dropped total: `0`
- target deficit: `17.56`
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
| camera startup ms | `11191.23` |
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
| time to first complete group s | `31.49` |
| time to first rendered group s | `31.51` |

## GPU Sampling

- backend requested: `nvml`
- backend used: `unavailable`
- device index: `0`
- interval s: `0.500`
- samples after warmup: `404`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `gpu_util_pct` | `54.00` | `62.00` | `64.00` | `67.00` |
| `memory_util_pct` | `14.00` | `49.00` | `52.00` | `54.00` |
| `memory_used_mb` | `3671.88` | `6031.06` | `6069.60` | `6114.31` |
| `power_w` | `170.77` | `295.06` | `297.35` | `304.57` |
| `sm_clock_mhz` | `2655.00` | `2685.00` | `2685.00` | `2685.00` |
| `mem_clock_mhz` | `10251.00` | `10251.00` | `10251.00` | `10501.00` |
| `temperature_c` | `61.00` | `66.00` | `67.00` | `68.00` |

## Throughput periods

| Event | median ms | p90 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| `capture_group_period_ms` | `33.74` | `48.08` | `51.19` | `209.87` |
| `display_packet_publish_period_ms` | `67.96` | `114.86` | `120.93` | `284.21` |
| `edgetam_stage_publish_period_ms` | `60.06` | `69.86` | `74.33` | `224.14` |
| `ffs_stage_publish_period_ms` | `45.60` | `59.68` | `63.01` | `224.19` |
| `filter_output_publish_period_ms` | `67.96` | `114.84` | `120.93` | `284.22` |
| `fusion_publish_period_ms` | `67.96` | `114.84` | `120.93` | `284.22` |
| `gpu_owner_publish_period_ms` | `66.86` | `115.86` | `121.88` | `226.40` |
| `raw_fusion_publish_period_ms` | `66.87` | `115.87` | `121.87` | `226.40` |
| `render_period_ms` | `68.46` | `114.41` | `121.10` | `282.17` |
| `stage_join_publish_period_ms` | `66.86` | `115.86` | `121.88` | `226.40` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `8.37` | `17.00` | `22.53` | `64.56` |
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
| `gpu_owner_total_ms` | `54.08` | `57.76` | `58.87` | `67.71` |
| `gpu_owner_ffs_cycle_ms` | `40.06` | `43.59` | `44.96` | `67.71` |
| `gpu_owner_edgetam_cycle_ms` | `54.08` | `57.72` | `58.85` | `66.75` |
| `raw_fusion_total_ms` | `12.38` | `22.42` | `25.86` | `49.30` |
| `fusion_total_ms` | `57.58` | `72.73` | `77.78` | `223.67` |
| `filter_total_ms` | `45.45` | `50.86` | `53.66` | `208.42` |
| `filter_input_age_ms` | `46.05` | `51.52` | `54.64` | `208.70` |
| `object_enhanced_pt_ms` | `27.90` | `33.28` | `36.14` | `192.10` |
| `controller_pt_filter_ms` | `17.06` | `20.77` | `22.01` | `26.19` |
| `render_total_ms` | `2.06` | `4.05` | `5.21` | `27.39` |
| `render_queue_wait_ms` | `8.81` | `9.71` | `10.13` | `20.59` |
| `render_gpu_to_cpu_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_combine_ms` | `0.08` | `0.16` | `0.24` | `9.54` |
| `render_cpu_format_ms` | `0.26` | `0.45` | `0.70` | `21.22` |
| `render_open3d_points_update_ms` | `0.08` | `0.10` | `0.12` | `6.60` |
| `render_open3d_colors_update_ms` | `0.08` | `0.19` | `0.23` | `6.46` |
| `render_open3d_update_geometry_ms` | `1.72` | `3.47` | `4.77` | `22.64` |
| `render_poll_events_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_update_renderer_ms` | `0.02` | `0.03` | `0.03` | `0.09` |
| `open3d_object_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_controller_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `1949` | `192.10` | `48972` | `9741` |
| `1687` | `188.98` | `49019` | `9783` |
| `1163` | `186.37` | `49002` | `9840` |
| `3234` | `185.09` | `49033` | `9856` |
| `2612` | `183.52` | `49007` | `9830` |
| `1872` | `182.71` | `48960` | `9908` |
| `2366` | `182.16` | `48953` | `9799` |
| `1514` | `181.29` | `49023` | `9811` |
| `3087` | `181.17` | `49061` | `9860` |
| `3003` | `180.95` | `49010` | `9824` |
