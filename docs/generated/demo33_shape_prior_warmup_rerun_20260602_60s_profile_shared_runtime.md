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
- render FPS after warmup: `0.00`
- raw fusion FPS after warmup: `0.00`
- filter output FPS after warmup: `0.00`
- fusion FPS after warmup: `0.00`
- stage period p50 after warmup: `0.00 ms`
- display packet period p50 after warmup: `0.00 ms`
- groups after warmup: `1609`
- complete fused groups after warmup: `0`
- rendered groups after warmup: `0`
- complete group ratio after warmup: `0.000`
- stage drop count after warmup: `0`
- raw fused pending replacements total: `0`
- render buffer dropped total: `0`
- target deficit: `30.00`
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
- status: `running`
- case dir: `/home/xinjie/proj-QQTT-v2/result/demo32_ffs_tapnextpp/demo33_shape_prior_warmup/20260602-200906/case`
- object points0: `0`
- surface points: `0`
- interior points: `0`
- structure points: `0`
- affects tracker input: `False`
- affects live observation PCD: `False`

Warning: this profile has no complete fused groups after warmup. Treat it as an initialization or missing-packet run, not as a valid visual FPS comparison.

## Init Profile

| Stage | value |
| --- | ---: |
| parallel init max wait ms | `n/a` |
| camera startup ms | `10886.73` |
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
| time to first complete group s | `n/a` |
| time to first rendered group s | `n/a` |

## GPU Sampling

- backend requested: `nvml`
- backend used: `unavailable`
- device index: `0`
- interval s: `0.500`
- samples after warmup: `256`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `gpu_util_pct` | `22.00` | `99.00` | `99.00` | `100.00` |
| `memory_util_pct` | `18.00` | `46.00` | `51.00` | `58.00` |
| `memory_used_mb` | `2935.47` | `10354.75` | `15832.12` | `17231.44` |
| `power_w` | `25.14` | `344.72` | `368.25` | `371.25` |
| `sm_clock_mhz` | `705.00` | `2655.00` | `2670.00` | `2670.00` |
| `mem_clock_mhz` | `810.00` | `10251.00` | `10251.00` | `10251.00` |
| `temperature_c` | `52.00` | `63.00` | `65.25` | `69.00` |

## Throughput periods

| Event | median ms | p90 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| `capture_group_period_ms` | `33.39` | `33.66` | `33.83` | `93.31` |
| `display_packet_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_stage_publish_period_ms` | `73.46` | `80.88` | `81.81` | `82.73` |
| `ffs_stage_publish_period_ms` | `44.43` | `55.27` | `61.22` | `85.46` |
| `filter_output_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `fusion_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `gpu_owner_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `raw_fusion_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `stage_join_publish_period_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `12.73` | `18.84` | `29.01` | `36.73` |
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
| `gpu_owner_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `gpu_owner_ffs_cycle_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `gpu_owner_edgetam_cycle_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `raw_fusion_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `fusion_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `filter_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `filter_input_age_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `object_enhanced_pt_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `controller_pt_filter_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_queue_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_gpu_to_cpu_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_combine_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_cpu_format_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_open3d_points_update_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_open3d_colors_update_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_open3d_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_poll_events_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_update_renderer_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_object_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_controller_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `0` | `0.00` | `0` | `0` |
| `1` | `0.00` | `0` | `0` |
| `2` | `0.00` | `0` | `0` |
| `3` | `0.00` | `0` | `0` |
| `4` | `0.00` | `0` | `0` |
| `5` | `0.00` | `0` | `0` |
| `6` | `0.00` | `0` | `0` |
| `7` | `0.00` | `0` | `0` |
| `8` | `0.00` | `0` | `0` |
| `9` | `0.00` | `0` | `0` |
