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
- render FPS after warmup: `5.65`
- raw fusion FPS after warmup: `5.65`
- filter output FPS after warmup: `5.65`
- fusion FPS after warmup: `5.65`
- stage period p50 after warmup: `104.16 ms`
- display packet period p50 after warmup: `155.96 ms`
- groups after warmup: `8789`
- complete fused groups after warmup: `520`
- rendered groups after warmup: `519`
- complete group ratio after warmup: `0.059`
- stage drop count after warmup: `2`
- raw fused pending replacements total: `0`
- render buffer dropped total: `0`
- target deficit: `24.35`
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
- status: `ready`
- case dir: `/home/xinjie/proj-QQTT-v2/result/demo32_ffs_tapnextpp/demo33_shape_prior_warmup/20260601-180112/case`
- object points0: `8248`
- surface points: `433`
- interior points: `45`
- structure points: `8726`
- affects tracker input: `False`
- affects live observation PCD: `False`

## Init Profile

| Stage | value |
| --- | ---: |
| parallel init max wait ms | `n/a` |
| camera startup ms | `10725.07` |
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
| time to first complete group s | `231.38` |
| time to first rendered group s | `231.43` |

## GPU Sampling

- backend requested: `nvml`
- backend used: `unavailable`
- device index: `0`
- interval s: `0.500`
- samples after warmup: `1232`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `gpu_util_pct` | `26.00` | `92.00` | `98.00` | `100.00` |
| `memory_util_pct` | `16.00` | `55.00` | `57.00` | `61.00` |
| `memory_used_mb` | `6105.44` | `15608.81` | `15608.81` | `24258.75` |
| `power_w` | `55.21` | `319.62` | `345.63` | `390.16` |
| `sm_clock_mhz` | `2565.00` | `2655.00` | `2670.00` | `2685.00` |
| `mem_clock_mhz` | `10251.00` | `10251.00` | `10251.00` | `10501.00` |
| `temperature_c` | `47.00` | `71.00` | `76.00` | `80.00` |

## Throughput periods

| Event | median ms | p90 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| `capture_group_period_ms` | `33.39` | `35.24` | `45.36` | `236.57` |
| `display_packet_publish_period_ms` | `155.96` | `291.60` | `319.87` | `764.07` |
| `edgetam_stage_publish_period_ms` | `68.82` | `95.84` | `103.72` | `197974.29` |
| `ffs_stage_publish_period_ms` | `60.07` | `98.54` | `107.40` | `198018.64` |
| `filter_output_publish_period_ms` | `155.91` | `291.90` | `321.06` | `761.32` |
| `fusion_publish_period_ms` | `155.91` | `291.91` | `321.06` | `761.33` |
| `gpu_owner_publish_period_ms` | `104.16` | `200.73` | `287.11` | `749.11` |
| `raw_fusion_publish_period_ms` | `156.10` | `278.03` | `294.70` | `762.53` |
| `render_period_ms` | `156.11` | `292.54` | `322.12` | `762.63` |
| `stage_join_publish_period_ms` | `104.16` | `200.73` | `287.11` | `749.11` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `2.52` | `10.90` | `28.48` | `62.69` |
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
| `gpu_owner_total_ms` | `64.22` | `68.55` | `70.46` | `100.15` |
| `gpu_owner_ffs_cycle_ms` | `62.12` | `65.09` | `66.08` | `76.85` |
| `gpu_owner_edgetam_cycle_ms` | `63.61` | `68.20` | `70.30` | `100.15` |
| `raw_fusion_total_ms` | `9.40` | `13.68` | `15.42` | `23.66` |
| `fusion_total_ms` | `86.13` | `96.47` | `241.59` | `277.47` |
| `filter_total_ms` | `76.39` | `85.60` | `230.64` | `267.40` |
| `filter_input_age_ms` | `76.42` | `85.63` | `230.66` | `267.42` |
| `object_enhanced_pt_ms` | `44.74` | `50.50` | `53.52` | `234.56` |
| `controller_pt_filter_ms` | `31.89` | `36.91` | `39.42` | `218.34` |
| `render_total_ms` | `3.57` | `4.60` | `5.21` | `21.90` |
| `render_queue_wait_ms` | `38.18` | `47.22` | `50.18` | `60.82` |
| `render_gpu_to_cpu_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_combine_ms` | `0.11` | `0.25` | `0.37` | `12.92` |
| `render_cpu_format_ms` | `0.35` | `0.65` | `0.97` | `18.47` |
| `render_open3d_points_update_ms` | `0.10` | `0.15` | `0.18` | `15.06` |
| `render_open3d_colors_update_ms` | `0.11` | `0.28` | `0.35` | `11.90` |
| `render_open3d_update_geometry_ms` | `3.09` | `3.92` | `4.04` | `6.73` |
| `render_poll_events_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_update_renderer_ms` | `0.02` | `0.03` | `0.03` | `0.05` |
| `open3d_object_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_controller_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `8837` | `234.56` | `24000` | `8887` |
| `8913` | `227.43` | `24000` | `8868` |
| `8605` | `227.14` | `24000` | `9345` |
| `7121` | `223.29` | `24000` | `9450` |
| `8294` | `221.40` | `24000` | `9362` |
| `7323` | `220.00` | `24000` | `9378` |
| `7928` | `217.93` | `24000` | `9436` |
| `7252` | `217.14` | `24000` | `9355` |
| `8378` | `216.57` | `24000` | `9436` |
| `8076` | `216.45` | `24000` | `9382` |
