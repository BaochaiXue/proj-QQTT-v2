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
- render FPS after warmup: `4.09`
- raw fusion FPS after warmup: `4.06`
- filter output FPS after warmup: `4.05`
- fusion FPS after warmup: `4.05`
- stage period p50 after warmup: `84.01 ms`
- display packet period p50 after warmup: `233.26 ms`
- groups after warmup: `1511`
- complete fused groups after warmup: `160`
- rendered groups after warmup: `159`
- complete group ratio after warmup: `0.106`
- stage drop count after warmup: `4`
- raw fused pending replacements total: `0`
- render buffer dropped total: `0`
- target deficit: `25.91`
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
| camera startup ms | `10768.59` |
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
| time to first complete group s | `34.33` |
| time to first rendered group s | `34.49` |

## GPU Sampling

GPU sampling disabled for this run.

## Throughput periods

| Event | median ms | p90 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| `capture_group_period_ms` | `33.39` | `43.78` | `57.86` | `220.94` |
| `display_packet_publish_period_ms` | `233.26` | `256.07` | `387.81` | `655.09` |
| `edgetam_stage_publish_period_ms` | `68.99` | `98.04` | `105.48` | `690.01` |
| `ffs_stage_publish_period_ms` | `59.44` | `96.44` | `107.07` | `735.96` |
| `filter_output_publish_period_ms` | `233.20` | `254.89` | `383.00` | `648.31` |
| `fusion_publish_period_ms` | `233.20` | `254.89` | `383.00` | `648.31` |
| `gpu_owner_publish_period_ms` | `84.01` | `218.85` | `269.57` | `478.79` |
| `raw_fusion_publish_period_ms` | `233.30` | `255.65` | `385.43` | `478.59` |
| `render_period_ms` | `234.39` | `258.23` | `379.88` | `398.10` |
| `stage_join_publish_period_ms` | `84.01` | `218.85` | `269.57` | `478.79` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `9.76` | `21.18` | `25.03` | `60.51` |
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
| `gpu_owner_total_ms` | `64.73` | `70.07` | `71.69` | `98.97` |
| `gpu_owner_ffs_cycle_ms` | `62.30` | `65.09` | `65.92` | `80.04` |
| `gpu_owner_edgetam_cycle_ms` | `64.08` | `70.00` | `71.03` | `98.97` |
| `raw_fusion_total_ms` | `12.52` | `18.13` | `19.71` | `25.52` |
| `fusion_total_ms` | `88.65` | `101.48` | `239.12` | `264.65` |
| `filter_total_ms` | `76.45` | `87.41` | `225.81` | `257.25` |
| `filter_input_age_ms` | `76.48` | `87.44` | `225.84` | `257.28` |
| `object_enhanced_pt_ms` | `47.19` | `52.94` | `56.98` | `224.62` |
| `controller_pt_filter_ms` | `29.45` | `34.73` | `37.63` | `191.09` |
| `render_total_ms` | `3.93` | `5.08` | `5.81` | `15.68` |
| `render_queue_wait_ms` | `149.33` | `162.16` | `168.00` | `171.83` |
| `render_gpu_to_cpu_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_combine_ms` | `0.15` | `0.47` | `0.83` | `1.69` |
| `render_cpu_format_ms` | `0.39` | `0.96` | `1.12` | `12.11` |
| `render_open3d_points_update_ms` | `0.11` | `0.16` | `0.22` | `1.40` |
| `render_open3d_colors_update_ms` | `0.11` | `0.23` | `0.34` | `11.56` |
| `render_open3d_update_geometry_ms` | `3.40` | `4.29` | `4.44` | `6.99` |
| `render_poll_events_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_update_renderer_ms` | `0.02` | `0.03` | `0.03` | `0.04` |
| `open3d_object_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_controller_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `1477` | `224.62` | `24000` | `9631` |
| `1683` | `214.09` | `24000` | `9538` |
| `1203` | `209.87` | `24000` | `9523` |
| `1292` | `206.07` | `24000` | `9476` |
| `935` | `197.33` | `24000` | `9478` |
| `671` | `193.69` | `24000` | `9447` |
| `1064` | `59.49` | `24000` | `9514` |
| `1428` | `57.27` | `24000` | `9567` |
| `1340` | `56.96` | `24000` | `9424` |
| `910` | `56.20` | `24000` | `9518` |
