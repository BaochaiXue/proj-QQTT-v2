# Demo 2.2 performance profile

- preset: `demo2.2-async-filter-5fps`
- canonical preset: `demo2.2-async-filter-5fps`
- target FPS: `15.00`
- capture group target FPS: `15.00`
- compile mode: `vision-reduce-overhead`
- dtype: `bfloat16`
- EdgeTAM input path: `pil`
- mask postprocess: `cuda-inline`
- render FPS after warmup: `1.83`
- raw fusion FPS after warmup: `5.80`
- filter output FPS after warmup: `5.80`
- fusion FPS after warmup: `5.80`
- Demo 2.2 PASS threshold: `14.40 FPS`
- Demo 2.2 result: `FAIL`
- groups after warmup: `967`
- complete fused groups after warmup: `408`
- rendered groups after warmup: `128`
- complete group ratio after warmup: `0.422`
- target deficit: `13.17`
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
| parallel init max wait ms | `n/a` |
| camera startup ms | `4491.60` |
| EdgeTAM model load ms | `881.93` |
| EdgeTAM compile wrap ms | `0.01` |
| EdgeTAM compile prewarm ms | `1551.32` |
| EdgeTAM warmup/first forward ms | `103.62` |
| SAM3.1 model load ms | `7840.79` |
| SAM3.1 cam0 segment ms | `8149.23` |
| SAM3.1 cam1 segment ms | `236.28` |
| SAM3.1 cam2 segment ms | `203.35` |
| FFS runner init ms | `2527.45` |
| FFS first run ms | `1093.64` |
| session init + prompt add ms | `6.76` |
| SAM3.1 release cleanup ms | `262.08` |
| time to first complete group s | `27.34` |
| time to first rendered group s | `27.91` |

## GPU Sampling

GPU sampling disabled for this run.

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `17.47` | `48.67` | `58.87` | `66.43` |
| `edgetam_model_ms` | `24.14` | `28.94` | `30.99` | `60.79` |
| `edgetam_preprocess_ms` | `1.03` | `1.28` | `1.37` | `1.76` |
| `edgetam_prompt_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_postprocess_ms` | `0.07` | `0.12` | `0.17` | `1.07` |
| `edgetam_mask_resize_ms` | `0.04` | `0.07` | `0.10` | `0.92` |
| `edgetam_mask_threshold_ms` | `0.03` | `0.05` | `0.07` | `1.03` |
| `edgetam_mask_to_cpu_ms` | `0.21` | `0.72` | `1.91` | `17.56` |
| `edgetam_total_ms` | `24.87` | `29.61` | `31.57` | `61.46` |
| `ffs_cycle_ms` | `70.20` | `75.16` | `78.67` | `272.97` |
| `ffs_batch_ms` | `51.11` | `54.72` | `56.12` | `245.73` |
| `ffs_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_model_ms` | `12.34` | `16.05` | `17.13` | `28.84` |
| `edgetam_batch_vision_total_ms` | `19.89` | `24.27` | `25.30` | `36.65` |
| `edgetam_batch_vision_preprocess_ms` | `3.09` | `3.83` | `4.11` | `5.28` |
| `edgetam_cam0_model_ms` | `24.53` | `29.34` | `31.26` | `60.79` |
| `edgetam_cam1_model_ms` | `23.62` | `28.53` | `30.27` | `40.63` |
| `edgetam_cam2_model_ms` | `24.17` | `28.58` | `31.24` | `57.48` |
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
| `ffs_stage_ms` | `1.93` | `3.46` | `4.26` | `9.84` |
| `ffs_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam0_stage_ms` | `1.93` | `3.46` | `4.26` | `9.84` |
| `ffs_cam1_stage_ms` | `1.93` | `3.46` | `4.26` | `9.84` |
| `ffs_cam2_stage_ms` | `1.93` | `3.46` | `4.26` | `9.84` |
| `ffs_cam0_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam1_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam2_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `gpu_owner_total_ms` | `165.10` | `180.67` | `187.96` | `374.81` |
| `gpu_owner_ffs_cycle_ms` | `70.20` | `75.16` | `78.67` | `272.97` |
| `gpu_owner_edgetam_cycle_ms` | `95.06` | `106.53` | `111.24` | `160.35` |
| `raw_fusion_total_ms` | `10.70` | `12.84` | `13.49` | `20.12` |
| `fusion_total_ms` | `49.20` | `54.59` | `56.09` | `252.72` |
| `filter_total_ms` | `38.09` | `43.00` | `44.83` | `243.80` |
| `filter_input_age_ms` | `38.72` | `43.62` | `45.31` | `244.50` |
| `object_enhanced_pt_ms` | `26.71` | `31.49` | `33.28` | `233.18` |
| `controller_pt_filter_ms` | `11.19` | `12.87` | `13.25` | `14.65` |
| `render_total_ms` | `0.29` | `0.47` | `1.18` | `2.28` |
| `open3d_object_update_geometry_ms` | `0.03` | `0.04` | `0.04` | `0.13` |
| `open3d_controller_update_geometry_ms` | `0.01` | `0.02` | `0.03` | `2.06` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `695` | `233.18` | `30000` | `11397` |
| `1052` | `224.99` | `30000` | `10793` |
| `610` | `223.95` | `30000` | `11545` |
| `872` | `222.63` | `30000` | `11094` |
| `521` | `219.80` | `30000` | `11489` |
| `1138` | `217.27` | `30000` | `10845` |
| `281` | `217.00` | `30000` | `11455` |
| `784` | `216.50` | `30000` | `11232` |
| `355` | `212.94` | `30000` | `11536` |
| `963` | `209.78` | `30000` | `10859` |
