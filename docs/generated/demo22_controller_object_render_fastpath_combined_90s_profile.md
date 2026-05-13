# Demo 2.2 performance profile

- preset: `demo2.2-async-filter-5fps`
- canonical preset: `demo2.2-async-filter-5fps`
- target FPS: `15.00`
- capture group target FPS: `15.00`
- compile mode: `vision-reduce-overhead`
- dtype: `bfloat16`
- EdgeTAM input path: `pil`
- mask postprocess: `cuda-inline`
- render backend: `legacy-inplace`
- render latest-only: `True`
- render copy mode: `async-pinned`
- render FPS after warmup: `5.69`
- Demo 2.2 PASS threshold: `14.40 FPS`
- Demo 2.2 result: `FAIL`
- raw fusion FPS after warmup: `5.69`
- filter output FPS after warmup: `5.69`
- fusion FPS after warmup: `5.69`
- groups after warmup: `933`
- complete fused groups after warmup: `400`
- rendered groups after warmup: `400`
- complete group ratio after warmup: `0.429`
- target deficit: `9.31`
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
| camera startup ms | `4531.01` |
| EdgeTAM model load ms | `751.39` |
| EdgeTAM compile wrap ms | `0.01` |
| EdgeTAM compile prewarm ms | `n/a` |
| EdgeTAM warmup/first forward ms | `115.68` |
| SAM3.1 model load ms | `7489.35` |
| SAM3.1 cam0 segment ms | `8010.23` |
| SAM3.1 cam1 segment ms | `175.33` |
| SAM3.1 cam2 segment ms | `175.52` |
| FFS runner init ms | `2425.58` |
| FFS first run ms | `1070.11` |
| session init + prompt add ms | `7.39` |
| SAM3.1 release cleanup ms | `235.10` |
| time to first complete group s | `25.75` |
| time to first rendered group s | `25.75` |

## GPU Sampling

GPU sampling disabled for this run.

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `28.69` | `48.67` | `55.82` | `66.68` |
| `edgetam_model_ms` | `23.93` | `28.06` | `29.61` | `41.96` |
| `edgetam_preprocess_ms` | `1.06` | `1.31` | `1.43` | `2.37` |
| `edgetam_prompt_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_postprocess_ms` | `0.07` | `0.11` | `0.14` | `0.58` |
| `edgetam_mask_resize_ms` | `0.04` | `0.07` | `0.09` | `0.53` |
| `edgetam_mask_threshold_ms` | `0.03` | `0.04` | `0.06` | `0.24` |
| `edgetam_mask_to_cpu_ms` | `0.21` | `0.33` | `0.59` | `7.32` |
| `edgetam_total_ms` | `24.47` | `28.59` | `30.15` | `42.43` |
| `ffs_cycle_ms` | `73.86` | `78.82` | `82.21` | `272.96` |
| `ffs_batch_ms` | `51.83` | `56.84` | `59.27` | `243.74` |
| `ffs_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_model_ms` | `12.48` | `15.89` | `16.98` | `21.96` |
| `edgetam_batch_vision_total_ms` | `20.12` | `24.89` | `26.42` | `30.83` |
| `edgetam_batch_vision_preprocess_ms` | `3.18` | `3.94` | `4.28` | `7.12` |
| `edgetam_cam0_model_ms` | `24.75` | `28.61` | `30.04` | `37.27` |
| `edgetam_cam1_model_ms` | `23.68` | `27.48` | `28.81` | `41.96` |
| `edgetam_cam2_model_ms` | `23.42` | `27.49` | `29.72` | `36.95` |
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
| `ffs_stage_ms` | `2.40` | `3.84` | `4.48` | `11.09` |
| `ffs_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam0_stage_ms` | `2.40` | `3.84` | `4.48` | `11.09` |
| `ffs_cam1_stage_ms` | `2.40` | `3.84` | `4.48` | `11.09` |
| `ffs_cam2_stage_ms` | `2.40` | `3.84` | `4.48` | `11.09` |
| `ffs_cam0_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam1_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam2_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `gpu_owner_total_ms` | `168.28` | `182.35` | `188.27` | `394.06` |
| `gpu_owner_ffs_cycle_ms` | `73.86` | `78.82` | `82.21` | `272.96` |
| `gpu_owner_edgetam_cycle_ms` | `94.23` | `104.99` | `107.81` | `127.32` |
| `raw_fusion_total_ms` | `11.30` | `12.96` | `13.54` | `16.12` |
| `fusion_total_ms` | `54.49` | `61.39` | `64.40` | `254.64` |
| `filter_total_ms` | `43.36` | `49.58` | `51.86` | `244.65` |
| `filter_input_age_ms` | `43.84` | `50.24` | `52.54` | `245.01` |
| `object_enhanced_pt_ms` | `29.12` | `34.26` | `36.61` | `227.86` |
| `controller_pt_filter_ms` | `13.88` | `15.99` | `16.46` | `18.53` |
| `render_total_ms` | `0.40` | `0.57` | `0.77` | `1.36` |
| `render_queue_wait_ms` | `9.11` | `9.72` | `9.85` | `10.18` |
| `render_gpu_to_cpu_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_combine_ms` | `0.10` | `0.18` | `0.24` | `0.91` |
| `render_cpu_format_ms` | `0.28` | `0.44` | `0.64` | `1.25` |
| `render_open3d_points_update_ms` | `0.09` | `0.13` | `0.16` | `1.09` |
| `render_open3d_colors_update_ms` | `0.06` | `0.14` | `0.17` | `0.78` |
| `render_open3d_update_geometry_ms` | `0.05` | `0.06` | `0.07` | `0.23` |
| `render_poll_events_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_update_renderer_ms` | `0.02` | `0.03` | `0.03` | `0.17` |
| `open3d_object_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_controller_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `423` | `227.86` | `47664` | `12575` |
| `1065` | `226.12` | `47658` | `12565` |
| `1147` | `220.48` | `47678` | `12492` |
| `988` | `220.13` | `47668` | `12559` |
| `905` | `219.88` | `47675` | `12570` |
| `827` | `215.93` | `47671` | `12639` |
| `746` | `213.35` | `47672` | `12624` |
| `505` | `208.22` | `47660` | `12485` |
| `670` | `208.16` | `47692` | `12561` |
| `585` | `207.59` | `47673` | `12560` |
