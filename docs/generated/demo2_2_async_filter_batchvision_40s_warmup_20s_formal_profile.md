# Demo 2.2 performance profile

- preset: `demo2.2-async-filter-5fps`
- canonical preset: `demo2.2-async-filter-5fps`
- target FPS: `15.00`
- capture group target FPS: `15.00`
- render FPS after warmup: `5.25`
- raw fusion FPS after warmup: `5.25`
- filter output FPS after warmup: `5.25`
- fusion FPS after warmup: `5.25`
- groups after warmup: `332`
- complete fused groups after warmup: `133`
- rendered groups after warmup: `132`
- complete group ratio after warmup: `0.401`
- Demo 2.2 PASS threshold: `14.40 FPS`
- Demo 2.2 result: `FAIL`
- target deficit: `9.75`
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
| camera startup ms | `4463.70` |
| EdgeTAM model load ms | `877.03` |
| EdgeTAM compile wrap ms | `0.01` |
| EdgeTAM compile prewarm ms | `1363.83` |
| EdgeTAM warmup/first forward ms | `111.69` |
| SAM3.1 model load ms | `9049.89` |
| SAM3.1 cam0 segment ms | `9560.74` |
| SAM3.1 cam1 segment ms | `236.37` |
| SAM3.1 cam2 segment ms | `270.36` |
| FFS runner init ms | `2516.25` |
| FFS first run ms | `998.33` |
| session init + prompt add ms | `7.78` |
| SAM3.1 release cleanup ms | `338.72` |
| time to first complete group s | `26.54` |
| time to first rendered group s | `26.56` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `26.05` | `49.77` | `51.40` | `66.07` |
| `ffs_cycle_ms` | `73.69` | `81.54` | `97.09` | `299.16` |
| `ffs_batch_ms` | `48.54` | `57.16` | `65.93` | `268.16` |
| `ffs_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_model_ms` | `13.34` | `17.74` | `19.13` | `42.88` |
| `edgetam_batch_vision_total_ms` | `20.97` | `27.25` | `28.67` | `49.76` |
| `edgetam_batch_vision_preprocess_ms` | `3.19` | `4.45` | `4.85` | `5.59` |
| `edgetam_cam0_model_ms` | `27.74` | `34.31` | `35.35` | `72.02` |
| `edgetam_cam1_model_ms` | `27.50` | `32.18` | `35.54` | `60.47` |
| `edgetam_cam2_model_ms` | `27.26` | `35.95` | `38.85` | `50.33` |
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
| `ffs_stage_ms` | `2.26` | `3.77` | `4.22` | `9.74` |
| `ffs_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam0_stage_ms` | `2.26` | `3.73` | `4.21` | `9.74` |
| `ffs_cam1_stage_ms` | `2.26` | `3.73` | `4.21` | `9.74` |
| `ffs_cam2_stage_ms` | `2.26` | `3.73` | `4.21` | `9.74` |
| `ffs_cam0_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam1_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam2_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `gpu_owner_total_ms` | `182.64` | `209.66` | `237.42` | `401.49` |
| `gpu_owner_ffs_cycle_ms` | `73.69` | `81.54` | `97.09` | `299.16` |
| `gpu_owner_edgetam_cycle_ms` | `107.46` | `125.00` | `130.14` | `176.25` |
| `raw_fusion_total_ms` | `12.25` | `14.44` | `15.55` | `17.17` |
| `fusion_total_ms` | `54.59` | `63.61` | `66.63` | `280.56` |
| `filter_total_ms` | `41.84` | `50.75` | `52.84` | `263.39` |
| `filter_input_age_ms` | `42.45` | `50.99` | `53.27` | `263.53` |
| `object_enhanced_pt_ms` | `27.56` | `34.99` | `36.47` | `249.32` |
| `controller_pt_filter_ms` | `14.12` | `16.93` | `17.77` | `27.48` |
| `render_total_ms` | `0.52` | `0.77` | `0.82` | `2.43` |
| `open3d_object_update_geometry_ms` | `0.03` | `0.04` | `0.06` | `1.93` |
| `open3d_controller_update_geometry_ms` | `0.01` | `0.02` | `0.02` | `1.25` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `327` | `278.95` | `45577` | `7981` |
| `410` | `249.32` | `45603` | `7945` |
| `669` | `249.08` | `45530` | `7947` |
| `497` | `216.87` | `45538` | `7926` |
| `579` | `216.74` | `45577` | `7911` |
| `666` | `56.73` | `45551` | `7899` |
| `255` | `56.34` | `45504` | `7878` |
| `642` | `45.54` | `45518` | `7901` |
| `286` | `43.26` | `45544` | `7927` |
| `292` | `37.79` | `45492` | `7935` |
