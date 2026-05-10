# Demo 2.2 performance profile

- preset: `demo2.2-async-filter-5fps`
- canonical preset: `demo2.2-async-filter-5fps`
- target FPS: `15.00`
- capture group target FPS: `15.00`
- render FPS after warmup: `4.99`
- raw fusion FPS after warmup: `4.98`
- filter output FPS after warmup: `4.98`
- fusion FPS after warmup: `4.98`
- groups after warmup: `347`
- complete fused groups after warmup: `80`
- rendered groups after warmup: `79`
- complete group ratio after warmup: `0.231`
- Demo 2.2 PASS threshold: `14.40 FPS`
- Demo 2.2 result: `FAIL`
- target deficit: `10.01`
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
| camera startup ms | `4434.37` |
| EdgeTAM model load ms | `813.05` |
| EdgeTAM compile wrap ms | `0.01` |
| EdgeTAM compile prewarm ms | `1609.29` |
| EdgeTAM warmup/first forward ms | `90.23` |
| SAM3.1 model load ms | `7686.35` |
| SAM3.1 cam0 segment ms | `7951.17` |
| SAM3.1 cam1 segment ms | `160.93` |
| SAM3.1 cam2 segment ms | `161.26` |
| FFS runner init ms | `3834.07` |
| FFS first run ms | `1211.36` |
| session init + prompt add ms | `3.94` |
| SAM3.1 release cleanup ms | `214.58` |
| time to first complete group s | `29.36` |
| time to first rendered group s | `29.37` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `25.95` | `46.16` | `55.34` | `64.68` |
| `ffs_cycle_ms` | `91.38` | `96.10` | `97.33` | `267.37` |
| `ffs_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_cam0_model_ms` | `31.25` | `38.05` | `40.10` | `42.41` |
| `edgetam_cam1_model_ms` | `30.64` | `35.52` | `37.37` | `50.09` |
| `edgetam_cam2_model_ms` | `30.38` | `35.64` | `37.12` | `42.14` |
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
| `ffs_stage_ms` | `1.11` | `7.94` | `8.15` | `9.19` |
| `ffs_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam0_stage_ms` | `0.79` | `1.39` | `1.59` | `2.24` |
| `ffs_cam1_stage_ms` | `7.59` | `8.41` | `8.56` | `9.19` |
| `ffs_cam2_stage_ms` | `0.86` | `1.90` | `2.30` | `7.62` |
| `ffs_cam0_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam1_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam2_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `gpu_owner_total_ms` | `193.34` | `211.36` | `219.72` | `371.23` |
| `gpu_owner_ffs_cycle_ms` | `91.38` | `96.10` | `97.33` | `267.37` |
| `gpu_owner_edgetam_cycle_ms` | `102.07` | `116.73` | `120.77` | `129.40` |
| `raw_fusion_total_ms` | `8.96` | `10.18` | `11.01` | `11.79` |
| `fusion_total_ms` | `45.91` | `49.49` | `50.88` | `223.29` |
| `filter_total_ms` | `37.12` | `40.78` | `41.60` | `214.13` |
| `filter_input_age_ms` | `37.44` | `40.99` | `42.10` | `214.56` |
| `object_enhanced_pt_ms` | `23.84` | `27.44` | `28.29` | `199.70` |
| `controller_pt_filter_ms` | `13.15` | `13.92` | `14.41` | `14.95` |
| `render_total_ms` | `0.37` | `0.92` | `2.74` | `4.28` |
| `open3d_object_update_geometry_ms` | `0.02` | `0.04` | `1.75` | `3.27` |
| `open3d_controller_update_geometry_ms` | `0.01` | `0.02` | `0.02` | `1.87` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `415` | `199.70` | `45604` | `7978` |
| `323` | `188.91` | `45588` | `7919` |
| `95` | `29.46` | `45273` | `7698` |
| `421` | `28.71` | `45587` | `7999` |
| `450` | `28.61` | `45672` | `7904` |
| `355` | `28.27` | `45634` | `7940` |
| `449` | `27.77` | `45655` | `7934` |
| `360` | `27.70` | `45591` | `7882` |
| `432` | `27.58` | `45577` | `7974` |
| `438` | `27.43` | `45598` | `7883` |
