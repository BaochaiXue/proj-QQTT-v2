# Demo 2.2 performance profile

- preset: `demo2.2-async-filter-5fps`
- canonical preset: `demo2.2-async-filter-5fps`
- target FPS: `15.00`
- capture group target FPS: `15.00`
- render FPS after warmup: `5.13`
- raw fusion FPS after warmup: `5.13`
- filter output FPS after warmup: `5.13`
- fusion FPS after warmup: `5.13`
- groups after warmup: `364`
- complete fused groups after warmup: `71`
- rendered groups after warmup: `71`
- complete group ratio after warmup: `0.195`
- Demo 2.2 PASS threshold: `14.40 FPS`
- Demo 2.2 result: `FAIL`
- target deficit: `9.87`
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
| camera startup ms | `4334.45` |
| EdgeTAM model load ms | `792.38` |
| EdgeTAM compile wrap ms | `0.01` |
| EdgeTAM warmup/first forward ms | `800.67` |
| SAM3.1 model load ms | `7360.88` |
| SAM3.1 cam0 segment ms | `7908.74` |
| SAM3.1 cam1 segment ms | `159.50` |
| SAM3.1 cam2 segment ms | `167.19` |
| FFS runner init ms | `3105.81` |
| FFS first run ms | `1018.58` |
| session init + prompt add ms | `2.81` |
| SAM3.1 release cleanup ms | `216.82` |
| time to first complete group s | `24.10` |
| time to first rendered group s | `24.11` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `21.97` | `45.64` | `52.95` | `65.52` |
| `ffs_cycle_ms` | `86.83` | `98.76` | `100.70` | `1055.05` |
| `ffs_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_cam0_model_ms` | `31.00` | `38.64` | `42.18` | `126.65` |
| `edgetam_cam1_model_ms` | `30.56` | `36.26` | `37.72` | `40.53` |
| `edgetam_cam2_model_ms` | `30.09` | `34.06` | `36.43` | `44.07` |
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
| `ffs_stage_ms` | `0.77` | `7.34` | `7.62` | `8.37` |
| `ffs_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam0_stage_ms` | `0.63` | `1.00` | `1.20` | `1.90` |
| `ffs_cam1_stage_ms` | `6.80` | `7.85` | `8.18` | `8.37` |
| `ffs_cam2_stage_ms` | `0.78` | `1.36` | `1.50` | `3.00` |
| `ffs_cam0_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam1_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam2_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `gpu_owner_total_ms` | `188.75` | `204.31` | `219.42` | `10436.22` |
| `gpu_owner_ffs_cycle_ms` | `86.83` | `98.76` | `100.70` | `1055.05` |
| `gpu_owner_edgetam_cycle_ms` | `102.04` | `113.52` | `115.49` | `9381.09` |
| `raw_fusion_total_ms` | `8.37` | `11.13` | `11.38` | `18.02` |
| `fusion_total_ms` | `43.05` | `46.73` | `48.99` | `209.47` |
| `filter_total_ms` | `34.40` | `37.60` | `39.32` | `201.44` |
| `filter_input_age_ms` | `34.69` | `38.38` | `39.52` | `201.91` |
| `object_enhanced_pt_ms` | `21.73` | `24.43` | `25.95` | `189.04` |
| `controller_pt_filter_ms` | `12.56` | `13.29` | `14.23` | `15.57` |
| `render_total_ms` | `0.38` | `2.22` | `2.63` | `5.60` |
| `open3d_object_update_geometry_ms` | `0.03` | `1.32` | `1.82` | `2.86` |
| `open3d_controller_update_geometry_ms` | `0.01` | `0.02` | `0.20` | `2.59` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `285` | `189.04` | `45740` | `7913` |
| `380` | `188.55` | `45802` | `7961` |
| `66` | `33.94` | `45240` | `7716` |
| `244` | `26.53` | `45744` | `7859` |
| `300` | `25.36` | `45776` | `8020` |
| `263` | `24.86` | `45753` | `7891` |
| `384` | `24.67` | `45746` | `7958` |
| `314` | `24.43` | `45748` | `7903` |
| `354` | `24.26` | `45772` | `7969` |
| `317` | `24.16` | `45754` | `7902` |
