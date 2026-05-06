# Demo 2.1 visual-5fps performance profile

- preset: `visual-5fps`
- target FPS: `5.00`
- render FPS after warmup: `0.24`
- fusion FPS after warmup: `0.24`
- groups after warmup: `380`
- complete fused groups after warmup: `16`
- rendered groups after warmup: `16`
- target deficit: `4.76`
- bottleneck class: `upstream_supply`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `17.18` | `24.56` | `27.66` | `33.36` |
| `ffs_cycle_ms` | `479.18` | `611.85` | `633.68` | `1252.47` |
| `ffs_gate_wait_ms` | `384.91` | `506.27` | `526.76` | `1121.48` |
| `edgetam_cam0_model_ms` | `163.74` | `189.75` | `201.08` | `227.21` |
| `edgetam_cam1_model_ms` | `162.74` | `187.88` | `197.80` | `692.19` |
| `edgetam_cam2_model_ms` | `160.10` | `190.43` | `203.81` | `699.77` |
| `edgetam_cam0_gate_wait_ms` | `86.34` | `159.08` | `170.14` | `186.64` |
| `edgetam_cam1_gate_wait_ms` | `64.54` | `149.23` | `167.45` | `222.14` |
| `edgetam_cam2_gate_wait_ms` | `52.16` | `147.73` | `158.02` | `191.71` |
| `fusion_total_ms` | `64.15` | `83.11` | `91.29` | `102.07` |
| `object_enhanced_pt_ms` | `30.25` | `38.51` | `42.22` | `47.28` |
| `controller_pt_filter_ms` | `18.73` | `23.13` | `24.17` | `26.28` |
| `render_total_ms` | `0.89` | `5.00` | `7.63` | `8.94` |
| `open3d_object_update_geometry_ms` | `0.04` | `1.93` | `3.01` | `4.19` |
| `open3d_controller_update_geometry_ms` | `0.02` | `1.36` | `1.86` | `2.43` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `199` | `47.28` | `47144` | `11564` |
| `229` | `40.54` | `47155` | `11630` |
| `372` | `36.49` | `47258` | `11606` |
| `379` | `33.59` | `47190` | `11564` |
| `420` | `33.31` | `47212` | `11578` |
| `411` | `31.69` | `47260` | `11579` |
| `475` | `30.73` | `47235` | `11615` |
| `296` | `30.73` | `47225` | `11518` |
| `224` | `29.77` | `47225` | `11601` |
| `337` | `29.61` | `47238` | `11566` |
