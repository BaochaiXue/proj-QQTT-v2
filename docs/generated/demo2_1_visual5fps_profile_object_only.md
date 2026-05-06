# Demo 2.1 visual-5fps performance profile

- preset: `visual-5fps`
- target FPS: `5.00`
- render FPS after warmup: `4.56`
- fusion FPS after warmup: `4.56`
- target deficit: `0.44`
- bottleneck class: `upstream_supply`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `ffs_cycle_ms` | `123.31` | `156.55` | `195.44` | `664.09` |
| `edgetam_cam0_model_ms` | `61.96` | `108.60` | `128.80` | `148.32` |
| `edgetam_cam1_model_ms` | `58.67` | `120.40` | `144.99` | `563.17` |
| `edgetam_cam2_model_ms` | `58.67` | `124.08` | `183.90` | `608.29` |
| `edgetam_cam0_gate_wait_ms` | `50.80` | `117.91` | `133.90` | `178.26` |
| `edgetam_cam1_gate_wait_ms` | `58.92` | `115.11` | `132.82` | `210.98` |
| `edgetam_cam2_gate_wait_ms` | `0.00` | `102.09` | `120.49` | `270.61` |
| `fusion_total_ms` | `40.29` | `47.72` | `53.15` | `62.68` |
| `object_enhanced_pt_ms` | `33.78` | `40.21` | `41.36` | `44.80` |
| `controller_pt_filter_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_total_ms` | `0.51` | `2.92` | `3.27` | `3.87` |
| `open3d_object_update_geometry_ms` | `0.03` | `1.85` | `2.24` | `2.48` |
| `open3d_controller_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `252` | `44.80` | `54156` | `9431` |
| `270` | `41.44` | `54246` | `9422` |
| `278` | `41.19` | `54194` | `9508` |
| `251` | `39.56` | `34182` | `4779` |
| `262` | `38.68` | `54181` | `9355` |
| `263` | `38.43` | `54203` | `9405` |
| `265` | `37.90` | `54166` | `9535` |
| `272` | `36.13` | `54174` | `9466` |
| `266` | `35.85` | `54223` | `9382` |
| `257` | `35.80` | `54196` | `9406` |
