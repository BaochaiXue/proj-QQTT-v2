# Demo 2.1 visual-5fps performance profile

- preset: `visual-5fps`
- target FPS: `5.00`
- render FPS after warmup: `1.37`
- fusion FPS after warmup: `1.37`
- groups after warmup: `387`
- complete fused groups after warmup: `111`
- rendered groups after warmup: `111`
- target deficit: `3.63`
- bottleneck class: `upstream_supply`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `10.82` | `23.05` | `28.26` | `33.26` |
| `ffs_cycle_ms` | `226.24` | `380.61` | `397.65` | `684.29` |
| `ffs_gate_wait_ms` | `109.90` | `257.81` | `270.18` | `554.39` |
| `edgetam_cam0_model_ms` | `150.34` | `177.21` | `186.48` | `369.52` |
| `edgetam_cam1_model_ms` | `150.87` | `177.24` | `190.54` | `718.88` |
| `edgetam_cam2_model_ms` | `147.16` | `178.86` | `194.09` | `724.79` |
| `edgetam_cam0_gate_wait_ms` | `0.00` | `18.49` | `25.11` | `46.28` |
| `edgetam_cam1_gate_wait_ms` | `0.00` | `14.49` | `19.90` | `39.14` |
| `edgetam_cam2_gate_wait_ms` | `0.00` | `15.72` | `21.30` | `38.73` |
| `fusion_total_ms` | `36.91` | `46.96` | `54.41` | `292.77` |
| `object_enhanced_pt_ms` | `28.60` | `35.72` | `39.42` | `287.83` |
| `controller_pt_filter_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_total_ms` | `0.48` | `0.87` | `1.55` | `4.29` |
| `open3d_object_update_geometry_ms` | `0.03` | `0.06` | `1.03` | `3.12` |
| `open3d_controller_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.01` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `305` | `287.83` | `46118` | `11097` |
| `425` | `234.62` | `46224` | `11159` |
| `517` | `227.52` | `46110` | `11085` |
| `350` | `46.14` | `46061` | `11070` |
| `437` | `42.24` | `46113` | `11115` |
| `174` | `39.96` | `46160` | `11081` |
| `179` | `38.88` | `46148` | `11131` |
| `202` | `38.62` | `46165` | `11097` |
| `241` | `37.76` | `46115` | `11046` |
| `518` | `37.22` | `46136` | `11107` |
