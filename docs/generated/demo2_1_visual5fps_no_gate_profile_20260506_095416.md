# Demo 2.1 visual-5fps performance profile

- preset: `visual-5fps-no-gate`
- target FPS: `5.00`
- render FPS after warmup: `3.74`
- fusion FPS after warmup: `3.74`
- groups after warmup: `376`
- complete fused groups after warmup: `293`
- rendered groups after warmup: `292`
- target deficit: `1.26`
- bottleneck class: `upstream_supply`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `10.55` | `27.79` | `30.20` | `33.17` |
| `ffs_cycle_ms` | `136.71` | `161.67` | `169.59` | `343.57` |
| `ffs_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_cam0_model_ms` | `212.66` | `242.46` | `258.46` | `449.46` |
| `edgetam_cam1_model_ms` | `211.58` | `245.79` | `261.10` | `677.46` |
| `edgetam_cam2_model_ms` | `209.61` | `243.11` | `254.63` | `669.69` |
| `edgetam_cam0_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_cam1_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_cam2_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `fusion_total_ms` | `40.53` | `49.65` | `52.52` | `246.41` |
| `object_enhanced_pt_ms` | `30.95` | `37.57` | `39.78` | `237.58` |
| `controller_pt_filter_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_total_ms` | `0.46` | `1.06` | `1.55` | `3.21` |
| `open3d_object_update_geometry_ms` | `0.03` | `0.04` | `0.05` | `2.78` |
| `open3d_controller_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.01` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `430` | `237.58` | `46093` | `11080` |
| `339` | `227.85` | `46086` | `11072` |
| `200` | `222.12` | `46081` | `11133` |
| `384` | `212.11` | `46074` | `11065` |
| `296` | `210.24` | `46151` | `11178` |
| `247` | `209.69` | `46055` | `11069` |
| `478` | `209.44` | `46104` | `11114` |
| `263` | `44.71` | `46045` | `11164` |
| `349` | `43.45` | `46109` | `11018` |
| `170` | `42.02` | `46118` | `11102` |
