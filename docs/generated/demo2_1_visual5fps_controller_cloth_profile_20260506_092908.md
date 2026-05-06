# Demo 2.1 visual-5fps performance profile

- preset: `visual-5fps`
- target FPS: `5.00`
- render FPS after warmup: `0.84`
- fusion FPS after warmup: `0.84`
- groups after warmup: `359`
- complete fused groups after warmup: `68`
- rendered groups after warmup: `68`
- target deficit: `4.16`
- bottleneck class: `upstream_supply`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `10.41` | `24.71` | `29.81` | `33.26` |
| `ffs_cycle_ms` | `399.09` | `513.85` | `537.91` | `845.74` |
| `ffs_gate_wait_ms` | `295.76` | `411.46` | `433.41` | `708.04` |
| `edgetam_cam0_model_ms` | `145.42` | `171.90` | `186.96` | `535.20` |
| `edgetam_cam1_model_ms` | `143.61` | `171.37` | `179.59` | `222.48` |
| `edgetam_cam2_model_ms` | `145.33` | `175.52` | `187.81` | `541.25` |
| `edgetam_cam0_gate_wait_ms` | `57.42` | `131.70` | `142.28` | `191.09` |
| `edgetam_cam1_gate_wait_ms` | `61.98` | `128.08` | `135.59` | `297.93` |
| `edgetam_cam2_gate_wait_ms` | `60.81` | `133.02` | `142.93` | `194.66` |
| `fusion_total_ms` | `56.82` | `67.50` | `74.46` | `230.69` |
| `object_enhanced_pt_ms` | `27.27` | `32.67` | `34.85` | `203.53` |
| `controller_pt_filter_ms` | `15.84` | `19.91` | `20.31` | `25.58` |
| `render_total_ms` | `0.60` | `2.23` | `2.63` | `5.30` |
| `open3d_object_update_geometry_ms` | `0.03` | `0.05` | `0.12` | `4.37` |
| `open3d_controller_update_geometry_ms` | `0.01` | `0.34` | `1.47` | `2.11` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `297` | `203.53` | `46187` | `11085` |
| `389` | `40.39` | `46109` | `11009` |
| `163` | `37.23` | `45965` | `11072` |
| `406` | `34.85` | `46155` | `11053` |
| `387` | `34.84` | `46131` | `11041` |
| `241` | `33.96` | `46121` | `11067` |
| `269` | `33.79` | `46085` | `11092` |
| `414` | `32.20` | `46140` | `11046` |
| `391` | `32.09` | `46118` | `11110` |
| `365` | `31.90` | `46191` | `11057` |
