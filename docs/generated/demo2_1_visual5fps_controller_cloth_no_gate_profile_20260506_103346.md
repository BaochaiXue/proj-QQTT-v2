# Demo 2.1 visual-5fps performance profile

- preset: `visual-5fps-no-gate`
- target FPS: `5.00`
- render FPS after warmup: `1.13`
- fusion FPS after warmup: `1.13`
- groups after warmup: `430`
- complete fused groups after warmup: `82`
- rendered groups after warmup: `82`
- target deficit: `3.87`
- bottleneck class: `upstream_supply`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `9.75` | `9.75` | `9.75` | `9.75` |
| `ffs_cycle_ms` | `119.03` | `145.49` | `154.87` | `321.60` |
| `ffs_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_cam0_model_ms` | `257.33` | `309.75` | `331.20` | `975.38` |
| `edgetam_cam1_model_ms` | `259.68` | `307.37` | `321.00` | `966.68` |
| `edgetam_cam2_model_ms` | `269.45` | `314.23` | `328.04` | `491.26` |
| `edgetam_cam0_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_cam1_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_cam2_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `fusion_total_ms` | `66.17` | `74.25` | `80.35` | `255.03` |
| `object_enhanced_pt_ms` | `29.35` | `34.95` | `38.66` | `214.97` |
| `controller_pt_filter_ms` | `17.67` | `20.69` | `21.80` | `26.82` |
| `render_total_ms` | `0.87` | `1.64` | `2.22` | `6.20` |
| `open3d_object_update_geometry_ms` | `0.03` | `0.04` | `0.10` | `2.29` |
| `open3d_controller_update_geometry_ms` | `0.01` | `0.02` | `0.03` | `1.30` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `337` | `214.97` | `47315` | `11615` |
| `469` | `191.29` | `47318` | `11618` |
| `190` | `43.38` | `47320` | `11619` |
| `485` | `39.56` | `47316` | `11619` |
| `265` | `38.81` | `47313` | `11615` |
| `331` | `35.81` | `47315` | `11619` |
| `514` | `35.74` | `47313` | `11618` |
| `251` | `35.21` | `47312` | `11617` |
| `289` | `34.98` | `47319` | `11618` |
| `536` | `34.69` | `47316` | `11616` |
