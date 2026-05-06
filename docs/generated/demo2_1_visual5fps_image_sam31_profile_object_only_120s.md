# Demo 2.1 visual-5fps performance profile

- preset: `visual-5fps`
- target FPS: `5.00`
- render FPS after warmup: `3.06`
- fusion FPS after warmup: `3.06`
- groups after warmup: `431`
- complete fused groups after warmup: `233`
- rendered groups after warmup: `233`
- target deficit: `1.94`
- bottleneck class: `upstream_supply`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `ffs_cycle_ms` | `197.49` | `309.45` | `332.98` | `977.97` |
| `edgetam_cam0_model_ms` | `96.99` | `130.99` | `141.07` | `443.92` |
| `edgetam_cam1_model_ms` | `97.52` | `132.62` | `143.51` | `497.76` |
| `edgetam_cam2_model_ms` | `91.81` | `129.30` | `136.23` | `348.09` |
| `edgetam_cam0_gate_wait_ms` | `13.11` | `112.98` | `120.44` | `298.42` |
| `edgetam_cam1_gate_wait_ms` | `11.57` | `107.88` | `119.53` | `321.26` |
| `edgetam_cam2_gate_wait_ms` | `16.23` | `118.42` | `123.74` | `311.99` |
| `fusion_total_ms` | `45.55` | `52.46` | `61.02` | `256.64` |
| `object_enhanced_pt_ms` | `37.47` | `43.58` | `49.30` | `247.44` |
| `controller_pt_filter_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_total_ms` | `0.43` | `0.66` | `0.75` | `3.23` |
| `open3d_object_update_geometry_ms` | `0.03` | `0.04` | `0.08` | `2.50` |
| `open3d_controller_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.01` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `548` | `247.44` | `54058` | `9908` |
| `421` | `228.68` | `54145` | `9798` |
| `469` | `226.05` | `53926` | `9793` |
| `344` | `223.94` | `54085` | `9897` |
| `381` | `223.00` | `53979` | `9931` |
| `269` | `220.51` | `53996` | `9841` |
| `511` | `220.34` | `54016` | `9715` |
| `235` | `220.10` | `54053` | `9744` |
| `306` | `218.73` | `54075` | `9850` |
| `195` | `61.06` | `54119` | `9972` |
