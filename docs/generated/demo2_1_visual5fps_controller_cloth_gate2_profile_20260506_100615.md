# Demo 2.1 visual-5fps performance profile

- preset: `visual-5fps`
- target FPS: `5.00`
- render FPS after warmup: `0.00`
- fusion FPS after warmup: `0.00`
- groups after warmup: `0`
- complete fused groups after warmup: `0`
- rendered groups after warmup: `0`
- target deficit: `5.00`
- bottleneck class: `upstream_supply`

Warning: this profile has no complete fused groups after warmup. Treat it as an initialization or missing-packet run, not as a valid visual FPS comparison.

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cycle_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_cam0_model_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_cam1_model_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_cam2_model_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_cam0_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_cam1_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_cam2_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `fusion_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `object_enhanced_pt_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `controller_pt_filter_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_object_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_controller_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
