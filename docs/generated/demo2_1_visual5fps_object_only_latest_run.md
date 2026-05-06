# Demo 2.1 visual-5fps performance profile

- preset: `visual-5fps`
- target FPS: `5.00`
- render FPS after warmup: `2.42`
- fusion FPS after warmup: `2.42`
- groups after warmup: `399`
- complete fused groups after warmup: `186`
- rendered groups after warmup: `186`
- target deficit: `2.58`
- bottleneck class: `upstream_supply`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `14.32` | `25.11` | `28.81` | `33.24` |
| `ffs_cycle_ms` | `193.17` | `312.14` | `324.02` | `691.97` |
| `ffs_gate_wait_ms` | `80.64` | `199.46` | `211.92` | `570.57` |
| `edgetam_cam0_model_ms` | `88.73` | `129.93` | `138.50` | `485.08` |
| `edgetam_cam1_model_ms` | `88.31` | `123.84` | `132.55` | `344.00` |
| `edgetam_cam2_model_ms` | `90.56` | `126.83` | `134.85` | `439.89` |
| `edgetam_cam0_gate_wait_ms` | `14.47` | `98.70` | `111.22` | `126.64` |
| `edgetam_cam1_gate_wait_ms` | `3.02` | `102.71` | `118.60` | `317.08` |
| `edgetam_cam2_gate_wait_ms` | `15.34` | `99.94` | `118.00` | `304.43` |
| `fusion_total_ms` | `46.10` | `53.27` | `57.87` | `241.28` |
| `object_enhanced_pt_ms` | `37.90` | `44.18` | `46.93` | `231.98` |
| `controller_pt_filter_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_total_ms` | `0.47` | `0.86` | `1.75` | `4.15` |
| `open3d_object_update_geometry_ms` | `0.03` | `0.05` | `0.11` | `3.24` |
| `open3d_controller_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `496` | `231.98` | `54332` | `14874` |
| `342` | `226.46` | `54317` | `14817` |
| `294` | `225.94` | `54349` | `14799` |
| `390` | `216.81` | `54324` | `14838` |
| `435` | `214.42` | `54314` | `14814` |
| `215` | `209.23` | `54340` | `14793` |
| `244` | `208.65` | `54335` | `14843` |
| `433` | `49.05` | `54332` | `14800` |
| `484` | `48.21` | `54347` | `14667` |
| `432` | `47.15` | `54330` | `14749` |
