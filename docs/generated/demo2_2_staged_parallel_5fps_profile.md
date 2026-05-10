# Demo 2.2 performance profile

- preset: `demo2.2-staged-parallel-5fps`
- canonical preset: `demo2.2-staged-parallel-5fps`
- target FPS: `5.00`
- render FPS after warmup: `2.70`
- raw fusion FPS after warmup: `2.71`
- filter output FPS after warmup: `2.71`
- fusion FPS after warmup: `2.71`
- groups after warmup: `223`
- complete fused groups after warmup: `218`
- rendered groups after warmup: `217`
- complete group ratio after warmup: `0.978`
- target deficit: `2.30`
- Demo 2.2 PASS threshold: `4.80 FPS`
- Demo 2.2 result: `FAIL`
- bottleneck class: `upstream_supply`
- GPU pipeline: `staged`
- single-owner order: `ffs-then-parallel-edgetam`
- filter scheduler: `async`
- render filtered only: `True`
- pin memory mode: `all`
- FFS input staging: `pinned`
- H2D stream mode: `dedicated`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `42.91` | `57.25` | `62.66` | `154.35` |
| `ffs_cycle_ms` | `87.44` | `93.90` | `98.39` | `267.79` |
| `ffs_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_cam0_model_ms` | `125.12` | `139.71` | `147.01` | `165.97` |
| `edgetam_cam1_model_ms` | `125.88` | `140.35` | `145.07` | `165.20` |
| `edgetam_cam2_model_ms` | `126.10` | `140.60` | `146.78` | `156.26` |
| `edgetam_cam0_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_cam1_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_cam2_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edge_pin_copy_ms` | `1.04` | `1.59` | `1.78` | `3.39` |
| `edge_h2d_wait_ms` | `0.01` | `0.06` | `0.17` | `1.13` |
| `edge_cam0_pin_copy_ms` | `1.05` | `1.67` | `1.76` | `2.98` |
| `edge_cam1_pin_copy_ms` | `1.00` | `1.58` | `1.85` | `3.39` |
| `edge_cam2_pin_copy_ms` | `1.05` | `1.57` | `1.75` | `2.56` |
| `edge_cam0_h2d_wait_ms` | `0.01` | `0.05` | `0.13` | `0.86` |
| `edge_cam1_h2d_wait_ms` | `0.01` | `0.06` | `0.25` | `1.13` |
| `edge_cam2_h2d_wait_ms` | `0.01` | `0.06` | `0.16` | `0.78` |
| `ffs_stage_ms` | `0.74` | `7.29` | `7.79` | `9.09` |
| `ffs_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam0_stage_ms` | `0.64` | `1.01` | `1.20` | `1.56` |
| `ffs_cam1_stage_ms` | `0.96` | `7.96` | `8.43` | `9.09` |
| `ffs_cam2_stage_ms` | `0.72` | `1.38` | `1.81` | `2.57` |
| `ffs_cam0_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam1_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam2_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `gpu_owner_total_ms` | `230.54` | `249.00` | `257.61` | `406.55` |
| `gpu_owner_ffs_cycle_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `gpu_owner_edgetam_cycle_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `raw_fusion_total_ms` | `8.08` | `9.59` | `10.44` | `12.81` |
| `fusion_total_ms` | `42.80` | `46.72` | `49.10` | `226.49` |
| `filter_total_ms` | `34.61` | `37.60` | `39.58` | `216.51` |
| `filter_input_age_ms` | `35.09` | `38.16` | `39.89` | `217.18` |
| `object_enhanced_pt_ms` | `22.31` | `24.85` | `25.88` | `203.84` |
| `controller_pt_filter_ms` | `12.18` | `13.43` | `13.61` | `29.41` |
| `render_total_ms` | `0.39` | `0.57` | `1.04` | `4.59` |
| `open3d_object_update_geometry_ms` | `0.03` | `0.03` | `0.04` | `1.71` |
| `open3d_controller_update_geometry_ms` | `0.01` | `0.01` | `0.02` | `3.89` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `275` | `203.84` | `45579` | `7725` |
| `162` | `197.30` | `45801` | `7893` |
| `236` | `195.06` | `45826` | `7929` |
| `199` | `193.97` | `45834` | `7877` |
| `125` | `179.17` | `45566` | `7696` |
| `18` | `30.84` | `44772` | `7575` |
| `261` | `29.00` | `45770` | `7861` |
| `264` | `28.29` | `45659` | `11505` |
| `270` | `27.18` | `45837` | `7845` |
| `260` | `27.05` | `45732` | `7814` |
