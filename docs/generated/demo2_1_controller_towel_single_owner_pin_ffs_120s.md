# Demo 2.1 visual-5fps performance profile

- preset: `visual-5fps-single-owner`
- target FPS: `5.00`
- render FPS after warmup: `3.59`
- fusion FPS after warmup: `3.59`
- groups after warmup: `383`
- complete fused groups after warmup: `299`
- rendered groups after warmup: `299`
- complete group ratio after warmup: `0.781`
- target deficit: `1.41`
- bottleneck class: `upstream_supply`
- GPU pipeline: `single-owner`
- single-owner order: `ffs-then-edgetam`
- pin memory mode: `ffs`
- FFS input staging: `pinned`
- H2D stream mode: `dedicated`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `12.71` | `25.99` | `29.64` | `32.54` |
| `ffs_cycle_ms` | `101.25` | `109.97` | `114.39` | `303.77` |
| `ffs_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_cam0_model_ms` | `49.54` | `68.83` | `73.98` | `93.65` |
| `edgetam_cam1_model_ms` | `49.83` | `67.25` | `72.69` | `94.13` |
| `edgetam_cam2_model_ms` | `50.33` | `69.65` | `74.24` | `102.53` |
| `edgetam_cam0_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_cam1_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_cam2_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edge_pin_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edge_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edge_cam0_pin_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edge_cam1_pin_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edge_cam2_pin_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edge_cam0_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edge_cam1_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edge_cam2_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_stage_ms` | `0.74` | `7.84` | `9.00` | `11.59` |
| `ffs_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam0_stage_ms` | `0.75` | `1.20` | `1.34` | `10.38` |
| `ffs_cam1_stage_ms` | `0.84` | `9.45` | `10.11` | `11.59` |
| `ffs_cam2_stage_ms` | `0.65` | `1.34` | `2.15` | `3.74` |
| `ffs_cam0_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam1_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam2_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `gpu_owner_total_ms` | `263.22` | `309.25` | `341.86` | `510.14` |
| `gpu_owner_ffs_cycle_ms` | `101.25` | `109.97` | `114.39` | `303.77` |
| `gpu_owner_edgetam_cycle_ms` | `161.22` | `200.75` | `218.93` | `281.78` |
| `fusion_total_ms` | `52.72` | `60.43` | `64.52` | `250.06` |
| `object_enhanced_pt_ms` | `26.57` | `31.91` | `34.10` | `221.85` |
| `controller_pt_filter_ms` | `15.45` | `17.93` | `18.58` | `25.24` |
| `render_total_ms` | `0.49` | `0.77` | `1.02` | `8.27` |
| `open3d_object_update_geometry_ms` | `0.03` | `0.04` | `0.06` | `0.96` |
| `open3d_controller_update_geometry_ms` | `0.01` | `0.02` | `0.03` | `6.64` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `494` | `221.85` | `47357` | `11601` |
| `349` | `221.06` | `47350` | `11525` |
| `305` | `218.69` | `47291` | `11649` |
| `176` | `198.34` | `47371` | `11616` |
| `401` | `197.08` | `47401` | `11621` |
| `447` | `196.83` | `47390` | `11599` |
| `220` | `195.71` | `47361` | `11643` |
| `265` | `193.45` | `47384` | `11559` |
| `34` | `36.07` | `46646` | `11415` |
| `260` | `35.62` | `47410` | `11662` |
