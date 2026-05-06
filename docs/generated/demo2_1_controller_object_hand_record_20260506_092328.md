# Demo 2.1 visual-5fps performance profile

- preset: `visual-5fps`
- target FPS: `5.00`
- render FPS after warmup: `0.16`
- fusion FPS after warmup: `0.16`
- groups after warmup: `462`
- complete fused groups after warmup: `15`
- rendered groups after warmup: `15`
- target deficit: `4.84`
- bottleneck class: `upstream_supply`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `13.86` | `22.29` | `23.90` | `32.44` |
| `ffs_cycle_ms` | `455.05` | `534.62` | `562.64` | `1078.51` |
| `ffs_gate_wait_ms` | `339.76` | `421.54` | `443.25` | `914.47` |
| `edgetam_cam0_model_ms` | `141.71` | `168.79` | `177.07` | `578.57` |
| `edgetam_cam1_model_ms` | `140.84` | `165.68` | `175.22` | `588.44` |
| `edgetam_cam2_model_ms` | `137.13` | `162.72` | `170.16` | `222.50` |
| `edgetam_cam0_gate_wait_ms` | `60.19` | `124.63` | `134.80` | `183.12` |
| `edgetam_cam1_gate_wait_ms` | `63.05` | `128.19` | `136.80` | `170.65` |
| `edgetam_cam2_gate_wait_ms` | `77.15` | `129.68` | `142.22` | `159.64` |
| `fusion_total_ms` | `52.67` | `69.99` | `225.36` | `585.68` |
| `object_enhanced_pt_ms` | `33.43` | `37.33` | `38.66` | `41.72` |
| `controller_pt_filter_ms` | `8.64` | `15.80` | `178.28` | `553.49` |
| `render_total_ms` | `1.41` | `6.69` | `6.90` | `6.94` |
| `open3d_object_update_geometry_ms` | `0.03` | `2.50` | `3.74` | `6.00` |
| `open3d_controller_update_geometry_ms` | `0.02` | `2.60` | `3.84` | `6.10` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `326` | `41.72` | `46330` | `12576` |
| `248` | `37.36` | `49267` | `12955` |
| `205` | `37.30` | `45064` | `11611` |
| `177` | `35.88` | `44387` | `11565` |
| `213` | `35.17` | `45258` | `11666` |
| `485` | `34.42` | `51435` | `12915` |
| `420` | `34.40` | `46843` | `12111` |
| `525` | `33.43` | `44528` | `11692` |
| `587` | `32.89` | `49743` | `13350` |
| `561` | `31.76` | `46890` | `12132` |
