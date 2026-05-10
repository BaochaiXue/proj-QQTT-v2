# Demo 2.1 performance profile

- preset: `demo2.1.5-async-filter-5fps`
- canonical preset: `demo2.1.5-async-filter-5fps`
- target FPS: `15.00`
- capture group target FPS: `15.00`
- render FPS after warmup: `6.25`
- raw fusion FPS after warmup: `6.14`
- filter output FPS after warmup: `6.22`
- fusion FPS after warmup: `6.22`
- groups after warmup: `283`
- complete fused groups after warmup: `6`
- rendered groups after warmup: `6`
- complete group ratio after warmup: `0.021`
- target deficit: `8.75`
- bottleneck class: `upstream_supply`
- GPU pipeline: `single-owner`
- single-owner order: `ffs-then-edgetam`
- filter scheduler: `async`
- render filtered only: `True`
- pin memory mode: `off`
- FFS input staging: `pinned`
- H2D stream mode: `default`

## Init Profile

| Stage | value |
| --- | ---: |
| parallel init max wait ms | `23230.98` |
| camera startup ms | `5717.65` |
| EdgeTAM model load ms | `927.72` |
| EdgeTAM compile wrap ms | `673.05` |
| EdgeTAM compile prewarm ms | `20792.78` |
| EdgeTAM warmup/first forward ms | `1079.01` |
| SAM3.1 model load ms | `21175.98` |
| SAM3.1 cam0 segment ms | `413.96` |
| SAM3.1 cam1 segment ms | `218.84` |
| SAM3.1 cam2 segment ms | `235.99` |
| FFS runner init ms | `n/a` |
| FFS first run ms | `n/a` |
| session init + prompt add ms | `7.49` |
| SAM3.1 release cleanup ms | `485.13` |
| time to first complete group s | `33.86` |
| time to first rendered group s | `33.88` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `25.23` | `39.23` | `54.73` | `65.74` |
| `ffs_cycle_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_batch_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_model_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_preprocess_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_cam0_model_ms` | `78.88` | `462.12` | `731.61` | `1001.11` |
| `edgetam_cam1_model_ms` | `33.19` | `39.50` | `42.04` | `44.59` |
| `edgetam_cam2_model_ms` | `32.16` | `42.15` | `46.85` | `51.56` |
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
| `ffs_stage_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam0_stage_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam1_stage_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam2_stage_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam0_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam1_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam2_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `gpu_owner_total_ms` | `150.95` | `1105.72` | `1774.26` | `2442.80` |
| `gpu_owner_ffs_cycle_ms` | `0.69` | `1.34` | `1.75` | `2.16` |
| `gpu_owner_edgetam_cycle_ms` | `150.24` | `1104.33` | `1772.45` | `2440.58` |
| `raw_fusion_total_ms` | `16.83` | `22.49` | `24.10` | `25.72` |
| `fusion_total_ms` | `65.24` | `75.56` | `79.60` | `83.65` |
| `filter_total_ms` | `47.99` | `56.45` | `57.19` | `57.93` |
| `filter_input_age_ms` | `48.83` | `57.18` | `57.97` | `58.75` |
| `object_enhanced_pt_ms` | `32.79` | `38.67` | `39.23` | `39.78` |
| `controller_pt_filter_ms` | `16.18` | `17.77` | `17.96` | `18.14` |
| `render_total_ms` | `1.35` | `5.06` | `6.08` | `7.11` |
| `open3d_object_update_geometry_ms` | `0.69` | `2.18` | `2.28` | `2.37` |
| `open3d_controller_update_geometry_ms` | `0.02` | `1.21` | `1.80` | `2.39` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `267` | `39.78` | `46008` | `8193` |
| `298` | `37.56` | `46368` | `8437` |
| `295` | `33.37` | `46360` | `8437` |
| `303` | `32.20` | `46549` | `8417` |
| `302` | `30.08` | `46394` | `8452` |
| `300` | `29.44` | `46439` | `8474` |
| `0` | `0.00` | `0` | `0` |
| `1` | `0.00` | `0` | `0` |
| `2` | `0.00` | `0` | `0` |
| `3` | `0.00` | `0` | `0` |
