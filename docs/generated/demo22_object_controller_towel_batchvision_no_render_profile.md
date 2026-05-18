# Demo 2.2 performance profile

- preset: `demo2.2-async-filter-5fps`
- canonical preset: `demo2.2-async-filter-5fps`
- target FPS: `15.00`
- capture group target FPS: `15.00`
- compile mode: `vision-reduce-overhead`
- dtype: `bfloat16`
- EdgeTAM input path: `pil`
- mask postprocess: `cuda-inline`
- render FPS after warmup: `0.00`
- raw fusion FPS after warmup: `6.37`
- filter output FPS after warmup: `6.37`
- fusion FPS after warmup: `6.37`
- Demo 2.2 PASS threshold: `14.40 FPS`
- Demo 2.2 result: `FAIL`
- groups after warmup: `580`
- complete fused groups after warmup: `264`
- rendered groups after warmup: `0`
- complete group ratio after warmup: `0.455`
- target deficit: `15.00`
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
| parallel init max wait ms | `n/a` |
| camera startup ms | `4508.14` |
| EdgeTAM model load ms | `807.50` |
| EdgeTAM compile wrap ms | `0.01` |
| EdgeTAM compile prewarm ms | `1128.25` |
| EdgeTAM warmup/first forward ms | `72.28` |
| SAM3.1 model load ms | `7408.01` |
| SAM3.1 cam0 segment ms | `7655.15` |
| SAM3.1 cam1 segment ms | `178.12` |
| SAM3.1 cam2 segment ms | `180.73` |
| FFS runner init ms | `6029.41` |
| FFS first run ms | `976.41` |
| session init + prompt add ms | `16.63` |
| SAM3.1 release cleanup ms | `206.96` |
| time to first complete group s | `23.08` |
| time to first rendered group s | `n/a` |

## GPU Sampling

GPU sampling disabled for this run.

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `32.68` | `52.33` | `57.26` | `65.38` |
| `edgetam_model_ms` | `22.26` | `29.53` | `32.80` | `51.15` |
| `edgetam_preprocess_ms` | `0.84` | `0.99` | `1.03` | `1.20` |
| `edgetam_prompt_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_postprocess_ms` | `0.06` | `0.11` | `0.13` | `0.72` |
| `edgetam_mask_resize_ms` | `0.03` | `0.06` | `0.08` | `0.66` |
| `edgetam_mask_threshold_ms` | `0.03` | `0.04` | `0.06` | `0.18` |
| `edgetam_mask_to_cpu_ms` | `0.20` | `0.24` | `0.25` | `0.44` |
| `edgetam_total_ms` | `20.49` | `27.06` | `30.02` | `46.61` |
| `ffs_cycle_ms` | `62.14` | `67.75` | `70.73` | `252.00` |
| `ffs_batch_ms` | `46.93` | `50.36` | `52.22` | `227.66` |
| `ffs_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_model_ms` | `18.44` | `21.02` | `22.39` | `29.80` |
| `edgetam_batch_vision_total_ms` | `22.78` | `25.54` | `27.04` | `34.33` |
| `edgetam_batch_vision_preprocess_ms` | `2.53` | `2.97` | `3.09` | `3.59` |
| `edgetam_cam0_model_ms` | `22.83` | `28.62` | `31.60` | `37.17` |
| `edgetam_cam1_model_ms` | `21.95` | `28.97` | `30.89` | `45.75` |
| `edgetam_cam2_model_ms` | `21.93` | `30.61` | `34.08` | `51.15` |
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
| `ffs_stage_ms` | `2.09` | `3.37` | `4.36` | `9.66` |
| `ffs_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam0_stage_ms` | `2.09` | `3.37` | `4.34` | `9.66` |
| `ffs_cam1_stage_ms` | `2.09` | `3.37` | `4.34` | `9.66` |
| `ffs_cam2_stage_ms` | `2.09` | `3.37` | `4.34` | `9.66` |
| `ffs_cam0_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam1_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam2_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `gpu_owner_total_ms` | `147.11` | `172.51` | `188.13` | `369.39` |
| `gpu_owner_ffs_cycle_ms` | `62.14` | `67.75` | `70.73` | `252.00` |
| `gpu_owner_edgetam_cycle_ms` | `84.87` | `104.11` | `108.97` | `137.61` |
| `raw_fusion_total_ms` | `9.87` | `11.50` | `12.05` | `14.23` |
| `fusion_total_ms` | `44.37` | `49.11` | `50.47` | `238.02` |
| `filter_total_ms` | `34.33` | `37.88` | `39.41` | `227.52` |
| `filter_input_age_ms` | `34.89` | `38.39` | `40.01` | `228.08` |
| `object_enhanced_pt_ms` | `22.18` | `25.35` | `26.13` | `215.15` |
| `controller_pt_filter_ms` | `11.97` | `13.18` | `13.56` | `14.38` |
| `render_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_object_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_controller_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `458` | `215.15` | `45775` | `7866` |
| `375` | `207.81` | `45771` | `7925` |
| `604` | `191.10` | `45749` | `7906` |
| `532` | `189.39` | `45770` | `7916` |
| `673` | `188.80` | `45777` | `7915` |
| `303` | `186.44` | `45693` | `7947` |
| `232` | `185.67` | `45745` | `7934` |
| `348` | `32.88` | `45751` | `7944` |
| `455` | `30.23` | `45787` | `8020` |
| `669` | `26.64` | `45741` | `7862` |
